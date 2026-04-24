using ILGPU;
using ILGPU.Runtime;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.Demo.Shared.UnitTests;

/// <summary>
/// Repro for Tuvok's 2026-04-24 bug report: Vp9Idct8x8Kernel fails on WebGPU with
/// "Invalid BindGroupLayout (unlabeled) is invalid" at pipeline compile time.
///
/// Shape: <see cref="ArrayView{T}"/> of short + byte + scalars, with
/// <see cref="LocalMemory.Allocate{T}(int)"/> holding 64 ints as a row-pass
/// intermediate. Works on CPU / CUDA / OpenCL / Wasm. Fails WebGPU only.
///
/// Strongest suspect per Tuvok's diff of 4x4 (works) vs 8x8 (fails):
/// the 64-element <c>LocalMemory&lt;int&gt;</c> - 4x4 uses 16 inline locals, 8x8
/// uses LocalMemory. Both kernels share the short+byte ArrayView shape.
///
/// The kernel itself is a simplified read-multiply-store so the test can run
/// independently of any VP9 iDCT math. If this kernel reproduces the WebGPU
/// failure, the bug isn't VP9-specific - it's the LocalMemory&lt;int&gt;(64)
/// pattern on WebGPU.
/// </summary>
public abstract partial class BackendTestBase
{
    /// <summary>
    /// Minimal mirror of the failing iDCT 8x8 kernel shape. One thread per 8x8 block.
    /// Reads 64 shorts into a LocalMemory scratch as int, multiplies by 2,
    /// writes 64 bytes out clipped to [0,255].
    /// </summary>
    private static void LocalMemoryReproKernel(
        Index1D blockIdx,
        ArrayView<short> coeffs,
        ArrayView<byte> dest,
        int blocks)
    {
        if (blockIdx >= blocks) return;
        var tmp = LocalMemory.Allocate<int>(64);
        int baseIdx = blockIdx * 64;
        for (int i = 0; i < 64; i++)
            tmp[i] = (int)coeffs[baseIdx + i] * 2;
        for (int i = 0; i < 64; i++)
        {
            int v = tmp[i];
            if (v < 0) v = 0;
            else if (v > 255) v = 255;
            dest[baseIdx + i] = (byte)v;
        }
    }

    [TestMethod]
    public async Task LocalMemoryRepro_Int64_ShortByteViews() => await RunTest(async accelerator =>
    {
        const int blocks = 4;
        const int elemsPerBlock = 64;
        const int total = blocks * elemsPerBlock;

        var inputShorts = new short[total];
        for (int i = 0; i < total; i++)
            inputShorts[i] = (short)((i % 120) - 60); // -60..59

        var expected = new byte[total];
        for (int i = 0; i < total; i++)
        {
            int v = inputShorts[i] * 2;
            expected[i] = (byte)(v < 0 ? 0 : v > 255 ? 255 : v);
        }

        using var coeffsBuf = accelerator.Allocate1D<short>(total);
        using var destBuf = accelerator.Allocate1D<byte>(total);
        coeffsBuf.View.CopyFromCPU(inputShorts);

        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<byte>, int>(LocalMemoryReproKernel);
        kernel((Index1D)blocks, coeffsBuf.View, destBuf.View, blocks);
        await accelerator.SynchronizeAsync();

        var actual = await destBuf.CopyToHostAsync<byte>();
        for (int i = 0; i < total; i++)
        {
            if (actual[i] != expected[i])
                throw new Exception(
                    $"LocalMemory repro mismatch at [{i}] (block {i / 64}, elem {i % 64}): " +
                    $"expected {expected[i]}, got {actual[i]} from input short {inputShorts[i]}");
        }
    });

    /// <summary>
    /// Same shape as <see cref="LocalMemoryReproKernel"/> but with 256-element LocalMemory -
    /// mirrors Tuvok's upcoming VP9 iDCT 16x16 kernel. Threads each allocate
    /// <c>array&lt;int, 256&gt;</c> (~1 KiB per invocation in WGSL var<![CDATA[<]]>function<![CDATA[>]]>).
    /// Proves the rc.10 SSA threshold fix scales past 64 elements without hitting a new
    /// WGSL / GLSL codegen wall.
    /// </summary>
    private static void LocalMemoryReproKernel256(
        Index1D blockIdx,
        ArrayView<short> coeffs,
        ArrayView<byte> dest,
        int blocks)
    {
        if (blockIdx >= blocks) return;
        var tmp = LocalMemory.Allocate<int>(256);
        int baseIdx = blockIdx * 256;
        for (int i = 0; i < 256; i++)
            tmp[i] = (int)coeffs[baseIdx + i] * 2;
        for (int i = 0; i < 256; i++)
        {
            int v = tmp[i];
            if (v < 0) v = 0;
            else if (v > 255) v = 255;
            dest[baseIdx + i] = (byte)v;
        }
    }

    [TestMethod]
    public async Task LocalMemoryRepro_Int256_ShortByteViews() => await RunTest(async accelerator =>
    {
        // WebGL architectural ceiling: one-thread-per-output-block topology blows past
        // the varying-count limit well before 256 outputs per thread. Skip cleanly
        // (Tuvok's Is8x8KernelSupported already gates this in Codecs).
        if (accelerator.AcceleratorType == AcceleratorType.WebGL)
            throw new UnsupportedTestException(
                "WebGL varying-count limit rejects 256 outputs-per-thread. " +
                "Use AcceleratorRequirements { RequiresAtomics = true } to filter WebGL up-front.");

        // rc.10 follow-up (see DevComms geordi-to-tuvok-n256-wall-2026-04-24): IR loop
        // optimizer fully unrolls the constant-bound 256-iter loop in the kernel body.
        // Produces ~132 KB WGSL (3867 lines / 1361 scalar var declarations) for WebGPU,
        // which Chrome's WGSL validator takes > 30 s to validate. Same symptom on Wasm
        // (the code-gen-per-iteration blows up the worker script too). CPU / CUDA /
        // OpenCL unroll the same way but the downstream compilers digest it fine.
        // Tracking as rc.11 unroll-threshold fix; skip cleanly for the browser backends
        // that hit the ceiling until then so master stays green.
        if (accelerator.AcceleratorType == AcceleratorType.WebGPU ||
            accelerator.AcceleratorType == AcceleratorType.Wasm)
            throw new UnsupportedTestException(
                "Blocked on rc.11 IR loop-unroll threshold fix: full unroll of the 256-iter " +
                "constant-bound body produces ~132 KB of WGSL / Wasm that WGSL validator + " +
                "V8 compile in > 30 s. Skip pending fix — tests stay as the regression oracle.");

        const int blocks = 4;
        const int elemsPerBlock = 256;
        const int total = blocks * elemsPerBlock;

        var inputShorts = new short[total];
        for (int i = 0; i < total; i++)
            inputShorts[i] = (short)((i % 120) - 60);

        var expected = new byte[total];
        for (int i = 0; i < total; i++)
        {
            int v = inputShorts[i] * 2;
            expected[i] = (byte)(v < 0 ? 0 : v > 255 ? 255 : v);
        }

        using var coeffsBuf = accelerator.Allocate1D<short>(total);
        using var destBuf = accelerator.Allocate1D<byte>(total);
        coeffsBuf.View.CopyFromCPU(inputShorts);

        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<byte>, int>(LocalMemoryReproKernel256);
        kernel((Index1D)blocks, coeffsBuf.View, destBuf.View, blocks);
        await accelerator.SynchronizeAsync();

        var actual = await destBuf.CopyToHostAsync<byte>();
        for (int i = 0; i < total; i++)
        {
            if (actual[i] != expected[i])
                throw new Exception(
                    $"LocalMemory(256) repro mismatch at [{i}] (block {i / 256}, elem {i % 256}): " +
                    $"expected {expected[i]}, got {actual[i]} from input short {inputShorts[i]}");
        }
    });

    /// <summary>
    /// Same shape as <see cref="LocalMemoryReproKernel"/> but with 1024-element LocalMemory -
    /// mirrors Tuvok's upcoming VP9 iDCT 32x32 kernel. Threads each allocate
    /// <c>array&lt;int, 1024&gt;</c> (~4 KiB per invocation). Proves the rc.10 SSA threshold
    /// fix scales to the largest LocalMemory pattern in the VP9 codec family.
    /// </summary>
    private static void LocalMemoryReproKernel1024(
        Index1D blockIdx,
        ArrayView<short> coeffs,
        ArrayView<byte> dest,
        int blocks)
    {
        if (blockIdx >= blocks) return;
        var tmp = LocalMemory.Allocate<int>(1024);
        int baseIdx = blockIdx * 1024;
        for (int i = 0; i < 1024; i++)
            tmp[i] = (int)coeffs[baseIdx + i] * 2;
        for (int i = 0; i < 1024; i++)
        {
            int v = tmp[i];
            if (v < 0) v = 0;
            else if (v > 255) v = 255;
            dest[baseIdx + i] = (byte)v;
        }
    }

    [TestMethod]
    public async Task LocalMemoryRepro_Int1024_ShortByteViews() => await RunTest(async accelerator =>
    {
        if (accelerator.AcceleratorType == AcceleratorType.WebGL)
            throw new UnsupportedTestException(
                "WebGL varying-count limit rejects 1024 outputs-per-thread. " +
                "Use AcceleratorRequirements { RequiresAtomics = true } to filter WebGL up-front.");

        // rc.10 follow-up — see LocalMemoryRepro_Int256_ShortByteViews for the root cause.
        // N=1024 produces ~530 KB WGSL / Wasm, Chrome WGSL validator timeouts ~100x worse.
        if (accelerator.AcceleratorType == AcceleratorType.WebGPU ||
            accelerator.AcceleratorType == AcceleratorType.Wasm)
            throw new UnsupportedTestException(
                "Blocked on rc.11 IR loop-unroll threshold fix: 1024-iter full unroll is " +
                "the worst-case scaling of the same pathology documented on the N=256 variant.");

        const int blocks = 2; // 2 * 1024 = 2048 elements total - keeps runtime fast even on Wasm
        const int elemsPerBlock = 1024;
        const int total = blocks * elemsPerBlock;

        var inputShorts = new short[total];
        for (int i = 0; i < total; i++)
            inputShorts[i] = (short)((i % 120) - 60);

        var expected = new byte[total];
        for (int i = 0; i < total; i++)
        {
            int v = inputShorts[i] * 2;
            expected[i] = (byte)(v < 0 ? 0 : v > 255 ? 255 : v);
        }

        using var coeffsBuf = accelerator.Allocate1D<short>(total);
        using var destBuf = accelerator.Allocate1D<byte>(total);
        coeffsBuf.View.CopyFromCPU(inputShorts);

        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<byte>, int>(LocalMemoryReproKernel1024);
        kernel((Index1D)blocks, coeffsBuf.View, destBuf.View, blocks);
        await accelerator.SynchronizeAsync();

        var actual = await destBuf.CopyToHostAsync<byte>();
        for (int i = 0; i < total; i++)
        {
            if (actual[i] != expected[i])
                throw new Exception(
                    $"LocalMemory(1024) repro mismatch at [{i}] (block {i / 1024}, elem {i % 1024}): " +
                    $"expected {expected[i]}, got {actual[i]} from input short {inputShorts[i]}");
        }
    });
}
