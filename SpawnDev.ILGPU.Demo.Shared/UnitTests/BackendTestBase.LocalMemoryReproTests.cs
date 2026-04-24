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
}
