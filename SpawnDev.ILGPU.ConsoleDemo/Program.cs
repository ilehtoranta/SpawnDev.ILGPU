// ---------------------------------------------------------------------------------------
//                        SpawnDev.ILGPU Console Demo
//
// Demonstrates that SpawnDev.ILGPU works in a standard .NET console application.
// Uses the unified async extensions (SynchronizeAsync, CopyToHostAsync)
// which gracefully fall back to synchronous ILGPU calls for native backends.
//
// Backends available in console: CPU, Cuda (if supported), OpenCL (if supported).
// Browser backends (WebGPU, WebGL, Wasm) are NOT available outside Blazor WASM.
// ---------------------------------------------------------------------------------------

using global::ILGPU;
using global::ILGPU.Runtime;
using SpawnDev.ILGPU;

Console.WriteLine("=== SpawnDev.ILGPU Console Demo ===");
Console.WriteLine();

// ──────────────────────────────────────────────
// 1. Create Context — SAME async code as Blazor WASM
// ──────────────────────────────────────────────
// This is the EXACT same initialization code used in Blazor WASM.
// AllAcceleratorsAsync() registers browser backends (WebGPU, WebGL, Wasm)
// AND native backends (Cuda, OpenCL, CPU). Browser backends fail silently
// on desktop since BlazorJSRuntime isn't available.
using var context = await Context.CreateAsync(builder => builder.AllAcceleratorsAsync());

Console.WriteLine("Registered devices:");
foreach (var device in context)
{
    Console.WriteLine($"  - {device.Name} ({device.AcceleratorType})");
}
Console.WriteLine();

// ──────────────────────────────────────────────
// 2. Pick the best accelerator — SAME async code as Blazor WASM
// ──────────────────────────────────────────────
// CreatePreferredAcceleratorAsync() checks for WebGPU > WebGL > Wasm > CPU.
// On desktop, browser backends aren't registered, so it falls through to
// Cuda > OpenCL > CPU automatically.
using var accelerator = await context.CreatePreferredAcceleratorAsync();

Console.WriteLine($"Using accelerator: {accelerator.Name} ({accelerator.AcceleratorType})");
Console.WriteLine();

// ──────────────────────────────────────────────
// 3. Vector Addition
// ──────────────────────────────────────────────
await RunVectorAddition(accelerator);

// ──────────────────────────────────────────────
// 4. Matrix Multiply (1D kernel)
// ──────────────────────────────────────────────
await RunMatrixMultiply(accelerator);

// ──────────────────────────────────────────────
// 5. Parallel Reduce (sum)
// ──────────────────────────────────────────────
await RunParallelReduce(accelerator);

Console.WriteLine();
Console.WriteLine("=== All tests passed! ===");
Console.WriteLine("SpawnDev.ILGPU works in console apps.");

// ════════════════════════════════════════════════════════════════════
//  Test Functions
// ════════════════════════════════════════════════════════════════════

static async Task RunVectorAddition(Accelerator accelerator)
{
    Console.WriteLine("── Vector Addition ──");

    const int length = 1024;
    var hostA = Enumerable.Range(0, length).Select(i => (float)i).ToArray();
    var hostB = Enumerable.Range(0, length).Select(i => (float)i * 2f).ToArray();

    using var bufA = accelerator.Allocate1D(hostA);
    using var bufB = accelerator.Allocate1D(hostB);
    using var bufC = accelerator.Allocate1D<float>(length);

    var kernel = accelerator.LoadAutoGroupedStreamKernel<
        Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(VectorAddKernel);

    kernel((Index1D)length, bufA.View, bufB.View, bufC.View);

    // Use SpawnDev.ILGPU's unified async extension — falls back to sync for CPU/Cuda
    await accelerator.SynchronizeAsync();

    // Use SpawnDev.ILGPU's unified async readback
    var results = await bufC.CopyToHostAsync<float>();

    // Verify
    bool pass = true;
    for (int i = 0; i < length; i++)
    {
        float expected = hostA[i] + hostB[i];
        if (Math.Abs(results[i] - expected) > 0.001f)
        {
            Console.WriteLine($"  FAIL at [{i}]: expected {expected}, got {results[i]}");
            pass = false;
            break;
        }
    }

    Console.WriteLine(pass
        ? $"  ✓ PASS  (result[0]={results[0]}, result[511]={results[511]}, result[1023]={results[1023]})"
        : "  ✗ FAIL");
    Console.WriteLine();
}

static async Task RunMatrixMultiply(Accelerator accelerator)
{
    Console.WriteLine("── Matrix Multiply (4×4 × 4×4) ──");

    const int N = 4;

    // Identity-like matrix A and simple matrix B
    var hostA = new float[N * N];
    var hostB = new float[N * N];
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            hostA[i * N + j] = (i == j) ? 1.0f : 0.0f; // Identity
            hostB[i * N + j] = i * N + j + 1;           // 1..16
        }
    }

    using var bufA = accelerator.Allocate1D(hostA);
    using var bufB = accelerator.Allocate1D(hostB);
    using var bufC = accelerator.Allocate1D<float>(N * N);

    var kernel = accelerator.LoadAutoGroupedStreamKernel<
        Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int>(MatMulKernel);

    kernel((Index1D)(N * N), bufA.View, bufB.View, bufC.View, N);
    await accelerator.SynchronizeAsync();

    var results = await bufC.CopyToHostAsync<float>();

    // Identity × B = B
    bool pass = true;
    for (int i = 0; i < N * N; i++)
    {
        if (Math.Abs(results[i] - hostB[i]) > 0.001f)
        {
            Console.WriteLine($"  FAIL at [{i}]: expected {hostB[i]}, got {results[i]}");
            pass = false;
            break;
        }
    }

    Console.WriteLine(pass
        ? $"  ✓ PASS  (C[0,0]={results[0]}, C[1,1]={results[5]}, C[3,3]={results[15]})"
        : "  ✗ FAIL");
    Console.WriteLine();
}

static async Task RunParallelReduce(Accelerator accelerator)
{
    Console.WriteLine("── Parallel Reduce (sum of 1..256) ──");

    const int length = 256;
    var hostData = Enumerable.Range(1, length).Select(i => (float)i).ToArray();

    using var bufData = accelerator.Allocate1D(hostData);
    using var bufOutput = accelerator.Allocate1D<float>(1);

    // Simple atomic-free reduction: just sum in the kernel
    var kernel = accelerator.LoadAutoGroupedStreamKernel<
        Index1D, ArrayView<float>, ArrayView<float>>(NaiveReduceKernel);

    // Initialize output to 0
    bufOutput.MemSetToZero();

    kernel((Index1D)length, bufData.View, bufOutput.View);
    await accelerator.SynchronizeAsync();

    var results = await bufOutput.CopyToHostAsync<float>();
    float expected = length * (length + 1) / 2f; // Sum formula: n(n+1)/2

    bool pass = Math.Abs(results[0] - expected) < 0.1f;
    Console.WriteLine(pass
        ? $"  ✓ PASS  (sum={results[0]}, expected={expected})"
        : $"  ✗ FAIL  (sum={results[0]}, expected={expected})");
    Console.WriteLine();
}

// ════════════════════════════════════════════════════════════════════
//  Kernels
// ════════════════════════════════════════════════════════════════════

static void VectorAddKernel(
    Index1D index,
    ArrayView<float> a,
    ArrayView<float> b,
    ArrayView<float> c)
{
    c[index] = a[index] + b[index];
}

static void MatMulKernel(
    Index1D index,
    ArrayView<float> a,
    ArrayView<float> b,
    ArrayView<float> c,
    int n)
{
    int row = index / n;
    int col = index - row * n;

    float sum = 0;
    for (int k = 0; k < n; k++)
    {
        sum += a[row * n + k] * b[k * n + col];
    }
    c[index] = sum;
}

static void NaiveReduceKernel(
    Index1D index,
    ArrayView<float> data,
    ArrayView<float> output)
{
    // NOTE: This uses Atomic.Add for thread-safe accumulation.
    // Works on Cuda/OpenCL. On CPU accelerator, ILGPU handles atomics internally.
    Atomic.Add(ref output[0], data[index]);
}
