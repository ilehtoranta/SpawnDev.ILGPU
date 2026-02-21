# Memory & Buffers

This guide covers GPU memory management in SpawnDev.ILGPU — allocating buffers, transferring data, and reading results back asynchronously.

## Core Concepts

GPU memory is separate from CPU memory. You allocate **buffers** on the GPU, pass **views** into kernels, and **copy** results back to CPU arrays when done.

| Concept | Description |
|---------|-------------|
| **`MemoryBuffer1D<T, Stride>`** | Host-side handle to GPU memory — allocated and disposed on the CPU |
| **`ArrayView<T>`** | GPU-side reference — passed to kernels, indexable like an array |
| **`CopyToHostAsync`** | Reads data from GPU back to a CPU array |
| **`SynchronizeAsync`** | Waits for all GPU work to complete |

## Allocating Buffers

### 1D Buffers

```csharp
// Allocate empty buffer
using var buffer = accelerator.Allocate1D<float>(1024);

// Allocate and initialize from an array
float[] hostData = { 1, 2, 3, 4, 5 };
using var buffer = accelerator.Allocate1D(hostData);
```

### 2D Buffers

```csharp
// Allocate 2D buffer (e.g., for image processing)
int width = 800, height = 600;
using var buffer = accelerator.Allocate2DDenseX<uint>(new Index2D(width, height));

// Access extent for kernel launch
kernel(buffer.IntExtent, buffer.View, ...);
```

### Stride Types

Stride controls how multi-dimensional data is laid out in memory:

| Stride | Use Case |
|--------|----------|
| `Stride1D.Dense` | 1D buffers (default) |
| `Stride2D.DenseX` | 2D buffers — rows are contiguous (X varies fastest) |

When in doubt, use `Dense` for 1D and `DenseX` for 2D. These match standard C# array layouts.

## Passing Data to the GPU

### Via Buffer Initialization

```csharp
var data = new float[] { 1, 2, 3, 4, 5 };
using var buffer = accelerator.Allocate1D(data);
// Buffer now contains the array data on the GPU
```

### Via CopyFromCPU

```csharp
using var buffer = accelerator.Allocate1D<float>(1024);
float[] newData = ComputeNewData();
buffer.CopyFromCPU(newData);
```

### Via Scalar Parameters

Scalar values (`int`, `float`, etc.) are passed directly to kernels as parameters — no buffer needed:

```csharp
static void MyKernel(Index1D index, ArrayView<float> data, float multiplier, int offset)
{
    data[index] = data[index] * multiplier + offset;
}

// Pass scalars directly
kernel((Index1D)length, buffer.View, 2.5f, 10);
```

## Reading Data from the GPU

### SynchronizeAsync (Wait for Completion)

After launching a kernel, you **must** wait for the GPU to finish before reading results:

```csharp
kernel((Index1D)length, bufA.View, bufB.View, bufC.View);
await accelerator.SynchronizeAsync();  // Wait for kernel to complete
```

> **Critical:** Always use `SynchronizeAsync()` in Blazor WASM. The synchronous `Synchronize()` will deadlock. See [Limitations](limitations.md).

### CopyToHostAsync — Unified Extension Method

The simplest way to read GPU data. Works with **all backends** (WebGPU, WebGL, Wasm, CPU):

```csharp
using SpawnDev.ILGPU;

// Read entire buffer to a new array
float[] results = await buffer.CopyToHostAsync<float>();

// Read 1D buffer directly
var results = await buffer1D.CopyToHostAsync<float>();
```

### CopyToHostAsync — Pre-allocated Destination

For render loops, avoid allocating a new array every frame:

```csharp
// Allocate once
float[] cachedResults = new float[bufferLength];

// Reuse every frame
await buffer.CopyToHostAsync(cachedResults);
```

### CopyToHostUint8ArrayAsync — JavaScript Interop

Returns a JavaScript `Uint8Array` for direct use with browser APIs (Canvas, WebGL textures, etc.):

```csharp
// Get raw bytes as a JS Uint8Array
using var bytes = await browserBuffer.CopyToHostUint8ArrayAsync(0, byteCount);

// Use directly with Canvas ImageData
destPixels.Set(bytes);
```

This is the fastest path for GPU → Canvas rendering. See [Advanced Patterns](advanced-patterns.md) for the full render pipeline.

### IBrowserMemoryBuffer Interface

All GPU-backed buffers implement `IBrowserMemoryBuffer`, which provides `CopyToHostUint8ArrayAsync`:

```csharp
var internalBuffer = ((IArrayView)outputBuffer).Buffer;

if (internalBuffer is IBrowserMemoryBuffer browserBuffer)
{
    // Fast JS-interop readback
    using var bytes = await browserBuffer.CopyToHostUint8ArrayAsync(0, byteCount);
    destPixels.Set(bytes);
}
else
{
    // CPU fallback
    outputBuffer.View.BaseView.CopyToCPU(resultArray);
}
```

## Buffer Lifecycle

### Disposal Order

Dispose in **reverse order** of creation:

```csharp
using var context = await Context.CreateAsync(builder => builder.AllAcceleratorsAsync());
using var accelerator = await context.CreatePreferredAcceleratorAsync();
using var buffer = accelerator.Allocate1D<float>(1024);

// ... use buffer ...

// `using` disposes in reverse: buffer → accelerator → context
```

### Key Rules

1. **Buffers are tied to their accelerator** — you cannot use a buffer from one accelerator with another
2. **Dispose buffers before their accelerator** — disposing out of order causes errors
3. **Views don't need disposal** — `ArrayView<T>` is a lightweight struct (like `Span<T>`)
4. **Don't access disposed buffers** — `CopyToHostAsync` on a disposed buffer throws `ObjectDisposedException`

## Zero-Allocation Hot Path

For real-time rendering, the buffer readback system supports a zero-allocation path:

```csharp
// Cache the staging buffer (created once, reused)
// This happens automatically inside WebGPUBuffer<T>.CopyToHostAsync

// Pre-allocate the result array
uint[] cachedResult = new uint[pixelCount];

// Each frame: no GC allocations
await nativeBuffer.CopyToHostAsync(cachedResult, 0, pixelCount);
```

The `WebGPUBuffer<T>` internally maintains a cached staging buffer for readback operations. The first call creates it; subsequent calls reuse it (as long as the size doesn't increase).

## Memory Tips

1. **Batch your uploads** — call `CopyFromCPU` before launching kernels, not inside render loops
2. **Reuse buffers** — allocate once, use many times, dispose at cleanup
3. **Avoid per-frame allocation** — pre-allocate host arrays and use the `CopyToHostAsync(destination)` overload
4. **Use `IBrowserMemoryBuffer`** for Canvas rendering — it skips the .NET array entirely when possible
5. **Buffer pooling is automatic** — the WebGPU backend pools scalar parameter buffers internally
