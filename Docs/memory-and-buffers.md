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

### Via CopyFromJS (Browser Backends — Zero-Copy from JS)

When data originates in JavaScript — fetched from a `WebSocket`, decoded by a `FileReader`, pulled from `IndexedDB`, returned by `fetch`, produced by `MediaRecorder`, etc. — going through `CopyFromCPU` would marshal every byte through .NET arrays for no reason. `CopyFromJS` is the zero-copy path: data stays in JS, gets pushed straight into the GPU buffer. Available on **all three browser backends** (WebGPU, WebGL, Wasm). Not available on desktop (CUDA / OpenCL / CPU don't have JS).

```csharp
using SpawnDev.ILGPU;
using SpawnDev.BlazorJS.JSObjects;

// Get the IBrowserMemoryBuffer for any GPU buffer
var browserBuffer = (IBrowserMemoryBuffer)((IArrayView)buffer.View).Buffer;

// From a JS TypedArray (e.g. data already in JS land)
Int16Array jsData = ...;            // 16-bit PCM samples from a WebSocket message
browserBuffer.CopyFromJS(jsData);

// From a raw ArrayBuffer (e.g. a fetch() result body)
ArrayBuffer rawBytes = ...;          // image bytes from a fetch response
browserBuffer.CopyFromJS(rawBytes);

// Optional target offset — write into the middle of an existing GPU buffer
browserBuffer.CopyFromJS(jsData, targetByteOffset: 1024);
```

**Per-backend behavior:**
- **WebGPU** — calls `queue.WriteBuffer` directly on the JS-side typed array. Zero copy through .NET, no managed allocation.
- **WebGL** — copies the JS bytes into the buffer's backing array and sets `NeedsUpload = true`; the data is uploaded to the GPU on the next dispatch (not immediately, by design — WebGL has no async write queue).
- **Wasm** — pure JS→JS copy within the SharedArrayBuffer that backs Wasm linear memory. Worker threads see the new bytes immediately.

**When to use it:**
- Streaming pipelines where JS produces data (audio decode, video frames, network bytes)
- Loading models / assets from `fetch` or `IndexedDB`
- Canvas → GPU pipelines where you read `ImageData` once on the JS side and want to skip the .NET round trip

If your data is already in a `byte[]` / `float[]` / etc., use `CopyFromCPU` — that's the right path for managed-array sources. `CopyFromJS` is specifically for when the data hasn't crossed into .NET yet and you want to keep it that way.

## Copying Between GPU Buffers (GPU→GPU)

Use `CopyFrom` to copy data between GPU buffers. This works on **all six backends** — it's a native GPU operation with no CPU involvement:

```csharp
using var source = accelerator.Allocate1D(new float[] { 1, 2, 3, 4, 5 });
using var dest = accelerator.Allocate1D<float>(5);

// Copy GPU→GPU (fast, native, works everywhere)
dest.CopyFrom(source);
```

On WebGPU this maps to `CopyBufferToBuffer`. On CUDA/OpenCL it's a device-to-device memcpy. On Wasm it copies within the SharedArrayBuffer. No shader compilation, no kernel dispatch.

### Cross-Backend Buffer Operations Reference

Not all copy operations work the same way on every backend. This table shows what works and what throws:

| Operation | Method | CPU / CUDA / OpenCL | WebGPU | WebGL | Wasm |
|-----------|--------|--------------------:|-------:|------:|-----:|
| **GPU→GPU** | `CopyFrom` | Sync | `CopyBufferToBuffer` | TF readback | SharedArrayBuffer |
| **CPU→GPU** | `CopyFromCPU` / `Allocate1D(data)` | Sync | `queue.WriteBuffer` | `texImage2D` | SharedArrayBuffer |
| **JS→GPU** | `IBrowserMemoryBuffer.CopyFromJS` | N/A | `queue.WriteBuffer` | Backing-array copy + `NeedsUpload` | JS→JS within SharedArrayBuffer |
| **GPU→CPU (async)** | `CopyToHostAsync` | Sync fallback | `mapAsync(Read)` | Readback | SharedArrayBuffer |
| **GPU→CPU (sync)** | `CopyTo` / `CopyToCPU` / `GetAsArray1D` | Sync | **THROWS** | **THROWS** | **THROWS** |

**Key rules:**
- **GPU→GPU copies: always use `CopyFrom`.** It's fast, native, and works on all backends.
- **GPU→CPU reads: always use `CopyToHostAsync`.** The sync methods (`CopyTo`, `CopyToCPU`, `GetAsArray1D`) throw `NotSupportedException` on browser backends (WebGPU, WebGL, Wasm) because they require async GPU readback (`mapAsync`).
- **Never replace `CopyFrom` with a kernel dispatch** (e.g., a Scale-by-1 kernel). `CopyFrom` is a hardware copy command — it's faster, simpler, and doesn't require shader compilation. Using a kernel dispatch for GPU→GPU copies can cause initialization errors on WebGPU when the accelerator isn't fully ready.

## Reading Data from the GPU

### Synchronization

After launching a kernel, you must synchronize before reading results:

```csharp
kernel((Index1D)length, bufA.View, bufB.View, bufC.View);

// Synchronize() flushes commands to the backend (non-blocking, safe in WASM)
accelerator.Synchronize();

// SynchronizeAsync() flushes AND waits for GPU completion
await accelerator.SynchronizeAsync();
```

> **Semantics:** `Synchronize()` flushes queued commands to the backend but does **not** wait for completion and does **not** transfer data. `SynchronizeAsync()` flushes and waits for all GPU operations to complete. Neither transfers data — use `CopyToHostAsync()` to read results back to the CPU.

### CopyToHostAsync — Unified Extension Method

The simplest way to read GPU data. Works with **all six backends** (WebGPU, WebGL, Wasm, CUDA, OpenCL, CPU):

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
