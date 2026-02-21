# API Reference

Public classes and extension methods organized by namespace.

## SpawnDev.ILGPU

Core extension methods that work across all backends.

### SpawnDevContextExtensions

Extension methods for the ILGPU `Context` class.

```csharp
using SpawnDev.ILGPU;
```

| Method | Return | Description |
|--------|--------|-------------|
| `Context.CreateAsync(Action<Builder>)` | `Task<Context>` | Creates a context using an async builder (required for WebGPU probing) |
| `builder.AllAcceleratorsAsync()` | `Builder` | Enables all WASM backends: WebGPU, WebGL, Wasm. Silently skips unavailable ones |
| `builder.WebGPU()` | `Builder` | Enables WebGPU backend only |
| `builder.WebGL()` | `Builder` | Enables WebGL backend only |
| `builder.Wasm()` | `Builder` | Enables Wasm backend only |
| `context.CreatePreferredAcceleratorAsync()` | `Task<Accelerator>` | Creates the best available accelerator (WebGPU > WebGL > Wasm) |
| `accelerator.SynchronizeAsync()` | `Task` | Async wait for all GPU work to complete |
| `buffer.CopyToHostAsync<T>()` | `Task<T[]>` | Copies buffer data to a new array (works with all backends) |
| `context.GetWebGPUDevices()` | `List<WebGPUILGPUDevice>` | Lists registered WebGPU devices |
| `context.GetWebGLDevices()` | `List<WebGLILGPUDevice>` | Lists registered WebGL devices |

### IBrowserMemoryBuffer

Interface implemented by GPU-backed memory buffers for browser interop.

```csharp
public interface IBrowserMemoryBuffer
{
    Task<Uint8Array> CopyToHostUint8ArrayAsync(long sourceByteOffset = 0, long? copyBytes = null);
}
```

---

## SpawnDev.ILGPU.WebGPU

### WebGPUAccelerator

The main WebGPU accelerator — extends ILGPU's `KernelAccelerator`.

| Member | Type | Description |
|--------|------|-------------|
| `CreateAsync(context, device)` | `Task<WebGPUAccelerator>` | Creates accelerator with default options |
| `CreateAsync(context, device, options)` | `Task<WebGPUAccelerator>` | Creates accelerator with custom options |
| `CreateFromExternalDevice(context, gpuDevice, options?)` | `WebGPUAccelerator` | Creates from an external `GPUDevice` (device sharing) |
| `NativeAccelerator` | `WebGPUNativeAccelerator` | Low-level WebGPU access |
| `Backend` | `WebGPUBackend` | The WGSL transpiler backend |
| `EnabledFeatures` | `HashSet<string>` | Detected WebGPU features (e.g., `subgroups`, `shader-f16`) |

### WebGPUBuffer\<T\>

Typed GPU memory buffer with staging buffer caching for efficient readback.

| Member | Type | Description |
|--------|------|-------------|
| `NativeBuffer` | `GPUBuffer?` | Underlying WebGPU buffer |
| `Length` | `long` | Number of elements |
| `ElementSize` | `int` | Size of each element in bytes |
| `LengthInBytes` | `long` | Total buffer size |
| `CopyFromHost(T[])` | `void` | Upload data from CPU to GPU |
| `CopyToHostAsync()` | `Task<T[]>` | Download data — allocates new array |
| `CopyToHostAsync(T[], offset, count)` | `Task<long>` | Download into pre-allocated array (zero-alloc) |
| `CopyToHostUint8ArrayAsync(offset, bytes)` | `Task<Uint8Array>` | Download as JS Uint8Array |
| `Fill(T)` | `void` | Fill buffer with a value |

### WebGPUDevice

Represents a WebGPU device available in the browser.

| Member | Type | Description |
|--------|------|-------------|
| `GetDevicesAsync()` | `Task<List<WebGPUDevice>>` | Detect all available WebGPU devices |
| `GetDefaultDeviceAsync()` | `Task<WebGPUDevice?>` | Get the first available device |
| `CreateAcceleratorAsync(context)` | `Task<WebGPUAccelerator>` | Create a WebGPU accelerator |
| `PrintInfo()` | `void` | Print device information to console |

### WebGPUNativeAccelerator

Low-level WebGPU operations — for advanced users who need direct GPU API access.

| Member | Type | Description |
|--------|------|-------------|
| `NativeDevice` | `GPUDevice?` | The underlying WebGPU device |
| `Queue` | `GPUQueue?` | The device's command queue |
| `EnabledFeatures` | `HashSet<string>` | Detected features |
| `FlushPendingCommands` | `Action?` | Callback to flush batched commands |
| `CreateFromExternalDevice(gpuDevice)` | `WebGPUNativeAccelerator` | Wrap an external device |
| `GetOrCreateComputeShader(source, entry, overrides?)` | `WebGPUComputeShader` | Compile/cache WGSL shader |
| `Dispatch(shader, bindGroup, dims)` | `void` | Dispatch a compute shader |

---

## SpawnDev.ILGPU.WebGPU.Backend

### WebGPUBackend

WGSL transpiler backend.

| Member | Type | Description |
|--------|------|-------------|
| `VerboseLogging` | `static bool` | Enable/disable debug output |
| `EnableReflectionCaching` | `static bool` | Enable reflection metadata caching |
| `EnableBufferPooling` | `static bool` | Enable scalar buffer pooling |

### WebGPUBackendOptions

Configuration for the WebGPU transpiler.

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `EnableF64Emulation` | `bool` | `true` | Emulate `double` as `vec2<f32>` |
| `UseOzakiF64Emulation` | `bool` | `false` | Use Ozaki (precise) vs Dekker (fast) |
| `EnableI64Emulation` | `bool` | `true` | Emulate `long`/`ulong` as `vec2<u32>` |

### ExternalWebGPUMemoryBuffer

Non-owning wrapper for externally-managed GPU buffers.

```csharp
// Wrap an external GPUBuffer for use in ILGPU kernels
var wrapped = new ExternalWebGPUMemoryBuffer(accelerator, gpuBuffer, elementCount, elementSize);

// Disposing does NOT destroy the underlying GPUBuffer
wrapped.Dispose();
```

---

## SpawnDev.ILGPU.WebGL

### WebGLAccelerator

WebGL2 accelerator — all GL calls are offloaded to a dedicated Web Worker.

| Member | Type | Description |
|--------|------|-------------|
| `Create(context, device, options?)` | `WebGLAccelerator` | Creates a WebGL accelerator |
| `SynchronizeAsync()` | `Task` | Async wait for GPU work (via extension method) |

### WebGLDevice

Represents a WebGL2 device.

| Member | Type | Description |
|--------|------|-------------|
| `GetDevicesAsync()` | `Task<List<WebGLDevice>>` | Detect available WebGL2 devices |
| `CreateAccelerator(context, options?)` | `WebGLAccelerator` | Create a WebGL accelerator |

---

## SpawnDev.ILGPU.WebGL.Backend

### WebGLBackend

GLSL ES 3.0 transpiler backend.

| Member | Type | Description |
|--------|------|-------------|
| `VerboseLogging` | `static bool` | Enable/disable debug output |

### WebGLBackendOptions

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `EnableF64Emulation` | `bool` | `true` | Emulate `double` precision |
| `EnableI64Emulation` | `bool` | `true` | Emulate `long`/`ulong` |

---

## SpawnDev.ILGPU.Wasm

### WasmAccelerator

WebAssembly accelerator — compiles kernels to native Wasm and dispatches across Web Workers.

| Member | Type | Description |
|--------|------|-------------|
| `SynchronizeAsync()` | `Task` | Async wait for all workers (via extension method) |

### WasmILGPUDevice

Represents a Wasm compute device.

| Member | Type | Description |
|--------|------|-------------|
| `CreateAcceleratorAsync(context)` | `Task<WasmAccelerator>` | Create a Wasm accelerator |

### WasmMemoryBuffer

Wasm-backed memory buffer using SharedArrayBuffer.

| Member | Type | Description |
|--------|------|-------------|
| `CopyToHostUint8ArrayAsync(offset, bytes)` | `Task<Uint8Array>` | Read data as JS Uint8Array |

---

## SpawnDev.ILGPU.Wasm.Backend

### WasmBackend

Wasm compilation backend.

| Member | Type | Description |
|--------|------|-------------|
| `VerboseLogging` | `static bool` | Enable/disable debug output |

### WasmBackendOptions

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `MaxWorkers` | `int?` | `null` (auto) | Maximum Web Workers for dispatch |
