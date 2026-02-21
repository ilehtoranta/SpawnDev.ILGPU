# Backends

SpawnDev.ILGPU supports multiple backends for running ILGPU kernels. In the browser, three backends (WebGPU, WebGL, Wasm) bring GPU compute to Blazor WebAssembly. On desktop and server, ILGPU's native Cuda and OpenCL backends are available. The CPU backend works everywhere. The same kernel code and async extensions work across all backends.

## Overview

### Browser Backends

| | 🎮 **WebGPU** | 🖼️ **WebGL** | 🧊 **Wasm** | 🐢 **CPU** |
|---|---|---|---|---|
| **Executes on** | GPU | GPU | Web Workers | Main thread |
| **Transpiles to** | WGSL | GLSL ES 3.0 | WebAssembly binary | — (interpreted) |
| **Technique** | Compute shader | Transform Feedback | Multi-worker | Single-threaded |
| **Blocking** | Non-blocking | Non-blocking | Non-blocking | ⚠️ Blocks UI |
| **Shared Memory** | ✅ | ❌ | ✅ | ⚠️ Barriers broken |
| **Atomics** | ✅ | ❌ | ✅ | ⚠️ Crashes in WASM |
| **64-bit (f64/i64)** | ✅ Emulated | ✅ Emulated | ✅ Native | ✅ Native |
| **Browser support** | Chrome/Edge 113+ | All modern browsers | All modern browsers | All modern browsers |

### Desktop/Server Backends

| | 🚀 **Cuda** | 🔧 **OpenCL** | 🐢 **CPU** |
|---|---|---|---|
| **Executes on** | NVIDIA GPU | AMD/Intel GPU | CPU cores |
| **Transpiles to** | PTX | OpenCL C | — (interpreted) |
| **Shared Memory** | ✅ | ✅ | ✅ |
| **Atomics** | ✅ | ✅ | ✅ |
| **64-bit** | ✅ Native | ✅ Native | ✅ Native |
| **Requirement** | NVIDIA GPU + driver | OpenCL 2.0+ GPU | None |

**Auto-selection priority (browser):** WebGPU → WebGL → Wasm
**Auto-selection priority (desktop):** Cuda → OpenCL → CPU

## Automatic Backend Selection

### Browser (Blazor WASM)

```csharp
using global::ILGPU;
using global::ILGPU.Runtime;
using SpawnDev.ILGPU;

// Register all available backends (browser + native)
using var context = await Context.CreateAsync(builder => builder.AllAcceleratorsAsync());

// Create the best available accelerator
using var accelerator = await context.CreatePreferredAcceleratorAsync();
```

### Desktop / Server (Console, WPF, etc.)

```csharp
using global::ILGPU;
using global::ILGPU.Runtime;
using SpawnDev.ILGPU;

// Register native backends (Cuda, OpenCL, CPU)
using var context = Context.Create(builder => builder.AllAccelerators());

// Pick the best device (Cuda > OpenCL > CPU)
using var accelerator = context.GetPreferredDevice(preferCPU: false)
    .CreateAccelerator(context);

// SpawnDev.ILGPU's async extensions work here too!
var kernel = accelerator.LoadAutoGroupedStreamKernel<...>(MyKernel);
kernel(extent, bufA.View, bufB.View, bufC.View);
await accelerator.SynchronizeAsync();  // Falls back to synchronous Synchronize()
var results = await bufC.CopyToHostAsync<float>();  // Falls back to CopyToCPU()
```

> **Cross-platform tip:** Use `SynchronizeAsync()` and `CopyToHostAsync()` everywhere. In the browser, they're truly async. On desktop, they gracefully fall back to synchronous ILGPU calls. Same code, both platforms.

## WebGPU Backend

The fastest backend. Uses GPU compute shaders via the WebGPU API, transpiling kernels to WGSL.

### Setup

```csharp
using ILGPU;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.WebGPU;

using var context = await Context.CreateAsync(builder => builder.WebGPU());
var devices = context.GetWebGPUDevices();

if (devices.Count > 0)
{
    using var accelerator = await devices[0].CreateAcceleratorAsync(context);
    // Use accelerator...
}
```

### Configuration Options

Use `WebGPUBackendOptions` to configure the transpiler:

```csharp
using SpawnDev.ILGPU.WebGPU.Backend;

var options = new WebGPUBackendOptions
{
    EnableF64Emulation = true,      // Default: true — emulate double precision
    UseOzakiF64Emulation = false,   // Default: false — use Dekker (faster) vs Ozaki (precise)
    EnableI64Emulation = true,      // Default: true — emulate long/ulong
};

using var accelerator = await devices[0].CreateAcceleratorAsync(context, options);
```

### Features

- **Compute shaders** — full GPU compute via `@compute @workgroup_size`
- **Shared memory** — `SharedMemory.Allocate<T>()` maps to `var<workgroup>`
- **Barriers** — `Group.Barrier()` maps to `workgroupBarrier()`
- **Atomics** — `Atomic.Add`, `Atomic.Min`, `Atomic.Max`, `Atomic.CompareExchange`
- **Subgroups** — `Group.Broadcast`, `Warp.Shuffle` (when the `subgroups` extension is available)
- **Auto-detected extensions** — probes adapter for `shader-f16`, `subgroups`, `timestamp-query`, etc.

### Browser Support

Chrome/Edge 113+, Firefox Nightly (with `dom.webgpu.enabled`).

## WebGL Backend

Universal GPU backend that works in virtually every modern browser. Transpiles kernels to GLSL ES 3.0 vertex shaders and uses Transform Feedback for GPU compute.

### Setup

```csharp
using ILGPU;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.WebGL;

using var context = await Context.CreateAsync(builder => builder.WebGL());
var devices = context.GetWebGLDevices();

if (devices.Count > 0)
{
    using var accelerator = devices[0].CreateAccelerator(context);
    // Use accelerator...
}
```

### Configuration Options

```csharp
using SpawnDev.ILGPU.WebGL.Backend;

var options = new WebGLBackendOptions
{
    EnableF64Emulation = true,   // Default: true
    EnableI64Emulation = true,   // Default: true
};

using var accelerator = devices[0].CreateAccelerator(context, options);
```

### Architecture

The WebGL backend is unique — all GL calls are dispatched to a dedicated Web Worker via `glWorker.js`. This keeps the main thread responsive even during intensive GPU compute.

### Limitations

- **No shared memory** — GLSL ES 3.0 vertex shaders don't support workgroup memory
- **No atomics** — not available in the vertex shader stage  
- **No barriers** — no workgroup synchronization

### Browser Support

All modern browsers — Chrome, Edge, Firefox, Safari, mobile browsers.

## Wasm Backend

Compiles kernels to native WebAssembly binary modules and dispatches them across Web Workers for parallel CPU execution.

### Setup

```csharp
using ILGPU;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.Wasm;

using var context = await Context.CreateAsync(builder => builder.Wasm());
var devices = context.GetDevices<WasmILGPUDevice>();

if (devices.Count > 0)
{
    using var accelerator = await devices[0].CreateAcceleratorAsync(context);
    // Use accelerator...
}
```

### Features

- **Multi-worker dispatch** — distributes work across all available CPU cores
- **Native 64-bit** — `double` and `long` work natively (no emulation needed)
- **Shared memory** — uses `SharedArrayBuffer` for zero-copy data sharing
- **Atomics** — supported via `SharedArrayBuffer`

### SharedArrayBuffer Requirement

For multi-worker mode, the page must be cross-origin isolated (COOP/COEP headers). The demo includes `coi-serviceworker.js` which handles this automatically. Without SharedArrayBuffer, the Wasm backend falls back to a single off-thread worker.

### Browser Support

All modern browsers that support Blazor WebAssembly.

## CPU Backend

Standard ILGPU CPU accelerator. Runs kernels synchronously on the main thread. Best for debugging and as a reference implementation.

### Setup

```csharp
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;

// CPU backend uses the synchronous API
using var context = Context.Create().CPU().ToContext();
using var accelerator = context.CreateCPUAccelerator(0);
```

### Limitations

- **Blocks the UI thread** — runs synchronously on the main thread
- **Barriers broken** in Blazor WASM single-threaded environment
- **Atomics crash** in Blazor WASM
- **Slowest backend** — single-threaded execution

Best used for debugging kernel logic.

## Desktop Backends (Cuda & OpenCL)

When running outside the browser (console apps, WPF, ASP.NET, etc.), SpawnDev.ILGPU uses ILGPU's native Cuda and OpenCL backends automatically. These are registered by the standard `builder.AllAccelerators()` call.

### Setup

```csharp
// Standard ILGPU context creation — works in any .NET app
using var context = Context.Create(builder => builder.AllAccelerators());

// Lists all detected devices
foreach (var device in context)
    Console.WriteLine($"{device.Name} ({device.AcceleratorType})");

// Pick the best GPU
using var accelerator = context.GetPreferredDevice(preferCPU: false)
    .CreateAccelerator(context);
```

### Cuda

- Requires an NVIDIA GPU with a supported driver
- Uses PTX intermediate representation
- Best performance for NVIDIA hardware
- Full ILGPU feature support (shared memory, atomics, warp ops)

### OpenCL

- Supports AMD and Intel GPUs (OpenCL 2.0+)
- Uses OpenCL C kernel language
- NVIDIA GPUs are limited to OpenCL 1.2 (not supported by ILGPU)

> **Note:** Cuda and OpenCL are not available in Blazor WebAssembly — they fail silently when the context builder tries to register them in the browser.

## 64-bit Emulation

GPU hardware typically supports only 32-bit operations. Both GPU backends provide software emulation for 64-bit types, **enabled by default**.

### `double` (f64) Emulation Schemes

| | **Dekker** (Default) | **Ozaki** |
|---|---|---|
| **Representation** | `vec2<f32>` (high + low) | `vec4<f32>` (quad-float) |
| **Precision** | ~48–53 bits mantissa | Strict IEEE 754 |
| **Memory** | 8 bytes | 16 bytes |
| **Performance** | ⚡ Faster | 🐢 ~2× slower |
| **Best for** | General compute, fractals | Scientific, financial |

### Configuration Examples

```csharp
// Default: Dekker double-float emulation
var options = new WebGPUBackendOptions();

// Strict IEEE 754 precision
var options = new WebGPUBackendOptions { UseOzakiF64Emulation = true };

// Disable emulation (double → float, max performance)
var options = new WebGPUBackendOptions { EnableF64Emulation = false };

// Disable all emulation
var options = new WebGPUBackendOptions { EnableF64Emulation = false, EnableI64Emulation = false };
```

## Runtime Backend Switching

You can switch backends at runtime by disposing old resources and creating new ones:

```csharp
// Dispose old resources
kernel = null;
outputBuffer?.Dispose();
accelerator?.Dispose();
context?.Dispose();

// Create new backend
context = await Context.CreateAsync(builder => builder.WebGL());
var devices = context.GetWebGLDevices();
accelerator = devices[0].CreateAccelerator(context);

// Reload kernel on new accelerator
kernel = accelerator.LoadAutoGroupedStreamKernel<...>(MyKernel);
outputBuffer = accelerator.Allocate1D<float>(length);
```

> **Important:** Kernels, buffers, and other resources are tied to their accelerator. You must recreate everything when switching backends.

## Verbose Logging

Enable debug logging per-backend:

```csharp
using SpawnDev.ILGPU.WebGPU.Backend;
using SpawnDev.ILGPU.WebGL.Backend;
using SpawnDev.ILGPU.Wasm.Backend;

WebGPUBackend.VerboseLogging = true;   // WebGPU
WebGLBackend.VerboseLogging = true;    // WebGL
WasmBackend.VerboseLogging = true;     // Wasm
```

This outputs compiled shader source, buffer binding details, and dispatch information to the browser console.

## Compiled Shader Inspection

After loading a kernel on a GPU backend, you can inspect the compiled shader source:

```csharp
// After kernel loads, the generated shader is available as a global JS variable
var wgslSource = JS.Get<string>("wgslDebug");  // WebGPU
var glslSource = JS.Get<string>("glslDebug");  // WebGL
```
