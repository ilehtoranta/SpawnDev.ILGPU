# SpawnDev.ILGPU

[![NuGet](https://img.shields.io/nuget/v/SpawnDev.ILGPU.svg?)](https://www.nuget.org/packages/SpawnDev.ILGPU)

**Run [ILGPU](https://github.com/m4rs-mt/ILGPU) kernels in the browser — on the GPU via WebGPU, natively via WebAssembly, across threads via Web Workers, or on the CPU.**  
Write parallel compute code in C# and let the library pick the best available backend automatically.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Your C# ILGPU Kernel                      │
├──────────────┬──────────────┬──────────────┬─────────────────┤
│   WebGPU     │     Wasm     │   Workers    │      CPU        │
│   Backend    │   Backend    │   Backend    │    Backend      │
├──────────────┼──────────────┼──────────────┼─────────────────┤
│ WGSL         │ WebAssembly  │ JavaScript   │ .NET runtime    │
│ transpile    │ binary       │ transpile +  │ (main thread,   │
│ → GPU        │ → Workers    │ Web Workers  │  blocking)      │
└──────────────┴──────────────┴──────────────┴─────────────────┘
```

## Backends at a Glance

| | 🎮 **WebGPU** | 🧊 **Wasm** | 🧵 **Workers** | 💻 **CPU** |
|---|---|---|---|---|
| **Executes on** | GPU | Web Workers | Web Workers | Main (UI) thread |
| **Transpiles to** | WGSL | WebAssembly binary | JavaScript | — (interpreted) |
| **Threading** | GPU parallelism | Multi-worker | Multi-worker | Single-threaded |
| **Blocking** | Non-blocking | Non-blocking | Non-blocking | ⚠️ Blocks UI thread |
| **SharedArrayBuffer** | Not required | Required for multi-worker | Required for multi-worker | Not required |
| **Performance** | ⚡⚡⚡ Fastest | ⚡⚡ Fast | ⚡ Moderate | 🐢 Slowest |
| **Shared Memory** | ✅ | ✅ | ✅ | ⚠️ Barriers broken in WASM |
| **Atomics** | ✅ | ✅ | ✅ | ⚠️ Crashes in WASM |
| **64-bit (f64/i64)** | ✅ Emulated | ✅ Native | ✅ Native | ✅ Native |
| **Browser support** | Chrome/Edge 113+ | All modern browsers | All modern browsers | All modern browsers |
| **Best for** | GPU compute, rendering | General compute | CPU-bound parallel work | Fallback / debugging |

**Auto-selection priority:** WebGPU → Wasm → Workers → CPU

## Features

- **Four backends** — WebGPU (GPU compute), Wasm (native WebAssembly), Workers (multi-threaded JS), and CPU (fallback)
- **Automatic backend selection** — `CreatePreferredAcceleratorAsync()` picks the best available
- **ILGPU-compatible** — Use familiar APIs (`ArrayView`, `Index1D/2D/3D`, math intrinsics, etc.)
- **WGSL transpilation** — C# kernels automatically compiled to WebGPU Shading Language
- **Wasm compilation** — C# kernels compiled to native WebAssembly binary modules
- **64-bit emulation** — Support for `double` (f64) and `long` (i64) via emulated WGSL logic
- **WebGPU extension auto-detection** — Probes adapter for `shader-f16`, `subgroups`, `timestamp-query`, and other features; conditionally enables them on the device
- **Subgroup operations** — `Group.Broadcast` and `Warp.Shuffle` are supported on the WebGPU backend when the browser supports the `subgroups` extension
- **Multi-worker dispatch** — Wasm and Workers backends distribute work across all available CPU cores via SharedArrayBuffer
- **Blazor WebAssembly** — Seamless integration via [SpawnDev.BlazorJS](https://github.com/LostBeard/SpawnDev.BlazorJS)
- **Shared memory & atomics** — Supports workgroup memory, barriers, and atomic operations
- **No native dependencies** — Entirely written in C#

## Installation

```bash
dotnet add package SpawnDev.ILGPU
```

### 1. Configure Program.cs

SpawnDev.ILGPU requires [SpawnDev.BlazorJS](https://github.com/LostBeard/SpawnDev.BlazorJS) for browser interop.

```csharp
using SpawnDev.BlazorJS;

var builder = WebAssemblyHostBuilder.CreateDefault(args);
builder.RootComponents.Add<App>("#app");
builder.RootComponents.Add<HeadOutlet>("head::after");

// Add BlazorJS services
builder.Services.AddBlazorJSRuntime();

await builder.Build().BlazorJSRunAsync();
```

### 2. Quick Start — Automatic Backend Selection

The simplest way to use SpawnDev.ILGPU is with automatic backend selection. The library discovers all available backends and picks the best one:

```csharp
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

// Initialize context with all available backends (WebGPU, Wasm, Workers, CPU)
using var context = await Context.CreateAsync(builder => builder.AllAcceleratorsAsync());

// Create the best available accelerator (WebGPU > Wasm > Workers > CPU)
using var accelerator = await context.CreatePreferredAcceleratorAsync();

// Allocate buffers and run a kernel — same API regardless of backend
int length = 256;
using var bufA = accelerator.Allocate1D(Enumerable.Range(0, length).Select(i => (float)i).ToArray());
using var bufB = accelerator.Allocate1D(Enumerable.Range(0, length).Select(i => (float)i * 2f).ToArray());
using var bufC = accelerator.Allocate1D<float>(length);

var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(VectorAddKernel);
kernel((Index1D)length, bufA.View, bufB.View, bufC.View);

await accelerator.SynchronizeAsync();
var results = await bufC.CopyToHostAsync<float>();

// The kernel — runs on GPU, Wasm, workers, or CPU transparently
static void VectorAddKernel(Index1D index, ArrayView<float> a, ArrayView<float> b, ArrayView<float> c)
{
    c[index] = a[index] + b[index];
}
```

### 3. Using a Specific Backend

You can also target a specific backend directly using `Context.CreateAsync`:

```csharp
// WebGPU — GPU compute via WGSL
using var context = await Context.CreateAsync(builder => builder.WebGPU());
var device = context.GetWebGPUDevices()[0];
using var accelerator = await device.CreateAcceleratorAsync(context);
```

```csharp
// Wasm — native WebAssembly binary
using var context = await Context.CreateAsync(builder => builder.Wasm());
var device = context.GetDevices<WasmILGPUDevice>()[0];
using var accelerator = await device.CreateAcceleratorAsync(context);
```

```csharp
// Workers — multi-threaded JavaScript
using var context = await Context.CreateAsync(builder => builder.Workers());
var device = context.GetDevices<WorkersILGPUDevice>()[0];
using var accelerator = await device.CreateAcceleratorAsync(context);
```

```csharp
// CPU — single-threaded fallback (runs on main thread)
using var context = Context.Create().CPU().ToContext();
using var accelerator = context.CreateCPUAccelerator(0);
```

## Demo Application

The demo application is located in [SpawnDev.ILGPU.Demo](SpawnDev.ILGPU.Demo) and showcases:
- Automatic device detection across all backends
- Interactive Mandelbrot / Fractal Explorer (WebGPU, Wasm, Workers)
- Comprehensive unit test suites for WebGPU, Workers, Wasm, and CPU backends
- View the [Live Demo](https://lostbeard.github.io/SpawnDev.ILGPU/)

### Running the Demo

```bash
cd SpawnDev.ILGPU.Demo
dotnet run
```

Navigate to `https://localhost:5181` in a WebGPU-capable browser (Chrome, Edge, or Firefox Nightly).

## Testing

### Browser Tests
Start the demo app and navigate to `/tests` to run the unit test suite.

### Automated Tests (Playwright)
```bash
# Windows
_test.bat

# Linux/macOS
./_test.sh
```

## Test Coverage

**360+ tests** across four test suites covering all core features.

### Test Suites

| Suite | Backend | What's Tested |
|-------|---------|---------------|
| **WebGPUTests** | WebGPU | Full ILGPU feature set on GPU via WGSL |
| **WasmTests** | Wasm | Native WebAssembly binary dispatch to workers |
| **WorkerTests** | Workers | Multi-threaded JS dispatch, parity with WebGPU |
| **CPUTests** | CPU | ILGPU CPU accelerator as reference (barriers/atomics excluded) |
| **DefaultTests** | Auto | Device enumeration, preferred backend, kernel execution |

### Coverage by Area

| Area | What's Tested | Status |
|------|---------------|--------|
| **Memory** | Allocation, transfer, copy, views | ✅ |
| **Indexing** | 1D, 2D, 3D kernels, boundary conditions | ✅ |
| **Arithmetic** | +, -, *, /, %, negation, complex expressions | ✅ |
| **Bitwise** | AND, OR, XOR, NOT, shifts (<<, >>) | ✅ |
| **Math Functions** | sin, cos, tan, exp, log, sqrt, pow, abs, min, max | ✅ |
| **Atomics** | Add, Min, Max, CompareExchange, Xor | ✅ |
| **Control Flow** | if/else, loops, nested, short-circuit | ✅ |
| **Structs** | Simple, nested, with arrays | ✅ |
| **Type Casting** | float↔int, uint, mixed precision | ✅ |
| **64-bit Emulation** | `double` and `long` via software emulation (WebGPU) | ✅ |
| **GPU Patterns** | Stencil, reduction, matrix multiply, lerp, smoothstep | ✅ |
| **Shared Memory** | Static and dynamic workgroup memory | ✅ |
| **Broadcast & Subgroups** | `Group.Broadcast`, `Warp.Shuffle` (WebGPU with subgroups extension) | ✅ |
| **Special Values** | NaN, Infinity detection | ✅ |
| **Backend Selection** | Auto-discovery, priority, cross-backend kernel execution | ✅ |

## Browser Requirements

- Chrome 113+ (WebGPU + all backends)
- Edge 113+ (WebGPU + all backends)
- Firefox Nightly with `dom.webgpu.enabled` (WebGPU); stable Firefox (Wasm + Workers + CPU)

> **Note:** For multi-worker SharedArrayBuffer support (used by Wasm and Workers backends), the page must be cross-origin isolated (COOP/COEP headers). The demo includes a service worker (`coi-serviceworker.js`) that handles this automatically. Without SharedArrayBuffer, both backends fall back to single-worker mode.

## WebGPU Backend Configuration

### 64-bit Emulation

WebGPU hardware typically only supports 32-bit operations. SpawnDev.ILGPU provides software emulation for 64-bit types (`double`/f64 and `long`/i64), **enabled by default** for full precision parity with CPU and Workers backends.

To disable emulation for better performance (at the cost of precision):

```csharp
using SpawnDev.ILGPU.WebGPU.Backend;

var options = new WebGPUBackendOptions { EnableF64Emulation = false, EnableI64Emulation = false };
using var accelerator = await device.CreateAcceleratorAsync(context, options);
```

| Option | Default | Description |
|--------|---------|-------------|
| `EnableF64Emulation` | `true` | 64-bit float (`double`) emulation via `vec2<f32>` |
| `EnableI64Emulation` | `true` | 64-bit integer (`long`) emulation via `vec2<u32>` |

## Workers Backend

The Workers backend transpiles ILGPU kernels to JavaScript and executes them across Web Workers for multi-threaded CPU computation in the browser.

### Execution Modes

| Mode | When | Workers | Memory |
|------|------|---------|--------|
| **Parallel** | `WorkerCount > 1` + SharedArrayBuffer | N workers | Zero-copy via SAB |
| **Off-thread** | `WorkerCount == 1` or no SAB | 1 worker | Zero-copy via transfer |

### Worker Count

By default, the Workers backend uses all available logical cores (`navigator.hardwareConcurrency`). You can configure this:

```csharp
// Use 4 workers
var device = context.GetWorkersDevices()[0];
using var accelerator = await device.CreateWorkersAcceleratorAsync(workerCount: 4);
```

## Wasm Backend

The Wasm backend compiles ILGPU kernels to native WebAssembly binary modules and dispatches them to Web Workers for parallel execution. This provides near-native performance for compute-intensive workloads.

- Kernels are compiled to `.wasm` binary format (not text)
- Compiled modules are cached and reused across dispatches
- Shared memory uses `SharedArrayBuffer` for zero-copy data sharing

## Async Synchronization

In Blazor WebAssembly, the main thread cannot block. Use `SynchronizeAsync()` instead of `Synchronize()`:

```csharp
// ❌ Don't use — causes deadlock in Blazor WASM
accelerator.Synchronize();

// ✅ Use async version — works with all backends
await accelerator.SynchronizeAsync();
```

## Verbose Logging

All backends include verbose debug logging, disabled by default. Enable per-backend when needed:

```csharp
using SpawnDev.ILGPU.WebGPU.Backend;
using SpawnDev.ILGPU.Workers.Backend;
using SpawnDev.ILGPU.Wasm.Backend;

WebGPUBackend.VerboseLogging = true;   // WebGPU backend
WasmBackend.VerboseLogging = true;     // Wasm backend
WorkersBackend.VerboseLogging = true;  // Workers backend
```

## Blazor WebAssembly Configuration

When publishing, specific MSBuild properties are required:

```xml
<PropertyGroup>
  <!-- Disable IL trimming to preserve ILGPU kernel methods and reflection metadata -->
  <PublishTrimmed>false</PublishTrimmed>
  <!-- Disable AOT compilation - ILGPU requires IL reflection -->
  <RunAOTCompilation>false</RunAOTCompilation>
</PropertyGroup>
```

## License

This project is licensed under the same terms as ILGPU. See [LICENSE](LICENSE) for details.

## Credits

SpawnDev.ILGPU is built upon the excellent [ILGPU](https://github.com/m4rs-mt/ILGPU) library. We would like to thank the original authors and contributors of ILGPU for their hard work in providing a high-performance, robust IL-to-GPU compiler for the .NET ecosystem.

- **ILGPU Project:** [https://github.com/m4rs-mt/ILGPU](https://github.com/m4rs-mt/ILGPU)
- **ILGPU Authors:** [Marcel Müller](https://github.com/m4rs-mt) and the [ILGPU contributors](https://github.com/m4rs-mt/ILGPU/graphs/contributors)

## Resources

- [ILGPU Documentation](https://ilgpu.net/)
- [WebGPU Specification](https://www.w3.org/TR/webgpu/)
- [SpawnDev.BlazorJS](https://github.com/LostBeard/SpawnDev.BlazorJS)
- [GitHub Repository](https://github.com/LostBeard/SpawnDev.ILGPU)
