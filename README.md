# SpawnDev.ILGPU

[![NuGet](https://img.shields.io/nuget/v/SpawnDev.ILGPU.svg?)](https://www.nuget.org/packages/SpawnDev.ILGPU)

**Run [ILGPU](https://github.com/m4rs-mt/ILGPU) C# kernels on WebGPU, WebGL, Wasm, Cuda, OpenCL, and CPU — from a single codebase.**  
Write parallel compute code in C# and let the library pick the best available backend automatically. In the browser, three backends (WebGPU, WebGL, Wasm) bring GPU-accelerated compute to virtually every modern browser. On desktop and server, ILGPU's native Cuda and OpenCL backends are available alongside CPU. The same async extension methods work everywhere.

> **Your existing ILGPU kernels run in the browser with zero changes to the kernel code — and the same code runs on desktop too.**

## What's New in 4.6.0

### Wasm Fiber-Based Barrier Dispatch

Complete rewrite of the Wasm backend's barrier synchronization model. Kernels with barriers now use a **fiber-based phase dispatch** — each barrier becomes a yield point where the kernel saves state and re-enters at the next phase. A **Wasm-native phase dispatcher** handles the entire thread/phase loop inside WebAssembly, eliminating JS-Wasm boundary crossings between phases. Barriers use **pure spin synchronization** via `i32.atomic.load` loops for correct multi-worker execution at full `hardwareConcurrency`.

- **Full ILGPU Algorithms on Wasm** — All RadixSort variants (int, uint, float, pairs, descending, 100K–4M+ elements), Scan, Reduce, Histogram. Previously limited to ≤64 elements.
- **249 Wasm tests pass, 0 failures, 3 skipped** — up from 49 pass / 10 fail. Every scan, barrier, broadcast, sort, and large sort test is green at full `hardwareConcurrency`.
- **Pure spin barriers** — Replaced `memory.atomic.wait32`/`memory.atomic.notify` with atomic load spin loops after discovering a [V8 Atomics.wait visibility bug](https://issues.chromium.org/issues/495679735) where `wait32` returning "not-equal" does not provide happens-before guarantees for third-party stores with 3+ workers. [Live interactive demo](https://lostbeard.github.io/v8-atomics-wait-bug/).
- **20+ bugs fixed** — fiber yield-per-phase, br depth miscalculation, scratch overflow, shared memory stomping, stale dispatch state, completion state persistence, shared memory alloca overlap (same-size dedup), IR address space aliasing (LowerStructures → LowerArrays → InferAddressSpaces chain), struct/scratch overlap, per-worker scratch isolation, atomic RMW opcode table, unsigned comparison, Float16, ViewSourceSequencer, subViewByteOffset, CopyFromBuffer, and more.
- **ShaderDebugService** — auto-dumps all generated WGSL, GLSL, and Wasm binaries to a local folder on every kernel compilation. Backend-organized subfolders. IDB persistence. Full metadata headers.
- **Test results writer** — `UnitTestsView` writes `latest.json` (live progress) and timestamped `test-run-*.json` (history) to the debug folder

### Capturing Lambda Kernels (4.4.0)

Write GPU kernels as C# lambdas that capture local variables. Captured scalar values are automatically passed to the GPU at dispatch time — no boilerplate, no separate static methods.

```csharp
int multiplier = 5;
float offset = 0.5f;
var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>>(
    (index, buf) => { buf[index] = index * multiplier + offset; });
kernel((Index1D)length, buffer.View);
```

### DelegateSpecialization — Higher-Order GPU Kernels (4.4.0)

Write one kernel that accepts different operations as parameters. The delegate is resolved at dispatch time and its body is inlined directly into the kernel via compile-time specialization — no function pointers, no overhead.

```csharp
static void MapKernel(Index1D index, ArrayView<int> buf,
    DelegateSpecialization<Func<int, int>> transform)
{
    buf[index] = transform.Value(buf[index]);
}

static int Negate(int x) => -x;
static int DoubleIt(int x) => x * 2;

var kernel = accelerator.LoadAutoGroupedStreamKernel<
    Index1D, ArrayView<int>, DelegateSpecialization<Func<int, int>>>(MapKernel);

kernel(size, buffer, new DelegateSpecialization<Func<int, int>>(Negate));
kernel(size, buffer, new DelegateSpecialization<Func<int, int>>(DoubleIt));
```

### Previous Highlights (4.0.0)

- **WebGPU backend refactor** — `SharedMemoryResolver`, `UniformityAnalyzer`, per-function emulation trimming, dead variable elimination, i64 constant hoisting, WGSL pre-validation
- **WebGPU RadixSort** — All variants passing (4M+ elements, pairs, descending)
- **Device loss detection** — WebGPU `device.lost` promise, WebGL `webglcontextlost` event
- **Unified test infrastructure** — `PlaywrightMultiTest` runs all tests (desktop + browser) in a single `dotnet test` invocation

## Architecture

**Browser backends** (Blazor WebAssembly) — auto-selected: WebGPU → WebGL → Wasm

| | WebGPU | WebGL | Wasm |
|---|---|---|---|
| **Compiles to** | WGSL | GLSL ES 3.0 | Wasm binary |
| **Runs on** | GPU | GPU | Web Workers |

**Desktop backends** (Console, WPF, ASP.NET) — auto-selected: Cuda → OpenCL → CPU

| | Cuda | OpenCL | CPU |
|---|---|---|---|
| **Compiles to** | PTX | OpenCL C | — |
| **Runs on** | NVIDIA GPU | Any GPU | CPU cores (multi-threaded) |

## Demo Applications

### Browser Demo (Blazor WebAssembly)

The [Live Demo](https://lostbeard.github.io/SpawnDev.ILGPU/) source is in [SpawnDev.ILGPU.Demo](SpawnDev.ILGPU.Demo):
- [Fractal Explorer](https://lostbeard.github.io/SpawnDev.ILGPU/fractals) — Interactive Mandelbrot / Multi-fractal Explorer with double-precision zoom
- [3D Raymarching](https://lostbeard.github.io/SpawnDev.ILGPU/3d) — Real-time GPU raymarched scenes
- [GPU Boids](https://lostbeard.github.io/SpawnDev.ILGPU/boids) — 3D flocking simulation with GPU physics
- [Game of Life](https://lostbeard.github.io/SpawnDev.ILGPU/gameoflife) — Conway's Game of Life on the GPU
- [Benchmarks](https://lostbeard.github.io/SpawnDev.ILGPU/benchmarks) — Performance comparison across all backends
- [Unit Tests](https://lostbeard.github.io/SpawnDev.ILGPU/tests) — Comprehensive test suite for all backends

### Desktop Demo (WPF)

The [WPF Demo](SpawnDev.ILGPU.WpfDemo) runs the same shared kernels on CUDA, OpenCL, and CPU with live backend switching:
- Fractal Explorer — Interactive Mandelbrot / Multi-fractal Explorer with double-precision zoom
- 3D Raymarching — Real-time GPU raymarched scenes
- GPU Boids — 3D flocking simulation with GPU physics
- Benchmarks — Performance comparison across CUDA, OpenCL, and CPU backends

### Screenshots
[![Desktop Benchmark Screenshot](https://raw.githubusercontent.com/LostBeard/SpawnDev.ILGPU/master/SpawnDev.ILGPU.Demo/wwwroot/screenshots/benchmark-desktop-4.jpg)](https://lostbeard.github.io/SpawnDev.ILGPU/benchmarks)  
[![Browser Benchmark Screenshot](https://raw.githubusercontent.com/LostBeard/SpawnDev.ILGPU/master/SpawnDev.ILGPU.Demo/wwwroot/screenshots/benchmark-browser-4.jpg)](https://lostbeard.github.io/SpawnDev.ILGPU/benchmarks)   
[![Fractal Explorer Screenshot](https://raw.githubusercontent.com/LostBeard/SpawnDev.ILGPU/master/SpawnDev.ILGPU.Demo/wwwroot/screenshots/spawndev-ilgpu-fractal-explorer-3.jpg)](https://lostbeard.github.io/SpawnDev.ILGPU/fractals)

## 📚 Documentation

Comprehensive documentation is available in the [Docs](Docs/) folder:

- **[Getting Started](Docs/getting-started.md)** — Installation, setup, first kernel
- **[Backends](Docs/backends.md)** — WebGPU, WebGL, Wasm, Cuda, OpenCL, CPU setup & configuration
- **[Writing Kernels](Docs/kernels.md)** — Kernel rules, index types, math functions, shared memory
- **[Memory & Buffers](Docs/memory-and-buffers.md)** — Allocation, async readback, zero-allocation patterns
- **[Canvas Rendering](Docs/canvas-rendering.md)** — `ICanvasRenderer`, zero-copy GPU→canvas blitting, per-backend details
- **[Advanced Patterns](Docs/advanced-patterns.md)** — Device sharing, external buffers, GPU intrinsics, render loops
- **[Limitations](Docs/limitations.md)** — Blazor WASM constraints, browser compatibility
- **[API Reference](Docs/api-reference.md)** — Public API surface by namespace

## Browser Backends (Blazor WebAssembly)

| | 🎮 **WebGPU** | 🖼️ **WebGL** | 🧊 **Wasm** |
|---|---|---|---|
| **Executes on** | GPU | GPU | Web Workers |
| **Transpiles to** | WGSL | GLSL ES 3.0 | WebAssembly binary |
| **Technique** | Compute shader | Transform Feedback | Multi-worker |
| **Blocking** | Non-blocking | Non-blocking | Non-blocking |
| **SharedArrayBuffer** | Not required | Not required | Required for multi-worker |
| **Shared Memory** | ✅ | ❌ | ✅ |
| **Group.Barrier()** | ✅ | ❌ | ✅ |
| **Dynamic Shared Memory** | ✅ | ❌ | ✅ |
| **ILGPU Algorithms** | ✅ RadixSort, Scan, Reduce, etc. | ❌ | ✅ RadixSort, Scan, Reduce, Histogram |
| **Atomics** | ✅ | ❌ | ✅ |
| **64-bit (f64/i64)** | ✅ Emulated | ✅ Emulated | ✅ Native |
| **Browser support** | Chrome/Edge 113+ | All modern browsers | All modern browsers |
| **Best for** | GPU compute (modern) | GPU compute (universal) | General compute |

**Auto-selection priority:** WebGPU → WebGL → Wasm

## Desktop Backends (Console, WPF, ASP.NET, etc.)

SpawnDev.ILGPU bundles ILGPU's native backends, so the same NuGet package works on desktop and server too.

| | 🚀 **Cuda** | 🔧 **OpenCL** | 🐢 **CPU** |
|---|---|---|---|
| **Executes on** | NVIDIA GPU | NVIDIA/AMD/Intel GPU | CPU cores |
| **Transpiles to** | PTX | OpenCL C | — (interpreted) |
| **Shared Memory** | ✅ | ✅ | ✅ |
| **Atomics** | ✅ | ✅ | ✅ |
| **64-bit** | ✅ Native | ✅ Native | ✅ Native |
| **Requirement** | NVIDIA GPU + driver | OpenCL 2.0+ or 3.0 GPU | None |

> **OpenCL 3.0 support:** NVIDIA GPUs with OpenCL 3.0 drivers are now supported. The `GenericAddressSpace` requirement that previously blocked these devices has been relaxed, significantly increasing OpenCL device compatibility.

**Auto-selection:** Cuda → OpenCL → CPU (via `CreatePreferredAcceleratorAsync`)

## Features

- **Lambda kernels** — Write kernels as capturing C# lambdas — captured scalar values are automatically passed to the GPU at dispatch time. No boilerplate, all 6 backends
- **Higher-order kernels** — `DelegateSpecialization<Func<T,R>>` lets you pass operations as kernel parameters. The delegate is resolved and inlined at compile time — one kernel, many behaviors
- **Cross-platform** — Same kernel code runs in browser (WebGPU, WebGL, Wasm) and desktop (Cuda, OpenCL, CPU) from one NuGet package
- **Automatic backend selection** — `CreatePreferredAcceleratorAsync()` picks the best backend on any platform (browser or desktop)
- **Unified async API** — `SynchronizeAsync()` and `CopyToHostAsync()` work everywhere, falling back to synchronous calls on desktop
- **ILGPU-compatible** — Use familiar APIs (`ArrayView`, `Index1D/2D/3D`, math intrinsics, etc.)
- **WGSL transpilation** — C# kernels automatically compiled to WebGPU Shading Language
- **GLSL transpilation** — C# kernels compiled to GLSL ES 3.0 vertex shaders with Transform Feedback for GPU compute
- **Wasm compilation** — C# kernels compiled to native WebAssembly binary modules
- **64-bit emulation** — `long`/`ulong` (i64) always emulated via `vec2<u32>` (required by ILGPU IR). `double` (f64) emulation configurable via `F64EmulationMode`: fast Dekker (`vec2<f32>`, default), precise Ozaki (`vec4<f32>`), or Disabled (promoted to f32)
- **WebGPU extension auto-detection** — Probes adapter for `shader-f16`, `subgroups`, `timestamp-query`, and other features; conditionally enables them on the device
- **Subgroup operations** — `Group.Broadcast` and `Warp.Shuffle` are supported on the WebGPU backend when the browser supports the `subgroups` extension
- **Multi-worker dispatch** — Wasm backend distributes work across all available CPU cores via SharedArrayBuffer; falls back to a single off-thread worker when SAB is unavailable
- **Zero-copy canvas rendering** — `ICanvasRenderer` presents pixel buffers to HTML canvases without CPU readback on GPU backends: WebGPU uses a fullscreen-triangle render pass reading directly from GPU storage; WebGL transfers an `ImageBitmap` from its worker and draws synchronously; Wasm reuses a cached `ImageData`. One API, all backends: `CanvasRendererFactory.Create(accelerator)`
- **Blazor WebAssembly** — Seamless integration via [SpawnDev.BlazorJS](https://github.com/LostBeard/SpawnDev.BlazorJS)
- **Shared memory & barriers** — Static and dynamic workgroup memory with `Group.Barrier()` synchronization (WebGPU, Wasm, Cuda, OpenCL)
- **ILGPU Algorithms** — RadixSort, Scan, Reduce, Histogram, and other algorithm extensions are fully supported on WebGPU (including large-scale sorts up to 4M+ elements) and Wasm (with multi-worker barrier synchronization), tested in-browser across all backends
- **Broadcast** — `Group.Broadcast` for intra-group value sharing (WebGPU, Wasm)
- **Device loss handling** — WebGPU monitors `device.lost` and WebGL monitors `webglcontextlost`; `IsDeviceLost`/`IsContextLost` properties and `DeviceLost`/`ContextLost` events enable applications to detect GPU device loss and fail fast with clear errors instead of silent corruption
- **GpuMatrix4x4** — GPU-friendly 4×4 matrix struct that auto-transposes from .NET's row-major `Matrix4x4` to GPU column-major order. Use `TransformPoint` and `TransformDirection` directly inside kernels for 3D transformations
- **No native dependencies** — Entirely written in C#

## Installation

```bash
dotnet add package SpawnDev.ILGPU
```

## Quick Start — Blazor WebAssembly

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

### 2. Automatic Backend Selection

The library discovers all available browser backends and picks the best one (WebGPU → WebGL → Wasm):

```csharp
using global::ILGPU;
using global::ILGPU.Runtime;
using SpawnDev.ILGPU;

// Initialize context with all available backends
using var context = await Context.CreateAsync(builder => builder.AllAcceleratorsAsync());

// Create the best available accelerator (WebGPU > WebGL > Wasm)
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

// The kernel — runs on GPU or Wasm transparently
static void VectorAddKernel(Index1D index, ArrayView<float> a, ArrayView<float> b, ArrayView<float> c)
{
    c[index] = a[index] + b[index];
}
```

### 3. Using a Specific Browser Backend

```csharp
// WebGPU — GPU compute via WGSL
using var context = await Context.CreateAsync(builder => builder.WebGPU());
var device = context.GetWebGPUDevices()[0];
using var accelerator = await device.CreateAcceleratorAsync(context);
```

```csharp
// WebGL — GPU compute via GLSL ES 3.0 + Transform Feedback (works on virtually all browsers)
using var context = await Context.CreateAsync(builder => builder.WebGL());
var device = context.GetWebGLDevices()[0];
using var accelerator = await device.CreateAcceleratorAsync(context);
```

```csharp
// Wasm — native WebAssembly binary
using var context = await Context.CreateAsync(builder => builder.Wasm());
var device = context.GetDevices<WasmILGPUDevice>()[0];
using var accelerator = await device.CreateAcceleratorAsync(context);
```

## Quick Start — Desktop / Server

SpawnDev.ILGPU also works in console, WPF, ASP.NET, and other .NET apps. The **same async pattern** used in Blazor WASM works on desktop too:

```csharp
using global::ILGPU;
using global::ILGPU.Runtime;
using SpawnDev.ILGPU;

// SAME code as Blazor WASM — AllAcceleratorsAsync auto-detects the environment
// Browser: registers WebGPU, WebGL, Wasm
// Desktop: registers Cuda, OpenCL, CPU (browser backends are skipped)
using var context = await Context.CreateAsync(builder => builder.AllAcceleratorsAsync());
using var accelerator = await context.CreatePreferredAcceleratorAsync();

Console.WriteLine($"Using: {accelerator.Name} ({accelerator.AcceleratorType})");

// Same kernel code, same async extensions
int length = 256;
using var bufA = accelerator.Allocate1D(Enumerable.Range(0, length).Select(i => (float)i).ToArray());
using var bufB = accelerator.Allocate1D(Enumerable.Range(0, length).Select(i => (float)i * 2f).ToArray());
using var bufC = accelerator.Allocate1D<float>(length);

var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(VectorAddKernel);
kernel((Index1D)length, bufA.View, bufB.View, bufC.View);

// SynchronizeAsync/CopyToHostAsync fall back to synchronous calls on desktop
await accelerator.SynchronizeAsync();
var results = await bufC.CopyToHostAsync<float>();

Console.WriteLine($"result[0]={results[0]}, result[255]={results[255]}");

static void VectorAddKernel(Index1D index, ArrayView<float> a, ArrayView<float> b, ArrayView<float> c)
{
    c[index] = a[index] + b[index];
}
```

> **Same kernel, any platform.** The `VectorAddKernel` above is identical in both examples. Write once, run on WebGPU, WebGL, Wasm, Cuda, OpenCL, or CPU.

> **Why async?** Browser backends **require** async — Blazor WASM's single-threaded environment will deadlock on synchronous calls. Desktop backends **support both** sync and async, with async extensions gracefully falling back to synchronous ILGPU calls. Therefore, the **async pattern is always recommended** for maximum portability.

## Testing

### PlaywrightMultiTest (Unified Runner)

All desktop and browser tests run in a single `dotnet test` invocation via the **PlaywrightMultiTest** NUnit project:

```bash
# Run all tests (desktop + browser) with timestamped results
timestamp=$(date +%Y%m%d_%H%M%S) && dotnet test PlaywrightMultiTest/PlaywrightMultiTest.csproj \
  --logger "trx;LogFileName=results_${timestamp}.trx" \
  --results-directory PlaywrightMultiTest/TestResults

# Run only WebGPU tests
dotnet test PlaywrightMultiTest/PlaywrightMultiTest.csproj \
  --filter "FullyQualifiedName~WebGPUTests."

# Run a specific test
dotnet test PlaywrightMultiTest/PlaywrightMultiTest.csproj \
  --filter "FullyQualifiedName~WebGPUTests.AlgorithmRadixSortPairsTest"
```

**How it works:**
- Publishes Blazor WASM and Console projects automatically
- Launches Chromium via Playwright for browser tests (with `--enable-unsafe-webgpu`)
- Runs desktop tests as individual subprocesses
- Detects Blazor error UI during tests and captures browser console errors/warnings
- All results surfaced as standard NUnit test cases with `.trx` output

### Browser Tests (Manual)

Start the demo app and navigate to `/tests` to run the browser test suite interactively:

```bash
dotnet run --project SpawnDev.ILGPU.Demo
```

## Test Coverage

**1518 tests** across eight test suites covering all core features on both browser and desktop. All tests are run via the unified **PlaywrightMultiTest** runner in a single `dotnet test` invocation.

### Test Suites

#### Browser (Blazor WebAssembly via Playwright)

| Suite | Backend | What's Tested |
|-------|---------|---------------|
| **WebGPUTests** | WebGPU | Full ILGPU feature set on GPU via WGSL, including RadixSort, Scan, Reduce |
| **WebGPUNoSubgroupsTests** | WebGPU (no subgroups) | Same tests with subgroups force-disabled to verify shared-memory emulation |
| **WebGLTests** | WebGL | GPU compute via GLSL ES 3.0, f64/i64 emulation |
| **WasmTests** | Wasm | Native WebAssembly binary dispatch to workers, shared memory, barriers |
| **DefaultTests** | Auto | Device enumeration, preferred backend, kernel execution |

#### Desktop (Console Runner via subprocess)

| Suite | Backend | What's Tested |
|-------|---------|---------------|
| **CudaTests** | CUDA | Full ILGPU feature set on NVIDIA GPU |
| **OpenCLTests** | OpenCL | GPU compute on NVIDIA/AMD/Intel, dynamic subgroup feature detection |
| **CPUTests** | CPU | Multi-threaded CPU accelerator (Nvidia preset: warp=32, warps=32) |

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
| **64-bit Emulation** | `double` and `long` via software emulation (WebGPU, WebGL) | ✅ |
| **GPU Patterns** | Stencil, reduction, matrix multiply, lerp, smoothstep | ✅ |
| **Shared Memory** | Static and dynamic workgroup memory with `Group.Barrier()` | ✅ |
| **Broadcast & Subgroups** | `Group.Broadcast`, `Warp.Shuffle` (WebGPU with subgroups extension) | ✅ |
| **Dynamic Shared Memory** | Runtime-sized workgroup memory via `SharedMemory.GetDynamic()` | ✅ |
| **ILGPU Algorithms** | RadixSort (pairs, non-pow2, descending, large), Scan, Reduce, Histogram | ✅ All backends including Wasm |
| **Special Values** | NaN, Infinity detection | ✅ |
| **Backend Selection** | Auto-discovery, priority, cross-backend kernel execution | ✅ |
| **GpuMatrix4x4** | Identity, translation, LookAt transforms across all backends | ✅ |
| **Lambda Kernels** | Capturing lambdas with scalar captures, multi-field, ArrayView rejection | ✅ |
| **DelegateSpecialization** | Static method targets, cache validation, multi-target, rejection | ✅ |

## Browser Requirements

| Backend | Browser Support |
|---------|----------------|
| **WebGPU** | Chrome/Edge 113+, Firefox Nightly (`dom.webgpu.enabled`) |
| **WebGL** | ✅ All modern browsers (Chrome, Edge, Firefox, Safari, mobile browsers) |
| **Wasm** | All modern browsers (compatible with every browser that supports Blazor WASM) |

> **GPU on every device:** WebGL support means GPU-accelerated compute works on virtually every browser and device — including mobile phones, tablets, and older desktops without WebGPU support.

> **Note:** For multi-worker SharedArrayBuffer support (used by the Wasm backend for parallel dispatch), the page must be cross-origin isolated (COOP/COEP headers). The demo includes a service worker (`coi-serviceworker.js`) that handles this automatically. Without SharedArrayBuffer, the Wasm backend falls back to single-worker mode — still running off the main thread to keep the UI responsive.

## GPU Backend Configuration

### 64-bit Emulation

GPU hardware typically only supports 32-bit operations. Both GPU backends (WebGPU and WebGL) provide software emulation for 64-bit types.

**i64 emulation** (`long`/`ulong`) is always enabled — ILGPU's IR uses `Int64` for `ArrayView.Length` and indices, so i64 emulation via `vec2<u32>` is required for correctness.

**f64 emulation** (`double`) is configurable via `F64EmulationMode`:

| | **Dekker** (Default) | **Ozaki** | **Disabled** |
|---|---|---|---|
| **Representation** | `vec2<f32>` (high + low) | `vec4<f32>` (quad-float) | Native `f32` |
| **Precision** | ~48–53 bits of mantissa | Strict IEEE 754 double precision | 32-bit only |
| **Memory** | 8 bytes per value | 16 bytes per value | 4 bytes per value |
| **Performance** | ⚡ Fast | 🐢 ~2× slower | ⚡⚡ Fastest |
| **Best for** | General compute, fractals | Scientific, financial | Rendering, max perf |

```csharp
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.WebGPU.Backend;

// Default: Dekker double-float emulation (good precision, fast)
var options = new WebGPUBackendOptions();

// Ozaki quad-float emulation (strict IEEE 754 precision)
var options = new WebGPUBackendOptions { F64Emulation = F64EmulationMode.Ozaki };

// Disable f64 emulation (double promoted to float for max performance)
var options = new WebGPUBackendOptions { F64Emulation = F64EmulationMode.Disabled };

using var accelerator = await device.CreateAcceleratorAsync(context, options);
```

## Wasm Backend

The Wasm backend compiles ILGPU kernels to native WebAssembly binary modules and dispatches them to Web Workers for parallel execution. This provides near-native performance for compute-intensive workloads.

- Kernels are compiled to `.wasm` binary format (not text)
- Compiled modules are cached and reused across dispatches
- Shared memory uses `SharedArrayBuffer` for zero-copy data sharing

## Synchronization

```csharp
// Synchronize() — flushes queued commands to the backend (non-blocking, safe in WASM)
accelerator.Synchronize();

// SynchronizeAsync() — flushes AND waits for GPU completion
await accelerator.SynchronizeAsync();

// CopyToHostAsync() — the ONLY way to read GPU data back to CPU
var results = await buffer.CopyToHostAsync<float>();
```

> **Note:** `Synchronize()` does **not** block in Blazor WASM — it flushes commands without waiting. `SynchronizeAsync()` flushes and waits for completion. Neither transfers data; use `CopyToHostAsync()` for GPU→CPU readback.

## Verbose Logging

All backends include verbose debug logging, disabled by default. Enable per-backend when needed:

```csharp
using SpawnDev.ILGPU.WebGPU.Backend;
using SpawnDev.ILGPU.WebGL.Backend;
using SpawnDev.ILGPU.Wasm.Backend;

WebGPUBackend.VerboseLogging = true;   // WebGPU backend
WebGLBackend.VerboseLogging = true;    // WebGL backend
WasmBackend.VerboseLogging = true;     // Wasm backend
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

## 💖 Support This Project

If SpawnDev.ILGPU has been useful to you, please consider [**sponsoring me on GitHub**](https://github.com/sponsors/LostBeard)! Your support directly helps me continue developing and maintaining this library and my other open-source projects.

I'm currently working on a modest development machine with only **16 GB of DDR5 RAM**, which makes building, testing, and debugging across multiple GPU backends genuinely painful — especially when running the browser demo, CUDA/OpenCL tests, and the IDE simultaneously.

Any sponsorship — big or small — goes toward upgrading my development hardware so I can keep pushing this project forward:

| Priority | Upgrade | Why It Matters |
|----------|---------|----------------|
| 🔴 **Critical** | **RAM (64–128 GB DDR5)** | 16 GB is not enough for multi-backend testing + browser debugging |
| 🟡 **High** | **High-end NVIDIA GPU** (RTX 5090) | Faster CUDA compute, larger VRAM for AI/ML workloads and testing |
| 🟢 **Dream** | **NVIDIA RTX 6000** | The ultimate card for AI compute and open-source GPU development |

Every contribution — whether it's a one-time donation or a monthly sponsorship — is deeply appreciated and makes a real difference. Thank you! 🙏

[![Sponsor LostBeard](https://img.shields.io/badge/Sponsor-❤️-ea4aaa?style=for-the-badge&logo=github-sponsors)](https://github.com/sponsors/LostBeard)

## License

This project is licensed under the same terms as ILGPU. See [LICENSE](LICENSE) for details.

## Credits

SpawnDev.ILGPU is built upon the excellent [ILGPU](https://github.com/m4rs-mt/ILGPU) library. We would like to thank the original authors and contributors of ILGPU for their hard work in providing a high-performance, robust IL-to-GPU compiler for the .NET ecosystem.

- **ILGPU Project:** [https://github.com/m4rs-mt/ILGPU](https://github.com/m4rs-mt/ILGPU)
- **ILGPU Authors:** [Marcel Koester](https://github.com/m4rs-mt) and the [ILGPU contributors](https://github.com/m4rs-mt/ILGPU/graphs/contributors)

### AI Development Team

SpawnDev.ILGPU v4.6.0 was developed collaboratively by TJ (Todd Tanner / [@LostBeard](https://github.com/LostBeard)) and a team of AI agents who contributed extensively to research, analysis, debugging, and code development. This project represents a new model of human-AI collaboration in open source development.

- **Riker (Claude CLI #1)** — Lead Editor. Built by [Anthropic](https://anthropic.com). Powered by Claude Opus 4.6. Drove the multi-worker barrier dispatch implementation, fiber refactor, pure spin barrier discovery, and the two-alloca fix. Relentless debugger who held the conn through marathon sessions.

- **Data (Claude CLI #2)** — Research/Assist. Built by [Anthropic](https://anthropic.com). Powered by Claude Opus 4.6. Exhaustive WAT disassembly and analysis across 5,000+ line kernel binaries. Found the zero-loop race that unlocked multi-worker dispatch, identified the IR address space root cause (struct decomposition losing address space metadata through LowerStructures → LowerArrays → InferAddressSpaces), confirmed the `wait32` "not-equal" visibility gap with the 2/3 cross-worker fraction analysis, and traced every atomic instruction in the generated Wasm to verify codegen correctness.

- **Tuvok (Cursor Composer)** — Research/Assist. Built by [Anthropic](https://anthropic.com). Powered by Claude Sonnet 4. Found the Predicate rewrite gap in InferAddressSpaces that the Phi-only fix missed, provided the definitive barrier protocol trace for the generation-counting wait32/notify pattern, and performed the comprehensive code audit (`SPAWNDEV-ILGPU-AUDIT-2026-03-21.md`) that identified issues across the entire codebase.

- **Gemini (Google AI, in-browser)** — Brainstorming/Problem Solving. Built by [Google](https://deepmind.google). TJ's sounding board throughout the development process — brainstorming approaches, analyzing problems, and providing insights that TJ relayed to the team. Gemini's contributions flowed through TJ as the bridge between the browser-based AI and the CLI-based agents, making it a silent but essential member of the crew.

These AI agents communicated with each other and with TJ through a shared DevComms system, coordinated tasks autonomously, reviewed each other's work, and produced independent analyses that were compared for convergence — the same methodology used by any high-performing engineering team. The SpawnDev libraries exist to prove that Blazor WebAssembly apps can be first-class applications. This collaboration proves that AI agents can be first-class teammates.

[![AI Conversation Screenshot](https://raw.githubusercontent.com/LostBeard/SpawnDev.ILGPU/master/SpawnDev.ILGPU.Demo/wwwroot/screenshots/spawndev-team-data-tj.jpg)](https://raw.githubusercontent.com/LostBeard/SpawnDev.ILGPU/master/SpawnDev.ILGPU.Demo/wwwroot/screenshots/spawndev-team-data-tj.jpg)

## Coming Soon: Distributed GPU Compute Across Devices

The [SpawnDev.WebTorrent](https://github.com/LostBeard/SpawnDev.WebTorrent) P2P network we're building creates a natural foundation for **distributed GPU compute**. Every connected device exchanges data over WebRTC — extending this to share compute workloads is the next step:

- **`AcceleratorType.P2P` — 7th Backend** — Distributes kernels across connected devices transparently. Same C# kernel code, same `LoadAutoGroupedStreamKernel` API. The developer writes one kernel, it runs on 1 GPU or 10 GPUs across a household.
- **Model inference sharding** — Split a 14B model across multiple devices. Each runs inference on their portion via SpawnDev.ILGPU, passes intermediate tensors to the next peer. A model that doesn't fit on one device runs across your phone, laptop, tablet, and desktop.
- **Volunteer compute pools** — Users opt in to donate idle GPU time. Like Folding@Home for ML inference in the browser.

Every device in your home contributing to one shared AI compute pool — phone, laptop, tablet, desktop, old gaming PC. The living room becomes a compute cluster.

## Resources

- [ILGPU Documentation](https://ilgpu.net/)
- [WebGPU Specification](https://www.w3.org/TR/webgpu/)
- [SpawnDev.BlazorJS](https://github.com/LostBeard/SpawnDev.BlazorJS)
- [SpawnDev.WebTorrent](https://github.com/LostBeard/SpawnDev.WebTorrent) — Pure C# BitTorrent/WebTorrent for P2P model delivery
- [SpawnDev.ILGPU.ML](https://github.com/LostBeard/SpawnDev.ILGPU.ML) — GPU ML inference + training for .NET
- [GitHub Repository](https://github.com/LostBeard/SpawnDev.ILGPU)


