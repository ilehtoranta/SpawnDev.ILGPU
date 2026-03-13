# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Build the main library (~17s, ~1049 pre-existing CS1591 warnings)
dotnet build SpawnDev.ILGPU/SpawnDev.ILGPU.csproj

# Build the full solution
dotnet build SpawnDev.ILGPU.slnx

# Run desktop tests (CUDA, OpenCL, CPU backends)
dotnet run --project SpawnDev.ILGPU.ConsoleDemo

# Run browser tests — launch the Blazor WASM demo and navigate to /tests
dotnet run --project SpawnDev.ILGPU.Demo
```

Target framework is **net10.0**. `PublishTrimmed` and `RunAOTCompilation` must remain **false** — ILGPU relies on IL reflection at runtime.

## Architecture

SpawnDev.ILGPU extends ILGPU with three browser GPU backends. It transpiles .NET IL into GPU shader languages (WGSL, GLSL, WASM binary) at runtime.

### Backends (6 total)

| Backend | Target | Shader Language | Key Constraint |
|---------|--------|----------------|----------------|
| **WebGPU** | Browser | WGSL | 4-byte alignment required for all buffer ops |
| **WebGL** | Browser | GLSL ES 3.0 | No shared memory, atomics, or barriers; uses Transform Feedback |
| **Wasm** | Browser | WebAssembly binary | SharedArrayBuffer + multi-worker dispatch |
| CUDA | Desktop | PTX | Via upstream ILGPU |
| OpenCL | Desktop | OpenCL C | Via upstream ILGPU |
| CPU | Desktop | .NET | Via upstream ILGPU |

### 64-bit Emulation

WebGPU/WebGL lack native f64 and i64. The library emulates them:
- **f64**: Dekker (`vec2<f32>`) and Ozaki (`vec4<f32>`) double-float arithmetic
- **i64**: `vec2<u32>` paired 32-bit operations
- Emulation library inclusion is controlled by flags set in `SetEmulationFlags()` which scans kernel + inlined helper IR for Int64/Float64 usage

### Key Code Paths

- **WebGPU transpiler**: `SpawnDev.ILGPU/WebGPU/Backend/` — `WGSLKernelFunctionGenerator.cs` (main kernel codegen), `WGSLCodeGenerator.cs`, `WGSLEmulationLibrary.cs`
- **WebGL transpiler**: `SpawnDev.ILGPU/WebGL/Backend/` — `GLSLKernelFunctionGenerator.cs`, `GLSLEmulationLibrary.cs`, `glWorker.js` (off-main-thread Web Worker)
- **Wasm compiler**: `SpawnDev.ILGPU/Wasm/Backend/` — `WasmCodeGenerator.cs`, `WasmKernelFunctionGenerator.cs`
- **Accelerators**: `WebGPUAccelerator.cs`, `WebGLAccelerator.cs`, `WasmAccelerator.cs` — runtime dispatch and buffer management
- **Rendering**: `SpawnDev.ILGPU/Rendering/ICanvasRenderer.cs` — zero-copy canvas blitting abstraction

### Forked ILGPU

The `ILGPU/` and `ILGPU.Algorithms/` directories contain a modified fork of ILGPU, included as private asset references. Modifying ILGPU internals (e.g., changing `internal` to `public`, adding intrinsics) is expected and preferred over workarounds in SpawnDev.ILGPU.

**T4 Templates**: Many `.cs` files in `ILGPU/` are generated from `.tt` templates. Before editing any `.cs` file in `ILGPU/`, check for a matching `.tt` file — edits to generated files are silently overwritten on rebuild.

## Test Infrastructure

Unit tests live in `SpawnDev.ILGPU.Demo.Shared/UnitTests/`. The abstract base class `BackendTestBase` (split across `BackendTestBase.Tests1-9.cs` partial files) defines all tests. Backend-specific test classes inherit from it:

- **Desktop** (`SpawnDev.ILGPU.ConsoleDemo/`): `CudaTests`, `OpenCLTests`, `DesktopCPUTests` — run via `dotnet run --project SpawnDev.ILGPU.ConsoleDemo`
- **Browser** (`SpawnDev.ILGPU.Demo/`): `WebGPUTests`, `WebGLTests`, `WasmTests`, `DefaultTests` — run via the Blazor WASM demo app's `/tests` route

### PlaywrightMultiTest (unified test runner)

`PlaywrightMultiTest/` is an NUnit + Playwright .NET project that runs **all** tests (desktop and browser) in a single `dotnet test` invocation.

```bash
# Run all desktop + browser tests together (with timestamped results)
timestamp=$(date +%Y%m%d_%H%M%S) && dotnet test PlaywrightMultiTest/PlaywrightMultiTest.csproj --logger "trx;LogFileName=results_${timestamp}.trx" --results-directory PlaywrightMultiTest/TestResults
```

Test results are persisted as timestamped `.trx` files in `PlaywrightMultiTest/TestResults/` to track changes between runs.

**How it works:**
- `ProjectDiscovery` scans the workspace for projects containing a `<PlaywrightMultiTest>` element in their `.csproj`
- **Blazor WASM projects**: publishes the app, starts an HTTPS static file server, launches Chromium via Playwright (with `--enable-unsafe-webgpu`), navigates to the test page, and enumerates tests from the DOM
- **Console/Exe projects**: publishes the app, runs the binary to enumerate tests, then runs each test individually as a subprocess
- All discovered tests are surfaced as NUnit `TestCaseSource` entries, so standard NUnit filtering works (`--filter`)
- The old `PlaywrightTestRunner/` project is deprecated and replaced by this

## Engineering Philosophy

**Treat performance as a first-class concern alongside correctness.** Correctness is non-negotiable, but performance is a close second — kernels may be dispatched thousands of times per second for real-time data processing. Every code path, generated shader, and runtime decision must be evaluated for both.

- **No workarounds that mask underlying problems.** If a bug exists, fix the root cause. Workarounds hide issues that will surface elsewhere under different conditions.
- **Quality, long-term code only.** Every change must handle edge cases robustly. Prefer solutions that work for all inputs over ones that work for common inputs.
- **Understand the runtime cost model.** Shader modules are cached after first compilation — WGSL/GLSL size affects first-dispatch latency, not steady-state throughput. The GPU shader compiler (Tint/naga/driver) performs its own optimization. The transpiler's job is to produce correct, reasonably-sized shader code and let the GPU compiler handle instruction-level optimization.
- **Generated shader quality matters.** Minimize bloat in generated WGSL/GLSL: deduplicate emulation libraries, hoist constants, eliminate dead variables. Cleaner shaders compile faster on first dispatch and are dramatically easier to debug.
- **Transpiler efficiency matters for dynamic compilation.** When kernels are compiled with varying type parameters at runtime, C# transpiler performance directly impacts user-visible latency. Avoid regex in hot paths, pool allocations, pre-compute what can be pre-computed.

## Critical Constraints

- **Blazor WASM is single-threaded** — all async, no blocking calls
- **WebGPU 4-byte alignment** — buffer sizes, writeBuffer, copyBufferToBuffer, bind group entry sizes must all use `WebGPUAlignment.AlignTo4()`
- **WebGPU batched submission** — `WebGPUStream` uses deferred encode + flush pattern
- **Cross-backend impact** — changes to shared code (especially in `ILGPU/`) affect all 6 backends; consider all of them before modifying
- **No quick fixes** — every change must be well thought out; for complex fixes, plan before implementing
- **Do not hardcode hardware limits that are evolving** — e.g., WebGPU's 4GB buffer limit is being lifted; preserve full i64 index paths rather than assuming 32-bit indices are always sufficient
- **Emulation paths must remain functional** — i64/u64 emulation cannot be disabled because ILGPU's IR uses Int64 for ArrayView.Length and indices; the choice is which emulation method to use (Dekker vs Ozaki for f64), not whether to emulate
