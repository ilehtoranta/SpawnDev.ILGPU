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

The `PlaywrightTestRunner/` project exists but should **not** be used for now. Test browser backends by running the demo app and navigating to `/tests` manually.

## Critical Constraints

- **Blazor WASM is single-threaded** — all async, no blocking calls
- **WebGPU 4-byte alignment** — buffer sizes, writeBuffer, copyBufferToBuffer, bind group entry sizes must all use `WebGPUAlignment.AlignTo4()`
- **WebGPU batched submission** — `WebGPUStream` uses deferred encode + flush pattern
- **Cross-backend impact** — changes to shared code (especially in `ILGPU/`) affect all 6 backends; consider all of them before modifying
- **No quick fixes** — every change must be well thought out; for complex fixes, plan before implementing
