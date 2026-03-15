# CLAUDE.md — Master Map

SpawnDev.ILGPU extends ILGPU with three browser GPU backends. It transpiles .NET IL into GPU shader languages (WGSL, GLSL, Wasm binary) at runtime.

## Build Commands

```bash
dotnet build SpawnDev.ILGPU/SpawnDev.ILGPU.csproj   # Main library (~2s)
dotnet build SpawnDev.ILGPU.slnx                     # Full solution
dotnet run --project SpawnDev.ILGPU.DemoConsole       # Desktop tests (CUDA, OpenCL, CPU)
dotnet run --project SpawnDev.ILGPU.Demo              # Browser tests (Blazor WASM → /tests)
```

Target: **net10.0**. `PublishTrimmed` and `RunAOTCompilation` must remain **false** — ILGPU relies on IL reflection at runtime.

## Context Map

Detailed constraints live in each directory's own `CLAUDE.md`. Read the relevant one when working in that area.

| Directory | What | Context File |
|-----------|------|-------------|
| `SpawnDev.ILGPU/WebGPU/` | WGSL transpiler, dispatch, buffers | [`WebGPU/CLAUDE.md`](SpawnDev.ILGPU/WebGPU/CLAUDE.md) |
| `SpawnDev.ILGPU/Wasm/` | Wasm binary compiler, worker dispatch | [`Wasm/CLAUDE.md`](SpawnDev.ILGPU/Wasm/CLAUDE.md) |
| `SpawnDev.ILGPU/WebGL/` | GLSL transpiler, Transform Feedback | [`WebGL/CLAUDE.md`](SpawnDev.ILGPU/WebGL/CLAUDE.md) |
| `ILGPU/` | Forked ILGPU core (IR, types, runtime) | [`ILGPU/CLAUDE.md`](ILGPU/CLAUDE.md) |
| `ILGPU.Algorithms/` | Forked algorithms (Scan, RadixSort) | [`ILGPU.Algorithms/CLAUDE.md`](ILGPU.Algorithms/CLAUDE.md) |
| `PlaywrightMultiTest/` | Unified test runner | [`PlaywrightMultiTest/CLAUDE.md`](PlaywrightMultiTest/CLAUDE.md) |
| `.claude/skills/ilgpu_transpiler/` | Hard-won transpiler mapping rules | [SKILL.md](.claude/skills/ilgpu_transpiler/SKILL.md) |

## Architecture Overview

### Backends (6 total)

| Backend | Target | Shader Language | Key Constraint |
|---------|--------|----------------|----------------|
| **WebGPU** | Browser | WGSL | 4-byte alignment, uniformity analysis |
| **WebGL** | Browser | GLSL ES 3.0 | No shared memory/atomics/barriers |
| **Wasm** | Browser | WebAssembly binary | SharedArrayBuffer + multi-worker dispatch |
| CUDA | Desktop | PTX | Via upstream ILGPU |
| OpenCL | Desktop | OpenCL C | Via upstream ILGPU |
| CPU | Desktop | .NET | Via upstream ILGPU |

### Test Infrastructure

Tests in `SpawnDev.ILGPU.Demo.Shared/UnitTests/BackendTestBase*.cs` (~211 tests, Tests1-10). Backend-specific classes inherit and override unsupported tests. See `PlaywrightMultiTest/CLAUDE.md` for running tests.

## Engineering Philosophy

- **Correctness is non-negotiable. Performance is a close second.** Kernels dispatch thousands of times/sec.
- **No workarounds that mask problems.** Fix root causes.
- **Cross-backend impact** — changes to `ILGPU/` affect all 6 backends. Consider all of them.
- **No quick fixes** — plan before implementing complex changes.
- **Do not hardcode evolving hardware limits** — preserve full i64 index paths.

## Global Constraints

These apply everywhere, not just one directory:

- **Blazor WASM is single-threaded** — all async, no blocking calls
- **T4 Templates in `ILGPU/`** — check for `.tt` before editing `.cs`. Generated files are silently overwritten.
- **Device loss detection** — WebGPU: `device.lost` promise. WebGL: `webglcontextlost` event. Guards on dispatch/synchronize. Intentional disposal filtered out.
