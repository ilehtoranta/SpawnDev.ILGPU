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

**Current results (March 2026):** Wasm: 249 pass / 0 fail / 3 skip (v4.6.0). Full `hardwareConcurrency` multi-worker barrier dispatch with pure spin barriers (i32.atomic.load loops) and in-Wasm phase dispatcher (no JS-Wasm boundary crossings between phases). All large sort tests (260K-4M) passing including SpawnSceneSimulation (1.4M elements, multi-frame). Key fixes: pure spin barrier (V8 wait32 visibility gap), shared memory alloca overlap (same-size dedup), IR address space aliasing (InferAddressSpaces guards), struct/scratch overlap, multi-pass scan, Float16, unsigned ops, 256 threads, memory.grow(), ViewSourceSequencer, subViewByteOffset, atomic RMW opcode table, CopyFromBuffer, onesComplementMask .tt template, per-worker scratch, atomic.fence at 3 sync points, float atomic stores, broadcast atomic store/load, barrier counter zeroing between groups.

## Debugging Pipeline — ShaderDebugService

Every kernel compilation auto-dumps generated code to a local folder via `ShaderDebugService` (registered in the demo's `Program.cs`). **Use this — do NOT ask TJ to manually run tests or capture output.**

### Setup
1. Run the demo, go to `/tests`
2. Click **"Set Debug Folder"** → pick a local folder (e.g., `_debugdump`)
3. Folder persists in IndexedDB across sessions — set once, works forever

### What Auto-Dumps (organized by backend)
```
debugfolder/
├── _DEBUG_README.md
├── latest.json                         ← live test results (updated each test)
├── test-run-YYYY-MM-DD_HH-mm-ss.json  ← permanent test run history
├── wgsl/                               ← WebGPU shaders with metadata headers
│   └── NNN_KernelName.wgsl
├── glsl/                               ← WebGL shaders with metadata headers
│   └── NNN_KernelName.glsl
└── wasm/                               ← Wasm binaries + compilation info
    ├── NNN_KernelName.wasm             ← disassemble: wasm2wat --enable-threads
    └── NNN_KernelName.txt              ← params, locals, barriers, shared mem size
```

### How to Use
- **Find a kernel:** Grep the `.txt` files for `hasBarriers=True`, `helpers=1`, etc.
- **Disassemble Wasm:** `wasm2wat --enable-threads NNN_kernel.wasm > kernel.wat`
- **Read WGSL/GLSL:** Files include metadata headers (kernel name, workgroup size, shared mem, bindings, timestamp)
- **Track test results:** `latest.json` updates after every test. Compare `test-run-*.json` across runs.
- **The files are on disk.** Do NOT ask TJ to capture output or run tests manually. Read the dump folder.

### Test Results (live via latest.json)
`UnitTestsView` writes results to the same debug folder via the `ResultsDirectory` parameter. **`latest.json` is overwritten after EVERY test completion** — it contains the full test suite state in real-time: pass/fail/skip/pending counts and per-test details (class, method, result, error, duration, stack trace). A timestamped `test-run-*.json` is written when the full run finishes.

**During test runs, read `latest.json` to see results as they happen.** Don't wait for the run to finish. Parse it with `node -e` to find failures:
```bash
node -e "const d=JSON.parse(require('fs').readFileSync('path/to/latest.json','utf8')); console.log('Pass:',d.passed,'Fail:',d.failed,'Skip:',d.skipped,'Pending:',d.pending); d.tests.filter(t=>t.result==='Error').forEach(t=>console.log('FAIL:',t.className+'.'+t.method,'-',(t.error||'?').substring(0,200)));"
```

## Engineering Philosophy

- **Correctness is non-negotiable. Performance is a close second.** Kernels dispatch thousands of times/sec.
- **No workarounds that mask problems.** Fix root causes.
- **Cross-backend impact** — changes to `ILGPU/` affect all 6 backends. Consider all of them.
- **No quick fixes** — plan before implementing complex changes.
- **Do not hardcode evolving hardware limits** — preserve full i64 index paths.

## Global Constraints

These apply everywhere, not just one directory:

- **No backend-specific kernel variants.** NEVER create backend-specific copies of algorithm kernels (e.g., `WasmRadixSortKernel1`) to work around bugs. The same kernel must work on all 6 backends. Fix bugs in the codegen, dispatch, or memory management — not by duplicating the algorithm. Only acceptable if it is absolutely IMPOSSIBLE to fix any other way.
- **Blazor WASM is single-threaded** — all async, no blocking calls
- **T4 Templates in `ILGPU/`** — check for `.tt` before editing `.cs`. Generated files are silently overwritten.
- **Device loss detection** — WebGPU: `device.lost` promise. WebGL: `webglcontextlost` event. Guards on dispatch/synchronize. Intentional disposal filtered out.
