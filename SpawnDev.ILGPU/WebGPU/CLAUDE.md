# WebGPU Backend

Transpiles ILGPU IR → WGSL shaders. Dispatches via `WebGPUAccelerator`.

## Key Files
- `Backend/WGSLKernelFunctionGenerator.cs` — main kernel codegen (~5900 lines)
- `Backend/WGSLCodeGenerator.cs` — base IR visitor, i64 constant dedup
- `Backend/WGSLEmulationLibrary.cs` — i64/f64 emulation functions with per-function trimming
- `Backend/SharedMemoryResolver.cs` — alloca→workgroup var matching, WGSL emission
- `Backend/UniformityAnalyzer.cs` — loop classification, PHI tracing, barrier detection
- `WebGPUAccelerator.cs` — dispatch, bind groups, buffer management, device loss monitoring
- `WebGPUStream.cs` — deferred encode + flush (batched submission)
- `WebGPUBackend.cs` — WGSLRegistry, WGSLDiagnostics, WGSLDumpPath, pre-validation

## Hard Constraints
- **4-byte alignment** on ALL buffer ops: sizes, writeBuffer, copyBufferToBuffer, bind group entries. Use `WebGPUAlignment.AlignTo4()`.
- **WGSL uniformity is syntactic** — browser traces variable origins through CFG. Anything touching `local_invocation_id` is non-uniform even if mathematically uniform. See `UniformityAnalyzer.cs`.
- **NaN/Inf** — use `bitcast<u32>()` bit-level checks, not `val != val`.
- **`__` prefix reserved** — never use double-underscore prefix in generated WGSL identifiers.
- **Emulation flags** — `SetEmulationFlags()` must run before `GenerateCode()`. Must scan helpers too.
- **KernelSpecialization** — required for algorithm kernels (RadixSort, Histogram, etc.) to bake workgroup size.
- **`TempViewManager.Allocate()`** — pad to 256 bytes for `minStorageBufferOffsetAlignment`.
- **Shared memory sizing** — RadixSort's ExclusiveScan calls share workgroup variables; trailing `Group.Barrier()` required after each scan call.
- **`_ilgpu_user_dim`** — override constant prevents excess threads corrupting buffers in auto-grouped kernels.

## Emulation
- **i64**: Always on. `vec2<u32>` paired 32-bit ops. `const _c_i64_N` hoisted constants.
- **f64**: Configurable via `F64EmulationMode` — Dekker (`vec2<f32>`), Ozaki (`vec4<f32>`), or Disabled.
- Library inclusion controlled by `SetEmulationFlags()` scanning kernel + helper IR.
- Per-function trimming via `GetMinimalEmulationLibrary()` BFS dependency graph.

## Diagnostics
- `WebGPUBackend.WGSLDumpPath` — dump shaders to files (desktop only)
- `WebGPUBackend.WGSLRegistry` — named registry of compiled shaders
- `WebGPUBackend.WGSLDiagnostics` — flags enum for per-category logging
- `WebGPUAccelerator.DispatchLog` — ring buffer of last 100 dispatches
- Shader header comments: kernel name, workgroup size, shared memory, emulation flags
