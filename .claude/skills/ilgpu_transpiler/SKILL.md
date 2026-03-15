---
user-invocable: false
description: "Deep logic for C# IL transpilation to WebGPU/Wasm"
---

# ILGPU Transpiler: Hard-Won Mapping Rules

This skill encodes the non-obvious rules discovered through debugging the IL → WGSL / GLSL / Wasm transpiler pipeline. **Every rule below was learned from a production bug.** When working on transpiler code, use extended reasoning to verify each mapping against these rules before emitting code.

## ILGPU IR Type System → Target Language Mapping

### ArrayView Representation
- `ArrayView<T>` in IR = **single `AddressSpaceType`** (a pointer). NOT decomposed into ptr+index+length.
- `ArrayView1D<T, TStride>` in IR = **StructureType** with `DirectFields = [AddressSpaceType, LongIndex1D, TStride]`
- To distinguish a view StructureType from a struct-with-embedded-view: check if `DirectFields[0] is AddressSpaceType`. If yes → view. If `DirectFields[0]` is another StructureType → struct-with-view.
- `StructureType` flattens nested structs via `Builder.Add()` (StructureType.cs line ~128). All `Fields[i]` are leaf types. `GetOffset(i)` gives byte offset. `NumFields` gives count.

### Struct Parameter Serialization (Wasm Backend)
- `Unsafe.Write` produces CLR layout. The kernel reads ILGPU IR layout. **These are different.**
- CLR has managed `MemoryBuffer` reference (GC handle). IR has `AddressSpaceType` (Wasm memory offset).
- CLR `ArrayView<T>` has 3 fields: `Buffer` (ref), `Index` (long), `Length` (long). IR has 1 field: pointer.
- Empty structs like `Stride1D.Dense` have no CLR instance fields but ILGPU adds an `Int8` padding field.
- `SpecializedValue<T>` wraps a value — IR lowers it to `PrimitiveType`. Dispatch must unwrap.
- **Fix**: Use `StructureType.GetOffset()` at compile time to record IR layout in `WasmParamInfo.StructFields`. At dispatch, walk CLR struct depth-first via `FlattenCLRStruct()`, write each value at its IR offset.

### WGSL Uniformity (WebGPU Backend)
- WGSL uniformity analysis is **syntactic**, not semantic. The browser traces variable origins through CFG.
- Even mathematically uniform expressions are rejected if ANY value in the chain traces to `local_invocation_id`.
- Synthetic loop counters for uniformity must be built entirely from `group_id`, `num_workgroups`, `workgroup_size`.
- Tile loops (step=GroupDim) need `_uf_tile_iter` counter stripped of GroupIndex. Grid-stride loops (step=GridDim) need `_uf_grid_iter`.
- `ClassifyLoopType()` in `UniformityAnalyzer.cs` distinguishes them by tracing the step PHI value.

### Emulation Library Inclusion
- `SetEmulationFlags()` must run BEFORE `GenerateCode()` body emission — it scans kernel + ALL helper methods' IR for Int64/Float64 usage.
- Helper methods are inlined at emission time (not IR level). Their IR values may contain types not present in the kernel's own IR. **Always scan helpers too.**
- i64 emulation is always on (ILGPU IR uses Int64 for ArrayView.Length). f64 emulation is configurable via `F64EmulationMode`.
- Per-function trimming: `GetMinimalEmulationLibrary()` BFS-walks used functions and their dependencies.

### NaN/Inf Detection
- WGSL: use `bitcast<u32>()` bit-level checks, NOT `val != val`. GPU shader compilers may optimize away self-comparisons or flush NaN.
- Pattern: `IsNaNF(x)` = `(bitcast<u32>(x) & 0x7F800000u) == 0x7F800000u && (bitcast<u32>(x) & 0x007FFFFFu) != 0u`

### Wasm Backend Specifics
- Wasm barrier dispatch: each Web Worker = one thread. All workers in a group synchronize via generation-counting barriers (atomic counter + atomic wait/notify).
- `stream.Synchronize()` is a no-op on Blazor WASM (single-threaded). Multi-pass algorithms must use `CreateSingleGroupScan` (single kernel dispatch, no inter-pass sync).
- `NativePtr` on `MemoryBuffer` is patched to Wasm memory offset before struct serialization, restored to 0 after. But this only helps for view params that read NativePtr at runtime — for serialized structs, must use IR-layout-aware manual serialization.
- Kernel function signature: 9 system params (globalIdx, dimX, dimY, scratchBase, groupDimX, threadIdX, sharedMemBase, barrierBase, dynamicSharedLen) + user params.
- For `LongIndex1D` as first param: it's the extent (loop bound), NOT the thread index. Set `_indexParam = null; startIdx = 0;` to treat all params as user params.

### WebGPU Backend Specifics
- 4-byte alignment required for ALL buffer operations. Use `WebGPUAlignment.AlignTo4()`.
- `WebGPUStream` uses deferred encode + flush (batched submission).
- `KernelSpecialization` bakes workgroup size into compiled shaders. Required for RadixSort, Histogram, etc.
- Shared memory sizing: RadixSort's ExclusiveScan calls share the same workgroup variable. Trailing `Group.Barrier()` needed after each scan call to prevent data corruption.
- `TempViewManager.Allocate()` must pad to 256 bytes for `minStorageBufferOffsetAlignment`.
- `_ilgpu_user_dim` override constant prevents excess threads from corrupting buffers in auto-grouped kernels.

### WebGL Backend Specifics
- No shared memory, atomics, or barriers. Uses Transform Feedback for output.
- `glWorker.js` runs kernels off-main-thread in a Web Worker.
- Context loss detection via `webglcontextlost`/`webglcontextrestored` events.

## Debugging Generated Code

### Wasm Binary Inspection
1. `WasmBackend.LastWasmBinary` captures last compiled kernel in-browser
2. Base64-encode in test error: `Convert.ToBase64String(WasmBackend.LastWasmBinary)`
3. Decode: `base64 -d > kernel.wasm`
4. Disassemble: `wasm2wat --enable-threads kernel.wasm` (MUST use `--enable-threads`)
5. Check atomic alignment, address computation chains, state machine `br_table` structure

### WGSL Shader Inspection
1. `WebGPUBackend.WGSLDumpPath` — auto-dump to files (desktop only)
2. `WebGPUBackend.WGSLRegistry` — named registry of compiled shaders
3. Shader header comments show kernel name, workgroup size, shared memory, emulation flags

### Dispatch Logging
- `WasmAccelerator._dispatchLog` / `_dispatchCount` for Wasm dispatch tracing
- `WebGPUAccelerator.DispatchLog` ring buffer (last 100 dispatches) for WebGPU
- CAUTION: `_dispatchCount` increments synchronously in `RunKernel()` but flat args are built asynchronously. Pass `dispNum` as parameter to async methods.
- CAUTION: Do NOT use LINQ (`args.Select(...)`) in Blazor WASM logging — silently fails. Use manual for-loops.
