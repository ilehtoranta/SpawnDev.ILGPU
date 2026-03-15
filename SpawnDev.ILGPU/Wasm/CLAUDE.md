# Wasm Backend

Compiles ILGPU IR → WebAssembly binary. Dispatches via Web Workers with SharedArrayBuffer.

## Key Files
- `Backend/WasmKernelFunctionGenerator.cs` — kernel codegen, parameter setup, helper functions
- `Backend/WasmCodeGenerator.cs` — base IR visitor, GetField, Store, Atomic handlers
- `Backend/WasmBackend.cs` — compilation orchestration, helper generation, `LastWasmBinary`
- `Backend/WasmModuleBuilder.cs` — Wasm binary format builder (sections, types, functions)
- `WasmAccelerator.cs` — dispatch to workers, buffer management, struct serialization
- `WasmMemoryBuffer.cs` — SharedArrayBuffer-backed memory, zero-copy sharing
- `WasmILGPUDevice.cs` — device config (MaxNumThreadsPerGroup=64)

## Hard Constraints
- **Blazor WASM is single-threaded** — all async, no blocking. `stream.Synchronize()` is a no-op.
- **Serialized dispatch** — `RunKernelAsync` awaits `_pendingWork` before each dispatch.
- **Struct-with-view serialization** — CLR layout ≠ IR layout. Use `WasmParamInfo.StructFields` + `FlattenCLRStruct()` for manual IR-layout serialization. See SKILL.md for details.
- **`IsViewType()` distinguishes views from struct-with-view** — checks if `DirectFields[0] is AddressSpaceType`.
- **Empty struct padding** — `Stride1D.Dense` has no CLR fields but IR adds Int8 padding.
- **`SpecializedValue<T>` unwrapping** — IR lowers to PrimitiveType; dispatch must extract inner value.
- **`LongIndex1D` as first param** — it's extent (loop bound), not thread index. Don't map to `_globalIdxLocal`.
- **Buffer deduplication** — SubViews of same buffer share one copy in Wasm memory.
- **NativePtr patching** — set to Wasm offset before struct serialization, restore to 0 after.
- **Multi-pass algorithms** — route to `CreateSingleGroupScan` (ScanExtensions.cs, AcceleratorType.Wasm).

## Tribal Knowledge: GetField View Field Mapping (March 2026)

**FIELD MAPPING RULE**: In the `GetField` handler for view parameters, field 1 is context-sensitive:
- **StructureType views** (ArrayView1D): field 1 = **Extent (Length)** → return `locals[1]`
- **AddressSpaceType views** (ArrayView): field 1 = **Index/Offset** → return 0

This was hardcoded to 0 for ALL views, which broke `view.Length` for ArrayView1D params.
The fix checks `param.Type is StructureType`. 152 existing tests pass with this change.

**TRACE RULE**: Both `GetViewLength` and `GetField` must trace the view source back to
the kernel Parameter through GetField/NewView/AddressSpaceCast chains (via `TraceToParameter()`).
ArrayView1D's BaseView access creates a GetField indirection that breaks direct Parameter lookup.

## Kernel Function Signature
9 system params + user params:
`kernel(globalIdx, dimX, dimY, scratchBase, groupDimX, threadIdX, sharedMemBase, barrierBase, dynamicSharedLen, ...userParams)`

## Barrier Dispatch
- Each Web Worker = one thread within a workgroup.
- Generation-counting barriers: `atomic.fence` + `i32.atomic.rmw.add` + `memory.atomic.wait32`/`memory.atomic.notify`.
- All workers iterate over groups: `for (g=0; g<numGroups; g++) kernel(g*groupSize+threadId, ...)`.

## Tribal Knowledge: GridIndex vs BucketIndex Bug (March 2026)

**RADIX RULE**: All atomic/store writes to shared histograms MUST verify the bucket index multiplier in the address computation. When the codegen unrolls a per-bucket loop, each iteration must use a DIFFERENT counter address — `counter_base + (numGroups * bucket + gridIndex) * sizeof(int)`. If the unrolled writes all share the same `local` for the index (e.g. `gridIndex` which is 0 for single-group), they all hit `counter[0]` and the histogram is silently wrong. The data appears unchanged (no crash, no trap) because the scan of an all-zero histogram produces all-zero offsets, so the scatter is a no-op.

**Current status**: RESOLVED. The counter addresses were actually correct (confirmed by WAT re-analysis with 15 blocks). The real issue was the scan helper's local alloca corruption — see Local Alloca rule below.

## Tribal Knowledge: Local Alloca Must Use Scratch (March 2026)

**LOCAL ALLOCA RULE**: The base `Alloca` handler in `WasmCodeGenerator.cs` sets local alloca addresses to `i32.const 0`. This causes the kernel to write to Wasm memory address 0 (the data buffer region). The `WasmKernelFunctionGenerator` MUST override this for non-shared allocas to allocate scratch memory (`scratchBaseLocal + offset`). Without this fix, the ExclusiveScan helper's output struct gets written to address 0, corrupting sorted data between RadixSort passes.

## Debugging
- `WasmBackend.LastWasmBinary` — capture last compiled kernel
- `WasmBackend.AllKernelInfos` — compilation summaries
- Disassemble: `wasm2wat --enable-threads kernel.wasm` (MUST use --enable-threads)
- Do NOT use LINQ in Blazor WASM logging — silently fails. Use for-loops.
