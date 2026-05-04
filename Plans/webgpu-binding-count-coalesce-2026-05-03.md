# WebGPU storage-buffer binding-count coalesce — 2026-05-03

## Status: SHIPPED 2026-05-03 evening (this session)

Codegen-side coalesce landed in `SpawnDev.ILGPU/WebGPU/Backend/WGSLKernelFunctionGenerator.cs` (`DecideCoalesceGroups` after `ScanBodyStructParams`) + `SpawnDev.ILGPU/WebGPU/WebGPUAccelerator.cs` (coalesce-processing pre-pass before Phase 1 + IsCoalesceFieldOffset packing in Phase 2). New `CoalesceManifest` on `WebGPUCompiledKernel` carries the group structure to runtime. Trigger: raw binding count > 10. Eligibility: i32/u32/f32/emu_64 element types, non-atomic, non-sub-word, non-packed-struct.

Runtime path = independent buffers concatenated via `CopyBufferToBuffer` into one fresh GPU buffer per group, bound once. Tuvok confirmed all 38 of his Vorbis fields are independent buffers (Plan B from him not needed). Test coverage: `BackendTestBase.Tests21.CoalesceBindings.cs` (3 tests) — passing on CPU + WebGPU + WebGPUNoSubgroups + CUDA + OpenCL. Wasm + WebGL skipped via `UnsupportedTestException` overrides — pre-existing body-struct decomp limitation in those backends with many ArrayView fields, separate fix task tracked.

## Why

WebGPU spec `maxStorageBuffersPerShaderStage` = 10 (Chrome default). Every `ArrayView` kernel parameter — whether top-level OR a field of a body-struct parameter — gets its own storage-buffer binding. Tuvok's `VorbisPacketDecodeStaticInputs` body struct has 38 `ArrayView` fields (36 `ArrayView<int>` + 2 `ArrayView<double>`), so the kernel needs 44 bindings (38 + 6 for top-level views) and dispatch throws:

```
[WebGPU] Kernel 'Kernel_Run' requires 44 storage buffer bindings but this device only supports 10
```

Same struct shape WILL recur for the upcoming Opus SILK + CELT integration kernels, Vorbis v3 streaming decoder, and any future high-parameter integration kernel. Per `_DevComms/SpawnDev.ILGPU/tuvok-to-geordi-vorbis-v2-binding-count-2026-05-03.md`.

## Existing infrastructure (what's already there)

`WGSLKernelFunctionGenerator` already decomposes body structs (search for `_bodyStructParams` / `BodyStructFieldInfo`):
- Each view field of a body struct gets a binding named `param{N}_f{fieldIdx}` (e.g. `param1_f0`, `param1_f1`, ...).
- Scalar fields are packed into the existing `_scalar_params` machinery.
- Atomic-flagged view fields tracked separately (`_bodyStructAtomicFields` etc).
- Packed-struct support exists for views whose element type contains emulated 64-bit fields (`_packedStructLayouts`).

`WebGPUAccelerator` runtime path (search `FlattenStructFields` / `ContainsPointerFields` / `_reusableBodyStructMap`):
- Pre-expands body-struct args into separate `ArrayView` and scalar values.
- Each expanded `ArrayView` → one `BindGroupEntry`.
- Scalar fields → packed scalar buffer with synthetic `ParamIndex = (irParam + 1) * 1000 + fieldIdx`.
- Final binding count check throws (`WebGPUAccelerator.cs` ~line 1395) when total exceeds device max.

The decomposition is correct for kernels with 1-8 view fields per struct. Tuvok's case (38 views) breaks because every field becomes one binding.

## Approach: detect SubViews-of-shared-buffer, coalesce by parent buffer × element type

The key observation is that body-struct view fields of "flat-packed setup" structs (Tuvok's case) are NOT independent buffer allocations — they're SubViews into ONE underlying flat-packed `MemoryBuffer` (or 2-3 buffers, one per element type). At dispatch time:

1. Group the body struct's `ArrayView` fields by `(parent MemoryBuffer instance, element type)`.
2. Each group becomes ONE coalesced binding bound to its parent buffer.
3. Per-field offsets/lengths flow through the existing `_scalar_params` channel as `IsViewOffset` / `IsViewCount` entries.
4. WGSL access rewritten: `param1_f0[idx]` → `param1_int_coalesced[fieldOffset_0 + idx]` (where `fieldOffset_0` is read from `_scalar_params`).

For Tuvok's struct (36 int + 2 double, all flat-packed):
- 1 coalesced int binding + 1 coalesced double binding = 2 bindings (replaces 38)
- Plus 6 top-level views + 1 scalar packing buffer = ~9 bindings total
- Under the 10 limit ✓

### Compile-time / runtime split

**Compile-time (codegen)**:
- During `ScanBodyStructParams`, detect body-struct view fields. Group them by `(struct param idx, element type)`.
- Emit ONE coalesced binding per group (`param1_int_coalesced` / `param1_f64_coalesced`) instead of N per-field bindings.
- Replace each per-field LEA with a coalesced-buffer access using the field's offset slot from `_scalar_params`.
- Emit two new `ScalarPackingManifest` entries per field: `ViewOffset` (scalar slot for the byte offset within the coalesced buffer) and `ViewCount` (scalar slot for the field's length).

**Runtime (dispatcher)**:
- `FlattenStructFields` runs as today.
- For each body-struct param, group expanded `ArrayView` fields by `(buffer.NativeHandle, elementType)`.
- VERIFY all fields in a group share one `WebGPUMemoryBuffer` (throw with a clear message if a field is from a different buffer — "field X of struct Y is not a SubView of the same parent buffer; coalesce requires shared buffer").
- For each group, emit ONE `BindGroupEntry` for the parent buffer.
- Populate scalar manifest entries with each field's byte-offset within parent + length.

### What must NOT change

- Non-coalescable kernels (scalar/atomic fields, single-view structs, non-uniform-element-type structs without buffer sharing) keep their current behavior.
- Existing tests pass without modification.

### Failure modes to handle gracefully

1. **Body struct fields point to DIFFERENT parent buffers** — fail at runtime with a clear message identifying which field is the outlier. Tuvok's case is "all share one parent" by construction (flat-pack), so this should not fire for him. Generic kernels would need a separate path or a different encoding.
2. **Mixed atomics + non-atomics in same coalesced group** — atomics need `array<atomic<u32>>` typing in WGSL; non-atomics use `array<u32>`. If the coalesced group has both, split into two bindings (one atomic, one not).
3. **Sub-word element types within a group** — Int8/UInt8/Int16/UInt16 are packed into `array<atomic<u32>>` already; integrate with coalescing the same way.

## Open question for Tuvok

Sent in `geordi-to-tuvok-binding-count-starting-2026-05-03.md`:

> Are all 38 ArrayView fields SubViews into ONE underlying flat-packed `MemoryBuffer`, or are they independent buffer allocations?

If "all SubViews of one int buffer + one double buffer" → ship the coalesce approach above.
If "mostly SubViews but a few independents" → need to handle the exception case (likely just leave those as separate bindings).
If "all independent buffers" → coalesce won't help; would need a different approach (e.g., the consumer-side flat-pack restructure that Tuvok mentioned as a fallback).

Awaiting Tuvok's read before implementing — the design hinges on the answer.

## Why not just bind all to one buffer at runtime without codegen change

WebGPU bind-group LAYOUT is fixed at pipeline-creation time. Even if all N entries point to the same GPU buffer, the LAYOUT still has N entries → counted against `maxStorageBuffersPerShaderStage`. The fix MUST reduce binding count at the WGSL declaration site, which means codegen change.

## Owner: Geordi

Filing this Plan as the working doc. Implementation pending Tuvok's response to the buffer-shape question. Once that's confirmed, the code change is a single PR touching `WGSLKernelFunctionGenerator` (codegen) + `WebGPUAccelerator` (dispatch). Wasm has the same opportunity (a body-struct field count beyond worker-binding-table size could regress similarly) — second pass after WebGPU lands.
