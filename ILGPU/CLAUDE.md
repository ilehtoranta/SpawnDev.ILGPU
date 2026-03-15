# Forked ILGPU

Modified fork of ILGPU, included as private asset reference. Modifying internals (e.g., `internal` → `public`, adding intrinsics) is expected and preferred over workarounds.

## T4 Templates — CRITICAL
Many `.cs` files are generated from `.tt` templates. **Before editing any `.cs` file, check for a matching `.tt` file.** Edits to generated files are silently overwritten on rebuild.

## Cross-Backend Impact
Changes here affect ALL 6 backends (CUDA, OpenCL, CPU, WebGPU, WebGL, Wasm). Consider all of them before modifying.

## Key Areas
- `IR/Types/StructureType.cs` — struct type flattening, `DirectFields` vs `Fields` (flattened), `GetOffset()`
- `IR/Values/` — IR value types (GetField, Store, GenericAtomic, Barrier, etc.)
- `Runtime/Accelerator.cs` — `AcceleratorType` enum (includes `.Wasm`)
- `Runtime/MemoryBuffer.cs` — `NativePtr` (public setter for Wasm dispatch patching)
- `Runtime/ArrayView*.cs` — ArrayView, ArrayView1D struct layouts
- `Stride.cs`, `StrideTypes.cs` — stride definitions (Dense has no instance fields)

## ILGPU IR Type System Quick Reference
- `AddressSpaceType` — pointer/view type (ArrayView<T> lowers to this)
- `StructureType` — flattened struct. `DirectFields` = top-level fields. `Fields` = all leaf fields.
- `PrimitiveType` — Int32, Int64, Float32, Float64, etc.
- `StructureType.Builder.Add()` recursively flattens nested structs (line ~128)
- `StructureType.GetOffset(FieldAccess)` returns byte offset of leaf field
