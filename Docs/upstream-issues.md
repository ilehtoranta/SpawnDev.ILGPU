# Upstream ILGPU Issues Analysis

Analysis of [open issues](https://github.com/m4rs-mt/ILGPU/issues) in the original ILGPU repo to determine which bugs are inherited and fixable in SpawnDev.ILGPU v3.3.0.

## Actionable Bugs (Testable & Potentially Fixable)

### 1. ✅ [#1361](https://github.com/m4rs-mt/ILGPU/issues/1361) — `MathF.CopySign` argument order swapped on GPU — **FIXED in v3.3.0**

| | |
|---|---|
| **Severity** | High — silent wrong results |
| **Affected** | All GPU backends (CUDA, OpenCL, WebGPU, WebGL) |
| **Reproducible?** | Yes — simple kernel: `CopySign(x, -1)` should return `-x` |
| **Root cause** | `XMath.CopySign` intrinsic passed `(sign, magnitude)` instead of `(magnitude, sign)` to the backend `copysign` instruction |
| **Fix complexity** | Low — swapped the two arguments in the intrinsic mapping |
| **Testable** | ✅ `CopySignTest` passes on all backends |

### 2. ✅ [#1309](https://github.com/m4rs-mt/ILGPU/issues/1309) — `uint` to `float` cast goes through `double` — **FIXED in v3.3.0**

| | |
|---|---|
| **Severity** | Medium — crashes on devices without fp64 support |
| **Affected** | OpenCL devices without double precision (Intel integrated GPUs) |
| **Reproducible?** | Yes — any `(float)someUint` cast in a kernel |
| **Root cause** | IL `conv.r.un` + `conv.r4` treated as uint→double→float instead of direct uint→float, emitting fp64 ops on devices that don't support them |
| **Fix complexity** | Low — added direct uint→float conversion path in the IL-to-IR converter |
| **Testable** | ✅ `UintToFloatCastTest` passes on all backends |

### 3. ✅ [#1479](https://github.com/m4rs-mt/ILGPU/issues/1479) — Infinite compilation with large local arrays — **FIXED in v3.3.0**

| | |
|---|---|
| **Severity** | High — 10+ minute compile, 10+ GB RAM for `new int[1_000_000]` |
| **Affected** | All backends |
| **Reproducible?** | Yes — any kernel with `new int[N]` where N is large |
| **Root cause** | `LowerArrays.Lower` unrolled zero-initialization into N individual store IR nodes regardless of array size |
| **Fix complexity** | Medium — added threshold (32 elements); small arrays keep unrolled stores, large arrays emit a proper IR loop |
| **Testable** | ✅ All 366 existing tests pass across CUDA/OpenCL/CPU |

### 4. ✅ [#1538](https://github.com/m4rs-mt/ILGPU/issues/1538) — Internal Compiler Error with nested struct properties — **FIXED in v3.3.0**

| | |
|---|---|
| **Severity** | Medium — prevents kernel compilation |
| **Affected** | All backends |
| **Reproducible?** | Yes — deeply nested `record struct` parameters + static struct member access |
| **Root cause** | `StructureType.Slice` used `SliceRecursive`/`DirectFields` to extract sub-spans, but type unification could merge types with different field orderings (e.g., `{float, Vec3}` vs `{Vec3, float}`), causing wrong slices |
| **Fix complexity** | Medium — changed `Slice` to use flat `Fields` directly instead of `SliceRecursive` |
| **Testable** | ✅ `NestedStructICETest` passes on all 8 backends (CPU, CUDA, OpenCL, WebGPU, WebGL, Wasm) |

### 5. ✅ [#1540](https://github.com/m4rs-mt/ILGPU/issues/1540) — H100/H200 not working — **Already fixed in our fork**

| | |
|---|---|
| **Severity** | High — H100/H200 GPUs crash immediately |
| **Affected** | CUDA (Hopper architecture SM_90) |
| **Reproducible?** | Only on H100/H200 hardware (we don't have this) |
| **Root cause** | Original ILGPU 1.5.x didn't include SM_90 in architecture tables |
| **Fix complexity** | Already fixed — our fork's `CudaArchitecture.Generated.cs` includes SM_90, SM_100, SM_101, SM_120 |
| **Testable** | Would need H100/H200 to verify, but code is clearly present |

### 6. ✅ [#1539](https://github.com/m4rs-mt/ILGPU/issues/1539) — OpenCL produces wrong results for complex kernels — **FIXED in v3.3.0**

| | |
|---|---|
| **Severity** | High — silent wrong results |
| **Affected** | All OpenCL backends (not just AMD — reproduces on NVIDIA too) |
| **Reproducible?** | Yes — BVH ray traversal kernel with while-loop stack-based traversal |
| **Root cause** | `intermediatePhiVariables` dictionary in `CLCodeGenerator.GenerateCodeInternal()` persisted across blocks, causing stale intermediate phi variables from one block's phi swap to be incorrectly used as source values in a different block's phi bindings |
| **Fix complexity** | Low (one-line fix) — added `intermediatePhiVariables.Clear()` at the start of each block's phi binding processing |
| **Testable** | ✅ `BVHRayTraversalTest` passes on all backends (CPU, CUDA, OpenCL, WebGPU, WASM) |

## Not Actionable (For Us)

| Issue | Why Not |
|---|---|
| [#1535](https://github.com/m4rs-mt/ILGPU/issues/1535) — .NET Standard 2.1 | We target .NET 10, not relevant |
| [#1359](https://github.com/m4rs-mt/ILGPU/issues/1359) — DebugInformationManager | PDB loading exception — very environment-specific |
| [#1263](https://github.com/m4rs-mt/ILGPU/issues/1263) — Radix sort too many resources | Algorithm-specific, configuration-dependent. Likely Debug vs Release IL verbosity. |
| [#1542](https://github.com/m4rs-mt/ILGPU/issues/1542), [#1476](https://github.com/m4rs-mt/ILGPU/issues/1476), [#1508](https://github.com/m4rs-mt/ILGPU/issues/1508) | Questions, not bugs |

## Fix Summary

1. ~~**#1361 (CopySign)**~~ ✅ Fixed — swapped argument order in PTX/OpenCL intrinsic
2. ~~**#1309 (uint→float)**~~ ✅ Fixed — direct uint→float conversion without double intermediate
3. ~~**#1479 (local array unrolling)**~~ ✅ Fixed — threshold-based loop for large arrays
4. ~~**#1538 (struct ICE)**~~ ✅ Fixed — `StructureType.Slice` uses flat fields instead of `SliceRecursive`
5. ~~**#1540 (H100/H200)**~~ ✅ Already fixed — SM_90+ architecture tables present in fork
6. ~~**#1539 (OpenCL wrong results)**~~ ✅ Fixed — `intermediatePhiVariables.Clear()` per-block in OpenCL code generator
