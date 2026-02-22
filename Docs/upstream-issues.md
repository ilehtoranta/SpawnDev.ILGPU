# Upstream ILGPU Issues Analysis

Analysis of [open issues](https://github.com/m4rs-mt/ILGPU/issues) in the original ILGPU repo to determine which bugs are inherited and fixable in SpawnDev.ILGPU v3.3.0.

## Actionable Bugs (Testable & Potentially Fixable)

### 1. ✅ [#1361](https://github.com/m4rs-mt/ILGPU/issues/1361) — `MathF.CopySign` argument order swapped on GPU — **FIXED in v3.3.0**

| | |
|---|---|
| **Severity** | High — silent wrong results |
| **Affected** | CUDA (PTX generation), likely OpenCL too |
| **Reproducible?** | Yes — simple kernel: `CopySign(x, -1)` should return `-x` |
| **Root cause** | Likely in PTX/OpenCL intrinsic mapping — arguments passed in wrong order to `copysign` |
| **Fix complexity** | Low — swap two arguments in the intrinsic implementation |
| **Testable** | ✅ Write a `CopySignTest` kernel that checks `CopySign(5f, -1f) == -5f` |

### 2. ✅ [#1309](https://github.com/m4rs-mt/ILGPU/issues/1309) — `uint` to `float` cast goes through `double` — **FIXED in v3.3.0**

| | |
|---|---|
| **Severity** | Medium — crashes on devices without fp64 support |
| **Affected** | OpenCL devices without double precision (Intel integrated GPUs) |
| **Reproducible?** | Yes on Intel iGPU, may not reproduce on NVIDIA (which has fp64) |
| **Root cause** | IL `conv.r.un` + `conv.r4` sequence treated as cast-to-double then cast-to-float, instead of direct uint→float |
| **Fix complexity** | Medium — needs change in IL-to-IR conversion for the `conv.r.un`/`conv.r4` pattern |
| **Testable** | ✅ Write a kernel: `float b = (float)someUint;` and verify on OpenCL |

### 3. ✅ [#1479](https://github.com/m4rs-mt/ILGPU/issues/1479) — Infinite compilation with large local arrays — **FIXED in v3.3.0**

| | |
|---|---|
| **Severity** | High — 10+ minute compile, 10+ GB RAM for `new int[1_000_000]` |
| **Affected** | All backends |
| **Reproducible?** | Yes — any kernel with `new int[N]` where N is large |
| **Root cause** | `LowerArrays.Lower` unrolled zero-initialization into N individual store IR nodes regardless of array size |
| **Fix complexity** | Medium — added threshold (32 elements); small arrays keep unrolled stores, large arrays emit a proper IR loop |
| **Testable** | ✅ All 366 existing tests pass across CUDA/OpenCL/CPU |

### 4. 🟡 [#1538](https://github.com/m4rs-mt/ILGPU/issues/1538) — Internal Compiler Error with nested struct properties

| | |
|---|---|
| **Severity** | Medium — prevents kernel compilation |
| **Affected** | All backends |
| **Reproducible?** | Yes — deeply nested `record struct` parameters + static struct member access |
| **Root cause** | ILGPU compiler fails on deeply nested value-type parameter layers (4 levels deep) combined with static struct property access |
| **Fix complexity** | High — deep in the ILGPU compiler frontend |
| **Testable** | ✅ Write a kernel with 4-level nested record structs accessing a static member |

### 4. 🟠 [#1539](https://github.com/m4rs-mt/ILGPU/issues/1539) — AMD OpenCL produces wrong results for complex kernels

| | |
|---|---|
| **Severity** | High — silent wrong results |
| **Affected** | AMD integrated GPU OpenCL only |
| **Reproducible?** | Only on AMD iGPU hardware (we don't have this) |
| **Root cause** | Likely an OpenCL compiler optimization issue with complex control flow (BVH ray traversal with stack) |
| **Fix complexity** | Unknown — may be driver bug, may be ILGPU emitting ambiguous OpenCL C |
| **Testable** | ❌ Requires AMD integrated GPU |

## Not Actionable (For Us)

| Issue | Why Not |
|---|---|
| [#1540](https://github.com/m4rs-mt/ILGPU/issues/1540) — H100/H200 not working | Requires SM_90 (Hopper) support — needs CUDA architecture table update. We don't have this hardware. |
| [#1535](https://github.com/m4rs-mt/ILGPU/issues/1535) — .NET Standard 2.1 | We target .NET 10, not relevant |
| [#1359](https://github.com/m4rs-mt/ILGPU/issues/1359) — DebugInformationManager | PDB loading exception — very environment-specific |
| [#1263](https://github.com/m4rs-mt/ILGPU/issues/1263) — Radix sort too many resources | Algorithm-specific, configuration-dependent |
| [#1542](https://github.com/m4rs-mt/ILGPU/issues/1542), [#1476](https://github.com/m4rs-mt/ILGPU/issues/1476), [#1508](https://github.com/m4rs-mt/ILGPU/issues/1508) | Questions, not bugs |

## Recommended Priority

1. ~~**#1361 (CopySign)**~~ ✅ Fixed
2. ~~**#1309 (uint→float)**~~ ✅ Fixed
3. ~~**#1479 (local array unrolling)**~~ ✅ Fixed
4. **#1538 (struct ICE)** — Interesting but complex compiler fix, lower priority
5. **#1539 (AMD OpenCL)** — Can't test without AMD hardware, skip for now
