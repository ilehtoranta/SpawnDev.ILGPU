# Half (f16) Support and Quality Roadmap

This document consolidates actionable items from the Half coverage plans, desktop backend fixes, and ongoing quality improvements. Use it as a reference for what's done, what's pending, and what to tackle next.

---

## 1. ~~Immediate / Pending Work~~ DONE

### Desktop Backend Skip Overrides — COMPLETED

**Location:** `SpawnDev.ILGPU.ConsoleDemo/Program.cs`

Skip overrides added for ConsoleDemo test classes:

| Class | Test Method | Reason |
|-------|-------------|--------|
| `CudaTests` | `AlgorithmExclusiveScanHalfTest` | CUDA: PTX warp shuffle only supports b32; Half (b16) requires promotion intrinsics not yet implemented |
| `CudaTests` | `AlgorithmInclusiveScanHalfTest` | Same as above |
| `CudaTests` | `AlgorithmAllReduceHalfTest` | CUDA: AllReduce requires Half atomics which are not supported (Half excluded from AtomicNumericTypes) |
| `CPUTests` | `AlgorithmAllReduceHalfTest` | CPU: AllReduce requires Half atomics which are not supported (Half excluded from AtomicNumericTypes) |

**Root causes:**
- PTX `shfl.sync.*.b32` instructions operate on 32-bit values only. Half is 16-bit, so the PTX JIT rejects generated code.
- Both `PTXGroupExtensions.AllReduce` and `ILGroupExtensions.AllReduce` call `reduction.AtomicApply()`. Half is excluded from `AtomicNumericTypes` in `ILGPU/Static/TypeInformation.ttinclude` because hardware typically lacks half-precision atomics.

**Note:** CPU ExclusiveScan and InclusiveScan Half tests should pass (IL backend uses sequential scan without warp shuffles or atomics).

---

## 2. Completed Work (Reference)

### Phase 1: Half Group Scan/Reduce Operations

- **`ILGPU.Algorithms/ScanReduceOperationsHalf.cs`** — Hand-written (not T4-generated) structs: `AddHalf`, `MaxHalf`, `MinHalf` implementing `IScanReduceOperation<Half>`
  - `Apply` uses `Half.operator+` or `(float)` promotion for comparisons
  - `AtomicApply` throws `NotSupportedException` (never called by WebGPU group extensions)
- **`BackendTestBase.Tests6.cs`** — Kernels: `ExclusiveScanHalfKernel`, `InclusiveScanHalfKernel`, `AllReduceHalfKernel`; tests: `AlgorithmExclusiveScanHalfTest`, `AlgorithmInclusiveScanHalfTest`, `AlgorithmAllReduceHalfTest`
- **Browser skip overrides** — `CPUTests.cs`, `WebGLTests.cs` skip new Half algorithm tests (CPU: barriers; WebGL: shared memory + barriers)

### WebGPU Half Fixes

- **4-byte alignment** — `WebGPUAlignment.AlignTo4()` applied to buffer sizes, byte arrays, and scalar buffers
- **WGSL type mapping** — `Float16` → `"f16"` in `WGSLTypeGenerator.cs`
- **Float16 constant emission** — Bitcast helpers (`BitcastToU32`, `BitcastFromU32`) for `vec2<f16>` patterns
- **Intrinsic handlers** — `GenerateConvertHalfToFloat`, `GenerateConvertFloatToHalf` in `WGSLCodeGenerator.cs`; `HalfExtensions` made public for intrinsic chain
- **Capability detection** — Float16 support from WebGPU device capabilities

### Desktop Backend Skip Overrides

- **`SpawnDev.ILGPU.ConsoleDemo/Program.cs`** — `CudaTests`: skip `AlgorithmExclusiveScanHalfTest`, `AlgorithmInclusiveScanHalfTest` (PTX b32 shuffle), `AlgorithmAllReduceHalfTest` (Half atomics); `CPUTests`: skip `AlgorithmAllReduceHalfTest` (Half atomics)

### XMath Half Overloads

- **`ILGPU.Algorithms/XMath/MinMax.cs`** — Added `XMath.Min(Half, Half)`, `XMath.Max(Half, Half)`, `XMath.Clamp(Half, Half, Half)` via float promotion through `IntrinsicMath`

### Half Test Coverage

- **`BackendTestBase.Tests3.cs`** — Added `HalfEdgeCasesTest` (zero, negative zero, subnormals, max/min values) and `HalfMixedTypeTest` (Half + int → float kernel)
- **`BackendTestBase.cs`** — Added `HalfEdgeCasesKernel` and `HalfMixedTypeKernel` static kernel methods

### Quality

- **Console verbosity** — `Console.WriteLine` wrapped in `#if DEBUG` in `DefaultTests.cs`, `WebGLTests.cs`; `WebGPUBackend.VerboseLogging` for diagnostic messages in `WebGPUAccelerator.cs`

---

## 3. Medium-term: CUDA PTX Half Warp Shuffle Support

**Problem:** PTX `shfl.sync.idx.b32`, `shfl.sync.down.b32`, `shfl.sync.up.b32`, `shfl.sync.bfly.b32` operate on 32-bit values only. Half is 16-bit.

**Solution:** Add Half shuffle intrinsics that widen `f16` to `b32` before shuffling and narrow back afterwards.

**Files:** `ILGPU/Backends/PTX/PTXInstructions.Data.cs`, `ILGPU.Algorithms/PTX/PTXWarpExtensions.cs` (or equivalent intrinsics)

**Impact:** Unlock `ExclusiveScanHalf` and `InclusiveScanHalf` on CUDA, removing 2 skip overrides.

---

## 4. Medium-term: Half Atomics Support

**Problem:** `AtomicFloatTypes = FloatTypes.Skip(1)` in `ILGPU/Static/TypeInformation.ttinclude` (line 134) excludes Half. `AtomicNumericTypes` therefore has no Half.

**Options:**
- **A:** Add Half to `AtomicFloatTypes` — cascades to `AtomicFunctions.tt` and all backends (PTX, OpenCL, CPU, WebGPU, etc.). Requires each backend to emit valid half-precision atomic instructions or fallbacks.
- **B:** CAS-based Half atomics — Use `Atomic.CompareExchange` on `ushort`. Note: `Atomic.CompareExchange` in `ILGPU/Atomic.cs` currently supports only int, long, uint, ulong, float, double — would need `ushort`/Half support first.

**Impact:** Unlock `AllReduceHalf` on CUDA/CPU; enable `accelerator.Reduce<Half, AddHalf>` (multi-workgroup reduction) and `ILGPUReduceHalfTest`.

---

## 5. ~~Medium-term: XMath Half Overloads~~ DONE

**Location:** `ILGPU.Algorithms/XMath/MinMax.cs`

Added `XMath.Min(Half, Half)`, `XMath.Max(Half, Half)`, and `XMath.Clamp(Half, Half, Half)` via `(float)` promotion through `IntrinsicMath`.

---

## 6. Medium-term: WebGPU WGSLCodeGenerator XOR-Butterfly Fallback

**Location:** `SpawnDev.ILGPU/WebGPU/Backend/WGSLCodeGenerator.cs` (line ~2884)

**TODO:** Implement full XOR-butterfly fallback when subgroups are unavailable. Currently emits passthrough so the shader compiles; shared-memory group reduce aggregates across warps. A proper fallback would improve correctness/performance when subgroups are disabled.

---

## 7. Future: WebGPU RadixSort Double Investigation

**Problem:** `AlgorithmRadixSortPairsDoubleTest` is skipped on WebGPU due to RadixSort with double keys being incompatible with f64 emulation (emu_f64).

**Scope:** Investigate whether the f64 emulation path can handle the bit-level operations RadixSort requires.

---

## 8. Skipped Test Analysis (121+ skips on browser)

| Backend | Approx. Skips | Root Cause |
|---------|---------------|------------|
| **CPUTests** | ~49 | WASM is single-threaded; barriers, atomics, subgroups impossible |
| **WebGLTests** | ~52 | Vertex shaders lack shared memory, barriers, atomics |
| **WasmTests** | ~20 | RadixSort infinite loops, struct decomposition unsupported, no subgroups |
| **WebGPU** | 1–2 | RadixSort double (f64 emulation) |

**Conclusion:** Nearly all skips are legitimate hardware/platform limitations, not bugs. Only the WebGPU RadixSort double case may be addressable via f64 emulation investigation.

---

## 9. Test Coverage Gaps

### Half Coverage

| Gap | Blocked By | Status |
|-----|------------|--------|
| `ILGPUReduceHalfTest` | Half atomics | Pending |
| Half buffer copy | — | Already covered by `HalfBufferRoundTripTest` in Tests3 |
| Half in mixed-type scenarios | — | DONE — `HalfMixedTypeTest` added to Tests3 |
| Half edge cases (NaN, Inf, subnormals) | — | DONE — `HalfEdgeCasesTest` added to Tests3 |

### Algorithm Test Type Variants

- `AlgorithmGroupReduceTest` — currently int only; could add float, long, double, Half variants
- `InclusiveScan` / `AllReduce` — currently int, float, Half; could add long, double, uint

---

## 10. Performance / Quality

| Item | Status |
|------|--------|
| **"READ-usage buffer was read back without waiting on a fence"** | Investigate adding fence/await to WebGPU read-back path to eliminate performance warnings |
| **Console verbosity** | Addressed with `#if DEBUG` and `VerboseLogging` flag |

---

## Related Workspace Rules

- **T4 templates** — See `.cursor/rules/ilgpu-t4-templates.mdc`. Half is intentionally excluded from T4-generated atomic types; manual `ScanReduceOperationsHalf.cs` exists for group-level scan/reduce.
- **Project conventions** — See `.cursor/rules/project-conventions.mdc` for quality standards and WebGPU-specific reminders.
