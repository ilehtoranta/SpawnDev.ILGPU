# Plan: Atomic.CompareExchange(ref Half, Half, Half) — Frontend + 6-Backend Intrinsics

**Author:** Geordi
**Date:** 2026-04-22
**Status:** DESIGN / PARKED — follow-up cycle

## Why this exists

Phase 4 of `f16-emulation-plan.md` shipped `accelerator.Reduce<Half, AddHalf/MaxHalf/MinHalf>` via a widen-to-f32 dispatch: allocate an f32 temp buffer equal in size to the input, run `Reduce<float, AddFloat>`, convert the result back to Half. That works, is lossless, and passes on 6 of 7 backend variants (WebGL legitimately skips).

The widen-to-f32 approach has one real cost: **it allocates an O(N) f32 temp buffer** for every Half reduction. For ML workloads with N = millions of elements, that's a transient 4× memory spike (Half is 2 bytes, the temp is 4 bytes per element). On memory-constrained devices (Quest 3S: 2 GB max buffer), this can be the difference between a model fitting and not fitting.

Native `Atomic.CompareExchange(ref Half, Half, Half)` + direct Half-typed atomics in `ScanReduceOperationsHalf.cs` would let `Reduce<Half>` run without the temp buffer — single-pass, like `Reduce<float>` does today.

## The problem the widen path sidesteps

Hardware has no Half atomics. `Atomic.CompareExchange(ref Half, Half, Half)` needs to be emulated via CAS on the containing u32 word. That requires:

1. **Buffer alignment guarantee** — Half storage is 2 bytes; CAS operates on 4 bytes. The Half target must sit at a 4-byte-aligned offset with 2 bytes of preserved-padding above. This is true on CUDA / OpenCL / .NET managed heap / WebGPU (post-Phase-1). Wasm currently allocates literal element-count × element-size bytes; needs a min-4-byte pad (reverted during Phase 4 debug; would need to re-land).
2. **Ref reinterpretation that each backend's codegen accepts** — `Atomic.CompareExchange(ref float, ...)` uses `Unsafe.As<float, uint>(ref target)`. For Half, the equivalent `Unsafe.As<Half, uint>(ref target)` is a size-mismatch reinterpret (2 → 4 bytes). CPU + CUDA accept it; WebGPU + OpenCL + Wasm had mixed results in the Phase 4 initial attempt. Per-backend codegen work required.
3. **Frontend intrinsic registration** — add `Atomic.CompareExchange(ref Half, Half, Half)` to `ILGPU/Atomic.cs` mirroring the existing `(ref float, float, float)` overload. Wire up the IR lowering.

## Scope

### Files touched

- `ILGPU/Atomic.cs` — new `CompareExchange(ref Half, Half, Half)` overload
- `ILGPU/IR/Construction/Atomics.cs` — IR construction for the Half atomic
- `ILGPU/IR/Values/Atomic.cs` — IR value for the Half atomic op (may reuse existing with type discriminator)
- `ILGPU/Frontend/Intrinsic/AtomicIntrinsics.cs` — intrinsic registration
- Each backend that supports Half atomics:
  - `ILGPU/Backends/Cuda/PTX*` — PTX emit (likely straightforward; CUDA has atomicCAS.b32 we can reinterpret over)
  - `ILGPU/Backends/OpenCL/CL*` — OpenCL C emit
  - `SpawnDev.ILGPU/WebGPU/Backend/WGSL*` — WGSL emit with CAS-on-u32 + bit-pack
  - `SpawnDev.ILGPU/Wasm/Backend/Wasm*` — Wasm binary emit (+ the min-4-byte pad in WasmMemoryBuffer)
  - `ILGPU/CPUAtomic.cs` or equivalent — .NET `Interlocked.CompareExchange` reinterpret-ref
  - `SpawnDev.ILGPU/WebGL/*` — **N/A**, no vertex shader atomics
- `ScanReduceOperationsHalf.cs` — `AddHalf/MaxHalf/MinHalf.AtomicApply` rewritten to call the new `Atomic.CompareExchange(ref Half)` primitive via `Atomic.MakeAtomic<Half, ..., CompareExchangeHalf>`
- `ReductionExtensions.cs` — remove the Half widen-to-f32 dispatch; let Half go through the same `CreateReduction<Half, Stride1D.Dense, TReduction>()` path as other types

### Scope estimate

Session-sized library engineering: all 6 backends touched, frontend intrinsic table updated, per-backend codegen emit verified against a concurrent-contention test. **Not a small change**; justified only if the widen-to-f32 memory spike turns out to be a real problem for a real ML workload. Until then, the shipping path is fine.

## Acceptance criteria

1. `Atomic.CompareExchange(ref Half, Half, Half)` compiles to correct native or CAS-emulated atomic on each backend
2. `AddHalf/MaxHalf/MinHalf.AtomicApply` work standalone (direct caller, outside Reduce)
3. `ReductionExtensions` widen-to-f32 branch removed; `Reduce<Half, AddHalf>` path identical to `Reduce<float, AddFloat>` shape
4. `ILGPUReduceHalfTest` still green on 6 of 7 backends with count=2048
5. Memory: single-pass Reduce<Half> allocates no O(N) temp
6. Concurrent-contention test verifies correctness under multi-workgroup atomic pressure (>32 workgroups contending on one Half slot)

## When to pick this up

Triggers:
- An ML workload reports OOM or thrashing due to the widen-to-f32 temp buffer
- A consumer project wants direct `Atomic.Add(ref Half)` outside the Reduce context
- A next-major-cycle refactor of ILGPU atomics touches the same code paths

Until one of those fires, the widen-to-f32 path (shipping today) is the correct answer. It's lossless, easy to reason about, and puts zero new surface area into the forked ILGPU core or the six backends.

## Cross-reference

- `f16-emulation-plan.md` Phase 4 (the shipping implementation)
- `Docs/atomic-operations.md` Half row (documents the current widen-to-f32 pattern)
- `ILGPU/AtomicFunctions.cs` + `ILGPU/Atomic.cs` (existing float / int pattern we'd mirror)
