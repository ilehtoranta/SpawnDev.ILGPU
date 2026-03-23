# Re-Enable Multi-Worker Barrier Dispatch

**Date:** 2026-03-20
**Status:** COMPLETE (v4.6.0, 2026-03-22) — Full `hardwareConcurrency` multi-worker barrier dispatch with pure spin barriers. 249/0/3 Wasm tests passing.
**Impact:** Full `hardwareConcurrency` speedup for ALL barrier kernels

## Current State

`WasmAccelerator.cs:834` hardcodes `workerCount = 1` for barrier kernels:
```csharp
workerCount = 1; // TODO: Fix cross-worker barrier to use multiple workers.
```

The cross-worker sync code exists (lines 1136-1148) but is gated behind `if (workerCount > 1)` — never runs.

## The Original Bug

Barrier kernels with multiple workers had **cross-group memory visibility issues**. Writes from group 0 weren't reliably visible to group 1 across parallel Web Workers, despite:
- `atomic.fence` in the Wasm kernel
- Atomic kernel loads/stores (`i32.atomic.store`, `i32.atomic.load`)
- JS-level `Atomics.store/load` fences between groups

This was documented in `Wasm/CLAUDE.md` as a known limitation. The workaround was `workerCount = 1`.

## Why It Needs Re-Investigation

1. **The fiber refactor changed the dispatch model.** The old model ran workers continuously with atomic barriers inside the Wasm. The new fiber model yields to JS between phases, where the cross-worker sync happens in JavaScript (`Atomics.wait/notify`). The JS-level sync might not have the same visibility issues as the Wasm-level atomic barriers.

2. **The original bug was pre-fiber.** The visibility issue was observed with the old barrier approach (atomic wait32/notify inside Wasm). The new approach syncs in JS between phases. Different code path = different behavior.

3. **Performance is significant.** On TJ's 12-core machine: barrier kernels could go from 1 worker to 12 workers. That's a 12x theoretical speedup for the parallel portion of scan/sort operations. Even with synchronization overhead, 4-8x is realistic.

4. **It affects the ML inference engine.** SpawnDev.ILGPU.ML uses barrier kernels for TiledMatMul, InstanceNorm, and other shared-memory operations. Multi-worker dispatch would directly accelerate Wasm backend inference.

## Investigation Plan

### Step 1: Reproduce the Original Bug
- Change `workerCount = 1` to `workerCount = _workerCount` (line 834)
- Run the passing barrier tests (PrefixSum, DotProduct, SharedMemoryBarrier, InclusiveScan)
- Do they pass with `workerCount > 1`?
- If YES: the fiber refactor fixed the visibility issue. Ship it.
- If NO: capture the specific failure and continue to step 2.

### Step 2: Diagnose Visibility
- Add per-worker logging: which worker, which phase, what values read/written
- Check: is the issue WITHIN a group (workers disagreeing on shared memory within one phase) or BETWEEN groups (group 0's writes invisible to group 1)?
- Check: does the JS-level `Atomics.wait/notify` between phases provide the memory fence needed?

### Step 3: Fix Options

**If within-group visibility:**
- The JS phase sync should handle this — all workers complete phase N before any starts phase N+1
- Check if the phase barrier (`_phaseBarrier` at lines 1136-1148) is correctly implemented
- The `Atomics.wait/Atomics.notify` pattern should provide a full memory barrier

**If between-group visibility:**
- The group barrier (`_groupBarrier` at lines 1152-1162) syncs between groups
- Check if `Atomics.store/load` on the group barrier provides visibility for non-atomic shared memory writes
- May need an explicit `Atomics.store(sharedView, 0, 0)` fence on a dummy location to flush writes

**If SharedArrayBuffer limitation:**
- Some browsers may not guarantee full memory model compliance for non-atomic accesses to SharedArrayBuffer
- Workaround: use `i32.atomic.store` / `i32.atomic.load` for ALL shared memory accesses in barrier kernels (the codegen already does this for some operations)
- Performance impact: minimal on modern hardware (atomic ops are cheap when uncontended)

### Step 4: Benchmark
- Compare single-worker vs multi-worker for:
  - InclusiveScan (simple, 2 barriers)
  - RadixSort (complex, 12+ barriers)
  - TiledMatMul (ML workload)
- Measure: wall time, phases per second, per-worker utilization

## Files to Modify

- `WasmAccelerator.cs:834` — remove `workerCount = 1` hardcode
- `WasmAccelerator.cs:1136-1162` — verify cross-worker sync logic
- Possibly: `WasmKernelFunctionGenerator.cs` — ensure atomic loads/stores for shared memory in barrier kernels

## Risk

Low — the change is one line (`workerCount = _workerCount`). If tests fail, revert. The fiber architecture already supports multi-worker; it's just disabled.

## Priority

HIGH. This is free performance for every barrier kernel on the Wasm backend. The fiber refactor was built to enable this. We should at least try it.
