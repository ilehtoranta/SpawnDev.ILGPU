# Divergent Barrier Support — Implementation Plan

**Date:** 2026-03-20
**Status:** Future work (post-v4.6.0). Prerequisite met: all uniform barrier tests pass (249/0/3).
**Prerequisite:** ✅ All uniform barrier tests passing (scan, broadcast, dot product, RadixSort)

## The Problem

Some kernels have CONDITIONAL code paths where different threads hit different numbers of barriers. The current fiber approach requires ALL threads to yield the same number of times per phase. When thread A completes in 12 phases and thread B needs 15, the worker can't synchronize them.

### Why It Happens

RadixSort scatter kernel example:
```csharp
if (threadIdx < numBuckets) {
    // Path A: 4 barriers (histogram accumulation)
    Group.Barrier();
    // ... work ...
    Group.Barrier();
} else {
    // Path B: 2 barriers (wait for results)
    Group.Barrier();
}
// Path C: all threads rejoin
Group.Barrier();
```

Path A has 4+1 = 5 barriers. Path B has 2+1 = 3 barriers. The per-phase yield model breaks because threads on Path A yield 5 times while Path B yields 3 times. After phase 3, Path B threads are "done" but get called again — their stale state causes garbage execution.

### What Works Today

Uniform barriers: all threads execute the same barriers in the same order. This covers:
- InclusiveScan / ExclusiveScan (all threads write → barrier → thread 0 scans → barrier → all read)
- DotProduct (all threads compute → barrier → reduce)
- SharedMemoryBarrier (all threads write → barrier → all read)
- Broadcast (all threads → broadcast → all threads)

## Options Analysis

### Option A: Convergent Barrier Insertion (RECOMMENDED)

At compile time, analyze all code paths through the kernel. If a branch has fewer barriers than the longest path, insert **dummy yields** (yield but do nothing) to equalize. All threads always yield the same number of times.

```
// Before:
if (cond) {
    barrier();  // real
    barrier();  // real
} else {
    barrier();  // real
}
barrier();      // real (all paths)

// After convergent insertion:
if (cond) {
    barrier();  // real
    barrier();  // real
} else {
    barrier();  // real
    DUMMY_YIELD; // inserted — no-op, just yields to keep phase count equal
}
barrier();      // real (all paths)
```

**Pros:**
- Worker script unchanged — still assumes uniform phases
- Correct by construction — all threads always yield the same number of times
- Compile-time analysis, no runtime overhead (dummy yield is just save/restore/return)

**Cons:**
- Requires barrier count analysis per branch at IR level
- Wasted phases (dummy yields add latency for threads that don't need them)
- Nested conditionals with barriers get complex

**Implementation:**
1. Walk IR CFG and count barriers per path from entry to exit
2. Find max barrier count across all paths
3. For paths with fewer barriers, insert dummy yield instructions before the convergence point
4. Adjust `totalBarriers`, `PhaseCount`, and `_blockCount` to match the max

### Option B: Per-Thread Phase Tracking

Each thread maintains its own phase counter. The worker runs threads independently — each thread advances its own phase until it completes. No global synchronization between phases.

```javascript
// Worker:
const threadDone = new Array(groupSize).fill(false);
const threadPhase = new Array(groupSize).fill(0);
while (true) {
    let allDone = true;
    for (let tid = 0; tid < groupSize; tid++) {
        if (threadDone[tid]) continue;
        allDone = false;
        const r = kernel(... threadPhase[tid] ...);
        if (r === 0) threadDone[tid] = true;
        else threadPhase[tid]++;
    }
    if (allDone) break;
}
```

**Pros:**
- Handles any divergence pattern — no compile-time analysis needed
- Simple worker change
- No dummy yields — each thread runs exactly the phases it needs

**Cons:**
- **BREAKS BARRIER SEMANTICS.** The whole point of barriers is that ALL threads sync at the same point. If thread 0 runs phase 3 while thread 1 is still on phase 1, shared memory is not synchronized. Thread 0 reads data thread 1 hasn't written yet.
- Only correct for kernels where threads DON'T share data across barriers (rare)

**Verdict: WRONG for most cases.** Barriers exist because threads need to sync. Running them independently defeats the purpose.

### Option C: Hybrid — Per-Thread Tracking with Sync Points

Like Option B but with explicit sync points. The worker tracks per-thread phases but also maintains sync barriers where ALL threads must reach before any can proceed.

```javascript
// Worker:
while (true) {
    // Run all threads for one phase each
    let anyYielded = false;
    for (let tid = 0; tid < groupSize; tid++) {
        if (threadDone[tid]) continue;
        const r = kernel(... threadPhase[tid] ...);
        if (r === 0) threadDone[tid] = true;
        else { threadPhase[tid]++; anyYielded = true; }
    }
    if (!anyYielded) break;
    // ALL non-done threads must yield before any advance
    // (this is already what we do — the current model)
}
```

Wait — this is exactly the current model. The issue is that "done" threads get called again with stale state. The **state persist fix** (#1 already applied) solves that: completed threads save exit state to scratch, and on re-entry they dispatch to the default exit and return 0.

**The state persist fix + the current model IS Option C.** If a thread finishes early (fewer barriers), it returns 0 on subsequent phases. The worker keeps running until ALL threads return 0.

**This should already work!** The only issue is: does the "done" thread's re-entry correctly return 0 without side effects?

### Option D: Detect Divergence at Compile Time

Analyze IR paths. If all paths have the same barrier count → use current uniform approach. If paths diverge → fall back to single-worker sequential execution (CPU-style, correct but slow).

**Pros:** Safe — never produces wrong results
**Cons:** Slow for divergent kernels, requires CFG analysis

## Recommended Approach: Option A + Option C Combined

1. **Option C (state persist) is already implemented** — completed threads return 0 on re-entry. This handles the "fast thread finishes early" case.

2. **The remaining issue is shared memory consistency.** When thread 0 is on phase 5 (past barrier 3) and thread 1 is on phase 3 (at barrier 2), thread 0 reads shared memory that thread 1 hasn't updated yet.

3. **Option A (convergent insertion) fixes this** by ensuring all threads are always at the same phase. No thread gets ahead because dummy yields keep them synchronized.

## Implementation Plan (Option A)

### Phase 1: CFG Barrier Analysis

Add a pass in `GenerateStateMachineCode` that:
1. Walks the IR CFG from entry to all exit paths
2. Counts barriers on each path (including helper barriers)
3. Computes `maxBarriers` across all paths
4. Identifies branches where paths diverge in barrier count

### Phase 2: Dummy Yield Insertion

For each path with fewer barriers than `maxBarriers`:
1. Identify the convergence point (where the divergent paths rejoin)
2. Insert `maxBarriers - pathBarriers` dummy yield instructions before the convergence
3. A dummy yield: save all locals → set yielded=1 → br $exit (same as real yield, but no barrier logic)

### Phase 3: Block Count and Phase Count Update

- `totalBarriers = maxBarriers` (not sum of all paths)
- `expandedBlockCount = IRblocks + maxBarriers`
- `PhaseCount = maxBarriers + 1`

### Phase 4: Testing

- RadixSort NonPairs (the canary test)
- RadixSort with various sizes (16, 64, 256, 1024)
- Verify no regression on uniform barrier tests

## Complexity Estimate

- CFG barrier analysis: moderate (traverse IR blocks, track per-path counts)
- Dummy yield insertion: straightforward (same codegen as real yield minus the barrier logic)
- Block count adjustment: already done for uniform case — extend to max-path
- Testing: RadixSort is the primary test case

## Files to Modify

- `WasmKernelFunctionGenerator.cs` — CFG analysis + dummy yield insertion
- `WasmBackend.cs` — possibly, for block count propagation
- `WasmTests.cs` — un-skip RadixSort tests

## Prerequisite

All uniform barrier tests must be green first. The ExclusiveScanWithSharedMemoryTest regression must be fixed before starting this work.
