# rc.11 — IR LoopUnrolling body-size heuristic — SHIPPED IN rc.12

**Status: COMPLETE.** Shipped 2026-04-25 in commit `767f826` as `SpawnDev.ILGPU 4.9.2-rc.12`
(prerelease, on nuget.org).

The implementation matched the design captured below. `MaxTotalUnrolledBodyCost = 320`,
applied only for `tripCount > maxUnrollFactor`; small loops always full-unroll. Verified
on quiet-machine PMT (Tuvok paused his concurrent Codecs sweep for the window):
`LocalMemoryRepro` 18 / 1 / 2 in 18 s (1 fail = pre-existing WebGL arch ceiling),
`WasmTests.RadixSort*` 18 / 0 / 0 in 7 m 33 s, `AlgorithmRadixSort + Reduce + Scan`
78 / 0 / 0 in 2 m 33 s. Zero regressions across 114 tests.

The "2026-04-24 late-evening tuning attempt (reverted)" section below documents the
naive cap=256 attempt that turned out to be contaminated by concurrent CPU load (per
`feedback_wasm_radix_spinwait_flaky_under_cpu_load.md`). The cap=320 that actually
shipped is conceptually the same heuristic with the small-loop full-unroll branch
left unconditional — the bug in the earlier attempt was applying the cap to small
loops too, which forced (32, 2) instead of (64, 1) and made Chrome run those kernels
slower (3 s → 31 s on the same hardware). Lesson preserved in the comment header on
`LoopUnrolling.MaxTotalUnrolledBodyCost`.

Original design notes preserved below for archaeology.

---

## Problem

`ILGPU/IR/Transformations/LoopUnrolling.cs:ComputeUnrollFactor` chooses an unroll factor from `tripCount` + `maxUnrollFactor` alone - it does not consider how many IR nodes the loop body contains. For loops with small bodies (RadixSort Half-packed inner loop) at `tripCount <= 64`, fully unrolling is correct and fast. For loops with large bodies (the `LocalMemoryReproKernel256` / `1024` test pattern, and by extension Tuvok's iDCT 16x16 / 32x32 kernel scratch-loop shape), the same full or partial unroll produces so much straight-line code that:

- Chrome WGSL validator exceeds its internal compile-time budget (30 s+ timeout on `LocalMemoryRepro_Int256_ShortByteViews`)
- V8 AsyncFunction compile (Wasm worker script) hits the same cliff
- Emitted shader reaches 132 KB WGSL (3867 lines / 1361 scalar var declarations) from a ~20-line C# kernel

## Regression oracle

`SpawnDev.ILGPU.Demo.Shared/UnitTests/BackendTestBase.LocalMemoryReproTests.cs`:

| Test | CPU/CUDA/OpenCL | Wasm | WebGPU | WebGL |
|------|-----------------|------|--------|-------|
| `LocalMemoryRepro_Int64_ShortByteViews` (N=64, shipped in rc.10) | green | green | green | red (arch ceiling) |
| `LocalMemoryRepro_Int256_ShortByteViews` (N=256) | green | **red** | **red** | skip (arch ceiling) |
| `LocalMemoryRepro_Int1024_ShortByteViews` (N=1024) | green | **red** | **red** | skip (arch ceiling) |

"red" = 30+ s timeout. No skip guards on WebGPU / Wasm at N=256 / N=1024 - per Captain 2026-04-24 "we're not skipping anything that can be fixed." Tests stay red until the fix lands.

## What did NOT work

Commit `4e6efc3` (reverted in `714f2ab`) tried the simplest possible fix: `ComputeUnrollFactor(tripCount, maxUnrollFactor)` returning `(1, tripCount)` for `tripCount > maxUnrollFactor` (no partial unroll above the threshold).

- `LocalMemoryRepro` filter turned green on WebGPU + Wasm at N=256 + N=1024 (runtime 2m 10s -> 16s).
- **But this regressed `AlgorithmRadixSortPairsHalfTest`**: wrong sort output on WebGPU + WebGPUNoSubgroups (`"expected=1, got=256"`). Partial unroll is load-bearing for correctness in the Half-packed sort inner loop.
- **And regressed Wasm RadixSort perf**: multiple 1-4M-element tests tipped from pass to >120 s timeout. Partial unroll is load-bearing for Wasm radix performance.

Conclusion: the right knob is NOT trip-count alone. It is `body_node_count * unroll_factor`. Small bodies can keep their 64x unroll; large bodies must get smaller unroll factors.

## The fix shape

At `TryUnroll` (line 494) after `tripCount` resolves, measure body size before calling `ComputeUnrollFactor`:

```csharp
// Measure the IR-node count of one loop iteration.
// BasicBlock.Count (BasicBlock.cs:354) gives per-block instruction count -
// O(nBlocks) traversal, not O(nValues). Fast at IR-optimization time.
var bodyBlocks = loopInfo.ComputeOrderedBodyBlocks();
int bodyCost = 0;
foreach (var block in bodyBlocks)
    bodyCost += block.Count;

var (unrolls, iterations) = ComputeUnrollFactor(
    tripCount.Value,
    maxUnrollFactor,
    bodyCost);
```

Confirmed API surface:
- `LoopInfo.ComputeOrderedBodyBlocks()` at `ILGPU/IR/Analyses/LoopInfo.cs:575` returns a `BasicBlockCollection<TOrder, TDirection>`.
- `BasicBlock.Count` at `ILGPU/IR/BasicBlock.cs:354` returns the int count of values in the block.
- `LoopSpecializer` already calls `ComputeOrderedBodyBlocks()` at `LoopUnrolling.cs:165`, so I am not adding a new traversal pass - just measuring during the already-existing body iteration.

And extend `ComputeUnrollFactor` with a `bodyCost` parameter plus a total-code-size cap:

```csharp
private const int MaxTotalUnrolledNodes = 1024; // tune — see below

private static (int unrolls, int iterations) ComputeUnrollFactor(
    int tripCount,
    int maxUnrollFactor,
    int bodyCost)
{
    // Fully unroll small loops IF the total unrolled body fits.
    if (tripCount <= maxUnrollFactor && bodyCost * tripCount <= MaxTotalUnrolledNodes)
        return (tripCount, 1);

    // Probe divisors, reject any that would produce too much code.
    for (int unrolls = Math.Min(maxUnrollFactor, tripCount); unrolls > 1; unrolls >>= 1)
    {
        if (tripCount % unrolls > 0) continue;
        if (bodyCost * unrolls > MaxTotalUnrolledNodes) continue;
        return (unrolls, tripCount / unrolls);
    }

    return (1, tripCount);
}
```

## Cap tuning

Target the cap so that:

- RadixSort Half body (estimated ~10 nodes) × 64 = 640 ≤ cap → preserves current full-unroll behavior on `AlgorithmRadixSortPairsHalfTest` (correctness).
- `LocalMemoryReproKernel256` body (estimated ~20 nodes) × 64 = 1280 > cap → partial-unroll drops to 32 or 16. At 32, body_cost × unrolls = 640, fits. Generated WGSL ~1/2 of current 132 KB. Chrome validator should digest it.
- Wasm radix sort hot loops: keep existing behavior. These should have body costs small enough that 64x fits within the cap.

**First trial cap: 1024.** If the Half sort regresses at this cap, the cap is too low (Half loop body is probably larger than 10 nodes). Raise it in 256-node increments. If N=256 WGSL still times out, lower it in 256-node increments until Chrome validator stays under 30 s.

## Verification plan before committing

**Do NOT commit the body-size heuristic without all of the following green (order matters):**

1. `dotnet build SpawnDev.ILGPU.slnx -c Release` clean (no new IR node-count API surface broken).
2. PlaywrightMultiTest filter `~LocalMemoryRepro` green on all 7 backend variants (CPU, CUDA, OpenCL, Wasm, WebGPU, WebGPUNoSubgroups, WebGL-skip-only). This proves the N=256/N=1024 fix.
3. PlaywrightMultiTest filter `~RadixSort` green on all 7 variants. **This is the Rule #1 gate** - the fix must not re-introduce the Half-packed sort regression or Wasm radix timeouts.
4. PlaywrightMultiTest filter `~ScanKernel|~AlgorithmReduce|~AlgorithmInclusiveScan|~AlgorithmExclusiveScan` green. These are the other heavy unrolling consumers.
5. **Optional: full PlaywrightMultiTest sweep.** Takes ~20 min. Covers Tests1-Tests10 etc. If the filter sweeps above are all green, this is confirmation, not hunting.

**If any of 2-4 goes red: revert immediately.** Do not push. Iterate in a local branch.

## Non-goals

- Do not change `DefaultMaxUnrollFactor = 64`. That knob is consumed by existing partial-unroll math and the value is load-bearing for at least the RadixSort Half case.
- Do not add a backend-specific override (WebGPU-only lower cap, etc.). The unroller is backend-agnostic by design; the fix should be too. If downstream compilers (CUDA, OpenCL) need a different cap, that is a separate issue.
- Do not ship without completing the full regression sweep above. The previous attempt (skipping partial unroll above threshold) was filter-green but broken in the broader sweep. IR-wide changes need IR-wide tests.

## Status

- Revert of naive fix: commit `714f2ab`, pushed to origin/master.
- LoopUnrolling.cs is at upstream HEAD again.
- LocalMemoryRepro N=256/N=1024 remain red on WebGPU + Wasm (accurate state).
- Pending: body-size heuristic implementation + full sweep verification. Next session.

## 2026-04-24 late-evening tuning attempt (reverted)

Implemented the body-size heuristic as designed in this plan. Ran diagnostic
instrumentation on the desktop console to measure actual body costs:

| Kernel (loop) | tripCount | bodyCost | Original (64) unroll total |
|---|---|---|---|
| `CPURadixSortKernel1_114` (3 inner loops) | 3, 4, 4 | 7, 9, 8 | 21, 36, 32 |
| `LocalMemoryReproKernel256` loop 1 | 256 | 15 | 15*64=960 |
| `LocalMemoryReproKernel256` loop 2 | 256 | 10 | 10*64=640 |

**cap=1024 attempt:** no-op for the N=256 kernel (both loops fit: 960 ≤ 1024
and 640 ≤ 1024, so partial-unroll stays at 64x × 4 outer iters). WGSL size
unchanged at 132 KB → Chrome validator still times out at 30 s.
`LocalMemoryRepro` filter: 4 failed, 13 passed (same as the revert state).

**cap=256 attempt:** forced N=256 loops down to 16x × 16 outer iters.
`LocalMemoryRepro` filter: **20 passed / 1 failed (WebGL N=64 architectural
ceiling, pre-existing) / 0 skipped in 45 s.** N=256 + N=1024 green on
WebGPU + WebGPUNoSubgroups + Wasm.

**But the full RadixSort sweep regressed with cap=256:** 8 failures / 179
passed in 25 m 33 s. Failures:

- `AlgorithmRadixSortPairsHalfTest` on WebGPU, WebGPUNoSubgroups, and **OpenCL**
  (correctness - Half-packed sort produces wrong output when the inner
  radix loop drops below some unroll factor I could not pin down from CPU
  diagnostic alone).
- `WasmTests.RadixSortThresholdProbeTest` / `RadixSortDescendingWithSentinelsTest`
  / `RadixSortRepeatedResortTest` / `RadixSortSpawnSceneSimulationTest` /
  `RadixSortDescending1_4MTest` (perf regression - Wasm large sorts
  (1.4M+ elements) exceed 2 min timeout because the radix inner loop's
  unroll factor drops too low to amortize dispatch/barrier overhead).

Conclusion: the cap has to be BETWEEN 256 and 1024, AND the WebGPU-side
Half-packed radix inner loop has some backend-specific path with a
bodyCost or tripCount that desktop-side measurement does not reveal. My
instrumentation on `CPURadixSortKernel1_114` showed only small trip counts
(3-4, body 7-9) - the ones that got clobbered at cap=256 live in a
separate code path my diagnostic did not catch.

## What tomorrow's iteration needs

Before the next cap attempt:

1. **Instrument the browser side.** Extend the diagnostic to emit to a
   file (from Blazor WASM, `Debug.WriteLine` -> browser console ->
   Playwright captures it into the per-test output folder; or write to
   a side file via `File.AppendAllText` from the desktop-only code
   paths reachable from WebGPU backend compile).
2. **Run `AlgorithmRadixSortPairsHalfTest` on WebGPU + OpenCL specifically
   with instrumentation.** Capture every loop's `(tripCount, bodyCost,
   unrolls, iterations)` tuple. Find the loop(s) where my cap=256 forced
   a smaller unroll and which broke. That reveals the minimum unroll
   factor the Half-packed sort requires for correctness.
3. **Also instrument `RadixSortDescending1_4MTest` on Wasm** to find the
   large-element-count loop that needs the full 64x unroll for perf.
4. **Only then pick a cap.** The cap must simultaneously satisfy:
   - Force partial-unroll on the N=256 `LocalMemoryReproKernel256` (body
     15, at cap must drop from 64x to at most 32x to fit Chrome
     validator budget).
   - Preserve the Half-sort correctness loop's minimum unroll factor.
   - Preserve the Wasm large-sort perf loop's unroll factor (likely
     needs to stay at 64x).

If those three constraints are mutually satisfiable with a single global
cap, we ship that cap and rc.11 lands. If they conflict (e.g. Half-sort
needs 64x on a body that N=256 also has bodyCost >15 for, so no single
cap works), the right answer may be backend-specific caps or a bodyCost
heuristic that weights some IR-node types differently.

## Outcome of 2026-04-24 session

LoopUnrolling.cs restored to upstream HEAD. Tests for N=256 and N=1024
remain red on WebGPU + Wasm (accurate state per Rule #1). The plan above
now carries the measured data needed to land the real fix on the next
attempt with clear success criteria and no more blind cap-guessing.
