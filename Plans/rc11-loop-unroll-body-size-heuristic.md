# rc.11 — IR LoopUnrolling body-size heuristic

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
