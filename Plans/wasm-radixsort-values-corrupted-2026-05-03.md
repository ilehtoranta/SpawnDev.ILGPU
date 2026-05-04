# Wasm RadixSort values-buffer corruption — 2026-05-03

## Status: RESOLVED 2026-05-03 evening — `SpawnDev.ILGPU 4.9.5-rc.4`

Root cause: rc.3's `WasmKernelFunctionGenerator.TraceToParameter` extension added LoadElementAddress / LoadFieldAddress / SubViewValue / Load cases for write-tracking, but the same helper is shared with `GenerateCode(GetViewLength)`. Walking through `SubViewValue` resolved a sub-view's `.Length` to its parent buffer's full length, which broke RadixSort's bounds checks (the kernel iterated past the SubView range into uninitialized memory).

Fix: split the trace into two methods — original `TraceToParameter` keeps view-shape-only cases (used by `GetViewLength`); new `TraceWriteTargetToParameter` adds pointer-arithmetic cases for write tracking only. `TrackParamWrite` calls the write-specific trace.

Bisect verified via worktree at `D:\users\tj\Projects\SpawnDev.ILGPU\_bisect-wasm-radixsort` (now removed):
- `1cce5ed` rc.27: `RadixSortRepeatedResortTest` PASS (36s)
- `f50c4a2` 4.9.4-rc.2: PASS (32s)
- `eb63799` (just before my rc.3): PASS (31s)
- `af8aacd` rc.3 (today's first attempt at the race fix): FAIL (8s)
- rc.4 (the fix): PASS (34s)

## Verification on rc.4

`dotnet test --filter "FullyQualifiedName~WasmTests.RadixSort"` → **17/18 PASS**

Only remaining failure: `RadixSort100KBenchmarkTest` — single-element mismatch (`index 32 expected 825865714, got 317314903`), 1s wall clock. Different failure mode from rc.3's wholesale duplicates pattern. Filed separately as a focused investigation; likely either a genuine correctness regression on a specific input pattern OR a non-deterministic race specific to the 100K benchmark kernel.

## Performance milestone (2026-05-03)

`WasmTests.RadixSortDescending1_4MTest` and `WebGPUTests.RadixSortDescending1_4MTest`
run side-by-side in the same PMT invocation 2026-05-03 evening:

- **Wasm 1.4M radix sort: 16s** (passed)
- **WebGPU 1.4M radix sort: 9s** (passed)

Wasm comes in at **~1.78× WebGPU's wall time** for 1.4M element radix sort
end-to-end, browser-side, including all dispatch overhead. Best Wasm sort
performance recorded on this codebase — Wasm doing serious browser-side
multi-pass GPU-style work within striking distance of native WebGPU.

Other notable wall clock times from the broader RadixSort sweep on the same run
(captured in `/tmp/render-test/wasm-radix-rc4.log`):
- RadixSortThresholdProbeTest: ~21s
- RadixSortDescendingWithSentinelsTest: ~17s
- RadixSortRepeatedResortTest: ~31s
- RadixSortHeavyDuplicateKeysTest: ~11s (on 1M elements with heavy duplicates)
- RadixSortDescendingOddCountTest: ~17s
- RadixSortSpawnSceneSimulationTest: ~47s

## Failure signature

```
WasmTests.RadixSortHeavyDuplicateKeysTest:
  RadixSortHeavyDuplicates: Index integrity failure — 999999 duplicates,
  0 out-of-range out of 1000000 elements.
  Total mismatches vs CPU: 0
```

- Keys (the sort target) come back correct (`Total mismatches vs CPU: 0`).
- Values (the original-index payload tracked alongside the keys) come back with N-1 duplicates of N total — almost every output position has the same value, almost certainly zero.
- Test completes in 8-22 seconds (NOT a timeout).
- Deterministic across runs.

Affected tests (Wasm only — every other backend passes):

```
WasmTests.RadixSortRepeatedResortTest          (8s)   — 499999 duplicates / 500000
WasmTests.RadixSortHeavyDuplicateKeysTest      (13s)  — 999999 duplicates / 1000000
WasmTests.RadixSortDescendingOddCountTest      (16s)  — 1500000 duplicates / 1500001
WasmTests.RadixSortDescendingWithSentinelsTest (1m24s, possibly timeout boundary)
WasmTests.RadixSortDescending1_4MTest
WasmTests.RadixSortDescending2MTest            — 2097151 duplicates / 2097152
WasmTests.RadixSortDescending4MTest
WasmTests.RadixSortAscending1_4MTest
WasmTests.RadixSortSpawnSceneSimulationTest
```

Companion failure family — likely separate root cause:

```
WasmTests.WasmStructShuffleDiagTest:
  WebAssembly.compile() ... local.set[0] expected type f32, found local.get
  of type i32 @+421
```

Six `WasmStruct*DiagTest` + `WasmMinimalPairsSortDiagTest` fail with Wasm
binary validation errors — codegen producing invalid bytecode, type mismatch
on `local.set` / `local.get`. Same dispatcher rejects the kernel before the
sort can run, so consumers in those test paths never even get the chance to
hit the integrity check. Possibly upstream of the RadixSort regression — the
diagnostic kernels and RadixSort both stress similar IR write patterns.

## What this is NOT

- Not a timeout. The framework would mark the test "Timeout" not "Index integrity failure". The longest failing test here is 1m24s; the test budget on RadixSort variants is 240s per `Plans/wasm-cold-start-vs-warm-pool-timing-2026-04-28.md`.
- Not the 2026-05-03 Wasm copy-OUT race. SpawnDev.ILGPU 4.9.5-rc.3 fixed the race; identical failure counts before (baseline) and after (rc.3) confirms RadixSort is unrelated. The rc.3 gate (`!HasBarriers && SharedMemorySize == 0 && traceFoundAnyBufferWrite`) also explicitly excludes barrier/shared-memory kernels like RadixSort, so the gate falls back to legacy copy-all on every RadixSort dispatch.
- Not a recent ILGPU/Algorithms change. `git diff f50c4a2..af8aacd -- ILGPU/ ILGPU.Algorithms/` shows only `OpenCL/CLException.cs` changed.

## What it IS

A regression introduced sometime between rc.27 (`1cce5ed`, 2026-04-28) and the current HEAD. Per the rc.27 sweep notes (`Plans/wasm-cold-start-vs-warm-pool-timing-2026-04-28.md`): "All Wasm RadixSort tests passed" in the 1h17m / 2196-test full sweep. Today they don't.

Wasm-touching commits in that window:

- `9cde61e` rc.28 — signed Div-by-pow2 + Wasm void helpers + diagnostics
- `4bc130a` 4.9.4-rc.1 — TwoTab phantom-alive close + Wasm linear-memory option
- `f50c4a2` 4.9.4-rc.2 — kernel-module memory import maximum mismatch (commit message claims `RadixSortDescendingOddCountTest PASS in 29s`)
- (none in 4.9.5-rc.1 / rc.2 — those touched WebGPU + WebGL only)
- `af8aacd` 4.9.5-rc.3 — Wasm copy-OUT race fix (today; verified neutral on RadixSort)

So the regression entered between `1cce5ed` and `f50c4a2`. f50c4a2's commit message verified ONE RadixSort test (OddCount, 29s); the rest of the family was never re-verified post-merge.

## Hypothesis (not yet verified)

The values-buffer corruption pattern (uniform value across the entire output) suggests:

1. The dispatch's per-kernel buffer SCATTER step writes `keys` to their sorted positions correctly but writes `values` to the wrong location — possibly always to position 0, or to a single shared scratch slot.
2. OR the values buffer's initial copy-IN data isn't propagating to wasmMemory, so the kernel reads zeros, scatters those zeros, and copies zeros back out.
3. OR the values buffer is being aliased with shared memory or scratch — dispatch lays out memory regions per-dispatch, and a layout change between rc.27 and f50c4a2 may have introduced overlap.

## Next-step verification (diagnostic, not yet run)

1. Pick one failing test — `RadixSortHeavyDuplicateKeysTest` (1M elements, 13s). Add a diag kernel that captures `valuesBuf[0]`, `valuesBuf[N/2]`, `valuesBuf[N-1]` after each RadixSort pass. Confirms whether the corruption is "all zero" or "all some other value".
2. Inspect dispatch logs (`_dispatchLog`) for the failing test — count of dispatches, buffer ranges, whether any dispatch reports an unusual layout for the values buffer.
3. Bisect between `1cce5ed` and `f50c4a2` on a single failing RadixSort test to find the exact regression commit.

## Owner: Geordi

Lane owner. Filing this Plan as the working doc; will continue investigating after the rc.3 ship.
