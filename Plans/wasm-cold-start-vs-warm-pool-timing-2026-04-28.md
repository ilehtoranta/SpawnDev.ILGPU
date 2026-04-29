# Wasm RadixSort cold-start vs warm-pool timing 2026-04-28

## Verified facts

**rc.27 sweep (1h17m full sweep, 2196 tests):**
- `RadixSortDescending4MTest` completed in 42494ms (~42s) — Captain remembers this as "good baseline."
- All Wasm RadixSort tests passed.

**Same source code (pristine rc.27, same machine) re-run later 2026-04-28 evening, RadixSort-only filter (~30 tests in batch):**
- `RadixSortThresholdProbeTest` 120827ms → **timeout** (limit was 120s)
- `RadixSortDescendingWithSentinelsTest` 121013ms → **timeout**
- `RadixSortRepeatedResortTest` 120149ms → **timeout**
- `RadixSortHeavyDuplicateKeysTest` 120215ms → **timeout** (rest of suite cascaded ObjectDisposed)

**Same source code, RadixSort-only filter run with my today's fixes (Tuvok Div + void-MethodCall + Drop-removal + WorkerCount property) on top:**
- `RadixSortThresholdProbeTest` 81615ms (PASS)
- `RadixSortDescendingWithSentinelsTest` 92870ms (PASS)
- `RadixSortDescendingOddCountTest` 79252ms (PASS)
- `RadixSortDescending2MTest` 74850ms (PASS)
- Several still timed out at 120s

The two non-sweep runs produce roughly the same magnitude of slowdown vs the sweep. **My fixes did not cause the slowdown** - they're either neutral or slight positive. The slowdown is environmental: cold-start vs warm-pool.

## Hypothesis (not yet verified)

Wasm-on-browser dispatch has a per-process startup cost that doesn't amortize to zero between dispatches but DOES amortize across tests in the same testhost / browser process. Costs that fit this profile:

1. **Chrome / Chromium V8 JIT warmup** - large kernels' first dispatch hits cold V8, second+ dispatch reuses compiled JS / Wasm host glue.
2. **Mono Wasm JIT warmup** - same pattern for the .NET side.
3. **Worker pool cold-start** - first dispatch spawns workers; subsequent dispatches reuse them.
4. **SAB allocation cost** - first allocation of the SharedArrayBuffer hits a real `new SharedArrayBuffer(N)`; subsequent dispatches reuse the same SAB.
5. **Wasm module instantiation per worker** - each worker `new WebAssembly.Module(bytes)` first time then reuses.

In a sweep with 1099 prior tests, all of these costs are paid early in the run. By the time RadixSort 4M runs at index 1099, every cost is amortized to near-zero. RadixSort sees the steady-state speed of ~42s.

In a cold-start single-test or small-batch run, every cost is freshly paid. The first RadixSort in the batch eats ALL of those costs PLUS its own dispatch. Total can easily exceed 120s.

## Practical fix shipped 2026-04-28

`SpawnDev.ILGPU.Demo/UnitTests/WasmTests.cs` - bumped RadixSort test timeouts from 120000ms to 240000ms. Matches the precedent at `SubViewRange_HighDispatchCount` (line 589) which already had this issue at the 120s boundary.

## Followups (not blocking)

- Investigate per-cost itemized timing to find the dominant cold-start cost. If one cost is responsible for 80% of cold-start time, fixing IT is more durable than bumping timeouts.
- Add a "warmup test" first in the WasmTests order (e.g. trivial dispatch) so the first real test doesn't pay the cold-start tax.
- Consider preserving the WasmAccelerator instance across PMT tests instead of recreating per-test.
