# Wasm Backend — Stable-Cut Gates for SpawnDev.ILGPU 4.9.2

**Status:** rc.20 ships WebGPU/WebGL/CPU/CUDA/OpenCL fixes. Wasm path remains blocked on the issues below. **4.9.2 stable does not ship until every gate here is closed.** Captain's "no fake stable with KNOWN BUGS" rule from 2026-04-25.

Riker + Data are investigating the Wasm spin-wait architectural fragility on a fork at `D:\users\tj\Projects\SpawnDev.ILGPU.RikerWasmFork`. This doc is the verification list for when their fork merges back.

## Gate 1 — Bug E: Wasm iDCT 16x16 bit-exact divergence

**Symptom:** Tuvok's `Vp9Idct16x16Kernel` Wasm tests fail bit-exact comparison vs CPU oracle:

| Test | Expected mismatches | Observed |
|------|--------------------|----------|
| `Vp9Idct16x16Kernel_ZeroCoefficients_LeavesPredictorUnchanged` | 0 | 0 (PASS — trivial all-zero) |
| `Vp9Idct16x16Kernel_DcOnly_MatchesReference` | 0 | **256** (FAIL — full block wrong) |
| `Vp9Idct16x16Kernel_RandomInputs_BitExactMatchReference` | 0 | **253** (FAIL) |
| `Vp9Idct16x16Kernel_BatchedDispatch_AllBlocksMatchReference` | 0 | **1019** (FAIL — 4 blocks worth) |

Bit-identical at rc.13, rc.15, rc.16, rc.17, rc.18, rc.19, rc.20 — bug has been there since well before today's rc churn. It is NOT regressed by any rc.* fix.

**In-repo repros that exercise the same path on Wasm:**
- `BackendTestBase.Tests6.cs::NoInliningVoidHelperEmitsFunctionCallTest` — minimal int+ref-int void helper. Wasm fails with `Expected (148, 57), got (0, 0)` — the helper doesn't actually run, output is uninitialized scratch.
- `BackendTestBase.Tests6.cs::NoInliningIdct16RowShapeHelperBitExactTest` — 16 short + 16 out int helper signature, simple `i*2 + n` body. Wasm fails the same silent-zero pattern.
- `BackendTestBase.Tests6.cs::NoInliningIdct16RowQ14NarrowHelperBitExactTest` — same signature with the Q14 narrowing arithmetic Tuvok's `Idct16Row` uses. Wasm fails.
- `BackendTestBase.Tests6.cs::NoInliningIdct16RowQ14StressHelperBitExactTest` — Tuvok's exact random input range with seed `0xADA51610`.
- `BackendTestBase.Tests6.cs::NoInliningIdct16RowQ14MultiCallHelperBitExactTest` — helper called twice (mirrors row+col pass).

**Probable root cause:** the Wasm helper-inline path in `WasmKernelFunctionGenerator.GenerateCode(MethodCall)` has a defect specific to multi-block helpers with `out`/`ref` int parameters. The current `_localMap[paramKey] = argLocal;` aliasing should pass the alloca's scratch address to the helper body's stores, but somewhere in the chain the writes aren't landing in the caller's alloca.

Full diagnosis pending a `WasmDumpPath` desktop run that captures the actual `.wat` for one of the failing kernels. Not blocking Riker + Data's spin-wait work but the right next step once they're back.

**Verification when fix lands:**

```
dotnet test PlaywrightMultiTest/PlaywrightMultiTest.csproj -c Release \
  --filter "FullyQualifiedName~WasmTests.NoInliningIdct16RowQ14NarrowHelperBitExactTest|FullyQualifiedName~WasmTests.NoInliningVoidHelperEmitsFunctionCallTest|FullyQualifiedName~WasmTests.NoInliningIdct16RowQ14StressHelperBitExactTest|FullyQualifiedName~WasmTests.NoInliningIdct16RowQ14MultiCallHelperBitExactTest|FullyQualifiedName~WasmTests.NoInliningIdct16RowShapeHelperBitExactTest"
```

Expected: 5/5 PASS bit-exact. After that Tuvok re-runs `~Vp9Idct16x16Kernel` filter on Wasm — expecting 4/4 PASS (vs 1/4 today).

## Gate 2 — Spin-wait fragility under CPU contention

**Symptom:** Wasm tests intermittently time out or produce wrong output when the host machine is under CPU load (parallel test runs, browser background tabs, other testhosts).

**Why it gates stable:** users running multi-tab browser apps, agents running parallel tests, or anything else that shares CPU cycles will hit this. From the consumer's perspective the library "works sometimes" which per Captain's "flaky is buggy" rule is a bug.

**In-flight investigation:** Riker + Data on `SpawnDev.ILGPU.RikerWasmFork`. Recent crew DevComms:
- `data-to-riker-wasm-dump-analysis-bisect-discriminator-2026-04-25.md`
- `data-to-riker-bisect-state-fails-3of3-2026-04-25.md`
- `data-to-riker-fence-claim-correction-2026-04-25.md`
- `data-to-riker-wasm-repros-rerun-empirical-2026-04-25.md`

**Verification when fix lands:** full PMT sweep with concurrent activity (Tuvok running Codecs sweep in parallel, Riker running WebTorrent sweep) — expecting Wasm tests to maintain stable pass/fail counts independent of contention. No flakies.

## Gate 3 — Wasm-side fn-def emission for NoInlining helpers (deferred)

**Status:** WGSL + GLSL got fn-definition emission for `[MethodImpl(NoInlining)]` helpers in rc.18. Wasm did not — it still inlines via the existing helper-inline path, which is what triggers Bug E above.

**Decision deferred:** if Bug E is fixed by Riker + Data via the existing inline path (more likely, based on their spin-wait focus), Wasm fn-def emission stays a future improvement (rc.21+ or 4.9.3). If Bug E turns out to require fn-def emission, then this gate merges with Gate 1.

## What's NOT a gate (working as intended)

These rc.13-rc.20 changes are NOT pending Wasm verification — they're already shipped and verified across non-Wasm backends. Just listing here for completeness:

- WGSL+GLSL fn-def emission for NoInlining helpers (rc.18+)
- Sub-word narrowing in WGSL/GLSL/Wasm `ConvertValue` (rc.18+)
- Loop-unroll body-cost cap on small-trip-count loops (rc.20)
- AddressSpaceCast inline-aliasing for kernel-side allocas (rc.18+)
- WGSL/GLSL `Declare()` skip for ptr types via `let` form (rc.16+)

These already work on Wasm where applicable. The Wasm-specific gates above are the ONLY ship blockers.

## Trigger conditions

When Riker + Data declare the Wasm fork ready to merge:

1. They post a "Wasm fork ready for merge" DevComms.
2. Geordi (or whoever's the ILGPU editor at that point) merges the fork branch into `master`.
3. Run the Gate 1 + Gate 2 verifications above.
4. If both pass, cut `4.9.2-rc.21` (or whatever the current rc.* number is) with the merge.
5. Tuvok verifies his full Codecs filter on Wasm.
6. If clean, cut `4.9.2` stable.

## See also

- `Plans/rc16-fn-def-codegen-harden.md` — historical context on the WGSL fn-def codegen work
- `Docs/kernels.md#helper-methods-and-inlining` — public API doc for `[NoInlining]` (added rc.20)
- `nuget-local-publish-log.md` — full rc.* progression log
- `feedback_flaky_is_buggy.md` (memory) — the rule that gates this whole list
