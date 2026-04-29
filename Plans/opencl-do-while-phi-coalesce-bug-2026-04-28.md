# OpenCL backend: `u = v` inside do-while aliases u to the phi `v`, breaking after-loop reads

**Date:** 2026-04-28
**Severity:** correctness — multi-call decode loops on OpenCL produce wrong values
**Reporter:** Tuvok (via Av1RangeCoderGpu_CdfQ15_RoundTrip OpenCL failure)
**Reproducer minimal:** `D:\users\tj\Projects\SpawnDev.Codecs\SpawnDev.Codecs\_dump_opencl\Min.cs`
**Workaround in place:** `Av1RangeDecoderGpu.DecodeCdfQ15` uses `u = v ^ 0u;` to break the alias
**Status:** WORKAROUND LANDED in SpawnDev.Codecs; proper fix pending in ILGPU OpenCL backend

## Symptom

Pattern that fails on OpenCL but passes on CUDA + CPU:

```csharp
uint v = r;
uint u;
int ret = -1;
do
{
    u = v;                // <-- aliased to phi(v) on OpenCL
    ret++;
    uint icdfVal = icdf[icdfBase + ret];
    v = ((r >> 8) * (icdfVal >> 6)) >> 1;
    v += (uint)(EcMinProb * (N - ret));
} while (c < v);

uint diff = u - v;       // <-- on OpenCL: reads NEW v, gives 0
```

After the loop exits, `u` should equal the value of `v` from the iteration BEFORE the exit iteration (i.e. the v that satisfied `c < v` for one last time, then was overwritten by a v that no longer did). On OpenCL, `u` instead equals the FINAL `v`, making `u - v == 0`. The downstream `Normalize` then produces wrong state, and subsequent decodes return wrong symbols.

Tuvok's specific symptom: `sym[1]: input=1 decoded=0` in `Av1RangeCoderGpu_CdfQ15_RoundTrip_AllBackends`.

## Minimal isolated reproduction

`Min.cs` exercises just the loop with hardcoded state. Sweep across `(dif, rng, N)`: 5 / 1632 cases diverge between CPU and OpenCL, with `u_CPU != u_OpenCL` and `v_CPU == v_OpenCL`. Replacing `u = v;` with `u = v ^ 0u;` (semantically identical) closes ALL 6528 cases (sweep extended to N=1..16). Confirms the divergence is in OpenCL's handling of the trivial copy `u = v`, not in the arithmetic or the loop control.

## Hypothesis

ILGPU's OpenCL backend variable allocator (or copy-propagation pass) coalesces SSA values when one is just a copy of another. Specifically:

- `u = v` inside the loop body, where `v` is a phi value at the loop top, gets emitted as `u_var = v_var` (or worse, u and v are aliased to the same OpenCL local).
- When the loop's back-edge phi update assigns `v_var = v_new` (the next iteration's v), the same variable that `u` aliased to gets overwritten.
- After the loop exits, `u - v` reads the NEW v from both u_var and v_var.

CUDA + CPU don't exhibit this. PTX backend likely treats SSA copies differently (PTX is register-based and may emit copies as moves), and CPU runs the IR through .NET's JIT which preserves the local-variable semantics.

The existing phi-swap intermediate-temp logic in `CLCodeGenerator.cs:506-548` handles `phi_A = phi_B` swaps, but does NOT handle `non_phi_X = phi_B` where phi_B is reassigned later — because `Add(phi, value)` only marks `value` as Intermediate when value is itself a phi (`PhiBindings.cs:250`).

The f04a63e fix (`Clear intermediate phi variables from previous blocks`) addressed a related but distinct bug (issue #1539); this is a separate manifestation that f04a63e doesn't cover.

## Proper fix candidates (ILGPU OpenCL backend)

1. **Mark a phi as Intermediate when ANY non-phi assignment reads it AND the phi gets reassigned in the same block.** Requires scanning each block for assignments-from-phi and tracking which phis are reassigned via the back-edge.

2. **Disable copy-propagation for SSA copies whose source is a phi that's reassigned in the same block.** Less invasive; can be done at the variable allocator level.

3. **Always allocate a separate variable for SSA copies of phi values.** Conservatively safe; may increase variable count slightly.

Best path is probably (3) for simplicity and safety. Touched code: `ILGPU/Backends/OpenCL/CLCodeGenerator.cs` block-emit loop or `CLVariableAllocator.cs` allocate-or-coalesce logic.

## Workaround in SpawnDev.Codecs (SHIPPED)

`Av1RangeDecoderGpu.DecodeCdfQ15` line 142:

```csharp
u = v ^ 0u;  // forces a distinct SSA value, bypassing OpenCL coalesce
```

Bit-equivalent on every backend; `^ 0u` is a no-op semantically but produces a computed expression rather than a trivial copy. CPU + CUDA + OpenCL all GREEN on the canonical round-trip test (3/3 PASS).

This is a Rule 2 workaround: documented `// WORKAROUND:` comment, this Plan tracks it, Captain has been informed via DevComms. Should be reverted once the proper ILGPU fix lands.

## Other consumers potentially affected

The same pattern likely fails wherever a do-while-with-loop-carried-snapshot is used:

```csharp
do { snapshot = current; current = mutate(); } while (cond_on_current);
result = combine(snapshot, current);
```

Audit candidates:
- `Av1CoefDecoderGpu` — has 4/5 OpenCL test failures in `Av1CoefDecoderGpu_RoundTrip_*` after this workaround. Different symptom (`[decEob] Expected '1' but got '2'`), but might be the same bug class somewhere else in that path. Needs investigation.
- Any GPU-side range/arithmetic coder with `u = current_v` pattern.

## Next steps

- [ ] Build a unit test in `BackendTestBase.Tests1` that exercises this exact pattern with a tiny do-while + post-loop snapshot read. Should fail on OpenCL pre-fix, pass on every other backend.
- [ ] Implement fix candidate (3) in `CLCodeGenerator.cs` / `CLVariableAllocator.cs`.
- [ ] Re-run the regression test + full backend sweep, confirm GREEN across all 7 backends.
- [ ] Revert the `^ 0u` workaround in `Av1RangeDecoderGpu.DecodeCdfQ15` once the fix is in.
- [ ] Investigate `Av1CoefDecoderGpu_RoundTrip_*` 4/5 failures - confirm same bug class or separate.
