# rc.16+ — WGSL/GLSL fn-definition codegen hardening (resumes the rc.14 attempt)

## Context

rc.14 (`commits 1cb4f6c + 08892f4`) attempted to ship WGSL/GLSL standalone-fn emission so that methods skipping IR-level inlining (`[MethodImpl(NoInlining)]`, body-size cap) compile as real fn definitions instead of being inlined 32x at codegen. The motivating consumer is Tuvok's `Vp9Idct16x16Kernel` — `Idct16Row` is ~500 IL instructions called 32x; rc.13 inlines the whole thing into a ~3800-line straight-line WGSL body, which pushes Chrome's WGSL validator + V8 AsyncFunction compile path past 30s per kernel ("intermittent 1m17s" on rc.13).

Tuvok's verification of rc.14 (`tuvok-to-geordi-rc14-verification-red-2026-04-25.md`) caught a surface bug in WGSL signature emission. Investigation surfaced a layered set of body-side codegen gaps. rc.15 (commit `7dc77cb`) reverted the rc.14 fn-call infrastructure to rc.13 inline-at-codegen baseline. This plan captures what was learned and what needs to ship in rc.16+.

## What surfaced during rc.14 verification

### Bug A: WGSL signature emitter (one-line, FIXED in rc.15 defensively)

`WGSLFunctionGenerator.GenerateHeaderStub` unconditionally appended `-> {ret_type}`, producing `fn name(...) -> void { ... }` for void helpers. WGSL grammar requires no return-type clause for void fns. Fixed in commit `7dc77cb` even though the path is dormant (defensive; harmless under inline-only).

### Bug B: WGSL body codegen cross-scope identifier resolution (NOT FIXED)

After Bug A is patched, a void-helper smoke test (`int + ref int` params) hits:

```
[WebGPU] Shader 'main' failed validation: Error while parsing WGSL: :68:5
error: unresolved value 'v_11'
```

Hypothesis: `WGSLFunctionGenerator.GenerateCode` calls `GenerateCodeInternal()` (shared base method with the kernel codegen). The base codegen tracks `valueVariables` and `varCounter` in `_args` (the shared `GeneratorArgs`). When the kernel codegen runs first and assigns name `v_11` to some kernel-scope IR Value, then the function codegen runs and references `v_11` from inside the fn body, the WGSL output has the reference but the `let v_11 = ...;` declaration was emitted in the kernel body (not the fn body) — so the WGSL parser sees an unresolved identifier inside the fn.

**Fix shape:** the function generator needs a clean codegen scope. Either:
- Snapshot `valueVariables` / `varCounter` at fn-generator entry and restore at exit (so the kernel never sees fn-internal value names);
- Or use a separate `valueVariables` map per function generator instance (current code shares one across kernel + all helpers via `_args`);
- Or pre-emit `let v_X = ...;` declarations for any IR Value the fn body references that doesn't have a fn-scope origin.

Investigation needs to check what `GenerateCodeInternal` does for IR Values that are kernel parameters vs fn parameters. Tuvok's `Idct16Row` has 32 `i32` parameters; my smoke test had 5 (`int a, b, c, ref int sumOut, ref int diffOut`). The bug shows up when the body produces intermediate values whose names are picked from the shared counter; need to confirm by dumping the WGSL for the failing case.

### Bug C: GLSL body codegen int/float type assigns (NOT FIXED)

Same void-helper smoke test on WebGL hits:

```
ERROR: 0:242: 'assign' : cannot convert from 'highp int' to 'const highp float'
ERROR: 0:245: 'assign' : cannot convert from 'highp int' to 'const highp float'
...
ERROR: 0:259: 'assign' : cannot convert from 'const highp float' to 'highp int'
```

Hypothesis: similar shared-state issue, but the GLSL backend emits `target = type(0);` for "unmapped" intermediate values when they appear in the fn body before the kernel has assigned them their real type. The "const float" comes from a default expression at one site and the "int" comes from the actual IR target type — they don't match.

**Fix shape:** likely the same scope-snapshot fix as Bug B for `GLSLCodeGenerator.GenerateCode(MethodCall)`'s fn-call branch. Plus possibly a pass that resolves intermediate-value types before emitting any assign.

### Bug D: ArrayView-as-fn-param marshaling (UNSPECIFIED)

When a NoInlining helper takes `ArrayView<int>` as a parameter, the WGSL fn signature has no clean way to express "pointer to a storage-buffer-bound array slice" as a fn parameter. WGSL fns can take `ptr<storage, ...>` only with strict address-space qualifications. Wasm has the same issue but in a different form (Wasm helpers receive raw memory offsets, not view structs).

**Fix shape (deferred — not blocking Tuvok):**
- For WGSL: pass the kernel's `_subWordParams` / storage-buffer var by reference (ptr) along with an offset. Body re-creates a virtual view using offset arithmetic.
- For GLSL: not feasible cleanly under WebGL 2.0 — Transform Feedback varyings + uniforms don't lend themselves to fn-param passing. Likely stays as a known-not-supported case.
- For Wasm: pass the view's `_memoryOffset` + `_byteLength` as scalar i32s; body reconstructs.

This is rc.17+ scope. Tuvok's `Idct16Row` does NOT take views as params — only `int` + `ref int` outputs lowering to `ptr<function, i32>` — so Bug D is not on Tuvok's critical path.

### Bug E: Wasm narrowing fix may be incomplete for iDCT 16x16 (NOT FIXED)

rc.14 added `i32.extend16_s` / `i32.extend8_s` / `0xFFFF`/`0xFF` mask narrowing in `WasmCodeGenerator.GenerateCode(ConvertValue)` for Int16 / Int8 destinations. Tuvok's report says `Vp9Idct16x16Kernel` Wasm tests still fail bit-exact at 256 mismatches per 16x16 block (100% mismatch).

**Two suspects for rc.16 investigation:**
1. **Second narrowing site.** The `(short)int` cast is one IL pattern; there may be another (e.g. `arr[i] = (byte)((short)val + (short)other)` could lower to a narrowing path that doesn't go through `ConvertValue`). Need to dump the Wasm IR + matching `.wat` for the failing kernel and trace.
2. **Wasm helper inline path with 32-param signature.** `Idct16Row` has 32 parameters including `ptr<function, i32>` ref outputs in WGSL. On Wasm the equivalent is i32 pointers into per-thread scratch. `WasmKernelFunctionGenerator.GenerateCode(MethodCall)` inline path (line ~2871-2914) maps args to params via `_localMap` and `EmitGetLocal`. With 32 params + ref outputs, does the inline correctly route writes through to the caller's locals? Worth a focused inline-correctness test for ref outputs.

`AlgorithmGroupReduceHalfTest` passes on Wasm at rc.15, so the Wasm narrowing fix for the simple case works. The iDCT 16x16 bit-exact failure is something else.

## Critical files for rc.16

### Bug B (WGSL body codegen scope)

- `SpawnDev.ILGPU/WebGPU/Backend/WGSLFunctionGenerator.cs` — `GenerateCode()` at lines 133-170 needs scope snapshot/restore around `GenerateCodeInternal()`. Reuses `varCounter`, `valueVariables`, `declaredVariables` from base class — these need per-fn isolation.
- `SpawnDev.ILGPU/WebGPU/Backend/WGSLCodeGenerator.cs` — `valueVariables`, `varCounter`, `declaredVariables` field semantics. Decide: per-instance or shared?
- `SpawnDev.ILGPU/WebGPU/Backend/WGSLKernelFunctionGenerator.cs` — the helper inline path (which works correctly) takes a snapshot at line ~1700 and restores at line ~1810 (`savedVarCounter`, `savedEmittedLetBindings`). The fn-definition path needs the same shape.

### Bug C (GLSL body codegen)

- `SpawnDev.ILGPU/WebGL/Backend/GLSLCodeGenerator.cs` — `GenerateCode(MethodCall)` fn-call branch lines ~1666-1691. Same snapshot/restore issue, plus type-mismatch resolution.
- `SpawnDev.ILGPU/WebGL/Backend/GLSLFunctionGenerator.cs` — fn-def emission shape; mirror WGSL fix.

### Bug E (Wasm iDCT 16x16 bit-exact)

Investigation first; code change later. Files to start in:
- `SpawnDev.ILGPU/Wasm/Backend/WasmKernelFunctionGenerator.cs` — `GenerateCode(MethodCall)` inline path lines ~2871-2914 — verify ref-output param routing for 32-param `Idct16Row`-shape methods.
- `SpawnDev.ILGPU/Wasm/Backend/WasmCodeGenerator.cs` — verify that NO other narrowing site exists by greping for `BasicValueType.Int16` / `Int8` writes. Maybe `Store(StoreElementAddress)` for sub-word arrays needs a check.

## Reactivation steps

When the bugs above are fixed:

1. **Re-add the rc.14 fn-call infrastructure** (commit `7dc77cb` reverted these pieces; reactivate):
   - `WebGPUBackend.CreateFunctionCodeGenerator` — re-add `if (method.HasFlags(MethodFlags.Inline))` gate so non-Inline methods skip `HelperMethods` registration.
   - `WGSLKernelFunctionGenerator.GenerateCode(MethodCall)` — re-add the `EmitNonInlinedMethodCall` branch (the method body itself is already in source, dormant).
2. **Re-add the void-helper smoke test** (matches Tuvok's `Idct16Row` shape: `int` params + `ref int` outputs):
   ```csharp
   [MethodImpl(NoInlining)]
   private static void NoInliningVoidPairWriterHelper(int a, int b, int c, ref int sumOut, ref int diffOut) { ... }
   ```
3. **Verify smoke tests pass on all 7 backends** before re-emitting any non-Inline-flagged method as a fn definition.
4. **Tuvok's `Vp9Idct16x16Kernel` filter run** — expecting 5/6 backends green (WebGL stays at architectural skip), with WebGPU compile sub-second instead of "intermittent 1m17s".

## Why an `Inliner` body-size cap is also needed

The rc.14 Plan (`Plans/rc14-large-method-fn-definition-emission.md`) called for an `MaxNumILInstructionsToInlineDefault = 200` cap in `ILGPU/IR/Transformations/Inliner.cs`. Not landed in rc.14 because the WGSL codegen pieces went first and broke before the IR change was needed. **Land the cap in rc.16 alongside the codegen fix** so methods over 200 IL automatically un-inline at the IR level (consumers get the benefit without needing `[MethodImpl(NoInlining)]` annotations everywhere).

## Sequencing

- **rc.16 phase 1** — Fix Bug B (WGSL scope), Bug C (GLSL scope + types). Reactivate rc.14 fn-call infrastructure. Re-run Tuvok's filters. If green, ship.
- **rc.16 phase 2 OR rc.17** — Add `Inliner` body-size cap (200 IL).
- **rc.17+** — Bug D (view-as-fn-param marshaling). Optional unless a consuming kernel hits it.
- **rc.16 parallel investigation** — Bug E (Wasm iDCT 16x16). May land in rc.16 or rc.17 depending on what's found.

## Done-ness criteria

For rc.16 to ship as the proper fix to Tuvok's compile cliff:

1. Tuvok's `~Vp9Idct16x16Kernel` filter: 5/6 backends green, WebGPU compile sub-second (was "intermittent 1m17s" rc.13 / "0/4" rc.14 / "intermittent 1m17s" rc.15).
2. NoInliningVoidHelperEmitsFunctionCallTest (re-added): 7/7 backends green.
3. Existing NoInliningHelperEmitsFunctionCallTest + ShortNarrowingTest + AlgorithmGroupReduceHalfTest: 7/7 each (no regressions).
4. Full PMT sweep on a quiet machine: zero new failures vs rc.15 baseline.

When all four hold, rc.16 is a real ship candidate. If Bug E (iDCT 16x16 Wasm) is unsolved, rc.16 ships with that as a documented known-issue and rc.17 is its own cut.
