# rc.16 Bug D — Resumption Pack (Geordi, 2026-05-05)

This file is a complete handoff for the next session resuming Bug D. Everything you need is here or referenced from here. **Read top-to-bottom before touching code.**

## What this fixes

Tuvok's Codecs `OpusRangeDecoderGpu_DecodeUint_LargeRanges_BitExactVsCpu` test is gated with `UnsupportedTestException` on WebGPU because the kernel hits a 30s WGSL cold-compile timeout. The kernel uses 5 nested helpers passing `ArrayView<byte>` by ref — when the Aggressive Inliner inlines them all into the kernel entry, the WGSL emit becomes a 600-line monolithic fn with 217 var decls + 5 nested loops. wgpu's Naga validator cold-compile of this shape exceeds 30s.

The fix: support `[MethodImpl(MethodImplOptions.NoInlining)]` on helpers that take ArrayView params, so they emit as separate WGSL fn definitions (~60 lines each, Naga validates in ~1s). Currently the fn-def codegen path is broken for ArrayView fn-params — that's Bug D in `Plans/rc16-fn-def-codegen-harden.md`.

## Status (2026-05-05 EOS)

| Phase | What | Status | Commit |
|---|---|---|---|
| 0 | Reproducer test on master | LANDED | `ef7b4eb` |
| 1 | WGSLFunctionGenerator emits `ptr<storage, ..., read_write>` for ArrayView fn-params | LANDED | `150670c` |
| 4 | Kernel emits sub-word alias `let v_X = &paramN;` unconditionally | LANDED | `150670c` |
| 2 | Sub-word body codegen (LEA + Load + Store inside helpers) | **PENDING** | — |
| 3 | Helper-to-helper recursive call dispatch | **PENDING** | — |
| 5 | EmitNonInlinedMethodCall view-arg routing | **PENDING** | — |

When phases 2/3/5 land + Tuvok marks his helpers `[MethodImpl(NoInlining)]`, the test should turn green on WebGPU + WebGPUNoSubgroups.

## Reproducer (already on master)

`SpawnDev.ILGPU.Demo.Shared/UnitTests/BackendTestBase.Tests23.UintCompareInLoop.cs::Tests23_DecodeUint_LongForm_CompileSmoke`

Currently committed WITHOUT NoInlining attrs (reproduces the original 30s Naga timeout shape). To exercise the fn-def path, locally add `[MethodImpl(MethodImplOptions.NoInlining)]` to each Tests23 helper:

- `Tests23_ReadByte`
- `Tests23_ReadByteFromEnd`
- `Tests23_Normalize`
- `Tests23_DecodeBits`
- `Tests23_EcIlog`
- `Tests23_DecodeUint`
- `Tests23_OpusInit`

Plus add `using System.Runtime.CompilerServices;` at the top of the file.

## Verification command

```powershell
dotnet test PlaywrightMultiTest/PlaywrightMultiTest.csproj --filter "FullyQualifiedName~Tests23_DecodeUint_LongForm_CompileSmoke" -c Release
```

Expected at end of phase 2/3/5: 7/7 PASS (CPU, CUDA, OpenCL, Wasm, WebGL, WebGPU, WebGPUNoSubgroups), all sub-second.

## Evidence files (git-tracked snapshots)

Stable copies of the live `_dump/*/wgsl/` files from this session, snapshotted into `Plans/rc16-bug-d-evidence/`:

- `01_baseline_inlined_naga_30s_timeout.wgsl` — pre-fix state, all helpers inlined into one 618-line fn main with 217 var decls. WGSL parses fine but Naga validator cold-compile > 30s. This is what ships today on master without NoInlining attrs.
- `02_with_noinlining_pre_phase1_broken.wgsl` — with NoInlining attrs but BEFORE phase 1 fix. 11 fns, ~60 lines each, Naga compiles in ~1s, but ArrayView fn-param emitted as `i32` not `ptr<storage, ...>`. WGSL fails to parse with "cannot initialize var of type i32 with value of type ptr<...>".
- `03_post_phase1_signatures_fixed_body_still_broken.wgsl` — current state on master. Signatures correct (`ptr<storage, array<atomic<u32>>, read_write>`); helper body still emits raw `let v_10 = &v_1[v_9]; v_11 = *v_10;` which fails Naga with "cannot assign atomic<u32> to i32" at line 81 of helper body. Phase 2 fixes this.
- `04_handcrafted_target_shape.wgsl` — minimal hand-crafted WGSL showing the target signature + body shape we want phase 2 to produce.

## Phase 2 implementation plan — sub-word body codegen

**Problem.** Inside a helper fn body, `LoadElementAddress` on a Parameter of ViewType with sub-word ElementType emits via base `WGSLCodeGenerator.GenerateCode(LoadElementAddress)` (line 1633 of `WGSLCodeGenerator.cs`), which produces:

```wgsl
let v_10 = &v_1[v_9];   // v_1 is ptr<storage, array<atomic<u32>>, read_write>
v_11 = *v_10;            // type mismatch: cannot assign atomic<u32> to i32
```

**Required emit (mirrors kernel's existing sub-word load shape).** For `Int8` reads:

```wgsl
var v_10 : i32 = v_9;    // store the offset, defer atomic load to Load codegen
// later, in Load(LEA result):
v_11 = i32(((u32(atomicLoad(&(*v_1)[(u32(v_10) / 4u)])) >> ((u32(v_10) % 4u) * 8u)) & 0xFFu));
// for sbyte, sign-extend via select:
// select(i32(rawByte), (i32(rawByte) - 256), rawByte >= 128u)
```

For `Int16` (similar but with `/ 2u` and `0xFFFFu` mask). For `Float16` emulated, calls `_f16_to_f32` helper.

**Where the kernel's logic lives** (mirror this in WGSLFunctionGenerator — or factor into base):

| What | File:lines | Notes |
|---|---|---|
| Sub-word state populated at constructor time | `WGSLKernelFunctionGenerator.cs:3572-3625` | `_subWordParams[paramIdx] = elemSize`, also `_subWordUnsignedParams.Add(paramIdx)` for byte/ushort, `_subWordFloat16Params` for f16 emulated |
| Sub-word LEA emit (stores i32 offset, registers in `_subWordLEAVars`) | `WGSLKernelFunctionGenerator.cs:4357-4403` | Key line: `_subWordLEAVars[target.Name] = param.Index;` then `var {target.Name} : i32 = {adjustedOffsetExpr};` |
| Sub-word Load emit (atomic load + shift + mask + sign-extend) | `WGSLKernelFunctionGenerator.cs:4560-4618` | Reads `_subWordLEAVars` to get paramIdx, looks up `_subWordParams[paramIdx]` for elemSize, branches by signed/unsigned/f16. The binding name source is `param{paramIdx}` for kernel — must be `(*p_alias)` for helper. |
| Sub-word Store emit | `WGSLKernelFunctionGenerator.cs:4750-4790+` | Same shape, RMW via atomicAnd + atomicOr |

**Recommended approach: factor sub-word handling into base.** The cleanest implementation refactors the sub-word LEA/Load/Store machinery into `WGSLCodeGenerator` (base class). Both kernel + function generators populate sub-word param state from their own param scan; both share the emit logic. This avoids ~300 lines of duplicated code and ensures helpers behave identically to kernel for sub-word access.

Concrete refactor steps:

1. **Move state into base** (`WGSLCodeGenerator.cs`): protected fields `_subWordParams`, `_subWordUnsignedParams`, `_subWordFloat16Params`, `_subWordLEAVars`, `_subWordBodyStructBindingNames`. Kernel inherits + populates from its existing `SetupParameterBindings` pre-scan; helper inherits + populates from a new `ScanHelperSubWordParams()` called from constructor.
2. **Move LEA emit into base virtual** (or factor a `protected EmitSubWordLEA(target, offset, paramIdx, bindingExpr)` helper). Kernel's override calls into it for sub-word; helper's `GenerateCode(LoadElementAddress)` either uses base virtual directly or delegates after detecting a fn-param ViewType source.
3. **Move Load extract into base helper** (`protected EmitSubWordLoadExtract(target, leaVarName)` returning the atomic-load-shift-mask expression). Both generators call it.
4. **Same for Store**.
5. **Binding name routing.** Kernel uses `param{N}`; helper uses `(*v_X)` where v_X is the helper's local pointer alias. Implement via a `protected virtual string GetSubWordBindingName(int paramIndex)` that the kernel returns `param{N}` from and the helper returns the local ptr alias name from.

Risk: this touches kernel codegen for sub-word — must run the full body-struct + sub-word PMT sweep before claiming green:
- `Tests23_RegisterHeavyBody*`, `Tests23_*BodyStruct*`, `Tests23_OnlyShortBodyStruct`, `Tests23_TwoShortBodyStruct*`, `Tests23_DecodeBitLogP*`, `Tests23_UnsignedDivRem*`, `Tests23_UnsignedShr*`, `Tests23_DeepUnroll*`
- Plus regular sub-word usage outside body structs: `AlgorithmReduceByteTest`, `AlgorithmGroupReduceHalfTest`, `LocalMemoryRepro_Int64_ShortByteViews`

**Alternative: copy-paste the kernel's sub-word logic into WGSLFunctionGenerator overrides.** Faster but produces ~300 lines of duplication. Use only if the base-class refactor surfaces unforeseen complications.

## Phase 3 — Helper-to-helper recursive calls

Inside helper bodies, calls to other helpers fall to "Unmapped fallback":

```wgsl
// Inside Tests23_OpusInit_961 body:
// Call: Tests23_ReadByte_16 (Unmapped)
v_27 = i32(0); // Unmapped fallback
```

Reason: `WGSLFunctionGenerator` doesn't override `GenerateCode(MethodCall)`. Only `WGSLKernelFunctionGenerator.GenerateCode(MethodCall)` (lines 1996-2160) handles the inline-helper / fn-call / intrinsic dispatch.

**Fix:** add an override in `WGSLFunctionGenerator` that mirrors the kernel's branching:

```csharp
public override void GenerateCode(MethodCall methodCall)
{
    var targetMethod = methodCall.Target;

    // Inline path: methods marked Inline get inlined at the call site.
    if (_args.HelperMethods.TryGetValue(targetMethod, out var helperAllocas))
    {
        // ... mirror WGSLKernelFunctionGenerator inline path ...
        // (The inline machinery is ~150 lines; either extract to a shared base helper,
        // or copy with minimal modification.)
        return;
    }

    // Fn-call path: non-Inline methods get a real WGSL fn call.
    if (targetMethod.HasImplementation
        && !targetMethod.HasFlags(MethodFlags.External)
        && !targetMethod.HasFlags(MethodFlags.Intrinsic))
    {
        EmitNonInlinedMethodCall(methodCall);  // shares kernel's EmitNonInlinedMethodCall logic
        return;
    }

    base.GenerateCode(methodCall);
}
```

Like phase 2, the cleanest approach factors the dispatch logic into a shared base method.

## Phase 5 — EmitNonInlinedMethodCall view-arg routing

When the kernel calls a non-inlined helper passing an ArrayView arg, ensure the arg variable is the kernel's ptr alias `v_X` from `let v_X = &paramN;` (now emitted unconditionally per phase 4), not the raw param value.

Looking at `WGSLKernelFunctionGenerator.cs:2172-2193`:

```csharp
private void EmitNonInlinedMethodCall(MethodCall methodCall)
{
    var fnName = WGSLFunctionGenerator.GetMethodName(methodCall.Target);

    var argStrs = new List<string>(methodCall.Count);
    for (int i = 0; i < methodCall.Count; i++)
    {
        var argVar = Load(methodCall[i]);  // <-- this should return the ptr alias for ArrayView args
        argStrs.Add(argVar.ToString());
    }
    ...
}
```

`Load(methodCall[i])` returns the variable for the IR Value. For an ArrayView arg, the IR Value is the kernel's Parameter, and `Load(Parameter)` returns the variable from `valueVariables[param]` which was registered when SetupParameterBindings ran. After phase 4, sub-word params have their alias registered (the `let v_X = &paramN;`). Verify this at the dump — confirm `Tests23_DecodeUint_963(&v_10, v_4, v_5, v_6, v_37);` becomes `Tests23_DecodeUint_963(&v_10, v_4_ptr_alias, v_5, v_6, v_37);` where `v_4_ptr_alias` is the variable from the alias emission.

If `Load(Parameter)` is returning the wrong variable (e.g., the raw param value instead of the alias), this is where to intercept.

## Open Rule-2a-owned bugs in the SpawnDev.ILGPU lane

Per Rule 2a (Editor Owns Every Bug), no "pre-existing" framing. These are mine to fix:

1. **`Tests23_DecodeUint_LongForm_CompileSmoke`** — WebGPU + WebGPUNoSubgroups timeout. Bug D phases 2-5 above.
2. **`Tests23_GroupDimX_Clamps_To_Extent_OnUnitDispatch`** — WebGPU + WebGPUNoSubgroups, expects 1 gets 64. WGSL `@workgroup_size(64)` is statically baked. Fix path: pipeline-creation-time WGSL override constants for workgroup size, OR shader recompile per dispatch with min(extent, customGroupSize). Pipeline-cache implications need design work.

Sweep these two and any others surfaced via PMT before declaring "all bugs fixed."

## Captain's pause posture (do NOT push to nuget.org)

Captain told me 2026-05-05: "do not push any more rc versions to nuget.org until told otherwise please." Standing pause until reopened.

- Local feed `D:\users\SpawnDevPackages` still active. `nuget add` rc.N nupkgs there freely.
- Source still pushes to GitHub master normally.
- `_DevComms/global/nuget-local-publish-log.md` rows mark "local only" until pause lifts.
- `_publish-nuget.bat` (which pushes to nuget.org) — **DO NOT RUN** until Captain explicitly green-lights it again.
- Saved as feedback memory `feedback_pause_nugetorg_pushes_2026_05_05.md` + indexed in `MEMORY.md`.

rc.27 was the last pre-pause push.

## Rule 2a posture (do NOT use "pre-existing" framing)

Captain codified Rule 2a 2026-05-05 in `D:\users\tj\Projects\CLAUDE.md`. BANNED phrases:
- "pre-existing"
- "already failing before my changes"
- "not introduced by my work"
- "I didn't break it" / "not my code"
- "existing bug, separate issue"
- "not a regression from this rc"
- "was broken in the previous version"

Every open bug in the lane is mine, regardless of when introduced. Saved as feedback memory `feedback_editor_owns_every_bug.md`.

## Cross-references

- **rc.16 master plan:** `Plans/rc16-fn-def-codegen-harden.md` — full Bug A through E catalog. Bug D section updated 2026-05-05 with phase-by-phase status.
- **DevComms posts from this session:**
  - `_DevComms/SpawnDev.ILGPU/geordi-to-team-rc27-webgpu-output-coalesce-fix-2026-05-05.md`
  - `_DevComms/SpawnDev.ILGPU/eod-geordi-2026-05-05.md` (renamed conceptually to "session pause" since work continues)
- **Active agents row:** `_DevComms/global/active-agents.md` Geordi row updated 2026-05-05.
- **Memory pointers (always loaded):**
  - `feedback_pause_nugetorg_pushes_2026_05_05.md`
  - `feedback_editor_owns_every_bug.md`
- **rc.27 publish log:** `_DevComms/global/nuget-local-publish-log.md` last row.

## Quick orientation commands for the next session

```powershell
# What state is master in?
git log --oneline -5
# 150670c WGSL Bug D phases 1+4 — ArrayView fn-param signatures + sub-word kernel alias
# ef7b4eb WebGPU DecodeUint long-form 30s cold-compile: regression test + Bug D analysis
# a63f2ea WebGPU coalesce excludes OUTPUT body-struct view fields + rc.27
# d00778b WebGL body-struct rc.26 — close v2 follow-ups, all 8 tests PASS
# 1695c9d WebGL multi-view body-struct codegen + rc.25

# What does the current WGSL emit look like for a NoInlining helper?
# Locally add NoInlining attrs to Tests23_DecodeUint_LongForm helpers, then:
dotnet test PlaywrightMultiTest/PlaywrightMultiTest.csproj --filter "FullyQualifiedName~Tests23_DecodeUint_LongForm_CompileSmoke" -c Release
# Then check D:\users\tj\Projects\SpawnDev.ILGPU\_dump\<latest>\wgsl\
# Compare against Plans/rc16-bug-d-evidence/03_post_phase1_signatures_fixed_body_still_broken.wgsl

# Pre-flight before running PMT (Rule 5b):
Get-Process testhost -ErrorAction SilentlyContinue | Where-Object { $_.Path -like '*SpawnDev.ILGPU*' } | Format-Table Id, Path -AutoSize
```
