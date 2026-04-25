# rc.14 — Large-method WGSL fn-definition emission (Bug 1 from Tuvok's Vp9Idct16x16 report)

## Summary

Fix the WebGPU shader compile cliff that triggers when a kernel calls a large user method many times. Currently every user method called from the kernel is inlined at IR level (`Inliner` in `Default` mode bypasses the size check), producing a single straight-line WGSL kernel body. Tuvok's `Vp9Idct16x16Kernel` calls `Idct16Row` (~500 IL instructions, 7-stage butterfly) at 32 call sites; the result is a ~3800-line straight-line WGSL function that pushes Chrome's WGSL validator + V8's AsyncFunction compile path past 30 s per kernel instance.

The fix is a body-size cap on the IR Inliner combined with WGSL codegen that emits a real `fn` definition + function call for methods that aren't inlined. WGSL already has `WGSLFunctionGenerator` that *can* emit fn definitions, but the gating logic always lands methods in `HelperMethods` (which forces codegen-time inlining), making the fn-definition path dormant. We light it up.

## Scope

**Bug 1 from `tuvok-to-geordi-idct16x16-two-bugs-2026-04-25.md`** — WebGPU compile cliff at large straight-line WGSL function bodies.

**Out of scope for this plan**: Bug 2 (Wasm bit-exact divergence on the 16-point butterfly). Filed as a separate plan.

## Critical files

### Changes

1. **`ILGPU/IR/Transformations/Inliner.cs`** (~10 lines)
   - Add `MaxNumILInstructionsToInlineDefault = 200` constant (Conservative cap is `32`).
   - `SetupInliningAttributes`: in `InliningMode.Default`, fall back to body-size cap (200) instead of bypassing the check.
   - `Aggressive` mode keeps the unconditional inline (escape hatch for code that explicitly opts in).
   - Methods marked `[MethodImpl(MethodImplOptions.AggressiveInlining)]` always inline regardless of mode (preserved).
   - Methods marked `[MethodImpl(MethodImplOptions.NoInlining)]` never inline (preserved).
   - Methods in `Context.FullAssemblyModuleName` (the IL frontend's auto-generated wrappers) always inline (preserved — these are tiny lambda dispatchers).

2. **`SpawnDev.ILGPU/WebGPU/Backend/WebGPUBackend.cs:819-827`** (`CreateFunctionCodeGenerator`)
   - Add condition: register method in `HelperMethods` only if it has the `Inline` flag set OR it uses workgroup barriers / shared memory in a way that requires inlining for WGSL barrier-uniformity safety.
   - Methods that DON'T qualify for inlining are skipped from `HelperMethods` and fall through to `WGSLFunctionGenerator`'s real fn-definition emission path (already implemented at lines 146-167 — currently dormant).

3. **`SpawnDev.ILGPU/WebGPU/Backend/WGSLKernelFunctionGenerator.cs:1667-1818`** (`GenerateCode(MethodCall)`)
   - After the existing `HelperMethods.TryGetValue` inline path, add a new branch:
     ```csharp
     // Method has an implementation but isn't a registered inlinable helper —
     // emit a real WGSL function call to the fn definition that
     // WGSLFunctionGenerator will emit at module scope.
     if (targetMethod.HasImplementation
         && !targetMethod.HasFlags(MethodFlags.External | MethodFlags.Intrinsic))
     {
         EmitNonInlinedMethodCall(methodCall);
         return;
     }
     ```
   - `EmitNonInlinedMethodCall`: looks up the WGSL fn name (uses `WGSLFunctionGenerator.GetMethodName`, made `internal`), allocates the return Variable, emits `let target = fn_name(arg1, arg2, ..., argN);`.
   - Handle the void-return case: `fn_name(args...);` (no `let target =`).

4. **`SpawnDev.ILGPU/WebGPU/Backend/WGSLFunctionGenerator.cs:82-96`** (`GetMethodName`)
   - Change visibility from `private static` to `internal static` so the kernel generator can produce identical names.

### Investigation deferred to verification phase

5. **`out` / `ref` parameter lowering** — `Idct16Row` has 16 `out int` parameters. The IR likely lowers these to pointer parameters (`Pointer<i32>` → WGSL `ptr<function, i32>`). The existing `GenerateHeaderStub` at `WGSLFunctionGenerator.cs:58-77` already handles pointer types for parameters via the `ptr<` startswith check. If verification shows the lowering produces clean WGSL pointers, no code change is needed. If the IR uses a struct-return form instead, we just need to handle the struct-return.

### No changes

- Other backends (CUDA, OpenCL, CPU, Wasm, WebGL) already support non-inlined method emission via their normal IR-MethodCall codegen paths. The Inliner change increases their non-inlined surface for large methods; expected behavior is correct because they're standard fn definitions in the target language.

## Why this approach

**Alternatives considered:**

- **WebGPU-specific IR transformation** (un-inline already-inlined methods) — impossible. Once the IR Inliner has specialized a call, the body is woven into the caller's IR; you can't un-weave it without the original method definition, and at that point you've lost the call structure.
- **Codegen-time WGSL deduplication** (detect repeated subgraphs in the kernel IR and emit them as a fn) — algorithmically complex, fragile to small variations, hard to test. Rejected.
- **Tuvok-side refactor** (option b in his bug report — 16 threads/block topology change) — works around the bug instead of fixing it. Rule #2 says fix the library. The performance shape ends up similar (Tuvok would still do the work), but the library stays correct for all consumers, not just the ones who know about the workaround.
- **Conservative InliningMode** — switching the entire IRContext to Conservative mode (32 IL cap) regresses smaller helpers that benefit from inlining. The 200-IL cap is a deliberate middle-ground.

**Why 200 IL instructions:**

Idct16Row body is ~500 IL instructions (estimated). Most "real" helpers are < 100 IL instructions. The 200 threshold catches Idct16Row-class methods while leaving small helpers in the inline path. We can tune later if the cap turns out to be too aggressive in either direction.

## Risk and mitigation

- **Other backends regress on perf**: Inlining matters for register allocation in CUDA / PTX. Risk: kernel hot paths now have function-call overhead. Mitigation: the cap is high enough (200 IL) that only LARGE methods un-inline. Verification: the existing PMT regression catches any perf-affecting kernel breakage; specifically the rc.12 LocalMemoryRepro N=64/256/1024 tests already exercise hand-tuned kernels.
- **`out`/`ref` parameter lowering surprises**: If the IR's lowering of out parameters doesn't produce clean WGSL fn parameters, the fn-definition emission may fail. Mitigation: write a small `[MethodImpl(MethodImplOptions.NoInlining)]` test on simple `int Sum(int a, int b)` first, verify the WGSL emit looks right, *then* test on Idct16Row's multi-out shape. If multi-out doesn't lower cleanly, fix in a follow-up commit (or split into two fixes — fn-def emission for value-return first, multi-out as phase 2).
- **Codegen-time inline path becomes partially dead**: Currently `WGSLKernelFunctionGenerator.GenerateCode(MethodCall)` lines 1672-1813 inline the helper at codegen time. After this fix, only barrier-using methods take that path. The path stays in place — barrier-using methods (Group.Reduce, etc.) still need it.
- **WGSLFunctionGenerator's fn-definition path was untested in production**: It's been dormant since the codegen was written. Verification will catch any latent bugs (parameter binding, return statement emission, scope of declarations). Bugs found get fixed in this same rc.

## Verification

### Phase 1 — Smoke test (desktop OpenCL, no Tuvok contention)

1. Add a temporary small test class with `[MethodImpl(MethodImplOptions.NoInlining)] int Helper(int a, int b) => a*a + b*b;` and a kernel calling it twice.
2. Run on OpenCL desktop. Verify result correct.
3. Confirm IR Inliner respected NoInlining (kernel IR has MethodCall, not inlined body).
4. Confirm OpenCL kernel source has a `static int Helper_NN(...)` definition + 2 call sites (not 2 inlined bodies).

### Phase 2 — WebGPU verification (after Tuvok's PMT clears)

5. Same test class, run on WebGPU (browser PMT). Verify WGSL has `fn Helper_NN(p_0: i32, p_1: i32) -> i32` + 2 call sites.
6. Verify `Vp9Idct16x16Kernel` WebGPU compile time drops below the 30 s test budget (Tuvok's 4 timing-out tests now pass).

### Phase 3 — Regression

7. Full SpawnDev.ILGPU PMT sweep (no P2P). Zero new failures vs rc.13 baseline. Specifically watch:
   - LocalMemoryRepro N=64 (still passes — small kernel, still inlines)
   - All RadixSort tests (medium-sized helpers — might cross the 200 cap; verify perf regression isn't significant)
   - Half / scan / reduce intrinsic tests (all pass)
8. Tuvok's `Vp9Idct16x16Kernel` filtered run via PMT. Specifically:
   - Vp9Idct16x16Kernel_DcOnly_MatchesReference WebGPU PASS in < 30 s
   - 4 previously-timing-out WebGPU tests now PASS
   - Wasm bit-exact tests still red (Bug 2, not addressed by this fix)

### Phase 4 — rc.14 cut

9. Bump csproj rc.13 → rc.14, build, local-publish.
10. DevComms ack to Tuvok with verification numbers.

## Why this is rc-not-stable territory

Per Captain's earlier rc-vs-stable question: this fix proves the rc.12 ship report's claim ("iDCT 16x16/32x32 family unblocked") was incomplete. rc.14 closes the gap; stable promotion can proceed once rc.14 + Bug 2 fix verify clean across the full sweep.
