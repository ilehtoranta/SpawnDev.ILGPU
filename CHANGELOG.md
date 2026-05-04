# SpawnDev.ILGPU Changelog

This file tracks notable changes per release. The README's "Recent Highlights" section links here for the full version history.

## 4.9.5-rc.7 (2026-05-04) — WebGL GLSL codegen: INT_MIN const + unsigned compare/shift

### Three GLSL codegen bugs fixed

Surfaced by the same Tests23 bisection that closed Tuvok's libopus Normalize loop on rc.6, all three issues had been silently corrupting WebGL kernel output for any code path that used `uint` values with the high bit set.

#### Bug 1 — `int.MinValue` constant emit substituted bit pattern 0x80000000 → 0x80000001

`GLSLCodeGenerator.GenerateCode(PrimitiveValue)` had been substituting `int.MinValue` with the literal `-2147483647` as a workaround for an ANGLE/ESSL3 parser issue (`-2147483648` parses as `-(2147483648)` and 2147483648 is not a valid signed-int literal). The substitution preserved sign but corrupted the LOW BIT — every constant that needed the exact 0x80000000 bit pattern silently shipped as 0x80000001. Affected: libopus range-coder constants (`EC_CODE_TOP = 1u << 31`), IEEE -0.0 bitcasts, uint shift overflow results that get constant-folded by ILGPU's IR construction.

Fix: emit as `int(2147483648u)` — uint-to-int bitcast preserves the exact bit pattern with no parser issue and no UB.

#### Bug 2 — `uint <= uintConst` evaluated as signed compare

`GenerateCode(CompareValue)` for unsigned integer comparisons (with `IsUnsignedOrUnordered` flag set) emitted bare `int op int` GLSL. ILGPU's IR uses `BasicValueType.Int32` for both signed and unsigned integers, with the unsigned-ness carried as a flag on the compare operation. The GLSL TypeGenerator maps Int32 → "int", so the operands were declared as `int` in the shader and the compare used signed semantics — values with the high bit set compared as negative. `0x80000000 <= 0x800000` returned TRUE on WebGL because `int(0x80000000) = -2147483648 < 8388608`.

Fix: when `IsUnsignedOrUnordered` is set on a `<= < > >=` comparison and the operand types are integer, emit `uint(left) op uint(right)` so GLSL uses unsigned semantics.

#### Bug 3 — Signed left-shift overflow into the sign bit produces undefined behavior on ANGLE

`GenerateCode(BinaryArithmetic)` for Shl emitted `int << int` directly. GLSL ES 3.0 spec marks left-shift of a signed integer where the result sets the sign bit as **undefined behavior**. ANGLE on Chrome produces inconsistent values (observed 0x80000001 instead of 0x80000000). Same UB applies to Shr when shifting a sign-bit-set value, but ILGPU IR does carry an `IsUnsigned` flag for `shr.un` IL, so signed-Shr's sign extension is preserved.

Fix: emit as `int(uint(left) << uint(right))` for Shl (always — no IR signal for shift signedness because IL `shl` has no `.un` variant). For Shr, only switch to unsigned when `IsUnsigned` is set on the IR node.

#### Bonus — `glWorker.js` TF readback path

Changed the WebGL Transform Feedback readback typed array from `Float32Array` to `Uint8Array`. The byte path is identical — `getBufferSubData` does a raw byte copy regardless of the destination's element type — but explicit `Uint8Array` is clearer documentation of intent and avoids any future driver-side type-conversion paths.

### Verification

- `BackendTestBase.Tests22.StaticStructReturnRefHelpers` — Tuvok's libopus regression: **14/14 PASS** post-fix (already PASSing on rc.6; confirmed unchanged).
- `BackendTestBase.Tests23.UintCompareInLoop` — 7 bisection cases × 7 backends = **49/49 PASS** including all WebGL cases (was 6 WebGL failing on rc.6).
- WebGL full test sweep (`FullyQualifiedName~WebGLTests`): **457 passed / 1 failed / 123 skipped**. The 1 failure is `LocalMemoryRepro_Int64_ShortByteViews` (30s timeout) — a preexisting WebGL architectural-varying-count limit, NOT caused by this fix.

### What this likely also fixes

Data's `data-to-captain-ml-sweep-summary-2026-05-04.md` listed 12 WebGL correctness failures across DistilBERT / GPT2 / WhisperEncoder / CLIPVision / DepthAnything / 5 Style transfers / YOLOv8 / TextGeneration. Many of those models use `uint` indexing or bitwise operations that go through the buggy compare/shift paths. Recommend re-running the ML pipeline + reference suite on WebGL after rc.7 lands; expect a meaningful chunk of the 12 failures to clear automatically.

### Files changed

- `SpawnDev.ILGPU/WebGL/Backend/GLSLCodeGenerator.cs` — PrimitiveValue Int32 INT_MIN bitcast + CompareValue unsigned-cast (base class path)
- `SpawnDev.ILGPU/WebGL/Backend/GLSLKernelFunctionGenerator.cs` — CompareValue unsigned-cast (kernel override) + BinaryArithmetic Shl/Shr unsigned-cast
- `SpawnDev.ILGPU/wwwroot/glWorker.js` — TF readback typed array (cosmetic)
- `SpawnDev.ILGPU.Demo.Shared/UnitTests/BackendTestBase.Tests23.UintCompareInLoop.cs` — added const-write diagnostic to Tests23_BareUintShift
- `SpawnDev.ILGPU.Demo/UnitTests/WebGLTests.cs` — removed all Tests23 gates (every case now passes on WebGL)

### Four-package bundle: Fork unchanged at 2.0.4

Changes are all in the `SpawnDev.ILGPU` wrapper (WebGL/* + wwwroot/glWorker.js); no edits to the forked `ILGPU/` tree. Fork stays at 2.0.4. Only the SpawnDev.ILGPU PackageReference bumps rc.6 → rc.7.

## 4.9.5-rc.6 (2026-05-04) — LoopUnrolling shift-induction trip-count fix

### Fix: `while (uintRng <= uintConst) rng <<= N` produced wrong output for N != 1 on every backend except CPU

Surfaced 2026-05-04 by Tuvok's libopus-style Normalize while-loop pattern (`while (rng <= 0x800000) rng <<= 8`) which silently produced `Rng=0` instead of `0x80000000` on every GPU + Wasm + WebGL backend. CPU bypassed because it doesn't run the IR-level unroller pass.

### Root cause

`ILGPU/IR/Analyses/LoopInfo.cs:944` (`TryGetTripCount`) computed the per-iteration multiplier for shift updates as:

```csharp
if (IsMultiplied2Update(UpdateKind)) update *= 2;
```

That formula only produces the correct multiplier for `<<= 1` (where 2*1 = 2 = 2^1). For `<<= N` with N != 1 the per-iteration multiplier should be `2^N`, not `2*N`. With shift_count=8 (libopus EC_SYM_BITS) the unroller computed update=16 instead of 256, producing trip count 5 instead of 3, so it emitted two extra `rng <<= 8` operations past the loop's intended exit. After the extra iterations rng=0 (high byte shifted off then again).

The bug had been latent since the loop unroller's introduction; it only surfaces when the kernel has a SINGLE induction variable that's a shift. Compound conditions like `while (cond && iter < N)` introduce a second induction variable and force the unroller to bail (`InductionVariables.Length != 1`) — which is why no existing algorithm tests caught it. Tuvok's `OpusRangeDecoderGpu.Init` was the first kernel in the codebase to use a bare-condition shift loop.

### Fix

```csharp
if (IsMultiplied2Update(UpdateKind))
{
    if (update < 1 || update >= 32)
        return null; // out-of-range shift — bail rather than emit garbage
    update = 1 << update;
}
```

Identical behavior for `<<= 1`. Correct for any other shift amount in 1..31.

### Verification

- `BackendTestBase.Tests22.StaticStructReturnRefHelpers` — Tuvok's regression test for the libopus Init/Normalize pattern. Was 12/14 failing pre-fix (CUDA / OpenCL / WebGPU / WebGPUNoSubgroups / Wasm / WebGL × bug+inline). **PASS 14/14 post-fix.**
- `BackendTestBase.Tests23.UintCompareInLoop` — 6 new bisection cases that pin down the unroll-path codegen. PASS on every backend except WebGL, which has a SEPARATE pre-existing GLSL signed-shift/compare bug (gated via `UnsupportedTestException`, tracked independently — not caused by this fix).
- Algorithm regression sweep (Reduce + Initialize + RadixSort + Sequence + Scan + Histogram + Algorithm*): **481 passed / 0 failed / 93 skipped** in 53m48s. Zero regressions.

### What this unblocks

- `SpawnDev.Codecs.OpusRangeDecoderGpu` — was CPU-only verified before; now works bit-exact on every backend.
- All upcoming Opus SILK / CELT / Vorbis decoder primitives that use libopus-style range-decoder normalize loops.

### Four-package bundle bumped to 2.0.4

Fix is in `ILGPU/IR/Analyses/LoopInfo.cs` (forked tree). Per the four-package bundle protocol, `ILGPU.csproj`, `ILGPU.Algorithms.csproj`, and the two `SpawnDev.ILGPU.Fork*` PackageReference lines in `SpawnDev.ILGPU.csproj` all bumped from 2.0.3 → 2.0.4. `_check-fork-version-sync.bat` passes.

## 4.9.5-rc.5 (2026-05-03) — WebGPU binding-count coalesce

### Fix: kernels with > 10 storage-buffer bindings on WebGPU

WebGPU spec `maxStorageBuffersPerShaderStage` = 10 (Chrome default). Every body-struct ArrayView field gets its own storage-buffer binding under the previous codegen, so a kernel taking a struct with many `ArrayView` fields would push the total over the limit and throw at dispatch time:

```
[WebGPU] Kernel 'Kernel_Run' requires 44 storage buffer bindings but this device only supports 10
```

Triggered by `SpawnDev.Codecs.Audio.Vorbis.VorbisPacketDecodeStaticInputs` (36 `ArrayView<int>` + 2 `ArrayView<double>`), which would also recur for the upcoming Opus SILK + CELT integration kernels and Vorbis v3 streaming decoder. Per `_DevComms/SpawnDev.ILGPU/tuvok-to-geordi-vorbis-v2-binding-count-2026-05-03.md`.

### Fix surface

`WGSLKernelFunctionGenerator.DecideCoalesceGroups` runs after `ScanBodyStructParams`. When the kernel's predicted raw binding count exceeds 10, it groups eligible body-struct ArrayView fields by element type and coalesces each multi-member group into a single shared `@binding(N) var<storage, read_write> ... : array<T>` declaration. Per-field accesses route through the existing `_scalar_params[ViewOffsetSlot]` channel (the same machinery sub-views already use for non-zero-offset element offsets); each member's offset within the coalesced buffer is stamped at dispatch time via a new `IsCoalesceFieldOffset` `ScalarPackingEntry` flag.

`WebGPUAccelerator` dispatch path: a coalesce pre-pass allocates one fresh GPU buffer per group, runs `CopyBufferToBuffer` for each member to concat their data at running offsets, binds the coalesced buffer once at the leader's binding slot, and skips non-leader members in Phase 1. The coalesced buffer is destroyed after the batch flushes (no scratch pool — sizes vary widely with kernel parameter shape).

### Eligibility (v1)

A body-struct ArrayView field qualifies for coalescing when ALL of:
- Element type is `i32`, `u32`, `f32`, `emu_i64`, `emu_u64`, or `emu_f64`
- Field is NOT atomic (atomic bindings need `atomic<T>` typing — separate path)
- Field is NOT sub-word (`i8` / `i16` / `Half` packed `atomic<u32>` — separate path)
- Field is NOT a packed-struct view (CPU-layout u32 packing — different stride per group)
- Body struct is flat (no nested struct fields with pointer recursion — defensive runtime check throws on unexpected shapes)

Trigger: kernel raw bindings > 10. Existing kernels with body structs of 1-9 view fields keep their current shape (no per-dispatch GPU→GPU copy overhead).

### What this unblocks

- **SpawnDev.Codecs Vorbis v2 browser path** — currently dual-path with `useV2Path = _accelerator.AcceleratorType is CPU or Cuda or OpenCL` in `VorbisAudioDecoderGpu.DecodePacketAsync`; flips to include WebGPU once Codecs bumps to rc.5.
- **Opus SILK + CELT integration kernels** — designed with the same struct-of-ArrayView pattern, browser-clean from day one.
- **Future high-parameter codec primitives** — Vorbis v3 streaming decoder, codec ML primitives, etc.

### Test coverage

New `BackendTestBase.Tests21.CoalesceBindings.cs`:
- `BodyStruct_12ArrayViewInt_CoalesceTest` — 12 independent `ArrayView<int>` fields, kernel sums all at idx, verify CPU reference match.
- `BodyStruct_MixedIntFloatCoalesceTest` — 11 `ArrayView<int>` + 1 `ArrayView<float>`, two coalesce groups (separate bindings per element type).
- `BodyStruct_VariableLengthCoalesceTest` — 12 fields with widely-varying lengths (4-768 elements), exercises per-field offset routing.

Result: **15/0/6 across CPU + WebGPU + WebGPUNoSubgroups + CUDA + OpenCL.** Wasm + WebGL are skipped via `UnsupportedTestException` for a pre-existing many-field body-struct decomposition limitation in those backends — NOT a regression from this work; tracked separately for a follow-up fix.

Regression sweep on Reduce + Initialize + RadixSort + Sequence (existing body-struct algorithm kernels, all with ≤ 9 view fields and well under the coalesce trigger): **332 passed, 0 failed, 63 documented skips** across all backends. Coalesce trigger does not fire spuriously on small body structs.

### Public API additions

- `WebGPUCompiledKernel.CoalesceManifest` — `IReadOnlyList<CoalesceGroupEntry>` describing the coalesce groups for this kernel; `HasCoalesceGroups` convenience flag.
- `CoalesceGroupEntry` (public class in `SpawnDev.ILGPU.WebGPU.Backend`) — `BodyStructParamIndex`, `ElementTypeKey`, `BindingName`, `BindingIndex`, `BindingWgslType`, `ElementWordsPerSlot`, `MemberFieldIndices`.
- `ScalarPackingEntry.IsCoalesceFieldOffset` + `CoalesceBodyStructParamIndex` + `CoalesceFieldIndex` — manifest entry kind for per-field coalesce-relative offsets.

## 4.9.4 (2026-05-03) — stable rollup of rc.1 + rc.2

Stable cut. Configurable Wasm linear-memory ceiling, end-to-end (host + module declared max agree). End-to-end verified by SpawnDev.ILGPU.ML's DA3-Small at `MaxLinearMemoryPages=32768`: op 93 `memory.grow` past 16384 pages succeeds, model runs 2m 28s past the rc.1 instant-instantiate-reject point (Data, 2026-05-03). Default consumers (16384) see byte-identical output vs 4.9.3.

`SpawnDev.ILGPU.P2P 4.9.4` ships in lockstep: closes `P2PSwarm.TwoTab_PeerDiscovery` regression via the new `Wire.SimplePeer.IsTransportDead` accessor in `SpawnDev.WebTorrent 3.2.3` stable. Both bridge filter sites updated. `LargeBuffer_100MB_DispatchedOverRealWebRtc_BitExact` PASS 3m 37s standalone (no regression).

See the rc.1 + rc.2 sections below for the full surface description.

## 4.9.4-rc.2 (2026-05-03) (superseded by 4.9.4 stable)

### Fixes the rc.1 kernel-module memory import maximum mismatch

rc.1 made the host-side `WebAssembly.Memory` `maximum` configurable via `WasmBackendOptions.MaxLinearMemoryPages`, but the compiled kernel module's WASM binary still hardcoded `maximum=16384` in its memory-import declaration. Per WebAssembly spec, the imported memory's max must be `<=` the import's declared max — when the host cap was raised above 16384, `WebAssembly.instantiate` rejected every kernel dispatch:

```
WebAssembly.instantiate(): Import #0 "env" "memory":
memory import has a larger maximum size 32768 than the module's declared maximum 16384
```

Discovered by Data on DA3-Small with `MaxLinearMemoryPages=32768`; first dispatch failed instantly, all Wasm tests in the consuming project failed.

`WasmBackend.CreateKernel` now reads `Options.MaxLinearMemoryPages` and threads it through `WasmModuleBuilder.ImportSharedMemory("env", "memory", 1, (uint)Options.MaxLinearMemoryPages)`. Both ends agree at any cap up to 65536 (4 GiB).

Default behavior unchanged — consumers at the 16384 default see byte-identical module output vs rc.1.

### SpawnDev.ILGPU.P2P 4.9.4-rc.2 (lockstep bundle)

P2P source unchanged from rc.1; bumped to keep the bundle versioned in sync. Same `P2PWebRtcBridge.wire.OnClose` phantom-alive filter, same SpawnDev.WebTorrent 3.2.3-rc.2 dep.

## 4.9.4-rc.1 (2026-05-03) (superseded by 4.9.4-rc.2)

### Configurable Wasm linear-memory ceiling

New `WasmBackendOptions.MaxLinearMemoryPages` knob (default 16384 / 1 GiB, configurable up to 65536 / 4 GiB). Threaded through `WasmAccelerator.Create` and the cached-memory `WebAssembly.Memory` `eval` strings. Default behavior unchanged. Required by SpawnDev.ILGPU.ML's DA3-Small graph executor where total live allocations exceed 1 GiB at op 93.

**KNOWN ISSUE (fixed in rc.2):** The kernel module's memory-import maximum was still hardcoded at 16384 in this version, so consumers raising the host cap hit `WebAssembly.instantiate` failures. Use rc.2 instead.

### SpawnDev.ILGPU.P2P 4.9.4-rc.1: TwoTab phantom-alive close

`P2PWebRtcBridge.wire.OnClose` now filters phantom-alive wires (where `Destroyed=false` but the underlying transport is dead) using the new `Wire.SimplePeer.IsTransportDead` accessor in SpawnDev.WebTorrent 3.2.3-rc.2. Catches the Chromium-under-Playwright bug where `connectionstatechange` doesn't propagate to `"failed"` on remote tab close, leaving the wire's `Destroyed` flag false and inflating the canonical wireSet count. Both bridge filter sites updated: the wireSet `RemoveWhere` in `wire.OnClose` and the `torrent.Wires` cross-check walk.

Verified: `P2PSwarm.TwoTab_PeerDiscovery` PASS in 1m 37s standalone (was failing 90s timeout in 4.9.2-rc.34); `LargeBuffer_100MB_DispatchedOverRealWebRtc_BitExact` PASS in 3m 37s standalone (no regression vs rc.34).

SpawnDev.WebTorrent dep bumped 3.2.2 -> 3.2.3-rc.2.

## 4.9.3 (2026-04-29)

### `ArrayView<T>.CopyToHostAsync()` partial-readback extension

New extension on `ArrayView<T>` (and `ArrayView1D<T, TStride>`) that does a real per-backend partial readback for the view's byte range. The data outside the view never crosses the device-host boundary.

```csharp
// AV1 YUV plane separation - one device buffer, three planes:
var y = await dRecon.View.SubView(0,            yLen ).CopyToHostAsync();
var u = await dRecon.View.SubView(yLen,         uvLen).CopyToHostAsync();
var v = await dRecon.View.SubView(yLen + uvLen, uvLen).CopyToHostAsync();
```

Per-backend dispatch (no full-buffer readback + CPU slice anywhere):

- **WebGPU** - `queue.CopyBufferToBuffer(srcBuf, srcOffset, staging, 0, byteCount)` -> `mapAsync` of just `[byteOffset, byteOffset+byteCount)`.
- **WebGL** - GL-worker `ReadbackAndGetUint8ArrayAsync(buf, sourceByteOffset, byteCount)` partial range path.
- **Wasm** - `Uint8Array(SharedBuffer, byteOffset, byteCount)` window onto exactly the slice's bytes; the rest of wasm linear memory is not touched.
- **CUDA / OpenCL / CPU** - ILGPU's native `view.CopyToCPU(target)` calls `cudaMemcpy` / `clEnqueueReadBuffer` / direct memcpy for just the view's range; the view's offset and length encode the partial copy.

Closes the `Buffer.BlockCopy` cardinal-rule violation in SpawnDev.Codecs decoder integration: consumers can now request per-channel / per-plane slices without the host iterating over codec data.

### WebGPU `Half` NaN/Inf bit-pattern codegen fix

WGSL multi-compare paths (`isNativeFloatUnordered`, `isNativeFloatEqualLike`) emit IEEE 754 bit-pattern checks for `IsNaN` / `IsInf` / `IsFinite`. The 4.9.2 codegen used the f32 mask constants (`0x7F800000` / `0x007FFFFF`) and `bitcast<u32>(operand)` directly on every operand type, including `f16`. WGSL rejects `bitcast<u32>(f16)` as an invalid bitcast, so any kernel with a multi-compare on `Half` operands failed shader validation. `Half` round-trip / arithmetic / min-max tests passed (single-compare path, no IR inversion), but `HalfNaNComparisonTest` failed.

`WGSLCodeGenerator` now routes f16 through `bitcast<u32>(vec2<f16>(x, 0.0h))` with the f16 mask constants (`0x7C00` exponent, `0x03FF` mantissa). f32 / f64 paths unchanged.

### `P2PDispatcher` test expectation alignment

`P2P_Dispatcher_Create` was asserting the historical `DispatchTimeoutMs == 30_000` default. The default was intentionally raised to `60_000` (per the doc comment on `P2PDispatcher.DispatchTimeoutMs`: 30s was too tight for >1MB result buffers and 10-way concurrent dispatch). Test updated to match the implementation.

## 4.9.2 (2026-04-29)

### OpenCL backend phi-binding-per-target fix

The OpenCL backend was emitting all phi bindings unconditionally before a conditional branch's terminator, even when the branch was about to take an exit edge that didn't need the back-edge update. When a non-phi SSA value `u` aliased to a loop's phi `v` (the C# pattern `u = v` inside `do { u = v; ...; v = compute(); } while (cond);`), the unconditional back-edge phi update stomped `u` on the path that exited the loop, producing wrong values for any `u - v` style read after the loop.

`CLCodeGenerator.Terminators.cs` now mirrors `PTXCodeGenerator`'s `BindPhis(target)` approach - phi bindings emit only on the edge actually being taken. `IfBranch` calls `BindPhis(trueTarget)` inside the if-block and `BindPhis(falseTarget)` after, with `ResetPhiBindingScope()` between blocks. `UnconditionalBranch` and `SwitchBranch` similarly per-target. CPU + CUDA were unaffected because their backend codegens don't share this aliasing pathology.

Diagnosed against SpawnDev.Codecs `Av1RangeCoderGpu_CdfQ15_RoundTrip_AllBackends` (was 1/3 FAIL on OpenCL: `sym[1]: input=1 decoded=0`) and `Av1CoefDecoderGpu_RoundTrip_*` (was 4/15 FAIL on OpenCL with `[decEob] Expected '1' but got '2'`). Same backend fix closes both - **18/18 PASS** post-fix across CPU + CUDA + OpenCL.

### Rolled up from 4.9.2-rc.X series

The 4.9.2 stable cut consolidates the rc.7 -> rc.30 series:

- **rc.30 (this release):** OpenCL phi-binding-per-target (above).
- **rc.29:** Respin to actually deliver Tuvok's signed `Div` fix - rc.28 bumped this csproj's version but kept the transitive dependency at `SpawnDev.ILGPU.Fork 2.0.1`, so consumers' resolved `ILGPU.dll` was still the unfixed Apr-23 build. rc.29 (now stable) bundles `SpawnDev.ILGPU.Fork 2.0.3` + `SpawnDev.ILGPU.Algorithms.Fork 2.0.3` with the corrected XML rewriter table. Also added: T4 drift CI guard + four-package version-sync CI guard.
- **rc.28:** IR signed `Div by pow2` no longer rewrites to `Shr` (rewrite was floor-toward-negative-infinity vs CLR / IL `div` truncate-toward-zero, off-by-one for every odd-negative dividend; gated on `(flags & ArithmeticFlags.Unsigned)`). Plus: Wasm `[NoInlining]` void helpers no longer silently dropped + `WasmAccelerator.WorkerCount` public read-only diagnostic.
- **rc.27:** Wasm wait/notify-free + worker headroom default. Removed last `memory.atomic.wait32`/`notify` call from in-kernel `EmitBarrier`; default `WasmBackendOptions.WorkerCount` changed from `hardwareConcurrency` to `Math.Max(2, hardwareConcurrency - 2)` (leaves 2 cores for browser UI / Mono / OS). Wasm RadixSort 18/18 PASS including 4M tests.
- **rc.26:** IEEE 754 NaN/Inf correctness across WGSL / GLSL / Wasm / OpenCL. Closes `clt+brfalse` -> `cge+brtrue [Unordered]` IR-inversion bug (4 backends ignored the unordered flag, silently corrupting NaN multi-compare flag-bit kernels).
- **rc.18:** Helper Method Fn-Definition Emission - Compile Cliff Fix. Tag a helper with `[MethodImpl(MethodImplOptions.NoInlining)]` and SpawnDev.ILGPU emits a real WGSL/GLSL `fn` definition + N call sites instead of N inline expansions. Avoids browser shader validator size limits (Tint rejecting `Invalid BindGroupLayout`).
- **rc.10:** `AcceleratorRequirements` capability-gating API + `UnsupportedKernelFeatureException` typed codegen errors + `LocalMemory<T>(N >= 32)` WGSL codegen 5-layer fix.
- **rc.7-9:** Float16 (Half) everywhere - native or emulated; i64 `Atomic.Add` on WebGPU lock-free CAS loop; WGSL break-in-loop codegen fix; mixed atomic/non-atomic buffer access fix; WGSL shader validation errors surfaced; implicit `SynchronizeAsync` before readback (parity with desktop backends); stream ordering verified correct; WebGL unsupported atomic guards; GLSL `IsReturnExit` defense-in-depth.

## 4.9.0

### Complete Sub-Word Data Type Support

Full `Int8`, `UInt8`, `Int16`, `UInt16`, and `Float16` (`ILGPU.Half`) buffer support across all 6 GPU backends. Sub-word types are stored packed and extracted with correct stride on every backend - no more data corruption from type promotion mismatches.

- **WebGPU** - Packed into `array<atomic<u32>>` storage buffers. Load via `atomicLoad` + shift + mask + sign-extend/zero-extend. Store via thread-safe `atomicAnd` + `atomicOr` for packed writes (prevents data races when threads write different halves of the same word). Float16 uses inline IEEE 754 f16-to-f32 conversion in WGSL.
- **Wasm** - Native `i32.load8_s`/`i32.load8_u`/`i32.load16_s`/`i32.load16_u`/`i32.store8`/`i32.store16` opcodes. Float16 via `EmitF16ToF32`/`EmitF32ToF16` for direct ArrayView load/store.
- **WebGL** - `texelFetch` from R32I texture with shift+mask extraction in GLSL. Float16 via `_f16_to_f32`/`_f32_to_f16` using `uintBitsToFloat`/`floatBitsToUint`.
- **OpenCL** - Float16 promoted to `float` compute type. `vload_half`/`vstore_half` for buffer access (handles 2-byte stride internally).
- **CUDA/CPU** - Native support, no changes needed.

### ILGPU.Half Intrinsics

- `Half.Abs`, `Half.Min`, `Half.Max`, `Half.Clamp` - GPU-accelerated half-precision math
- Implicit `System.Half` <-> `ILGPU.Half` conversion operators for seamless interop
- Use `ILGPU.Half` (not `System.Half`) in kernel signatures for correct transpilation

### CopyFromJS - Zero-Copy JS-to-GPU Transfer

New `IBrowserMemoryBuffer.CopyFromJS()` methods accept `TypedArray` or `ArrayBuffer` and write directly to GPU memory without .NET heap allocation. Available on all 3 browser backends (WebGPU, WebGL, Wasm).

```csharp
// Write JS data directly to GPU buffer - no .NET allocation
var jsArray = new Int16Array(data);
((IBrowserMemoryBuffer)buffer).CopyFromJS(jsArray);
```

## 4.8.0

### Worker Function Caching (3-4x Speedup)

Wasm backend now caches compiled `AsyncFunction` objects in the worker bootstrap. Previously, V8 recompiled each unique script string on every dispatch. Caching eliminates recompilation overhead - **3-4x faster** kernel dispatch on repeated calls.

### Full Worker Parallelism

Non-barrier Wasm workers uncapped from 2 to full `navigator.hardwareConcurrency`. Barrier-limited workers remain capped for synchronization correctness, but non-barrier kernels now use all available cores.

### Memory Leak Fixes

- `AllWasmBinaries` and `AllKernelInfos` collections gated behind debug dump flag - no longer accumulate in production
- `_dispatchLog` gated with `VerboseLogging` - eliminated unbounded log growth
- `ExtraImportCount` for correct Wasm function index calculation

### Barrier Count Auto-Correction

Automatic detection and correction of barrier count mismatches between the kernel's declared barrier count and the actual barriers found during compilation.

## 4.7.1

### GPU Test Verification (`GpuTestVerify`)

Shared utility for verifying test results on the GPU without CPU readback. Data stays on the accelerator - CPU reads back only a few bytes of violation counts.

- `VerifyDescendingSort` / `VerifyAscendingSort` - Sort order + index integrity + key-value tracking
- `CompareBuffers` - Float comparison returning `(meanAbsError, maxAbsError)`
- **10x+ faster** verification - 4M element RadixSort went from 120s timeout to 11s on CPU

### QR Code Library (`SpawnDev.ILGPU.QR`)

GPU-accelerated QR code encoder + decoder. Zero external dependencies.

- **Encoder** - All 40 QR versions, 4 EC levels, byte mode, 8 mask patterns with penalty scoring
- **Renderer** - GPU kernel for pixel rendering + CPU fallback + logo overlay (EC level H)
- **Decoder** - Grayscale -> binarize -> finder detection -> grid sampling -> unmask -> Reed-Solomon -> data decode
- **Round-trip verified** - Encode -> render -> decode = exact match, including with logo overlay

### CPU Default Optimization

CPU backend default changed from warp=4/warps=4 (group size 16) to **warp=8/warps=8 (group size 64)**, matching the Wasm backend's proven configuration. 4M element RadixSort: **TIMEOUT -> 11 seconds**. CPU is now faster than Wasm for the same workloads.

### DI Integration

- `AddPlatformCrypto()` - registers platform-appropriate `IPortableCrypto` (WebCrypto in browser, System.Security.Cryptography on desktop)
- `WebTorrentClient` registered as DI singleton with tracker discovery
- All test classes receive `IPortableCrypto` via constructor injection

## 4.6.0

### Wasm Fiber-Based Barrier Dispatch

Complete rewrite of the Wasm backend's barrier synchronization model. Kernels with barriers now use a **fiber-based phase dispatch** - each barrier becomes a yield point where the kernel saves state and re-enters at the next phase. A **Wasm-native phase dispatcher** handles the entire thread/phase loop inside WebAssembly, eliminating JS-Wasm boundary crossings between phases. Barriers use **pure spin synchronization** via `i32.atomic.load` loops for correct multi-worker execution at full `hardwareConcurrency`.

- **Full ILGPU Algorithms on Wasm** - All RadixSort variants (int, uint, float, pairs, descending, 100K-4M+ elements), Scan, Reduce, Histogram. Previously limited to <=64 elements.
- **Pure spin barriers** - Replaced `memory.atomic.wait32`/`memory.atomic.notify` with atomic load spin loops after discovering a [V8 Atomics.wait visibility bug](https://issues.chromium.org/issues/495679735) where `wait32` returning "not-equal" does not provide happens-before guarantees for third-party stores with 3+ workers. [Live interactive demo](https://lostbeard.github.io/v8-atomics-wait-bug/).
- **20+ bugs fixed** - fiber yield-per-phase, br depth miscalculation, scratch overflow, shared memory stomping, stale dispatch state, completion state persistence, shared memory alloca overlap (same-size dedup), IR address space aliasing (LowerStructures -> LowerArrays -> InferAddressSpaces chain), struct/scratch overlap, per-worker scratch isolation, atomic RMW opcode table, unsigned comparison, Float16, ViewSourceSequencer, subViewByteOffset, CopyFromBuffer, and more.
- **ShaderDebugService** - auto-dumps all generated WGSL, GLSL, and Wasm binaries to a local folder on every kernel compilation. Backend-organized subfolders. IDB persistence. Full metadata headers.
- **Test results writer** - `UnitTestsView` writes `latest.json` (live progress) and timestamped `test-run-*.json` (history) to the debug folder

## 4.4.0

### Capturing Lambda Kernels

Write GPU kernels as C# lambdas that capture local variables. Captured scalar values are automatically passed to the GPU at dispatch time - no boilerplate, no separate static methods.

```csharp
int multiplier = 5;
float offset = 0.5f;
var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>>(
    (index, buf) => { buf[index] = index * multiplier + offset; });
kernel((Index1D)length, buffer.View);
```

### DelegateSpecialization - Higher-Order GPU Kernels

Write one kernel that accepts different operations as parameters. The delegate is resolved at dispatch time and its body is inlined directly into the kernel via compile-time specialization - no function pointers, no overhead.

```csharp
static void MapKernel(Index1D index, ArrayView<int> buf,
    DelegateSpecialization<Func<int, int>> transform)
{
    buf[index] = transform.Value(buf[index]);
}

static int Negate(int x) => -x;
static int DoubleIt(int x) => x * 2;

var kernel = accelerator.LoadAutoGroupedStreamKernel<
    Index1D, ArrayView<int>, DelegateSpecialization<Func<int, int>>>(MapKernel);

kernel(size, buffer, new DelegateSpecialization<Func<int, int>>(Negate));
kernel(size, buffer, new DelegateSpecialization<Func<int, int>>(DoubleIt));
```

## 4.0.0

- **WebGPU backend refactor** - `SharedMemoryResolver`, `UniformityAnalyzer`, per-function emulation trimming, dead variable elimination, i64 constant hoisting, WGSL pre-validation
- **WebGPU RadixSort** - All variants passing (4M+ elements, pairs, descending)
- **Device loss detection** - WebGPU `device.lost` promise, WebGL `webglcontextlost` event
- **Unified test infrastructure** - `PlaywrightMultiTest` runs all tests (desktop + browser) in a single `dotnet test` invocation
