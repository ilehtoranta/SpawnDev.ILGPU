# SpawnDev.ILGPU Changelog

This file tracks notable changes per release. The README's "Recent Highlights" section links here for the full version history.

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
