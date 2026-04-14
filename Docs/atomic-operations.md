# Atomic Operations by Backend

Tracks support for all atomic operations across all backends and data types.
Updated: 2026-04-14

**Legend:**
- [x] PASS - native or correctly emulated, verified with tests
- [CAS] - emulated via compare-and-swap loop (correct, slightly slower)
- [!] NOT SUPPORTED - throws `NotSupportedException` at kernel compilation time
- [V] VOTE - WebGL Transform Feedback vote pattern (accumulation only, return value always 0)
- [-] NOT TESTED - no dedicated tests yet

---

## Int32 / UInt32 Atomics

Natively supported on all backends with atomics capability.

| Operation | WebGPU | Wasm | WebGL | CUDA | OpenCL | CPU |
|-----------|:------:|:----:|:-----:|:----:|:------:|:---:|
| Add | [x] | [x] | [V] | [x] | [x] | [x] |
| And | [x] | [x] | [!] | [x] | [x] | [x] |
| Or | [x] | [x] | [!] | [x] | [x] | [x] |
| Xor | [x] | [x] | [!] | [x] | [x] | [x] |
| Min | [x] | [x] | [!] | [x] | [x] | [x] |
| Max | [x] | [x] | [!] | [x] | [x] | [x] |
| Exchange | [x] | [x] | [!] | [x] | [x] | [x] |
| CompareExchange | [x] | [x] | [!] | [x] | [x] | [x] |

**WebGL note:** Only `Atomic.Add` is partially supported via the Transform Feedback vote pattern. Each thread emits its increment to a TF varying; JavaScript sums all per-vertex contributions post-draw. The return value (old value before add) is always 0 - kernels using the return value for slot allocation will silently produce wrong results. All other atomic operations throw `NotSupportedException`.

---

## Int64 / UInt64 Atomics

i64 is emulated as `vec2<u32>` on WebGPU and WebGL. Native on Wasm, CUDA, OpenCL, CPU.

| Operation | WebGPU | Wasm | WebGL | CUDA | OpenCL | CPU |
|-----------|:------:|:----:|:-----:|:----:|:------:|:---:|
| Add | [x] CAS | [x] | [!] | [x] | [x] | [x] |
| And | [x] Dual | [x] | [!] | [x] | [x] | [x] |
| Or | [x] Dual | [x] | [!] | [x] | [x] | [x] |
| Xor | [x] Dual | [x] | [!] | [x] | [x] | [x] |
| Min | [!] | [CAS] | [!] | [x] | [x] | [x] |
| Max | [!] | [CAS] | [!] | [x] | [x] | [x] |
| Exchange | [!] | [x] | [!] | [x] | [x] | [x] |
| CompareExchange | [!] | [x] | [!] | [x] | [x] | [x] |

### WebGPU i64 emulation details

**Bitwise (And/Or/Xor) - "Dual i32":** Two independent `atomicAnd`/`atomicOr`/`atomicXor` on lo and hi halves. Correct because bitwise operations are component-independent - the result of each half doesn't depend on the other.

**Add - "CAS loop":** Lock-free CAS loop on the lo half + `atomicAdd` on the hi half with carry. The CAS loop (`atomicCompareExchangeWeak`) on the lo half serializes low-half updates. When the lo half wraps (unsigned overflow), a carry of 1 is added to the hi half via `atomicAdd`. This works because `atomicAdd` is commutative - multiple threads can add their carry values in any order and the final result is correct.

**Min/Max/Exchange/CompareExchange - NOT SUPPORTED:** These operations require both halves of the i64 value to be read and/or written atomically as a single 64-bit unit. WGSL only has 32-bit atomic operations. There is no way to atomically compare or exchange two u32 words simultaneously without hardware support for 64-bit atomics. Attempting to use these operations on i64 with a WebGPU accelerator throws `NotSupportedException` at kernel compilation time with a clear error message.

---

## Float32 Atomics

| Operation | WebGPU | Wasm | WebGL | CUDA | OpenCL | CPU |
|-----------|:------:|:----:|:-----:|:----:|:------:|:---:|
| Add | [CAS] | [CAS] | [!] | [x] | [x] | [x] |
| Min | [CAS] | [CAS] | [!] | [CAS] | [x] | [x] |
| Max | [CAS] | [CAS] | [!] | [CAS] | [x] | [x] |
| Exchange | [CAS] | [CAS] | [!] | [x] | [x] | [x] |

**CAS loop:** Buffer is declared as `atomic<u32>`. Values are `bitcast` between `f32` and `u32`. The CAS loop reads the current u32, bitcasts to f32, computes the new value, bitcasts back to u32, and attempts `atomicCompareExchangeWeak`. Retries on contention.

---

## Float64 Atomics

| Operation | WebGPU | Wasm | WebGL | CUDA | OpenCL | CPU |
|-----------|:------:|:----:|:-----:|:----:|:------:|:---:|
| Add | [!] | [CAS] | [!] | [x] | [x] | [x] |
| Min | [!] | [CAS] | [!] | [x] | [x] | [x] |
| Max | [!] | [CAS] | [!] | [x] | [x] | [x] |
| Exchange | [!] | [CAS] | [!] | [x] | [x] | [x] |

**WebGPU:** f64 is emulated as two 32-bit words (Dekker `vec2<f32>` or Ozaki `vec4<f32>`). Like i64, atomic arithmetic requires both words to update atomically. Throws `NotSupportedException`.

**Wasm:** f64 is native. CAS loop uses `i64.atomic.rmw.cmpxchg` with `f64.reinterpret_i64` / `i64.reinterpret_f64` for bitwise conversion.

---

## Usage Examples

### i32 Atomic.Add (works everywhere with atomics)

```csharp
static void CountKernel(Index1D index, ArrayView<int> data, ArrayView<int> counter)
{
    if (data[index] > 0)
        Atomic.Add(ref counter[0], 1);
}
```

### i64 Atomic.And (bitwise - works on WebGPU, Wasm, desktop)

```csharp
static void ClearBitKernel(Index1D index, ArrayView<long> faceMasks)
{
    // Clear this thread's bit from the shared mask
    Atomic.And(ref faceMasks[0], ~(1L << index));
}
```

### i64 Atomic.Add (CAS loop on WebGPU, native on Wasm/desktop)

```csharp
static void AccumulateKernel(Index1D index, ArrayView<long> accumulator, ArrayView<long> values)
{
    Atomic.Add(ref accumulator[0], values[index]);
}
```

### f32 Atomic.Add (CAS loop on browser backends, native on desktop)

```csharp
static void SumKernel(Index1D index, ArrayView<float> data, ArrayView<float> sum)
{
    Atomic.Add(ref sum[0], data[index]);
}
```

---

## WebGPU Binding Limit Impact

WebGPU has a `maxStorageBuffersPerShaderStage` limit of 10 (Chrome). Each `ArrayView` kernel parameter uses one storage buffer binding, plus one for `_scalar_params`.

Atomic operations do NOT add extra bindings - they operate on existing buffer bindings using `atomicLoad`, `atomicStore`, `atomicAdd`, `atomicAnd`, etc.

If a buffer has BOTH atomic and non-atomic access patterns (e.g., `Atomic.And(ref buf[x], mask)` and `long val = buf[y]`), the codegen declares the buffer as `array<atomic<u32>>` and emits `atomicLoad` for all reads - not just atomic operations. This ensures WGSL validation passes (Chrome rejects mixed atomic/non-atomic access on the same buffer).

---

## Test Coverage

| Test | File | What It Verifies |
|------|------|-----------------|
| AtomicAndLongTest | Tests19 | i64 Atomic.And with dual i32 atomics |
| AtomicAddLongConcurrentTest | Tests19 | i64 Atomic.Add CAS loop with 64 concurrent writers |
| AtomicAddLongCarryTest | Tests19 | i64 Atomic.Add CAS loop carry between lo/hi halves |
| FullCombinedPatternTest | Tests19 | i64 Atomic.And + Add + bit packing combined |
| ChainedDispatch3KernelTest | Tests18 | Implicit stream ordering (no sync between dispatches) |
| ImplicitSyncOnReadbackTest | Tests18 | CopyToHostAsync without explicit SynchronizeAsync |
