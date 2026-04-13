# Data Type Support by Backend

Tracks verified support for all data types across all 6 backends.
Updated: 2026-04-12

**Legend:**
- [x] PASS - verified with unit tests (real data, real kernels, real verification)
- [ ] FAIL - tests exist, currently failing
- [!] KNOWN LIMITATION - architectural constraint, not a bug (e.g., WebGL Transform Feedback has no sub-word output)
- [-] NOT TESTED - no tests yet, status unknown
- [N/A] - not applicable to this backend

---

## Buffer Read (Load from ArrayView)

| Type | C# Type | Size | WebGPU | WebGPU NoSub | Wasm | WebGL | CUDA | OpenCL | CPU |
|------|---------|------|:------:|:------------:|:----:|:-----:|:----:|:------:|:---:|
| Int8 | sbyte | 1B | [-] | [-] | [-] | [-] | [-] | [-] | [-] |
| UInt8 | byte | 1B | [-] | [-] | [-] | [-] | [-] | [-] | [-] |
| Int16 | short | 2B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| UInt16 | ushort | 2B | [-] | [-] | [-] | [-] | [-] | [-] | [-] |
| Int32 | int | 4B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| UInt32 | uint | 4B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| Int64 | long | 8B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| UInt64 | ulong | 8B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| Float16 | Half | 2B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| Float32 | float | 4B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| Float64 | double | 8B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |

## Buffer Write (Store to ArrayView)

| Type | C# Type | Size | WebGPU | WebGPU NoSub | Wasm | WebGL | CUDA | OpenCL | CPU |
|------|---------|------|:------:|:------------:|:----:|:-----:|:----:|:------:|:---:|
| Int8 | sbyte | 1B | [-] | [-] | [-] | [-] | [-] | [-] | [-] |
| UInt8 | byte | 1B | [-] | [-] | [-] | [-] | [-] | [-] | [-] |
| Int16 | short | 2B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| UInt16 | ushort | 2B | [-] | [-] | [-] | [-] | [-] | [-] | [-] |
| Int32 | int | 4B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| UInt32 | uint | 4B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| Int64 | long | 8B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| UInt64 | ulong | 8B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| Float16 | Half | 2B | [-] | [-] | [-] | [-] | [-] | [-] | [-] |
| Float32 | float | 4B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| Float64 | double | 8B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |

## End-to-End (Read + Kernel Process + Write)

| Type | C# Type | Size | WebGPU | WebGPU NoSub | Wasm | WebGL | CUDA | OpenCL | CPU |
|------|---------|------|:------:|:------------:|:----:|:-----:|:----:|:------:|:---:|
| Int8 | sbyte | 1B | [-] | [-] | [-] | [-] | [-] | [-] | [-] |
| UInt8 | byte | 1B | [-] | [-] | [-] | [-] | [-] | [-] | [-] |
| Int16 | short | 2B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| UInt16 | ushort | 2B | [-] | [-] | [-] | [-] | [-] | [-] | [-] |
| Int32 | int | 4B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| UInt32 | uint | 4B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| Int64 | long | 8B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| UInt64 | ulong | 8B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| Float16 | Half | 2B | [-] | [-] | [-] | [-] | [-] | [-] | [-] |
| Float32 | float | 4B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| Float64 | double | 8B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |

## Buffer RoundTrip (CopyFromCPU -> CopyToHostAsync, no kernel)

| Type | C# Type | Size | WebGPU | WebGPU NoSub | Wasm | WebGL | CUDA | OpenCL | CPU |
|------|---------|------|:------:|:------------:|:----:|:-----:|:----:|:------:|:---:|
| Int16 | short | 2B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| Float16 | Half | 2B | [-] | [-] | [-] | [-] | [-] | [-] | [-] |

## CopyFromJS (Browser-only: JS TypedArray/ArrayBuffer -> GPU)

| Type | C# Type | Size | WebGPU | Wasm | WebGL |
|------|---------|------|:------:|:----:|:-----:|
| Int16 | short | 2B | [-] | [-] | [-] |
| Int32 | int | 4B | [x] | [x] | [x] |
| Float32 | float | 4B | [x] | [x] | [x] |

## Math Intrinsics

| Function | sbyte | short | byte | ushort | Half | float | int | long | double |
|----------|:-----:|:-----:|:----:|:------:|:----:|:-----:|:---:|:----:|:------:|
| Abs | [x] | [x] | N/A | N/A | [-] | [x] | [x] | [x] | [x] |
| Min | [x] | [x] | [-] | [-] | [-] | [x] | [x] | [x] | [x] |
| Max | [x] | [x] | [-] | [-] | [-] | [x] | [x] | [x] | [x] |
| Clamp | [-] | [-] | [-] | [-] | [-] | [x] | [x] | [-] | [-] |

Note: Math intrinsics for sub-word types (sbyte, short, byte, ushort) are C# implementations that get promoted to i32 operations in WGSL/GLSL/Wasm. They work on all backends where the core i32 operations work, but need explicit testing to verify.

## Atomic Operations

| Type | C# Type | WebGPU | Wasm | WebGL | CUDA | OpenCL | CPU |
|------|---------|:------:|:----:|:-----:|:----:|:------:|:---:|
| Int32 | int | [x] | [x] | N/A | [x] | [x] | [x] |
| UInt32 | uint | [x] | [x] | N/A | [x] | [x] | [x] |
| Int64 | long | [-] | [x] | N/A | [x] | [x] | [x] |
| Float32 | float | [x] | [x] | N/A | [x] | [x] | [x] |

Note: WebGPU sub-word atomic (atomicAnd/atomicOr on packed u32) is used for Int16 Store. Not true 16-bit atomics but correct behavior.

---

## Implementation Notes

### Sub-word buffer access (Int8, Int16, UInt16, Float16)

These types are smaller than the minimum buffer element size on GPU backends:
- **WebGPU:** Minimum storage buffer element is u32 (4 bytes). Sub-word values packed 2 or 4 per u32. Extraction via shift+mask in WGSL. Stores via atomic CAS loop.
- **WebGL:** Minimum texel size is 32-bit (R32I/R32F). Same packing approach needed in GLSL.
- **Wasm:** Native sub-word load/store instructions (i32.load16_s, i32.store16, etc.). No packing needed.
- **Desktop (CUDA/OpenCL/CPU):** Native sub-word support. No special handling.

### Float16 emulation

- **With shader-f16:** Native f16 type in WGSL. Buffer uses array<f16>.
- **Without shader-f16:** Promoted to f32 in shader. Buffer packed as u16 in u32 words (same as Int16). IEEE 754 half<->single conversion at load/store.
- **Wasm:** Uses i32.load16_u + F16ToF32 conversion function. Native 16-bit load, software conversion.

### Float64 emulation

- **WebGPU:** Dekker (vec2<f32>) or Ozaki (vec4<f32>) emulation. ~90 FPS Mandelbrot proven.
- **WebGL:** Same emulation approach.
- **Wasm:** Native f64 support.

---

## Related Documentation

- **Intrinsic parity audit:** `Docs/intrinsic_parity_audit.md` - UnaryArithmetic (27), BinaryArithmetic, Compare, Convert, Atomic operations across all backends
- **Backend limitations:** `Docs/limitations.md`

## Test File Reference

- Int16/Float16/CopyFromJS tests: `SpawnDev.ILGPU.Demo.Shared/UnitTests/BackendTestBase.Tests17.BrowserBuffer.cs`
- RadixSort tests: `BackendTestBase.Tests9.cs`
- Core tests (Int32/Float32/etc): `BackendTestBase.Tests1-8.cs`

## How to Run

```bash
# All backends, all tests
dotnet test PlaywrightMultiTest/PlaywrightMultiTest.csproj

# Specific test
dotnet test PlaywrightMultiTest/PlaywrightMultiTest.csproj --filter "FullyQualifiedName~Int16"

# Specific backend
dotnet test PlaywrightMultiTest/PlaywrightMultiTest.csproj --filter "FullyQualifiedName~WebGPUTests.Int16"
```
