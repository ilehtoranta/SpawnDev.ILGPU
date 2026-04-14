# Data Type Support by Backend

Tracks verified support for all data types across all 7 backends.
Updated: 2026-04-13

**Legend:**
- [x] PASS - verified with unit tests (real data, real kernels, real verification)
- [ ] FAIL - tests exist, currently failing
- [!] KNOWN LIMITATION - architectural constraint, not a bug
- [-] NOT TESTED - no tests yet, status unknown
- [N/A] - not applicable to this backend

---

## Buffer Read (Load from ArrayView)

| Type | C# Type | Size | WebGPU | WebGPU NoSub | Wasm | WebGL | CUDA | OpenCL | CPU |
|------|---------|------|:------:|:------------:|:----:|:-----:|:----:|:------:|:---:|
| Int8 | sbyte | 1B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| UInt8 | byte | 1B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| Int16 | short | 2B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| UInt16 | ushort | 2B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
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
| Int8 | sbyte | 1B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| UInt8 | byte | 1B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| Int16 | short | 2B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| UInt16 | ushort | 2B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| Int32 | int | 4B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| UInt32 | uint | 4B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| Int64 | long | 8B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| UInt64 | ulong | 8B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| Float16 | Half | 2B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| Float32 | float | 4B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| Float64 | double | 8B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |

## End-to-End (Read + Kernel Process + Write)

| Type | C# Type | Size | WebGPU | WebGPU NoSub | Wasm | WebGL | CUDA | OpenCL | CPU |
|------|---------|------|:------:|:------------:|:----:|:-----:|:----:|:------:|:---:|
| Int8 | sbyte | 1B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| UInt8 | byte | 1B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| Int16 | short | 2B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| UInt16 | ushort | 2B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| Int32 | int | 4B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| UInt32 | uint | 4B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| Int64 | long | 8B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| UInt64 | ulong | 8B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| Float16 | Half | 2B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| Float32 | float | 4B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| Float64 | double | 8B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |

## Buffer RoundTrip (CopyFromCPU -> CopyToHostAsync, no kernel)

| Type | C# Type | Size | WebGPU | WebGPU NoSub | Wasm | WebGL | CUDA | OpenCL | CPU |
|------|---------|------|:------:|:------------:|:----:|:-----:|:----:|:------:|:---:|
| Int8 | sbyte | 1B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| UInt8 | byte | 1B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| Int16 | short | 2B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| UInt16 | ushort | 2B | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| Float16 | Half | 2B | [-] | [-] | [-] | [-] | [-] | [-] | [-] |

## Half Math Intrinsics

| Function | WebGPU | WebGPU NoSub | Wasm | WebGL | CUDA | OpenCL | CPU |
|----------|:------:|:------------:|:----:|:-----:|:----:|:------:|:---:|
| Abs | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| Min/Max | [x] | [x] | [x] | [x] | [x] | [x] | [x] |
| Clamp | [-] | [-] | [-] | [-] | [-] | [-] | [-] |

## CopyFromJS (Browser-only: JS TypedArray/ArrayBuffer -> GPU)

| Type | C# Type | Size | WebGPU | Wasm | WebGL |
|------|---------|------|:------:|:----:|:-----:|
| Int32 | int | 4B | [x] | [x] | [x] |
| Float32 | float | 4B | [x] | [x] | [x] |

## Atomic Operations

See **[Docs/atomic-operations.md](atomic-operations.md)** for the complete per-operation support matrix.

| Type | C# Type | WebGPU | Wasm | WebGL | CUDA | OpenCL | CPU |
|------|---------|:------:|:----:|:-----:|:----:|:------:|:---:|
| Int32 | int | [x] | [x] | [!] Add only (vote TF) | [x] | [x] | [x] |
| UInt32 | uint | [x] | [x] | [!] Add only (vote TF) | [x] | [x] | [x] |
| Int64 | long | [x] Add/bitwise, [!] Min/Max/Exch/CAS | [x] | [!] | [x] | [x] | [x] |
| UInt64 | ulong | [x] Add/bitwise, [!] Min/Max/Exch/CAS | [x] | [!] | [x] | [x] | [x] |
| Float32 | float | [x] CAS loop | [x] CAS loop | [!] | [x] | [x] | [x] |
| Float64 | double | [!] | [x] CAS loop | [!] | [x] | [x] | [x] |

**[!]** = Throws `NotSupportedException` at kernel compilation time. See [atomic-operations.md](atomic-operations.md) for details.

---

## Implementation Summary

### Sub-word buffer access (Int8, UInt8, Int16, UInt16, Float16)

All sub-word types now have **complete Read/Write/EndToEnd support on ALL 7 backends**.

| Backend | Mechanism | Signed/Unsigned Detection |
|---------|-----------|--------------------------|
| **WebGPU** | `array<atomic<u32>>` + atomicAnd/atomicOr for Store, atomicLoad for Read. IEEE 754 f16<->f32 inline conversion for Float16. | `EntryPoint.Parameters[N].GetGenericArguments()[0]` CLR type check |
| **Wasm** | Native `i32.load8_s/u`, `i32.load16_s/u`, `i32.store8`, `i32.store16` opcodes. Float16 via EmitF16ToF32/EmitF32ToF16. | CLR type trace via `_generatorArgs.EntryPoint.Parameters` |
| **WebGL** | `texelFetch` from R32I texture, shift+mask extraction. TF output with sub-word packing in `glWorker.js`. Float16 via GLSL f16<->f32 bit manipulation. | `EntryPoint.Parameters[N]` CLR type check |
| **OpenCL** | Native types for Int8/UInt8/Int16/UInt16. Float16 via `vload_half`/`vstore_half` with tracked LEA base pointer. | Native type support |
| **CPU/CUDA** | Native sub-word support, no special handling needed. | Native |

### Test Coverage

**147 tests total** across 10 sub-word test methods x 7 backends + Half intrinsics:
- Int8: 28 tests (RoundTrip + Read + Write + EndToEnd x 7 backends)
- UInt8: 28 tests
- Int16: 35 tests (+ existing CopyFromJS tests)
- UInt16: 28 tests
- Float16: 21 tests (Read + Write + EndToEnd x 7 backends)
- Half Abs: 7 tests
- Half MinMax: 7 tests

### Test File
`SpawnDev.ILGPU.Demo.Shared/UnitTests/BackendTestBase.Tests17.BrowserBuffer.cs`

### How to Run
```bash
# All sub-word tests
dotnet test PlaywrightMultiTest/PlaywrightMultiTest.csproj --filter "FullyQualifiedName~Int8|FullyQualifiedName~UInt8|FullyQualifiedName~Int16|FullyQualifiedName~UInt16|FullyQualifiedName~Float16|FullyQualifiedName~Half_"
```
