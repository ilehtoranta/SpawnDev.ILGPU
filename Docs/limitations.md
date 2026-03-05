# Limitations & Constraints

SpawnDev.ILGPU runs ILGPU across both Blazor WebAssembly and desktop environments. Each platform introduces specific constraints. This page documents all known limitations and their workarounds.

## Critical: No Blocking Calls

**The #1 rule of Blazor WASM:** Never block the main thread.

Blazor WebAssembly runs on a single thread. Blocking that thread prevents JavaScript promises from resolving, causing a permanent deadlock.

### ❌ Will Deadlock

```csharp
buffer.GetAsArray1D();               // DEADLOCKS — calls synchronous readback internally
var result = task.Result;            // DEADLOCKS — blocks on async result
task.Wait();                         // DEADLOCKS
task.GetAwaiter().GetResult();       // DEADLOCKS
```

### ✅ Correct Async Pattern

```csharp
// Synchronize() flushes commands to the backend (non-blocking, safe in WASM)
accelerator.Synchronize();

// SynchronizeAsync() flushes AND waits for completion
await accelerator.SynchronizeAsync();

// CopyToHostAsync() is the only way to read GPU data back to CPU
var results = await buffer.CopyToHostAsync<float>();
```

> **Rule:** Always propagate `async/await` through your entire call stack. Never use `.Result`, `.Wait()`, or `.GetAwaiter().GetResult()` on the main thread.

## `throw` Not Supported in Kernels

The WGSL and GLSL transpilers cannot translate the IL `throw` instruction. If any code in your kernel (or methods it calls) contains a `throw`, compilation will fail.

### Common Offenders

Many `System.Math` methods contain implicit argument validation with `throw`. All browser backends (WebGPU, WebGL, Wasm) include **throw-free redirects** that handle the most common cases automatically:

| Method | Contains `throw`? | Auto-redirected? | Notes |
|--------|-------------------|:----------------:|-------|
| `Math.Clamp(val, min, max)` | ✅ Yes | ✅ Yes | Redirected to `Min(Max(val, min), max)` |
| `Math.Round(x)` | ✅ Yes | ✅ Yes | Redirected to throw-free wrapper |
| `Math.Truncate(x)` | ✅ Yes | ✅ Yes | Redirected to throw-free wrapper |
| `Math.Sign(x)` | ✅ Yes | ✅ Yes | Redirected to throw-free wrapper |
| `MathF.FusedMultiplyAdd` | ✅ Yes | ✅ Yes | Redirected to throw-free wrapper |
| `XMath.Rsqrt(x)` | ✅ Yes | ✅ Yes | Redirected to throw-free wrapper |
| `XMath.Rcp(x)` | ✅ Yes | ✅ Yes | Redirected to throw-free wrapper |
| `MathF.Sin(x)` | ❌ No | — | Safe to use directly |
| `MathF.Sqrt(x)` | ❌ No | — | Safe to use directly |
| `Math.Min(a, b)` | ❌ No | — | Safe to use directly |
| `Math.Max(a, b)` | ❌ No | — | Safe to use directly |

> **Auto-redirects**: The `RegisterMathIntrinsics()` infrastructure in each browser backend automatically intercepts calls to problematic .NET methods and replaces them with throw-free equivalents at compile time. You can use `Math.Clamp`, `Math.Round`, `Math.Truncate`, and `Math.Sign` directly in kernels — they will work on all backends.

### General Rule

Avoid calling any helper method that might throw exceptions. If you're not sure, check the .NET source for the method — if it contains `throw new ArgumentException(...)` or similar, it won't work unless a redirect is registered for it.

## No Reference Types in Kernels

Kernels can only work with **value types** (structs, primitives). Reference types (`class`, `string`, arrays) are not supported.

| ❌ Not Allowed | ✅ Allowed |
|----------------|-----------|
| `string` | `int`, `float`, `double`, `long` |
| `class` instances | `struct` instances |
| `int[]` (managed array) | `ArrayView<int>` |
| `object` | Primitives and value-type structs |

## No `ref` / `out` Parameters

Kernel parameters are passed by value. Use `ArrayView<T>` for output:

```csharp
// ❌ Won't work
static void Bad(Index1D i, ref int result) { }

// ✅ Use a buffer
static void Good(Index1D i, ArrayView<int> result) { result[0] = 42; }
```

## No Recursion

GPU hardware doesn't support call stacks. Recursive functions must be rewritten as iterative loops.

## Float Precision

### f32 (Default for GPU Backends)

WGSL and GLSL natively support 32-bit floats (`f32` / `float`). Using `float` and `MathF` in kernels gives native GPU precision and performance.

### f64 (Double Precision)

`double` is **not natively supported** on most GPU hardware. Both GPU backends provide software emulation:

- **Dekker** (default): `vec2<f32>` — ~48–53 bits mantissa, fast
- **Ozaki**: `vec4<f32>` — full IEEE 754, ~2x slower

Emulated doubles work well for many use cases (fractals, scientific compute) but have performance overhead. For rendering and visual applications, prefer `float`.

> **Deep zoom limitation:** f32 precision limits useful Mandelbrot zoom to ~10⁶× magnification. Emulated f64 extends this significantly.

### i64 (Long / ULong)

`long` and `ulong` are emulated as `vec2<u32>` when `EnableI64Emulation` is true.

## IL Trimming & AOT

ILGPU compiles kernels at runtime by reading .NET IL (Intermediate Language). Both trimming and AOT compilation will break this:

```xml
<PropertyGroup>
  <!-- REQUIRED: ILGPU needs IL reflection at runtime -->
  <PublishTrimmed>false</PublishTrimmed>
  <RunAOTCompilation>false</RunAOTCompilation>
</PropertyGroup>
```

## SharedArrayBuffer Requirements (Wasm Backend)

The Wasm backend uses Web Workers for parallel dispatch. `SharedArrayBuffer` is required for zero-copy data sharing between workers.

### Cross-Origin Isolation

The page must be served with these HTTP headers:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

### Automatic Setup

The demo includes `coi-serviceworker.js` which auto-injects these headers via a service worker — no server configuration needed for development.

### Fallback

Without `SharedArrayBuffer`, the Wasm backend still works but falls back to a single off-thread worker (no multi-worker parallelism).

## Backend Feature Availability

Not all ILGPU features work on all backends:

| Feature | WebGPU | WebGL | Wasm | CPU (Browser) | Cuda | OpenCL | CPU (Desktop) |
|---------|--------|-------|------|---------------|------|--------|---------------|
| Basic kernels | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 1D/2D/3D index | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Scalar params | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Struct params | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| SharedMemory | ✅ | ❌ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| Group.Barrier() | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ |
| Dynamic SharedMemory | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ |
| Group.Broadcast | ✅ | ❌ | ✅ | ❌ | ✅ | ✅¹ | ✅ |
| Atomics | ✅ | ❌ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| Warp/Subgroup ops | ✅² | ❌ | ❌ | ❌ | ✅ | ✅¹ | ✅ |
| f64 emulation | ✅ | ✅ | N/A (native) | N/A (native) | N/A (native) | N/A (native) | N/A (native) |
| i64 emulation | ✅ | ✅ | N/A (native) | N/A (native) | N/A (native) | N/A (native) | N/A (native) |
| ILGPU Algorithms | ✅⁴ | ❌³ | ⚠️⁵ | ⚠️ | ✅ | ✅ | ✅ |

¹ Requires device subgroup support (dynamically detected; some OpenCL 3.0 devices like NVIDIA may lack subgroups)  
² Requires `subgroups` WebGPU extension  
³ Most algorithms require shared memory or atomics  
⁴ WebGPU: RadixSort, Scan, Reduce, Histogram fully supported and tested  
⁵ Wasm: Scan and Reduce work; RadixSort excluded (infinite-loop bug in generated Wasm)  
⚠️ CPU backend works in theory but may crash or have limitations in Blazor WASM single-threaded environment

## Browser Compatibility

| Browser | WebGPU | WebGL | Wasm |
|---------|--------|-------|------|
| Chrome 113+ | ✅ | ✅ | ✅ |
| Edge 113+ | ✅ | ✅ | ✅ |
| Firefox 128+ | ✅ (Nightly) | ✅ | ✅ |
| Safari 18+ | 🧪 Experimental | ✅ | ✅ |
| Mobile Chrome | ✅ (Android) | ✅ | ✅ |
| Mobile Safari | ❌ | ✅ | ✅ |

## Namespace Collision

SpawnDev.ILGPU uses the `SpawnDev.ILGPU` namespace, which can collide with the `ILGPU` namespace from the forked ILGPU library. When both are in scope, use the `global::` prefix:

```csharp
using SpawnDev.ILGPU;       // SpawnDev extensions
using global::ILGPU;         // Original ILGPU types
using global::ILGPU.Runtime; // Original ILGPU runtime
```

This is a known issue preserved for backward compatibility.

## Maximum Parameter Count

ILGPU supports up to ~19 kernel parameters. If you approach this limit, pack related values into structs:

```csharp
// ❌ Too many parameters
static void Bad(Index1D i, ArrayView<float> d, float a, float b, float c, float d2, ...) { }

// ✅ Pack into a struct
public struct Config { public float A; public float B; public float C; public float D; }
static void Good(Index1D i, ArrayView<float> data, Config config) { }
```
