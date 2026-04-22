# WebGL Backend

Transpiles ILGPU IR → GLSL ES 3.0 shaders. Uses Transform Feedback for output.

## Key Files
- `Backend/GLSLKernelFunctionGenerator.cs` — kernel codegen
- `Backend/GLSLEmulationLibrary.cs` — i64/f64/f16 emulation for GLSL
- `WebGLAccelerator.cs` — dispatch, context management, device loss handling
- `glWorker.js` — off-main-thread Web Worker for GL calls

## Hard Constraints
- **No shared memory, atomics, or barriers** — fundamentally limited by WebGL 2.0 / GLSL ES 3.0.
- **Transform Feedback** for kernel output — output data written via varying variables.
- **Context loss** — `glWorker.js` monitors `webglcontextlost`/`webglcontextrestored`. `IsContextLost` guards dispatch.
- **Output varying index** — `BuildOutputVaryingIndex` dictionaries for O(1) lookup.
- **i64/f64 emulation** — same as WebGPU but in GLSL syntax.

## Float16 (Half) — Emulated Only

`Capabilities.Float16 = true` on WebGL via emulation. `Capabilities.Float16Native = false` — WebGL 2.0 / GLSL ES 3.0 has no hardware Float16 path.

**How it works:**
- **Type mapping:** Half → `float` in GLSL (f32 arithmetic). See `GLSLTypeGenerator.cs:113`.
- **Storage:** 2 halves packed per `int` texel in R32I buffer textures (same layout as Int16 sub-word).
- **Load:** `texelFetch` the u32 word, bit-extract the u16 via shift+mask, call `_f16_to_f32(uint)` from `GLSLEmulationLibrary.F16Functions`.
- **Store:** call `_f32_to_f16(float)` on the f32 value, cast the returned uint to int, write to the Transform Feedback varying. Host-side readback reassembles the packed u16 stream into the original `Half[]` buffer.

**Algorithm-family Half tests (RadixSort/Scan/Reduce) still skip on WebGL** because they require shared memory + barriers — those limitations are structural to WebGL, not Half-specific. The 5 non-algorithm Half tests (`HalfBufferRoundTrip`, `HalfArithmetic`, `HalfMinMax`, `HalfEdgeCases`, `HalfMixedType`) run.

**Why emulation is lossless:** every f16 value is exactly representable as f32 (f16 is a strict subset of f32's encoding). The WGSL, GLSL, and Wasm emulation paths all match the same IEEE 754 bit conversion behavior so results on emulated WebGL and emulated WebGPU agree byte-for-byte on the same inputs. Denormals flush to signed zero, Inf/NaN propagate via mantissa preservation.
