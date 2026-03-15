# WebGL Backend

Transpiles ILGPU IR → GLSL ES 3.0 shaders. Uses Transform Feedback for output.

## Key Files
- `Backend/GLSLKernelFunctionGenerator.cs` — kernel codegen
- `Backend/GLSLEmulationLibrary.cs` — i64/f64 emulation for GLSL
- `WebGLAccelerator.cs` — dispatch, context management, device loss handling
- `glWorker.js` — off-main-thread Web Worker for GL calls

## Hard Constraints
- **No shared memory, atomics, or barriers** — fundamentally limited by WebGL 2.0 / GLSL ES 3.0.
- **Transform Feedback** for kernel output — output data written via varying variables.
- **Context loss** — `glWorker.js` monitors `webglcontextlost`/`webglcontextrestored`. `IsContextLost` guards dispatch.
- **Output varying index** — `BuildOutputVaryingIndex` dictionaries for O(1) lookup.
- **i64/f64 emulation** — same as WebGPU but in GLSL syntax.
