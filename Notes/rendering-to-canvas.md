# Unified Canvas Renderer API

## Goal Description
The `SpawnDev.ILGPU.Demo` currently uses `PutImageData` indiscriminately to present rendered results (e.g., Mandelbrot) to the canvas. This forces a massive CPU read-back bottleneck every frame for GPU backends.
The goal is to design a unified `ICanvasRenderer` interface that allows SpawnDev.ILGPU backends (WebGPU, WebGL, and Wasm) to present memory buffers directly to the screen natively.
- **WebGPU**: Zero-copy render directly from the WebGPU storage buffer to the canvas context via a fullscreen quad.
- **WebGL**: Zero-copy render (or GPU-side copy) from the WebGL buffer to the canvas context via a textured quad.
- **Wasm (CPU)**: Optimized `PutImageData` path reusing shared memory arrays to minimize allocation overhead.

## Architecture & Proposed Changes

We will introduce a new namespace: `SpawnDev.ILGPU.Rendering`.

### 1. Unified Interfaces
Create `ICanvasRenderer` and `CanvasRendererFactory`.

#### [NEW] `SpawnDev.ILGPU/Rendering/ICanvasRenderer.cs`
Defines the standard API for presenting an ILGPU `MemoryBuffer2D` to a canvas.
```csharp
public interface ICanvasRenderer : IDisposable
{
    /// <summary>Attaches or re-attaches the renderer to an HTML canvas.</summary>
    void AttachCanvas(HTMLCanvasElement canvas);

    /// <summary>Presents a 2D integer/uint pixel buffer to the canvas.</summary>
    Task PresentAsync(MemoryBuffer2D<uint, Stride2D.DenseX> buffer);
    Task PresentAsync(MemoryBuffer2D<int, Stride2D.DenseX> buffer);
}
```

#### [NEW] `SpawnDev.ILGPU/Rendering/CanvasRendererFactory.cs`
A helper that creates the optimal renderer given an [Accelerator](file:///d:/users/tj/Projects/SpawnDev.ILGPU/SpawnDev.ILGPU/SpawnDev.ILGPU/WebGPU/WebGPUAccelerator.cs#241-242).
```csharp
public static class CanvasRendererFactory
{
    public static ICanvasRenderer Create(Accelerator accelerator)
    {
        // Switch on accelerator type and return the respective WebGPU, WebGL, or CPU renderer.
    }
}
```

### 2. WebGPU Implementation

#### [NEW] `SpawnDev.ILGPU.WebGPU/Rendering/WebGPUCanvasRenderer.cs`
Zero-copy rendering via `GPUCanvasContext`.
- Gets `GPUCanvasContext` ("webgpu") from the canvas.
- Sets up a `GPURenderPipeline` with a hardcoded fullscreen triangle vertex shader.
- The fragment shader takes a specific `var<storage, read> buffer : array<u32>;` binding.
- Reads `buffer[y * canvasWidth + x]` and converts the packed uint32 RGBA to `vec4<f32>`.
- Draws directly to the `GetCurrentTexture()`.

### 3. CPU / Wasm Implementation

#### [NEW] `SpawnDev.ILGPU/Rendering/CPUCanvasRenderer.cs`
For the traditional Wasm CPU multi-threading backend.
- Gets `CanvasRenderingContext2D` ("2d") from the canvas.
- Reuses a pre-allocated `ImageData` object to prevent GC pressure.
- [CopyToHostAsync()](file:///d:/users/tj/Projects/SpawnDev.ILGPU/SpawnDev.ILGPU/SpawnDev.ILGPU/WebGPU/WebGPUBuffer.cs#138-203) from the memory buffer (which may just be a quick wrap if the buffer is CPU backed).
- Calls `context.PutImageData()`.

### 4. WebGL Implementation

#### [NEW] `SpawnDev.ILGPU.WebGL/Rendering/WebGLCanvasRenderer.cs`
- Gets `WebGL2RenderingContext` ("webgl2").
- Sets up a standard GLSL quad shader.
- Since WebGL backend [MemoryBuffer](file:///d:/users/tj/Projects/SpawnDev.ILGPU/SpawnDev.ILGPU/SpawnDev.ILGPU/WebGPU/Backend/WebGPUMemoryBuffer.cs#18-27) is backed by a WebGLBuffer, we can bind it to `PIXEL_UNPACK_BUFFER` and call `texImage2D` to populate a texture instantly on the GPU, then render the quad.

### 5. Demos Update

#### [MODIFY] `SpawnDev.ILGPU.Demo/Pages/*.razor`
- Remove all `PutImageData` and raw WebGPU boilerplate from all graphical demos involving canvas rendering (e.g., [Mandelbrot.razor](file:///d:/users/tj/Projects/SpawnDev.ILGPU/SpawnDev.ILGPU/SpawnDev.ILGPU.Demo/Pages/Mandelbrot.razor), [Boids3D.razor](file:///d:/users/tj/Projects/SpawnDev.ILGPU/SpawnDev.ILGPU/SpawnDev.ILGPU.Demo/Pages/Boids3D.razor), [Compute3D.razor](file:///d:/users/tj/Projects/SpawnDev.ILGPU/SpawnDev.ILGPU/SpawnDev.ILGPU.Demo/Pages/Compute3D.razor), [FractalExplorer.razor](file:///d:/users/tj/Projects/SpawnDev.ILGPU/SpawnDev.ILGPU/SpawnDev.ILGPU.Demo/Pages/FractalExplorer.razor), [GameOfLife.razor](file:///d:/users/tj/Projects/SpawnDev.ILGPU/SpawnDev.ILGPU/SpawnDev.ILGPU.Demo/Pages/GameOfLife.razor)).
- Use `CanvasRendererFactory.Create(_accelerator)` upon GPU initialization.
- In each demo's render loop, replace readback logic with `await _renderer.PresentAsync(_buffer)`.

## Verification Plan

### Automated/Compilation Tests
- `dotnet build` the entire solution to ensure the new classes and factory integrate correctly with all backends.

### Manual Verification
1. Run `SpawnDev.ILGPU.Demo`.
2. Open each graphical demo page (`Mandelbrot Explorer`, `Boids3D`, `GameOfLife`, etc.).
3. Switch between **WebGPU** mode and **CPU Wasm** mode (when supported by the demo). All should render correctly.
4. Verify the "GPU Time" vs "Total Time" significantly improves compared to the old CPU readback method.
