# Canvas Rendering

SpawnDev.ILGPU ships a unified canvas rendering API that presents an ILGPU pixel buffer to an HTML `<canvas>` element with the lowest possible overhead on every backend.

## Overview

```
ICanvasRenderer
├── WebGPUCanvasRenderer   — zero-copy fullscreen-triangle render pass, no CPU readback
├── WebGLCanvasRenderer    — ImageBitmap blit from the GL worker, draw inside callback
└── CPUCanvasRenderer      — reused ImageData object, fast Uint8Array copy (Wasm / desktop CPU)
```

`CanvasRendererFactory.Create(accelerator)` returns the right implementation automatically.

## Quick Start

```csharp
using SpawnDev.ILGPU.Rendering;
using SpawnDev.BlazorJS.JSObjects;

// Create the best renderer for the active accelerator (call once)
ICanvasRenderer _renderer = CanvasRendererFactory.Create(accelerator);

// Attach to the canvas element (call once, or again when the canvas changes)
using var canvas = new HTMLCanvasElement(_canvasRef);
_renderer.AttachCanvas(canvas);

// Each frame: run kernel → present
_kernel(_outputBuffer.IntExtent, _outputBuffer.View /*, ...args */);
await accelerator.SynchronizeAsync();
await _renderer.PresentAsync(_outputBuffer);
```

`PresentAsync` accepts both `MemoryBuffer2D<uint, Stride2D.DenseX>` and `MemoryBuffer2D<int, Stride2D.DenseX>`. Pixels are packed RGBA little-endian: R in bits 0–7, G 8–15, B 16–23, A 24–31.

---

## How Each Implementation Works

### WebGPU — `WebGPUCanvasRenderer`

No CPU readback at all. On every `PresentAsync` call:

1. `FlushPendingCommands()` ensures all queued kernel dispatches have been submitted.
2. A cached **fullscreen-triangle render pipeline** reads the pixel buffer directly from a `read-only-storage` binding.
3. A `GPURenderPass` rasterises a 3-vertex triangle that covers the entire viewport, with the fragment shader unpacking each `uint32` pixel into RGBA.
4. The result is blitted from an off-DOM internal canvas to the user-visible canvas via `CanvasRenderingContext2D.drawImage`.

The render pipeline and bind-group layout are built once in `AttachCanvas` and reused every frame. The uniform buffer (width/height) is only re-uploaded when the resolution changes.

```
kernel output buffer (GPUBuffer)
        │  storage read
        ▼
  fullscreen triangle renderpass   ← no CPU round-trip
        │
  internal WebGPU canvas
        │  drawImage (zero-copy GPU blit)
        ▼
  display canvas (2d context)
```

### WebGL — `WebGLCanvasRenderer`

The WebGL backend runs in a dedicated Web Worker. Blitting to a visible canvas therefore requires getting an `ImageBitmap` across the worker boundary. The implementation avoids a race where the browser could clear the canvas between the blit and the draw:

1. `BlitAndDrawAsync` posts a `blit` message to the GL worker.
2. The worker renders the texture to an offscreen framebuffer and calls `transferToImageBitmap()`.
3. The `ImageBitmap` is transferred back to the main thread.
4. **Synchronously inside the message-handler callback** — before any JS event-loop turn can run — `ctx.drawImage(bitmap)` paints the bitmap onto the canvas.
5. Only after the draw does `BlitAndDrawAsync` resolve its `Task`.

The synchronous draw is the critical detail. Without it, Blazor's render cycle can overwrite the canvas between frames.

```
kernel output buffer (WebGL texture in worker)
        │  texelFetch + offscreen FBO
        ▼
  worker: transferToImageBitmap()
        │  postMessage (structured clone, zero-copy)
        ▼
  main thread callback: ctx.drawImage(bitmap)   ← synchronous, in the handler
        │
  display canvas (2d context)
```

### Wasm — `CPUCanvasRenderer`

Used for any accelerator that is neither WebGPU nor WebGL (Wasm in the browser; CPU on desktop). It reuses a single `ImageData` object to avoid GC churn:

1. If the buffer is browser-backed (`IBrowserMemoryBuffer`) — true for Wasm buffers — it calls `CopyToHostUint8ArrayAsync` for a fast JS-side copy with no managed allocation.
2. Otherwise it falls back to synchronous `CopyToCPU` into a pooled `uint[]` array.
3. `ctx.putImageData` writes the `ImageData` to the canvas.

---

## Pixel Format

Kernels should pack pixels as `uint32` little-endian RGBA:

```csharp
static void PixelKernel(Index2D idx, ArrayView2D<uint, Stride2D.DenseX> output)
{
    byte r = /* red   0–255 */;
    byte g = /* green 0–255 */;
    byte b = /* blue  0–255 */;
    byte a = 255;

    // Little-endian: R in byte 0, A in byte 3
    output[idx] = (uint)((a << 24) | (b << 16) | (g << 8) | r);
}
```

The same packing works identically on all three renderer implementations.

---

## Complete Blazor Page Example

```razor
@page "/mypage"
@implements IAsyncDisposable
@inject IJSRuntime JS

<canvas @ref="_canvasRef" width="800" height="600" />

@code {
    private ElementReference _canvasRef;
    private Context? _context;
    private Accelerator? _accelerator;
    private MemoryBuffer2D<uint, Stride2D.DenseX>? _output;
    private Action<Index2D, ArrayView2D<uint, Stride2D.DenseX>>? _kernel;
    private ICanvasRenderer? _renderer;
    private bool _running;

    protected override async Task OnAfterRenderAsync(bool firstRender)
    {
        if (!firstRender) return;

        _context    = await Context.CreateAsync(b => b.AllAcceleratorsAsync());
        _accelerator = await _context.CreatePreferredAcceleratorAsync();
        _output     = _accelerator.Allocate2DDenseX<uint>(new LongIndex2D(800, 600));
        _kernel     = _accelerator.LoadAutoGroupedStreamKernel<
                          Index2D, ArrayView2D<uint, Stride2D.DenseX>>(PixelKernel);

        _renderer = CanvasRendererFactory.Create(_accelerator);
        using var canvas = new HTMLCanvasElement(_canvasRef);
        _renderer.AttachCanvas(canvas);

        _running = true;
        _ = RenderLoop();
    }

    private async Task RenderLoop()
    {
        while (_running)
        {
            _kernel!(_output!.IntExtent, _output.View);
            await _accelerator!.SynchronizeAsync();
            await _renderer!.PresentAsync(_output);
            await Task.Yield();  // yield to keep the browser responsive
        }
    }

    static void PixelKernel(Index2D idx, ArrayView2D<uint, Stride2D.DenseX> output)
    {
        byte r = (byte)(255 * idx.X / 800);
        byte g = (byte)(255 * idx.Y / 600);
        output[idx] = (uint)(0xFF000000u | ((uint)128 << 16) | ((uint)g << 8) | r);
    }

    public async ValueTask DisposeAsync()
    {
        _running = false;
        _renderer?.Dispose();
        _output?.Dispose();
        _accelerator?.Dispose();
        _context?.Dispose();
    }
}
```

---

## Performance Characteristics

| | WebGPU | WebGL | Wasm (+ desktop CPU) |
|---|---|---|---|
| **CPU readback** | ❌ None | ❌ None (ImageBitmap) | ✅ Required |
| **Extra allocations per frame** | Bind group only | None | None (cached ImageData) |
| **GPU stall** | None | None | Sync on CPU path |
| **Main-thread work** | `drawImage` | `drawImage` | `putImageData` |

The WebGPU and WebGL paths avoid all CPU ↔ GPU data transfers during rendering. The pixel data stays GPU-resident from kernel output to the canvas.

---

## Lifecycle

```csharp
// Create once alongside the accelerator
ICanvasRenderer renderer = CanvasRendererFactory.Create(accelerator);

// Attach whenever the canvas element is available/changes
renderer.AttachCanvas(canvas);

// Call every frame
await renderer.PresentAsync(buffer);

// Dispose alongside the accelerator
renderer.Dispose();
```

`AttachCanvas` can be called multiple times (e.g., when the Blazor component re-renders and the `ElementReference` changes). The previous context is disposed and replaced.

---

## Advanced: Manual Approach

If you need direct control over the blit pipeline — or need to support a buffer layout that does not fit `MemoryBuffer2D<uint>` — you can replicate what the renderers do:

**WebGPU:**

```csharp
// Flush batched dispatches first
webGpuAccelerator.FlushPendingCommands();

// Then execute your own render pass using NativeAccelerator.NativeDevice
var gpuBuffer = ((WebGPUMemoryBuffer)rawBuffer).NativeBuffer.NativeBuffer;
```

**WebGL:**

```csharp
// BlitAndDrawAsync posts to the worker and calls your callback synchronously
// with the ImageBitmap before resolving, keeping the draw in the same event-loop turn.
await webGlAccelerator.BlitAndDrawAsync(memBuf, width, height, bitmap =>
{
    ctx.DrawImage(bitmap);
    // Additional compositing can go here
});
```
