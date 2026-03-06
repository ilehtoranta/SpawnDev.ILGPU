using global::ILGPU;
using global::ILGPU.Runtime;
using SpawnDev.BlazorJS.JSObjects;
using SpawnDev.ILGPU.Rendering;

namespace SpawnDev.ILGPU.WebGL.Rendering
{
    /// <summary>
    /// WebGL2 canvas renderer. Uploads the pixel buffer directly via texImage2D to a GPU texture
    /// and renders it with a fullscreen triangle — avoiding PutImageData's 2D rasterizer path.
    /// </summary>
    public sealed class WebGLCanvasRenderer : ICanvasRenderer
    {
        private readonly WebGLAccelerator _accelerator;

        private HTMLCanvasElement? _internalCanvas;
        private CanvasRenderingContext2D? _displayCtx;
        private WebGL2RenderingContext? _gl;
        private WebGLProgram? _program;
        private WebGLTexture? _texture;
        private WebGLBuffer? _vbo;
        private bool _disposed;

        // GLSL ES 3.00 fullscreen triangle
        private const string VertexSource = @"#version 300 es
layout(location = 0) in vec2 aPos;
out vec2 vTex;
void main() {
    vTex = aPos * 0.5 + 0.5;
    vTex.y = 1.0 - vTex.y;
    gl_Position = vec4(aPos, 0.0, 1.0);
}";

        private const string FragmentSource = @"#version 300 es
precision mediump float;
in vec2 vTex;
uniform sampler2D uTexture;
out vec4 fragColor;
void main() {
    fragColor = texture(uTexture, vTex);
}";

        private static readonly float[] FullscreenTriangle = { -1f, -1f, 3f, -1f, -1f, 3f };

        public WebGLCanvasRenderer(WebGLAccelerator accelerator)
        {
            _accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
        }

        public void AttachCanvas(HTMLCanvasElement canvas)
        {
            DisposeGlResources();

            // Display canvas always uses 2d context — no context-type conflict when switching backends.
            _displayCtx = canvas.GetContext<CanvasRenderingContext2D>("2d");

            // Internal off-DOM canvas owns the webgl2 context.
            // preserveDrawingBuffer=true is required so DrawImage can read the buffer after gl.drawArrays.
            _internalCanvas = new HTMLCanvasElement();
            _gl = _internalCanvas.GetContext<WebGL2RenderingContext>("webgl2", new WebGLContextAttributes { PreserveDrawingBuffer = true })
                ?? throw new InvalidOperationException("Failed to get WebGL2 rendering context.");

            _program = _gl.CreateProgram(VertexSource, FragmentSource);

            _texture = _gl.CreateTexture();
            _gl.BindTexture(GL.TEXTURE_2D, _texture);
            _gl.TexParameteri(GL.TEXTURE_2D, GL.TEXTURE_MIN_FILTER, GL.NEAREST);
            _gl.TexParameteri(GL.TEXTURE_2D, GL.TEXTURE_MAG_FILTER, GL.NEAREST);
            _gl.TexParameteri(GL.TEXTURE_2D, GL.TEXTURE_WRAP_S, GL.CLAMP_TO_EDGE);
            _gl.TexParameteri(GL.TEXTURE_2D, GL.TEXTURE_WRAP_T, GL.CLAMP_TO_EDGE);

            _vbo = _gl.CreateBuffer();
            _gl.BindBuffer(GL.ARRAY_BUFFER, _vbo);
            using var verts = new Float32Array(FullscreenTriangle);
            _gl.BufferData(GL.ARRAY_BUFFER, verts, GL.STATIC_DRAW);
        }

        public async Task PresentAsync(MemoryBuffer2D<uint, Stride2D.DenseX> buffer)
        {
            int width = (int)buffer.Extent.X, height = (int)buffer.Extent.Y;
            var internalBuf = ((IArrayView)buffer).Buffer;
            if (internalBuf is IBrowserMemoryBuffer browserBuf)
            {
                using var src = await browserBuf.CopyToHostUint8ArrayAsync(0, (long)width * height * 4);
                BlitTexture(src, width, height);
            }
            else
            {
                _accelerator.Synchronize();
                var tmp = new uint[width * height];
                buffer.View.BaseView.CopyToCPU(tmp);
                using var u32 = new Uint32Array(tmp);
                using var src = new Uint8Array(u32.Buffer);
                BlitTexture(src, width, height);
            }
        }

        public async Task PresentAsync(MemoryBuffer2D<int, Stride2D.DenseX> buffer)
        {
            int width = (int)buffer.Extent.X, height = (int)buffer.Extent.Y;
            var internalBuf = ((IArrayView)buffer).Buffer;
            if (internalBuf is IBrowserMemoryBuffer browserBuf)
            {
                using var src = await browserBuf.CopyToHostUint8ArrayAsync(0, (long)width * height * 4);
                BlitTexture(src, width, height);
            }
            else
            {
                _accelerator.Synchronize();
                var tmp = new int[width * height];
                buffer.View.BaseView.CopyToCPU(tmp);
                using var i32 = new Int32Array(tmp);
                using var src = new Uint8Array(i32.Buffer);
                BlitTexture(src, width, height);
            }
        }

        private void BlitTexture(Uint8Array pixels, int width, int height)
        {
            if (_gl == null || _program == null || _texture == null || _vbo == null
                || _internalCanvas == null || _displayCtx == null) return;

            if (_internalCanvas.Width != width || _internalCanvas.Height != height)
            {
                _internalCanvas.Width = width;
                _internalCanvas.Height = height;
            }

            var gl = _gl;
            gl.Viewport(0, 0, width, height);

            gl.ActiveTexture(GL.TEXTURE0);
            gl.BindTexture(GL.TEXTURE_2D, _texture);
            gl.TexImage2D(GL.TEXTURE_2D, 0, GL.RGBA, width, height, 0, GL.RGBA, GL.UNSIGNED_BYTE, pixels);

            gl.UseProgram(_program);
            using var texUniform = gl.GetUniformLocation(_program, "uTexture");
            if (texUniform != null) gl.Uniform1i(texUniform, 0);

            gl.BindBuffer(GL.ARRAY_BUFFER, _vbo);
            gl.EnableVertexAttribArray(0);
            gl.VertexAttribPointer(0, 2, GL.FLOAT, false, 0, 0);
            gl.DrawArrays(GL.TRIANGLES, 0, 3);
            gl.DisableVertexAttribArray(0);

            // Blit internal WebGL canvas to the display canvas via 2d context.
            _displayCtx.DrawImage(_internalCanvas);
        }

        private void DisposeGlResources()
        {
            _program?.Dispose(); _program = null;
            _texture?.Dispose(); _texture = null;
            _vbo?.Dispose(); _vbo = null;
            _gl?.Dispose(); _gl = null;
            _internalCanvas?.Dispose(); _internalCanvas = null;
            _displayCtx?.Dispose(); _displayCtx = null;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            DisposeGlResources();
        }
    }
}
