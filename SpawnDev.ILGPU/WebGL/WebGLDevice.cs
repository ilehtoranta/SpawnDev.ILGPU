// ---------------------------------------------------------------------------------------
//                               SpawnDev.ILGPU.WebGL
//                 WebGL2 Compute Library for Blazor WebAssembly
//
// File: WebGLDevice.cs
// ---------------------------------------------------------------------------------------

using SpawnDev.BlazorJS;
using SpawnDev.BlazorJS.JSObjects;
using GL = SpawnDev.BlazorJS.JSObjects.GL;
using System.Collections.Immutable;

namespace SpawnDev.ILGPU.WebGL
{
    /// <summary>
    /// Represents a WebGL2 device available in the browser.
    /// Creates an OffscreenCanvas and obtains a WebGL2RenderingContext for GPGPU
    /// via Transform Feedback.
    /// </summary>
    public sealed class WebGLDevice : IDisposable
    {
        #region Static

        /// <summary>
        /// Checks if WebGL2 is supported in the current browser.
        /// </summary>
        public static bool IsSupported
        {
            get
            {
                try
                {
                    using var canvas = new OffscreenCanvas(1, 1);
                    using var gl = canvas.GetWebGL2Context();
                    return gl != null;
                }
                catch
                {
                    return false;
                }
            }
        }

        /// <summary>
        /// Asynchronously detects all available WebGL2 devices.
        /// WebGL2 typically exposes a single device per browser context.
        /// </summary>
        public static Task<ImmutableArray<WebGLDevice>> GetDevicesAsync()
        {
            var devices = ImmutableArray.CreateBuilder<WebGLDevice>();

            if (!IsSupported)
                return Task.FromResult(devices.ToImmutable());

            try
            {
                var device = new WebGLDevice(0);
                devices.Add(device);
            }
            catch
            {
                // WebGL2 not available
            }

            return Task.FromResult(devices.ToImmutable());
        }

        /// <summary>
        /// Gets the default WebGL2 device if available.
        /// </summary>
        public static async Task<WebGLDevice?> GetDefaultDeviceAsync()
        {
            var devices = await GetDevicesAsync();
            return devices.Length > 0 ? devices[0] : null;
        }

        #endregion

        #region Instance

        private OffscreenCanvas? _canvas;
        private WebGL2RenderingContext? _gl;
        private readonly int _deviceIndex;
        private bool _disposed;

        internal WebGLDevice(int deviceIndex)
        {
            _deviceIndex = deviceIndex;

            // Create a small offscreen canvas for the WebGL2 context
            _canvas = new OffscreenCanvas(1, 1);
            _gl = _canvas.GetWebGL2Context();

            if (_gl == null)
                throw new InvalidOperationException("WebGL2 is not supported in this browser.");

            // Probe device capabilities
            Name = GetRendererString() ?? "WebGL2 Device";
            Vendor = GetVendorString() ?? "Unknown";

            // Get limits via MAX parameters
            MaxTextureSize = _gl.GetParameter<int>(GL.MAX_TEXTURE_SIZE);
            MaxUniformBlockSize = _gl.GetParameter<int>(GL.MAX_UNIFORM_BLOCK_SIZE);
            MaxTransformFeedbackSeparateComponents = _gl.GetParameter<int>(GL.MAX_TRANSFORM_FEEDBACK_SEPARATE_COMPONENTS);
            MaxTransformFeedbackInterleavedComponents = _gl.GetParameter<int>(GL.MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS);

            // Estimate max vertex count for GPGPU dispatch
            // WebGL2 guarantees at least 2^24 − 1 vertices
            MaxVertexCount = 16777215; // 2^24 - 1
        }

        private string? GetRendererString()
        {
            try
            {
                // Try WEBGL_debug_renderer_info for unmasked renderer
                var ext = _gl!.GetExtension("WEBGL_debug_renderer_info");
                if (ext != null)
                {
                    return _gl.GetParameter<string>(GL.UNMASKED_RENDERER_WEBGL);
                }
                return _gl.GetParameter<string>(GL.RENDERER);
            }
            catch
            {
                return null;
            }
        }

        private string? GetVendorString()
        {
            try
            {
                var ext = _gl!.GetExtension("WEBGL_debug_renderer_info");
                if (ext != null)
                {
                    return _gl.GetParameter<string>(GL.UNMASKED_VENDOR_WEBGL);
                }
                return _gl.GetParameter<string>(GL.VENDOR);
            }
            catch
            {
                return null;
            }
        }

        #endregion

        #region Properties

        /// <summary>
        /// Gets a new WebGL2 context (creates a fresh OffscreenCanvas).
        /// Each accelerator should call this to get its own context.
        /// </summary>
        public (OffscreenCanvas canvas, WebGL2RenderingContext gl) CreateContext()
        {
            var canvas = new OffscreenCanvas(1, 1);
            var gl = canvas.GetWebGL2Context();
            if (gl == null)
            {
                canvas.Dispose();
                throw new InvalidOperationException("Failed to create WebGL2 context.");
            }
            return (canvas, gl);
        }

        /// <summary>
        /// Returns the device name (GPU renderer string).
        /// </summary>
        public string Name { get; }

        /// <summary>
        /// Returns the GPU vendor string.
        /// </summary>
        public string Vendor { get; }

        /// <summary>
        /// Gets the maximum texture size (used for TBO data width).
        /// </summary>
        public int MaxTextureSize { get; }

        /// <summary>
        /// Gets the maximum uniform block size in bytes.
        /// </summary>
        public int MaxUniformBlockSize { get; }

        /// <summary>
        /// Gets the max transform feedback separate components.
        /// </summary>
        public int MaxTransformFeedbackSeparateComponents { get; }

        /// <summary>
        /// Gets the max transform feedback interleaved components.
        /// </summary>
        public int MaxTransformFeedbackInterleavedComponents { get; }

        /// <summary>
        /// Gets the maximum vertex count for a single drawArrays call.
        /// </summary>
        public int MaxVertexCount { get; }

        #endregion

        #region Methods

        /// <summary>
        /// Prints device information to the console.
        /// </summary>
        public void PrintInfo(TextWriter writer)
        {
            writer.WriteLine($"WebGL2 Device: {Name}");
            writer.WriteLine($"  Vendor:             {Vendor}");
            writer.WriteLine($"  Max Texture Size:   {MaxTextureSize}");
            writer.WriteLine($"  Max UBO Size:       {MaxUniformBlockSize} bytes");
            writer.WriteLine($"  Max TF Components:  {MaxTransformFeedbackInterleavedComponents} (interleaved)");
            writer.WriteLine($"  Max Vertex Count:   {MaxVertexCount}");
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _gl?.Dispose();
            _gl = null;
            _canvas?.Dispose();
            _canvas = null;
        }

        #endregion
    }
}
