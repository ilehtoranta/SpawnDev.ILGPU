using ILGPU.Runtime;

namespace SpawnDev.ILGPU.WebGL
{
    /// <summary>
    /// Represents the capabilities of the WebGL2 backend.
    /// </summary>
    public sealed class WebGLCapabilityContext : CapabilityContext
    {
        /// <summary>
        /// Creates a new WebGL capability context.
        /// </summary>
        public WebGLCapabilityContext()
        {
            // WebGL2 / GLSL ES 3.0 has no native double support
            Float16 = false;
            Float64 = false;
            Int64 = false;
        }

        /// <summary>
        /// Supports Float64 (double) data type via emulation.
        /// </summary>
        public bool Float64 { get; internal set; }

        /// <summary>
        /// Supports Int64 (long) data type via emulation.
        /// </summary>
        public bool Int64 { get; internal set; }
    }
}
