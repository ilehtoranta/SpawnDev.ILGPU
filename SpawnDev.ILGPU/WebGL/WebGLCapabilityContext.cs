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
        /// WebGL2 / GLSL ES 3.0 has no native f16/f64/i64 support. Float16 is emulated
        /// via inline IEEE 754 bit conversion at buffer load/store boundaries (see
        /// <c>GLSLEmulationLibrary.F16Functions</c>). Float64 and Int64 emulation is
        /// configured by the backend itself.
        /// </summary>
        public WebGLCapabilityContext()
        {
            // Float16 is available via f32 emulation (no hardware f16 on WebGL).
            Float16 = true;
            Float16Native = false;
            Float64 = false;
            Int64 = false;
        }

        /// <summary>
        /// Always false on WebGL. WebGL2 / GLSL ES 3.0 has no hardware Float16 support;
        /// the capability is satisfied via f32 arithmetic plus inline IEEE 754 bit
        /// conversion at buffer load/store boundaries. Exposed for parity with
        /// <c>WebGPUCapabilityContext.Float16Native</c> so consumers can detect the
        /// underlying path without backend-specific branching.
        /// </summary>
        public bool Float16Native { get; internal set; }

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
