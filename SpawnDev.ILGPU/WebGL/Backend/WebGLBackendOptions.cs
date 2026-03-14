// ---------------------------------------------------------------------------------------
//                                 SpawnDev.ILGPU.WebGL
//                        Copyright (c) 2024 SpawnDev Project
//
// File: WebGLBackendOptions.cs
//
// Configuration options for the WebGL2 backend.
// ---------------------------------------------------------------------------------------

namespace SpawnDev.ILGPU.WebGL.Backend
{
    /// <summary>
    /// Configuration options for the WebGL2 backend.
    /// </summary>
    public record WebGLBackendOptions
    {
        /// <summary>
        /// Controls how 64-bit float (double) values are emulated.
        /// GPU hardware lacks native f64 support, so doubles are emulated using f32 pairs.
        /// Default: <see cref="F64EmulationMode.Dekker"/> (fast, good precision).
        /// </summary>
        public F64EmulationMode F64Emulation { get; init; } = F64EmulationMode.Dekker;

        /// <summary>
        /// Default options.
        /// </summary>
        public static WebGLBackendOptions Default { get; } = new();
    }
}
