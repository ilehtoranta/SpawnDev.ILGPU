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
    /// Pass to the context builder to configure backend behavior.
    /// By default, F64 and I64 emulation are enabled for full 64-bit precision parity
    /// with CPU and Workers backends. Disable them for better performance if 32-bit precision is acceptable.
    /// </summary>
    public record WebGLBackendOptions
    {
        /// <summary>
        /// Enables emu_f64 (double) emulation using two f32 values (double-float technique).
        /// When enabled (default), emu_f64 operations use vec2 with software emulation.
        /// When disabled, emu_f64 is promoted to float (loses precision but improves performance).
        /// </summary>
        public bool EnableF64Emulation { get; init; } = true;

        /// <summary>
        /// Enables emu_i64 (long) emulation using two u32 values (double-word technique).
        /// When enabled (default), emu_i64 operations use uvec2 with software emulation.
        /// When disabled, emu_i64 is promoted to int (loses range but improves performance).
        /// </summary>
        public bool EnableI64Emulation { get; init; } = true;

        /// <summary>
        /// Default options with F64 and I64 emulation enabled for full precision.
        /// </summary>
        public static WebGLBackendOptions Default { get; } = new();
    }
}
