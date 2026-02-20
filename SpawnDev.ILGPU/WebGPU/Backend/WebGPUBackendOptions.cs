// ---------------------------------------------------------------------------------------
//                                 SpawnDev.ILGPU.WebGPU
//                        Copyright (c) 2024 SpawnDev Project
//
// File: WebGPUBackendOptions.cs
//
// Configuration options for the WebGPU backend.
// ---------------------------------------------------------------------------------------

namespace SpawnDev.ILGPU.WebGPU.Backend
{
    /// <summary>
    /// Configuration options for the WebGPU backend.
    /// Pass to the context builder to configure backend behavior.
    /// By default, F64 and I64 emulation are enabled for full 64-bit precision parity
    /// with CPU and Workers backends. Disable them for better performance if 32-bit precision is acceptable.
    /// </summary>
    public record WebGPUBackendOptions
    {
        /// <summary>
        /// Enables emu_f64 (double) emulation using two f32 values (double-float technique).
        /// When enabled (default), emu_f64 operations use vec2&lt;f32&gt; with software emulation.
        /// When disabled, emu_f64 is promoted to f32 (loses precision but improves performance).
        /// </summary>
        public bool EnableF64Emulation { get; init; } = true;

        /// <summary>
        /// When true, and EnableF64Emulation is true, uses the Ozaki Scheme (vec4&lt;f32&gt;) for 64-bit emulation
        /// to achieve strict IEEE 754 precision compliance at the cost of performance.
        /// Defaults to false (uses the faster Dekker double-float vec2&lt;f32&gt; technique).
        /// </summary>
        public bool UseOzakiF64Emulation { get; init; } = false;

        /// <summary>
        /// Enables emu_i64 (long) emulation using two u32 values (double-word technique).
        /// When enabled (default), emu_i64 operations use vec2&lt;u32&gt; with software emulation.
        /// When disabled, emu_i64 is promoted to i32 (loses range but improves performance).
        /// </summary>
        public bool EnableI64Emulation { get; init; } = true;

        /// <summary>
        /// When true, disables native subgroup operations (subgroupShuffle, subgroupBroadcastFirst, etc.)
        /// even if the GPU device supports them, and falls back to workgroup shared-memory emulation instead.
        /// <para>
        /// This is useful for developers who want to verify their ILGPU kernel code works correctly
        /// on both the native-subgroup path and the shared-memory emulation path, without needing
        /// hardware that lacks subgroup support.
        /// </para>
        /// Default: false (use native subgroups when available).
        /// </summary>
        public bool ForceDisableSubgroups { get; init; } = false;

        /// <summary>
        /// Default options with F64 and I64 emulation enabled for full precision.
        /// </summary>
        public static WebGPUBackendOptions Default { get; } = new();
    }
}
