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
    /// </summary>
    public record WebGPUBackendOptions
    {
        /// <summary>
        /// Controls how 64-bit float (double) values are emulated.
        /// GPU hardware lacks native f64 support, so doubles are emulated using f32 pairs.
        /// Default: <see cref="F64EmulationMode.Dekker"/> (fast, good precision).
        /// </summary>
        public F64EmulationMode F64Emulation { get; init; } = F64EmulationMode.Dekker;

        /// <summary>
        /// When true, disables native subgroup operations (subgroupShuffle, subgroupBroadcastFirst, etc.)
        /// even if the GPU device supports them, and falls back to workgroup shared-memory emulation instead.
        /// Useful for verifying kernel correctness on both the native-subgroup and emulation paths.
        /// Default: false (use native subgroups when available).
        /// </summary>
        public bool ForceDisableSubgroups { get; init; } = false;

        /// <summary>
        /// Default options.
        /// </summary>
        public static WebGPUBackendOptions Default { get; } = new();
    }
}
