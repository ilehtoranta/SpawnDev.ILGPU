// ---------------------------------------------------------------------------------------
//                               SpawnDev.ILGPU
//                    Copyright (c) 2024-2026 SpawnDev / LostBeard
//
// File: WebGPUAlgorithmExtensions.cs
//
// Public extension method for enabling WebGPU algorithm intrinsics.
// Usage: Context.Create().EnableAlgorithms().EnableWebGPUAlgorithms()
// ---------------------------------------------------------------------------------------

using ILGPU;
using ILGPU.IR;
using SpawnDev.ILGPU.WebGPU.Algorithms;

namespace ILGPU
{
    /// <summary>
    /// Extension methods for enabling WebGPU-specific algorithm intrinsics.
    /// </summary>
    public static class WebGPUAlgorithmExtensions
    {
        /// <summary>
        /// Enables WebGPU-specific algorithm intrinsics (scan, reduce, etc.)
        /// required for ILGPU.Algorithms operations like RadixSort to work
        /// on the WebGPU backend.
        /// </summary>
        /// <param name="builder">The context builder.</param>
        /// <returns>The same builder for chaining.</returns>
        /// <remarks>
        /// Call this after <see cref="AlgorithmContext.EnableAlgorithms"/>:
        /// <code>
        /// var context = Context.Create()
        ///     .EnableAlgorithms()
        ///     .EnableWebGPUAlgorithms()
        ///     .CreateContext();
        /// </code>
        /// </remarks>
        public static Context.Builder EnableWebGPUAlgorithms(
            this Context.Builder builder)
        {
            var intrinsicManager = builder.GetIntrinsicManager();
            WebGPUAlgorithmContext.EnableWebGPUAlgorithms(intrinsicManager);
            return builder;
        }
    }
}
