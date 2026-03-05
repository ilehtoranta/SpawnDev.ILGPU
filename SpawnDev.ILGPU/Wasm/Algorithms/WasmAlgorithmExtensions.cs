// ---------------------------------------------------------------------------------------
//                               SpawnDev.ILGPU
//                    Copyright (c) 2024-2026 SpawnDev / LostBeard
//
// File: WasmAlgorithmExtensions.cs
//
// Public extension method to enable Wasm-specific algorithm intrinsics.
// Usage: Context.Create().EnableAlgorithms().EnableWasmAlgorithms()
// ---------------------------------------------------------------------------------------

using ILGPU;
using ILGPU.IR;
using SpawnDev.ILGPU.Wasm.Algorithms;

namespace ILGPU
{
    /// <summary>
    /// Extension methods for enabling Wasm-specific algorithm intrinsics.
    /// </summary>
    public static class WasmAlgorithmExtensions
    {
        /// <summary>
        /// Enables Wasm-specific algorithm intrinsics (scan, reduce, etc.)
        /// required for ILGPU.Algorithms operations like RadixSort to work
        /// on the Wasm backend.
        /// </summary>
        /// <param name="builder">The context builder.</param>
        /// <returns>The same builder for chaining.</returns>
        /// <remarks>
        /// Call this after <see cref="AlgorithmContext.EnableAlgorithms"/>:
        /// <code>
        /// var context = Context.Create()
        ///     .EnableAlgorithms()
        ///     .EnableWasmAlgorithms()
        ///     .CreateContext();
        /// </code>
        /// </remarks>
        public static Context.Builder EnableWasmAlgorithms(
            this Context.Builder builder)
        {
            var intrinsicManager = builder.GetIntrinsicManager();
            WasmAlgorithmContext.EnableWasmAlgorithms(intrinsicManager);
            return builder;
        }
    }
}
