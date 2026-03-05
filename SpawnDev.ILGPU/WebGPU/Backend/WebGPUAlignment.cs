// ---------------------------------------------------------------------------------------
//                               SpawnDev.ILGPU.WebGPU
//                 WebGPU Compute Library for Blazor WebAssembly
//
// File: WebGPUAlignment.cs
// ---------------------------------------------------------------------------------------

namespace SpawnDev.ILGPU.WebGPU.Backend
{
    /// <summary>
    /// WebGPU alignment utilities.
    /// </summary>
    /// <remarks>
    /// The WebGPU spec requires buffer sizes, writeBuffer byte counts, and CopyBufferToBuffer
    /// sizes to be multiples of 4 bytes. Use these helpers when interfacing with raw WebGPU APIs.
    /// </remarks>
    internal static class WebGPUAlignment
    {
        /// <summary>
        /// Rounds up to the next multiple of 4 bytes, as required by WebGPU for buffer operations.
        /// </summary>
        public static long AlignTo4(long bytes) => (bytes + 3) & ~3L;
    }
}
