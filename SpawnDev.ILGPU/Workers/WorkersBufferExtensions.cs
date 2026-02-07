// ---------------------------------------------------------------------------------------
//                               SpawnDev.ILGPU.Workers
//                 Web Worker Compute Library for Blazor WebAssembly
//
// File: WorkersBufferExtensions.cs
//
// Extension methods for asynchronous data readback from Workers-backed buffers.
// ---------------------------------------------------------------------------------------

using global::ILGPU;
using global::ILGPU.Runtime;
using System.Runtime.InteropServices;

namespace SpawnDev.ILGPU.Workers
{
    /// <summary>
    /// Extension methods for ILGPU buffer types to support readback from Workers buffers.
    /// Workers buffers are backed by SharedArrayBuffer/ArrayBuffer and can be read synchronously,
    /// but async API is provided for consistency with the WebGPU backend.
    /// </summary>
    public static class WorkersBufferExtensions
    {
        /// <summary>
        /// Copies data from a Workers-backed buffer back to the host.
        /// Since Workers buffers are in CPU memory (SharedArrayBuffer/ArrayBuffer),
        /// this is effectively a synchronous copy wrapped in a Task for API consistency.
        /// </summary>
        /// <typeparam name="T">The element type of the buffer.</typeparam>
        /// <param name="buffer">The MemoryBuffer1D to read from.</param>
        /// <returns>An array containing the buffer data.</returns>
        public static Task<T[]> CopyToHostAsync<T>(this MemoryBuffer1D<T, Stride1D.Dense> buffer) where T : unmanaged
        {
            var workersBuffer = GetWorkersBuffer((MemoryBuffer)buffer);
            return Task.FromResult(ReadData<T>(workersBuffer, buffer.Length));
        }

        /// <summary>
        /// Copies data from a Workers-backed buffer back to the host.
        /// This overload works with any MemoryBuffer.
        /// </summary>
        /// <typeparam name="T">The element type of the buffer.</typeparam>
        /// <param name="buffer">The MemoryBuffer to read from.</param>
        /// <returns>An array containing the buffer data.</returns>
        public static Task<T[]> CopyToHostAsync<T>(this MemoryBuffer buffer) where T : unmanaged
        {
            var workersBuffer = GetWorkersBuffer(buffer);
            return Task.FromResult(ReadData<T>(workersBuffer, buffer.Length));
        }

        /// <summary>
        /// Reads typed data from a WorkersMemoryBuffer via its Uint8Array view.
        /// </summary>
        private static T[] ReadData<T>(WorkersMemoryBuffer workersBuffer, long length) where T : unmanaged
        {
            var uint8View = workersBuffer.Uint8View
                ?? throw new ObjectDisposedException(nameof(WorkersMemoryBuffer));

            // Read the raw bytes from the Uint8Array
            var byteData = uint8View.ReadBytes();
            var result = new T[length];
            MemoryMarshal.Cast<byte, T>(byteData).CopyTo(new Span<T>(result));
            return result;
        }

        /// <summary>
        /// Gets the underlying WorkersMemoryBuffer from an ILGPU MemoryBuffer.
        /// </summary>
        private static WorkersMemoryBuffer GetWorkersBuffer(MemoryBuffer buffer)
        {
            var iView = (IArrayView)buffer;
            if (iView.Buffer is WorkersMemoryBuffer workersBuffer)
                return workersBuffer;
            throw new InvalidOperationException("CopyToHostAsync is only supported for Workers-backed or WebGPU-backed buffers.");
        }
    }
}
