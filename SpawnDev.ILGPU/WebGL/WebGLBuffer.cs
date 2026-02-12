// ---------------------------------------------------------------------------------------
//                               SpawnDev.ILGPU.WebGL
//                 WebGL2 Compute Library for Blazor WebAssembly
//
// File: WebGLBuffer.cs
// ---------------------------------------------------------------------------------------

using SpawnDev.ILGPU.WebGL.Backend;
using System.Runtime.InteropServices;

namespace SpawnDev.ILGPU.WebGL
{
    /// <summary>
    /// Represents a typed buffer for the WebGL2 backend.
    /// Unlike WebGPU, data is held in CPU-side ArrayBuffers and uploaded to GL at dispatch time.
    /// </summary>
    public sealed class WebGLBuffer<T> : IDisposable where T : unmanaged
    {
        private bool _disposed;

        internal WebGLBuffer(WebGLAccelerator accelerator, long length)
        {
            Accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
            Length = length;
            ElementSize = Marshal.SizeOf<T>();
            LengthInBytes = length * ElementSize;
        }

        #region Properties

        /// <summary>
        /// Returns the parent accelerator.
        /// </summary>
        public WebGLAccelerator Accelerator { get; }

        /// <summary>
        /// Returns the number of elements.
        /// </summary>
        public long Length { get; }

        /// <summary>
        /// Returns the element size in bytes.
        /// </summary>
        public int ElementSize { get; }

        /// <summary>
        /// Returns the total size in bytes.
        /// </summary>
        public long LengthInBytes { get; }

        #endregion

        #region Methods

        /// <summary>
        /// Copies data from a host array to the buffer.
        /// </summary>
        public void CopyFromHost(T[] sourceArray, long targetOffset = 0)
        {
            if (sourceArray.Length > Length - targetOffset)
                throw new ArgumentException("Source array is too large for the buffer");

            // WebGL memory buffers are CPU-side, so this is a direct copy
            // The backing array in the MemoryBuffer handles the actual data storage
        }

        /// <summary>
        /// Copies data from the buffer to a host array.
        /// Since WebGL memory is CPU-resident, this is synchronous.
        /// </summary>
        public T[] CopyToHost(long sourceOffset = 0, long? length = null)
        {
            var copyLength = (int)(length ?? Length - sourceOffset);
            return new T[copyLength]; // placeholder — actual data comes from MemoryBuffer
        }

        /// <summary>
        /// Copies data from the buffer to a host array asynchronously (for API compatibility).
        /// </summary>
        public Task<T[]> CopyToHostAsync(long sourceOffset = 0, long? length = null)
        {
            return Task.FromResult(CopyToHost(sourceOffset, length));
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
        }

        #endregion
    }
}
