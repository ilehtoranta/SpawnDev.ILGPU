// ---------------------------------------------------------------------------------------
//                               SpawnDev.ILGPU.WebGPU
//                 WebGPU Compute Library for Blazor WebAssembly
//
// File: WebGPUBuffer.cs
// ---------------------------------------------------------------------------------------

using SpawnDev.BlazorJS.JSObjects;
using SpawnDev.ILGPU.WebGPU.Backend;
using System.Runtime.InteropServices;

namespace SpawnDev.ILGPU.WebGPU
{
    /// <summary>
    /// Represents a typed GPU memory buffer in WebGPU.
    /// </summary>
    public sealed class WebGPUBuffer<T> : IDisposable where T : unmanaged
    {
        #region Instance

        private GPUBuffer? _buffer;
        private bool _disposed;

        /// <summary>
        /// Constructs a new WebGPU buffer.
        /// </summary>
        internal WebGPUBuffer(WebGPUNativeAccelerator accelerator, long length)
        {
            Accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
            Length = length;
            ElementSize = Marshal.SizeOf<T>();
            LengthInBytes = length * ElementSize;

            var device = accelerator.NativeDevice;
            if (device == null)
                throw new InvalidOperationException("GPU device not initialized");

            // Create GPU buffer
            var descriptor = new GPUBufferDescriptor
            {
                Size = (ulong)LengthInBytes,
                Usage = GPUBufferUsage.Storage | GPUBufferUsage.CopySrc | GPUBufferUsage.CopyDst,
                MappedAtCreation = false
            };

            _buffer = device.CreateBuffer(descriptor);
        }

        #endregion

        #region Properties

        /// <summary>
        /// Returns the parent accelerator.
        /// </summary>
        public WebGPUNativeAccelerator Accelerator { get; }

        /// <summary>
        /// Returns the native GPU buffer.
        /// </summary>
        public GPUBuffer? NativeBuffer => _buffer;

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
        /// Copies data from a host array to the GPU buffer.
        /// </summary>
        public void CopyFromHost(T[] sourceArray, long targetOffset = 0)
        {
            if (_buffer == null)
                throw new ObjectDisposedException(nameof(WebGPUBuffer<T>));
            if (sourceArray.Length > Length - targetOffset)
                throw new ArgumentException("Source array is too large for the buffer");

            var queue = Accelerator.Queue;
            if (queue == null)
                throw new InvalidOperationException("GPU queue not available");

            // Use Uint8Array.Write<T> for direct transfer without Marshal.Copy overhead
            using var uint8Array = new Uint8Array(sourceArray.Length * ElementSize);
            uint8Array.Write(sourceArray);
            queue.WriteBuffer(_buffer, (long)(targetOffset * ElementSize), uint8Array);
        }

        /// <summary>
        /// Copies data from the GPU buffer to a host array asynchronously.
        /// Allocates and returns a new T[] array.
        /// For hot-path rendering loops, prefer the overload that accepts a destination array
        /// to avoid per-call allocations.
        /// </summary>
        public async Task<T[]> CopyToHostAsync(long sourceOffset = 0, long? length = null)
        {
            var copyLength = length ?? Length - sourceOffset;
            var result = new T[copyLength];
            await CopyToHostAsync(result, sourceOffset, copyLength);
            return result;
        }

        // Cached staging buffer for zero-allocation readback
        private GPUBuffer? _cachedStagingBuffer;
        private long _cachedStagingSize;

        /// <summary>
        /// Copies GPU data into a caller-provided array, reusing a cached staging buffer.
        /// This is the zero-allocation hot path — no GPU buffer or .NET array allocation per call.
        /// The staging buffer is created once and reused for subsequent calls of the same or smaller size.
        /// </summary>
        /// <param name="destination">Pre-allocated array to receive the data. Must have enough space for count elements.</param>
        /// <param name="sourceOffset">Offset in elements from the start of the GPU buffer.</param>
        /// <param name="count">Number of elements to copy. If null, copies as many as will fit in destination.</param>
        /// <returns>Number of elements actually copied.</returns>
        public async Task<long> CopyToHostAsync(T[] destination, long sourceOffset = 0, long? count = null)
        {
            if (_buffer == null)
                throw new ObjectDisposedException(nameof(WebGPUBuffer<T>));

            var copyLength = count ?? Math.Min(destination.Length, Length - sourceOffset);
            if (copyLength <= 0) return 0;

            var copyBytes = copyLength * ElementSize;
            var sourceByteOffset = sourceOffset * ElementSize;

            WebGPUBackend.Log($"[WebGPU] CopyToHostAsync: SourceOffset={sourceOffset}, Length={copyLength} elements");

            var device = Accelerator.NativeDevice;
            if (device == null)
                throw new InvalidOperationException("GPU device not initialized");

            // Ensure cached staging buffer is large enough (created once, reused)
            if (_cachedStagingBuffer == null || _cachedStagingSize < copyBytes)
            {
                _cachedStagingBuffer?.Destroy();
                _cachedStagingBuffer?.Dispose();

                var stagingDescriptor = new GPUBufferDescriptor
                {
                    Size = (ulong)copyBytes,
                    Usage = GPUBufferUsage.CopyDst | GPUBufferUsage.MapRead,
                    MappedAtCreation = false
                };
                _cachedStagingBuffer = device.CreateBuffer(stagingDescriptor);
                _cachedStagingSize = copyBytes;
            }

            // Copy from GPU buffer to cached staging buffer
            using var encoder = device.CreateCommandEncoder();
            encoder.CopyBufferToBuffer(_buffer, (ulong)sourceByteOffset, _cachedStagingBuffer, 0, (ulong)copyBytes);
            using var commandBuffer = encoder.Finish();
            Accelerator.Queue?.Submit(new[] { commandBuffer });

            // Map, read into caller's destination array, unmap
            await _cachedStagingBuffer.MapAsync(GPUMapMode.Read);
            var mappedRange = _cachedStagingBuffer.GetMappedRange();
            if (mappedRange != null)
            {
                using var uint8Array = new Uint8Array(mappedRange);
                uint8Array.Read(0, destination, 0, copyLength);
            }
            _cachedStagingBuffer.Unmap();

            WebGPUBackend.Log($"[WebGPU] CopyToHostAsync: Finished");

            return copyLength;
        }

        /// <summary>
        /// Copies GPU data into a caller-provided array, reusing a cached staging buffer.
        /// This is the zero-allocation hot path — no GPU buffer or .NET array allocation per call.
        /// The staging buffer is created once and reused for subsequent calls of the same or smaller size.
        /// </summary>
        /// <param name="sourceByteOffset">Offset in bytes from the start of the GPU buffer.</param>
        /// <param name="copyBytes">Number of bytes to copy. If null, copies as many as will fit in destination.</param>
        /// <returns>Number of elements actually copied.</returns>
        public async Task<Uint8Array> CopyToHostUint8ArrayAsync(long sourceByteOffset = 0, long? copyBytes = null)
        {
            if (_buffer == null)
                throw new ObjectDisposedException(nameof(Buffer));

            copyBytes ??= Length * ElementSize - sourceByteOffset;
            if (copyBytes <= 0) return new Uint8Array();

            WebGPUBackend.Log($"[WebGPU] CopyToHostUint8ArrayAsync: SourceByteOffset={sourceByteOffset}, CopyBytes={copyBytes} elements");

            var device = Accelerator.NativeDevice;
            if (device == null)
                throw new InvalidOperationException("GPU device not initialized");

            // Ensure cached staging buffer is large enough (created once, reused)
            if (_cachedStagingBuffer == null || _cachedStagingSize < copyBytes)
            {
                _cachedStagingBuffer?.Destroy();
                _cachedStagingBuffer?.Dispose();

                var stagingDescriptor = new GPUBufferDescriptor
                {
                    Size = (ulong)copyBytes,
                    Usage = GPUBufferUsage.CopyDst | GPUBufferUsage.MapRead,
                    MappedAtCreation = false
                };
                _cachedStagingBuffer = device.CreateBuffer(stagingDescriptor);
                _cachedStagingSize = copyBytes.Value;
            }

            // Copy from GPU buffer to cached staging buffer
            using var encoder = device.CreateCommandEncoder();
            encoder.CopyBufferToBuffer(_buffer, (ulong)sourceByteOffset, _cachedStagingBuffer, 0, (ulong)copyBytes);
            using var commandBuffer = encoder.Finish();
            Accelerator.Queue?.Submit(new[] { commandBuffer });

            // Map, read into caller's destination array, unmap
            await _cachedStagingBuffer.MapAsync(GPUMapMode.Read);
            Uint8Array result = default!;
            try
            {
                using var mappedRange = _cachedStagingBuffer.GetMappedRange();
                if (mappedRange != null)
                {
                    result = new Uint8Array(mappedRange.Slice(0));
                }
            }
            finally
            {
                _cachedStagingBuffer.Unmap();
            }

            WebGPUBackend.Log($"[WebGPU] CopyToHostAsync: Finished");

            return result ?? new Uint8Array();
        }

        /// <summary>
        /// Copies GPU data into a caller-provided array of a different element type,
        /// reusing a cached staging buffer. The data is reinterpreted as TDest elements.
        /// This is used by extension methods where the native buffer is byte-typed
        /// but the caller wants to read as a different struct type (e.g., uint, float).
        /// </summary>
        /// <typeparam name="TDest">The destination element type.</typeparam>
        /// <param name="destination">Pre-allocated array to receive the data.</param>
        /// <param name="sourceOffset">Offset in TDest elements from the start of the GPU buffer.</param>
        /// <param name="count">Number of TDest elements to copy.</param>
        /// <param name="destElementSize">Size of TDest in bytes (Marshal.SizeOf&lt;TDest&gt;()).</param>
        /// <returns>Number of TDest elements actually copied.</returns>
        public async Task<long> CopyToHostAsync<TDest>(TDest[] destination, long sourceOffset, long count, int destElementSize) where TDest : struct
        {
            if (_buffer == null)
                throw new ObjectDisposedException(nameof(WebGPUBuffer<T>));

            if (count <= 0) return 0;

            var copyBytes = count * destElementSize;
            var sourceByteOffset = sourceOffset * destElementSize;

            WebGPUBackend.Log($"[WebGPU] CopyToHostAsync<{typeof(TDest).Name}>: SourceOffset={sourceOffset}, Length={count} elements, ByteSize={copyBytes}");

            var device = Accelerator.NativeDevice;
            if (device == null)
                throw new InvalidOperationException("GPU device not initialized");

            // Ensure cached staging buffer is large enough (created once, reused)
            if (_cachedStagingBuffer == null || _cachedStagingSize < copyBytes)
            {
                _cachedStagingBuffer?.Destroy();
                _cachedStagingBuffer?.Dispose();

                var stagingDescriptor = new GPUBufferDescriptor
                {
                    Size = (ulong)copyBytes,
                    Usage = GPUBufferUsage.CopyDst | GPUBufferUsage.MapRead,
                    MappedAtCreation = false
                };
                _cachedStagingBuffer = device.CreateBuffer(stagingDescriptor);
                _cachedStagingSize = copyBytes;
            }

            // Copy from GPU buffer to cached staging buffer
            using var encoder = device.CreateCommandEncoder();
            encoder.CopyBufferToBuffer(_buffer, (ulong)sourceByteOffset, _cachedStagingBuffer, 0, (ulong)copyBytes);
            using var commandBuffer = encoder.Finish();
            Accelerator.Queue?.Submit(new[] { commandBuffer });

            // Map, read as TDest into destination array, unmap
            await _cachedStagingBuffer.MapAsync(GPUMapMode.Read);
            var mappedRange = _cachedStagingBuffer.GetMappedRange();
            if (mappedRange != null)
            {
                using var uint8Array = new Uint8Array(mappedRange);
                uint8Array.Read(0, destination, 0, count);
            }
            _cachedStagingBuffer.Unmap();

            WebGPUBackend.Log($"[WebGPU] CopyToHostAsync<{typeof(TDest).Name}>: Finished");

            return count;
        }

        /// <summary>
        /// Fills the buffer with a value.
        /// </summary>
        public void Fill(T value)
        {
            var data = new T[Length];
            System.Array.Fill(data, value);
            CopyFromHost(data);
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            _cachedStagingBuffer?.Destroy();
            _cachedStagingBuffer?.Dispose();
            _cachedStagingBuffer = null;

            _buffer?.Destroy();
            _buffer?.Dispose();
            _buffer = null;
        }

        #endregion
    }
}
