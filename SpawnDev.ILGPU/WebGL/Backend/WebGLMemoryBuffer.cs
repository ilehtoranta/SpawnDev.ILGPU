using global::ILGPU;
using global::ILGPU.Runtime;
using SpawnDev.BlazorJS.JSObjects;
using System.Runtime.InteropServices;

namespace SpawnDev.ILGPU.WebGL.Backend
{
    /// <summary>
    /// ILGPU MemoryBuffer implementation backed by a JavaScript ArrayBuffer.
    /// WebGL2 does not have persistent GPU buffers like WebGPU — data lives in
    /// CPU-side ArrayBuffers and is uploaded to GL buffers at dispatch time.
    /// </summary>
    public class WebGLMemoryBuffer : MemoryBuffer, IBrowserMemoryBuffer
    {
        private Uint8Array? _backingArray;
        private bool _disposed;

        public WebGLMemoryBuffer(Accelerator accelerator, long length, int elementSize)
            : base(accelerator, length, elementSize)
        {
            _backingArray = new Uint8Array((int)LengthInBytes);
        }

        /// <summary>
        /// Gets the backing Uint8Array that holds the data.
        /// </summary>
        public Uint8Array? BackingArray => _backingArray;

        /// <summary>
        /// Gets the underlying ArrayBuffer backing this memory buffer.
        /// Used for zero-copy transfer to/from the GL worker.
        /// </summary>
        public ArrayBuffer? UnderlyingBuffer => _backingArray?.Buffer;

        /// <summary>
        /// Replaces the underlying ArrayBuffer with a new one (used after transfer back from Worker).
        /// Enables zero-copy ArrayBuffer transfers without requiring SharedArrayBuffer/COI.
        /// </summary>
        internal void ReplaceArrayBuffer(ArrayBuffer newBuffer)
        {
            _backingArray?.Dispose();
            _backingArray = new Uint8Array(newBuffer);
        }

        public Task<Uint8Array> CopyToHostUint8ArrayAsync(long sourceByteOffset = 0, long? copyBytes = null)
        {
            if (BackingArray == null) return Task.FromResult(new Uint8Array());
            return copyBytes == null ? Task.FromResult(BackingArray.SubArray(sourceByteOffset)) : Task.FromResult(BackingArray.SubArray(sourceByteOffset, copyBytes.Value + sourceByteOffset));
        }

        protected override void CopyFrom(
            AcceleratorStream stream,
            in ArrayView<byte> source,
            in ArrayView<byte> destination)
        {
            if (source.GetAcceleratorType() == AcceleratorType.CPU)
            {
                var length = (int)source.Length;
                var sourceContiguous = (IContiguousArrayView)source;
                var sourceBuffer = sourceContiguous.Buffer;
                var srcPtr = sourceBuffer.NativePtr + (int)sourceContiguous.Index;

                var byteArray = new byte[length];
                Marshal.Copy(srcPtr, byteArray, 0, length);

                var destContiguous = (IContiguousArrayView)destination;
                _backingArray!.Write(byteArray, (int)destContiguous.Index);
            }
            else
            {
                throw new NotSupportedException("Peer-to-peer copies not supported in WebGL backend.");
            }
        }

        protected override void CopyTo(
            AcceleratorStream stream,
            in ArrayView<byte> source,
            in ArrayView<byte> destination)
        {
            if (destination.GetAcceleratorType() == AcceleratorType.CPU)
            {
                var sourceContiguous = (IContiguousArrayView)source;
                var destContiguous = (IContiguousArrayView)destination;
                var destBuffer = destContiguous.Buffer;
                var destPtr = destBuffer.NativePtr + (int)destContiguous.Index;
                var length = (int)source.Length;

                var byteArray = _backingArray!.Read<byte>((int)sourceContiguous.Index, length);
                Marshal.Copy(byteArray, 0, destPtr, length);
            }
            else
            {
                throw new NotSupportedException("Peer-to-peer copies not supported in WebGL backend.");
            }
        }

        protected override void MemSet(
            AcceleratorStream stream,
            byte value,
            in ArrayView<byte> view)
        {
            var viewContiguous = (IContiguousArrayView)view;
            var data = new byte[view.Length];
            if (value != 0) global::System.Array.Fill(data, value);
            _backingArray!.Write(data, (int)viewContiguous.Index);
        }

        protected override void DisposeAcceleratorObject(bool disposing)
        {
            if (_disposed) return;
            _disposed = true;
            if (disposing)
            {
                _backingArray?.Dispose();
                _backingArray = null;
            }
        }
    }
}
