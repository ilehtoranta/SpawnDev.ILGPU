using global::ILGPU;
using global::ILGPU.Runtime;
using SpawnDev.BlazorJS.JSObjects;
using System.Runtime.InteropServices;

namespace SpawnDev.ILGPU.WebGPU.Backend
{
    public class WebGPUMemoryBuffer : MemoryBuffer, IBrowserMemoryBuffer
    {
        private readonly WebGPUBuffer<byte> _buffer;

        public WebGPUMemoryBuffer(WebGPUAccelerator accelerator, long length, int elementSize)
            : base(accelerator, length, elementSize)
        {
            _buffer = accelerator.NativeAccelerator.Allocate<byte>(LengthInBytes);
        }

        public Task<Uint8Array> CopyToHostUint8ArrayAsync(long sourceByteOffset = 0, long? copyBytes = null) => NativeBuffer.CopyToHostUint8ArrayAsync(sourceByteOffset, copyBytes);

        public WebGPUBuffer<byte> NativeBuffer => _buffer;
        // Implementation of abstract members
        protected override void CopyFrom(AcceleratorStream stream, in ArrayView<byte> source, in ArrayView<byte> destination)
        {
            if (source.GetAcceleratorType() == AcceleratorType.CPU)
            {
                var length = (int)source.Length;

                // Use IContiguousArrayView to access internal members
                var sourceContiguous = (IContiguousArrayView)source;
                var sourceBuffer = sourceContiguous.Buffer;
                var srcPtr = sourceBuffer.NativePtr + (int)sourceContiguous.Index;

                // Read from CPU buffer to managed byte array (still required for NativePtr access)
                var byteArray = new byte[length];
                Marshal.Copy(srcPtr, byteArray, 0, length);

                // Flush pending dispatches before writing to the buffer
                var accelerator = (WebGPUAccelerator)Accelerator;
                accelerator.FlushPendingCommands();

                var destContiguous = (IContiguousArrayView)destination;
                using var typedArray = new Uint8Array(byteArray);
                accelerator.NativeAccelerator.Queue!.WriteBuffer(_buffer.NativeBuffer!, (long)destContiguous.Index, typedArray);
            }
            else
            {
                throw new NotSupportedException("Peer-to-peer copies not yet implemented");
            }
        }

        protected override void CopyTo(AcceleratorStream stream, in ArrayView<byte> source, in ArrayView<byte> destination)
        {
            // GPU to CPU - This is inherently async in WebGPU.
            // For now, we throw as ILGPU expects sync behavior here.
            // Users should use CopyToHostAsync in WebGPUBuffer for now.
            throw new NotSupportedException("Synchronous GPU to CPU copies are not supported in WebGPU backend. Use CopyToHostAsync.");
        }

        protected override void MemSet(AcceleratorStream stream, byte value, in ArrayView<byte> view)
        {
            // Use GPU queue WriteBuffer with filled array
            var data = new byte[view.Length];
            if (value != 0) global::System.Array.Fill(data, value);
            var accelerator = (WebGPUAccelerator)Accelerator;
            // Flush pending dispatches before writing to the buffer
            accelerator.FlushPendingCommands();
            var viewContiguous = (IContiguousArrayView)view;
            accelerator.NativeAccelerator.Queue!.WriteBuffer(_buffer.NativeBuffer!, (long)viewContiguous.Index, data);
        }

        // DisposeAcceleratorObject is protected (not protected internal) in base AcceleratorObject
        protected override void DisposeAcceleratorObject(bool disposing)
        {
            if (disposing) _buffer.Dispose();
        }



    }
}
