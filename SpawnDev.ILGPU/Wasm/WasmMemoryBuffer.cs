// ---------------------------------------------------------------------------------------
//                               SpawnDev.ILGPU.Wasm
//                    WebAssembly Compute Backend for Blazor WebAssembly
//
// File: WasmMemoryBuffer.cs
//
// Manages GPU memory buffers backed by SharedArrayBuffer regions.
// Each buffer is a slice of a SharedArrayBuffer for zero-copy sharing across workers.
// ---------------------------------------------------------------------------------------

using global::ILGPU;
using global::ILGPU.Runtime;
using SpawnDev.BlazorJS;
using SpawnDev.BlazorJS.JSObjects;

namespace SpawnDev.ILGPU.Wasm
{
    /// <summary>
    /// Wasm memory buffer backed by a SharedArrayBuffer for zero-copy sharing across workers.
    /// </summary>
    public class WasmMemoryBuffer : MemoryBuffer
    {
        /// <summary>
        /// The SharedArrayBuffer backing this buffer.
        /// </summary>
        public SharedArrayBuffer SharedBuffer { get; private set; }

        /// <summary>
        /// The typed array view for this buffer (e.g., Int32Array, Float32Array).
        /// </summary>
        public TypedArray TypedArrayView { get; private set; }

        /// <summary>
        /// Byte offset within the SharedArrayBuffer where this buffer starts.
        /// When the buffer owns its own SharedArrayBuffer, this is 0.
        /// </summary>
        public int ByteOffset { get; set; } = 0;

        /// <summary>
        /// Creates a new Wasm memory buffer.
        /// </summary>
        /// <param name="accelerator">The associated accelerator.</param>
        /// <param name="length">The number of elements to allocate.</param>
        /// <param name="elementSize">The size of each element in bytes.</param>
        public WasmMemoryBuffer(
            Accelerator accelerator,
            long length,
            int elementSize)
            : base(accelerator, length, elementSize)
        {
            // Compute total bytes: length (elements) × elementSize (bytes per element).
            int totalBytes = (int)(length * elementSize);
            SharedBuffer = new SharedArrayBuffer(totalBytes);

            // Create a Uint8Array view for raw data access
            TypedArrayView = new Uint8Array(SharedBuffer);
        }

        /// <summary>
        /// Copies data from the host to this buffer.
        /// </summary>
        public void CopyFromHost<T>(T[] data) where T : unmanaged
        {
            TypedArrayView.Write(data);
        }

        /// <summary>
        /// Copies data from this buffer to the host.
        /// </summary>
        public T[] CopyToHost<T>(long length) where T : unmanaged
        {
            return TypedArrayView.Read<T>(0, length);
        }

        /// <summary>
        /// Copies data from this buffer to host asynchronously.
        /// </summary>
        public Task<T[]> CopyToHostAsync<T>(long length) where T : unmanaged
        {
            return Task.FromResult(CopyToHost<T>(length));
        }

        /// <inheritdoc/>
        protected override void MemSet(
            AcceleratorStream stream,
            byte value,
            in ArrayView<byte> targetView)
        {
            // Phase 1: Set all bytes to the given value via JS typed array fill
            int offset = (int)targetView.LoadEffectiveAddressAsPtr();
            int length = (int)targetView.LengthInBytes;
            using var view = new Uint8Array(SharedBuffer, offset, length);
            view.JSRef!.CallVoid("fill", (int)value);
        }

        /// <inheritdoc/>
        protected override void CopyTo(
            AcceleratorStream stream,
            in ArrayView<byte> sourceView,
            in ArrayView<byte> targetView)
        {
            // sourceView is in THIS (Wasm) buffer, targetView is in a CPU buffer.
            // Read bytes from our SharedArrayBuffer and write to the CPU target.
            int srcOffset = (int)sourceView.LoadEffectiveAddressAsPtr();
            int length = (int)sourceView.LengthInBytes;

            // Read bytes from SharedArrayBuffer
            using var srcUint8 = new Uint8Array(SharedBuffer, srcOffset, length);
            byte[] data = srcUint8.ReadBytes();

            // Write to target CPU memory
            unsafe
            {
                var targetPtr = targetView.LoadEffectiveAddressAsPtr();
                System.Runtime.InteropServices.Marshal.Copy(data, 0, targetPtr, length);
            }
        }

        /// <inheritdoc/>
        protected override void CopyFrom(
            AcceleratorStream stream,
            in ArrayView<byte> sourceView,
            in ArrayView<byte> targetView)
        {
            // sourceView is in a CPU buffer, targetView is in THIS (Wasm) buffer.
            // Read bytes from the CPU source and write to our SharedArrayBuffer.
            int length = (int)sourceView.LengthInBytes;

            // Read bytes from CPU memory
            byte[] data = new byte[length];
            unsafe
            {
                var sourcePtr = sourceView.LoadEffectiveAddressAsPtr();
                System.Runtime.InteropServices.Marshal.Copy(sourcePtr, data, 0, length);
            }

            // Write to SharedArrayBuffer
            int dstOffset = (int)targetView.LoadEffectiveAddressAsPtr();
            using var dstUint8 = new Uint8Array(SharedBuffer, dstOffset, length);
            dstUint8.WriteBytes(data);
        }

        /// <inheritdoc/>
        protected override void DisposeAcceleratorObject(bool disposing)
        {
            if (disposing)
            {
                TypedArrayView?.Dispose();
                SharedBuffer?.Dispose();
            }
        }
    }
}
