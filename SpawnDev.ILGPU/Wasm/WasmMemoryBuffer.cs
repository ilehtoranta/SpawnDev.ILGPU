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
    public class WasmMemoryBuffer : MemoryBuffer, IBrowserMemoryBuffer
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

            // NativePtr = 0: Wasm buffers don't use native pointers.
            // ArrayView.LoadEffectiveAddressAsPtr() returns NativePtr + Index * ElementSize.
            // With NativePtr=0, SubView offsets are purely Index-based (correct for Wasm).
            // The multi-pass scan (which needs non-zero NativePtr) is not used for Wasm —
            // Wasm routes to single-group scan via AcceleratorType.Wasm in ScanExtensions.
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

        public Task<Uint8Array> CopyToHostUint8ArrayAsync(long sourceByteOffset = 0, long? copyBytes = null)
        {
            if (SharedBuffer == null) return Task.FromResult(new Uint8Array());
            return copyBytes == null ? 
                Task.FromResult(new Uint8Array(SharedBuffer, sourceByteOffset)) : 
                Task.FromResult(new Uint8Array(SharedBuffer, sourceByteOffset, copyBytes.Value));
        }

        /// <inheritdoc/>
        protected override void MemSet(
            AcceleratorStream stream,
            byte value,
            in ArrayView<byte> targetView)
        {
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
            int srcOffset = (int)sourceView.LoadEffectiveAddressAsPtr();
            int length = (int)sourceView.LengthInBytes;

            using var srcUint8 = new Uint8Array(SharedBuffer, srcOffset, length);
            byte[] data = srcUint8.ReadBytes();

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
            int length = (int)sourceView.LengthInBytes;

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

        /// <summary>
        /// Copies data from the source buffer to this buffer.
        /// Handles Wasm-to-Wasm copies via SharedArrayBuffer directly,
        /// bypassing Marshal.Copy which requires native pointers.
        /// </summary>
        protected override void CopyFromBuffer(
            AcceleratorStream stream,
            MemoryBuffer sourceBuffer,
            long sourceOffsetInBytes,
            long targetOffsetInBytes,
            long lengthInBytes)
        {
            if (sourceBuffer is WasmMemoryBuffer wasmSource)
            {
                // Wasm-to-Wasm: copy between SharedArrayBuffers via JS TypedArray
                using var srcView = new Uint8Array(
                    wasmSource.SharedBuffer,
                    (int)sourceOffsetInBytes,
                    (int)lengthInBytes);
                using var dstView = new Uint8Array(
                    SharedBuffer,
                    (int)targetOffsetInBytes,
                    (int)lengthInBytes);
                dstView.JSRef!.CallVoid("set", srcView);
                return;
            }
            // Non-Wasm source: fall back to default (via native pointer)
            base.CopyFromBuffer(
                stream, sourceBuffer,
                sourceOffsetInBytes, targetOffsetInBytes, lengthInBytes);
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
