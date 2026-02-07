// ---------------------------------------------------------------------------------------
//                               SpawnDev.ILGPU.Workers
//                 Web Worker Compute Library for Blazor WebAssembly
//
// File: WorkersMemoryBuffer.cs
//
// MemoryBuffer backed by SharedArrayBuffer for zero-copy worker access.
// ---------------------------------------------------------------------------------------

using global::ILGPU;
using global::ILGPU.Runtime;
using SpawnDev.BlazorJS;
using SpawnDev.BlazorJS.JSObjects;
using SpawnDev.ILGPU.Workers.Backend;
using System.Runtime.InteropServices;

namespace SpawnDev.ILGPU.Workers
{
    /// <summary>
    /// ILGPU MemoryBuffer backed by a SharedArrayBuffer (or ArrayBuffer fallback).
    /// Enables zero-copy sharing between the main thread and Web Workers.
    /// When Cross-Origin Isolation is not available, falls back to a regular ArrayBuffer
    /// which still supports single-threaded kernel execution.
    /// </summary>
    public class WorkersMemoryBuffer : MemoryBuffer
    {
        private SharedArrayBuffer? _sharedBuffer;
        private ArrayBuffer? _arrayBuffer;
        private Uint8Array? _uint8View;
        private bool _disposed;
        private bool _isShared;

        /// <summary>
        /// Checks if the browser supports SharedArrayBuffer (requires Cross-Origin Isolation).
        /// </summary>
        public static bool IsCrossOriginIsolated
        {
            get
            {
                try
                {
                    return BlazorJSRuntime.JS.Get<bool>("crossOriginIsolated");
                }
                catch
                {
                    return false;
                }
            }
        }

        /// <summary>
        /// Constructs a new Workers memory buffer backed by SharedArrayBuffer (or ArrayBuffer fallback).
        /// </summary>
        public WorkersMemoryBuffer(WorkersAccelerator accelerator, long length, int elementSize)
            : base(accelerator, length, elementSize)
        {
            _isShared = IsCrossOriginIsolated;

            if (_isShared)
            {
                // SharedArrayBuffer: enables zero-copy transfer to Web Workers
                _sharedBuffer = new SharedArrayBuffer((int)LengthInBytes);
                _uint8View = new Uint8Array(_sharedBuffer);
                WorkersBackend.Log($"[Workers] Allocated SharedArrayBuffer: {LengthInBytes} bytes ({length} × {elementSize})");
            }
            else
            {
                // Fallback: regular ArrayBuffer for single-threaded execution
                _arrayBuffer = new ArrayBuffer((int)LengthInBytes);
                _uint8View = new Uint8Array(_arrayBuffer);
                WorkersBackend.Log($"[Workers] Allocated ArrayBuffer (no COI): {LengthInBytes} bytes ({length} × {elementSize})");
            }
        }

        #region Properties

        /// <summary>
        /// Gets the underlying SharedArrayBuffer for transfer to workers.
        /// Returns null when using ArrayBuffer fallback.
        /// </summary>
        public SharedArrayBuffer? SharedBuffer => _sharedBuffer;

        /// <summary>
        /// Gets the underlying buffer (SharedArrayBuffer or ArrayBuffer) as a JSObject.
        /// Used by the execution pipeline to set the buffer on globalThis.
        /// </summary>
        public object UnderlyingBuffer => _isShared ? (object)_sharedBuffer! : (object)_arrayBuffer!;

        /// <summary>
        /// Gets whether this buffer uses SharedArrayBuffer (requires Cross-Origin Isolation).
        /// </summary>
        public bool IsShared => _isShared;

        /// <summary>
        /// Gets a Uint8Array view over the buffer.
        /// </summary>
        public Uint8Array? Uint8View => _uint8View;

        #endregion

        #region MemoryBuffer Implementation

        /// <summary>
        /// Copies data from a source view to a destination view (CPU → Workers buffer).
        /// </summary>
        protected override void CopyFrom(
            AcceleratorStream stream,
            in ArrayView<byte> source,
            in ArrayView<byte> destination)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(WorkersMemoryBuffer));

            if (source.GetAcceleratorType() == AcceleratorType.CPU)
            {
                var length = (int)source.Length;
                var sourceContiguous = (IContiguousArrayView)source;
                var sourceBuffer = sourceContiguous.Buffer;
                var srcPtr = sourceBuffer.NativePtr + (int)sourceContiguous.Index;

                // Read from CPU buffer to managed byte array
                var byteArray = new byte[length];
                Marshal.Copy(srcPtr, byteArray, 0, length);

                // Write to SharedArrayBuffer via Uint8Array
                var destContiguous = (IContiguousArrayView)destination;
                var destOffset = (int)destContiguous.Index;

                using var srcTypedArray = new Uint8Array(byteArray);
                _uint8View!.Set(srcTypedArray, destOffset);

                WorkersBackend.Log($"[Workers] CopyFrom (CPU→SAB): {length} bytes at offset {destOffset}");
            }
            else
            {
                throw new NotSupportedException("Peer-to-peer copies not yet implemented for Workers backend.");
            }
        }

        /// <summary>
        /// Copies data from Workers buffer to destination (Workers buffer → CPU).
        /// Since SharedArrayBuffer is directly accessible, this creates a copy.
        /// </summary>
        protected override void CopyTo(
            AcceleratorStream stream,
            in ArrayView<byte> source,
            in ArrayView<byte> destination)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(WorkersMemoryBuffer));

            var sourceContiguous = (IContiguousArrayView)source;
            var destContiguous = (IContiguousArrayView)destination;
            var length = (int)source.Length;
            var srcOffset = (int)sourceContiguous.Index;

            // Read from SharedArrayBuffer
            using var slice = _uint8View!.Slice(srcOffset, srcOffset + length);
            var byteArray = slice.ReadBytes();

            // Write to destination CPU buffer
            var destBuffer = destContiguous.Buffer;
            var dstPtr = destBuffer.NativePtr + (int)destContiguous.Index;
            Marshal.Copy(byteArray, 0, dstPtr, length);

            WorkersBackend.Log($"[Workers] CopyTo (SAB→CPU): {length} bytes from offset {srcOffset}");
        }

        /// <summary>
        /// Fills the buffer with a byte value.
        /// </summary>
        protected override void MemSet(
            AcceleratorStream stream,
            byte value,
            in ArrayView<byte> view)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(WorkersMemoryBuffer));

            var viewContiguous = (IContiguousArrayView)view;
            var offset = (int)viewContiguous.Index;
            var length = (int)view.Length;

            var data = new byte[length];
            if (value != 0) global::System.Array.Fill(data, value);

            using var typedArray = new Uint8Array(data);
            _uint8View!.Set(typedArray, offset);
        }

        /// <summary>
        /// Disposes the buffer and associated views.
        /// </summary>
        protected override void DisposeAcceleratorObject(bool disposing)
        {
            if (_disposed) return;
            _disposed = true;

            if (disposing)
            {
                _uint8View?.Dispose();
                _uint8View = null;
                _sharedBuffer?.Dispose();
                _sharedBuffer = null;
                _arrayBuffer?.Dispose();
                _arrayBuffer = null;
            }
        }

        #endregion

        #region Convenience Methods

        /// <summary>
        /// Reads the entire buffer content as a typed array.
        /// This is a synchronous operation since SharedArrayBuffer is directly accessible.
        /// </summary>
        public T[] ReadAll<T>() where T : unmanaged
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(WorkersMemoryBuffer));

            return _uint8View!.Read<T>();
        }

        /// <summary>
        /// Reads a portion of the buffer content as a typed array.
        /// </summary>
        public T[] Read<T>(int byteOffset, int count) where T : unmanaged
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(WorkersMemoryBuffer));

            var elementSize = Marshal.SizeOf<T>();
            using var slice = _uint8View!.Slice(byteOffset, byteOffset + count * elementSize);
            return slice.Read<T>();
        }

        /// <summary>
        /// Writes typed data to the buffer at a byte offset.
        /// </summary>
        public void Write<T>(T[] data, int byteOffset = 0) where T : unmanaged
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(WorkersMemoryBuffer));

            using var typedArray = new Uint8Array(data.Length * Marshal.SizeOf<T>());
            typedArray.Write(data);
            _uint8View!.Set(typedArray, byteOffset);
        }

        #endregion
    }
}
