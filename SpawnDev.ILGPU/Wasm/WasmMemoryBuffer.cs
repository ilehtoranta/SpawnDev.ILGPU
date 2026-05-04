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
        /// Monotonic counter bumped on every host-side write to <see cref="SharedBuffer"/>
        /// (CopyFromHost / CopyFromJS / CopyFrom override). The dispatch's copy-OUT
        /// phase compares the buffer's current <see cref="HostWriteCounter"/> to the
        /// snapshot it took at copy-IN time; if they differ, the host has overwritten
        /// SharedBuffer during the in-flight dispatch and copy-OUT skips that buffer
        /// to preserve the host's write. Closes the 2026-05-03 Wasm copy-OUT race
        /// (per `_DevComms/SpawnDev.ILGPU/geordi-to-team-wasm-copy-out-race-2026-05-03.md`).
        /// </summary>
        public int HostWriteCounter { get; private set; }

        /// <summary>
        /// Bumps <see cref="HostWriteCounter"/>. Called from every CopyFromCPU /
        /// CopyFromJS / CopyFromHost path. Use whenever the host writes to
        /// SharedBuffer outside a dispatch's copy-IN.
        /// </summary>
        internal void NotifyHostWrite() => HostWriteCounter++;

        // ── Queued-dispatch snapshot to close the host-write-vs-copy-IN race ──
        // Surfaced 2026-05-04 by Tests23_HostWriteVsQueuedDispatchRace and Data's
        // YOLOv8 Wasm Softmax bug. The race fires when:
        //   1. CopyFromCPU writes data1 to SharedBuffer.
        //   2. RunKernel queues dispatch D1 (returns immediately, async task in flight).
        //   3. CopyFromCPU writes data2 to SharedBuffer.
        //   4. RunKernel queues dispatch D2.
        //   5. D1 resumes after awaiting prior tasks. Its copy-IN reads SharedBuffer
        //      = data2 (already overwritten by step 3). D1 runs with WRONG data.
        //
        // Fix: at RunKernel SYNC time (before queueing the task), each arg buffer
        // calls GetOrCreateSnapshotForDispatch(). If the buffer was host-written
        // since the last snapshot, a fresh SAB is allocated and SharedBuffer's
        // current content is copied into it. The dispatch's task captures the
        // snapshot reference; copy-IN reads from snapshot instead of SharedBuffer.
        //
        // For unchanged buffers (e.g. ML weights uploaded once and read many times),
        // the SAME snapshot is shared across dispatches — no per-dispatch allocation
        // overhead. Only buffers that get host-written between dispatches (e.g.
        // TransposeKernel's reused paramsBuf) trigger fresh snapshots.

        /// <summary>
        /// HostWriteCounter at the most recent <see cref="GetOrCreateSnapshotForDispatch"/> call.
        /// </summary>
        private int _lastSnapshottedHostWriteCounter = -1;

        /// <summary>
        /// Snapshot SAB capturing <see cref="SharedBuffer"/> at counter
        /// <see cref="_lastSnapshottedHostWriteCounter"/>. Pinned dispatches reference
        /// this; replaced on next host write. Multiple dispatches with the same
        /// counter share the same snapshot instance.
        /// </summary>
        private SharedArrayBuffer? _currentSnapshot;

        /// <summary>
        /// GPU-write sequence counter — incremented every time the dispatcher's
        /// copy-OUT path writes back to <see cref="SharedBuffer"/>. Used to invalidate
        /// the cached snapshot when a buffer is host-written-then-GPU-written-then-read
        /// (snapshot would be pre-GPU-write contents but SharedBuffer has post-GPU-write
        /// contents — cached snapshot is stale).
        /// </summary>
        private int _gpuWriteSeq;

        /// <summary>
        /// <see cref="_gpuWriteSeq"/> at the most recent snapshot. If the current value
        /// is higher, the cached snapshot is stale.
        /// </summary>
        private int _lastSnapshottedGpuWriteSeq;

        /// <summary>
        /// Bumps <see cref="_gpuWriteSeq"/>. Called from the dispatcher's copy-OUT path
        /// after writing kernel output back to <see cref="SharedBuffer"/>.
        /// </summary>
        internal void NotifyGpuWrite() => _gpuWriteSeq++;

        /// <summary>
        /// Returns a SAB whose contents reflect <see cref="SharedBuffer"/> at the moment
        /// of this call (or at the last call if no host or GPU writes occurred since).
        /// Returns null when this buffer has never been host-written — in that case the
        /// caller falls back to <see cref="SharedBuffer"/> directly, which by run-time
        /// reflects all prior dispatches' copy-OUT (intermediate GPU buffers don't
        /// participate in the host-write-vs-queued-dispatch race so they don't need a
        /// snapshot, and snapshotting them at queue time would capture pre-D1-write
        /// zeros). 2026-05-04 Data BlazeFace zeros regression closure.
        /// </summary>
        internal SharedArrayBuffer? GetOrCreateSnapshotForDispatch()
        {
            // Buffers that have never been host-written can't be involved in the
            // host-write-vs-queued-dispatch race — fall back to SharedBuffer.
            if (HostWriteCounter == 0)
                return null;

            // Cache the snapshot across dispatches when nothing has changed; invalidate
            // on either a new host write OR a new GPU write (the latter handles the
            // host-write -> kernel-write -> kernel-read sequence where the cached
            // snapshot would otherwise return pre-GPU-write contents).
            bool needsRefresh =
                HostWriteCounter > _lastSnapshottedHostWriteCounter
                || _gpuWriteSeq > _lastSnapshottedGpuWriteSeq
                || _currentSnapshot == null;
            if (needsRefresh)
            {
                var fresh = new SharedArrayBuffer((int)LengthInBytes);
                using var dst = new Uint8Array(fresh);
                using var src = new Uint8Array(SharedBuffer);
                dst.JSRef!.CallVoid("set", src);
                _currentSnapshot = fresh;
                _lastSnapshottedHostWriteCounter = HostWriteCounter;
                _lastSnapshottedGpuWriteSeq = _gpuWriteSeq;
            }
            return _currentSnapshot;
        }

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
            long totalBytesLong = length * elementSize;
            if (totalBytesLong > int.MaxValue || totalBytesLong < 0)
                throw new ArgumentOutOfRangeException(nameof(length),
                    $"Buffer size {totalBytesLong} bytes exceeds maximum SharedArrayBuffer capacity (2GB)");
            int totalBytes = (int)totalBytesLong;
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
        /// Data crosses the .NET/JS boundary. For browser backends, prefer
        /// <see cref="CopyFromJS(TypedArray, long)"/> when data is already in JS.
        /// </summary>
        public void CopyFromHost<T>(T[] data) where T : unmanaged
        {
            TypedArrayView.Write(data);
            NotifyHostWrite();
        }

        /// <inheritdoc/>
        public void CopyFromJS(TypedArray source, long targetByteOffset = 0)
        {
            if (TypedArrayView == null)
                throw new ObjectDisposedException(nameof(WasmMemoryBuffer));
            // Use the typed Set(TypedArray, long) overload - zero .NET copy, JS-to-JS
            using var srcBytes = new Uint8Array(source.Buffer, (int)source.ByteOffset, (int)source.ByteLength);
            TypedArrayView.Set(srcBytes, targetByteOffset);
            NotifyHostWrite();
        }

        /// <inheritdoc/>
        public void CopyFromJS(ArrayBuffer source, long targetByteOffset = 0)
        {
            if (TypedArrayView == null)
                throw new ObjectDisposedException(nameof(WasmMemoryBuffer));
            using var srcBytes = new Uint8Array(source);
            TypedArrayView.Set(srcBytes, targetByteOffset);
            NotifyHostWrite();
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
        /// Awaits all pending kernel dispatches before reading, matching
        /// desktop backends' implicit synchronization on readback.
        /// </summary>
        public async Task<T[]> CopyToHostAsync<T>(long length) where T : unmanaged
        {
            // Implicit sync before readback - match CUDA/OpenCL behavior where
            // CopyToCPU calls stream.Synchronize() before reading data
            if (Accelerator is WasmAccelerator wasmAccel)
                await wasmAccel.SynchronizeAsync();
            return CopyToHost<T>(length);
        }

        public async Task<Uint8Array> CopyToHostUint8ArrayAsync(long sourceByteOffset = 0, long? copyBytes = null)
        {
            if (SharedBuffer == null) return new Uint8Array();
            // Implicit sync before readback - match desktop behavior
            if (Accelerator is WasmAccelerator wasmAccel)
                await wasmAccel.SynchronizeAsync();
            long bufferSize = SharedBuffer.ByteLength;
            long actualCopyBytes = copyBytes ?? (bufferSize - sourceByteOffset);
            if (sourceByteOffset < 0 || sourceByteOffset > bufferSize)
                throw new ArgumentOutOfRangeException(nameof(sourceByteOffset),
                    $"Source offset {sourceByteOffset} is outside buffer bounds [0, {bufferSize})");
            if (actualCopyBytes < 0 || sourceByteOffset + actualCopyBytes > bufferSize)
                throw new ArgumentOutOfRangeException(nameof(copyBytes),
                    $"Copy range [{sourceByteOffset}, {sourceByteOffset + actualCopyBytes}) exceeds buffer size {bufferSize}");
            return copyBytes == null ?
                new Uint8Array(SharedBuffer, sourceByteOffset) :
                new Uint8Array(SharedBuffer, sourceByteOffset, copyBytes.Value);
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
            NotifyHostWrite();
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
                // Bounds validation
                if (sourceOffsetInBytes + lengthInBytes > wasmSource.SharedBuffer.ByteLength)
                    throw new ArgumentOutOfRangeException(nameof(sourceOffsetInBytes),
                        $"Source copy range [{sourceOffsetInBytes}, {sourceOffsetInBytes + lengthInBytes}) exceeds source buffer size {wasmSource.SharedBuffer.ByteLength}");
                if (targetOffsetInBytes + lengthInBytes > SharedBuffer.ByteLength)
                    throw new ArgumentOutOfRangeException(nameof(targetOffsetInBytes),
                        $"Target copy range [{targetOffsetInBytes}, {targetOffsetInBytes + lengthInBytes}) exceeds target buffer size {SharedBuffer.ByteLength}");

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
