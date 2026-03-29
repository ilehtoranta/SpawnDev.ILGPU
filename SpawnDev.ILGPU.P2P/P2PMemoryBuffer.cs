using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.P2P;

/// <summary>
/// P2P memory buffer — represents data that may reside on a remote peer device.
/// Tracks data locality for intelligent kernel dispatch routing.
///
/// The buffer maintains a local shadow copy of the data. When the coordinator
/// copies data to this buffer (CopyTo), the shadow is updated and marked dirty.
/// When a kernel dispatch targets the buffer's resident peer, the dirty shadow
/// is sent via WebRTC. When the coordinator reads back (CopyFrom), the latest
/// data is retrieved from the shadow (updated after kernel execution).
///
/// This design avoids blocking the synchronous MemoryBuffer API on async WebRTC.
/// </summary>
public class P2PMemoryBuffer : MemoryBuffer
{
    /// <summary>
    /// Which peer this buffer's data currently resides on (null = local/host).
    /// </summary>
    public RemotePeer? ResidentPeer { get; set; }

    /// <summary>
    /// Whether the local copy is current (not stale from remote writes).
    /// </summary>
    public bool IsLocalCurrent { get; internal set; } = true;

    /// <summary>
    /// Local shadow copy of the buffer data (for send/receive staging).
    /// Access must be synchronized via ShadowLock.
    /// </summary>
    public byte[] ShadowData { get; set; }

    /// <summary>
    /// Whether the shadow has been modified locally and needs to be sent to the peer.
    /// </summary>
    public bool IsDirty { get; set; }

    /// <summary>Lock for thread-safe shadow data access.</summary>
    internal readonly object ShadowLock = new();

    /// <summary>
    /// Unique buffer ID for transport-level tracking.
    /// </summary>
    public string BufferId { get; } = Guid.NewGuid().ToString("N");

    public P2PMemoryBuffer(Accelerator accelerator, long length, int elementSize)
        : base(accelerator, length, elementSize)
    {
        NativePtr = IntPtr.Zero;
        ShadowData = new byte[length * elementSize];
    }

    /// <inheritdoc/>
    protected override void MemSet(
        AcceleratorStream stream, byte value, in ArrayView<byte> targetView)
    {
        lock (ShadowLock)
        {
            Array.Fill(ShadowData, value, 0, (int)Math.Min(targetView.Length, ShadowData.Length));
            IsDirty = true;
            IsLocalCurrent = true;
        }
    }

    /// <inheritdoc/>
    protected override void CopyTo(
        AcceleratorStream stream,
        in ArrayView<byte> sourceView,
        in ArrayView<byte> targetView)
    {
        lock (ShadowLock)
        {
            if (sourceView.Length > 0 && sourceView.Length <= ShadowData.Length)
            {
                unsafe
                {
                    var srcPtr = sourceView.LoadEffectiveAddressAsPtr();
                    if (srcPtr != IntPtr.Zero)
                    {
                        System.Runtime.InteropServices.Marshal.Copy(
                            srcPtr, ShadowData, 0, (int)Math.Min(sourceView.Length, ShadowData.Length));
                    }
                }
                IsDirty = true;
                IsLocalCurrent = true;
            }
        }
    }

    /// <inheritdoc/>
    protected override void CopyFrom(
        AcceleratorStream stream,
        in ArrayView<byte> sourceView,
        in ArrayView<byte> targetView)
    {
        lock (ShadowLock)
        {
            if (targetView.Length > 0 && ShadowData.Length > 0)
            {
                unsafe
                {
                    var dstPtr = targetView.LoadEffectiveAddressAsPtr();
                    if (dstPtr != IntPtr.Zero)
                    {
                        System.Runtime.InteropServices.Marshal.Copy(
                            ShadowData, 0, dstPtr, (int)Math.Min(targetView.Length, ShadowData.Length));
                    }
                }
            }
        }
    }

    /// <summary>
    /// Update the shadow data from bytes received from a remote peer (after kernel execution).
    /// </summary>
    public void UpdateFromRemote(byte[] data)
    {
        lock (ShadowLock)
        {
            Array.Copy(data, ShadowData, Math.Min(data.Length, ShadowData.Length));
            IsDirty = false;
            IsLocalCurrent = true;
        }
    }

    /// <summary>
    /// Get the current shadow data for transmission to a remote peer.
    /// </summary>
    public byte[] GetShadowForTransmission()
    {
        lock (ShadowLock)
        {
            IsDirty = false;
            return ShadowData.ToArray();
        }
    }

    /// <inheritdoc/>
    protected override void DisposeAcceleratorObject(bool disposing)
    {
        if (disposing)
        {
            ResidentPeer = null;
            ShadowData = Array.Empty<byte>();
        }
    }
}
