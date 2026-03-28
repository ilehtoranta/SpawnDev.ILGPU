using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.P2P;

/// <summary>
/// P2P memory buffer — represents data that may reside on a remote peer device.
/// Tracks data locality for intelligent kernel dispatch routing.
/// </summary>
public class P2PMemoryBuffer : MemoryBuffer
{
    /// <summary>
    /// Which peer this buffer's data currently resides on (null = local/host).
    /// </summary>
    public RemotePeer? ResidentPeer { get; internal set; }

    /// <summary>
    /// Whether the local copy is current (not stale from remote writes).
    /// </summary>
    public bool IsLocalCurrent { get; internal set; } = true;

    public P2PMemoryBuffer(Accelerator accelerator, long length, int elementSize)
        : base(accelerator, length, elementSize)
    {
        NativePtr = IntPtr.Zero;
    }

    /// <inheritdoc/>
    protected override void MemSet(
        AcceleratorStream stream, byte value, in ArrayView<byte> targetView)
    {
        // Will be implemented with remote peer communication
    }

    /// <inheritdoc/>
    protected override void CopyTo(
        AcceleratorStream stream,
        in ArrayView<byte> sourceView,
        in ArrayView<byte> targetView)
    {
        // Will send data to remote peer via WebRTC
    }

    /// <inheritdoc/>
    protected override void CopyFrom(
        AcceleratorStream stream,
        in ArrayView<byte> sourceView,
        in ArrayView<byte> targetView)
    {
        // Will receive data from remote peer via WebRTC
    }

    /// <inheritdoc/>
    protected override void DisposeAcceleratorObject(bool disposing)
    {
        if (disposing)
        {
            ResidentPeer = null;
        }
    }
}
