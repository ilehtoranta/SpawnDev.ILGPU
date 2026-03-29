using System.Reflection;
using ILGPU;
using ILGPU.Backends;
using ILGPU.Runtime;
using SpawnDev.WebTorrent;

namespace SpawnDev.ILGPU.P2P;

/// <summary>
/// P2P accelerator — distributes kernel dispatch across connected peers
/// via SpawnDev.WebTorrent WebRTC data channels.
///
/// Extends KernelAccelerator with P2PCompiledKernel/P2PKernel types.
/// Delegates execution to remote peers running their own local backends.
/// </summary>
public class P2PAccelerator : KernelAccelerator<P2PCompiledKernel, P2PKernel>
{
    private readonly P2PDevice _device;
    private readonly System.Collections.Concurrent.ConcurrentDictionary<string, RemotePeer> _peers = new();

    /// <summary>
    /// Connected remote peers.
    /// </summary>
    public IReadOnlyList<RemotePeer> Peers => _peers.Values.ToList();

    /// <summary>
    /// The WebTorrent client used for P2P communication.
    /// </summary>
    public WebTorrentClient? TorrentClient { get; set; }

    /// <summary>
    /// Creates a new P2P accelerator.
    /// </summary>
    internal P2PAccelerator(Context context, P2PDevice device)
        : base(context, device)
    {
        _device = device;
        DefaultStream = CreateStreamInternal();
        Init(new P2PBackend(context));
    }

    /// <summary>
    /// Add a remote peer to the accelerator.
    /// </summary>
    public void AddPeer(RemotePeer peer)
    {
        _peers[peer.PeerId] = peer;
        peer.Accelerator = this;
    }

    /// <summary>
    /// Remove a remote peer.
    /// </summary>
    public void RemovePeer(RemotePeer peer)
    {
        _peers.TryRemove(peer.PeerId, out _);
        peer.Accelerator = null;
    }

    /// <summary>
    /// Select the best peer for executing a kernel, based on data locality.
    /// </summary>
    public RemotePeer? SelectPeer(P2PMemoryBuffer[]? dataBuffers = null)
    {
        var peers = _peers.Values.ToList();
        if (peers.Count == 0) return null;
        if (dataBuffers != null)
        {
            foreach (var peer in peers)
            {
                if (dataBuffers.All(b => b.ResidentPeer == peer))
                    return peer;
            }
        }
        return peers[Random.Shared.Next(peers.Count)];
    }

    #region KernelAccelerator Implementation

    /// <inheritdoc/>
    protected override P2PKernel CreateKernel(P2PCompiledKernel compiledKernel)
    {
        return new P2PKernel(this, compiledKernel, null);
    }

    /// <inheritdoc/>
    protected override P2PKernel CreateKernel(
        P2PCompiledKernel compiledKernel,
        MethodInfo launcher)
    {
        return new P2PKernel(this, compiledKernel, launcher);
    }

    /// <inheritdoc/>
    protected override MethodInfo GenerateKernelLauncherMethod(
        P2PCompiledKernel kernel, int customGroupSize)
    {
        // P2P doesn't generate IL launchers — dispatch is handled via
        // remote peer communication. Return a placeholder method.
        // TODO: Implement proper dispatch delegation to remote peers.
        throw new NotImplementedException(
            "P2P kernel launcher generation — requires remote dispatch protocol");
    }

    #endregion

    #region Accelerator Implementation

    /// <inheritdoc/>
    protected override AcceleratorStream CreateStreamInternal()
    {
        return new P2PStream(this);
    }

    /// <inheritdoc/>
    protected override void SynchronizeInternal() { }

    /// <inheritdoc/>
    protected override MemoryBuffer AllocateRawInternal(long length, int elementSize)
    {
        return new P2PMemoryBuffer(this, length, elementSize);
    }

    /// <inheritdoc/>
    protected override int EstimateMaxActiveGroupsPerMultiprocessorInternal(
        Kernel kernel, int groupSize, int dynamicSharedMemorySizeInBytes) => 1;

    /// <inheritdoc/>
    protected override int EstimateGroupSizeInternal(
        Kernel kernel, int dynamicSharedMemorySizeInBytes, int maxGroupSize, out int minGridSize)
    {
        minGridSize = 1;
        return Math.Min(256, maxGroupSize);
    }

    /// <inheritdoc/>
    protected override int EstimateGroupSizeInternal(
        Kernel kernel, Func<int, int> computeSharedMemorySize, int maxGroupSize, out int minGridSize)
    {
        minGridSize = 1;
        return Math.Min(256, maxGroupSize);
    }

    /// <inheritdoc/>
    protected override PageLockScope<T> CreatePageLockFromPinnedInternal<T>(IntPtr ptr, long numElements) =>
        throw new NotSupportedException("Page locking not supported for P2P accelerator");

    /// <inheritdoc/>
    protected override void EnablePeerAccessInternal(Accelerator otherAccelerator) { }

    /// <inheritdoc/>
    protected override void DisablePeerAccessInternal(Accelerator otherAccelerator) { }

    /// <inheritdoc/>
    protected override bool CanAccessPeerInternal(Accelerator otherAccelerator) => false;

    /// <inheritdoc/>
    public override TExtension CreateExtension<TExtension, TExtensionProvider>(TExtensionProvider provider) =>
        throw new NotSupportedException("Extensions not supported for P2P accelerator");

    /// <inheritdoc/>
    protected override void OnBind() { }

    /// <inheritdoc/>
    protected override void OnUnbind() { }

    /// <inheritdoc/>
    protected override void DisposeAccelerator_SyncRoot(bool disposing)
    {
        if (disposing)
        {
            foreach (var peer in _peers.Values)
                peer.Disconnect();
            _peers.Clear();
        }
    }

    #endregion
}

/// <summary>
/// A kernel wrapper for P2P dispatch.
/// </summary>
public class P2PKernel : Kernel
{
    public new P2PCompiledKernel CompiledKernel { get; }

    public P2PKernel(P2PAccelerator accelerator, P2PCompiledKernel compiledKernel, MethodInfo? launcher)
        : base(accelerator, compiledKernel, launcher)
    {
        CompiledKernel = compiledKernel;
    }

    protected override void DisposeAcceleratorObject(bool disposing) { }
}

/// <summary>
/// Represents a remote peer device connected via WebRTC.
/// </summary>
public class RemotePeer
{
    public string PeerId { get; set; } = "";
    public AcceleratorType RemoteBackend { get; set; }
    public long MemorySize { get; set; }
    public bool IsConnected { get; set; }
    private int _pendingOperations;
    public int PendingOperations
    {
        get => _pendingOperations;
        set => Interlocked.Exchange(ref _pendingOperations, value);
    }
    public void IncrementPending() => Interlocked.Increment(ref _pendingOperations);
    public void DecrementPending() => Interlocked.Decrement(ref _pendingOperations);
    public PeerCapabilities? Capabilities { get; set; }
    public DateTime LastHeartbeat { get; set; } = DateTime.MinValue;
    internal P2PAccelerator? Accelerator { get; set; }

    public void Disconnect()
    {
        IsConnected = false;
    }
}
