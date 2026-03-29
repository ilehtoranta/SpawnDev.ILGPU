using SpawnDev.WebTorrent;
using SpawnDev.WebTorrent.Torrent;
using SpawnDev.WebTorrent.Wire;

namespace SpawnDev.ILGPU.P2P;

/// <summary>
/// Bridges WebTorrent peer connections to the P2P compute system.
/// Registers sd_compute extension on each new peer, wires message routing.
///
/// Usage:
///   var bridge = new P2PWebRtcBridge(transport);
///   // When a WebTorrent peer connects:
///   bridge.AttachToPeer(wire, peerId);
///   // Now compute messages flow through the real WebRTC data channel.
///
/// Or attach to a whole swarm:
///   bridge.AttachToSwarm(torrentSwarm);
///   // Automatically wires sd_compute to every peer that connects.
/// </summary>
public class P2PWebRtcBridge
{
    private readonly P2PTransport _transport;
    private readonly System.Collections.Concurrent.ConcurrentDictionary<string, SdComputeExtension> _extensions = new();

    /// <summary>
    /// Number of peers with sd_compute wired.
    /// </summary>
    public int ComputePeerCount => _extensions.Count;

    /// <summary>
    /// Fired when a new compute-capable peer connects.
    /// </summary>
    public event Action<string>? OnComputePeerConnected;

    public P2PWebRtcBridge(P2PTransport transport)
    {
        _transport = transport;
    }

    /// <summary>
    /// Attach sd_compute extension to a specific wire protocol connection.
    /// Call before the extension handshake is exchanged.
    /// </summary>
    public SdComputeExtension AttachToPeer(WireProtocol wire, string peerId)
    {
        var ext = new SdComputeExtension(_transport);
        ext.SetPeerId(peerId);

        // Wire the extension's send requests to the wire protocol
        ext.OnSendRequested += async (remoteExtId, data) =>
        {
            await wire.SendExtensionMessageAsync(remoteExtId, data);
        };

        // Register in the wire protocol's extension manager
        wire.Extensions.Register(ext);

        _extensions[peerId] = ext;

        // Notify when compute handshake completes
        ext.OnComputeMessage += (msg) =>
        {
            if (msg.Type == P2PMessageType.CapabilityResponse && _notified.TryAdd(peerId, true))
            {
                OnComputePeerConnected?.Invoke(peerId);
            }
        };

        return ext;
    }
    private readonly System.Collections.Concurrent.ConcurrentDictionary<string, bool> _notified = new();

    /// <summary>
    /// Attach to a TorrentSwarm — automatically wires sd_compute
    /// to every peer that connects to this swarm.
    /// </summary>
    public void AttachToSwarm(TorrentSwarm swarm)
    {
        swarm.OnPeerConnect += (peerConnection) =>
        {
            var peerId = peerConnection.Info.Address ?? Guid.NewGuid().ToString("N");
            AttachToPeer(peerConnection.Wire, peerId);
        };
    }

    /// <summary>
    /// Send a compute message to a specific peer via the real WebRTC channel.
    /// </summary>
    public async Task SendAsync(string peerId, P2PMessage message)
    {
        if (_extensions.TryGetValue(peerId, out var ext) && ext.IsSupported)
        {
            await ext.SendAsync(message);
        }
    }

    /// <summary>
    /// Check if a peer has sd_compute support.
    /// </summary>
    public bool IsComputeCapable(string peerId)
    {
        return _extensions.TryGetValue(peerId, out var ext) && ext.IsSupported;
    }
}
