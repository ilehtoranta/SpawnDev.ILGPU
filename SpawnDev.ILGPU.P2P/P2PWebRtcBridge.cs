using SpawnDev.WebTorrent;

namespace SpawnDev.ILGPU.P2P;

/// <summary>
/// Bridges WebTorrent peer connections to the P2P compute system.
/// Registers sd_compute extension on each new peer, wires message routing.
///
/// Usage:
///   var bridge = new P2PWebRtcBridge(transport);
///   // Attach to a torrent — automatically wires sd_compute to every peer:
///   bridge.AttachToSwarm(torrent);
/// </summary>
public class P2PWebRtcBridge : IAsyncDisposable
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

    /// <summary>
    /// Fired when a peer's capabilities are received.
    /// </summary>
    public event Action<string, PeerCapabilities?>? OnComputePeerCapabilities;

    public P2PWebRtcBridge(P2PTransport transport)
    {
        _transport = transport;
    }

    /// <summary>
    /// Attach to a Torrent — automatically wires sd_compute
    /// to every peer that connects to this torrent.
    /// Uses the OnWire event (replaces old OnPeerConnect pattern).
    /// </summary>
    public void AttachToSwarm(Torrent torrent)
    {
        torrent.OnWire += (wire, peerId) =>
        {
            if (string.IsNullOrEmpty(peerId))
                peerId = Guid.NewGuid().ToString("N");

            // Check if the wire already has an SdComputeExtension (from UseExtension factory)
            var existing = wire.GetExtension<SdComputeExtension>();
            if (existing != null)
            {
                // Use the factory-created extension (already in BEP 10 handshake)
                existing.SetPeerId(peerId);
                _extensions[peerId] = existing;

                // Wire up the capability tracking for future messages
                existing.OnComputeMessage += (msg) =>
                {
                    if (msg.Type == P2PMessageType.CapabilityResponse && _notified.TryAdd(peerId, true))
                    {
                        PeerCapabilities? caps = null;
                        try
                        {
                            if (msg.Payload.HasValue)
                                caps = System.Text.Json.JsonSerializer.Deserialize<PeerCapabilities>(msg.Payload.Value);
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"[P2PBridge] Failed to deserialize capabilities from peer {peerId}: {ex.Message}");
                        }
                        OnComputePeerConnected?.Invoke(peerId);
                        OnComputePeerCapabilities?.Invoke(peerId, caps);
                    }
                };

                // Check if CapabilityResponse already arrived before we wired the handler
                if (existing.LastCapabilityResponse != null && _notified.TryAdd(peerId, true))
                {
                    PeerCapabilities? caps = null;
                    try
                    {
                        if (existing.LastCapabilityResponse.Payload.HasValue)
                            caps = System.Text.Json.JsonSerializer.Deserialize<PeerCapabilities>(
                                existing.LastCapabilityResponse.Payload.Value);
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"[P2PBridge] Failed to deserialize buffered capabilities from peer {peerId}: {ex.Message}");
                    }
                    OnComputePeerConnected?.Invoke(peerId);
                    OnComputePeerCapabilities?.Invoke(peerId, caps);
                }
            }
            else
            {
                // No factory extension — UseExtension was NOT called before the torrent was created.
                Console.WriteLine($"[P2PBridge] WARNING: No factory extension for peer {peerId}. " +
                    "UseExtension must be called BEFORE creating/joining the torrent. " +
                    "sd_compute will not work for this peer.");
            }
        };
    }

    private readonly System.Collections.Concurrent.ConcurrentDictionary<string, bool> _notified = new();

    /// <summary>
    /// Send a compute message to a specific peer via the real WebRTC channel.
    /// </summary>
    public async Task SendAsync(string peerId, P2PMessage message)
    {
        if (_extensions.TryGetValue(peerId, out var ext) && ext.IsSupported)
        {
            await ext.SendP2PMessageAsync(message);
        }
    }

    /// <summary>
    /// Check if a peer has sd_compute support.
    /// </summary>
    public bool IsComputeCapable(string peerId)
    {
        return _extensions.TryGetValue(peerId, out var ext) && ext.IsSupported;
    }

    /// <inheritdoc/>
    public ValueTask DisposeAsync()
    {
        _extensions.Clear();
        _notified.Clear();
        return ValueTask.CompletedTask;
    }
}
