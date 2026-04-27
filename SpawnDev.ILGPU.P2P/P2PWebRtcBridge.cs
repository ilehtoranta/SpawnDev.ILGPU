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
    /// Active wires grouped by canonical (remote BitTorrent) peer ID. Two wires from
    /// the same remote during a handshake-race duplicate scenario share the same
    /// `wire.PeerId`, so we can dedupe at the bridge layer and present ONE peer to
    /// the dispatcher regardless of how many transient wires the WebRTC layer churns
    /// through. When the set for a canonical ID goes empty, the peer is fully gone
    /// and UnregisterPeer fires.
    /// </summary>
    private readonly System.Collections.Concurrent.ConcurrentDictionary<string, HashSet<SpawnDev.WebTorrent.Wire>> _wiresByBtPeerId = new();
    private readonly object _wiresByBtPeerIdLock = new();

    /// <summary>
    /// Reverse map: wire reference -> the canonical peer id it was registered under in
    /// NotifyCanonical. Captured at registration time so wire.OnClose has the correct
    /// id even if wire.PeerId is cleared by the destroy path.
    /// </summary>
    private readonly System.Collections.Concurrent.ConcurrentDictionary<SpawnDev.WebTorrent.Wire, string> _wireToCanonical = new();

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

            // Resolve the canonical (cross-wire-stable) peer key. After BEP-10 handshake
            // wire.PeerId is set to the remote BitTorrent peerId, which two wires from the
            // same remote (handshake-race duplicate scenario) share. Falls back to the
            // per-wire generated id only until handshake completes.
            string CanonicalPeerId() =>
                !string.IsNullOrEmpty(wire.PeerId) ? wire.PeerId! : peerId;

            // Wire-close detection: when a peer's underlying transport dies (tab closed,
            // network drop, browser navigated away, etc.), we remove that wire from the
            // bridge's tracking. UnregisterPeer fires only when EVERY wire for the
            // canonical peer is gone - the rc.21+ handshake-race duplicate-destroy from
            // SpawnDev.WebTorrent.Torrent.OnHandshake closes the loser wire while the
            // surviving wire is still live; we must not let that close path mistakenly
            // surface as a peer-departure to the dispatcher.
            wire.OnClose += () =>
            {
                _extensions.TryRemove(peerId, out _);
                _wireToCanonical.TryRemove(wire, out var canonical);
                // Try every plausible identifier this wire could have been registered
                // under. UnregisterPeer is idempotent (TryRemove no-ops on missing keys)
                // so this is safe. Defensive coverage closes the phantom-peer leak where
                // a registration path used one id and the close path resolves a different
                // id - which happens during the BEP-10 vs JSON-CapabilityResponse
                // handshake-race window.
                var ids = new HashSet<string>(StringComparer.Ordinal);
                if (!string.IsNullOrEmpty(canonical)) ids.Add(canonical!);
                if (!string.IsNullOrEmpty(wire.PeerId)) ids.Add(wire.PeerId!);
                if (!string.IsNullOrEmpty(peerId)) ids.Add(peerId);
                foreach (var id in ids)
                {
                    _notified.TryRemove(id, out _);
                    _transport.UnregisterPeer(id);
                }

                // Bookkeeping: remove this wire from canonical-to-wires map for completeness.
                if (!string.IsNullOrEmpty(canonical))
                {
                    lock (_wiresByBtPeerIdLock)
                    {
                        if (_wiresByBtPeerId.TryGetValue(canonical!, out var wireSet))
                        {
                            wireSet.Remove(wire);
                            if (wireSet.Count == 0)
                                _wiresByBtPeerId.TryRemove(canonical!, out _);
                        }
                    }
                }
            };

            // Check if the wire already has an SdComputeExtension (from UseExtension factory)
            var existing = wire.GetExtension<SdComputeExtension>();
            if (existing != null)
            {
                // Do NOT pre-seed the extension's _peerId with the OnWire-time per-wire id.
                // SdComputeExtension.OnHandshake will set _peerId to the canonical (remote
                // BitTorrent) peerId once BEP-10 handshake completes; that is the same id
                // both sides see and the same id the bridge uses for OnComputePeerCapabilities.
                // Pre-seeding here would lock the extension to the per-wire id, which causes
                // the extension's transport.RegisterPeer call to register a different peerId
                // than the bridge surfaces - so HandlePeerConnected fires twice (once via the
                // bridge with canonical id, once via P2PTransport.HandleCapabilityResponse with
                // the per-wire id) and the dispatcher's _peers ends up with two entries for
                // one physical worker.
                _extensions[peerId] = existing;

                // Helper that registers the wire under the canonical peer-id and fires
                // the bridge events ONCE per canonical peer regardless of how many wires
                // a handshake race produces.
                void NotifyCanonical(PeerCapabilities? caps)
                {
                    var canonical = CanonicalPeerId();
                    // Capture both directions of the wire <-> canonical mapping so OnClose
                    // can resolve the correct canonical id even after the wire's destroy
                    // clears wire.PeerId.
                    _wireToCanonical[wire] = canonical;
                    lock (_wiresByBtPeerIdLock)
                    {
                        var wireSet = _wiresByBtPeerId.GetOrAdd(canonical, _ => new HashSet<SpawnDev.WebTorrent.Wire>());
                        wireSet.Add(wire);
                    }
                    if (_notified.TryAdd(canonical, true))
                    {
                        OnComputePeerConnected?.Invoke(canonical);
                        OnComputePeerCapabilities?.Invoke(canonical, caps);
                    }
                }

                // Wire up the capability tracking for future messages
                existing.OnComputeMessage += (msg) =>
                {
                    if (msg.Type == P2PMessageType.CapabilityResponse)
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
                        NotifyCanonical(caps);
                    }
                };

                // Check if CapabilityResponse already arrived before we wired the handler
                if (existing.LastCapabilityResponse != null)
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
                    NotifyCanonical(caps);
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
        _wireToCanonical.Clear();
        lock (_wiresByBtPeerIdLock)
        {
            _wiresByBtPeerId.Clear();
        }
        return ValueTask.CompletedTask;
    }
}
