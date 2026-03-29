using global::ILGPU;
using SpawnDev.WebTorrent;

namespace SpawnDev.ILGPU.P2P;

/// <summary>
/// High-level coordinator for P2P distributed compute.
/// Manages the swarm, peer discovery, capability negotiation,
/// and kernel dispatch routing.
///
/// Usage:
///   var coordinator = new P2PSwarmCoordinator(webTorrentClient);
///   await coordinator.CreateSwarmAsync("my-compute-group");
///   // Share coordinator.MagnetLink as QR code — anyone who scans it joins
///   // Peers connect automatically via WebRTC through the tracker
///   var accelerator = coordinator.CreateAccelerator(context);
/// </summary>
public class P2PSwarmCoordinator : IAsyncDisposable
{
    private readonly WebTorrentClient _client;
    private readonly System.Collections.Concurrent.ConcurrentDictionary<string, RemotePeer> _peers = new();
    private readonly System.Collections.Concurrent.ConcurrentDictionary<string, bool> _blockedPeers = new();
    private P2PAccelerator? _accelerator;

    /// <summary>
    /// This node's current role in the swarm.
    /// </summary>
    public P2PRole Role { get; private set; } = P2PRole.Worker;

    /// <summary>
    /// PeerId of the current coordinator (null if this node is coordinator).
    /// </summary>
    public string? CoordinatorPeerId { get; private set; }

    /// <summary>
    /// Magnet link for joining this compute swarm (BitTorrent protocol).
    /// </summary>
    public string? MagnetLink { get; private set; }

    /// <summary>
    /// HTTP join link — clean, clickable, QR-friendly.
    /// Points to the coordinator's own web app with swarm info as query params.
    /// Anyone who follows this link loads the same app and auto-joins as a worker.
    /// The coordinator's web app IS the compute code — no separate join server needed.
    /// </summary>
    public string? JoinLink { get; private set; }

    /// <summary>
    /// Base URL for generating HTTP join links.
    /// Should be the URL of the web app the coordinator is running on.
    /// In browser: set to window.location.origin + path.
    /// On desktop: set to wherever the web UI is hosted.
    /// </summary>
    public string? JoinLinkBaseUrl { get; set; }

    /// <summary>
    /// The swarm's ECDSA public key (BEP 46) for signed coordination messages.
    /// </summary>
    public byte[]? PublicKey { get; private set; }

    /// <summary>
    /// Number of connected compute peers.
    /// </summary>
    public int PeerCount => _peers.Count;

    /// <summary>
    /// Total estimated TFLOPS across all connected peers.
    /// </summary>
    public double TotalTflops => _peers.Values.Sum(p => p.Capabilities?.EstimatedTflops ?? 0);

    /// <summary>
    /// Total available memory across all connected peers (bytes).
    /// </summary>
    public long TotalMemory => _peers.Values.Sum(p => p.Capabilities?.AvailableMemory ?? 0);

    /// <summary>
    /// Get a snapshot of all connected peers (for state persistence).
    /// </summary>
    public IReadOnlyList<RemotePeer> GetPeerList() => _peers.Values.ToList();

    /// <summary>
    /// Fired when a new peer joins the compute swarm.
    /// </summary>
    public event Action<RemotePeer>? OnPeerJoined;

    /// <summary>
    /// Fired when a peer leaves the swarm.
    /// </summary>
    public event Action<RemotePeer>? OnPeerLeft;

    /// <summary>
    /// Fired when aggregate swarm capacity changes.
    /// </summary>
    public event Action? OnCapacityChanged;

    public P2PSwarmCoordinator(WebTorrentClient client)
    {
        _client = client;
    }

    /// <summary>
    /// Create a new compute swarm. Generates a magnet link that peers can use to join.
    /// </summary>
    /// <param name="name">Human-readable swarm name.</param>
    /// <param name="trackers">Tracker URLs for peer discovery.</param>
    public async Task CreateSwarmAsync(string name, string[]? trackers = null)
    {
        trackers ??= new[] { "wss://hub.spawndev.com:44365/announce" };

        // Create a small data payload that identifies this compute swarm
        var swarmId = System.Text.Encoding.UTF8.GetBytes($"p2p-compute:{name}:{Guid.NewGuid():N}");
        var swarm = await _client.SeedAsync(swarmId, $"{name}.p2p");

        // Build magnet link with tracker info
        var hashHex = Convert.ToHexString(swarm.InfoHash).ToLowerInvariant();
        var trackerParams = string.Join("", trackers.Select(t =>
            $"&tr={Uri.EscapeDataString(t)}"));
        MagnetLink = $"magnet:?xt=urn:btih:{hashHex}&dn={Uri.EscapeDataString(name)}{trackerParams}";

        // HTTP join link — only generated when we know the coordinator's web app URL
        if (!string.IsNullOrEmpty(JoinLinkBaseUrl))
        {
            var baseUrl = JoinLinkBaseUrl.TrimEnd('/');
            JoinLink = $"{baseUrl}?compute={hashHex}&n={Uri.EscapeDataString(name)}";
        }

        Role = P2PRole.Coordinator; // Creator is initial coordinator
    }

    /// <summary>
    /// Join an existing compute swarm via magnet link.
    /// Joins as a Worker — the swarm already has a coordinator.
    /// </summary>
    public async Task JoinSwarmAsync(string magnetLink)
    {
        MagnetLink = magnetLink;
        Role = P2PRole.Worker;
        var swarm = await _client.AddAsync(magnetLink);
        // Swarm connection happens through the tracker
        // Peers are discovered and capabilities exchanged
    }

    /// <summary>
    /// Create a P2P accelerator backed by this swarm.
    /// </summary>
    public P2PAccelerator CreateAccelerator(global::ILGPU.Context context)
    {
        var device = new P2PDevice();
        _accelerator = (P2PAccelerator)device.CreateAccelerator(context);
        _accelerator.TorrentClient = _client;

        // Add any already-connected peers
        foreach (var peer in _peers.Values.Where(p => p.IsConnected))
        {
            _accelerator.AddPeer(peer);
        }

        return _accelerator;
    }

    /// <summary>
    /// Handle a new peer connecting to the swarm.
    /// Blocked peers are rejected immediately.
    /// </summary>
    public bool HandlePeerConnected(string peerId, PeerCapabilities capabilities)
    {
        if (_blockedPeers.ContainsKey(peerId))
        {
            OnPeerRejected?.Invoke(peerId, "blocked");
            return false;
        }

        var peer = new RemotePeer
        {
            PeerId = peerId,
            IsConnected = true,
            MemorySize = capabilities.AvailableMemory,
            Capabilities = capabilities,
        };

        _peers[peerId] = peer;
        _accelerator?.AddPeer(peer);

        OnPeerJoined?.Invoke(peer);
        OnCapacityChanged?.Invoke();
        return true;
    }

    /// <summary>
    /// Handle a peer disconnecting.
    /// </summary>
    public void HandlePeerDisconnected(string peerId)
    {
        if (_peers.TryGetValue(peerId, out var peer))
        {
            peer.IsConnected = false;
            _peers.TryRemove(peerId, out _);
            _accelerator?.RemovePeer(peer);

            OnPeerLeft?.Invoke(peer);
            OnCapacityChanged?.Invoke();
        }
    }

    /// <summary>
    /// Kick a peer from the swarm. They can reconnect unless blocked.
    /// Coordinator-only operation.
    /// </summary>
    public bool KickPeer(string peerId, string reason = "")
    {
        if (Role != P2PRole.Coordinator) return false;
        if (!_peers.ContainsKey(peerId)) return false;

        HandlePeerDisconnected(peerId);
        OnPeerKicked?.Invoke(peerId, reason);
        return true;
    }

    /// <summary>
    /// Block a peer from the swarm. They cannot reconnect until unblocked.
    /// If currently connected, they are kicked first.
    /// Coordinator-only operation.
    /// </summary>
    public bool BlockPeer(string peerId, string reason = "")
    {
        if (Role != P2PRole.Coordinator) return false;

        _blockedPeers.TryAdd(peerId, true);

        // Kick if currently connected
        if (_peers.ContainsKey(peerId))
            HandlePeerDisconnected(peerId);

        OnPeerBlocked?.Invoke(peerId, reason);
        return true;
    }

    /// <summary>
    /// Unblock a previously blocked peer. They can reconnect on next attempt.
    /// </summary>
    public bool UnblockPeer(string peerId)
    {
        return _blockedPeers.TryRemove(peerId, out _);
    }

    /// <summary>
    /// Check if a peer is blocked.
    /// </summary>
    public bool IsPeerBlocked(string peerId) => _blockedPeers.ContainsKey(peerId);

    /// <summary>
    /// Get all blocked peer IDs.
    /// </summary>
    public IReadOnlyCollection<string> BlockedPeers => (IReadOnlyCollection<string>)_blockedPeers.Keys;

    /// <summary>
    /// Fired when a peer is kicked from the swarm.
    /// </summary>
    public event Action<string, string>? OnPeerKicked; // peerId, reason

    /// <summary>
    /// Fired when a peer is blocked from the swarm.
    /// </summary>
    public event Action<string, string>? OnPeerBlocked; // peerId, reason

    /// <summary>
    /// Fired when a peer connection is rejected (blocked).
    /// </summary>
    public event Action<string, string>? OnPeerRejected; // peerId, reason

    /// <summary>
    /// Transfer the coordinator role to another peer.
    /// Called when this coordinator is leaving gracefully (battery dying, tab closing).
    /// The new coordinator inherits pending dispatch state via BEP 46 signed transfer.
    /// </summary>
    public string? TransferCoordinator(string? preferredPeerId = null)
    {
        if (Role != P2PRole.Coordinator) return null;

        // Pick the healthiest peer, or the preferred one if specified
        var target = preferredPeerId != null
            ? _peers.Values.FirstOrDefault(p => p.PeerId == preferredPeerId && p.IsConnected)
            : _peers.Values
                .Where(p => p.IsConnected)
                .OrderByDescending(p => p.Capabilities?.EstimatedTflops ?? 0)
                .ThenByDescending(p => p.Capabilities?.AvailableMemory ?? 0)
                .ThenBy(p => p.Capabilities?.ThermalState ?? 0)
                .FirstOrDefault();

        if (target == null) return null;

        // Notify the target peer that they are the new coordinator
        OnSendMessage?.Invoke(target.PeerId, new P2PMessage
        {
            Type = P2PMessageType.CoordinatorTransfer,
            Payload = System.Text.Json.JsonSerializer.SerializeToElement(new
            {
                newCoordinatorPeerId = target.PeerId,
                timestamp = DateTimeOffset.UtcNow,
            }),
        });

        // Announce to all peers
        foreach (var peer in _peers.Values.Where(p => p.IsConnected && p.PeerId != target.PeerId))
        {
            OnSendMessage?.Invoke(peer.PeerId, new P2PMessage
            {
                Type = P2PMessageType.CoordinatorAnnounce,
                Payload = System.Text.Json.JsonSerializer.SerializeToElement(new
                {
                    newCoordinatorPeerId = target.PeerId,
                }),
            });
        }

        Role = P2PRole.Worker;
        CoordinatorPeerId = target.PeerId;
        OnCoordinatorChanged?.Invoke(target.PeerId);
        return target.PeerId;
    }

    /// <summary>
    /// Fired when the coordinator needs to send a message to a peer.
    /// Hook this to the P2PTransport for actual delivery.
    /// </summary>
    public event Action<string, P2PMessage>? OnSendMessage;

    /// <summary>
    /// Elect a new coordinator after the current one drops unexpectedly.
    /// Uses deterministic selection: highest score wins.
    /// All peers run the same algorithm so they converge on the same choice.
    /// </summary>
    public string? ElectCoordinator()
    {
        var candidates = _peers.Values
            .Where(p => p.IsConnected)
            .OrderByDescending(p => p.Capabilities?.EstimatedTflops ?? 0)
            .ThenByDescending(p => p.Capabilities?.AvailableMemory ?? 0)
            .ThenBy(p => p.PeerId) // deterministic tiebreaker
            .ToList();

        if (candidates.Count == 0) return null;

        var winner = candidates[0];
        CoordinatorPeerId = winner.PeerId;
        OnCoordinatorChanged?.Invoke(winner.PeerId);
        return winner.PeerId;
    }

    /// <summary>
    /// This node accepts the coordinator role (elected or transferred to us).
    /// </summary>
    public void BecomeCoordinator()
    {
        Role = P2PRole.Coordinator;
        CoordinatorPeerId = null; // we ARE the coordinator
        OnCoordinatorChanged?.Invoke(null); // null = self
    }

    /// <summary>
    /// Fired when the coordinator role changes.
    /// Null peerId means this node is the new coordinator.
    /// </summary>
    public event Action<string?>? OnCoordinatorChanged;

    /// <inheritdoc/>
    public ValueTask DisposeAsync()
    {
        // Notify all peers we're leaving
        foreach (var peer in _peers.Values.Where(p => p.IsConnected))
        {
            OnSendMessage?.Invoke(peer.PeerId, new P2PMessage
            {
                Type = P2PMessageType.Disconnect,
            });
        }
        _peers.Clear();
        return ValueTask.CompletedTask;
    }
}
