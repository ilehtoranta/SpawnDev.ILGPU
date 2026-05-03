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
    /// The underlying WebTorrent torrent/swarm (for bridge attachment).
    /// </summary>
    public Torrent? Swarm { get; private set; }

    /// <summary>
    /// This node's cryptographic identity (Ed25519 key pair).
    /// Null until CreateSwarmAsync or SetIdentity is called.
    /// </summary>
    public SwarmIdentity? Identity { get; private set; }

    /// <summary>
    /// Swarm access policy (join permissions, limits).
    /// </summary>
    public SwarmPolicy Policy { get; set; } = new();

    /// <summary>
    /// Set of peer fingerprints that have been seen before (for KnownOnly mode).
    /// </summary>
    private readonly System.Collections.Concurrent.ConcurrentDictionary<string, bool> _knownPeers = new();

    /// <summary>
    /// Peers waiting for owner approval (Approval mode).
    /// </summary>
    public System.Collections.Concurrent.ConcurrentDictionary<string, RemotePeer> PendingApproval { get; } = new();

    /// <summary>
    /// The swarm's key registry (authorized keys + roles).
    /// </summary>
    public KeyRegistry? Registry { get; private set; }

    /// <summary>
    /// Set the cryptographic identity for this node.
    /// Call before CreateSwarmAsync for owner identity, or after JoinSwarmAsync for worker identity.
    /// </summary>
    public void SetIdentity(SwarmIdentity identity)
    {
        Identity = identity;
    }

    /// <summary>
    /// This node's current role in the swarm.
    /// </summary>
    public P2PRole Role { get; internal set; } = P2PRole.Worker;

    /// <summary>
    /// PeerId of the current coordinator (null if this node is coordinator).
    /// </summary>
    public string? CoordinatorPeerId { get; internal set; }

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
    /// The swarm's Ed25519 public key (BEP 46) for signed coordination messages.
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
    /// Saturates at <see cref="long.MaxValue"/> instead of throwing on overflow -
    /// .NET 10's <c>IEnumerable&lt;long&gt;.Sum</c> uses checked arithmetic and will
    /// throw <see cref="OverflowException"/> when a peer reports an extreme value
    /// (e.g. a sentinel representing "unknown/unlimited"). Saturation keeps the
    /// state-publish path alive and gives downstream aggregators a sensible ceiling.
    /// </summary>
    public long TotalMemory
    {
        get
        {
            long total = 0;
            foreach (var peer in _peers.Values)
            {
                var mem = peer.Capabilities?.AvailableMemory ?? 0;
                if (mem <= 0) continue;
                // Saturate at long.MaxValue - defensive, since no realistic sum of peer
                // memory should overflow Int64 (would require ~9 exabytes combined).
                if (mem > long.MaxValue - total) return long.MaxValue;
                total += mem;
            }
            return total;
        }
    }

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
        var createOptions = new TorrentCreatorOptions
        {
            Trackers = trackers,
        };
        var torrent = await _client.SeedAsync($"{name}.p2p", swarmId, createOptions);
        Swarm = torrent;

        // Build magnet link with tracker info (InfoHash is already hex string in _Alt)
        var hashHex = torrent.InfoHash ?? "";
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

        // Initialize key registry with owner's identity (if set)
        if (Identity != null)
        {
            Registry = new KeyRegistry();
            Registry.AddKey(Identity.PublicKeySpki, SwarmRole.Owner, Identity.Label);
        }
    }

    /// <summary>
    /// Join an existing compute swarm via magnet link.
    /// Joins as a Worker — the swarm already has a coordinator.
    /// </summary>
    public Task JoinSwarmAsync(string magnetLink)
    {
        MagnetLink = magnetLink;
        Role = P2PRole.Worker;
        var torrent = _client.Add(magnetLink);
        Swarm = torrent;
        // Swarm connection happens through the tracker
        // Peers are discovered and capabilities exchanged
        return Task.CompletedTask;
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
        // Dedup: a single logical remote peer can fire HandlePeerConnected multiple
        // times in steady state because the BT-layer DUP-OBSERVED branch (rc.1)
        // intentionally keeps every duplicate wire alive (Torrent.cs:949+), and
        // each wire's sd_compute extension fires its own capability-response
        // event. Without this guard, the "Peer joined" UI activity log gets a
        // burst of N entries (one per duplicate wire) every time a peer connects,
        // each with potentially-different trust labels because the DHT identity
        // arrives async relative to the wire's first handshake — early handshakes
        // see pre-identity capabilities, later ones see post-identity. Symptom on
        // the coord UI is e.g. 13 "Peer joined" events at the same timestamp
        // with shifting trust=Anonymous/Identified labels (Captain's 2026-05-03
        // RenderMandelbrot live repro on rc.7).
        //
        // If we already have this peer registered, this is a duplicate-wire
        // arrival, not a new join. Update the latest capabilities (so trust
        // labels and TFLOPS converge to the most recent values) and quietly
        // return success without firing OnPeerJoined a second time. Real
        // re-joins after disconnect happen via Wire.OnClose -> _peers removal,
        // so a fresh ContainsKey check after that path naturally fires the
        // re-join.
        if (_peers.TryGetValue(peerId, out var existing))
        {
            existing.Capabilities = capabilities;
            existing.MemorySize = capabilities.AvailableMemory;
            existing.IsConnected = true;
            return true;
        }

        // Check block list
        if (_blockedPeers.ContainsKey(peerId))
        {
            OnPeerRejected?.Invoke(peerId, "blocked");
            return false;
        }

        // Check peer limit
        if (Policy.MaxPeers > 0 && _peers.Count >= Policy.MaxPeers)
        {
            OnPeerRejected?.Invoke(peerId, "swarm full");
            return false;
        }

        // Check TFLOPS limit
        if (Policy.MaxTflops > 0 && TotalTflops >= Policy.MaxTflops)
        {
            OnPeerRejected?.Invoke(peerId, "TFLOPS limit reached");
            return false;
        }

        // Identity check: use cryptographic fingerprint if available, fall back to peerId
        var fingerprint = capabilities.Fingerprint ?? "";
        var isAnonymous = string.IsNullOrEmpty(capabilities.PublicKey);

        // Anonymous peer check
        if (!Policy.AllowAnonymous && isAnonymous)
        {
            OnPeerRejected?.Invoke(peerId, "anonymous peers not allowed");
            return false;
        }

        // Identity key for known-peer lookups: fingerprint if available, else peerId
        var identityKey = !string.IsNullOrEmpty(fingerprint) ? fingerprint : peerId;

        // Check join mode
        switch (Policy.JoinPermission)
        {
            case JoinMode.InviteOnly:
                // Must have key in registry (requires non-anonymous)
                if (isAnonymous || Registry == null ||
                    !Registry.HasRole(capabilities.PublicKey!, SwarmRole.Worker))
                {
                    OnPeerRejected?.Invoke(peerId, "invite only");
                    return false;
                }
                break;

            case JoinMode.KnownOnly:
                if (!_knownPeers.ContainsKey(identityKey))
                {
                    OnPeerRejected?.Invoke(peerId, "unknown device");
                    return false;
                }
                break;

            case JoinMode.Approval:
                if (!_knownPeers.ContainsKey(identityKey))
                {
                    // Add to pending — owner must approve
                    var pendingPeer = new RemotePeer
                    {
                        PeerId = peerId,
                        IsConnected = true,
                        MemorySize = capabilities.AvailableMemory,
                        Capabilities = capabilities,
                    };
                    PendingApproval[peerId] = pendingPeer;
                    OnPeerPendingApproval?.Invoke(pendingPeer);
                    return false; // Not admitted yet
                }
                break;

            // JoinMode.Open — always admit
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

        // Remember this peer for future joins (by fingerprint if available, else peerId)
        if (Policy.RememberPeers)
            _knownPeers.TryAdd(identityKey, true);

        OnPeerJoined?.Invoke(peer);
        OnCapacityChanged?.Invoke();
        return true;
    }

    /// <summary>
    /// Approve a pending peer (Approval mode).
    /// Moves them from pending to active.
    /// Requires Coordinator or higher authority.
    /// </summary>
    public bool ApprovePeer(string peerId)
    {
        if (!HasLocalAuthority(SwarmRole.Coordinator)) return false;
        if (!PendingApproval.TryRemove(peerId, out var pending)) return false;

        _knownPeers.TryAdd(peerId, true);
        _peers[peerId] = pending;
        _accelerator?.AddPeer(pending);

        OnPeerJoined?.Invoke(pending);
        OnCapacityChanged?.Invoke();
        return true;
    }

    /// <summary>
    /// Reject a pending peer (Approval mode).
    /// </summary>
    public bool RejectPendingPeer(string peerId, string reason = "")
    {
        if (!PendingApproval.TryRemove(peerId, out _)) return false;
        OnPeerRejected?.Invoke(peerId, reason);
        return true;
    }

    /// <summary>
    /// Fired when a peer is waiting for approval (Approval mode).
    /// </summary>
    public event Action<RemotePeer>? OnPeerPendingApproval;

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
    /// Check if the local identity has at least the given role in the registry.
    /// Falls back to P2PRole check for backward compatibility with open swarms.
    /// </summary>
    private bool HasLocalAuthority(SwarmRole minimumRole)
    {
        // If we have a registry and identity, use RBAC
        if (Registry != null && Identity != null)
        {
            var pubKey = Convert.ToBase64String(Identity.PublicKeySpki);
            return Registry.HasRole(pubKey, minimumRole);
        }
        // Fallback: coordinator role implies Coordinator authority, but not Owner
        if (minimumRole <= SwarmRole.Coordinator)
            return Role == P2PRole.Coordinator;
        return false;
    }

    /// <summary>
    /// Kick a peer from the swarm. They can reconnect unless blocked.
    /// Requires Coordinator or higher authority in the registry.
    /// </summary>
    public bool KickPeer(string peerId, string reason = "")
    {
        if (!HasLocalAuthority(SwarmRole.Coordinator)) return false;
        if (!_peers.ContainsKey(peerId)) return false;

        HandlePeerDisconnected(peerId);
        OnPeerKicked?.Invoke(peerId, reason);
        return true;
    }

    /// <summary>
    /// Block a peer from the swarm. They cannot reconnect until unblocked.
    /// If currently connected, they are kicked first.
    /// Requires Coordinator or higher authority in the registry.
    /// </summary>
    public bool BlockPeer(string peerId, string reason = "")
    {
        if (!HasLocalAuthority(SwarmRole.Coordinator)) return false;

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
    /// Requires Coordinator or higher authority (Owner/Admin can always transfer).
    /// </summary>
    public string? TransferCoordinator(string? preferredPeerId = null)
    {
        if (!HasLocalAuthority(SwarmRole.Coordinator)) return null;

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

        // Get pending dispatch state from the accelerator's dispatcher
        PendingDispatchInfo[]? pendingState = null;
        if (_accelerator?.Dispatcher != null)
            pendingState = _accelerator.Dispatcher.GetPendingSnapshot();

        // Notify the target peer that they are the new coordinator (with pending state)
        OnSendMessage?.Invoke(target.PeerId, new P2PMessage
        {
            Type = P2PMessageType.CoordinatorTransfer,
            Payload = System.Text.Json.JsonSerializer.SerializeToElement(new CoordinatorTransferData
            {
                NewCoordinatorPeerId = target.PeerId,
                Timestamp = DateTimeOffset.UtcNow,
                PendingDispatches = pendingState,
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
    /// Prefers registry-assigned coordinators (Owner > Admin > Coordinator).
    /// Falls back to TFLOPS-based election if no registry assignments match.
    /// All peers run the same algorithm so they converge on the same choice.
    /// </summary>
    public string? ElectCoordinator()
    {
        var connected = _peers.Values.Where(p => p.IsConnected).ToList();
        if (connected.Count == 0) return null;

        RemotePeer? winner = null;

        // Phase 1: Prefer registry-assigned peers (highest role first)
        if (Registry != null)
        {
            winner = connected
                .Where(p => !string.IsNullOrEmpty(p.Capabilities?.PublicKey) &&
                            Registry.HasRole(p.Capabilities!.PublicKey!, SwarmRole.Coordinator))
                .OrderByDescending(p => GetRegistryRole(p))
                .ThenByDescending(p => p.Capabilities?.EstimatedTflops ?? 0)
                .ThenBy(p => p.PeerId)
                .FirstOrDefault();
        }

        // Phase 2: Fallback to TFLOPS-based election
        winner ??= connected
            .OrderByDescending(p => p.Capabilities?.EstimatedTflops ?? 0)
            .ThenByDescending(p => p.Capabilities?.AvailableMemory ?? 0)
            .ThenBy(p => p.PeerId)
            .First();

        CoordinatorPeerId = winner.PeerId;
        OnCoordinatorChanged?.Invoke(winner.PeerId);
        return winner.PeerId;
    }

    /// <summary>
    /// Get a peer's role from the registry (for election ordering).
    /// </summary>
    private SwarmRole GetRegistryRole(RemotePeer peer)
    {
        if (Registry == null || string.IsNullOrEmpty(peer.Capabilities?.PublicKey))
            return SwarmRole.Worker;
        var key = Registry.Keys.FirstOrDefault(k => k.PublicKey == peer.Capabilities!.PublicKey);
        return key?.Role ?? SwarmRole.Worker;
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
    /// Assign a role to a peer. Creates a signed RoleAssignment and updates the registry.
    /// Requires the granter to have a higher role than the one being assigned.
    /// </summary>
    /// <param name="peerId">The peer to assign the role to.</param>
    /// <param name="peerPublicKeySpki">The peer's public key (SPKI bytes).</param>
    /// <param name="role">The role to assign.</param>
    /// <returns>The signed assignment, or null if authority check fails.</returns>
    public async Task<RoleAssignment?> AssignRoleAsync(
        string peerId, byte[] peerPublicKeySpki, SwarmRole role)
    {
        if (Identity == null || Registry == null) return null;

        // Granter must have higher authority than the role being assigned
        var granterKey = Convert.ToBase64String(Identity.PublicKeySpki);
        var granterEntry = Registry.Keys.FirstOrDefault(k => k.PublicKey == granterKey);
        if (granterEntry == null || granterEntry.Role <= role) return null;

        // Create signed assignment
        var assignment = await RoleAssignment.CreateAsync(
            Identity, peerId, peerPublicKeySpki, role);

        // Update registry
        Registry.AddKey(peerPublicKeySpki, role, peerId);
        await Registry.SignAsync(Identity);

        return assignment;
    }

    /// <summary>
    /// Revoke a peer's key from the registry. Requires Owner authority.
    /// </summary>
    /// <param name="publicKeySpki">The key to revoke.</param>
    /// <param name="reason">Reason for revocation.</param>
    /// <returns>True if revoked, false if denied (last owner, or insufficient authority).</returns>
    public async Task<bool> RevokeKeyAsync(byte[] publicKeySpki, string reason = "")
    {
        if (Identity == null || Registry == null) return false;
        if (!HasLocalAuthority(SwarmRole.Owner)) return false;

        if (!Registry.RevokeKey(publicKeySpki, reason)) return false;

        await Registry.SignAsync(Identity);
        return true;
    }

    /// <summary>
    /// Get the current registry for distribution to peers.
    /// </summary>
    public KeyRegistry? GetRegistry() => Registry;

    /// <summary>
    /// Update the registry (received from owner or another coordinator).
    /// Only accepts registries with a higher sequence number than the current one.
    /// </summary>
    public void UpdateRegistry(KeyRegistry newRegistry)
    {
        if (Registry == null || newRegistry.Sequence > Registry.Sequence)
        {
            Registry = newRegistry;
            OnRegistryUpdated?.Invoke(newRegistry);
        }
    }

    /// <summary>
    /// Fired when the key registry is updated.
    /// </summary>
    public event Action<KeyRegistry>? OnRegistryUpdated;

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
