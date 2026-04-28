using System.Text.Json;
using SpawnDev.WebTorrent;

namespace SpawnDev.ILGPU.P2P;

/// <summary>
/// Manages persistent swarm state via BEP 46 DHT mutable items.
/// State survives coordinator loss — the DHT is the persistent brain.
///
/// Published state includes:
///   - Peer table (who's in the swarm, their capabilities)
///   - Current coordinator identity
///   - Active dispatches and their assignments
///   - Swarm configuration (name, purpose, scoring weights)
///
/// Any peer can read the state from the DHT by the swarm's public key.
/// Only the coordinator (who holds the private key) can write.
/// On coordinator transfer, the signing key is transferred via BEP 46.
/// </summary>
public class P2PStateManager
{
    private readonly AgentChannel _channel;
    private readonly P2PSwarmCoordinator _coordinator;

    /// <summary>
    /// The swarm's public key identity (from BEP 46 signing key).
    /// </summary>
    public byte[] PublicKey => _channel.PublicKey;

    /// <summary>
    /// Hex string of public key (for sharing in join links).
    /// </summary>
    public string PublicKeyHex => _channel.PublicKeyHex;

    /// <summary>
    /// Current sequence number for published state.
    /// </summary>
    public long Sequence => _channel.Sequence;

    /// <summary>
    /// Fired when swarm state is updated from the DHT
    /// (e.g., after coordinator transfer or recovery).
    /// </summary>
    public event Action<SwarmState>? OnStateUpdated;

    public P2PStateManager(AgentChannel channel, P2PSwarmCoordinator coordinator)
    {
        _channel = channel;
        _coordinator = coordinator;

        _channel.OnAgentUpdate += HandleStateUpdate;
    }

    /// <summary>
    /// Publish current swarm state to the DHT (coordinator only).
    /// Called after peer joins/leaves, dispatch changes, or coordinator transfer.
    /// </summary>
    public async Task PublishStateAsync()
    {
        if (_coordinator.Role != P2PRole.Coordinator) return;

        var state = BuildCurrentState();
        var json = JsonSerializer.SerializeToUtf8Bytes(state, _jsonOptions);
        await _channel.PublishStateAsync(json);
    }

    /// <summary>
    /// Subscribe to state updates for this swarm (workers call this).
    /// Recovers state from DHT after coordinator loss. <paramref name="pollIntervalMs"/>
    /// controls how often the underlying BEP 46 subscription re-queries the DHT for
    /// updated values; defaults to 30s which matches BEP 46 expected behaviour but is
    /// overridable for tests or for latency-sensitive paths.
    /// </summary>
    public async Task SubscribeAsync(byte[] swarmPublicKey, int pollIntervalMs = 30000)
    {
        await _channel.SubscribeAsync(swarmPublicKey, pollIntervalMs: pollIntervalMs);
    }

    /// <summary>
    /// Publish a coordinator election announcement.
    /// The new coordinator signs "I am now the coordinator" with the swarm key.
    /// </summary>
    public async Task AnnounceCoordinatorAsync(string newCoordinatorPeerId)
    {
        var announcement = new CoordinatorAnnouncement
        {
            CoordinatorPeerId = newCoordinatorPeerId,
            Timestamp = DateTimeOffset.UtcNow,
        };
        var json = JsonSerializer.SerializeToUtf8Bytes(announcement, _jsonOptions);
        await _channel.PublishStateAsync(json);
    }

    private SwarmState BuildCurrentState()
    {
        // Include full peer table so all nodes have the same view for elections
        var peers = _coordinator.GetPeerList().Select(p => new PeerInfo
        {
            PeerId = p.PeerId,
            EstimatedTflops = p.Capabilities?.EstimatedTflops ?? 0,
            AvailableMemory = p.Capabilities?.AvailableMemory ?? 0,
            IsConnected = p.IsConnected,
        }).ToList();

        // Resolve a stable identifier for "who is the coordinator" in the published state.
        // Internally CoordinatorPeerId is null when self-is-coordinator (by design - the
        // field names a REMOTE peer), but a published state consumed by workers needs SOME
        // identifier. Use the coord's SwarmIdentity fingerprint (SHA-256 of its Ed25519
        // public key) - stable across reconnects, same identity every time this node is
        // the coordinator, survives WebTorrent peer-id rotation.
        var coordId = !string.IsNullOrEmpty(_coordinator.CoordinatorPeerId)
            ? _coordinator.CoordinatorPeerId
            : _coordinator.Role == P2PRole.Coordinator
                ? _coordinator.Identity?.Fingerprint ?? ""
                : "";

        return new SwarmState
        {
            SwarmName = "compute-swarm",
            CoordinatorPeerId = coordId,
            Timestamp = DateTimeOffset.UtcNow,
            PeerCount = _coordinator.PeerCount,
            TotalTflops = _coordinator.TotalTflops,
            TotalMemory = _coordinator.TotalMemory,
            Peers = peers,
        };
    }

    private void HandleStateUpdate(byte[] publicKey, byte[] value, long sequence)
    {
        try
        {
            // Check discriminator to determine message type
            var doc = JsonDocument.Parse(value);
            var type = doc.RootElement.TryGetProperty("type", out var typeProp)
                ? typeProp.GetString() : null;

            if (type == "SwarmState")
            {
                var state = JsonSerializer.Deserialize<SwarmState>(value, _jsonOptions);
                if (state != null)
                    OnStateUpdated?.Invoke(state);
            }
            else if (type == "CoordinatorAnnouncement")
            {
                var announcement = JsonSerializer.Deserialize<CoordinatorAnnouncement>(value, _jsonOptions);
                if (announcement != null)
                    OnCoordinatorAnnounced?.Invoke(announcement);
            }
        }
        catch (Exception ex)
        {
            if (P2PCompute.VerboseLogging) Console.WriteLine($"[P2PState] Failed to process state update: {ex.Message}");
        }
    }

    /// <summary>
    /// Fired when a coordinator announcement is received from the DHT.
    /// </summary>
    public event Action<CoordinatorAnnouncement>? OnCoordinatorAnnounced;

    private static readonly JsonSerializerOptions _jsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
    };
}

/// <summary>
/// Serializable swarm state published to BEP 46 DHT.
/// </summary>
public class SwarmState
{
    /// <summary>Type discriminator — prevents deserialization confusion with other message types on same DHT key.</summary>
    public string Type { get; set; } = "SwarmState";
    public string SwarmName { get; set; } = "";
    public string? CoordinatorPeerId { get; set; }
    public DateTimeOffset Timestamp { get; set; }
    public int PeerCount { get; set; }
    public double TotalTflops { get; set; }
    public long TotalMemory { get; set; }
    /// <summary>Full peer table for election consistency (prevents split-brain).</summary>
    public List<PeerInfo> Peers { get; set; } = new();
}

/// <summary>
/// Peer info stored in swarm state for consistent election.
/// </summary>
public class PeerInfo
{
    public string PeerId { get; set; } = "";
    public double EstimatedTflops { get; set; }
    public long AvailableMemory { get; set; }
    public bool IsConnected { get; set; }
}

/// <summary>
/// Coordinator election announcement published to BEP 46 DHT.
/// </summary>
public class CoordinatorAnnouncement
{
    /// <summary>Type discriminator.</summary>
    public string Type { get; set; } = "CoordinatorAnnouncement";
    public string CoordinatorPeerId { get; set; } = "";
    public DateTimeOffset Timestamp { get; set; }
}
