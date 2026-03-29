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
    /// Recovers state from DHT after coordinator loss.
    /// </summary>
    public async Task SubscribeAsync(byte[] swarmPublicKey)
    {
        await _channel.SubscribeAsync(swarmPublicKey);
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
        return new SwarmState
        {
            SwarmName = "compute-swarm", // TODO: from coordinator config
            CoordinatorPeerId = _coordinator.CoordinatorPeerId,
            Timestamp = DateTimeOffset.UtcNow,
            PeerCount = _coordinator.PeerCount,
            TotalTflops = _coordinator.TotalTflops,
            TotalMemory = _coordinator.TotalMemory,
        };
    }

    private void HandleStateUpdate(byte[] publicKey, byte[] value, long sequence)
    {
        try
        {
            var state = JsonSerializer.Deserialize<SwarmState>(value, _jsonOptions);
            if (state != null)
                OnStateUpdated?.Invoke(state);
        }
        catch { }
    }

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
    public string SwarmName { get; set; } = "";
    public string? CoordinatorPeerId { get; set; }
    public DateTimeOffset Timestamp { get; set; }
    public int PeerCount { get; set; }
    public double TotalTflops { get; set; }
    public long TotalMemory { get; set; }
}

/// <summary>
/// Coordinator election announcement published to BEP 46 DHT.
/// </summary>
public class CoordinatorAnnouncement
{
    public string CoordinatorPeerId { get; set; } = "";
    public DateTimeOffset Timestamp { get; set; }
}
