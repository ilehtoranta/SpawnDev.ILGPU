using System.Text.Json;
using SpawnDev.WebTorrent.Wire;

namespace SpawnDev.ILGPU.P2P;

/// <summary>
/// BEP 10 wire protocol extension for P2P distributed compute.
/// Registered as "sd_compute" in the extension handshake.
///
/// When two WebTorrent peers both support sd_compute, they can exchange
/// P2P compute messages (capability exchange, kernel dispatch, buffer transfer)
/// over the same WebRTC data channel used for BitTorrent piece exchange.
///
/// Usage:
///   var ext = new SdComputeExtension(transport);
///   wire.Extensions.Register(ext);
///   // After handshake, ext.IsSupported tells you if peer has compute capability
/// </summary>
public class SdComputeExtension : WireExtension
{
    private readonly P2PTransport _transport;
    private string _peerId = "";

    /// <summary>
    /// Extension name in BEP 10 handshake.
    /// </summary>
    public override string Name => "sd_compute";

    /// <summary>
    /// Fired when a compute message is received from this peer.
    /// </summary>
    public event Action<P2PMessage>? OnComputeMessage;

    /// <summary>
    /// Last CapabilityResponse received from this peer (buffered for late subscribers).
    /// </summary>
    public P2PMessage? LastCapabilityResponse { get; private set; }

    public SdComputeExtension(P2PTransport transport)
    {
        _transport = transport;
    }

    /// <summary>
    /// Set the peer ID for this connection (from the wire handshake).
    /// </summary>
    public void SetPeerId(string peerId)
    {
        _peerId = peerId;
    }

    /// <summary>
    /// Handle incoming sd_compute extension message.
    /// The payload is a P2P protocol message (JSON).
    /// </summary>
    public override async Task HandleMessageAsync(byte[] payload)
    {
        Console.WriteLine($"[sd_compute] HandleMessageAsync: peerId={_peerId}, {payload.Length} bytes, IsSupported={IsSupported}, RemoteId={RemoteId}");
        // Route to the P2P transport for handling
        await _transport.HandleIncomingDataAsync(_peerId, payload);

        // Also fire local event + buffer capabilities
        var message = P2PProtocol.Deserialize(payload);
        if (message != null)
        {
            if (message.Type == P2PMessageType.CapabilityResponse)
                LastCapabilityResponse = message;
            OnComputeMessage?.Invoke(message);
        }
    }

    /// <summary>
    /// Include compute capabilities in the extension handshake.
    /// </summary>
    public override Dictionary<string, object>? GetHandshakeData()
    {
        return new Dictionary<string, object>
        {
            ["sd_compute_version"] = P2PProtocol.Version,
        };
    }

    /// <summary>
    /// Process the peer's compute extension handshake data.
    /// </summary>
    public override void ProcessHandshakeData(Dictionary<string, object> handshake)
    {
        Console.WriteLine($"[sd_compute] ProcessHandshakeData: peerId={_peerId}, keys=[{string.Join(",", handshake.Keys)}], IsSupported={IsSupported}, RemoteId={RemoteId}");
        // Check if peer announced sd_compute support
        if (handshake.TryGetValue("sd_compute_version", out var version))
        {
            // Peer supports compute — register in transport
            _transport.RegisterPeer(_peerId, async (data) =>
            {
                // Send via the wire protocol's extension message system
                await SendComputeMessageAsync(data);
            });

            // Initiate capability exchange — send our capabilities to the peer
            _ = Task.Run(async () =>
            {
                try
                {
                    var capMsg = new P2PMessage
                    {
                        Type = P2PMessageType.CapabilityResponse,
                        Payload = System.Text.Json.JsonSerializer.SerializeToElement(
                            _transport.GetLocalCapabilities()),
                    };
                    await SendP2PMessageAsync(capMsg);
                }
                catch { }
            });
        }
    }

    /// <summary>
    /// Send a compute message to the peer via the BEP 10 extension channel.
    /// Uses WireExtension.SendAsync which sends directly through the wire.
    /// </summary>
    public new async Task SendComputeMessageAsync(byte[] data)
    {
        if (!IsSupported) return;
        await SendAsync(data);
    }

    /// <summary>
    /// Send a typed P2P message to the peer.
    /// </summary>
    public async Task SendP2PMessageAsync(P2PMessage message)
    {
        var data = P2PProtocol.Serialize(message);
        await SendComputeMessageAsync(data);
    }
}
