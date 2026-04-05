using System.Text.Json;

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
///   client.UseExtension((wire) =>
///   {
///       var ext = new SdComputeExtension(transport);
///       ext.SetWire(wire);
///       return ext;
///   });
/// </summary>
public class SdComputeExtension : SpawnDev.WebTorrent.IWireExtension
{
    private readonly P2PTransport _transport;
    private SpawnDev.WebTorrent.Wire? _wire;
    private string _peerId = "";

    /// <summary>
    /// Extension name in BEP 10 handshake.
    /// </summary>
    public string Name => "sd_compute";

    /// <summary>
    /// True if the remote peer also supports sd_compute (determined after BEP 10 handshake).
    /// </summary>
    public bool IsSupported { get; private set; }

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
    /// Set the Wire reference for sending messages.
    /// Called by the UseExtension factory.
    /// </summary>
    public void SetWire(SpawnDev.WebTorrent.Wire wire)
    {
        _wire = wire;
        // Add our version to the extended handshake
        wire.ExtendedHandshake["sd_compute_version"] = P2PProtocol.Version;
    }

    /// <summary>
    /// Set the peer ID for this connection.
    /// </summary>
    public void SetPeerId(string peerId)
    {
        _peerId = peerId;
    }

    /// <summary>
    /// Called when the BEP 3 handshake completes (before BEP 10 extended handshake).
    /// </summary>
    public void OnHandshake(string infoHash, string peerId, SpawnDev.WebTorrent.WireExtensions extensions)
    {
        if (string.IsNullOrEmpty(_peerId))
            _peerId = peerId;
    }

    /// <summary>
    /// Called when the BEP 10 extended handshake arrives from the remote peer.
    /// Check if peer supports sd_compute.
    /// </summary>
    public void OnExtendedHandshake(Dictionary<string, object> handshake)
    {
        // The Wire class already filters: OnExtendedHandshake is only called
        // if the peer's 'm' dict includes our extension name. So if we get here,
        // the peer supports sd_compute.
        IsSupported = true;

        Console.WriteLine($"[sd_compute] OnExtendedHandshake: peerId={_peerId}, IsSupported={IsSupported}");

        // Check for version info
        if (handshake.TryGetValue("sd_compute_version", out var version))
        {
            Console.WriteLine($"[sd_compute] Peer {_peerId} supports sd_compute version {version}");
        }

        // Register peer in transport with a send function that uses the wire
        _transport.RegisterPeer(_peerId, async (data) =>
        {
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
                    Payload = JsonSerializer.SerializeToElement(
                        _transport.GetLocalCapabilities()),
                };
                await SendP2PMessageAsync(capMsg);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[sd_compute] Capability exchange failed for peer {_peerId}: {ex.Message}");
            }
        });
    }

    /// <summary>
    /// Handle incoming sd_compute extension message.
    /// Called by the Wire framework when a BEP 10 message with our extension ID arrives.
    /// </summary>
    public void OnMessage(byte[] payload)
    {
        Console.WriteLine($"[sd_compute] OnMessage: peerId={_peerId}, {payload.Length} bytes, IsSupported={IsSupported}");
        // Route to the P2P transport for handling (fire-and-forget since OnMessage is sync)
        _ = _transport.HandleIncomingDataAsync(_peerId, payload);

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
    /// Send a compute message to the peer via the BEP 10 extension channel.
    /// </summary>
    public async Task SendComputeMessageAsync(byte[] data)
    {
        if (_wire == null || !IsSupported) return;
        await _wire.Extended("sd_compute", data);
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
