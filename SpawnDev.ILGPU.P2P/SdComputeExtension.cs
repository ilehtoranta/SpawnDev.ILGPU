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

    /// <summary>
    /// Local BEP 44/46 DHT public key (raw 32-byte Ed25519). Advertised to the remote peer
    /// in the BEP 10 extended handshake under key <c>sd_compute_dht_pubkey</c>, hex-encoded.
    /// On the coordinator side this is <c>P2PStateManager.PublicKey</c> so workers can find
    /// our DHT mutable-item channel. Set BEFORE <see cref="SetWire"/> runs or the handshake
    /// will go out without the key. Workers generally leave this null - only coordinators
    /// publish DHT state.
    /// </summary>
    public byte[]? DhtPublicKey { get; set; }

    /// <summary>
    /// Remote peer's advertised DHT public key (raw 32-byte Ed25519), if any. Populated from
    /// the remote's <c>sd_compute_dht_pubkey</c> handshake entry. Null when the remote did
    /// not advertise one (typical for workers). Subscribers call
    /// <c>P2PStateManager.SubscribeAsync(RemoteDhtPublicKey)</c> on the value to receive
    /// BEP 46 state updates from that peer.
    /// </summary>
    public byte[]? RemoteDhtPublicKey { get; private set; }

    /// <summary>
    /// Fired once the remote peer's DHT public key arrives in the extended handshake. The
    /// byte[] is the raw 32-byte Ed25519 key. Used by <c>P2PCompute.JoinSwarmAsync</c> to
    /// drive <c>StateManager.SubscribeAsync</c> after the sd_compute handshake settles.
    /// </summary>
    public event Action<byte[]>? OnRemoteDhtPublicKeyReceived;

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

        // If we have a DHT identity to publish under, advertise it hex-encoded so the
        // remote peer can subscribe to our BEP 46 mutable-item channel. Hex instead of
        // raw bytes because bencoded extended-handshake values round-trip as strings
        // cleanly; an ASCII-safe 64-char hex string avoids any binary-in-dict quirks.
        if (DhtPublicKey != null && DhtPublicKey.Length == 32)
        {
            wire.ExtendedHandshake["sd_compute_dht_pubkey"] =
                Convert.ToHexString(DhtPublicKey).ToLowerInvariant();
        }
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

        if (P2PCompute.VerboseLogging) Console.WriteLine($"[sd_compute] OnExtendedHandshake: peerId={_peerId}, IsSupported={IsSupported}");

        // Check for version info
        if (handshake.TryGetValue("sd_compute_version", out var version))
        {
            if (P2PCompute.VerboseLogging) Console.WriteLine($"[sd_compute] Peer {_peerId} supports sd_compute version {version}");
        }

        // Extract remote DHT pubkey if advertised. Values come through as either string
        // or byte[] depending on the wire's bencode decoder - handle both. Only raw 32-byte
        // Ed25519 keys are honored; anything else is ignored and RemoteDhtPublicKey stays
        // null so subscribers don't call SubscribeAsync with garbage.
        if (handshake.TryGetValue("sd_compute_dht_pubkey", out var pubKeyObj))
        {
            byte[]? decoded = null;
            try
            {
                decoded = pubKeyObj switch
                {
                    string s when s.Length == 64 => Convert.FromHexString(s),
                    byte[] b when b.Length == 64 =>
                        Convert.FromHexString(System.Text.Encoding.ASCII.GetString(b)),
                    byte[] b when b.Length == 32 => b,
                    _ => null,
                };
            }
            catch (FormatException)
            {
                decoded = null;
            }
            if (decoded != null && decoded.Length == 32)
            {
                RemoteDhtPublicKey = decoded;
                if (P2PCompute.VerboseLogging) Console.WriteLine(
                    $"[sd_compute] Peer {_peerId} advertised DHT pubkey " +
                    $"{Convert.ToHexString(decoded)[..16].ToLowerInvariant()}...");
                try { OnRemoteDhtPublicKeyReceived?.Invoke(decoded); }
                catch (Exception ex)
                {
                    if (P2PCompute.VerboseLogging) Console.WriteLine($"[sd_compute] OnRemoteDhtPublicKeyReceived handler threw: {ex.Message}");
                }
            }
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
                if (P2PCompute.VerboseLogging) Console.WriteLine($"[sd_compute] Capability exchange failed for peer {_peerId}: {ex.Message}");
            }
        });
    }

    /// <summary>
    /// Handle incoming sd_compute extension message.
    /// Called by the Wire framework when a BEP 10 message with our extension ID arrives.
    /// </summary>
    public void OnMessage(byte[] payload)
    {
        if (P2PCompute.VerboseLogging) Console.WriteLine($"[sd_compute] OnMessage: peerId={_peerId}, {payload.Length} bytes, IsSupported={IsSupported}");
        // Route to the P2P transport for handling (fire-and-forget since OnMessage is sync).
        // The transport owns the single primary parse (binary fast-path or JSON).
        _ = _transport.HandleIncomingDataAsync(_peerId, payload);

        // Binary buffer-chunk frames carry no P2PMessage envelope and no consumers
        // of OnComputeMessage care about per-chunk events - skip the deserialize.
        // This removes the redundant second JSON parse that previously ran on every
        // tensor chunk in parallel with the transport's parse.
        if (P2PBinaryFrame.IsBinaryFrame(payload))
            return;

        // Fire local event + cache capabilities for handshake completion detection.
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
