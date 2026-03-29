using System.Collections.Concurrent;
using System.Text.Json;
using SpawnDev.WebTorrent;
using SpawnDev.WebTorrent.Torrent;

namespace SpawnDev.ILGPU.P2P;

/// <summary>
/// Bridges SpawnDev.WebTorrent peer connections to the P2P compute protocol.
/// Handles message routing between the P2PDispatcher and WebRTC data channels.
///
/// Each connected WebTorrent peer gets a P2P compute channel layered on top
/// of the existing BitTorrent wire protocol connection.
/// </summary>
public class P2PTransport : IAsyncDisposable
{
    private readonly WebTorrentClient _client;
    private readonly P2PSwarmCoordinator _coordinator;
    private readonly P2PDispatcher _dispatcher;
    private readonly ConcurrentDictionary<string, PeerChannel> _channels = new();
    private readonly P2PBufferTransfer _bufferTransfer = new();

    /// <summary>
    /// Fired when a compute message is received from a peer.
    /// </summary>
    public event Action<string, P2PMessage>? OnMessageReceived;

    public P2PTransport(
        WebTorrentClient client,
        P2PSwarmCoordinator coordinator,
        P2PDispatcher dispatcher)
    {
        _client = client;
        _coordinator = coordinator;
        _dispatcher = dispatcher;
    }

    /// <summary>
    /// Register a peer connection for P2P compute messaging.
    /// Called when a WebTorrent peer connects via WebRTC.
    /// </summary>
    public void RegisterPeer(string peerId, Func<byte[], Task> sendFunc)
    {
        var channel = new PeerChannel
        {
            PeerId = peerId,
            SendAsync = sendFunc,
        };
        _channels[peerId] = channel;

        // Request capabilities from the new peer
        _ = SendMessageAsync(peerId, new P2PMessage
        {
            Type = P2PMessageType.CapabilityRequest,
        });
    }

    /// <summary>
    /// Unregister a peer (disconnected).
    /// </summary>
    public void UnregisterPeer(string peerId)
    {
        _channels.TryRemove(peerId, out _);
        _coordinator.HandlePeerDisconnected(peerId);
        _dispatcher.HandlePeerLost(peerId);
    }

    /// <summary>
    /// Handle incoming data from a peer's compute channel.
    /// </summary>
    public async Task HandleIncomingDataAsync(string peerId, byte[] data)
    {
        P2PMessage? message;
        try
        {
            message = P2PProtocol.Deserialize(data);
        }
        catch
        {
            return; // Malformed message — ignore
        }

        if (message == null) return;

        OnMessageReceived?.Invoke(peerId, message);

        switch (message.Type)
        {
            case P2PMessageType.CapabilityResponse:
                HandleCapabilityResponse(peerId, message);
                break;

            case P2PMessageType.KernelResult:
                HandleKernelResult(peerId, message);
                break;

            case P2PMessageType.BufferData:
                HandleBufferData(peerId, message);
                break;

            case P2PMessageType.Heartbeat:
                _dispatcher.HandleHeartbeat(peerId);
                break;

            case P2PMessageType.StatusUpdate:
                HandleStatusUpdate(peerId, message);
                break;

            case P2PMessageType.Disconnect:
                UnregisterPeer(peerId);
                break;

            // Peer-side handling (when this node is a worker):
            case P2PMessageType.CapabilityRequest:
                await HandleCapabilityRequest(peerId);
                break;

            case P2PMessageType.KernelDispatch:
                await HandleKernelDispatchRequest(peerId, message);
                break;

            case P2PMessageType.BufferSend:
                HandleBufferReceive(peerId, message);
                break;
        }
    }

    /// <summary>
    /// Send a P2P compute message to a specific peer.
    /// </summary>
    public async Task SendMessageAsync(string peerId, P2PMessage message)
    {
        if (_channels.TryGetValue(peerId, out var channel))
        {
            var data = P2PProtocol.Serialize(message);
            await channel.SendAsync(data);
        }
    }

    /// <summary>
    /// Broadcast a message to all connected compute peers.
    /// </summary>
    public async Task BroadcastAsync(P2PMessage message)
    {
        var data = P2PProtocol.Serialize(message);
        foreach (var channel in _channels.Values)
        {
            try { await channel.SendAsync(data); } catch { }
        }
    }

    #region Message Handlers — Coordinator Side

    private void HandleCapabilityResponse(string peerId, P2PMessage message)
    {
        if (message.Payload == null) return;
        var caps = message.Payload.Value.Deserialize<PeerCapabilities>();
        if (caps != null)
        {
            _coordinator.HandlePeerConnected(peerId, caps);
        }
    }

    private void HandleKernelResult(string peerId, P2PMessage message)
    {
        if (message.Payload == null) return;
        var result = message.Payload.Value.Deserialize<KernelDispatchResult>();
        if (result != null)
        {
            _dispatcher.HandleResult(result.DispatchId, result);
        }
    }

    private void HandleBufferData(string peerId, P2PMessage message)
    {
        if (message.Payload == null) return;
        var chunk = message.Payload.Value.Deserialize<BufferChunk>();
        if (chunk != null)
        {
            _bufferTransfer.ReceiveChunk(chunk);
        }
    }

    /// <summary>
    /// Send buffer data to a peer in chunks.
    /// </summary>
    public async Task SendBufferAsync(string peerId, string bufferId, byte[] data)
    {
        var chunks = _bufferTransfer.CreateChunks(bufferId, data);
        foreach (var chunk in chunks)
        {
            await SendMessageAsync(peerId, new P2PMessage
            {
                Type = P2PMessageType.BufferSend,
                Payload = JsonSerializer.SerializeToElement(chunk),
            });
        }
    }

    /// <summary>
    /// Access the buffer transfer for progress tracking and events.
    /// </summary>
    public P2PBufferTransfer BufferTransfer => _bufferTransfer;

    private void HandleStatusUpdate(string peerId, P2PMessage message)
    {
        if (message.Payload == null) return;
        var caps = message.Payload.Value.Deserialize<PeerCapabilities>();
        if (caps != null)
        {
            _dispatcher.UpdatePeerCapabilities(peerId, caps);
        }
    }

    #endregion

    #region Message Handlers — Worker Side

    private async Task HandleCapabilityRequest(string peerId)
    {
        // Report our capabilities to the requesting coordinator
        var caps = BuildLocalCapabilities();
        await SendMessageAsync(peerId, new P2PMessage
        {
            Type = P2PMessageType.CapabilityResponse,
            Payload = JsonSerializer.SerializeToElement(caps),
        });
    }

    private async Task HandleKernelDispatchRequest(string peerId, P2PMessage message)
    {
        // TODO: Deserialize kernel dispatch, compile locally, execute, return result
        // This is where the peer uses its own local ILGPU backend
        if (message.Payload == null) return;
        var request = message.Payload.Value.Deserialize<KernelDispatchRequest>();
        if (request == null) return;

        // For now, report not implemented
        await SendMessageAsync(peerId, new P2PMessage
        {
            Type = P2PMessageType.KernelResult,
            ReplyTo = message.MessageId,
            Payload = JsonSerializer.SerializeToElement(new KernelDispatchResult
            {
                DispatchId = request.DispatchId,
                Success = false,
                Error = "Worker-side kernel execution not yet implemented",
            }),
        });
    }

    private void HandleBufferReceive(string peerId, P2PMessage message)
    {
        if (message.Payload == null) return;
        var chunk = message.Payload.Value.Deserialize<BufferChunk>();
        if (chunk != null)
        {
            _bufferTransfer.ReceiveChunk(chunk);
        }
    }

    #endregion

    /// <summary>
    /// Build capabilities for the local device.
    /// </summary>
    private PeerCapabilities BuildLocalCapabilities()
    {
        return new PeerCapabilities
        {
            PeerId = _client.PeerId != null ? Convert.ToHexString(_client.PeerId) : Guid.NewGuid().ToString("N"),
            Platform = OperatingSystem.IsBrowser() ? "browser" : "desktop",
            IlgpuVersion = "4.7.0",
            // TODO: Detect actual backends, VRAM, TFLOPS
            AvailableBackends = new[] { "CPU" },
            PreferredBackend = "CPU",
            AvailableMemory = Environment.WorkingSet,
            EstimatedTflops = 1.0,
            MaxThreadsPerGroup = 256,
            IsCharging = true, // TODO: navigator.getBattery()
            BatteryLevel = -1,
            ThermalState = 0,
        };
    }

    /// <inheritdoc/>
    public ValueTask DisposeAsync()
    {
        _channels.Clear();
        return ValueTask.CompletedTask;
    }
}

/// <summary>
/// Represents a compute messaging channel to a specific peer.
/// </summary>
internal class PeerChannel
{
    public string PeerId { get; set; } = "";
    public Func<byte[], Task> SendAsync { get; set; } = _ => Task.CompletedTask;
}
