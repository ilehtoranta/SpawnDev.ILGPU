using System.Text.Json;
using System.Text.Json.Serialization;

namespace SpawnDev.ILGPU.P2P;

/// <summary>
/// P2P compute protocol messages — sent between coordinator and peers
/// over WebRTC data channels via SpawnDev.WebTorrent.
///
/// Message flow:
///   Coordinator → Peer: CAPABILITY_REQUEST, KERNEL_DISPATCH, BUFFER_SEND
///   Peer → Coordinator: CAPABILITY_RESPONSE, KERNEL_RESULT, BUFFER_DATA
/// </summary>
public static class P2PProtocol
{
    public const string Version = "1.0";

    /// <summary>
    /// Serialize a message to JSON bytes for transmission over WebRTC.
    /// </summary>
    public static byte[] Serialize(P2PMessage message)
    {
        return JsonSerializer.SerializeToUtf8Bytes(message, _jsonOptions);
    }

    /// <summary>
    /// Deserialize a message from JSON bytes.
    /// </summary>
    public static P2PMessage? Deserialize(byte[] data)
    {
        return JsonSerializer.Deserialize<P2PMessage>(data, _jsonOptions);
    }

    private static readonly JsonSerializerOptions _jsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        Converters = { new JsonStringEnumConverter() },
    };
}

/// <summary>
/// Base message type for P2P compute protocol.
/// </summary>
public class P2PMessage
{
    /// <summary>
    /// Message type identifier.
    /// </summary>
    public P2PMessageType Type { get; set; }

    /// <summary>
    /// Protocol version.
    /// </summary>
    public string Version { get; set; } = P2PProtocol.Version;

    /// <summary>
    /// Unique message ID for request/response correlation.
    /// </summary>
    public string MessageId { get; set; } = Guid.NewGuid().ToString("N");

    /// <summary>
    /// ID of the message this is responding to (for responses).
    /// </summary>
    public string? ReplyTo { get; set; }

    /// <summary>
    /// JSON payload (message-type-specific).
    /// </summary>
    public JsonElement? Payload { get; set; }
}

/// <summary>
/// P2P compute message types.
/// </summary>
public enum P2PMessageType
{
    /// <summary>Request peer's hardware capabilities.</summary>
    CapabilityRequest,

    /// <summary>Peer reports its available backends, VRAM, TFLOPS estimate.</summary>
    CapabilityResponse,

    /// <summary>Send a kernel to a peer for compilation and execution.</summary>
    KernelDispatch,

    /// <summary>Peer reports kernel execution result (success/error, timing).</summary>
    KernelResult,

    /// <summary>Send buffer data to a peer (tensor transfer).</summary>
    BufferSend,

    /// <summary>Request buffer data from a peer.</summary>
    BufferRequest,

    /// <summary>Peer sends buffer data back.</summary>
    BufferData,

    /// <summary>Heartbeat — peer is alive and working.</summary>
    Heartbeat,

    /// <summary>Peer is leaving the swarm.</summary>
    Disconnect,

    /// <summary>Peer reports updated status (thermal, battery, TFLOPS degradation).</summary>
    StatusUpdate,

    /// <summary>Coordinator initiates graceful handoff — peer should flush state via BEP 46.</summary>
    GracefulHandoff,

    /// <summary>Coordinator transfers leadership to another peer.</summary>
    CoordinatorTransfer,

    /// <summary>Peer announces it is the new coordinator (after election or transfer).</summary>
    CoordinatorAnnounce,

    /// <summary>Peer requests coordinator election (current coordinator is unresponsive).</summary>
    ElectionRequest,

    /// <summary>Coordinator kicks a peer from the swarm.</summary>
    Kick,

    /// <summary>Coordinator blocks a peer from the swarm (permanent until unblocked).</summary>
    Block,
}

/// <summary>
/// Roles in the P2P compute swarm.
/// The coordinator role is transferable — not a permanent assignment.
/// </summary>
public enum P2PRole
{
    /// <summary>
    /// Worker — executes kernels dispatched by the coordinator.
    /// Can be promoted to coordinator via transfer or election.
    /// </summary>
    Worker,

    /// <summary>
    /// Coordinator — dispatches work, tracks state, scores peers.
    /// Transferable. If coordinator drops, workers elect a new one.
    /// </summary>
    Coordinator,
}

/// <summary>
/// Capability manifest — sent by peers when they join the swarm.
/// </summary>
public class PeerCapabilities
{
    /// <summary>Peer's unique ID.</summary>
    public string PeerId { get; set; } = "";

    /// <summary>Available accelerator backends on this peer.</summary>
    public string[] AvailableBackends { get; set; } = Array.Empty<string>();

    /// <summary>Preferred backend (what the peer will use for compute).</summary>
    public string PreferredBackend { get; set; } = "";

    /// <summary>Available GPU memory in bytes.</summary>
    public long AvailableMemory { get; set; }

    /// <summary>Estimated TFLOPS (single precision).</summary>
    public double EstimatedTflops { get; set; }

    /// <summary>Maximum threads per group supported.</summary>
    public int MaxThreadsPerGroup { get; set; }

    /// <summary>Maximum shared memory per group (bytes).</summary>
    public int MaxSharedMemory { get; set; }

    /// <summary>Platform (browser/desktop).</summary>
    public string Platform { get; set; } = "";

    /// <summary>SpawnDev.ILGPU version.</summary>
    public string IlgpuVersion { get; set; } = "";

    /// <summary>Battery level (0.0-1.0). -1 = plugged in / unknown.</summary>
    public double BatteryLevel { get; set; } = -1;

    /// <summary>Whether the device is charging.</summary>
    public bool IsCharging { get; set; } = true;

    /// <summary>Thermal state (0=nominal, 1=fair, 2=serious, 3=critical).</summary>
    public int ThermalState { get; set; } = 0;

    /// <summary>
    /// Peer's ECDSA public key in SPKI format (base64).
    /// Null/empty = anonymous peer (no cryptographic identity).
    /// </summary>
    public string? PublicKey { get; set; }

    /// <summary>
    /// SHA-256 fingerprint of the peer's public key (hex, lowercase).
    /// Used for quick identity lookups without full key comparison.
    /// </summary>
    public string? Fingerprint { get; set; }
}

/// <summary>
/// Kernel dispatch request — sent to a peer to execute a kernel.
/// </summary>
public class KernelDispatchRequest
{
    /// <summary>Unique dispatch ID for tracking.</summary>
    public string DispatchId { get; set; } = Guid.NewGuid().ToString("N");

    /// <summary>Kernel entry point method name (for the peer to compile from its own assembly).</summary>
    public string KernelMethod { get; set; } = "";

    /// <summary>Coordinator's public key (base64 SPKI) for authority verification.</summary>
    public string? CoordinatorPublicKey { get; set; }

    /// <summary>Kernel entry point type name.</summary>
    public string KernelType { get; set; } = "";

    /// <summary>Grid dimension (total work items).</summary>
    public long GridDimX { get; set; }
    public long GridDimY { get; set; }
    public long GridDimZ { get; set; }

    /// <summary>Group dimension (threads per group).</summary>
    public int GroupDimX { get; set; }
    public int GroupDimY { get; set; }
    public int GroupDimZ { get; set; }

    /// <summary>Buffer bindings — which buffers to bind to which kernel parameters.</summary>
    public BufferBinding[] Buffers { get; set; } = Array.Empty<BufferBinding>();

    /// <summary>Scalar parameter values (serialized).</summary>
    public byte[]? ScalarParams { get; set; }
}

/// <summary>
/// Associates a buffer with a kernel parameter slot.
/// </summary>
public class BufferBinding
{
    /// <summary>Parameter index in the kernel signature.</summary>
    public int ParameterIndex { get; set; }

    /// <summary>Buffer ID (matches a previously sent BufferSend).</summary>
    public string BufferId { get; set; } = "";

    /// <summary>Offset into the buffer (bytes).</summary>
    public long Offset { get; set; }

    /// <summary>Length of the view (elements).</summary>
    public long Length { get; set; }

    /// <summary>Element size (bytes).</summary>
    public int ElementSize { get; set; }
}

/// <summary>
/// Kernel execution result — sent by peer after dispatch completes.
/// </summary>
public class KernelDispatchResult
{
    /// <summary>Dispatch ID this result is for.</summary>
    public string DispatchId { get; set; } = "";

    /// <summary>Whether execution succeeded.</summary>
    public bool Success { get; set; }

    /// <summary>Error message if failed.</summary>
    public string? Error { get; set; }

    /// <summary>Execution time in milliseconds.</summary>
    public double DurationMs { get; set; }

    /// <summary>IDs of buffers that were modified (need to be read back).</summary>
    public string[] ModifiedBuffers { get; set; } = Array.Empty<string>();
}

/// <summary>
/// Data included in a CoordinatorTransfer message — pending state for the new coordinator.
/// </summary>
public class CoordinatorTransferData
{
    /// <summary>Peer ID of the new coordinator.</summary>
    public string NewCoordinatorPeerId { get; set; } = "";

    /// <summary>Timestamp of the transfer.</summary>
    public DateTimeOffset Timestamp { get; set; }

    /// <summary>Pending dispatch requests that the new coordinator must track.</summary>
    public PendingDispatchInfo[]? PendingDispatches { get; set; }
}

/// <summary>
/// Serializable snapshot of a pending dispatch (for coordinator transfer).
/// </summary>
public class PendingDispatchInfo
{
    /// <summary>The dispatch ID.</summary>
    public string DispatchId { get; set; } = "";

    /// <summary>The original dispatch request.</summary>
    public KernelDispatchRequest Request { get; set; } = new();

    /// <summary>Peer currently executing this dispatch.</summary>
    public string AssignedPeerId { get; set; } = "";

    /// <summary>Number of retry attempts so far.</summary>
    public int Attempts { get; set; }
}

/// <summary>
/// Data in a CoordinatorAnnounce message.
/// </summary>
public class CoordinatorAnnounceData
{
    /// <summary>Peer ID of the new coordinator.</summary>
    public string NewCoordinatorPeerId { get; set; } = "";
}
