using ILGPU;
using ILGPU.Runtime;
using SpawnDev.BlazorJS.Cryptography;
using SpawnDev.WebTorrent;

namespace SpawnDev.ILGPU.P2P;

/// <summary>
/// High-level facade for P2P distributed compute.
/// Simplifies the setup from ~20 lines to ~3 lines.
///
/// Usage — Coordinator (creates swarm):
///   var compute = await P2PCompute.CreateSwarmAsync(crypto, client, "My Inference");
///   Console.WriteLine(compute.JoinLink); // Share as QR code
///   // compute.Accelerator is ready for kernel dispatch
///
/// Usage — Worker (joins swarm):
///   var compute = await P2PCompute.JoinSwarmAsync(crypto, client, accelerator, magnetOrJoinLink);
///   // Automatically contributes compute power
///
/// Usage — Self-contained (both in one process, for testing):
///   var compute = await P2PCompute.CreateSelfTestAsync(crypto, client, context, accelerator);
///   // Full pipeline in-process
/// </summary>
public class P2PCompute : IAsyncDisposable
{
    /// <summary>The swarm coordinator.</summary>
    public P2PSwarmCoordinator Coordinator { get; }

    /// <summary>The P2P accelerator (coordinator side).</summary>
    public P2PAccelerator? Accelerator { get; }

    /// <summary>The fault-tolerant dispatcher.</summary>
    public P2PDispatcher? Dispatcher { get; }

    /// <summary>The transport layer.</summary>
    public P2PTransport? Transport { get; }

    /// <summary>The WebRTC bridge — auto-wires sd_compute to real peer connections.</summary>
    public P2PWebRtcBridge? Bridge { get; }

    /// <summary>The worker (worker side).</summary>
    public P2PWorker? Worker { get; }

    /// <summary>The cryptographic identity.</summary>
    public SwarmIdentity Identity { get; }

    /// <summary>Magnet link for joining.</summary>
    public string? MagnetLink => Coordinator.MagnetLink;

    /// <summary>HTTP join link (QR-friendly).</summary>
    public string? JoinLink => Coordinator.JoinLink;

    /// <summary>Number of connected peers.</summary>
    public int PeerCount => Coordinator.PeerCount;

    /// <summary>Total TFLOPS across the swarm.</summary>
    public double TotalTflops => Coordinator.TotalTflops;

    /// <summary>This node's role.</summary>
    public P2PRole Role => Coordinator.Role;

    /// <summary>
    /// Dispatch a kernel to the swarm. Coordinator-side convenience method.
    ///
    /// Usage:
    ///   compute.DispatchToSwarm(typeof(MyKernels), "VectorAdd", 1024,
    ///       ("a", aData, 4), ("b", bData, 4), ("result", null, 4));
    /// </summary>
    public string DispatchToSwarm(Type kernelType, string methodName, long gridDimX,
        params (string bufferId, byte[]? data, int elementSize)[] buffers)
    {
        if (Accelerator == null)
            throw new InvalidOperationException("Not a coordinator node");

        var helper = Accelerator.CreateDispatcher(kernelType, methodName);
        return helper.Execute(gridDimX, buffers);
    }

    private readonly WebTorrentClient _client;
    private readonly global::ILGPU.Context? _context;

    private P2PCompute(
        WebTorrentClient client,
        SwarmIdentity identity,
        P2PSwarmCoordinator coordinator,
        P2PAccelerator? accelerator = null,
        P2PDispatcher? dispatcher = null,
        P2PTransport? transport = null,
        P2PWebRtcBridge? bridge = null,
        P2PWorker? worker = null,
        global::ILGPU.Context? context = null)
    {
        _client = client;
        _context = context;
        Identity = identity;
        Coordinator = coordinator;
        Accelerator = accelerator;
        Dispatcher = dispatcher;
        Transport = transport;
        Bridge = bridge;
        Worker = worker;
    }

    /// <summary>
    /// Create a new compute swarm. This node becomes the coordinator/owner.
    /// </summary>
    /// <param name="crypto">Crypto provider (DotNetCrypto for desktop, BrowserCrypto for browser).</param>
    /// <param name="client">WebTorrent client for P2P transport.</param>
    /// <param name="name">Human-readable swarm name.</param>
    /// <param name="joinLinkBaseUrl">Base URL for HTTP join links (your web app URL).</param>
    public static async Task<P2PCompute> CreateSwarmAsync(
        IPortableCrypto crypto,
        WebTorrentClient client,
        string name,
        string? joinLinkBaseUrl = null)
    {
        var identity = await SwarmIdentity.CreateAsync(crypto, name + "-owner");
        var coordinator = new P2PSwarmCoordinator(client);
        coordinator.SetIdentity(identity);

        // Auto-detect join link URL: explicit > browser location > null (desktop)
        if (joinLinkBaseUrl != null)
            coordinator.JoinLinkBaseUrl = joinLinkBaseUrl;
        else if (OperatingSystem.IsBrowser())
            coordinator.JoinLinkBaseUrl = GetBrowserBaseUrl();
        await coordinator.CreateSwarmAsync(name);

        var context = global::ILGPU.Context.CreateDefault();
        var accelerator = coordinator.CreateAccelerator(context);
        accelerator.Dispatcher = new P2PDispatcher(accelerator);
        var dispatcher = accelerator.Dispatcher;
        var transport = new P2PTransport(client, coordinator, dispatcher);

        // Wire coordinator messages to transport
        coordinator.OnSendMessage += async (peerId, msg) =>
        {
            await transport.SendMessageAsync(peerId, msg);
        };

        // Wire dispatcher dispatch messages to transport
        dispatcher.OnSendMessage += (peerId, msg) =>
        {
            _ = transport.SendMessageAsync(peerId, msg);
        };

        // Bridge WebRTC peer connections to sd_compute extension
        var bridge = new P2PWebRtcBridge(transport);
        if (coordinator.Swarm != null)
            bridge.AttachToSwarm(coordinator.Swarm);

        return new P2PCompute(client, identity, coordinator, accelerator, dispatcher, transport, bridge, context: context);
    }

    /// <summary>
    /// Join an existing compute swarm as a worker.
    /// </summary>
    /// <param name="crypto">Crypto provider.</param>
    /// <param name="client">WebTorrent client.</param>
    /// <param name="accelerator">Local accelerator for kernel execution.</param>
    /// <param name="magnetOrJoinLink">Magnet link or HTTP join link.</param>
    public static async Task<P2PCompute> JoinSwarmAsync(
        IPortableCrypto crypto,
        WebTorrentClient client,
        Accelerator accelerator,
        string magnetOrJoinLink)
    {
        var identity = await SwarmIdentity.CreateAsync(crypto, "worker");
        var coordinator = new P2PSwarmCoordinator(client);
        coordinator.SetIdentity(identity);

        // Extract magnet from join link if needed
        string magnet;
        if (magnetOrJoinLink.StartsWith("magnet:"))
        {
            magnet = magnetOrJoinLink;
        }
        else
        {
            // Parse HTTP join link → build magnet from hash
            var (hash, name) = ParseJoinLink(magnetOrJoinLink);
            magnet = hash != null
                ? BuildMagnetFromHash(hash, name)
                : magnetOrJoinLink; // fallback: treat as raw magnet
        }
        await coordinator.JoinSwarmAsync(magnet);

        var p2pAccel = coordinator.CreateAccelerator(accelerator.Context);
        var dispatcher = new P2PDispatcher(p2pAccel);
        var transport = new P2PTransport(client, coordinator, dispatcher);
        var worker = new P2PWorker(transport);
        worker.Initialize(accelerator.Context, accelerator);
        transport.SetWorker(worker);

        // Bridge WebRTC peer connections to sd_compute extension
        var bridge = new P2PWebRtcBridge(transport);
        if (coordinator.Swarm != null)
            bridge.AttachToSwarm(coordinator.Swarm);

        return new P2PCompute(client, identity, coordinator, p2pAccel, dispatcher, transport, bridge, worker);
    }

    /// <summary>
    /// Create a self-contained test setup (coordinator + worker in one process).
    /// </summary>
    public static async Task<P2PCompute> CreateSelfTestAsync(
        IPortableCrypto crypto,
        WebTorrentClient client,
        global::ILGPU.Context context,
        Accelerator accelerator)
    {
        var compute = await CreateSwarmAsync(crypto, client, "SelfTest");

        // Create worker in same process
        var worker = new P2PWorker(compute.Transport!);
        worker.Initialize(context, accelerator);
        compute.Transport!.SetWorker(worker);

        // Connect worker as peer
        var caps = worker.BuildCapabilities("self-worker");
        compute.Coordinator.HandlePeerConnected("self-worker", caps);
        compute.Accelerator!.AddPeer(new RemotePeer
        {
            PeerId = "self-worker",
            IsConnected = true,
            Capabilities = caps,
        });

        return new P2PCompute(client, compute.Identity, compute.Coordinator,
            compute.Accelerator, compute.Dispatcher, compute.Transport, compute.Bridge, worker, context);
    }

    /// <summary>
    /// Get the current browser URL (origin + path) for auto-detecting join link base.
    /// Returns null on desktop.
    /// </summary>
    private static string? GetBrowserBaseUrl()
    {
        try
        {
            if (!OperatingSystem.IsBrowser()) return null;
            // In Blazor WASM: window.location.origin + window.location.pathname
            // Access via SpawnDev.BlazorJS if available, fallback to null
            var origin = SpawnDev.BlazorJS.BlazorJSRuntime.JS?.Get<string>("window.location.origin");
            var path = SpawnDev.BlazorJS.BlazorJSRuntime.JS?.Get<string>("window.location.pathname");
            if (!string.IsNullOrEmpty(origin))
                return origin + (path ?? "");
        }
        catch { }
        return null;
    }

    /// <summary>
    /// Parse a join link URL to extract the compute hash and swarm name.
    /// Returns (hash, name) or (null, null) if not a join link.
    /// </summary>
    public static (string? computeHash, string? swarmName) ParseJoinLink(string url)
    {
        try
        {
            var uri = new Uri(url);
            var query = System.Web.HttpUtility.ParseQueryString(uri.Query);
            var hash = query["compute"];
            var name = query["n"];
            return (hash, name);
        }
        catch
        {
            return (null, null);
        }
    }

    /// <summary>
    /// Build a magnet link from a compute hash and default tracker.
    /// Used by workers joining via HTTP join link.
    /// </summary>
    public static string BuildMagnetFromHash(string computeHash, string? swarmName = null,
        string tracker = "wss://hub.spawndev.com:44365/announce")
    {
        var dn = !string.IsNullOrEmpty(swarmName)
            ? $"&dn={Uri.EscapeDataString(swarmName)}" : "";
        return $"magnet:?xt=urn:btih:{computeHash}{dn}&tr={Uri.EscapeDataString(tracker)}";
    }

    /// <inheritdoc/>
    public async ValueTask DisposeAsync()
    {
        if (Worker != null) await Worker.DisposeAsync();
        if (Bridge != null) await Bridge.DisposeAsync();
        if (Transport != null) await Transport.DisposeAsync();
        Dispatcher?.Dispose();
        await Coordinator.DisposeAsync();
        await Identity.DisposeAsync();
    }
}
