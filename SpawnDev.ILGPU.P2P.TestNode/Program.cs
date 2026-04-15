using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using SpawnDev.ILGPU.P2P;
using SpawnDev.WebTorrent;

/// <summary>
/// P2P Test Node - runs as coordinator, worker, or selftest for integration testing.
///
/// Emits structured output protocol for TestNodeProcess.cs parsing:
///   MAGNET:{link}
///   READY
///   PEER_JOINED:{peerId}
///   PEER_LEFT:{peerId}
///   DISPATCH_SENT:{dispatchId}
///   DISPATCH_COMPLETE:{dispatchId}:{success}:{durationMs}
///   RESULT:{bufferId}:{base64data}
///   ERROR:{message}
///
/// Usage:
///   dotnet run -- coordinator [flags]
///   dotnet run -- worker {magnetLink} [flags]
///   dotnet run -- selftest [flags]
///
/// Flags:
///   --tracker {url}       Override tracker URL
///   --tflops {value}      Override advertised TFLOPS
///   --thermal {state}     Override thermal state (0-3)
///   --verify              Output result buffers as base64
///   --policy {mode}       Set join policy (open/known/invite/maxpeers:N)
///   --always-fail         Worker always returns dispatch failure
/// </summary>

// Parse CLI flags
var flags = ParseFlags(args);
var mode = args.Length > 0 ? args[0] : "selftest";

switch (mode)
{
    case "coordinator":
        await RunCoordinator(flags);
        break;
    case "worker" when args.Length > 1:
        await RunWorker(args[1], flags);
        break;
    case "selftest":
        await RunSelfTest(flags);
        break;
    default:
        Console.WriteLine("Usage:");
        Console.WriteLine("  coordinator              - create swarm, wait for workers");
        Console.WriteLine("  worker <magnetLink>      - join swarm as worker");
        Console.WriteLine("  selftest                 - in-process coordinator + worker test");
        Console.WriteLine();
        Console.WriteLine("Flags:");
        Console.WriteLine("  --tracker <url>          Override tracker URL");
        Console.WriteLine("  --tflops <value>         Override advertised TFLOPS");
        Console.WriteLine("  --thermal <state>        Override thermal state (0-3)");
        Console.WriteLine("  --verify                 Output result buffers as base64");
        Console.WriteLine("  --policy <mode>          Join policy: open/known/invite/maxpeers:N");
        Console.WriteLine("  --always-fail            Worker always returns dispatch failure");
        break;
}

async Task RunCoordinator(TestFlags flags)
{
    Console.Error.WriteLine("[Coordinator] Starting...");
    await using var client = new WebTorrentClient();
    await using var coordinator = new P2PSwarmCoordinator(client);

    // Apply policy from flags
    ApplyPolicy(coordinator, flags);

    var trackers = flags.TrackerUrl != null
        ? new[] { flags.TrackerUrl }
        : null;
    await coordinator.CreateSwarmAsync("P2P-TestNode", trackers);

    // Emit structured magnet link
    Console.WriteLine($"MAGNET:{coordinator.MagnetLink}");

    coordinator.OnPeerJoined += peer =>
    {
        Console.WriteLine($"PEER_JOINED:{peer.PeerId}");
        Console.Error.WriteLine($"[Coordinator] Worker joined: {peer.PeerId} ({peer.Capabilities?.EstimatedTflops} TFLOPS)");
    };

    coordinator.OnPeerLeft += peer =>
    {
        Console.WriteLine($"PEER_LEFT:{peer.PeerId}");
    };

    coordinator.OnCapacityChanged += () =>
    {
        Console.Error.WriteLine($"[Coordinator] Capacity: {coordinator.PeerCount} peers, {coordinator.TotalTflops} TFLOPS");
    };

    // Signal ready
    Console.WriteLine("READY");

    // Wait for at least one worker
    while (coordinator.PeerCount == 0)
        await Task.Delay(500);

    Console.Error.WriteLine("[Coordinator] Worker connected. Dispatching test kernel...");

    // Create accelerator and dispatch
    using var context = Context.CreateDefault();
    var accelerator = coordinator.CreateAccelerator(context);
    var dispatcher = new P2PDispatcher(accelerator);

    // Register test kernels
    P2PKernelSerializer.RegisterKernelType(typeof(TestKernels));

    // Prepare test data: a[i] = i, b[i] = i * 2, expected result[i] = i * 3
    int n = 1024;
    var aFloats = new float[n];
    var bFloats = new float[n];
    for (int i = 0; i < n; i++)
    {
        aFloats[i] = i;
        bFloats[i] = i * 2;
    }

    var method = typeof(TestKernels).GetMethod(nameof(TestKernels.VectorAdd))!;
    var request = P2PKernelSerializer.CreateDispatch(method, gridDimX: n);
    request.Buffers = new[]
    {
        new BufferBinding { ParameterIndex = 1, BufferId = "a", Length = n, ElementSize = 4 },
        new BufferBinding { ParameterIndex = 2, BufferId = "b", Length = n, ElementSize = 4 },
        new BufferBinding { ParameterIndex = 3, BufferId = "result", Length = n, ElementSize = 4 },
    };

    Console.WriteLine($"DISPATCH_SENT:{request.DispatchId}");
    var dispatchId = dispatcher.Dispatch(request);

    // Wire result handler
    dispatcher.OnDispatchFailed += (id, error) =>
    {
        Console.WriteLine($"DISPATCH_COMPLETE:{id}:False:0");
        Console.WriteLine($"ERROR:{error}");
    };

    // Wait for result
    await Task.Delay(10000);
    Console.Error.WriteLine("[Coordinator] Done.");
}

async Task RunWorker(string magnetLink, TestFlags flags)
{
    Console.Error.WriteLine("[Worker] Starting...");
    await using var client = new WebTorrentClient();
    await using var coordinator = new P2PSwarmCoordinator(client);
    await coordinator.JoinSwarmAsync(magnetLink);

    Console.Error.WriteLine("[Worker] Joined swarm. Initializing compute...");

    using var context = Context.Create(builder => builder.CPU());
    using var accelerator = context.CreateCPUAccelerator(0);

    var p2pAccel = coordinator.CreateAccelerator(context);
    var dispatcher = new P2PDispatcher(p2pAccel);
    await using var transport = new P2PTransport(client, coordinator, dispatcher);
    await using var worker = new P2PWorker(transport);
    worker.Initialize(context, accelerator);

    // Register test kernels for security allowlist
    P2PKernelSerializer.RegisterKernelType(typeof(TestKernels));

    worker.OnKernelCompiled += name =>
    {
        Console.Error.WriteLine($"[Worker] Compiled: {name}");
    };

    worker.OnKernelCompleted += (id, success, ms) =>
    {
        // Override success if --always-fail
        if (flags.AlwaysFail)
            success = false;

        Console.WriteLine($"DISPATCH_COMPLETE:{id}:{success}:{ms:F1}");
        Console.Error.WriteLine($"[Worker] Completed: {id} success={success} {ms:N0}ms");

        // Output result buffers if --verify
        if (flags.Verify && success)
        {
            var resultData = worker.GetBuffer("result");
            if (resultData != null)
                Console.WriteLine($"RESULT:result:{Convert.ToBase64String(resultData)}");
        }
    };

    // Build capabilities with flag overrides
    var caps = worker.BuildCapabilities("worker");
    if (flags.Tflops.HasValue)
        caps.EstimatedTflops = flags.Tflops.Value;
    if (flags.ThermalState.HasValue)
        caps.ThermalState = flags.ThermalState.Value;

    // Signal ready
    Console.WriteLine("READY");

    // Keep running until parent kills us or timeout
    await Task.Delay(60000);
    Console.Error.WriteLine("[Worker] Shutting down.");
}

async Task RunSelfTest(TestFlags flags)
{
    Console.Error.WriteLine("[SelfTest] Running in-process coordinator + worker...");

    await using var client = new WebTorrentClient();
    await using var coordinator = new P2PSwarmCoordinator(client);

    // Apply policy from flags
    ApplyPolicy(coordinator, flags);

    await coordinator.CreateSwarmAsync("SelfTest");
    Console.WriteLine($"MAGNET:{coordinator.MagnetLink}");

    using var context = Context.Create(builder => builder.CPU());
    using var accelerator = context.CreateCPUAccelerator(0);

    var p2pAccel = coordinator.CreateAccelerator(context);
    var dispatcher = new P2PDispatcher(p2pAccel);
    await using var transport = new P2PTransport(client, coordinator, dispatcher);
    await using var worker = new P2PWorker(transport);
    worker.Initialize(context, accelerator);

    // Simulate peer connection
    var caps = worker.BuildCapabilities("self-worker");
    if (flags.Tflops.HasValue)
        caps.EstimatedTflops = flags.Tflops.Value;
    if (flags.ThermalState.HasValue)
        caps.ThermalState = flags.ThermalState.Value;

    coordinator.HandlePeerConnected("self-worker", caps);
    p2pAccel.AddPeer(new RemotePeer
    {
        PeerId = "self-worker",
        IsConnected = true,
        Capabilities = caps,
    });
    Console.WriteLine("PEER_JOINED:self-worker");

    // Register kernel type for P2P security allowlist
    P2PKernelSerializer.RegisterKernelType(typeof(TestKernels));

    // Pre-compile kernel
    var method = typeof(TestKernels).GetMethod(nameof(TestKernels.VectorAdd))!;
    var compiled = worker.PreCompileKernel(method);
    Console.Error.WriteLine($"[SelfTest] Kernel compiled: {compiled}");

    // Signal ready
    Console.WriteLine("READY");

    // Create dispatch
    int n = 1024;
    var request = P2PKernelSerializer.CreateDispatch(method, gridDimX: n);
    request.Buffers = new[]
    {
        new BufferBinding { ParameterIndex = 1, BufferId = "a", Length = n, ElementSize = 4 },
        new BufferBinding { ParameterIndex = 2, BufferId = "b", Length = n, ElementSize = 4 },
        new BufferBinding { ParameterIndex = 3, BufferId = "result", Length = n, ElementSize = 4 },
    };

    // Prepare real data: a[i] = i, b[i] = i * 2, expected result[i] = i * 3
    var aData = new byte[n * 4];
    var bData = new byte[n * 4];
    var aFloats = new float[n];
    var bFloats = new float[n];
    for (int i = 0; i < n; i++)
    {
        aFloats[i] = i;
        bFloats[i] = i * 2;
    }
    Buffer.BlockCopy(aFloats, 0, aData, 0, n * 4);
    Buffer.BlockCopy(bFloats, 0, bData, 0, n * 4);

    worker.ReceiveBuffer("a", aData);
    worker.ReceiveBuffer("b", bData);

    Console.WriteLine($"DISPATCH_SENT:{request.DispatchId}");

    // Dispatch and handle
    bool completed = false;
    bool success = false;
    double durationMs = 0;
    worker.OnKernelCompleted += (id, s, ms) =>
    {
        // Override success if --always-fail
        if (flags.AlwaysFail)
            s = false;

        completed = true;
        success = s;
        durationMs = ms;
        Console.WriteLine($"DISPATCH_COMPLETE:{id}:{s}:{ms:F1}");
    };

    await worker.HandleDispatchAsync("coordinator", request);

    if (!completed || !success)
    {
        Console.WriteLine($"ERROR:Dispatch did not complete successfully");
        return;
    }

    // Verify result buffer: result[i] should equal a[i] + b[i] = i + i*2 = i*3
    var resultData = worker.GetBuffer("result");
    if (resultData == null || resultData.Length != n * 4)
    {
        Console.WriteLine($"ERROR:Result buffer missing or wrong size (expected {n * 4}, got {resultData?.Length ?? 0})");
        return;
    }

    // Output result buffer if --verify
    if (flags.Verify)
        Console.WriteLine($"RESULT:result:{Convert.ToBase64String(resultData)}");

    var resultFloats = new float[n];
    Buffer.BlockCopy(resultData, 0, resultFloats, 0, n * 4);

    int violations = 0;
    for (int i = 0; i < n; i++)
    {
        float expected = i * 3.0f;
        if (Math.Abs(resultFloats[i] - expected) > 0.001f)
        {
            if (violations < 5)
                Console.Error.WriteLine($"[SelfTest] Mismatch at [{i}]: expected {expected}, got {resultFloats[i]}");
            violations++;
        }
    }

    if (violations > 0)
    {
        Console.WriteLine($"ERROR:{violations}/{n} elements incorrect");
    }
    else
    {
        Console.Error.WriteLine($"[SelfTest] PASS - All {n} elements verified correct. VectorAdd dispatched in {durationMs:F1}ms.");
    }
}

static void ApplyPolicy(P2PSwarmCoordinator coordinator, TestFlags flags)
{
    if (flags.PolicyMode == null) return;

    var mode = flags.PolicyMode.ToLowerInvariant();
    if (mode.StartsWith("maxpeers:"))
    {
        if (int.TryParse(mode["maxpeers:".Length..], out var max))
            coordinator.Policy = new SwarmPolicy { MaxPeers = max };
    }
    else
    {
        coordinator.Policy = mode switch
        {
            "open" => new SwarmPolicy { JoinPermission = JoinMode.Open },
            "known" => new SwarmPolicy { JoinPermission = JoinMode.KnownOnly },
            "invite" => new SwarmPolicy { JoinPermission = JoinMode.InviteOnly },
            "approval" => new SwarmPolicy { JoinPermission = JoinMode.Approval },
            _ => new SwarmPolicy(),
        };
    }
}

static TestFlags ParseFlags(string[] args)
{
    var flags = new TestFlags();
    for (int i = 0; i < args.Length; i++)
    {
        switch (args[i])
        {
            case "--tracker" when i + 1 < args.Length:
                flags.TrackerUrl = args[++i];
                break;
            case "--tflops" when i + 1 < args.Length:
                if (double.TryParse(args[++i], out var tflops))
                    flags.Tflops = tflops;
                break;
            case "--thermal" when i + 1 < args.Length:
                if (int.TryParse(args[++i], out var thermal))
                    flags.ThermalState = thermal;
                break;
            case "--verify":
                flags.Verify = true;
                break;
            case "--policy" when i + 1 < args.Length:
                flags.PolicyMode = args[++i];
                break;
            case "--always-fail":
                flags.AlwaysFail = true;
                break;
        }
    }
    return flags;
}

/// <summary>
/// Parsed CLI flags for TestNode.
/// </summary>
class TestFlags
{
    public string? TrackerUrl { get; set; }
    public double? Tflops { get; set; }
    public int? ThermalState { get; set; }
    public bool Verify { get; set; }
    public string? PolicyMode { get; set; }
    public bool AlwaysFail { get; set; }
}

/// <summary>
/// Test kernels for P2P dispatch testing.
/// </summary>
public static class TestKernels
{
    public static void VectorAdd(Index1D index,
        ArrayView<float> a, ArrayView<float> b, ArrayView<float> result)
    {
        result[index] = a[index] + b[index];
    }
}
