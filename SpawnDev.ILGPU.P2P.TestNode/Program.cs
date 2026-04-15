using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using SpawnDev.ILGPU.P2P;
using SpawnDev.WebTorrent;

/// <summary>
/// P2P Test Node — runs as either coordinator or worker for integration testing.
///
/// Usage:
///   dotnet run -- coordinator              → creates swarm, prints magnet, waits for workers
///   dotnet run -- worker {magnetLink}      → joins swarm, executes dispatched kernels
///   dotnet run -- selftest                 → runs both in-process (mock transport)
/// </summary>

var mode = args.Length > 0 ? args[0] : "selftest";

switch (mode)
{
    case "coordinator":
        await RunCoordinator();
        break;
    case "worker" when args.Length > 1:
        await RunWorker(args[1]);
        break;
    case "selftest":
        await RunSelfTest();
        break;
    default:
        Console.WriteLine("Usage:");
        Console.WriteLine("  coordinator              — create swarm, wait for workers");
        Console.WriteLine("  worker <magnetLink>      — join swarm as worker");
        Console.WriteLine("  selftest                 — in-process coordinator + worker test");
        break;
}

async Task RunCoordinator()
{
    Console.WriteLine("[Coordinator] Starting...");
    await using var client = new WebTorrentClient();
    await using var coordinator = new P2PSwarmCoordinator(client);
    await coordinator.CreateSwarmAsync("P2P-TestNode");

    Console.WriteLine($"MAGNET:{coordinator.MagnetLink}");
    Console.WriteLine($"[Coordinator] Waiting for workers...");

    coordinator.OnPeerJoined += peer =>
    {
        Console.WriteLine($"[Coordinator] Worker joined: {peer.PeerId} ({peer.Capabilities?.EstimatedTflops} TFLOPS)");
    };

    coordinator.OnCapacityChanged += () =>
    {
        Console.WriteLine($"[Coordinator] Capacity: {coordinator.PeerCount} peers, {coordinator.TotalTflops} TFLOPS");
    };

    // Wait for at least one worker
    while (coordinator.PeerCount == 0)
        await Task.Delay(500);

    Console.WriteLine("[Coordinator] Worker connected. Dispatching test kernel...");

    // Create accelerator and dispatch
    using var context = Context.CreateDefault();
    var accelerator = coordinator.CreateAccelerator(context);
    var dispatcher = new P2PDispatcher(accelerator);
    var request = P2PKernelSerializer.CreateDispatch(
        typeof(TestKernels).GetMethod(nameof(TestKernels.VectorAdd))!,
        gridDimX: 1024);

    var dispatchId = dispatcher.Dispatch(request);
    Console.WriteLine($"[Coordinator] Dispatched: {dispatchId}");

    // Wait for result
    await Task.Delay(5000);
    Console.WriteLine("[Coordinator] Done.");
}

async Task RunWorker(string magnetLink)
{
    Console.WriteLine("[Worker] Starting...");
    await using var client = new WebTorrentClient();
    await using var coordinator = new P2PSwarmCoordinator(client);
    await coordinator.JoinSwarmAsync(magnetLink);

    Console.WriteLine("[Worker] Joined swarm. Waiting for dispatches...");

    using var context = Context.Create(builder => builder.CPU());
    using var accelerator = context.CreateCPUAccelerator(0);

    var p2pAccel = coordinator.CreateAccelerator(context);
    var dispatcher = new P2PDispatcher(p2pAccel);
    await using var transport = new P2PTransport(client, coordinator, dispatcher);
    await using var worker = new P2PWorker(transport);
    worker.Initialize(context, accelerator);

    worker.OnKernelCompiled += name => Console.WriteLine($"[Worker] Compiled: {name}");
    worker.OnKernelCompleted += (id, success, ms) =>
        Console.WriteLine($"[Worker] Completed: {id} success={success} {ms:N0}ms");

    // Keep running
    await Task.Delay(30000);
    Console.WriteLine("[Worker] Shutting down.");
}

async Task RunSelfTest()
{
    Console.WriteLine("[SelfTest] Running in-process coordinator + worker...");

    await using var client = new WebTorrentClient();
    await using var coordinator = new P2PSwarmCoordinator(client);
    await coordinator.CreateSwarmAsync("SelfTest");

    using var context = Context.Create(builder => builder.CPU());
    using var accelerator = context.CreateCPUAccelerator(0);

    var p2pAccel = coordinator.CreateAccelerator(context);
    var dispatcher = new P2PDispatcher(p2pAccel);
    await using var transport = new P2PTransport(client, coordinator, dispatcher);
    await using var worker = new P2PWorker(transport);
    worker.Initialize(context, accelerator);

    // Simulate peer connection
    var caps = worker.BuildCapabilities("self-worker");
    coordinator.HandlePeerConnected("self-worker", caps);
    p2pAccel.AddPeer(new RemotePeer
    {
        PeerId = "self-worker",
        IsConnected = true,
        Capabilities = caps,
    });

    // Register kernel type for P2P security allowlist
    P2PKernelSerializer.RegisterKernelType(typeof(TestKernels));

    // Pre-compile kernel
    var method = typeof(TestKernels).GetMethod(nameof(TestKernels.VectorAdd))!;
    var compiled = worker.PreCompileKernel(method);
    Console.WriteLine($"[SelfTest] Kernel compiled: {compiled}");

    // Create dispatch
    var request = P2PKernelSerializer.CreateDispatch(method, gridDimX: 1024);
    // Index1D is param 0 (auto-filled), ArrayView params start at index 1
    request.Buffers = new[]
    {
        new BufferBinding { ParameterIndex = 1, BufferId = "a", Length = 1024, ElementSize = 4 },
        new BufferBinding { ParameterIndex = 2, BufferId = "b", Length = 1024, ElementSize = 4 },
        new BufferBinding { ParameterIndex = 3, BufferId = "result", Length = 1024, ElementSize = 4 },
    };

    // Prepare real data: a[i] = i, b[i] = i * 2, expected result[i] = i * 3
    int n = 1024;
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

    // Dispatch and handle
    bool completed = false;
    bool success = false;
    double durationMs = 0;
    worker.OnKernelCompleted += (id, s, ms) =>
    {
        Console.WriteLine($"[SelfTest] Dispatch {id}: success={s}, {ms:N1}ms");
        completed = true;
        success = s;
        durationMs = ms;
    };

    await worker.HandleDispatchAsync("coordinator", request);

    if (!completed || !success)
    {
        Console.WriteLine("[SelfTest] FAIL - Dispatch did not complete successfully.");
        return;
    }

    // Verify result buffer: result[i] should equal a[i] + b[i] = i + i*2 = i*3
    var resultData = worker.GetBuffer("result");
    if (resultData == null || resultData.Length != n * 4)
    {
        Console.WriteLine($"[SelfTest] FAIL - Result buffer missing or wrong size (expected {n * 4}, got {resultData?.Length ?? 0}).");
        return;
    }

    var resultFloats = new float[n];
    Buffer.BlockCopy(resultData, 0, resultFloats, 0, n * 4);

    int violations = 0;
    for (int i = 0; i < n; i++)
    {
        float expected = i * 3.0f;
        if (Math.Abs(resultFloats[i] - expected) > 0.001f)
        {
            if (violations < 5)
                Console.WriteLine($"[SelfTest] Mismatch at [{i}]: expected {expected}, got {resultFloats[i]}");
            violations++;
        }
    }

    if (violations > 0)
        Console.WriteLine($"[SelfTest] FAIL - {violations} / {n} elements incorrect.");
    else
        Console.WriteLine($"[SelfTest] PASS - All {n} elements verified correct. VectorAdd dispatched in {durationMs:N1}ms.");
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
