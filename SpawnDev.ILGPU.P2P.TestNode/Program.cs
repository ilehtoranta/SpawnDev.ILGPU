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

    using var context = Context.Create(builder => builder.CPU(CPUDevice.Nvidia));
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

    using var context = Context.Create(builder => builder.CPU(CPUDevice.Nvidia));
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

    // Pre-compile kernel
    var method = typeof(TestKernels).GetMethod(nameof(TestKernels.VectorAdd))!;
    var compiled = worker.PreCompileKernel(method);
    Console.WriteLine($"[SelfTest] Kernel compiled: {compiled}");

    // Create dispatch
    var request = P2PKernelSerializer.CreateDispatch(method, gridDimX: 1024);
    request.Buffers = new[]
    {
        new BufferBinding { ParameterIndex = 0, BufferId = "a", Length = 1024, ElementSize = 4 },
        new BufferBinding { ParameterIndex = 1, BufferId = "b", Length = 1024, ElementSize = 4 },
        new BufferBinding { ParameterIndex = 2, BufferId = "result", Length = 1024, ElementSize = 4 },
    };

    // Send buffers
    worker.ReceiveBuffer("a", new byte[4096]);
    worker.ReceiveBuffer("b", new byte[4096]);

    // Dispatch and handle
    bool completed = false;
    worker.OnKernelCompleted += (id, success, ms) =>
    {
        Console.WriteLine($"[SelfTest] Dispatch {id}: success={success}, {ms:N0}ms");
        completed = true;
    };

    await worker.HandleDispatchAsync("coordinator", request);

    if (completed)
        Console.WriteLine("[SelfTest] PASS — Full pipeline works.");
    else
        Console.WriteLine("[SelfTest] FAIL — Dispatch did not complete.");
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
