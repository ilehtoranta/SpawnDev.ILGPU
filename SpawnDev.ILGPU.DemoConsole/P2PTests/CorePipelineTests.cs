using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using SpawnDev.ILGPU.P2P;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.DemoConsole.P2PTests;

/// <summary>
/// Core pipeline integration tests. Phase 2 of the test plan.
/// Proves the fundamental dispatch pipeline works - first in-process,
/// then over real WebRTC between separate processes.
/// </summary>
public class CorePipelineTests
{
    /// <summary>
    /// In-process baseline: VectorAdd with real CPU accelerator, no WebRTC.
    /// If this fails, nothing else can work.
    /// </summary>
    [TestMethod]
    public async Task CorePipeline_InProcess_Baseline()
    {
        // Register kernel type in allowlist
        P2PKernelSerializer.RegisterKernelType(typeof(P2PTestKernels));

        using var context = Context.Create(builder => builder.CPU());
        using var accelerator = context.CreateCPUAccelerator(0);

        var coordinator = new P2PSwarmCoordinator(new SpawnDev.WebTorrent.WebTorrentClient());
        var p2pAccel = coordinator.CreateAccelerator(context);
        var dispatcher = new P2PDispatcher(p2pAccel);
        var transport = new P2PTransport(
            new SpawnDev.WebTorrent.WebTorrentClient(), coordinator, dispatcher);
        var worker = new P2PWorker(transport);
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

        // Prepare test data
        int n = 1024;
        var (aBytes, bBytes, expected) = DataIntegrityHelper.GenerateVectorAddData(n);

        // Send buffers to worker
        worker.ReceiveBuffer("a", aBytes);
        worker.ReceiveBuffer("b", bBytes);
        worker.ReceiveBuffer("result", new byte[n * 4]);

        // Create dispatch request
        var method = typeof(P2PTestKernels).GetMethod(nameof(P2PTestKernels.VectorAdd))!;
        var request = P2PKernelSerializer.CreateDispatch(method, gridDimX: n);
        request.Buffers = new[]
        {
            new BufferBinding { ParameterIndex = 1, BufferId = "a", Length = n, ElementSize = 4 },
            new BufferBinding { ParameterIndex = 2, BufferId = "b", Length = n, ElementSize = 4 },
            new BufferBinding { ParameterIndex = 3, BufferId = "result", Length = n, ElementSize = 4 },
        };

        // Dispatch
        bool completed = false;
        bool success = false;
        worker.OnKernelCompleted += (id, s, ms) =>
        {
            completed = true;
            success = s;
        };

        await worker.HandleDispatchAsync("coordinator", request);

        if (!completed) throw new Exception("Dispatch did not complete");
        if (!success) throw new Exception("Dispatch failed");

        // Verify result
        var resultBytes = worker.GetBuffer("result");
        if (resultBytes == null) throw new Exception("Result buffer missing");
        if (resultBytes.Length != n * 4) throw new Exception($"Result buffer wrong size: {resultBytes.Length}");

        var actual = DataIntegrityHelper.BytesToFloats(resultBytes);
        var (violations, firstIdx, firstExp, firstAct) =
            DataIntegrityHelper.VerifyFloats(actual, expected);

        if (violations != 0)
            throw new Exception(
                $"VectorAdd failed: {violations}/{n} violations. " +
                $"First at [{firstIdx}]: expected {firstExp}, got {firstAct}");
    }

    /// <summary>
    /// In-process baseline: integer Identity kernel - verifies non-float types.
    /// </summary>
    [TestMethod]
    public async Task CorePipeline_InProcess_IntegerKernel()
    {
        P2PKernelSerializer.RegisterKernelType(typeof(P2PTestKernels));

        using var context = Context.Create(builder => builder.CPU());
        using var accelerator = context.CreateCPUAccelerator(0);

        var coordinator = new P2PSwarmCoordinator(new SpawnDev.WebTorrent.WebTorrentClient());
        var p2pAccel = coordinator.CreateAccelerator(context);
        var dispatcher = new P2PDispatcher(p2pAccel);
        var transport = new P2PTransport(
            new SpawnDev.WebTorrent.WebTorrentClient(), coordinator, dispatcher);
        var worker = new P2PWorker(transport);
        worker.Initialize(context, accelerator);

        var caps = worker.BuildCapabilities("self-worker");
        coordinator.HandlePeerConnected("self-worker", caps);
        p2pAccel.AddPeer(new RemotePeer
        {
            PeerId = "self-worker",
            IsConnected = true,
            Capabilities = caps,
        });

        int n = 512;
        var input = new int[n];
        var expectedOutput = new int[n];
        for (int i = 0; i < n; i++)
        {
            input[i] = i * 7 + 13;
            expectedOutput[i] = input[i] * 2;
        }
        var inputBytes = new byte[n * 4];
        Buffer.BlockCopy(input, 0, inputBytes, 0, n * 4);

        worker.ReceiveBuffer("input", inputBytes);
        worker.ReceiveBuffer("result", new byte[n * 4]);

        var method = typeof(P2PTestKernels).GetMethod(nameof(P2PTestKernels.IntDoubler))!;
        var request = P2PKernelSerializer.CreateDispatch(method, gridDimX: n);
        request.Buffers = new[]
        {
            new BufferBinding { ParameterIndex = 1, BufferId = "input", Length = n, ElementSize = 4 },
            new BufferBinding { ParameterIndex = 2, BufferId = "result", Length = n, ElementSize = 4 },
        };

        bool completed = false;
        bool success = false;
        worker.OnKernelCompleted += (id, s, ms) => { completed = true; success = s; };

        await worker.HandleDispatchAsync("coordinator", request);

        if (!completed) throw new Exception("Dispatch did not complete");
        if (!success) throw new Exception("Dispatch failed");

        var resultBytes = worker.GetBuffer("result");
        if (resultBytes == null) throw new Exception("Result buffer missing");

        var actual = DataIntegrityHelper.BytesToInts(resultBytes);
        var (violations, firstIdx, firstExp, firstAct) =
            DataIntegrityHelper.VerifyInts(actual, expectedOutput);

        if (violations != 0)
            throw new Exception(
                $"IntDoubler failed: {violations}/{n} violations. " +
                $"First at [{firstIdx}]: expected {firstExp}, got {firstAct}");
    }

    /// <summary>
    /// In-process baseline: SHA256 data integrity - random data, Identity kernel,
    /// hash comparison before/after. Any bit flip causes failure.
    /// </summary>
    [TestMethod]
    public async Task CorePipeline_InProcess_DataIntegrity_SHA256()
    {
        P2PKernelSerializer.RegisterKernelType(typeof(P2PTestKernels));

        using var context = Context.Create(builder => builder.CPU());
        using var accelerator = context.CreateCPUAccelerator(0);

        var coordinator = new P2PSwarmCoordinator(new SpawnDev.WebTorrent.WebTorrentClient());
        var p2pAccel = coordinator.CreateAccelerator(context);
        var dispatcher = new P2PDispatcher(p2pAccel);
        var transport = new P2PTransport(
            new SpawnDev.WebTorrent.WebTorrentClient(), coordinator, dispatcher);
        var worker = new P2PWorker(transport);
        worker.Initialize(context, accelerator);

        var caps = worker.BuildCapabilities("self-worker");
        coordinator.HandlePeerConnected("self-worker", caps);
        p2pAccel.AddPeer(new RemotePeer
        {
            PeerId = "self-worker",
            IsConnected = true,
            Capabilities = caps,
        });

        int n = 16384;
        var rng = new Random(42);
        var input = new int[n];
        for (int i = 0; i < n; i++)
            input[i] = rng.Next();

        var inputBytes = new byte[n * 4];
        Buffer.BlockCopy(input, 0, inputBytes, 0, n * 4);
        var hashBefore = DataIntegrityHelper.ComputeSha256(inputBytes);

        worker.ReceiveBuffer("input", inputBytes);
        worker.ReceiveBuffer("result", new byte[n * 4]);

        var method = typeof(P2PTestKernels).GetMethod(nameof(P2PTestKernels.Identity))!;
        var request = P2PKernelSerializer.CreateDispatch(method, gridDimX: n);
        request.Buffers = new[]
        {
            new BufferBinding { ParameterIndex = 1, BufferId = "input", Length = n, ElementSize = 4 },
            new BufferBinding { ParameterIndex = 2, BufferId = "result", Length = n, ElementSize = 4 },
        };

        bool completed = false;
        bool success = false;
        worker.OnKernelCompleted += (id, s, ms) => { completed = true; success = s; };

        await worker.HandleDispatchAsync("coordinator", request);

        if (!(completed && success)) throw new Exception("Identity dispatch failed");

        var resultBytes = worker.GetBuffer("result");
        var hashAfter = DataIntegrityHelper.ComputeSha256(resultBytes!);

        if (hashAfter != hashBefore)
            throw new Exception(
                $"SHA256 mismatch: data corrupted during Identity kernel dispatch. " +
                $"Before: {hashBefore[..16]}... After: {hashAfter[..16]}...");
    }

    /// <summary>
    /// In-process baseline for scalar kernel parameter transmission.
    /// Dispatches VectorScale(input, result, scalar=7.5f) and proves the scalar value
    /// sent by the coordinator actually reaches the worker's kernel invocation, rather
    /// than silently defaulting to 0 as it did before the ScalarParams fix.
    /// </summary>
    [TestMethod]
    public async Task CorePipeline_InProcess_ScalarParams()
    {
        P2PKernelSerializer.RegisterKernelType(typeof(P2PTestKernels));

        using var context = Context.Create(builder => builder.CPU());
        using var accelerator = context.CreateCPUAccelerator(0);

        var coordinator = new P2PSwarmCoordinator(new SpawnDev.WebTorrent.WebTorrentClient());
        var p2pAccel = coordinator.CreateAccelerator(context);
        var dispatcher = new P2PDispatcher(p2pAccel);
        var transport = new P2PTransport(
            new SpawnDev.WebTorrent.WebTorrentClient(), coordinator, dispatcher);
        var worker = new P2PWorker(transport);
        worker.Initialize(context, accelerator);

        var caps = worker.BuildCapabilities("self-worker");
        coordinator.HandlePeerConnected("self-worker", caps);
        p2pAccel.AddPeer(new RemotePeer
        {
            PeerId = "self-worker",
            IsConnected = true,
            Capabilities = caps,
        });

        const int n = 1024;
        const float scalar = 7.5f;
        var input = new float[n];
        var expected = new float[n];
        for (int i = 0; i < n; i++)
        {
            input[i] = i;
            expected[i] = i * scalar;
        }
        var inputBytes = new byte[n * 4];
        Buffer.BlockCopy(input, 0, inputBytes, 0, n * 4);

        worker.ReceiveBuffer("input", inputBytes);
        worker.ReceiveBuffer("result", new byte[n * 4]);

        var method = typeof(P2PTestKernels).GetMethod(nameof(P2PTestKernels.VectorScale))!;
        var request = P2PKernelSerializer.CreateDispatch(
            method,
            gridDimX: n,
            scalarValues: new Dictionary<int, object> { [3] = scalar });
        request.Buffers = new[]
        {
            new BufferBinding { ParameterIndex = 1, BufferId = "input", Length = n, ElementSize = 4 },
            new BufferBinding { ParameterIndex = 2, BufferId = "result", Length = n, ElementSize = 4 },
        };

        if (request.ScalarParams == null)
            throw new Exception("CreateDispatch should populate ScalarParams when scalarValues is provided.");

        bool completed = false;
        bool success = false;
        worker.OnKernelCompleted += (id, s, ms) => { completed = true; success = s; };

        await worker.HandleDispatchAsync("coordinator", request);

        if (!(completed && success)) throw new Exception("VectorScale dispatch failed");

        var resultBytes = worker.GetBuffer("result");
        if (resultBytes == null) throw new Exception("Result buffer missing");

        var actual = DataIntegrityHelper.BytesToFloats(resultBytes);
        var (violations, firstIdx, firstExp, firstAct) =
            DataIntegrityHelper.VerifyFloats(actual, expected);

        // Regression: if ScalarParams is silently defaulted to 0, every result[i] comes back as 0
        // and violations == n-1 (result[0] = 0 * 7.5 = 0 = expected[0], rest all mismatch).
        if (violations != 0)
            throw new Exception(
                $"VectorScale scalar param not transmitted: {violations}/{n} violations. " +
                $"First at [{firstIdx}]: expected {firstExp}, got {firstAct}. " +
                $"If firstAct is 0, the worker silently defaulted scalar to 0.");
    }
}
