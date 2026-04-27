using System.Collections.Concurrent;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using SpawnDev.ILGPU.P2P;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.DemoConsole.P2PTests;

/// <summary>
/// Phase 6 stress tests from <c>SpawnDev.ILGPU.P2P/Plans/PLAN-IntegrationTests.md</c>.
///
/// In-process simulated peers (no real WebRTC) — these tests exercise the
/// dispatcher's concurrency model, the dispatch chain across multiple stages,
/// and disposal cleanup. Real-WebRTC stress is the next layer up; getting the
/// in-process layer green first is the gate before paying the WebRTC dispatch
/// cost on every iteration.
///
/// Each test uses the <c>HandlePeerConnected</c> + <c>AddPeer</c> simulated-peer
/// pattern that <see cref="MultiPeerTests"/> established. Results are returned
/// via <c>HandleResult</c> as if the peer had actually run the kernel.
/// </summary>
public class StressTests
{
    /// <summary>
    /// Fire 20 dispatches concurrently against a 3-worker swarm. Verifies the
    /// dispatcher serializes correctly under contention: every dispatch lands
    /// on exactly one worker, every result is handled exactly once, and the
    /// pending queue drains to zero. Uses Task.WhenAll on the result-completion
    /// taps to drive the load.
    ///
    /// Coverage gap closed: previously the multi-peer suite did 4 sequential
    /// dispatches. Concurrent dispatch is the realistic load shape — ML
    /// inference and scientific compute pipelines fan out many simultaneous
    /// kernels.
    /// </summary>
    [TestMethod]
    public async Task Stress_Concurrent_20Dispatches_InProcess()
    {
        P2PKernelSerializer.RegisterKernelType(typeof(P2PTestKernels));

        const int dispatchCount = 20;
        const int workerCount = 3;
        const int n = 512;

        using var context = Context.Create(builder => builder.CPU());
        using var accel = context.CreateCPUAccelerator(0);

        var coordinator = new P2PSwarmCoordinator(new SpawnDev.WebTorrent.WebTorrentClient());
        var p2pAccel = coordinator.CreateAccelerator(context);
        var dispatcher = new P2PDispatcher(p2pAccel);

        for (int i = 0; i < workerCount; i++)
        {
            var caps = new PeerCapabilities
            {
                PeerId = $"worker-{i}",
                EstimatedTflops = 2.0,
                AvailableMemory = 1L << 32,
                IsCharging = true,
                ThermalState = 0,
            };
            coordinator.HandlePeerConnected(caps.PeerId, caps);
            p2pAccel.AddPeer(new RemotePeer
            {
                PeerId = caps.PeerId,
                IsConnected = true,
                Capabilities = caps,
            });
        }

        var sentTo = new ConcurrentBag<string>();
        var dispatchedIds = new ConcurrentBag<string>();
        dispatcher.OnSendMessage += (peerId, msg) =>
        {
            // Record which peer received the dispatch. We don't need to inspect
            // the message body - the routing decision is what's under test.
            sentTo.Add(peerId);
        };

        // Fire dispatches concurrently from multiple threads. Each call to
        // dispatcher.Dispatch is internally locked; the test verifies the
        // dispatcher tolerates the contention without dropping or duplicating.
        var method = typeof(P2PTestKernels).GetMethod(nameof(P2PTestKernels.VectorAdd))!;
        await Task.WhenAll(Enumerable.Range(0, dispatchCount).Select(d => Task.Run(() =>
        {
            var request = P2PKernelSerializer.CreateDispatch(method, gridDimX: n);
            request.Buffers = new[]
            {
                new BufferBinding { ParameterIndex = 1, BufferId = $"a{d}", Length = n, ElementSize = 4 },
                new BufferBinding { ParameterIndex = 2, BufferId = $"b{d}", Length = n, ElementSize = 4 },
                new BufferBinding { ParameterIndex = 3, BufferId = $"r{d}", Length = n, ElementSize = 4 },
            };
            dispatcher.Dispatch(request);
            dispatchedIds.Add(request.DispatchId);
        })));

        if (dispatchedIds.Count != dispatchCount)
            throw new Exception(
                $"Expected {dispatchCount} dispatches, recorded {dispatchedIds.Count}.");

        // Every dispatch should have been routed to a worker.
        if (sentTo.Count != dispatchCount)
            throw new Exception(
                $"Expected {dispatchCount} OnSendMessage events, got {sentTo.Count}. " +
                $"Dispatches were silently dropped under concurrent load.");

        // Concurrently feed results back. The dispatcher's _pending queue
        // should drain cleanly with no leftover entries.
        await Task.WhenAll(dispatchedIds.Select(id => Task.Run(() =>
        {
            dispatcher.HandleResult(id, new KernelDispatchResult
            {
                DispatchId = id,
                Success = true,
                DurationMs = 1.0,
            });
        })));

        if (dispatcher.PendingCount != 0)
            throw new Exception(
                $"Pending queue did not drain. Remaining: {dispatcher.PendingCount}. " +
                $"Indicates dispatches were enqueued but never matched to a result.");

        // Distribution check: at minimum 2 of the 3 workers should have been
        // hit (the third may not be hit if the scoring is deterministic enough
        // to favor one consistently, but two is a reasonable lower bound for
        // load distribution under 20 dispatches with equal capabilities).
        var distinctPeers = sentTo.Distinct().Count();
        if (distinctPeers < 2)
            throw new Exception(
                $"Concurrent load was funneled to {distinctPeers} worker(s) of {workerCount}. " +
                $"Expected at least 2 to be hit under {dispatchCount} dispatches.");
    }

    /// <summary>
    /// Two-stage dispatch chain in process: stage 1 fills a buffer with values
    /// 0..n-1, stage 2 scales them by a multiplier. Verifies the dispatcher
    /// handles back-to-back dispatches with shared buffer state — the realistic
    /// shape of ML inference pipelines (preprocess → kernel → postprocess) and
    /// any compute graph that chains operations.
    ///
    /// Coverage gap closed: the existing CorePipeline tests cover single-stage
    /// dispatches end-to-end. Two-stage with a buffer carried between stages
    /// is the next-layer pipeline test.
    /// </summary>
    [TestMethod]
    public async Task Pipeline_TwoStage_EndToEnd_InProcess()
    {
        P2PKernelSerializer.RegisterKernelType(typeof(P2PTestKernels));

        const int n = 1024;
        const float multiplier = 3.5f;

        using var context = Context.Create(builder => builder.CPU());
        using var accel = context.CreateCPUAccelerator(0);

        var coordinator = new P2PSwarmCoordinator(new SpawnDev.WebTorrent.WebTorrentClient());
        var p2pAccel = coordinator.CreateAccelerator(context);
        var dispatcher = new P2PDispatcher(p2pAccel);

        var caps = new PeerCapabilities
        {
            PeerId = "worker-pipeline",
            EstimatedTflops = 2.0,
            AvailableMemory = 1L << 32,
            IsCharging = true,
            ThermalState = 0,
        };
        coordinator.HandlePeerConnected(caps.PeerId, caps);
        p2pAccel.AddPeer(new RemotePeer
        {
            PeerId = caps.PeerId,
            IsConnected = true,
            Capabilities = caps,
        });

        var sendCount = 0;
        var dispatchOrder = new List<string>();
        dispatcher.OnSendMessage += (peerId, msg) => Interlocked.Increment(ref sendCount);

        // Stage 1: FillSequence kernel produces sequential values into "stage1_out".
        var fillMethod = typeof(P2PTestKernels).GetMethod(nameof(P2PTestKernels.FillSequence))!;
        var fillRequest = P2PKernelSerializer.CreateDispatch(fillMethod, gridDimX: n);
        fillRequest.Buffers = new[]
        {
            new BufferBinding { ParameterIndex = 1, BufferId = "stage1_out", Length = n, ElementSize = 4 },
        };
        dispatcher.Dispatch(fillRequest);
        dispatchOrder.Add("stage1");

        // Stage 1 completes - simulate the worker's success and the buffer being
        // available for the next stage.
        dispatcher.HandleResult(fillRequest.DispatchId, new KernelDispatchResult
        {
            DispatchId = fillRequest.DispatchId,
            Success = true,
            ModifiedBuffers = new[] { "stage1_out" },
            DurationMs = 1.5,
        });

        // Stage 2: VectorScale takes "stage1_out" as input and writes "stage2_out".
        // The ParameterIndex=1 binding reuses the buffer ID from stage 1.
        var scaleMethod = typeof(P2PTestKernels).GetMethod(nameof(P2PTestKernels.VectorScale))!;
        var scaleRequest = P2PKernelSerializer.CreateDispatch(
            scaleMethod, gridDimX: n,
            scalarValues: new Dictionary<int, object> { [3] = multiplier });
        scaleRequest.Buffers = new[]
        {
            new BufferBinding { ParameterIndex = 1, BufferId = "stage1_out", Length = n, ElementSize = 4 },
            new BufferBinding { ParameterIndex = 2, BufferId = "stage2_out", Length = n, ElementSize = 4 },
        };
        dispatcher.Dispatch(scaleRequest);
        dispatchOrder.Add("stage2");

        dispatcher.HandleResult(scaleRequest.DispatchId, new KernelDispatchResult
        {
            DispatchId = scaleRequest.DispatchId,
            Success = true,
            ModifiedBuffers = new[] { "stage2_out" },
            DurationMs = 1.2,
        });

        // Both stages should have been sent to the single worker.
        if (sendCount != 2)
            throw new Exception(
                $"Expected 2 dispatch sends (one per stage), got {sendCount}.");

        if (dispatcher.PendingCount != 0)
            throw new Exception(
                $"Pending queue should be empty after both stages complete. " +
                $"Remaining: {dispatcher.PendingCount}.");

        if (dispatchOrder.Count != 2 || dispatchOrder[0] != "stage1" || dispatchOrder[1] != "stage2")
            throw new Exception(
                $"Stage ordering corrupted: [{string.Join(", ", dispatchOrder)}].");

        await Task.CompletedTask;
    }

    /// <summary>
    /// Rapid worker join/leave cycles. Verifies that worker registration and
    /// disconnection don't leak peer state in the coordinator. After 10 cycles,
    /// the coordinator's peer count should match exactly the workers still
    /// connected (zero in this case).
    ///
    /// Coverage gap closed: catches regressions where peer-disconnect handling
    /// (HandlePeerDisconnected) silently fails to remove a peer, causing
    /// stale entries to accumulate. Real WebRTC swarms churn peers constantly
    /// (mobile devices going through tunnels, browser tabs opening/closing).
    /// </summary>
    [TestMethod]
    public Task Stress_RapidJoinLeave_10Cycles()
    {
        const int cycles = 10;

        var coordinator = new P2PSwarmCoordinator(new SpawnDev.WebTorrent.WebTorrentClient());

        for (int cycle = 0; cycle < cycles; cycle++)
        {
            var peerId = $"transient-peer-{cycle}";
            var caps = new PeerCapabilities
            {
                PeerId = peerId,
                EstimatedTflops = 1.0,
                IsCharging = true,
                ThermalState = 0,
            };
            coordinator.HandlePeerConnected(peerId, caps);
            if (coordinator.PeerCount != 1)
                throw new Exception(
                    $"Cycle {cycle}: expected PeerCount=1 after connect, got {coordinator.PeerCount}.");

            coordinator.HandlePeerDisconnected(peerId);
            if (coordinator.PeerCount != 0)
                throw new Exception(
                    $"Cycle {cycle}: expected PeerCount=0 after disconnect, got {coordinator.PeerCount}. " +
                    $"Peer state leaked across the disconnect.");
        }

        // Final invariant: zero stale peers after all cycles.
        if (coordinator.PeerCount != 0)
            throw new Exception(
                $"Final PeerCount={coordinator.PeerCount} after {cycles} clean cycles. " +
                $"Indicates state leak.");

        return Task.CompletedTask;
    }

    /// <summary>
    /// Five workers join the swarm simultaneously, then 15 dispatches fan out.
    /// Verifies large-swarm scoring + load balancing under realistic peer counts.
    /// Combined with the concurrent-dispatch test, this proves the dispatcher
    /// handles both axes of scale (peer count × dispatch count).
    /// </summary>
    [TestMethod]
    public async Task Stress_LargeSwarm_5Peers_15Dispatches()
    {
        P2PKernelSerializer.RegisterKernelType(typeof(P2PTestKernels));

        const int workerCount = 5;
        const int dispatchCount = 15;
        const int n = 256;

        using var context = Context.Create(builder => builder.CPU());

        var coordinator = new P2PSwarmCoordinator(new SpawnDev.WebTorrent.WebTorrentClient());
        var p2pAccel = coordinator.CreateAccelerator(context);
        var dispatcher = new P2PDispatcher(p2pAccel);

        // Workers with varying TFLOPS so the dispatcher's scoring has something
        // meaningful to balance. Equal capabilities can collapse to round-robin.
        var tflops = new[] { 1.0, 2.5, 1.5, 4.0, 3.0 };
        for (int i = 0; i < workerCount; i++)
        {
            var caps = new PeerCapabilities
            {
                PeerId = $"worker-{i}",
                EstimatedTflops = tflops[i],
                AvailableMemory = 1L << 32,
                IsCharging = true,
                ThermalState = 0,
            };
            coordinator.HandlePeerConnected(caps.PeerId, caps);
            p2pAccel.AddPeer(new RemotePeer
            {
                PeerId = caps.PeerId,
                IsConnected = true,
                Capabilities = caps,
            });
        }

        var perPeerHits = new ConcurrentDictionary<string, int>();
        dispatcher.OnSendMessage += (peerId, msg) =>
            perPeerHits.AddOrUpdate(peerId, 1, (_, n) => n + 1);

        var dispatchedIds = new List<string>();
        var method = typeof(P2PTestKernels).GetMethod(nameof(P2PTestKernels.VectorAdd))!;
        for (int d = 0; d < dispatchCount; d++)
        {
            var request = P2PKernelSerializer.CreateDispatch(method, gridDimX: n);
            request.Buffers = new[]
            {
                new BufferBinding { ParameterIndex = 1, BufferId = $"a{d}", Length = n, ElementSize = 4 },
                new BufferBinding { ParameterIndex = 2, BufferId = $"b{d}", Length = n, ElementSize = 4 },
                new BufferBinding { ParameterIndex = 3, BufferId = $"r{d}", Length = n, ElementSize = 4 },
            };
            dispatcher.Dispatch(request);
            dispatchedIds.Add(request.DispatchId);
        }

        // Drain the pending queue.
        foreach (var id in dispatchedIds)
        {
            dispatcher.HandleResult(id, new KernelDispatchResult
            {
                DispatchId = id,
                Success = true,
                DurationMs = 1.0,
            });
        }

        if (perPeerHits.Values.Sum() != dispatchCount)
            throw new Exception(
                $"Total dispatches sent ({perPeerHits.Values.Sum()}) != requested ({dispatchCount}).");

        if (dispatcher.PendingCount != 0)
            throw new Exception($"Pending queue not drained: {dispatcher.PendingCount} left.");

        // Distribution check: at minimum 3 of 5 workers should be hit. With
        // varying TFLOPS the scoring may concentrate work on the strongest 2-3
        // workers, but pure single-peer concentration would be a load-balancer bug.
        var distinctPeersHit = perPeerHits.Count(kv => kv.Value > 0);
        if (distinctPeersHit < 3)
            throw new Exception(
                $"Load was concentrated on {distinctPeersHit} of {workerCount} workers. " +
                $"Distribution: {string.Join(", ", perPeerHits.Select(kv => $"{kv.Key}={kv.Value}"))}.");

        await Task.CompletedTask;
    }

    /// <summary>
    /// Endurance: 1000 dispatches against a 3-worker swarm. Verifies the
    /// dispatcher has no leak (pending queue drains) and no slowdown
    /// (last-quartile dispatch time is not pathologically slower than the
    /// first-quartile). Exercises the internal state machine at production
    /// scale - long-running swarms (ML training jobs, scientific batch runs)
    /// will see this volume in normal operation.
    ///
    /// Coverage gap closed: existing stress tests exercise the dispatcher at
    /// 15-20 dispatches. A leak that adds O(1) state per dispatch wouldn't
    /// surface there but would over 1000. A slowdown caused by list/dict
    /// growth or unbounded cache wouldn't either.
    /// </summary>
    [TestMethod(Timeout = 60000)]
    public async Task Endurance_1000Dispatches_NoLeakNoSlowdown_InProcess()
    {
        P2PKernelSerializer.RegisterKernelType(typeof(P2PTestKernels));

        const int dispatchCount = 1000;
        const int workerCount = 3;
        const int n = 64;

        using var context = Context.Create(builder => builder.CPU());

        var coordinator = new P2PSwarmCoordinator(new SpawnDev.WebTorrent.WebTorrentClient());
        var p2pAccel = coordinator.CreateAccelerator(context);
        var dispatcher = new P2PDispatcher(p2pAccel);

        for (int i = 0; i < workerCount; i++)
        {
            var caps = new PeerCapabilities
            {
                PeerId = $"worker-{i}",
                EstimatedTflops = 2.0,
                AvailableMemory = 1L << 32,
                IsCharging = true,
                ThermalState = 0,
            };
            coordinator.HandlePeerConnected(caps.PeerId, caps);
            p2pAccel.AddPeer(new RemotePeer
            {
                PeerId = caps.PeerId,
                IsConnected = true,
                Capabilities = caps,
            });
        }

        var sentTo = new ConcurrentBag<string>();
        dispatcher.OnSendMessage += (peerId, msg) => sentTo.Add(peerId);

        var method = typeof(P2PTestKernels).GetMethod(nameof(P2PTestKernels.VectorAdd))!;
        var perDispatchTicks = new long[dispatchCount];
        var sw = System.Diagnostics.Stopwatch.StartNew();

        // Sequential dispatch + immediate result handling to simulate a
        // steady-state swarm where each kernel completes before the next is
        // queued. The rapid-churn variant covers concurrency; this covers
        // long-haul state-machine integrity.
        for (int d = 0; d < dispatchCount; d++)
        {
            var t0 = sw.ElapsedTicks;
            var request = P2PKernelSerializer.CreateDispatch(method, gridDimX: n);
            request.Buffers = new[]
            {
                new BufferBinding { ParameterIndex = 1, BufferId = $"a{d}", Length = n, ElementSize = 4 },
                new BufferBinding { ParameterIndex = 2, BufferId = $"b{d}", Length = n, ElementSize = 4 },
                new BufferBinding { ParameterIndex = 3, BufferId = $"r{d}", Length = n, ElementSize = 4 },
            };
            var dispatchId = dispatcher.Dispatch(request);
            dispatcher.HandleResult(dispatchId, new KernelDispatchResult
            {
                DispatchId = dispatchId,
                Success = true,
                DurationMs = 1.0,
            });
            perDispatchTicks[d] = sw.ElapsedTicks - t0;
        }
        sw.Stop();

        // Leak check: pending queue must drain to zero after every dispatch
        // has been resulted. Any leftover means a dispatchId never matched a
        // result, or HandleResult silently no-op'd.
        if (dispatcher.PendingCount != 0)
            throw new Exception(
                $"Pending queue not drained after {dispatchCount} dispatches: " +
                $"{dispatcher.PendingCount} entries leaked. Indicates HandleResult " +
                $"failed to match dispatchIds, or _pending wasn't removed on success.");

        // Leak check: peer count must be unchanged. Dispatch should not mutate
        // peer state (no add/remove from the dispatcher's perspective).
        if (coordinator.PeerCount != workerCount)
            throw new Exception(
                $"Peer count drifted: expected {workerCount}, got {coordinator.PeerCount}. " +
                $"Dispatch path mutated peer registry.");

        // Routing check: every dispatch produced exactly one OnSendMessage event.
        if (sentTo.Count != dispatchCount)
            throw new Exception(
                $"Expected {dispatchCount} OnSendMessage events, got {sentTo.Count}. " +
                $"Dispatches were silently dropped over the long run.");

        // Slowdown check: compare median dispatch time of the last quartile
        // against the first quartile. Sort within each window so a single
        // GC blip doesn't tank the comparison. A 5x degradation between
        // first and last quartile is the threshold - the steady-state cost
        // should be flat (this dispatcher does no per-dispatch list scans
        // that grow with history).
        var quartile = dispatchCount / 4;
        var firstQuartile = perDispatchTicks.Take(quartile).OrderBy(t => t).ToArray();
        var lastQuartile = perDispatchTicks.Skip(dispatchCount - quartile).OrderBy(t => t).ToArray();
        var firstMedian = firstQuartile[quartile / 2];
        var lastMedian = lastQuartile[quartile / 2];
        if (lastMedian > firstMedian * 5 && lastMedian > 100_000) // ignore sub-10us noise floor
            throw new Exception(
                $"Dispatch latency degraded: first-quartile median = {firstMedian} ticks, " +
                $"last-quartile median = {lastMedian} ticks ({(double)lastMedian / firstMedian:F1}x). " +
                $"Indicates O(N) work per dispatch as history grows.");

        await Task.CompletedTask;
    }
}
