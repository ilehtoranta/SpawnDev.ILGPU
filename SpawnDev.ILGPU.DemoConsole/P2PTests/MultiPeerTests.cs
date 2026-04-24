using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using SpawnDev.ILGPU.P2P;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.DemoConsole.P2PTests;

/// <summary>
/// Multi-Peer and Fault Tolerance tests.
/// Proves the dispatcher scoring, load balancing, retry, election, and transfer
/// logic is correct. These use simulated peer connections (HandlePeerConnected);
/// real-WebRTC end-to-end dispatch is covered by RealWebRtcPipelineTests.
/// </summary>
public class MultiPeerTests
{
    /// <summary>
    /// Two workers connected, 4 dispatches sent. Both workers should receive work.
    /// Verifies the dispatcher doesn't always pick the same peer.
    /// </summary>
    [TestMethod]
    public async Task MultiPeer_TwoWorkers_BothReceive()
    {
        P2PKernelSerializer.RegisterKernelType(typeof(P2PTestKernels));

        using var context = Context.Create(builder => builder.CPU());
        using var accel1 = context.CreateCPUAccelerator(0);
        using var accel2 = context.CreateCPUAccelerator(0);

        var coordinator = new P2PSwarmCoordinator(new SpawnDev.WebTorrent.WebTorrentClient());
        var p2pAccel = coordinator.CreateAccelerator(context);
        var dispatcher = new P2PDispatcher(p2pAccel);
        var transport = new P2PTransport(
            new SpawnDev.WebTorrent.WebTorrentClient(), coordinator, dispatcher);

        var workerA = new P2PWorker(transport);
        workerA.Initialize(context, accel1);
        var capsA = workerA.BuildCapabilities("worker-A");
        capsA.EstimatedTflops = 2.0;
        coordinator.HandlePeerConnected("worker-A", capsA);
        p2pAccel.AddPeer(new RemotePeer
        {
            PeerId = "worker-A",
            IsConnected = true,
            Capabilities = capsA,
        });

        var workerB = new P2PWorker(transport);
        workerB.Initialize(context, accel2);
        var capsB = workerB.BuildCapabilities("worker-B");
        capsB.EstimatedTflops = 2.0;
        coordinator.HandlePeerConnected("worker-B", capsB);
        p2pAccel.AddPeer(new RemotePeer
        {
            PeerId = "worker-B",
            IsConnected = true,
            Capabilities = capsB,
        });

        var dispatchedTo = new List<string>();
        dispatcher.OnSendMessage += (peerId, msg) =>
        {
            dispatchedTo.Add(peerId);
        };

        int n = 256;
        var dispatchIds = new List<string>();
        for (int d = 0; d < 4; d++)
        {
            var method = typeof(P2PTestKernels).GetMethod(nameof(P2PTestKernels.VectorAdd))!;
            var request = P2PKernelSerializer.CreateDispatch(method, gridDimX: n);
            request.Buffers = new[]
            {
                new BufferBinding { ParameterIndex = 1, BufferId = $"a{d}", Length = n, ElementSize = 4 },
                new BufferBinding { ParameterIndex = 2, BufferId = $"b{d}", Length = n, ElementSize = 4 },
                new BufferBinding { ParameterIndex = 3, BufferId = $"result{d}", Length = n, ElementSize = 4 },
            };

            dispatcher.Dispatch(request);
            dispatchIds.Add(request.DispatchId);
        }

        foreach (var id in dispatchIds)
        {
            dispatcher.HandleResult(id, new KernelDispatchResult
            {
                DispatchId = id,
                Success = true,
                DurationMs = 1.0,
            });
        }

        if (dispatchedTo.Count != 4)
            throw new Exception($"Expected 4 dispatches, got {dispatchedTo.Count}");
        if (dispatchedTo.Distinct().Count() < 2)
            throw new Exception(
                "Expected work distributed to at least 2 workers, " +
                $"but only dispatched to: {string.Join(", ", dispatchedTo.Distinct())}");
        await Task.CompletedTask;
    }

    /// <summary>
    /// Three workers with different TFLOPS. The highest-TFLOPS worker
    /// should receive the first dispatch (it scores highest).
    /// </summary>
    [TestMethod]
    public Task MultiPeer_ThreeWorkers_LoadBalance()
    {
        P2PKernelSerializer.RegisterKernelType(typeof(P2PTestKernels));

        using var context = Context.Create(builder => builder.CPU());
        var coordinator = new P2PSwarmCoordinator(new SpawnDev.WebTorrent.WebTorrentClient());
        var p2pAccel = coordinator.CreateAccelerator(context);
        var dispatcher = new P2PDispatcher(p2pAccel);

        var tflopsValues = new[] { 1.0, 5.0, 10.0 };
        var peerIds = new[] { "peer-low", "peer-mid", "peer-high" };
        for (int i = 0; i < 3; i++)
        {
            var caps = new PeerCapabilities
            {
                PeerId = peerIds[i],
                EstimatedTflops = tflopsValues[i],
                AvailableMemory = 4L * 1024 * 1024 * 1024,
                IsCharging = true,
                ThermalState = 0,
            };
            coordinator.HandlePeerConnected(peerIds[i], caps);
            p2pAccel.AddPeer(new RemotePeer
            {
                PeerId = peerIds[i],
                IsConnected = true,
                Capabilities = caps,
            });
        }

        string? firstTarget = null;
        dispatcher.OnSendMessage += (peerId, msg) =>
        {
            firstTarget ??= peerId;
        };

        var method = typeof(P2PTestKernels).GetMethod(nameof(P2PTestKernels.VectorAdd))!;
        var request = P2PKernelSerializer.CreateDispatch(method, gridDimX: 256);
        request.Buffers = new[]
        {
            new BufferBinding { ParameterIndex = 1, BufferId = "a", Length = 256, ElementSize = 4 },
            new BufferBinding { ParameterIndex = 2, BufferId = "b", Length = 256, ElementSize = 4 },
            new BufferBinding { ParameterIndex = 3, BufferId = "result", Length = 256, ElementSize = 4 },
        };
        dispatcher.Dispatch(request);

        if (firstTarget != "peer-high")
            throw new Exception($"Expected highest-TFLOPS peer to get first dispatch, got '{firstTarget}'");
        return Task.CompletedTask;
    }

    /// <summary>
    /// When a worker disconnects mid-dispatch, the dispatcher retries on another peer.
    /// </summary>
    [TestMethod]
    public Task Fault_WorkerDies_RetrySucceeds()
    {
        P2PKernelSerializer.RegisterKernelType(typeof(P2PTestKernels));

        using var context = Context.Create(builder => builder.CPU());
        var coordinator = new P2PSwarmCoordinator(new SpawnDev.WebTorrent.WebTorrentClient());
        var p2pAccel = coordinator.CreateAccelerator(context);
        var dispatcher = new P2PDispatcher(p2pAccel);

        for (int i = 0; i < 2; i++)
        {
            var id = $"peer-{i}";
            var caps = new PeerCapabilities
            {
                PeerId = id,
                EstimatedTflops = 2.0,
                IsCharging = true,
                ThermalState = 0,
            };
            coordinator.HandlePeerConnected(id, caps);
            p2pAccel.AddPeer(new RemotePeer
            {
                PeerId = id,
                IsConnected = true,
                Capabilities = caps,
            });
        }

        string? retriedTo = null;
        string? failedPeer = null;
        dispatcher.OnDispatchRetried += (id, failed, newPeer) =>
        {
            failedPeer = failed;
            retriedTo = newPeer;
        };

        var method = typeof(P2PTestKernels).GetMethod(nameof(P2PTestKernels.VectorAdd))!;
        var request = P2PKernelSerializer.CreateDispatch(method, gridDimX: 256);
        request.Buffers = new[]
        {
            new BufferBinding { ParameterIndex = 1, BufferId = "a", Length = 256, ElementSize = 4 },
            new BufferBinding { ParameterIndex = 2, BufferId = "b", Length = 256, ElementSize = 4 },
            new BufferBinding { ParameterIndex = 3, BufferId = "result", Length = 256, ElementSize = 4 },
        };
        var dispatchId = dispatcher.Dispatch(request);

        var assignedPeerId = p2pAccel.Peers.First().PeerId;
        dispatcher.HandlePeerLost(assignedPeerId);

        if (failedPeer == null) throw new Exception("Expected retry event to fire");
        if (retriedTo == null) throw new Exception("Expected dispatch to be retried on another peer");
        if (retriedTo == failedPeer) throw new Exception("Retried dispatch should go to a DIFFERENT peer");
        return Task.CompletedTask;
    }

    /// <summary>
    /// When all workers disconnect, dispatch fails with a clear error.
    /// </summary>
    [TestMethod]
    public Task Fault_AllWorkersDie_DispatchFails()
    {
        P2PKernelSerializer.RegisterKernelType(typeof(P2PTestKernels));

        using var context = Context.Create(builder => builder.CPU());
        var coordinator = new P2PSwarmCoordinator(new SpawnDev.WebTorrent.WebTorrentClient());
        var p2pAccel = coordinator.CreateAccelerator(context);
        var dispatcher = new P2PDispatcher(p2pAccel);

        var caps = new PeerCapabilities
        {
            PeerId = "lone-peer",
            EstimatedTflops = 2.0,
            IsCharging = true,
            ThermalState = 0,
        };
        coordinator.HandlePeerConnected("lone-peer", caps);
        p2pAccel.AddPeer(new RemotePeer
        {
            PeerId = "lone-peer",
            IsConnected = true,
            Capabilities = caps,
        });

        string? failedId = null;
        string? failError = null;
        dispatcher.OnDispatchFailed += (id, error) =>
        {
            failedId = id;
            failError = error;
        };

        var method = typeof(P2PTestKernels).GetMethod(nameof(P2PTestKernels.VectorAdd))!;
        var request = P2PKernelSerializer.CreateDispatch(method, gridDimX: 256);
        request.Buffers = new[]
        {
            new BufferBinding { ParameterIndex = 1, BufferId = "a", Length = 256, ElementSize = 4 },
            new BufferBinding { ParameterIndex = 2, BufferId = "b", Length = 256, ElementSize = 4 },
            new BufferBinding { ParameterIndex = 3, BufferId = "result", Length = 256, ElementSize = 4 },
        };
        var dispatchId = dispatcher.Dispatch(request);

        dispatcher.HandlePeerLost("lone-peer");

        if (failedId != dispatchId) throw new Exception($"Expected dispatch failure event, got failedId={failedId}");
        if (failError == null || !failError.Contains("No healthy peers"))
            throw new Exception($"Expected 'No healthy peers' error, got: {failError}");
        return Task.CompletedTask;
    }

    /// <summary>
    /// Coordinator election: when coordinator drops, highest-TFLOPS worker wins.
    /// </summary>
    [TestMethod]
    public Task Fault_CoordinatorDies_ElectionHappens()
    {
        using var context = Context.Create(builder => builder.CPU());
        var coordinator = new P2PSwarmCoordinator(new SpawnDev.WebTorrent.WebTorrentClient());

        coordinator.HandlePeerConnected("peer-weak", new PeerCapabilities
        {
            PeerId = "peer-weak",
            EstimatedTflops = 1.0,
            IsCharging = true,
        });
        coordinator.HandlePeerConnected("peer-strong", new PeerCapabilities
        {
            PeerId = "peer-strong",
            EstimatedTflops = 10.0,
            IsCharging = true,
        });

        var winner = coordinator.ElectCoordinator();

        if (winner != "peer-strong")
            throw new Exception($"Expected strongest peer to win election, got '{winner}'");
        return Task.CompletedTask;
    }

    /// <summary>
    /// Graceful coordinator transfer preserves pending dispatch state.
    /// </summary>
    [TestMethod]
    public async Task Fault_GracefulTransfer_PendingPreserved()
    {
        P2PKernelSerializer.RegisterKernelType(typeof(P2PTestKernels));

        using var context = Context.Create(builder => builder.CPU());
        var client = new SpawnDev.WebTorrent.WebTorrentClient();
        var coordinator = new P2PSwarmCoordinator(client);
        await coordinator.CreateSwarmAsync("transfer-test");
        var p2pAccel = coordinator.CreateAccelerator(context);
        var dispatcher = new P2PDispatcher(p2pAccel);
        p2pAccel.Dispatcher = dispatcher;

        for (int i = 0; i < 2; i++)
        {
            var id = $"peer-{i}";
            var caps = new PeerCapabilities
            {
                PeerId = id,
                EstimatedTflops = 2.0 + i,
                IsCharging = true,
                ThermalState = 0,
            };
            coordinator.HandlePeerConnected(id, caps);
            p2pAccel.AddPeer(new RemotePeer
            {
                PeerId = id,
                IsConnected = true,
                Capabilities = caps,
            });
        }

        var method = typeof(P2PTestKernels).GetMethod(nameof(P2PTestKernels.VectorAdd))!;
        var request = P2PKernelSerializer.CreateDispatch(method, gridDimX: 256);
        request.Buffers = new[]
        {
            new BufferBinding { ParameterIndex = 1, BufferId = "a", Length = 256, ElementSize = 4 },
            new BufferBinding { ParameterIndex = 2, BufferId = "b", Length = 256, ElementSize = 4 },
            new BufferBinding { ParameterIndex = 3, BufferId = "result", Length = 256, ElementSize = 4 },
        };
        dispatcher.Dispatch(request);
        if (dispatcher.PendingCount != 1) throw new Exception($"Should have 1 pending dispatch, got {dispatcher.PendingCount}");

        P2PMessage? transferMsg = null;
        coordinator.OnSendMessage += (peerId, msg) =>
        {
            if (msg.Type == P2PMessageType.CoordinatorTransfer)
                transferMsg = msg;
        };

        var transferTarget = coordinator.TransferCoordinator();

        if (transferTarget == null) throw new Exception("Transfer should succeed");
        if (coordinator.Role != P2PRole.Worker)
            throw new Exception($"Transferring coordinator should become Worker, got {coordinator.Role}");
        if (transferMsg == null) throw new Exception("Transfer message should be sent to target");
    }

    /// <summary>
    /// Always-fail worker exhausts max retries, dispatch fails permanently.
    /// </summary>
    [TestMethod]
    public Task Fault_MaxRetries_Exhausted()
    {
        P2PKernelSerializer.RegisterKernelType(typeof(P2PTestKernels));

        using var context = Context.Create(builder => builder.CPU());
        var coordinator = new P2PSwarmCoordinator(new SpawnDev.WebTorrent.WebTorrentClient());
        var p2pAccel = coordinator.CreateAccelerator(context);
        var dispatcher = new P2PDispatcher(p2pAccel);
        dispatcher.MaxRetries = 2;

        var caps = new PeerCapabilities
        {
            PeerId = "fail-peer",
            EstimatedTflops = 2.0,
            IsCharging = true,
            ThermalState = 0,
        };
        coordinator.HandlePeerConnected("fail-peer", caps);
        p2pAccel.AddPeer(new RemotePeer
        {
            PeerId = "fail-peer",
            IsConnected = true,
            Capabilities = caps,
        });

        string? failedId = null;
        dispatcher.OnDispatchFailed += (id, error) => { failedId = id; };

        var method = typeof(P2PTestKernels).GetMethod(nameof(P2PTestKernels.VectorAdd))!;
        var request = P2PKernelSerializer.CreateDispatch(method, gridDimX: 256);
        request.Buffers = new[]
        {
            new BufferBinding { ParameterIndex = 1, BufferId = "a", Length = 256, ElementSize = 4 },
            new BufferBinding { ParameterIndex = 2, BufferId = "b", Length = 256, ElementSize = 4 },
            new BufferBinding { ParameterIndex = 3, BufferId = "result", Length = 256, ElementSize = 4 },
        };
        var dispatchId = dispatcher.Dispatch(request);

        for (int i = 0; i < 3; i++)
        {
            dispatcher.HandleResult(request.DispatchId, new KernelDispatchResult
            {
                DispatchId = request.DispatchId,
                Success = false,
                Error = "Simulated failure",
            });
        }

        if (failedId != dispatchId)
            throw new Exception($"Dispatch should fail permanently after max retries, failedId={failedId}");
        return Task.CompletedTask;
    }

    /// <summary>
    /// Thermal eviction: a peer at thermal state 3 (critical) gets score 0
    /// and should not receive new dispatches.
    /// </summary>
    [TestMethod]
    public Task Fault_ThermalCritical_NoDispatch()
    {
        P2PKernelSerializer.RegisterKernelType(typeof(P2PTestKernels));

        using var context = Context.Create(builder => builder.CPU());
        var coordinator = new P2PSwarmCoordinator(new SpawnDev.WebTorrent.WebTorrentClient());
        var p2pAccel = coordinator.CreateAccelerator(context);
        var dispatcher = new P2PDispatcher(p2pAccel);

        var healthyCaps = new PeerCapabilities
        {
            PeerId = "healthy",
            EstimatedTflops = 1.0,
            IsCharging = true,
            ThermalState = 0,
        };
        coordinator.HandlePeerConnected("healthy", healthyCaps);
        p2pAccel.AddPeer(new RemotePeer
        {
            PeerId = "healthy",
            IsConnected = true,
            Capabilities = healthyCaps,
        });

        var hotCaps = new PeerCapabilities
        {
            PeerId = "overheating",
            EstimatedTflops = 10.0,
            IsCharging = true,
            ThermalState = 3,
        };
        coordinator.HandlePeerConnected("overheating", hotCaps);
        p2pAccel.AddPeer(new RemotePeer
        {
            PeerId = "overheating",
            IsConnected = true,
            Capabilities = hotCaps,
        });

        var hotPeer = p2pAccel.Peers.First(p => p.PeerId == "overheating");
        var healthyPeer = p2pAccel.Peers.First(p => p.PeerId == "healthy");
        var hotScore = dispatcher.ScorePeer(hotPeer);
        var healthyScore = dispatcher.ScorePeer(healthyPeer);

        if (hotScore != 0.0)
            throw new Exception($"Thermally critical peer should score 0, got {hotScore}");
        if (!(healthyScore > 0.0))
            throw new Exception("Healthy peer should score > 0");

        string? target = null;
        dispatcher.OnSendMessage += (peerId, msg) => { target ??= peerId; };

        var method = typeof(P2PTestKernels).GetMethod(nameof(P2PTestKernels.VectorAdd))!;
        var request = P2PKernelSerializer.CreateDispatch(method, gridDimX: 256);
        request.Buffers = new[]
        {
            new BufferBinding { ParameterIndex = 1, BufferId = "a", Length = 256, ElementSize = 4 },
            new BufferBinding { ParameterIndex = 2, BufferId = "b", Length = 256, ElementSize = 4 },
            new BufferBinding { ParameterIndex = 3, BufferId = "result", Length = 256, ElementSize = 4 },
        };
        dispatcher.Dispatch(request);

        if (target != "healthy")
            throw new Exception($"Dispatch should go to healthy peer, not overheating. Got '{target}'");
        return Task.CompletedTask;
    }
}
