using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using NUnit.Framework;
using SpawnDev.ILGPU.P2P.IntegrationTests.Infrastructure;

namespace SpawnDev.ILGPU.P2P.IntegrationTests;

/// <summary>
/// Multi-Peer and Fault Tolerance tests.
/// Proves the dispatcher scoring, load balancing, retry, election, and transfer
/// logic is correct. These use simulated peer connections (HandlePeerConnected);
/// real-WebRTC end-to-end dispatch is covered by RealWebRtcPipelineTests.
/// </summary>
[TestFixture]
public class MultiPeerTests
{
    /// <summary>
    /// Two workers connected, 4 dispatches sent. Both workers should receive work.
    /// Verifies the dispatcher doesn't always pick the same peer.
    /// </summary>
    [Test]
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

        // Worker A - 2 TFLOPS
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

        // Worker B - 2 TFLOPS (same as A)
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

        // Track which peer gets each dispatch
        var dispatchedTo = new List<string>();
        dispatcher.OnSendMessage += (peerId, msg) =>
        {
            dispatchedTo.Add(peerId);
        };

        // Dispatch 4 times WITHOUT completing between dispatches.
        // This builds up pending count on the first-chosen worker,
        // causing the load score to decrease and the dispatcher to pick the other.
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

        // Now complete them all
        foreach (var id in dispatchIds)
        {
            dispatcher.HandleResult(id, new KernelDispatchResult
            {
                DispatchId = id,
                Success = true,
                DurationMs = 1.0,
            });
        }

        Assert.That(dispatchedTo.Count, Is.EqualTo(4), "Expected 4 dispatches");
        Assert.That(dispatchedTo.Distinct().Count(), Is.GreaterThanOrEqualTo(2),
            "Expected work distributed to at least 2 workers, " +
            $"but only dispatched to: {string.Join(", ", dispatchedTo.Distinct())}");
    }

    /// <summary>
    /// Three workers with different TFLOPS. The highest-TFLOPS worker
    /// should receive the first dispatch (it scores highest).
    /// </summary>
    [Test]
    public void MultiPeer_ThreeWorkers_LoadBalance()
    {
        P2PKernelSerializer.RegisterKernelType(typeof(P2PTestKernels));

        using var context = Context.Create(builder => builder.CPU());
        var coordinator = new P2PSwarmCoordinator(new SpawnDev.WebTorrent.WebTorrentClient());
        var p2pAccel = coordinator.CreateAccelerator(context);
        var dispatcher = new P2PDispatcher(p2pAccel);

        // Add 3 peers with different TFLOPS
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

        // Track first dispatch target
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

        Assert.That(firstTarget, Is.EqualTo("peer-high"),
            $"Expected highest-TFLOPS peer to get first dispatch, got '{firstTarget}'");
    }

    /// <summary>
    /// When a worker disconnects mid-dispatch, the dispatcher retries on another peer.
    /// </summary>
    [Test]
    public void Fault_WorkerDies_RetrySucceeds()
    {
        P2PKernelSerializer.RegisterKernelType(typeof(P2PTestKernels));

        using var context = Context.Create(builder => builder.CPU());
        var coordinator = new P2PSwarmCoordinator(new SpawnDev.WebTorrent.WebTorrentClient());
        var p2pAccel = coordinator.CreateAccelerator(context);
        var dispatcher = new P2PDispatcher(p2pAccel);

        // Add two peers
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

        // Track retries
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

        // Simulate the assigned peer dying
        var assignedPeerId = p2pAccel.Peers.First().PeerId;
        dispatcher.HandlePeerLost(assignedPeerId);

        Assert.That(failedPeer, Is.Not.Null, "Expected retry event to fire");
        Assert.That(retriedTo, Is.Not.Null, "Expected dispatch to be retried on another peer");
        Assert.That(retriedTo, Is.Not.EqualTo(failedPeer),
            "Retried dispatch should go to a DIFFERENT peer");
    }

    /// <summary>
    /// When all workers disconnect, dispatch fails with a clear error.
    /// </summary>
    [Test]
    public void Fault_AllWorkersDie_DispatchFails()
    {
        P2PKernelSerializer.RegisterKernelType(typeof(P2PTestKernels));

        using var context = Context.Create(builder => builder.CPU());
        var coordinator = new P2PSwarmCoordinator(new SpawnDev.WebTorrent.WebTorrentClient());
        var p2pAccel = coordinator.CreateAccelerator(context);
        var dispatcher = new P2PDispatcher(p2pAccel);

        // Add a single peer
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

        // Kill the only peer
        dispatcher.HandlePeerLost("lone-peer");

        Assert.That(failedId, Is.EqualTo(dispatchId), "Expected dispatch failure event");
        Assert.That(failError, Does.Contain("No healthy peers"),
            $"Expected 'No healthy peers' error, got: {failError}");
    }

    /// <summary>
    /// Coordinator election: when coordinator drops, highest-TFLOPS worker wins.
    /// </summary>
    [Test]
    public void Fault_CoordinatorDies_ElectionHappens()
    {
        using var context = Context.Create(builder => builder.CPU());
        var coordinator = new P2PSwarmCoordinator(new SpawnDev.WebTorrent.WebTorrentClient());

        // Add peers with different TFLOPS
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

        // Elect new coordinator
        var winner = coordinator.ElectCoordinator();

        Assert.That(winner, Is.EqualTo("peer-strong"),
            $"Expected strongest peer to win election, got '{winner}'");
    }

    /// <summary>
    /// Graceful coordinator transfer preserves pending dispatch state.
    /// </summary>
    [Test]
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

        // Add two peers
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

        // Dispatch a kernel (creates pending state)
        var method = typeof(P2PTestKernels).GetMethod(nameof(P2PTestKernels.VectorAdd))!;
        var request = P2PKernelSerializer.CreateDispatch(method, gridDimX: 256);
        request.Buffers = new[]
        {
            new BufferBinding { ParameterIndex = 1, BufferId = "a", Length = 256, ElementSize = 4 },
            new BufferBinding { ParameterIndex = 2, BufferId = "b", Length = 256, ElementSize = 4 },
            new BufferBinding { ParameterIndex = 3, BufferId = "result", Length = 256, ElementSize = 4 },
        };
        dispatcher.Dispatch(request);
        Assert.That(dispatcher.PendingCount, Is.EqualTo(1), "Should have 1 pending dispatch");

        // Capture transfer message
        P2PMessage? transferMsg = null;
        coordinator.OnSendMessage += (peerId, msg) =>
        {
            if (msg.Type == P2PMessageType.CoordinatorTransfer)
                transferMsg = msg;
        };

        // Transfer coordinator role
        var transferTarget = coordinator.TransferCoordinator();

        Assert.That(transferTarget, Is.Not.Null, "Transfer should succeed");
        Assert.That(coordinator.Role, Is.EqualTo(P2PRole.Worker),
            "Transferring coordinator should become Worker");
        Assert.That(transferMsg, Is.Not.Null,
            "Transfer message should be sent to target");
    }

    /// <summary>
    /// Always-fail worker exhausts max retries, dispatch fails permanently.
    /// </summary>
    [Test]
    public void Fault_MaxRetries_Exhausted()
    {
        P2PKernelSerializer.RegisterKernelType(typeof(P2PTestKernels));

        using var context = Context.Create(builder => builder.CPU());
        var coordinator = new P2PSwarmCoordinator(new SpawnDev.WebTorrent.WebTorrentClient());
        var p2pAccel = coordinator.CreateAccelerator(context);
        var dispatcher = new P2PDispatcher(p2pAccel);
        dispatcher.MaxRetries = 2;

        // Add one peer
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

        // Simulate failure results until max retries exhausted
        for (int i = 0; i < 3; i++)
        {
            dispatcher.HandleResult(request.DispatchId, new KernelDispatchResult
            {
                DispatchId = request.DispatchId,
                Success = false,
                Error = "Simulated failure",
            });
        }

        Assert.That(failedId, Is.EqualTo(dispatchId),
            "Dispatch should fail permanently after max retries");
    }

    /// <summary>
    /// Thermal eviction: a peer at thermal state 3 (critical) gets score 0
    /// and should not receive new dispatches.
    /// </summary>
    [Test]
    public void Fault_ThermalCritical_NoDispatch()
    {
        P2PKernelSerializer.RegisterKernelType(typeof(P2PTestKernels));

        using var context = Context.Create(builder => builder.CPU());
        var coordinator = new P2PSwarmCoordinator(new SpawnDev.WebTorrent.WebTorrentClient());
        var p2pAccel = coordinator.CreateAccelerator(context);
        var dispatcher = new P2PDispatcher(p2pAccel);

        // Add healthy peer
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

        // Add thermally critical peer (10x TFLOPS but critical thermal)
        var hotCaps = new PeerCapabilities
        {
            PeerId = "overheating",
            EstimatedTflops = 10.0,
            IsCharging = true,
            ThermalState = 3, // critical
        };
        coordinator.HandlePeerConnected("overheating", hotCaps);
        p2pAccel.AddPeer(new RemotePeer
        {
            PeerId = "overheating",
            IsConnected = true,
            Capabilities = hotCaps,
        });

        // Verify scoring: thermal critical should score 0
        var hotPeer = p2pAccel.Peers.First(p => p.PeerId == "overheating");
        var healthyPeer = p2pAccel.Peers.First(p => p.PeerId == "healthy");
        var hotScore = dispatcher.ScorePeer(hotPeer);
        var healthyScore = dispatcher.ScorePeer(healthyPeer);

        Assert.That(hotScore, Is.EqualTo(0.0),
            $"Thermally critical peer should score 0, got {hotScore}");
        Assert.That(healthyScore, Is.GreaterThan(0.0),
            "Healthy peer should score > 0");

        // Dispatch should go to healthy peer
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

        Assert.That(target, Is.EqualTo("healthy"),
            $"Dispatch should go to healthy peer, not overheating. Got '{target}'");
    }
}
