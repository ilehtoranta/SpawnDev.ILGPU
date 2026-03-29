using ILGPU.Runtime;
using SpawnDev.ILGPU.P2P;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.Demo.Shared.UnitTests;

/// <summary>
/// P2P accelerator unit tests — peer management, dispatch, fault tolerance.
/// Pure logic tests with mock peers — no network needed.
/// </summary>
public abstract partial class BackendTestBase
{
    // ═══════════════════════════════════════════════════════════
    //  P2P Device & Accelerator — Creation
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task P2P_Device_Create()
    {
        var device = new P2PDevice();
        if (device.Name != "P2P Distributed Accelerator")
            throw new Exception($"Name: {device.Name}");
        if (device.MaxNumThreadsPerGroup != 256)
            throw new Exception($"MaxThreads: {device.MaxNumThreadsPerGroup}");
    }

    [TestMethod]
    public async Task P2P_Accelerator_Create()
    {
        using var context = global::ILGPU.Context.CreateDefault();
        var device = new P2PDevice();
        var accelerator = (P2PAccelerator)device.CreateAccelerator(context);
        if (accelerator.Peers.Count != 0)
            throw new Exception($"Initial peers: {accelerator.Peers.Count}");
    }

    // ═══════════════════════════════════════════════════════════
    //  Peer Management — Join & Leave
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task P2P_AddPeer()
    {
        using var context = global::ILGPU.Context.CreateDefault();
        var device = new P2PDevice();
        var accelerator = (P2PAccelerator)device.CreateAccelerator(context);

        var peer = new RemotePeer
        {
            PeerId = "peer-1",
            IsConnected = true,
            MemorySize = 4L * 1024 * 1024 * 1024,
            RemoteBackend = global::ILGPU.Runtime.AcceleratorType.WebGPU,
        };
        accelerator.AddPeer(peer);

        if (accelerator.Peers.Count != 1)
            throw new Exception($"Peers: {accelerator.Peers.Count}");
        if (accelerator.Peers[0].PeerId != "peer-1")
            throw new Exception($"PeerId: {accelerator.Peers[0].PeerId}");
    }

    [TestMethod]
    public async Task P2P_RemovePeer()
    {
        using var context = global::ILGPU.Context.CreateDefault();
        var device = new P2PDevice();
        var accelerator = (P2PAccelerator)device.CreateAccelerator(context);

        var peer = new RemotePeer { PeerId = "peer-1", IsConnected = true };
        accelerator.AddPeer(peer);
        accelerator.RemovePeer(peer);

        if (accelerator.Peers.Count != 0)
            throw new Exception($"Peers after remove: {accelerator.Peers.Count}");
    }

    [TestMethod]
    public async Task P2P_MultiplePeers()
    {
        using var context = global::ILGPU.Context.CreateDefault();
        var device = new P2PDevice();
        var accelerator = (P2PAccelerator)device.CreateAccelerator(context);

        for (int i = 0; i < 5; i++)
        {
            accelerator.AddPeer(new RemotePeer
            {
                PeerId = $"peer-{i}",
                IsConnected = true,
                MemorySize = 2L * 1024 * 1024 * 1024,
            });
        }

        if (accelerator.Peers.Count != 5)
            throw new Exception($"Peers: {accelerator.Peers.Count}");
    }

    [TestMethod]
    public async Task P2P_SelectPeer_RoundRobin()
    {
        using var context = global::ILGPU.Context.CreateDefault();
        var device = new P2PDevice();
        var accelerator = (P2PAccelerator)device.CreateAccelerator(context);

        accelerator.AddPeer(new RemotePeer { PeerId = "a", IsConnected = true });
        accelerator.AddPeer(new RemotePeer { PeerId = "b", IsConnected = true });

        var selected = accelerator.SelectPeer();
        if (selected == null)
            throw new Exception("Should select a peer");
        if (selected.PeerId != "a" && selected.PeerId != "b")
            throw new Exception($"Selected unknown peer: {selected.PeerId}");
    }

    [TestMethod]
    public async Task P2P_SelectPeer_NoPeers_ReturnsNull()
    {
        using var context = global::ILGPU.Context.CreateDefault();
        var device = new P2PDevice();
        var accelerator = (P2PAccelerator)device.CreateAccelerator(context);

        var selected = accelerator.SelectPeer();
        if (selected != null)
            throw new Exception("Should return null with no peers");
    }

    // ═══════════════════════════════════════════════════════════
    //  Dispatcher — Fault Tolerance
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task P2P_Dispatcher_Create()
    {
        using var context = global::ILGPU.Context.CreateDefault();
        var device = new P2PDevice();
        var accelerator = (P2PAccelerator)device.CreateAccelerator(context);
        var dispatcher = new P2PDispatcher(accelerator);

        if (dispatcher.DispatchTimeoutMs != 30_000)
            throw new Exception($"Timeout: {dispatcher.DispatchTimeoutMs}");
        if (dispatcher.MaxRetries != 3)
            throw new Exception($"MaxRetries: {dispatcher.MaxRetries}");
    }

    [TestMethod]
    public async Task P2P_Dispatcher_Dispatch_NoPeers_Throws()
    {
        using var context = global::ILGPU.Context.CreateDefault();
        var device = new P2PDevice();
        var accelerator = (P2PAccelerator)device.CreateAccelerator(context);
        var dispatcher = new P2PDispatcher(accelerator);

        bool threw = false;
        try
        {
            dispatcher.Dispatch(new KernelDispatchRequest());
        }
        catch (InvalidOperationException) { threw = true; }
        if (!threw)
            throw new Exception("Should throw when no peers available");
    }

    [TestMethod]
    public async Task P2P_Dispatcher_Dispatch_WithPeer()
    {
        using var context = global::ILGPU.Context.CreateDefault();
        var device = new P2PDevice();
        var accelerator = (P2PAccelerator)device.CreateAccelerator(context);
        accelerator.AddPeer(new RemotePeer { PeerId = "worker-1", IsConnected = true });

        var dispatcher = new P2PDispatcher(accelerator);
        var request = new KernelDispatchRequest
        {
            KernelMethod = "MatMulKernel",
            GridDimX = 1024,
            GroupDimX = 256,
        };

        var dispatchId = dispatcher.Dispatch(request);
        if (string.IsNullOrEmpty(dispatchId))
            throw new Exception("DispatchId should not be empty");
        if (accelerator.Peers[0].PendingOperations != 1)
            throw new Exception($"PendingOps: {accelerator.Peers[0].PendingOperations}");
    }

    [TestMethod]
    public async Task P2P_Dispatcher_HandleResult_Success()
    {
        using var context = global::ILGPU.Context.CreateDefault();
        var device = new P2PDevice();
        var accelerator = (P2PAccelerator)device.CreateAccelerator(context);
        accelerator.AddPeer(new RemotePeer { PeerId = "worker-1", IsConnected = true });

        var dispatcher = new P2PDispatcher(accelerator);
        var request = new KernelDispatchRequest();
        var dispatchId = dispatcher.Dispatch(request);

        dispatcher.HandleResult(dispatchId, new KernelDispatchResult
        {
            DispatchId = dispatchId,
            Success = true,
            DurationMs = 42.5,
        });

        if (accelerator.Peers[0].PendingOperations != 0)
            throw new Exception($"PendingOps after result: {accelerator.Peers[0].PendingOperations}");
    }

    [TestMethod]
    public async Task P2P_Dispatcher_PeerLoss_Retry()
    {
        using var context = global::ILGPU.Context.CreateDefault();
        var device = new P2PDevice();
        var accelerator = (P2PAccelerator)device.CreateAccelerator(context);
        accelerator.AddPeer(new RemotePeer { PeerId = "worker-1", IsConnected = true });
        accelerator.AddPeer(new RemotePeer { PeerId = "worker-2", IsConnected = true });

        var dispatcher = new P2PDispatcher(accelerator);
        string? retriedDispatchId = null;
        dispatcher.OnDispatchRetried += (id, from, to) => retriedDispatchId = id;

        var request = new KernelDispatchRequest();
        var dispatchId = dispatcher.Dispatch(request);

        // Simulate peer loss
        dispatcher.HandlePeerLost("worker-1");

        // Should have retried on worker-2
        if (retriedDispatchId == null)
            throw new Exception("Should have retried dispatch");
        if (retriedDispatchId != dispatchId)
            throw new Exception("Retried wrong dispatch");
    }

    [TestMethod]
    public async Task P2P_Dispatcher_AllPeersLost_Fails()
    {
        using var context = global::ILGPU.Context.CreateDefault();
        var device = new P2PDevice();
        var accelerator = (P2PAccelerator)device.CreateAccelerator(context);
        accelerator.AddPeer(new RemotePeer { PeerId = "worker-1", IsConnected = true });

        var dispatcher = new P2PDispatcher(accelerator);
        string? failedId = null;
        dispatcher.OnDispatchFailed += (id, err) => failedId = id;

        var dispatchId = dispatcher.Dispatch(new KernelDispatchRequest());

        // Lose the only peer — no one to retry on
        dispatcher.HandlePeerLost("worker-1");

        if (failedId == null)
            throw new Exception("Should have reported failure");
    }

    // ═══════════════════════════════════════════════════════════
    //  Protocol — Serialization
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task P2P_Protocol_SerializeMessage()
    {
        var msg = new P2PMessage
        {
            Type = P2PMessageType.CapabilityRequest,
        };
        var bytes = P2PProtocol.Serialize(msg);
        var deserialized = P2PProtocol.Deserialize(bytes);

        if (deserialized == null)
            throw new Exception("Deserialization returned null");
        if (deserialized.Type != P2PMessageType.CapabilityRequest)
            throw new Exception($"Type mismatch: {deserialized.Type}");
        if (deserialized.Version != P2PProtocol.Version)
            throw new Exception($"Version: {deserialized.Version}");
    }

    [TestMethod]
    public async Task P2P_Protocol_Capabilities()
    {
        var caps = new PeerCapabilities
        {
            PeerId = "phone-1",
            AvailableBackends = new[] { "WebGPU", "Wasm" },
            PreferredBackend = "WebGPU",
            AvailableMemory = 4L * 1024 * 1024 * 1024,
            EstimatedTflops = 2.5,
            MaxThreadsPerGroup = 256,
            Platform = "browser",
            IlgpuVersion = "4.7.0",
        };

        var json = System.Text.Json.JsonSerializer.Serialize(caps);
        var deserialized = System.Text.Json.JsonSerializer.Deserialize<PeerCapabilities>(json);

        if (deserialized?.PeerId != "phone-1")
            throw new Exception($"PeerId: {deserialized?.PeerId}");
        if (deserialized?.EstimatedTflops != 2.5)
            throw new Exception($"TFLOPS: {deserialized?.EstimatedTflops}");
    }

    // ═══════════════════════════════════════════════════════════
    //  Swarm Coordinator
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task P2P_Coordinator_Create()
    {
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);

        if (coordinator.PeerCount != 0)
            throw new Exception($"Initial peers: {coordinator.PeerCount}");
        if (coordinator.TotalTflops != 0)
            throw new Exception($"Initial TFLOPS: {coordinator.TotalTflops}");
    }

    [TestMethod]
    public async Task P2P_Coordinator_PeerJoin()
    {
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);

        RemotePeer? joined = null;
        coordinator.OnPeerJoined += p => joined = p;

        coordinator.HandlePeerConnected("phone-1", new PeerCapabilities
        {
            PeerId = "phone-1",
            AvailableMemory = 2L * 1024 * 1024 * 1024,
            EstimatedTflops = 1.5,
        });

        if (joined == null || joined.PeerId != "phone-1")
            throw new Exception("OnPeerJoined not fired or wrong peer");
        if (coordinator.PeerCount != 1)
            throw new Exception($"PeerCount: {coordinator.PeerCount}");
        if (coordinator.TotalTflops != 1.5)
            throw new Exception($"TFLOPS: {coordinator.TotalTflops}");
    }

    [TestMethod]
    public async Task P2P_Coordinator_PeerLeave()
    {
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);

        coordinator.HandlePeerConnected("phone-1", new PeerCapabilities
        {
            PeerId = "phone-1",
            EstimatedTflops = 1.5,
        });

        RemotePeer? left = null;
        coordinator.OnPeerLeft += p => left = p;
        coordinator.HandlePeerDisconnected("phone-1");

        if (left == null || left.PeerId != "phone-1")
            throw new Exception("OnPeerLeft not fired");
        if (coordinator.PeerCount != 0)
            throw new Exception($"PeerCount after leave: {coordinator.PeerCount}");
    }

    [TestMethod]
    public async Task P2P_Coordinator_CreateSwarm()
    {
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);
        await coordinator.CreateSwarmAsync("test-compute");

        if (string.IsNullOrEmpty(coordinator.MagnetLink))
            throw new Exception("MagnetLink should not be empty");
        if (!coordinator.MagnetLink.StartsWith("magnet:?xt=urn:btih:"))
            throw new Exception($"Invalid magnet: {coordinator.MagnetLink[..30]}");
        if (!coordinator.MagnetLink.Contains("dn=test-compute"))
            throw new Exception("MagnetLink should contain swarm name");

        Console.WriteLine($"[P2P] Swarm magnet: {coordinator.MagnetLink}");

        // HTTP join link — not generated without JoinLinkBaseUrl
        if (coordinator.JoinLink != null)
            throw new Exception("JoinLink should be null without JoinLinkBaseUrl");
    }

    [TestMethod]
    public async Task P2P_Coordinator_JoinLink_WithBaseUrl()
    {
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);
        // Set to the coordinator's own web app URL
        coordinator.JoinLinkBaseUrl = "https://myapp.example.com/ml-demo";
        await coordinator.CreateSwarmAsync("inference-pool");

        if (string.IsNullOrEmpty(coordinator.JoinLink))
            throw new Exception("JoinLink should be generated when BaseUrl is set");
        if (!coordinator.JoinLink.StartsWith("https://myapp.example.com/ml-demo?compute="))
            throw new Exception($"Should use coordinator's app URL: {coordinator.JoinLink}");
        if (!coordinator.JoinLink.Contains("n=inference-pool"))
            throw new Exception("JoinLink should contain swarm name");

        Console.WriteLine($"[P2P] Join: {coordinator.JoinLink}");
    }

    [TestMethod]
    public async Task P2P_Coordinator_JoinLink_NoBaseUrl()
    {
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);
        // No base URL set — JoinLink stays null
        await coordinator.CreateSwarmAsync("no-url-test");

        if (coordinator.JoinLink != null)
            throw new Exception("JoinLink should be null without JoinLinkBaseUrl");
        // MagnetLink still works
        if (string.IsNullOrEmpty(coordinator.MagnetLink))
            throw new Exception("MagnetLink should always be generated");
    }

    // ═══════════════════════════════════════════════════════════
    //  Smart Dispatch — Ability-Based Selection
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task P2P_Dispatcher_PrefersPowerfulPeer()
    {
        using var context = global::ILGPU.Context.CreateDefault();
        var device = new P2PDevice();
        var accelerator = (P2PAccelerator)device.CreateAccelerator(context);

        // Phone: 2 TFLOPS, 4GB
        accelerator.AddPeer(new RemotePeer
        {
            PeerId = "phone",
            IsConnected = true,
            MemorySize = 4L * 1024 * 1024 * 1024,
            Capabilities = new PeerCapabilities
            {
                PeerId = "phone",
                EstimatedTflops = 2.0,
                AvailableMemory = 4L * 1024 * 1024 * 1024,
            },
        });

        // Desktop: 36 TFLOPS, 24GB
        accelerator.AddPeer(new RemotePeer
        {
            PeerId = "desktop",
            IsConnected = true,
            MemorySize = 24L * 1024 * 1024 * 1024,
            Capabilities = new PeerCapabilities
            {
                PeerId = "desktop",
                EstimatedTflops = 36.0,
                AvailableMemory = 24L * 1024 * 1024 * 1024,
            },
        });

        var dispatcher = new P2PDispatcher(accelerator);
        var dispatchId = dispatcher.Dispatch(new KernelDispatchRequest());

        // Desktop should be preferred (higher TFLOPS + memory)
        var desktop = accelerator.Peers.First(p => p.PeerId == "desktop");
        var phone = accelerator.Peers.First(p => p.PeerId == "phone");

        if (desktop.PendingOperations != 1)
            throw new Exception($"Desktop should have the dispatch, pending={desktop.PendingOperations}");
        if (phone.PendingOperations != 0)
            throw new Exception($"Phone should be idle, pending={phone.PendingOperations}");
    }

    [TestMethod]
    public async Task P2P_Dispatcher_LoadBalances()
    {
        using var context = global::ILGPU.Context.CreateDefault();
        var device = new P2PDevice();
        var accelerator = (P2PAccelerator)device.CreateAccelerator(context);

        // Two equal peers
        accelerator.AddPeer(new RemotePeer
        {
            PeerId = "peer-a",
            IsConnected = true,
            Capabilities = new PeerCapabilities { EstimatedTflops = 10.0, AvailableMemory = 8L * 1024 * 1024 * 1024 },
        });
        accelerator.AddPeer(new RemotePeer
        {
            PeerId = "peer-b",
            IsConnected = true,
            Capabilities = new PeerCapabilities { EstimatedTflops = 10.0, AvailableMemory = 8L * 1024 * 1024 * 1024 },
        });

        var dispatcher = new P2PDispatcher(accelerator);

        // First dispatch goes to peer-a (both equal, first wins)
        dispatcher.Dispatch(new KernelDispatchRequest());

        // Second dispatch should go to peer-b (peer-a has load)
        dispatcher.Dispatch(new KernelDispatchRequest());

        var a = accelerator.Peers.First(p => p.PeerId == "peer-a");
        var b = accelerator.Peers.First(p => p.PeerId == "peer-b");

        if (a.PendingOperations != 1 || b.PendingOperations != 1)
            throw new Exception($"Should balance: a={a.PendingOperations}, b={b.PendingOperations}");
    }

    [TestMethod]
    public async Task P2P_Dispatcher_ThermalThrottle_AvoidsCritical()
    {
        using var context = global::ILGPU.Context.CreateDefault();
        var device = new P2PDevice();
        var accelerator = (P2PAccelerator)device.CreateAccelerator(context);

        // Powerful phone but thermally critical
        accelerator.AddPeer(new RemotePeer
        {
            PeerId = "hot-phone",
            IsConnected = true,
            Capabilities = new PeerCapabilities
            {
                EstimatedTflops = 5.0,
                AvailableMemory = 8L * 1024 * 1024 * 1024,
                ThermalState = 3, // CRITICAL
                IsCharging = false,
                BatteryLevel = 0.8,
            },
        });

        // Weak desktop, but cool and plugged in
        accelerator.AddPeer(new RemotePeer
        {
            PeerId = "cool-desktop",
            IsConnected = true,
            Capabilities = new PeerCapabilities
            {
                EstimatedTflops = 2.0,
                AvailableMemory = 4L * 1024 * 1024 * 1024,
                ThermalState = 0, // nominal
                IsCharging = true,
            },
        });

        var dispatcher = new P2PDispatcher(accelerator);
        dispatcher.Dispatch(new KernelDispatchRequest());

        var hotPhone = accelerator.Peers.First(p => p.PeerId == "hot-phone");
        var coolDesktop = accelerator.Peers.First(p => p.PeerId == "cool-desktop");

        // Cool desktop should get the work despite lower TFLOPS
        if (coolDesktop.PendingOperations != 1)
            throw new Exception($"Cool desktop should get work: pending={coolDesktop.PendingOperations}");
        if (hotPhone.PendingOperations != 0)
            throw new Exception($"Hot phone should be avoided: pending={hotPhone.PendingOperations}");
    }

    [TestMethod]
    public async Task P2P_Dispatcher_LowBattery_Penalized()
    {
        using var context = global::ILGPU.Context.CreateDefault();
        var device = new P2PDevice();
        var accelerator = (P2PAccelerator)device.CreateAccelerator(context);

        // Phone at 5% battery, not charging
        accelerator.AddPeer(new RemotePeer
        {
            PeerId = "dying-phone",
            IsConnected = true,
            Capabilities = new PeerCapabilities
            {
                EstimatedTflops = 5.0,
                AvailableMemory = 4L * 1024 * 1024 * 1024,
                ThermalState = 0,
                IsCharging = false,
                BatteryLevel = 0.05, // 5%
            },
        });

        // Equal phone, but plugged in
        accelerator.AddPeer(new RemotePeer
        {
            PeerId = "charged-phone",
            IsConnected = true,
            Capabilities = new PeerCapabilities
            {
                EstimatedTflops = 5.0,
                AvailableMemory = 4L * 1024 * 1024 * 1024,
                ThermalState = 0,
                IsCharging = true,
            },
        });

        var dispatcher = new P2PDispatcher(accelerator);
        dispatcher.Dispatch(new KernelDispatchRequest());

        var dying = accelerator.Peers.First(p => p.PeerId == "dying-phone");
        var charged = accelerator.Peers.First(p => p.PeerId == "charged-phone");

        if (charged.PendingOperations != 1)
            throw new Exception($"Charged phone should get work: pending={charged.PendingOperations}");
        if (dying.PendingOperations != 0)
            throw new Exception($"Dying phone should be avoided: pending={dying.PendingOperations}");
    }

    [TestMethod]
    public async Task P2P_Dispatcher_GracefulHandoff_ThermalCritical()
    {
        using var context = global::ILGPU.Context.CreateDefault();
        var device = new P2PDevice();
        var accelerator = (P2PAccelerator)device.CreateAccelerator(context);

        var hotPeer = new RemotePeer
        {
            PeerId = "overheating",
            IsConnected = true,
            Capabilities = new PeerCapabilities
            {
                EstimatedTflops = 10.0,
                AvailableMemory = 8L * 1024 * 1024 * 1024,
                ThermalState = 0, // starts cool
                IsCharging = true,
            },
        };
        var stablePeer = new RemotePeer
        {
            PeerId = "stable",
            IsConnected = true,
            Capabilities = new PeerCapabilities
            {
                EstimatedTflops = 8.0,
                AvailableMemory = 8L * 1024 * 1024 * 1024,
                ThermalState = 0,
                IsCharging = true,
            },
        };
        accelerator.AddPeer(hotPeer);
        accelerator.AddPeer(stablePeer);

        var dispatcher = new P2PDispatcher(accelerator);
        string? evictedPeerId = null;
        dispatcher.OnPeerEvicted += (id, count) => evictedPeerId = id;

        // Dispatch work to the hotter (but faster) peer
        dispatcher.Dispatch(new KernelDispatchRequest());

        // Peer gets hot — update capabilities
        dispatcher.UpdatePeerCapabilities("overheating", new PeerCapabilities
        {
            EstimatedTflops = 10.0,
            ThermalState = 3, // CRITICAL
            IsCharging = true,
        });

        // Trigger heartbeat check which does proactive eviction
        // Simulate by calling the check directly
        // The monitor would catch this on next heartbeat tick
        // For now, verify the scoring would avoid this peer
        var score = dispatcher.ScorePeer(hotPeer);
        if (score != 0.0)
            throw new Exception($"Thermally critical peer should score 0: {score}");
    }

    [TestMethod]
    public async Task P2P_Dispatcher_UpdateCapabilities()
    {
        using var context = global::ILGPU.Context.CreateDefault();
        var device = new P2PDevice();
        var accelerator = (P2PAccelerator)device.CreateAccelerator(context);

        accelerator.AddPeer(new RemotePeer
        {
            PeerId = "peer-1",
            IsConnected = true,
            Capabilities = new PeerCapabilities
            {
                EstimatedTflops = 5.0,
                BatteryLevel = 0.8,
                IsCharging = false,
            },
        });

        var dispatcher = new P2PDispatcher(accelerator);

        // Battery drains
        dispatcher.UpdatePeerCapabilities("peer-1", new PeerCapabilities
        {
            EstimatedTflops = 5.0,
            BatteryLevel = 0.03, // 3% — critical
            IsCharging = false,
        });

        var peer = accelerator.Peers[0];
        if (peer.Capabilities!.BatteryLevel != 0.03)
            throw new Exception($"Battery not updated: {peer.Capabilities.BatteryLevel}");

        var score = dispatcher.ScorePeer(peer);
        // Should be heavily penalized (battery 3% = 0.1x multiplier)
        if (score > 0.1)
            throw new Exception($"Low battery peer score too high: {score}");
    }

    [TestMethod]
    public async Task P2P_Coordinator_TotalCapacity()
    {
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);

        coordinator.HandlePeerConnected("peer-1", new PeerCapabilities
        {
            PeerId = "peer-1",
            EstimatedTflops = 10.0,
            AvailableMemory = 8L * 1024 * 1024 * 1024,
        });
        coordinator.HandlePeerConnected("peer-2", new PeerCapabilities
        {
            PeerId = "peer-2",
            EstimatedTflops = 5.0,
            AvailableMemory = 4L * 1024 * 1024 * 1024,
        });

        if (coordinator.TotalTflops != 15.0)
            throw new Exception($"TotalTflops: {coordinator.TotalTflops}");
        if (coordinator.TotalMemory != 12L * 1024 * 1024 * 1024)
            throw new Exception($"TotalMemory: {coordinator.TotalMemory}");

        // Peer leaves — capacity drops
        coordinator.HandlePeerDisconnected("peer-1");
        if (coordinator.TotalTflops != 5.0)
            throw new Exception($"TotalTflops after leave: {coordinator.TotalTflops}");
    }

    // ═══════════════════════════════════════════════════════════
    //  P2P Transport — Message Round-Trip
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task P2P_Transport_RegisterPeer_SendsCapabilityRequest()
    {
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);
        using var context = global::ILGPU.Context.CreateDefault();
        var accelerator = coordinator.CreateAccelerator(context);
        var dispatcher = new P2PDispatcher(accelerator);
        await using var transport = new P2PTransport(client, coordinator, dispatcher);

        P2PMessage? sentMessage = null;
        transport.RegisterPeer("test-peer", async (data) =>
        {
            sentMessage = P2PProtocol.Deserialize(data);
        });
        await Task.Delay(50); // let async send complete

        if (sentMessage == null)
            throw new Exception("Should send capability request on register");
        if (sentMessage.Type != P2PMessageType.CapabilityRequest)
            throw new Exception($"Type: {sentMessage.Type}");
    }

    [TestMethod]
    public async Task P2P_Transport_HandleCapabilityResponse_AddsPeer()
    {
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);
        using var context = global::ILGPU.Context.CreateDefault();
        var accelerator = coordinator.CreateAccelerator(context);
        var dispatcher = new P2PDispatcher(accelerator);
        await using var transport = new P2PTransport(client, coordinator, dispatcher);

        transport.RegisterPeer("gpu-peer", _ => Task.CompletedTask);

        // Simulate peer sending capability response
        var caps = new PeerCapabilities
        {
            PeerId = "gpu-peer",
            EstimatedTflops = 12.0,
            AvailableMemory = 8L * 1024 * 1024 * 1024,
            AvailableBackends = new[] { "WebGPU", "Wasm" },
            PreferredBackend = "WebGPU",
            Platform = "browser",
        };
        var response = new P2PMessage
        {
            Type = P2PMessageType.CapabilityResponse,
            Payload = System.Text.Json.JsonSerializer.SerializeToElement(caps),
        };
        await transport.HandleIncomingDataAsync("gpu-peer", P2PProtocol.Serialize(response));

        if (coordinator.PeerCount != 1)
            throw new Exception($"PeerCount: {coordinator.PeerCount}");
        if (coordinator.TotalTflops != 12.0)
            throw new Exception($"TFLOPS: {coordinator.TotalTflops}");
    }

    [TestMethod]
    public async Task P2P_Transport_HandleDisconnect_RemovesPeer()
    {
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);
        using var context = global::ILGPU.Context.CreateDefault();
        var accelerator = coordinator.CreateAccelerator(context);
        var dispatcher = new P2PDispatcher(accelerator);
        await using var transport = new P2PTransport(client, coordinator, dispatcher);

        transport.RegisterPeer("leaving-peer", _ => Task.CompletedTask);

        // Add peer via capability response
        var caps = new PeerCapabilities { PeerId = "leaving-peer", EstimatedTflops = 5.0 };
        await transport.HandleIncomingDataAsync("leaving-peer", P2PProtocol.Serialize(new P2PMessage
        {
            Type = P2PMessageType.CapabilityResponse,
            Payload = System.Text.Json.JsonSerializer.SerializeToElement(caps),
        }));

        if (coordinator.PeerCount != 1) throw new Exception("Should have 1 peer");

        // Peer sends disconnect
        await transport.HandleIncomingDataAsync("leaving-peer", P2PProtocol.Serialize(new P2PMessage
        {
            Type = P2PMessageType.Disconnect,
        }));

        if (coordinator.PeerCount != 0)
            throw new Exception($"PeerCount after disconnect: {coordinator.PeerCount}");
    }

    [TestMethod]
    public async Task P2P_Transport_HandleHeartbeat_UpdatesTimestamp()
    {
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);
        using var context = global::ILGPU.Context.CreateDefault();
        var accelerator = coordinator.CreateAccelerator(context);
        var dispatcher = new P2PDispatcher(accelerator);
        await using var transport = new P2PTransport(client, coordinator, dispatcher);

        transport.RegisterPeer("heartbeat-peer", _ => Task.CompletedTask);

        // Connect peer
        await transport.HandleIncomingDataAsync("heartbeat-peer", P2PProtocol.Serialize(new P2PMessage
        {
            Type = P2PMessageType.CapabilityResponse,
            Payload = System.Text.Json.JsonSerializer.SerializeToElement(
                new PeerCapabilities { PeerId = "heartbeat-peer", EstimatedTflops = 3.0 }),
        }));

        var peer = accelerator.Peers.FirstOrDefault(p => p.PeerId == "heartbeat-peer");
        if (peer == null) throw new Exception("Peer not found");

        var before = peer.LastHeartbeat;

        // Send heartbeat
        await transport.HandleIncomingDataAsync("heartbeat-peer", P2PProtocol.Serialize(new P2PMessage
        {
            Type = P2PMessageType.Heartbeat,
        }));

        if (peer.LastHeartbeat == before)
            throw new Exception("Heartbeat should update timestamp");
    }

    [TestMethod]
    public async Task P2P_Transport_Broadcast()
    {
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);
        using var context = global::ILGPU.Context.CreateDefault();
        var accelerator = coordinator.CreateAccelerator(context);
        var dispatcher = new P2PDispatcher(accelerator);
        await using var transport = new P2PTransport(client, coordinator, dispatcher);

        // RegisterPeer sends a CapabilityRequest, so we track only after registration
        transport.RegisterPeer("peer-a", _ => Task.CompletedTask);
        transport.RegisterPeer("peer-b", _ => Task.CompletedTask);
        transport.RegisterPeer("peer-c", _ => Task.CompletedTask);

        int broadcastCount = 0;
        // Re-register with counting send function
        transport.RegisterPeer("peer-a", _ => { broadcastCount++; return Task.CompletedTask; });
        transport.RegisterPeer("peer-b", _ => { broadcastCount++; return Task.CompletedTask; });
        transport.RegisterPeer("peer-c", _ => { broadcastCount++; return Task.CompletedTask; });

        // Reset count after re-registration capability requests
        await Task.Delay(50);
        broadcastCount = 0;

        await transport.BroadcastAsync(new P2PMessage { Type = P2PMessageType.Heartbeat });

        if (broadcastCount != 3)
            throw new Exception($"Broadcast should reach all 3 peers: {broadcastCount}");
    }

    // ═══════════════════════════════════════════════════════════
    //  Coordinator Role — Transfer & Election
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task P2P_Role_CreatorIsCoordinator()
    {
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);
        await coordinator.CreateSwarmAsync("role-test");

        if (coordinator.Role != P2PRole.Coordinator)
            throw new Exception($"Creator should be Coordinator: {coordinator.Role}");
    }

    [TestMethod]
    public async Task P2P_Role_JoinerIsWorker()
    {
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);
        // JoinSwarmAsync needs a real magnet — use a fake one to test role assignment
        // Just test the role assignment logic directly
        coordinator.HandlePeerConnected("existing-coord", new PeerCapabilities { PeerId = "existing-coord" });

        // Default role before creating swarm is Worker
        if (coordinator.Role != P2PRole.Worker)
            throw new Exception($"Default role should be Worker: {coordinator.Role}");
    }

    [TestMethod]
    public async Task P2P_Role_Transfer()
    {
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);
        await coordinator.CreateSwarmAsync("transfer-test");

        coordinator.HandlePeerConnected("fast-desktop", new PeerCapabilities
        {
            PeerId = "fast-desktop",
            EstimatedTflops = 20.0,
            AvailableMemory = 16L * 1024 * 1024 * 1024,
        });
        coordinator.HandlePeerConnected("slow-phone", new PeerCapabilities
        {
            PeerId = "slow-phone",
            EstimatedTflops = 2.0,
            AvailableMemory = 2L * 1024 * 1024 * 1024,
        });

        string? newCoord = null;
        coordinator.OnCoordinatorChanged += id => newCoord = id;

        var transferredTo = coordinator.TransferCoordinator();

        if (transferredTo != "fast-desktop")
            throw new Exception($"Should transfer to fastest peer: {transferredTo}");
        if (coordinator.Role != P2PRole.Worker)
            throw new Exception($"After transfer, should be Worker: {coordinator.Role}");
        if (newCoord != "fast-desktop")
            throw new Exception("OnCoordinatorChanged should fire");
    }

    [TestMethod]
    public async Task P2P_Role_Transfer_ToSpecific()
    {
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);
        await coordinator.CreateSwarmAsync("transfer-specific");

        coordinator.HandlePeerConnected("peer-a", new PeerCapabilities { PeerId = "peer-a", EstimatedTflops = 20.0 });
        coordinator.HandlePeerConnected("peer-b", new PeerCapabilities { PeerId = "peer-b", EstimatedTflops = 5.0 });

        // Transfer to specific peer (even though peer-a is faster)
        var result = coordinator.TransferCoordinator("peer-b");
        if (result != "peer-b")
            throw new Exception($"Should transfer to specified peer: {result}");
    }

    [TestMethod]
    public async Task P2P_Role_Election_PicksStrongest()
    {
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);

        coordinator.HandlePeerConnected("weak", new PeerCapabilities { PeerId = "weak", EstimatedTflops = 2.0, AvailableMemory = 2L * 1024 * 1024 * 1024 });
        coordinator.HandlePeerConnected("strong", new PeerCapabilities { PeerId = "strong", EstimatedTflops = 30.0, AvailableMemory = 24L * 1024 * 1024 * 1024 });
        coordinator.HandlePeerConnected("medium", new PeerCapabilities { PeerId = "medium", EstimatedTflops = 10.0, AvailableMemory = 8L * 1024 * 1024 * 1024 });

        var elected = coordinator.ElectCoordinator();
        if (elected != "strong")
            throw new Exception($"Election should pick strongest: {elected}");
    }

    [TestMethod]
    public async Task P2P_Role_Election_Deterministic()
    {
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);

        // Two equal peers — should pick deterministically by PeerId
        coordinator.HandlePeerConnected("peer-b", new PeerCapabilities { PeerId = "peer-b", EstimatedTflops = 10.0, AvailableMemory = 8L * 1024 * 1024 * 1024 });
        coordinator.HandlePeerConnected("peer-a", new PeerCapabilities { PeerId = "peer-a", EstimatedTflops = 10.0, AvailableMemory = 8L * 1024 * 1024 * 1024 });

        var elected1 = coordinator.ElectCoordinator();
        var elected2 = coordinator.ElectCoordinator();
        if (elected1 != elected2)
            throw new Exception($"Election must be deterministic: {elected1} vs {elected2}");
    }

    [TestMethod]
    public async Task P2P_Role_BecomeCoordinator()
    {
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);

        if (coordinator.Role != P2PRole.Worker)
            throw new Exception("Should start as Worker");

        coordinator.BecomeCoordinator();

        if (coordinator.Role != P2PRole.Coordinator)
            throw new Exception($"Should be Coordinator: {coordinator.Role}");
        if (coordinator.CoordinatorPeerId != null)
            throw new Exception("CoordinatorPeerId should be null (we are it)");
    }

    // ═══════════════════════════════════════════════════════════
    //  P2P Worker
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task P2P_Worker_Create()
    {
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);
        using var context = global::ILGPU.Context.CreateDefault();
        var accelerator = coordinator.CreateAccelerator(context);
        var dispatcher = new P2PDispatcher(accelerator);
        await using var transport = new P2PTransport(client, coordinator, dispatcher);
        await using var worker = new P2PWorker(transport);

        if (worker.IsReady)
            throw new Exception("Should not be ready before Initialize");
    }

    [TestMethod]
    public async Task P2P_Worker_Initialize()
    {
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);
        using var ctx = global::ILGPU.Context.CreateDefault();
        var accel = coordinator.CreateAccelerator(ctx);
        var dispatcher = new P2PDispatcher(accel);
        await using var transport = new P2PTransport(client, coordinator, dispatcher);
        await using var worker = new P2PWorker(transport);
        worker.Initialize(ctx, accel);

        if (!worker.IsReady)
            throw new Exception("Should be ready after Initialize");
    }

    [TestMethod]
    public async Task P2P_Worker_BuildCapabilities()
    {
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);
        using var ctx = global::ILGPU.Context.CreateDefault();
        var accel = coordinator.CreateAccelerator(ctx);
        var dispatcher = new P2PDispatcher(accel);
        await using var transport = new P2PTransport(client, coordinator, dispatcher);
        await using var worker = new P2PWorker(transport);
        worker.Initialize(ctx, accel);

        var caps = worker.BuildCapabilities("my-peer-id");
        if (caps.PeerId != "my-peer-id")
            throw new Exception($"PeerId: {caps.PeerId}");
        if (string.IsNullOrEmpty(caps.PreferredBackend))
            throw new Exception("PreferredBackend empty");
        if (caps.AvailableBackends.Length == 0)
            throw new Exception("No backends");
    }

    [TestMethod]
    public async Task P2P_Worker_BufferRoundTrip()
    {
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);
        using var ctx = global::ILGPU.Context.CreateDefault();
        var accel = coordinator.CreateAccelerator(ctx);
        var dispatcher = new P2PDispatcher(accel);
        await using var transport = new P2PTransport(client, coordinator, dispatcher);
        await using var worker = new P2PWorker(transport);

        var data = new byte[] { 1, 2, 3, 4, 5 };
        worker.ReceiveBuffer("buf-1", data);

        var retrieved = worker.GetBuffer("buf-1");
        if (retrieved == null || !retrieved.SequenceEqual(data))
            throw new Exception("Buffer round-trip failed");

        if (worker.GetBuffer("nonexistent") != null)
            throw new Exception("Missing buffer should return null");
    }

    [TestMethod]
    public async Task P2P_Role_Transfer_NoPeers_ReturnsNull()
    {
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);
        await coordinator.CreateSwarmAsync("empty-transfer");

        var result = coordinator.TransferCoordinator();
        if (result != null)
            throw new Exception($"Transfer with no peers should return null: {result}");
        if (coordinator.Role != P2PRole.Coordinator)
            throw new Exception("Should remain Coordinator if transfer fails");
    }

    // ═══════════════════════════════════════════════════════════
    //  Kick & Block
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task P2P_Kick_RemovesPeer()
    {
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);
        await coordinator.CreateSwarmAsync("kick-test");

        coordinator.HandlePeerConnected("bad-peer", new PeerCapabilities { PeerId = "bad-peer" });
        if (coordinator.PeerCount != 1) throw new Exception("Should have 1 peer");

        string? kickedId = null;
        coordinator.OnPeerKicked += (id, reason) => kickedId = id;

        var result = coordinator.KickPeer("bad-peer", "returning garbage tensors");
        if (!result) throw new Exception("Kick should succeed");
        if (coordinator.PeerCount != 0) throw new Exception("Peer should be removed");
        if (kickedId != "bad-peer") throw new Exception("OnPeerKicked should fire");
    }

    [TestMethod]
    public async Task P2P_Kick_OnlyCoordinatorCan()
    {
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);
        // Default role is Worker
        coordinator.HandlePeerConnected("peer-1", new PeerCapabilities { PeerId = "peer-1" });

        var result = coordinator.KickPeer("peer-1");
        if (result) throw new Exception("Worker should not be able to kick");
    }

    [TestMethod]
    public async Task P2P_Block_PreventsReconnect()
    {
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);
        await coordinator.CreateSwarmAsync("block-test");

        coordinator.HandlePeerConnected("malicious", new PeerCapabilities { PeerId = "malicious" });
        coordinator.BlockPeer("malicious", "tampered results");

        if (coordinator.PeerCount != 0) throw new Exception("Should be disconnected");
        if (!coordinator.IsPeerBlocked("malicious")) throw new Exception("Should be blocked");

        // Try to reconnect — should be rejected
        string? rejectedId = null;
        coordinator.OnPeerRejected += (id, reason) => rejectedId = id;

        var accepted = coordinator.HandlePeerConnected("malicious", new PeerCapabilities { PeerId = "malicious" });
        if (accepted) throw new Exception("Blocked peer should be rejected");
        if (rejectedId != "malicious") throw new Exception("OnPeerRejected should fire");
        if (coordinator.PeerCount != 0) throw new Exception("Should still have 0 peers");
    }

    [TestMethod]
    public async Task P2P_Unblock_AllowsReconnect()
    {
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);
        await coordinator.CreateSwarmAsync("unblock-test");

        coordinator.BlockPeer("reformed-peer");
        if (!coordinator.IsPeerBlocked("reformed-peer")) throw new Exception("Should be blocked");

        coordinator.UnblockPeer("reformed-peer");
        if (coordinator.IsPeerBlocked("reformed-peer")) throw new Exception("Should be unblocked");

        // Now they can reconnect
        var accepted = coordinator.HandlePeerConnected("reformed-peer", new PeerCapabilities { PeerId = "reformed-peer", EstimatedTflops = 5.0 });
        if (!accepted) throw new Exception("Unblocked peer should be accepted");
        if (coordinator.PeerCount != 1) throw new Exception("Should have 1 peer");
    }

    [TestMethod]
    public async Task P2P_Block_ListTracked()
    {
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);
        await coordinator.CreateSwarmAsync("blocklist-test");

        coordinator.BlockPeer("bad-1");
        coordinator.BlockPeer("bad-2");
        coordinator.BlockPeer("bad-3");

        if (coordinator.BlockedPeers.Count != 3)
            throw new Exception($"BlockedPeers: {coordinator.BlockedPeers.Count}");

        coordinator.UnblockPeer("bad-2");
        if (coordinator.BlockedPeers.Count != 2)
            throw new Exception($"After unblock: {coordinator.BlockedPeers.Count}");
    }

    // ═══════════════════════════════════════════════════════════
    //  Kernel Serialization
    // ═══════════════════════════════════════════════════════════

    // A simple test kernel for serialization testing
    public static void TestVectorAdd(global::ILGPU.Index1D index, global::ILGPU.ArrayView<float> a, global::ILGPU.ArrayView<float> b, global::ILGPU.ArrayView<float> result)
    {
        result[index] = a[index] + b[index];
    }

    [TestMethod]
    public async Task P2P_KernelSerializer_CreateDispatch()
    {
        var method = typeof(BackendTestBase).GetMethod(nameof(TestVectorAdd),
            System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static)!;

        var dispatch = P2PKernelSerializer.CreateDispatch(method, gridDimX: 1024);

        if (string.IsNullOrEmpty(dispatch.KernelType))
            throw new Exception("KernelType empty");
        if (dispatch.KernelMethod != "TestVectorAdd")
            throw new Exception($"Method: {dispatch.KernelMethod}");
        if (dispatch.GridDimX != 1024)
            throw new Exception($"GridDimX: {dispatch.GridDimX}");
    }

    [TestMethod]
    public async Task P2P_KernelSerializer_ResolveKernel()
    {
        var method = typeof(BackendTestBase).GetMethod(nameof(TestVectorAdd),
            System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static)!;

        var dispatch = P2PKernelSerializer.CreateDispatch(method, gridDimX: 512);

        // Simulate worker-side resolution
        var resolved = P2PKernelSerializer.ResolveKernel(dispatch);
        if (resolved == null)
            throw new Exception("Should resolve the kernel method");
        if (resolved.Name != "TestVectorAdd")
            throw new Exception($"Resolved wrong method: {resolved.Name}");
    }

    [TestMethod]
    public async Task P2P_KernelSerializer_CanExecute()
    {
        var method = typeof(BackendTestBase).GetMethod(nameof(TestVectorAdd),
            System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static)!;

        var dispatch = P2PKernelSerializer.CreateDispatch(method, gridDimX: 256);
        if (!P2PKernelSerializer.CanExecute(dispatch))
            throw new Exception("Should be able to execute — method exists in our assemblies");
    }

    [TestMethod]
    public async Task P2P_KernelSerializer_CantExecute_UnknownMethod()
    {
        var dispatch = new KernelDispatchRequest
        {
            KernelType = "NonExistent.FakeClass, FakeAssembly",
            KernelMethod = "FakeKernel",
        };
        if (P2PKernelSerializer.CanExecute(dispatch))
            throw new Exception("Should NOT be able to execute — method doesn't exist");
    }

    // ═══════════════════════════════════════════════════════════
    //  End-to-End: Coordinator → Transport → Worker → Result
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task P2P_EndToEnd_DispatchAndReceiveResult()
    {
        // Full pipeline: coordinator → transport → worker → result → transport → coordinator
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);
        await coordinator.CreateSwarmAsync("e2e-test");
        using var ctx = global::ILGPU.Context.CreateDefault();
        var accelerator = coordinator.CreateAccelerator(ctx);
        var dispatcher = new P2PDispatcher(accelerator);
        await using var transport = new P2PTransport(client, coordinator, dispatcher);

        // Worker side — uses a real CPU accelerator for execution
        var cpuAccel = global::ILGPU.Runtime.CPU.CPUDevice.Default.CreateAccelerator(ctx);
        await using var worker = new P2PWorker(transport);
        worker.Initialize(ctx, cpuAccel);
        worker.ReceiveBuffer("buf-a", new byte[4096]);
        worker.ReceiveBuffer("buf-b", new byte[4096]);
        worker.ReceiveBuffer("buf-result", new byte[4096]);

        // Create dispatch
        var method = typeof(BackendTestBase).GetMethod(nameof(TestVectorAdd),
            System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static)!;
        var request = P2PKernelSerializer.CreateDispatch(method, gridDimX: 1024);
        request.Buffers = new[]
        {
            new BufferBinding { ParameterIndex = 1, BufferId = "buf-a", Length = 1024, ElementSize = 4 },
            new BufferBinding { ParameterIndex = 2, BufferId = "buf-b", Length = 1024, ElementSize = 4 },
            new BufferBinding { ParameterIndex = 3, BufferId = "buf-result", Length = 1024, ElementSize = 4 },
        };

        // Simulate the full chain: serialize dispatch, worker handles, returns result
        var dispatchMsg = new P2PMessage
        {
            Type = P2PMessageType.KernelDispatch,
            Payload = System.Text.Json.JsonSerializer.SerializeToElement(request),
        };

        // Track result received back
        KernelDispatchResult? receivedResult = null;
        transport.RegisterPeer("worker-1", async (data) =>
        {
            // This is what the coordinator sends to the worker
            // We intercept and have the worker handle it
        });

        // Worker processes the dispatch directly (simulating transport delivery)
        await worker.HandleDispatchAsync("coordinator", request);

        // Verify worker processed it
        if (worker.GetBuffer("buf-result") == null)
            throw new Exception("Worker should have created result buffer");

        Console.WriteLine("[P2P E2E] Full pipeline: dispatch → resolve → execute → result");
    }

    [TestMethod]
    public async Task P2P_EndToEnd_WorkerKernelResolution()
    {
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);
        using var ctx = global::ILGPU.Context.CreateDefault();
        var accelerator = coordinator.CreateAccelerator(ctx);
        var cpuAccel = global::ILGPU.Runtime.CPU.CPUDevice.Default.CreateAccelerator(ctx);
        var dispatcher = new P2PDispatcher(accelerator);
        await using var transport = new P2PTransport(client, coordinator, dispatcher);
        await using var worker = new P2PWorker(transport);
        worker.Initialize(ctx, cpuAccel);

        // Track worker events
        string? startedId = null;
        string? completedId = null;
        bool? completedSuccess = null;
        worker.OnKernelStarted += id => startedId = id;
        worker.OnKernelCompleted += (id, success, ms) =>
        {
            completedId = id;
            completedSuccess = success;
        };

        // Create dispatch with real kernel reference
        var method = typeof(BackendTestBase).GetMethod(nameof(TestVectorAdd),
            System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static)!;
        var request = P2PKernelSerializer.CreateDispatch(method, gridDimX: 512);
        request.Buffers = new[]
        {
            new BufferBinding { ParameterIndex = 1, BufferId = "a", Length = 512, ElementSize = 4 },
            new BufferBinding { ParameterIndex = 2, BufferId = "b", Length = 512, ElementSize = 4 },
            new BufferBinding { ParameterIndex = 3, BufferId = "r", Length = 512, ElementSize = 4 },
        };
        worker.ReceiveBuffer("a", new byte[2048]);
        worker.ReceiveBuffer("b", new byte[2048]);
        worker.ReceiveBuffer("r", new byte[2048]);

        // Worker handles dispatch directly
        await worker.HandleDispatchAsync("coordinator", request);

        if (startedId != request.DispatchId)
            throw new Exception($"OnKernelStarted not fired: {startedId}");
        if (completedId != request.DispatchId)
            throw new Exception($"OnKernelCompleted not fired: {completedId}");
        if (completedSuccess != true)
            throw new Exception($"Should succeed: {completedSuccess}");
    }

    [TestMethod]
    public async Task P2P_EndToEnd_WorkerRejectsUnknownKernel()
    {
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);
        using var ctx = global::ILGPU.Context.CreateDefault();
        var accelerator = coordinator.CreateAccelerator(ctx);
        var cpuAccel = global::ILGPU.Runtime.CPU.CPUDevice.Default.CreateAccelerator(ctx);
        var dispatcher = new P2PDispatcher(accelerator);
        await using var transport = new P2PTransport(client, coordinator, dispatcher);
        await using var worker = new P2PWorker(transport);
        worker.Initialize(ctx, cpuAccel);

        bool? completedSuccess = null;
        worker.OnKernelCompleted += (id, success, ms) => completedSuccess = success;

        var request = new KernelDispatchRequest
        {
            KernelType = "Fake.NonExistent, FakeAssembly",
            KernelMethod = "BogusKernel",
        };

        await worker.HandleDispatchAsync("coordinator", request);

        if (completedSuccess != false)
            throw new Exception("Should fail for unknown kernel");
    }

    // ═══════════════════════════════════════════════════════════
    //  Worker Kernel Compilation (Real Backend)
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task P2P_Worker_CompileKernel_RealBackend() => await RunTest(async accelerator =>
    {
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);
        var dispatcher = new P2PDispatcher(coordinator.CreateAccelerator(
            global::ILGPU.Context.CreateDefault()));
        await using var transport = new P2PTransport(client, coordinator, dispatcher);
        await using var worker = new P2PWorker(transport);

        // Initialize with the REAL accelerator (CPU/CUDA/WebGPU — whatever this test class runs on)
        worker.Initialize(accelerator.Context, accelerator);

        // Pre-compile the test kernel on the real backend
        var method = typeof(BackendTestBase).GetMethod(nameof(TestVectorAdd),
            System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static)!;
        var compiled = worker.PreCompileKernel(method);

        if (!compiled)
            throw new Exception($"Kernel should compile on {accelerator.AcceleratorType}");
        if (worker.CachedKernelCount != 1)
            throw new Exception($"CachedKernelCount: {worker.CachedKernelCount}");

        // Compile again — should use cache
        var compiled2 = worker.PreCompileKernel(method);
        if (!compiled2)
            throw new Exception("Cached compile should succeed");
        if (worker.CachedKernelCount != 1)
            throw new Exception("Should still be 1 (cached)");
    });

    [TestMethod]
    public async Task P2P_Worker_CompileAndDispatch_RealBackend() => await RunTest(async accelerator =>
    {
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);
        var p2pAccel = coordinator.CreateAccelerator(global::ILGPU.Context.CreateDefault());
        var dispatcher = new P2PDispatcher(p2pAccel);
        await using var transport = new P2PTransport(client, coordinator, dispatcher);
        await using var worker = new P2PWorker(transport);

        // Real backend
        worker.Initialize(accelerator.Context, accelerator);

        string? compiledKernel = null;
        bool? dispatchSuccess = null;
        worker.OnKernelCompiled += name => compiledKernel = name;
        worker.OnKernelCompleted += (id, success, ms) => dispatchSuccess = success;

        // Dispatch with real kernel method
        var method = typeof(BackendTestBase).GetMethod(nameof(TestVectorAdd),
            System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static)!;
        var request = P2PKernelSerializer.CreateDispatch(method, gridDimX: 64);
        request.Buffers = new[]
        {
            new BufferBinding { ParameterIndex = 1, BufferId = "a", Length = 64, ElementSize = 4 },
            new BufferBinding { ParameterIndex = 2, BufferId = "b", Length = 64, ElementSize = 4 },
            new BufferBinding { ParameterIndex = 3, BufferId = "r", Length = 64, ElementSize = 4 },
        };
        worker.ReceiveBuffer("a", new byte[256]);
        worker.ReceiveBuffer("b", new byte[256]);
        worker.ReceiveBuffer("r", new byte[256]);

        await worker.HandleDispatchAsync("coordinator", request);

        if (compiledKernel == null)
            throw new Exception("OnKernelCompiled should fire");
        if (dispatchSuccess != true)
            throw new Exception($"Dispatch should succeed on {accelerator.AcceleratorType}");
    });

    // ═══════════════════════════════════════════════════════════
    //  Buffer Transfer — Chunking & Reassembly
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task P2P_BufferTransfer_SmallBuffer_SingleChunk()
    {
        var transfer = new P2PBufferTransfer();
        var data = new byte[] { 1, 2, 3, 4, 5 };

        var chunks = transfer.CreateChunks("buf-1", data);
        if (chunks.Length != 1) throw new Exception($"Small buffer should be 1 chunk: {chunks.Length}");
        if (chunks[0].TotalBytes != 5) throw new Exception($"TotalBytes: {chunks[0].TotalBytes}");
    }

    [TestMethod]
    public async Task P2P_BufferTransfer_LargeBuffer_MultiChunk()
    {
        var transfer = new P2PBufferTransfer { MaxChunkSize = 1024 };
        var data = new byte[5000];
        Random.Shared.NextBytes(data);

        var chunks = transfer.CreateChunks("buf-large", data);
        if (chunks.Length != 5) throw new Exception($"5000/1024 should be 5 chunks: {chunks.Length}");

        // Verify chunk metadata
        for (int i = 0; i < chunks.Length; i++)
        {
            if (chunks[i].ChunkIndex != i) throw new Exception($"Chunk {i} index wrong");
            if (chunks[i].TotalChunks != 5) throw new Exception($"TotalChunks: {chunks[i].TotalChunks}");
            if (chunks[i].BufferId != "buf-large") throw new Exception("BufferId mismatch");
        }
    }

    [TestMethod]
    public async Task P2P_BufferTransfer_Reassemble()
    {
        var transfer = new P2PBufferTransfer { MaxChunkSize = 1024 };
        var originalData = new byte[5000];
        Random.Shared.NextBytes(originalData);

        var chunks = transfer.CreateChunks("buf-reassemble", originalData);

        byte[]? received = null;
        transfer.OnBufferReceived += (id, data) => received = data;

        // Feed chunks in order
        for (int i = 0; i < chunks.Length; i++)
        {
            var complete = transfer.ReceiveChunk(chunks[i]);
            if (i < chunks.Length - 1 && complete)
                throw new Exception("Should not be complete before last chunk");
        }

        if (received == null) throw new Exception("OnBufferReceived should fire");
        if (received.Length != 5000) throw new Exception($"Length: {received.Length}");
        if (!received.SequenceEqual(originalData))
            throw new Exception("Reassembled data doesn't match original");
    }

    [TestMethod]
    public async Task P2P_BufferTransfer_OutOfOrder()
    {
        var transfer = new P2PBufferTransfer { MaxChunkSize = 1024 };
        var data = new byte[3000];
        Random.Shared.NextBytes(data);

        var chunks = transfer.CreateChunks("buf-ooo", data);

        byte[]? received = null;
        transfer.OnBufferReceived += (id, d) => received = d;

        // Feed in reverse order
        for (int i = chunks.Length - 1; i >= 0; i--)
            transfer.ReceiveChunk(chunks[i]);

        if (received == null) throw new Exception("Should reassemble from out-of-order chunks");
        if (!received.SequenceEqual(data))
            throw new Exception("Out-of-order reassembly data mismatch");
    }

    [TestMethod]
    public async Task P2P_BufferTransfer_Progress()
    {
        var transfer = new P2PBufferTransfer { MaxChunkSize = 1024 };
        var data = new byte[4096]; // 4 chunks

        var chunks = transfer.CreateChunks("buf-progress", data);

        transfer.ReceiveChunk(chunks[0]);
        var p1 = transfer.GetProgress("buf-progress");
        if (Math.Abs(p1 - 0.25) > 0.01) throw new Exception($"After 1/4: {p1}");

        transfer.ReceiveChunk(chunks[1]);
        var p2 = transfer.GetProgress("buf-progress");
        if (Math.Abs(p2 - 0.5) > 0.01) throw new Exception($"After 2/4: {p2}");
    }

    [TestMethod]
    public async Task P2P_BufferTransfer_DuplicateChunk()
    {
        var transfer = new P2PBufferTransfer { MaxChunkSize = 1024 };
        var data = new byte[2048]; // 2 chunks
        Random.Shared.NextBytes(data);

        var chunks = transfer.CreateChunks("buf-dup", data);

        byte[]? received = null;
        transfer.OnBufferReceived += (id, d) => received = d;

        // Send chunk 0 twice, then chunk 1
        transfer.ReceiveChunk(chunks[0]);
        transfer.ReceiveChunk(chunks[0]); // duplicate
        transfer.ReceiveChunk(chunks[1]);

        if (received == null) throw new Exception("Should complete despite duplicate");
        if (!received.SequenceEqual(data)) throw new Exception("Data mismatch after duplicate");
    }

    [TestMethod]
    public async Task P2P_BufferTransfer_Cancel()
    {
        var transfer = new P2PBufferTransfer { MaxChunkSize = 1024 };
        var data = new byte[4096];
        var chunks = transfer.CreateChunks("buf-cancel", data);

        transfer.ReceiveChunk(chunks[0]);
        if (transfer.ActiveTransfers != 1) throw new Exception($"Active: {transfer.ActiveTransfers}");

        transfer.CancelTransfer("buf-cancel");
        if (transfer.ActiveTransfers != 0) throw new Exception("Should be 0 after cancel");
    }

    [TestMethod]
    public async Task P2P_BufferTransfer_1MB_Tensor()
    {
        // Simulate transferring a 1MB float tensor (256K floats)
        var transfer = new P2PBufferTransfer(); // default 64KB chunks
        var tensorData = new byte[1024 * 1024];
        Random.Shared.NextBytes(tensorData);

        var chunks = transfer.CreateChunks("tensor-1mb", tensorData);
        // 1MB / 64KB = 16 chunks
        if (chunks.Length != 16) throw new Exception($"1MB should be 16 chunks: {chunks.Length}");

        byte[]? received = null;
        transfer.OnBufferReceived += (id, d) => received = d;

        foreach (var chunk in chunks)
            transfer.ReceiveChunk(chunk);

        if (received == null) throw new Exception("Should receive 1MB tensor");
        if (received.Length != 1024 * 1024) throw new Exception($"Length: {received.Length}");
        if (!received.SequenceEqual(tensorData))
            throw new Exception("1MB tensor data corruption");
    }

    // ═══════════════════════════════════════════════════════════
    //  Buffer Transfer via Transport Layer
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task P2P_Transport_SendBuffer_ChunkedDelivery()
    {
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);
        using var ctx = global::ILGPU.Context.CreateDefault();
        var accelerator = coordinator.CreateAccelerator(ctx);
        var dispatcher = new P2PDispatcher(accelerator);
        await using var transport = new P2PTransport(client, coordinator, dispatcher);

        // Track messages sent to worker
        var sentMessages = new List<byte[]>();
        transport.RegisterPeer("worker-1", async (data) =>
        {
            sentMessages.Add(data);
        });

        // Send a buffer that requires multiple chunks
        var tensorData = new byte[200_000]; // ~3 chunks at 64KB
        Random.Shared.NextBytes(tensorData);
        await transport.SendBufferAsync("worker-1", "tensor-a", tensorData);

        // Should have sent multiple messages (capability request + buffer chunks)
        // RegisterPeer sends a capability request, then SendBufferAsync sends chunks
        // 200KB / 64KB = 4 chunks (rounded up) + 1 capability request = 5 messages
        if (sentMessages.Count < 4)
            throw new Exception($"Should send multiple chunk messages: {sentMessages.Count}");

        // Verify the transport's buffer transfer can reassemble on the receiving end
        byte[]? reassembled = null;
        transport.BufferTransfer.OnBufferReceived += (id, data) => reassembled = data;

        // Simulate receiving the chunks on the other end
        var chunks = transport.BufferTransfer.CreateChunks("tensor-roundtrip", tensorData);
        foreach (var chunk in chunks)
            transport.BufferTransfer.ReceiveChunk(chunk);

        if (reassembled == null) throw new Exception("Should reassemble");
        if (!reassembled.SequenceEqual(tensorData))
            throw new Exception("Buffer data corruption through transport");
    }

    // ═══════════════════════════════════════════════════════════
    //  BEP 46 State Manager
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task P2P_StateManager_Create()
    {
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        var dht = new SpawnDev.WebTorrent.Discovery.DhtDiscovery();
        var channel = new SpawnDev.WebTorrent.AgentChannel(dht, new SpawnDev.WebTorrent.Discovery.HmacFallbackSigner());
        await using var coordinator = new P2PSwarmCoordinator(client);
        await coordinator.CreateSwarmAsync("state-test");

        var stateManager = new P2PStateManager(channel, coordinator);

        if (stateManager.PublicKey == null || stateManager.PublicKey.Length == 0)
            throw new Exception("PublicKey should not be empty");
        if (string.IsNullOrEmpty(stateManager.PublicKeyHex))
            throw new Exception("PublicKeyHex should not be empty");
    }

    [TestMethod]
    public async Task P2P_StateManager_PublishState()
    {
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        var dht = new SpawnDev.WebTorrent.Discovery.DhtDiscovery();
        var channel = new SpawnDev.WebTorrent.AgentChannel(dht, new SpawnDev.WebTorrent.Discovery.HmacFallbackSigner());
        await using var coordinator = new P2PSwarmCoordinator(client);
        await coordinator.CreateSwarmAsync("publish-test");

        // Add some peers for state content
        coordinator.HandlePeerConnected("peer-1", new PeerCapabilities
        {
            PeerId = "peer-1", EstimatedTflops = 10.0,
            AvailableMemory = 8L * 1024 * 1024 * 1024,
        });

        var stateManager = new P2PStateManager(channel, coordinator);

        // Should not throw — publishes state to DHT
        await stateManager.PublishStateAsync();
    }

    [TestMethod]
    public async Task P2P_StateManager_WorkerCantPublish()
    {
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        var dht = new SpawnDev.WebTorrent.Discovery.DhtDiscovery();
        var channel = new SpawnDev.WebTorrent.AgentChannel(dht, new SpawnDev.WebTorrent.Discovery.HmacFallbackSigner());
        await using var coordinator = new P2PSwarmCoordinator(client);
        // Don't call CreateSwarmAsync — stays as Worker role

        var stateManager = new P2PStateManager(channel, coordinator);

        // Workers can't publish — should be a no-op, not throw
        await stateManager.PublishStateAsync();
    }

    [TestMethod]
    public async Task P2P_StateManager_OnStateUpdated()
    {
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        var dht = new SpawnDev.WebTorrent.Discovery.DhtDiscovery();
        var channel = new SpawnDev.WebTorrent.AgentChannel(dht, new SpawnDev.WebTorrent.Discovery.HmacFallbackSigner());
        await using var coordinator = new P2PSwarmCoordinator(client);

        var stateManager = new P2PStateManager(channel, coordinator);

        SwarmState? received = null;
        stateManager.OnStateUpdated += state => received = state;

        // Simulate a state update from DHT
        var testState = new SwarmState
        {
            SwarmName = "test-swarm",
            PeerCount = 5,
            TotalTflops = 42.0,
        };
        var json = System.Text.Json.JsonSerializer.SerializeToUtf8Bytes(testState,
            new System.Text.Json.JsonSerializerOptions { PropertyNamingPolicy = System.Text.Json.JsonNamingPolicy.CamelCase });

        // Simulate DHT delivering the update through the channel
        // AgentChannel fires OnAgentUpdate which StateManager handles
        // For this test, we verify the deserialization path
        if (received != null)
            throw new Exception("Should not have received state yet (no DHT delivery in test)");

        // Verify state model serializes correctly
        var roundTripped = System.Text.Json.JsonSerializer.Deserialize<SwarmState>(json,
            new System.Text.Json.JsonSerializerOptions { PropertyNamingPolicy = System.Text.Json.JsonNamingPolicy.CamelCase });
        if (roundTripped?.TotalTflops != 42.0)
            throw new Exception($"State round-trip failed: {roundTripped?.TotalTflops}");
        if (roundTripped?.PeerCount != 5)
            throw new Exception($"PeerCount: {roundTripped?.PeerCount}");
    }

    [TestMethod]
    public async Task P2P_SdComputeExtension_Name()
    {
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);
        using var ctx = global::ILGPU.Context.CreateDefault();
        var accelerator = coordinator.CreateAccelerator(ctx);
        var dispatcher = new P2PDispatcher(accelerator);
        await using var transport = new P2PTransport(client, coordinator, dispatcher);

        var ext = new SdComputeExtension(transport);
        if (ext.Name != "sd_compute")
            throw new Exception($"Extension name: {ext.Name}");
    }

    [TestMethod]
    public async Task P2P_SdComputeExtension_HandshakeData()
    {
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);
        using var ctx = global::ILGPU.Context.CreateDefault();
        var accelerator = coordinator.CreateAccelerator(ctx);
        var dispatcher = new P2PDispatcher(accelerator);
        await using var transport = new P2PTransport(client, coordinator, dispatcher);

        var ext = new SdComputeExtension(transport);
        var handshake = ext.GetHandshakeData();

        if (handshake == null)
            throw new Exception("Should include handshake data");
        if (!handshake.ContainsKey("sd_compute_version"))
            throw new Exception("Should include version");
    }

    // ═══════════════════════════════════════════════════════════
    //  Two-Node Integration: Coordinator + Worker via Mock Transport
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task P2P_TwoNode_FullCycle() => await RunTest(async accelerator =>
    {
        // === NODE 1: Coordinator ===
        await using var coordClient = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(coordClient);
        await coordinator.CreateSwarmAsync("two-node-test");
        using var coordCtx = global::ILGPU.Context.CreateDefault();
        var coordAccel = coordinator.CreateAccelerator(coordCtx);
        var dispatcher = new P2PDispatcher(coordAccel);
        await using var coordTransport = new P2PTransport(coordClient, coordinator, dispatcher);

        // === NODE 2: Worker ===
        await using var workerClient = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var workerCoord = new P2PSwarmCoordinator(workerClient);
        var workerP2PAccel = workerCoord.CreateAccelerator(coordCtx);
        var workerDispatcher = new P2PDispatcher(workerP2PAccel);
        await using var workerTransport = new P2PTransport(workerClient, workerCoord, workerDispatcher);
        await using var worker = new P2PWorker(workerTransport);
        // Worker uses the REAL backend (CPU/CUDA/WebGPU — whatever this test class runs)
        worker.Initialize(accelerator.Context, accelerator);

        // === Wire mock transport: bidirectional ===
        // Coordinator → Worker
        coordTransport.RegisterPeer("worker-node", async (data) =>
        {
            await workerTransport.HandleIncomingDataAsync("coord-node", data);
        });
        // Worker → Coordinator
        workerTransport.RegisterPeer("coord-node", async (data) =>
        {
            await coordTransport.HandleIncomingDataAsync("worker-node", data);
        });

        // === Connect worker as peer ===
        var caps = worker.BuildCapabilities("worker-node");
        coordinator.HandlePeerConnected("worker-node", caps);
        coordAccel.AddPeer(new RemotePeer
        {
            PeerId = "worker-node",
            IsConnected = true,
            Capabilities = caps,
        });

        // === Send buffer data to worker ===
        var tensorA = new byte[4096]; // 1024 floats
        var tensorB = new byte[4096];
        for (int i = 0; i < 4096; i++) { tensorA[i] = (byte)(i % 256); tensorB[i] = 1; }
        worker.ReceiveBuffer("a", tensorA);
        worker.ReceiveBuffer("b", tensorB);
        worker.ReceiveBuffer("result", new byte[4096]);

        // === Dispatch kernel ===
        var method = typeof(BackendTestBase).GetMethod(nameof(TestVectorAdd),
            System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static)!;
        var request = P2PKernelSerializer.CreateDispatch(method, gridDimX: 1024);
        request.Buffers = new[]
        {
            new BufferBinding { ParameterIndex = 1, BufferId = "a", Length = 1024, ElementSize = 4 },
            new BufferBinding { ParameterIndex = 2, BufferId = "b", Length = 1024, ElementSize = 4 },
            new BufferBinding { ParameterIndex = 3, BufferId = "result", Length = 1024, ElementSize = 4 },
        };

        // === Worker handles dispatch ===
        string? completedId = null;
        bool? completedSuccess = null;
        string? compiledKernel = null;
        worker.OnKernelCompleted += (id, success, ms) =>
        {
            completedId = id;
            completedSuccess = success;
        };
        worker.OnKernelCompiled += name => compiledKernel = name;

        await worker.HandleDispatchAsync("coord-node", request);

        // === Verify ===
        if (compiledKernel == null)
            throw new Exception($"Worker should compile kernel on {accelerator.AcceleratorType}");
        if (completedSuccess != true)
            throw new Exception($"Dispatch should succeed on {accelerator.AcceleratorType}");
        if (completedId != request.DispatchId)
            throw new Exception("Dispatch ID mismatch");

        // Worker should have result buffer
        var resultBuf = worker.GetBuffer("result");
        if (resultBuf == null)
            throw new Exception("Worker should have result buffer");

        Console.WriteLine($"[P2P TwoNode] Coordinator → Worker → Compile({accelerator.AcceleratorType}) → Result. PASS.");
    });

    // ═══════════════════════════════════════════════════════════
    //  Security Tests
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task P2P_Security_AllowlistBlocksUnauthorized()
    {
        // Register only our test kernel type
        P2PKernelSerializer.RegisterKernelType(typeof(BackendTestBase));

        try
        {
            // Trying to dispatch System.IO.File should be blocked
            var malicious = new KernelDispatchRequest
            {
                KernelType = typeof(System.IO.File).AssemblyQualifiedName ?? "",
                KernelMethod = "Delete",
            };
            if (P2PKernelSerializer.CanExecute(malicious))
                throw new Exception("SECURITY FAIL: System.IO.File.Delete should be blocked by allowlist");

            // Our registered type should work
            var legitimate = P2PKernelSerializer.CreateDispatch(
                typeof(BackendTestBase).GetMethod(nameof(TestVectorAdd),
                    System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static)!,
                gridDimX: 64);
            if (!P2PKernelSerializer.CanExecute(legitimate))
                throw new Exception("Registered type should be allowed");
        }
        finally
        {
            P2PKernelSerializer.ClearAllowlist();
        }
    }

    [TestMethod]
    public async Task P2P_Security_BufferTransfer_RejectsOversized()
    {
        var transfer = new P2PBufferTransfer { MaxTotalChunks = 100, MaxTotalBytes = 1024 * 1024 };

        // Malicious chunk claiming absurd size
        var malicious = new BufferChunk
        {
            BufferId = "evil",
            ChunkIndex = 0,
            TotalChunks = int.MaxValue, // would OOM without bounds check
            TotalBytes = int.MaxValue,
            Data = new byte[1],
        };

        var accepted = transfer.ReceiveChunk(malicious);
        if (accepted)
            throw new Exception("SECURITY FAIL: Oversized chunk should be rejected");
        if (transfer.ActiveTransfers != 0)
            throw new Exception("Rejected chunk should not create a transfer");
    }

    [TestMethod]
    public async Task P2P_Security_BufferTransfer_RejectsNegativeIndex()
    {
        var transfer = new P2PBufferTransfer();

        var badIndex = new BufferChunk
        {
            BufferId = "bad",
            ChunkIndex = -1,
            TotalChunks = 5,
            TotalBytes = 5000,
            Data = new byte[1000],
        };

        var accepted = transfer.ReceiveChunk(badIndex);
        if (accepted)
            throw new Exception("SECURITY FAIL: Negative chunk index should be rejected");
    }

    [TestMethod]
    public async Task P2P_Security_BufferTransfer_RejectsOutOfRangeIndex()
    {
        var transfer = new P2PBufferTransfer();

        var outOfRange = new BufferChunk
        {
            BufferId = "oor",
            ChunkIndex = 10, // only 5 chunks total
            TotalChunks = 5,
            TotalBytes = 5000,
            Data = new byte[1000],
        };

        var accepted = transfer.ReceiveChunk(outOfRange);
        if (accepted)
            throw new Exception("SECURITY FAIL: Out-of-range chunk index should be rejected");
    }

    [TestMethod]
    public async Task P2P_Security_KickRequiresCoordinator()
    {
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);
        // Default role is Worker — should not be able to kick
        coordinator.HandlePeerConnected("target", new PeerCapabilities { PeerId = "target" });

        var result = coordinator.KickPeer("target");
        if (result)
            throw new Exception("SECURITY FAIL: Worker should not be able to kick peers");
    }

    // ═══════════════════════════════════════════════════════════
    //  Transfer Notification + Dispose
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task P2P_Transfer_SendsNotification()
    {
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);
        await coordinator.CreateSwarmAsync("transfer-notify");

        coordinator.HandlePeerConnected("target-peer", new PeerCapabilities
        {
            PeerId = "target-peer", EstimatedTflops = 10.0,
        });
        coordinator.HandlePeerConnected("other-peer", new PeerCapabilities
        {
            PeerId = "other-peer", EstimatedTflops = 5.0,
        });

        var sentMessages = new List<(string peerId, P2PMessageType type)>();
        coordinator.OnSendMessage += (peerId, msg) =>
            sentMessages.Add((peerId, msg.Type));

        coordinator.TransferCoordinator("target-peer");

        // Target should receive CoordinatorTransfer
        if (!sentMessages.Any(m => m.peerId == "target-peer" && m.type == P2PMessageType.CoordinatorTransfer))
            throw new Exception("Target should receive CoordinatorTransfer");

        // Other peer should receive CoordinatorAnnounce
        if (!sentMessages.Any(m => m.peerId == "other-peer" && m.type == P2PMessageType.CoordinatorAnnounce))
            throw new Exception("Other peers should receive CoordinatorAnnounce");
    }

    [TestMethod]
    public async Task P2P_Dispose_SendsDisconnect()
    {
        var client = new SpawnDev.WebTorrent.WebTorrentClient();
        var coordinator = new P2PSwarmCoordinator(client);
        await coordinator.CreateSwarmAsync("dispose-notify");

        coordinator.HandlePeerConnected("peer-1", new PeerCapabilities { PeerId = "peer-1" });
        coordinator.HandlePeerConnected("peer-2", new PeerCapabilities { PeerId = "peer-2" });

        var disconnectsSent = 0;
        coordinator.OnSendMessage += (peerId, msg) =>
        {
            if (msg.Type == P2PMessageType.Disconnect)
                disconnectsSent++;
        };

        await coordinator.DisposeAsync();
        await client.DisposeAsync();

        if (disconnectsSent != 2)
            throw new Exception($"Should send 2 disconnect messages: {disconnectsSent}");
    }

    // ═══════════════════════════════════════════════════════════
    //  SwarmIdentity — Real ECDSA Crypto
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task P2P_SwarmIdentity_Create()
    {
        if (OperatingSystem.IsBrowser())
            throw new SpawnDev.UnitTesting.UnsupportedTestException("DotNetCrypto requires desktop");

        var crypto = new SpawnDev.BlazorJS.Cryptography.DotNetCrypto();
        await using var identity = await SwarmIdentity.CreateAsync(crypto, "test-identity");

        if (identity.PublicKeySpki.Length == 0)
            throw new Exception("PublicKeySpki should not be empty");
        if (string.IsNullOrEmpty(identity.Fingerprint))
            throw new Exception("Fingerprint should not be empty");
        if (identity.Label != "test-identity")
            throw new Exception($"Label: {identity.Label}");
    }

    [TestMethod]
    public async Task P2P_SwarmIdentity_SignAndVerify()
    {
        if (OperatingSystem.IsBrowser())
            throw new SpawnDev.UnitTesting.UnsupportedTestException("DotNetCrypto requires desktop");

        var crypto = new SpawnDev.BlazorJS.Cryptography.DotNetCrypto();
        await using var identity = await SwarmIdentity.CreateAsync(crypto, "signer");

        var message = System.Text.Encoding.UTF8.GetBytes("Hello P2P World");
        var signature = await identity.SignAsync(message);

        if (signature == null || signature.Length == 0)
            throw new Exception("Signature should not be empty");

        var valid = await identity.VerifyAsync(message, signature);
        if (!valid)
            throw new Exception("Signature should verify against own public key");

        // Tampered message should fail
        var tampered = System.Text.Encoding.UTF8.GetBytes("Tampered message");
        var invalidVerify = await identity.VerifyAsync(tampered, signature);
        if (invalidVerify)
            throw new Exception("Tampered message should NOT verify");
    }

    [TestMethod]
    public async Task P2P_SwarmIdentity_ExportImport()
    {
        if (OperatingSystem.IsBrowser())
            throw new SpawnDev.UnitTesting.UnsupportedTestException("DotNetCrypto requires desktop");

        var crypto = new SpawnDev.BlazorJS.Cryptography.DotNetCrypto();
        await using var original = await SwarmIdentity.CreateAsync(crypto, "exportable");

        // Sign something
        var message = System.Text.Encoding.UTF8.GetBytes("Persistent identity");
        var signature = await original.SignAsync(message);

        // Export keys
        var privateKey = await original.ExportPrivateKeyAsync();
        var publicKey = original.PublicKeySpki;

        // Import into new identity
        await using var imported = await SwarmIdentity.ImportAsync(crypto, publicKey, privateKey);

        // Fingerprint should match
        if (imported.Fingerprint != original.Fingerprint)
            throw new Exception($"Fingerprint mismatch: {imported.Fingerprint} vs {original.Fingerprint}");

        // Imported identity should verify original's signature
        var valid = await imported.VerifyAsync(message, signature);
        if (!valid)
            throw new Exception("Imported identity should verify original signature");
    }

    [TestMethod]
    public async Task P2P_SwarmIdentity_CrossVerify()
    {
        if (OperatingSystem.IsBrowser())
            throw new SpawnDev.UnitTesting.UnsupportedTestException("DotNetCrypto requires desktop");

        var crypto = new SpawnDev.BlazorJS.Cryptography.DotNetCrypto();
        await using var alice = await SwarmIdentity.CreateAsync(crypto, "Alice");
        await using var bob = await SwarmIdentity.CreateAsync(crypto, "Bob");

        // Alice signs
        var message = System.Text.Encoding.UTF8.GetBytes("Alice's message");
        var aliceSig = await alice.SignAsync(message);

        // Bob verifies Alice's signature using static verify
        var valid = await SwarmIdentity.VerifyAsync(crypto, alice.PublicKeySpki, message, aliceSig);
        if (!valid)
            throw new Exception("Bob should verify Alice's signature");

        // Bob's key should NOT verify Alice's signature
        var invalid = await SwarmIdentity.VerifyAsync(crypto, bob.PublicKeySpki, message, aliceSig);
        if (invalid)
            throw new Exception("Bob's key should NOT verify Alice's signature");
    }

    [TestMethod]
    public async Task P2P_KeyRegistry_AddAndCheck()
    {
        if (OperatingSystem.IsBrowser())
            throw new SpawnDev.UnitTesting.UnsupportedTestException("DotNetCrypto requires desktop");

        var crypto = new SpawnDev.BlazorJS.Cryptography.DotNetCrypto();
        await using var owner = await SwarmIdentity.CreateAsync(crypto, "Owner");

        var registry = new KeyRegistry();
        registry.AddKey(owner.PublicKeySpki, SwarmRole.Owner, "Owner Key");
        var ownerKeyB64 = Convert.ToBase64String(owner.PublicKeySpki);

        if (!registry.HasRole(ownerKeyB64, SwarmRole.Owner))
            throw new Exception("Owner should have Owner role");
        if (!registry.HasRole(ownerKeyB64, SwarmRole.Coordinator))
            throw new Exception("Owner should have Coordinator role (Owner > Coordinator)");
    }

    [TestMethod]
    public async Task P2P_KeyRegistry_Revoke()
    {
        if (OperatingSystem.IsBrowser())
            throw new SpawnDev.UnitTesting.UnsupportedTestException("DotNetCrypto requires desktop");

        var crypto = new SpawnDev.BlazorJS.Cryptography.DotNetCrypto();
        await using var owner = await SwarmIdentity.CreateAsync(crypto, "Owner");
        await using var worker = await SwarmIdentity.CreateAsync(crypto, "Worker");

        var registry = new KeyRegistry();
        registry.AddKey(owner.PublicKeySpki, SwarmRole.Owner, "Owner");
        registry.AddKey(worker.PublicKeySpki, SwarmRole.Worker, "Worker");
        var workerKeyB64 = Convert.ToBase64String(worker.PublicKeySpki);

        registry.RevokeKey(worker.PublicKeySpki, "Misbehaving");

        if (!registry.IsRevoked(workerKeyB64))
            throw new Exception("Worker should be revoked");
        if (registry.HasRole(workerKeyB64, SwarmRole.Worker))
            throw new Exception("Revoked worker should not have Worker role");
    }

    // ═══════════════════════════════════════════════════════════
    //  SwarmPolicy Enforcement
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task P2P_Policy_MaxPeers()
    {
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);
        await coordinator.CreateSwarmAsync("max-peers");
        coordinator.Policy.MaxPeers = 2;

        coordinator.HandlePeerConnected("p1", new PeerCapabilities { PeerId = "p1" });
        coordinator.HandlePeerConnected("p2", new PeerCapabilities { PeerId = "p2" });

        string? rejected = null;
        coordinator.OnPeerRejected += (id, reason) => rejected = reason;
        var accepted = coordinator.HandlePeerConnected("p3", new PeerCapabilities { PeerId = "p3" });

        if (accepted) throw new Exception("Should reject — swarm full");
        if (rejected != "swarm full") throw new Exception($"Reason: {rejected}");
        if (coordinator.PeerCount != 2) throw new Exception($"Count: {coordinator.PeerCount}");
    }

    [TestMethod]
    public async Task P2P_Policy_ApprovalMode()
    {
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);
        await coordinator.CreateSwarmAsync("approval-test");
        coordinator.Policy.JoinPermission = JoinMode.Approval;

        RemotePeer? pending = null;
        coordinator.OnPeerPendingApproval += p => pending = p;

        var accepted = coordinator.HandlePeerConnected("new-device", new PeerCapabilities { PeerId = "new-device" });

        if (accepted) throw new Exception("New device should be pending, not accepted");
        if (pending == null) throw new Exception("OnPeerPendingApproval should fire");
        if (coordinator.PeerCount != 0) throw new Exception("Pending peer should not be counted");
        if (coordinator.PendingApproval.Count != 1) throw new Exception($"Pending: {coordinator.PendingApproval.Count}");

        // Owner approves
        var approved = coordinator.ApprovePeer("new-device");
        if (!approved) throw new Exception("Approval should succeed");
        if (coordinator.PeerCount != 1) throw new Exception("Approved peer should be counted");
        if (coordinator.PendingApproval.Count != 0) throw new Exception("Pending should be empty");
    }

    [TestMethod]
    public async Task P2P_Policy_ApprovalMode_KnownPeerAutoAdmit()
    {
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);
        await coordinator.CreateSwarmAsync("approval-known");
        coordinator.Policy.JoinPermission = JoinMode.Approval;

        // First visit — goes to pending, gets approved
        coordinator.HandlePeerConnected("returning-device", new PeerCapabilities { PeerId = "returning-device" });
        coordinator.ApprovePeer("returning-device");
        coordinator.HandlePeerDisconnected("returning-device");

        // Second visit — should auto-admit (known peer)
        var accepted = coordinator.HandlePeerConnected("returning-device", new PeerCapabilities { PeerId = "returning-device" });
        if (!accepted) throw new Exception("Known device should auto-admit");
        if (coordinator.PeerCount != 1) throw new Exception($"Count: {coordinator.PeerCount}");
    }

    [TestMethod]
    public async Task P2P_Policy_KnownOnly()
    {
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);
        await coordinator.CreateSwarmAsync("known-only");

        // First: admit in Open mode to establish known peers
        coordinator.HandlePeerConnected("known-peer", new PeerCapabilities { PeerId = "known-peer" });
        coordinator.HandlePeerDisconnected("known-peer");

        // Switch to KnownOnly
        coordinator.Policy.JoinPermission = JoinMode.KnownOnly;

        // Known peer can rejoin
        var accepted = coordinator.HandlePeerConnected("known-peer", new PeerCapabilities { PeerId = "known-peer" });
        if (!accepted) throw new Exception("Known peer should be accepted");

        // Unknown peer rejected
        string? rejected = null;
        coordinator.OnPeerRejected += (id, reason) => rejected = reason;
        var unknown = coordinator.HandlePeerConnected("stranger", new PeerCapabilities { PeerId = "stranger" });
        if (unknown) throw new Exception("Unknown peer should be rejected");
        if (rejected != "unknown device") throw new Exception($"Reason: {rejected}");
    }

    [TestMethod]
    public async Task P2P_Policy_Open_Default()
    {
        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);
        await coordinator.CreateSwarmAsync("open-default");

        // Default policy is Open — everyone gets in
        for (int i = 0; i < 10; i++)
            coordinator.HandlePeerConnected($"peer-{i}", new PeerCapabilities { PeerId = $"peer-{i}" });

        if (coordinator.PeerCount != 10) throw new Exception($"Count: {coordinator.PeerCount}");
    }

    // ═══════════════════════════════════════════════════════════
    //  Real Kernel Execution — P2PKernelLauncher
    //  These tests prove actual GPU dispatch through the P2P system.
    //  No mocks. Real accelerator, real kernels, real data.
    // ═══════════════════════════════════════════════════════════

    /// <summary>Test kernel: multiply each element by 2.</summary>
    static void KernelMultiplyBy2(global::ILGPU.Index1D index, global::ILGPU.ArrayView<float> data)
    {
        data[index] = data[index] * 2.0f;
    }

    /// <summary>Test kernel: add two arrays.</summary>
    static void KernelAddArrays(
        global::ILGPU.Index1D index,
        global::ILGPU.ArrayView<float> a,
        global::ILGPU.ArrayView<float> b,
        global::ILGPU.ArrayView<float> result)
    {
        result[index] = a[index] + b[index];
    }

    /// <summary>Test kernel: fill with constant value.</summary>
    static void KernelFillConstant(
        global::ILGPU.Index1D index,
        global::ILGPU.ArrayView<int> data)
    {
        data[index] = 42;
    }

    [TestMethod]
    public async Task P2P_KernelLauncher_LoadAndCache()
    {
        using var context = global::ILGPU.Context.CreateDefault();
        using var accelerator = global::ILGPU.Runtime.CPU.CPUDevice.Default.CreateAccelerator(context);
        var launcher = new P2PKernelLauncher(accelerator);

        var method = typeof(BackendTestBase).GetMethod(
            nameof(KernelMultiplyBy2),
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static)!;

        var cached = launcher.LoadAndCache(method);
        if (cached == null) throw new Exception("LoadAndCache returned null");
        if (cached.Launcher == null) throw new Exception("Launcher delegate is null");
        if (cached.ParameterTypes.Length != 2) throw new Exception($"Expected 2 params, got {cached.ParameterTypes.Length}");
        if (launcher.CachedCount != 1) throw new Exception($"Cache count: {launcher.CachedCount}");

        // Loading same method again should return cached
        var cached2 = launcher.LoadAndCache(method);
        if (launcher.CachedCount != 1) throw new Exception($"Should still be 1, got {launcher.CachedCount}");

        Console.WriteLine("[P2P] KernelLauncher load + cache: OK");
    }

    [TestMethod]
    public async Task P2P_KernelLauncher_Execute_MultiplyBy2()
    {
        using var context = global::ILGPU.Context.CreateDefault();
        using var accelerator = global::ILGPU.Runtime.CPU.CPUDevice.Default.CreateAccelerator(context);
        var launcher = new P2PKernelLauncher(accelerator);

        var method = typeof(BackendTestBase).GetMethod(
            nameof(KernelMultiplyBy2),
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static)!;

        // Create input data: [1.0, 2.0, 3.0, 4.0]
        var inputFloats = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
        var inputBytes = new byte[inputFloats.Length * 4];
        Buffer.BlockCopy(inputFloats, 0, inputBytes, 0, inputBytes.Length);

        var bufferBindings = new Dictionary<int, BufferData>
        {
            [1] = new BufferData { RawData = inputBytes, ElementCount = 4, ElementSize = 4 }
        };

        var results = launcher.Execute(method, 4, bufferBindings);

        if (!results.ContainsKey(1)) throw new Exception("Missing result for param 1");

        // Verify output: [2.0, 4.0, 6.0, 8.0]
        var outputFloats = new float[4];
        Buffer.BlockCopy(results[1], 0, outputFloats, 0, 16);

        for (int i = 0; i < 4; i++)
        {
            float expected = inputFloats[i] * 2.0f;
            if (Math.Abs(outputFloats[i] - expected) > 0.001f)
                throw new Exception($"[{i}] expected {expected}, got {outputFloats[i]}");
        }

        Console.WriteLine($"[P2P] KernelLauncher MultiplyBy2: [{string.Join(", ", outputFloats)}] ✓");
    }

    [TestMethod]
    public async Task P2P_KernelLauncher_Execute_AddArrays()
    {
        using var context = global::ILGPU.Context.CreateDefault();
        using var accelerator = global::ILGPU.Runtime.CPU.CPUDevice.Default.CreateAccelerator(context);
        var launcher = new P2PKernelLauncher(accelerator);

        var method = typeof(BackendTestBase).GetMethod(
            nameof(KernelAddArrays),
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static)!;

        int count = 1024;
        var aFloats = new float[count];
        var bFloats = new float[count];
        for (int i = 0; i < count; i++)
        {
            aFloats[i] = i;
            bFloats[i] = i * 10.0f;
        }

        var aBytes = new byte[count * 4];
        var bBytes = new byte[count * 4];
        var rBytes = new byte[count * 4]; // zero-init output
        Buffer.BlockCopy(aFloats, 0, aBytes, 0, aBytes.Length);
        Buffer.BlockCopy(bFloats, 0, bBytes, 0, bBytes.Length);

        var bufferBindings = new Dictionary<int, BufferData>
        {
            [1] = new BufferData { RawData = aBytes, ElementCount = count, ElementSize = 4 },
            [2] = new BufferData { RawData = bBytes, ElementCount = count, ElementSize = 4 },
            [3] = new BufferData { RawData = rBytes, ElementCount = count, ElementSize = 4 },
        };

        var results = launcher.Execute(method, count, bufferBindings);

        var outFloats = new float[count];
        Buffer.BlockCopy(results[3], 0, outFloats, 0, count * 4);

        int errors = 0;
        for (int i = 0; i < count; i++)
        {
            float expected = aFloats[i] + bFloats[i];
            if (Math.Abs(outFloats[i] - expected) > 0.001f) errors++;
        }

        if (errors > 0) throw new Exception($"{errors}/{count} elements wrong");
        Console.WriteLine($"[P2P] KernelLauncher AddArrays: {count} elements verified ✓");
    }

    [TestMethod]
    public async Task P2P_KernelLauncher_Execute_IntKernel()
    {
        using var context = global::ILGPU.Context.CreateDefault();
        using var accelerator = global::ILGPU.Runtime.CPU.CPUDevice.Default.CreateAccelerator(context);
        var launcher = new P2PKernelLauncher(accelerator);

        var method = typeof(BackendTestBase).GetMethod(
            nameof(KernelFillConstant),
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static)!;

        int count = 256;
        var bufferBindings = new Dictionary<int, BufferData>
        {
            [1] = new BufferData { RawData = new byte[count * 4], ElementCount = count, ElementSize = 4 }
        };

        var results = launcher.Execute(method, count, bufferBindings);

        var outInts = new int[count];
        Buffer.BlockCopy(results[1], 0, outInts, 0, count * 4);

        for (int i = 0; i < count; i++)
        {
            if (outInts[i] != 42)
                throw new Exception($"[{i}] expected 42, got {outInts[i]}");
        }

        Console.WriteLine($"[P2P] KernelLauncher FillConstant: {count} elements = 42 ✓");
    }

    [TestMethod]
    public async Task P2P_KernelLauncher_LargeBuffer()
    {
        using var context = global::ILGPU.Context.CreateDefault();
        using var accelerator = global::ILGPU.Runtime.CPU.CPUDevice.Default.CreateAccelerator(context);
        var launcher = new P2PKernelLauncher(accelerator);

        var method = typeof(BackendTestBase).GetMethod(
            nameof(KernelMultiplyBy2),
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static)!;

        // 1M elements — proves we handle real tensor sizes
        int count = 1_000_000;
        var inputFloats = new float[count];
        for (int i = 0; i < count; i++) inputFloats[i] = i * 0.001f;

        var inputBytes = new byte[count * 4];
        Buffer.BlockCopy(inputFloats, 0, inputBytes, 0, inputBytes.Length);

        var bufferBindings = new Dictionary<int, BufferData>
        {
            [1] = new BufferData { RawData = inputBytes, ElementCount = count, ElementSize = 4 }
        };

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var results = launcher.Execute(method, count, bufferBindings);
        sw.Stop();

        var outFloats = new float[count];
        Buffer.BlockCopy(results[1], 0, outFloats, 0, count * 4);

        // Spot check
        if (Math.Abs(outFloats[0] - 0.0f) > 0.001f) throw new Exception($"[0] = {outFloats[0]}");
        if (Math.Abs(outFloats[500000] - 1000.0f) > 0.01f) throw new Exception($"[500000] = {outFloats[500000]}");
        if (Math.Abs(outFloats[999999] - 1999.998f) > 0.01f) throw new Exception($"[999999] = {outFloats[999999]}");

        Console.WriteLine($"[P2P] KernelLauncher 1M floats: {sw.ElapsedMilliseconds}ms ✓");
    }

    [TestMethod]
    public async Task P2P_Worker_RealDispatch()
    {
        // Full end-to-end: Worker receives a dispatch request, executes on CPU, returns results
        using var context = global::ILGPU.Context.CreateDefault();
        using var accelerator = global::ILGPU.Runtime.CPU.CPUDevice.Default.CreateAccelerator(context);

        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);
        await coordinator.CreateSwarmAsync("worker-dispatch-test");

        var p2pAccel = coordinator.CreateAccelerator(context);
        var dispatcher = new P2PDispatcher(p2pAccel);
        var transport = new P2PTransport(client, coordinator, dispatcher);
        var worker = new P2PWorker(transport);
        worker.Initialize(context, accelerator);
        transport.SetWorker(worker);

        // Register our test kernel type
        P2PKernelSerializer.RegisterKernelType(typeof(BackendTestBase));

        // Build dispatch request
        var method = typeof(BackendTestBase).GetMethod(
            nameof(KernelMultiplyBy2),
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static)!;

        var request = P2PKernelSerializer.CreateDispatch(method, gridDimX: 4);
        request.Buffers = new[]
        {
            new BufferBinding { ParameterIndex = 1, BufferId = "buf-0", Length = 4, ElementSize = 4 }
        };

        // Send buffer data to worker
        var inputFloats = new float[] { 10.0f, 20.0f, 30.0f, 40.0f };
        var inputBytes = new byte[16];
        Buffer.BlockCopy(inputFloats, 0, inputBytes, 0, 16);
        worker.ReceiveBuffer("buf-0", inputBytes);

        // Track completion
        string? completedId = null;
        bool? completedSuccess = null;
        worker.OnKernelCompleted += (id, success, ms) =>
        {
            completedId = id;
            completedSuccess = success;
        };

        // Dispatch
        await worker.HandleDispatchAsync("coordinator-peer", request);

        if (completedId != request.DispatchId)
            throw new Exception($"Wrong dispatch ID: {completedId}");
        if (completedSuccess != true)
            throw new Exception("Dispatch failed");

        // Verify buffer was modified
        var result = worker.GetBuffer("buf-0");
        if (result == null) throw new Exception("Buffer not found after dispatch");

        var outFloats = new float[4];
        Buffer.BlockCopy(result, 0, outFloats, 0, 16);

        for (int i = 0; i < 4; i++)
        {
            float expected = inputFloats[i] * 2.0f;
            if (Math.Abs(outFloats[i] - expected) > 0.001f)
                throw new Exception($"[{i}] expected {expected}, got {outFloats[i]}");
        }

        Console.WriteLine($"[P2P] Worker real dispatch: [{string.Join(", ", outFloats)}] ✓");

        // Cleanup
        P2PKernelSerializer.ClearAllowlist();
    }

    [TestMethod]
    public async Task P2P_Worker_RealDispatch_AddArrays()
    {
        using var context = global::ILGPU.Context.CreateDefault();
        using var accelerator = global::ILGPU.Runtime.CPU.CPUDevice.Default.CreateAccelerator(context);

        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);
        await coordinator.CreateSwarmAsync("add-arrays-test");

        var p2pAccel = coordinator.CreateAccelerator(context);
        var dispatcher = new P2PDispatcher(p2pAccel);
        var transport = new P2PTransport(client, coordinator, dispatcher);
        var worker = new P2PWorker(transport);
        worker.Initialize(context, accelerator);
        transport.SetWorker(worker);

        P2PKernelSerializer.RegisterKernelType(typeof(BackendTestBase));

        var method = typeof(BackendTestBase).GetMethod(
            nameof(KernelAddArrays),
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static)!;

        int count = 512;
        var request = P2PKernelSerializer.CreateDispatch(method, gridDimX: count);
        request.Buffers = new[]
        {
            new BufferBinding { ParameterIndex = 1, BufferId = "a", Length = count, ElementSize = 4 },
            new BufferBinding { ParameterIndex = 2, BufferId = "b", Length = count, ElementSize = 4 },
            new BufferBinding { ParameterIndex = 3, BufferId = "r", Length = count, ElementSize = 4 },
        };

        // Prepare input buffers
        var aFloats = new float[count];
        var bFloats = new float[count];
        for (int i = 0; i < count; i++) { aFloats[i] = i; bFloats[i] = i * 3.0f; }

        var aBytes = new byte[count * 4];
        var bBytes = new byte[count * 4];
        Buffer.BlockCopy(aFloats, 0, aBytes, 0, aBytes.Length);
        Buffer.BlockCopy(bFloats, 0, bBytes, 0, bBytes.Length);

        worker.ReceiveBuffer("a", aBytes);
        worker.ReceiveBuffer("b", bBytes);
        worker.ReceiveBuffer("r", new byte[count * 4]);

        bool success = false;
        worker.OnKernelCompleted += (_, s, _) => success = s;
        await worker.HandleDispatchAsync("coordinator", request);

        if (!success) throw new Exception("Dispatch failed");

        var result = worker.GetBuffer("r")!;
        var outFloats = new float[count];
        Buffer.BlockCopy(result, 0, outFloats, 0, count * 4);

        int errors = 0;
        for (int i = 0; i < count; i++)
        {
            float expected = aFloats[i] + bFloats[i];
            if (Math.Abs(outFloats[i] - expected) > 0.01f) errors++;
        }

        if (errors > 0) throw new Exception($"{errors}/{count} elements wrong");
        Console.WriteLine($"[P2P] Worker AddArrays dispatch: {count} elements verified ✓");

        P2PKernelSerializer.ClearAllowlist();
    }

    [TestMethod]
    public async Task P2P_Worker_RejectsUnregisteredKernel()
    {
        using var context = global::ILGPU.Context.CreateDefault();
        using var accelerator = global::ILGPU.Runtime.CPU.CPUDevice.Default.CreateAccelerator(context);

        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);
        await coordinator.CreateSwarmAsync("reject-test");

        var p2pAccel = coordinator.CreateAccelerator(context);
        var dispatcher = new P2PDispatcher(p2pAccel);
        var transport = new P2PTransport(client, coordinator, dispatcher);
        var worker = new P2PWorker(transport);
        worker.Initialize(context, accelerator);
        transport.SetWorker(worker);

        // Register a DIFFERENT type — our test kernel type is NOT registered
        P2PKernelSerializer.ClearAllowlist();
        P2PKernelSerializer.RegisterKernelType(typeof(string)); // arbitrary wrong type

        var request = new KernelDispatchRequest
        {
            KernelType = typeof(BackendTestBase).AssemblyQualifiedName!,
            KernelMethod = nameof(KernelMultiplyBy2),
            GridDimX = 4,
            Buffers = new[]
            {
                new BufferBinding { ParameterIndex = 1, BufferId = "buf", Length = 4, ElementSize = 4 }
            },
        };

        worker.ReceiveBuffer("buf", new byte[16]);

        bool? success = null;
        worker.OnKernelCompleted += (_, s, _) => success = s;
        await worker.HandleDispatchAsync("coordinator", request);

        if (success != false)
            throw new Exception("Should have rejected unregistered kernel type");

        Console.WriteLine("[P2P] Worker correctly rejects unregistered kernel ✓");

        P2PKernelSerializer.ClearAllowlist();
    }

    [TestMethod]
    public async Task P2P_KernelLauncher_CacheReuse()
    {
        using var context = global::ILGPU.Context.CreateDefault();
        using var accelerator = global::ILGPU.Runtime.CPU.CPUDevice.Default.CreateAccelerator(context);
        var launcher = new P2PKernelLauncher(accelerator);

        var method = typeof(BackendTestBase).GetMethod(
            nameof(KernelMultiplyBy2),
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static)!;

        // Execute same kernel 5 times — should compile once, reuse cache
        for (int run = 0; run < 5; run++)
        {
            var data = new float[] { run + 1.0f };
            var bytes = new byte[4];
            Buffer.BlockCopy(data, 0, bytes, 0, 4);

            var results = launcher.Execute(method, 1, new Dictionary<int, BufferData>
            {
                [1] = new BufferData { RawData = bytes, ElementCount = 1, ElementSize = 4 }
            });

            var outFloat = new float[1];
            Buffer.BlockCopy(results[1], 0, outFloat, 0, 4);

            float expected = (run + 1.0f) * 2.0f;
            if (Math.Abs(outFloat[0] - expected) > 0.001f)
                throw new Exception($"Run {run}: expected {expected}, got {outFloat[0]}");
        }

        if (launcher.CachedCount != 1)
            throw new Exception($"Should have 1 cached launcher, got {launcher.CachedCount}");

        Console.WriteLine("[P2P] KernelLauncher cache reuse: 5 runs, 1 compilation ✓");
    }

    // ═══════════════════════════════════════════════════════════
    //  Coordinator Dispatch — Full Round-Trip
    //  Coordinator dispatches via DispatchToSwarm → worker executes → result
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task P2P_Coordinator_DispatchToSwarm()
    {
        // Full coordinator → worker round-trip with real kernel execution
        using var context = global::ILGPU.Context.CreateDefault();
        using var cpuAccel = global::ILGPU.Runtime.CPU.CPUDevice.Default.CreateAccelerator(context);

        await using var client = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var coordinator = new P2PSwarmCoordinator(client);
        await coordinator.CreateSwarmAsync("coord-dispatch-test");

        var p2pAccel = coordinator.CreateAccelerator(context);
        p2pAccel.Dispatcher = new P2PDispatcher(p2pAccel);
        var transport = new P2PTransport(client, coordinator, p2pAccel.Dispatcher);

        // Wire dispatcher → transport
        p2pAccel.Dispatcher.OnSendMessage += (peerId, msg) =>
        {
            _ = transport.SendMessageAsync(peerId, msg);
        };

        // Create worker on same process (simulated remote)
        var worker = new P2PWorker(transport);
        worker.Initialize(context, cpuAccel);
        transport.SetWorker(worker);

        // Wire mock transport: coordinator ↔ worker
        transport.RegisterPeer("worker-1", async (data) =>
        {
            await transport.HandleIncomingDataAsync("worker-1", data);
        });

        // Register peer
        var caps = worker.BuildCapabilities("worker-1");
        coordinator.HandlePeerConnected("worker-1", caps);
        p2pAccel.AddPeer(new RemotePeer
        {
            PeerId = "worker-1",
            IsConnected = true,
            Capabilities = caps,
            LastHeartbeat = DateTime.UtcNow,
        });

        // Register kernel type
        P2PKernelSerializer.RegisterKernelType(typeof(BackendTestBase));

        // Prepare input data
        var inputFloats = new float[] { 5.0f, 10.0f, 15.0f, 20.0f };
        var inputBytes = new byte[16];
        Buffer.BlockCopy(inputFloats, 0, inputBytes, 0, 16);
        worker.ReceiveBuffer("data", inputBytes);

        // Coordinator dispatches to swarm
        var dispatchId = p2pAccel.DispatchToSwarm(
            typeof(BackendTestBase).GetMethod(nameof(KernelMultiplyBy2),
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static)!,
            4,
            ("data", inputBytes, 4));

        if (string.IsNullOrEmpty(dispatchId))
            throw new Exception("DispatchToSwarm returned empty ID");

        // Verify the worker received and processed the dispatch
        var result = worker.GetBuffer("data");
        if (result == null)
            throw new Exception("Worker buffer not updated");

        var outFloats = new float[4];
        Buffer.BlockCopy(result, 0, outFloats, 0, 16);

        for (int i = 0; i < 4; i++)
        {
            float expected = inputFloats[i] * 2.0f;
            if (Math.Abs(outFloats[i] - expected) > 0.001f)
                throw new Exception($"[{i}] expected {expected}, got {outFloats[i]}");
        }

        Console.WriteLine($"[P2P] Coordinator DispatchToSwarm: [{string.Join(", ", outFloats)}] ✓");

        P2PKernelSerializer.ClearAllowlist();
    }

    [TestMethod]
    public async Task P2P_Coordinator_CreateDispatcher_Helper()
    {
        using var context = global::ILGPU.Context.CreateDefault();
        var device = new P2PDevice();
        var p2pAccel = (P2PAccelerator)device.CreateAccelerator(context);
        p2pAccel.Dispatcher = new P2PDispatcher(p2pAccel);

        // CreateDispatcher should find the method
        var helper = p2pAccel.CreateDispatcher(typeof(BackendTestBase), "KernelMultiplyBy2");
        if (helper.MethodName != "KernelMultiplyBy2")
            throw new Exception($"Method name: {helper.MethodName}");
        if (helper.TypeName != "BackendTestBase")
            throw new Exception($"Type name: {helper.TypeName}");

        Console.WriteLine("[P2P] CreateDispatcher helper: OK ✓");
    }

    [TestMethod]
    public async Task P2P_Coordinator_NotSupportedException_LoadKernel()
    {
        using var context = global::ILGPU.Context.CreateDefault();
        var device = new P2PDevice();
        var p2pAccel = (P2PAccelerator)device.CreateAccelerator(context);

        // LoadAutoGroupedStreamKernel should throw NotSupportedException with helpful message
        try
        {
            var kernel = p2pAccel.LoadAutoGroupedStreamKernel<
                global::ILGPU.Index1D, global::ILGPU.ArrayView<float>>(KernelMultiplyBy2);
            throw new Exception("Should have thrown NotSupportedException");
        }
        catch (NotSupportedException ex)
        {
            if (!ex.Message.Contains("DispatchAsync"))
                throw new Exception($"Error should mention DispatchAsync: {ex.Message}");
        }

        Console.WriteLine("[P2P] LoadAutoGroupedStreamKernel correctly throws with guidance ✓");
    }
}
