using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using NUnit.Framework;
using SpawnDev.BlazorJS.Cryptography;
using SpawnDev.ILGPU.P2P.IntegrationTests.Infrastructure;

namespace SpawnDev.ILGPU.P2P.IntegrationTests;

/// <summary>
/// Phase 4: Security, RBAC, Identity, and Policy tests.
/// Proves Ed25519 signatures, coordinator authority verification,
/// kick/block enforcement, key registry, and join policies all work correctly.
/// Uses real crypto (DotNetCrypto) - no mocks.
/// </summary>
[TestFixture]
public class SecurityTests
{
    private IPortableCrypto _crypto = null!;

    [SetUp]
    public void SetUp()
    {
        _crypto = new DotNetCrypto();
    }

    /// <summary>
    /// Worker verifies that a dispatch comes from an authorized coordinator
    /// via Ed25519 signature on the dispatch request.
    /// </summary>
    [Test]
    public async Task Security_SignedDispatch_Verified()
    {
        P2PKernelSerializer.RegisterKernelType(typeof(P2PTestKernels));

        // Create owner identity and registry
        var ownerIdentity = await SwarmIdentity.CreateAsync(_crypto, "owner");
        var registry = new KeyRegistry();
        registry.AddKey(ownerIdentity.PublicKeySpki, SwarmRole.Owner, "owner");
        await registry.SignAsync(ownerIdentity);

        // Create worker with registry
        using var context = Context.Create(builder => builder.CPU());
        using var accelerator = context.CreateCPUAccelerator(0);
        var coordinator = new P2PSwarmCoordinator(new SpawnDev.WebTorrent.WebTorrentClient());
        var p2pAccel = coordinator.CreateAccelerator(context);
        var dispatcher = new P2PDispatcher(p2pAccel);
        var transport = new P2PTransport(
            new SpawnDev.WebTorrent.WebTorrentClient(), coordinator, dispatcher);
        var worker = new P2PWorker(transport);
        worker.Initialize(context, accelerator);
        worker.SetKeyRegistry(registry);

        // Create dispatch with coordinator's public key
        var method = typeof(P2PTestKernels).GetMethod(nameof(P2PTestKernels.Identity))!;
        var request = P2PKernelSerializer.CreateDispatch(method, gridDimX: 64);
        request.CoordinatorPublicKey = Convert.ToBase64String(ownerIdentity.PublicKeySpki);
        request.Buffers = new[]
        {
            new BufferBinding { ParameterIndex = 1, BufferId = "in", Length = 64, ElementSize = 4 },
            new BufferBinding { ParameterIndex = 2, BufferId = "out", Length = 64, ElementSize = 4 },
        };

        // Provide input data
        var inputBytes = new byte[64 * 4];
        new Random(42).NextBytes(inputBytes);
        worker.ReceiveBuffer("in", inputBytes);

        bool completed = false;
        bool success = false;
        worker.OnKernelCompleted += (id, s, ms) => { completed = true; success = s; };

        await worker.HandleDispatchAsync("coordinator", request);

        Assert.That(completed, Is.True, "Dispatch should complete");
        Assert.That(success, Is.True, "Dispatch from authorized coordinator should succeed");
    }

    /// <summary>
    /// Worker rejects a dispatch from an unauthorized peer (no public key provided).
    /// </summary>
    [Test]
    public async Task Security_UnsignedDispatch_Rejected()
    {
        P2PKernelSerializer.RegisterKernelType(typeof(P2PTestKernels));

        // Create owner identity and registry with entries
        var ownerIdentity = await SwarmIdentity.CreateAsync(_crypto, "owner");
        var registry = new KeyRegistry();
        registry.AddKey(ownerIdentity.PublicKeySpki, SwarmRole.Owner, "owner");
        await registry.SignAsync(ownerIdentity);

        using var context = Context.Create(builder => builder.CPU());
        using var accelerator = context.CreateCPUAccelerator(0);
        var coordinator = new P2PSwarmCoordinator(new SpawnDev.WebTorrent.WebTorrentClient());
        var p2pAccel = coordinator.CreateAccelerator(context);
        var dispatcher = new P2PDispatcher(p2pAccel);
        var transport = new P2PTransport(
            new SpawnDev.WebTorrent.WebTorrentClient(), coordinator, dispatcher);
        var worker = new P2PWorker(transport);
        worker.Initialize(context, accelerator);
        worker.SetKeyRegistry(registry);

        // Create dispatch WITHOUT coordinator public key - should be rejected
        var method = typeof(P2PTestKernels).GetMethod(nameof(P2PTestKernels.Identity))!;
        var request = P2PKernelSerializer.CreateDispatch(method, gridDimX: 64);
        // Intentionally NOT setting CoordinatorPublicKey
        request.Buffers = new[]
        {
            new BufferBinding { ParameterIndex = 1, BufferId = "in", Length = 64, ElementSize = 4 },
            new BufferBinding { ParameterIndex = 2, BufferId = "out", Length = 64, ElementSize = 4 },
        };

        worker.ReceiveBuffer("in", new byte[64 * 4]);

        bool completed = false;
        bool success = false;
        worker.OnKernelCompleted += (id, s, ms) => { completed = true; success = s; };

        await worker.HandleDispatchAsync("attacker", request);

        Assert.That(completed, Is.True, "Dispatch event should fire");
        Assert.That(success, Is.False,
            "Dispatch without coordinator public key should be REJECTED when registry has entries");
    }

    /// <summary>
    /// A revoked key cannot dispatch work - the worker checks the revocation list.
    /// </summary>
    [Test]
    public async Task Security_RevokedKey_Rejected()
    {
        P2PKernelSerializer.RegisterKernelType(typeof(P2PTestKernels));

        // Create owner + coordinator identities
        var ownerIdentity = await SwarmIdentity.CreateAsync(_crypto, "owner");
        var coordIdentity = await SwarmIdentity.CreateAsync(_crypto, "coordinator");

        var registry = new KeyRegistry();
        registry.AddKey(ownerIdentity.PublicKeySpki, SwarmRole.Owner, "owner");
        registry.AddKey(coordIdentity.PublicKeySpki, SwarmRole.Coordinator, "coordinator");
        // Now revoke the coordinator
        registry.RevokeKey(coordIdentity.PublicKeySpki, "compromised");
        await registry.SignAsync(ownerIdentity);

        using var context = Context.Create(builder => builder.CPU());
        using var accelerator = context.CreateCPUAccelerator(0);
        var coordinator = new P2PSwarmCoordinator(new SpawnDev.WebTorrent.WebTorrentClient());
        var p2pAccel = coordinator.CreateAccelerator(context);
        var dispatcher = new P2PDispatcher(p2pAccel);
        var transport = new P2PTransport(
            new SpawnDev.WebTorrent.WebTorrentClient(), coordinator, dispatcher);
        var worker = new P2PWorker(transport);
        worker.Initialize(context, accelerator);
        worker.SetKeyRegistry(registry);

        var method = typeof(P2PTestKernels).GetMethod(nameof(P2PTestKernels.Identity))!;
        var request = P2PKernelSerializer.CreateDispatch(method, gridDimX: 64);
        request.CoordinatorPublicKey = Convert.ToBase64String(coordIdentity.PublicKeySpki);
        request.Buffers = new[]
        {
            new BufferBinding { ParameterIndex = 1, BufferId = "in", Length = 64, ElementSize = 4 },
            new BufferBinding { ParameterIndex = 2, BufferId = "out", Length = 64, ElementSize = 4 },
        };
        worker.ReceiveBuffer("in", new byte[64 * 4]);

        bool completed = false;
        bool success = false;
        worker.OnKernelCompleted += (id, s, ms) => { completed = true; success = s; };

        await worker.HandleDispatchAsync("revoked-coord", request);

        Assert.That(completed, Is.True, "Dispatch event should fire");
        Assert.That(success, Is.False,
            "Dispatch from revoked key should be REJECTED");
    }

    /// <summary>
    /// Kick removes a connected peer from the swarm.
    /// Requires Coordinator authority.
    /// </summary>
    [Test]
    public async Task Security_Kick_RemovesPeer()
    {
        var coordinator = new P2PSwarmCoordinator(new SpawnDev.WebTorrent.WebTorrentClient());
        await coordinator.CreateSwarmAsync("kick-test");

        coordinator.HandlePeerConnected("peer-1", new PeerCapabilities
        {
            PeerId = "peer-1",
            EstimatedTflops = 2.0,
            IsCharging = true,
        });

        Assert.That(coordinator.PeerCount, Is.EqualTo(1), "Should have 1 peer");

        string? kickedPeer = null;
        coordinator.OnPeerKicked += (peerId, reason) => { kickedPeer = peerId; };

        bool kicked = coordinator.KickPeer("peer-1", "test kick");

        Assert.That(kicked, Is.True, "Kick should succeed for coordinator");
        Assert.That(kickedPeer, Is.EqualTo("peer-1"), "Kicked event should fire");
        Assert.That(coordinator.PeerCount, Is.EqualTo(0), "Peer should be removed");
    }

    /// <summary>
    /// Blocked peer cannot reconnect.
    /// </summary>
    [Test]
    public async Task Security_Block_PreventsReconnect()
    {
        var coordinator = new P2PSwarmCoordinator(new SpawnDev.WebTorrent.WebTorrentClient());
        await coordinator.CreateSwarmAsync("block-test");

        // Connect and then block
        coordinator.HandlePeerConnected("bad-peer", new PeerCapabilities
        {
            PeerId = "bad-peer",
            EstimatedTflops = 1.0,
            IsCharging = true,
        });

        bool blocked = coordinator.BlockPeer("bad-peer", "misbehavior");
        Assert.That(blocked, Is.True, "Block should succeed");
        Assert.That(coordinator.PeerCount, Is.EqualTo(0), "Blocked peer should be disconnected");
        Assert.That(coordinator.IsPeerBlocked("bad-peer"), Is.True, "Peer should be in block list");

        // Try to reconnect - should be rejected
        string? rejectedPeer = null;
        string? rejectedReason = null;
        coordinator.OnPeerRejected += (peerId, reason) =>
        {
            rejectedPeer = peerId;
            rejectedReason = reason;
        };

        bool reconnected = coordinator.HandlePeerConnected("bad-peer", new PeerCapabilities
        {
            PeerId = "bad-peer",
            EstimatedTflops = 1.0,
            IsCharging = true,
        });

        Assert.That(reconnected, Is.False, "Blocked peer should not be admitted");
        Assert.That(rejectedPeer, Is.EqualTo("bad-peer"), "Rejection event should fire");
        Assert.That(rejectedReason, Is.EqualTo("blocked"), "Rejection reason should be 'blocked'");
    }

    /// <summary>
    /// Ed25519 identity: create, sign, and verify - proves real crypto works end-to-end.
    /// </summary>
    [Test]
    public async Task Identity_CrossVerify_Ed25519()
    {
        // Create two identities
        var identity1 = await SwarmIdentity.CreateAsync(_crypto, "peer-1");
        var identity2 = await SwarmIdentity.CreateAsync(_crypto, "peer-2");

        // Identity 1 signs some data
        var data = System.Text.Encoding.UTF8.GetBytes("dispatch:vectoradd:1024");
        var signature = await identity1.SignAsync(data);

        Assert.That(signature.Length, Is.EqualTo(64),
            "Ed25519 signature should always be 64 bytes");

        // Identity 2 verifies using identity 1's public key
        var verified = await SwarmIdentity.VerifyAsync(
            _crypto, identity1.PublicKeySpki, data, signature);
        Assert.That(verified, Is.True,
            "Verification of valid signature should succeed");

        // Tampered data should fail verification
        var tampered = System.Text.Encoding.UTF8.GetBytes("dispatch:vectoradd:9999");
        var tamperedVerified = await SwarmIdentity.VerifyAsync(
            _crypto, identity1.PublicKeySpki, tampered, signature);
        Assert.That(tamperedVerified, Is.False,
            "Verification of tampered data should fail");

        // Wrong key should fail verification
        var wrongKeyVerified = await SwarmIdentity.VerifyAsync(
            _crypto, identity2.PublicKeySpki, data, signature);
        Assert.That(wrongKeyVerified, Is.False,
            "Verification with wrong key should fail");

        await identity1.DisposeAsync();
        await identity2.DisposeAsync();
    }

    /// <summary>
    /// Signed role assignment: granter creates assignment, recipient verifies.
    /// </summary>
    [Test]
    public async Task Identity_RoleAssignment_Signed()
    {
        var owner = await SwarmIdentity.CreateAsync(_crypto, "owner");
        var worker = await SwarmIdentity.CreateAsync(_crypto, "new-coordinator");

        var assignment = await RoleAssignment.CreateAsync(
            owner, "peer-1", worker.PublicKeySpki, SwarmRole.Coordinator);

        Assert.That(assignment.Role, Is.EqualTo(SwarmRole.Coordinator));
        Assert.That(assignment.PeerId, Is.EqualTo("peer-1"));
        Assert.That(assignment.Signature, Is.Not.Empty);

        // Verify the signature
        var verified = await assignment.VerifyAsync(_crypto);
        Assert.That(verified, Is.True,
            "Role assignment signature should verify correctly");

        await owner.DisposeAsync();
        await worker.DisposeAsync();
    }

    /// <summary>
    /// Election respects registry: Admin wins over higher-TFLOPS Worker.
    /// </summary>
    [Test]
    public async Task Identity_Election_RespectsRegistry()
    {
        var owner = await SwarmIdentity.CreateAsync(_crypto, "owner");
        var adminIdentity = await SwarmIdentity.CreateAsync(_crypto, "admin");
        var workerIdentity = await SwarmIdentity.CreateAsync(_crypto, "fast-worker");

        var registry = new KeyRegistry();
        registry.AddKey(owner.PublicKeySpki, SwarmRole.Owner, "owner");
        registry.AddKey(adminIdentity.PublicKeySpki, SwarmRole.Admin, "admin");
        registry.AddKey(workerIdentity.PublicKeySpki, SwarmRole.Worker, "fast-worker");
        await registry.SignAsync(owner);

        var coordinator = new P2PSwarmCoordinator(new SpawnDev.WebTorrent.WebTorrentClient());
        coordinator.UpdateRegistry(registry);

        // Admin with low TFLOPS
        coordinator.HandlePeerConnected("admin-peer", new PeerCapabilities
        {
            PeerId = "admin-peer",
            EstimatedTflops = 1.0,
            PublicKey = Convert.ToBase64String(adminIdentity.PublicKeySpki),
            IsCharging = true,
        });

        // Worker with high TFLOPS
        coordinator.HandlePeerConnected("worker-peer", new PeerCapabilities
        {
            PeerId = "worker-peer",
            EstimatedTflops = 20.0,
            PublicKey = Convert.ToBase64String(workerIdentity.PublicKeySpki),
            IsCharging = true,
        });

        var winner = coordinator.ElectCoordinator();

        // Admin should win because registry-based election prefers higher roles,
        // even if the worker has more TFLOPS
        Assert.That(winner, Is.EqualTo("admin-peer"),
            $"Admin should win election over higher-TFLOPS Worker, got '{winner}'");

        await owner.DisposeAsync();
        await adminIdentity.DisposeAsync();
        await workerIdentity.DisposeAsync();
    }

    /// <summary>
    /// Open policy: any peer joins without credentials.
    /// </summary>
    [Test]
    public void Policy_Open_AnyPeerJoins()
    {
        var coordinator = new P2PSwarmCoordinator(new SpawnDev.WebTorrent.WebTorrentClient());
        coordinator.Policy = new SwarmPolicy { JoinPermission = JoinMode.Open };

        var result = coordinator.HandlePeerConnected("anonymous", new PeerCapabilities
        {
            PeerId = "anonymous",
            EstimatedTflops = 1.0,
            IsCharging = true,
        });

        Assert.That(result, Is.True, "Open policy should admit any peer");
        Assert.That(coordinator.PeerCount, Is.EqualTo(1));
    }

    /// <summary>
    /// KnownOnly policy: unknown peer is rejected.
    /// </summary>
    [Test]
    public void Policy_KnownOnly_NewRejected()
    {
        var coordinator = new P2PSwarmCoordinator(new SpawnDev.WebTorrent.WebTorrentClient());
        coordinator.Policy = new SwarmPolicy { JoinPermission = JoinMode.KnownOnly };

        string? rejectedPeer = null;
        coordinator.OnPeerRejected += (peerId, reason) => { rejectedPeer = peerId; };

        var result = coordinator.HandlePeerConnected("stranger", new PeerCapabilities
        {
            PeerId = "stranger",
            EstimatedTflops = 1.0,
            IsCharging = true,
        });

        Assert.That(result, Is.False, "KnownOnly should reject unknown peers");
        Assert.That(rejectedPeer, Is.EqualTo("stranger"));
        Assert.That(coordinator.PeerCount, Is.EqualTo(0));
    }

    /// <summary>
    /// MaxPeers policy: excess peers are rejected.
    /// </summary>
    [Test]
    public void Policy_MaxPeers_ExcessRejected()
    {
        var coordinator = new P2PSwarmCoordinator(new SpawnDev.WebTorrent.WebTorrentClient());
        coordinator.Policy = new SwarmPolicy { MaxPeers = 1 };

        // First peer admitted
        var result1 = coordinator.HandlePeerConnected("first", new PeerCapabilities
        {
            PeerId = "first",
            EstimatedTflops = 1.0,
            IsCharging = true,
        });
        Assert.That(result1, Is.True, "First peer should be admitted");

        // Second peer rejected (max 1)
        string? rejectedReason = null;
        coordinator.OnPeerRejected += (peerId, reason) => { rejectedReason = reason; };

        var result2 = coordinator.HandlePeerConnected("second", new PeerCapabilities
        {
            PeerId = "second",
            EstimatedTflops = 1.0,
            IsCharging = true,
        });
        Assert.That(result2, Is.False, "Second peer should be rejected (MaxPeers=1)");
        Assert.That(rejectedReason, Does.Contain("full"),
            $"Rejection reason should mention 'full', got: {rejectedReason}");
    }

    /// <summary>
    /// InviteOnly policy: peer without registry entry is rejected.
    /// </summary>
    [Test]
    public async Task Policy_InviteOnly_RequiresRegistry()
    {
        var owner = await SwarmIdentity.CreateAsync(_crypto, "owner");
        var registry = new KeyRegistry();
        registry.AddKey(owner.PublicKeySpki, SwarmRole.Owner, "owner");
        await registry.SignAsync(owner);

        var coordinator = new P2PSwarmCoordinator(new SpawnDev.WebTorrent.WebTorrentClient());
        coordinator.SetIdentity(owner);
        coordinator.Policy = new SwarmPolicy { JoinPermission = JoinMode.InviteOnly };
        coordinator.UpdateRegistry(registry);

        // Random peer with a key but not in registry
        var randomIdentity = await SwarmIdentity.CreateAsync(_crypto, "random");
        string? rejectedReason = null;
        coordinator.OnPeerRejected += (peerId, reason) => { rejectedReason = reason; };

        var result = coordinator.HandlePeerConnected("uninvited", new PeerCapabilities
        {
            PeerId = "uninvited",
            EstimatedTflops = 5.0,
            PublicKey = Convert.ToBase64String(randomIdentity.PublicKeySpki),
            IsCharging = true,
        });

        Assert.That(result, Is.False, "InviteOnly should reject peer not in registry");
        Assert.That(rejectedReason, Is.EqualTo("invite only"));

        await owner.DisposeAsync();
        await randomIdentity.DisposeAsync();
    }

    /// <summary>
    /// Key registry: last owner cannot be revoked (prevents lockout).
    /// </summary>
    [Test]
    public async Task Security_LastOwner_CannotBeRevoked()
    {
        var owner = await SwarmIdentity.CreateAsync(_crypto, "owner");
        var registry = new KeyRegistry();
        registry.AddKey(owner.PublicKeySpki, SwarmRole.Owner, "owner");

        bool revoked = registry.RevokeKey(owner.PublicKeySpki, "test");

        Assert.That(revoked, Is.False,
            "Revoking the last owner should fail (prevents lockout)");
        Assert.That(registry.Revocations.Count, Is.EqualTo(0),
            "No revocation should be recorded");

        await owner.DisposeAsync();
    }

    /// <summary>
    /// Key registry: sequence number increases on modifications (replay protection).
    /// </summary>
    [Test]
    public async Task Security_RegistrySequence_Monotonic()
    {
        var owner = await SwarmIdentity.CreateAsync(_crypto, "owner");
        var registry = new KeyRegistry();

        long seq0 = registry.Sequence;
        registry.AddKey(owner.PublicKeySpki, SwarmRole.Owner, "owner");
        long seq1 = registry.Sequence;

        var worker = await SwarmIdentity.CreateAsync(_crypto, "worker");
        registry.AddKey(worker.PublicKeySpki, SwarmRole.Worker, "worker");
        long seq2 = registry.Sequence;

        registry.RevokeKey(worker.PublicKeySpki, "test");
        long seq3 = registry.Sequence;

        Assert.That(seq1, Is.GreaterThan(seq0), "AddKey should increment sequence");
        Assert.That(seq2, Is.GreaterThan(seq1), "Second AddKey should increment");
        Assert.That(seq3, Is.GreaterThan(seq2), "RevokeKey should increment");

        await owner.DisposeAsync();
        await worker.DisposeAsync();
    }

    /// <summary>
    /// P2PProtocol message signing and verification round-trip.
    /// </summary>
    [Test]
    public async Task Security_MessageSignVerify_RoundTrip()
    {
        var identity = await SwarmIdentity.CreateAsync(_crypto, "signer");

        var message = new P2PMessage
        {
            Type = P2PMessageType.KernelDispatch,
            Payload = System.Text.Json.JsonSerializer.SerializeToElement(new { test = "data" }),
        };

        await P2PProtocol.SignMessageAsync(message, identity);

        Assert.That(message.SenderPublicKey, Is.Not.Null.And.Not.Empty);
        Assert.That(message.Signature, Is.Not.Null.And.Not.Empty);

        var verified = await P2PProtocol.VerifyMessageAsync(message, _crypto);
        Assert.That(verified, Is.True, "Signed message should verify successfully");

        // Tamper with the message
        message.Type = P2PMessageType.Kick;
        var tamperedResult = await P2PProtocol.VerifyMessageAsync(message, _crypto);
        Assert.That(tamperedResult, Is.False,
            "Tampered message should fail verification");

        await identity.DisposeAsync();
    }

    /// <summary>
    /// Kernel allowlist: unregistered types cannot be resolved.
    /// </summary>
    [Test]
    public void Security_KernelAllowlist_EnforcedOnResolve()
    {
        // Clear and register only P2PTestKernels
        P2PKernelSerializer.ClearAllowlist();
        P2PKernelSerializer.RegisterKernelType(typeof(P2PTestKernels));

        // Test kernels should resolve
        var method = typeof(P2PTestKernels).GetMethod(nameof(P2PTestKernels.VectorAdd))!;
        var request = P2PKernelSerializer.CreateDispatch(method, gridDimX: 256);
        var resolved = P2PKernelSerializer.ResolveKernel(request);
        Assert.That(resolved, Is.Not.Null, "Registered kernel type should resolve");

        // Unregistered type should NOT resolve
        var evilRequest = new KernelDispatchRequest
        {
            KernelType = typeof(Environment).AssemblyQualifiedName!,
            KernelMethod = "Exit",
        };
        var evilResolved = P2PKernelSerializer.ResolveKernel(evilRequest);
        Assert.That(evilResolved, Is.Null,
            "Unregistered kernel type should NOT resolve - security violation");

        // Re-register for other tests
        P2PKernelSerializer.RegisterKernelType(typeof(P2PTestKernels));
    }
}
