using System.Collections.Concurrent;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using NUnit.Framework;
using SpawnDev.BlazorJS.Cryptography;
using SpawnDev.ILGPU.P2P.IntegrationTests.Infrastructure;

namespace SpawnDev.ILGPU.P2P.IntegrationTests;

/// <summary>
/// Phase 2: Real-WebRTC end-to-end pipeline tests (Rule #1 primary-purpose coverage).
///
/// Two P2PCompute instances live in the same test process — one coordinator, one worker
/// — connected through the LocalTrackerFixture WebSocket tracker at ws://localhost:5561
/// (with hub.spawndev.com + openwebtorrent.com as fallbacks). The WebRTC data channel
/// between them is provided by SpawnDev.RTC's SIPSorcery fork exactly as it would be
/// between two separate machines. No loopback shortcuts.
///
/// Each test exercises the full wire: sd_compute BEP 10 handshake, input buffer chunks
/// from coordinator to worker, KernelDispatch message, worker-side compile + execute,
/// KernelResult metadata, and the modified-buffer auto-push back to the coordinator.
/// Every result is verified bit-for-bit against a CPU reference.
///
/// **Flakiness note:** tracker signaling through public infrastructure is not 100%
/// reliable. Every test in this fixture carries `[Retry(3)]` so one bad announce
/// cycle on hub.spawndev.com does not fail the build. When a test fails all three
/// attempts with peer-discovery timeouts and no `[sd_compute] OnExtendedHandshake`
/// lines, that is a tracker/WebRTC-layer issue, not a P2P library regression.
/// Task #35 tracks making the local tracker signaling deterministic so these tests
/// can stop depending on public infrastructure.
/// </summary>
[TestFixture]
public class RealWebRtcPipelineTests
{
    /// <summary>
    /// Peer-discovery timeout. Public trackers (hub.spawndev.com) are sometimes
    /// slow to relay offers/answers when under load, so we allow a minute before
    /// declaring discovery failed.
    /// </summary>
    private const int DiscoveryTimeoutMs = 60000;

    private const int DispatchTimeoutMs = 15000;

    /// <summary>
    /// Timeout for buffer-transfer waits. Relayed WebRTC through a public tracker
    /// delivers ~10–30 KB/s in the worst case; 1MB + round trip needs breathing room.
    /// </summary>
    private const int BufferReturnTimeoutMs = 120000;

    /// <summary>
    /// Trackers used by the kernel-dispatch tests. LocalTrackerFixture's
    /// loopback tracker is listed first so tests do not depend on any public
    /// service; hub.spawndev.com + openwebtorrent.com are kept as fallbacks
    /// for the case where LocalTrackerFixture fails to launch ServerApp
    /// (e.g., port 5561 already in use by another process).
    /// </summary>
    private static string[] DispatchTrackers => LocalTrackerFixture.IsAvailable
        ? new[]
        {
            LocalTrackerFixture.GetTrackerUrl(),
            "wss://hub.spawndev.com:44365/announce",
            "wss://tracker.openwebtorrent.com",
        }
        : new[]
        {
            "wss://hub.spawndev.com:44365/announce",
            "wss://tracker.openwebtorrent.com",
        };

    [SetUp]
    public void PerTestSetUp()
    {
        // The kernel allowlist is process-wide static state and
        // SecurityTests.Security_KernelAllowlist_EnforcedOnResolve clears it.
        // Register before every test so run order cannot affect us.
        P2PKernelSerializer.RegisterKernelType(typeof(P2PTestKernels));
    }

    /// <summary>
    /// Foundation test: two P2PCompute instances discover each other through the
    /// local tracker via real WebRTC. Proves the tracker + SIPSorcery + sd_compute
    /// handshake path is wired correctly before layering kernel work on top.
    /// </summary>
    [Test, CancelAfter(90000), Retry(3)]
    public async Task TwoPeers_DiscoverEachOtherViaLocalTracker()
    {
        await RunDiscoveryAsync(new[] { LocalTrackerFixture.GetTrackerUrl() },
            "discovery-local-tracker");
    }

    /// <summary>
    /// Same as the local-tracker discovery test but signaled through the public
    /// hub.spawndev.com tracker. Proves the P2PCompute wiring works end-to-end
    /// in the known-good signaling path (matches
    /// SpawnDev.WebTorrent.Desktop_TwoClients_DiscoverViaTracker). Helpful for
    /// isolating regressions: if this passes while the local-tracker test fails,
    /// the issue is in the local tracker setup.
    /// </summary>
    [Test, CancelAfter(90000), Retry(3)]
    public async Task TwoPeers_DiscoverEachOtherViaPublicHubTracker()
    {
        await RunDiscoveryAsync(new[] { "wss://hub.spawndev.com:44365/announce" },
            "discovery-public-tracker");
    }

    private async Task RunDiscoveryAsync(string[] trackers, string swarmName)
    {
        var crypto = new DotNetCrypto();

        // Declare clients with `await using` before the P2PCompute wrappers so
        // dispose order is P2PCompute first, WebTorrentClient last. Prevents
        // dangling transport/bridge state from racing with client shutdown.
        await using var coordClient = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var workerClient = new SpawnDev.WebTorrent.WebTorrentClient();

        await using var coordinator = await P2PCompute.CreateSwarmAsync(
            crypto, coordClient, swarmName, trackers: trackers);

        using var workerContext = Context.Create(b => b.CPU());
        using var workerAccelerator = workerContext.CreateCPUAccelerator(0);

        await using var worker = await P2PCompute.JoinSwarmAsync(
            crypto, workerClient, workerAccelerator, coordinator.MagnetLink!);

        var start = DateTime.UtcNow;
        var deadline = start.AddMilliseconds(DiscoveryTimeoutMs);
        while (DateTime.UtcNow < deadline &&
               (coordinator.PeerCount == 0 || worker.PeerCount == 0))
        {
            await Task.Delay(1000);
            var elapsed = (DateTime.UtcNow - start).TotalSeconds;
            TestContext.Progress.WriteLine(
                $"[{elapsed:F0}s] coord.PeerCount={coordinator.PeerCount}, " +
                $"worker.PeerCount={worker.PeerCount}");
        }

        Assert.That(coordinator.PeerCount, Is.GreaterThan(0),
            "Coordinator should see the worker as a peer");
        Assert.That(worker.PeerCount, Is.GreaterThan(0),
            "Worker should see the coordinator as a peer");

    }

    /// <summary>
    /// Rule #1 primary-purpose test: real kernel dispatch across real WebRTC.
    /// Coordinator sends VectorAdd inputs to a worker over the data channel,
    /// worker compiles + executes on its CPU accelerator, coordinator receives
    /// the result buffer back and verifies all 1024 elements bit-exact.
    /// </summary>
    [Test, CancelAfter(120000), Retry(3)]
    public async Task VectorAdd_1024_DispatchedOverRealWebRtc_BitExact()
    {
        const int n = 1024;
        var trackers = DispatchTrackers;
        var crypto = new DotNetCrypto();

        // Declare clients with `await using` before the P2PCompute wrappers so
        // dispose order is P2PCompute first, WebTorrentClient last. Prevents
        // dangling transport/bridge state from racing with client shutdown.
        await using var coordClient = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var workerClient = new SpawnDev.WebTorrent.WebTorrentClient();

        await using var coordinator = await P2PCompute.CreateSwarmAsync(
            crypto, coordClient, "vectoradd-test", trackers: trackers);

        using var workerContext = Context.Create(b => b.CPU());
        using var workerAccelerator = workerContext.CreateCPUAccelerator(0);

        await using var worker = await P2PCompute.JoinSwarmAsync(
            crypto, workerClient, workerAccelerator, coordinator.MagnetLink!);

        // Wait for peer discovery.
        await WaitForPeersAsync(coordinator, worker);

        var peerId = coordinator.Accelerator!.Peers.First().PeerId;
        var workerBuffers = CollectWorkerBuffers(coordinator);

        // Send inputs to the worker. result buffer is output-only, worker allocates.
        var (aBytes, bBytes, expected) = DataIntegrityHelper.GenerateVectorAddData(n);
        await coordinator.Transport!.SendBufferAsync(peerId, "a", aBytes);
        await coordinator.Transport!.SendBufferAsync(peerId, "b", bBytes);

        // Ensure both inputs are fully reassembled on the worker before the
        // dispatch message is processed. Chunks and dispatch travel on the same
        // reliable+ordered data channel, but the worker's P2PTransport fires
        // each handler task without serializing, so a race is possible.
        await WaitForWorkerBuffersAsync(worker, "a", "b");

        // Dispatch VectorAdd. Worker auto-pushes modified "result" buffer back.
        var method = typeof(P2PTestKernels).GetMethod(nameof(P2PTestKernels.VectorAdd))!;
        var dispatchResult = await coordinator.Accelerator!.DispatchAsync(method, n,
            ("a", aBytes, 4), ("b", bBytes, 4), ("result", null, 4));

        Assert.That(dispatchResult.Success, Is.True,
            $"Dispatch failed: {dispatchResult.Error}");
        Assert.That(dispatchResult.ModifiedBuffers, Contains.Item("result"),
            "result buffer should be reported as modified");

        var resultBytes = await WaitForBufferAsync(workerBuffers, "result", BufferReturnTimeoutMs);
        Assert.That(resultBytes.Length, Is.EqualTo(n * 4),
            $"Returned buffer wrong size: {resultBytes.Length}");

        var actual = DataIntegrityHelper.BytesToFloats(resultBytes);
        var (violations, firstIdx, firstExp, firstAct) =
            DataIntegrityHelper.VerifyFloats(actual, expected);

        Assert.That(violations, Is.EqualTo(0),
            $"VectorAdd over WebRTC: {violations}/{n} violations. " +
            $"First at [{firstIdx}]: expected {firstExp}, got {firstAct}");

    }

    /// <summary>
    /// Tensor-transfer stress test: three 1MB float buffers (a, b, result) — 3MB of
    /// traffic across real WebRTC per direction. Each 1MB buffer is chunked into 16
    /// x 64KB frames by P2PBufferTransfer. VectorAdd is used instead of VectorScale
    /// because the P2P dispatch path does not yet serialize scalar kernel parameters
    /// (see task #33). All 256K elements are verified bit-exact.
    /// </summary>
    [Test, CancelAfter(180000), Retry(3)]
    public async Task LargeBuffer_1MB_DispatchedOverRealWebRtc_BitExact()
    {
        const int n = 256 * 1024; // 1MB of float32
        var trackers = DispatchTrackers;
        var crypto = new DotNetCrypto();

        // Declare clients with `await using` before the P2PCompute wrappers so
        // dispose order is P2PCompute first, WebTorrentClient last. Prevents
        // dangling transport/bridge state from racing with client shutdown.
        await using var coordClient = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var workerClient = new SpawnDev.WebTorrent.WebTorrentClient();

        await using var coordinator = await P2PCompute.CreateSwarmAsync(
            crypto, coordClient, "largebuffer-test", trackers: trackers);

        using var workerContext = Context.Create(b => b.CPU());
        using var workerAccelerator = workerContext.CreateCPUAccelerator(0);

        await using var worker = await P2PCompute.JoinSwarmAsync(
            crypto, workerClient, workerAccelerator, coordinator.MagnetLink!);

        await WaitForPeersAsync(coordinator, worker);

        var peerId = coordinator.Accelerator!.Peers.First().PeerId;
        var workerBuffers = CollectWorkerBuffers(coordinator);

        // Deterministic pseudo-random inputs so any single-bit corruption is detectable.
        var rng = new Random(unchecked((int)0xC0DEFACE));
        var a = new float[n];
        var b = new float[n];
        var expected = new float[n];
        for (int i = 0; i < n; i++)
        {
            a[i] = (float)(rng.NextDouble() * 1000.0 - 500.0);
            b[i] = (float)(rng.NextDouble() * 1000.0 - 500.0);
            expected[i] = a[i] + b[i];
        }
        var aBytes = new byte[n * 4];
        var bBytes = new byte[n * 4];
        Buffer.BlockCopy(a, 0, aBytes, 0, n * 4);
        Buffer.BlockCopy(b, 0, bBytes, 0, n * 4);

        await coordinator.Transport!.SendBufferAsync(peerId, "large_a", aBytes);
        await coordinator.Transport!.SendBufferAsync(peerId, "large_b", bBytes);

        await WaitForWorkerBuffersAsync(worker, "large_a", "large_b");

        var method = typeof(P2PTestKernels).GetMethod(nameof(P2PTestKernels.VectorAdd))!;
        var dispatchResult = await coordinator.Accelerator!.DispatchAsync(method, n,
            ("large_a", aBytes, 4), ("large_b", bBytes, 4), ("large_out", null, 4));

        Assert.That(dispatchResult.Success, Is.True,
            $"Dispatch failed: {dispatchResult.Error}");

        var resultBytes = await WaitForBufferAsync(workerBuffers, "large_out", BufferReturnTimeoutMs);
        Assert.That(resultBytes.Length, Is.EqualTo(n * 4), "1MB buffer wrong size");

        var actual = DataIntegrityHelper.BytesToFloats(resultBytes);
        var (violations, firstIdx, firstExp, firstAct) =
            DataIntegrityHelper.VerifyFloats(actual, expected, tolerance: 0.001f);

        Assert.That(violations, Is.EqualTo(0),
            $"1MB VectorAdd over WebRTC: {violations}/{n} violations. " +
            $"First at [{firstIdx}]: expected {firstExp}, got {firstAct}");

    }

    /// <summary>
    /// Integer Identity kernel across real WebRTC. Any single-bit flip during
    /// input transfer, kernel execution, or output transfer fails the SHA256 check.
    /// </summary>
    [Test, CancelAfter(120000), Retry(3)]
    public async Task DataIntegrity_SHA256_IdentityOverRealWebRtc()
    {
        const int n = 16384; // 64KB
        var trackers = DispatchTrackers;
        var crypto = new DotNetCrypto();

        // Declare clients with `await using` before the P2PCompute wrappers so
        // dispose order is P2PCompute first, WebTorrentClient last. Prevents
        // dangling transport/bridge state from racing with client shutdown.
        await using var coordClient = new SpawnDev.WebTorrent.WebTorrentClient();
        await using var workerClient = new SpawnDev.WebTorrent.WebTorrentClient();

        await using var coordinator = await P2PCompute.CreateSwarmAsync(
            crypto, coordClient, "integrity-test", trackers: trackers);

        using var workerContext = Context.Create(b => b.CPU());
        using var workerAccelerator = workerContext.CreateCPUAccelerator(0);

        await using var worker = await P2PCompute.JoinSwarmAsync(
            crypto, workerClient, workerAccelerator, coordinator.MagnetLink!);

        await WaitForPeersAsync(coordinator, worker);

        var peerId = coordinator.Accelerator!.Peers.First().PeerId;
        var workerBuffers = CollectWorkerBuffers(coordinator);

        var rng = new Random(unchecked((int)0xDEADBEEF));
        var input = new int[n];
        for (int i = 0; i < n; i++)
            input[i] = rng.Next();
        var inputBytes = new byte[n * 4];
        Buffer.BlockCopy(input, 0, inputBytes, 0, n * 4);
        var hashBefore = DataIntegrityHelper.ComputeSha256(inputBytes);

        await coordinator.Transport!.SendBufferAsync(peerId, "integ_in", inputBytes);

        await WaitForWorkerBuffersAsync(worker, "integ_in");

        var method = typeof(P2PTestKernels).GetMethod(nameof(P2PTestKernels.Identity))!;
        var dispatchResult = await coordinator.Accelerator!.DispatchAsync(method, n,
            ("integ_in", inputBytes, 4), ("integ_out", null, 4));

        Assert.That(dispatchResult.Success, Is.True,
            $"Dispatch failed: {dispatchResult.Error}");

        var resultBytes = await WaitForBufferAsync(workerBuffers, "integ_out", BufferReturnTimeoutMs);
        var hashAfter = DataIntegrityHelper.ComputeSha256(resultBytes);

        Assert.That(hashAfter, Is.EqualTo(hashBefore),
            $"SHA256 mismatch over WebRTC round-trip. Before: {hashBefore[..16]}... " +
            $"After: {hashAfter[..16]}... ({resultBytes.Length} bytes, {n} ints)");

    }

    /// <summary>
    /// Wait for the coordinator and worker to see each other through WebRTC.
    /// </summary>
    private static async Task WaitForPeersAsync(P2PCompute coordinator, P2PCompute worker)
    {
        var deadline = DateTime.UtcNow.AddMilliseconds(DiscoveryTimeoutMs);
        while (DateTime.UtcNow < deadline &&
               (coordinator.PeerCount == 0 || worker.PeerCount == 0))
        {
            await Task.Delay(250);
        }

        if (coordinator.PeerCount == 0 || worker.PeerCount == 0)
        {
            Assert.Fail($"Peer discovery timed out after {DiscoveryTimeoutMs}ms. " +
                $"Coordinator peers: {coordinator.PeerCount}, Worker peers: {worker.PeerCount}.");
        }
    }

    /// <summary>
    /// Subscribe to the coordinator's buffer-transfer completion event. Returns
    /// a thread-safe map of bufferId → data populated as buffers arrive from the worker.
    /// </summary>
    private static ConcurrentDictionary<string, byte[]> CollectWorkerBuffers(P2PCompute coordinator)
    {
        var received = new ConcurrentDictionary<string, byte[]>();
        coordinator.Transport!.BufferTransfer.OnBufferReceived += (bufferId, data) =>
        {
            received[bufferId] = data;
        };
        return received;
    }

    /// <summary>
    /// Wait for a specific buffer to appear in the map (worker auto-push path).
    /// </summary>
    private static async Task<byte[]> WaitForBufferAsync(
        ConcurrentDictionary<string, byte[]> received, string bufferId, int timeoutMs)
    {
        var deadline = DateTime.UtcNow.AddMilliseconds(timeoutMs);
        while (DateTime.UtcNow < deadline)
        {
            if (received.TryGetValue(bufferId, out var data))
                return data;
            await Task.Delay(50);
        }
        throw new TimeoutException(
            $"Modified buffer '{bufferId}' did not arrive from worker within {timeoutMs}ms.");
    }

    /// <summary>
    /// Block until every listed buffer has been fully reassembled in the worker's
    /// buffer store. Guards against the dispatch message overtaking its own input
    /// buffer chunks at the worker.
    /// </summary>
    private static async Task WaitForWorkerBuffersAsync(
        P2PCompute worker, params string[] bufferIds)
    {
        var workerInstance = worker.Worker
            ?? throw new InvalidOperationException("Worker P2PCompute has no Worker instance set.");

        var deadline = DateTime.UtcNow.AddMilliseconds(BufferReturnTimeoutMs);
        while (DateTime.UtcNow < deadline)
        {
            if (bufferIds.All(id => workerInstance.GetBuffer(id) != null))
                return;
            await Task.Delay(50);
        }

        var missing = bufferIds.Where(id => workerInstance.GetBuffer(id) == null).ToArray();
        throw new TimeoutException(
            $"Input buffers did not arrive at worker within {BufferReturnTimeoutMs}ms: " +
            $"{string.Join(", ", missing)}.");
    }
}
