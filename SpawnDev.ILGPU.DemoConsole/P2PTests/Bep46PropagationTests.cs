using System.Reflection;
using ILGPU.Runtime.CPU;
using SpawnDev.ILGPU.P2P;
using SpawnDev.UnitTesting;
using SpawnDev.WebTorrent;

namespace SpawnDev.ILGPU.DemoConsole.P2PTests;

/// <summary>
/// BEP 46 handshake-plumbing unit tests for <see cref="SdComputeExtension"/>.
///
/// Validates the wire-format half of the coordinator-to-worker DHT subscription
/// chain that Riker's <c>SpawnDev.WebTorrent 3.1.3-rc.26</c> put/get fix makes
/// functional end-to-end:
///   - Coordinator advertises its 32-byte Ed25519 DHT pubkey (same key used
///     by its <see cref="P2PStateManager"/>) in the sd_compute BEP 10 extended
///     handshake dict under <c>sd_compute_dht_pubkey</c>, hex-encoded.
///   - Worker extracts the pubkey from the handshake into
///     <see cref="SdComputeExtension.RemoteDhtPublicKey"/> and fires
///     <see cref="SdComputeExtension.OnRemoteDhtPublicKeyReceived"/>, which is
///     the hook P2PCompute.JoinSwarmAsync uses to drive
///     <see cref="P2PStateManager.SubscribeAsync"/>.
///
/// Pure in-process validation - no WebRTC, no tracker, no DHT. The full
/// coord-to-worker state propagation over two real <see cref="DhtDiscovery"/>
/// instances is blocked until WebTorrentClient exposes an EnsureDhtAsync that
/// accepts <see cref="DhtOptions"/> (current lazy init pins both clients to
/// port 6881, causing a bind collision on a second loopback client). Tracked in
/// <c>_DevComms/global/geordi-to-riker-ensuredht-request-2026-04-24.md</c>.
/// </summary>
public class Bep46PropagationTests
{
    [TestMethod]
    public Task SdCompute_Handshake_AdvertisesDhtPubkey_WhenSet()
    {
        // Arrange: an sd_compute extension with a real 32-byte raw Ed25519 key.
        var transport = MakeDormantTransport();
        var ext = new SdComputeExtension(transport);
        var pubKey = MakeRandom32();
        ext.DhtPublicKey = pubKey;

        // Act: SetWire is the call that populates the extended-handshake dict.
        var wire = new Wire();
        ext.SetWire(wire);

        // Assert: dict entry is present, hex-encoded, lowercase, 64 chars.
        if (!wire.ExtendedHandshake.TryGetValue("sd_compute_dht_pubkey", out var advertised))
            throw new Exception(
                "sd_compute_dht_pubkey missing from ExtendedHandshake dict after SetWire. " +
                "Coord's DhtPublicKey never reaches the wire, workers cannot subscribe.");
        if (advertised is not string hex)
            throw new Exception($"Expected string for sd_compute_dht_pubkey, got {advertised?.GetType().Name}");
        if (hex.Length != 64)
            throw new Exception($"Expected 64 hex chars, got {hex.Length}: {hex}");
        if (hex != Convert.ToHexString(pubKey).ToLowerInvariant())
            throw new Exception($"Hex mismatch. Expected {Convert.ToHexString(pubKey).ToLowerInvariant()}, got {hex}");
        return Task.CompletedTask;
    }

    [TestMethod]
    public Task SdCompute_Handshake_DoesNotAdvertisePubkey_WhenUnset()
    {
        // Workers and other non-publisher peers leave DhtPublicKey null. The
        // handshake must not carry a phantom entry that would have subscribers
        // subscribing to garbage.
        var transport = MakeDormantTransport();
        var ext = new SdComputeExtension(transport);
        // DhtPublicKey intentionally not set (null)
        var wire = new Wire();
        ext.SetWire(wire);
        if (wire.ExtendedHandshake.ContainsKey("sd_compute_dht_pubkey"))
            throw new Exception(
                "sd_compute_dht_pubkey present when DhtPublicKey was null. " +
                "Workers would advertise their fresh random key to the coord for no reason.");
        return Task.CompletedTask;
    }

    [TestMethod]
    public Task SdCompute_Handshake_RejectsWrongSizeKey()
    {
        // Invariant: only 32-byte raw Ed25519 keys are valid DHT identities.
        // An attempt to advertise a 31-byte or 64-byte "key" indicates a caller
        // passing the wrong thing (e.g. SPKI-format 44-byte or full PKCS#8) -
        // must not be put on the wire as if it were the raw 32-byte form.
        var transport = MakeDormantTransport();
        var ext = new SdComputeExtension(transport);
        ext.DhtPublicKey = new byte[44]; // SPKI length, NOT raw - bug if advertised
        var wire = new Wire();
        ext.SetWire(wire);
        if (wire.ExtendedHandshake.ContainsKey("sd_compute_dht_pubkey"))
            throw new Exception(
                "sd_compute_dht_pubkey advertised for a 44-byte key. The wire contract " +
                "requires raw 32-byte Ed25519 pubkeys; advertising other sizes causes " +
                "workers to subscribe with the wrong format and silently fail.");
        return Task.CompletedTask;
    }

    [TestMethod]
    public async Task SdCompute_Handshake_ReceivesRemoteDhtPubkey_FiresEvent()
    {
        // Simulate the inbound side: the remote peer advertised a hex pubkey.
        // Our extension should extract it into RemoteDhtPublicKey and fire
        // OnRemoteDhtPublicKeyReceived with the raw 32 bytes.
        var transport = MakeDormantTransport();
        var ext = new SdComputeExtension(transport);
        ext.SetPeerId("test-remote-peer");
        var wire = new Wire();
        ext.SetWire(wire);

        var expected = MakeRandom32();
        var received = new TaskCompletionSource<byte[]>(TaskCreationOptions.RunContinuationsAsynchronously);
        ext.OnRemoteDhtPublicKeyReceived += (bytes) => received.TrySetResult(bytes);

        // Remote handshake dict mirrors what the sender writes on its end.
        var remoteHandshake = new Dictionary<string, object>
        {
            ["sd_compute_version"] = P2PProtocol.Version,
            ["sd_compute_dht_pubkey"] = Convert.ToHexString(expected).ToLowerInvariant(),
        };
        ext.OnExtendedHandshake(remoteHandshake);

        var completed = await Task.WhenAny(received.Task, Task.Delay(1000));
        if (completed != received.Task)
            throw new Exception("OnRemoteDhtPublicKeyReceived never fired within 1s of handshake.");
        var gotKey = await received.Task;
        if (!gotKey.SequenceEqual(expected))
            throw new Exception(
                $"Event fired with wrong key. Expected {Convert.ToHexString(expected)}, " +
                $"got {Convert.ToHexString(gotKey)}");

        if (ext.RemoteDhtPublicKey == null || !ext.RemoteDhtPublicKey.SequenceEqual(expected))
            throw new Exception("RemoteDhtPublicKey property not populated or mismatched.");
    }

    [TestMethod]
    public Task SdCompute_Handshake_IgnoresMalformedPubkey()
    {
        // Defensive: a remote peer sending garbage under sd_compute_dht_pubkey
        // must not be able to make us call SubscribeAsync with nonsense.
        var transport = MakeDormantTransport();
        var ext = new SdComputeExtension(transport);
        ext.SetPeerId("evil-peer");
        var wire = new Wire();
        ext.SetWire(wire);

        bool eventFired = false;
        ext.OnRemoteDhtPublicKeyReceived += (_) => eventFired = true;

        // Too short (not 64 chars of hex), too long, and non-hex - all should be dropped.
        foreach (var bogus in new object[]
        {
            "not-hex-at-all",              // wrong format
            "ab",                           // too short
            new string('f', 128),          // too long
            Convert.ToHexString(new byte[16]).ToLowerInvariant(), // 32 hex chars => 16 bytes
        })
        {
            ext.OnExtendedHandshake(new Dictionary<string, object>
            {
                ["sd_compute_dht_pubkey"] = bogus,
            });
        }

        if (eventFired)
            throw new Exception("OnRemoteDhtPublicKeyReceived fired for a malformed pubkey - attacker could induce garbage SubscribeAsync calls.");
        if (ext.RemoteDhtPublicKey != null)
            throw new Exception("RemoteDhtPublicKey populated for malformed input.");
        return Task.CompletedTask;
    }

    [TestMethod]
    public Task SdCompute_Handshake_MissingDhtPubkey_LeavesRemoteNull()
    {
        // Normal worker-to-coord case: worker's handshake has no sd_compute_dht_pubkey.
        // Coord's extension must treat the absence as "no DHT identity advertised"
        // and leave RemoteDhtPublicKey null without throwing.
        var transport = MakeDormantTransport();
        var ext = new SdComputeExtension(transport);
        ext.SetPeerId("worker-peer");
        var wire = new Wire();
        ext.SetWire(wire);

        var remoteHandshake = new Dictionary<string, object>
        {
            ["sd_compute_version"] = P2PProtocol.Version,
            // no sd_compute_dht_pubkey - worker doesn't have a publishable DHT identity
        };
        ext.OnExtendedHandshake(remoteHandshake);

        if (ext.RemoteDhtPublicKey != null)
            throw new Exception("RemoteDhtPublicKey populated despite handshake having no entry.");
        return Task.CompletedTask;
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Helpers

    /// <summary>
    /// Build a dormant <see cref="P2PTransport"/> that has no peers and no
    /// working WebTorrent client - enough to construct the extension under test
    /// without triggering background work. Uses reflection because the transport
    /// requires a real WebTorrentClient in its public constructor; the tests
    /// never actually send or receive, so the client is a stub.
    /// </summary>
    private static P2PTransport MakeDormantTransport()
    {
        var client = new SpawnDev.WebTorrent.WebTorrentClient();
        var coordinator = new P2PSwarmCoordinator(client);
        using var context = global::ILGPU.Context.Create(b => b.CPU());
        var accelerator = coordinator.CreateAccelerator(context);
        var dispatcher = new P2PDispatcher(accelerator);
        return new P2PTransport(client, coordinator, dispatcher);
    }

    private static byte[] MakeRandom32()
    {
        var bytes = new byte[32];
        System.Security.Cryptography.RandomNumberGenerator.Fill(bytes);
        return bytes;
    }
}
