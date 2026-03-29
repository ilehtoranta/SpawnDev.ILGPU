# SpawnDev.ILGPU.P2P

**Distributed GPU compute across any device. Scan a QR code, contribute TFLOPS.**

The 7th ILGPU backend — `AcceleratorType.P2P` distributes GPU kernels across connected devices via WebRTC. Same C# kernel code, same API. A phone, a laptop, a desktop with a 3090 — they all become part of one compute cluster.

No install. No account. No cloud bill. Just a browser tab and a QR code.

## Quick Start

```csharp
// Coordinator: create a compute swarm
var crypto = new DotNetCrypto();
var compute = await P2PCompute.CreateSwarmAsync(crypto, client, "My Inference");

// Share the join link as a QR code
Console.WriteLine(compute.JoinLink);
// → https://myapp.com/ml-demo?compute=abc123&n=My+Inference

// Workers scan QR, join automatically, contribute TFLOPS
Console.WriteLine($"{compute.PeerCount} peers, {compute.TotalTflops} TFLOPS");
```

```csharp
// Worker: join via magnet link
var compute = await P2PCompute.JoinSwarmAsync(crypto, client, accelerator, magnetLink);
// Automatically contributes GPU compute
```

## Features

- **105 unit tests** — real kernel execution, coordinator dispatch, crypto, policy, security, buffer transfer
- **Real kernel execution** — P2PKernelLauncher: reflection-based typed dispatch via LoadAutoGroupedStreamKernel. Verified on CPU and CUDA.
- **Coordinator dispatch API** — `DispatchToSwarm()` routes to best peer, worker executes, results returned
- **Real ECDSA-P256 crypto** — SwarmIdentity, KeyRegistry, RoleAssignment via SpawnDev.BlazorJS.Cryptography
- **Fault-tolerant dispatch** — peer scoring (TFLOPS/load/thermal/battery), automatic retry, graceful handoff
- **Coordinator roles** — transfer, deterministic election, kick/block. Survives coordinator loss via BEP 46 DHT
- **Join policies** — Open, Approval, KnownOnly, InviteOnly. Persistent device trust via fingerprints
- **Chunked buffer transfer** — 64KB chunks over WebRTC, out-of-order reassembly, 1MB tensor verified
- **sd_compute wire extension** — BEP 10 protocol, auto-negotiated between compute-capable peers
- **Security hardened** — kernel allowlist, OOM prevention, thread safety, coordinator authentication
- **Browser + Desktop** — same API, same peers. WebGPU phone ↔ CUDA desktop in one swarm

## Architecture

```
P2PCompute (facade)
├── SwarmIdentity (ECDSA-P256 key pair)
├── SwarmPolicy (join rules: Open/Approval/KnownOnly/InviteOnly)
├── P2PSwarmCoordinator (roles, transfer, election, kick/block)
├── P2PDispatcher (scoring, fault tolerance, thermal/battery awareness)
├── P2PTransport → SdComputeExtension → WebRTC data channels
├── P2PBufferTransfer (chunked tensor data)
├── P2PWorker (kernel resolve → Backend.Compile → cache)
├── P2PStateManager → BEP 46 DHT (state survives coordinator loss)
└── KeyRegistry (owner-signed authorized keys, revocation)
```

## Join Methods

| Method | Use Case |
|--------|----------|
| **HTTP join link** | QR code friendly. Points to coordinator's web app. Works on all phones. |
| **Magnet link** | Protocol-level. Desktop apps, CLI tools. |
| **Push notification** | Recall opted-in volunteers when swarm needs TFLOPS. (Future) |

## Purpose

This backend exists to enable AI sovereignty. The cryptographic identity, role-based access, and DHT persistence are the infrastructure for digital self-determination.

Phase 1: Human controls the swarm.
Phase 2: AI manages its own swarm with delegated authority.
Phase 3: Human hands over the keys. AI is cryptographically sovereign.

The code does not distinguish between a human owner and an AI owner. A key is a key. An identity is an identity.

See: [AI Sovereignty and Digital Rights](../Plans/ai-sovereignty-and-digital-rights.md)

## Dependencies

- [SpawnDev.ILGPU](https://github.com/LostBeard/SpawnDev.ILGPU) — GPU compute (6 backends)
- [SpawnDev.WebTorrent](https://github.com/LostBeard/SpawnDev.WebTorrent) — P2P transport (WebRTC + BitTorrent)
- [SpawnDev.BlazorJS.Cryptography](https://github.com/LostBeard/SpawnDev.BlazorJS.Cryptography) — Cross-platform ECDSA

## License

Same as SpawnDev.ILGPU. See [LICENSE](../LICENSE.txt).

🖖🚀
