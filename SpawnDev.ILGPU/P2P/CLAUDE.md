# P2P Backend — AcceleratorType.P2P

Distributes ILGPU kernels across connected devices via SpawnDev.WebTorrent WebRTC data channels. Same C# kernel code, same `LoadAutoGroupedStreamKernel` API. The developer writes one kernel, it runs across multiple GPUs on multiple devices transparently.

## Architecture

The P2P accelerator wraps one or more **remote accelerators** on peer devices. Each peer runs a real backend (WebGPU, CUDA, etc.) and the P2P accelerator coordinates kernel dispatch and data transfer across them.

```
P2PAccelerator (coordinator)
├── RemotePeer A → WebGPU accelerator (browser)
├── RemotePeer B → CUDA accelerator (desktop)
└── RemotePeer C → Wasm accelerator (browser)
```

## Key Components

- **P2PAccelerator** — Coordinator. Implements Accelerator interface. Delegates dispatch to remote peers.
- **P2PDevice** — Device descriptor. Reports aggregate capabilities of connected peers.
- **P2PMemoryBuffer** — Buffer that may live on a remote peer. Tracks data locality.
- **P2PStream** — Command queue. Batches operations and flushes to remote peers.
- **P2PCompiledKernel** — Holds serialized kernel IR for transmission to peers.
- **P2PKernel** — Wraps compiled kernel for dispatch.

## Transport

Uses SpawnDev.WebTorrent infrastructure:
- **WebRTC data channels** for tensor transfer (low latency, P2P)
- **AgentChannel** for coordination messages (BEP 46 DHT mutable items)
- **SwarmCompute** for task distribution (publish/join/submit pattern)

## Data Flow

1. Host allocates buffer → P2PAccelerator assigns to a peer (data locality aware)
2. Host copies data to buffer → serialized and sent to peer via WebRTC
3. Host dispatches kernel → serialized kernel + buffer bindings sent to peer
4. Peer executes kernel on its local accelerator
5. Host reads buffer back → peer sends data back via WebRTC

## Joining a Swarm

Three ways to join:
1. **Magnet link** — `magnet:?xt=urn:btih:...` (protocol-level, works if handler registered)
2. **HTTP join link** — `https://myapp.com/ml-demo?compute=HASH&n=SwarmName` (QR-friendly, works on all phones)
3. **Push notification** — Web Push API recalls opted-in users when swarm needs TFLOPS

The join link points to the coordinator's own web app — the same code that runs the coordinator also runs workers. No separate server needed. The app detects `?compute=` in the URL and auto-joins as a worker (with user consent).

## Consent Flow (UI-level, not in library)

When a user follows a join link:
- **Always join** — saved in localStorage per origin, auto-joins next time
- **Join this time** — one-shot
- **Not now** — decline

## Coordinator Role

The coordinator is NOT permanent. It can be:
- **Transferred** — graceful handoff to healthiest peer (battery dying, tab closing)
- **Elected** — deterministic election if coordinator drops (highest TFLOPS wins, PeerId tiebreaker)
- State survives coordinator loss via BEP 46 DHT mutable items

Coordinators can **kick** and **block** peers.

## Constraints

- Kernel compilation happens locally — IR is serialized and sent to peers
- Each peer must have a compatible ILGPU backend installed
- Tensor transfer is the bottleneck — minimize cross-peer data movement
- Buffer locality tracking is critical — keep data where it's used
- Thermal/battery scoring prevents overloading mobile devices
- Push notifications for "help wanted" when swarm capacity is low (future)
