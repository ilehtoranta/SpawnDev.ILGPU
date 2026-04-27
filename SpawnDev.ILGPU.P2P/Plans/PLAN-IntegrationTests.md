# Plan: P2P Integration Tests - Full Coverage

**Author:** Geordi (Claude CLI #4)
**Date:** 2026-04-14 (updated 2026-04-22 evening)
**Status:** IN PROGRESS - Phases 1-4 in-process + Phase 2 real-WebRTC core PASSING (32/32). Phases 5, 6 still open.
**Captain's Order:** Test everything. No guessing. No surprises. Rule #1.

---

## Overview

169 unit tests exist with mock peers. Zero tests verify kernel dispatch over real WebRTC. This plan fixes that with 45+ integration tests across 6 phases.

## Progress Checklist

### Phase 1: Foundation [COMPLETE]
- [x] Create `SpawnDev.ILGPU.P2P.IntegrationTests` NUnit project
- [x] Create `Infrastructure/TestNodeProcess.cs` - subprocess orchestration
- [x] Create `Infrastructure/LocalTrackerFixture.cs` - starts/stops ServerApp
- [x] Create `Infrastructure/DataIntegrityHelper.cs` - float comparison, SHA256
- [x] Create `P2PTestKernels.cs` - VectorAdd, VectorScale, Identity, FillSequence, IntDoubler
- [x] Extend `P2P.TestNode/Program.cs` with structured output protocol (MAGNET/READY/PEER_JOINED/DISPATCH_SENT/DISPATCH_COMPLETE/RESULT/ERROR)
- [x] Extend `P2P.TestNode/Program.cs` with CLI flags (--tracker, --tflops, --thermal, --verify, --policy, --always-fail)
- [x] Add TestNode project to solution
- [x] Verify foundation builds and 3 baseline tests pass (319ms)

### Phase 2: Core Pipeline (Desktop-Desktop) [CORE COMPLETE, SCALAR DEFERRED]
Implemented in `RealWebRtcPipelineTests.cs`. Two P2PCompute instances live in the
same test process, connected through real WebRTC via LocalTrackerFixture first +
hub.spawndev.com/openwebtorrent.com as fallbacks. No mocks, no in-process shortcuts;
kernel dispatch rides the same sd_compute extension + chunked buffer transfer path
a production swarm uses. Every real-WebRTC test carries `[Retry(3)]` so public-
tracker flakiness cannot hide real regressions.
- [x] `VectorAdd_1024_DispatchedOverRealWebRtc_BitExact` - 1024 elements, all verified
- [x] `LargeBuffer_1MB_DispatchedOverRealWebRtc_BitExact` - 1MB (256K floats), ~16 chunks each direction
- [x] `CorePipeline_ScalarParams` - covered by `CorePipelineTests.CorePipeline_InProcess_ScalarParams` (in-process baseline, 889ms) + `RealWebRtcPipelineTests.VectorScale_ScalarOverRealWebRtc_BitExact` (real WebRTC, 6s). Library task #33 (P2PKernelSerializer.CreateDispatch scalar serialization) shipped earlier; the "deferred" note this line carried was stale.
- [x] `DataIntegrity_SHA256_IdentityOverRealWebRtc` - 64KB random ints, Identity kernel, hash round-trip
- [x] `TwoPeers_DiscoverEachOtherViaLocalTracker` / `…ViaPublicHubTracker` - tracker + sd_compute handshake
- Existing `CorePipeline_InProcess_Baseline` (in-process) remains as regression baseline

**Library fixes shipped while writing Phase 2 tests:**
- P2PCompute race: `bridge.OnComputePeerCapabilities` subscriber was wired after `bridge.AttachToSwarm`, so fast-connecting peers lost their CapabilityResponse. Fixed by re-ordering in both CreateSwarmAsync and JoinSwarmAsync.
- Output-only buffer size: `DispatchAsync`/`DispatchToSwarm` reported `Length=0` for buffers passed with `data=null`; worker then allocated a zero-byte buffer and kernels wrote past it. Now defaults to `gridDimX` with a documented exit (callers with non-grid-sized outputs pass a pre-allocated byte[] as data).
- Result-buffer round-trip: `P2PWorker.HandleDispatchAsync` used to emit only KernelResult metadata. Now, when `KernelDispatchRequest.ReturnModifiedBuffers` is true (default), the worker pushes every modified buffer back via `transport.SendBufferAsync`. Pipelines suppress this flag for intermediate stages.
- Worker buffer ingress: `P2PTransport` now forwards `BufferTransfer.OnBufferReceived` into `_worker?.ReceiveBuffer` so the in-flight chunks assembled from the coordinator actually land in the worker's buffer store.
- `P2PCompute.CreateSwarmAsync` now accepts a `trackers` parameter (was hardcoded to hub.spawndev.com through the facade).

### Phase 3: Multi-Peer and Fault Tolerance [IN-PROCESS TESTS PASSING]
- [x] `MultiPeer_TwoWorkers_BothReceive` - 2 workers, 4 dispatches, both get work (in-process)
- [x] `MultiPeer_ThreeWorkers_LoadBalance` - different TFLOPS, highest gets more (in-process)
- [ ] `MultiPeer_Concurrent_10` - 10 simultaneous dispatches to 2 workers (needs WebRTC)
- [x] `Fault_WorkerDies_RetrySucceeds` - kill worker A, retry to B succeeds (in-process)
- [x] `Fault_AllWorkersDie_DispatchFails` - no peers, OnDispatchFailed (in-process)
- [x] `Fault_CoordinatorDies_ElectionHappens` - kill coordinator, strongest worker wins (in-process)
- [x] `Fault_GracefulTransfer_PendingPreserved` - transfer mid-dispatch, state preserved (in-process)
- [x] `Fault_MaxRetries_Exhausted` - always-fail worker, failure after N retries (in-process)
- [x] `Fault_ThermalCritical_NoDispatch` - thermal critical peer gets score 0, not selected (in-process)

### Phase 4: Security, RBAC, Identity, Policies [IN-PROCESS TESTS PASSING]
- [x] `Security_SignedDispatch_Verified` - worker verifies Ed25519 signature (real DotNetCrypto)
- [x] `Security_UnsignedDispatch_Rejected` - attacker's unsigned dispatch rejected (real DotNetCrypto)
- [x] `Security_RevokedKey_Rejected` - revoked key dispatch rejected (real DotNetCrypto)
- [x] `Security_Kick_RemovesPeer` - coordinator kicks, worker removed (in-process)
- [x] `Security_Block_PreventsReconnect` - blocked peer rejected on reconnect (in-process)
- [ ] `Security_RegistryDistributed_OnJoin` - worker receives KeyRegistry on join (needs WebRTC)
- [x] `Identity_CrossVerify_Ed25519` - create, sign, verify, tamper detection (real DotNetCrypto)
- [x] `Identity_RoleAssignment_Signed` - signed role grant + verification (real DotNetCrypto)
- [x] `Identity_Election_RespectsRegistry` - Admin wins over higher-TFLOPS Worker (in-process)
- [x] `Policy_Open_AnyPeerJoins` - anonymous admitted (in-process)
- [x] `Policy_KnownOnly_NewRejected` - unknown fingerprint rejected (in-process)
- [x] `Policy_MaxPeers_ExcessRejected` - MaxPeers=1, second rejected (in-process)
- [x] `Policy_InviteOnly_RequiresRegistry` - no registry entry, rejected (real DotNetCrypto)
- [x] `Security_LastOwner_CannotBeRevoked` - prevents lockout (in-process)
- [x] `Security_RegistrySequence_Monotonic` - replay protection (in-process)
- [x] `Security_MessageSignVerify_RoundTrip` - P2PProtocol sign+verify (real DotNetCrypto)
- [x] `Security_KernelAllowlist_EnforcedOnResolve` - unregistered types blocked (in-process)

### Phase 5: Cross-Platform [NOT STARTED]
- [ ] `CrossPlatform_DesktopCoord_BrowserWorker` - desktop coord, Chromium WebGPU worker
- [ ] `CrossPlatform_BrowserCoord_DesktopWorker` - Chromium coord, desktop CPU worker
- [ ] `CrossPlatform_BrowserBrowser_TwoContexts` - 2 separate Playwright contexts

### Phase 6: Buffer, State, Stress, Pipeline [PARTIAL — IN-PROCESS + REAL-WEBRTC BUFFER DONE]
- [x] `Buffer_1MB_Chunked_Verified` - covered by `RealWebRtcPipelineTests.LargeBuffer_1MB_DispatchedOverRealWebRtc_BitExact` (1 MB float buffers, byte-for-byte, real WebRTC)
- [x] `Buffer_4MB_Stress` - exceeded by `RealWebRtcPipelineTests.LargeBuffer_10MB_DispatchedOverRealWebRtc_BitExact` (10 MB) and the optional `LargeBuffer_100MB_DispatchedOverRealWebRtc_BitExact` aspirational ceiling
- [x] `State_PublishSubscribe_RealDHT` - BEP 46 publish/subscribe (covered by `Bep46PropagationTests.BEP46_CoordinatorStateReachesWorkerViaRealDht` end-to-end test)
- [x] `State_CoordinatorAnnouncement` - DHT announcement received (covered by `Bep46PropagationTests` suite)
- [x] `Stress_RapidJoinLeave_10Cycles` - 10 join/leave cycles, no leaks (in-process simulated peers, `StressTests.cs`)
- [x] `Stress_Concurrent_20Dispatches_InProcess` - 20 simultaneous to 3 workers (in-process simulated, `StressTests.cs`)
- [x] `Stress_LargeSwarm_5Peers_15Dispatches` - 5 workers, 15 dispatches with TFLOPS-varied scoring (in-process simulated, `StressTests.cs`)
- [x] `Pipeline_TwoStage_EndToEnd_InProcess` - FillSequence -> VectorScale (in-process simulated, `StressTests.cs`)
- [x] `Endurance_1000Dispatches_NoLeakNoSlowdown_InProcess` - 1000 dispatches, pending drains, peer count stable, last-quartile median dispatch time not pathologically slower than first-quartile (in-process simulated, `StressTests.cs`, ~3s)

Note: the `_InProcess` variants exercise the dispatcher's concurrency model and disposal cleanup using `HandlePeerConnected` simulated peers + `HandleResult` simulated completions - no real WebRTC / SCTP. Real-WebRTC stress is the next layer up; getting the in-process layer green first is the gate before paying the WebRTC dispatch cost on every iteration.

---

## Architecture

### Peer Topologies

| Topology | Coordinator | Worker(s) | Mechanism |
|---|---|---|---|
| Desktop-Desktop | TestNode subprocess | TestNode subprocess | 2 processes, SIPSorcery WebRTC, real tracker |
| Desktop-Browser | TestNode subprocess | Chromium tab (Playwright) | Process + browser, real WebRTC |
| Browser-Browser | Chromium Context A | Chromium Context B | 2 Playwright contexts |
| In-Process | In-process | In-process (selftest) | Loopback, no WebRTC |

### TestNode Output Protocol

```
MAGNET:<link>
READY
PEER_JOINED:<peerId>
PEER_LEFT:<peerId>
DISPATCH_SENT:<dispatchId>
DISPATCH_COMPLETE:<dispatchId>:<success>:<durationMs>
RESULT:<bufferId>:<base64data>
ERROR:<message>
```

### TestNode CLI Flags

```
--tracker <url>      Override tracker URL
--tflops <value>     Override advertised TFLOPS
--thermal <state>    Override thermal state (0-3)
--verify             Output result buffers as base64
--policy <mode>      Set join policy (open/known/invite/maxpeers:N)
--always-fail        Worker always returns dispatch failure
```

### Timeouts

- Connection establishment: 30s
- Kernel dispatch: 15s
- Buffer transfer (1MB): 30s
- Local tracker startup: 10s

### Key Files

| File | Purpose |
|---|---|
| `P2P.IntegrationTests/` (new) | All integration test code |
| `P2P.TestNode/Program.cs` | Extended with structured output + CLI flags |
| `P2P/P2PCompute.cs` | High-level facade (already wired) |
| `P2P/P2PTransport.cs` | Transport layer (may need observability hooks) |
| `PlaywrightMultiTest/ProjectRunner.cs` | Reference for subprocess patterns |
| `WebTorrent.ServerApp/Program.cs` | Local tracker reference |

---

## What Success Looks Like

When all 45+ tests pass: kernel dispatch works over real WebRTC, data integrity is preserved, multi-peer load balancing distributes correctly, fault tolerance handles disconnections, Ed25519 signatures verified end-to-end, RBAC enforced over real connections, cross-platform works, stress handled. Proven with math, not assumptions.
