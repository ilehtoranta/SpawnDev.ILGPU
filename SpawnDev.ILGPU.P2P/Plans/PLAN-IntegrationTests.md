# Plan: P2P Integration Tests - Full Coverage

**Author:** Geordi (Claude CLI #4)
**Date:** 2026-04-14
**Status:** IN PROGRESS - Phases 1-4 in-process tests PASSING (27/27). Phase 2 WebRTC tests blocked on WebTorrent (Riker).
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

### Phase 2: Core Pipeline (Desktop-Desktop) [NOT STARTED]
- [ ] `CorePipeline_VectorAdd_1024` - a[i]=i, b[i]=i*2, verify result[i]==i*3 for ALL 1024 elements over real WebRTC
- [ ] `CorePipeline_LargeBuffer_256K` - 1MB float buffer, 16 chunks, full integrity
- [ ] `CorePipeline_ScalarParams` - VectorScale with float scalar across WebRTC
- [ ] `CorePipeline_IntegerKernel` - ArrayView<int> Identity kernel
- [ ] `CorePipeline_DataIntegrity_SHA256` - random 64KB, Identity kernel, SHA256 before/after
- [ ] `CorePipeline_InProcess_Baseline` - selftest pattern, no WebRTC, regression baseline

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

### Phase 6: Buffer, State, Stress, Pipeline [NOT STARTED]
- [ ] `Buffer_1MB_Chunked_Verified` - 16 chunks, byte-for-byte
- [ ] `Buffer_4MB_Stress` - data channel queuing under load
- [ ] `State_PublishSubscribe_RealDHT` - BEP 46 publish/subscribe
- [ ] `State_CoordinatorAnnouncement` - DHT announcement received
- [ ] `Stress_RapidJoinLeave_10Cycles` - 10 join/leave cycles, no leaks
- [ ] `Stress_Concurrent_20Dispatches` - 20 simultaneous to 3 workers
- [ ] `Stress_LargeSwarm_5Peers` - 5 workers, 15 dispatches
- [ ] `Pipeline_TwoStage_EndToEnd` - FillSequence -> VectorScale

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
