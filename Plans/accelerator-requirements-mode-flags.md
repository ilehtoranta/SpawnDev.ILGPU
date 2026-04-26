# Plan: Mode-Shape Capability Flags in `AcceleratorRequirements` (local + P2P)

**Status:** Design / not yet implemented. Draft 2026-04-25 in response to TJ's question about "user needs strict IEEE 754 f64 — can a single flag handle backend selection AND mode configuration end to end."

**Updated 2026-04-25 (post-TJ corrections):**
> 1. "ALL peers support Ozaki and Dekker on WebGPU and WebGL"
> 2. "IEEE 754 f64 is universal on all of our current backends"
> 3. "Dekker f64 is only available on WebGL and WebGPU"

These three facts collapse the original "rejective filter" Phase 1 P2P design — every current peer can satisfy `RequiresFloat64Strict` because every backend can produce IEEE 754 f64 (natively on CPU/CUDA/OpenCL/Wasm, via Ozaki on WebGPU/WebGL). The remaining work is **configuration**, not selection: tell WebGPU/WebGL accelerators to switch to Ozaki for the dispatch. Plan rewritten below to match.

## The architectural distinction

`AcceleratorRequirements` today is a **filter** — it rejects backends that don't have a feature, but doesn't change how a backend is configured. That fits two requirement shapes; a third shape doesn't fit and is the gap this plan fills.

| Shape | Examples | How `AcceleratorRequirements` expresses it today |
|---|---|---|
| **Binary feature, hardware-only** | atomics, shared memory, barriers, sub-groups, native f64 hardware | `RequiresAtomics` etc. — pure filter, rejects backends without the feature |
| **Binary feature, always-available** | Float16 (lossless on every backend via emulation that fits in f32) | `RequiresFloat16` is a documentation no-op; `RequiresFloat16Native` is the real filter when emulation overhead is unacceptable |
| **Configurable mode (always-satisfiable)** | f64 IEEE 754-strict (Dekker ~48-53 bits vs Ozaki strict 53-bit), future: deterministic FMA, strict NaN propagation | **Not expressible today.** Every backend CAN satisfy strict-f64, but WebGPU/WebGL need to be told to use Ozaki instead of Dekker. Currently the user has to manually set `WebGPUBackendOptions.F64Emulation = Ozaki` at backend registration time, which is global, immutable, and doesn't compose with `AcceleratorRequirements`. |

The third shape is what this plan addresses. Crucially: **it is not a filter shape**. Every current backend supports strict IEEE 754; the flag's job is to configure the browser-backend mode, not to reject peers.

## Mode availability matrix

This is the fact that the rest of the plan rides on. **Strict IEEE 754 f64 is universal**; Dekker mode is a WebGPU/WebGL-only fast-but-lossy mode.

| Backend | Native strict f64 | Dekker (~48-53 bit fast) | Ozaki (strict 53-bit) | Default mode |
|---|---|---|---|---|
| CPU | Yes | N/A | N/A | Native f64 |
| CUDA | Yes | N/A | N/A | Native f64 |
| OpenCL | Yes | N/A | N/A | Native f64 |
| Wasm | Yes | N/A | N/A | Native f64 |
| WebGPU | No | **Yes** | **Yes** | Dekker |
| WebGL | No | **Yes** | **Yes** | Dekker |

`RequiresFloat64Strict = true` accepts every row of this table; on the WebGPU/WebGL rows it changes the column from "Dekker" to "Ozaki" via the configuration plumbing in Phase 1.

## The flag

```csharp
public sealed class AcceleratorRequirements
{
    // ... existing flags ...

    /// <summary>
    /// Kernel requires strict IEEE 754 double-precision arithmetic.
    /// All current backends can satisfy this:
    ///   - CPU / CUDA / OpenCL / Wasm: accepted unchanged - they are always strict f64.
    ///   - WebGPU / WebGL: accepted, with the accelerator's F64 emulation mode
    ///     promoted to Ozaki for the dispatch (overrides the backend default of Dekker).
    /// Dekker mode (the WebGPU / WebGL default) is REJECTED at codegen time when this
    /// flag is set, because its ~48-53 bit mantissa diverges from IEEE 754 53-bit.
    ///
    /// This is a CORRECTNESS CONTRACT: the flag is the user's promise that they
    /// would rather a dispatch fail than receive Dekker-precision math under the
    /// Float64Strict label. P2P coordinators use the flag to ensure every peer
    /// in the swarm runs the kernel under the same numeric model.
    /// </summary>
    public bool RequiresFloat64Strict { get; init; }
}
```

Same shape generalizes to future configurable-mode flags (`RequiresDeterministicFloat32`, `RequiresStrictNanPropagation`, etc.) without further architectural changes.

## Phase 1 — Local: dynamic F64 mode on a single backend instance

Today `WebGPUBackendOptions.F64Emulation` is set at `Context.Builder.WebGPU(opts)` registration time and immutable thereafter. The fix: make F64 mode a property on the WebGPU **accelerator instance** that codegen reads at shader-compile time, with a clean lifecycle hook in `CreateAccelerator`.

### Changes required

1. **`SpawnDev.ILGPU/WebGPU/WebGPUAccelerator.cs`** — add a public `F64EmulationMode F64Mode { get; set; }` property, default initialized from the backend's options. Codegen path (`WGSLTypeGenerator`, `WGSLEmulationLibrary`) reads from the accelerator instance instead of from `Backend.Options.F64Emulation`. Same shape applies to `WebGLAccelerator.F64Mode`.

2. **Shader cache key** — `F64Mode` joins the cache key for compiled shaders. A single backend instance can hold both a Dekker-cached shader and an Ozaki-cached shader for the same kernel C# without collision. Adds one component to whatever generates the cache key today.

3. **`AcceleratorRequirementsExtensions.CreatePreferredAccelerator`** — when `RequiresFloat64Strict = true` and the picked device is `WebGPU` or `WebGL`, set the accelerator's `F64Mode = Ozaki` immediately after creation, before any kernel is loaded.

4. **Filter logic in `Satisfies()`** — `RequiresFloat64Strict` accepts every device unconditionally (because every backend can satisfy it). Native-f64 devices (CPU/CUDA/OpenCL/Wasm) accept directly; WebGPU/WebGL devices accept on the understanding that the runtime will configure Ozaki on the resulting accelerator.

5. **`UnsupportedKernelFeatureException` integration** — if a user pins to `WebGPUAccelerator.F64Mode = Disabled` (the f32-promotion mode) and dispatches a kernel that uses `double` AND `RequiresFloat64Strict` is set, throw at compile-time with `Feature = "double in F64Mode.Disabled when RequiresFloat64Strict is set"`.

### Why dynamic-mode-on-single-backend instead of "register WebGPU twice"

The "register WebGPU once for Dekker and once for Ozaki" alternative was discussed earlier; rejected because:

- Two devices in `context.Devices` with the same `AcceleratorType.WebGPU` confuses every iteration site in user code.
- Doubles the device-enumeration cost at startup (`navigator.gpu.requestAdapter()` JS interop, ~50-200 ms first request, cached after).
- Doubles the test surface: PMT discovers both, tests targeting "WebGPU" become ambiguous.

Dynamic mode on a single accelerator: single device entry, single registration, runtime config via accelerator property. Cache key adds one component (~one-line change); no API surface duplication.

## Phase 1 — P2P: configuration request, not capability filter

This is where the original draft was wrong and TJ's three corrections matter most. Because every browser peer (running rc.10 or newer) supports both Dekker and Ozaki, the coordinator does not filter peers by configured mode — instead, the dispatch carries the desired mode and the peer configures its accelerator on receipt.

### Wire format addition

`ComputePeerCapabilities` (the message a peer returns during BEP 10 `sd_compute` handshake) currently reports backend type and basic features. Add a per-backend "supported f64 modes" advertisement so coordinators can detect ancient peers (pre-Ozaki) that genuinely cannot satisfy `RequiresFloat64Strict`:

```
ComputePeerCapabilities {
    backends: [
        {
            type: "WebGPU",
            atomics: true,
            shared_memory: true,
            barriers: true,
            float16_native: false,
            float64_native: false,
            // NEW: modes this backend can be configured into.
            // Current rc.10+ peers always advertise ["Dekker", "Ozaki"].
            // A pre-Ozaki peer would advertise ["Dekker"] only.
            float64_modes_supported: ["Dekker", "Ozaki"],
        },
        {
            type: "Wasm",
            atomics: true,
            ...
            float64_native: true,
            // Native-f64 backends do not need a modes list - native means always strict.
        }
    ]
}
```

The capability advertisement is **a precondition check, not a runtime config**. The peer's ACTIVE F64 mode is whatever the dispatch tells it to use.

### Dispatch carries the mode

`KernelDispatchRequest` (the message a coordinator sends to a peer to run a kernel) gains a per-backend `f64_mode` field:

```
KernelDispatchRequest {
    kernel_id: "...",
    f64_mode: "Ozaki",    // or "Dekker" - peer's WebGPU/WebGL accelerator runs in this mode for this dispatch
    // ... other fields
}
```

When the dispatch arrives, the peer:

1. Sets its WebGPU/WebGL accelerator's `F64Mode` to the requested value (Phase 1 plumbing).
2. Compiles + runs the kernel under that mode.
3. Sends the result back as normal.

For native-f64 peers (CPU/CUDA/OpenCL/Wasm), the `f64_mode` field is informational — they are always strict regardless. The coordinator sends it; the peer logs it and proceeds.

### Coordinator pre-flight check

When a coordinator dispatches a kernel with `RequiresFloat64Strict = true`:

- Loop peer capabilities. For each backend on each peer, call `Satisfies(requirements)`.
- For WebGPU/WebGL backends: accept if `float64_modes_supported` contains "Ozaki". (All rc.10+ peers do; only deeply old peers would be filtered out here.)
- For native-f64 backends: accept unconditionally.
- For unrecognized old peers that don't include `float64_modes_supported` at all: treat as "Dekker only" (conservative) and filter out under `RequiresFloat64Strict`.
- The dispatch then emits `f64_mode: "Ozaki"` to every selected WebGPU/WebGL peer; native-f64 peers ignore it.

### Non-negotiable correctness rules

These fall out of "configurable-mode flags are correctness contracts not preferences":

1. **Every selected peer runs in the requested mode.** The coordinator does not silently demote some peers to Dekker for performance and others to Ozaki for strictness within the same dispatch. One mode for the whole swarm-dispatch.

2. **Coordinator-local fallback obeys the same rule.** If the coordinator dispatches some work locally rather than to peers, its local accelerator MUST also be in the requested mode. Otherwise the swarm produces some results from strict peers and some from local Dekker — bit-exact verification then fails non-deterministically. The local accelerator factory in P2P swarm setup must use the same `AcceleratorRequirements` as the peer-selection logic, and must apply the same `F64Mode = Ozaki` configuration when `RequiresFloat64Strict` is set.

3. **No "best effort" fallback.** If `RequiresFloat64Strict = true` and a peer's capabilities don't list Ozaki support, that peer is excluded from the dispatch. The dispatch does not silently downgrade to Dekker. If zero peers + local can satisfy, the dispatch throws. The whole point of declaring the requirement is to refuse to ship wrong numbers. Callers that want a fallback can catch the exception and dispatch a different kernel (or accept different precision) explicitly — that's their decision, not the swarm's.

4. **Ack on mode change is not required for Phase 1.** Because the mode is set per-dispatch and the result-validation happens at result-receipt time (CPU oracle compare), the coordinator detects mode-mismatch bugs as failed verification. A future Phase 2 ack handshake can close the gap before computation starts; Phase 1 catches it after.

## Phase 2 — Per-dispatch ack handshake (deferred)

Phase 1 sends the mode in the dispatch and trusts the peer's reconfigure to land before kernel execution. Phase 2 adds an explicit ack so the coordinator can fail fast if the peer's reconfigure didn't take effect:

1. Coordinator sends `KernelDispatchRequest { f64_mode: "Ozaki" }`.
2. Peer reconfigures its accelerator's `F64Mode`.
3. Peer sends `KernelDispatchAck { configured_modes: { WebGPU: "Ozaki" } }` confirming the active mode.
4. Coordinator validates the ack — if mode mismatch, reject the dispatch entirely.
5. Then dispatch begins.

Without this ack there is a verification gap: peer could claim "ok done" but actually still run in the old mode (e.g. due to a code path that didn't honor the request). The ack closes the gap by making the active mode part of the dispatch contract.

**When to ship Phase 2:** when bit-exact verification on Phase 1 surfaces a real-world peer-side mode-honor bug. Until then, result-side bit-exact compare against the CPU oracle catches incorrect modes downstream — which is sufficient for current single-coordinator + single-swarm-dispatch usage.

## Generalization beyond f64 strict

Same plan shape (capability-advertised + per-dispatch mode field) handles future configurable-mode flags:

- **`RequiresDeterministicFloat32`**: turn off FMA fusion on backends that allow it. CUDA + WebGPU support this via codegen flags. CPU/Wasm are deterministic by default.
- **`RequiresStrictNanPropagation`**: prevent NaN-flushing optimizations. Mostly relevant to CUDA + WebGPU's fast-math options.
- **`RequiresF16NoFlush`**: prevent denormal-flush-to-zero on Float16 — relevant when ML quantization is sensitive.

Each is a configurable backend mode. Same `AcceleratorRequirements` flag → same `Satisfies()` filter (always-accept for the always-satisfiable modes; reject only for legitimately-incapable backends) → same dynamic-mode plumbing → same wire format addition for P2P.

## Verification

A `RequiresFloat64Strict` test in `BackendTestBase` would:

1. Compute a known IEEE-754-strict result on the accelerator.
2. Compare against a CPU reference computed in C# with `double`.
3. Bit-exact match required.

Test passes on every backend the accelerator-requirements path picks:

- Native-f64 backends always pass (always strict).
- WebGPU/WebGL pass when configured for Ozaki, fail when configured for Dekker. The `RequiresFloat64Strict` path triggers Ozaki configuration, so the test naturally passes — and by the same token, removing the requirement reverts to Dekker and the test naturally fails (negative test confirms the path actually mode-shifts).

For P2P: a swarm test where the coordinator sets `RequiresFloat64Strict` and dispatches to multiple browser peers. Each peer's WebGPU/WebGL accelerator should arrive at the dispatch in Ozaki mode; both swarm-result and coordinator-local-fallback path verified to bit-exact CPU reference. A second test with a deliberately old (synthetic) peer capability advertising no Ozaki support should be filtered out by the pre-flight check.

## Sequencing / ship gate

- 4.9.2 stable: ships without this plan. WebGPU/WebGL F64 mode stays at registration-time-immutable, users either accept Dekker or set `WebGPUBackendOptions.F64Emulation = Ozaki` at startup.
- 4.9.3 candidate: Phase 1 implementation (dynamic F64 mode on accelerator + `RequiresFloat64Strict` flag in local AND P2P).
- 4.9.4 or later: Phase 2 ack handshake — only if a real consumer surfaces mode-honor bug.

## See also

- [`Docs/capabilities-and-backend-selection.md`](../Docs/capabilities-and-backend-selection.md) — public-facing API doc for `AcceleratorRequirements` (added 2026-04-25)
- [`Plans/wasm-backend-stable-gates.md`](wasm-backend-stable-gates.md) — companion doc for the OTHER multi-backend gate currently blocking 4.9.2 stable
- [`SpawnDev.ILGPU/F64EmulationMode.cs`](../SpawnDev.ILGPU/F64EmulationMode.cs) — the existing enum (Dekker/Ozaki/Disabled)
- [`SpawnDev.ILGPU/AcceleratorRequirements.cs`](../SpawnDev.ILGPU/AcceleratorRequirements.cs) — the existing filter API to extend
