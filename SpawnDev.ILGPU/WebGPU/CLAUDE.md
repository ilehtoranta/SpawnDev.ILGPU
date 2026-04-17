# WebGPU Backend

Transpiles ILGPU IR → WGSL shaders. Dispatches via `WebGPUAccelerator`.

## Key Files
- `Backend/WGSLKernelFunctionGenerator.cs` — main kernel codegen (~5900 lines)
- `Backend/WGSLCodeGenerator.cs` — base IR visitor, i64 constant dedup
- `Backend/WGSLEmulationLibrary.cs` — i64/f64 emulation functions with per-function trimming
- `Backend/SharedMemoryResolver.cs` — alloca→workgroup var matching, WGSL emission
- `Backend/UniformityAnalyzer.cs` — loop classification, PHI tracing, barrier detection
- `WebGPUAccelerator.cs` — dispatch, bind groups, buffer management, device loss monitoring
- `WebGPUStream.cs` — deferred encode + flush (batched submission)
- `WebGPUBackend.cs` — WGSLRegistry, WGSLDiagnostics, WGSLDumpPath, pre-validation

## Hard Constraints
- **4-byte alignment** on ALL buffer ops: sizes, writeBuffer, copyBufferToBuffer, bind group entries. Use `WebGPUAlignment.AlignTo4()`.
- **WGSL uniformity is syntactic** — browser traces variable origins through CFG. Anything touching `local_invocation_id` is non-uniform even if mathematically uniform. See `UniformityAnalyzer.cs`.
- **NaN/Inf** — use `bitcast<u32>()` bit-level checks, not `val != val`.
- **`__` prefix reserved** — never use double-underscore prefix in generated WGSL identifiers.
- **Emulation flags** — `SetEmulationFlags()` must run before `GenerateCode()`. Must scan helpers too.
- **KernelSpecialization** — required for algorithm kernels (RadixSort, Histogram, etc.) to bake workgroup size.
- **`TempViewManager.Allocate()`** — pad to 256 bytes for `minStorageBufferOffsetAlignment`.
- **Shared memory sizing** — RadixSort's ExclusiveScan calls share workgroup variables; trailing `Group.Barrier()` required after each scan call.
- **`_ilgpu_user_dim`** — override constant prevents excess threads corrupting buffers in auto-grouped kernels.

## Emulation
- **i64**: Always on. `vec2<u32>` paired 32-bit ops. `const _c_i64_N` hoisted constants.
- **f64**: Configurable via `F64EmulationMode` — Dekker (`vec2<f32>`), Ozaki (`vec4<f32>`), or Disabled.
- Library inclusion controlled by `SetEmulationFlags()` scanning kernel + helper IR.
- Per-function trimming via `GetMinimalEmulationLibrary()` BFS dependency graph.

## Buffer Copy Operations — What Works vs What Throws

| Operation | Method | WebGPU Implementation | Status |
|-----------|--------|----------------------|--------|
| GPU→GPU | `CopyFrom` / `ArrayView.CopyTo(ArrayView)` | `CopyBufferToBuffer` | **WORKS** |
| CPU→GPU | `CopyFromCPU` | `queue.WriteBuffer` | **WORKS** |
| GPU→CPU (sync) | `CopyTo` / `CopyToCPU` / `GetAsArray1D` | N/A | **THROWS NotSupportedException** |
| GPU→CPU (async) | `CopyToHostAsync` | `mapAsync(Read)` | **WORKS** |

**NEVER replace `CopyFrom` with `Scale(×1)` kernel dispatch.** `CopyFrom` is a native GPU command (`CopyBufferToBuffer`) — no shader compilation, no dispatch overhead. `Scale(×1)` requires kernel loading and dispatch, which causes "obj null or undefined" errors during early session initialization when accelerator state isn't fully wired. This was proven in ML commit 45b7cba (13+ WebGPU failures, reverted).

**The confusion:** `CopyTo` (GPU→**CPU**) throws on WebGPU. `CopyFrom` (GPU→**GPU**) works perfectly. They are different operations going different directions. When you need GPU→CPU, use `CopyToHostAsync`. When you need GPU→GPU, use `CopyFrom`.

## Command Batching & Synchronization

**WebGPUStream batches compute passes into one command encoder.** `Synchronize()` = `Flush()` = finishes the encoder and submits to the GPU queue. This is NON-BLOCKING — it submits but does NOT wait for completion.

**`SynchronizeAsync()`** calls `FlushPendingCommands()` then `queue.OnSubmittedWorkDone()` — this DOES wait. But `OnSubmittedWorkDone` can deadlock in Blazor WASM if too much GPU work is queued (100+ compute passes → Chrome GPU watchdog timeout).

**Rule: Flush periodically for large workloads.** If dispatching many kernels (>50), call `Synchronize()` every 16-32 dispatches to submit smaller batches. This prevents the GPU command buffer from growing too large. Example:
```csharp
for (int i = 0; i < 112; i++) {
    accelerator.LaunchKernel(...);
    if (i % 16 == 0) accelerator.Synchronize(); // Flush batch
}
// Final flush + async wait
accelerator.Synchronize();
await accelerator.SynchronizeAsync(); // Safe — only waits for last small batch
```

**`CopyToHostAsync` internally:** FlushPendingCommands → CopyBufferToBuffer → Submit → `MapAsync(Read)`. The `MapAsync` waits for the copy to finish, which is queued behind all prior work. If prior work is large, `MapAsync` may timeout.

## i64/f64 Atomic Operations (v4.9.2-rc.5+)

WGSL only has 32-bit atomics. i64 is emulated as `vec2<u32>`. Atomic support:

| Operation | i64 | f64 | Method |
|-----------|-----|-----|--------|
| And/Or/Xor | Supported | N/A | Dual i32 atomics on lo/hi halves (independent) |
| Add | Supported | Supported | i64: CAS on lo + atomicAdd on hi (lock-free). f64: spinlock + f64_add. |
| Min/Max | Supported | Supported | Spinlock companion buffer + dual-u32 atomicLoad/atomicStore critical section |
| Exchange | Supported | Supported | Spinlock companion buffer + dual-u32 atomicStore critical section |
| CAS | Not supported | Not supported | Throws `NotSupportedException` - WGSL has no 64-bit CAS |

**i32/f32 atomics are fully supported** via native WGSL atomics (i32) or CAS loops (f32).

**Spinlock pattern:** For operations that need atomicity across both u32 words (Min/Max/Exchange on i64/f64, and Add on f64), a companion `array<atomic<u32>>` lock buffer is auto-provisioned by `ScanForAtomicUsage`. Each 64-bit slot gets its own lock word; threads `atomicCompareExchangeWeak` to acquire, perform `atomicLoad`/`atomicStore` on both halves inside the critical section, then release with `atomicStore(lock, 0u)`. i64 Add uses a lock-free dual-atomic path instead (commutative carry).

## Diagnostics
- `WebGPUBackend.WGSLDumpPath` — dump shaders to files (desktop only)
- `WebGPUBackend.WGSLRegistry` — named registry of compiled shaders
- `WebGPUBackend.WGSLDiagnostics` — flags enum for per-category logging
- `WebGPUAccelerator.DispatchLog` — ring buffer of last 100 dispatches
- Shader header comments: kernel name, workgroup size, shared memory, emulation flags
