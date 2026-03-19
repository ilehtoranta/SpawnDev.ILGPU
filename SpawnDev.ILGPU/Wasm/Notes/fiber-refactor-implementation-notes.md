# Wasm Fiber Refactor — Implementation Notes

## Status: In Progress

### What's Done (Infrastructure)
- `WasmCompiledKernel.cs`: Added `PhaseCount` property
- `WasmKernelFunctionGenerator.cs`:
  - Added `_phaseParamLocal` (param 9: phaseId) to kernel AND helper function signatures
  - Added `_phaseMode` flag and `_phaseStateOffset`
  - Added `EmitSaveAllLocals()` — spills all non-param locals to scratch
  - Added `EmitRestoreAllLocals()` — restores all non-param locals from scratch
  - Added `ComputePhaseStateSize()` — calculates spill region size
  - Added phase entry code in `GenerateStateMachineCode`: if phaseId > 0, restore locals
  - Added phase-mode path in `EmitBarrier`: save locals + br $exit (NEEDS BLOCK SPLITTING)
- `WasmAccelerator.cs`: Cross-worker barrier (busy-wait version, needs replacement with phase-based script)

### What's Left (Critical)

#### 1. Block Splitting at Barrier Points
**The problem:** Barriers can appear IN THE MIDDLE of an IR basic block. The state machine's `br_table` dispatches to block starts. If we save+return at a barrier and re-enter the block, the pre-barrier code re-executes (wrong — side effects on shared memory).

**The solution:** Before generating the state machine, scan the IR blocks for barrier instructions. Split any block that contains a barrier into:
- Sub-block A: code before the barrier
- Sub-block B: code after the barrier (the continuation)

The state machine's `br_table` gets expanded to include entries for all sub-blocks. At barrier points: save locals, set `_stateLocal = sub-block B's index`, `br $exit`. On phase re-entry: restore locals, `br_table` dispatches to sub-block B.

**Implementation location:** `GenerateStateMachineCode` — add a preprocessing step that:
1. Walks each IR block's values
2. If a `Barrier` instruction is found, records its position
3. Creates a split plan: `List<(BasicBlock block, int splitIndex, int subBlockId)>`
4. Expands `_blockMap` and `_blockCount` to include sub-blocks
5. Modified code emission: for each IR block, emit up to the barrier point, then close the sub-block; open the next sub-block for the continuation

#### 2. Worker Script Changes
Replace the worker script for barrier kernels to use phase-based execution:
```javascript
for (let g = 0; g < numGroups; g++) {
  for (let p = 0; p < phaseCount; p++) {
    for (let tid = threadStart; tid < threadEnd; tid++) {
      kernel(globalIdx, dimX, dimY, scratchBase+tid*spt, groupSize, tid,
             sharedMemBase, barrierBase, dynSharedLen, p, ...args);
    }
    // Cross-worker sync between phases
    interGroupBarrier(barrierView, workerCount);
  }
}
```
Note the `p` parameter passed as phaseId.

#### 3. Dispatch Changes
- `workerCount = Math.Min(groupSize, hardwareConcurrency)`
- Each worker gets `threadStart..threadEnd` (fiber range)
- `scratchPerThread` includes `_phaseStateOffset + ComputePhaseStateSize()`
- `PhaseCount` propagated to `WasmCompiledKernel`

#### 4. Helper Functions with Barriers
Helpers that contain barriers (like ExclusiveScan) also need phase splitting. Two options:
- **Inline helpers** into the kernel before phase splitting
- **Phase-split helpers independently** and have the worker script execute helper phases

The inlining approach is simpler for the initial implementation.

### Key Insight: The State Machine Is Our Friend

The existing state machine (`block/loop/br_table`) already supports jumping to arbitrary blocks. Phase re-entry just needs to set `_stateLocal` to the right sub-block index and enter the loop. All the infrastructure for dispatching to blocks is already there — we just need more entries in the `br_table`.

### Kernel Signature (10 system params + user params)
```
kernel(globalIdx, dimX, dimY, scratchBase, groupDimX, threadIdX,
       sharedMemBase, barrierBase, dynamicSharedLen, phaseId, ...userParams)
```
The `phaseId` param is always present (even for non-barrier kernels, where it's ignored). This simplifies the worker script — it always passes phaseId=0 for non-barrier kernels.

### Scratch Layout Per Thread
```
[alloca usage: 0.._scratchNextOffset]
[phase state: _phaseStateOffset..+stateSize]
  - local_0 (i32/i64/f32/f64)
  - local_1
  - ...
  - local_N
```
Total `ScratchPerThread` = max(alloca usage + state size, 4096 for non-barrier).
