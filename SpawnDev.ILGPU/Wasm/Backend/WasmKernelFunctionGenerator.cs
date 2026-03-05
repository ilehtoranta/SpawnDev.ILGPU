// ---------------------------------------------------------------------------------------
//                               SpawnDev.ILGPU.Wasm
//                    WebAssembly Compute Backend for Blazor WebAssembly
//
// File: WasmKernelFunctionGenerator.cs
//
// Generates the kernel entry-point function in WebAssembly binary format.
// Handles parameter binding, index computation, memory layout, and
// the main multi-block state machine for kernel execution.
// ---------------------------------------------------------------------------------------

using global::ILGPU;
using global::ILGPU.IR;
using global::ILGPU.IR.Analyses;
using global::ILGPU.IR.Analyses.ControlFlowDirection;
using global::ILGPU.IR.Analyses.TraversalOrders;
using global::ILGPU.IR.Types;
using global::ILGPU.IR.Values;
using System.Text;

namespace SpawnDev.ILGPU.Wasm.Backend
{
    /// <summary>
    /// Generates the Wasm kernel function.
    ///
    /// Function signature (1D): kernel(globalIdx, dimX, param0_offset, param0_len, param1_value, ...)
    ///
    /// The kernel imports a shared WebAssembly.Memory. Buffer parameters are passed as
    /// byte offsets into this linear memory. Scalar parameters are passed directly.
    ///
    /// Memory layout:
    ///   [0..sync_region] [buffer0_offset..] [buffer1_offset..] [scalar_area..]
    ///   Offsets are computed by the accelerator at dispatch time.
    /// </summary>
    public class WasmKernelFunctionGenerator : WasmCodeGenerator
    {
        /// <summary>
        /// Wasm function parameter types (built during Setup).
        /// </summary>
        public readonly List<byte> FuncParamTypes = new();

        /// <summary>
        /// Maps ILGPU parameter index to its Wasm local indices.
        /// For ArrayView: [offset_local, length_local]
        /// For scalars: [value_local]
        /// </summary>
        private readonly Dictionary<int, uint[]> _paramLocals = new();

        /// <summary>
        /// For view parameters, maps ILGPU param index to the first stride field index in the IR struct.
        /// Strides are always the trailing Int32 fields after the pointer + Int64 extent fields.
        /// </summary>
        private readonly Dictionary<int, int> _viewStrideStartField = new();

        /// <summary>
        /// The parameter info for marshaling (written to GeneratorArgs).
        /// </summary>
        private readonly List<WasmParamInfo> _paramInfos = new();

        /// <summary>
        /// Wasm local index for the global index.
        /// </summary>
        private uint _globalIdxLocal;

        /// <summary>
        /// Wasm local index for dimension X.
        /// </summary>
        private uint _dimXLocal;

        /// <summary>
        /// Wasm local index for dimension Y (used for 2D/3D kernels).
        /// </summary>
        private uint _dimYLocal;

        /// <summary>
        /// Wasm local index for scratch memory base address.
        /// Used by StructureValue for temporary struct construction in linear memory.
        /// </summary>
        private uint _scratchBaseLocal;

        /// <summary>
        /// Tracks the next available byte offset within the scratch memory area.
        /// Reset for each kernel compilation.
        /// </summary>
        private int _scratchNextOffset = 0;

        /// <summary>
        /// The index type of the kernel (Index1D, Index2D, Index3D).
        /// </summary>
        private IndexType _indexType = IndexType.Index1D;

        /// <summary>
        /// The IR parameter that represents the implicit index (for 2D/3D decomposition).
        /// </summary>
        private global::ILGPU.IR.Values.Parameter? _indexParam;

        /// <summary>
        /// Whether this kernel uses views (ArrayView).
        /// </summary>
        private bool _hasViews = false;

        /// <summary>
        /// Maps BasicBlock to its index in the state machine.
        /// </summary>
        private readonly Dictionary<BasicBlock, int> _blockMap = new();

        /// <summary>
        /// State local for the state machine (_block variable).
        /// </summary>
        private uint _stateLocal;

        /// <summary>
        /// Whether the state machine is active (multi-block kernel).
        /// </summary>
        private bool _isStateMachine = false;

        /// <summary>
        /// Number of blocks in the state machine (used for br_table nesting depth).
        /// </summary>
        private int _blockCount = 0;

        /// <summary>
        /// Index of the block currently being emitted (0-indexed). Used for br depth calculation.
        /// </summary>
        private int _currentBlockEmitIndex = 0;

        // === Group execution params (for shared memory + barrier support) ===

        /// <summary>
        /// Wasm local index for group dimension X (number of threads in workgroup).
        /// </summary>
        private uint _groupDimXLocal;

        /// <summary>
        /// Wasm local index for thread index within the workgroup.
        /// </summary>
        private uint _threadIdXLocal;

        /// <summary>
        /// Wasm local index for the base address of shared memory region in linear memory.
        /// </summary>
        private uint _sharedMemBaseLocal;

        /// <summary>
        /// Wasm local index for the base address of barrier slots in linear memory.
        /// </summary>
        private uint _barrierBaseLocal;

        /// <summary>
        /// Wasm local index for the dynamic shared memory element count (passed at dispatch time).
        /// </summary>
        private uint _dynamicSharedLengthLocal;

        /// <summary>
        /// Counter for barrier synchronization points in the kernel.
        /// </summary>
        private int _barrierCounter = 0;

        /// <summary>
        /// Whether this kernel uses shared memory or barriers.
        /// </summary>
        private bool _hasBarriers = false;

        /// <summary>
        /// Total bytes allocated for shared memory in this kernel.
        /// </summary>
        private int _sharedMemorySize = 0;

        /// <summary>
        /// Maps Alloca IR values to their byte offsets within the shared memory region.
        /// </summary>
        private readonly Dictionary<string, int> _sharedAllocaOffsets = new();

        /// <summary>
        /// Element size in bytes for dynamic shared memory allocations.
        /// </summary>
        private int _dynamicSharedElementSize = 0;

        public WasmKernelFunctionGenerator(
            in GeneratorArgs args,
            Method method,
            Allocas allocas)
            : base(args, method, allocas)
        {
        }

        /// <summary>
        /// Sets up parameter-to-local mappings. Must be called before visiting IR blocks,
        /// since ILGPU calls GenerateCode() BEFORE GenerateHeader().
        /// </summary>
        private void SetupParameters()
        {
            if (_parametersInitialized) return;
            _parametersInitialized = true;

            var entryPoint = _generatorArgs.EntryPoint;
            var parameters = Method.Parameters;

            WasmBackend.Log($"[Wasm-Setup] Parameters.Count={parameters.Count}, IsExplicitlyGrouped={entryPoint.IsExplicitlyGrouped}, _nextLocalIndex={_nextLocalIndex}");

            // Reset local tracking — params occupy indices 0..N-1
            // in the Wasm function. We track the param count so that
            // AllocateLocal knows when to emit extra locals.
            _locals.Clear();
            _localMap.Clear();
            _nextLocalIndex = 0;
            _paramCount = 0;

            // Store the index type for decomposition in GetField
            _indexType = entryPoint.IndexType;

            // Fixed params: globalIdx (i32), dimX (i32), dimY (i32), scratchBase (i32),
            //               groupDimX (i32), threadIdX (i32), sharedMemBase (i32), barrierBase (i32)
            _globalIdxLocal = _nextLocalIndex++;
            _paramCount++;

            _dimXLocal = _nextLocalIndex++;
            _paramCount++;

            _dimYLocal = _nextLocalIndex++;
            _paramCount++;

            _scratchBaseLocal = _nextLocalIndex++;
            _paramCount++;
            _scratchNextOffset = 0;

            _groupDimXLocal = _nextLocalIndex++;
            _paramCount++;

            _threadIdXLocal = _nextLocalIndex++;
            _paramCount++;

            _sharedMemBaseLocal = _nextLocalIndex++;
            _paramCount++;

            _barrierBaseLocal = _nextLocalIndex++;
            _paramCount++;

            _dynamicSharedLengthLocal = _nextLocalIndex++;
            _paramCount++;

            // FuncParamTypes tracks Wasm type signatures for the module builder
            FuncParamTypes.Clear();
            FuncParamTypes.Add(WasmOpCodes.I32); // param 0: globalIdx
            FuncParamTypes.Add(WasmOpCodes.I32); // param 1: dimX
            FuncParamTypes.Add(WasmOpCodes.I32); // param 2: dimY
            FuncParamTypes.Add(WasmOpCodes.I32); // param 3: scratchBase
            FuncParamTypes.Add(WasmOpCodes.I32); // param 4: groupDimX
            FuncParamTypes.Add(WasmOpCodes.I32); // param 5: threadIdX
            FuncParamTypes.Add(WasmOpCodes.I32); // param 6: sharedMemBase
            FuncParamTypes.Add(WasmOpCodes.I32); // param 7: barrierBase
            FuncParamTypes.Add(WasmOpCodes.I32); // param 8: dynamicSharedLength

            // CRITICAL: Determine if param 0 is the implicit index parameter.
            // IndexType is unreliable: ILGPU overrides it to KernelConfig for ALL kernels
            // using SharedMemory, even those with an actual Index1D first parameter.
            // Instead, we check: (1) IndexType != None (kernel expects some index), AND
            // (2) parameters[0] is NOT an ArrayView type (views are always user params).
            // This correctly handles:
            //   SharedMemoryKernel(Index1D, ArrayView) → param[0]=PrimitiveType → startIdx=1
            //   PrefixSumKernel(ArrayView)             → param[0]=View → startIdx=0
            //   Kernel2D(Index2D, ArrayView2D)          → param[0]=StructureType(non-view) → startIdx=1
            bool hasImplicitIndex = entryPoint.IndexType != IndexType.None
                && parameters.Count > 0
                && !IsViewType(parameters[0].Type);
            int startIdx = hasImplicitIndex ? 1 : 0;

            // Map the implicit index param to globalIdx
            if (startIdx == 1 && parameters.Count > 0)
            {
                _indexParam = parameters[0];
                _localMap[GetValueKey(_indexParam)] = _globalIdxLocal;
                WasmBackend.Log($"[Wasm-Setup] Index param {GetValueKey(_indexParam)} -> local_{_globalIdxLocal}, IndexType={_indexType}");
            }

            for (int i = startIdx; i < parameters.Count; i++)
            {
                var param = parameters[i];
                var paramType = param.Type;
                bool isView = IsViewType(paramType);

                WasmBackend.Log($"[Wasm-Setup] param[{i}] id={param.Id} type={paramType} isView={isView} _nextLocalIndex={_nextLocalIndex}");

                if (isView)
                {
                    _hasViews = true;

                    FuncParamTypes.Add(WasmOpCodes.I32); // byte offset
                    uint offsetLocal = _nextLocalIndex++;
                    _paramCount++;

                    FuncParamTypes.Add(WasmOpCodes.I32); // element count
                    uint lengthLocal = _nextLocalIndex++;
                    _paramCount++;

                    FuncParamTypes.Add(WasmOpCodes.I32); // stride (YStride for 2D, YStride for 3D)
                    uint strideLocal = _nextLocalIndex++;
                    _paramCount++;

                    FuncParamTypes.Add(WasmOpCodes.I32); // stride2 (ZStride for 3D, 0 for 1D/2D)
                    uint stride2Local = _nextLocalIndex++;
                    _paramCount++;

                    _paramLocals[i] = new[] { offsetLocal, lengthLocal, strideLocal, stride2Local };
                    _localMap[GetValueKey(param)] = offsetLocal;

                    WasmBackend.Log($"[Wasm-Setup]   View: {GetValueKey(param)}=local_{offsetLocal} (length=local_{lengthLocal}, stride=local_{strideLocal}, stride2=local_{stride2Local})");

                    // Determine stride field start index from the IR struct type
                    // The struct layout is: [View, Int64..., Int32...] where Int32 fields are strides
                    int strideStart = 1; // default: right after field 0
                    if (paramType is StructureType structType)
                    {
                        int fieldCount = structType.NumFields;
                        // Find first Int32 field after field 0 - these are stride fields
                        strideStart = fieldCount; // default to end (no strides)
                        for (int fi = 1; fi < fieldCount; fi++)
                        {
                            var fieldType = structType.Fields[fi];
                            if (fieldType is PrimitiveType pft && pft.BasicValueType == BasicValueType.Int32)
                            {
                                strideStart = fi;
                                break;
                            }
                        }
                    }
                    _viewStrideStartField[i] = strideStart;
                    WasmBackend.Log($"[Wasm-Setup]   View strideStartField={strideStart}");

                    var elemType = GetViewElementType(paramType);
                    int elemSize = 4;
                    if (elemType is PrimitiveType pt)
                        elemSize = GetElementSize(pt.BasicValueType);

                    _paramInfos.Add(new WasmParamInfo
                    {
                        Index = i,
                        Name = $"param{i}",
                        IsView = true,
                        WasmType = WasmOpCodes.I32,
                        ElementSize = elemSize,
                    });
                }
                else
                {
                    var wasmType = GetWasmTypeFromIR(paramType);
                    FuncParamTypes.Add(wasmType);
                    uint valLocal = _nextLocalIndex++;
                    _paramCount++;

                    _paramLocals[i] = new[] { valLocal };
                    _localMap[GetValueKey(param)] = valLocal;

                    WasmBackend.Log($"[Wasm-Setup]   Scalar: {GetValueKey(param)}=local_{valLocal}");

                    _paramInfos.Add(new WasmParamInfo
                    {
                        Index = i,
                        Name = $"param{i}",
                        IsScalar = true,
                        WasmType = wasmType,
                    });
                }
            }

            // Store param infos for the backend
            _generatorArgs.ParamInfos = _paramInfos;

            // Setup shared memory allocations
            SetupSharedAllocations(Allocas.SharedAllocations, isDynamic: false);
            SetupSharedAllocations(Allocas.DynamicSharedAllocations, isDynamic: true);

            // Propagate metadata to GeneratorArgs for WasmCompiledKernel
            // NOTE: Only set SharedMemorySize and DynamicSharedElementSize here.
            // BarrierCount and HasBarriers are set AFTER code generation in GenerateCode()
            // because _barrierCounter is incremented during IR visiting (GenerateCode(Barrier)).
            _generatorArgs.SharedMemorySize = _sharedMemorySize;
            _generatorArgs.DynamicSharedElementSize = _dynamicSharedElementSize;

            WasmBackend.Log($"[Wasm-Setup] Final: _nextLocalIndex={_nextLocalIndex}, _paramCount={_paramCount}, FuncParamTypes={FuncParamTypes.Count}");
            WasmBackend.Log($"[Wasm-Setup] _localMap: {string.Join(", ", _localMap.Select(kv => $"{kv.Key}={kv.Value}"))}");
            WasmBackend.Log($"[Wasm-Setup] SharedMemorySize={_sharedMemorySize}, HasBarriers={_hasBarriers}");
        }

        private bool _parametersInitialized = false;

        /// <summary>
        /// GenerateHeader is called AFTER GenerateCode by ILGPU.
        /// It only needs to ensure FuncParamTypes is populated (which SetupParameters already did).
        /// </summary>
        public override void GenerateHeader(StringBuilder builder)
        {
            // SetupParameters was already called from GenerateCode.
            // FuncParamTypes is already populated. Nothing else to do here.
            SetupParameters(); // safe to call again (idempotent)
        }

        /// <summary>
        /// Gets the block index for a basic block in the state machine.
        /// </summary>
        private int GetBlockIndex(BasicBlock block)
        {
            if (_blockMap.TryGetValue(block, out var idx))
                return idx;
            return 0;
        }

        /// <summary>
        /// Pushes phi values before a branch.
        /// For each target block, find any phi values that source from the current block
        /// and assign them before transitioning.
        /// </summary>
        private void PushPhiValues(Branch branch)
        {
            for (int i = 0; i < branch.NumTargets; i++)
            {
                var target = branch.Targets[i];
                foreach (var valueEntry in target)
                {
                    if (valueEntry.Value is PhiValue phi)
                    {
                        // Find the value from the current block for this phi
                        for (int j = 0; j < phi.Count; j++)
                        {
                            if (phi.Sources[j] == branch.BasicBlock)
                            {
                                var phiLocal = GetLocal(phi);
                                EmitGetLocal(phi[j].Resolve());
                                WasmModuleBuilder.EmitLocalSet(Code, phiLocal);
                            }
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Computes the br depth to reach $loop from the code section of the current block.
        /// After block K's End has been emitted, remaining nesting above is:
        ///   (N-K-1) remaining blocks + $loop + $exit
        /// The label stack at Block K's code position:
        ///   depth 0: B[K+1], depth 1: B[K+2], ..., depth (N-K-2): B[N-1],
        ///   depth (N-K-1): $loop, depth (N-K): $exit
        /// So depth to $loop = N - K - 1 = _blockCount - _currentBlockEmitIndex - 1.
        /// </summary>
        private uint GetBrDepthToLoop(int extraNesting = 0)
        {
            return (uint)(_blockCount - _currentBlockEmitIndex - 1 + extraNesting);
        }

        /// <summary>
        /// Emits state transition: set _state = blockIndex, then br $loop.
        /// extraNesting accounts for additional nesting from if/else blocks.
        /// </summary>
        private void EmitBranchToBlock(int targetBlockIndex, int extraNesting = 0)
        {
            WasmModuleBuilder.EmitI32Const(Code, targetBlockIndex);
            WasmModuleBuilder.EmitLocalSet(Code, _stateLocal);
            Code.Add(WasmOpCodes.Br);
            WasmModuleBuilder.EmitU32Leb128(Code, GetBrDepthToLoop(extraNesting));
        }

        /// <summary>
        /// Generates the function body by visiting all blocks.
        /// For multi-block kernels, uses a state machine with loop/block/br_table.
        /// </summary>
        public override void GenerateCode()
        {
            // CRITICAL: Set up parameter mappings FIRST, before any IR visiting.
            // ILGPU calls GenerateCode() BEFORE GenerateHeader().
            SetupParameters();

            var blocks = Method.Blocks;
            _blockCount = blocks.Count;

            if (_blockCount == 1)
            {
                // Single block: no state machine needed
                _isStateMachine = false;
                var singleBlock = blocks.First();
                foreach (var value in singleBlock)
                    GenerateCodeFor(value);
                if (singleBlock.Terminator != null)
                    GenerateCodeFor(singleBlock.Terminator);
            }
            else
            {
                // Multi-block: state machine
                GenerateStateMachineCode(blocks);
            }

            // CRITICAL: Propagate BarrierCount, HasBarriers, and SharedMemorySize AFTER code generation.
            // _barrierCounter is incremented during IR visiting by GenerateCode(Barrier),
            // and _sharedMemorySize is incremented by GenerateCode(Broadcast) for broadcast slots,
            // so they must be read after all blocks have been visited.
            _generatorArgs.BarrierCount = _barrierCounter;
            _generatorArgs.HasBarriers = _hasBarriers;
            _generatorArgs.SharedMemorySize = _sharedMemorySize; // Re-propagate (broadcast may have added slots)

            WasmBackend.Log($"[Wasm-CodeGen] Final BarrierCount={_barrierCounter}, HasBarriers={_hasBarriers}, SharedMemorySize={_sharedMemorySize}");
        }

        /// <summary>
        /// Generates state machine code for multi-block kernels.
        /// 
        /// Wasm structured control flow pattern:
        ///   block $exit
        ///     loop $loop
        ///       block $blockN
        ///         block $blockN-1
        ///           ...
        ///           block $block0
        ///             local.get $state
        ///             br_table $block0 $block1 ... $blockN $exit
        ///           end ;; $block0
        ///           [code for block 0]
        ///           br $loop
        ///         end ;; $block1
        ///         [code for block 1]
        ///         br $loop
        ///       end ;; $blockN
        ///       [code for block N]
        ///       br $loop
        ///     end ;; $loop
        ///   end ;; $exit
        /// </summary>
        private void GenerateStateMachineCode(BasicBlockCollection<ReversePostOrder, Forwards> blocks)
        {
            _isStateMachine = true;

            // Allocate state local
            _stateLocal = AllocateNewLocal(WasmOpCodes.I32);

            // Build block map
            int blockIndex = 0;
            foreach (var block in blocks)
            {
                _blockMap[block] = blockIndex;
                blockIndex++;
            }

            WasmBackend.Log($"[Wasm-SM] State machine with {_blockCount} blocks, _stateLocal=local_{_stateLocal}");

            // Initialize state to 0 (first block)
            WasmModuleBuilder.EmitI32Const(Code, 0);
            WasmModuleBuilder.EmitLocalSet(Code, _stateLocal);

            // block $exit
            Code.Add(WasmOpCodes.Block);
            Code.Add(WasmOpCodes.Void);

            // loop $loop
            Code.Add(WasmOpCodes.Loop);
            Code.Add(WasmOpCodes.Void);

            // Nested blocks for dispatch: block $blockN { block $blockN-1 { ... block $block0 { ... } } }
            for (int i = 0; i < _blockCount; i++)
            {
                Code.Add(WasmOpCodes.Block);
                Code.Add(WasmOpCodes.Void);
            }

            // br_table dispatch: local.get $state; br_table 0 1 2 ... N-1 N
            WasmModuleBuilder.EmitLocalGet(Code, _stateLocal);
            Code.Add(WasmOpCodes.BrTable);
            WasmModuleBuilder.EmitU32Leb128(Code, (uint)_blockCount); // number of targets (not counting default)
            for (int i = 0; i < _blockCount; i++)
                WasmModuleBuilder.EmitU32Leb128(Code, (uint)i); // target labels (0..N-1 break out of corresponding block)
            WasmModuleBuilder.EmitU32Leb128(Code, (uint)(_blockCount + 1)); // default: br to $exit

            // Now emit code for each block
            blockIndex = 0;
            foreach (var block in blocks)
            {
                // End the innermost remaining block — this is where br_table lands for this index
                Code.Add(WasmOpCodes.End);

                _currentBlockEmitIndex = blockIndex;

                WasmBackend.Log($"[Wasm-SM] Block {blockIndex}: {block}");

                // Generate code for all values in this block
                foreach (var value in block)
                    GenerateCodeFor(value);

                // Handle terminator (branches/return)
                if (block.Terminator != null)
                    GenerateCodeFor(block.Terminator);

                blockIndex++;
            }

            // end $loop
            Code.Add(WasmOpCodes.End);

            // end $exit
            Code.Add(WasmOpCodes.End);
        }

        // === Control Flow Overrides ===

        public override void GenerateCode(ReturnTerminator returnTerminator)
        {
            if (_isStateMachine)
            {
                // Set state to an invalid value and br to $loop.
                // $loop will re-dispatch via br_table which hits the default target → $exit.
                WasmModuleBuilder.EmitI32Const(Code, _blockCount); // out-of-range state
                WasmModuleBuilder.EmitLocalSet(Code, _stateLocal);
                Code.Add(WasmOpCodes.Br);
                WasmModuleBuilder.EmitU32Leb128(Code, GetBrDepthToLoop());
            }
            else
            {
                Code.Add(WasmOpCodes.Return);
            }
        }

        public override void GenerateCode(UnconditionalBranch branch)
        {
            if (!_isStateMachine) return;

            PushPhiValues(branch);
            int targetBlock = GetBlockIndex(branch.Target);
            EmitBranchToBlock(targetBlock);
        }

        public override void GenerateCode(IfBranch branch)
        {
            if (!_isStateMachine) return;

            PushPhiValues(branch);

            int trueBlock = GetBlockIndex(branch.TrueTarget);
            int falseBlock = GetBlockIndex(branch.FalseTarget);

            EmitGetLocal(branch.Condition.Resolve());
            Code.Add(WasmOpCodes.If);
            Code.Add(WasmOpCodes.Void);
            // True branch — inside if{}, adds 1 nesting level
            EmitBranchToBlock(trueBlock, 1);
            Code.Add(WasmOpCodes.Else);
            // False branch — still inside if/else{}, same extra nesting
            EmitBranchToBlock(falseBlock, 1);
            Code.Add(WasmOpCodes.End);
        }

        public override void GenerateCode(SwitchBranch branch)
        {
            if (!_isStateMachine) return;

            PushPhiValues(branch);

            int defaultBlock = GetBlockIndex(branch.DefaultBlock);

            // For each case, use if chain — each if adds +1 nesting
            for (int i = 0; i < branch.NumCasesWithoutDefault; i++)
            {
                EmitGetLocal(branch.Condition.Resolve());
                WasmModuleBuilder.EmitI32Const(Code, i);
                Code.Add(WasmOpCodes.I32Eq);
                Code.Add(WasmOpCodes.If);
                Code.Add(WasmOpCodes.Void);
                int caseBlock = GetBlockIndex(branch.GetCaseTarget(i));
                EmitBranchToBlock(caseBlock, 1);
                Code.Add(WasmOpCodes.End);
            }

            // Default — no extra nesting
            EmitBranchToBlock(defaultBlock);
        }

        /// <summary>
        /// Merges the generated code into the StringBuilder (for debug output)
        /// and builds the Wasm module binary.
        /// </summary>
        public override void Merge(StringBuilder builder)
        {
            builder.AppendLine($"// Wasm kernel: {FuncParamTypes.Count} params, {_locals.Count} locals, {Code.Count} instruction bytes");
        }

        /// <summary>
        /// Gets the Wasm function body (locals + code) for this kernel.
        /// </summary>
        public WasmFuncBody GetFunctionBody()
        {
            return new WasmFuncBody
            {
                Locals = _locals,
                Code = Code.ToArray()
            };
        }

        /// <summary>
        /// Gets the function parameter types.
        /// </summary>
        public byte[] GetParamTypes()
        {
            return FuncParamTypes.ToArray();
        }

        #region Override: Memory Operations

        public override void GenerateCode(Load value)
        {
            // For struct types, the "value" is a memory address (i32 byte offset).
            // Don't actually load from memory — just pass through the address.
            // GetField will do the actual typed loads from memory at the correct offsets.
            if (value.Type is StructureType)
            {
                var target = AllocateLocal(value, WasmOpCodes.I32);
                var source = value.Source.Resolve();
                EmitGetLocal(source);
                WasmModuleBuilder.EmitLocalSet(Code, target);
                return;
            }

            var target2 = AllocateLocal(value, GetWasmType(value));
            var source2 = value.Source.Resolve();
            var wasmType = GetWasmType(value);

            // Push the address (byte offset in linear memory)
            EmitGetLocal(source2);

            // Emit the appropriate load instruction
            byte loadOp;
            uint align;
            switch (wasmType)
            {
                case WasmOpCodes.I64:
                    loadOp = WasmOpCodes.I64Load;
                    align = 3; // 2^3 = 8 byte alignment
                    break;
                case WasmOpCodes.F32:
                    loadOp = WasmOpCodes.F32Load;
                    align = 2; // 2^2 = 4 byte alignment
                    break;
                case WasmOpCodes.F64:
                    loadOp = WasmOpCodes.F64Load;
                    align = 3;
                    break;
                default: // I32
                    loadOp = WasmOpCodes.I32Load;
                    align = 2;
                    break;
            }

            WasmModuleBuilder.EmitLoad(Code, loadOp, align, 0);
            WasmModuleBuilder.EmitLocalSet(Code, target2);
        }


        public override void GenerateCode(LoadElementAddress value)
        {
            var target = AllocateLocal(value);
            var source = value.Source.Resolve();
            var index = value.Offset.Resolve();

            var sourceLocal = GetLocal(source);
            var indexLocal = GetLocal(index);
            WasmBackend.Log($"[Wasm-LEA] target=local_{target}, source=local_{sourceLocal} (IR={source.GetType().Name} id={source.Id} type={source.Type}), index=local_{indexLocal} (IR={index.GetType().Name} id={index.Id})");
            WasmBackend.Log($"[Wasm-LEA] _localMap dump: {string.Join(", ", _localMap.Select(kv => $"{kv.Key}={kv.Value}"))}");

            // Determine element size
            int elemSize = 4;
            if (value.Type is AddressSpaceType addrType)
            {
                if (addrType.ElementType is PrimitiveType pt)
                    elemSize = GetElementSize(pt.BasicValueType);
                else if (addrType.ElementType is StructureType st)
                    elemSize = st.Size;
            }

            // addr = source + index * elemSize
            // Source is an i32 memory address
            EmitGetLocal(source);
            // Wrap i64 source to i32 if needed (pointers are i32 in Wasm)
            if (GetWasmTypeFromIR(source.Type) == WasmOpCodes.I64)
                Code.Add(WasmOpCodes.I32WrapI64);

            // Index may be i64 (ILGPU uses Int64 for linearized 2D/3D array indices)
            EmitGetLocal(index);
            if (GetWasmTypeFromIR(index.Type) == WasmOpCodes.I64)
                Code.Add(WasmOpCodes.I32WrapI64);

            WasmBackend.Log($"[Wasm-LEA] value.Type={value.Type}, elemSize={elemSize}");
            WasmModuleBuilder.EmitI32Const(Code, elemSize);
            Code.Add(WasmOpCodes.I32Mul);
            Code.Add(WasmOpCodes.I32Add);

            WasmModuleBuilder.EmitLocalSet(Code, target);
        }

        #endregion

        #region Override: GetField for ArrayView decomposition

        public override void GenerateCode(GetField value)
        {
            var target = AllocateLocal(value, GetWasmType(value));
            var source = value.ObjectValue.Resolve();
            var fieldIndex = (int)value.FieldSpan.Index;

            // Check if source is the implicit index parameter for 2D/3D decomposition
            if (source is global::ILGPU.IR.Values.Parameter indexParam && indexParam == _indexParam)
            {
                switch (_indexType)
                {
                    case IndexType.Index2D:
                        if (fieldIndex == 0) // X = globalIdx % dimX
                        {
                            WasmModuleBuilder.EmitLocalGet(Code, _globalIdxLocal);
                            WasmModuleBuilder.EmitLocalGet(Code, _dimXLocal);
                            Code.Add(WasmOpCodes.I32RemS);
                        }
                        else // Y = globalIdx / dimX
                        {
                            WasmModuleBuilder.EmitLocalGet(Code, _globalIdxLocal);
                            WasmModuleBuilder.EmitLocalGet(Code, _dimXLocal);
                            Code.Add(WasmOpCodes.I32DivS);
                        }
                        WasmModuleBuilder.EmitLocalSet(Code, target);
                        return;

                    case IndexType.Index3D:
                        if (fieldIndex == 0) // X = globalIdx % dimX
                        {
                            WasmModuleBuilder.EmitLocalGet(Code, _globalIdxLocal);
                            WasmModuleBuilder.EmitLocalGet(Code, _dimXLocal);
                            Code.Add(WasmOpCodes.I32RemS);
                        }
                        else if (fieldIndex == 1) // Y = (globalIdx / dimX) % dimY
                        {
                            WasmModuleBuilder.EmitLocalGet(Code, _globalIdxLocal);
                            WasmModuleBuilder.EmitLocalGet(Code, _dimXLocal);
                            Code.Add(WasmOpCodes.I32DivS);
                            WasmModuleBuilder.EmitLocalGet(Code, _dimYLocal);
                            Code.Add(WasmOpCodes.I32RemS);
                        }
                        else // Z = globalIdx / (dimX * dimY)
                        {
                            WasmModuleBuilder.EmitLocalGet(Code, _globalIdxLocal);
                            WasmModuleBuilder.EmitLocalGet(Code, _dimXLocal);
                            WasmModuleBuilder.EmitLocalGet(Code, _dimYLocal);
                            Code.Add(WasmOpCodes.I32Mul);
                            Code.Add(WasmOpCodes.I32DivS);
                        }
                        WasmModuleBuilder.EmitLocalSet(Code, target);
                        return;

                    default: // Index1D - just use globalIdx directly
                        WasmModuleBuilder.EmitLocalGet(Code, _globalIdxLocal);
                        WasmModuleBuilder.EmitLocalSet(Code, target);
                        return;
                }
            }

            // Check if source is a Parameter that maps to an ArrayView
            if (source is global::ILGPU.IR.Values.Parameter param && IsViewType(param.Type))
            {
                int paramIdx = -1;
                for (int pi = 0; pi < Method.Parameters.Count; pi++)
                {
                    if (Method.Parameters[pi] == param) { paramIdx = pi; break; }
                }
                if (_paramLocals.TryGetValue(paramIdx, out var locals))
                {
                    var targetWasmType = GetWasmType(value);
                    WasmBackend.Log($"[Wasm-GetField] View param[{paramIdx}] field={fieldIndex} targetType={targetWasmType:X} localsCount={locals.Length} sourceType={param.Type}");

                    // Determine stride start field for this view parameter
                    int strideStartField = _viewStrideStartField.GetValueOrDefault(paramIdx, 3);
                    int strideFieldOffset = fieldIndex - strideStartField; // 0=stride1, 1=stride2, etc.

                    if (fieldIndex == 0)
                    {
                        // Ptr (byte offset)
                        WasmModuleBuilder.EmitLocalGet(Code, locals[0]);
                    }
                    else if (fieldIndex < strideStartField)
                    {
                        // Extent/index/length fields — these are i64 fields
                        // Field 1 = offset (always 0 for views)
                        // Field 2+ = extent dimensions or length
                        if (fieldIndex == 1)
                        {
                            // Index/Offset - always zero
                            if (targetWasmType == WasmOpCodes.I64)
                                WasmModuleBuilder.EmitI64Const(Code, 0);
                            else
                                WasmModuleBuilder.EmitI32Const(Code, 0);
                        }
                        else
                        {
                            // Length/extent fields - return length from locals[1]
                            WasmModuleBuilder.EmitLocalGet(Code, locals[1]);
                            if (targetWasmType == WasmOpCodes.I64)
                                Code.Add(WasmOpCodes.I64ExtendI32S);
                        }
                    }
                    else if (strideFieldOffset == 0)
                    {
                        // Stride1 (YStride)
                        if (locals.Length > 2)
                        {
                            WasmModuleBuilder.EmitLocalGet(Code, locals[2]);
                            if (targetWasmType == WasmOpCodes.I64)
                                Code.Add(WasmOpCodes.I64ExtendI32S);
                        }
                        else
                        {
                            if (targetWasmType == WasmOpCodes.I64)
                                WasmModuleBuilder.EmitI64Const(Code, 1);
                            else
                                WasmModuleBuilder.EmitI32Const(Code, 1);
                        }
                    }
                    else if (strideFieldOffset == 1)
                    {
                        // Stride2 (ZStride)
                        if (locals.Length > 3)
                        {
                            WasmModuleBuilder.EmitLocalGet(Code, locals[3]);
                            if (targetWasmType == WasmOpCodes.I64)
                                Code.Add(WasmOpCodes.I64ExtendI32S);
                        }
                        else
                        {
                            if (targetWasmType == WasmOpCodes.I64)
                                WasmModuleBuilder.EmitI64Const(Code, 0);
                            else
                                WasmModuleBuilder.EmitI32Const(Code, 0);
                        }
                    }
                    else
                    {
                        // Unknown/higher stride field
                        if (targetWasmType == WasmOpCodes.I64)
                            WasmModuleBuilder.EmitI64Const(Code, 0);
                        else
                            WasmModuleBuilder.EmitI32Const(Code, 0);
                    }
                    WasmModuleBuilder.EmitLocalSet(Code, target);
                    return;
                }
            }

            // Handle struct field access from memory
            // Source is a memory address (i32), and we need to load the specific field
            if (source.Type is StructureType structType)
            {
                var fieldAccess = new FieldAccess(fieldIndex);
                int byteOffset = structType.GetOffset(fieldAccess);
                var fieldType = structType.Fields[fieldIndex];
                byte fieldWasmType = GetWasmTypeFromIR(fieldType);

                // Push address: source + byteOffset
                EmitGetLocal(source);
                if (byteOffset > 0)
                {
                    WasmModuleBuilder.EmitI32Const(Code, byteOffset);
                    Code.Add(WasmOpCodes.I32Add);
                }

                // Emit typed load for the field
                byte loadOp;
                uint align;
                switch (fieldWasmType)
                {
                    case WasmOpCodes.I64:
                        loadOp = WasmOpCodes.I64Load;
                        align = 3;
                        break;
                    case WasmOpCodes.F32:
                        loadOp = WasmOpCodes.F32Load;
                        align = 2;
                        break;
                    case WasmOpCodes.F64:
                        loadOp = WasmOpCodes.F64Load;
                        align = 3;
                        break;
                    default:
                        loadOp = WasmOpCodes.I32Load;
                        align = 2;
                        break;
                }
                WasmModuleBuilder.EmitLoad(Code, loadOp, align, 0);
                WasmModuleBuilder.EmitLocalSet(Code, target);
                return;
            }

            // Fallback: pass through
            EmitGetLocal(source);
            WasmModuleBuilder.EmitLocalSet(Code, target);
        }

        public override void GenerateCode(SetField value)
        {
            // SetField modifies a field of a struct and returns the modified struct.
            // In our Wasm model, structs are memory addresses, so we write the field
            // value to memory at base + offset and return the base address.
            var objectValue = value.ObjectValue.Resolve();

            if (objectValue.Type is StructureType structType)
            {
                var target = AllocateLocal(value, WasmOpCodes.I32);
                int fieldIdx = (int)value.FieldSpan.Index;
                var fieldAccess = new FieldAccess(fieldIdx);
                int byteOffset = structType.GetOffset(fieldAccess);
                var fieldType = structType.Fields[fieldIdx];
                byte fieldWasmType = GetWasmTypeFromIR(fieldType);

                // Store the new field value to memory: mem[base + offset] = newValue
                EmitGetLocal(objectValue); // push base address
                if (byteOffset > 0)
                {
                    WasmModuleBuilder.EmitI32Const(Code, byteOffset);
                    Code.Add(WasmOpCodes.I32Add);
                }
                EmitGetLocal(value.Value.Resolve()); // push new value

                // Emit typed store
                byte storeOp;
                uint align;
                switch (fieldWasmType)
                {
                    case WasmOpCodes.I64:
                        storeOp = WasmOpCodes.I64Store;
                        align = 3;
                        break;
                    case WasmOpCodes.F32:
                        storeOp = WasmOpCodes.F32Store;
                        align = 2;
                        break;
                    case WasmOpCodes.F64:
                        storeOp = WasmOpCodes.F64Store;
                        align = 3;
                        break;
                    default:
                        storeOp = WasmOpCodes.I32Store;
                        align = 2;
                        break;
                }
                WasmModuleBuilder.EmitStore(Code, storeOp, align, 0);

                // Return the base address (unchanged — struct was modified in-place)
                EmitGetLocal(objectValue);
                WasmModuleBuilder.EmitLocalSet(Code, target);
                return;
            }

            // Non-struct fallback: pass through the value
            var fallbackTarget = AllocateLocal(value, GetWasmType(value));
            EmitGetLocal(value.Value.Resolve());
            WasmModuleBuilder.EmitLocalSet(Code, fallbackTarget);
        }

        public override void GenerateCode(Store value)
        {
            var target = value.Target.Resolve();
            var storeValue = value.Value.Resolve();

            // For struct types, we need to copy each field from source to destination
            if (storeValue.Type is StructureType structType)
            {
                int structSize = structType.Size;
                // Copy field by field
                for (int i = 0; i < structType.NumFields; i++)
                {
                    var fieldAccess = new FieldAccess(i);
                    int byteOffset = structType.GetOffset(fieldAccess);
                    var fieldType = structType.Fields[i];
                    byte fieldWasmType = GetWasmTypeFromIR(fieldType);

                    byte loadOp, storeOp;
                    uint align;
                    switch (fieldWasmType)
                    {
                        case WasmOpCodes.I64:
                            loadOp = WasmOpCodes.I64Load;
                            storeOp = WasmOpCodes.I64Store;
                            align = 3;
                            break;
                        case WasmOpCodes.F32:
                            loadOp = WasmOpCodes.F32Load;
                            storeOp = WasmOpCodes.F32Store;
                            align = 2;
                            break;
                        case WasmOpCodes.F64:
                            loadOp = WasmOpCodes.F64Load;
                            storeOp = WasmOpCodes.F64Store;
                            align = 3;
                            break;
                        default:
                            loadOp = WasmOpCodes.I32Load;
                            storeOp = WasmOpCodes.I32Store;
                            align = 2;
                            break;
                    }

                    // Push destination address + offset
                    EmitGetLocal(target);
                    if (byteOffset > 0)
                    {
                        WasmModuleBuilder.EmitI32Const(Code, byteOffset);
                        Code.Add(WasmOpCodes.I32Add);
                    }

                    // Load field value from source address + offset
                    EmitGetLocal(storeValue);
                    if (byteOffset > 0)
                    {
                        WasmModuleBuilder.EmitI32Const(Code, byteOffset);
                        Code.Add(WasmOpCodes.I32Add);
                    }
                    WasmModuleBuilder.EmitLoad(Code, loadOp, align, 0);

                    // Store to destination
                    WasmModuleBuilder.EmitStore(Code, storeOp, align, 0);
                }
                return;
            }

            // Non-struct types: standard typed store
            var wasmType = GetWasmTypeFromIR(storeValue.Type);

            // Push address, then value
            EmitGetLocal(target);
            EmitGetLocal(storeValue);

            byte sOp;
            uint sAlign;
            switch (wasmType)
            {
                case WasmOpCodes.I64:
                    sOp = WasmOpCodes.I64Store;
                    sAlign = 3;
                    break;
                case WasmOpCodes.F32:
                    sOp = WasmOpCodes.F32Store;
                    sAlign = 2;
                    break;
                case WasmOpCodes.F64:
                    sOp = WasmOpCodes.F64Store;
                    sAlign = 3;
                    break;
                default:
                    sOp = WasmOpCodes.I32Store;
                    sAlign = 2;
                    break;
            }

            WasmModuleBuilder.EmitStore(Code, sOp, sAlign, 0);
        }

        public override void GenerateCode(StructureValue value)
        {
            // StructureValue constructs a struct from field values.
            // Allocate space in scratch memory, store each field, return the base address.
            var target = AllocateLocal(value, WasmOpCodes.I32);

            if (value.Type is StructureType structType && value.Count > 0)
            {
                // Compute this struct's base address in scratch memory
                int baseOffset = _scratchNextOffset;
                _scratchNextOffset += structType.Size;
                // Align to 8 bytes for next allocation
                _scratchNextOffset = (_scratchNextOffset + 7) & ~7;

                // Compute base address = scratchBaseLocal + baseOffset
                WasmModuleBuilder.EmitLocalGet(Code, _scratchBaseLocal);
                if (baseOffset > 0)
                {
                    WasmModuleBuilder.EmitI32Const(Code, baseOffset);
                    Code.Add(WasmOpCodes.I32Add);
                }
                WasmModuleBuilder.EmitLocalSet(Code, target);

                // Store each field to memory at the correct offset
                for (int i = 0; i < value.Count && i < structType.NumFields; i++)
                {
                    var fieldAccess = new FieldAccess(i);
                    int fieldOffset = structType.GetOffset(fieldAccess);
                    var fieldType = structType.Fields[i];
                    byte fieldWasmType = GetWasmTypeFromIR(fieldType);

                    // Push destination address: target + fieldOffset
                    WasmModuleBuilder.EmitLocalGet(Code, target);
                    if (fieldOffset > 0)
                    {
                        WasmModuleBuilder.EmitI32Const(Code, fieldOffset);
                        Code.Add(WasmOpCodes.I32Add);
                    }

                    // Push field value
                    EmitGetLocal(value[i].Resolve());

                    // Emit typed store
                    byte storeOp;
                    uint align;
                    switch (fieldWasmType)
                    {
                        case WasmOpCodes.I64:
                            storeOp = WasmOpCodes.I64Store;
                            align = 3;
                            break;
                        case WasmOpCodes.F32:
                            storeOp = WasmOpCodes.F32Store;
                            align = 2;
                            break;
                        case WasmOpCodes.F64:
                            storeOp = WasmOpCodes.F64Store;
                            align = 3;
                            break;
                        default:
                            storeOp = WasmOpCodes.I32Store;
                            align = 2;
                            break;
                    }
                    WasmModuleBuilder.EmitStore(Code, storeOp, align, 0);
                }
            }
            else
            {
                // Non-struct or empty: pass through first field or zero
                if (value.Count > 0)
                    EmitGetLocal(value[0].Resolve());
                else
                    WasmModuleBuilder.EmitI32Const(Code, 0);
                WasmModuleBuilder.EmitLocalSet(Code, target);
            }
        }

        #endregion

        #region Override: MethodCall (Inline Helper Functions)

        /// <summary>
        /// Overrides MethodCall to inline helper functions (e.g., redirected algorithm intrinsics)
        /// directly into the kernel. The Wasm backend produces a single-function module, so
        /// helper function calls must be inlined rather than emitted as separate functions.
        /// 
        /// For multi-block helpers, generates a nested state machine with its own
        /// block map, state local, and dispatch loop.
        /// </summary>
        public override void GenerateCode(MethodCall methodCall)
        {
            var targetMethod = methodCall.Target;

            // Check if this is a helper function we can inline
            if (_generatorArgs.HelperMethods.TryGetValue(targetMethod, out var helperAllocas))
            {
                WasmBackend.Log($"[Wasm-Inline] Inlining helper: {targetMethod.Name} ({targetMethod.Parameters.Count} params, {methodCall.Nodes.Length} args)");

                // Step 1: Set up the helper's shared memory allocations
                SetupSharedAllocations(helperAllocas.SharedAllocations, isDynamic: false);
                SetupSharedAllocations(helperAllocas.DynamicSharedAllocations, isDynamic: true);

                // Step 2: Map call arguments to helper's parameters
                for (int i = 0; i < targetMethod.Parameters.Count && i < methodCall.Nodes.Length; i++)
                {
                    var param = targetMethod.Parameters[i];
                    var arg = methodCall.Nodes[i].Resolve();
                    var paramKey = GetValueKey(param);
                    
                    if (_localMap.TryGetValue(GetValueKey(arg), out uint argLocal))
                    {
                        _localMap[paramKey] = argLocal;
                        WasmBackend.Log($"[Wasm-Inline]   param[{i}] '{paramKey}' -> existing local_{argLocal}");
                    }
                    else
                    {
                        var paramWasmType = GetWasmType(param);
                        var local = AllocateLocal(param, paramWasmType);
                        EmitGetLocal(arg);
                        WasmModuleBuilder.EmitLocalSet(Code, local);
                        WasmBackend.Log($"[Wasm-Inline]   param[{i}] '{paramKey}' -> new local_{local} (copied)");
                    }
                }

                // Step 3: Allocate result local if non-void
                uint? resultLocal = null;
                if (!methodCall.Type.IsVoidType)
                {
                    resultLocal = AllocateLocal(methodCall, GetWasmType(methodCall));
                }

                // Step 4: Visit the helper's IR blocks
                var blocks = targetMethod.Blocks;
                var blockList = blocks.ToList();

                if (blockList.Count <= 1)
                {
                    // Single block: visit inline (no state machine needed)
                    if (blockList.Count == 1)
                    {
                        var block = blockList[0];
                        foreach (var value in block)
                            GenerateCodeFor(value);

                        // Handle return terminator
                        if (block.Terminator is ReturnTerminator ret && resultLocal.HasValue && !ret.IsVoidReturn)
                        {
                            EmitGetLocal(ret.ReturnValue.Resolve());
                            WasmModuleBuilder.EmitLocalSet(Code, resultLocal.Value);
                        }
                    }
                }
                else
                {
                    // Multi-block: nested state machine
                    // Save the kernel's state machine context
                    var savedBlockMap = new Dictionary<BasicBlock, int>(_blockMap);
                    var savedStateLocal = _stateLocal;
                    var savedBlockCount = _blockCount;
                    var savedIsStateMachine = _isStateMachine;
                    var savedCurrentBlockEmitIndex = _currentBlockEmitIndex;

                    // Set up a new state machine for the helper
                    _blockMap.Clear();
                    _isStateMachine = true;
                    _stateLocal = AllocateNewLocal(WasmOpCodes.I32);
                    int helperBlockCount = blockList.Count;
                    _blockCount = helperBlockCount;

                    int blockIndex = 0;
                    foreach (var block in blockList)
                    {
                        _blockMap[block] = blockIndex++;
                    }

                    WasmBackend.Log($"[Wasm-Inline] Nested state machine with {helperBlockCount} blocks, stateLocal=local_{_stateLocal}");

                    // Initialize state to 0
                    WasmModuleBuilder.EmitI32Const(Code, 0);
                    WasmModuleBuilder.EmitLocalSet(Code, _stateLocal);

                    // block $exit
                    Code.Add(WasmOpCodes.Block);
                    Code.Add(WasmOpCodes.Void);

                    // loop $loop
                    Code.Add(WasmOpCodes.Loop);
                    Code.Add(WasmOpCodes.Void);

                    // Nested dispatch blocks
                    for (int i = 0; i < helperBlockCount; i++)
                    {
                        Code.Add(WasmOpCodes.Block);
                        Code.Add(WasmOpCodes.Void);
                    }

                    // br_table dispatch
                    WasmModuleBuilder.EmitLocalGet(Code, _stateLocal);
                    Code.Add(WasmOpCodes.BrTable);
                    WasmModuleBuilder.EmitU32Leb128(Code, (uint)helperBlockCount);
                    for (int i = 0; i < helperBlockCount; i++)
                        WasmModuleBuilder.EmitU32Leb128(Code, (uint)i);
                    WasmModuleBuilder.EmitU32Leb128(Code, (uint)(helperBlockCount + 1)); // default -> exit

                    // Generate code for each block
                    int blockIdx = 0;
                    foreach (var block in blockList)
                    {
                        Code.Add(WasmOpCodes.End); // end of dispatch block

                        // CRITICAL: update _currentBlockEmitIndex so GetBrDepthToLoop() works
                        _currentBlockEmitIndex = blockIdx;

                        WasmBackend.Log($"[Wasm-Inline] Block {blockIdx}: {block.Id}");

                        // Visit all values
                        foreach (var value in block)
                            GenerateCodeFor(value);

                        // Handle terminator
                        if (block.Terminator is ReturnTerminator ret)
                        {
                            // Capture return value and break out of the state machine
                            if (resultLocal.HasValue && !ret.IsVoidReturn)
                            {
                                EmitGetLocal(ret.ReturnValue.Resolve());
                                WasmModuleBuilder.EmitLocalSet(Code, resultLocal.Value);
                            }
                            // br $exit: After closing dispatch block k, remaining nesting is:
                            //   $block(k+1)...$block(N-1) = N-k-1 scopes
                            //   + $loop = 1 scope
                            //   + $exit = target
                            // So depth to $exit = (N-k-1) + 1 + 0 = N-k
                            Code.Add(WasmOpCodes.Br);
                            WasmModuleBuilder.EmitU32Leb128(Code, (uint)(helperBlockCount - blockIdx)); // br to $exit
                        }
                        else if (block.Terminator != null)
                        {
                            // The terminator handlers (IfBranch, UnconditionalBranch)
                            // already emit EmitBranchToBlock which includes br $loop.
                            // Do NOT add an extra br here.
                            GenerateCodeFor(block.Terminator);
                        }

                        blockIdx++;
                    }

                    // end loop $loop
                    Code.Add(WasmOpCodes.End);
                    // end block $exit
                    Code.Add(WasmOpCodes.End);

                    // Restore the kernel's state machine context
                    _blockMap.Clear();
                    foreach (var kv in savedBlockMap)
                        _blockMap[kv.Key] = kv.Value;
                    _stateLocal = savedStateLocal;
                    _blockCount = savedBlockCount;
                    _isStateMachine = savedIsStateMachine;
                    _currentBlockEmitIndex = savedCurrentBlockEmitIndex;
                }

                WasmBackend.Log($"[Wasm-Inline] Done inlining {targetMethod.Name}");
                return;
            }

            // Fall through to base class for non-helper calls
            base.GenerateCode(methodCall);
        }

        /// <summary>
        /// Emits phi value assignments for a block (used during inlining).
        /// </summary>
        private void EmitPhiValues(BasicBlock block)
        {
            // Phi values are handled by PushPhiValues calls in branch terminators.
            // Nothing extra needed here.
        }

        #endregion

        #region Override: Device Constants

        public override void GenerateCode(GroupIndexValue value)
        {
            var target = AllocateLocal(value);
            // threadIdX = thread's index within the workgroup
            WasmModuleBuilder.EmitLocalGet(Code, _threadIdXLocal);
            WasmModuleBuilder.EmitLocalSet(Code, target);
        }

        public override void GenerateCode(GridIndexValue value)
        {
            var target = AllocateLocal(value);
            // gridIndex = globalIdx / groupDimX (which group this thread belongs to)
            WasmModuleBuilder.EmitLocalGet(Code, _globalIdxLocal);
            WasmModuleBuilder.EmitLocalGet(Code, _groupDimXLocal);
            Code.Add(WasmOpCodes.I32DivU);
            WasmModuleBuilder.EmitLocalSet(Code, target);
        }

        public override void GenerateCode(GroupDimensionValue value)
        {
            var target = AllocateLocal(value);
            // groupDimX = number of threads per workgroup
            WasmModuleBuilder.EmitLocalGet(Code, _groupDimXLocal);
            WasmModuleBuilder.EmitLocalSet(Code, target);
        }

        public override void GenerateCode(GridDimensionValue value)
        {
            var target = AllocateLocal(value);
            // gridDim = dimX / groupDimX (number of workgroups)
            WasmModuleBuilder.EmitLocalGet(Code, _dimXLocal);
            WasmModuleBuilder.EmitLocalGet(Code, _groupDimXLocal);
            Code.Add(WasmOpCodes.I32DivU);
            WasmModuleBuilder.EmitLocalSet(Code, target);
        }

        public override void GenerateCode(DynamicMemoryLengthValue value)
        {
            // Dynamic shared memory length = element count (passed at dispatch time via kernel param 8)
            var target = AllocateLocal(value);
            WasmModuleBuilder.EmitLocalGet(Code, _dynamicSharedLengthLocal);
            WasmModuleBuilder.EmitLocalSet(Code, target);
            WasmBackend.Log($"[Wasm-DynShared] DynamicMemoryLengthValue -> local_{target} = local_{_dynamicSharedLengthLocal}");
        }

        /// <summary>
        /// Handles ArrayView.Length: traces the view back to its kernel parameter
        /// and reads the pre-stored length local (locals[1]).
        /// This is the correct cross-backend approach — the length is already passed
        /// at dispatch time as a Wasm function parameter.
        /// </summary>
        public override void GenerateCode(GetViewLength value)
        {
            // GetViewLength returns i64 (long) in ILGPU, but our length local is i32.
            // We must extend i32 → i64 using I64ExtendI32S.
            var target = AllocateLocal(value, WasmOpCodes.I64);

            // Trace the view back to its kernel parameter
            var viewSource = value.View.Resolve();
            int paramIdx = -1;
            for (int pi = 0; pi < Method.Parameters.Count; pi++)
            {
                if (Method.Parameters[pi] == viewSource) { paramIdx = pi; break; }
            }

            if (paramIdx >= 0 && _paramLocals.TryGetValue(paramIdx, out var locals) && locals.Length > 1)
            {
                // locals[1] is the element count (i32) for this view parameter.
                // Extend to i64 since GetViewLength returns long.
                WasmModuleBuilder.EmitLocalGet(Code, locals[1]);
                Code.Add(WasmOpCodes.I64ExtendI32S);
                WasmBackend.Log($"[Wasm-GetViewLength] param[{paramIdx}] length -> local_{locals[1]} (i32 extended to i64)");
            }
            else
            {
                // Fallback: emit 0 as i64 (should not happen for well-formed kernels)
                WasmModuleBuilder.EmitI64Const(Code, 0);
                WasmBackend.Log($"[Wasm-GetViewLength] WARN: could not resolve view to parameter, emitting 0L");
            }

            WasmModuleBuilder.EmitLocalSet(Code, target);
        }

        #endregion

        #region Shared Memory

        /// <summary>
        /// Sets up shared memory allocations by computing byte offsets within the shared memory region.
        /// Each allocation gets a fixed offset from sharedMemBase that the Alloca handler will use.
        /// </summary>
        private void SetupSharedAllocations(AllocaKindInformation allocas, bool isDynamic)
        {
            foreach (var allocaInfo in allocas)
            {
                var key = GetValueKey(allocaInfo.Alloca);
                int elemSize = 4; // default i32

                // Determine element size from the alloca type
                if (allocaInfo.Alloca.Type is AddressSpaceType addrType)
                {
                    if (addrType.ElementType is PrimitiveType pt)
                        elemSize = GetElementSize(pt.BasicValueType);
                    else if (addrType.ElementType is StructureType st)
                        elemSize = st.Size;
                }

                if (isDynamic)
                {
                    // Dynamic shared memory: size determined at runtime
                    // Placed at the end of static shared allocations
                    _sharedAllocaOffsets[key] = _sharedMemorySize;
                    _dynamicSharedElementSize = elemSize;
                    _hasBarriers = true;
                    WasmBackend.Log($"[Wasm-SharedMem] Dynamic alloca {key}: offset={_sharedMemorySize}, elemSize={elemSize}");
                    // Don't advance _sharedMemorySize — dynamic size is not known at compile time
                }
                else if (allocaInfo.IsArray)
                {
                    int arrayBytes = allocaInfo.ArraySize * elemSize;
                    // Align to 4 bytes
                    arrayBytes = (arrayBytes + 3) & ~3;
                    _sharedAllocaOffsets[key] = _sharedMemorySize;
                    _sharedMemorySize += arrayBytes;
                    _hasBarriers = true;
                    WasmBackend.Log($"[Wasm-SharedMem] Static array alloca {key}: offset={_sharedMemorySize - arrayBytes}, size={arrayBytes}, arrayLen={allocaInfo.ArraySize}, elemSize={elemSize}");
                }
                else
                {
                    // Single scalar shared
                    int scalarBytes = (elemSize + 3) & ~3; // align to 4
                    _sharedAllocaOffsets[key] = _sharedMemorySize;
                    _sharedMemorySize += scalarBytes;
                    _hasBarriers = true;
                    WasmBackend.Log($"[Wasm-SharedMem] Scalar alloca {key}: offset={_sharedMemorySize - scalarBytes}, size={scalarBytes}");
                }
            }
        }

        /// <summary>
        /// Override Alloca for shared memory: emit address = sharedMemBase + offset
        /// </summary>
        public override void GenerateCode(Alloca value)
        {
            var key = GetValueKey(value);
            if (_sharedAllocaOffsets.TryGetValue(key, out int offset))
            {
                // Shared memory allocation: compute address in linear memory
                var target = AllocateLocal(value);
                WasmModuleBuilder.EmitLocalGet(Code, _sharedMemBaseLocal);
                if (offset > 0)
                {
                    WasmModuleBuilder.EmitI32Const(Code, offset);
                    Code.Add(WasmOpCodes.I32Add);
                }
                WasmModuleBuilder.EmitLocalSet(Code, target);
                WasmBackend.Log($"[Wasm-SharedMem] Alloca {key}: sharedMemBase + {offset}");
            }
            else
            {
                // Local alloca (not shared memory) — fall through to base
                base.GenerateCode(value);
            }
        }

        #endregion

        #region Barrier Synchronization

        /// <summary>
        /// Emits a sense-reversing barrier using Wasm atomic instructions.
        /// Each barrier uses 8 bytes at barrierBase + (barrierIdx * 8):
        ///   - offset +0: arrival counter (i32)
        ///   - offset +4: sense flag (i32)
        /// 
        /// Algorithm:
        ///   old = i32.atomic.rmw.add(barrierAddr, 1)
        ///   if (old + 1 == groupDimX):   // last thread
        ///     i32.atomic.store(barrierAddr, 0)   // reset counter
        ///     new_sense = 1 - i32.atomic.load(senseAddr)
        ///     i32.atomic.store(senseAddr, new_sense)
        ///     memory.atomic.notify(senseAddr, MAX)
        ///   else:
        ///     sense = i32.atomic.load(senseAddr)
        ///     memory.atomic.wait32(senseAddr, sense, -1)  // wait until sense flips
        /// </summary>
        public override void GenerateCode(global::ILGPU.IR.Values.Barrier barrier)
        {
            int barrierIdx = _barrierCounter++;
            _hasBarriers = true;
            EmitBarrier(barrierIdx);
        }

        /// <summary>
        /// Counter for broadcast shared memory slots.
        /// </summary>
        private int _broadcastCounter = 0;

        /// <summary>
        /// Override for Broadcast: emulates Group.Broadcast using shared memory + atomic barriers.
        /// Pattern:
        ///   1. Origin thread stores its value to a dedicated shared memory slot
        ///   2. Barrier (all threads sync)
        ///   3. All threads load from the shared slot
        ///   4. Barrier (prevent origin from overwriting before others finish reading)
        /// </summary>
        public override void GenerateCode(Broadcast broadcast)
        {
            var wasmType = GetWasmType(broadcast);
            var target = AllocateLocal(broadcast, wasmType);
            var source = broadcast.Variable.Resolve();
            var origin = broadcast.Origin.Resolve();

            // Allocate a slot in shared memory for this broadcast
            int broadcastSlotOffset = _sharedMemorySize;
            int slotSize = wasmType switch
            {
                WasmOpCodes.I64 => 8,
                WasmOpCodes.F64 => 8,
                _ => 4
            };
            _sharedMemorySize += (slotSize + 3) & ~3; // align to 4
            _hasBarriers = true;

            // Determine store/load ops and alignment for this type
            byte storeOp, loadOp;
            uint align;
            switch (wasmType)
            {
                case WasmOpCodes.I64:
                    storeOp = WasmOpCodes.I64Store;
                    loadOp = WasmOpCodes.I64Load;
                    align = 3;
                    break;
                case WasmOpCodes.F32:
                    storeOp = WasmOpCodes.F32Store;
                    loadOp = WasmOpCodes.F32Load;
                    align = 2;
                    break;
                case WasmOpCodes.F64:
                    storeOp = WasmOpCodes.F64Store;
                    loadOp = WasmOpCodes.F64Load;
                    align = 3;
                    break;
                default: // I32
                    storeOp = WasmOpCodes.I32Store;
                    loadOp = WasmOpCodes.I32Load;
                    align = 2;
                    break;
            }

            // Step 1: if (threadIdX == origin) { mem[sharedMemBase + offset] = source }
            WasmModuleBuilder.EmitLocalGet(Code, _threadIdXLocal);
            EmitGetLocal(origin);
            Code.Add(WasmOpCodes.I32Eq);
            Code.Add(WasmOpCodes.If);
            Code.Add(WasmOpCodes.Void);

            // Store: address = sharedMemBase + broadcastSlotOffset
            WasmModuleBuilder.EmitLocalGet(Code, _sharedMemBaseLocal);
            if (broadcastSlotOffset > 0)
            {
                WasmModuleBuilder.EmitI32Const(Code, broadcastSlotOffset);
                Code.Add(WasmOpCodes.I32Add);
            }
            EmitGetLocal(source);
            WasmModuleBuilder.EmitStore(Code, storeOp, align, 0);

            Code.Add(WasmOpCodes.End); // end if

            // Step 2: Barrier — ensure the origin's store is visible to all threads
            int barrier1 = _barrierCounter++;
            EmitBarrier(barrier1);

            // Step 3: All threads load from the shared slot
            WasmModuleBuilder.EmitLocalGet(Code, _sharedMemBaseLocal);
            if (broadcastSlotOffset > 0)
            {
                WasmModuleBuilder.EmitI32Const(Code, broadcastSlotOffset);
                Code.Add(WasmOpCodes.I32Add);
            }
            WasmModuleBuilder.EmitLoad(Code, loadOp, align, 0);
            WasmModuleBuilder.EmitLocalSet(Code, target);

            // Step 4: Barrier — prevent broadcast slot reuse before all threads have read
            int barrier2 = _barrierCounter++;
            EmitBarrier(barrier2);

            _broadcastCounter++;
            WasmBackend.Log($"[Wasm-Broadcast] Broadcast #{_broadcastCounter - 1}: slot offset={broadcastSlotOffset}, barriers={barrier1},{barrier2}");
        }

        public override void GenerateCode(MemoryBarrier barrier)
        {
            // MemoryBarrier (fence) — emit atomic.fence or just a barrier
            // For Wasm with shared memory, we use the same sense-reversing barrier
            // since all threads need to synchronize memory visibility
            // However, MemoryBarrier is typically just a fence, not a full barrier
            // For now, emit atomic.fence (0x03) 
            Code.Add(WasmOpCodes.AtomicPrefix);
            WasmModuleBuilder.EmitU32Leb128(Code, 0x03); // atomic.fence
            Code.Add(0x00); // memory index 0
        }

        /// <summary>
        /// Per-barrier local sense variables. Each barrier instruction gets a Wasm local
        /// that tracks this thread's expected sense value (0 or 1). This is essential for
        /// barriers inside loops — without per-thread sense tracking, a fast thread can
        /// re-enter the barrier before slow threads exit, causing stale sense reads.
        /// </summary>
        private readonly Dictionary<int, uint> _barrierSenseLocals = new();

        /// <summary>
        /// Emits a loop-safe sense-reversing barrier using Wasm atomic instructions.
        /// 
        /// Each barrier uses 8 bytes at barrierBase + (barrierIdx * 8):
        ///   - offset +0: arrival counter (i32)
        ///   - offset +4: sense flag (i32)
        /// 
        /// Each barrier also has a per-thread local variable (mySense) that starts at 0
        /// and flips after each use. This makes the barrier safe for re-entry in loops.
        /// 
        /// Algorithm:
        ///   atomic.fence                                    // flush prior stores
        ///   mySense = 1 - mySense                          // flip local sense
        ///   old = i32.atomic.rmw.add(arrivalAddr, 1)
        ///   if (old + 1 == groupDimX):                      // last thread
        ///     i32.atomic.store(arrivalAddr, 0)              // reset counter
        ///     i32.atomic.store(senseAddr, mySense)          // publish release sense
        ///     memory.atomic.notify(senseAddr, MAX)          // wake all waiters
        ///   else:
        ///     loop:                                         // spin-wait
        ///       cur = i32.atomic.load(senseAddr)
        ///       if (cur == mySense) break                   // released
        ///       memory.atomic.wait32(senseAddr, cur, -1)    // block until change
        ///   atomic.fence                                    // acquire: see others' stores
        /// </summary>
        private void EmitBarrier(int barrierIdx)
        {
            int byteOffset = barrierIdx * 8;

            // Get or create the per-thread local sense variable for this barrier.
            // The local persists across loop iterations, giving each thread its own
            // sense tracking for this specific barrier instruction.
            if (!_barrierSenseLocals.TryGetValue(barrierIdx, out uint mySenseLocal))
            {
                mySenseLocal = AllocateNewLocal(WasmOpCodes.I32);
                _barrierSenseLocals[barrierIdx] = mySenseLocal;
            }

            // Temp locals for addresses and arrival count
            var arrivalAddrLocal = AllocateNewLocal(WasmOpCodes.I32);
            var senseAddrLocal = AllocateNewLocal(WasmOpCodes.I32);
            var arrivedLocal = AllocateNewLocal(WasmOpCodes.I32);

            // === Step 0: Fence — flush all prior non-atomic stores ===
            // Regular i32.store to shared memory is NOT ordered across workers.
            // atomic.fence ensures that all stores made before the barrier
            // (e.g., shared[tid] = value) are visible in the SharedArrayBuffer
            // before we announce our arrival.
            Code.Add(WasmOpCodes.AtomicPrefix);
            WasmModuleBuilder.EmitU32Leb128(Code, 0x03); // atomic.fence
            Code.Add(0x00); // memory index 0

            // === Step 1: Flip local sense (0→1 or 1→0) ===
            // mySense = 1 - mySense
            WasmModuleBuilder.EmitI32Const(Code, 1);
            WasmModuleBuilder.EmitLocalGet(Code, mySenseLocal);
            Code.Add(WasmOpCodes.I32Sub);
            WasmModuleBuilder.EmitLocalSet(Code, mySenseLocal);

            // === Step 2: Compute barrier addresses ===
            // arrivalAddr = barrierBase + byteOffset
            WasmModuleBuilder.EmitLocalGet(Code, _barrierBaseLocal);
            WasmModuleBuilder.EmitI32Const(Code, byteOffset);
            Code.Add(WasmOpCodes.I32Add);
            WasmModuleBuilder.EmitLocalSet(Code, arrivalAddrLocal);

            // senseAddr = barrierBase + byteOffset + 4
            WasmModuleBuilder.EmitLocalGet(Code, _barrierBaseLocal);
            WasmModuleBuilder.EmitI32Const(Code, byteOffset + 4);
            Code.Add(WasmOpCodes.I32Add);
            WasmModuleBuilder.EmitLocalSet(Code, senseAddrLocal);

            // === Step 3: Atomically increment arrival counter ===
            // arrived = i32.atomic.rmw.add(arrivalAddr, 1) + 1
            WasmModuleBuilder.EmitLocalGet(Code, arrivalAddrLocal);
            WasmModuleBuilder.EmitI32Const(Code, 1);
            Code.Add(WasmOpCodes.AtomicPrefix);
            WasmModuleBuilder.EmitU32Leb128(Code, WasmOpCodes.I32AtomicRmwAdd);
            Code.Add(0x02); // alignment = 4 bytes (2^2)
            Code.Add(0x00); // offset = 0
            WasmModuleBuilder.EmitI32Const(Code, 1);
            Code.Add(WasmOpCodes.I32Add);
            WasmModuleBuilder.EmitLocalSet(Code, arrivedLocal);

            // === Step 4: Branch on last thread ===
            // if (arrived == groupDimX)
            WasmModuleBuilder.EmitLocalGet(Code, arrivedLocal);
            WasmModuleBuilder.EmitLocalGet(Code, _groupDimXLocal);
            Code.Add(WasmOpCodes.I32Eq);

            Code.Add(WasmOpCodes.If);
            Code.Add(WasmOpCodes.Void);

            // === LAST THREAD: reset counter, publish sense, notify ===

            // i32.atomic.store(arrivalAddr, 0) — reset counter
            WasmModuleBuilder.EmitLocalGet(Code, arrivalAddrLocal);
            WasmModuleBuilder.EmitI32Const(Code, 0);
            Code.Add(WasmOpCodes.AtomicPrefix);
            WasmModuleBuilder.EmitU32Leb128(Code, WasmOpCodes.I32AtomicStore);
            Code.Add(0x02);
            Code.Add(0x00);

            // i32.atomic.store(senseAddr, mySense) — publish the target sense
            WasmModuleBuilder.EmitLocalGet(Code, senseAddrLocal);
            WasmModuleBuilder.EmitLocalGet(Code, mySenseLocal);
            Code.Add(WasmOpCodes.AtomicPrefix);
            WasmModuleBuilder.EmitU32Leb128(Code, WasmOpCodes.I32AtomicStore);
            Code.Add(0x02);
            Code.Add(0x00);

            // memory.atomic.notify(senseAddr, MAX_WAITERS)
            WasmModuleBuilder.EmitLocalGet(Code, senseAddrLocal);
            WasmModuleBuilder.EmitI32Const(Code, int.MaxValue);
            Code.Add(WasmOpCodes.AtomicPrefix);
            WasmModuleBuilder.EmitU32Leb128(Code, WasmOpCodes.MemoryAtomicNotify);
            Code.Add(0x02);
            Code.Add(0x00);
            Code.Add(WasmOpCodes.Drop);

            Code.Add(WasmOpCodes.Else);

            // === NOT LAST: spin-wait until global sense matches local sense ===
            // This is a spin-wait loop: we must keep checking because wait32 may
            // return spuriously or with not-equal if another iteration's release
            // changed the sense before we entered the wait.

            // block $exit
            Code.Add(WasmOpCodes.Block);
            Code.Add(WasmOpCodes.Void);

            // loop $spin
            Code.Add(WasmOpCodes.Loop);
            Code.Add(WasmOpCodes.Void);

            // cur = i32.atomic.load(senseAddr)
            var curSenseLocal = AllocateNewLocal(WasmOpCodes.I32);
            WasmModuleBuilder.EmitLocalGet(Code, senseAddrLocal);
            Code.Add(WasmOpCodes.AtomicPrefix);
            WasmModuleBuilder.EmitU32Leb128(Code, WasmOpCodes.I32AtomicLoad);
            Code.Add(0x02);
            Code.Add(0x00);
            WasmModuleBuilder.EmitLocalSet(Code, curSenseLocal);

            // if (cur == mySense) break out of spin loop
            WasmModuleBuilder.EmitLocalGet(Code, curSenseLocal);
            WasmModuleBuilder.EmitLocalGet(Code, mySenseLocal);
            Code.Add(WasmOpCodes.I32Eq);
            Code.Add(WasmOpCodes.BrIf);
            WasmModuleBuilder.EmitU32Leb128(Code, 1); // br_if $exit (depth 1 from loop)

            // memory.atomic.wait32(senseAddr, cur, -1) — block until sense changes
            WasmModuleBuilder.EmitLocalGet(Code, senseAddrLocal);
            WasmModuleBuilder.EmitLocalGet(Code, curSenseLocal);
            WasmModuleBuilder.EmitI64Const(Code, -1);
            Code.Add(WasmOpCodes.AtomicPrefix);
            WasmModuleBuilder.EmitU32Leb128(Code, WasmOpCodes.MemoryAtomicWait32);
            Code.Add(0x02);
            Code.Add(0x00);
            Code.Add(WasmOpCodes.Drop);

            // br $spin — go back and re-check
            Code.Add(WasmOpCodes.Br);
            WasmModuleBuilder.EmitU32Leb128(Code, 0); // br $spin (depth 0 = innermost loop)

            Code.Add(WasmOpCodes.End); // end loop $spin
            Code.Add(WasmOpCodes.End); // end block $exit

            Code.Add(WasmOpCodes.End); // end if/else

            // === Step 5: Fence — acquire: ensure loads after the barrier
            // see all stores made by other threads before the barrier.
            Code.Add(WasmOpCodes.AtomicPrefix);
            WasmModuleBuilder.EmitU32Leb128(Code, 0x03); // atomic.fence
            Code.Add(0x00); // memory index 0

            WasmBackend.Log($"[Wasm-Barrier] Emitted loop-safe barrier #{barrierIdx} at byteOffset={byteOffset}");
        }

        #endregion

        #region Helpers

        /// <summary>
        /// Checks if an IR type is an ArrayView type.
        /// </summary>
        protected bool IsViewType(TypeNode type)
        {
            if (type is StructureType structType)
            {
                var typeName = structType.ToString();
                if (typeName.Contains("ArrayView") || typeName.Contains("View"))
                    return true;
            }
            if (type is AddressSpaceType)
                return true;
            return false;
        }

        /// <summary>
        /// Gets the element type of an ArrayView.
        /// </summary>
        protected TypeNode? GetViewElementType(TypeNode type)
        {
            if (type is AddressSpaceType addrType)
                return addrType.ElementType;
            if (type is StructureType structType && structType.NumFields > 0)
            {
                var firstField = structType.Fields[0];
                if (firstField is AddressSpaceType addr)
                    return addr.ElementType;
            }
            return null;
        }

        #endregion
    }
}
