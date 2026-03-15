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
        /// Stores (elementType string, arraySize) metadata for each shared alloca entry,
        /// keyed by the same key as _sharedAllocaOffsets. Used for fallback matching
        /// when the Alloca IR node encountered during code generation is a different
        /// instance from the one registered during SetupSharedAllocations().
        /// </summary>
        private readonly Dictionary<string, (string ElemType, int ArraySize)> _sharedAllocaMetadata = new();

        /// <summary>
        /// Element size in bytes for dynamic shared memory allocations.
        /// </summary>
        private int _dynamicSharedElementSize = 0;

        // === Helper function mode ===

        /// <summary>
        /// Whether this generator is producing a helper function (not the kernel entry point).
        /// Helper functions have different parameter setup and return value handling.
        /// </summary>
        private bool _isHelperFunction = false;

        /// <summary>
        /// Wasm local index for the helper's return value (non-void helpers only).
        /// </summary>
        private uint _helperResultLocal;

        /// <summary>
        /// Wasm type of the helper's return value, or null for void helpers.
        /// </summary>
        private byte? _helperResultType;

        // === Exposed state for helper generation ===

        /// <summary>
        /// Exposes shared memory alloca offsets for use by helper function generators.
        /// </summary>
        internal IReadOnlyDictionary<string, int> SharedAllocaOffsets => _sharedAllocaOffsets;

        /// <summary>
        /// Exposes shared memory alloca metadata for use by helper function generators.
        /// </summary>
        internal IReadOnlyDictionary<string, (string ElemType, int ArraySize)> SharedAllocaMetadata => _sharedAllocaMetadata;

        /// <summary>
        /// Exposes the current shared memory size for use by helper function generators.
        /// </summary>
        internal int SharedMemorySizeValue => _sharedMemorySize;

        public WasmKernelFunctionGenerator(
            in GeneratorArgs args,
            Method method,
            Allocas allocas)
            : base(args, method, allocas)
        {
        }

        /// <summary>
        /// Result of generating a helper function body.
        /// </summary>
        internal class HelperFunctionResult
        {
            public byte[] ParamTypes { get; set; } = Array.Empty<byte>();
            public byte[] ResultTypes { get; set; } = Array.Empty<byte>();
            public List<WasmLocal> Locals { get; set; } = new();
            public byte[] Code { get; set; } = Array.Empty<byte>();
            public int BarrierCount { get; set; }
            public int SharedMemorySize { get; set; }
        }

        /// <summary>
        /// Generates this method as a standalone helper function (not the kernel entry point).
        /// Called from WasmBackend.CreateKernel() for multi-block helpers.
        ///
        /// The helper function receives the same fixed context params as the kernel
        /// (globalIdx, dimX, dimY, scratchBase, groupDimX, threadIdX, sharedMemBase,
        /// barrierBase, dynamicSharedLength), followed by the helper's IR parameters.
        /// Returns a value via the Wasm function return mechanism.
        /// </summary>
        internal HelperFunctionResult GenerateAsHelper(
            IReadOnlyDictionary<string, int> sharedAllocaOffsets,
            IReadOnlyDictionary<string, (string ElemType, int ArraySize)> sharedAllocaMetadata,
            int sharedMemorySize,
            Dictionary<string, uint> mathImports)
        {
            _isHelperFunction = true;
            MathImports = mathImports;

            // Copy shared memory allocation state from the kernel generator.
            // The kernel has already computed offsets for all static shared allocas
            // (including this helper's). The helper starts from this base and may
            // add more allocations (e.g., Broadcast slots) during code generation.
            foreach (var kv in sharedAllocaOffsets)
                _sharedAllocaOffsets[kv.Key] = kv.Value;
            foreach (var kv in sharedAllocaMetadata)
                _sharedAllocaMetadata[kv.Key] = kv.Value;
            _sharedMemorySize = sharedMemorySize;
            if (sharedMemorySize > 0)
                _hasBarriers = true;

            // Set up helper-specific parameters
            SetupHelperParameters();

            // Generate code (state machine for multi-block)
            var blocks = Method.Blocks;
            _blockCount = blocks.Count;

            if (_blockCount <= 1)
            {
                _isStateMachine = false;
                if (_blockCount == 1)
                {
                    var singleBlock = blocks.First();
                    foreach (var value in singleBlock)
                        GenerateCodeFor(value);
                    if (singleBlock.Terminator != null)
                        GenerateCodeFor(singleBlock.Terminator);
                }
            }
            else
            {
                GenerateStateMachineCode(blocks);
            }

            // After the state machine, push the return value onto the stack
            // (Wasm returns the value on the stack when the function ends)
            if (_helperResultType.HasValue)
            {
                WasmModuleBuilder.EmitLocalGet(Code, _helperResultLocal);
            }

            // Determine result types
            byte[] resultTypes = _helperResultType.HasValue
                ? new[] { _helperResultType.Value }
                : Array.Empty<byte>();

            return new HelperFunctionResult
            {
                ParamTypes = FuncParamTypes.ToArray(),
                ResultTypes = resultTypes,
                Locals = _locals,
                Code = Code.ToArray(),
                BarrierCount = _barrierCounter,
                SharedMemorySize = _sharedMemorySize,
            };
        }

        /// <summary>
        /// Sets up parameter-to-local mappings for a helper function.
        /// Fixed context params first (same as kernel), then the helper's IR params.
        /// </summary>
        private void SetupHelperParameters()
        {
            _locals.Clear();
            _localMap.Clear();
            _nextLocalIndex = 0;
            _paramCount = 0;

            // Fixed context params (same order as kernel — 9 params)
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

            FuncParamTypes.Clear();
            FuncParamTypes.Add(WasmOpCodes.I32); // globalIdx
            FuncParamTypes.Add(WasmOpCodes.I32); // dimX
            FuncParamTypes.Add(WasmOpCodes.I32); // dimY
            FuncParamTypes.Add(WasmOpCodes.I32); // scratchBase
            FuncParamTypes.Add(WasmOpCodes.I32); // groupDimX
            FuncParamTypes.Add(WasmOpCodes.I32); // threadIdX
            FuncParamTypes.Add(WasmOpCodes.I32); // sharedMemBase
            FuncParamTypes.Add(WasmOpCodes.I32); // barrierBase
            FuncParamTypes.Add(WasmOpCodes.I32); // dynamicSharedLength

            // Helper's IR parameters
            var parameters = Method.Parameters;
            for (int i = 0; i < parameters.Count; i++)
            {
                var param = parameters[i];
                var wasmType = GetWasmTypeFromIR(param.Type);
                FuncParamTypes.Add(wasmType);
                uint paramLocal = _nextLocalIndex++;
                _paramCount++;
                _localMap[GetValueKey(param)] = paramLocal;

                WasmBackend.Log($"[Wasm-Helper] param[{i}] '{GetValueKey(param)}' -> local_{paramLocal} (type={wasmType:X2})");
            }

            // Determine return type from the method's return type
            var returnType = Method.ReturnType;
            if (returnType != null && !returnType.IsVoidType)
            {
                _helperResultType = GetWasmTypeFromIR(returnType);
                _helperResultLocal = AllocateNewLocal(_helperResultType.Value);
                WasmBackend.Log($"[Wasm-Helper] Return type: {_helperResultType:X2}, resultLocal=local_{_helperResultLocal}");
            }

            // Re-register shared memory allocations. The offsets were pre-computed by the
            // kernel generator and copied in GenerateAsHelper(). SetupSharedAllocations
            // will skip keys that already exist (guard in that method), ensuring no
            // double-counting. However, if the alloca IR values here have different keys
            // from the kernel's copy (different instances), this re-registration ensures
            // they're properly mapped.
            SetupSharedAllocations(Allocas.SharedAllocations, isDynamic: false);
            SetupSharedAllocations(Allocas.DynamicSharedAllocations, isDynamic: true);

            WasmBackend.Log($"[Wasm-Helper] Setup complete: {_nextLocalIndex} locals, {FuncParamTypes.Count} params, sharedMem={_sharedMemorySize}");
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

                // Check if the implicit index is Int64 (LongIndex1D).
                // globalIdx is i32, but the IR parameter might be i64.
                // Create an i64 local that extends globalIdx for i64 params.
                if (GetWasmTypeFromIR(parameters[0].Type) == WasmOpCodes.I64)
                {
                    // Int64 implicit index (LongIndex1D) = paddedNumDataElements.
                    // This is NOT the thread index — it's the loop bound / extent.
                    // Don't map to _globalIdxLocal. Instead, let it be a regular user param
                    // by NOT mapping it and NOT incrementing startIdx.
                    _indexParam = null; // not an index param
                    startIdx = 0; // all params are user params
                    WasmBackend.Log($"[Wasm-Setup] Int64 param is extent, not index — treating as user param");
                }
                else
                {
                    _localMap[GetValueKey(_indexParam)] = _globalIdxLocal;
                    WasmBackend.Log($"[Wasm-Setup] Index param {GetValueKey(_indexParam)} -> local_{_globalIdxLocal}, IndexType={_indexType}");
                }
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

                    // Record struct layout info for dispatch-side serialization
                    List<StructFieldInfo> structFields = null;
                    if (paramType is StructureType sType)
                    {
                        structFields = new List<StructFieldInfo>();
                        WasmBackend.Log($"[Wasm-Setup]   Struct layout: fields={sType.NumFields}, size={sType.Size}, alignment={sType.Alignment}");
                        FlattenStructLayout(sType, 0, structFields);
                        foreach (var sf in structFields)
                            WasmBackend.Log($"[Wasm-Setup]     leaf: offset={sf.Offset}, wasmType=0x{sf.WasmType:X2}, size={sf.Size}, isViewPtr={sf.IsViewPtr}");
                    }

                    _paramInfos.Add(new WasmParamInfo
                    {
                        Index = i,
                        Name = $"param{i}",
                        IsScalar = true,
                        WasmType = wasmType,
                        StructFields = structFields,
                        StructSize = (paramType is StructureType st) ? st.Size : 0,
                        IRTypeName = paramType?.GetType().Name + ":" + paramType?.ToString(),
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

            // (i64 fixup removed — Int64 params are now handled as user params)

            WasmBackend.Log($"[Wasm-Setup] Final: _nextLocalIndex={_nextLocalIndex}, _paramCount={_paramCount}, FuncParamTypes={FuncParamTypes.Count}");
            WasmBackend.Log($"[Wasm-Setup] _localMap: {string.Join(", ", _localMap.Select(kv => $"{kv.Key}={kv.Value}"))}");
            WasmBackend.Log($"[Wasm-Setup] SharedMemorySize={_sharedMemorySize}, HasBarriers={_hasBarriers}");
        }

        private bool _parametersInitialized = false;
        private bool _needsI64IndexFixup = false;

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
                                var srcValue = phi[j].Resolve();
                                EmitGetLocal(srcValue);

                                // Coerce type if source and PHI local differ
                                var srcType = GetWasmTypeFromIR(srcValue.Type);
                                var phiType = GetLocalType(phiLocal);
                                if (srcType == WasmOpCodes.I64 && phiType == WasmOpCodes.I32)
                                    Code.Add(WasmOpCodes.I32WrapI64);
                                else if (srcType == WasmOpCodes.I32 && phiType == WasmOpCodes.I64)
                                    Code.Add(WasmOpCodes.I64ExtendI32S);

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
        /// Assigns Wasm function indices to multi-block helper methods.
        /// Called at the start of kernel GenerateCode(), after all helpers have been
        /// registered in data.HelperMethods by CreateFunctionCodeGenerator().
        /// </summary>
        private void AssignHelperFunctionIndices()
        {
            if (_generatorArgs.HelperFunctionIndices.Count > 0)
                return; // Already assigned (e.g., called twice)

            // Log helper info
            foreach (var kvp in _generatorArgs.HelperMethods)
                WasmBackend.Log($"Wasm: Helper '{kvp.Key.Name}' blocks={kvp.Key.Blocks.Count}");

            if (_generatorArgs.HelperMethods.Count == 0)
                return; // No helpers

            int importCount = WasmBackend.UnaryMathFuncs.Length + WasmBackend.BinaryMathFuncs.Length;
            int nextHelperFuncIdx = importCount + 1; // +1 for the kernel function

            foreach (var kvp in _generatorArgs.HelperMethods)
            {
                int barrierCount = ComputeEffectiveBarrierCount(kvp.Key);
                // Promote to separate Wasm function if:
                // - Multi-block (needs its own state machine, can't nest), OR
                // - Contains barriers directly or in sub-helpers (sense locals must be
                //   isolated per call to prevent stale sense values when called
                //   multiple times in a loop)
                if (kvp.Key.Blocks.Count > 1 || barrierCount > 0)
                {
                    _generatorArgs.HelperFunctionIndices[kvp.Key] = nextHelperFuncIdx++;
                    _generatorArgs.HelperBarrierCounts[kvp.Key] = barrierCount;
                    _generatorArgs.HelperFunctionOrder.Add(kvp.Key);
                    string reason = kvp.Key.Blocks.Count > 1 ? "multi-block" : "has-barriers";
                    WasmBackend.Log($"Wasm: Helper '{kvp.Key.Name}' promoted ({reason}): funcIdx={_generatorArgs.HelperFunctionIndices[kvp.Key]}, barriers={barrierCount}, blocks={kvp.Key.Blocks.Count}");
                }
            }
        }

        /// <summary>
        /// Computes the effective barrier count for a method, recursively counting
        /// barriers from sub-helper MethodCalls. This is necessary because a helper's
        /// barriers may not be direct IR Barrier values — they may be in sub-methods
        /// that are called via MethodCall within the helper's blocks.
        ///
        /// Each Barrier value counts as 1, each Broadcast as 2 (store barrier + load barrier).
        /// Each MethodCall to a known helper adds that helper's effective barrier count.
        /// </summary>
        private int ComputeEffectiveBarrierCount(Method method, HashSet<Method>? visited = null)
        {
            visited ??= new HashSet<Method>();
            if (!visited.Add(method))
                return 0; // Prevent infinite recursion for cyclic calls

            int count = 0;
            foreach (var block in method.Blocks)
            {
                foreach (var entry in block)
                {
                    var value = entry.Value;
                    if (value is global::ILGPU.IR.Values.Barrier)
                        count++;
                    else if (value is global::ILGPU.IR.Values.Broadcast)
                        count += 2;
                    else if (value is global::ILGPU.IR.Values.MethodCall call)
                    {
                        // Recursively count barriers in sub-helpers
                        if (_generatorArgs.HelperMethods.ContainsKey(call.Target))
                            count += ComputeEffectiveBarrierCount(call.Target, visited);
                    }
                }
            }
            return count;
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

            // Assign function indices for multi-block helpers. This must happen here
            // (not in CreateKernelCodeGenerator) because ILGPU calls CreateKernelCodeGenerator
            // BEFORE CreateFunctionCodeGenerator, so data.HelperMethods is empty at that point.
            // By the time GenerateCode() runs, all helpers have been registered.
            AssignHelperFunctionIndices();

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
            _generatorArgs.SharedMemorySize = _sharedMemorySize;
            // Record scratch bytes used per thread. For barrier kernels, each worker needs
            // its own scratch region to avoid races on StructureValue/Alloca memory.
            _generatorArgs.ScratchPerThread = (_scratchNextOffset + 7) & ~7; // 8-byte aligned

            // NOTE: Barrier sense flag reset for multi-group dispatch is handled in the
            // JavaScript worker loop (BuildWasmWorkerScript) to avoid generating
            // unreachable code after the Wasm return instruction.

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

            // Log block map for diagnostics (visible in AllKernelInfos via WasmBackend.Log)
            foreach (var kvp in _blockMap)
            {
                var termName = kvp.Key.Terminator?.GetType().Name ?? "none";
                WasmBackend.Log($"[Wasm-SM]   Block {kvp.Value}: {kvp.Key} terminator={termName}");
            }

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
            // Helper functions: store return value before exiting
            if (_isHelperFunction && _helperResultType.HasValue && !returnTerminator.IsVoidReturn)
            {
                EmitGetLocal(returnTerminator.ReturnValue.Resolve());
                WasmModuleBuilder.EmitLocalSet(Code, _helperResultLocal);
            }

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
            WasmBackend.Log($"[Wasm-SM] IfBranch: true→{trueBlock} false→{falseBlock} (blockCount={_blockCount})");

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

            // For barrier kernels, use atomic loads for cross-worker visibility
            if (_hasBarriers && (wasmType == WasmOpCodes.I32 || wasmType == WasmOpCodes.I64))
            {
                byte atomicLoadOp = wasmType == WasmOpCodes.I64
                    ? WasmOpCodes.I64AtomicLoad : WasmOpCodes.I32AtomicLoad;
                uint atomicAlign = wasmType == WasmOpCodes.I64 ? 3u : 2u;
                WasmModuleBuilder.EmitAtomicRmw(Code, atomicLoadOp, atomicAlign, 0);
            }
            else
            {
                byte loadOp;
                uint align;
                switch (wasmType)
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
            }
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

            // Check if source is (or traces to) a Parameter that maps to an ArrayView.
            // For ArrayView1D params, the source might be a GetField(param, field)
            // rather than the Parameter directly. Use TraceToParameter to resolve.
            int resolvedParamIdx = -1;
            if (source is global::ILGPU.IR.Values.Parameter directParam && IsViewType(directParam.Type))
            {
                for (int pi = 0; pi < Method.Parameters.Count; pi++)
                    if (Method.Parameters[pi] == directParam) { resolvedParamIdx = pi; break; }
            }
            else
            {
                // Trace through GetField/NewView/etc. to find the underlying view param
                resolvedParamIdx = TraceToParameter(source);
                // Only use if the resolved param is actually a view
                if (resolvedParamIdx >= 0 && resolvedParamIdx < Method.Parameters.Count
                    && !IsViewType(Method.Parameters[resolvedParamIdx].Type))
                    resolvedParamIdx = -1;
            }

            if (resolvedParamIdx >= 0)
            {
                var param = Method.Parameters[resolvedParamIdx];
                int paramIdx = resolvedParamIdx;
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
                        // Context-sensitive field handling:
                        // - ArrayView1D (StructureType): fields = [ptr, Extent(i64), Stride...]
                        //   Field 1 = Extent = Length → return locals[1]
                        // - ArrayView (AddressSpaceType): fields = [ptr, Index(i64), Length(i64)]
                        //   Field 1 = Index/Offset → always 0 for whole-buffer views
                        //   Field 2 = Length → return locals[1]
                        bool isStructView = param.Type is StructureType;
                        if (fieldIndex == 1 && !isStructView)
                        {
                            // ArrayView's Index/Offset field — always 0
                            if (targetWasmType == WasmOpCodes.I64)
                                WasmModuleBuilder.EmitI64Const(Code, 0);
                            else
                                WasmModuleBuilder.EmitI32Const(Code, 0);
                        }
                        else
                        {
                            // Length/Extent — return element count from locals[1]
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

                    // Load field value from source address + offset (atomic for barrier kernels)
                    EmitGetLocal(storeValue);
                    if (byteOffset > 0)
                    {
                        WasmModuleBuilder.EmitI32Const(Code, byteOffset);
                        Code.Add(WasmOpCodes.I32Add);
                    }
                    if (_hasBarriers && (fieldWasmType == WasmOpCodes.I32 || fieldWasmType == WasmOpCodes.I64))
                    {
                        byte atomicLoadOp = fieldWasmType == WasmOpCodes.I64
                            ? WasmOpCodes.I64AtomicLoad : WasmOpCodes.I32AtomicLoad;
                        WasmModuleBuilder.EmitAtomicRmw(Code, atomicLoadOp, align, 0);
                    }
                    else
                        WasmModuleBuilder.EmitLoad(Code, loadOp, align, 0);

                    // Store to destination (atomic for barrier kernels)
                    if (_hasBarriers && (fieldWasmType == WasmOpCodes.I32 || fieldWasmType == WasmOpCodes.I64))
                        WasmModuleBuilder.EmitAtomicRmw(Code,
                            fieldWasmType == WasmOpCodes.I64 ? WasmOpCodes.I64AtomicStore : WasmOpCodes.I32AtomicStore,
                            align, 0);
                    else
                        WasmModuleBuilder.EmitStore(Code, storeOp, align, 0);
                }
                return;
            }

            // Non-struct types: typed store.
            // For barrier kernels (multi-worker dispatch), use ATOMIC stores to ensure
            // cross-worker visibility on SharedArrayBuffer. Non-atomic i32.store writes
            // are not guaranteed to be visible to other workers after atomic.fence.
            var wasmType = GetWasmTypeFromIR(storeValue.Type);

            EmitGetLocal(target);
            EmitGetLocal(storeValue);

            if (_hasBarriers && (wasmType == WasmOpCodes.I32 || wasmType == WasmOpCodes.I64))
            {
                // Atomic store for integer types in barrier kernels
                byte atomicStoreOp = wasmType == WasmOpCodes.I64
                    ? WasmOpCodes.I64AtomicStore : WasmOpCodes.I32AtomicStore;
                uint atomicAlign = wasmType == WasmOpCodes.I64 ? 3u : 2u;
                WasmModuleBuilder.EmitAtomicRmw(Code, atomicStoreOp, atomicAlign, 0);
            }
            else
            {
                // Non-barrier kernels or float types: regular store
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

        #region Override: MethodCall (Helper Functions)

        /// <summary>
        /// Overrides MethodCall to handle helper functions:
        /// - Single-block helpers are inlined directly (fast, no state machine needed).
        /// - Multi-block helpers are emitted as separate Wasm functions and called via
        ///   the 'call' instruction. This avoids nested state machines which corrupt
        ///   control flow when both the kernel and helper have multiple blocks.
        /// </summary>
        public override void GenerateCode(MethodCall methodCall)
        {
            var targetMethod = methodCall.Target;

            // Check if this is a helper function
            if (!_generatorArgs.HelperMethods.TryGetValue(targetMethod, out var helperAllocas))
            {
                // Not a helper — fall through to base class
                base.GenerateCode(methodCall);
                return;
            }

            // Set up the helper's shared memory allocations (needed for both paths
            // so that the kernel's _sharedMemorySize accounts for the helper's allocas)
            SetupSharedAllocations(helperAllocas.SharedAllocations, isDynamic: false);
            SetupSharedAllocations(helperAllocas.DynamicSharedAllocations, isDynamic: true);

            // Allocate result local if non-void
            uint? resultLocal = null;
            if (!methodCall.Type.IsVoidType)
            {
                resultLocal = AllocateLocal(methodCall, GetWasmType(methodCall));
            }

            // Check if this is a multi-block helper with a pre-assigned function index
            if (_generatorArgs.HelperFunctionIndices.TryGetValue(targetMethod, out int helperFuncIdx))
            {
                // Multi-block helper: emit function call
                int helperBarrierCount = _generatorArgs.HelperBarrierCounts[targetMethod];

                WasmBackend.Log($"[Wasm-Call] Calling helper '{targetMethod.Name}' funcIdx={helperFuncIdx}, barriers={helperBarrierCount}, barrierOffset={_barrierCounter}");

                // No barrier reset needed — generation-counting barriers require no
                // per-call reset. The generation counter monotonically increases, so
                // fresh function locals always read the correct current generation.

                // Push context params (same order as the helper's fixed params)
                WasmModuleBuilder.EmitLocalGet(Code, _globalIdxLocal);
                WasmModuleBuilder.EmitLocalGet(Code, _dimXLocal);
                WasmModuleBuilder.EmitLocalGet(Code, _dimYLocal);
                WasmModuleBuilder.EmitLocalGet(Code, _scratchBaseLocal);
                WasmModuleBuilder.EmitLocalGet(Code, _groupDimXLocal);
                WasmModuleBuilder.EmitLocalGet(Code, _threadIdXLocal);
                WasmModuleBuilder.EmitLocalGet(Code, _sharedMemBaseLocal);

                // Adjusted barrier base: offset by current barrier counter
                WasmModuleBuilder.EmitLocalGet(Code, _barrierBaseLocal);
                if (_barrierCounter > 0)
                {
                    WasmModuleBuilder.EmitI32Const(Code, _barrierCounter * 8);
                    Code.Add(WasmOpCodes.I32Add);
                }

                WasmModuleBuilder.EmitLocalGet(Code, _dynamicSharedLengthLocal);

                // Push the helper's IR arguments
                for (int i = 0; i < targetMethod.Parameters.Count && i < methodCall.Nodes.Length; i++)
                {
                    EmitGetLocal(methodCall.Nodes[i].Resolve());
                }

                // Emit call instruction
                WasmModuleBuilder.EmitCall(Code, (uint)helperFuncIdx);

                // Pop result if non-void
                if (resultLocal.HasValue)
                {
                    WasmModuleBuilder.EmitLocalSet(Code, resultLocal.Value);
                }

                // Advance barrier counter for subsequent barriers/calls
                _barrierCounter += helperBarrierCount;
                if (helperBarrierCount > 0)
                {
                    _hasBarriers = true;

                    // CRITICAL: Emit an extra barrier AFTER each helper call that uses barriers.
                    // The helper's internal barriers ensure all workers reach the end of the
                    // helper, but workers resume at different speeds after the last barrier.
                    // Without this post-call barrier, a fast worker can start the NEXT helper
                    // call while a slow worker is still inside the previous one. Since the
                    // helper uses shared memory at fixed offsets, this causes a data race.
                    // (See Wasm/CLAUDE.md "Post-helper barrier" rule.)
                    EmitBarrier(_barrierCounter);
                    _barrierCounter++;
                }

                WasmBackend.Log($"[Wasm-Call] Done calling '{targetMethod.Name}', barrierCounter now={_barrierCounter}");
            }
            else
            {
                // Inline path: single-block or multi-block (nested SM fallback)
                WasmBackend.Log($"[Wasm-Inline] Inlining helper: {targetMethod.Name} ({targetMethod.Parameters.Count} params, {methodCall.Nodes.Length} args)");

                // Map call arguments to helper's parameters
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

                        if (block.Terminator is ReturnTerminator ret && resultLocal.HasValue && !ret.IsVoidReturn)
                        {
                            EmitGetLocal(ret.ReturnValue.Resolve());
                            WasmModuleBuilder.EmitLocalSet(Code, resultLocal.Value);
                        }
                    }
                }
                else
                {
                    // Multi-block: nested state machine (fallback for when function call is unavailable)
                    var savedBlockMap = new Dictionary<BasicBlock, int>(_blockMap);
                    var savedStateLocal = _stateLocal;
                    var savedBlockCount = _blockCount;
                    var savedIsStateMachine = _isStateMachine;
                    var savedCurrentBlockEmitIndex = _currentBlockEmitIndex;

                    _blockMap.Clear();
                    _isStateMachine = true;
                    _stateLocal = AllocateNewLocal(WasmOpCodes.I32);
                    int helperBlockCount = blockList.Count;
                    _blockCount = helperBlockCount;

                    int blockIndex = 0;
                    foreach (var block in blockList)
                        _blockMap[block] = blockIndex++;

                    WasmModuleBuilder.EmitI32Const(Code, 0);
                    WasmModuleBuilder.EmitLocalSet(Code, _stateLocal);

                    Code.Add(WasmOpCodes.Block); Code.Add(WasmOpCodes.Void);
                    Code.Add(WasmOpCodes.Loop); Code.Add(WasmOpCodes.Void);
                    for (int i = 0; i < helperBlockCount; i++)
                    {
                        Code.Add(WasmOpCodes.Block); Code.Add(WasmOpCodes.Void);
                    }

                    WasmModuleBuilder.EmitLocalGet(Code, _stateLocal);
                    Code.Add(WasmOpCodes.BrTable);
                    WasmModuleBuilder.EmitU32Leb128(Code, (uint)helperBlockCount);
                    for (int i = 0; i < helperBlockCount; i++)
                        WasmModuleBuilder.EmitU32Leb128(Code, (uint)i);
                    WasmModuleBuilder.EmitU32Leb128(Code, (uint)(helperBlockCount + 1));

                    int blockIdx = 0;
                    foreach (var block in blockList)
                    {
                        Code.Add(WasmOpCodes.End);
                        _currentBlockEmitIndex = blockIdx;

                        foreach (var value in block)
                            GenerateCodeFor(value);

                        if (block.Terminator is ReturnTerminator ret)
                        {
                            if (resultLocal.HasValue && !ret.IsVoidReturn)
                            {
                                EmitGetLocal(ret.ReturnValue.Resolve());
                                WasmModuleBuilder.EmitLocalSet(Code, resultLocal.Value);
                            }
                            Code.Add(WasmOpCodes.Br);
                            WasmModuleBuilder.EmitU32Leb128(Code, (uint)(helperBlockCount - blockIdx));
                        }
                        else if (block.Terminator != null)
                        {
                            GenerateCodeFor(block.Terminator);
                        }

                        blockIdx++;
                    }

                    Code.Add(WasmOpCodes.End);
                    Code.Add(WasmOpCodes.End);

                    _blockMap.Clear();
                    foreach (var kv in savedBlockMap)
                        _blockMap[kv.Key] = kv.Value;
                    _stateLocal = savedStateLocal;
                    _blockCount = savedBlockCount;
                    _isStateMachine = savedIsStateMachine;
                    _currentBlockEmitIndex = savedCurrentBlockEmitIndex;
                }

                WasmBackend.Log($"[Wasm-Inline] Done inlining {targetMethod.Name}");
            }
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

            // Trace the view back to its kernel parameter.
            // For ArrayView<T> params, value.View.Resolve() IS the Parameter directly.
            // For ArrayView1D<T, TStride> params, the view goes through GetField
            // (to extract BaseView from the struct) — we must trace through the
            // chain of operations to find the original Parameter.
            var viewSource = value.View.Resolve();
            int paramIdx = TraceToParameter(viewSource);

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
                WasmBackend.Log($"[Wasm-GetViewLength] WARN: could not resolve view to parameter (source={viewSource?.GetType().Name}), emitting 0L");
            }

            WasmModuleBuilder.EmitLocalSet(Code, target);
        }

        /// <summary>
        /// Traces an IR value back through GetField, NewView, AddressSpaceCast, etc.
        /// to find the underlying kernel Parameter. Returns the parameter index, or -1.
        /// This is needed because ArrayView1D params wrap ArrayView via GetField(BaseView),
        /// so the view doesn't directly resolve to the Parameter.
        /// </summary>
        private int TraceToParameter(Value source)
        {
            var current = source;
            int depth = 0;
            while (current != null && depth < 20)
            {
                if (current is global::ILGPU.IR.Values.Parameter)
                {
                    for (int pi = 0; pi < Method.Parameters.Count; pi++)
                    {
                        if (Method.Parameters[pi] == current) return pi;
                    }
                    return -1;
                }

                if (current is GetField gf)
                    current = gf.ObjectValue.Resolve();
                else if (current is NewView nv)
                    current = nv.Pointer.Resolve();
                else if (current is AddressSpaceCast asc)
                    current = asc.Value.Resolve();
                else if (current is PointerCast pc)
                    current = pc.Value.Resolve();
                else
                    break;

                depth++;
            }
            return -1;
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
                string elemTypeStr = "i32";

                // Determine element size from the alloca type
                if (allocaInfo.Alloca.Type is AddressSpaceType addrType)
                {
                    elemTypeStr = addrType.ElementType.ToString();
                    if (addrType.ElementType is PrimitiveType pt)
                        elemSize = GetElementSize(pt.BasicValueType);
                    else if (addrType.ElementType is StructureType st)
                        elemSize = st.Size;
                }

                int arraySize = allocaInfo.IsArray ? allocaInfo.ArraySize : 1;

                if (isDynamic)
                {
                    // Dynamic shared memory: size determined at runtime
                    // Placed at the end of static shared allocations
                    _sharedAllocaOffsets[key] = _sharedMemorySize;
                    _sharedAllocaMetadata[key] = (elemTypeStr, arraySize);
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
                    _sharedAllocaMetadata[key] = (elemTypeStr, arraySize);
                    _sharedMemorySize += arrayBytes;
                    _hasBarriers = true;
                    WasmBackend.Log($"[Wasm-SharedMem] Static array alloca {key}: offset={_sharedMemorySize - arrayBytes}, size={arrayBytes}, arrayLen={allocaInfo.ArraySize}, elemSize={elemSize}");
                }
                else
                {
                    // Single scalar shared
                    int scalarBytes = (elemSize + 3) & ~3; // align to 4
                    _sharedAllocaOffsets[key] = _sharedMemorySize;
                    _sharedAllocaMetadata[key] = (elemTypeStr, arraySize);
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
            if (!_sharedAllocaOffsets.TryGetValue(key, out int offset))
            {
                // Primary key miss — try fallback matching by element type + array size.
                // The Alloca IR node from GenerateCode may be a different instance
                // from the one registered during SetupSharedAllocations().
                if (value.AddressSpace == MemoryAddressSpace.Shared)
                {
                    string allocaElemType = value.AllocaType.ToString();
                    long allocaArrayLen = value.ArrayLength.Resolve() is global::ILGPU.IR.Values.PrimitiveValue pv
                        ? pv.Int64Value : -1;

                    string? matchedKey = null;
                    foreach (var kvp in _sharedAllocaMetadata)
                    {
                        if (kvp.Value.ElemType == allocaElemType && kvp.Value.ArraySize == allocaArrayLen)
                        {
                            matchedKey = kvp.Key;
                            break;
                        }
                    }

                    if (matchedKey != null && _sharedAllocaOffsets.TryGetValue(matchedKey, out offset))
                    {
                        WasmBackend.Log($"[Wasm-SharedMem] Alloca {key}: fallback match to {matchedKey} (type={allocaElemType}, size={allocaArrayLen})");
                    }
                    else
                    {
                        // No match found — emit base alloca as a safety fallback
                        WasmBackend.Log($"[Wasm-SharedMem] WARNING: Alloca {key} (type={allocaElemType}, size={allocaArrayLen}) has no matching shared entry");
                        base.GenerateCode(value);
                        return;
                    }
                }
                else
                {
                    // Not shared memory — local alloca.
                    // Allocate in scratch memory (NOT address 0, which would corrupt
                    // the data buffer region — see Wasm/CLAUDE.md RADIX RULE).
                    int allocSize = 8; // default
                    if (value.Type is AddressSpaceType localAddrType)
                    {
                        if (localAddrType.ElementType is PrimitiveType localPt)
                            allocSize = GetElementSize(localPt.BasicValueType);
                        else if (localAddrType.ElementType is StructureType localSt)
                            allocSize = localSt.Size;
                    }
                    if (value.ArrayLength.Resolve() is global::ILGPU.IR.Values.PrimitiveValue localPv)
                        allocSize *= localPv.Int32Value;

                    int baseOff = _scratchNextOffset;
                    _scratchNextOffset += allocSize;
                    _scratchNextOffset = (_scratchNextOffset + 7) & ~7;

                    var localTarget = AllocateLocal(value, WasmOpCodes.I32);
                    WasmModuleBuilder.EmitLocalGet(Code, _scratchBaseLocal);
                    if (baseOff > 0)
                    {
                        WasmModuleBuilder.EmitI32Const(Code, baseOff);
                        Code.Add(WasmOpCodes.I32Add);
                    }
                    WasmModuleBuilder.EmitLocalSet(Code, localTarget);
                    WasmBackend.Log($"[Wasm-Alloca] Local alloca: key={key}, size={allocSize}, scratchOffset={baseOff}");
                    return;
                }
            }

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
        /// Emits a generation-counting barrier using Wasm atomic instructions.
        ///
        /// Each barrier uses 8 bytes at barrierBase + (barrierIdx * 8):
        ///   - offset +0: arrival counter (i32)
        ///   - offset +4: generation counter (i32)
        ///
        /// No per-thread local state is needed. The generation counter monotonically
        /// increases. Each invocation reads the current generation, waits for it to
        /// increment. This is safe for:
        ///   - Loop re-entry (generation keeps increasing)
        ///   - Cross-function calls (no stale per-thread sense locals)
        ///   - Multi-group dispatch (no reset needed between groups)
        ///
        /// Algorithm:
        ///   atomic.fence                                    // flush prior stores
        ///   myGen = i32.atomic.load(genAddr)                // read current gen
        ///   old = i32.atomic.rmw.add(arrivalAddr, 1)
        ///   if (old + 1 == groupDimX):                      // last thread
        ///     i32.atomic.store(arrivalAddr, 0)              // reset counter
        ///     i32.atomic.rmw.add(genAddr, 1)                // bump generation
        ///     memory.atomic.notify(genAddr, MAX)            // wake all waiters
        ///   else:
        ///     loop:                                         // spin-wait
        ///       curGen = i32.atomic.load(genAddr)
        ///       if (curGen != myGen) break                  // generation advanced
        ///       memory.atomic.wait32(genAddr, curGen, -1)   // block until change
        ///   atomic.fence                                    // acquire: see others' stores
        /// </summary>
        private void EmitBarrier(int barrierIdx)
        {
            int byteOffset = barrierIdx * 8;

            var arrivalAddrLocal = AllocateNewLocal(WasmOpCodes.I32);
            var genAddrLocal = AllocateNewLocal(WasmOpCodes.I32);
            var myGenLocal = AllocateNewLocal(WasmOpCodes.I32);
            var arrivedLocal = AllocateNewLocal(WasmOpCodes.I32);

            // === Step 0: Fence — flush all prior non-atomic stores ===
            Code.Add(WasmOpCodes.AtomicPrefix);
            WasmModuleBuilder.EmitU32Leb128(Code, 0x03); // atomic.fence
            Code.Add(0x00);

            // === Step 1: Compute barrier addresses ===
            // arrivalAddr = barrierBase + byteOffset
            WasmModuleBuilder.EmitLocalGet(Code, _barrierBaseLocal);
            WasmModuleBuilder.EmitI32Const(Code, byteOffset);
            Code.Add(WasmOpCodes.I32Add);
            WasmModuleBuilder.EmitLocalSet(Code, arrivalAddrLocal);

            // genAddr = barrierBase + byteOffset + 4
            WasmModuleBuilder.EmitLocalGet(Code, _barrierBaseLocal);
            WasmModuleBuilder.EmitI32Const(Code, byteOffset + 4);
            Code.Add(WasmOpCodes.I32Add);
            WasmModuleBuilder.EmitLocalSet(Code, genAddrLocal);

            // === Step 2: Read current generation ===
            // myGen = i32.atomic.load(genAddr)
            WasmModuleBuilder.EmitLocalGet(Code, genAddrLocal);
            Code.Add(WasmOpCodes.AtomicPrefix);
            WasmModuleBuilder.EmitU32Leb128(Code, WasmOpCodes.I32AtomicLoad);
            Code.Add(0x02);
            Code.Add(0x00);
            WasmModuleBuilder.EmitLocalSet(Code, myGenLocal);

            // === Step 3: Atomically increment arrival counter ===
            // arrived = i32.atomic.rmw.add(arrivalAddr, 1) + 1
            WasmModuleBuilder.EmitLocalGet(Code, arrivalAddrLocal);
            WasmModuleBuilder.EmitI32Const(Code, 1);
            Code.Add(WasmOpCodes.AtomicPrefix);
            WasmModuleBuilder.EmitU32Leb128(Code, WasmOpCodes.I32AtomicRmwAdd);
            Code.Add(0x02);
            Code.Add(0x00);
            WasmModuleBuilder.EmitI32Const(Code, 1);
            Code.Add(WasmOpCodes.I32Add);
            WasmModuleBuilder.EmitLocalSet(Code, arrivedLocal);

            // === Step 4: Branch on last thread ===
            WasmModuleBuilder.EmitLocalGet(Code, arrivedLocal);
            WasmModuleBuilder.EmitLocalGet(Code, _groupDimXLocal);
            Code.Add(WasmOpCodes.I32Eq);

            Code.Add(WasmOpCodes.If);
            Code.Add(WasmOpCodes.Void);

            // === LAST THREAD: reset counter, bump generation, notify ===

            // i32.atomic.store(arrivalAddr, 0)
            WasmModuleBuilder.EmitLocalGet(Code, arrivalAddrLocal);
            WasmModuleBuilder.EmitI32Const(Code, 0);
            Code.Add(WasmOpCodes.AtomicPrefix);
            WasmModuleBuilder.EmitU32Leb128(Code, WasmOpCodes.I32AtomicStore);
            Code.Add(0x02);
            Code.Add(0x00);

            // i32.atomic.rmw.add(genAddr, 1) — bump generation
            WasmModuleBuilder.EmitLocalGet(Code, genAddrLocal);
            WasmModuleBuilder.EmitI32Const(Code, 1);
            Code.Add(WasmOpCodes.AtomicPrefix);
            WasmModuleBuilder.EmitU32Leb128(Code, WasmOpCodes.I32AtomicRmwAdd);
            Code.Add(0x02);
            Code.Add(0x00);
            Code.Add(WasmOpCodes.Drop); // discard old value

            // memory.atomic.notify(genAddr, MAX_WAITERS)
            WasmModuleBuilder.EmitLocalGet(Code, genAddrLocal);
            WasmModuleBuilder.EmitI32Const(Code, int.MaxValue);
            Code.Add(WasmOpCodes.AtomicPrefix);
            WasmModuleBuilder.EmitU32Leb128(Code, WasmOpCodes.MemoryAtomicNotify);
            Code.Add(0x02);
            Code.Add(0x00);
            Code.Add(WasmOpCodes.Drop);

            Code.Add(WasmOpCodes.Else);

            // === NOT LAST: spin-wait until generation advances ===
            Code.Add(WasmOpCodes.Block);
            Code.Add(WasmOpCodes.Void);
            Code.Add(WasmOpCodes.Loop);
            Code.Add(WasmOpCodes.Void);

            // curGen = i32.atomic.load(genAddr)
            var curGenLocal = AllocateNewLocal(WasmOpCodes.I32);
            WasmModuleBuilder.EmitLocalGet(Code, genAddrLocal);
            Code.Add(WasmOpCodes.AtomicPrefix);
            WasmModuleBuilder.EmitU32Leb128(Code, WasmOpCodes.I32AtomicLoad);
            Code.Add(0x02);
            Code.Add(0x00);
            WasmModuleBuilder.EmitLocalSet(Code, curGenLocal);

            // if (curGen != myGen) break — generation advanced
            WasmModuleBuilder.EmitLocalGet(Code, curGenLocal);
            WasmModuleBuilder.EmitLocalGet(Code, myGenLocal);
            Code.Add(WasmOpCodes.I32Ne);
            Code.Add(WasmOpCodes.BrIf);
            WasmModuleBuilder.EmitU32Leb128(Code, 1); // br_if $exit

            // memory.atomic.wait32(genAddr, curGen, -1)
            WasmModuleBuilder.EmitLocalGet(Code, genAddrLocal);
            WasmModuleBuilder.EmitLocalGet(Code, curGenLocal);
            WasmModuleBuilder.EmitI64Const(Code, -1);
            Code.Add(WasmOpCodes.AtomicPrefix);
            WasmModuleBuilder.EmitU32Leb128(Code, WasmOpCodes.MemoryAtomicWait32);
            Code.Add(0x02);
            Code.Add(0x00);
            Code.Add(WasmOpCodes.Drop);

            // br $spin
            Code.Add(WasmOpCodes.Br);
            WasmModuleBuilder.EmitU32Leb128(Code, 0);

            Code.Add(WasmOpCodes.End); // end loop $spin
            Code.Add(WasmOpCodes.End); // end block $exit
            Code.Add(WasmOpCodes.End); // end if/else

            // === Step 5: Fence — acquire ===
            Code.Add(WasmOpCodes.AtomicPrefix);
            WasmModuleBuilder.EmitU32Leb128(Code, 0x03); // atomic.fence
            Code.Add(0x00);

            WasmBackend.Log($"[Wasm-Barrier] Emitted generation barrier #{barrierIdx} at byteOffset={byteOffset}");
        }

        #endregion

        #region Helpers

        /// <summary>
        /// Checks if an IR type is an ArrayView type.
        /// </summary>
        /// <summary>
        /// Recursively flattens an IR StructureType into its leaf fields.
        /// ILGPU's StructureType already stores flattened fields (no nested structs),
        /// so we can enumerate them directly.
        /// </summary>
        private void FlattenStructLayout(StructureType sType, int baseOffset, List<StructFieldInfo> result)
        {
            for (int i = 0; i < sType.NumFields; i++)
            {
                var fieldType = sType[i]; // Already a leaf type (flattened by ILGPU)
                var fieldOffset = sType.GetOffset(new FieldAccess(i));
                var wasmType = GetWasmTypeFromIR(fieldType);

                result.Add(new StructFieldInfo
                {
                    Offset = baseOffset + fieldOffset,
                    WasmType = wasmType,
                    Size = fieldType.Size,
                    IsViewPtr = fieldType is AddressSpaceType,
                });
            }
        }

        protected bool IsViewType(TypeNode type)
        {
            if (type is AddressSpaceType)
                return true;
            if (type is StructureType structType)
            {
                // A view StructureType (ArrayView1D<T, TStride>) has AddressSpaceType as
                // its first DirectField. This is the pointer field from ArrayView<T>.
                //
                // A struct-with-embedded-view (InitializerImplementation<T, TStride>) has
                // a StructureType as its first DirectField (the ArrayView1D wrapper),
                // NOT a direct AddressSpaceType.
                //
                // This distinction works because ILGPU represents ArrayView<T> as
                // AddressSpaceType (a pointer), while ArrayView1D wraps it by having
                // AddressSpaceType as its first direct field plus extent + stride.
                // User structs containing views have the view as a nested struct field.
                if (structType.DirectFields.Length > 0
                    && structType.DirectFields[0] is AddressSpaceType)
                    return true;
            }
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
