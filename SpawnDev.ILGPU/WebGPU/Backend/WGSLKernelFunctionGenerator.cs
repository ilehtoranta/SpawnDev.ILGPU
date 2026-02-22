// ---------------------------------------------------------------------------------------
//                                  SpawnDev.ILGPU.WebGPU
//                         Copyright (c) 2024 SpawnDev Project
//
// File: WGSLKernelFunctionGenerator.cs
// Force Update
// ---------------------------------------------------------------------------------------

using global::ILGPU;
using global::ILGPU.Backends.EntryPoints;
using global::ILGPU.IR;
using global::ILGPU.IR.Analyses;
using global::ILGPU.IR.Analyses.ControlFlowDirection;
using global::ILGPU.IR.Analyses.TraversalOrders;
using global::ILGPU.IR.Types;
using global::ILGPU.IR.Values;
using System.Text;

namespace SpawnDev.ILGPU.WebGPU.Backend
{
    internal sealed class WGSLKernelFunctionGenerator : WGSLCodeGenerator
    {
        #region Instance

        private HashSet<int> _atomicParameters = new HashSet<int>();
        // Subset of _atomicParameters where the element type is float (f32/f16).
        // WGSL does not support atomic<f32>, so these are declared as atomic<u32> (raw bits)
        // and emulated via a CAS loop in WGSLCodeGenerator.GenerateCode(GenericAtomic).
        private HashSet<int> _floatAtomicParameters = new HashSet<int>();
        // Subset of _atomicParameters where the atomic has unsigned semantics.
        // WGSL's atomicMin/atomicMax on i32 do signed comparisons, so unsigned
        // reductions need atomic<u32> buffers and u32 values.
        private HashSet<int> _unsignedAtomicParameters = new HashSet<int>();
        private HashSet<Value> _hoistedPrimitives = new HashSet<Value>();
        // Tracks WGSL variable names that hold pointer aliases (e.g., "v_27_f0" which is 'let v_27_f0 = &param1_f0;').
        // The post-processor checks this to avoid converting 'let v_28 = v_27_f0;' to 'v_28 = v_27_f0;'
        // which would be a type error (can't assign a pointer to a 'var' variable in WGSL).
        private HashSet<string> _viewPointerVarNames = new HashSet<string>();
        private HashSet<int> _emulatedF64Params = new HashSet<int>();
        private HashSet<int> _emulatedI64Params = new HashSet<int>();
        private List<DynamicSharedOverrideInfo> _dynamicSharedOverrides = new List<DynamicSharedOverrideInfo>();
        private bool _usesBroadcast = false;
        private bool _usesSubgroups = false;
        // Set to true by GenerateGroupAllReduce so the module-scope atomic<i32> workgroup var is emitted.
        // The static handler sets this via the public property to signal the need for _grp_reduce_i32.
        private bool _usesGroupReduce = false;
        /// <summary>Set to true by GenerateGroupAllReduce to request emission of <c>var&lt;workgroup&gt; _grp_reduce_i32: atomic&lt;i32&gt;</c>.</summary>
        public override bool UsesGroupReduce { get => _usesGroupReduce; set => _usesGroupReduce = value; }
        // Maps variable names to their emulation info (param index, is emu_f64)
        private Dictionary<string, (int ParamIndex, bool IsF64)> _emulatedVarMappings = new Dictionary<string, (int, bool)>();
        // Maps cross-block pointer variable names to inline expressions (e.g. "param1[v_3_idx]")
        // This fixes WGSL scoping: pointers declared in one switch case are not visible in another.
        private Dictionary<string, string> _crossBlockPointerExprs = new Dictionary<string, string>();
        // Set of LoadElementAddress Values that cross block boundaries
        private HashSet<Value> _crossBlockPointers = new HashSet<Value>();
        // Scalar packing manifest — populated during GenerateHeader(), used by SetupParameterBindings()
        private List<ScalarPackingEntry> _scalarManifest = new List<ScalarPackingEntry>();
        private CFG<ReversePostOrder, Forwards> _cfg;
        private Dominators<Backwards> _postDominators;
        private Loops<ReversePostOrder, Forwards> _loops;
        private GeneratorArgs _generatorArgs;

        // --- Body Struct Decomposition ---
        // A "body struct" is a struct parameter (e.g. IGridStrideKernelBody implementations like
        // ReductionImplementation) that contains ArrayView fields mixed with scalar fields.
        // WGSL cannot represent such a struct as a single buffer binding, so we decompose it:
        // - Each view field gets its own storage buffer binding: param{N}_f{fieldIdx}
        // - Scalar fields are packed into _scalar_params
        //
        // _bodyStructParams: paramIndex → list of field info
        // _bodyStructAtomicFields: (paramIndex, fieldIndex) → true if this view field is atomic
        // _bodyStructAtomicFloatFields: (paramIndex, fieldIndex) → true if float atomic
        private Dictionary<int, List<BodyStructFieldInfo>> _bodyStructParams = new Dictionary<int, List<BodyStructFieldInfo>>();
        private HashSet<(int paramIdx, int fieldIdx)> _bodyStructAtomicFields = new HashSet<(int, int)>();
        private HashSet<(int paramIdx, int fieldIdx)> _bodyStructAtomicFloatFields = new HashSet<(int, int)>();
        private HashSet<(int paramIdx, int fieldIdx)> _bodyStructUnsignedAtomicFields = new HashSet<(int, int)>();
        // Subset of _bodyStructAtomicFields where the element type is emulated 64-bit (emu_i64, emu_u64, emu_f64).
        // WGSL doesn't support atomic on vec2<u32>, so these buffers use plain storage and
        // the reduction's final write uses a plain store instead of an atomic operation.
        private HashSet<(int paramIdx, int fieldIdx)> _bodyStructEmulated64AtomicFields = new HashSet<(int, int)>();

        /// <summary>
        /// Describes one field of a decomposed body struct parameter.
        /// </summary>
        private sealed class BodyStructFieldInfo
        {
            public int FieldIndex { get; init; }
            public TypeNode FieldType { get; init; } = null!;
            public bool IsView { get; init; }   // true → gets its own buffer binding
            public bool IsScalar { get; init; } // true → packed into _scalar_params OR is view metadata
            // For view fields: the WGSL binding name (e.g. "param1_f0")
            public string BindingName { get; set; } = "";
            // For scalar fields: the slot index in _scalar_params (-1 if view metadata)
            public int ScalarSlot { get; set; } = -1;
            // True if this scalar field is metadata of a preceding view field (e.g. ArrayView1D.Length)
            // These fields use arrayLength(&bindingName) instead of _scalar_params
            public bool IsViewMetadata { get; set; } = false;
            // For IsViewMetadata fields: the binding name of the associated view field
            public string AssociatedViewBindingName { get; set; } = "";
            // True if this is a length field (Int64 following a view), false if flag (Int8 following a view)
            public bool IsLengthField { get; set; } = false;
        }
        // _bodyStructFieldVars: (paramIndex, fieldIndex) → WGSL variable name
        // Used by GetField code generation to redirect field accesses to the appropriate binding.
        private Dictionary<(int, int), string> _bodyStructFieldVars = new Dictionary<(int, int), string>();

        public WGSLKernelFunctionGenerator(
            in GeneratorArgs args,
            Method method,
            Allocas allocas)
            : base(args, method, allocas)
        {
            _generatorArgs = args;
            EntryPoint = args.EntryPoint;
            DynamicSharedAllocations = args.DynamicSharedAllocations;
            _cfg = method.Blocks.CreateCFG();
            _postDominators = _cfg.Blocks.CreatePostDominators();
            _loops = _cfg.CreateLoops();

            ScanForAtomicUsage();
            // CRITICAL: ScanBodyStructParams must run before body generation
            // because SetupParameterBindings (called during body generation) needs _bodyStructParams.
            // GenerateHeader runs AFTER body generation, so we cannot rely on GenerateHeader
            // to populate _bodyStructParams.
            ScanBodyStructParams();

            if (!EntryPoint.IsExplicitlyGrouped && method.Parameters.Count > 0)
            {
                var indexParam = method.Parameters[0];
                foreach (var block in method.Blocks)
                {
                    foreach (var entry in block)
                    {
                        if (entry.Value is Store store && store.Value == indexParam)
                        {
                            _indexParameterAddress = store.Target;
                            goto FoundIndexParam;
                        }
                    }
                }
            FoundIndexParam:;
            }
        }

        private void ScanForAtomicUsage()
        {
            foreach (var block in Method.Blocks)
            {
                foreach (var entry in block)
                {
                    if (entry.Value is global::ILGPU.IR.Values.GenericAtomic atomic)
                    {
                        // Resolve the atomic target to a parameter, tracking GetField chain
                        if (ResolveToParameterWithFieldChain(atomic.Target, out var param, out var fieldIdx))
                        {
                            if (fieldIdx >= 0)
                            {
                                // Atomic on a field of a body struct — track the specific field
                                _bodyStructAtomicFields.Add((param!.Index, fieldIdx));
                                var elemType = atomic.Value.Type;
                                if (elemType is global::ILGPU.IR.Types.PrimitiveType pt &&
                                    (pt.BasicValueType == global::ILGPU.BasicValueType.Float32 ||
                                     pt.BasicValueType == global::ILGPU.BasicValueType.Float16))
                                    _bodyStructAtomicFloatFields.Add((param.Index, fieldIdx));
                                if (atomic.IsUnsigned)
                                    _bodyStructUnsignedAtomicFields.Add((param.Index, fieldIdx));
                            }
                            else
                            {
                                // Atomic directly on a parameter
                                _atomicParameters.Add(param!.Index);
                                var elemType = atomic.Value.Type;
                                if (elemType is global::ILGPU.IR.Types.PrimitiveType pt &&
                                    (pt.BasicValueType == global::ILGPU.BasicValueType.Float32 ||
                                     pt.BasicValueType == global::ILGPU.BasicValueType.Float16))
                                    _floatAtomicParameters.Add(param.Index);
                                if (atomic.IsUnsigned)
                                    _unsignedAtomicParameters.Add(param.Index);
                            }
                        }
                    }
                    else if (entry.Value is global::ILGPU.IR.Values.AtomicCAS cas)
                    {
                        if (ResolveToParameterWithFieldChain(cas.Target, out var param, out var fieldIdx))
                        {
                            if (fieldIdx >= 0)
                                _bodyStructAtomicFields.Add((param!.Index, fieldIdx));
                            else
                                _atomicParameters.Add(param!.Index);
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Pre-populates _bodyStructParams by classifying struct parameters as body structs.
        /// MUST be called from the constructor, before body generation and SetupParameterBindings.
        /// GenerateHeader runs AFTER body generation in ILGPU's pipeline, so we cannot rely on
        /// GenerateHeader to populate _bodyStructParams. We do the field classification here so
        /// SetupParameterBindings can correctly handle body struct parameters.
        /// </summary>
        private void ScanBodyStructParams()
        {
            int paramOffset = KernelParamOffset;
            // Track global scalar slot offset across all params (as GenerateHeader does)
            int globalScalarSlotOffset = 0;

            foreach (var param in Method.Parameters)
            {
                if (param.Index < paramOffset) continue;

                var elementType = GetBufferElementType(param.ParameterType);
                var wgslType = TypeGenerator[elementType];
                bool isEmuF64 = Backend.Options.EnableF64Emulation && wgslType == "emu_f64";
                bool isEmuI64 = Backend.Options.EnableI64Emulation && (wgslType == "emu_i64" || wgslType == "emu_u64");

                // Body struct detection: structs containing view fields (e.g., ReductionImplementation,
                // InitializerImplementation) must be decomposed into individual bindings.
                //
                // Challenge: ArrayView2D/3D are also structs with view fields but should NOT be decomposed.
                // They use the IsMultiDim / view binding path which matches the runtime's binding layout.
                //
                // Solution: For grid-stride kernels (KernelParamOffset == 0), ALL struct params with view
                // fields are body structs — grid-stride kernels never receive raw ArrayView2D/3D params.
                // For stream kernels (KernelParamOffset > 0), use IsMultiDim to distinguish view wrappers.
                IsMultiDim(param.ParameterType, out var earlyIsMultiDimInit, out _, out _, out _, out _);
                if (param.ParameterType is global::ILGPU.IR.Types.StructureType bodyStructType
                    && IsBodyStruct(bodyStructType)
                    && (KernelParamOffset == 0 || !earlyIsMultiDimInit))
                {
                    var fieldInfos = new List<BodyStructFieldInfo>();
                    string lastViewBindingName = "";
                    int viewMetadataPhase = 0; // 0=idle, 1=after view ptr, 2=after Int64 (expect optional Int8)

                    for (int fi = 0; fi < bodyStructType.NumFields; fi++)
                    {
                        var fieldType = bodyStructType.Fields[fi];
                        bool fieldIsView = IsViewFieldType(fieldType);
                        // Capture before any reset
                        string capturedViewBindingName = lastViewBindingName;

                        // State machine for view metadata detection (see GenerateHeader for full comments)
                        bool isViewMetadata = false;
                        bool isLengthField = false;
                        if (fieldIsView)
                        {
                            viewMetadataPhase = 1;
                        }
                        else if (viewMetadataPhase == 1)
                        {
                            string typeStr = fieldType.ToString();
                            // Only mark as view-length metadata if:
                            // 1. The type is Int64 (view lengths are always Int64), AND
                            // 2. This is NOT the last field in the struct.
                            // Reason: ILGPU's DCE can remove the Output view's Length field when it's
                            // never accessed (e.g., Finish() only writes to Output[0], never reads Output.Length).
                            // When DCE removes Output.Length, the ReducedValue field (e.g., long/ulong) becomes
                            // the field immediately after the Output view ptr — and ReducedValue can also be
                            // Int64! With no next field following it, we can reliably distinguish:
                            // - If last field: it's ReducedValue (user scalar), NOT Output.length
                            // - If not last field: it's the view length (some field follows it)
                            bool isLastField = (fi == bodyStructType.NumFields - 1);
                            if (typeStr.Contains("Int64") && !isLastField)
                            {
                                isViewMetadata = true;
                                isLengthField = true;
                                viewMetadataPhase = 2;
                            }
                            else
                            {
                                viewMetadataPhase = 0;
                                lastViewBindingName = "";
                            }
                        }
                        else if (viewMetadataPhase == 2)
                        {
                            string typeStr = fieldType.ToString();
                            if (typeStr.Contains("Int8"))
                            {
                                isViewMetadata = true;
                                isLengthField = false;
                                viewMetadataPhase = 0;
                                lastViewBindingName = "";
                            }
                            else
                            {
                                viewMetadataPhase = 0;
                                lastViewBindingName = "";
                            }
                        }

                        var info = new BodyStructFieldInfo
                        {
                            FieldIndex = fi,
                            FieldType = fieldType,
                            IsView = fieldIsView,
                            IsScalar = !fieldIsView,
                            IsViewMetadata = isViewMetadata,
                            IsLengthField = isLengthField,
                            BindingName = fieldIsView ? $"param{param.Index}_f{fi}" : "",
                            AssociatedViewBindingName = isViewMetadata ? capturedViewBindingName : "",
                        };
                        fieldInfos.Add(info);

                        if (fieldIsView)
                        {
                            lastViewBindingName = $"param{param.Index}_f{fi}";
                        }
                        else if (!isViewMetadata)
                        {
                            // User scalar field: assign scalar slot
                            var scalarElemType = GetBufferElementType(fieldType);
                            var scalarWgslType = TypeGenerator[scalarElemType];
                            bool fieldIsEmuF64 = Backend.Options.EnableF64Emulation && scalarWgslType == "emu_f64";
                            bool fieldIsEmuI64 = Backend.Options.EnableI64Emulation && (scalarWgslType == "emu_i64" || scalarWgslType == "emu_u64");
                            int slotCount = (fieldIsEmuF64 || fieldIsEmuI64) ? 2 : 1;
                            info.ScalarSlot = globalScalarSlotOffset;
                            globalScalarSlotOffset += slotCount;
                            lastViewBindingName = "";
                        }
                    }
                    _bodyStructParams[param.Index] = fieldInfos;
                }
                else
                {
                    // Non-body-struct param: track if it's a packed scalar
                    IsMultiDim(param.ParameterType, out _, out var isView, out _, out _, out _);
                    bool isStruct = param.ParameterType is global::ILGPU.IR.Types.StructureType && !isView;
                    bool isAtomic = _atomicParameters.Contains(param.Index);
                    bool ownBinding = isView || isStruct || isAtomic;
                    if (!ownBinding)
                    {
                        // This is a packed scalar — advance the slot offset
                        int slotCount = (isEmuF64 || isEmuI64) ? 2 : 1;
                        globalScalarSlotOffset += slotCount;
                    }
                }
            }
        }


        /// <summary>
        /// Resolves a value to a parameter, also tracking if the access goes through a GetField
        /// on a body struct. Returns true if resolved, with fieldIdx=-1 for direct param access
        /// or fieldIdx>=0 for a field of a body struct.
        /// </summary>
        private bool ResolveToParameterWithFieldChain(Value target, out global::ILGPU.IR.Values.Parameter? param, out int fieldIdx)
        {
            param = null;
            fieldIdx = -1;
            var current = target;
            int lastFieldIdx = -1;
            while (current != null)
            {
                if (current is global::ILGPU.IR.Values.Parameter p)
                {
                    param = p;
                    fieldIdx = lastFieldIdx;
                    return true;
                }
                if (current is global::ILGPU.IR.Values.LoadElementAddress lea)
                {
                    current = lea.Source;
                    continue;
                }
                if (current is global::ILGPU.IR.Values.GetField gf)
                {
                    lastFieldIdx = gf.FieldSpan.Index;
                    current = gf.ObjectValue;
                    continue;
                }
                if (current is global::ILGPU.IR.Values.AddressSpaceCast cast)
                {
                    current = cast.Value;
                    continue;
                }
                break;
            }
            // Fallback: use original ResolveToParameter
            param = ResolveToParameter(target) as global::ILGPU.IR.Values.Parameter;
            return param != null;
        }

        /// <summary>
        /// Returns true if the given StructureType is a "body struct" — i.e., it contains
        /// at least one view field (ViewType or struct with View as first field).
        /// Body structs (like ReductionImplementation, InitializerImplementation) cannot be
        /// represented as a single WGSL buffer binding and must be decomposed.
        /// </summary>
        private bool IsBodyStruct(global::ILGPU.IR.Types.StructureType st)
        {
            for (int i = 0; i < st.NumFields; i++)
            {
                var field = st.Fields[i];
                if (IsViewFieldType(field)) return true;
            }
            return false;
        }

        /// <summary>
        /// Returns true if the given type is a view type (ViewType or struct whose first field is a view).
        /// </summary>
        private bool IsViewFieldType(TypeNode type)
        {
            if (type is ViewType) return true;
            if (type is global::ILGPU.IR.Types.StructureType st)
            {
                if (st.NumFields > 0 && (st.Fields[0] is ViewType || st.Fields[0].ToString().Contains("View")))
                    return true;
            }
            if (type.ToString().Contains("View")) return true;
            return false;
        }

        #endregion

        #region Properties

        public EntryPoint EntryPoint { get; }
        public AllocaKindInformation DynamicSharedAllocations { get; }

        /// <summary>
        /// Returns the list of dynamic shared memory override constant info generated during code emission.
        /// </summary>
        public IReadOnlyList<DynamicSharedOverrideInfo> DynamicSharedOverrides => _dynamicSharedOverrides;

        private Value? _indexParameterAddress;

        // Returns 1 if the first IR parameter is an index that should be skipped
        // (for bindings/scalar packing), 0 if it's a user-supplied parameter.
        //
        // For auto-grouped kernels (IndexType = Index1D/2D/3D/LongIndex*),
        // ILGPU always puts the index as Method.Parameters[0] and it should be skipped.
        // Note: Index2D/3D are StructureType in IR (they have X,Y,Z fields).
        //
        // For KernelConfig (stream/explicitly-grouped kernels), the first IR param
        // may or may not be an index. LoadStreamKernel<Index1D, ...> puts Index1D
        // (PrimitiveType) at param 0. LoadStreamKernel<ArrayView<T>> puts a ViewType.
        // We only skip primitives (actual index types), not views or structs.
        private int KernelParamOffset
        {
            get
            {
                if (EntryPoint.IndexType == IndexType.None) return 0;

                // For auto-grouped kernels, the first param is always the index — skip it.
                // This covers Index1D (PrimitiveType), Index2D/3D (StructureType), and LongIndex* types.
                if (EntryPoint.IndexType != IndexType.KernelConfig) return 1;

                // KernelConfig: explicitly grouped / grid-stride kernels.
                // ILGPU's KernelIndexParameterOffset is 0 for KernelConfig, meaning ALL
                // params are in EntryPoint.Parameters[]. However, stream kernels loaded
                // via LoadStreamKernel<Index1D, ...> have Index1D as Parameters[0] —
                // this is the thread index and should be skipped (mapped to global_id.x).
                //
                // Grid-stride kernels (e.g. reduce, initialize) use LongIndex1D as their
                // first param — this is the padded data count, NOT a thread index.
                // It must NOT be skipped; it needs a buffer binding.
                //
                // Distinction: Index1D/2D/3D → stream kernel index → skip (offset=1)
                //              LongIndex1D/2D/3D → grid-stride data count → don't skip (offset=0)
                //
                // This mirrors the runtime's runtimeIndexSkip logic in WebGPUAccelerator.cs.
                if (EntryPoint.Parameters.Count > 0)
                {
                    var firstParamType = EntryPoint.Parameters[0];
                    if (firstParamType == typeof(Index1D) || firstParamType == typeof(Index2D) || firstParamType == typeof(Index3D))
                        return 1;
                }
                return 0;
            }
        }


        #endregion

        #region Methods

        /// <summary>
        /// Returns true if the atomic target pointer resolves to a parameter that was
        /// detected as float-typed during ScanForAtomicUsage. Used by the base class
        /// GenerateCode(GenericAtomic) to select the CAS-loop emulation path.
        /// </summary>
        protected override bool IsFloatAtomicTarget(global::ILGPU.IR.Value target)
        {
            // Check direct parameter float atomics
            if (ResolveToParameter(target) is global::ILGPU.IR.Values.Parameter param
                && _floatAtomicParameters.Contains(param.Index))
                return true;

            // Check body struct float atomic fields (e.g. ReductionImplementation.Output view)
            if (ResolveToParameterWithFieldChain(target, out var bsParam, out var fieldIdx)
                && fieldIdx >= 0
                && _bodyStructAtomicFloatFields.Contains((bsParam!.Index, fieldIdx)))
                return true;

            return false;
        }

        /// <summary>
        /// Returns true if the atomic target pointer resolves to a parameter that was
        /// detected as unsigned during ScanForAtomicUsage. Used by the base class
        /// GenerateCode(GenericAtomic) to use atomic&lt;u32&gt; operations for unsigned semantics.
        /// </summary>
        protected override bool IsUnsignedAtomicTarget(global::ILGPU.IR.Value target)
        {
            // Check direct parameter unsigned atomics
            if (ResolveToParameter(target) is global::ILGPU.IR.Values.Parameter param
                && _unsignedAtomicParameters.Contains(param.Index))
                return true;

            // Check body struct unsigned atomic fields
            if (ResolveToParameterWithFieldChain(target, out var bsParam, out var fieldIdx)
                && fieldIdx >= 0
                && _bodyStructUnsignedAtomicFields.Contains((bsParam!.Index, fieldIdx)))
                return true;

            return false;
        }

        /// <summary>
        /// Returns true if the atomic target pointer resolves to a parameter with an emulated
        /// 64-bit element type (emu_i64, emu_u64, emu_f64). WGSL doesn't support atomic
        /// operations on vec2&lt;u32&gt;, so these must use plain storage and a direct store.
        /// </summary>
        protected override bool IsEmulated64BitAtomicTarget(global::ILGPU.IR.Value target)
        {
            // Check direct parameter emulated 64-bit atomics
            if (ResolveToParameter(target) is global::ILGPU.IR.Values.Parameter param
                && (_emulatedI64Params.Contains(param.Index) || _emulatedF64Params.Contains(param.Index))
                && _atomicParameters.Contains(param.Index))
                return true;

            // Check body struct emulated 64-bit atomic fields
            if (ResolveToParameterWithFieldChain(target, out var bsParam, out var fieldIdx)
                && fieldIdx >= 0
                && _bodyStructEmulated64AtomicFields.Contains((bsParam!.Index, fieldIdx)))
                return true;

            return false;
        }

        private TypeNode UnwrapType(TypeNode type)
        {
            var current = type;
            while (current != null)
            {
                if (current is global::ILGPU.IR.Types.PointerType ptr)
                {
                    current = ptr.ElementType;
                    continue;
                }
                return current;
            }
            return type;
        }

        private bool IsViewStructure(TypeNode type)
        {
            if (type is ViewType) return true;
            if (type is StructureType st)
            {
                if (st.ToString().Contains("View")) return true;
                if (st.NumFields >= 2 && st.Fields[0] is PointerType) return true;
            }
            return false;
        }

        public override void GenerateHeader(StringBuilder builder)
        {
            // Pre-scan for Broadcast and subgroup usage
            ScanForSubgroupAndBroadcastUsage();

            // Emit WGSL enable directives at the very top of the module
            // These must appear before any other declarations
            // For subgroups: emit a placeholder that will be resolved after body generation.
            // The IR pre-scan may flag subgroup usage for nodes that are later handled
            // via shared-memory emulation (e.g., 64-bit reductions), so we defer the
            // decision until we can inspect the generated body.
            bool mayUseSubgroups = Backend.HasSubgroups && _usesSubgroups;
            if (mayUseSubgroups)
                builder.AppendLine("/*__WGSL_ENABLE_SUBGROUPS_PLACEHOLDER__*/");
            if (Backend.HasShaderF16)
                builder.AppendLine("enable f16;");
            if (mayUseSubgroups || Backend.HasShaderF16)
                builder.AppendLine();

            // Pre-scan parameters to detect if any actually use f64/i64 types.
            // IMPORTANT: We check IR types directly (BasicValueType) instead of going through
            // TypeGenerator[...] which would populate the mapping dictionary and cause
            // GenerateTypeDefinitions to emit struct definitions prematurely.
            bool needsF64Emulation = Backend.Options.EnableF64Emulation && ContainsBasicValueType(BasicValueType.Float64);
            bool containsI64 = ContainsBasicValueType(BasicValueType.Int64);
            bool needsI64Emulation = Backend.Options.EnableI64Emulation && containsI64;

            // Set per-kernel overrides on the TypeGenerator so that intermediate IR values
            // (e.g., Grid.GlobalIndex.X → Int64) map to i32 instead of emu_i64 when the
            // kernel parameters don't actually use 64-bit data types.
            TypeGenerator.KernelUsesI64 = needsI64Emulation;
            TypeGenerator.KernelUsesF64 = needsF64Emulation;

            // Only emit emulation library if it's both enabled AND actually needed by this kernel
            if (needsF64Emulation || needsI64Emulation)
            {
                builder.AppendLine("// ============ 64-bit Emulation Library ============");
                builder.AppendLine(WGSLEmulationLibrary.GetEmulationLibrary(
                    needsF64Emulation,
                    Backend.Options.UseOzakiF64Emulation,
                    needsI64Emulation));
            }

            // Emit struct definitions (may reference emu_i64/emu_f64 from the library above)
            TypeGenerator.GenerateTypeDefinitions(builder);

            int bindingIdx = 0;
            int paramOffset = KernelParamOffset;

            // --- SCALAR PACKING ---
            // Instead of emitting one @binding per param, we pack all non-view, non-struct
            // scalar params into a single array<u32> storage buffer. This drastically reduces
            // the binding count, staying within WebGPU's maxStorageBuffersPerShaderStage limit.
            //
            // Phase 1: Classify params and emit view bindings. Collect scalar packing info.
            // Phase 2: Emit single packed scalar binding (if any).

            var scalarManifest = new List<ScalarPackingEntry>();
            int scalarSlotOffset = 0; // current u32 slot index in packed buffer

            foreach (var param in Method.Parameters)
            {
                if (param.Index < paramOffset) continue;

                var elementType = GetBufferElementType(param.ParameterType);
                var wgslType = TypeGenerator[elementType];

                // Debug info
                builder.AppendLine($"// Param {param.Index}: {param.ParameterType} (Element: {elementType})");

                // --- BODY STRUCT DECOMPOSITION ---
                // We must check for body structs (e.g. ReductionImplementation<T>, InitializerImplementation<T>)
                // that contain view fields mixed with scalar fields.
                // These get decomposed into individual bindings.
                //
                // Challenge: ArrayView2D/3D are also StructureTypes with a view field, but they
                // should NOT be treated as body structs. They are handled by the IsMultiDim / view
                // binding path, which matches the runtime's binding layout.
                //
                // Solution: For grid-stride kernels (KernelParamOffset == 0), ALL struct params with
                // view fields are body structs — grid-stride kernels never receive raw view params.
                // For stream kernels (KernelParamOffset > 0), use IsMultiDim to exclude view wrappers.
                IsMultiDim(param.ParameterType, out var earlyIsMultiDim, out var earlyIsView, out var earlyIs1D, out var earlyIs2D, out var earlyIs3D);
                if (param.ParameterType is global::ILGPU.IR.Types.StructureType earlyBodyStructType
                    && IsBodyStruct(earlyBodyStructType)
                    && (KernelParamOffset == 0 || !earlyIsMultiDim))
                {
                    var fieldInfos = new List<BodyStructFieldInfo>();
                    string lastViewBindingName = "";
                    int viewMetadataPhase = 0; // 0=idle, 1=after view ptr (expect Int64), 2=after Int64 (expect optional Int8)

                    for (int fi = 0; fi < earlyBodyStructType.NumFields; fi++)
                    {
                        var fieldType = earlyBodyStructType.Fields[fi];
                        bool fieldIsView = IsViewFieldType(fieldType);
                        WebGPUBackend.Log($"[WGSL-Field] param.Index={param.Index} fi={fi} type={fieldType} IsView={fieldIsView} phase={viewMetadataPhase} lastViewName='{lastViewBindingName}'");
                        // Capture lastViewBindingName BEFORE any potential reset in the state machine below,
                        // so AssociatedViewBindingName is correctly set for metadata fields.
                        string capturedViewBindingName = lastViewBindingName;

                        // Detect view metadata: Int64 or Int8 fields immediately following a view field
                        // These are internal fields of ArrayView1D (Length, flag) — not user scalars
                        //
                        // ArrayView1D<T, Dense> IR structure: [ptr(ViewType), Int64(length), Int8(valid_flag)]
                        // ArrayView<T>          IR structure: [ptr(ViewType), Int64(length)] (no flag)
                        //
                        // We use a state machine (viewMetadataPhase) to track position in this sequence:
                        //   0 = no view being tracked (reset state)
                        //   1 = saw a view ptr; expecting Int64 length next
                        //   2 = saw the Int64 length; expecting optional Int8 flag next
                        // Any field outside expected sequence → reset to 0.
                        //
                        // CRITICAL: This prevents ReducedValue:long (Int64) from being misclassified
                        // as a second view's length metadata when it immediately follows a view's
                        // Int64 length field (which occurs in ArrayView<T> with no Int8 flag).
                        bool isViewMetadata = false;
                        bool isLengthField = false;
                        if (fieldIsView)
                        {
                            // View pointer encountered — next field should be Int64 length
                            viewMetadataPhase = 1;
                        }
                        else if (viewMetadataPhase == 1)
                        {
                            string typeStr = fieldType.ToString();
                            // Only mark as view-length metadata if:
                            // 1. The type is Int64 (view lengths are always Int64), AND
                            // 2. This is NOT the last field in the struct.
                            // Reason: ILGPU's DCE can remove the Output view's Length field when it's
                            // never accessed. When DCE removes Output.Length, ReducedValue (e.g., long/ulong)
                            // becomes the field immediately after the Output view ptr — and it's also Int64!
                            // If there's no next field, this must be ReducedValue, not a view length.
                            bool isLastField = (fi == earlyBodyStructType.NumFields - 1);
                            if (typeStr.Contains("Int64") && !isLastField)
                            {
                                isViewMetadata = true;
                                isLengthField = true;
                                viewMetadataPhase = 2; // Now expecting optional Int8 flag
                            }
                            else
                            {
                                // Unexpected type where length should be, OR last field — reset
                                viewMetadataPhase = 0;
                                lastViewBindingName = "";
                            }
                        }
                        else if (viewMetadataPhase == 2)
                        {
                            // Just after Int64 length: expecting optional Int8 flag
                            string typeStr = fieldType.ToString();
                            if (typeStr.Contains("Int8"))
                            {
                                isViewMetadata = true;
                                isLengthField = false;
                                viewMetadataPhase = 0; // Int8 is the last metadata field — reset
                                lastViewBindingName = "";
                            }
                            else
                            {
                                // No Int8 flag (e.g., ArrayView<T> vs ArrayView1D) — this is a user scalar
                                // Reset phase and treat this field normally (user scalar)
                                viewMetadataPhase = 0;
                                lastViewBindingName = "";
                            }
                        }
                        else
                        {
                            // Not in metadata tracking (viewMetadataPhase == 0) — user scalar or another view
                            // lastViewBindingName is already "" here; no action needed
                        }

                        var info = new BodyStructFieldInfo
                        {
                            FieldIndex = fi,
                            FieldType = fieldType,
                            IsView = fieldIsView,
                            IsScalar = !fieldIsView,
                            IsViewMetadata = isViewMetadata,
                            IsLengthField = isLengthField,
                            AssociatedViewBindingName = isViewMetadata ? capturedViewBindingName : "",
                        };
                        fieldInfos.Add(info);

                        if (fieldIsView)
                        {
                            // View field: create a separate buffer binding
                            string bindingName = $"param{param.Index}_f{fi}";
                            info.BindingName = bindingName;
                            lastViewBindingName = bindingName;

                            // Determine element type for this view field
                            var viewElemType = GetBufferElementType(fieldType);
                            var viewWgslType = TypeGenerator[viewElemType];

                            // Check if this view field is atomic
                            bool fieldIsAtomic = _bodyStructAtomicFields.Contains((param.Index, fi));
                            bool fieldIsFloatAtomic = _bodyStructAtomicFloatFields.Contains((param.Index, fi));
                            bool fieldIsUnsignedAtomic = _bodyStructUnsignedAtomicFields.Contains((param.Index, fi));

                            string bindingWgslType = viewWgslType;
                            bool is64Emu = (Backend.Options.EnableF64Emulation && viewWgslType == "emu_f64") ||
                                           (Backend.Options.EnableI64Emulation && (viewWgslType == "emu_i64" || viewWgslType == "emu_u64"));

                            if (is64Emu)
                            {
                                bindingWgslType = "u32";
                            }

                            if (fieldIsAtomic)
                            {
                                if (is64Emu)
                                {
                                    // WGSL doesn't support atomic on 64-bit structs.
                                    // Use plain storage (u32); only thread 0 writes the final value.
                                    bindingWgslType = "u32";
                                    _bodyStructEmulated64AtomicFields.Add((param.Index, fi));
                                }
                                else if (fieldIsFloatAtomic || fieldIsUnsignedAtomic)
                                    bindingWgslType = "atomic<u32>";
                                else
                                    bindingWgslType = $"atomic<{viewWgslType}>";
                            }

                            builder.AppendLine($"// Body struct field {fi}: {fieldType} (view)");
                            builder.AppendLine($"@group(0) @binding({bindingIdx}) var<storage, read_write> {bindingName} : array<{bindingWgslType}>;");
                            bindingIdx++;
                        }
                        else if (isViewMetadata)
                        {
                            // View metadata field: NOT packed into _scalar_params
                            // Will use arrayLength(&bindingName) or 0 in SetupParameterBindings
                            builder.AppendLine($"// Body struct field {fi}: {fieldType} (view metadata, {(isLengthField ? "length" : "flag")})");
                            // Don't reset lastViewBindingName — there may be more metadata fields
                        }
                        else
                        {
                            // User scalar field: pack into _scalar_params
                            lastViewBindingName = ""; // Reset view tracking
                            var scalarElemType = GetBufferElementType(fieldType);
                            var scalarWgslType = TypeGenerator[scalarElemType];

                            bool isEmuF64 = Backend.Options.EnableF64Emulation && scalarWgslType == "emu_f64";
                            bool isEmuI64 = Backend.Options.EnableI64Emulation && (scalarWgslType == "emu_i64" || scalarWgslType == "emu_u64");
                            int slotCount = (isEmuF64 || isEmuI64) ? 2 : 1;

                            info.ScalarSlot = scalarSlotOffset;

                            // Use a synthetic param index for the scalar manifest entry
                            // We encode it as (paramIndex + 1) * 1000 + fieldIndex to make it unique
                            // IMPORTANT: use (param.Index + 1) so that grid-stride kernels with
                            // param.Index == 0 still produce a synthetic >= 1000, which is how the
                            // runtime decoder distinguishes body-struct scalars from real params.
                            int syntheticParamIdx = (param.Index + 1) * 1000 + fi;
                            WebGPUBackend.Log($"[WGSL-BodyStruct] param.Index={param.Index}, fi={fi}, syntheticParamIdx={syntheticParamIdx}, paramOffset={paramOffset}, wgslType={scalarWgslType}");
                            var entry = new ScalarPackingEntry
                            {
                                ParamIndex = syntheticParamIdx,
                                ByteOffset = scalarSlotOffset * 4,
                                ByteSize = slotCount * 4,
                                WgslType = scalarWgslType,
                                IsEmulatedF64 = isEmuF64,
                                IsEmulatedI64 = isEmuI64,
                                IsStruct = false,
                            };
                            scalarManifest.Add(entry);
                            builder.AppendLine($"// Body struct field {fi}: {fieldType} (user scalar, slot {scalarSlotOffset})");
                            scalarSlotOffset += slotCount;
                        }
                    }
                    _bodyStructParams[param.Index] = fieldInfos;
                    continue; // Skip normal binding logic for this param
                }

                // Determine if this is a view, multidim view, struct, or primitive scalar
                IsMultiDim(param.ParameterType, out var isMultiDim, out var isView, out var is1DView, out var is2DView, out var is3DView);

                bool isStruct = param.ParameterType is global::ILGPU.IR.Types.StructureType && !isView;
                bool isAtomic = _atomicParameters.Contains(param.Index);

                // Check emulation
                bool isEmulatedF64 = Backend.Options.EnableF64Emulation && wgslType == "emu_f64";
                bool isEmulatedI64 = Backend.Options.EnableI64Emulation && (wgslType == "emu_i64" || wgslType == "emu_u64");

                // If parameter is a view, check its element type for emulation tracking
                if (isView)
                {
                    var paramElemType = GetBufferElementType(param.ParameterType);
                    string elemWgslType = TypeGenerator[paramElemType];
                    if (Backend.Options.EnableF64Emulation && elemWgslType == "emu_f64")
                        isEmulatedF64 = true;
                    if (Backend.Options.EnableI64Emulation && (elemWgslType == "emu_i64" || elemWgslType == "emu_u64"))
                        isEmulatedI64 = true;
                }

                if (isEmulatedF64)
                    _emulatedF64Params.Add(param.Index);
                else if (isEmulatedI64)
                    _emulatedI64Params.Add(param.Index);

                // DECISION: Does this param get its own binding, or is it packed?
                bool ownBinding = isView || isStruct || isAtomic;

                if (ownBinding)
                {
                    // --- OWN BINDING (views, structs, atomics) ---
                    string accessMode = "read_write";
                    string bindingWgslType = wgslType;

                    if (isEmulatedF64 || isEmulatedI64)
                    {
                        bindingWgslType = "u32"; // Raw bits storage
                    }

                    if (isAtomic)
                    {
                        // WGSL only supports atomic<i32> and atomic<u32>.
                        // Float atomics (f32/f16) must be stored as atomic<u32> (raw bits)
                        // and emulated via a CAS loop.
                        // Unsigned atomics also need atomic<u32> for correct min/max semantics.
                        if (_floatAtomicParameters.Contains(param.Index) ||
                            _unsignedAtomicParameters.Contains(param.Index))
                            bindingWgslType = "atomic<u32>";
                        else
                            bindingWgslType = $"atomic<{bindingWgslType}>";
                    }

                    var bindingDecl = $"@group(0) @binding({bindingIdx}) var<storage, {accessMode}> param{param.Index} : array<{bindingWgslType}>;";
                    builder.AppendLine(bindingDecl);
                    bindingIdx++;

                    // Stride buffer for 2D/3D views
                    if (isView && isMultiDim && (is2DView || is3DView))
                    {
                        var strideDecl = $"@group(0) @binding({bindingIdx}) var<storage, read> param{param.Index}_stride : array<i32>;";
                        builder.AppendLine(strideDecl);
                        bindingIdx++;
                    }
                }

                else
                {
                    // --- PACKED SCALAR ---
                    // Determine how many u32 slots this scalar needs
                    int slotCount = 1; // default: 4 bytes
                    int byteSize = 4;
                    string scalarWgslType = wgslType;

                    if (isEmulatedF64)
                    {
                        slotCount = 2;
                        byteSize = 8;
                        scalarWgslType = "emu_f64";
                    }
                    else if (isEmulatedI64)
                    {
                        slotCount = 2;
                        byteSize = 8;
                        scalarWgslType = wgslType; // emu_i64 or emu_u64
                    }

                    var entry = new ScalarPackingEntry
                    {
                        ParamIndex = param.Index,
                        ByteOffset = scalarSlotOffset * 4,
                        ByteSize = byteSize,
                        WgslType = scalarWgslType,
                        IsEmulatedF64 = isEmulatedF64,
                        IsEmulatedI64 = isEmulatedI64,
                        IsStruct = false,
                    };
                    scalarManifest.Add(entry);
                    builder.AppendLine($"// -> Packed scalar at slot {scalarSlotOffset} ({slotCount} slots, type: {scalarWgslType})");

                    scalarSlotOffset += slotCount;
                }
            }

            // Emit the single packed scalar binding (if any scalars were packed)
            if (scalarManifest.Count > 0)
            {
                builder.AppendLine($"@group(0) @binding({bindingIdx}) var<storage, read> _scalar_params : array<u32>;");
                bindingIdx++;

                // Store manifest in generator args for the compiled kernel
                foreach (var entry in scalarManifest)
                    _generatorArgs.ScalarPackingManifest.Add(entry);

                // DIAGNOSTIC: dump manifest entries so we can verify ParamIndex values
                foreach (var entry in scalarManifest)
                    WebGPUBackend.Log($"[WGSL-Manifest] ParamIndex={entry.ParamIndex}, ByteOffset={entry.ByteOffset}, ByteSize={entry.ByteSize}, WgslType={entry.WgslType}");
            }

            // Store manifest locally for SetupParameterBindings()
            _scalarManifest = scalarManifest;

            builder.AppendLine();

            // Emit shared memory allocations (static)
            foreach (var alloca in Allocas.SharedAllocations)
            {
                var variable = Load(alloca.Alloca);
                declaredVariables.Add(variable.Name);

                var elementType = alloca.ElementType;
                int entryCount = (int)alloca.ArraySize;

                var wgslType = TypeGenerator[elementType];
                builder.AppendLine($"var<workgroup> {variable.Name} : array<{wgslType}, {entryCount}>;");
            }

            // Emit dynamic shared memory allocations using pipeline-overridable constants
            if (DynamicSharedAllocations.Length > 0)
            {
                builder.AppendLine();
                builder.AppendLine("// Dynamic shared memory (sized via pipeline override constants)");
                foreach (var alloca in DynamicSharedAllocations)
                {
                    var variable = Load(alloca.Alloca);
                    declaredVariables.Add(variable.Name);

                    var elementType = alloca.ElementType;
                    var wgslType = TypeGenerator[elementType];
                    int elementSize = alloca.ElementSize;

                    // Override constant name — the accelerator will set this at pipeline creation
                    string overrideConstName = $"DYNAMIC_SHARED_SIZE_{alloca.Index}";

                    // Default to 1 element (will be overridden at pipeline creation)
                    builder.AppendLine($"override {overrideConstName} : u32 = 1u;");
                    builder.AppendLine($"var<workgroup> {variable.Name} : array<{wgslType}, {overrideConstName}>;");

                    // Track the override constant info for the compiled kernel
                    var overrideInfo = new DynamicSharedOverrideInfo(
                        overrideConstName,
                        variable.Name,
                        alloca.Index,
                        elementSize);
                    _dynamicSharedOverrides.Add(overrideInfo);
                    _generatorArgs.DynamicSharedOverrides.Add(overrideInfo);
                }
            }

            // Emit workgroup-level broadcast temp variable if needed
            if (_usesBroadcast)
            {
                builder.AppendLine();
                builder.AppendLine("// Workgroup broadcast shared memory");
                builder.AppendLine("var<workgroup> _broadcast_temp : i32;");
            }

            // Emit warp shuffle shared memory buffer when:
            // 1. Subgroups are NOT available but shuffle ops are used (shared-memory emulation path), OR
            // 2. UsesWarpShuffleEmulation is explicitly set (e.g., 64-bit reduce via shared memory)
            if ((_usesSubgroups && !Backend.HasSubgroups) || UsesWarpShuffleEmulation)
            {
                builder.AppendLine();
                builder.AppendLine("// Warp shuffle emulation shared memory (256 u32 slots = 4 per thread for Ozaki vec4<f32> emulated types)");
                builder.AppendLine("var<workgroup> _warp_shuffle_buf : array<u32, 256>;");
            }

            // Emit workgroup-level group-reduce atomic variable when GenerateGroupAllReduce is used.
            // Declared as atomic<i32> for integer max/min/add, plus an auxiliary slot array
            // holding per-subgroup partial results (max 32 subgroups for workgroup_size 2048).
            if (_usesGroupReduce)
            {
                builder.AppendLine();
                builder.AppendLine("// Group-level reduction shared memory (for GenerateGroupAllReduce)");
                builder.AppendLine("var<workgroup> _grp_reduce_i32 : atomic<i32>;");
                builder.AppendLine("var<workgroup> _grp_reduce_u32 : atomic<u32>;");
                builder.AppendLine("var<workgroup> _grp_sg_results : array<u32, 64>; // per-subgroup partial results (2 slots per sg for 64-bit types)");
            }

            builder.AppendLine();
        }

        /// <summary>
        /// Scans the method for Broadcast and subgroup IR nodes to determine what features are needed.
        /// Sets _usesBroadcast (for shared memory temp) and _usesSubgroups (for enable subgroups directive).
        /// </summary>
        private void ScanForSubgroupAndBroadcastUsage()
        {
            _usesBroadcast = false;
            _usesSubgroups = false;

            foreach (var block in Method.Blocks)
            {
                foreach (var entry in block)
                {
                    switch (entry.Value)
                    {
                        case global::ILGPU.IR.Values.Broadcast broadcast:
                            if (broadcast.Kind == global::ILGPU.IR.Values.BroadcastKind.GroupLevel)
                                _usesBroadcast = true;
                            else
                                _usesSubgroups = true; // warp-level broadcast uses subgroupBroadcastFirst
                            break;
                        case global::ILGPU.IR.Values.WarpShuffle:
                        case global::ILGPU.IR.Values.SubWarpShuffle:
                        case global::ILGPU.IR.Values.LaneIdxValue:
                        case global::ILGPU.IR.Values.WarpSizeValue:
                            _usesSubgroups = true;
                            break;
                    }
                }
            }
        }

        public override void GenerateCode()
        {
            string workgroupSize = GetWorkgroupSize();

            // --- BODY-FIRST GENERATION ---
            // We need to know if the body uses subgroup builtins BEFORE emitting the signature.
            // The IR pre-scan (_usesSubgroups) may miss inlined methods. Instead, generate the
            // body into a temporary StringBuilder, check if it references subgroup_size or
            // subgroup_invocation_id, then emit the correct function signature.

            // Save the current builder position so we can inject the signature before the body.
            int signatureInsertPosition = Builder.Length;

            // Emit a placeholder for the signature (we'll replace it after generating the body).
            // We use a unique sentinel that we'll replace with the real signature.
            const string signatureSentinel = "/*__WGSL_SIGNATURE_PLACEHOLDER__*/";
            Builder.AppendLine(signatureSentinel);

            PushIndent();

            // Declare workgroup_size constant for access by intrinsics
            string wgDims = EntryPoint.IndexType switch
            {
                IndexType.Index1D => "64, 1, 1",
                IndexType.Index2D => "8, 8, 1",
                IndexType.Index3D => "4, 4, 4",
                _ => "64, 1, 1"
            };
            AppendLine($"let workgroup_size = vec3<u32>({wgDims});");

            // 0. Pre-scan parameters to identify emulated 64-bit types
            // This MUST happen before hoisting so that Load/Store know which buffers are emulated
            PreScanEmulatedParameters();

            // 1. Scan and declare hoisted variables
            HoistCrossBlockVariables();

            // 2. Setup standard bindings
            SetupIndexVariables();
            SetupParameterBindings();

            // 3. START CONTROL FLOW
            // Save position before code generation for post-processing
            int codeGenStartPosition = Builder.Length;

            if (Method.Blocks.Count == 1)
            {
                GenerateCodeInternal();
            }
            else if (_loops.Count == 0) // No Loops -> Acyclic
            {
                // ACYCLIC KERNEL: Use Structured Control Flow (Nested IFs)
                var pd = global::ILGPU.IR.Analyses.Dominators.CreatePostDominators(Method.Blocks);
                GenerateStructuredCode(Method.EntryBlock, null, pd);
            }
            else
            {
                // CYCLIC KERNEL: Use Structured Control Flow with Loop constructs
                // Instead of the state machine (loop { switch { case: ... } }) which is
                // incompatible with workgroupBarrier() due to WGSL's uniform control flow
                // requirements, we generate proper WGSL loop {} constructs using the
                // loop analysis data from ILGPU. This allows barriers to be in uniform
                // control flow (inside the loop body at the top level, not inside a switch).
                var pd = global::ILGPU.IR.Analyses.Dominators.CreatePostDominators(Method.Blocks);
                GenerateStructuredCode(Method.EntryBlock, null, pd);
            }

            // CRITICAL: Post-process to fix variable scoping issues.
            // HoistCrossBlockVariables() declares cross-block vars at function scope, but its
            // analysis may miss variables used by terminators in other blocks (e.g., CompareValue
            // used as IfBranch condition). Code generators also emit "let v_N = expr;" which creates
            // block-scoped bindings that shadow hoisted vars. Fix: unconditionally convert ALL
            // "let v_N = expr;" to assignments, and add missing var declarations at function scope.
            // Pointer references (&param, &temp_) must remain as 'let' since they're immutable bindings.
            if (Method.Blocks.Count > 1)
            {
                string generatedCode = Builder.ToString(codeGenStartPosition, Builder.Length - codeGenStartPosition);
                var letPattern = new System.Text.RegularExpressions.Regex(
                    @"^(\s*)let\s+(v_\d+(?:_\w+)?)\s*=\s*(.+);",
                    System.Text.RegularExpressions.RegexOptions.Multiline);
                var missingDeclarations = new List<string>(); // var declarations to add
                var hoistedLetDeclarations = new List<string>(); // pointer-alias lets to hoist
                bool anyReplacements = false;
                string processed = letPattern.Replace(generatedCode, match =>
                {
                    string indent = match.Groups[1].Value;
                    string varName = match.Groups[2].Value;
                    string expr = match.Groups[3].Value;

                    // Pointer aliases (expressions starting with '&') can't be declared as 'var'
                    // in WGSL — they must remain as 'let'. But 'let' has block scope, so if they're
                    // inside a case block and used from other case blocks, we must hoist them to
                    // function scope. ONLY hoist simple pointer aliases like &v_67 (references to
                    // module-scope shared memory). DO NOT hoist indexed expressions like &v_66[v_82]
                    // since those depend on block-local variables.
                    if (expr.TrimStart().StartsWith("&") || _viewPointerVarNames.Contains(expr.Trim()))
                    {
                        var trimmed = expr.TrimStart();
                        if (_viewPointerVarNames.Contains(expr.Trim()))
                        {
                            // The RHS is a pointer alias variable (e.g. v_27_f0, which is 'let v_27_f0 = &param1_f0;').
                            // WGSL cannot assign pointers to 'var' variables — keep as 'let' and hoist to function scope.
                            hoistedLetDeclarations.Add($"    let {varName} = {trimmed};");
                            _viewPointerVarNames.Add(varName); // Also track v_28 as a pointer alias
                            anyReplacements = true;
                            return ""; // Remove from case block; will be emitted at function scope
                        }
                        // Simple pointer alias: &identifier (no brackets, no dots, no operators)
                        // e.g. &v_67, &param0 — these reference module-scope or function-scope vars
                        bool isSimpleRef = System.Text.RegularExpressions.Regex.IsMatch(trimmed, @"^\&\w+$");
                        if (isSimpleRef)
                        {
                            hoistedLetDeclarations.Add($"    let {varName} = {trimmed};");
                            anyReplacements = true;
                            return ""; // Remove from case block; will be emitted at function scope
                        }
                        // Complex pointer expression (e.g. &v_66[v_82]) — keep as-is in block
                        return match.Value;
                    }

                    anyReplacements = true;

                    // If this variable wasn't already declared by HoistCrossBlockVariables,
                    // we need to add a var declaration at function scope
                    if (!declaredVariables.Contains(varName))
                    {
                        // Infer type from expression patterns
                        string inferredType = InferWgslType(expr);

                        // Try valueVariables lookup for more accurate type
                        foreach (var kvp in valueVariables)
                        {
                            if (kvp.Value.Name == varName)
                            {
                                inferredType = kvp.Value.Type;
                                break;
                            }
                        }

                        missingDeclarations.Add($"    var {varName} : {inferredType};");
                        declaredVariables.Add(varName);
                    }

                    // Convert "let v_N = expr;" to "v_N = expr;" (assignment only)
                    return $"{indent}{varName} = {expr};";
                });

                if (anyReplacements)
                {
                    Builder.Remove(codeGenStartPosition, Builder.Length - codeGenStartPosition);

                    // Insert missing var declarations at the function scope (before generated code)
                    if (missingDeclarations.Count > 0)
                    {
                        foreach (var decl in missingDeclarations)
                            Builder.AppendLine(decl);
                    }

                    // Insert hoisted pointer-alias let declarations at function scope
                    if (hoistedLetDeclarations.Count > 0)
                    {
                        foreach (var decl in hoistedLetDeclarations)
                            Builder.AppendLine(decl);
                    }

                    Builder.Append(processed);
                }

                // Note: Barrier extraction post-processing is no longer needed.
                // With structured loop generation, barriers are emitted directly
                // in the loop body at the correct scope, ensuring uniform control flow.

            }

            PopIndent();
            Builder.AppendLine("}"); // End Function

            // --- NOW BUILD THE REAL SIGNATURE ---
            // Check the generated body to see if it actually uses subgroup builtins.
            // This is more reliable than the IR pre-scan which may miss inlined methods.
            string bodyText = Builder.ToString(signatureInsertPosition + signatureSentinel.Length + Environment.NewLine.Length,
                Builder.Length - signatureInsertPosition - signatureSentinel.Length - Environment.NewLine.Length);
            bool bodyUsesSubgroups = Backend.HasSubgroups &&
                (bodyText.Contains("subgroup_size") || bodyText.Contains("subgroup_invocation_id") ||
                 bodyText.Contains("subgroupShuffle") || bodyText.Contains("subgroupAdd") ||
                 bodyText.Contains("subgroupBroadcastFirst") || bodyText.Contains("subgroup_id"));

            // Include subgroup_id builtin when group reduce or subgroup ops are used.
            // subgroup_id is needed by GenerateGroupAllReduce to index the per-subgroup results array.
            bool bodyNeedsSubgroupId = bodyUsesSubgroups || _usesGroupReduce;

            string signature = $"@compute @workgroup_size({workgroupSize})\n" +
                "fn main(@builtin(global_invocation_id) global_id : vec3<u32>, " +
                "@builtin(local_invocation_id) local_id : vec3<u32>, " +
                "@builtin(workgroup_id) group_id : vec3<u32>, " +
                "@builtin(num_workgroups) num_workgroups : vec3<u32>, " +
                "@builtin(local_invocation_index) local_index : u32" +
                (bodyNeedsSubgroupId ? ", @builtin(subgroup_invocation_id) subgroup_invocation_id : u32, @builtin(subgroup_size) subgroup_size : u32, @builtin(subgroup_id) subgroup_id : u32" : "") +
                ") {";

            // Replace the sentinel with the real signature
            string fullOutput = Builder.ToString(signatureInsertPosition, Builder.Length - signatureInsertPosition);
            fullOutput = fullOutput.Replace(signatureSentinel, signature);
            Builder.Remove(signatureInsertPosition, Builder.Length - signatureInsertPosition);
            Builder.Append(fullOutput);
            // NOTE: The enable subgroups placeholder is resolved in WebGPUBackend.CreateKernel()
            // because the header is written to the upstream builder, not this.Builder.
        }
        private string GetPrefix(Value value)
        {
            // If the variable was hoisted to the top of main(), it's already a 'var'
            // and we must use a direct assignment (no prefix).
            // Otherwise, we use 'let ' to declare it locally.
            return _hoistedPrimitives.Contains(value) ? "" : "let ";
        }

        /// <summary>
        /// Infers the WGSL type from an expression string for post-processing var declarations.
        /// </summary>
        private static string InferWgslType(string expr)
        {
            // Comparison operators and emulated comparison functions produce bool
            if (expr.Contains("f64_lt") || expr.Contains("f64_le") || expr.Contains("f64_gt") ||
                expr.Contains("f64_ge") || expr.Contains("f64_eq") || expr.Contains("f64_ne") ||
                expr.Contains("i64_lt") || expr.Contains("i64_le") || expr.Contains("i64_gt") ||
                expr.Contains("i64_ge") || expr.Contains("i64_eq") || expr.Contains("i64_ne") ||
                expr.Contains("u64_lt") || expr.Contains("u64_le") || expr.Contains("u64_gt") ||
                expr.Contains("u64_ge"))
                return "bool";
            if (expr.Contains(" != ") || expr.Contains(" == ") || expr.Contains(" >= ") ||
                expr.Contains(" <= ") || expr.Contains(" > ") || expr.Contains(" < ") ||
                expr.Contains(" | ") || expr.Contains(" & "))
                return "bool";
            // emu_f64 emulation functions return vec2<f32>
            if (expr.Contains("f64_from_f32") || expr.Contains("f64_add") || expr.Contains("f64_sub") ||
                expr.Contains("f64_mul") || expr.Contains("f64_div") || expr.Contains("f64_neg") ||
                expr.Contains("f64_from_ieee754") || expr.Contains("emu_f64(") ||
                expr.Contains("f64_abs") || expr.Contains("f64_min") || expr.Contains("f64_max"))
                return "vec2<f32>";
            // emu_i64 emulation functions return vec2<u32>
            if (expr.Contains("i64_from_i32") || expr.Contains("u64_from_u32") ||
                expr.Contains("i64_add") || expr.Contains("i64_sub") || expr.Contains("i64_mul") ||
                expr.Contains("u64_mul") || expr.Contains("i64_neg") || expr.Contains("i64_abs") ||
                expr.Contains("i64_and") || expr.Contains("i64_or") || expr.Contains("i64_xor") ||
                expr.Contains("i64_shl") || expr.Contains("i64_shr") || expr.Contains("u64_shr") ||
                expr.Contains("i64_not") || expr.Contains("emu_i64(") || expr.Contains("emu_u64("))
                return "vec2<u32>";
            // Extraction functions return scalars
            if (expr.Contains("f64_to_f32") || expr.Contains("f32("))
                return "f32";
            if (expr.Contains("i64_to_i32"))
                return "i32";
            if (expr.Contains("u64_to_u32"))
                return "u32";
            if (expr.Contains("i32("))
                return "i32";
            if (expr.Contains("u32("))
                return "u32";
            // Default to bool (most common for unlisted patterns like comparisons)
            return "bool";
        }

        private string GetWorkgroupSize()
        {
            return EntryPoint.IndexType switch
            {
                IndexType.Index1D => "64",
                IndexType.Index2D => "8, 8",
                IndexType.Index3D => "4, 4, 4",
                _ => "64"
            };
        }

        /// <summary>
        /// Checks whether ANY value in the kernel IR (parameters AND internal values)
        /// contains the specified BasicValueType.
        /// This scans the full IR because intermediate values (e.g. Grid.GlobalIndex.X → Int64,
        /// loop counters, comparisons) generate 64-bit IR operations that require the
        /// emulation library even when no kernel parameter uses 64-bit types.
        /// </summary>
        private bool ContainsBasicValueType(BasicValueType targetType)
        {
            // First check parameters
            int paramOffset = KernelParamOffset;
            foreach (var param in Method.Parameters)
            {
                if (param.Index < paramOffset) continue;
                if (TypeContainsBasicValue(param.ParameterType, targetType, new HashSet<TypeNode>()))
                    return true;
            }
            // Then check ALL values in all blocks (catches intermediate Int64/Float64 from
            // Grid.GlobalIndex.X, loop counters, comparison results, etc.)
            foreach (var block in Method.Blocks)
            {
                foreach (var value in block)
                {
                    if (value.Value.Type is PrimitiveType pt && pt.BasicValueType == targetType)
                        return true;
                }
            }
            return false;
        }

        private bool TypeContainsBasicValue(TypeNode type, BasicValueType targetType, HashSet<TypeNode> visited)
        {
            if (type == null || !visited.Add(type)) return false;
            if (type is PrimitiveType pt)
                return pt.BasicValueType == targetType;
            if (type is ViewType vt)
                return TypeContainsBasicValue(vt.ElementType, targetType, visited);
            if (type is global::ILGPU.IR.Types.PointerType ptr)
                return TypeContainsBasicValue(ptr.ElementType, targetType, visited);
            if (type is StructureType st)
            {
                foreach (var field in st.Fields)
                {
                    if (TypeContainsBasicValue(field, targetType, visited))
                        return true;
                }
            }
            return false;
        }

        private TypeNode GetBufferElementType(TypeNode type)
        {
            var current = type;
            while (current != null)
            {
                if (current is ViewType viewType) return viewType.ElementType;
                if (current is global::ILGPU.IR.Types.PointerType ptrType) return ptrType.ElementType;
                if (current is global::ILGPU.IR.Types.StructureType structType)
                {
                    // Refined Logic:
                    // 1. If this is a Wrapper Struct (like ArrayView2D/3D), its first field is the underlying View.
                    //    We MUST drill down to get the primitive element type (e.g., float).
                    // 2. If this is a Data Struct (user-defined struct), it is the element type itself.
                    //    We MUST return it as-is so we get 'array<MyStruct>'.

                    if (structType.NumFields > 0 && (structType.Fields[0] is ViewType || structType.Fields[0].ToString().Contains("View")))
                    {
                        current = structType.Fields[0];
                        continue;
                    }

                    return structType;
                }
                break;
            }
            return current; // Return whatever we resolved to
        }
        // Add this to your Properties region
        private HashSet<Value> _hoistedIndexFields = new HashSet<Value>();

        private void SetupIndexVariables()
        {
            // If the kernel is explicitly grouped (Shared Memory / Advanced), the user handles indices via Group.* intrinsics
            // However, we still need to map the main Kernel Index parameter if it exists.

            if (Method.Parameters.Count == 0) return;

            var indexParam = Method.Parameters[0];

            // Only map if strictly implicit OR if we detected an IndexType
            if (KernelParamOffset == 0) return;

            var indexVar = Allocate(indexParam);
            _hoistedIndexFields.Clear();

            // 1D Kernel
            if (EntryPoint.IndexType == IndexType.Index1D)
            {
                // Map to global linear index: local_index + group_id.x * workgroup_size.x
                AppendLine($"var {indexVar.Name} : i32 = i32(local_index + group_id.x * workgroup_size.x);");
            }
            // 2D Kernel
            else if (EntryPoint.IndexType == IndexType.Index2D)
            {
                // Map global_id.xy to vec2<i32>
                AppendLine($"var {indexVar.Name} : vec2<i32> = vec2<i32>(i32(global_id.x), i32(global_id.y));");

                // Handle struct field access (index.X, index.Y) by pre-calculating them
                // This prevents "GetField" later from trying to access a struct field on a vec2
                foreach (var use in indexParam.Uses)
                {
                    if (use.Target is global::ILGPU.IR.Values.GetField gf)
                    {
                        var componentVar = Allocate(gf);
                        _hoistedIndexFields.Add(gf);
                        declaredVariables.Add(componentVar.Name); // Ensure it's marked as declared

                        string comp = gf.FieldSpan.Index == 0 ? "x" : "y";
                        AppendLine($"var {componentVar.Name} : i32 = {indexVar.Name}.{comp};"); // Use vector component
                    }
                }
            }
            // 3D Kernel
            else if (EntryPoint.IndexType == IndexType.Index3D)
            {
                // Map global_id.xyz to vec3<i32>
                AppendLine($"var {indexVar.Name} : vec3<i32> = vec3<i32>(i32(global_id.x), i32(global_id.y), i32(global_id.z));");

                foreach (var use in indexParam.Uses)
                {
                    if (use.Target is global::ILGPU.IR.Values.GetField gf)
                    {
                        var componentVar = Allocate(gf);
                        _hoistedIndexFields.Add(gf);
                        declaredVariables.Add(componentVar.Name);

                        string comp = gf.FieldSpan.Index == 0 ? "x" : (gf.FieldSpan.Index == 1 ? "y" : "z");
                        AppendLine($"var {componentVar.Name} : i32 = {indexVar.Name}.{comp};");
                    }
                }
            }
            // LongIndex1D Kernel (grid-stride kernels, e.g. from LoadGridStrideKernel/accelerator.Reduce)
            // LongIndex1D is an emulated i64 (vec2<u32>) representing the 64-bit thread index.
            else if (EntryPoint.IndexType == IndexType.LongIndex1D)
            {
                // Map to emulated i64: pack the 32-bit global linear index into the low word
                AppendLine($"var {indexVar.Name} : vec2<u32> = vec2<u32>(u32(local_index) + u32(group_id.x) * u32(workgroup_size.x), 0u); // LongIndex1D (emu_i64)");
            }
            // LongIndex2D Kernel
            else if (EntryPoint.IndexType == IndexType.LongIndex2D)
            {
                // Map to pair of emulated i64 (vec2<u32> per dimension)
                // This is rare - map to a struct with two emu_i64 fields for X and Y
                AppendLine($"var {indexVar.Name}_x : vec2<u32> = vec2<u32>(u32(global_id.x), 0u); // LongIndex2D X");
                AppendLine($"var {indexVar.Name}_y : vec2<u32> = vec2<u32>(u32(global_id.y), 0u); // LongIndex2D Y");
                // Note: LongIndex2D field accesses may still need handling in GetField
            }
            // LongIndex3D Kernel
            else if (EntryPoint.IndexType == IndexType.LongIndex3D)
            {
                AppendLine($"var {indexVar.Name}_x : vec2<u32> = vec2<u32>(u32(global_id.x), 0u); // LongIndex3D X");
                AppendLine($"var {indexVar.Name}_y : vec2<u32> = vec2<u32>(u32(global_id.y), 0u); // LongIndex3D Y");
                AppendLine($"var {indexVar.Name}_z : vec2<u32> = vec2<u32>(u32(global_id.z), 0u); // LongIndex3D Z");
            }
            else
            {
                // Fallback: Default to 1D if we skipped the param but didn't match specific types
                // This protects against weird IndexType states
                AppendLine($"var {indexVar.Name} : i32 = i32(global_id.x); // Fallback mapping (IndexType={EntryPoint.IndexType})");
            }
        }


        /// <summary>
        /// Pre-scans parameters to identify which buffers use emulated 64-bit types.
        /// This MUST run before HoistCrossBlockVariables and body generation so that
        /// LoadElementAddress and Load/Store can check if a param needs emulation.
        /// </summary>
        private void PreScanEmulatedParameters()
        {
            int paramOffset = KernelParamOffset;
            foreach (var param in Method.Parameters)
            {
                if (param.Index < paramOffset) continue;

                var elementType = GetBufferElementType(param.ParameterType);
                var wgslType = TypeGenerator[elementType];

                if (Backend.Options.EnableF64Emulation && wgslType == "emu_f64")
                {
                    _emulatedF64Params.Add(param.Index);
                }
                else if (Backend.Options.EnableI64Emulation && (wgslType == "emu_i64" || wgslType == "emu_u64"))
                {
                    _emulatedI64Params.Add(param.Index);
                }
            }
        }

        private void HoistCrossBlockVariables()
        {
            var defBlocks = new Dictionary<Value, BasicBlock>();
            _hoistedPrimitives.Clear(); // Initialize the restored set
            _crossBlockPointers.Clear();
            _crossBlockPointerExprs.Clear();

            foreach (var block in Method.Blocks)
                foreach (var value in block)
                    defBlocks[value.Value] = block;

            foreach (var block in Method.Blocks)
            {
                foreach (var value in block)
                {
                    // CRITICAL: Skip NewView to prevent hoisting pointer logic
                    if (value.Value is global::ILGPU.IR.Values.NewView) continue;

                    if (value.Value is PhiValue)
                    {
                        _hoistedPrimitives.Add(value.Value);
                    }

                    // FIX: Hoist UnaryArithmeticValue and BinaryArithmeticValue results
                    // These are math intrinsics (Abs, Min, Max, etc.) that may be reused
                    // across branches in structured control flow, causing scope issues
                    if (value.Value is global::ILGPU.IR.Values.UnaryArithmeticValue ||
                        value.Value is global::ILGPU.IR.Values.BinaryArithmeticValue)
                    {
                        if (!value.Value.Type.IsPointerType && !value.Value.Type.IsVoidType)
                        {
                            _hoistedPrimitives.Add(value.Value);
                        }
                    }

                    foreach (var use in value.Value.Uses)
                    {
                        if (defBlocks.TryGetValue(use.Target, out var defBlock) && defBlock != block)
                        {
                            // CRITICAL FIX: Skip hoisting values whose IR type is ViewType.
                            // ViewType values (ArrayView references extracted via GetField from
                            // parameter structs) will be aliased to buffer bindings by
                            // GenerateCode(LoadElementAddress) which emits "let v_X = &paramY[offset];".
                            // Hoisting them as "var v_X : <element_type> = 0;" causes a redeclaration
                            // conflict because TypeGenerator maps ViewType to its element type.
                            bool isViewType = value.Value.Type is global::ILGPU.IR.Types.ViewType;

                            if (!value.Value.Type.IsPointerType && !value.Value.Type.IsVoidType && !isViewType)
                            {
                                _hoistedPrimitives.Add(value.Value);
                            }
                            // Track cross-block LoadElementAddress pointers for inline substitution
                            else if (value.Value.Type.IsPointerType && value.Value is LoadElementAddress)
                            {
                                _crossBlockPointers.Add(value.Value);
                            }
                        }
                    }
                }
            }

            AppendLine("// HOIST-FIX-V2-ACTIVE");
            foreach (var val in _hoistedPrimitives)
            {
                var variable = Allocate(val);
                var wgslType = TypeGenerator[val.Type];
                var basicType = val.Type.BasicValueType;


                string init = " = 0";
                if (basicType == BasicValueType.Int1) init = " = false";
                else if (basicType == BasicValueType.Float16 || basicType == BasicValueType.Float32) init = " = 0.0";
                else if (basicType == BasicValueType.Float64)
                {
                    // emu_f64 is emulated as vec2<f32> when emulation is enabled
                    if (wgslType == "emu_f64")
                    {
                        if (Backend.Options.UseOzakiF64Emulation)
                            init = " = emu_f64(0.0, 0.0, 0.0, 0.0)";
                        else
                            init = " = emu_f64(0.0, 0.0)";
                    }
                    else
                        init = " = 0.0";
                }
                else if (basicType == BasicValueType.Int64)
                {
                    // emu_i64 is emulated as vec2<u32> when emulation is enabled
                    if (wgslType == "emu_i64" || wgslType == "emu_u64")
                        init = " = emu_i64(0u, 0u)";
                    else
                        init = " = 0";
                }

                AppendLine($"var {variable.Name} : {wgslType}{init};");
                declaredVariables.Add(variable.Name);
            }
        }

        void IsMultiDim(TypeNode ParameterType, out bool isMultiDim, out bool isView, out bool is1DView, out bool is2DView, out bool is3DView)
        {
            is1DView = false;
            is2DView = false;
            is3DView = false;
            isMultiDim = false;
            isView = false;

            // 0. Direct ViewType Check — for stream kernels where the IR type is View<T> directly
            if (ParameterType is ViewType)
            {
                isView = true;
                is1DView = true;
                return;
            }

            string typeName = ParameterType.ToString();

            // 1. Direct String Check (works for C# types if available, but unreliable for IR strings)
            bool isStringView = typeName.Contains("ArrayView");

            // 2. Structural Check on ParameterType (The Wrapper Struct)
            if (ParameterType is global::ILGPU.IR.Types.StructureType st)
            {
                // Check if it looks like an ArrayView wrapper (Field 0 is View)
                if (st.NumFields > 0 && (st.Fields[0] is ViewType || st.Fields[0].ToString().Contains("View")))
                {
                    isView = true;

                    if (st.NumFields == 6 || st.NumFields == 4) // 3D (6 fields usually), 2D (4 fields)
                    {
                        isMultiDim = true;
                        if (st.NumFields == 6) is3DView = true;
                        else is2DView = true;
                    }
                    else if (st.NumFields == 3)
                    {
                        is1DView = true;
                        isMultiDim = false; // 1D doesn't need stride buffers, so we treat as "not multi dim" for stride purposes
                    }
                }
            }

            // 3. Fallback: Combine String check with Structural properties
            if (!isView && isStringView)
            {
                isView = true;
                if (typeName.Contains("2D") || typeName.Contains("3D")) isMultiDim = true;
                if (typeName.Contains("3D")) is3DView = true;
                else if (typeName.Contains("2D")) is2DView = true;
                else if (typeName.Contains("1D")) is1DView = true;
            }
        }

        private void SetupParameterBindings()
        {
            // NOTE: GenerateHeader() runs AFTER body generation in ILGPU's pipeline,
            // so _scalarManifest is not yet populated. We must replicate the same
            // classification logic here to determine which params are packed scalars.
            // Both this method and GenerateHeader iterate Method.Parameters in the same
            // order, so the slot offsets will match.

            int paramOffset = KernelParamOffset;

            // First pass: classify all params and compute slot offsets for packed scalars
            var packedScalarSlots = new Dictionary<int, (int slot, int slotCount, string wgslType, bool isEmuF64, bool isEmuI64)>();
            int scalarSlotOffset = 0;

            foreach (var param in Method.Parameters)
            {
                if (param.Index < paramOffset) continue;

                // Skip body struct params — they're handled separately in _bodyStructParams
                if (_bodyStructParams.ContainsKey(param.Index)) continue;

                IsMultiDim(param.ParameterType, out var isMultiDim, out var isView, out var is1DView, out var is2DView, out var is3DView);
                bool isStruct = param.ParameterType is global::ILGPU.IR.Types.StructureType && !isView;
                bool isAtomic = _atomicParameters.Contains(param.Index);

                // Does this param get its own binding? (same logic as GenerateHeader)
                bool ownBinding = isView || isStruct || isAtomic;

                if (!ownBinding)
                {
                    // This is a packed scalar
                    var elementType = GetBufferElementType(param.ParameterType);
                    var wgslType = TypeGenerator[elementType];

                    bool isEmuF64 = Backend.Options.EnableF64Emulation && wgslType == "emu_f64";
                    bool isEmuI64 = Backend.Options.EnableI64Emulation && (wgslType == "emu_i64" || wgslType == "emu_u64");

                    int slotCount = (isEmuF64 || isEmuI64) ? 2 : 1;
                    packedScalarSlots[param.Index] = (scalarSlotOffset, slotCount, wgslType, isEmuF64, isEmuI64);
                    scalarSlotOffset += slotCount;
                }
            }

            // Second pass: emit variable declarations
            foreach (var param in Method.Parameters)
            {
                if (param.Index < paramOffset) continue;
                var variable = Allocate(param);

                IsMultiDim(param.ParameterType, out var isMultiDim, out var isView, out var is1DView, out var is2DView, out var is3DView);

                // --- BODY STRUCT: create individual field variables ---
                if (_bodyStructParams.TryGetValue(param.Index, out var fieldInfos))
                {
                    // The param variable itself is allocated but not used directly.
                    // Instead, create individual variables for each field.
                    for (int fi = 0; fi < fieldInfos.Count; fi++)
                    {
                        var fieldInfo = fieldInfos[fi];
                        string fieldVarName = $"{variable.Name}_f{fi}";
                        _bodyStructFieldVars[(param.Index, fi)] = fieldVarName;
                        declaredVariables.Add(fieldVarName);

                        if (fieldInfo.IsView)
                        {
                            // View field: create a pointer to the view buffer (a 'let' pointer alias)
                            // Track the var name as a pointer alias so the post-processor won't
                            // convert 'let v_28 = v_27_f0;' to 'v_28 = v_27_f0;' (pointer→i32 error).
                            _viewPointerVarNames.Add(fieldVarName);
                            AppendLine($"let {fieldVarName} = &{fieldInfo.BindingName};");
                        }
                        else if (fieldInfo.IsViewMetadata)
                        {
                            // View metadata field: use arrayLength() for length, 0 for flags
                            if (fieldInfo.IsLengthField)
                            {
                                // Length field: use arrayLength(&bindingName) cast to i32
                                // The IR type is Int64, but we use i32 since WGSL doesn't have i64 natively
                                var scalarElemType = GetBufferElementType(fieldInfo.FieldType);
                                var scalarWgslType = TypeGenerator[scalarElemType];
                                bool isEmuI64 = Backend.Options.EnableI64Emulation && (scalarWgslType == "emu_i64" || scalarWgslType == "emu_u64");
                                if (isEmuI64)
                                    AppendLine($"var {fieldVarName} = i64_from_i32(i32(arrayLength(&{fieldInfo.AssociatedViewBindingName})));");
                                else
                                    AppendLine($"var {fieldVarName} = i32(arrayLength(&{fieldInfo.AssociatedViewBindingName}));");
                            }
                            else
                            {
                                // Flag field: use 0 (unused in WGSL)
                                AppendLine($"var {fieldVarName} = 0;");
                            }
                        }
                        else
                        {
                            // User scalar field: read from _scalar_params
                            int slot = fieldInfo.ScalarSlot;
                            var scalarElemType = GetBufferElementType(fieldInfo.FieldType);
                            var scalarWgslType = TypeGenerator[scalarElemType];
                            bool isEmuF64 = Backend.Options.EnableF64Emulation && scalarWgslType == "emu_f64";
                            bool isEmuI64 = Backend.Options.EnableI64Emulation && (scalarWgslType == "emu_i64" || scalarWgslType == "emu_u64");

                            if (isEmuF64)
                                AppendLine($"var {fieldVarName} = f64_from_ieee754_bits(_scalar_params[{slot}], _scalar_params[{slot + 1}]);");
                            else if (isEmuI64)
                                // Use the correct type constructor (emu_i64 or emu_u64) so the WGSL type
                                // tracker assigns the right signedness for subsequent comparisons.
                                // Using emu_i64 for ulong would cause unsigned comparisons (u64_lt) to fall
                                // back to signed (i64_lt), breaking MinUInt64 (0xFFFF... as signed = -1).
                                AppendLine($"var {fieldVarName} = {scalarWgslType}(_scalar_params[{slot}], _scalar_params[{slot + 1}]);");
                            else if (scalarWgslType == "u32")
                                AppendLine($"var {fieldVarName} = _scalar_params[{slot}];");
                            else if (scalarWgslType == "i32")
                                AppendLine($"var {fieldVarName} = bitcast<i32>(_scalar_params[{slot}]);");
                            else if (scalarWgslType == "f32")
                                AppendLine($"var {fieldVarName} = bitcast<f32>(_scalar_params[{slot}]);");
                            else
                                AppendLine($"var {fieldVarName} = bitcast<{scalarWgslType}>(_scalar_params[{slot}]);");
                        }
                    }
                    continue; // Skip normal variable declaration for this param
                }


                // Check if this param is packed
                if (packedScalarSlots.TryGetValue(param.Index, out var packInfo))
                {
                    // --- PACKED SCALAR: read from _scalar_params buffer ---
                    int slot = packInfo.slot;

                    if (packInfo.isEmuF64)
                    {
                        AppendLine($"var {variable.Name} = f64_from_ieee754_bits(_scalar_params[{slot}], _scalar_params[{slot + 1}]);");
                    }
                    else if (packInfo.isEmuI64)
                    {
                        AppendLine($"var {variable.Name} = emu_i64(_scalar_params[{slot}], _scalar_params[{slot + 1}]);");
                    }
                    else
                    {
                        // Standard scalar: bitcast from u32 to the target type
                        string wgslType = packInfo.wgslType;
                        if (wgslType == "u32")
                        {
                            AppendLine($"var {variable.Name} = _scalar_params[{slot}];");
                        }
                        else if (wgslType == "i32")
                        {
                            AppendLine($"var {variable.Name} = bitcast<i32>(_scalar_params[{slot}]);");
                        }
                        else if (wgslType == "f32")
                        {
                            AppendLine($"var {variable.Name} = bitcast<f32>(_scalar_params[{slot}]);");
                        }
                        else
                        {
                            AppendLine($"var {variable.Name} = bitcast<{wgslType}>(_scalar_params[{slot}]);");
                        }
                    }
                    continue;
                }

                // --- NON-PACKED: original binding logic ---
                if (!isView)
                {
                    bool isAtomic = _atomicParameters.Contains(param.Index);
                    bool isStruct = param.ParameterType is global::ILGPU.IR.Types.StructureType;

                    if (isAtomic)
                    {
                        AppendLine($"let {variable.Name} = &param{param.Index};");
                    }
                    else if (isStruct)
                    {
                        if (isMultiDim)
                        {
                            AppendLine($"let {variable.Name} = &param{param.Index};");
                        }
                        else
                        {
                            AppendLine($"let {variable.Name} = &param{param.Index}[0];");
                        }
                    }
                    else
                    {
                        // Non-packed scalar that has its own binding (shouldn't normally happen,
                        // but kept for safety/fallback)
                        if (_emulatedF64Params.Contains(param.Index))
                        {
                            AppendLine($"var {variable.Name} = f64_from_ieee754_bits(param{param.Index}[0], param{param.Index}[1]);");
                        }
                        else if (_emulatedI64Params.Contains(param.Index))
                        {
                            AppendLine($"var {variable.Name} = emu_i64(param{param.Index}[0], param{param.Index}[1]);");
                        }
                        else
                        {
                            AppendLine($"var {variable.Name} = param{param.Index}[0];");
                        }
                    }
                }
                else
                {
                    AppendLine($"let {variable.Name} = &param{param.Index};");

                    if (isView && isMultiDim && (is2DView || is3DView))
                    {
                        AppendLine($"let {variable.Name}_stride = &param{param.Index}_stride;");
                    }
                }
            }
        }

        public override void GenerateCode(global::ILGPU.IR.Values.NewView value)
        {
            var target = Load(value);
            var source = Load(value.Pointer);

            // NewView result is strictly a pointer (reference) in WGSL
            string refPrefix = "";
            if (value.Pointer.Type is global::ILGPU.IR.Types.PointerType ptrType &&
                ptrType.AddressSpace == MemoryAddressSpace.Shared)
            {
                refPrefix = "&";
            }

            // We use 'let' to alias the pointer, ensuring we don't copy the array
            // Optimization: If source is already a pointer (likely), we might not need &
            // But for Shared Memory (var<workgroup> arr), it's treated as value, so we need &

            declaredVariables.Add(target.Name);

            if (IsStateMachineActive)
            {
                // Hoist let declarations to function scope (VariableBuilder) so they
                // are accessible across all case blocks in the state machine.
                // Without this, the let has block scope inside a case and becomes
                // "unresolved value" in other case blocks that reference it.
                VariableBuilder.AppendLine($"    let {target.Name} = {refPrefix}{source};");
            }
            else
            {
                AppendLine($"let {target.Name} = {refPrefix}{source};");
            }
        }

        public override void GenerateCode(DynamicMemoryLengthValue value)
        {
            var target = Load(value);
            Declare(target);

            // Use the override constant for dynamic shared memory length
            // The override constant is declared as u32 but DynamicMemoryLengthValue is i32, so cast
            if (_dynamicSharedOverrides.Count > 0)
            {
                // For now, all dynamic shared allocations share the same size from SharedMemoryConfig
                // Use the first (and typically only) override constant
                var overrideName = _dynamicSharedOverrides[0].ConstantName;
                AppendLine($"{target} = i32({overrideName}); // dynamic memory length from override constant");
            }
            else
            {
                // Fallback: no dynamic shared memory overrides registered
                AppendLine($"{target} = 0; // dynamic memory length (no override available)");
            }
        }

        public override void GenerateCode(Parameter parameter) { }

        public override void GenerateCode(LoadElementAddress value)
        {
            var target = Load(value);
            var offset = Load(value.Offset);

            // When i64 emulation is active, Int64 indices become emu_i64 (vec2<u32>)
            // which cannot be used as WGSL array indices. Wrap with i64_to_i32().
            bool offsetIsEmulatedI64 = Backend.Options.EnableI64Emulation
                && value.Offset.Resolve().BasicValueType == BasicValueType.Int64;
            string offsetExpr = offsetIsEmulatedI64 ? $"i64_to_i32({offset})" : $"{offset}";

            // --- BODY STRUCT VIEW FIELD ACCESS ---
            // When LoadElementAddress is called on a GetField result from a body struct parameter,
            // redirect to the correct per-field binding (e.g. param1_f0[offset]).
            // This handles: GetField(bodyStruct, viewFieldIdx) → LoadElementAddress(viewField, offset)
            if (value.Source.Resolve() is global::ILGPU.IR.Values.GetField gf
                && gf.ObjectValue.Resolve() is global::ILGPU.IR.Values.Parameter bsParam
                && _bodyStructParams.TryGetValue(bsParam.Index, out var bsFields))
            {
                int fieldIdx = gf.FieldSpan.Index;
                if (fieldIdx < bsFields.Count && bsFields[fieldIdx].IsView)
                {
                    string bindingName = bsFields[fieldIdx].BindingName;
                    var elemType = GetBufferElementType(bsFields[fieldIdx].FieldType);
                    string fieldTypeStr = TypeGenerator[elemType];
                    
                    bool isEmuF64Field = Backend.Options.EnableF64Emulation && fieldTypeStr == "emu_f64";
                    bool isEmuI64Field = Backend.Options.EnableI64Emulation && (fieldTypeStr == "emu_i64" || fieldTypeStr == "emu_u64");

                    if (isEmuF64Field || isEmuI64Field)
                    {
                        _emulatedVarMappings[target.Name] = (bsParam.Index, isEmuF64Field);
                        
                        AppendIndent();
                        Builder.Append($"let {target.Name}_base_idx = i32({offsetExpr}) * 2;");
                        Builder.AppendLine();
                        
                        AppendIndent();
                        Builder.Append($"let {target.Name} = &{bindingName};");
                        Builder.AppendLine();
                        return;
                    }

                    if (_crossBlockPointers.Contains(value))
                    {
                        _crossBlockPointerExprs[target.Name] = $"{bindingName}[{offsetExpr}]";
                        AppendLine($"let {target.Name} = &{bindingName}[{offsetExpr}];");
                    }
                    else
                    {
                        AppendLine($"let {target.Name} = &{bindingName}[{offsetExpr}];");
                    }
                    return;
                }
            }
            if (ResolveToParameter(value.Source) is global::ILGPU.IR.Values.Parameter param)
            {
                int paramOffset = KernelParamOffset;
                if (param.Index >= paramOffset)
                {
                    // Check if this is an emulated 64-bit buffer
                    if (_emulatedF64Params.Contains(param.Index) || _emulatedI64Params.Contains(param.Index))
                    {
                        bool isF64 = _emulatedF64Params.Contains(param.Index);
                        // Register for Load/Store to know this needs conversion
                        _emulatedVarMappings[target.Name] = (param.Index, isF64);

                        // For emulated buffers, we need to address 2 u32 per element
                        // Store the base index (multiplied by 2) for later use in Load/Store
                        AppendIndent();
                        Builder.Append($"let {target.Name}_base_idx = i32({offsetExpr}) * 2;");
                        Builder.AppendLine();
                        // Also create an alias for compatibility
                        AppendIndent();
                        Builder.Append($"let {target.Name} = &param{param.Index};");
                        Builder.AppendLine();
                        return;
                    }

                    // CROSS-BLOCK POINTER FIX:
                    // If this LoadElementAddress is used in a different switch-case block,
                    // its `let v_X = &paramN[offset]` will be out of scope at the use site.
                    // Instead, register an inline expression so Load/Store can substitute it.
                    if (_crossBlockPointers.Contains(value))
                    {
                        // Register the inline array access expression
                        _crossBlockPointerExprs[target.Name] = $"param{param.Index}[{offsetExpr}]";
                        // Still emit a local declaration for same-block uses
                        AppendIndent();
                        Builder.Append($"let {target.Name} = &param{param.Index}[{offsetExpr}];");
                        Builder.AppendLine();
                        return;
                    }

                    AppendIndent();
                    Builder.Append($"let {target.Name} = &param{param.Index}[{offsetExpr}];");
                    Builder.AppendLine();
                    return;
                }
            }

            var sourceVal = Load(value.Source);
            AppendIndent();
            Builder.Append($"let {target.Name} = ");

            // POINTER DEREFERENCE LOGIC
            // If the source comes from NewView, it's a pointer (let v_3 = &v_4).
            // To access element at offset: (*ptr)[offset] -> &(*ptr)[offset] gives the address
            if (value.Source is global::ILGPU.IR.Values.NewView || value.Source.Type.IsPointerType)
            {
                Builder.Append($"&(*{sourceVal})[{offsetExpr}];");
            }
            else
            {
                Builder.Append($"&{sourceVal}[{offsetExpr}];");
            }
            Builder.AppendLine();
        }

        public override void GenerateCode(global::ILGPU.IR.Values.Load loadVal)
        {
            var target = Load(loadVal);
            var source = Load(loadVal.Source);

            // Special handling for loading an ArrayView parameter (which is a struct).
            // We cannot 'load' the struct from the buffer binding (which is array<T>).
            // Instead, we treat the 'loaded' value as a pointer/alias to the binding.

            // CRITICAL FIX: Only alias if the source IS the parameter node itself.
            // Do NOT alias if the source is a derived pointer (e.g. LoadElementAddress), 
            // as that would prevent loading actual data values.
            // Note: loadVal.Source is a ValueReference struct. To pattern match against Value types (classes),
            // we must use .Resolve() to get the underlying Value object.
            if (loadVal.Source.Resolve() is global::ILGPU.IR.Values.Parameter param &&
                param.Type is global::ILGPU.IR.Types.ViewType)
            {
                // Verify index (skip implicit kernel index if any)
                int paramOffset = KernelParamOffset;
                if (param.Index >= paramOffset)
                {
                    // Alias: let v_X = &paramY;
                    AppendLine($"let {target} = &param{param.Index};");
                    return;
                }
            }

            // Check for emulated 64-bit buffer access
            if (_emulatedVarMappings.TryGetValue(source.ToString(), out var emulInfo))
            {
                // This is loading from an emulated buffer - need to read 2 u32 values and convert
                string baseIdxVar = $"{source}_base_idx";
                if (emulInfo.IsF64)
                {
                    // emu_f64: convert IEEE 754 bits to double-float
                    Declare(target);
                    AppendLine($"{target} = f64_from_ieee754_bits((*{source})[u32({baseIdxVar})], (*{source})[u32({baseIdxVar}) + 1u]);");
                }
                else
                {
                    // emu_i64: just combine two u32 into vec2<u32>
                    Declare(target);
                    AppendLine($"{target} = emu_i64((*{source})[u32({baseIdxVar})], (*{source})[u32({baseIdxVar}) + 1u]);");
                }
                return;
            }

            // CROSS-BLOCK POINTER FIX: Use inline expression instead of out-of-scope pointer
            if (_crossBlockPointerExprs.TryGetValue(source.ToString(), out var inlineExpr))
            {
                if (_hoistedPrimitives.Contains(loadVal))
                {
                    AppendLine($"{target} = {inlineExpr};");
                }
                else
                {
                    Declare(target);
                    AppendLine($"{target} = {inlineExpr};");
                }
                return;
            }

            // Check if we already declared this at the top
            if (_hoistedPrimitives.Contains(loadVal))
            {
                AppendLine($"{target} = *{source};");
            }
            else
            {
                // Fallback to your stable declaration logic
                Declare(target);
                AppendLine($"{target} = *{source};");
            }
        }

        public override void GenerateCode(global::ILGPU.IR.Values.Store storeVal)
        {
            var address = Load(storeVal.Target);
            var val = Load(storeVal.Value);

            // Check for emulated 64-bit buffer store (Parameter arrays registered earlier)
            if (_emulatedVarMappings.TryGetValue(address.ToString(), out var emulInfo))
            {
                string baseIdxVar = $"{address}_base_idx";
                if (emulInfo.IsF64)
                {
                    // emu_f64: convert double-float back to IEEE 754 bits
                    AppendLine($"let _bits_{address} = f64_to_ieee754_bits({val});");
                    AppendLine($"(*{address})[u32({baseIdxVar})] = _bits_{address}.x;");
                    AppendLine($"(*{address})[u32({baseIdxVar}) + 1u] = _bits_{address}.y;");
                }
                else
                {
                    // emu_i64: split vec2<u32> into two u32 values
                    AppendLine($"(*{address})[u32({baseIdxVar})] = {val}.x;");
                    AppendLine($"(*{address})[u32({baseIdxVar}) + 1u] = {val}.y;");
                }
                return;
            }

            // CROSS-BLOCK POINTER FIX: Use inline expression instead of out-of-scope pointer
            if (_crossBlockPointerExprs.TryGetValue(address.ToString(), out var inlineExpr))
            {
                AppendLine($"{inlineExpr} = {val};");
                return;
            }

            AppendLine($"*{address} = {val};");
        }

        public override void GenerateCode(global::ILGPU.IR.Values.GenericAtomic value)
        {
            string ptrStr = Load(value.Target).ToString();

            if (_emulatedVarMappings.TryGetValue(ptrStr, out var emulInfo))
            {
                 var target = Load(value);
                 var val = Load(value.Value);
                 Declare(target);

                 string valWgslType = TypeGenerator[value.Value.Type];
                 
                 string prefix = valWgslType switch
                 {
                     "emu_f64" => "f64",
                     "emu_u64" => "u64",
                     "emu_i64" => value.IsUnsigned ? "u64" : "i64",
                     _ => value.IsUnsigned ? "u64" : "i64"
                 };

                 string emuOp = value.Kind switch
                 {
                     global::ILGPU.IR.Values.AtomicKind.Max => $"{prefix}_max",
                     global::ILGPU.IR.Values.AtomicKind.Min => $"{prefix}_min",
                     global::ILGPU.IR.Values.AtomicKind.Add => $"{prefix}_add",
                     global::ILGPU.IR.Values.AtomicKind.Exchange => null,
                     _ => $"{prefix}_add"
                 };

                 string baseIdxVar = $"{ptrStr}_base_idx";

                 if (emuOp != null)
                 {
                     string oldVar = $"_emu64_old_{target.Name}";
                     if (valWgslType == "emu_f64")
                     {
                         string rawOldVarX = $"_raw_old_x_{target.Name}";
                         string rawOldVarY = $"_raw_old_y_{target.Name}";
                         string resultVar = $"_result_{target.Name}";
                         string ieeeBitsVar = $"_ieee_bits_{target.Name}";
                         
                         AppendLine($"let {rawOldVarX} = (*{ptrStr})[u32({baseIdxVar})];");
                         AppendLine($"let {rawOldVarY} = (*{ptrStr})[u32({baseIdxVar}) + 1u];");
                         AppendLine($"let {oldVar} = f64_from_ieee754_bits(bitcast<u32>({rawOldVarX}), bitcast<u32>({rawOldVarY}));");
                         AppendLine($"let {resultVar} = {emuOp}({oldVar}, {val});");
                         AppendLine($"let {ieeeBitsVar} = f64_to_ieee754_bits({resultVar});");
                         AppendLine($"(*{ptrStr})[u32({baseIdxVar})] = {ieeeBitsVar}.x;");
                         AppendLine($"(*{ptrStr})[u32({baseIdxVar}) + 1u] = {ieeeBitsVar}.y;");
                         AppendLine($"{target} = {oldVar};");
                     }
                     else
                     {
                         // emu_i64 / emu_u64
                         string resultVar = $"_result_{target.Name}";
                         AppendLine($"let {oldVar} = {valWgslType}((*{ptrStr})[u32({baseIdxVar})], (*{ptrStr})[u32({baseIdxVar}) + 1u]);");
                         AppendLine($"let {resultVar} = {emuOp}({oldVar}, {val});");
                         AppendLine($"(*{ptrStr})[u32({baseIdxVar})] = {resultVar}.x;");
                         AppendLine($"(*{ptrStr})[u32({baseIdxVar}) + 1u] = {resultVar}.y;");
                         AppendLine($"{target} = {oldVar};");
                     }
                 }
                 else
                 {
                     if (valWgslType == "emu_f64")
                     {
                         string ieeeBitsVar = $"_ieee_bits_{target.Name}";
                         AppendLine($"let {ieeeBitsVar} = f64_to_ieee754_bits({val});");
                         AppendLine($"(*{ptrStr})[u32({baseIdxVar})] = {ieeeBitsVar}.x;");
                         AppendLine($"(*{ptrStr})[u32({baseIdxVar}) + 1u] = {ieeeBitsVar}.y;");
                     }
                     else
                     {
                         AppendLine($"(*{ptrStr})[u32({baseIdxVar})] = {val}.x;");
                         AppendLine($"(*{ptrStr})[u32({baseIdxVar}) + 1u] = {val}.y;");
                     }
                     AppendLine($"{target} = {val};");
                 }
                 return;
            }
            
            base.GenerateCode(value);
        }



        public override void GenerateCode(BinaryArithmeticValue value)
        {
            var target = Load(value);
            var left = Load(value.Left);
            var right = Load(value.Right);
            string prefix = GetPrefix(value);

            // Check if this is an emulated 64-bit operation
            var leftType = TypeGenerator[value.Left.Type];
            var rightType = TypeGenerator[value.Right.Type];
            bool isEmulatedF64 = Backend.Options.EnableF64Emulation && (leftType == "emu_f64" || rightType == "emu_f64");
            bool isEmulatedI64 = Backend.Options.EnableI64Emulation && (leftType == "emu_i64" || leftType == "emu_u64" || rightType == "emu_i64" || rightType == "emu_u64");
            
            // Fallback: check BasicValueType when emulation options are enabled.
            // TypeGenerator may not return "emu_i64" for IR intermediate values (e.g., reduce accumulators).
            if (!isEmulatedF64 && !isEmulatedI64)
            {
                if (Backend.Options.EnableI64Emulation && value.BasicValueType == BasicValueType.Int64)
                {
                    isEmulatedI64 = true;
                    leftType = value.IsUnsigned ? "emu_u64" : "emu_i64";
                }
                else if (Backend.Options.EnableF64Emulation && value.BasicValueType == BasicValueType.Float64)
                {
                    isEmulatedF64 = true;
                    leftType = "emu_f64";
                }
            }
            
            // DIAGNOSTIC: Log all Max/Min operations to trace 64-bit detection
            if (value.Kind == BinaryArithmeticKind.Max || value.Kind == BinaryArithmeticKind.Min)
            {
                WebGPUBackend.Log($"[DIAG-BinaryArith] Kind={value.Kind} BVT={value.BasicValueType} leftType={leftType} rightType={rightType} isEmuI64={isEmulatedI64} isEmuF64={isEmulatedF64} I64Emu={Backend.Options.EnableI64Emulation} LeftIRType={value.Left.Type} IsUnsigned={value.IsUnsigned}");
            }

            if (isEmulatedF64)
            {
                // Use emu_f64 emulation functions
                string? emulFunc = value.Kind switch
                {
                    BinaryArithmeticKind.Add => "f64_add",
                    BinaryArithmeticKind.Sub => "f64_sub",
                    BinaryArithmeticKind.Mul => "f64_mul",
                    BinaryArithmeticKind.Div => "f64_div",
                    BinaryArithmeticKind.Min => "f64_min",
                    BinaryArithmeticKind.Max => "f64_max",
                    _ => null
                };

                if (emulFunc != null)
                {
                    AppendLine($"{prefix}{target} = {emulFunc}({left}, {right});");
                    return;
                }
                // Fall through to standard ops for unsupported kinds
            }

            if (isEmulatedI64)
            {
                // Use emu_i64 emulation functions
                // NOTE: TypeGenerator returns "emu_i64" for BOTH long AND ulong, because ILGPU's
                // BasicValueType enum has no UInt64 member (only Int64). Signedness is tracked
                // by the arithmetic type, so we must also check value.IsUnsigned.
                bool isUnsigned = leftType == "emu_u64" || rightType == "emu_u64" || value.IsUnsigned;
                string? emulFunc = value.Kind switch
                {
                    BinaryArithmeticKind.Add => "i64_add",
                    BinaryArithmeticKind.Sub => "i64_sub",
                    BinaryArithmeticKind.Mul => isUnsigned ? "u64_mul" : "i64_mul",
                    BinaryArithmeticKind.And => "i64_and",
                    BinaryArithmeticKind.Or => "i64_or",
                    BinaryArithmeticKind.Xor => "i64_xor",
                    BinaryArithmeticKind.Shl => "i64_shl",
                    BinaryArithmeticKind.Shr => isUnsigned ? "u64_shr" : "i64_shr",
                    BinaryArithmeticKind.Min => isUnsigned ? "u64_min" : "i64_min",
                    BinaryArithmeticKind.Max => isUnsigned ? "u64_max" : "i64_max",
                    _ => null
                };

                if (emulFunc != null)
                {
                    if (value.Kind == BinaryArithmeticKind.Shl || value.Kind == BinaryArithmeticKind.Shr)
                    {
                        // Shift amount should be u32
                        AppendLine($"{prefix}{target} = {emulFunc}({left}, u32({right}));");
                    }
                    else
                    {
                        AppendLine($"{prefix}{target} = {emulFunc}({left}, {right});");
                    }
                    return;
                }
                // Fall through to standard ops for unsupported kinds
            }

            // Standard (non-emulated) path
            if (value.Kind == BinaryArithmeticKind.Min || value.Kind == BinaryArithmeticKind.Max || value.Kind == BinaryArithmeticKind.PowF)
            {
                string func = value.Kind switch
                {
                    BinaryArithmeticKind.Min => "min",
                    BinaryArithmeticKind.Max => "max",
                    BinaryArithmeticKind.PowF => "pow",
                    _ => "min"
                };
                // For unsigned integer min/max, WGSL's min()/max() are type-overloaded:
                // min(i32, i32) is signed, min(u32, u32) is unsigned.
                // When ILGPU flags the operation as unsigned, bitcast operands to u32
                // and the result back to i32.
                if (value.IsUnsigned && value.BasicValueType == BasicValueType.Int32
                    && (value.Kind == BinaryArithmeticKind.Min || value.Kind == BinaryArithmeticKind.Max))
                {
                    AppendLine($"{prefix}{target} = bitcast<i32>({func}(bitcast<u32>({left}), bitcast<u32>({right})));");
                }
                else
                {
                    AppendLine($"{prefix}{target} = {func}({left}, {right});");
                }
                return;
            }

            // CopySign: WGSL has no copysign built-in.
            // Can't use abs(x)*sign(y) because sign(0)=0 which zeroes the result.
            // Use select() to handle the zero case correctly.
            if (value.Kind == BinaryArithmeticKind.CopySignF)
            {
                AppendLine($"{prefix}{target} = select(-abs({left}), abs({left}), {right} >= 0.0);");
                return;
            }

            var op = GetArithmeticOp(value.Kind);

            if (value.Kind == BinaryArithmeticKind.Shl || value.Kind == BinaryArithmeticKind.Shr)
                AppendLine($"{prefix}{target} = {left} {op} u32({right});");
            else
                AppendLine($"{prefix}{target} = {left} {op} {right};");
        }


        private void PushPhiValues(BasicBlock targetBlock, BasicBlock sourceBlock)
        {
            foreach (var value in targetBlock)
            {
                if (value.Value is PhiValue phi)
                {
                    var targetVar = Load(phi);
                    // PhiValue in ILGPU implements a collection of (SourceBlock, Value) pairs
                    for (int i = 0; i < phi.Count; i++)
                    {
                        if (phi.Sources[i] == sourceBlock)
                        {
                            var sourceVal = Load(phi[i]);
                            AppendLine($"{targetVar} = {sourceVal};");
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Pushes PHI values transitively through intermediate blocks that sit between
        /// a loop break point and the final merge/exit block.
        ///
        /// When ILGPU generates IR for a loop body break (e.g., from C# `hitT = t; steps = i; break;`),
        /// it may insert intermediate basic blocks between the break source and the final merge
        /// block. These intermediate blocks have unconditional branches and PHI nodes that
        /// carry the assigned values forward.
        ///
        /// CRITICAL: This method must NOT add blocks to the visited set or emit code for them.
        /// The exit/merge blocks will be emitted by GenerateStructuredCodeRecursive AFTER the
        /// loop construct. If we mark them visited here, the post-loop code (conditionals, stores)
        /// will never be generated, causing the "grey screen" bug.
        /// </summary>
        private void PushPhiValuesTransitive(BasicBlock exitTarget, BasicBlock sourceBlock, 
            Loops<ReversePostOrder, Forwards>.Node currentLoop, HashSet<BasicBlock> visited)
        {
            // Push any PHIs in the immediate exit target 
            PushPhiValues(exitTarget, sourceBlock);

            // Follow the chain of unconditional branches through intermediate blocks.
            // Only push PHI values — do NOT emit code or mark blocks as visited.
            // The post-loop blocks will be emitted naturally after the loop construct.
            var current = exitTarget;
            int maxDepth = 8; // Safety limit
            for (int depth = 0; depth < maxDepth; depth++)
            {
                if (current.Terminator is global::ILGPU.IR.Values.UnconditionalBranch uBranch)
                {
                    var nextBlock = uBranch.Target;
                    PushPhiValues(nextBlock, current);
                    current = nextBlock;
                }
                else
                {
                    break; // Not an unconditional branch, stop tracing
                }
            }
        }
        public override void GenerateCode(UnaryArithmeticValue value)
        {
            var target = Load(value);
            var source = Load(value.Value);
            string prefix = GetPrefix(value);

            // Handle math intrinsics that need function calls
            string? funcCall = value.Kind switch
            {
                UnaryArithmeticKind.Abs => $"abs({source})",
                UnaryArithmeticKind.SinF => $"sin({source})",
                UnaryArithmeticKind.CosF => $"cos({source})",
                UnaryArithmeticKind.TanF => $"tan({source})",
                UnaryArithmeticKind.AsinF => $"asin({source})",
                UnaryArithmeticKind.AcosF => $"acos({source})",
                UnaryArithmeticKind.AtanF => $"atan({source})",
                UnaryArithmeticKind.SinhF => $"sinh({source})",
                UnaryArithmeticKind.CoshF => $"cosh({source})",
                UnaryArithmeticKind.TanhF => $"tanh({source})",
                UnaryArithmeticKind.ExpF => $"exp({source})",
                UnaryArithmeticKind.Exp2F => $"exp2({source})",
                UnaryArithmeticKind.LogF => $"log({source})",
                UnaryArithmeticKind.Log2F => $"log2({source})",
                UnaryArithmeticKind.SqrtF => $"sqrt({source})",
                UnaryArithmeticKind.RsqrtF => $"1.0 / sqrt({source})",
                UnaryArithmeticKind.RcpF => $"1.0 / {source}",
                UnaryArithmeticKind.FloorF => $"floor({source})",
                UnaryArithmeticKind.CeilingF => $"ceil({source})",
                // IsNaN: NaN is the only value where x != x
                UnaryArithmeticKind.IsNaNF => $"({source} != {source})",
                // IsInf: infinity is unchanged by abs and equals itself
                // Use abs(x) == abs(x) && abs(x) > 1e38 as a heuristic
                // Or use (abs(x) * 0.0 != 0.0) - but that includes NaN
                // Safest WGSL approach: (x != 0.0 && x == x * 2.0) - works for infinity
                UnaryArithmeticKind.IsInfF => $"({source} != 0.0 && {source} == {source} * 2.0 && {source} == {source})",
                _ => null
            };

            if (funcCall != null)
            {
                AppendLine($"{prefix}{target} = {funcCall};");
                return;
            }

            // Handle emulated emu_i64/emu_u64 negation
            var sourceType = TypeGenerator[value.Value.Type];
            if (Backend.Options.EnableI64Emulation && (sourceType == "emu_i64" || sourceType == "emu_u64") && value.Kind == UnaryArithmeticKind.Neg)
            {
                AppendLine($"{prefix}{target} = i64_neg({source});");
                return;
            }

            // Handle simple unary operators
            string op = value.Kind switch
            {
                UnaryArithmeticKind.Neg => "-",
                UnaryArithmeticKind.Not => TypeGenerator[value.Value.Type] == "bool" ? "!" : "~",
                _ => ""
            };

            if (!string.IsNullOrEmpty(op))
            {
                AppendLine($"{prefix}{target} = {op}({source});");
            }
            else
            {
                // Fallback for unsupported operations
                AppendLine($"// [WGSL] Unhandled UnaryArithmeticKind: {value.Kind}");
                AppendLine($"{prefix}{target} = {source};");
            }
        }
        public override void GenerateCode(UnconditionalBranch branch)
        {
            // 1. Move Phi data (v_11 = v_19, etc.)
            PushPhiValues(branch.Target, branch.BasicBlock);

            // 2. Force the State Machine transition
            var targetIndex = GetBlockIndex(branch.Target);
            AppendIndent();
            Builder.AppendLine($"current_block = {targetIndex};");
            // Note: NO continue; here. In WGSL switch, cases don't fall through.
            // Control naturally flows to after the switch, where workgroupBarrier()
            // is placed for uniform control flow. The loop's implicit continuation
            // returns to the top for the next state machine iteration.
        }
        private new void GenerateCodeInternal()
        {
            // Optimization for single-block (Linear)
            if (Method.Blocks.Count == 1)
            {
                GenerateBlockCode(Method.Blocks.First());
                return;
            }

            // Structured Control Flow (Acyclic Multi-block)
            if (_loops.Count == 0)
            {
                _visitedBlocks.Clear();
                GenerateStructuredCode(Method.EntryBlock, null);
                return;
            }

            // Fallback: State Machine (Cyclic)
            // Note: Barriers inside loops will likely fail validation or execution uniformity quirks
            foreach (var block in Method.Blocks)
            {
                AppendIndent();
                Builder.AppendLine($"case {GetBlockIndex(block)}: {{");
                PushIndent();

                GenerateBlockCode(block);

                PopIndent();
                AppendIndent();
                Builder.AppendLine("}");
            }
        }

        private HashSet<BasicBlock> _visitedBlocks = new HashSet<BasicBlock>();

        private void GenerateBlockCode(BasicBlock block)
        {
            foreach (var value in block)
            {
                if (value.Value is TerminatorValue) continue;

                this.GenerateCodeFor(value.Value);
            }

            // Handle terminator explicitly ONLY if we are in state machine or single block
            // For structured code, the recursion handles branching logic, but we might need to emit 'return'
            if (!(_loops.Count == 0 && Method.Blocks.Count > 1))
            {
                // State Machine / Single Block
                if (block.Terminator is UnconditionalBranch ub) GenerateCode(ub);
                else if (block.Terminator is IfBranch ib) GenerateCode(ib);
                else if (block.Terminator is SwitchBranch sb) GenerateCode(sb); // Add Switch support
                else if (block.Terminator is ReturnTerminator rt2) GenerateCode(rt2);
                else if (block.Terminator != null) this.GenerateCodeFor(block.Terminator);
            }
        }

        private void GenerateStructuredCode(BasicBlock current, BasicBlock? stop)
        {
            if (current == null || current == stop || _visitedBlocks.Contains(current)) return;
            _visitedBlocks.Add(current);

            // 1. Emit instructions (excluding terminator)
            GenerateBlockCode(current);

            // 2. Handle Control Flow
            var terminator = current.Terminator;

            if (terminator is UnconditionalBranch ub)
            {
                // Linear flow -> Recurse to target
                PushPhiValues(ub.Target, current);
                GenerateStructuredCode(ub.Target, stop);
            }
            else if (terminator is IfBranch ib)
            {
                // Divergence point
                var trueTarget = ib.TrueTarget;
                var falseTarget = ib.FalseTarget;
                var merge = _postDominators.GetImmediateDominator(current);

                // Resolve Phi nodes for outgoing branches
                PushPhiValues(trueTarget, current);

                // SHORT-CIRCUIT FIX: Save visited state before true branch so that
                // blocks visited during the true path can be re-visited by the false
                // path. This is essential for || short-circuit patterns where both
                // branches converge on a shared body block (e.g. if (a || b) { body }).
                var visitedBeforeTrueBranch = new HashSet<BasicBlock>(_visitedBlocks);

                var cond = Load(ib.Condition);
                AppendLine($"if ({cond}) {{");
                PushIndent();
                GenerateStructuredCode(trueTarget, merge);
                PopIndent();
                AppendLine("} else {");
                PushIndent();
                PushPhiValues(falseTarget, current);
                // Restore visited state: remove blocks that were only visited during
                // the true branch, so the false branch can reach shared targets.
                _visitedBlocks.IntersectWith(visitedBeforeTrueBranch);
                GenerateStructuredCode(falseTarget, merge);
                PopIndent();
                AppendLine("}");

                // Continue from merge point
                if (merge != null && merge != stop)
                {
                    GenerateStructuredCode(merge, stop);
                }
            }
            else if (terminator is SwitchBranch sb)
            {
                // Switch support for structured flow
                var selector = Load(sb.Condition);
                AppendLine($"switch ({selector}) {{");

                var merge = _postDominators.GetImmediateDominator(current);

                for (int i = 0; i < sb.NumCasesWithoutDefault; i++)
                {
                    AppendLine($"case {i}: {{");
                    PushIndent();
                    PushPhiValues(sb.GetCaseTarget(i), current);
                    GenerateStructuredCode(sb.GetCaseTarget(i), merge);
                    AppendLine("break;");
                    PopIndent();
                    AppendLine("}");
                }

                AppendLine("default: {");
                PushIndent();
                PushPhiValues(sb.DefaultBlock, current);
                GenerateStructuredCode(sb.DefaultBlock, merge);
                AppendLine("break;");
                PopIndent();
                AppendLine("}");
                AppendLine("}");

                if (merge != null && merge != stop)
                    GenerateStructuredCode(merge, stop);
            }
            else if (terminator is ReturnTerminator rt)
            {
                // Already handled in GenerateBlockCode for the return statement itself
            }
        }
        public override void GenerateCode(IfBranch branch)
        {
            var condition = Load(branch.Condition);
            var trueIndex = GetBlockIndex(branch.TrueTarget);
            var falseIndex = GetBlockIndex(branch.FalseTarget);

            AppendIndent();
            Builder.AppendLine($"if ({condition}) {{");
            PushIndent();
            PushPhiValues(branch.TrueTarget, branch.BasicBlock);
            AppendIndent();
            Builder.AppendLine($"current_block = {trueIndex};");
            PopIndent();

            AppendIndent();
            Builder.AppendLine("} else {");
            PushIndent();
            PushPhiValues(branch.FalseTarget, branch.BasicBlock);
            AppendIndent();
            Builder.AppendLine($"current_block = {falseIndex};");
            PopIndent();

            AppendIndent();
            Builder.AppendLine("}");
            // Note: NO continue; here — same rationale as UnconditionalBranch.
            // Control flows to after the switch where barriers are placed.
        }
        // 3. Handles 'return' to exit the kernel
        public override void GenerateCode(ReturnTerminator returnTerminator)
        {
            // In state machine mode, we can't use 'return;' directly because it would
            // skip the workgroupBarrier() placed after the switch statement, violating
            // WGSL's uniform control flow requirement. Instead, set the exit signal
            // and let the loop's exit check handle the termination.
            if (_loops.Count > 0 && Method.Blocks.Count > 1)
            {
                AppendIndent();
                Builder.AppendLine("current_block = -1;");
            }
            else
            {
                // Single-block or non-state-machine: direct return
                AppendIndent();
                AppendLine("return;");
            }
        }

        public override void GenerateCode(CompareValue value)
        {
            var target = Load(value);
            var left = Load(value.Left);
            var right = Load(value.Right);
            var op = GetCompareOp(value.Kind);

            string prefix = GetPrefix(value);

            // Fix: Handle Vector vs Scalar comparison (e.g. vec2 >= i32)
            string leftType = TypeGenerator[value.Left.Type];
            string rightType = TypeGenerator[value.Right.Type];

            // Check for emulated 64-bit compare
            bool isEmulatedF64 = Backend.Options.EnableF64Emulation && (leftType == "emu_f64" || rightType == "emu_f64");
            bool isEmulatedI64 = Backend.Options.EnableI64Emulation && (leftType == "emu_i64" || leftType == "emu_u64" || rightType == "emu_i64" || rightType == "emu_u64");

            if (isEmulatedF64)
            {
                string? emulFunc = value.Kind switch
                {
                    CompareKind.LessThan => "f64_lt",
                    CompareKind.LessEqual => "f64_le",
                    CompareKind.GreaterThan => "f64_gt",
                    CompareKind.GreaterEqual => "f64_ge",
                    CompareKind.Equal => "f64_eq",
                    CompareKind.NotEqual => "f64_ne",
                    _ => null
                };

                if (emulFunc != null)
                {
                    AppendLine($"{prefix}{target} = {emulFunc}({left}, {right});");
                    return;
                }
            }

            if (isEmulatedI64)
            {
                // NOTE: TypeGenerator returns "emu_i64" for BOTH long AND ulong, because ILGPU's
                // BasicValueType enum has no UInt64 member. Must also check value.IsUnsignedOrUnordered.
                bool isUnsigned = leftType == "emu_u64" || rightType == "emu_u64" || value.IsUnsignedOrUnordered;
                string? emulFunc = value.Kind switch
                {
                    CompareKind.LessThan => isUnsigned ? "u64_lt" : "i64_lt",
                    CompareKind.LessEqual => isUnsigned ? "u64_le" : "i64_le",
                    CompareKind.GreaterThan => isUnsigned ? "u64_gt" : "i64_gt",
                    CompareKind.GreaterEqual => isUnsigned ? "u64_ge" : "i64_ge",
                    CompareKind.Equal => "i64_eq",
                    CompareKind.NotEqual => "i64_ne",
                    _ => null
                };

                if (emulFunc != null)
                {
                    AppendLine($"{prefix}{target} = {emulFunc}({left}, {right});");
                    return;
                }
            }

            // Standard comparison:
            bool leftIsVec = leftType.StartsWith("vec");
            bool rightIsVec = rightType.StartsWith("vec");

            // For unsigned integer comparisons, WGSL operators are type-sensitive:
            // i32 < i32 is signed, u32 < u32 is unsigned.
            // When ILGPU flags the comparison as unsigned, bitcast operands to u32.
            bool needsUnsignedCast = value.IsUnsignedOrUnordered
                && value.Left.BasicValueType == BasicValueType.Int32
                && !leftIsVec && !rightIsVec
                && value.Kind != CompareKind.Equal
                && value.Kind != CompareKind.NotEqual;

            if (needsUnsignedCast)
            {
                AppendLine($"{prefix}{target} = bitcast<u32>({left}) {op} bitcast<u32>({right});");
            }
            else if (leftIsVec && !rightIsVec)
            {
                // vec op scalar -> all(vec op vec(scalar))
                // Use vector type of the vector operand to splat the scalar
                AppendLine($"{prefix}{target} = all({left} {op} {leftType}({right}));");
            }
            else if (!leftIsVec && rightIsVec)
            {
                // scalar op vec -> all(vec(scalar) op vec)
                AppendLine($"{prefix}{target} = all({rightType}({left}) {op} {right});");
            }
            else
            {
                AppendLine($"{prefix}{target} = {left} {op} {right};");
            }
        }
        public override void GenerateCode(ConvertValue value)
        {
            var target = Load(value);
            var source = Load(value.Value);
            var targetType = TypeGenerator[value.Type];

            string prefix = GetPrefix(value);

            // Fix: Handle Vector to Scalar conversion (e.g. i32(vec2)) which WGSL forbids
            var sourceType = TypeGenerator[value.Value.Type];

            bool isVectorSource = sourceType.StartsWith("vec");
            bool isScalarTarget = !targetType.StartsWith("vec") && !targetType.StartsWith("mat") && !targetType.StartsWith("array")
                                  && targetType != "emu_f64" && targetType != "emu_i64" && targetType != "emu_u64";

            // Detect unsigned source conversion (e.g. uint → float)
            bool isSourceUnsigned = (value.Flags & ConvertFlags.SourceUnsigned) == ConvertFlags.SourceUnsigned;

            // Emulated type detection
            bool isEmulatedF64Target = Backend.Options.EnableF64Emulation && targetType == "emu_f64";
            bool isEmulatedI64Target = Backend.Options.EnableI64Emulation && (targetType == "emu_i64" || targetType == "emu_u64");
            bool isEmulatedF64Source = Backend.Options.EnableF64Emulation && sourceType == "emu_f64";
            bool isEmulatedI64Source = Backend.Options.EnableI64Emulation && (sourceType == "emu_i64" || sourceType == "emu_u64");

            // ---- TARGET is emulated emu_f64 ----
            if (isEmulatedF64Target)
            {
                if (isEmulatedF64Source)
                {
                    // emu_f64 → emu_f64: just assign
                    AppendLine($"{prefix}{target} = {source};");
                }
                else if (isEmulatedI64Source)
                {
                    // emu_i64 → emu_f64: extract i32 then convert through f32
                    AppendLine($"{prefix}{target} = f64_from_f32(f32(i64_to_i32({source})));");
                }
                else if (isVectorSource)
                {
                    // vec → emu_f64: extract .x component
                    AppendLine($"{prefix}{target} = f64_from_f32(f32({source}.x));");
                }
                else if (sourceType == "f32")
                {
                    AppendLine($"{prefix}{target} = f64_from_f32({source});");
                }
                else if (isSourceUnsigned && sourceType == "i32")
                {
                    // unsigned int → emu_f64: bitcast to u32 first to preserve unsigned value
                    AppendLine($"{prefix}{target} = f64_from_f32(f32(bitcast<u32>({source})));");
                }
                else
                {
                    // Integer or other scalar → emu_f64: convert to f32 first
                    AppendLine($"{prefix}{target} = f64_from_f32(f32({source}));");
                }
                return;
            }

            // ---- TARGET is emulated emu_i64/emu_u64 ----
            if (isEmulatedI64Target)
            {
                if (isEmulatedI64Source)
                {
                    // emu_i64 → emu_i64 (or emu_u64 → emu_u64): just assign
                    AppendLine($"{prefix}{target} = {source};");
                }
                else if (isEmulatedF64Source)
                {
                    // emu_f64 → emu_i64: extract f32, cast to i32, then widen
                    AppendLine($"{prefix}{target} = i64_from_i32(i32(f64_to_f32({source})));");
                }
                else if (isVectorSource)
                {
                    // Generic vec → emu_i64: extract .x, cast to i32, then widen
                    if (targetType == "emu_u64")
                        AppendLine($"{prefix}{target} = u64_from_u32(u32({source}.x));");
                    else
                        AppendLine($"{prefix}{target} = i64_from_i32(i32({source}.x));");
                }
                else if (sourceType == "i32")
                {
                    AppendLine($"{prefix}{target} = i64_from_i32({source});");
                }
                else if (sourceType == "u32")
                {
                    if (targetType == "emu_u64")
                        AppendLine($"{prefix}{target} = u64_from_u32({source});");
                    else
                        AppendLine($"{prefix}{target} = i64_from_i32(i32({source}));");
                }
                else if (sourceType == "f32")
                {
                    // f32 → emu_i64: cast to i32 first
                    AppendLine($"{prefix}{target} = i64_from_i32(i32({source}));");
                }
                else
                {
                    // Other scalar → emu_i64: cast to i32 first
                    AppendLine($"{prefix}{target} = i64_from_i32(i32({source}));");
                }
                return;
            }

            // ---- SOURCE is emulated emu_f64, target is scalar ----
            if (isEmulatedF64Source && isScalarTarget)
            {
                if (targetType == "f32")
                {
                    AppendLine($"{prefix}{target} = f64_to_f32({source});");
                }
                else
                {
                    // emu_f64 → i32/u32/etc: extract to f32, then cast
                    AppendLine($"{prefix}{target} = {targetType}(f64_to_f32({source}));");
                }
                return;
            }

            // ---- SOURCE is emulated emu_i64/emu_u64, target is scalar ----
            if (isEmulatedI64Source && isScalarTarget)
            {
                if (targetType == "i32")
                {
                    AppendLine($"{prefix}{target} = i64_to_i32({source});");
                }
                else if (targetType == "u32")
                {
                    AppendLine($"{prefix}{target} = u64_to_u32({source});");
                }
                else if (targetType == "f32")
                {
                    // emu_i64 → f32: extract low word as i32, then cast to f32
                    AppendLine($"{prefix}{target} = f32(i64_to_i32({source}));");
                }
                else
                {
                    // emu_i64 → other scalar: extract low word and cast
                    AppendLine($"{prefix}{target} = {targetType}(i64_to_i32({source}));");
                }
                return;
            }

            // ---- Generic vector → scalar (non-emulated) ----
            if (isVectorSource && isScalarTarget)
            {
                AppendLine($"{prefix}{target} = {targetType}({source}.x);");
                return;
            }

            // ---- Unsigned int → float: must bitcast to u32 first to preserve unsigned value ----
            if (isSourceUnsigned && sourceType == "i32" && targetType == "f32")
            {
                AppendLine($"{prefix}{target} = f32(bitcast<u32>({source}));");
                return;
            }

            // ---- Standard scalar → scalar ----
            AppendLine($"{prefix}{target} = {targetType}({source});");
        }

        public override void GenerateCode(global::ILGPU.IR.Values.GetField value)
        {
            // 1. Safety check: If this is a kernel index component (X, Y, Z) 
            // already hoisted in SetupIndexVariables, skip it.
            if (_hoistedIndexFields.Contains(value)) return;

            // 1b. Body struct field access: redirect to the per-field variable.
            // When a body struct parameter (e.g. ReductionImplementation) is accessed,
            // we redirect GetField to the individual field variable created in SetupParameterBindings.
            if (value.ObjectValue.Resolve() is global::ILGPU.IR.Values.Parameter bodyParam
                && _bodyStructParams.ContainsKey(bodyParam.Index))
            {
                int fieldIdx = value.FieldSpan.Index;
                if (_bodyStructFieldVars.TryGetValue((bodyParam.Index, fieldIdx), out var fieldVarName)
                    && fieldIdx < _bodyStructParams[bodyParam.Index].Count)
                {
                    var fieldInfo = _bodyStructParams[bodyParam.Index][fieldIdx];

                    if (fieldInfo.IsView)
                    {
                        // View field: the fieldVarName is a pointer (&param1_f0).
                        // WGSL does NOT support assigning pointers to 'var' variables — only 'let' bindings.
                        // If this GetField result was hoisted (declared as 'var v_28 : i32'), we CANNOT
                        // assign a pointer to it. This would cause a WGSL validation error.
                        // Instead, skip the assignment — the 'var v_28' stays at 0 (unused placeholder).
                        // The actual buffer access is handled by LoadElementAddress, which detects
                        // body struct view field accesses through the IR chain and uses param{N}_f{M}[idx]
                        // directly, without needing the v_28 intermediate variable.
                        if (!_hoistedPrimitives.Contains(value))
                        {
                            var target = Load(value);
                            AppendLine($"let {target} = {fieldVarName};");
                        }
                        // If hoisted: skip the assignment to avoid pointer→i32 type error.
                        return;
                    }
                    else
                    {
                        // Scalar or metadata field: the fieldVarName is a value (i32, etc.).
                        // Safe to assign to a hoisted var.
                        var target = Load(value);
                        string prefix = _hoistedPrimitives.Contains(value) ? "" : "let ";
                        AppendLine($"{prefix}{target} = {fieldVarName};");
                        return;
                    }
                }
            }

            // 2. Identify if the object being accessed is DIRECTLY a Parameter (likely an ArrayView)
            // We use direct type check to avoid confusing hierarchical access (View.Stride.X) with Root access (View.Stride)
            if (ResolveToParameter(value.ObjectValue) is global::ILGPU.IR.Values.Parameter param)
            {
                int paramOffset = KernelParamOffset;
                if (param.Index >= paramOffset)
                {
                    IsMultiDim(param.ParameterType, out var isMultiDim, out var isView, out var is1DView, out var is2DView, out var is3DView);

                    if (isView)
                    {
                        var target = Load(value);
                        var rawType = UnwrapType(param.ParameterType);

                        // ROBUST TYPE DETECTION
                        string typeName = param.ParameterType.ToString();

                        if (rawType is global::ILGPU.IR.Types.StructureType st)
                        {


                            // Field 0 is always the actual pointer to the data
                            if (value.FieldSpan.Index == 0)
                            {
                                AppendLine($"let {target} = &param{param.Index}[0];");
                                return;
                            }

                            // Metadata handling (Length and Strides)
                            if (isMultiDim && rawType is StructureType st1)
                            {
                                var totalLen = $"i32(arrayLength(&param{param.Index}))";

                                // Check hoisting to prevent shadowing
                                string prefix = _hoistedPrimitives.Contains(value) ? "" : "let ";

                                // Helper: wrap i32 expression with i64_from_i32 when target expects emu_i64
                                var targetWgslType = TypeGenerator[value.Type];
                                bool needsI64Wrap = Backend.Options.EnableI64Emulation &&
                                    (targetWgslType == "emu_i64" || targetWgslType == "emu_u64");
                                string WrapI32(string expr) => needsI64Wrap ? $"i64_from_i32({expr})" : expr;

                                if (is3DView)
                                {
                                    var width = $"param{param.Index}_stride[0]";
                                    var height = $"param{param.Index}_stride[1]";
                                    var depth = $"param{param.Index}_stride[2]";

                                    // Flattened Access:
                                    // 1: Width
                                    // 2: Height
                                    // 3: Depth
                                    // 4: StrideY (Width)
                                    // 5: StrideZ (Width*Height)

                                    switch (value.FieldSpan.Index)
                                    {
                                        case 1: AppendLine($"{prefix}{target} = {WrapI32(width)};"); return;
                                        case 2: AppendLine($"{prefix}{target} = {WrapI32(height)};"); return;
                                        case 3: AppendLine($"{prefix}{target} = {WrapI32(depth)};"); return;
                                        case 4: AppendLine($"{prefix}{target} = {WrapI32(width)};"); return;
                                        case 5: AppendLine($"{prefix}{target} = {WrapI32($"{width} * {height}")};"); return;
                                    }
                                }
                                else if (is2DView)
                                {
                                    var width = $"param{param.Index}_stride[0]";
                                    var height = $"param{param.Index}_stride[1]";

                                    // Flattened Access:
                                    // 1: Width
                                    // 2: Height
                                    // 3: StrideY (Width)

                                    switch (value.FieldSpan.Index)
                                    {
                                        case 1: AppendLine($"{prefix}{target} = {WrapI32(width)};"); return;
                                        case 2: AppendLine($"{prefix}{target} = {WrapI32(height)};"); return;
                                        case 3: AppendLine($"{prefix}{target} = {WrapI32(width)};"); return;
                                    }
                                }
                                else if (is1DView)
                                {
                                    // For 1D, we verify if mapped to standard structure
                                    // 1D ArrayView is (View, Index, Length) or (View, Length)?
                                    // If base View, it has (Buffer, Index, Length).
                                    // Usually accessed: Field 2 (Length).

                                    switch (value.FieldSpan.Index)
                                    {
                                        case 1: AppendLine($"{prefix}{target} = {WrapI32("0")};"); return;       // Index (Assume 0 for base view)
                                        case 2: AppendLine($"{prefix}{target} = {WrapI32(totalLen)};"); return; // Length
                                    }
                                }

                                // Fallback for unknown length access
                                AppendLine($"{prefix}{target} = {WrapI32(totalLen)};");
                                return;
                            }
                        }
                    }
                }
            }

            // 3. Special handling for Kernel Index Parameter (X, Y components)
            if (ResolveToParameter(value.ObjectValue) is global::ILGPU.IR.Values.Parameter kernelParam)
            {
                int paramOffset = KernelParamOffset;
                if (kernelParam.Index < paramOffset)
                {
                    var target = Load(value);
                    var source = Load(value.ObjectValue);
                    string prefix = _hoistedPrimitives.Contains(value) ? "" : "let ";

                    if (EntryPoint.IndexType == IndexType.Index2D)
                    {
                        // Standard: Field 0 is X, Field 1 is Y
                        string comp = value.FieldSpan.Index == 0 ? "x" : "y";
                        AppendLine($"{prefix}{target} = {source}.{comp};");
                        return;
                    }
                    else if (EntryPoint.IndexType == IndexType.Index3D)
                    {
                        string comp = value.FieldSpan.Index == 0 ? "x" : (value.FieldSpan.Index == 1 ? "y" : "z");
                        AppendLine($"{prefix}{target} = {source}.{comp};");
                        return;
                    }
                    else if (EntryPoint.IndexType == IndexType.Index1D)
                    {
                        if (value.FieldSpan.Index == 0) AppendLine($"{prefix}{target} = {source};");
                        else AppendLine($"{prefix}{target} = 0;");
                        return;
                    }
                }
            }

            // 4. Standard Field Access (not a View or Kernel Index)
            var standardTarget = Load(value);
            var standardSource = Load(value.ObjectValue);

            // Check hoisting for the final result
            string finalPrefix = _hoistedPrimitives.Contains(value) ? "" : "let ";

            if (value.Type.IsPointerType)
            {
                AppendLine($"{finalPrefix}{standardTarget} = &({standardSource}).field_{value.FieldSpan.Index};");
            }
            else
            {
                // WGSL supports vector/struct swizzles/access
                // If source is a vec2/vec3 and we want field 0/1, use x/y/z
                // BUT ILGPU IR 'Index2D' is a struct, so it might be treating it as such
                // If we forced Index2D to be vec2<i32>, we need .x/.y access.
                // However, standard structs use struct members.
                // WE MUST CHECK TYPE.

                string fieldAccess = $".field_{value.FieldSpan.Index}";

                // Heuristic: If source is a vector type string, use x/y/z/w
                var typeStr = TypeGenerator[value.ObjectValue.Type];
                if (typeStr.Contains("vec2"))
                {
                    fieldAccess = value.FieldSpan.Index == 0 ? ".x" : ".y";
                }

                AppendLine($"{finalPrefix}{standardTarget} = {standardSource}{fieldAccess};");
            }
        }

        public override void GenerateCode(global::ILGPU.IR.Values.SetField value)
        {
            var target = Load(value.ObjectValue);
            var val = Load(value.Value);
            // Directly update the field of the hoisted variable
            // Note: This relies on 'target' being a mutable 'var' (hoisted primitive)
            AppendLine($"{target}.field_{value.FieldSpan.Index} = {val};");

            // Define the result value to maintain connectivity for downstream users (like Phi)
            // Since we mutated 'target' in place, the result 'value' is logically equivalent to 'target'
            var res = Allocate(value);
            AppendLine($"let {res.Name} = {target};");
        }

        private static string GetArithmeticOp(BinaryArithmeticKind kind)
        {
            switch (kind)
            {
                case BinaryArithmeticKind.Add: return "+";
                case BinaryArithmeticKind.Sub: return "-";
                case BinaryArithmeticKind.Mul: return "*";
                case BinaryArithmeticKind.Div: return "/";
                case BinaryArithmeticKind.Rem: return "%";
                case BinaryArithmeticKind.And: return "&";
                case BinaryArithmeticKind.Or: return "|";
                case BinaryArithmeticKind.Xor: return "^";
                case BinaryArithmeticKind.Shl: return "<<";
                case BinaryArithmeticKind.Shr: return ">>";
                default: throw new NotSupportedException($"Binary op {kind} not supported.");
            }
        }
        private static string GetCompareOp(CompareKind kind)
        {
            switch (kind)
            {
                case CompareKind.Equal: return "==";
                case CompareKind.NotEqual: return "!=";
                case CompareKind.LessThan: return "<";
                case CompareKind.LessEqual: return "<=";
                case CompareKind.GreaterThan: return ">";
                case CompareKind.GreaterEqual: return ">=";
                default: throw new NotSupportedException($"Compare op {kind} not supported.");
            }
        }
        public override void GenerateCode(LoadFieldAddress value)
        {
            if (ResolveToParameter(value.Source) is global::ILGPU.IR.Values.Parameter param)
            {
                int paramOffset = KernelParamOffset;
                if (param.Index < paramOffset)
                {
                    var target = Load(value);
                    var source = Load(value.Source);
                    if (EntryPoint.IndexType == IndexType.Index2D)
                    {
                        string comp = value.FieldSpan.Index == 0 ? "x" : "y";
                        AppendLine($"var temp_{target.Name} : i32 = {source}.{comp};");
                        AppendLine($"let {target} = &temp_{target.Name};");
                        return;
                    }
                }
            }
            base.GenerateCode(value);
        }





        public override void GenerateCode(PhiValue phiValue)
        {
            // Phi values are handled at the branch level in this state-machine model.
            // We don't emit code here, but the values will be pulled by the branching logic.
        }

        private global::ILGPU.IR.Values.Parameter? ResolveToParameter(global::ILGPU.IR.Value value)
        {
            if (value is global::ILGPU.IR.Values.Parameter p) return p;
            if (value is GetField gf) return ResolveToParameter(gf.ObjectValue);

            if (value is global::ILGPU.IR.Values.Load load) return ResolveToParameter(load.Source);
            if (value is LoadElementAddress lea) return ResolveToParameter(lea.Source);
            if (value is LoadFieldAddress lfa) return ResolveToParameter(lfa.Source);
            if (value is global::ILGPU.IR.Values.NewView nv) return ResolveToParameter(nv.Pointer);

            if (_indexParameterAddress != null && value == _indexParameterAddress)
            {
                return Method.Parameters[0];
            }

            return null;
        }

        /// <summary>
        /// Kernel-specific GetViewLength: resolves the view back to a kernel parameter
        /// and emits i32(arrayLength(&paramN)) for storage buffer bindings.
        /// </summary>
        public override void GenerateCode(global::ILGPU.IR.Values.GetViewLength value)
        {
            var target = Load(value);

            // Try to resolve the view to a kernel parameter
            if (ResolveToParameter(value.View) is global::ILGPU.IR.Values.Parameter param
                && param.Index >= KernelParamOffset)
            {
                string prefix = _hoistedPrimitives.Contains(value) ? "" : "let ";
                var lengthExpr = $"i32(arrayLength(&param{param.Index}))";

                // Handle emulated i64 case
                var targetWgslType = TypeGenerator[value.Type];
                if (Backend.Options.EnableI64Emulation &&
                    (targetWgslType == "emu_i64" || targetWgslType == "emu_u64"))
                {
                    lengthExpr = $"i64_from_i32({lengthExpr})";
                }

                AppendLine($"{prefix}{target} = {lengthExpr};");
                return;
            }

            // Fallback to base implementation
            base.GenerateCode(value);
        }

        protected override bool IsAtomicPointer(Value ptr)
        {
            if (ResolveToParameter(ptr) is global::ILGPU.IR.Values.Parameter param)
            {
                return _atomicParameters.Contains(param.Index);
            }
            return base.IsAtomicPointer(ptr);
        }

        #endregion


        private void GenerateBasicBlockCode(BasicBlock block)
        {
            foreach (var value in block)
            {
                // Skip Terminators (handled by Structured Logic)
                if (value.Value is global::ILGPU.IR.Values.TerminatorValue) continue;
                GenerateCodeFor(value.Value);
            }
        }

        private void GenerateStructuredCode(BasicBlock block, BasicBlock? stopBlock, global::ILGPU.IR.Analyses.Dominators<global::ILGPU.IR.Analyses.ControlFlowDirection.Backwards> pd)
        {
            var visited = new HashSet<BasicBlock>(new BasicBlock.Comparer());

            GenerateStructuredCodeRecursive(block, stopBlock, pd, visited);
        }

        private void GenerateStructuredCodeRecursive(BasicBlock? block, BasicBlock? stopBlock, global::ILGPU.IR.Analyses.Dominators<global::ILGPU.IR.Analyses.ControlFlowDirection.Backwards> pd, HashSet<BasicBlock> visited, Loops<ReversePostOrder, Forwards>.Node? currentLoop = null)
        {
            while (block != null && block != stopBlock)
            {
                // Check if this block is a loop header that we're entering for the first time
                Loops<ReversePostOrder, Forwards>.Node? loopNode = null;
                foreach (var loop in _loops)
                {
                    foreach (var header in loop.Headers)
                    {
                        if (header == block)
                        {
                            loopNode = loop;
                            break;
                        }
                    }
                    if (loopNode != null) break;
                }

                if (loopNode != null && !visited.Contains(block))
                {
                    // Entering a loop — emit a WGSL loop {} construct
                    visited.Add(block);
                    GenerateLoopConstruct(block, loopNode, stopBlock, pd, visited);

                    // After the loop, continue with the exit block.
                    // DEBUG IL FIX: The loop's break paths already pushed PHI values
                    // for all post-loop blocks via EmitIntermediateBlocksToExit /
                    // PushPhiValuesTransitive. Skip through any exit chain blocks
                    // that are just pass-through (UnconditionalBranch only) since
                    // re-processing them would call PushPhiValues again and overwrite
                    // the break-path values with stale "normal exit" values.
                    if (loopNode.Exits.Length > 0)
                    {
                        block = loopNode.Exits[0];
                        // Skip pass-through blocks in the exit chain
                        int skipLimit = 10;
                        while (block != null && skipLimit-- > 0)
                        {
                            if (block.Terminator is global::ILGPU.IR.Values.UnconditionalBranch exitUBranch
                                && !HasNonPhiInstructions(block))
                            {
                                // Pure pass-through block — mark visited and skip
                                visited.Add(block);
                                block = exitUBranch.Target;
                            }
                            else
                            {
                                break; // Found a block with real code or conditional branch
                            }
                        }
                    }
                    else
                    {
                        block = null;
                    }
                    continue;
                }

                // Back-edge check: if this is a loop header we've already visited,
                // this is a back-edge — the loop handles re-iteration implicitly.
                if (visited.Contains(block))
                {
                    // If we're inside a loop and this is the header, push phi values for the
                    // back-edge and let the loop iterate naturally.
                    break;
                }
                visited.Add(block);

                // If we're inside a loop and this block is OUTSIDE the loop,
                // we shouldn't be here. This shouldn't happen if stopBlock is set correctly.
                if (currentLoop != null && !currentLoop.Contains(block))
                {
                    break;
                }

                // Emit Block Body
                GenerateBasicBlockCode(block);

                var terminator = block.Terminator;

                if (terminator is global::ILGPU.IR.Values.IfBranch branch)
                {
                    var trueTarget = branch.TrueTarget;
                    var falseTarget = branch.FalseTarget;

                    // Check if we're inside a loop and one target exits the loop
                    if (currentLoop != null)
                    {
                        // DEBUG IL FIX: Use transitive exit detection.
                        // In Debug builds, Roslyn inserts intermediate blocks between
                        // branches and the actual loop exit. We follow unconditional
                        // branch chains to detect indirect exits.
                        bool trueExitsLoop = ExitsLoopTransitively(trueTarget, currentLoop, out var trueExitBlock);
                        bool falseExitsLoop = ExitsLoopTransitively(falseTarget, currentLoop, out var falseExitBlock);
                        bool trueIsBackEdge = IsLoopHeader(trueTarget, currentLoop);
                        bool falseIsBackEdge = IsLoopHeader(falseTarget, currentLoop);

                        // Case: if (cond) break; — true exits loop (directly or transitively)
                        if (trueExitsLoop && !falseExitsLoop)
                        {
                            AppendLine($"if ({Load(branch.Condition)}) {{");
                            PushIndent();
                            // Emit intermediate blocks' code and PHI values before break
                            EmitIntermediateBlocksToExit(trueTarget, trueExitBlock, block, currentLoop, visited);
                            AppendLine("break;");
                            PopIndent();
                            AppendLine("}");
                            PushPhiValues(falseTarget, block);
                            block = falseTarget;
                            continue;
                        }
                        // Case: if (!cond) break; — false exits loop (directly or transitively)
                        else if (falseExitsLoop && !trueExitsLoop)
                        {
                            AppendLine($"if (!{Load(branch.Condition)}) {{");
                            PushIndent();
                            // Emit intermediate blocks' code and PHI values before break
                            EmitIntermediateBlocksToExit(falseTarget, falseExitBlock, block, currentLoop, visited);
                            AppendLine("break;");
                            PopIndent();
                            AppendLine("}");
                            PushPhiValues(trueTarget, block);
                            block = trueTarget;
                            continue;
                        }
                        // Case: if (cond) continue; — true goes back to header
                        else if (trueIsBackEdge && !falseIsBackEdge)
                        {
                            AppendLine($"if ({Load(branch.Condition)}) {{");
                            PushIndent();
                            PushPhiValues(trueTarget, block);
                            AppendLine("continue;");
                            PopIndent();
                            AppendLine("}");
                            PushPhiValues(falseTarget, block);
                            block = falseTarget;
                            continue;
                        }
                        // Case: if (!cond) continue; — false goes back to header
                        else if (falseIsBackEdge && !trueIsBackEdge)
                        {
                            AppendLine($"if (!{Load(branch.Condition)}) {{");
                            PushIndent();
                            PushPhiValues(falseTarget, block);
                            AppendLine("continue;");
                            PopIndent();
                            AppendLine("}");
                            PushPhiValues(trueTarget, block);
                            block = trueTarget;
                            continue;
                        }
                    }

                    // Standard if/else handling (no loop break/continue)
                    BasicBlock? mergeNode = pd.GetImmediateDominator(block);
                    if (mergeNode == block) mergeNode = null;

                    if (falseTarget == mergeNode)
                    {
                        AppendLine($"if ({Load(branch.Condition)}) {{");
                        PushIndent();
                        PushPhiValues(trueTarget, block);
                        GenerateStructuredCodeRecursive(trueTarget, mergeNode, pd, visited, currentLoop);
                        PopIndent();
                        // When falseTarget == merge, the false path goes directly to merge.
                        // We must still push the merge's phi values for this path, otherwise
                        // variables that should retain their pre-branch values will stay at
                        // their hoisted defaults (e.g., 0.0 instead of pixel coordinates).
                        AppendLine("} else {");
                        PushIndent();
                        PushPhiValues(mergeNode, block);
                        PopIndent();
                        AppendLine("}");
                    }
                    else if (trueTarget == mergeNode)
                    {
                        AppendLine($"if (!{Load(branch.Condition)}) {{");
                        PushIndent();
                        PushPhiValues(falseTarget, block);
                        GenerateStructuredCodeRecursive(falseTarget, mergeNode, pd, visited, currentLoop);
                        PopIndent();
                        // When trueTarget == merge, the true path goes directly to merge.
                        // Push the merge's phi values for the true-path (condition was true
                        // but negated above, so this else is the "original true" path).
                        AppendLine("} else {");
                        PushIndent();
                        PushPhiValues(mergeNode, block);
                        PopIndent();
                        AppendLine("}");
                    }
                    else
                    {
                        // SHORT-CIRCUIT FIX: For || short-circuit patterns where both
                        // branches converge on a shared body block (e.g. if (a || b) { body }),
                        // we need to allow the false branch to re-visit blocks that were
                        // visited by the true branch. However, we must NOT blindly restore
                        // all visited state, as that would cause duplicate code emission in
                        // normal if-else branches (breaking the f64 emulation library etc).
                        //
                        // Strategy: After the true branch, check if the false target has a
                        // direct branch to the same block as trueTarget. If so, un-visit
                        // that shared target so the false path can emit it too.
                        var visitedBeforeTrueBranch = new HashSet<BasicBlock>(visited);

                        AppendLine($"if ({Load(branch.Condition)}) {{");
                        PushIndent();
                        PushPhiValues(trueTarget, block);
                        GenerateStructuredCodeRecursive(trueTarget, mergeNode, pd, visited, currentLoop);
                        PopIndent();
                        AppendLine("} else {");
                        PushIndent();
                        PushPhiValues(falseTarget, block);
                        // Targeted restore: only un-visit blocks that are shared convergence
                        // targets reachable from the false branch. This handles || patterns
                        // without duplicating code in normal if-else branches.
                        UnvisitSharedTargets(falseTarget, visitedBeforeTrueBranch, visited);
                        GenerateStructuredCodeRecursive(falseTarget, mergeNode, pd, visited, currentLoop);
                        PopIndent();
                        AppendLine("}");
                    }

                    block = mergeNode;
                }
                else if (terminator is global::ILGPU.IR.Values.UnconditionalBranch uBranch)
                {
                    PushPhiValues(uBranch.Target, block);

                    // Check if this is a back-edge to the loop header (implicit continue)
                    if (currentLoop != null && IsLoopHeader(uBranch.Target, currentLoop))
                    {
                        // Back-edge: phi values pushed, loop continues naturally
                        block = null;
                    }
                    else
                    {
                        block = uBranch.Target;
                    }
                }
                else if (terminator is global::ILGPU.IR.Values.ReturnTerminator)
                {
                    AppendLine("return;");
                    block = null;
                }
                else
                {
                    block = null;
                }
            }
        }

        /// <summary>
        /// For || short-circuit patterns: un-visit blocks that are shared convergence
        /// targets between the true and false branches of an if-else. This walks the
        /// false target's CFG (following unconditional branches and if-branch targets)
        /// to find blocks that were newly visited by the true branch. Only those blocks
        /// are removed from the visited set, avoiding duplicate code emission in normal
        /// if-else branches.
        /// </summary>
        private void UnvisitSharedTargets(BasicBlock falseTarget, HashSet<BasicBlock> visitedBefore, HashSet<BasicBlock> visited)
        {
            // Collect blocks newly visited by the true branch
            var newlyVisited = new HashSet<BasicBlock>(visited, new BasicBlock.Comparer());
            newlyVisited.ExceptWith(visitedBefore);
            if (newlyVisited.Count == 0) return;

            // Walk the false path to find reachable blocks that were also visited
            // by the true path. Only walk a few steps to find immediate convergence
            // (the || pattern typically converges within 1-2 blocks).
            var toCheck = new Queue<BasicBlock>();
            toCheck.Enqueue(falseTarget);
            var checked_ = new HashSet<BasicBlock>(new BasicBlock.Comparer());

            while (toCheck.Count > 0)
            {
                var bb = toCheck.Dequeue();
                if (!checked_.Add(bb)) continue;

                // If this block was newly visited by the true branch, un-visit it
                // so the false branch can re-emit it
                if (newlyVisited.Contains(bb))
                {
                    visited.Remove(bb);
                    // Continue checking successors — the shared body might chain
                    // to more shared blocks via unconditional branches
                }

                // Only follow edges from blocks we haven't un-visited yet or that
                // are the starting block (to find its successors)
                var term = bb.Terminator;
                if (term is global::ILGPU.IR.Values.IfBranch ib)
                {
                    if (newlyVisited.Contains(ib.TrueTarget))
                        toCheck.Enqueue(ib.TrueTarget);
                    if (newlyVisited.Contains(ib.FalseTarget))
                        toCheck.Enqueue(ib.FalseTarget);
                }
                else if (term is global::ILGPU.IR.Values.UnconditionalBranch ub)
                {
                    if (newlyVisited.Contains(ub.Target))
                        toCheck.Enqueue(ub.Target);
                }
            }
        }

        /// <summary>
        /// Checks if a block is a header of the given loop.
        /// </summary>
        private static bool IsLoopHeader(BasicBlock block, Loops<ReversePostOrder, Forwards>.Node loopNode)
        {
            foreach (var header in loopNode.Headers)
            {
                if (header == block) return true;
            }
            return false;
        }

        /// <summary>
        /// DEBUG IL FIX: Checks if a block contains any non-PHI, non-terminator instructions.
        /// Pure pass-through blocks (containing only PHI values and a terminator) can be
        /// skipped in post-loop exit chain processing since their PHI values were already
        /// handled by the loop's break paths.
        /// </summary>
        private static bool HasNonPhiInstructions(BasicBlock block)
        {
            foreach (var value in block)
            {
                if (value.Value is global::ILGPU.IR.Values.TerminatorValue) continue;
                if (value.Value is PhiValue) continue;
                return true; // Found a real instruction
            }
            return false;
        }

        /// <summary>
        /// DEBUG IL FIX: Checks whether a block exits the loop, either directly
        /// or transitively through a chain of unconditional branches.
        /// 
        /// In Debug builds, Roslyn inserts intermediate basic blocks between
        /// control flow decisions and their actual targets. These blocks contain
        /// only unconditional branches (and sometimes PHI assignments). This
        /// method follows such chains to determine if the eventual destination
        /// is outside the loop.
        /// </summary>
        /// <param name="target">The immediate branch target to check.</param>
        /// <param name="loop">The current loop node.</param>
        /// <param name="exitBlock">The actual exit block (outside the loop), or the target itself if it directly exits.</param>
        /// <returns>True if the target eventually exits the loop.</returns>
        private static bool ExitsLoopTransitively(
            BasicBlock target,
            Loops<ReversePostOrder, Forwards>.Node loop,
            out BasicBlock exitBlock)
        {
            // Fast path: direct exit
            if (!loop.Contains(target))
            {
                exitBlock = target;
                return true;
            }

            // Follow unconditional branch chains through intermediate blocks
            var current = target;
            int maxDepth = 10; // Safety limit to prevent infinite loops
            for (int i = 0; i < maxDepth; i++)
            {
                // Only follow unconditional branches (intermediate pass-through blocks)
                if (current.Terminator is global::ILGPU.IR.Values.UnconditionalBranch uBranch)
                {
                    var next = uBranch.Target;
                    if (!loop.Contains(next))
                    {
                        // Found the exit!
                        exitBlock = next;
                        return true;
                    }
                    current = next;
                }
                else
                {
                    // Hit a conditional branch or other terminator — not a simple chain
                    break;
                }
            }

            exitBlock = target;
            return false;
        }

        /// <summary>
        /// DEBUG IL FIX: Emits intermediate blocks' instructions and PHI values
        /// along the chain from startBlock to exitBlock.
        /// 
        /// When Debug IL inserts intermediate blocks between a branch and the
        /// loop exit, those blocks may contain PHI assignments that carry values
        /// (like hitT, steps) forward. This method walks the chain, emitting
        /// each block's instructions and pushing PHI values, then finally pushes
        /// PHI values for the exit block itself.
        /// </summary>
        private void EmitIntermediateBlocksToExit(
            BasicBlock startBlock,
            BasicBlock exitBlock,
            BasicBlock sourceBlock,
            Loops<ReversePostOrder, Forwards>.Node currentLoop,
            HashSet<BasicBlock> visited)
        {
            if (!currentLoop.Contains(startBlock))
            {
                // startBlock IS the exit — no intermediate blocks.
                // Use transitive push for the exit chain.
                PushPhiValuesTransitive(startBlock, sourceBlock, currentLoop, visited);
                return;
            }

            // Walk through intermediate blocks, emitting their code and PHI values
            var current = startBlock;
            var prevBlock = sourceBlock;
            int maxDepth = 10;
            for (int i = 0; i < maxDepth; i++)
            {
                // Push PHI values from the previous block into this one
                PushPhiValues(current, prevBlock);
                // Mark visited so we don't re-emit
                visited.Add(current);
                // Emit the block's instructions (excluding terminator)
                GenerateBasicBlockCode(current);

                if (current.Terminator is global::ILGPU.IR.Values.UnconditionalBranch uBranch)
                {
                    var next = uBranch.Target;
                    if (!currentLoop.Contains(next))
                    {
                        // Reached the exit — push PHI values transitively
                        PushPhiValuesTransitive(next, current, currentLoop, visited);
                        return;
                    }
                    prevBlock = current;
                    current = next;
                }
                else
                {
                    // Shouldn't happen given ExitsLoopTransitively passed, but be safe
                    break;
                }
            }
        }

        /// <summary>
        /// Generates a WGSL loop {} construct for a natural loop detected by ILGPU's loop analysis.
        /// The loop header block is emitted first, followed by a break condition check,
        /// then the loop body blocks. This ensures barriers are in uniform control flow.
        /// </summary>
        private void GenerateLoopConstruct(
            BasicBlock headerBlock,
            Loops<ReversePostOrder, Forwards>.Node loopNode,
            BasicBlock? outerStopBlock,
            global::ILGPU.IR.Analyses.Dominators<global::ILGPU.IR.Analyses.ControlFlowDirection.Backwards> pd,
            HashSet<BasicBlock> visited)
        {
            AppendLine("loop {");
            PushIndent();

            // Emit the header block's body (instructions)
            GenerateBasicBlockCode(headerBlock);

            var terminator = headerBlock.Terminator;

            if (terminator is global::ILGPU.IR.Values.IfBranch headerBranch)
            {
                var trueTarget = headerBranch.TrueTarget;
                var falseTarget = headerBranch.FalseTarget;

                // DEBUG IL FIX: Use transitive exit detection for loop headers too.
                bool trueIsExit = ExitsLoopTransitively(trueTarget, loopNode, out var trueHeaderExit);
                bool falseIsExit = ExitsLoopTransitively(falseTarget, loopNode, out var falseHeaderExit);

                if (trueIsExit)
                {
                    // True branch exits → break when condition is true
                    AppendLine($"if ({Load(headerBranch.Condition)}) {{");
                    PushIndent();
                    EmitIntermediateBlocksToExit(trueTarget, trueHeaderExit, headerBlock, loopNode, visited);
                    AppendLine("break;");
                    PopIndent();
                    AppendLine("}");

                    // False branch is the loop body
                    PushPhiValues(falseTarget, headerBlock);
                    GenerateStructuredCodeRecursive(falseTarget, headerBlock, pd, visited, loopNode);
                }
                else if (falseIsExit)
                {
                    // False branch exits → break when condition is false
                    AppendLine($"if (!{Load(headerBranch.Condition)}) {{");
                    PushIndent();
                    EmitIntermediateBlocksToExit(falseTarget, falseHeaderExit, headerBlock, loopNode, visited);
                    AppendLine("break;");
                    PopIndent();
                    AppendLine("}");

                    // True branch is the loop body
                    PushPhiValues(trueTarget, headerBlock);
                    GenerateStructuredCodeRecursive(trueTarget, headerBlock, pd, visited, loopNode);
                }
                else
                {
                    // Both branches stay in the loop — emit if/else inside loop
                    AppendLine($"if ({Load(headerBranch.Condition)}) {{");
                    PushIndent();
                    PushPhiValues(trueTarget, headerBlock);
                    GenerateStructuredCodeRecursive(trueTarget, headerBlock, pd, visited, loopNode);
                    PopIndent();
                    AppendLine("} else {");
                    PushIndent();
                    PushPhiValues(falseTarget, headerBlock);
                    GenerateStructuredCodeRecursive(falseTarget, headerBlock, pd, visited, loopNode);
                    PopIndent();
                    AppendLine("}");
                }
            }
            else if (terminator is global::ILGPU.IR.Values.UnconditionalBranch uBranch)
            {
                // Unconditional branch in header — emit body, loop continues implicitly
                PushPhiValues(uBranch.Target, headerBlock);
                GenerateStructuredCodeRecursive(uBranch.Target, headerBlock, pd, visited, loopNode);
            }

            PopIndent();
            AppendLine("}");
        }


    }

    /// <summary>
    /// Describes a pipeline-overridable constant for dynamic shared memory sizing.
    /// </summary>
    public readonly struct DynamicSharedOverrideInfo
    {
        /// <summary>The WGSL override constant name (e.g. "DYNAMIC_SHARED_SIZE_0").</summary>
        public string ConstantName { get; }

        /// <summary>The WGSL variable name for the shared memory array.</summary>
        public string VariableName { get; }

        /// <summary>The allocation index within ILGPU's dynamic shared allocation list.</summary>
        public int AllocaIndex { get; }

        /// <summary>The size of one element in bytes.</summary>
        public int ElementSize { get; }

        public DynamicSharedOverrideInfo(string constantName, string variableName, int allocaIndex, int elementSize)
        {
            ConstantName = constantName;
            VariableName = variableName;
            AllocaIndex = allocaIndex;
            ElementSize = elementSize;
        }
    }
}
