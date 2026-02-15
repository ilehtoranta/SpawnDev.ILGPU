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
        private HashSet<Value> _hoistedPrimitives = new HashSet<Value>();
        private HashSet<int> _emulatedF64Params = new HashSet<int>();
        private HashSet<int> _emulatedI64Params = new HashSet<int>();
        private List<DynamicSharedOverrideInfo> _dynamicSharedOverrides = new List<DynamicSharedOverrideInfo>();
        private bool _usesBroadcast = false;
        private bool _usesSubgroups = false;
        // Maps variable names to their emulation info (param index, is emu_f64)
        private Dictionary<string, (int ParamIndex, bool IsF64)> _emulatedVarMappings = new Dictionary<string, (int, bool)>();
        // Maps cross-block pointer variable names to inline expressions (e.g. "param1[v_3_idx]")
        // This fixes WGSL scoping: pointers declared in one switch case are not visible in another.
        private Dictionary<string, string> _crossBlockPointerExprs = new Dictionary<string, string>();
        // Set of LoadElementAddress Values that cross block boundaries
        private HashSet<Value> _crossBlockPointers = new HashSet<Value>();
        private CFG<ReversePostOrder, Forwards> _cfg;
        private Dominators<Backwards> _postDominators;
        private Loops<ReversePostOrder, Forwards> _loops;
        private GeneratorArgs _generatorArgs;
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
                        if (ResolveToParameter(atomic.Target) is global::ILGPU.IR.Values.Parameter param)
                        {
                            _atomicParameters.Add(param.Index);
                        }
                    }
                    else if (entry.Value is global::ILGPU.IR.Values.AtomicCAS cas)
                    {
                        if (ResolveToParameter(cas.Target) is global::ILGPU.IR.Values.Parameter param)
                        {
                            _atomicParameters.Add(param.Index);
                        }
                    }
                }
            }
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

        // Returns 1 if the first IR parameter is an implicit index (auto-grouped kernels),
        // 0 if the first IR parameter is a user-supplied parameter (stream/explicitly-grouped kernels).
        // The old logic (IndexType != None ? 1 : 0) was wrong for explicitly grouped kernels
        // that still report a non-None IndexType, because those kernels' first parameter is a user buffer.
        private int KernelParamOffset
        {
            get
            {
                if (EntryPoint.IndexType == IndexType.None) return 0;
                // Check if Method.Parameters[0] is actually an index-like type (not a ViewType/buffer)
                if (Method.Parameters.Count > 0)
                {
                    var firstParamType = Method.Parameters[0].Type;
                    if (firstParamType is global::ILGPU.IR.Types.ViewType)
                    {
                        // First param is a buffer (ArrayView) — it's a user param, not an index
                        return 0;
                    }
                }
                return 1;
            }
        }

        #endregion

        #region Methods

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
            // Only enable subgroups if the shader actually uses subgroup operations
            bool emitSubgroups = Backend.HasSubgroups && _usesSubgroups;
            if (emitSubgroups)
                builder.AppendLine("enable subgroups;");
            if (Backend.HasShaderF16)
                builder.AppendLine("enable f16;");
            if (emitSubgroups || Backend.HasShaderF16)
                builder.AppendLine();

            // Pre-scan parameters to detect if any actually use f64/i64 types.
            // IMPORTANT: We check IR types directly (BasicValueType) instead of going through
            // TypeGenerator[...] which would populate the mapping dictionary and cause
            // GenerateTypeDefinitions to emit struct definitions prematurely.
            bool needsF64Emulation = Backend.Options.EnableF64Emulation && ContainsBasicValueType(BasicValueType.Float64);
            bool needsI64Emulation = Backend.Options.EnableI64Emulation && ContainsBasicValueType(BasicValueType.Int64);

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
                    needsI64Emulation));
            }

            // Emit struct definitions (may reference emu_i64/emu_f64 from the library above)
            TypeGenerator.GenerateTypeDefinitions(builder);

            int bindingIdx = 0;
            int paramOffset = KernelParamOffset;

            foreach (var param in Method.Parameters)
            {
                if (param.Index < paramOffset) continue;

                var elementType = GetBufferElementType(param.ParameterType);
                var wgslType = TypeGenerator[elementType];
                string accessMode = "read_write";

                // Debug info
                builder.AppendLine($"// Param {param.Index}: {param.ParameterType} (Element: {elementType})");

                // Track if this is an emulated 64-bit buffer
                bool isEmulated64Bit = false;
                if (Backend.Options.EnableF64Emulation && wgslType == "emu_f64")
                {
                    wgslType = "u32"; // Raw bits storage (2 u32 per emu_f64)
                    isEmulated64Bit = true;
                    _emulatedF64Params.Add(param.Index);
                }
                else if (Backend.Options.EnableI64Emulation && (wgslType == "emu_i64" || wgslType == "emu_u64"))
                {
                    wgslType = "u32"; // Raw bits storage (2 u32 per emu_i64)
                    isEmulated64Bit = true;
                    _emulatedI64Params.Add(param.Index);
                }

                if (_atomicParameters.Contains(param.Index))
                {
                    wgslType = $"atomic<{wgslType}>";
                }

                var bindingDecl = $"@group(0) @binding({bindingIdx}) var<storage, {accessMode}> param{param.Index} : array<{wgslType}>;";
                builder.AppendLine(bindingDecl);
                bindingIdx++;

                IsMultiDim(param.ParameterType, out var isMultiDim, out var isView, out var is1DView, out var is2DView, out var is3DView);

                // STRICT STRIDE LOGIC: Only emit stride buffer for 2D/3D Views.
                // General Structs (even if they mimic views) should NOT have a stride buffer unless they ARE views.
                if (isView && isMultiDim && (is2DView || is3DView))
                {
                    var strideDecl = $"@group(0) @binding({bindingIdx}) var<storage, read> param{param.Index}_stride : array<i32>;";
                    builder.AppendLine(strideDecl);
                    bindingIdx++;
                }
            }

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
                            _usesSubgroups = true;
                            break;
                    }
                }
            }
        }

        public override void GenerateCode()
        {
            string workgroupSize = GetWorkgroupSize();
            Builder.AppendLine($"@compute @workgroup_size({workgroupSize})");
            Builder.Append("fn main(@builtin(global_invocation_id) global_id : vec3<u32>, @builtin(local_invocation_id) local_id : vec3<u32>, @builtin(workgroup_id) group_id : vec3<u32>, @builtin(num_workgroups) num_workgroups : vec3<u32>, @builtin(local_invocation_index) local_index : u32");
            // Add subgroup builtins only when the shader actually uses subgroup operations
            if (Backend.HasSubgroups && _usesSubgroups)
                Builder.Append(", @builtin(subgroup_invocation_id) subgroup_invocation_id : u32, @builtin(subgroup_size) subgroup_size : u32");
            Builder.AppendLine(") {");
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
                    if (expr.TrimStart().StartsWith("&"))
                    {
                        var trimmed = expr.TrimStart();
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
            else
            {
                // Fallback: Default to 1D if we skipped the param but didn't match specific types
                // This protects against weird IndexType states
                AppendLine($"var {indexVar.Name} : i32 = i32(global_id.x); // Fallback mapping");
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
                            if (!value.Value.Type.IsPointerType && !value.Value.Type.IsVoidType)
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
                        init = " = emu_f64(0.0, 0.0)";
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
            int paramOffset = KernelParamOffset;
            foreach (var param in Method.Parameters)
            {
                if (param.Index < paramOffset) continue;
                var variable = Allocate(param);

                IsMultiDim(param.ParameterType, out var isMultiDim, out var isView, out var is1DView, out var is2DView, out var is3DView);

                if (!isView)
                {
                    // Check if it's an atomic parameter (even if not explicitly a ViewType in C# reflection terms)
                    // Atomic operations in WGSL work on pointers to storage buffer elements.
                    bool isAtomic = _atomicParameters.Contains(param.Index);

                    // Check if it's a Structure (that might contain Views). 
                    // If we try to 'load' a struct from a buffer that is array<f32>, it fails.
                    // We should treat structs as pointers/references so GetField can handle them.
                    bool isStruct = param.ParameterType is global::ILGPU.IR.Types.StructureType;

                    if (isAtomic)
                    {
                        // Treat as pointer/reference
                        AppendLine($"let {variable.Name} = &param{param.Index};");
                    }
                    else if (isStruct)
                    {
                        // Optimization:
                        // If this is a View-Like struct (ArrayView2D/3D), we want to alias it as the Buffer Pointer (&param)
                        // so that GetField can detect it as a Parameter and applying Field 0 logic.
                        // If it is a PURE data struct, we alias as &param[0].

                        if (isMultiDim)
                        {
                            AppendLine($"let {variable.Name} = &param{param.Index};");
                        }
                        else
                        {
                            // Structs are loaded as pointers to the first element of the array
                            // param is array<MyStruct>, so &param is ptr<array<MyStruct>>.
                            // We want ptr<MyStruct>, which is &param[0].
                            AppendLine($"let {variable.Name} = &param{param.Index}[0];");
                        }
                    }
                    else
                    {
                        // Scalar load - check for emulated emu_f64/emu_i64
                        if (_emulatedF64Params.Contains(param.Index))
                        {
                            // emu_f64 emulation: read 2 u32 values and convert to emulated emu_f64
                            AppendLine($"var {variable.Name} = f64_from_ieee754_bits(param{param.Index}[0], param{param.Index}[1]);");
                        }
                        else if (_emulatedI64Params.Contains(param.Index))
                        {
                            // emu_i64 emulation: combine 2 u32 values into vec2<u32>
                            AppendLine($"var {variable.Name} = emu_i64(param{param.Index}[0], param{param.Index}[1]);");
                        }
                        else
                        {
                            // Standard scalar load
                            AppendLine($"var {variable.Name} = param{param.Index}[0];");
                        }
                    }
                }
                else
                {
                    AppendLine($"let {variable.Name} = &param{param.Index};");

                    // STRICT STRIDE INITIALIZATION: Only for 2D/3D Views
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
            if (ResolveToParameter(value.Source) is global::ILGPU.IR.Values.Parameter param)
            {
                int paramOffset = KernelParamOffset;
                Console.WriteLine($"[LEA-PARAM] Resolved param.Index={param.Index} paramOffset={paramOffset} param.Type={param.Type} check={param.Index >= paramOffset}");
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
                        Builder.Append($"let {target.Name}_base_idx = i32({offset}) * 2;");
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
                        _crossBlockPointerExprs[target.Name] = $"param{param.Index}[{offset}]";
                        // Still emit a local declaration for same-block uses
                        AppendIndent();
                        Builder.Append($"let {target.Name} = &param{param.Index}[{offset}];");
                        Builder.AppendLine();
                        return;
                    }

                    AppendIndent();
                    Builder.Append($"let {target.Name} = &param{param.Index}[{offset}];");
                    Builder.AppendLine();
                    return;
                }
            }

            var sourceVal = Load(value.Source);
            AppendIndent();
            Builder.Append($"let {target.Name} = ");

            // TEMP DEBUG: trace what generates &sourceVal[offset]
            Console.WriteLine($"[LEA-DEBUG] {target.Name} = &{sourceVal}[{offset}] | Source={value.Source} SourceType={value.Source.Type} IsPointer={value.Source.Type.IsPointerType} SourceValueKind={value.Source.GetType().Name} WGSL-Type={TypeGenerator[value.Source.Type]}");

            // POINTER DEREFERENCE LOGIC
            // If the source comes from NewView, it's a pointer (let v_3 = &v_4).
            // To access element at offset: (*ptr)[offset] -> &(*ptr)[offset] gives the address
            if (value.Source is global::ILGPU.IR.Values.NewView || value.Source.Type.IsPointerType)
            {
                Builder.Append($"&(*{sourceVal})[{offset}];");
            }
            else
            {
                Builder.Append($"&{sourceVal}[{offset}];");
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

            // Check for emulated 64-bit buffer store
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
                bool isUnsigned = leftType == "emu_u64" || rightType == "emu_u64";
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
                AppendLine($"{prefix}{target} = {func}({left}, {right});");
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
                bool isUnsigned = leftType == "emu_u64" || rightType == "emu_u64";
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

            if (leftIsVec && !rightIsVec)
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

            // ---- Standard scalar → scalar ----
            AppendLine($"{prefix}{target} = {targetType}({source});");
        }

        public override void GenerateCode(global::ILGPU.IR.Values.GetField value)
        {
            // 1. Safety check: If this is a kernel index component (X, Y, Z) 
            // already hoisted in SetupIndexVariables, skip it.
            if (_hoistedIndexFields.Contains(value)) return;

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
            Console.WriteLine($"[RTP-DEBUG] Type={value.GetType().Name} IRType={value.Type} Value={value}");
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

            Console.WriteLine($"[RTP-DEBUG] FAILED to resolve: {value.GetType().Name}");
            return null;
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

                    // After the loop, continue with the exit block
                    if (loopNode.Exits.Length > 0)
                    {
                        block = loopNode.Exits[0];
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
                        bool trueExitsLoop = !currentLoop.Contains(trueTarget);
                        bool falseExitsLoop = !currentLoop.Contains(falseTarget);
                        bool trueIsBackEdge = IsLoopHeader(trueTarget, currentLoop);
                        bool falseIsBackEdge = IsLoopHeader(falseTarget, currentLoop);

                        // Case: if (cond) break; — true exits loop
                        if (trueExitsLoop && !falseExitsLoop)
                        {
                            AppendLine($"if ({Load(branch.Condition)}) {{");
                            PushIndent();
                            PushPhiValues(trueTarget, block);
                            AppendLine("break;");
                            PopIndent();
                            AppendLine("}");
                            PushPhiValues(falseTarget, block);
                            block = falseTarget;
                            continue;
                        }
                        // Case: if (!cond) break; — false exits loop (i.e., condition is true → continue)
                        else if (falseExitsLoop && !trueExitsLoop)
                        {
                            AppendLine($"if (!{Load(branch.Condition)}) {{");
                            PushIndent();
                            PushPhiValues(falseTarget, block);
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
                        AppendLine("}");
                    }
                    else if (trueTarget == mergeNode)
                    {
                        AppendLine($"if (!{Load(branch.Condition)}) {{");
                        PushIndent();
                        PushPhiValues(falseTarget, block);
                        GenerateStructuredCodeRecursive(falseTarget, mergeNode, pd, visited, currentLoop);
                        PopIndent();
                        AppendLine("}");
                    }
                    else
                    {
                        // SHORT-CIRCUIT FIX: Save visited state before true branch so that
                        // blocks visited during the true path can be re-visited by the false
                        // path. This is essential for || short-circuit patterns where both
                        // branches converge on a shared body block (e.g. if (a || b) { body }).
                        var visitedBeforeTrueBranch = new HashSet<BasicBlock>(visited);

                        AppendLine($"if ({Load(branch.Condition)}) {{");
                        PushIndent();
                        PushPhiValues(trueTarget, block);
                        GenerateStructuredCodeRecursive(trueTarget, mergeNode, pd, visited, currentLoop);
                        PopIndent();
                        AppendLine("} else {");
                        PushIndent();
                        PushPhiValues(falseTarget, block);
                        // Restore visited state: remove blocks that were only visited during
                        // the true branch, so the false branch can reach shared targets.
                        visited.IntersectWith(visitedBeforeTrueBranch);
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

                // Determine which branch exits the loop and which continues
                bool trueIsExit = !loopNode.Contains(trueTarget);
                bool falseIsExit = !loopNode.Contains(falseTarget);

                if (trueIsExit)
                {
                    // True branch exits → break when condition is true
                    AppendLine($"if ({Load(headerBranch.Condition)}) {{");
                    PushIndent();
                    PushPhiValues(trueTarget, headerBlock);
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
                    PushPhiValues(falseTarget, headerBlock);
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
