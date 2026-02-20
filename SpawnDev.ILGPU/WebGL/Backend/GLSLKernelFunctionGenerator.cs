// ---------------------------------------------------------------------------------------
//                                 SpawnDev.ILGPU.WebGL
//                        Copyright (c) 2024 SpawnDev Project
//
// File: GLSLKernelFunctionGenerator.cs
//
// Generates GLSL ES 3.0 vertex shader code for kernel entry points.
// Uses Transform Feedback to capture output buffers and gl_VertexID for invocation index.
// Parameters are passed via Uniform Buffer Objects (UBOs) for scalars and
// Texture Buffer Objects (TBOs) for array data.
// ---------------------------------------------------------------------------------------

using global::ILGPU;
using global::ILGPU.Backends;
using global::ILGPU.Backends.EntryPoints;
using global::ILGPU.IR;
using global::ILGPU.IR.Analyses;
using global::ILGPU.IR.Analyses.ControlFlowDirection;
using global::ILGPU.IR.Analyses.TraversalOrders;
using global::ILGPU.IR.Types;
using global::ILGPU.IR.Values;
using System.Text;

namespace SpawnDev.ILGPU.WebGL.Backend
{
    /// <summary>
    /// Generates GLSL ES 3.0 vertex shader code for kernel (entry point) functions.
    /// Transform Feedback captures output varyings; gl_VertexID provides the invocation index.
    /// </summary>
    internal sealed class GLSLKernelFunctionGenerator : GLSLCodeGenerator
    {
        #region Fields

        private readonly EntryPoint EntryPoint;
        private readonly AllocaKindInformation SharedAllocations;
        private readonly AllocaKindInformation DynamicSharedAllocations;

        // Emulated 64-bit parameter tracking
        private readonly HashSet<int> _emulatedF64Params = new();
        private readonly HashSet<int> _emulatedI64Params = new();
        private readonly Dictionary<string, (int ParamIndex, bool IsF64)> _emulatedVarMappings = new();

        // Cross-block pointer fix
        private readonly HashSet<Value> _crossBlockPointers = new();
        private readonly Dictionary<string, string> _crossBlockPointerExprs = new();

        // Output buffer detection: only params with Store targets get TF varyings
        private readonly HashSet<int> _outputParamIndices = new();

        // Input buffer detection: params with Load sources get sampler uniforms
        private readonly HashSet<int> _inputParamIndices = new();

        // Hoisting
        private readonly HashSet<Value> _hoistedPrimitives = new();
        private readonly HashSet<Value> _hoistedIndexFields = new();

        // CFG and analysis
        private CFG<ReversePostOrder, Forwards> _cfg;
        private Dominators<Backwards> _postDominators;
        private Loops<ReversePostOrder, Forwards> _loops;
        private HashSet<BasicBlock> _visitedBlocks = new();
        private readonly Stack<BasicBlock> _activeLoopHeaders = new();

        // Parameter binding info for runtime
        private readonly List<KernelParameterBinding> _parameterBindings = new();

        // Output buffer info
        private readonly List<OutputVaryingInfo> _outputVaryings = new();

        // Struct buffer field counts: paramIndex → number of flattened scalar fields
        private readonly Dictionary<int, int> _structFieldCounts = new();
        // Struct buffer field GLSL types: paramIndex → [fieldType0, fieldType1, ...]
        private readonly Dictionary<int, List<string>> _structFieldTypes = new();

        /// <summary>Kernel parameter offset: 1 for implicit index parameter.</summary>
        private int KernelParamOffset => 1;

        #endregion

        #region Constructor

        public GLSLKernelFunctionGenerator(in GeneratorArgs args, Method method, Allocas allocas)
            : base(args, method, allocas)
        {
            _generatorArgs = args;
            EntryPoint = args.EntryPoint;
            SharedAllocations = args.SharedAllocations;
            DynamicSharedAllocations = args.DynamicSharedAllocations;

            _cfg = method.Blocks.CreateCFG();
            _postDominators = _cfg.Blocks.CreatePostDominators();
            _loops = _cfg.CreateLoops();

            AnalyzeCrossBlockPointers();
        }

        /// <summary>Stored args for accessing shared OutputVaryings collection.</summary>
        private readonly GeneratorArgs _generatorArgs;

        #endregion

        #region Analysis

        private void AnalyzeCrossBlockPointers()
        {
            // Find LEA values used in different blocks
            var leaBlocks = new Dictionary<Value, BasicBlock>();
            foreach (var block in Method.Blocks)
            {
                foreach (var value in block)
                {
                    if (value.Value is LoadElementAddress lea)
                        leaBlocks[lea] = block;
                }
            }

            foreach (var block in Method.Blocks)
            {
                foreach (var value in block)
                {
                    if (value.Value is global::ILGPU.IR.Values.Load load)
                    {
                        var src = load.Source.Resolve();
                        if (src is LoadElementAddress lea && leaBlocks.TryGetValue(lea, out var defBlock) && defBlock != block)
                            _crossBlockPointers.Add(lea);
                    }
                    else if (value.Value is Store store)
                    {
                        var tgt = store.Target.Resolve();
                        if (tgt is LoadElementAddress lea && leaBlocks.TryGetValue(lea, out var defBlock) && defBlock != block)
                            _crossBlockPointers.Add(lea);
                    }
                }
            }
        }

        /// <summary>
        /// Pre-scan all IR blocks to detect which buffer parameters are targets of
        /// Store instructions. Only those params need TF output varyings. This avoids
        /// wasting TF slots on read-only input buffers, which causes interleaving
        /// bugs in multi-buffer kernels.
        /// </summary>
        private void AnalyzeOutputBuffers()
        {
            int paramOffset = KernelParamOffset;
            foreach (var block in Method.Blocks)
            {
                foreach (var value in block)
                {
                    if (value.Value is Store store)
                    {
                        // Trace: Store.Target → LEA → ... → Parameter
                        var param = ResolveToParameterStatic(store.Target.Resolve());
                        if (param != null && param.Index >= paramOffset)
                            _outputParamIndices.Add(param.Index);
                    }
                }
            }
        }

        /// <summary>
        /// Pre-scan all IR blocks to detect which buffer parameters are sources of
        /// Load instructions. Only those params need sampler uniform declarations.
        /// Output-only buffers (in _outputParamIndices but NOT in _inputParamIndices)
        /// must NOT have sampler uniforms, because WebGL 2 generates GL_INVALID_OPERATION
        /// when an active sampler has no valid texture bound.
        /// </summary>
        private void AnalyzeInputBuffers()
        {
            int paramOffset = KernelParamOffset;
            foreach (var block in Method.Blocks)
            {
                foreach (var value in block)
                {
                    if (value.Value is global::ILGPU.IR.Values.Load load)
                    {
                        // Trace: Load.Source → LEA → ... → Parameter
                        var param = ResolveToParameterStatic(load.Source.Resolve());
                        if (param != null && param.Index >= paramOffset)
                            _inputParamIndices.Add(param.Index);
                    }
                }
            }
        }

        /// <summary>
        /// Static version of ResolveToParameter that doesn't depend on code gen state.
        /// Used during pre-scan analysis before code generation starts.
        /// </summary>
        private static global::ILGPU.IR.Values.Parameter? ResolveToParameterStatic(Value value)
        {
            if (value is global::ILGPU.IR.Values.Parameter p) return p;
            if (value is GetField gf) return ResolveToParameterStatic(gf.ObjectValue);
            if (value is global::ILGPU.IR.Values.Load load) return ResolveToParameterStatic(load.Source);
            if (value is LoadElementAddress lea) return ResolveToParameterStatic(lea.Source);
            if (value is LoadFieldAddress lfa) return ResolveToParameterStatic(lfa.Source);
            if (value is PhiValue phi)
            {
                // For phi nodes, check all source values — if any traces to a buffer param, include it
                for (int i = 0; i < phi.Count; i++)
                {
                    var result = ResolveToParameterStatic(phi[i]);
                    if (result != null) return result;
                }
            }
            return null;
        }

        #endregion

        #region Code Generation

        public override void GenerateHeader(StringBuilder builder)
        {
            // Precision and version are added in WebGLBackend.CreateKernelBuilder
        }

        public override void GenerateCode()
        {
            // 1. Detect emulated 64-bit parameters
            DetectEmulatedParameters();

            // 1.5. Detect which buffer params are store targets (need TF outputs)
            AnalyzeOutputBuffers();

            // 1.6. Detect which buffer params are load sources (need sampler uniforms)
            AnalyzeInputBuffers();

            // 2. Analyze hoisting needs
            AnalyzeHoisting();

            // 3. Post-dominators and loops already created in constructor

            // 4. Emit emulation library only if the kernel actually uses f64/i64 types.
            //    Including the library unconditionally produces a massive vertex shader
            //    that ANGLE's D3D11 backend cannot compile with Transform Feedback.
            var (kernelNeedsF64, kernelNeedsI64) = KernelUsesEmulatedTypes();
            if ((Backend.Options.EnableF64Emulation && kernelNeedsF64) ||
                (Backend.Options.EnableI64Emulation && kernelNeedsI64))
            {
                Builder.AppendLine("// ============ 64-bit Emulation Library ============");
                Builder.AppendLine(GLSLEmulationLibrary.GetEmulationLibrary(
                    Backend.Options.EnableF64Emulation && kernelNeedsF64,
                    Backend.Options.EnableI64Emulation && kernelNeedsI64));
            }

            // 5. Insert a placeholder for struct type definitions.
            //    Struct types are discovered lazily during EmitParameterDeclarations() and
            //    later code generation, so we can't emit them here yet. The placeholder
            //    will be replaced with actual struct definitions in WebGLBackend.CreateKernel()
            //    after all code generation is complete.
            Builder.AppendLine("// __STRUCT_DEFS_PLACEHOLDER__");

            // 6. Emit parameter declarations (uniforms and samplers)
            EmitParameterDeclarations();

            // 7. Emit output varyings for Transform Feedback
            EmitOutputVaryings();

            // 8. Emit main() vertex shader entry point
            Builder.AppendLine("void main() {");
            PushIndent();

            // 9. Setup index from gl_VertexID
            SetupIndexVariables();

            // 10. Load parameters into local variables
            SetupParameterBindings();

            // 11. Emit hoisted variable declarations
            EmitHoistedDeclarations();

            // 12. Generate kernel body
            GenerateKernelBody();

            PopIndent();
            Builder.AppendLine("}");
        }

        #endregion

        #region Emulated Parameter Detection

        private void DetectEmulatedParameters()
        {
            int paramOffset = KernelParamOffset;
            foreach (var param in Method.Parameters)
            {
                if (param.Index < paramOffset) continue;

                var type = param.ParameterType;
                var unwrapped = UnwrapType(type);

                if (unwrapped is PrimitiveType pt)
                {
                    if (Backend.Options.EnableF64Emulation && pt.BasicValueType == BasicValueType.Float64)
                        _emulatedF64Params.Add(param.Index);
                    else if (Backend.Options.EnableI64Emulation && pt.BasicValueType == BasicValueType.Int64)
                        _emulatedI64Params.Add(param.Index);
                }
            }
        }

        /// <summary>
        /// Scans the kernel's parameters and IR values to determine if f64/i64
        /// emulation types are actually used. This prevents unconditionally including
        /// the entire emulation library for simple float32 kernels, which causes
        /// ANGLE D3D11 to fail compiling the large vertex shader.
        /// </summary>
        private (bool needsF64, bool needsI64) KernelUsesEmulatedTypes()
        {
            bool needsF64 = _emulatedF64Params.Count > 0;
            bool needsI64 = _emulatedI64Params.Count > 0;

            // Early exit if both already detected from parameters
            if (needsF64 && needsI64) return (needsF64, needsI64);

            // Scan all values in the kernel's IR blocks for f64/i64 usage
            foreach (var block in Method.Blocks)
            {
                foreach (Value value in block)
                {
                    var bvt = value.BasicValueType;
                    if (!needsF64 && bvt == BasicValueType.Float64)
                        needsF64 = true;
                    if (!needsI64 && bvt == BasicValueType.Int64)
                        needsI64 = true;

                    if (needsF64 && needsI64) return (needsF64, needsI64);
                }
            }

            return (needsF64, needsI64);
        }

        private static TypeNode UnwrapType(TypeNode type)
        {
            while (type is PointerType pt) type = pt.ElementType;
            while (type is ViewType vt) type = vt.ElementType;
            // For ArrayView2D/3D: the IR type is a StructureType wrapping a ViewType + stride info.
            // Use structural detection (matching IsMultiDim approach) instead of string check,
            // because StructureType.ToString() doesn't contain "ArrayView".
            if (type is StructureType st && st.NumFields > 0 &&
                (st.Fields[0] is ViewType || st.Fields[0] is PointerType || st.Fields[0].ToString().Contains("View")))
            {
                // The first flattened field is the buffer pointer — unwrap it to get element type
                return UnwrapType(st.Fields[0]);
            }
            return type;
        }

        #endregion

        #region Hoisting Analysis

        private void AnalyzeHoisting()
        {
            // For single-block kernels, no hoisting is needed.
            if (Method.Blocks.Count <= 1) return;

            // For multi-block kernels (state machine path), GLSL switch/case blocks
            // have strict scoping — variables declared inside one case {} block are not
            // visible in other case blocks. Hoist ALL values to function scope to avoid
            // undeclared identifier errors.
            foreach (var block in Method.Blocks)
            {
                foreach (var value in block)
                {
                    if (value.Value.Type.IsPrimitiveType || value.Value.Type.IsStructureType)
                        _hoistedPrimitives.Add(value.Value);
                }
            }
        }

        #endregion

        #region Parameter Declarations

        private void EmitParameterDeclarations()
        {
            int paramOffset = KernelParamOffset;
            int bindingIndex = 0;

            // Emit dimension uniforms for multi-dimensional kernels
            if (EntryPoint.IndexType == IndexType.Index2D || EntryPoint.IndexType == IndexType.Index3D)
            {
                Builder.AppendLine("uniform highp int u_dimWidth; // grid X dimension");
                Builder.AppendLine("uniform highp int u_dimHeight; // grid Y dimension");
            }

            foreach (var param in Method.Parameters)
            {
                if (param.Index < paramOffset) continue;

                var type = param.ParameterType;
                IsMultiDim(type, out var isMultiDim, out var isView, out var is1DView, out var is2DView, out var is3DView);
                var elementType = UnwrapType(type);
                bool isStruct = elementType is StructureType && !isView;

                if (isView || type is ViewType)
                {
                    // Array-like parameter: use a highp sampler or SSBO-like mechanism
                    // In WebGL2, we use a uniform sampler for input and TF varying for output
                    // Multi-dim views (ArrayView2D/3D) are NOT struct buffers even though
                    // their IR param type is StructureType — they're container structs
                    // holding a ViewType + stride info, not user-defined struct element types
                    bool isStructBuffer = !isMultiDim
                        && elementType is StructureType structElType
                        && !TypeGenerator[elementType].StartsWith("vec")
                        && !TypeGenerator[elementType].StartsWith("ivec")
                        && !TypeGenerator[elementType].StartsWith("uvec");
                    
                    string glslType;
                    if (isStructBuffer)
                    {
                        // Struct element buffer: flatten to per-field layout
                        // All data stored as R32I texture, reinterpret float fields with intBitsToFloat
                        var flatFields = FlattenStructFields(elementType);
                        _structFieldCounts[param.Index] = flatFields.Count;
                        _structFieldTypes[param.Index] = flatFields;
                        glslType = "int"; // R32I texture for raw bit access
                    }
                    else
                    {
                        glslType = GetBufferElementType(elementType);
                    }

                    // Only declare a sampler uniform for buffers that are actually READ.
                    // Output-only buffers (only in _outputParamIndices, not in _inputParamIndices)
                    // must NOT get sampler uniforms — WebGL2 generates GL_INVALID_OPERATION
                    // when an active sampler has no valid texture bound at draw time.
                    bool isInputBuffer = _inputParamIndices.Contains(param.Index);

                    if (isInputBuffer)
                    {
                        // Input buffer: use a uniform sampler (TBO emulated as texture)
                        // Use isampler2D for int/struct buffers, usampler2D for uint, sampler2D for float
                        string samplerType = glslType switch
                        {
                            "int" => "isampler2D",
                            "uint" => "usampler2D",
                            _ => "sampler2D"
                        };
                        Builder.AppendLine($"uniform highp {samplerType} u_param{param.Index}; // buffer param[{param.Index}]{(isStructBuffer ? " (struct, flattened)" : "")}");
                        // Tile width for 2D texture tiling (supports buffers > MAX_TEXTURE_SIZE)
                        Builder.AppendLine($"uniform highp int u_param{param.Index}_tileW;");
                    }
                    _parameterBindings.Add(new KernelParameterBinding(param.Index, bindingIndex++, KernelParamKind.Buffer, glslType,
                        structFieldCount: isStructBuffer ? _structFieldCounts[param.Index] : 0));
                    _generatorArgs.ParameterBindings.Add(_parameterBindings[^1]);
                    _bufferGlslTypes[param.Index] = glslType;

                    if (isStructBuffer)
                    {
                        // Emit length uniform for struct buffers (element count, not texel count)
                        Builder.AppendLine($"uniform highp int u_param{param.Index}_length; // struct element count");
                    }
                    else
                    {
                        // Emit length uniform for all view params so GetViewLength can reference it
                        Builder.AppendLine($"uniform highp int u_param{param.Index}_length; // element count");
                    }

                    // Multi-dim stride buffer
                    if (isMultiDim && (is2DView || is3DView))
                    {
                        int strideCount = is3DView ? 3 : 2;
                        Builder.AppendLine($"uniform highp int u_param{param.Index}_stride[{strideCount}]; // stride");
                        _parameterBindings.Add(new KernelParameterBinding(param.Index, bindingIndex++, KernelParamKind.Stride, "int"));
                        _generatorArgs.ParameterBindings.Add(_parameterBindings[^1]);
                    }
                }
                else if (isStruct)
                {
                    // Struct parameter: UBO (no precision qualifier allowed on structs)
                    string glslType = TypeGenerator[elementType];
                    Builder.AppendLine($"uniform {glslType} u_param{param.Index}; // struct param[{param.Index}]");
                    _parameterBindings.Add(new KernelParameterBinding(param.Index, bindingIndex++, KernelParamKind.Scalar, glslType));
                    _generatorArgs.ParameterBindings.Add(_parameterBindings[^1]);
                }
                else
                {
                    // Scalar parameter
                    string glslType = TypeGenerator[elementType];

                    if (_emulatedF64Params.Contains(param.Index))
                    {
                        // f64 emulation: pass as 2 uint uniforms
                        Builder.AppendLine($"uniform highp uint u_param{param.Index}_lo; // emu_f64 low bits");
                        Builder.AppendLine($"uniform highp uint u_param{param.Index}_hi; // emu_f64 high bits");
                        _parameterBindings.Add(new KernelParameterBinding(param.Index, bindingIndex++, KernelParamKind.EmulatedF64, "uint"));
                        _generatorArgs.ParameterBindings.Add(_parameterBindings[^1]);
                    }
                    else if (_emulatedI64Params.Contains(param.Index))
                    {
                        // i64 emulation: pass as 2 uint uniforms
                        Builder.AppendLine($"uniform highp uint u_param{param.Index}_lo; // emu_i64 low bits");
                        Builder.AppendLine($"uniform highp uint u_param{param.Index}_hi; // emu_i64 high bits");
                        _parameterBindings.Add(new KernelParameterBinding(param.Index, bindingIndex++, KernelParamKind.EmulatedI64, "uint"));
                        _generatorArgs.ParameterBindings.Add(_parameterBindings[^1]);
                    }
                    else
                    {
                        Builder.AppendLine($"uniform highp {glslType} u_param{param.Index}; // scalar param[{param.Index}]");
                        _parameterBindings.Add(new KernelParameterBinding(param.Index, bindingIndex++, KernelParamKind.Scalar, glslType));
                        _generatorArgs.ParameterBindings.Add(_parameterBindings[^1]);
                    }
                }
            }

            Builder.AppendLine();
        }

        private string GetBufferElementType(TypeNode elementType)
        {
            if (elementType is PrimitiveType pt)
            {
                return pt.BasicValueType switch
                {
                    BasicValueType.Float32 => "float",
                    BasicValueType.Int32 => "int",
                    BasicValueType.Int16 => "int",
                    BasicValueType.Int8 => "int",
                    BasicValueType.Float64 when Backend.Options.EnableF64Emulation => "uint",
                    BasicValueType.Int64 when Backend.Options.EnableI64Emulation => "uint",
                    _ => "float"
                };
            }
            return TypeGenerator[elementType];
        }

        /// <summary>
        /// Recursively flattens a struct type into its primitive GLSL field types.
        /// For example, OuterStruct { InnerStruct { float Val }, int ID } → ["float", "int"]
        /// </summary>
        private List<string> FlattenStructFields(TypeNode type)
        {
            var fields = new List<string>();
            if (type is StructureType st)
            {
                foreach (var fieldType in st.Fields)
                {
                    if (fieldType is StructureType nested)
                        fields.AddRange(FlattenStructFields(nested));
                    else
                        fields.Add(GetBufferElementType(fieldType));
                }
            }
            else
            {
                fields.Add(GetBufferElementType(type));
            }
            return fields;
        }

        /// <summary>
        /// Recursively generates (fieldAccessPath, glslType) pairs mapping flat field index
        /// to the hierarchical GLSL struct field access path.
        /// For OuterStruct { InnerStruct { float Val }, int ID }:
        ///   → [(".field_0.field_0", "float"), (".field_1", "int")]
        /// </summary>
        private List<(string Path, string GlslType)> GenerateStructFieldPaths(TypeNode type, string prefix = "")
        {
            var result = new List<(string, string)>();
            if (type is StructureType st)
            {
                int fieldIdx = 0;
                foreach (var fieldType in st.Fields)
                {
                    string fieldPath = $"{prefix}.field_{fieldIdx}";
                    if (fieldType is StructureType nested)
                        result.AddRange(GenerateStructFieldPaths(nested, fieldPath));
                    else
                        result.Add((fieldPath, GetBufferElementType(fieldType)));
                    fieldIdx++;
                }
            }
            else
            {
                result.Add((prefix, GetBufferElementType(type)));
            }
            return result;
        }

        #endregion

        #region Output Varyings (Transform Feedback)

        private void EmitOutputVaryings()
        {
            // For Transform Feedback, output buffers must be declared as 'out' varyings
            // The runtime will configure TF to capture these
            int paramOffset = KernelParamOffset;
            int outIndex = 0;

            foreach (var param in Method.Parameters)
            {
                if (param.Index < paramOffset) continue;

                var type = param.ParameterType;
                // Use the same detection as EmitParameterDeclarations — ArrayView2D/3D
                // appear as struct types in the IR, not ViewType, so we need IsMultiDim
                IsMultiDim(type, out var isMultiDim, out var isView, out var is1DView, out var is2DView, out var is3DView);

                if (isView || type is ViewType || type is PointerType)
                {
                    // Only emit TF varyings for buffer params that actually have Store targets.
                    // Read-only input buffers don't need TF outputs, and including them wastes
                    // TF slots and causes interleaving bugs in multi-buffer readback.
                    if (!_outputParamIndices.Contains(param.Index))
                        continue;
                    var elementType = UnwrapType(type);
                    bool isEmulatedF64 = _emulatedF64Params.Contains(param.Index);
                    bool isEmulatedI64 = _emulatedI64Params.Contains(param.Index);

                    if (isEmulatedF64 || isEmulatedI64)
                    {
                        // Emulated 64-bit types need TWO uint TF outputs (lo and hi words)
                        Builder.AppendLine($"flat out highp uint tf_out_{param.Index}_lo; // TF output for param[{param.Index}] (emu 64-bit lo)");
                        Builder.AppendLine($"flat out highp uint tf_out_{param.Index}_hi; // TF output for param[{param.Index}] (emu 64-bit hi)");
                        var loInfo = new OutputVaryingInfo(param.Index, outIndex++, $"tf_out_{param.Index}_lo", "uint",
                            isEmulated: true, emulatedSuffix: "lo");
                        var hiInfo = new OutputVaryingInfo(param.Index, outIndex++, $"tf_out_{param.Index}_hi", "uint",
                            isEmulated: true, emulatedSuffix: "hi");
                        _outputVaryings.Add(loInfo);
                        _outputVaryings.Add(hiInfo);
                        _generatorArgs.OutputVaryings.Add(loInfo);
                        _generatorArgs.OutputVaryings.Add(hiInfo);
                    }
                    else if (_structFieldCounts.TryGetValue(param.Index, out var fieldCount) && fieldCount > 0)
                    {
                        // Struct element buffer: emit per-field TF varyings
                        var fieldTypes = _structFieldTypes[param.Index];
                        for (int fi = 0; fi < fieldCount; fi++)
                        {
                            string fieldGlslType = fieldTypes[fi];
                            bool needsFlat = fieldGlslType != "float";
                            string flatPrefix = needsFlat ? "flat " : "";
                            string varyingName = $"tf_out_{param.Index}_f{fi}";
                            Builder.AppendLine($"{flatPrefix}out highp {fieldGlslType} {varyingName}; // TF output for param[{param.Index}] field {fi}");
                            var varyingInfo = new OutputVaryingInfo(param.Index, outIndex++, varyingName, fieldGlslType, fieldIndex: fi);
                            _outputVaryings.Add(varyingInfo);
                            _generatorArgs.OutputVaryings.Add(varyingInfo);
                        }
                    }
                    else
                    {
                        // Use the already-computed GLSL type from EmitParameterDeclarations,
                        // which correctly handles 2D/3D views (StructureType in IR).
                        // Falling back to GetBufferElementType(UnwrapType) for safety.
                        string glslType = _bufferGlslTypes.TryGetValue(param.Index, out var cached)
                            ? cached
                            : GetBufferElementType(UnwrapType(type));

                        // TF varyings must be flat for non-float types AND structs (which may contain int fields)
                        bool needsFlat = glslType != "float";
                        string flatPrefix = needsFlat ? "flat " : "";
                        // GLSL ES 3.0: precision qualifiers are illegal on struct types
                        string precisionPrefix = glslType.StartsWith("struct_") ? "" : "highp ";
                        Builder.AppendLine($"{flatPrefix}out {precisionPrefix}{glslType} tf_out_{param.Index}; // TF output for param[{param.Index}]");
                        var varyingInfo = new OutputVaryingInfo(param.Index, outIndex++, $"tf_out_{param.Index}", glslType);
                        _outputVaryings.Add(varyingInfo);
                        _generatorArgs.OutputVaryings.Add(varyingInfo);
                    }
                }
            }

            Builder.AppendLine();
        }

        #endregion

        #region Index Setup

        private void SetupIndexVariables()
        {
            // gl_VertexID provides the flat invocation index
            switch (EntryPoint.IndexType)
            {
                case IndexType.Index1D:
                    AppendLine("int _idx = gl_VertexID;");
                    break;
                case IndexType.Index2D:
                    AppendLine("int _flat_idx = gl_VertexID;");
                    // Need width from a uniform
                    AppendLine("int _idx_x = _flat_idx % u_dimWidth;");
                    AppendLine("int _idx_y = _flat_idx / u_dimWidth;");
                    AppendLine("ivec2 _idx = ivec2(_idx_x, _idx_y);");
                    break;
                case IndexType.Index3D:
                    AppendLine("int _flat_idx = gl_VertexID;");
                    AppendLine("int _idx_x = _flat_idx % u_dimWidth;");
                    AppendLine("int _idx_y = (_flat_idx / u_dimWidth) % u_dimHeight;");
                    AppendLine("int _idx_z = _flat_idx / (u_dimWidth * u_dimHeight);");
                    AppendLine("ivec3 _idx = ivec3(_idx_x, _idx_y, _idx_z);");
                    break;
            }

            // Bind the implicit index parameter
            if (Method.Parameters.Count > 0)
            {
                var indexParam = Method.Parameters[0];
                string idxType = EntryPoint.IndexType switch
                {
                    IndexType.Index1D => "int",
                    IndexType.Index2D => "ivec2",
                    IndexType.Index3D => "ivec3",
                    _ => "int"
                };
                var variable = new Variable("_idx", idxType);
                Bind(indexParam, variable);
            }

            // Set gl_Position to dummy value (required for vertex shader)
            AppendLine("gl_Position = vec4(0.0, 0.0, 0.0, 1.0);");
            AppendLine("");
        }

        #endregion

        #region Parameter Binding Load

        private void SetupParameterBindings()
        {
            int paramOffset = KernelParamOffset;

            foreach (var param in Method.Parameters)
            {
                if (param.Index < paramOffset) continue;

                var type = param.ParameterType;
                var variable = Load(param);
                IsMultiDim(type, out var isMultiDim, out var isView, out var is1DView, out var is2DView, out var is3DView);
                bool isViewType = isView || type is ViewType;

                if (isViewType)
                {
                    // Buffer parameters: the variable represents the sampler/texture
                    // Actual access is done via texelFetch in LoadElementAddress
                    declaredVariables.Add(variable.Name);
                    AppendLine($"// param{param.Index} is buffer (accessed via u_param{param.Index})");
                }
                else if (_emulatedF64Params.Contains(param.Index))
                {
                    Declare(variable);
                    AppendLine($"{variable.Name} = f64_from_ieee754_bits(u_param{param.Index}_lo, u_param{param.Index}_hi);");
                }
                else if (_emulatedI64Params.Contains(param.Index))
                {
                    Declare(variable);
                    AppendLine($"{variable.Name} = uvec2(u_param{param.Index}_lo, u_param{param.Index}_hi);");
                }
                else
                {
                    Declare(variable);
                    AppendLine($"{variable.Name} = u_param{param.Index};");
                }
            }

            AppendLine("");
        }

        #endregion

        #region Hoisted Declarations

        private void EmitHoistedDeclarations()
        {
            foreach (var value in _hoistedPrimitives)
            {
                var variable = Load(value);
                if (!declaredVariables.Contains(variable.Name))
                {
                    declaredVariables.Add(variable.Name);
                    string type = TypeGenerator[value.Type];
                    // Track boolean variables for logical operator selection
                    if (type == "bool")
                        booleanVariables.Add(variable.Name);
                    string init = GetDefaultValue(type);
                    AppendLine($"{type} {variable.Name} = {init};");
                }
            }
        }

        private string GetDefaultValue(string glslType) => glslType switch
        {
            "bool" => "false",
            "int" => "0",
            "uint" => "0u",
            "float" => "0.0",
            "vec2" => "vec2(0.0)",
            "vec3" => "vec3(0.0)",
            "vec4" => "vec4(0.0)",
            "ivec2" => "ivec2(0)",
            "ivec3" => "ivec3(0)",
            "uvec2" => "uvec2(0u)",
            _ => $"{glslType}(0)"
        };

        #endregion

        #region Kernel Body Generation

        private void GenerateKernelBody()
        {
            if (Method.Blocks.Count == 1)
            {
                GenerateBlockCode(Method.Blocks.First());
                return;
            }

            // Always use structured code gen — the state machine (switch/case
            // inside a loop) triggers ANGLE D3D11 "Error compiling dynamic
            // vertex executable" when Transform Feedback is active.
            _visitedBlocks.Clear();
            _activeLoopHeaders.Clear();
            GenerateStructuredCode(Method.EntryBlock, null);
        }

        private void GenerateBlockCode(BasicBlock block)
        {
            // Emit all non-terminator values in the block
            foreach (var value in block)
            {
                if (value.Value is TerminatorValue) continue;
                GenerateCodeFor(value.Value);
            }
        }

        /// <summary>
        /// Finds the innermost loop that has the given block as a header.
        /// </summary>
        private Loops<ReversePostOrder, Forwards>.Node? FindLoopForHeader(BasicBlock block)
        {
            for (int i = 0; i < _loops.Count; i++)
            {
                var loop = _loops[i];
                foreach (var header in loop.Headers)
                {
                    if (header == block)
                        return loop;
                }
            }
            return null;
        }

        /// <summary>
        /// Checks if target is a back-edge to an active loop header (should emit 'continue').
        /// </summary>
        private bool IsBackEdgeToActiveLoop(BasicBlock target)
        {
            return _activeLoopHeaders.Contains(target);
        }

        /// <summary>
        /// Emit a branch to target, handling loop continue/break/fall-through.
        /// Returns true if the branch was emitted as continue/break (caller should not recurse).
        /// </summary>
        private bool EmitBranchTarget(BasicBlock target, BasicBlock source, BasicBlock? stop)
        {
            PushPhiValues(target, source);

            // Back-edge to active loop header → continue
            if (IsBackEdgeToActiveLoop(target))
            {
                AppendLine("continue;");
                return true;
            }

            // Target is the stop block → let caller handle it
            if (target == stop)
                return true;

            // Already visited → skip
            if (_visitedBlocks.Contains(target))
                return true;

            return false;
        }

        private void GenerateStructuredCode(BasicBlock current, BasicBlock? stop)
        {
            if (current == null || current == stop || _visitedBlocks.Contains(current)) return;
            _visitedBlocks.Add(current);

            // Check if this block is a loop header
            var loop = FindLoopForHeader(current);
            if (loop != null)
            {
                // Emit while(true) for the loop
                AppendLine("while (true) {");
                PushIndent();

                _activeLoopHeaders.Push(current);

                // Remove current from visited so we can re-enter it for the loop body
                _visitedBlocks.Remove(current);

                // Generate the loop body starting from the header
                GenerateLoopBody(current, loop, stop);

                _activeLoopHeaders.Pop();

                PopIndent();
                AppendLine("}");

                // Continue with exit blocks after the loop
                // DEBUG IL FIX: Skip through pure pass-through exit blocks
                // (only PHI values + unconditional branch) since the loop's break
                // paths already pushed their PHI values. Re-processing would call
                // PushPhiValues again and overwrite break-path values.
                foreach (var exit in loop.Exits)
                {
                    var exitBlock = exit;
                    // Skip pass-through blocks
                    int skipLimit = 10;
                    while (exitBlock != null && exitBlock != stop
                        && !_visitedBlocks.Contains(exitBlock)
                        && skipLimit-- > 0)
                    {
                        if (exitBlock.Terminator is UnconditionalBranch exitUBranch
                            && !HasNonPhiInstructions(exitBlock))
                        {
                            _visitedBlocks.Add(exitBlock);
                            exitBlock = exitUBranch.Target;
                        }
                        else
                        {
                            break;
                        }
                    }
                    if (exitBlock != null && exitBlock != stop && !_visitedBlocks.Contains(exitBlock))
                    {
                        GenerateStructuredCode(exitBlock, stop);
                        break; // Only one exit continuation in structured flow
                    }
                }
                return;
            }

            // Not a loop header — emit the block's values
            GenerateBlockCode(current);

            // Handle terminator
            var terminator = current.Terminator;

            if (terminator is ReturnTerminator rt)
            {
                GenerateCode(rt);
            }
            else if (terminator is UnconditionalBranch ub)
            {
                if (!EmitBranchTarget(ub.Target, current, stop))
                    GenerateStructuredCode(ub.Target, stop);
            }
            else if (terminator is IfBranch ib)
            {
                EmitIfBranch(ib, current, stop);
            }
        }

        /// <summary>
        /// Generates the body of a loop (all blocks inside the loop).
        /// </summary>
        private void GenerateLoopBody(BasicBlock current, Loops<ReversePostOrder, Forwards>.Node loop, BasicBlock? outerStop)
        {
            if (current == null || _visitedBlocks.Contains(current)) return;
            _visitedBlocks.Add(current);

            // Emit the block's values
            GenerateBlockCode(current);

            // Handle terminator with loop-aware logic
            var terminator = current.Terminator;

            if (terminator is ReturnTerminator rt)
            {
                GenerateCode(rt);
            }
            else if (terminator is UnconditionalBranch ub)
            {
                PushPhiValues(ub.Target, current);

                if (IsBackEdgeToActiveLoop(ub.Target))
                {
                    AppendLine("continue;");
                }
                else if (ExitsLoopTransitively(ub.Target, loop))
                {
                    // Exit the loop — trace through intermediate blocks first
                    EmitBreakWithIntermediateCode(ub.Target, current);
                }
                else
                {
                    GenerateLoopBody(ub.Target, loop, outerStop);
                }
            }
            else if (terminator is IfBranch ib)
            {
                EmitLoopIfBranch(ib, current, loop, outerStop);
            }
        }

        /// <summary>
        /// Emits an if/else branch inside a loop body.
        /// </summary>
        private void EmitLoopIfBranch(IfBranch ib, BasicBlock source, Loops<ReversePostOrder, Forwards>.Node loop, BasicBlock? outerStop)
        {
            var trueTarget = ib.TrueTarget;
            var falseTarget = ib.FalseTarget;
            var cond = Load(ib.Condition);

            bool trueIsBackEdge = IsBackEdgeToActiveLoop(trueTarget);
            bool falseIsBackEdge = IsBackEdgeToActiveLoop(falseTarget);
            // DEBUG IL FIX: Use transitive exit detection.
            // In Debug builds, Roslyn inserts intermediate blocks between
            // branches and the actual loop exit. Follow unconditional branch
            // chains to detect indirect exits.
            bool trueIsExit = ExitsLoopTransitively(trueTarget, loop);
            bool falseIsExit = ExitsLoopTransitively(falseTarget, loop);

            // Case 1: Loop condition check — one side continues, other exits
            if (trueIsBackEdge && falseIsExit)
            {
                // if (cond) continue; else break;
                AppendLine($"if (!{cond}) {{");
                PushIndent();
                PushPhiValues(falseTarget, source);
                EmitBreakWithIntermediateCode(falseTarget, source);
                PopIndent();
                AppendLine("}");
                PushPhiValues(trueTarget, source);
                AppendLine("continue;");
                return;
            }
            if (falseIsBackEdge && trueIsExit)
            {
                AppendLine($"if ({cond}) {{");
                PushIndent();
                PushPhiValues(trueTarget, source);
                EmitBreakWithIntermediateCode(trueTarget, source);
                PopIndent();
                AppendLine("}");
                PushPhiValues(falseTarget, source);
                AppendLine("continue;");
                return;
            }

            // Case 2: One side is a back-edge, other continues in-loop
            if (trueIsBackEdge)
            {
                AppendLine($"if ({cond}) {{");
                PushIndent();
                PushPhiValues(trueTarget, source);
                AppendLine("continue;");
                PopIndent();
                AppendLine("}");
                PushPhiValues(falseTarget, source);
                if (falseIsExit)
                    EmitBreakWithIntermediateCode(falseTarget, source);
                else
                    GenerateLoopBody(falseTarget, loop, outerStop);
                return;
            }
            if (falseIsBackEdge)
            {
                AppendLine($"if (!{cond}) {{");
                PushIndent();
                PushPhiValues(falseTarget, source);
                AppendLine("continue;");
                PopIndent();
                AppendLine("}");
                PushPhiValues(trueTarget, source);
                if (trueIsExit)
                    EmitBreakWithIntermediateCode(trueTarget, source);
                else
                    GenerateLoopBody(trueTarget, loop, outerStop);
                return;
            }

            // Case 3: One side exits the loop, other stays
            if (trueIsExit && !falseIsExit)
            {
                AppendLine($"if ({cond}) {{");
                PushIndent();
                PushPhiValues(trueTarget, source);
                EmitBreakWithIntermediateCode(trueTarget, source);
                PopIndent();
                AppendLine("}");
                PushPhiValues(falseTarget, source);
                GenerateLoopBody(falseTarget, loop, outerStop);
                return;
            }
            if (falseIsExit && !trueIsExit)
            {
                AppendLine($"if (!{cond}) {{");
                PushIndent();
                PushPhiValues(falseTarget, source);
                EmitBreakWithIntermediateCode(falseTarget, source);
                PopIndent();
                AppendLine("}");
                PushPhiValues(trueTarget, source);
                GenerateLoopBody(trueTarget, loop, outerStop);
                return;
            }

            // Case 4: Both sides stay in loop — use post-dominator for merge
            var merge = _postDominators?.GetImmediateDominator(source);

            PushPhiValues(trueTarget, source);
            AppendLine($"if ({cond}) {{");
            PushIndent();
            if (trueTarget == merge)
            {
                // True goes directly to merge — nothing to generate
            }
            else
            {
                GenerateLoopBody(trueTarget, loop, outerStop);
            }
            PopIndent();
            AppendLine("} else {");
            PushIndent();
            PushPhiValues(falseTarget, source);
            if (falseTarget == merge)
            {
                // False goes directly to merge — nothing to generate
            }
            else
            {
                GenerateLoopBody(falseTarget, loop, outerStop);
            }
            PopIndent();
            AppendLine("}");

            // Continue with merge block if it's in the loop
            if (merge != null && loop.Contains(merge) && !_visitedBlocks.Contains(merge))
            {
                GenerateLoopBody(merge, loop, outerStop);
            }
        }

        /// <summary>
        /// Emits an if/else branch outside a loop (acyclic structured flow).
        /// </summary>
        private void EmitIfBranch(IfBranch ib, BasicBlock source, BasicBlock? stop)
        {
            var trueTarget = ib.TrueTarget;
            var falseTarget = ib.FalseTarget;
            var merge = _postDominators?.GetImmediateDominator(source);

            PushPhiValues(trueTarget, source);

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
            PushPhiValues(falseTarget, source);
            // Restore visited state: remove blocks that were only visited during
            // the true branch, so the false branch can reach shared targets.
            _visitedBlocks.IntersectWith(visitedBeforeTrueBranch);
            GenerateStructuredCode(falseTarget, merge);
            PopIndent();
            AppendLine("}");

            if (merge != null && merge != stop)
                GenerateStructuredCode(merge, stop);
        }

        private void PushPhiValues(BasicBlock targetBlock, BasicBlock sourceBlock)
        {
            foreach (var value in targetBlock)
            {
                if (value.Value is PhiValue phi)
                {
                    var targetVar = Load(phi);
                    for (int i = 0; i < phi.Count; i++)
                    {
                        if (phi.Sources[i] == sourceBlock)
                        {
                            var sourceVal = Load(phi[i]);
                            AppendLine($"{targetVar} = {sourceVal};");

                            // Propagate buffer pointer mappings through phi nodes.
                            // If the source is a LEA-derived pointer, the phi target
                            // must inherit the param mapping so Stores can find TF outputs.
                            if (_leaParamMap.TryGetValue(sourceVal.Name, out var phiParamIdx))
                                _leaParamMap[targetVar.Name] = phiParamIdx;
                            if (_crossBlockPointerExprs.TryGetValue(sourceVal.Name, out var phiExpr))
                                _crossBlockPointerExprs[targetVar.Name] = phiExpr;
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Emits code for intermediate blocks between a break source and the loop exit,
        /// then emits the break statement. When ILGPU generates IR for `hitT = t; steps = i; break;`,
        /// the assignments end up in intermediate blocks between the break source and the
        /// merge/exit block. These blocks must have their code emitted before the GLSL break.
        /// 
        /// CRITICAL: Does NOT add intermediate blocks to _visitedBlocks — they may need
        /// to be re-entered by the post-loop code generator.
        /// </summary>
        private void EmitBreakWithIntermediateCode(BasicBlock exitTarget, BasicBlock sourceBlock)
        {
            // Trace through intermediate blocks that have unconditional branches
            var current = exitTarget;
            int maxDepth = 8; // Safety limit
            for (int depth = 0; depth < maxDepth; depth++)
            {
                if (current.Terminator is UnconditionalBranch uBranch)
                {
                    // Emit this intermediate block's code (assignments like hitT = t)
                    GenerateBlockCode(current);
                    // Push PHI values to the next block
                    PushPhiValues(uBranch.Target, current);
                    current = uBranch.Target;
                }
                else
                {
                    break; // Not an unconditional branch, stop tracing
                }
            }
            AppendLine("break;");
        }

        /// <summary>
        /// DEBUG IL FIX: Checks whether a block exits the loop, either directly
        /// or transitively through a chain of unconditional branches.
        /// 
        /// In Debug builds, Roslyn inserts intermediate basic blocks between
        /// control flow decisions and their actual targets. This method follows
        /// such chains to determine if the eventual destination is outside the loop.
        /// </summary>
        private static bool ExitsLoopTransitively(
            BasicBlock target,
            Loops<ReversePostOrder, Forwards>.Node loop)
        {
            // Fast path: direct exit
            if (!loop.Contains(target))
                return true;

            // Follow unconditional branch chains through intermediate blocks
            var current = target;
            int maxDepth = 10; // Safety limit
            for (int i = 0; i < maxDepth; i++)
            {
                if (current.Terminator is UnconditionalBranch uBranch)
                {
                    if (!loop.Contains(uBranch.Target))
                        return true;
                    current = uBranch.Target;
                }
                else
                {
                    break;
                }
            }
            return false;
        }

        /// <summary>
        /// DEBUG IL FIX: Checks if a block contains any non-PHI, non-terminator instructions.
        /// Pure pass-through blocks can be skipped in post-loop exit chain processing
        /// since their PHI values were already handled by the loop's break paths.
        /// </summary>
        private static bool HasNonPhiInstructions(BasicBlock block)
        {
            foreach (var value in block)
            {
                if (value.Value is TerminatorValue) continue;
                if (value.Value is PhiValue) continue;
                return true;
            }
            return false;
        }

        /// <summary>
        /// Override EmitPhiAssignments for state machine path to propagate
        /// _leaParamMap and _crossBlockPointerExprs through phi nodes.
        /// Without this, multi-block kernels lose their buffer pointer mappings
        /// after phi resolution, causing Stores to miss TF outputs.
        /// </summary>
        protected override void EmitPhiAssignments(BasicBlock sourceBlock, BasicBlock targetBlock)
        {
            foreach (var valueEntry in targetBlock)
            {
                if (valueEntry.Value is PhiValue phi)
                {
                    var phiVar = Load(phi);
                    var srcValue = phi.GetValue(sourceBlock);
                    if (srcValue != null)
                    {
                        var srcVar = Load(srcValue);
                        // GLSL ES 3.0: cast to phi variable type if needed
                        string srcExpr = CastIfNeeded(srcVar, phiVar.Type);
                        AppendLine($"{phiVar} = {srcExpr};");

                        // Propagate buffer pointer mappings through phi nodes
                        if (_leaParamMap.TryGetValue(srcVar.Name, out var phiParamIdx))
                            _leaParamMap[phiVar.Name] = phiParamIdx;
                        if (_crossBlockPointerExprs.TryGetValue(srcVar.Name, out var phiExpr))
                            _crossBlockPointerExprs[phiVar.Name] = phiExpr;
                    }
                }
            }
        }

        #endregion

        #region Overrides for Kernel-Specific Code Gen

        public override void GenerateCode(global::ILGPU.IR.Values.Parameter parameter) { }

        public override void GenerateCode(ReturnTerminator value)
        {
            // Kernel: just return from main()
            AppendLine("return;");
        }

        public override void GenerateCode(LoadElementAddress value)
        {
            var target = Load(value);
            var offset = Load(value.Offset);

            if (ResolveToParameter(value.Source) is global::ILGPU.IR.Values.Parameter param)
            {
                int paramOffset = KernelParamOffset;
                if (param.Index >= paramOffset)
                {
                    // Buffer access via texelFetch from sampler2D
                    if (_emulatedF64Params.Contains(param.Index) || _emulatedI64Params.Contains(param.Index))
                    {
                        bool isF64 = _emulatedF64Params.Contains(param.Index);
                        _emulatedVarMappings[target.Name] = (param.Index, isF64);
                        declaredVariables.Add(target.Name);

                        AppendLine($"int {target.Name}_base_idx = int({offset}) * 2;");
                        return;
                    }

                    if (_crossBlockPointers.Contains(value))
                        _crossBlockPointerExprs[target.Name] = $"texelFetch(u_param{param.Index}, ivec2(int({offset}) % u_param{param.Index}_tileW, int({offset}) / u_param{param.Index}_tileW), 0)";

                    declaredVariables.Add(target.Name);
                    AppendLine($"int {target.Name} = int({offset}); // LEA into param{param.Index}");
                    // Re-bind with correct type: the GLSL variable is declared as 'int',
                    // but the IR type is a pointer whose element type may differ (e.g. float).
                    Bind(value, new Variable(target.Name, "int"));
                    // Store the param index for Load/Store to use
                    _leaParamMap[target.Name] = param.Index;
                    return;
                }
            }

            // Fallback: check if the source variable is already in _leaParamMap
            // This happens for 2D/3D views where GetField(viewParam, 0) creates a buffer
            // alias that is in _leaParamMap, but ResolveToParameter can't trace through
            // the StructureType → GetField → ViewType chain
            var sourceVal = Load(value.Source);
            if (_leaParamMap.TryGetValue(sourceVal.ToString(), out var sourceParamIdx))
            {
                if (_crossBlockPointers.Contains(value))
                    _crossBlockPointerExprs[target.Name] = $"texelFetch(u_param{sourceParamIdx}, ivec2(int({offset}) % u_param{sourceParamIdx}_tileW, int({offset}) / u_param{sourceParamIdx}_tileW), 0)";

                declaredVariables.Add(target.Name);
                AppendLine($"int {target.Name} = int({offset}); // LEA into param{sourceParamIdx} (via alias)");
                Bind(value, new Variable(target.Name, "int"));
                _leaParamMap[target.Name] = sourceParamIdx;
                return;
            }

            Declare(target);
            AppendLine($"{target} = {sourceVal} + {offset}; // generic LEA");
        }

        private readonly Dictionary<string, int> _leaParamMap = new();
        /// <summary>Maps parameter index → GLSL element type ("int", "uint", "float") for buffer params.</summary>
        private readonly Dictionary<int, string> _bufferGlslTypes = new();

        public override void GenerateCode(global::ILGPU.IR.Values.Load loadVal)
        {
            var target = Load(loadVal);
            var source = Load(loadVal.Source);

            // Check for direct parameter load (View alias)
            if (loadVal.Source.Resolve() is global::ILGPU.IR.Values.Parameter param && param.Type is ViewType)
            {
                int paramOffset = KernelParamOffset;
                if (param.Index >= paramOffset)
                {
                    declaredVariables.Add(target.Name);
                    AppendLine($"// {target} aliases buffer param{param.Index}");
                    _leaParamMap[target.Name] = param.Index;
                    return;
                }
            }

            // Emulated 64-bit buffer access
            if (_emulatedVarMappings.TryGetValue(source.ToString(), out var emulInfo))
            {
                string baseIdxVar = $"{source}_base_idx";
                Declare(target);
                // Check if this sampler is already integer-typed
                bool emuSamplerIsInt = _bufferGlslTypes.TryGetValue(emulInfo.ParamIndex, out var emuSamplerType)
                    && (emuSamplerType == "int" || emuSamplerType == "uint");
                string emuFetchLo = $"texelFetch(u_param{emulInfo.ParamIndex}, ivec2({baseIdxVar} % u_param{emulInfo.ParamIndex}_tileW, {baseIdxVar} / u_param{emulInfo.ParamIndex}_tileW), 0).r";
                string emuFetchHi = $"texelFetch(u_param{emulInfo.ParamIndex}, ivec2(({baseIdxVar} + 1) % u_param{emulInfo.ParamIndex}_tileW, ({baseIdxVar} + 1) / u_param{emulInfo.ParamIndex}_tileW), 0).r";
                // Integer samplers already return int/uint; float samplers need floatBitsToUint
                string emuValLo = emuSamplerIsInt ? $"uint({emuFetchLo})" : $"floatBitsToUint({emuFetchLo})";
                string emuValHi = emuSamplerIsInt ? $"uint({emuFetchHi})" : $"floatBitsToUint({emuFetchHi})";
                if (emulInfo.IsF64)
                {
                    AppendLine($"{target} = f64_from_ieee754_bits(");
                    PushIndent();
                    AppendLine($"{emuValLo},");
                    AppendLine($"{emuValHi});");
                    PopIndent();
                }
                else
                {
                    AppendLine($"{target} = uvec2(");
                    PushIndent();
                    AppendLine($"{emuValLo},");
                    AppendLine($"{emuValHi});");
                    PopIndent();
                }
                return;
            }

            // Cross-block pointer fix
            if (_crossBlockPointerExprs.TryGetValue(source.ToString(), out var inlineExpr))
            {
                string crossExpr = $"{inlineExpr}.r";
                // For integer samplers, texelFetch already returns int/uint; just cast to target type
                string castExpr = CastIfNeeded(crossExpr, target.Type);
                if (_hoistedPrimitives.Contains(loadVal))
                    AppendLine($"{target} = {castExpr};");
                else
                {
                    Declare(target);
                    AppendLine($"{target} = {castExpr};");
                }
                return;
            }

            // LEA-based buffer read via texelFetch
            if (_leaParamMap.TryGetValue(source.ToString(), out var leaParamIdx))
            {
                // Check if this is a struct buffer load
                if (_structFieldCounts.TryGetValue(leaParamIdx, out var structFieldCount) && structFieldCount > 0)
                {
                    // Struct buffer: construct struct from per-field texelFetch
                    // Find the element type to get field paths
                    var leaParam = Method.Parameters.FirstOrDefault(p => p.Index == leaParamIdx);
                    if (leaParam != null)
                    {
                        var elemType = UnwrapType(leaParam.ParameterType);
                        var fieldPaths = GenerateStructFieldPaths(elemType);
                        string structType = TypeGenerator[elemType];
                        Declare(target);
                        // Use intBitsToFloat for float fields since texture is R32I
                        for (int fi = 0; fi < fieldPaths.Count; fi++)
                        {
                            string fetchExpr = $"texelFetch(u_param{leaParamIdx}, ivec2((int({source}) * {structFieldCount} + {fi}) % u_param{leaParamIdx}_tileW, (int({source}) * {structFieldCount} + {fi}) / u_param{leaParamIdx}_tileW), 0).r";
                            string valExpr = fieldPaths[fi].GlslType == "float"
                                ? $"intBitsToFloat({fetchExpr})"
                                : fetchExpr;
                            AppendLine($"{target}{fieldPaths[fi].Path} = {valExpr};");
                        }
                        return;
                    }
                }

                // Non-struct: standard single-texel load
                string fetchExprStd = $"texelFetch(u_param{leaParamIdx}, ivec2(int({source}) % u_param{leaParamIdx}_tileW, int({source}) / u_param{leaParamIdx}_tileW), 0).r";
                // For integer samplers (isampler2D/usampler2D), texelFetch already returns int/uint.
                // For float samplers, texelFetch returns float; we may need floatBitsToInt/floatBitsToUint.
                bool samplerIsInt = _bufferGlslTypes.TryGetValue(leaParamIdx, out var samplerGlslType)
                    && (samplerGlslType == "int" || samplerGlslType == "uint");
                string castExpr;
                if (samplerIsInt)
                {
                    // Integer sampler: texelFetch returns int (isampler2D) or uint (usampler2D)
                    // Just cast to target type normally (int→int is identity, uint→int is cast)
                    castExpr = CastIfNeeded(fetchExprStd, target.Type);
                }
                else
                {
                    // Float sampler: texelFetch returns float
                    // For integer targets, we need bit-level reinterpretation
                    if (target.Type == "int")
                        castExpr = $"floatBitsToInt({fetchExprStd})";
                    else if (target.Type == "uint")
                        castExpr = $"floatBitsToUint({fetchExprStd})";
                    else
                        castExpr = CastIfNeeded(fetchExprStd, target.Type);
                }
                if (_hoistedPrimitives.Contains(loadVal))
                    AppendLine($"{target} = {castExpr};");
                else
                {
                    Declare(target);
                    AppendLine($"{target} = {castExpr};");
                }
                return;
            }

            // Standard load
            if (_hoistedPrimitives.Contains(loadVal))
                AppendLine($"{target} = {source};");
            else
            {
                Declare(target);
                AppendLine($"{target} = {source};");
            }
        }

        public override void GenerateCode(global::ILGPU.IR.Values.Store storeVal)
        {
            var address = Load(storeVal.Target);
            var val = Load(storeVal.Value);


            // Emulated 64-bit buffer store via TF output
            if (_emulatedVarMappings.TryGetValue(address.ToString(), out var emulInfo))
            {
                // Find the lo and hi TF varyings for this param
                var loOutput = _outputVaryings.FirstOrDefault(o => o.ParamIndex == emulInfo.ParamIndex && o.EmulatedSuffix == "lo");
                var hiOutput = _outputVaryings.FirstOrDefault(o => o.ParamIndex == emulInfo.ParamIndex && o.EmulatedSuffix == "hi");
                if (loOutput != null && hiOutput != null)
                {
                    if (emulInfo.IsF64)
                    {
                        // f64: vec2(hi_float, lo_float) double-float representation
                        // Must convert back to IEEE 754 bits via f64_to_ieee754_bits() → uvec2(lo, hi)
                        AppendLine($"{{");
                        AppendLine($"  uvec2 _ieee_bits = f64_to_ieee754_bits({val});");
                        AppendLine($"  {loOutput.VaryingName} = _ieee_bits.x;");
                        AppendLine($"  {hiOutput.VaryingName} = _ieee_bits.y;");
                        AppendLine($"}}");
                    }
                    else
                    {
                        // i64/u64: uvec2(lo, hi) — already uint components
                        AppendLine($"{loOutput.VaryingName} = {val}.x;");
                        AppendLine($"{hiOutput.VaryingName} = {val}.y;");
                    }
                }
                return;
            }

            // LEA-based buffer write → TF output
            if (_leaParamMap.TryGetValue(address.ToString(), out var leaParamIdx))
            {
                // Check if this is a struct buffer store
                if (_structFieldCounts.TryGetValue(leaParamIdx, out var structFieldCount) && structFieldCount > 0)
                {
                    // Struct buffer: decompose struct into per-field TF outputs
                    var structOutputs = _outputVaryings.Where(o => o.ParamIndex == leaParamIdx && o.FieldIndex >= 0).OrderBy(o => o.FieldIndex).ToList();
                    var leaParam = Method.Parameters.FirstOrDefault(p => p.Index == leaParamIdx);
                    if (leaParam != null && structOutputs.Count > 0)
                    {
                        var elemType = UnwrapType(leaParam.ParameterType);
                        var fieldPaths = GenerateStructFieldPaths(elemType);
                        for (int fi = 0; fi < Math.Min(fieldPaths.Count, structOutputs.Count); fi++)
                        {
                            string fieldExpr = $"{val}{fieldPaths[fi].Path}";
                            // TF varying type matches the field's native type, so direct assignment
                            AppendLine($"{structOutputs[fi].VaryingName} = {fieldExpr};");
                        }
                        return;
                    }
                }

                var output = _outputVaryings.FirstOrDefault(o => o.ParamIndex == leaParamIdx);
                if (output != null)
                {
                    // Single-element TF output (one value per vertex invocation)
                    // GLSL ES 3.0: cast value to match output type if needed
                    string tfValExpr = CastIfNeeded(val, output.GlslType ?? "float");
                    AppendLine($"{output.VaryingName} = {tfValExpr};");
                    return;
                }
            }

            // Cross-block pointer → resolve to TF output if available
            if (_crossBlockPointerExprs.TryGetValue(address.ToString(), out _))
            {
                // Cross-block pointers should also map to TF outputs via _leaParamMap
                // (propagated through PushPhiValues). Try one more time.
                if (_leaParamMap.TryGetValue(address.ToString(), out var cbParamIdx))
                {
                    var cbOutput = _outputVaryings.FirstOrDefault(o => o.ParamIndex == cbParamIdx);
                    if (cbOutput != null)
                    {
                        string cbValExpr = CastIfNeeded(val, cbOutput.GlslType ?? "float");
                        AppendLine($"{cbOutput.VaryingName} = {cbValExpr};");
                        return;
                    }
                }
                AppendLine($"// Store to cross-block pointer (no TF output found)");
                return;
            }

            // GLSL ES 3.0: cast value to match address type if needed
            string valExpr = CastIfNeeded(val, address.Type);
            AppendLine($"{address} = {valExpr};");
        }

        public override void GenerateCode(NewView value)
        {
            var target = Load(value);
            var source = Load(value.Pointer);
            declaredVariables.Add(target.Name);
            // Propagate the param mapping if source is a param alias
            if (_leaParamMap.TryGetValue(source.ToString(), out var paramIdx))
                _leaParamMap[target.Name] = paramIdx;
            AppendLine($"// NewView: {target} aliases {source}");
        }

        /// <summary>
        /// Handles ArrayView.Length: traces the view back to its kernel parameter
        /// and emits a reference to the u_param{N}_length uniform.
        /// This uniform is already declared by EmitParameterDeclarations and set at dispatch time.
        /// </summary>
        public override void GenerateCode(GetViewLength value)
        {
            var target = Load(value);
            string prefix = _hoistedPrimitives.Contains(value) ? "" : $"{TypeGenerator[value.Type]} ";

            // Trace the view back to its kernel parameter
            var param = ResolveToParameter(value.View);
            if (param != null && param.Index >= KernelParamOffset)
            {
                // GetViewLength returns long (Int64). In emulated i64 mode, the GLSL type is uvec2.
                // We must construct uvec2(uint(length), 0u) rather than assigning int directly.
                string glslType = TypeGenerator[value.Type];
                string lengthExpr;
                if (glslType == "uvec2")
                {
                    // Emulated i64: wrap the int length as uvec2(uint(len), 0u)
                    lengthExpr = $"uvec2(uint(u_param{param.Index}_length), 0u)";
                }
                else
                {
                    // Non-emulated path (shouldn't happen for long, but be safe)
                    lengthExpr = $"u_param{param.Index}_length";
                }
                AppendLine($"{prefix}{target} = {lengthExpr};");
            }
            else
            {
                // Fallback: emit 0 (should not happen for well-formed kernels)
                Declare(target);
                AppendLine($"{target} = 0; // GetViewLength: could not resolve view to parameter");
            }
        }

        public override void GenerateCode(GetField value)
        {
            if (_hoistedIndexFields.Contains(value)) return;

            // Check if this is a kernel parameter (View) field access
            // Guard: when the object is a struct loaded from a struct buffer, we must use
            // standard field access. ResolveToParameter would incorrectly traverse through
            // the struct Load back to the View parameter. We detect this by checking if the
            // object's Load source traces back to a LEA param that has struct field counts.
            bool isLoadedStructFromBuffer = false;
            if (value.ObjectValue.Resolve() is global::ILGPU.IR.Values.Load objLoad
                && _leaParamMap.TryGetValue(Load(objLoad.Source).Name, out var objLeaParamIdx)
                && _structFieldCounts.ContainsKey(objLeaParamIdx))
            {
                isLoadedStructFromBuffer = true;
            }
            if (!isLoadedStructFromBuffer &&
                ResolveToParameter(value.ObjectValue) is global::ILGPU.IR.Values.Parameter param)
            {
                int paramOffset = KernelParamOffset;
                if (param.Index >= paramOffset)
                {
                    IsMultiDim(param.ParameterType, out var isMultiDim, out var isView, out var is1DView, out var is2DView, out var is3DView);
                    if (isView)
                    {
                        var target = Load(value);
                        string prefix = _hoistedPrimitives.Contains(value) ? "" : $"{TypeGenerator[value.Type]} ";

                        if (value.FieldSpan.Index == 0)
                        {
                            // Field 0: buffer pointer alias
                            declaredVariables.Add(target.Name);
                            _leaParamMap[target.Name] = param.Index;
                            AppendLine($"// {target} = buffer alias for param{param.Index}");
                            return;
                        }

                        // Buffer length via uniform
                        string totalLen = $"u_param{param.Index}_length";

                        if (is2DView)
                        {
                            // Stride fields: some may be Int64 (→ uvec2), some Int32 (→ int).
                            // Wrap with i64_from_i32() only when the target type is uvec2.
                            string glslType = TypeGenerator[value.Type];
                            bool needsI64 = glslType == "uvec2";
                            switch (value.FieldSpan.Index)
                            {
                                case 1:
                                    if (needsI64) AppendLine($"{prefix}{target} = i64_from_i32(u_param{param.Index}_stride[0]);");
                                    else AppendLine($"{prefix}{target} = u_param{param.Index}_stride[0];");
                                    return;
                                case 2:
                                    if (needsI64) AppendLine($"{prefix}{target} = i64_from_i32(u_param{param.Index}_stride[1]);");
                                    else AppendLine($"{prefix}{target} = u_param{param.Index}_stride[1];");
                                    return;
                                case 3:
                                    if (needsI64) AppendLine($"{prefix}{target} = i64_from_i32(u_param{param.Index}_stride[0]);");
                                    else AppendLine($"{prefix}{target} = u_param{param.Index}_stride[0];");
                                    return;
                            }
                        }
                        else if (is3DView)
                        {
                            // Same type-aware conversion as 2D
                            string glslType = TypeGenerator[value.Type];
                            bool needsI64 = glslType == "uvec2";
                            switch (value.FieldSpan.Index)
                            {
                                case 1:
                                    if (needsI64) AppendLine($"{prefix}{target} = i64_from_i32(u_param{param.Index}_stride[0]);");
                                    else AppendLine($"{prefix}{target} = u_param{param.Index}_stride[0];");
                                    return;
                                case 2:
                                    if (needsI64) AppendLine($"{prefix}{target} = i64_from_i32(u_param{param.Index}_stride[1]);");
                                    else AppendLine($"{prefix}{target} = u_param{param.Index}_stride[1];");
                                    return;
                                case 3:
                                    if (needsI64) AppendLine($"{prefix}{target} = i64_from_i32(u_param{param.Index}_stride[2]);");
                                    else AppendLine($"{prefix}{target} = u_param{param.Index}_stride[2];");
                                    return;
                                case 4:
                                    if (needsI64) AppendLine($"{prefix}{target} = i64_from_i32(u_param{param.Index}_stride[0]);");
                                    else AppendLine($"{prefix}{target} = u_param{param.Index}_stride[0];");
                                    return;
                                case 5:
                                    if (needsI64) AppendLine($"{prefix}{target} = i64_from_i32(u_param{param.Index}_stride[0] * u_param{param.Index}_stride[1]);");
                                    else AppendLine($"{prefix}{target} = u_param{param.Index}_stride[0] * u_param{param.Index}_stride[1];");
                                    return;
                            }
                        }
                        else if (is1DView)
                        {
                            switch (value.FieldSpan.Index)
                            {
                                case 1: AppendLine($"{prefix}{target} = 0;"); return; // Index
                                case 2: AppendLine($"{prefix}{target} = {totalLen};"); return; // Length
                            }
                        }

                        AppendLine($"{prefix}{target} = {totalLen};");
                        return;
                    }
                }

                // Kernel index parameter field access
                if (param.Index < paramOffset)
                {
                    var target = Load(value);
                    var source = Load(value.ObjectValue);
                    string prefix = _hoistedPrimitives.Contains(value) ? "" : "int ";

                    if (EntryPoint.IndexType == IndexType.Index2D)
                    {
                        string comp = value.FieldSpan.Index == 0 ? "x" : "y";
                        AppendLine($"{prefix}{target} = {source}.{comp};");
                        return;
                    }
                    else if (EntryPoint.IndexType == IndexType.Index3D)
                    {
                        string comp = value.FieldSpan.Index switch { 0 => "x", 1 => "y", _ => "z" };
                        AppendLine($"{prefix}{target} = {source}.{comp};");
                        return;
                    }
                    else if (EntryPoint.IndexType == IndexType.Index1D)
                    {
                        if (value.FieldSpan.Index == 0)
                            AppendLine($"{prefix}{target} = {source};");
                        else
                            AppendLine($"{prefix}{target} = 0;");
                        return;
                    }
                }
            }

            // Standard field access
            base.GenerateCode(value);
        }

        public override void GenerateCode(BinaryArithmeticValue value)
        {
            var target = Load(value);
            var left = Load(value.Left);
            var right = Load(value.Right);
            string prefix = _hoistedPrimitives.Contains(value) ? "" : $"{TypeGenerator[value.Type]} ";

            var leftType = TypeGenerator[value.Left.Type];
            var rightType = TypeGenerator[value.Right.Type];
            bool isEmulatedF64 = Backend.Options.EnableF64Emulation && (leftType == "vec2" || rightType == "vec2");
            bool isEmulatedI64 = Backend.Options.EnableI64Emulation && (leftType == "uvec2" || rightType == "uvec2");

            if (isEmulatedF64)
            {
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
                if (emulFunc != null) { AppendLine($"{prefix}{target} = {emulFunc}({left}, {right});"); return; }
            }

            if (isEmulatedI64)
            {
                string? emulFunc = value.Kind switch
                {
                    BinaryArithmeticKind.Add => "i64_add",
                    BinaryArithmeticKind.Sub => "i64_sub",
                    BinaryArithmeticKind.Mul => "i64_mul",
                    BinaryArithmeticKind.And => "i64_and",
                    BinaryArithmeticKind.Or => "i64_or",
                    BinaryArithmeticKind.Xor => "i64_xor",
                    BinaryArithmeticKind.Shl => "i64_shl",
                    BinaryArithmeticKind.Shr => "i64_shr",
                    _ => null
                };
                if (emulFunc != null)
                {
                    if (value.Kind == BinaryArithmeticKind.Shl || value.Kind == BinaryArithmeticKind.Shr)
                        AppendLine($"{prefix}{target} = {emulFunc}({left}, uint({right}));");
                    else
                        AppendLine($"{prefix}{target} = {emulFunc}({left}, {right});");
                    return;
                }
            }

            // Standard ops
            if (value.Kind == BinaryArithmeticKind.Min || value.Kind == BinaryArithmeticKind.Max)
            {
                string func = value.Kind == BinaryArithmeticKind.Min ? "min" : "max";
                AppendLine($"{prefix}{target} = {func}({left}, {right});");
                return;
            }

            if (value.Kind == BinaryArithmeticKind.PowF)
            {
                AppendLine($"{prefix}{target} = pow({left}, {right});");
                return;
            }

            if (value.Kind == BinaryArithmeticKind.Atan2F)
            {
                AppendLine($"{prefix}{target} = atan({left}, {right});");
                return;
            }

            // Check if this is a boolean operation. GLSL requires logical operators
            // (&&, ||) for booleans, not bitwise (&, |).
            string op;
            var resultGlslType = TypeGenerator[value.Type];
            bool isBoolOp = resultGlslType == "bool"
                || TypeGenerator[value.Left.Type] == "bool"
                || booleanVariables.Contains(target.Name);
            if (isBoolOp && value.Kind == BinaryArithmeticKind.And)
                op = "&&";
            else if (isBoolOp && value.Kind == BinaryArithmeticKind.Or)
                op = "||";
            else
                op = GetArithmeticOp(value.Kind);
            AppendLine($"{prefix}{target} = {left} {op} {right};");
        }

        public override void GenerateCode(UnaryArithmeticValue value)
        {
            var target = Load(value);
            var operand = Load(value.Value);
            var operandType = TypeGenerator[value.Value.Type];
            string prefix = _hoistedPrimitives.Contains(value) ? "" : $"{TypeGenerator[value.Type]} ";

            bool isEmulatedF64 = Backend.Options.EnableF64Emulation && operandType == "vec2";
            bool isEmulatedI64 = Backend.Options.EnableI64Emulation && operandType == "uvec2";

            if (isEmulatedF64)
            {
                string? emulFunc = value.Kind switch
                {
                    UnaryArithmeticKind.Neg => "f64_neg",
                    UnaryArithmeticKind.Abs => "f64_abs",
                    _ => null
                };
                if (emulFunc != null) { AppendLine($"{prefix}{target} = {emulFunc}({operand});"); return; }
            }

            if (isEmulatedI64)
            {
                string? emulFunc = value.Kind switch
                {
                    UnaryArithmeticKind.Neg => "i64_neg",
                    UnaryArithmeticKind.Abs => "i64_abs",
                    _ => null
                };
                if (emulFunc != null) { AppendLine($"{prefix}{target} = {emulFunc}({operand});"); return; }
            }

            // Fall back to base class for non-emulated types
            base.GenerateCode(value);
        }

        public override void GenerateCode(TernaryArithmeticValue value)
        {
            var target = Load(value);
            var first = Load(value.First);
            var second = Load(value.Second);
            var third = Load(value.Third);
            string prefix = _hoistedPrimitives.Contains(value) ? "" : $"{TypeGenerator[value.Type]} ";

            var firstType = TypeGenerator[value.First.Type];
            bool isEmulatedF64 = Backend.Options.EnableF64Emulation && firstType == "vec2";

            if (isEmulatedF64)
            {
                // FMA: a*b+c using double-float emulation
                AppendLine($"{prefix}{target} = f64_add(f64_mul({first}, {second}), {third});");
                return;
            }

            // Fall back: GLSL ES 3.0 has no fma() — emulate with a*b+c
            AppendLine($"{prefix}{target} = ({first} * {second} + {third});");
        }

        public override void GenerateCode(CompareValue value)
        {
            var target = Load(value);
            var left = Load(value.Left);
            var right = Load(value.Right);
            string prefix = _hoistedPrimitives.Contains(value) ? "" : "bool ";

            var leftType = TypeGenerator[value.Left.Type];
            bool isEmulatedF64 = Backend.Options.EnableF64Emulation && leftType == "vec2";
            bool isEmulatedI64 = Backend.Options.EnableI64Emulation && leftType == "uvec2";

            if (isEmulatedF64)
            {
                string? f = value.Kind switch
                {
                    CompareKind.LessThan => "f64_lt", CompareKind.LessEqual => "f64_le",
                    CompareKind.GreaterThan => "f64_gt", CompareKind.GreaterEqual => "f64_ge",
                    CompareKind.Equal => "f64_eq", CompareKind.NotEqual => "f64_ne", _ => null
                };
                if (f != null) { AppendLine($"{prefix}{target} = {f}({left}, {right});"); return; }
            }

            if (isEmulatedI64)
            {
                string? f = value.Kind switch
                {
                    CompareKind.LessThan => "i64_lt", CompareKind.LessEqual => "i64_le",
                    CompareKind.GreaterThan => "i64_gt", CompareKind.GreaterEqual => "i64_ge",
                    CompareKind.Equal => "i64_eq", CompareKind.NotEqual => "i64_ne", _ => null
                };
                if (f != null) { AppendLine($"{prefix}{target} = {f}({left}, {right});"); return; }
            }

            string op = GetCompareOp(value.Kind);
            AppendLine($"{prefix}{target} = {left} {op} {right};");
        }

        public override void GenerateCode(ConvertValue value)
        {
            var target = Load(value);
            var source = Load(value.Value);
            var targetType = TypeGenerator[value.Type];
            var sourceType = TypeGenerator[value.Value.Type];
            string prefix = _hoistedPrimitives.Contains(value) ? "" : $"{targetType} ";

            bool isEmulatedF64Target = Backend.Options.EnableF64Emulation && targetType == "vec2";
            bool isEmulatedI64Target = Backend.Options.EnableI64Emulation && targetType == "uvec2";
            bool isEmulatedF64Source = Backend.Options.EnableF64Emulation && sourceType == "vec2";
            bool isEmulatedI64Source = Backend.Options.EnableI64Emulation && sourceType == "uvec2";

            if (isEmulatedF64Target)
            {
                if (isEmulatedF64Source) AppendLine($"{prefix}{target} = {source};");
                else if (isEmulatedI64Source) AppendLine($"{prefix}{target} = f64_from_f32(float(i64_to_i32({source})));");
                else if (sourceType == "float") AppendLine($"{prefix}{target} = f64_from_f32({source});");
                else AppendLine($"{prefix}{target} = f64_from_f32(float({source}));");
                return;
            }
            if (isEmulatedI64Target)
            {
                if (isEmulatedI64Source) AppendLine($"{prefix}{target} = {source};");
                else if (isEmulatedF64Source) AppendLine($"{prefix}{target} = i64_from_i32(int(f64_to_f32({source})));");
                else if (sourceType == "int") AppendLine($"{prefix}{target} = i64_from_i32({source});");
                else if (sourceType == "uint") AppendLine($"{prefix}{target} = u64_from_u32({source});");
                else AppendLine($"{prefix}{target} = i64_from_i32(int({source}));");
                return;
            }
            if (isEmulatedF64Source)
            {
                if (targetType == "float") AppendLine($"{prefix}{target} = f64_to_f32({source});");
                else AppendLine($"{prefix}{target} = {targetType}(f64_to_f32({source}));");
                return;
            }
            if (isEmulatedI64Source)
            {
                if (targetType == "int") AppendLine($"{prefix}{target} = i64_to_i32({source});");
                else if (targetType == "uint") AppendLine($"{prefix}{target} = u64_to_u32({source});");
                else if (targetType == "float") AppendLine($"{prefix}{target} = float(i64_to_i32({source}));");
                else AppendLine($"{prefix}{target} = {targetType}(i64_to_i32({source}));");
                return;
            }

            AppendLine($"{prefix}{target} = {targetType}({source});");
        }

        #endregion

        #region Helpers

        private static string GetArithmeticOp(BinaryArithmeticKind kind) => kind switch
        {
            BinaryArithmeticKind.Add => "+",
            BinaryArithmeticKind.Sub => "-",
            BinaryArithmeticKind.Mul => "*",
            BinaryArithmeticKind.Div => "/",
            BinaryArithmeticKind.Rem => "%",
            BinaryArithmeticKind.And => "&",
            BinaryArithmeticKind.Or => "|",
            BinaryArithmeticKind.Xor => "^",
            BinaryArithmeticKind.Shl => "<<",
            BinaryArithmeticKind.Shr => ">>",
            _ => "+"
        };

        private static string GetCompareOp(CompareKind kind) => kind switch
        {
            CompareKind.Equal => "==",
            CompareKind.NotEqual => "!=",
            CompareKind.LessThan => "<",
            CompareKind.LessEqual => "<=",
            CompareKind.GreaterThan => ">",
            CompareKind.GreaterEqual => ">=",
            _ => "=="
        };

        private static void IsMultiDim(TypeNode type, out bool isMultiDim, out bool isView,
            out bool is1DView, out bool is2DView, out bool is3DView)
        {
            is1DView = false;
            is2DView = false;
            is3DView = false;
            isMultiDim = false;
            isView = false;

            // 1. Direct type check: ViewType is a 1D view
            if (type is ViewType)
            {
                isView = true;
                is1DView = true;
                return;
            }

            string typeName = type.ToString();

            // 2. Structural check: ArrayView2D/3D are StructureType wrappers around a ViewType
            //    Port from the WGSL backend's proven detection logic
            if (type is StructureType st)
            {
                // Check if first flattened field is a ViewType or its string contains "View"
                if (st.NumFields > 0 && (st.Fields[0] is ViewType || st.Fields[0] is PointerType || st.Fields[0].ToString().Contains("View")))
                {
                    isView = true;

                    // Distinguish 1D/2D/3D by flattened field count:
                    //   1D ArrayView<T> = 3 fields (pointer, index, length) but 1D is ViewType not StructureType
                    //   2D ArrayView2D<T> = 4 fields (pointer, index, length, stride)
                    //   3D ArrayView3D<T> = 6 fields (pointer, index, length, strideX, strideY, strideZ)
                    if (st.NumFields == 6)
                    {
                        is3DView = true;
                        isMultiDim = true;
                    }
                    else if (st.NumFields == 4)
                    {
                        is2DView = true;
                        isMultiDim = true;
                    }
                    else if (st.NumFields == 3)
                    {
                        is1DView = true;
                        isMultiDim = false;
                    }
                }
            }

            // 3. Fallback: string check (covers any edge cases)
            if (!isView && typeName.Contains("ArrayView"))
            {
                isView = true;
                if (typeName.Contains("2D") || typeName.Contains("3D")) isMultiDim = true;
                if (typeName.Contains("3D")) is3DView = true;
                else if (typeName.Contains("2D")) is2DView = true;
                else is1DView = true;
            }
        }

        private global::ILGPU.IR.Values.Parameter? ResolveToParameter(Value value)
        {
            if (value is global::ILGPU.IR.Values.Parameter p) return p;
            if (value is GetField gf) return ResolveToParameter(gf.ObjectValue);
            if (value is global::ILGPU.IR.Values.Load load) return ResolveToParameter(load.Source);
            if (value is LoadElementAddress lea) return ResolveToParameter(lea.Source);
            if (value is LoadFieldAddress lfa) return ResolveToParameter(lfa.Source);
            return null;
        }

        #endregion
    }

    #region Supporting Types

    public enum KernelParamKind
    {
        Buffer,
        Scalar,
        Stride,
        EmulatedF64,
        EmulatedI64
    }

    public readonly struct KernelParameterBinding
    {
        public int ParamIndex { get; }
        public int BindingIndex { get; }
        public KernelParamKind Kind { get; }
        public string GlslType { get; }
        /// <summary>For struct element buffers: number of flattened scalar fields per element. 0 for non-struct.</summary>
        public int StructFieldCount { get; }

        public KernelParameterBinding(int paramIndex, int bindingIndex, KernelParamKind kind, string glslType, int structFieldCount = 0)
        {
            ParamIndex = paramIndex;
            BindingIndex = bindingIndex;
            Kind = kind;
            GlslType = glslType;
            StructFieldCount = structFieldCount;
        }
    }

    public class OutputVaryingInfo
    {
        public int ParamIndex { get; }
        public int OutputIndex { get; }
        public string VaryingName { get; }
        public string GlslType { get; }
        /// <summary>True if this varying is one half of an emulated 64-bit value (lo or hi).</summary>
        public bool IsEmulated { get; }
        /// <summary>For emulated varyings: "lo" or "hi". Null for normal varyings.</summary>
        public string? EmulatedSuffix { get; }
        /// <summary>For struct field varyings: the field index within the struct. -1 for non-struct.</summary>
        public int FieldIndex { get; }

        public OutputVaryingInfo(int paramIndex, int outputIndex, string varyingName, string glslType,
            bool isEmulated = false, string? emulatedSuffix = null, int fieldIndex = -1)
        {
            ParamIndex = paramIndex;
            OutputIndex = outputIndex;
            VaryingName = varyingName;
            GlslType = glslType;
            IsEmulated = isEmulated;
            EmulatedSuffix = emulatedSuffix;
            FieldIndex = fieldIndex;
        }
    }

    #endregion
}
