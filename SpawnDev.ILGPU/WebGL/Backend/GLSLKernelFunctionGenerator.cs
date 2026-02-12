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

        // Hoisting
        private readonly HashSet<Value> _hoistedPrimitives = new();
        private readonly HashSet<Value> _hoistedIndexFields = new();

        // CFG and analysis
        private CFG<ReversePostOrder, Forwards> _cfg;
        private Dominators<Backwards> _postDominators;
        private Loops<ReversePostOrder, Forwards> _loops;
        private HashSet<BasicBlock> _visitedBlocks = new();

        // Parameter binding info for runtime
        private readonly List<KernelParameterBinding> _parameterBindings = new();

        // Output buffer info
        private readonly List<OutputVaryingInfo> _outputVaryings = new();

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

            // 2. Analyze hoisting needs
            AnalyzeHoisting();

            // 3. Post-dominators and loops already created in constructor

            // 4. Emit emulation library (type aliases + helper functions must precede
            //    struct definitions that may reference vec2/uvec2 emulated types)
            if (Backend.Options.EnableF64Emulation || Backend.Options.EnableI64Emulation)
            {
                Builder.AppendLine("// ============ 64-bit Emulation Library ============");
                Builder.AppendLine(GLSLEmulationLibrary.GetEmulationLibrary(
                    Backend.Options.EnableF64Emulation,
                    Backend.Options.EnableI64Emulation));
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

        private static TypeNode UnwrapType(TypeNode type)
        {
            while (type is PointerType pt) type = pt.ElementType;
            while (type is ViewType vt) type = vt.ElementType;
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
                    string glslType = GetBufferElementType(elementType);

                    // Input buffer: use a uniform sampler (TBO emulated as texture)
                    // Use isampler2D for int buffers, usampler2D for uint, sampler2D for float
                    string samplerType = glslType switch
                    {
                        "int" => "isampler2D",
                        "uint" => "usampler2D",
                        _ => "sampler2D"
                    };
                    Builder.AppendLine($"uniform highp {samplerType} u_param{param.Index}; // buffer param[{param.Index}]");
                    _parameterBindings.Add(new KernelParameterBinding(param.Index, bindingIndex++, KernelParamKind.Buffer, glslType));
                    _generatorArgs.ParameterBindings.Add(_parameterBindings[^1]);
                    _bufferGlslTypes[param.Index] = glslType;

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
                    // This could be an output buffer — we mark all view params as potential TF outputs
                    // The actual write determination is done in the kernel body
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
                    else
                    {
                        string glslType = GetBufferElementType(elementType);

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

            // Structured code gen for acyclic graphs
            if (_loops.Count == 0 && _postDominators != null)
            {
                _visitedBlocks.Clear();
                GenerateStructuredCode(Method.EntryBlock, null);
                return;
            }

            // State machine for cyclic graphs
            int deferredPos = Builder.Length;
            AppendLine("int current_block = 0;");
            AppendLine("for (;;) {");
            PushIndent();
            AppendLine("switch (current_block) {");
            PushIndent();

            foreach (var block in Method.Blocks)
            {
                AppendLine($"case {GetBlockIndex(block)}: {{");
                PushIndent();
                GenerateBlockCode(block);
                PopIndent();
                AppendLine("}");
            }

            AppendLine("default: break;");
            PopIndent();
            AppendLine("}"); // switch
            AppendLine("if (current_block == -1) break;");
            PopIndent();
            AppendLine("}"); // for

            // Insert deferred declarations
            if (VariableBuilder.Length > 0)
                Builder.Insert(deferredPos, VariableBuilder.ToString());
        }

        private void GenerateBlockCode(BasicBlock block)
        {
            foreach (var value in block)
            {
                if (value.Value is TerminatorValue) continue;
                GenerateCodeFor(value.Value);
            }

            // Handle terminator
            if (!(_loops.Count == 0 && Method.Blocks.Count > 1 && _postDominators != null))
            {
                var term = block.Terminator;
                if (term is UnconditionalBranch ub) GenerateCode(ub);
                else if (term is IfBranch ib) GenerateCode(ib);
                else if (term is SwitchBranch sb) GenerateCode(sb);
                else if (term is ReturnTerminator rt) GenerateCode(rt);
                else if (term != null) GenerateCodeFor(term);
            }
        }

        private void GenerateStructuredCode(BasicBlock current, BasicBlock? stop)
        {
            if (current == null || current == stop || _visitedBlocks.Contains(current)) return;
            _visitedBlocks.Add(current);

            GenerateBlockCode(current);

            var terminator = current.Terminator;

            if (terminator is UnconditionalBranch ub)
            {
                PushPhiValues(ub.Target, current);
                GenerateStructuredCode(ub.Target, stop);
            }
            else if (terminator is IfBranch ib)
            {
                var trueTarget = ib.TrueTarget;
                var falseTarget = ib.FalseTarget;
                var merge = _postDominators?.GetImmediateDominator(current);

                PushPhiValues(trueTarget, current);
                var cond = Load(ib.Condition);
                AppendLine($"if ({cond}) {{");
                PushIndent();
                GenerateStructuredCode(trueTarget, merge);
                PopIndent();
                AppendLine("} else {");
                PushIndent();
                PushPhiValues(falseTarget, current);
                GenerateStructuredCode(falseTarget, merge);
                PopIndent();
                AppendLine("}");

                if (merge != null && merge != stop)
                    GenerateStructuredCode(merge, stop);
            }
            else if (terminator is ReturnTerminator rt)
            {
                GenerateCode(rt);
            }
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
                        _crossBlockPointerExprs[target.Name] = $"texelFetch(u_param{param.Index}, ivec2({offset}, 0), 0)";

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

            Declare(target);
            var sourceVal = Load(value.Source);
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
                string emuFetchLo = $"texelFetch(u_param{emulInfo.ParamIndex}, ivec2({baseIdxVar}, 0), 0).r";
                string emuFetchHi = $"texelFetch(u_param{emulInfo.ParamIndex}, ivec2({baseIdxVar} + 1, 0), 0).r";
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
                string fetchExpr = $"texelFetch(u_param{leaParamIdx}, ivec2({source}, 0), 0).r";
                // For integer samplers (isampler2D/usampler2D), texelFetch already returns int/uint.
                // For float samplers, texelFetch returns float; we may need floatBitsToInt/floatBitsToUint.
                bool samplerIsInt = _bufferGlslTypes.TryGetValue(leaParamIdx, out var samplerGlslType)
                    && (samplerGlslType == "int" || samplerGlslType == "uint");
                string castExpr;
                if (samplerIsInt)
                {
                    // Integer sampler: texelFetch returns int (isampler2D) or uint (usampler2D)
                    // Just cast to target type normally (int→int is identity, uint→int is cast)
                    castExpr = CastIfNeeded(fetchExpr, target.Type);
                }
                else
                {
                    // Float sampler: texelFetch returns float
                    // For integer targets, we need bit-level reinterpretation
                    if (target.Type == "int")
                        castExpr = $"floatBitsToInt({fetchExpr})";
                    else if (target.Type == "uint")
                        castExpr = $"floatBitsToUint({fetchExpr})";
                    else
                        castExpr = CastIfNeeded(fetchExpr, target.Type);
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

        public override void GenerateCode(GetField value)
        {
            if (_hoistedIndexFields.Contains(value)) return;

            // Check if this is a kernel parameter (View) field access
            if (ResolveToParameter(value.ObjectValue) is global::ILGPU.IR.Values.Parameter param)
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
                            switch (value.FieldSpan.Index)
                            {
                                case 1: AppendLine($"{prefix}{target} = u_param{param.Index}_stride[0];"); return;
                                case 2: AppendLine($"{prefix}{target} = u_param{param.Index}_stride[1];"); return;
                                case 3: AppendLine($"{prefix}{target} = u_param{param.Index}_stride[0];"); return;
                            }
                        }
                        else if (is3DView)
                        {
                            switch (value.FieldSpan.Index)
                            {
                                case 1: AppendLine($"{prefix}{target} = u_param{param.Index}_stride[0];"); return;
                                case 2: AppendLine($"{prefix}{target} = u_param{param.Index}_stride[1];"); return;
                                case 3: AppendLine($"{prefix}{target} = u_param{param.Index}_stride[2];"); return;
                                case 4: AppendLine($"{prefix}{target} = u_param{param.Index}_stride[0];"); return;
                                case 5: AppendLine($"{prefix}{target} = u_param{param.Index}_stride[0] * u_param{param.Index}_stride[1];"); return;
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
            string typeName = type.ToString();
            isView = type is ViewType || typeName.Contains("ArrayView");
            is1DView = isView && !typeName.Contains("2D") && !typeName.Contains("3D");
            is2DView = typeName.Contains("2D");
            is3DView = typeName.Contains("3D");
            isMultiDim = is2DView || is3DView;
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

        public KernelParameterBinding(int paramIndex, int bindingIndex, KernelParamKind kind, string glslType)
        {
            ParamIndex = paramIndex;
            BindingIndex = bindingIndex;
            Kind = kind;
            GlslType = glslType;
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

        public OutputVaryingInfo(int paramIndex, int outputIndex, string varyingName, string glslType,
            bool isEmulated = false, string? emulatedSuffix = null)
        {
            ParamIndex = paramIndex;
            OutputIndex = outputIndex;
            VaryingName = varyingName;
            GlslType = glslType;
            IsEmulated = isEmulated;
            EmulatedSuffix = emulatedSuffix;
        }
    }

    #endregion
}
