// ---------------------------------------------------------------------------------------
//                                 SpawnDev.ILGPU.WebGPU
//                        Copyright (c) 2024 SpawnDev Project
//
// File: WGSLCodeGenerator.cs
//
// Base WGSL code generator implementing IBackendCodeGenerator for WebGPU backend.
// ---------------------------------------------------------------------------------------

using global::ILGPU;
using global::ILGPU.Backends;
using global::ILGPU.Backends.EntryPoints;
using global::ILGPU.IR;
using global::ILGPU.IR.Analyses;
using global::ILGPU.IR.Types;
using global::ILGPU.IR.Values;
using System.Reflection;
using System.Text;

namespace SpawnDev.ILGPU.WebGPU.Backend
{
    /// <summary>
    /// Base class for WGSL code generation. Generates WGSL compute shader source
    /// from ILGPU IR values by implementing the IBackendCodeGenerator interface.
    /// </summary>
    public abstract partial class WGSLCodeGenerator : IBackendCodeGenerator<StringBuilder>
    {
        #region Nested Types

        /// <summary>
        /// Generation arguments for WGSL code generator construction.
        /// </summary>
        public readonly struct GeneratorArgs
        {
            /// <summary>
            /// Creates new generator args.
            /// </summary>
            public GeneratorArgs(
                WebGPUBackend backend,
                WGSLTypeGenerator typeGenerator,
                EntryPoint entryPoint,
                AllocaKindInformation sharedAllocations,
                AllocaKindInformation dynamicSharedAllocations,
                int? maxWorkgroupSize = null)
            {
                Backend = backend;
                TypeGenerator = typeGenerator;
                EntryPoint = entryPoint;
                SharedAllocations = sharedAllocations;
                DynamicSharedAllocations = dynamicSharedAllocations;
                MaxWorkgroupSize = maxWorkgroupSize;
                DynamicSharedOverrides = new List<DynamicSharedOverrideInfo>();
                ScalarPackingManifest = new List<ScalarPackingEntry>();
                HelperMethods = new Dictionary<Method, Allocas>();
                
                // Pre-populate SharedMemoryVarNames with deterministic names.
                // This MUST happen before helper function generators run (which is
                // before the kernel generator), so both can use the same names.
                // We store both the var name AND the full declaration string because
                // the helper function generator needs to emit the declaration but
                // doesn't have access to AllocaKindInformation metadata.
                SharedMemoryVarNames = new Dictionary<Value, (string Name, string Declaration)>();
                int sharedIdx = 0;
                foreach (var alloca in sharedAllocations)
                {
                    var name = $"shared_{sharedIdx}";
                    var wgslType = typeGenerator[alloca.ElementType];
                    int entryCount = (int)alloca.ArraySize;
                    var declaration = $"var<workgroup> {name} : array<{wgslType}, {entryCount}>;";
                    SharedMemoryVarNames[alloca.Alloca] = (name, declaration);
                    sharedIdx++;
                }
            }

            /// <summary>The parent backend.</summary>
            public WebGPUBackend Backend { get; }

            /// <summary>The type generator.</summary>
            public WGSLTypeGenerator TypeGenerator { get; }

            /// <summary>The kernel entry point.</summary>
            public EntryPoint EntryPoint { get; }

            /// <summary>Shared memory allocations.</summary>
            public AllocaKindInformation SharedAllocations { get; }

            /// <summary>Dynamic shared memory allocations.</summary>
            public AllocaKindInformation DynamicSharedAllocations { get; }

            /// <summary>
            /// The maximum workgroup size for this kernel, derived from KernelSpecialization.
            /// When set, this overrides the hardcoded default workgroup sizes (64 for 1D, etc.)
            /// to match the actual group dimension used during kernel compilation.
            /// This is critical for explicitly grouped kernels where SpecializedValue<int>(groupDim)
            /// triggers recompilation with a specific group size that must match @workgroup_size.
            /// </summary>
            public int? MaxWorkgroupSize { get; }

            /// <summary>
            /// Populated by the kernel code generator during GenerateHeader() with
            /// the WGSL override constant names for dynamic shared memory.
            /// </summary>
            public List<DynamicSharedOverrideInfo> DynamicSharedOverrides { get; }

            /// <summary>
            /// Populated by the kernel code generator during GenerateHeader() with
            /// the scalar parameter packing layout for this kernel.
            /// </summary>
            public List<ScalarPackingEntry> ScalarPackingManifest { get; }

            /// <summary>
            /// Populated by the kernel code generator at the end of GenerateHeader() with
            /// the total number of @group(0) bindings emitted. Used at runtime to validate
            /// that the bind group entry count matches the WGSL layout. List with 1 element (mutable in readonly struct).
            /// </summary>
            public List<int> ExpectedBindingCountHolder { get; } = new() { 0 };

            /// <summary>
            /// Maps shared memory Alloca IR values to their module-scope WGSL variable names
            /// and full declarations. Pre-populated in constructor from backendContext.
            /// The tuple contains (VarName, FullDeclaration) where Declaration is
            /// the complete WGSL var<workgroup> statement ready to emit.
            /// </summary>
            public Dictionary<Value, (string Name, string Declaration)> SharedMemoryVarNames { get; }

            /// <summary>
            /// Maps helper methods to their allocas for inlining at call sites.
            /// Populated by WebGPUBackend.CreateFunctionCodeGenerator().
            /// </summary>
            public Dictionary<Method, Allocas> HelperMethods { get; }
        }

        /// <summary>
        /// Represents a variable in WGSL code.
        /// </summary>
        /// <summary>
        /// Represents a variable in WGSL code.
        /// </summary>
        public class Variable
        {
            public Variable(string name, string type)
            {
                Name = name;
                Type = type;
            }

            public string Name { get; }
            public string Type { get; }

            public override string ToString() => Name;
        }

        #endregion

        #region Instance

        protected int varCounter = 0;
        protected int labelCounter = 0;
        protected readonly Dictionary<Value, Variable> valueVariables = new();
        private readonly Dictionary<BasicBlock, string> blockLabels = new();
        protected readonly GeneratorArgs _baseArgs;

        // Local array support (NewArray + LoadArrayElementAddress)
        protected readonly Dictionary<string, string> _allocaArrayNames = new();
        protected int _localArrayCounter = 0;

        // Track local alloca variable names (declared as WGSL 'var', NOT pointers).
        // Store/Load/LoadFieldAddress must NOT use pointer dereference (*) on these.
        protected readonly HashSet<string> _localAllocaVarNames = new();

        // Flag to tracking if we are generating code within the state machine loop
        protected bool IsStateMachineActive { get; set; } = false;

        private StringBuilder prefixBuilder = new StringBuilder();
        private StringBuilder suffixBuilder = new StringBuilder();

        /// <summary>
        /// Constructs a new WGSL code generator.
        /// </summary>
        protected WGSLCodeGenerator(in GeneratorArgs args, Method method, Allocas allocas)
        {
            Backend = args.Backend;
            TypeGenerator = args.TypeGenerator;
            Method = method;
            Allocas = allocas;
            _baseArgs = args;

            Builder = prefixBuilder;
        }

        #endregion

        #region Properties

        /// <summary>The parent backend.</summary>
        public WebGPUBackend Backend { get; }

        /// <summary>The type generator.</summary>
        public WGSLTypeGenerator TypeGenerator { get; }

        /// <summary>The current method being generated.</summary>
        public Method Method { get; }

        /// <summary>All local allocas.</summary>
        public Allocas Allocas { get; }

        /// <summary>The current string builder.</summary>
        public StringBuilder Builder { get; protected set; }

        /// <summary>Builder for variable declarations.</summary>
        public StringBuilder VariableBuilder { get; } = new StringBuilder();

        /// <summary>The current intrinsic provider for code-generation purposes.</summary>
        public global::ILGPU.IR.Intrinsics.IntrinsicImplementationProvider<WGSLIntrinsic.Handler> ImplementationProvider => Backend.IntrinsicProvider;

        /// <summary>Current indentation level.</summary>
        protected int IndentLevel { get; set; } = 0;

        /// <summary>
        /// Returns true if the 'subgroups' WebGPU feature is enabled on the device.
        /// When true, native subgroup intrinsics (subgroupShuffle, subgroup_invocation_id, etc.) are used.
        /// When false, workgroup shared-memory emulation is used instead.
        /// </summary>
        protected bool HasSubgroups => Backend.HasSubgroups;

        /// <summary>
        /// When set to true, the kernel generator emits the group-reduce workgroup variables
        /// (atomic&lt;i32&gt;, atomic&lt;u32&gt;, per-subgroup result array) at module scope.
        /// Set by GenerateGroupAllReduce during body generation so the header knows to include them.
        /// </summary>
        public virtual bool UsesGroupReduce { get; set; } = false;

        /// <summary>
        /// When set to true, the kernel generator emits the warp shuffle emulation shared memory buffer
        /// (_warp_shuffle_buf) at module scope. Set by the Reduce handler when subgroups are unavailable
        /// and a workgroup-level shared-memory reduction is needed.
        /// </summary>
        public virtual bool UsesWarpShuffleEmulation { get; set; } = false;

        #endregion

        #region IBackendCodeGenerator

        /// <summary>
        /// Generates a function declaration header.
        /// </summary>
        public abstract void GenerateHeader(StringBuilder builder);

        /// <summary>
        /// Generates the function body code.
        /// </summary>
        public abstract void GenerateCode();

        /// <summary>
        /// Generates constant declarations (not used for WGSL).
        /// </summary>
        public void GenerateConstants(StringBuilder builder) { }

        /// <summary>
        /// Merges the generated code into the main builder.
        /// </summary>
        public void Merge(StringBuilder builder) => builder.Append(Builder);

        #endregion

        #region Variable Management

        /// <summary>
        /// Allocates a variable for a value.
        /// </summary>
        protected Variable Allocate(Value value)
        {
            var name = $"v_{varCounter++}";
            var type = TypeGenerator[value.Type];
            if (type.StartsWith("ptr<"))
                if (WebGPUBackend.VerboseLogging) Console.WriteLine($"[DIAG-ALLOCATE] Allocated {name} with ptr type '{type}' for IR value type={value.Type} kind={value.GetType().Name} hash={value.GetHashCode()}");
            var variable = new Variable(name, type);
            valueVariables[value] = variable;
            return variable;
        }

        /// <summary>
        /// Allocates a variable for a type.
        /// </summary>
        protected Variable AllocateType(TypeNode type)
        {
            var name = $"v_{varCounter++}";
            var wgslType = TypeGenerator[type];
            return new Variable(name, wgslType);
        }

        /// <summary>
        /// Gets the variable for a value, allocating if necessary.
        /// </summary>
        public Variable Load(Value value)
        {
            if (!valueVariables.TryGetValue(value, out var variable))
            {
                // CRITICAL: Intercept shared-address-space Alloca values.
                // The Alloca node from backendContext.SharedAllocations (stored in
                // SharedMemoryVarNames) may be a DIFFERENT object instance from the one
                // referenced in the function body's IR. We detect shared Alloca values
                // by type and redirect them to the pre-registered module-scope variable.
                if (value is Alloca alloca && alloca.AddressSpace == MemoryAddressSpace.Shared)
                {
                    // Match by element type + array size to find the correct SharedMemoryVarNames
                    // entry. The old "first unassigned" strategy can grab the wrong entry when
                    // multiple shared allocations exist (e.g. RadixSort's histogram buffer vs
                    // ExclusiveScan's workspace), swapping their array sizes.
                    string allocaElemStr = alloca.AllocaType.ToString();
                    long allocaSize = alloca.ArrayLength.Resolve() is global::ILGPU.IR.Values.PrimitiveValue pv
                        ? pv.Int64Value : -1;
                    // First pass: try to match by element type + array size
                    foreach (var kvp in _baseArgs.SharedMemoryVarNames)
                    {
                        var (varName, declaration) = kvp.Value;
                        if (valueVariables.Any(kv => kv.Value.Name == varName && kv.Key != value))
                            continue;
                        if (kvp.Key is Alloca candidate
                            && candidate.AllocaType.ToString() == allocaElemStr
                            && candidate.ArrayLength.Resolve() is global::ILGPU.IR.Values.PrimitiveValue cpv
                            && cpv.Int64Value == allocaSize)
                        {
                            var wgslType = TypeGenerator[value.Type];
                            var sharedVar = new Variable(varName, wgslType);
                            valueVariables[value] = sharedVar;
                            if (WebGPUBackend.VerboseLogging) Console.WriteLine($"[DIAG-LOAD-SHARED] Matched shared Alloca (hash={value.GetHashCode()}, size={allocaSize}) to '{varName}' by type+size");
                            return sharedVar;
                        }
                    }
                    // Second pass: fall back to first unassigned entry (single-allocation case)
                    foreach (var kvp in _baseArgs.SharedMemoryVarNames)
                    {
                        var (varName, declaration) = kvp.Value;
                        if (valueVariables.Any(kv => kv.Value.Name == varName && kv.Key != value))
                            continue;
                        var wgslType = TypeGenerator[value.Type];
                        var sharedVar = new Variable(varName, wgslType);
                        valueVariables[value] = sharedVar;
                        if (WebGPUBackend.VerboseLogging) Console.WriteLine($"[DIAG-LOAD-SHARED] Fallback: redirected shared Alloca (hash={value.GetHashCode()}) to '{varName}'");
                        return sharedVar;
                    }
                }
                variable = Allocate(value);
            }
            return variable;
        }

        public Variable LoadIntrinsicValue(Value value) => Load(value);

        /// <summary>
        /// Binds a value to a variable.
        /// </summary>
        protected void Bind(Value value, Variable variable)
        {
            valueVariables[value] = variable;
        }

        protected readonly HashSet<string> declaredVariables = new();

        /// <summary>
        /// Declares a variable in the current scope.
        /// </summary>
        protected virtual void Declare(Variable variable)
        {
            if (declaredVariables.Contains(variable.Name)) return;
            declaredVariables.Add(variable.Name);

            // WGSL: pointer types (ptr<workgroup, ...>, ptr<storage, ...>, ptr<function, ...>)
            // are not constructible and cannot be declared with 'var'. Skip declaration — these
            // are bound via 'let' at the point of assignment or come from function parameters.
            if (variable.Type.StartsWith("ptr<"))
            {
                if (WebGPUBackend.VerboseLogging) Console.WriteLine($"[DIAG-DECLARE] Skipping ptr type declaration for {variable.Name} : {variable.Type}");
                return;
            }

            if (IsStateMachineActive)
            {
                // When in state machine mode, defer declarations to function scope
                // instead of emitting them inside case blocks where they'd be scoped.
                VariableBuilder.Append("    var ");
                VariableBuilder.Append(variable.Name);
                VariableBuilder.Append(" : ");
                VariableBuilder.Append(variable.Type);
                VariableBuilder.AppendLine(";");
            }
            else
            {
                AppendIndent();
                Builder.Append("var ");
                Builder.Append(variable.Name);
                Builder.Append(" : ");
                Builder.Append(variable.Type);
                Builder.AppendLine(";");
            }
        }

        #endregion

        #region Label Management

        /// <summary>
        /// Declares a new label for a block.
        /// </summary>
        protected string DeclareLabel()
        {
            return $"L_{Method.Id}_{labelCounter++}";
        }

        /// <summary>
        /// Gets or creates a label for a block.
        /// </summary>
        protected string GetBlockLabel(BasicBlock block)
        {
            if (!blockLabels.TryGetValue(block, out var label))
            {
                label = DeclareLabel();
                blockLabels[block] = label;
            }
            return label;
        }

        /// <summary>
        /// Marks a label in the output.
        /// </summary>
        protected void MarkLabel(string label)
        {
            // WGSL doesn't have labels like C, use block IDs in switch
        }

        #endregion

        #region Code Emission Helpers

        /// <summary>
        /// Appends indentation to the builder.
        /// </summary>
        protected void AppendIndent()
        {
            for (int i = 0; i < IndentLevel; i++)
                Builder.Append("    ");
        }

        /// <summary>
        /// Increases indentation.
        /// </summary>
        protected void PushIndent() => IndentLevel++;

        /// <summary>
        /// Decreases indentation.
        /// </summary>
        protected void PopIndent() => IndentLevel--;

        /// <summary>
        /// Appends a line with indentation.
        /// </summary>
        protected void AppendLine(string line)
        {
            AppendIndent();
            Builder.AppendLine(line);
        }

        /// <summary>
        /// Appends a line without indentation.
        /// </summary>
        protected void AppendLineRaw(string line)
        {
            Builder.AppendLine(line);
        }

        /// <summary>
        /// Begins a function body.
        /// </summary>
        protected void BeginFunctionBody()
        {
            Builder.AppendLine("{");
            PushIndent();
        }

        /// <summary>
        /// Finishes a function body.
        /// </summary>
        protected void FinishFunctionBody()
        {
            PopIndent();
            Builder.AppendLine("}");
        }

        #endregion

        #region IR Traversal

        /// <summary>
        /// Generates code for all blocks in the method.
        /// </summary>
        protected void GenerateCodeInternal()
        {
            var blocks = Method.Blocks;

            // Setup local allocations
            SetupAllocations(Allocas.LocalAllocations, MemoryAddressSpace.Local);

            bool hasReturnValue = !Method.ReturnType.IsVoidType;
            string returnType = hasReturnValue ? TypeGenerator[Method.ReturnType] : "void";

            // Simple case: single block, no control flow
            if (blocks.Count == 1)
            {
                IsStateMachineActive = false;
                foreach (var valueEntry in blocks.First())
                {
                    GenerateCodeFor(valueEntry.Value);
                }
                return;
            }

            // Multiple blocks: use loop/switch pattern
            IsStateMachineActive = true;

            // Declare return variable if needed
            if (hasReturnValue)
            {
                // Initialize to zero/default to avoid WGSL "variable used before assignment" errors
                // though logic should ensure assignment before exit.
                string zeroVal = returnType.StartsWith("bool") ? "false" : (returnType.StartsWith("f") ? "0.0" : "0");
                if (returnType.StartsWith("vec")) zeroVal = $"{returnType}()"; // zero init vector
                // emu_i64/emu_u64/emu_f64 are type aliases for vec types — use the alias as constructor
                // which automatically matches the correct component count (vec2 or vec4 depending on mode)
                if (returnType == "emu_i64" || returnType == "emu_u64" || returnType == "emu_f64") zeroVal = $"{returnType}()";

                AppendLine($"var _ilgpu_return_val : {returnType} = {zeroVal};");
            }

            // Save position before loop - deferred variable declarations will be inserted here
            int deferredInsertPosition = Builder.Length;

            AppendLine("var current_block : i32 = 0;");
            AppendLine("loop {");
            PushIndent();
            AppendLine("switch (current_block) {");
            PushIndent();

            int blockIndex = 0;
            foreach (var block in blocks)
            {
                AppendLine($"case {blockIndex}: {{");
                PushIndent();

                foreach (var valueEntry in block)
                {
                    GenerateCodeFor(valueEntry.Value);
                }

                PopIndent();
                AppendLine("}");
                blockIndex++;
            }

            AppendLine("default: { break; }");
            PopIndent();
            AppendLine("}"); // end switch

            AppendLine("if (current_block == -1) { break; }");

            PopIndent();
            AppendLine("}"); // end loop

            // CRITICAL: Post-process to hoist ALL variable declarations to function scope.
            // Some code generators bypass Declare() and write "let v_N = expr;" directly.
            // In WGSL, 'let' inside a 'case' block is scoped to that block, causing
            // "unresolved value" errors when that variable is used in another block.
            // We convert these to assignments and add var declarations at function scope.
            {
                string loopCode = Builder.ToString(deferredInsertPosition, Builder.Length - deferredInsertPosition);
                var letPattern = new System.Text.RegularExpressions.Regex(@"^(\s*)let\s+(v_\d+)\s*=\s*(.+);", System.Text.RegularExpressions.RegexOptions.Multiline);
                var hoistedVars = new Dictionary<string, string>(); // name -> inferred type
                string processed = letPattern.Replace(loopCode, match =>
                {
                    string indent = match.Groups[1].Value;
                    string varName = match.Groups[2].Value;
                    string expr = match.Groups[3].Value;

                    // Infer type from the expression for the var declaration
                    string inferredType = "bool"; // default for comparisons
                    // Pointer aliases — expressions starting with '&' create references
                    // that can't be declared as 'var'. Handle based on complexity:
                    // - Simple refs (&identifier): hoist to function scope
                    // - Complex refs (&identifier[index]): keep in block (index may be block-local)
                    if (expr.TrimStart().StartsWith("&"))
                    {
                        var trimmed = expr.TrimStart();
                        bool isSimpleRef = System.Text.RegularExpressions.Regex.IsMatch(trimmed, @"^&\w+$");
                        if (isSimpleRef)
                        {
                            VariableBuilder.AppendLine($"    let {varName} = {trimmed};");
                            return ""; // Remove from case block; hoisted to function scope
                        }
                        // Complex pointer ref — keep as 'let' in the block
                        return match.Value;
                    }
                    else if (expr.Contains("f64_from_f32") || expr.Contains("f64_add") || expr.Contains("f64_sub") || expr.Contains("f64_mul") || expr.Contains("f64_div") ||
                             expr.Contains("f64_neg") || expr.Contains("f64_from_ieee754") || expr.Contains("emu_f64(") ||
                             expr.Contains("f64_abs") || expr.Contains("f64_min") || expr.Contains("f64_max"))
                    {
                        inferredType = "emu_f64"; // vec2<f32> alias
                    }
                    else if (expr.Contains("i64_from_i32") || expr.Contains("u64_from_u32") ||
                             expr.Contains("i64_add") || expr.Contains("i64_sub") || expr.Contains("i64_mul") ||
                             expr.Contains("u64_mul") || expr.Contains("i64_neg") || expr.Contains("i64_abs") ||
                             expr.Contains("i64_and") || expr.Contains("i64_or") || expr.Contains("i64_xor") ||
                             expr.Contains("i64_shl") || expr.Contains("i64_shr") || expr.Contains("u64_shr") ||
                             expr.Contains("i64_not") || expr.Contains("emu_i64(") || expr.Contains("emu_u64("))
                    {
                        inferredType = "emu_i64"; // vec2<u32> alias
                    }
                    else if (expr.Contains("f64_to_f32") || expr.Contains("f32("))
                    {
                        inferredType = "f32";
                    }
                    else if (expr.Contains("i64_to_i32"))
                    {
                        inferredType = "i32";
                    }
                    else if (expr.Contains("u64_to_u32"))
                    {
                        inferredType = "u32";
                    }
                    else if (expr.Contains("i32("))
                    {
                        inferredType = "i32";
                    }
                    else if (expr.Contains("u32("))
                    {
                        inferredType = "u32";
                    }
                    else
                    {
                        // Try to look up type from valueVariables
                        foreach (var kvp in valueVariables)
                        {
                            if (kvp.Value.Name == varName)
                            {
                                inferredType = kvp.Value.Type;
                                break;
                            }
                        }
                    }

                    // WGSL: pointer types are not constructible and can't be declared as 'var'.
                    // Hoist simple identifier pointers; keep complex expressions in block.
                    if (inferredType.StartsWith("ptr<"))
                    {
                        bool isSimpleIdent = System.Text.RegularExpressions.Regex.IsMatch(expr.Trim(), @"^\w+$");
                        if (isSimpleIdent)
                        {
                            VariableBuilder.AppendLine($"    let {varName} = {expr.Trim()};");
                            return ""; // Remove from case block; hoisted to function scope
                        }
                        return match.Value; // Keep complex pointer expressions in block
                    }

                    if (!hoistedVars.ContainsKey(varName))
                    {
                        hoistedVars[varName] = inferredType;
                    }

                    // Convert "let v_N = expr;" to "v_N = expr;" (assignment only)
                    return $"{indent}{varName} = {expr};";
                });

                // Rebuild the code: replace the loop portion with processed version
                Builder.Remove(deferredInsertPosition, Builder.Length - deferredInsertPosition);
                Builder.Append(processed);

                // Add hoisted var declarations to VariableBuilder
                foreach (var kvp in hoistedVars)
                {
                    VariableBuilder.AppendLine($"    var {kvp.Key} : {kvp.Value};");
                }
            }

            // Insert all deferred variable declarations at the function scope
            // (before the loop). This ensures all variables are accessible from any case block.
            if (VariableBuilder.Length > 0)
            {
                Builder.Insert(deferredInsertPosition, VariableBuilder.ToString());
            }

            // Emit final return
            if (hasReturnValue)
            {
                AppendLine("return _ilgpu_return_val;");
            }
            else
            {
                AppendLine("return;");
            }
        }

        /// <summary>
        /// Gets the block index for control flow.
        /// </summary>
        protected int GetBlockIndex(BasicBlock block)
        {
            int index = 0;
            foreach (var b in Method.Blocks)
            {
                if (b == block) return index;
                index++;
            }
            return -1;
        }

        /// <summary>
        /// Sets up local/shared memory allocations.
        /// </summary>
        protected void SetupAllocations(AllocaKindInformation allocas, MemoryAddressSpace addressSpace)
        {
            foreach (var allocaInfo in allocas)
            {
                var variable = Allocate(allocaInfo.Alloca);
                var elementType = TypeGenerator[allocaInfo.ElementType];

                // Track this variable as a local alloca value type (not a pointer).
                // Store/Load/LoadFieldAddress must NOT dereference these with '*'.
                _localAllocaVarNames.Add(variable.Name);

                if (allocaInfo.IsArray)
                {
                    AppendLine($"var {variable.Name} : array<{elementType}, {allocaInfo.ArraySize}>;");
                }
                else
                {
                    AppendLine($"var {variable.Name} : {elementType};");
                }
            }
        }

        #endregion

        #region Value Visitors - Dispatch

        /// <summary>
        /// Generates code for a value using the visitor pattern.
        /// </summary>
        protected void GenerateCodeFor(Value value)
        {
            // Skip void values and already-handled values
            if (value.Type.IsVoidType &&
                !(value is TerminatorValue) &&
                !(value is Store) &&
                !(value is MemoryBarrier) &&
                !(value is global::ILGPU.IR.Values.Barrier) &&
                !(value is PredicateBarrier))
                return;

            // Debug logging to trace instruction generation
            WebGPUBackend.Log($"[WGSL] Generating code for: {value.GetType().FullName} - {value}");

            // Handle inaccessible Throw instruction (likely internal)
            // Use Contains("Throw") to be safer
            if (value.GetType().Name.Contains("Throw"))
            {
                WebGPUBackend.Log($"[WGSL] HANDLING THROW: {value}");
                GenerateThrow(value);
                return;
            }

            // Check for intrinsic implementation
            if (ImplementationProvider.TryGetCodeGenerator(value, out var intrinsicCodeGenerator))
            {
                intrinsicCodeGenerator(Backend, this, value);
                return;
            }

            switch (value)
            {
                // Parameters
                case global::ILGPU.IR.Values.Parameter p:
                    GenerateCode(p);
                    break;

                case global::ILGPU.IR.Values.MethodCall v:
                    GenerateCode(v);
                    break;

                // Arithmetic
                case global::ILGPU.IR.Values.BinaryArithmeticValue v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.UnaryArithmeticValue v:
                    GenerateUnOp(v);
                    break;
                case global::ILGPU.IR.Values.TernaryArithmeticValue v:
                    GenerateCode(v);
                    break;

                // Comparisons
                case global::ILGPU.IR.Values.CompareValue v:
                    GenerateCode(v);
                    break;

                // Conversions
                case global::ILGPU.IR.Values.ConvertValue v:
                    GenerateCode(v);
                    break;

                // Memory Operations
                case global::ILGPU.IR.Values.Load v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.Store v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.LoadElementAddress v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.LoadFieldAddress v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.Alloca v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.NewArray v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.LoadArrayElementAddress v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.NewView v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.GetViewLength v:
                    GenerateCode(v);
                    break;

                // Constants
                case global::ILGPU.IR.Values.PrimitiveValue v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.NullValue v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.StringValue v:
                    GenerateCode(v);
                    break;

                // Phi
                case global::ILGPU.IR.Values.PhiValue v:
                    GenerateCode(v);
                    break;

                // Structures
                case global::ILGPU.IR.Values.StructureValue v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.GetField v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.SetField v:
                    GenerateCode(v);
                    break;

                // Device Constants
                case global::ILGPU.IR.Values.GridIndexValue v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.GroupIndexValue v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.GridDimensionValue v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.GroupDimensionValue v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.WarpSizeValue v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.LaneIdxValue v:
                    GenerateCode(v);
                    break;


                // Control Flow
                case global::ILGPU.IR.Values.ReturnTerminator v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.UnconditionalBranch v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.IfBranch v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.SwitchBranch v:
                    GenerateCode(v);
                    break;



                // Casts
                case global::ILGPU.IR.Values.IntAsPointerCast v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.PointerAsIntCast v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.PointerCast v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.AddressSpaceCast v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.FloatAsIntCast v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.IntAsFloatCast v:
                    GenerateCode(v);
                    break;


                // Atomics & Barriers
                case global::ILGPU.IR.Values.GenericAtomic v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.AtomicCAS v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.MemoryBarrier v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.Barrier v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.PredicateBarrier v:
                    GenerateCode(v);
                    break;

                // Warp Operations
                case global::ILGPU.IR.Values.Broadcast v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.WarpShuffle v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.SubWarpShuffle v:
                    GenerateCode(v);
                    break;

                // Debug/IO
                case global::ILGPU.IR.Values.DebugAssertOperation v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.WriteToOutput v:
                    GenerateCode(v);
                    break;

                // Other
                case global::ILGPU.IR.Values.Predicate v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.DynamicMemoryLengthValue v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.AlignTo v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.AsAligned v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.LanguageEmitValue v:
                    GenerateCode(v);
                    break;

                default:
                    AppendLine($"// Unhandled value type: {value.GetType().Name}");
                    break;
            }
        }

        #endregion

        #region Value Visitors - Implementation

        // Parameters
        public virtual void GenerateCode(Parameter parameter)
        {
            // Parameters are typically handled in the kernel generator
        }

        // Arithmetic
        public virtual void GenerateCode(BinaryArithmeticValue value)
        {
            var target = Load(value);
            var left = Load(value.Left);
            var right = Load(value.Right);

            string op = value.Kind switch
            {
                BinaryArithmeticKind.Add => "+",
                BinaryArithmeticKind.Sub => "-",
                BinaryArithmeticKind.Mul => "*",
                BinaryArithmeticKind.Div => "/",
                BinaryArithmeticKind.And => "&",
                BinaryArithmeticKind.Or => "|",
                BinaryArithmeticKind.Xor => "^",
                BinaryArithmeticKind.Shl => "<<",
                BinaryArithmeticKind.Shr => ">>",
                BinaryArithmeticKind.Min => "min",
                BinaryArithmeticKind.Max => "max",
                BinaryArithmeticKind.Rem when TypeGenerator[value.Left.Type].StartsWith("f") => "frem",
                BinaryArithmeticKind.Rem => "%",
                BinaryArithmeticKind.PowF => "pow_func",
                BinaryArithmeticKind.Atan2F => "atan2_func",
                BinaryArithmeticKind.BinaryLogF => "binarylog_func",
                BinaryArithmeticKind.CopySignF => "copysign_func",
                _ => "+"
            };

            Declare(target);

            if (op == "frem")
            {
                AppendLine($"{target} = {left} - {right} * trunc({left} / {right});");
                return;
            }

            // Handle pointer arithmetic (e.g., &(*ptr)[offset])
            if (value.Kind == BinaryArithmeticKind.Add)
            {
                if (value.Left.Type.IsPointerType)
                {
                    AppendLine($"{target} = &(*{left})[{right}];");
                    return;
                }
                else if (value.Right.Type.IsPointerType)
                {
                    AppendLine($"{target} = &(*{right})[{left}];");
                    return;
                }
            }

            if (value.Kind == BinaryArithmeticKind.Min || value.Kind == BinaryArithmeticKind.Max)
            {
                // Check for emulated 64-bit types first.
                // WGSL's min()/max() are component-wise on vec2, which is WRONG for 64-bit values.
                // Use the emulation functions (i64_max, u64_max, f64_max, etc.) instead.
                //
                // Two detection paths:
                // 1. TypeGenerator lookup → "emu_i64" / "emu_u64" / "emu_f64"
                // 2. BasicValueType fallback → Int64/Float64 when emulation is enabled
                //    This catches cases where IR intermediate values may not map correctly via TypeGenerator.
                var leftType = TypeGenerator[value.Left.Type];
                bool isEmu64 = leftType == "emu_i64" || leftType == "emu_u64" || leftType == "emu_f64";
                
                // Fallback: check BasicValueType when emulation options are enabled
                if (!isEmu64 && Backend.Options.EnableI64Emulation && 
                    (value.BasicValueType == BasicValueType.Int64))
                {
                    isEmu64 = true;
                    leftType = value.IsUnsigned ? "emu_u64" : "emu_i64";
                }
                else if (!isEmu64 && Backend.Options.EnableF64Emulation && 
                    value.BasicValueType == BasicValueType.Float64)
                {
                    isEmu64 = true;
                    leftType = "emu_f64";
                }
                
                if (isEmu64)
                {
                    string prefix = leftType switch
                    {
                        "emu_f64" => "f64",
                        "emu_u64" => "u64",
                        "emu_i64" => "i64",
                        _ => "i64"
                    };
                    string emuOp = value.Kind == BinaryArithmeticKind.Max ? "max" : "min";
                    AppendLine($"{target} = {prefix}_{emuOp}({left}, {right});");
                }
                // For unsigned integer min/max, WGSL's min()/max() are type-overloaded:
                // min(i32, i32) is signed, min(u32, u32) is unsigned.
                // When ILGPU flags the operation as unsigned, bitcast operands to u32
                // and the result back to i32.
                else if (value.IsUnsigned && value.BasicValueType == BasicValueType.Int32)
                {
                    AppendLine($"{target} = bitcast<i32>({op}(bitcast<u32>({left}), bitcast<u32>({right})));");
                }
                else
                {
                    AppendLine($"{target} = {op}({left}, {right});");
                }
            }
            else if (op == "pow_func")
            {
                AppendLine($"{target} = pow({left}, {right});");
            }
            else if (op == "atan2_func")
            {
                AppendLine($"{target} = atan2({left}, {right});");
            }
            else if (op == "binarylog_func")
            {
                AppendLine($"{target} = log({left}) / log({right});");
            }
            else if (op == "copysign_func")
            {
                // WGSL has no copysign builtin; emulate: sign(right) * abs(left)
                AppendLine($"{target} = sign({right}) * abs({left});");
            }
            else
            {
                AppendLine($"{target} = {left} {op} {right};");
            }
        }

        public virtual void GenerateCode(UnaryArithmeticValue value)
        {
            GenerateUnOp(value);
        }

        private void GenerateUnOp(UnaryArithmeticValue value)
        {
            var target = Load(value);
            var operand = Load(value.Value);
            Declare(target);

            // Handle emulated emu_i64/emu_u64 negation
            var operandType = TypeGenerator[value.Value.Type];
            if (Backend.Options.EnableI64Emulation && (operandType == "emu_i64" || operandType == "emu_u64") && value.Kind == UnaryArithmeticKind.Neg)
            {
                AppendLine($"{target} = i64_neg({operand});");
                return;
            }

            string result = value.Kind switch
            {
                UnaryArithmeticKind.Neg => $"-{operand}",
                UnaryArithmeticKind.Not => TypeGenerator[value.Value.Type] == "bool" ? $"!{operand}" : $"~{operand}",

                // Math Intrinsics (Float)
                UnaryArithmeticKind.Abs => $"abs({operand})",
                UnaryArithmeticKind.SinF => $"sin({operand})",
                UnaryArithmeticKind.CosF => $"cos({operand})",
                UnaryArithmeticKind.TanF => $"tan({operand})",
                UnaryArithmeticKind.AsinF => $"asin({operand})",
                UnaryArithmeticKind.AcosF => $"acos({operand})",
                UnaryArithmeticKind.AtanF => $"atan({operand})",
                UnaryArithmeticKind.SinhF => $"sinh({operand})",
                UnaryArithmeticKind.CoshF => $"cosh({operand})",
                UnaryArithmeticKind.TanhF => $"tanh({operand})",
                UnaryArithmeticKind.ExpF => $"exp({operand})",
                UnaryArithmeticKind.Exp2F => $"exp2({operand})",
                UnaryArithmeticKind.LogF => $"log({operand})",
                UnaryArithmeticKind.Log2F => $"log2({operand})",
                UnaryArithmeticKind.SqrtF => $"sqrt({operand})",
                UnaryArithmeticKind.RsqrtF => $"1.0 / sqrt({operand})", // Manual inverseSqrt
                UnaryArithmeticKind.RcpF => $"1.0 / {operand}",
                UnaryArithmeticKind.FloorF => $"floor({operand})",
                UnaryArithmeticKind.CeilingF => $"ceil({operand})",
                UnaryArithmeticKind.Log10F => $"(log({operand}) / 2.302585093)",
                UnaryArithmeticKind.IsNaNF => $"({operand} != {operand})",
                UnaryArithmeticKind.IsInfF => $"(abs({operand}) == (1.0 / 0.0))",
                UnaryArithmeticKind.IsFinF => $"(({operand} == {operand}) && (abs({operand}) != (1.0 / 0.0)))",

                // Bit Operations
                UnaryArithmeticKind.PopC => $"i32(countOneBits({operand}))",
                UnaryArithmeticKind.CLZ => $"i32(countLeadingZeros({operand}))",
                UnaryArithmeticKind.CTZ => $"i32(countTrailingZeros({operand}))",

                _ => "DEBUG_MISSING"
            };

            if (result == "DEBUG_MISSING")
            {
                AppendLine($"// [WGSL] Unhandled UnaryArithmeticKind: {value.Kind}");
                result = $"unhandled_unary({operand})";
            }

            AppendLine($"{target} = {result};");
        }

        public virtual void GenerateCode(TernaryArithmeticValue value)
        {
            var target = Load(value);
            var first = Load(value.First);
            var second = Load(value.Second);
            var third = Load(value.Third);
            Declare(target);

            string result = value.Kind switch
            {
                TernaryArithmeticKind.MultiplyAdd => $"fma({first}, {second}, {third})",
                _ => $"({first} * {second} + {third})"
            };

            AppendLine($"{target} = {result};");
        }

        // Comparisons
        public virtual void GenerateCode(CompareValue value)
        {
            var target = Load(value);
            var left = Load(value.Left);
            var right = Load(value.Right);
            Declare(target);

            string op = value.Kind switch
            {
                CompareKind.Equal => "==",
                CompareKind.NotEqual => "!=",
                CompareKind.LessThan => "<",
                CompareKind.LessEqual => "<=",
                CompareKind.GreaterThan => ">",
                CompareKind.GreaterEqual => ">=",
                _ => "=="
            };

            // Check for emulated 64-bit operands
            string leftType = TypeGenerator[value.Left.Type];
            string rightType = TypeGenerator[value.Right.Type];
            bool isEmulatedI64 = (leftType == "emu_i64" || leftType == "emu_u64" ||
                                  rightType == "emu_i64" || rightType == "emu_u64");

            if (isEmulatedI64)
            {
                // Use 64-bit emulation comparison functions instead of raw operators
                // (raw >= on vec2<u32> returns vec2<bool>, not bool)
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
                    AppendLine($"{target} = {emulFunc}({left}, {right});");
                    return;
                }
            }

            // For unsigned integer comparisons, WGSL operators are type-sensitive:
            // i32 < i32 is signed, u32 < u32 is unsigned.
            // When ILGPU flags the comparison as unsigned, bitcast operands to u32.
            bool needsUnsignedCast = value.IsUnsignedOrUnordered
                && value.Left.BasicValueType == BasicValueType.Int32
                && value.Kind != CompareKind.Equal
                && value.Kind != CompareKind.NotEqual;

            if (needsUnsignedCast)
            {
                AppendLine($"{target} = bitcast<u32>({left}) {op} bitcast<u32>({right});");
            }
            else
            {
                AppendLine($"{target} = {left} {op} {right};");
            }
        }

        // Conversions
        public virtual void GenerateCode(ConvertValue value)
        {
            var target = Load(value);
            var source = Load(value.Value);
            var targetType = TypeGenerator[value.Type];
            var sourceType = TypeGenerator[value.Value.Type];
            Declare(target);

            // Handle emulated 64-bit type conversions.
            // emu_i64/emu_u64 are type aliases for vec2<u32> (lo, hi).
            bool targetIsEmu64 = targetType == "emu_i64" || targetType == "emu_u64";
            bool sourceIsEmu64 = sourceType == "emu_i64" || sourceType == "emu_u64";

            if (targetIsEmu64 && !sourceIsEmu64)
            {
                // Widening: i32/u32/f32 → emu_i64/emu_u64
                // Convert source to u32, then zero-extend to vec2<u32>(lo, 0u)
                AppendLine($"{target} = vec2<u32>(u32({source}), 0u);");
            }
            else if (!targetIsEmu64 && sourceIsEmu64)
            {
                // Narrowing: emu_i64/emu_u64 → i32/u32/f32
                // Take low word (.x) and cast to target type
                AppendLine($"{target} = {targetType}({source}.x);");
            }
            else if (targetType == "emu_f64" && !sourceIsEmu64)
            {
                // emu_f64 widening from f32: use vec2<u32> with bitcast
                AppendLine($"{target} = vec2<u32>(bitcast<u32>(f32({source})), 0u);");
            }
            else
            {
                // Standard conversion (or emu_i64 → emu_u64 which are same repr)
                AppendLine($"{target} = {targetType}({source});");
            }
        }

        // Memory Operations
        public virtual void GenerateCode(global::ILGPU.IR.Values.Load loadVal)
        {
            var target = Load(loadVal);
            var source = Load(loadVal.Source);
            Declare(target);

            var targetType = TypeGenerator[loadVal.Type];
            string sourceType = targetType;
            bool isAtomic = IsAtomicPointer(loadVal.Source);

            if (loadVal.Source.Type is global::ILGPU.IR.Types.PointerType ptrType)
            {
                var elemType = ptrType.ElementType;
                sourceType = TypeGenerator[elemType];
            }

            // Local alloca variables are WGSL 'var' value types, not pointers — no dereference needed.
            bool isAllocaVar = _localAllocaVarNames.Contains(source.Name);

            if (isAtomic)
            {
                AppendLine($"{target} = atomicLoad({source});");
            }
            else if (isAllocaVar)
            {
                AppendLine($"{target} = {source};");
            }
            else if (targetType != sourceType)
            {
                AppendLine($"{target} = bitcast<{targetType}>(*{source});");
            }
            else
            {
                AppendLine($"{target} = *{source};");
            }
        }

        public virtual void GenerateCode(global::ILGPU.IR.Values.Store storeVal)
        {
            var address = Load(storeVal.Target);
            var val = Load(storeVal.Value);

            bool isAtomic = IsAtomicPointer(storeVal.Target);

            // Detect emu_f64 store: buffer stores IEEE-754 bytes, must convert from Dekker.
            string targetElemType = "";
            if (storeVal.Target.Type is global::ILGPU.IR.Types.PointerType storePtrType)
                targetElemType = TypeGenerator[storePtrType.ElementType];

            // Local alloca variables are WGSL 'var' value types, not pointers — no dereference needed.
            bool isAllocaVar = _localAllocaVarNames.Contains(address.Name);

            if (isAtomic)
            {
                AppendLine($"atomicStore({address}, {val});");
            }
            else if (isAllocaVar)
            {
                AppendLine($"{address} = {val};");
            }
            else
            {
                AppendLine($"*{address} = {val};");
            }
        }

        protected virtual bool IsAtomicPointer(Value ptr)
        {
            if (ptr.Type is global::ILGPU.IR.Types.PointerType ptrType)
            {
                var elemTypeStr = TypeGenerator[ptrType.ElementType];
                if (elemTypeStr.Contains("atomic")) return true;
                var elemType = ptrType.ElementType;
                if (elemType.ToString().Contains("Index1D") || elemType.ToString().Contains("LongIndex1D"))
                    return true;
            }
            return false;
        }

        public virtual void GenerateCode(LoadElementAddress value)
        {
            var target = Load(value);
            var source = Load(value.Source);
            var offset = Load(value.Offset);
            AppendIndent();
            Builder.Append("let ");
            Builder.Append(target.Name);
            Builder.Append(" = ");
            Builder.Append($"&{source}[{offset}];");
            Builder.AppendLine();
        }

        public virtual void GenerateCode(global::ILGPU.IR.Values.NewView value)
        {
            var target = Load(value);
            var source = Load(value.Pointer);
            // Detect shared memory by checking if the resolved source name
            // matches any shared variable name from SharedMemoryVarNames.
            bool isSharedSource = false;
            foreach (var kvp in _baseArgs.SharedMemoryVarNames)
            {
                if (kvp.Value.Name == source.Name)
                {
                    isSharedSource = true;
                    break;
                }
            }

            if (isSharedSource)
            {
                // For shared memory: alias the target to the source array name (no WGSL emission).
                // LoadElementAddress will then directly produce &shared_0[idx].
                valueVariables[value] = new Variable(source.Name, source.Type);
                if (WebGPUBackend.VerboseLogging) Console.WriteLine($"[DIAG-NEWVIEW-SHARED] Aliased {target} -> {source.Name} (no WGSL emit)");
            }
            else
            {
                // For non-shared buffer sources: emit a let binding to alias the pointer.
                // CRITICAL: if a state machine is active, the let must be hoisted to
                // function scope (VariableBuilder) so it's accessible across all case
                // blocks. Without hoisting, v_22 declared in case 3 is unresolved in case 5.
                declaredVariables.Add(target.Name);
                if (IsStateMachineActive)
                {
                    VariableBuilder.AppendLine($"    let {target.Name} = {source}; // newView (hoisted)");
                }
                else
                {
                    AppendLine($"let {target.Name} = {source}; // newView");
                }
            }
        }

        /// <summary>
        /// Generates code for GetViewLength (ArrayView.Length).
        /// Base implementation emits a comment; override in kernel generator
        /// to emit arrayLength(&paramN) for storage buffer bindings.
        /// </summary>
        public virtual void GenerateCode(global::ILGPU.IR.Values.GetViewLength value)
        {
            var target = Load(value);
            var source = Load(value.View);
            Declare(target);
            // In non-kernel contexts (helper functions), length might
            // not be directly available. For kernel params, the kernel
            // function generator overrides this.
            string lengthType = value.LengthType == BasicValueType.Int64 ? "i64" : "i32";
            AppendLine($"{target} = {lengthType}(arrayLength(&{source}));");
        }

        public virtual void GenerateCode(LoadFieldAddress value)
        {
            var target = Load(value);
            var source = Load(value.Source);
            AppendIndent();
            Builder.Append("let ");
            Builder.Append(target.Name);
            Builder.Append(" = ");

            string fieldName = $"field_{value.FieldSpan.Index}";
            if (IsIndexType(value.Source.Type))
            {
                fieldName = value.FieldSpan.Index switch
                {
                    0 => "x",
                    1 => "y",
                    2 => "z",
                    _ => fieldName
                };
            }

            // Local alloca variables are WGSL 'var' value types — use &source.field, not &(*source).field
            if (_localAllocaVarNames.Contains(source.Name))
                Builder.Append($"&{source}.{fieldName};");
            else
                Builder.Append($"&(*{source}).{fieldName};");
            Builder.AppendLine();
        }

        public virtual void GenerateCode(Alloca value)
        {
            // Already handled in SetupAllocations
        }

        public virtual void GenerateCode(global::ILGPU.IR.Values.NewArray value)
        {
            // NewArray creates a local array — declare it
            var arrayType = value.Type;
            string elementType = TypeGenerator[arrayType.ElementType];
            int arraySize = 1;
            foreach (var dim in value.Nodes)
            {
                if (dim.Resolve() is PrimitiveValue pv)
                    arraySize *= pv.Int32Value;
            }
            string arrayName = $"local_arr_{_localArrayCounter++}";
            var target = Load(value);
            _allocaArrayNames[target.Name] = arrayName;
            if (IsStateMachineActive)
            {
                VariableBuilder.AppendLine($"    var {arrayName} : array<{elementType}, {arraySize}>;");
            }
            else
            {
                AppendLine($"var {arrayName} : array<{elementType}, {arraySize}>;");
            }
        }

        public virtual void GenerateCode(global::ILGPU.IR.Values.LoadArrayElementAddress value)
        {
            var target = Load(value);
            var arraySource = Load(value.ArrayValue);
            var indexVar = Load(value.Dimensions[0]);
            string arrayName = _allocaArrayNames.TryGetValue(arraySource.Name, out var name)
                ? name : arraySource.Name;
            // Emit a WGSL pointer: let v_N = &array[index];
            AppendIndent();
            Builder.Append($"let {target.Name} = &{arrayName}[{indexVar}];");
            Builder.AppendLine();
        }

        // Constants
        public virtual void GenerateCode(PrimitiveValue value)
        {
            var target = Load(value);
            var type = TypeGenerator[value.Type];
            Declare(target);

            // Check if we need emulation for this constant
            bool isEmulatedF64 = Backend.Options.EnableF64Emulation && value.BasicValueType == BasicValueType.Float64;
            bool isEmulatedI64 = Backend.Options.EnableI64Emulation && value.BasicValueType == BasicValueType.Int64;

            if (isEmulatedF64)
            {
                // CRITICAL FIX: Use IEEE-754 bits to preserve full 64-bit precision!
                // Casting to float loses precision (24-bit mantissa vs 52-bit)
                // Instead, we pass the raw 64-bit representation as two u32 values
                double doubleVal = value.Float64Value;
                ulong bits = BitConverter.DoubleToUInt64Bits(doubleVal);
                uint lo = (uint)(bits & 0xFFFFFFFF);
                uint hi = (uint)(bits >> 32);
                AppendLine($"{target} = f64_from_ieee754_bits({lo}u, {hi}u);");
                return;
            }

            if (isEmulatedI64)
            {
                // For emu_i64 emulation, we need to split the 64-bit value into two 32-bit parts
                long longVal = value.Int64Value;
                uint lo = (uint)(longVal & 0xFFFFFFFF);
                uint hi = (uint)((ulong)longVal >> 32);
                AppendLine($"{target} = vec2<u32>({lo}u, {hi}u);");
                return;
            }

            string valStrStd = value.BasicValueType switch
            {
                BasicValueType.Int1 => value.Int1Value ? "true" : "false",
                BasicValueType.Int8 => value.Int8Value.ToString(),
                BasicValueType.Int16 => value.Int16Value.ToString(),
                BasicValueType.Int32 => value.Int32Value.ToString(),
                BasicValueType.Int64 => value.Int64Value.ToString(),
                BasicValueType.Float16 => FormatFloat((float)value.Float16Value),
                BasicValueType.Float32 => FormatFloat(value.Float32Value),
                BasicValueType.Float64 => FormatFloat((float)value.Float64Value),
                _ => "0"
            };

            if (value.BasicValueType != BasicValueType.Int1)
            {
                AppendLine($"{target} = {type}({valStrStd});");
            }
            else
            {
                AppendLine($"{target} = {valStrStd};");
            }
        }

        private string FormatFloat(float value)
        {
            if (float.IsNaN(value)) return "0.0";
            if (float.IsPositiveInfinity(value)) return "3.402823e+38";
            if (float.IsNegativeInfinity(value)) return "-3.402823e+38";

            // float.MaxValue / float.MinValue exceed what the WGSL parser
            // can represent as a decimal literal.  Use bitcast instead.
            if (value == float.MaxValue || value == float.MinValue ||
                value == -float.MaxValue)
            {
                uint bits = BitConverter.SingleToUInt32Bits(value);
                return $"bitcast<f32>(0x{bits:X8}u)";
            }

            var str = value.ToString("G9");
            if (!str.Contains('.') && !str.Contains('e') && !str.Contains('E'))
                str += ".0";
            return str;
        }

        public virtual void GenerateCode(NullValue value)
        {
            var target = Load(value);
            Declare(target);
            AppendLine($"{target} = {target.Type}(); // null");
        }

        public virtual void GenerateCode(global::ILGPU.IR.Values.Barrier value)
        {
            if (value.Kind == global::ILGPU.IR.Values.BarrierKind.GroupLevel)
            {
                AppendLine("workgroupBarrier();");
                AppendLine("storageBarrier();");
            }
        }

        public virtual void GenerateCode(StringValue value)
        {
            AppendLine($"// String: {value.String}");
        }

        // Phi
        public virtual void GenerateCode(PhiValue value)
        {
            var target = Load(value);
            Declare(target);
        }

        // Structures
        public virtual void GenerateCode(StructureValue value)
        {
            var target = Load(value);
            Declare(target);
            var sb = new StringBuilder();
            sb.Append($"{target} = {target.Type}(");
            for (int i = 0; i < value.Count; i++)
            {
                if (i > 0) sb.Append(", ");
                sb.Append(Load(value[i]));
            }
            sb.Append(");");
            AppendLine(sb.ToString());
        }

        public virtual void GenerateCode(GetField value)
        {
            var target = Load(value);
            var source = Load(value.ObjectValue);
            Declare(target);

            string fieldName = $"field_{value.FieldSpan.Index}";
            if (IsIndexType(value.ObjectValue.Type))
            {
                fieldName = value.FieldSpan.Index switch
                {
                    0 => "x",
                    1 => "y",
                    2 => "z",
                    _ => fieldName
                };
            }

            AppendLine($"{target} = {source}.{fieldName};");
        }

        public virtual void GenerateCode(SetField value)
        {
            var target = Load(value);
            var source = Load(value.ObjectValue);
            var fieldValue = Load(value.Value);
            Declare(target);
            AppendLine($"{target} = {source};");

            string fieldName = $"field_{value.FieldSpan.Index}";
            if (IsIndexType(value.ObjectValue.Type))
            {
                fieldName = value.FieldSpan.Index switch
                {
                    0 => "x",
                    1 => "y",
                    2 => "z",
                    _ => fieldName
                };
            }

            AppendLine($"{target}.{fieldName} = {fieldValue};");
        }

        protected bool IsIndexType(TypeNode type)
        {
            var typeName = type.ToString();
            return typeName.Contains("Index") &&
                   (typeName.Contains("1D") || typeName.Contains("2D") || typeName.Contains("3D"));
        }

        // Device Constants
        public virtual void GenerateCode(GridIndexValue value)
        {
            var target = Load(value);
            Declare(target);
            string dim = value.Dimension switch
            {
                DeviceConstantDimension3D.X => "x",
                DeviceConstantDimension3D.Y => "y",
                DeviceConstantDimension3D.Z => "z",
                _ => "x"
            };
            AppendLine($"{target} = i32(group_id.{dim});");
        }

        public virtual void GenerateCode(GroupIndexValue value)
        {
            var target = Load(value);
            Declare(target);
            string dim = value.Dimension switch
            {
                DeviceConstantDimension3D.X => "x",
                DeviceConstantDimension3D.Y => "y",
                DeviceConstantDimension3D.Z => "z",
                _ => "x"
            };
            AppendLine($"{target} = i32(local_id.{dim});");
        }

        public virtual void GenerateCode(GridDimensionValue value)
        {
            var target = Load(value);
            Declare(target);
            string dim = value.Dimension switch
            {
                DeviceConstantDimension3D.X => "x",
                DeviceConstantDimension3D.Y => "y",
                DeviceConstantDimension3D.Z => "z",
                _ => "x"
            };
            // GridDimensionValue = Grid.DimX (number of workgroups in dimension, NOT total threads).
            // In WebGPU WGSL this is num_workgroups.x.
            // The previous code multiplied by workgroup_size, erroneously computing total thread count,
            // which caused GridStrideLoopStride = Grid.DimX * Group.DimX to be 64x too large.
            AppendLine($"{target} = i32(num_workgroups.{dim});");
        }


        public virtual void GenerateCode(GroupDimensionValue value)
        {
            var target = Load(value);
            Declare(target);
            string dim = value.Dimension switch
            {
                DeviceConstantDimension3D.X => "x",
                DeviceConstantDimension3D.Y => "y",
                DeviceConstantDimension3D.Z => "z",
                _ => "x"
            };
            AppendLine($"{target} = i32(workgroup_size.{dim});");
        }

        public virtual void GenerateCode(WarpSizeValue value)
        {
            var target = Load(value);
            Declare(target);
            if (HasSubgroups)
                AppendLine($"{target} = i32(subgroup_size);");
            else
                AppendLine($"{target} = i32(workgroup_size.x);");
        }

        public virtual void GenerateCode(LaneIdxValue value)
        {
            var target = Load(value);
            Declare(target);
            if (HasSubgroups)
                AppendLine($"{target} = i32(subgroup_invocation_id);");
            else
                AppendLine($"{target} = i32(local_index % workgroup_size.x);");
        }

        // Control Flow
        public virtual void GenerateCode(ReturnTerminator value)
        {
            if (IsStateMachineActive)
            {
                // In state machine: Assign to return var and break loop
                if (!value.IsVoidReturn)
                {
                    var retVal = Load(value.ReturnValue);
                    AppendLine($"_ilgpu_return_val = {retVal};");
                }

                AppendLine("current_block = -1;");
                AppendLine("break;");
            }
            else
            {
                // Direct return (Single block)
                if (value.IsVoidReturn)
                {
                    AppendLine("return;");
                }
                else
                {
                    var retVal = Load(value.ReturnValue);
                    AppendLine($"return {retVal};");
                }
            }
        }

        public virtual void GenerateCode(UnconditionalBranch branch)
        {
            EmitPhiAssignments(branch.BasicBlock, branch.Target);
            int targetIdx = GetBlockIndex(branch.Target);
            AppendLine($"current_block = {targetIdx};");
            AppendLine("continue;");
        }

        public virtual void GenerateCode(IfBranch branch)
        {
            var cond = Load(branch.Condition);
            int trueIdx = GetBlockIndex(branch.TrueTarget);
            int falseIdx = GetBlockIndex(branch.FalseTarget);

            AppendLine($"if ({cond}) {{");
            PushIndent();
            EmitPhiAssignments(branch.BasicBlock, branch.TrueTarget);
            AppendLine($"current_block = {trueIdx};");
            PopIndent();
            AppendLine("} else {");
            PushIndent();
            EmitPhiAssignments(branch.BasicBlock, branch.FalseTarget);
            AppendLine($"current_block = {falseIdx};");
            PopIndent();
            AppendLine("}");
            AppendLine("continue;");
        }

        public virtual void GenerateCode(SwitchBranch branch)
        {
            var selector = Load(branch.Condition);
            AppendLine($"switch ({selector}) {{");
            PushIndent();

            for (int i = 0; i < branch.NumCasesWithoutDefault; i++)
            {
                var target = branch.GetCaseTarget(i);
                int targetIdx = GetBlockIndex(target);
                AppendLine($"case {i}: {{");
                PushIndent();
                EmitPhiAssignments(branch.BasicBlock, target);
                AppendLine($"current_block = {targetIdx};");
                PopIndent();
                AppendLine("}");
            }

            int defaultIdx = GetBlockIndex(branch.DefaultBlock);
            AppendLine("default: {");
            PushIndent();
            EmitPhiAssignments(branch.BasicBlock, branch.DefaultBlock);
            AppendLine($"current_block = {defaultIdx};");
            PopIndent();
            AppendLine("}");

            PopIndent();
            AppendLine("}");
            AppendLine("continue;");
        }

        protected void EmitPhiAssignments(BasicBlock sourceBlock, BasicBlock targetBlock)
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
                        AppendLine($"{phiVar} = {srcVar};");
                    }
                }
            }
        }

        // Method Calls
        public virtual void GenerateCode(MethodCall methodCall)
        {
            var target = Load(methodCall);
            Declare(target);

            string name = methodCall.Target.Name;
            // Map common intrinsics if they appear as method calls
            string? wgslFunc = name switch
            {
                // Specialized Intrinsics (Check these first to avoid sub-string matches)
                var n when n.Contains("Rsqrt") => "rsqrt_custom",
                var n when n.Contains("Rcp") => "rcp_custom", // Special handling
                var n when n.Contains("Asin") => "asin",
                var n when n.Contains("Acos") => "acos",
                var n when n.Contains("Atan2") => "atan2",
                var n when n.Contains("Atan") => "atan",
                var n when n.Contains("Sinh") => "sinh",
                var n when n.Contains("Cosh") => "cosh",
                var n when n.Contains("Tanh") => "tanh",
                var n when n.Contains("Step") && !n.Contains("Smooth") => "step",
                var n when n.Contains("SmoothStep") => "smoothstep",
                var n when n.Contains("FusedMultiplyAdd") => "fma",

                var n when n.Contains("PopCount") => "countOneBits",
                var n when n.Contains("TrailingZeroCount") => "countTrailingZeros",
                var n when n.Contains("LeadingZeroCount") => "countLeadingZeros",
                var n when n.Contains("Reverse") => "reverseBits",

                // Standard Math
                var n when n.Contains("Sin") => "sin",
                var n when n.Contains("Cos") => "cos",
                var n when n.Contains("Tan") => "tan",
                var n when n.Contains("Sqrt") => "sqrt",
                var n when n.Contains("Abs") => "abs",
                var n when n.Contains("Pow") => "pow",
                var n when n.Contains("Exp") => "exp",
                var n when n.Contains("Log") => "log",
                var n when n.Contains("Floor") => "floor",
                var n when n.Contains("Ceiling") => "ceil",
                var n when n.Contains("Min") => "min",
                var n when n.Contains("Max") => "max",
                var n when n.Contains("Clamp") => "clamp",
                var n when n.Contains("Sign") => "sign",
                var n when n.Contains("Round") => "round",
                var n when n.Contains("Truncate") => "trunc",
                var n when n.Contains("Lerp") || n.Contains("Mix") => "mix",
                var n when n.Contains("Select") => "select_custom",

                var n when n.Contains("Sinh") => "sinh",
                var n when n.Contains("Cosh") => "cosh",
                var n when n.Contains("Tanh") => "tanh",

                _ => null
            };

            if (wgslFunc != null)
            {
                if (methodCall.Count == 1)
                {
                    var arg = Load(methodCall[0]);

                    if (wgslFunc == "rcp_custom")
                    {
                        AppendLine($"{target} = 1.0 / {arg};");
                        return;
                    }
                    if (wgslFunc == "rsqrt_custom")
                    {
                        AppendLine($"{target} = 1.0 / sqrt({arg});");
                        return;
                    }

                    string call = $"{wgslFunc}({arg})";
                    if (wgslFunc == "sign" && (TypeGenerator[methodCall.Type] == "i32" || TypeGenerator[methodCall.Type] == "u32"))
                    {
                        call = $"i32({call})";
                    }
                    AppendLine($"{target} = {call};");
                    return;
                }
                else if (methodCall.Count == 2)
                {
                    var arg1 = Load(methodCall[0]);
                    var arg2 = Load(methodCall[1]);
                    AppendLine($"{target} = {wgslFunc}({arg1}, {arg2});");
                    return;
                }
                else if (methodCall.Count == 3)
                {
                    var arg1 = Load(methodCall[0]);
                    var arg2 = Load(methodCall[1]);
                    var arg3 = Load(methodCall[2]);

                    // Select(cond, trueVal, falseVal) -> select(falseVal, trueVal, cond)
                    if (wgslFunc == "select_custom")
                    {
                        AppendLine($"{target} = select({arg3}, {arg2}, {arg1});");
                    }
                    else
                    {
                        // FMA or Mix usually
                        AppendLine($"{target} = {wgslFunc}({arg1}, {arg2}, {arg3});");
                    }
                    return;
                }
                else if (methodCall.Count == 3)
                {
                    var arg1 = Load(methodCall[0]);
                    var arg2 = Load(methodCall[1]);
                    var arg3 = Load(methodCall[2]);
                    AppendLine($"{target} = {wgslFunc}({arg1}, {arg2}, {arg3});");
                    return;
                }
            }


            // --- Smart Fallback: Reduction Method Detection ---
            // When GroupExtensions.Reduce / WarpExtensions.Reduce is not registered as an
            // intrinsic for the current backend, ILGPU falls back to the IL implementation
            // (PTXWarpExtensions.Reduce, ILWarpExtensions.Reduce, etc.). The resulting IR
            // MethodCall has a source MethodInfo for the specialized generic variant (e.g.
            // Reduce<int, MaxInt32>). We detect this pattern and emit the appropriate WGSL
            // subgroup intrinsic based on TReduction.CLCommand.
            //
            // CRITICAL: ILGroupExtensions.AllReduce compiles Warp.IsFirstLane as Group.IsFirstThread
            // (i.e. local_id.x==0 && local_id.y==0 && local_id.z==0), which is only TRUE for thread 0
            // of the workgroup. This means only subgroup 0's result ever reaches shared memory.
            // FIX: After the warp-level subgroup op, we emit a complete cross-subgroup barrier
            // accumulation using _grp_sg_results[]. ALL threads then hold the workgroup-wide result,
            // so the subsequent (broken) thread-0-only atomicMax correctly writes the final value.
            var sourceMethodInfo = methodCall.Target.Source as MethodInfo;
            if (sourceMethodInfo != null &&
                sourceMethodInfo.Name == "Reduce" &&
                sourceMethodInfo.IsGenericMethod)
            {
                string? subgroupOp = GetSubgroupReduceOp(sourceMethodInfo);
                // Early-compute wgslType to check if 64-bit emulated
                string earlyIrType = TypeGenerator[methodCall.Type];
                string earlyWgslType = GetReductionWgslType(sourceMethodInfo) ?? earlyIrType;
                bool is64BitReduce = earlyWgslType.StartsWith("emu_");

                // CRITICAL: subgroupMax/Min/Add do NOT work on vec2<u32> (emulated 64-bit types).
                // Force 64-bit types to the no-subgroups shared-memory path.
                if (subgroupOp != null && HasSubgroups && !is64BitReduce)
                {
                    AppendLine($"// Reduce fallback (group-level): {methodCall.Target.Name} -> {subgroupOp}");
                    var reduceSrc = methodCall.Count > 0 ? Load(methodCall[0]) : null;
                    if (reduceSrc != null)
                    {
                        // Signal kernel generator to declare _grp_sg_results workgroup array
                        UsesGroupReduce = true;

                        // Determine effective WGSL type from TReduction (handles u32, f32, emu_* correctly)
                        string irType = earlyIrType;
                        string wgslType = earlyWgslType;
                        bool needsBitcast = wgslType != irType;

                        // Determine scalar accumulation op from subgroupOp name
                        string wgslAccumOp;
                        if (subgroupOp.EndsWith("Max", StringComparison.Ordinal))
                            wgslAccumOp = "max";
                        else if (subgroupOp.EndsWith("Min", StringComparison.Ordinal))
                            wgslAccumOp = "min";
                        else
                            wgslAccumOp = "+";

                        // Step 1: intra-subgroup reduce (with correct type for subgroup op)
                        string warpResult = $"_warp_sg_result_{target.Name}";
                        string srcExpr = needsBitcast ? $"bitcast<{wgslType}>({reduceSrc})" : $"{reduceSrc}";
                        AppendLine($"var {warpResult} : {wgslType} = {subgroupOp}({srcExpr});");

                        // Step 2: first lane of each subgroup stores to _grp_sg_results
                        AppendLine($"if (subgroup_invocation_id == 0u) {{");
                        PushIndent();
                        AppendLine($"_grp_sg_results[subgroup_id] = {BitcastToU32(warpResult, wgslType)};");
                        PopIndent();
                        AppendLine("}");

                        // Step 3: workgroup barrier so all subgroups have written
                        AppendLine("workgroupBarrier();");

                        // Step 4: all threads cooperatively reduce across subgroups
                        string load0Expr = BitcastFromU32("_grp_sg_results[0u]", wgslType);
                        string loadNExpr = BitcastFromU32("_grp_sg_results[_sgi]", wgslType);
                        AppendLine("{");
                        PushIndent();
                        AppendLine($"var _grp_accum : {wgslType} = {load0Expr};");
                        AppendLine("let _num_sgs : u32 = workgroup_size.x / subgroup_size;");
                        AppendLine("for (var _sgi : u32 = 1u; _sgi < _num_sgs; _sgi++) {");
                        PushIndent();
                        string elemExpr = loadNExpr;
                        if (wgslAccumOp == "+")
                            AppendLine($"_grp_accum = _grp_accum + {elemExpr};");
                        else
                            AppendLine($"_grp_accum = {wgslAccumOp}(_grp_accum, {elemExpr});");
                        PopIndent();
                        AppendLine("}");
                        // Write back, bitcasting to IR type if needed
                        if (needsBitcast)
                            AppendLine($"{target} = bitcast<{irType}>(_grp_accum);");
                        else
                            AppendLine($"{target} = _grp_accum;");
                        PopIndent();
                        AppendLine("}");
                        return;
                    }
                }
                else if (subgroupOp != null || (subgroupOp != null && is64BitReduce))
                {
                    // No subgroups (or 64-bit emulated type): workgroup-level shared-memory reduction.
                    var reduceSrc = methodCall.Count > 0 ? Load(methodCall[0]) : null;
                    if (reduceSrc != null)
                    {
                        UsesWarpShuffleEmulation = true;

                        // Determine effective WGSL type from TReduction
                        string irType = earlyIrType;
                        string wgslType = earlyWgslType;
                        bool needsBitcast = wgslType != irType;

                        string wgslAccumOp;
                        if (subgroupOp.EndsWith("Max", StringComparison.Ordinal))
                            wgslAccumOp = "max";
                        else if (subgroupOp.EndsWith("Min", StringComparison.Ordinal))
                            wgslAccumOp = "min";
                        else
                            wgslAccumOp = "+";

                        AppendLine($"// Reduce via shared memory{(is64BitReduce ? " (64-bit emulated)" : " (subgroups unavailable)")}: {methodCall.Target.Name}");

                        bool is64Bit = is64BitReduce;

                        // Step 1: all threads write their value to shared memory
                        string storeVal = needsBitcast ? $"bitcast<{wgslType}>({reduceSrc})" : $"{reduceSrc}";
                        bool isOzakiF64 = wgslType == "emu_f64" && Backend.Options.UseOzakiF64Emulation;
                        if (is64Bit)
                        {
                            if (isOzakiF64)
                            {
                                // Ozaki emu_f64 = vec4<f32>: store all 4 components
                                AppendLine($"_warp_shuffle_buf[local_index * 4u] = bitcast<u32>({storeVal}.x);");
                                AppendLine($"_warp_shuffle_buf[local_index * 4u + 1u] = bitcast<u32>({storeVal}.y);");
                                AppendLine($"_warp_shuffle_buf[local_index * 4u + 2u] = bitcast<u32>({storeVal}.z);");
                                AppendLine($"_warp_shuffle_buf[local_index * 4u + 3u] = bitcast<u32>({storeVal}.w);");
                            }
                            else
                            {
                                // Dekker emu_f64 = vec2<f32> or emu_i64/emu_u64 = vec2<u32>: store 2 components
                                AppendLine($"_warp_shuffle_buf[local_index * 2u] = bitcast<u32>({storeVal}.x);");
                                AppendLine($"_warp_shuffle_buf[local_index * 2u + 1u] = bitcast<u32>({storeVal}.y);");
                            }
                        }
                        else
                        {
                            AppendLine($"_warp_shuffle_buf[local_index] = {BitcastToU32(storeVal, wgslType)};");
                        }
                        AppendLine("workgroupBarrier();");

                        // Step 2: all threads cooperatively reduce across the workgroup
                        string componentType = wgslType == "emu_f64" ? "f32" : "u32";
                        string load0, loadN;
                        if (is64Bit)
                        {
                            if (isOzakiF64)
                            {
                                // Ozaki: reconstruct vec4<f32> from 4 u32 slots
                                load0 = $"{wgslType}(bitcast<{componentType}>(_warp_shuffle_buf[0u]), bitcast<{componentType}>(_warp_shuffle_buf[1u]), bitcast<{componentType}>(_warp_shuffle_buf[2u]), bitcast<{componentType}>(_warp_shuffle_buf[3u]))";
                                loadN = $"{wgslType}(bitcast<{componentType}>(_warp_shuffle_buf[_ri * 4u]), bitcast<{componentType}>(_warp_shuffle_buf[_ri * 4u + 1u]), bitcast<{componentType}>(_warp_shuffle_buf[_ri * 4u + 2u]), bitcast<{componentType}>(_warp_shuffle_buf[_ri * 4u + 3u]))";
                            }
                            else
                            {
                                // Dekker or integer: reconstruct from 2 u32 slots
                                load0 = $"{wgslType}(bitcast<{componentType}>(_warp_shuffle_buf[0u]), bitcast<{componentType}>(_warp_shuffle_buf[1u]))";
                                loadN = $"{wgslType}(bitcast<{componentType}>(_warp_shuffle_buf[_ri * 2u]), bitcast<{componentType}>(_warp_shuffle_buf[_ri * 2u + 1u]))";
                            }
                        }
                        else
                        {
                            load0 = BitcastFromU32("_warp_shuffle_buf[0u]", wgslType);
                            loadN = BitcastFromU32("_warp_shuffle_buf[_ri]", wgslType);
                        }
                        AppendLine("{");
                        PushIndent();
                        AppendLine($"var _reduce_accum : {wgslType} = {load0};");
                        AppendLine("for (var _ri : u32 = 1u; _ri < workgroup_size.x; _ri++) {");
                        PushIndent();
                        // For 64-bit emulated types, use emulated math helpers instead of
                        // WGSL built-ins (max/min do component-wise on vec2, + lacks carry propagation)
                        string accumExpr = GetEmulated64BitAccumExpr(is64Bit, wgslType, wgslAccumOp, "_reduce_accum", loadN);
                        AppendLine($"_reduce_accum = {accumExpr};");
                        PopIndent();
                        AppendLine("}");
                        if (needsBitcast)
                            AppendLine($"{target} = bitcast<{irType}>(_reduce_accum);");
                        else
                            AppendLine($"{target} = _reduce_accum;");
                        PopIndent();
                        AppendLine("}");
                        AppendLine("workgroupBarrier();");
                        return;
                    }
                }
            }


            // Final fallback: emit a type-correct zero value to avoid WGSL type errors.
            // The 12345.0 literal previously used caused 'abstract-float to i32' type errors.
            AppendLine($"// Call: {methodCall.Target.Name} (Unmapped)");
            string fallbackType = TypeGenerator[methodCall.Type];
            AppendLine($"{target} = {fallbackType}(0); // Unmapped fallback");
        }


        // Casts
        public virtual void GenerateCode(IntAsPointerCast value)
        {
            var target = Load(value);
            var source = Load(value.Value);
            Declare(target);
            AppendLine($"{target} = {source}; // intAsPtr");
        }

        public virtual void GenerateCode(PointerAsIntCast value)
        {
            var target = Load(value);
            var source = Load(value.Value);
            Declare(target);
            AppendLine($"{target} = {target.Type}({source}); // ptrAsInt");
        }

        public virtual void GenerateCode(PointerCast value)
        {
            var target = Load(value);
            var source = Load(value.Value);
            Declare(target);
            AppendLine($"{target} = {source}; // ptrCast");
        }

        public virtual void GenerateCode(AddressSpaceCast value)
        {
            var target = Load(value);
            var source = Load(value.Value);
            Declare(target);
            AppendLine($"{target} = {source}; // addrSpaceCast");
        }

        public virtual void GenerateCode(FloatAsIntCast value)
        {
            var target = Load(value);
            var source = Load(value.Value);
            Declare(target);
            if (source.Type == "f16")
            {
                // f16 is 2 bytes, target is i32 (4 bytes) -- size-mismatched bitcast is invalid in WGSL.
                // Pack f16 into the low 16 bits of a u32 via vec2<f16>, then cast to i32.
                AppendLine($"{target} = i32(bitcast<u32>(vec2<f16>({source}, f16(0.0))));");
            }
            else
            {
                AppendLine($"{target} = bitcast<{target.Type}>({source});");
            }
        }

        public virtual void GenerateCode(IntAsFloatCast value)
        {
            var target = Load(value);
            var source = Load(value.Value);
            Declare(target);
            if (target.Type == "f16")
            {
                // i32 is 4 bytes, target f16 is 2 bytes -- size-mismatched bitcast is invalid in WGSL.
                // Reinterpret u32 as vec2<f16> and extract the low half.
                AppendLine($"{target} = bitcast<vec2<f16>>(u32({source})).x;");
            }
            else
            {
                AppendLine($"{target} = bitcast<{target.Type}>({source});");
            }
        }

        /// <summary>
        /// Produces a WGSL expression that packs a scalar value into a u32 via bitcast.
        /// For f16 (2 bytes), a direct <c>bitcast&lt;u32&gt;(f16)</c> is invalid because WGSL
        /// requires bitcast operands to have matching byte sizes.  Instead we pack via
        /// <c>bitcast&lt;u32&gt;(vec2&lt;f16&gt;(val, f16(0.0)))</c>.
        /// </summary>
        protected static string BitcastToU32(string expr, string sourceWgslType)
        {
            if (sourceWgslType == "f16")
                return $"bitcast<u32>(vec2<f16>({expr}, f16(0.0)))";
            return $"bitcast<u32>({expr})";
        }

        /// <summary>
        /// Produces a WGSL expression that unpacks a u32 into the given scalar type via bitcast.
        /// For f16 (2 bytes), a direct <c>bitcast&lt;f16&gt;(u32)</c> is invalid.  Instead we
        /// unpack via <c>bitcast&lt;vec2&lt;f16&gt;&gt;(u32).x</c>.
        /// </summary>
        protected static string BitcastFromU32(string expr, string targetWgslType)
        {
            if (targetWgslType == "f16")
                return $"bitcast<vec2<f16>>({expr}).x";
            return $"bitcast<{targetWgslType}>({expr})";
        }

        // Atomics & Barriers

        /// <summary>
        /// Override in subclasses to indicate whether the atomic target pointer refers to a
        /// float-typed parameter. WGSL has no atomic&lt;f32&gt;, so float atomics are stored as
        /// atomic&lt;u32&gt; (raw bits) and emulated via a CAS loop.
        /// </summary>
        protected virtual bool IsFloatAtomicTarget(Value target) => false;

        /// <summary>
        /// Returns true if the atomic target pointer resolves to an unsigned-typed parameter.
        /// WGSL's atomicMin/atomicMax on i32 do signed comparisons, so unsigned reductions need
        /// atomic&lt;u32&gt; buffers and bitcast values to u32 for correct unsigned semantics.
        /// </summary>
        protected virtual bool IsUnsignedAtomicTarget(Value target) => false;

        /// <summary>
        /// Returns true if the atomic target pointer resolves to a buffer with emulated 64-bit
        /// element type (emu_i64, emu_u64, emu_f64). WGSL doesn't support atomic operations on
        /// vec2&lt;u32&gt;, so these use plain storage and direct stores.
        /// </summary>
        protected virtual bool IsEmulated64BitAtomicTarget(Value target) => false;

        public virtual void GenerateCode(GenericAtomic value)
        {
            var target = Load(value);
            var ptr = Load(value.Target);
            var val = Load(value.Value);
            Declare(target);

            // WGSL only supports atomicAdd/Min/Max/etc. on i32 and u32.
            // Emulated 64-bit types (emu_i64, emu_u64, emu_f64 = vec2<u32>) cannot use WGSL atomics.
            // The reduction algorithm already accumulates via shared memory; only thread 0 writes
            // the final result, so a plain store is safe (no race condition).
            //
            // Detection: check both the explicit body-struct tracking AND the value's IR type.
            // The IR value type for emulated 64-bit is vec2<u32> (BasicValueType.None for struct types),
            // but we can also check via TypeGenerator which maps to emu_i64/emu_u64/emu_f64.
            string valWgslType = TypeGenerator[value.Value.Type];
            bool isEmu64 = IsEmulated64BitAtomicTarget(value.Target)
                || valWgslType == "emu_i64" || valWgslType == "emu_u64" || valWgslType == "emu_f64";
            if (isEmu64)
            {
                // For emulated 64-bit: read-compare-write using emulation functions.
                // Only thread 0 writes, so no race, but we still emit a compare-and-store
                // pattern since the output buffer may have the identity value.
                string prefix = valWgslType switch
                {
                    "emu_f64" => "f64",
                    "emu_u64" => "u64",
                    // NOTE: TypeGenerator returns "emu_i64" for BOTH long AND ulong, because ILGPU's
                    // BasicValueType enum has no UInt64 member. Must also check value.IsUnsigned.
                    "emu_i64" => value.IsUnsigned ? "u64" : "i64",
                    _ => value.IsUnsigned ? "u64" : "i64"
                };

                string emuOp = value.Kind switch
                {
                    AtomicKind.Max => $"{prefix}_max",
                    AtomicKind.Min => $"{prefix}_min",
                    AtomicKind.Add => $"{prefix}_add",
                    AtomicKind.Exchange => null,  // plain store
                    _ => $"{prefix}_add"
                };
                if (emuOp != null)
                {
                    // Read current value, apply op, write back
                    string oldVar = $"_emu64_old_{target.Name}";
                    if (valWgslType == "emu_f64")
                    {
                        // emu_f64 (Dekker vec2<f32>) is NOT bit-compatible with IEEE-754 double.
                        // The output buffer stores IEEE-754 bytes written by the C# host.
                        // Must: (1) read raw f32 pair, bitcast to u32, convert to Dekker for arithmetic,
                        //        (2) after op, convert Dekker result back to IEEE-754 bits before storing.
                        string rawOldVar = $"_raw_old_{target.Name}";
                        string resultVar = $"_result_{target.Name}";
                        string ieeeBitsVar = $"_ieee_bits_{target.Name}";
                        AppendLine($"let {rawOldVar} = *{ptr};");
                        AppendLine($"let {oldVar} = f64_from_ieee754_bits(bitcast<u32>({rawOldVar}.x), bitcast<u32>({rawOldVar}.y));");
                        AppendLine($"let {resultVar} = {emuOp}({oldVar}, {val});");
                        AppendLine($"let {ieeeBitsVar} = f64_to_ieee754_bits({resultVar});");
                        AppendLine($"*{ptr} = emu_f64(bitcast<f32>({ieeeBitsVar}.x), bitcast<f32>({ieeeBitsVar}.y));");
                        AppendLine($"{target} = {oldVar};");
                    }
                    else
                    {
                        AppendLine($"let {oldVar} = *{ptr};");
                        AppendLine($"*{ptr} = {emuOp}({oldVar}, {val});");
                        AppendLine($"{target} = {oldVar};");
                    }
                }

                else
                {
                    // Exchange: plain store
                    AppendLine($"*{ptr} = {val};");
                    AppendLine($"{target} = {val};");
                }
                return;
            }

            // Float atomics (f32/f16) must be emulated using a CAS loop on atomic<u32>.
            if (IsFloatAtomicTarget(value.Target))
            {
                // The buffer is declared as atomic<u32>. We bitcast f32 <-> u32.
                // CAS loop: atomically read-modify-write the float value.
                string casOld = $"_cas_old_{target.Name}";
                string casNew = $"_cas_new_{target.Name}";
                string casRes = $"_cas_res_{target.Name}";

                string floatType = target.Type;
                AppendLine($"var {casOld} = atomicLoad({ptr});");
                AppendLine($"loop {{");
                PushIndent();

                string floatOld = BitcastFromU32(casOld, floatType);
                string newFloat = value.Kind switch
                {
                    AtomicKind.Add => $"{floatOld} + {val}",
                    AtomicKind.Max => $"max({floatOld}, {val})",
                    AtomicKind.Min => $"min({floatOld}, {val})",
                    AtomicKind.Exchange => $"{val}",
                    _ => $"{floatOld} + {val}"  // fallback: Add
                };

                AppendLine($"let {casNew} = {BitcastToU32(newFloat, floatType)};");
                AppendLine($"let {casRes} = atomicCompareExchangeWeak({ptr}, {casOld}, {casNew});");
                AppendLine($"if ({casRes}.exchanged) {{ break; }}");
                AppendLine($"{casOld} = {casRes}.old_value;");
                PopIndent();
                AppendLine($"}}");
                AppendLine($"{target} = {BitcastFromU32(casOld, floatType)};");
            }
            else if (IsUnsignedAtomicTarget(value.Target))
            {
                // Buffer is atomic<u32>. Values from ILGPU IR are i32, need bitcast to u32.
                string op = value.Kind switch
                {
                    AtomicKind.Add => "atomicAdd",
                    AtomicKind.And => "atomicAnd",
                    AtomicKind.Or => "atomicOr",
                    AtomicKind.Xor => "atomicXor",
                    AtomicKind.Max => "atomicMax",
                    AtomicKind.Min => "atomicMin",
                    AtomicKind.Exchange => "atomicExchange",
                    _ => "atomicAdd"
                };

                // Bitcast value to u32 for unsigned atomic semantics
                AppendLine($"{target} = bitcast<i32>({op}({ptr}, bitcast<u32>({val})));");
            }
            else
            {
                string op = value.Kind switch
                {
                    AtomicKind.Add => "atomicAdd",
                    AtomicKind.And => "atomicAnd",
                    AtomicKind.Or => "atomicOr",
                    AtomicKind.Xor => "atomicXor",
                    AtomicKind.Max => "atomicMax",
                    AtomicKind.Min => "atomicMin",
                    AtomicKind.Exchange => "atomicExchange",
                    _ => "atomicAdd"
                };

                AppendLine($"{target} = {op}({ptr}, {val});");
            }
        }


        public virtual void GenerateCode(AtomicCAS value)
        {
            var target = Load(value);
            var ptr = Load(value.Target);
            var cmp = Load(value.Value);
            var val = Load(value.CompareValue);
            Declare(target);
            AppendLine($"{target} = atomicCompareExchangeWeak({ptr}, {cmp}, {val}).old_value;");
        }



        public virtual void GenerateCode(MemoryBarrier value)
        {
            AppendLine("workgroupBarrier();");
        }

        public virtual void GenerateCode(PredicateBarrier value)
        {
            AppendLine("workgroupBarrier();");
        }

        // Warp Operations
        public virtual void GenerateCode(Broadcast value)
        {
            var target = Load(value);
            var source = Load(value.Variable);
            var origin = Load(value.Origin);
            Declare(target);

            if (value.Kind == global::ILGPU.IR.Values.BroadcastKind.GroupLevel)
            {
                // Workgroup-level broadcast: use shared memory + barrier
                // 1. Origin thread writes value to shared memory
                // 2. All threads synchronize
                // 3. All threads read from shared memory
                AppendLine($"if (i32(local_index) == {origin}) {{");
                PushIndent();
                AppendLine($"_broadcast_temp = {source};");
                PopIndent();
                AppendLine("}");
                AppendLine("workgroupBarrier();");
                AppendLine($"{target} = _broadcast_temp;");
                AppendLine("workgroupBarrier();");
            }
            else
            {
                // WarpLevel broadcast — broadcast from lane 0 of the warp
                if (HasSubgroups)
                {
                    AppendLine($"{target} = subgroupBroadcastFirst({source});");
                }
                else
                {
                    // Shared-memory emulation: write all values, read from warp-base (lane 0)
                    var warpBase = $"_warp_base_{target.Name}";
                    AppendLine($"_warp_shuffle_buf[local_index] = {BitcastToU32($"{source}", source.Type)};");
                    AppendLine("workgroupBarrier();");
                    AppendLine($"let {warpBase} = (local_index / workgroup_size.x) * workgroup_size.x;");
                    AppendLine($"{target} = {BitcastFromU32($"_warp_shuffle_buf[{warpBase}]", target.Type)};");
                    AppendLine("workgroupBarrier();");
                }
            }
        }

        public virtual void GenerateCode(WarpShuffle value)
        {
            var target = Load(value);
            var source = Load(value.Variable);
            var origin = Load(value.Origin);
            Declare(target);
            if (HasSubgroups)
            {
                AppendLine($"{target} = subgroupShuffle({source}, u32({origin}));");
            }
            else
            {
                // Shared-memory emulation:
                // 'origin' is a lane index (0..warpSize-1), not a global thread index.
                // The buffer is indexed by local_index, so we must compute:
                //   warp_base = (local_index / warp_size) * warp_size
                //   src_idx   = warp_base + origin
                var warpBase = $"_wb_{target.Name}";
                AppendLine($"_warp_shuffle_buf[local_index] = {BitcastToU32($"{source}", source.Type)};");
                AppendLine("workgroupBarrier();");
                AppendLine($"let {warpBase} = (local_index / u32(workgroup_size.x)) * u32(workgroup_size.x);");
                AppendLine($"{target} = {BitcastFromU32($"_warp_shuffle_buf[{warpBase} + u32({origin})]", target.Type)};");
                AppendLine("workgroupBarrier();");
            }
        }

        public virtual void GenerateCode(SubWarpShuffle value)
        {
            var target = Load(value);
            var source = Load(value.Variable);
            var origin = Load(value.Origin);
            Declare(target);
            if (HasSubgroups)
            {
                AppendLine($"{target} = subgroupShuffle({source}, u32({origin}));");
            }
            else
            {
                // Same fix as WarpShuffle: origin is lane-relative, buffer is global.
                var warpBase = $"_wb_{target.Name}";
                AppendLine($"_warp_shuffle_buf[local_index] = {BitcastToU32($"{source}", source.Type)};");
                AppendLine("workgroupBarrier();");
                AppendLine($"let {warpBase} = (local_index / u32(workgroup_size.x)) * u32(workgroup_size.x);");
                AppendLine($"{target} = {BitcastFromU32($"_warp_shuffle_buf[{warpBase} + u32({origin})]", target.Type)};");
                AppendLine("workgroupBarrier();");
            }
        }

        // Debug/IO
        public virtual void GenerateCode(DebugAssertOperation value)
        {
            // WGSL doesn't support debug assertions
        }

        public virtual void GenerateCode(WriteToOutput value)
        {
            // WGSL doesn't support stdout
        }

        // Other
        public virtual void GenerateCode(Predicate value)
        {
            var target = Load(value);
            var cond = Load(value.Condition);
            Declare(target);

            bool trueHasLoad = value.TrueValue is global::ILGPU.IR.Values.Load;
            bool falseHasLoad = value.FalseValue is global::ILGPU.IR.Values.Load;

            if (trueHasLoad || falseHasLoad)
            {
                var trueVal = Load(value.TrueValue);
                var falseVal = Load(value.FalseValue);
                AppendLine($"if ({cond}) {{");
                AppendLine($"    {target} = {trueVal};");
                AppendLine($"}} else {{");
                AppendLine($"    {target} = {falseVal};");
                AppendLine($"}}");
            }
            else
            {
                var trueVal = Load(value.TrueValue);
                var falseVal = Load(value.FalseValue);
                AppendLine($"{target} = select({falseVal}, {trueVal}, {cond});");
            }
        }

        public virtual void GenerateCode(DynamicMemoryLengthValue value)
        {
            var target = Load(value);
            Declare(target);
            // Use i32 literal (not u32) since the target variable is i32
            AppendLine($"{target} = 0; // dynamic memory length (placeholder)");
        }

        public virtual void GenerateCode(AlignTo value)
        {
            var target = Load(value);
            var source = Load(value.Source);
            Declare(target);
            AppendLine($"{target} = {source}; // alignTo");
        }

        public virtual void GenerateCode(AsAligned value)
        {
            var target = Load(value);
            var source = Load(value.Source);
            Declare(target);
            AppendLine($"{target} = {source}; // asAligned");
        }

        public virtual void GenerateCode(LanguageEmitValue value)
        {
            // Raw WGSL emission - not commonly used
        }

        public virtual void GenerateThrow(Value value)
        {
            // WebGPU does not support exceptions. 
            // We emit a trap/unreachable, or just a comment if we assume it's unreachable in valid code.
            AppendLine($"// [WGSL] Throw encountered: {value} (Ignored/Unreachable)");

            if (IsStateMachineActive)
            {
                // Break out of the loop
                AppendLine("current_block = -1;");
                AppendLine("break;");
            }
            else
            {
                AppendLine("return;");
            }
        }

        #endregion

        #region Half Conversion Intrinsics

        /// <summary>
        /// Handles <c>HalfExtensions.ConvertHalfToFloat(Half)</c> by emitting a native
        /// WGSL <c>f32(source)</c> conversion instead of the lookup-table implementation
        /// that the CPU path uses. The lookup tables are static .NET arrays that cannot
        /// be accessed from GPU code, so inlining the default implementation produces
        /// all-zero results.
        /// </summary>
        public static void GenerateConvertHalfToFloat(WebGPUBackend backend, WGSLCodeGenerator codeGenerator, Value value)
        {
            if (value is MethodCall methodCall)
            {
                var target = codeGenerator.LoadIntrinsicValue(value);
                var operand = codeGenerator.LoadIntrinsicValue(methodCall[0].Resolve());
                codeGenerator.Declare(target);
                codeGenerator.AppendLine($"{target} = f32({operand});");
            }
        }

        /// <summary>
        /// Handles <c>HalfExtensions.ConvertFloatToHalf(float)</c> by emitting a native
        /// WGSL <c>f16(source)</c> conversion.
        /// </summary>
        public static void GenerateConvertFloatToHalf(WebGPUBackend backend, WGSLCodeGenerator codeGenerator, Value value)
        {
            if (value is MethodCall methodCall)
            {
                var target = codeGenerator.LoadIntrinsicValue(value);
                var operand = codeGenerator.LoadIntrinsicValue(methodCall[0].Resolve());
                codeGenerator.Declare(target);
                codeGenerator.AppendLine($"{target} = f16({operand});");
            }
        }

        #endregion

        #region Math Intrinsics

        public static void GenerateAbs(WebGPUBackend backend, WGSLCodeGenerator codeGenerator, Value value)
        {
            if (value is MethodCall methodCall)
            {
                var target = codeGenerator.LoadIntrinsicValue(value);
                var operand = codeGenerator.LoadIntrinsicValue(methodCall[0].Resolve());
                codeGenerator.Declare(target);
                codeGenerator.AppendLine($"{target} = abs({operand});");
            }
        }

        public static void GenerateSign(WebGPUBackend backend, WGSLCodeGenerator codeGenerator, Value value)
        {
            if (value is MethodCall methodCall)
            {
                var target = codeGenerator.LoadIntrinsicValue(value);
                var operand = codeGenerator.LoadIntrinsicValue(methodCall[0].Resolve());
                codeGenerator.Declare(target);
                string call = $"sign({operand})";
                if (codeGenerator.TypeGenerator[value.Type] == "i32" || codeGenerator.TypeGenerator[value.Type] == "u32")
                {
                    call = $"i32({call})";
                }
                codeGenerator.AppendLine($"{target} = {call};");
            }
        }

        public static void GenerateRound(WebGPUBackend backend, WGSLCodeGenerator codeGenerator, Value value)
        {
            if (value is MethodCall methodCall)
            {
                var target = codeGenerator.LoadIntrinsicValue(value);
                var operand = codeGenerator.LoadIntrinsicValue(methodCall[0].Resolve());
                codeGenerator.Declare(target);
                codeGenerator.AppendLine($"{target} = round({operand});");
            }
        }

        public static void GenerateTruncate(WebGPUBackend backend, WGSLCodeGenerator codeGenerator, Value value)
        {
            if (value is MethodCall methodCall)
            {
                var target = codeGenerator.LoadIntrinsicValue(value);
                var operand = codeGenerator.LoadIntrinsicValue(methodCall[0].Resolve());
                codeGenerator.Declare(target);
                codeGenerator.AppendLine($"{target} = trunc({operand});");
            }
        }

        public static void GenerateAtan2(WebGPUBackend backend, WGSLCodeGenerator codeGenerator, Value value)
        {
            if (value is MethodCall methodCall)
            {
                var target = codeGenerator.LoadIntrinsicValue(value);
                var y = codeGenerator.LoadIntrinsicValue(methodCall[0].Resolve());
                var x = codeGenerator.LoadIntrinsicValue(methodCall[1].Resolve());
                codeGenerator.Declare(target);
                codeGenerator.AppendLine($"{target} = atan2({y}, {x});");
            }
        }

        public static void GenerateMax(WebGPUBackend backend, WGSLCodeGenerator codeGenerator, Value value)
        {
            if (value is MethodCall methodCall)
            {
                var target = codeGenerator.LoadIntrinsicValue(value);
                var a = codeGenerator.LoadIntrinsicValue(methodCall[0].Resolve());
                var b = codeGenerator.LoadIntrinsicValue(methodCall[1].Resolve());
                codeGenerator.Declare(target);
                codeGenerator.AppendLine($"{target} = max({a}, {b});");
            }
        }

        public static void GenerateMin(WebGPUBackend backend, WGSLCodeGenerator codeGenerator, Value value)
        {
            if (value is MethodCall methodCall)
            {
                var target = codeGenerator.LoadIntrinsicValue(value);
                var a = codeGenerator.LoadIntrinsicValue(methodCall[0].Resolve());
                var b = codeGenerator.LoadIntrinsicValue(methodCall[1].Resolve());
                codeGenerator.Declare(target);
                codeGenerator.AppendLine($"{target} = min({a}, {b});");
            }
        }

        public static void GeneratePow(WebGPUBackend backend, WGSLCodeGenerator codeGenerator, Value value)
        {
            if (value is MethodCall methodCall)
            {
                var target = codeGenerator.LoadIntrinsicValue(value);
                var b = codeGenerator.LoadIntrinsicValue(methodCall[0].Resolve());
                var e = codeGenerator.LoadIntrinsicValue(methodCall[1].Resolve());
                codeGenerator.Declare(target);
                codeGenerator.AppendLine($"{target} = pow({b}, {e});");
            }
        }

        public static void GenerateClamp(WebGPUBackend backend, WGSLCodeGenerator codeGenerator, Value value)
        {
            if (value is MethodCall methodCall)
            {
                var target = codeGenerator.LoadIntrinsicValue(value);
                var val = codeGenerator.LoadIntrinsicValue(methodCall[0].Resolve());
                var min = codeGenerator.LoadIntrinsicValue(methodCall[1].Resolve());
                var max = codeGenerator.LoadIntrinsicValue(methodCall[2].Resolve());
                codeGenerator.Declare(target);
                codeGenerator.AppendLine($"{target} = clamp({val}, {min}, {max});");
            }
        }

        public static void GenerateFusedMultiplyAdd(WebGPUBackend backend, WGSLCodeGenerator codeGenerator, Value value)
        {
            if (value is MethodCall methodCall)
            {
                var target = codeGenerator.LoadIntrinsicValue(value);
                var x = codeGenerator.LoadIntrinsicValue(methodCall[0].Resolve());
                var y = codeGenerator.LoadIntrinsicValue(methodCall[1].Resolve());
                var z = codeGenerator.LoadIntrinsicValue(methodCall[2].Resolve());
                codeGenerator.Declare(target);
                codeGenerator.AppendLine($"{target} = fma({x}, {y}, {z});");
            }
        }

        /// <summary>
        /// Generates WGSL code for XMath.Rsqrt (reciprocal square root: 1/sqrt(x)).
        /// </summary>
        public static void GenerateRsqrt(WebGPUBackend backend, WGSLCodeGenerator codeGenerator, Value value)
        {
            if (value is MethodCall methodCall)
            {
                var target = codeGenerator.LoadIntrinsicValue(value);
                var operand = codeGenerator.LoadIntrinsicValue(methodCall[0].Resolve());
                codeGenerator.Declare(target);
                // WGSL has inverseSqrt but we use 1.0/sqrt() for clarity
                codeGenerator.AppendLine($"{target} = 1.0 / sqrt({operand});");
            }
        }

        /// <summary>
        /// Generates WGSL code for XMath.Rcp (reciprocal: 1/x).
        /// </summary>
        public static void GenerateRcp(WebGPUBackend backend, WGSLCodeGenerator codeGenerator, Value value)
        {
            if (value is MethodCall methodCall)
            {
                var target = codeGenerator.LoadIntrinsicValue(value);
                var operand = codeGenerator.LoadIntrinsicValue(methodCall[0].Resolve());
                codeGenerator.Declare(target);
                codeGenerator.AppendLine($"{target} = 1.0 / {operand};");
            }
        }

        /// <summary>
        /// Generates WGSL code for a warp-level reduction (WarpExtensions.Reduce).
        /// Emits subgroupMax/Min/Add when subgroups are available, otherwise falls back
        /// to a shared-memory butterfly reduction within the subgroup.
        /// The reduction operation type is determined by TReduction.CLCommand.
        /// </summary>
        public static void GenerateWarpReduce(WebGPUBackend backend, WGSLCodeGenerator codeGenerator, Value value)
        {
            if (value is MethodCall methodCall)
            {
                var target = codeGenerator.LoadIntrinsicValue(value);
                var operand = codeGenerator.LoadIntrinsicValue(methodCall[0].Resolve());
                codeGenerator.Declare(target);

                // Determine the reduction operation from the TReduction type argument.
                string wgslOp = GetSubgroupReduceOp(methodCall.Target.Source as MethodInfo);

                if (backend.HasSubgroups && wgslOp != null)
                {
                    // Use native subgroup reduce intrinsic
                    codeGenerator.AppendLine($"{target} = {wgslOp}({operand});");
                }
                else
                {
                    // Fallback: software butterfly warp reduce using shared memory would require
                    // complex code generation. For now, emit a simple passthrough so the shader
                    // at least compiles. The shared-memory group reduce (in ILGroupExtensions)
                    // will aggregate these values across warps in a workgroup barrier.
                    // TODO: Implement full XOR-butterfly fallback when subgroups are unavailable.
                    codeGenerator.AppendLine($"{target} = {operand}; // warp reduce (subgroups unavailable)");
                }
            }
        }

        /// <summary>
        /// Generates WGSL code for a group-level reduction (GroupExtensions.Reduce/AllReduce,
        /// ILGroupExtensions.Reduce/AllReduce).
        /// Emits a full cross-subgroup reduction: per-subgroup native reduce (subgroupMax/Min/Add),
        /// then aggregates partial results across all subgroups via the _grp_sg_results workgroup array.
        /// Requires subgroup_id and subgroup_invocation_id builtins (added to kernel signature automatically).
        /// </summary>
        public static void GenerateGroupReduce(WebGPUBackend backend, WGSLCodeGenerator codeGenerator, Value value)
            => GenerateGroupAllReduce(backend, codeGenerator, value);

        /// <summary>
        /// Core implementation for group-level AllReduce. Emits inline WGSL that:
        /// <list type="number">
        ///   <item>Does a per-subgroup native reduce (subgroupMax / subgroupMin / subgroupAdd).</item>
        ///   <item>Lane 0 of each subgroup stores its partial result to _grp_sg_results[subgroup_id].</item>
        ///   <item>workgroupBarrier() synchronises.</item>
        ///   <item>All threads in subgroup 0 load from _grp_sg_results[their_lane] (padded with identity) and do a
        ///         second subgroup reduce — thread 0 of subgroup 0 then holds the workgroup-wide result.</item>
        ///   <item>Thread 0,0 writes the result back to slot 0; another workgroupBarrier(); all threads read slot 0.</item>
        /// </list>
        /// Sets UsesGroupReduce = true so the kernel header emits _grp_sg_results/atomic vars and the
        /// signature includes subgroup_id / subgroup_size / subgroup_invocation_id.
        /// </summary>
        public static void GenerateGroupAllReduce(WebGPUBackend backend, WGSLCodeGenerator codeGenerator, Value value)
        {
            WebGPUBackend.Log($"[WGSL-DBG] GenerateGroupAllReduce called, value type={value?.GetType().Name}, isMethodCall={value is MethodCall}");
            if (value is not MethodCall methodCall)
            {
                WebGPUBackend.Log("[WGSL-DBG] GenerateGroupAllReduce: early exit - not a MethodCall");
                return;
            }

            // Signal that the kernel needs the workgroup-level shared arrays and subgroup_id builtin.
            codeGenerator.UsesGroupReduce = true;

            var target = codeGenerator.LoadIntrinsicValue(value);
            var operand = codeGenerator.LoadIntrinsicValue(methodCall[0].Resolve());
            codeGenerator.Declare(target);

            // Determine the WGSL subgroup operation and identity value from TReduction type argument.
            var srcMethodInfo = methodCall.Target.Source as MethodInfo;
            WebGPUBackend.Log($"[WGSL-DBG] srcMethodInfo={srcMethodInfo?.Name}, isGeneric={srcMethodInfo?.IsGenericMethod}, genericArgs={string.Join(",", srcMethodInfo?.GetGenericArguments()?.Select(t => t.Name) ?? Array.Empty<string>())}");
            string? wgslOp = GetSubgroupReduceOp(srcMethodInfo);
            string? identityExpr = GetReductionIdentityExpr(srcMethodInfo);

            // Determine the WGSL scalar type for this reduction early (needed for is64Bit check)
            string irType = codeGenerator.TypeGenerator[methodCall.Type];
            string wgslType = codeGenerator.GetReductionWgslType(srcMethodInfo) ?? irType;
            bool needsBitcast = wgslType != irType;
            bool is64Bit = wgslType.StartsWith("emu_"); // emu_i64, emu_u64, emu_f64 are vec2 types

            WebGPUBackend.Log($"[WGSL-DBG] HasSubgroups={backend.HasSubgroups}, wgslOp={wgslOp}, identityExpr={identityExpr}");

            if (is64Bit && wgslOp != null && identityExpr != null)
            {
                WebGPUBackend.Log($"[WGSL-DBG] GenerateGroupAllReduce: 64-bit shared-memory reduction, wgslType={wgslType}");
                // For 64-bit emulated types, perform a complete shared-memory reduction
                // since subgroup ops don't support vec2<u32>.
                codeGenerator.UsesWarpShuffleEmulation = true;

                // Determine accumulation op
                string wgslAccumOp;
                if (wgslOp.EndsWith("Max", StringComparison.Ordinal))
                    wgslAccumOp = "max";
                else if (wgslOp.EndsWith("Min", StringComparison.Ordinal))
                    wgslAccumOp = "min";
                else
                    wgslAccumOp = "+";

                string sfx64 = target.Name.Replace("v_", "_").TrimStart('_');

                // Step 1: all threads write their value to shared memory (2 u32 slots per thread)
                string storeVal = needsBitcast ? $"bitcast<{wgslType}>({operand})" : $"{operand}";

                bool isOzakiF64 = wgslType == "emu_f64" && backend.Options.UseOzakiF64Emulation;
                if (isOzakiF64)
                {
                    // Ozaki emu_f64 = vec4<f32>: store all 4 components
                    codeGenerator.AppendLine($"{{ _warp_shuffle_buf[local_index * 4u] = bitcast<u32>({storeVal}.x); _warp_shuffle_buf[local_index * 4u + 1u] = bitcast<u32>({storeVal}.y); _warp_shuffle_buf[local_index * 4u + 2u] = bitcast<u32>({storeVal}.z); _warp_shuffle_buf[local_index * 4u + 3u] = bitcast<u32>({storeVal}.w); }}");
                }
                else if (wgslType == "emu_f64")
                {
                    // Dekker emu_f64 = vec2<f32>: convert to IEEE-754 bits (2 u32 slots)
                    codeGenerator.AppendLine($"{{ let _bits = f64_to_ieee754_bits({storeVal}); _warp_shuffle_buf[local_index * 2u] = _bits.x; _warp_shuffle_buf[local_index * 2u + 1u] = _bits.y; }}");
                }
                else
                {
                    codeGenerator.AppendLine($"_warp_shuffle_buf[local_index * 2u] = {storeVal}.x;");
                    codeGenerator.AppendLine($"_warp_shuffle_buf[local_index * 2u + 1u] = {storeVal}.y;");
                }
                codeGenerator.AppendLine("workgroupBarrier();");

                // Step 2: all threads cooperatively reduce across the workgroup
                string load0;
                string loadN;
                if (isOzakiF64)
                {
                    // Ozaki: reconstruct vec4<f32> from 4 u32 slots
                    load0 = $"emu_f64(bitcast<f32>(_warp_shuffle_buf[0u]), bitcast<f32>(_warp_shuffle_buf[1u]), bitcast<f32>(_warp_shuffle_buf[2u]), bitcast<f32>(_warp_shuffle_buf[3u]))";
                    loadN = $"emu_f64(bitcast<f32>(_warp_shuffle_buf[_ri_{sfx64} * 4u]), bitcast<f32>(_warp_shuffle_buf[_ri_{sfx64} * 4u + 1u]), bitcast<f32>(_warp_shuffle_buf[_ri_{sfx64} * 4u + 2u]), bitcast<f32>(_warp_shuffle_buf[_ri_{sfx64} * 4u + 3u]))";
                }
                else if (wgslType == "emu_f64")
                {
                    // Dekker: convert from IEEE-754 bits (2 u32 slots)
                    load0 = $"f64_from_ieee754_bits(_warp_shuffle_buf[0u], _warp_shuffle_buf[1u])";
                    loadN = $"f64_from_ieee754_bits(_warp_shuffle_buf[_ri_{sfx64} * 2u], _warp_shuffle_buf[_ri_{sfx64} * 2u + 1u])";
                }
                else if (wgslType.Contains("emu_"))
                {
                    load0 = $"{wgslType}(_warp_shuffle_buf[0u], _warp_shuffle_buf[1u])";
                    loadN = $"{wgslType}(_warp_shuffle_buf[_ri_{sfx64} * 2u], _warp_shuffle_buf[_ri_{sfx64} * 2u + 1u])";
                }
                else
                {
                    load0 = $"{wgslType}(_warp_shuffle_buf[0u], _warp_shuffle_buf[1u])";
                    loadN = $"{wgslType}(_warp_shuffle_buf[_ri_{sfx64} * 2u], _warp_shuffle_buf[_ri_{sfx64} * 2u + 1u])";
                }

                codeGenerator.AppendLine("{");
                codeGenerator.AppendLine($"    var _reduce_accum_{sfx64} : {wgslType} = {load0};");
                codeGenerator.AppendLine($"    for (var _ri_{sfx64} : u32 = 1u; _ri_{sfx64} < workgroup_size.x; _ri_{sfx64}++) {{");
                string accumExpr = GetEmulated64BitAccumExpr(true, wgslType, wgslAccumOp, $"_reduce_accum_{sfx64}", loadN);
                codeGenerator.AppendLine($"        _reduce_accum_{sfx64} = {accumExpr};");
                codeGenerator.AppendLine("    }");
                if (needsBitcast)
                    codeGenerator.AppendLine($"    {target} = bitcast<{irType}>(_reduce_accum_{sfx64});");
                else
                    codeGenerator.AppendLine($"    {target} = _reduce_accum_{sfx64};");
                codeGenerator.AppendLine("}");
                codeGenerator.AppendLine("workgroupBarrier();");
                return;
            }

            if (!backend.HasSubgroups || wgslOp == null || identityExpr == null)
            {
                WebGPUBackend.Log($"[WGSL-DBG] GenerateGroupAllReduce: fallback, reason: HasSubgroups={backend.HasSubgroups}, wgslOp={wgslOp}, identityExpr={identityExpr}");
                // Fallback: can't do cross-subgroup reduce without subgroups — emit identity.
                codeGenerator.AppendLine($"{target} = {identityExpr ?? "0"}; // group reduce (subgroups unavailable)");
                return;
            }

            // Generate unique variable names using target name as a suffix to avoid clashes in inlined code.
            string sfx = target.Name.Replace("v_", "_").TrimStart('_');

            // Wrap operand in bitcast if needed (e.g., i32 -> u32 for unsigned reductions)
            string operandExpr = needsBitcast ? $"bitcast<{wgslType}>({operand})" : $"{operand}";

            // === Step 1: Per-subgroup reduction ===
            codeGenerator.AppendLine($"let _sg_part_{sfx} = {wgslOp}({operandExpr}); // per-subgroup partial reduce");

            // === Step 2: Lane 0 of each subgroup stores partial result ===
            codeGenerator.AppendLine($"if (subgroup_invocation_id == 0u) {{");
            if (is64Bit)
            {
                // 64-bit emulated types need 2 consecutive u32 slots per subgroup
                codeGenerator.AppendLine($"    _grp_sg_results[subgroup_id * 2u] = bitcast<u32>(_sg_part_{sfx}.x);");
                codeGenerator.AppendLine($"    _grp_sg_results[subgroup_id * 2u + 1u] = bitcast<u32>(_sg_part_{sfx}.y);");
            }
            else
            {
                codeGenerator.AppendLine($"    _grp_sg_results[subgroup_id] = {BitcastToU32($"_sg_part_{sfx}", wgslType)};");
            }
            codeGenerator.AppendLine($"}}");

            // === Step 3: Synchronise so all subgroup slots are visible ===
            codeGenerator.AppendLine($"workgroupBarrier();");

            // === Step 4: Subgroup 0's threads reduce across all subgroup slots ===
            // Number of subgroups = ceil(workgroup_size.x / subgroup_size)
            // Lane i of subgroup 0 reads slot i (padded with identity if i >= num_subgroups)
            codeGenerator.AppendLine($"let _num_sg_{sfx} = (workgroup_size.x + subgroup_size - 1u) / subgroup_size;");
            if (is64Bit)
            {
                string loadExpr = $"{wgslType}(bitcast<{(wgslType == "emu_f64" ? "f32" : "u32")}>(_grp_sg_results[subgroup_invocation_id * 2u]), bitcast<{(wgslType == "emu_f64" ? "f32" : "u32")}>(_grp_sg_results[subgroup_invocation_id * 2u + 1u]))";
                codeGenerator.AppendLine($"let _sg0_val_{sfx} = select({identityExpr}, {loadExpr},");
            }
            else
            {
                string loadExpr = BitcastFromU32("_grp_sg_results[subgroup_invocation_id]", wgslType);
                codeGenerator.AppendLine($"let _sg0_val_{sfx} = select({identityExpr}, {loadExpr},");
            }
            codeGenerator.AppendLine($"    subgroup_invocation_id < _num_sg_{sfx});");
            codeGenerator.AppendLine($"let _final_{sfx} = {wgslOp}(_sg0_val_{sfx}); // second-level reduce across subgroup slots");

            // === Step 5: Thread (0,0) writes final result back so all threads can read it ===
            codeGenerator.AppendLine($"if (subgroup_id == 0u && subgroup_invocation_id == 0u) {{");
            if (is64Bit)
            {
                codeGenerator.AppendLine($"    _grp_sg_results[0] = bitcast<u32>(_final_{sfx}.x);");
                codeGenerator.AppendLine($"    _grp_sg_results[1] = bitcast<u32>(_final_{sfx}.y);");
            }
            else
            {
                codeGenerator.AppendLine($"    _grp_sg_results[0] = {BitcastToU32($"_final_{sfx}", wgslType)};");
            }
            codeGenerator.AppendLine($"}}");

            // === Step 6: Sync and all threads read the final group-wide result ===
            codeGenerator.AppendLine($"workgroupBarrier();");
            if (is64Bit)
            {
                string comp = wgslType == "emu_f64" ? "f32" : "u32";
                string readExpr = $"{wgslType}(bitcast<{comp}>(_grp_sg_results[0]), bitcast<{comp}>(_grp_sg_results[1]))";
                codeGenerator.AppendLine($"{target} = {(needsBitcast ? $"bitcast<{irType}>({readExpr})" : readExpr)};");
            }
            else
            {
                string readExpr = BitcastFromU32("_grp_sg_results[0]", wgslType);
                codeGenerator.AppendLine($"{target} = {(needsBitcast ? $"bitcast<{irType}>({readExpr})" : readExpr)};");
            };
        }

        /// <summary>
        /// Returns the WGSL identity literal for the IScanReduceOperation TReduction type.
        /// E.g. MaxInt32 → "-2147483648", MinInt32 → "2147483647", AddInt32 → "0".
        /// Returns null if the identity cannot be determined.
        /// </summary>
        private static string? GetReductionIdentityExpr(MethodInfo? sourceMethod)
        {
            if (sourceMethod == null) return null;
            try
            {
                Type[] genericArgs = sourceMethod.IsGenericMethod
                    ? sourceMethod.GetGenericArguments()
                    : null;
                if (genericArgs == null || genericArgs.Length < 2) return null;

                Type treductionType = genericArgs[1];
                var instance = Activator.CreateInstance(treductionType);

                // IScanReduceOperation<T>.Identity property
                var identityProp = treductionType.GetProperty(
                    "Identity",
                    System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance);
                if (identityProp == null) return null;

                object? identity = identityProp.GetValue(instance);
                return identity switch
                {
                    int i => i.ToString(),
                    uint u => $"{u}u",
                    float f => float.IsNegativeInfinity(f)
                        ? "-3.40282347e+38f"
                        : float.IsPositiveInfinity(f)
                            ? "3.40282347e+38f"
                            : f.ToString("G9") + "f",
                    long l => LongToEmuI64(l),
                    ulong ul => ULongToEmuU64(ul),
                    double d => DoubleToEmuF64(d),
                    _ => identity?.ToString() ?? "0",
                };
            }
            catch
            {
                return null;
            }
        }


        /// <summary>
        /// Converts a long value to its WGSL emu_i64 (vec2&lt;u32&gt;) literal representation.
        /// The low 32 bits go in .x, the high 32 bits in .y.
        /// </summary>
        private static string LongToEmuI64(long value)
        {
            ulong bits = unchecked((ulong)value);
            uint lo = (uint)(bits & 0xFFFFFFFF);
            uint hi = (uint)(bits >> 32);
            return $"emu_i64({lo}u, {hi}u)";
        }

        /// <summary>
        /// Converts a ulong value to its WGSL emu_u64 (vec2&lt;u32&gt;) literal representation.
        /// The low 32 bits go in .x, the high 32 bits in .y.
        /// </summary>
        private static string ULongToEmuU64(ulong value)
        {
            uint lo = (uint)(value & 0xFFFFFFFF);
            uint hi = (uint)(value >> 32);
            return $"emu_u64({lo}u, {hi}u)";
        }

        /// <summary>
        /// Converts a double value to its WGSL emu_f64 (vec2&lt;f32&gt;) literal representation.
        /// Uses bitcast to split the 64-bit IEEE 754 double into two u32 words,
        /// then bitcasts each to f32 for vec2&lt;f32&gt; storage.
        /// </summary>
        private static string DoubleToEmuF64(double value)
        {
            ulong bits = BitConverter.DoubleToUInt64Bits(value);
            uint lo = (uint)(bits & 0xFFFFFFFF);
            uint hi = (uint)(bits >> 32);
            return $"emu_f64(bitcast<f32>({lo}u), bitcast<f32>({hi}u))";
        }

        /// <summary>
        /// Extracts the WGSL subgroup operation name from an open or specialized generic
        /// MethodInfo whose second generic argument is a TReduction implementing
        /// IScanReduceOperation, using the CLCommand property ("max", "min", "add").
        /// Returns null if the operation cannot be determined.
        /// </summary>
        private static string? GetSubgroupReduceOp(MethodInfo? sourceMethod)
        {
            if (sourceMethod == null) return null;
            try
            {
                Type[] genericArgs = sourceMethod.IsGenericMethod
                    ? sourceMethod.GetGenericArguments()
                    : null;

                if (genericArgs == null || genericArgs.Length < 2) return null;

                Type treductionType = genericArgs[1];

                // Instantiate the TReduction struct to read CLCommand
                var instance = Activator.CreateInstance(treductionType);
                var clCommandProp = treductionType.GetProperty(
                    "CLCommand",
                    System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance);
                if (clCommandProp == null) return null;

                string? clCommand = clCommandProp.GetValue(instance) as string;
                return clCommand switch
                {
                    "max" => "subgroupMax",
                    "min" => "subgroupMin",
                    "add" => "subgroupAdd",
                    "and" => "subgroupAnd",
                    "or"  => "subgroupOr",
                    "xor" => "subgroupXor",
                    _ => null,
                };
            }
            catch
            {
                return null;
            }
        }

        /// <summary>
        /// Determines the effective WGSL element type for a reduction based on the TReduction
        /// generic type argument (e.g., MaxUInt32 → "u32", MaxFloat32 → "f32").
        /// ILGPU's IR doesn't distinguish i32/u32, so we must inspect TReduction to get
        /// the correct WGSL type for subgroup and atomic operations that are type-sensitive.
        /// Returns null if the type cannot be determined (caller should fall back to TypeGenerator).
        /// </summary>
        private string? GetReductionWgslType(MethodInfo? sourceMethod)
        {
            if (sourceMethod == null || !sourceMethod.IsGenericMethod) return null;
            try
            {
                var genericArgs = sourceMethod.GetGenericArguments();
                if (genericArgs.Length < 2) return null;
                var tReductionName = genericArgs[1].Name; // e.g. "MaxUInt32", "MinFloat32", "AddInt64"

                if (tReductionName.Contains("UInt32")) return "u32";
                if (tReductionName.Contains("Int32")) return "i32";
                if (tReductionName.Contains("Float32")) return "f32";
                if (tReductionName.Contains("UInt64"))
                    return Backend.Options.EnableI64Emulation ? "emu_u64" : "u32";
                if (tReductionName.Contains("Int64"))
                    return Backend.Options.EnableI64Emulation ? "emu_i64" : "i32";
                if (tReductionName.Contains("Float64") || tReductionName.Contains("Double"))
                    return Backend.Options.EnableF64Emulation ? "emu_f64" : "f32";

                return null;
            }
            catch
            {
                return null;
            }
        }

        /// <summary>
        /// Returns the correct accumulation expression for reduce operations.
        /// For 64-bit emulated types, uses emulated math helpers (i64_max, f64_add, etc.)
        /// instead of WGSL built-ins which would do incorrect component-wise operations on vec2.
        /// </summary>
        private static string GetEmulated64BitAccumExpr(bool is64Bit, string wgslType, string wgslAccumOp, string accumVar, string elemExpr)
        {
            if (!is64Bit)
            {
                // Standard 32-bit path
                if (wgslAccumOp == "+")
                    return $"{accumVar} + {elemExpr}";
                else
                    return $"{wgslAccumOp}({accumVar}, {elemExpr})";
            }

            // 64-bit emulated path: use the emulated helper functions
            // wgslType is one of: emu_i64, emu_u64, emu_f64
            string prefix = wgslType switch
            {
                "emu_f64" => "f64",
                "emu_u64" => "u64",
                "emu_i64" => "i64",
                _ => "i64"
            };

            if (wgslAccumOp == "+")
                return $"{prefix}_add({accumVar}, {elemExpr})";
            else if (wgslAccumOp == "max")
                return $"{prefix}_max({accumVar}, {elemExpr})";
            else if (wgslAccumOp == "min")
                return $"{prefix}_min({accumVar}, {elemExpr})";
            else
                return $"{prefix}_add({accumVar}, {elemExpr})"; // fallback to add
        }

        #endregion
    }
}
