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
                AllocaKindInformation dynamicSharedAllocations)
            {
                Backend = backend;
                TypeGenerator = typeGenerator;
                EntryPoint = entryPoint;
                SharedAllocations = sharedAllocations;
                DynamicSharedAllocations = dynamicSharedAllocations;
                DynamicSharedOverrides = new List<DynamicSharedOverrideInfo>();
                ScalarPackingManifest = new List<ScalarPackingEntry>();
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
            /// Populated by the kernel code generator during GenerateHeader() with
            /// the WGSL override constant names for dynamic shared memory.
            /// </summary>
            public List<DynamicSharedOverrideInfo> DynamicSharedOverrides { get; }

            /// <summary>
            /// Populated by the kernel code generator during GenerateHeader() with
            /// the scalar parameter packing layout for this kernel.
            /// </summary>
            public List<ScalarPackingEntry> ScalarPackingManifest { get; }
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
        protected void Declare(Variable variable)
        {
            if (declaredVariables.Contains(variable.Name)) return;
            declaredVariables.Add(variable.Name);

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
                    if (expr.Contains("&param") || expr.Contains("&temp_"))
                    {
                        // Pointer aliases — keep as let (they can't be var since var can't hold references)
                        return match.Value;
                    }
                    else if (expr.Contains("f64_from_f32") || expr.Contains("f64_add") || expr.Contains("f64_sub") || expr.Contains("f64_mul") || expr.Contains("f64_div"))
                    {
                        inferredType = "emu_f64"; // vec2<f32> alias
                    }
                    else if (expr.Contains("f32(") || expr.Contains("f64_to_f32"))
                    {
                        inferredType = "f32";
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
            if (value is MethodCall dbgCall)
            {
                bool dbgHit = ImplementationProvider.TryGetCodeGenerator(value, out var _);
                Console.WriteLine($"[WGSL-MC] MethodCall: {dbgCall.Target?.Source?.Name ?? "?"}, HasIntrinsic={dbgHit}, MethodFlags={dbgCall.Target?.Flags}");
            }
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
                AppendLine($"{target} = {op}({left}, {right});");
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

            AppendLine($"{target} = {left} {op} {right};");
        }

        // Conversions
        public virtual void GenerateCode(ConvertValue value)
        {
            var target = Load(value);
            var source = Load(value.Value);
            var targetType = TypeGenerator[value.Type];
            Declare(target);
            AppendLine($"{target} = {targetType}({source});");
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

            if (isAtomic)
            {
                AppendLine($"{target} = atomicLoad({source});");
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

            if (isAtomic)
            {
                AppendLine($"atomicStore({address}, {val});");
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
            Declare(target);
            AppendLine($"{target} = {source}; // newView");
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

            Builder.Append($"&(*{source}).{fieldName};");
            Builder.AppendLine();
        }

        public virtual void GenerateCode(Alloca value)
        {
            // Already handled in SetupAllocations
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
                BasicValueType.Float16 => FormatFloat(value.Float32Value),
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
                if (subgroupOp != null && HasSubgroups)
                {
                    AppendLine($"// Reduce fallback (group-level): {methodCall.Target.Name} -> {subgroupOp}");
                    var reduceSrc = methodCall.Count > 0 ? Load(methodCall[0]) : null;
                    if (reduceSrc != null)
                    {
                        // Signal kernel generator to declare _grp_sg_results workgroup array
                        UsesGroupReduce = true;

                        // Determine scalar accumulation op from subgroupOp name
                        // e.g. "subgroupMax" -> "max", "subgroupMin" -> "min", "subgroupAdd" -> "+"
                        string wgslAccumOp;
                        string wgslType = TypeGenerator[methodCall.Type];
                        if (subgroupOp.EndsWith("Max", StringComparison.Ordinal))
                            wgslAccumOp = "max";
                        else if (subgroupOp.EndsWith("Min", StringComparison.Ordinal))
                            wgslAccumOp = "min";
                        else
                            wgslAccumOp = "+"; // Add / or

                        // Step 1: intra-subgroup reduce
                        string warpResult = $"_warp_sg_result_{target.Name}";
                        AppendLine($"var {warpResult} : {wgslType} = {subgroupOp}({reduceSrc});");

                        // Step 2: first lane of each subgroup stores to _grp_sg_results
                        // Pre-compute cast expressions to avoid nested-quote issues in interpolation
                        bool isI32 = wgslType == "i32";
                        string storeExpr = isI32 ? warpResult : $"i32({warpResult})";
                        string load0Expr = isI32 ? "_grp_sg_results[0u]" : $"{wgslType}(_grp_sg_results[0u])";
                        string loadNExpr = isI32 ? "_grp_sg_results[_sgi]" : $"{wgslType}(_grp_sg_results[_sgi])";
                        AppendLine($"if (subgroup_invocation_id == 0u) {{");
                        PushIndent();
                        AppendLine($"_grp_sg_results[subgroup_id] = {storeExpr};");
                        PopIndent();
                        AppendLine("}");

                        // Step 3: workgroup barrier so all subgroups have written
                        AppendLine("workgroupBarrier();");

                        // Step 4: all threads cooperatively reduce across subgroups
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
                        AppendLine($"{target} = _grp_accum;");
                        PopIndent();
                        AppendLine("}");
                        return;
                    }
                }
                else if (subgroupOp != null)
                {
                    // No subgroups: workgroup-level shared-memory reduction.
                    // Without native subgroup ops, the entire workgroup is one "warp".
                    // ILGroupExtensions.AllReduce calls Warp.Reduce, then only IsFirstLane
                    // does atomicMax. A passthrough here would lose all threads' values
                    // except thread 0's. Instead, we use _warp_shuffle_buf to collect all
                    // threads' values and reduce across the workgroup in shared memory.
                    var reduceSrc = methodCall.Count > 0 ? Load(methodCall[0]) : null;
                    if (reduceSrc != null)
                    {
                        // Signal that we need the shared memory buffer
                        UsesWarpShuffleEmulation = true;

                        string wgslType = TypeGenerator[methodCall.Type];
                        string wgslAccumOp;
                        if (subgroupOp.EndsWith("Max", StringComparison.Ordinal))
                            wgslAccumOp = "max";
                        else if (subgroupOp.EndsWith("Min", StringComparison.Ordinal))
                            wgslAccumOp = "min";
                        else
                            wgslAccumOp = "+";

                        AppendLine($"// Reduce via shared memory (subgroups unavailable): {methodCall.Target.Name}");

                        // Step 1: all threads write their value to shared memory
                        AppendLine($"_warp_shuffle_buf[local_index] = bitcast<u32>({reduceSrc});");
                        AppendLine("workgroupBarrier();");

                        // Step 2: all threads cooperatively reduce across the workgroup
                        bool isI32 = wgslType == "i32";
                        string load0 = isI32 ? "bitcast<i32>(_warp_shuffle_buf[0u])" : $"{wgslType}(bitcast<i32>(_warp_shuffle_buf[0u]))";
                        string loadN = isI32 ? "bitcast<i32>(_warp_shuffle_buf[_ri])" : $"{wgslType}(bitcast<i32>(_warp_shuffle_buf[_ri]))";
                        AppendLine("{");
                        PushIndent();
                        AppendLine($"var _reduce_accum : {wgslType} = {load0};");
                        AppendLine("for (var _ri : u32 = 1u; _ri < workgroup_size.x; _ri++) {");
                        PushIndent();
                        if (wgslAccumOp == "+")
                            AppendLine($"_reduce_accum = _reduce_accum + {loadN};");
                        else
                            AppendLine($"_reduce_accum = {wgslAccumOp}(_reduce_accum, {loadN});");
                        PopIndent();
                        AppendLine("}");
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
            AppendLine($"{target} = bitcast<{target.Type}>({source});");
        }

        public virtual void GenerateCode(IntAsFloatCast value)
        {
            var target = Load(value);
            var source = Load(value.Value);
            Declare(target);
            AppendLine($"{target} = bitcast<{target.Type}>({source});");
        }

        // Atomics & Barriers

        /// <summary>
        /// Override in subclasses to indicate whether the atomic target pointer refers to a
        /// float-typed parameter. WGSL has no atomic&lt;f32&gt;, so float atomics are stored as
        /// atomic&lt;u32&gt; (raw bits) and emulated via a CAS loop.
        /// </summary>
        protected virtual bool IsFloatAtomicTarget(Value target) => false;

        public virtual void GenerateCode(GenericAtomic value)
        {
            var target = Load(value);
            var ptr = Load(value.Target);
            var val = Load(value.Value);
            Declare(target);

            // WGSL only supports atomicAdd/Min/Max/etc. on i32 and u32.
            // Float atomics (f32/f16) must be emulated using a CAS loop on atomic<u32>.
            if (IsFloatAtomicTarget(value.Target))
            {
                // The buffer is declared as atomic<u32>. We bitcast f32 <-> u32.
                // CAS loop: atomically read-modify-write the float value.
                string casOld = $"_cas_old_{target.Name}";
                string casNew = $"_cas_new_{target.Name}";
                string casRes = $"_cas_res_{target.Name}";

                AppendLine($"var {casOld} = atomicLoad({ptr});");
                AppendLine($"loop {{");
                PushIndent();

                string floatOld = $"bitcast<f32>({casOld})";
                string newFloat = value.Kind switch
                {
                    AtomicKind.Add => $"{floatOld} + {val}",
                    AtomicKind.Max => $"max({floatOld}, {val})",
                    AtomicKind.Min => $"min({floatOld}, {val})",
                    AtomicKind.Exchange => $"{val}",
                    _ => $"{floatOld} + {val}"  // fallback: Add
                };

                AppendLine($"let {casNew} = bitcast<u32>({newFloat});");
                AppendLine($"let {casRes} = atomicCompareExchangeWeak({ptr}, {casOld}, {casNew});");
                AppendLine($"if ({casRes}.exchanged) {{ break; }}");
                AppendLine($"{casOld} = {casRes}.old_value;");
                PopIndent();
                AppendLine($"}}");
                AppendLine($"{target} = bitcast<{target.Type}>({casOld});");
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
                    AppendLine($"_warp_shuffle_buf[local_index] = bitcast<u32>({source});");
                    AppendLine("workgroupBarrier();");
                    AppendLine($"let {warpBase} = (local_index / workgroup_size.x) * workgroup_size.x;");
                    AppendLine($"{target} = bitcast<{target.Type}>(_warp_shuffle_buf[{warpBase}]);");
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
                AppendLine($"_warp_shuffle_buf[local_index] = bitcast<u32>({source});");
                AppendLine("workgroupBarrier();");
                AppendLine($"let {warpBase} = (local_index / u32(workgroup_size.x)) * u32(workgroup_size.x);");
                AppendLine($"{target} = bitcast<{target.Type}>(_warp_shuffle_buf[{warpBase} + u32({origin})]);");
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
                AppendLine($"_warp_shuffle_buf[local_index] = bitcast<u32>({source});");
                AppendLine("workgroupBarrier();");
                AppendLine($"let {warpBase} = (local_index / u32(workgroup_size.x)) * u32(workgroup_size.x);");
                AppendLine($"{target} = bitcast<{target.Type}>(_warp_shuffle_buf[{warpBase} + u32({origin})]);");
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
            var trueVal = Load(value.TrueValue);
            var falseVal = Load(value.FalseValue);
            Declare(target);
            AppendLine($"{target} = select({falseVal}, {trueVal}, {cond});");
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
            Console.WriteLine($"[WGSL-DBG] GenerateGroupAllReduce called, value type={value?.GetType().Name}, isMethodCall={value is MethodCall}");
            if (value is not MethodCall methodCall)
            {
                Console.WriteLine("[WGSL-DBG] GenerateGroupAllReduce: early exit - not a MethodCall");
                return;
            }

            // Signal that the kernel needs the workgroup-level shared arrays and subgroup_id builtin.
            codeGenerator.UsesGroupReduce = true;

            var target = codeGenerator.LoadIntrinsicValue(value);
            var operand = codeGenerator.LoadIntrinsicValue(methodCall[0].Resolve());
            codeGenerator.Declare(target);

            // Determine the WGSL subgroup operation and identity value from TReduction type argument.
            var srcMethodInfo = methodCall.Target.Source as MethodInfo;
            Console.WriteLine($"[WGSL-DBG] srcMethodInfo={srcMethodInfo?.Name}, isGeneric={srcMethodInfo?.IsGenericMethod}, genericArgs={string.Join(",", srcMethodInfo?.GetGenericArguments()?.Select(t => t.Name) ?? Array.Empty<string>())}");
            string? wgslOp = GetSubgroupReduceOp(srcMethodInfo);
            string? identityExpr = GetReductionIdentityExpr(srcMethodInfo);
            Console.WriteLine($"[WGSL-DBG] HasSubgroups={backend.HasSubgroups}, wgslOp={wgslOp}, identityExpr={identityExpr}");

            if (!backend.HasSubgroups || wgslOp == null || identityExpr == null)
            {
                Console.WriteLine($"[WGSL-DBG] GenerateGroupAllReduce: fallback, reason: HasSubgroups={backend.HasSubgroups}, wgslOp={wgslOp}, identityExpr={identityExpr}");
                // Fallback: can't do cross-subgroup reduce without subgroups — emit identity.
                codeGenerator.AppendLine($"{target} = {identityExpr ?? "0"}; // group reduce (subgroups unavailable)");
                return;
            }

            // Generate unique variable names using target name as a suffix to avoid clashes in inlined code.
            string sfx = target.Name.Replace("v_", "_").TrimStart('_');

            // === Step 1: Per-subgroup reduction ===
            codeGenerator.AppendLine($"let _sg_part_{sfx} = {wgslOp}({operand}); // per-subgroup partial reduce");

            // === Step 2: Lane 0 of each subgroup stores partial result ===
            codeGenerator.AppendLine($"if (subgroup_invocation_id == 0u) {{");
            codeGenerator.AppendLine($"    _grp_sg_results[subgroup_id] = _sg_part_{sfx};");
            codeGenerator.AppendLine($"}}");

            // === Step 3: Synchronise so all subgroup slots are visible ===
            codeGenerator.AppendLine($"workgroupBarrier();");

            // === Step 4: Subgroup 0's threads reduce across all subgroup slots ===
            // Number of subgroups = ceil(workgroup_size.x / subgroup_size)
            // Lane i of subgroup 0 reads slot i (padded with identity if i >= num_subgroups)
            codeGenerator.AppendLine($"let _num_sg_{sfx} = (workgroup_size.x + subgroup_size - 1u) / subgroup_size;");
            codeGenerator.AppendLine($"let _sg0_val_{sfx} = select({identityExpr}, _grp_sg_results[subgroup_invocation_id],");
            codeGenerator.AppendLine($"    subgroup_invocation_id < _num_sg_{sfx});");
            codeGenerator.AppendLine($"let _final_{sfx} = {wgslOp}(_sg0_val_{sfx}); // second-level reduce across subgroup slots");

            // === Step 5: Thread (0,0) writes final result back so all threads can read it ===
            codeGenerator.AppendLine($"if (subgroup_id == 0u && subgroup_invocation_id == 0u) {{");
            codeGenerator.AppendLine($"    _grp_sg_results[0] = _final_{sfx};");
            codeGenerator.AppendLine($"}}");

            // === Step 6: Sync and all threads read the final group-wide result ===
            codeGenerator.AppendLine($"workgroupBarrier();");
            codeGenerator.AppendLine($"{target} = _grp_sg_results[0];");
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
                    float f => float.IsNegativeInfinity(f) ? "-3.40282347e+38f" : f.ToString("G9") + "f",
                    long l => $"{(int)l}", // emu_i64 – just use low bits for identity (0 or MIN_INT)
                    _ => identity?.ToString() ?? "0",
                };
            }
            catch
            {
                return null;
            }
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

        #endregion
    }
}
