// ---------------------------------------------------------------------------------------
//                               SpawnDev.ILGPU.Workers
//                 Web Worker Compute Library for Blazor WebAssembly
//
// File: JSCodeGenerator.cs
//
// Base JavaScript code generator implementing IBackendCodeGenerator.
// Translates ILGPU IR values into JavaScript source code.
// ---------------------------------------------------------------------------------------

using global::ILGPU;
using global::ILGPU.Backends;
using global::ILGPU.Backends.EntryPoints;
using global::ILGPU.IR;
using global::ILGPU.IR.Analyses;
using global::ILGPU.IR.Types;
using global::ILGPU.IR.Values;
using System.Text;

namespace SpawnDev.ILGPU.Workers.Backend
{
    /// <summary>
    /// Base class for JavaScript code generation. Generates JavaScript source
    /// from ILGPU IR values by implementing the IBackendCodeGenerator interface.
    /// </summary>
    public abstract partial class JSCodeGenerator : IBackendCodeGenerator<StringBuilder>
    {
        #region Nested Types

        /// <summary>
        /// Generation arguments for JS code generator construction.
        /// </summary>
        public readonly struct GeneratorArgs
        {
            /// <summary>
            /// Creates new generator args.
            /// </summary>
            public GeneratorArgs(
                WorkersBackend backend,
                JSTypeGenerator typeGenerator,
                EntryPoint entryPoint,
                AllocaKindInformation sharedAllocations,
                AllocaKindInformation dynamicSharedAllocations)
            {
                Backend = backend;
                TypeGenerator = typeGenerator;
                EntryPoint = entryPoint;
                SharedAllocations = sharedAllocations;
                DynamicSharedAllocations = dynamicSharedAllocations;
            }

            /// <summary>The parent backend.</summary>
            public WorkersBackend Backend { get; }

            /// <summary>The type generator.</summary>
            public JSTypeGenerator TypeGenerator { get; }

            /// <summary>The kernel entry point.</summary>
            public EntryPoint EntryPoint { get; }

            /// <summary>Shared memory allocations.</summary>
            public AllocaKindInformation SharedAllocations { get; }

            /// <summary>Dynamic shared memory allocations.</summary>
            public AllocaKindInformation DynamicSharedAllocations { get; }
        }

        /// <summary>
        /// Represents a variable in JavaScript code.
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

        /// <summary>
        /// Tracks element addresses for JS (no pointer semantics).
        /// Maps a variable name to its (arrayVar, indexVar) so Store/Load can
        /// emit correct array[index] = value / value = array[index].
        /// </summary>
        protected readonly Dictionary<string, (Variable Array, Variable Index)> _elementAddresses = new();

        // Flag to track if we are generating code within the state machine loop
        protected bool IsStateMachineActive { get; set; } = false;

        private StringBuilder prefixBuilder = new StringBuilder();

        /// <summary>
        /// Constructs a new JS code generator.
        /// </summary>
        protected JSCodeGenerator(in GeneratorArgs args, Method method, Allocas allocas)
        {
            Backend = args.Backend;
            Method = method;
            Allocas = allocas;
            Builder = prefixBuilder;
        }

        #endregion

        #region Properties

        /// <summary>The parent backend.</summary>
        public WorkersBackend Backend { get; }

        /// <summary>The current method being generated.</summary>
        public Method Method { get; }

        /// <summary>All local allocas.</summary>
        public Allocas Allocas { get; }

        /// <summary>The current string builder.</summary>
        public StringBuilder Builder { get; protected set; }

        /// <summary>Builder for variable declarations.</summary>
        public StringBuilder VariableBuilder { get; } = new StringBuilder();

        /// <summary>Current indentation level.</summary>
        protected int IndentLevel { get; set; } = 0;

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
        /// Generates constant declarations (not used for JS).
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
            var type = GetJSType(value.Type);
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
            var jsType = GetJSType(type);
            return new Variable(name, jsType);
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

        /// <summary>
        /// Public alias used by intrinsic code generators.
        /// </summary>
        public Variable LoadVariable(Value value) => Load(value);

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
        /// In JS, we simply emit "let varName = defaultValue;".
        /// </summary>
        protected void Declare(Variable variable)
        {
            if (declaredVariables.Contains(variable.Name)) return;
            declaredVariables.Add(variable.Name);

            if (IsStateMachineActive)
            {
                // When in state machine mode, defer declarations to function scope
                VariableBuilder.Append("  let ");
                VariableBuilder.Append(variable.Name);
                VariableBuilder.AppendLine(";");
            }
            else
            {
                AppendLine($"let {variable.Name};");
            }
        }

        #endregion

        #region Type Helpers

        /// <summary>
        /// Gets a JS-friendly type name for an ILGPU type.
        /// </summary>
        protected string GetJSType(TypeNode type)
        {
            if (type is PrimitiveType pt)
            {
                return pt.BasicValueType switch
                {
                    BasicValueType.Int1 => "bool",
                    BasicValueType.Int8 or BasicValueType.Int16 or BasicValueType.Int32 => "i32",
                    BasicValueType.Int64 => "i64",
                    BasicValueType.Float16 or BasicValueType.Float32 => "f32",
                    BasicValueType.Float64 => "f64",
                    _ => "i32"
                };
            }
            return "i32";
        }

        /// <summary>
        /// Checks if a type represents an ILGPU Index type (Index1D, Index2D, etc.)
        /// </summary>
        protected bool IsIndexType(TypeNode type)
        {
            var typeStr = type.ToString();
            return typeStr.Contains("Index1D") || typeStr.Contains("Index2D") || typeStr.Contains("Index3D");
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

        #endregion

        #region Code Emission Helpers

        /// <summary>
        /// Appends indentation to the builder.
        /// </summary>
        protected void AppendIndent()
        {
            for (int i = 0; i < IndentLevel; i++)
                Builder.Append("  ");
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
        public void AppendLine(string line)
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

            // Simple case: single block, no control flow
            if (blocks.Count == 1)
            {
                IsStateMachineActive = false;
                foreach (var valueEntry in blocks.First())
                {
                    GenerateCodeFor(valueEntry.Value);
                }
                // The Terminator (branch/return) is stored separately from the block's values
                var firstBlock = blocks.First();
                if (firstBlock.Terminator != null)
                    GenerateCodeFor(firstBlock.Terminator);
                return;
            }

            // Multiple blocks: use loop/switch pattern (state machine)
            IsStateMachineActive = true;

            // Declare return variable if needed
            if (hasReturnValue)
            {
                AppendLine("let _return_val;");
            }

            // Save position before loop - deferred variable declarations will be inserted here
            int deferredInsertPosition = Builder.Length;

            AppendLine("let _block = 0;");
            AppendLine("_loop: while (true) {");
            PushIndent();
            AppendLine("switch (_block) {");
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

                // The Terminator (branch/return) is stored separately from the block's values
                if (block.Terminator != null)
                    GenerateCodeFor(block.Terminator);

                PopIndent();
                AppendLine("}");
                AppendLine("break;");
                blockIndex++;
            }

            AppendLine("default: break _loop;");
            PopIndent();
            AppendLine("}"); // end switch

            PopIndent();
            AppendLine("}"); // end while

            // Insert all deferred variable declarations at the function scope
            if (VariableBuilder.Length > 0)
            {
                Builder.Insert(deferredInsertPosition, VariableBuilder.ToString());
            }

            // Emit final return
            if (hasReturnValue)
            {
                AppendLine("return _return_val;");
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

                if (allocaInfo.IsArray)
                {
                    AppendLine($"let {variable.Name} = new Array({allocaInfo.ArraySize}).fill(0);");
                }
                else
                {
                    AppendLine($"let {variable.Name} = 0;");
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

            WorkersBackend.Log($"[JS] Generating code for: {value.GetType().FullName} - {value}");

            // Handle inaccessible Throw instruction
            if (value.GetType().Name.Contains("Throw"))
            {
                WorkersBackend.Log($"[JS] HANDLING THROW: {value}");
                AppendLine("// throw (unsupported in workers kernel)");
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

                // Alignment
                case global::ILGPU.IR.Values.AlignTo v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.AsAligned v:
                    GenerateCode(v);
                    break;

                // Warp/Group operations
                case global::ILGPU.IR.Values.Broadcast v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.WarpShuffle v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.SubWarpShuffle v:
                    GenerateCode(v);
                    break;

                // Dynamic memory
                case global::ILGPU.IR.Values.DynamicMemoryLengthValue v:
                    GenerateCode(v);
                    break;

                // Language emit
                case global::ILGPU.IR.Values.LanguageEmitValue v:
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

            // Determine if this is an integer operation
            bool isIntegerOp = value.BasicValueType is BasicValueType.Int8 or BasicValueType.Int16
                or BasicValueType.Int32 or BasicValueType.Int64;

            Declare(target);

            // Handle function-call-style binary operations first
            switch (value.Kind)
            {
                case BinaryArithmeticKind.Min:
                    AppendLine($"{target} = Math.min({left}, {right});");
                    return;
                case BinaryArithmeticKind.Max:
                    AppendLine($"{target} = Math.max({left}, {right});");
                    return;
                case BinaryArithmeticKind.Atan2F:
                    AppendLine($"{target} = Math.atan2({left}, {right});");
                    return;
                case BinaryArithmeticKind.PowF:
                    AppendLine($"{target} = Math.pow({left}, {right});");
                    return;
            }

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
                BinaryArithmeticKind.Rem => "%",
                _ => "+"
            };

            if (op == "/" && isIntegerOp)
            {
                // JS division always produces float; truncate for integer semantics
                AppendLine($"{target} = Math.trunc({left} / {right});");
            }
            else
            {
                AppendLine($"{target} = ({left} {op} {right});");
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

            string result = value.Kind switch
            {
                UnaryArithmeticKind.Neg => $"-{operand}",
                UnaryArithmeticKind.Not => value.Value.Type is PrimitiveType pt && pt.BasicValueType == BasicValueType.Int1
                    ? $"!{operand}"
                    : $"~{operand}",

                // Math Intrinsics
                UnaryArithmeticKind.Abs => $"Math.abs({operand})",
                UnaryArithmeticKind.SinF => $"Math.sin({operand})",
                UnaryArithmeticKind.CosF => $"Math.cos({operand})",
                UnaryArithmeticKind.TanF => $"Math.tan({operand})",
                UnaryArithmeticKind.AsinF => $"Math.asin({operand})",
                UnaryArithmeticKind.AcosF => $"Math.acos({operand})",
                UnaryArithmeticKind.AtanF => $"Math.atan({operand})",
                UnaryArithmeticKind.SinhF => $"Math.sinh({operand})",
                UnaryArithmeticKind.CoshF => $"Math.cosh({operand})",
                UnaryArithmeticKind.TanhF => $"Math.tanh({operand})",
                UnaryArithmeticKind.ExpF => $"Math.exp({operand})",
                UnaryArithmeticKind.Exp2F => $"Math.pow(2, {operand})",
                UnaryArithmeticKind.LogF => $"Math.log({operand})",
                UnaryArithmeticKind.Log2F => $"Math.log2({operand})",
                UnaryArithmeticKind.SqrtF => $"Math.sqrt({operand})",
                UnaryArithmeticKind.RsqrtF => $"(1.0 / Math.sqrt({operand}))",
                UnaryArithmeticKind.RcpF => $"(1.0 / {operand})",
                UnaryArithmeticKind.FloorF => $"Math.floor({operand})",
                UnaryArithmeticKind.CeilingF => $"Math.ceil({operand})",

                _ => $"/* unhandled unary: {value.Kind} */ {operand}"
            };

            AppendLine($"{target} = {result};");
        }

        public virtual void GenerateCode(TernaryArithmeticValue value)
        {
            var target = Load(value);
            var first = Load(value.First);
            var second = Load(value.Second);
            var third = Load(value.Third);
            Declare(target);

            // FMA: a * b + c
            AppendLine($"{target} = ({first} * {second} + {third});");
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
                CompareKind.Equal => "===",
                CompareKind.NotEqual => "!==",
                CompareKind.LessThan => "<",
                CompareKind.LessEqual => "<=",
                CompareKind.GreaterThan => ">",
                CompareKind.GreaterEqual => ">=",
                _ => "==="
            };

            AppendLine($"{target} = ({left} {op} {right});");
        }

        // Conversions
        public virtual void GenerateCode(ConvertValue value)
        {
            var target = Load(value);
            var source = Load(value.Value);
            Declare(target);

            var sourceType = value.Value.Type is PrimitiveType spt ? spt.BasicValueType : BasicValueType.Int32;
            var targetType = value.Type is PrimitiveType tpt ? tpt.BasicValueType : BasicValueType.Int32;

            var expr = JSTypeGenerator.GetConversionExpression(sourceType, targetType, source.Name);
            AppendLine($"{target} = {expr};");
        }

        // Memory Operations
        public virtual void GenerateCode(global::ILGPU.IR.Values.Load loadVal)
        {
            var target = Load(loadVal);
            var source = Load(loadVal.Source);
            Declare(target);

            // Check if the source is an element address (array[index] reference)
            if (_elementAddresses.TryGetValue(source.Name, out var elemAddr))
            {
                // Read from array: target = array[index]
                AppendLine($"{target} = {elemAddr.Array}[{elemAddr.Index}];");
            }
            // Check if the source is a field address (struct.field reference)
            else if (_fieldAddresses.TryGetValue(source.Name, out var fieldAddr))
            {
                // Read from struct field: target = struct.f{index}
                AppendLine($"{target} = {fieldAddr.Struct}.f{fieldAddr.FieldIndex};");
            }
            else
            {
                // Simple variable read
                AppendLine($"{target} = {source};");
            }
        }

        public virtual void GenerateCode(global::ILGPU.IR.Values.Store storeVal)
        {
            var address = Load(storeVal.Target);
            var val = Load(storeVal.Value);

            // Check if the target is an element address (array[index] reference)
            if (_elementAddresses.TryGetValue(address.Name, out var elemAddr))
            {
                // Write to array: array[index] = value
                AppendLine($"{elemAddr.Array}[{elemAddr.Index}] = {val};");
            }
            // Check if the target is a field address (struct.field reference)
            else if (_fieldAddresses.TryGetValue(address.Name, out var fieldAddr))
            {
                // Write to struct field: struct.f{index} = value
                AppendLine($"{fieldAddr.Struct}.f{fieldAddr.FieldIndex} = {val};");
            }
            else
            {
                // Simple variable write
                AppendLine($"{address} = {val};");
            }
        }

        public virtual void GenerateCode(LoadElementAddress value)
        {
            var target = Load(value);
            var source = Load(value.Source);
            var offset = Load(value.Offset);

            // JS has no pointer semantics — we track the array+index pair
            // so that subsequent Load/Store can resolve to array[index]
            _elementAddresses[target.Name] = (source, offset);

            // Don't emit any code — the actual read/write happens in Load/Store
            WorkersBackend.Log($"[JS] LoadElementAddress: {target.Name} => {source}[{offset}]");
        }

        /// <summary>
        /// Tracks field addresses for JS struct field access.
        /// Maps a variable name to its (structVar, fieldIndex) so Store/Load can
        /// emit correct struct.f{index} = value / value = struct.f{index}.
        /// </summary>
        protected readonly Dictionary<string, (Variable Struct, int FieldIndex)> _fieldAddresses = new();

        public virtual void GenerateCode(LoadFieldAddress value)
        {
            var target = Load(value);
            var source = Load(value.Source);
            int fieldIndex = value.FieldSpan.Index;

            // Track the field address for subsequent Load/Store
            _fieldAddresses[target.Name] = (source, fieldIndex);

            // Don't emit code — reads/writes happen in Load/Store
            WorkersBackend.Log($"[JS] LoadFieldAddress: {target.Name} => {source}.f{fieldIndex}");
        }

        public virtual void GenerateCode(Alloca value)
        {
            // Already handled in SetupAllocations
        }

        public virtual void GenerateCode(global::ILGPU.IR.Values.NewView value)
        {
            var target = Load(value);
            var source = Load(value.Pointer);
            Declare(target);
            AppendLine($"{target} = {source}; // newView");
        }

        // Constants
        public virtual void GenerateCode(PrimitiveValue value)
        {
            var target = Load(value);
            Declare(target);

            string valStr = value.BasicValueType switch
            {
                BasicValueType.Int1 => value.Int1Value ? "true" : "false",
                BasicValueType.Int8 => value.Int8Value.ToString(),
                BasicValueType.Int16 => value.Int16Value.ToString(),
                BasicValueType.Int32 => value.Int32Value.ToString(),
                BasicValueType.Int64 => $"BigInt({value.Int64Value})",
                BasicValueType.Float16 => ((float)value.Float16Value).ToString("G9"),
                BasicValueType.Float32 => value.Float32Value.ToString("G9"),
                BasicValueType.Float64 => value.Float64Value.ToString("G17"),
                _ => "0"
            };

            AppendLine($"{target} = {valStr};");
        }

        public virtual void GenerateCode(NullValue value)
        {
            var target = Load(value);
            Declare(target);
            AppendLine($"{target} = 0;");
        }

        public virtual void GenerateCode(StringValue value)
        {
            var target = Load(value);
            Declare(target);
            AppendLine($"{target} = \"{value.String}\";");
        }

        // Phi values
        public virtual void GenerateCode(PhiValue value)
        {
            // Phi nodes are handled by pushing values at branch points
            var target = Load(value);
            Declare(target);
        }

        // Structures
        public virtual void GenerateCode(StructureValue value)
        {
            var target = Load(value);
            Declare(target);
            // In JS, structures become objects or arrays
            AppendLine($"{target} = {{}};");
            for (int i = 0; i < value.Count; i++)
            {
                var field = Load(value[i]);
                AppendLine($"{target}.f{i} = {field};");
            }
        }

        public virtual void GenerateCode(GetField value)
        {
            var target = Load(value);
            var source = Load(value.ObjectValue);
            int fieldIndex = value.FieldSpan.Index;
            Declare(target);
            AppendLine($"{target} = {source}.f{fieldIndex};");
        }

        public virtual void GenerateCode(SetField value)
        {
            var target = Load(value);
            var source = Load(value.ObjectValue);
            var val = Load(value.Value);
            int fieldIndex = value.FieldSpan.Index;
            Declare(target);
            // Copy all fields then update the target field
            AppendLine($"{target} = Object.assign({{}}, {source});");
            AppendLine($"{target}.f{fieldIndex} = {val};");
        }

        // Device Constants
        public virtual void GenerateCode(GridIndexValue value)
        {
            var target = Load(value);
            Declare(target);
            // Will be overridden in JSKernelFunctionGenerator
            AppendLine($"{target} = _gridIndex;");
        }

        public virtual void GenerateCode(GroupIndexValue value)
        {
            var target = Load(value);
            Declare(target);
            AppendLine($"{target} = _groupIndex;");
        }

        public virtual void GenerateCode(GridDimensionValue value)
        {
            var target = Load(value);
            Declare(target);
            AppendLine($"{target} = _gridDim;");
        }

        public virtual void GenerateCode(GroupDimensionValue value)
        {
            var target = Load(value);
            Declare(target);
            AppendLine($"{target} = _groupDim;");
        }

        public virtual void GenerateCode(WarpSizeValue value)
        {
            var target = Load(value);
            Declare(target);
            AppendLine($"{target} = 1;"); // Single-threaded per worker
        }

        public virtual void GenerateCode(LaneIdxValue value)
        {
            var target = Load(value);
            Declare(target);
            AppendLine($"{target} = 0;"); // Lane 0 always
        }

        // Control Flow
        public virtual void GenerateCode(ReturnTerminator value)
        {
            if (value.IsVoidReturn)
            {
                if (IsStateMachineActive)
                {
                    AppendLine("_block = -1; continue;");
                }
                else
                {
                    AppendLine("return;");
                }
            }
            else
            {
                var retVal = Load(value.ReturnValue);
                if (IsStateMachineActive)
                {
                    AppendLine($"_return_val = {retVal};");
                    AppendLine("_block = -1; continue;");
                }
                else
                {
                    AppendLine($"return {retVal};");
                }
            }
        }

        public virtual void GenerateCode(UnconditionalBranch value)
        {
            // Push phi values before branching
            PushPhiValues(value);

            int targetBlock = GetBlockIndex(value.Target);
            AppendLine($"_block = {targetBlock}; continue;");
        }

        public virtual void GenerateCode(IfBranch value)
        {
            var condition = Load(value.Condition);

            // Push phi values before branching
            PushPhiValues(value);

            int trueBlock = GetBlockIndex(value.TrueTarget);
            int falseBlock = GetBlockIndex(value.FalseTarget);

            AppendLine($"if ({condition}) {{");
            PushIndent();
            AppendLine($"_block = {trueBlock}; continue;");
            PopIndent();
            AppendLine("} else {");
            PushIndent();
            AppendLine($"_block = {falseBlock}; continue;");
            PopIndent();
            AppendLine("}");
        }

        public virtual void GenerateCode(SwitchBranch value)
        {
            var condition = Load(value.Condition);
            PushPhiValues(value);

            AppendLine($"switch ({condition}) {{");
            PushIndent();

            for (int i = 0; i < value.NumCasesWithoutDefault; i++)
            {
                int caseBlock = GetBlockIndex(value.GetCaseTarget(i));
                AppendLine($"case {i}: _block = {caseBlock}; continue _loop;");
            }

            int defaultBlock = GetBlockIndex(value.DefaultBlock);
            AppendLine($"default: _block = {defaultBlock}; continue _loop;");

            PopIndent();
            AppendLine("}");
        }

        /// <summary>
        /// Pushes phi values before a branch.
        /// </summary>
        protected void PushPhiValues(Branch branch)
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
                                var phiVar = Load(phi);
                                var sourceVal = Load(phi[j]);
                                AppendLine($"{phiVar} = {sourceVal};");
                            }
                        }
                    }
                }
            }
        }

        // Casts (mostly pass-through for JS)
        public virtual void GenerateCode(IntAsPointerCast value)
        {
            var target = Load(value);
            var source = Load(value.Value);
            Declare(target);
            AppendLine($"{target} = {source};");
        }

        public virtual void GenerateCode(PointerAsIntCast value)
        {
            var target = Load(value);
            var source = Load(value.Value);
            Declare(target);
            AppendLine($"{target} = {source};");
        }

        public virtual void GenerateCode(PointerCast value)
        {
            var target = Load(value);
            var source = Load(value.Value);
            Declare(target);
            AppendLine($"{target} = {source};");
        }

        public virtual void GenerateCode(AddressSpaceCast value)
        {
            var target = Load(value);
            var source = Load(value.Value);
            Declare(target);
            AppendLine($"{target} = {source};");
        }

        public virtual void GenerateCode(FloatAsIntCast value)
        {
            var target = Load(value);
            var source = Load(value.Value);
            Declare(target);
            // Use DataView for reinterpret cast
            AppendLine($"{target} = _floatAsInt({source});");
        }

        public virtual void GenerateCode(IntAsFloatCast value)
        {
            var target = Load(value);
            var source = Load(value.Value);
            Declare(target);
            AppendLine($"{target} = _intAsFloat({source});");
        }

        // Atomics
        public virtual void GenerateCode(GenericAtomic value)
        {
            var target = Load(value);
            var address = Load(value.Target);
            var val = Load(value.Value);
            Declare(target);

            string atomicOp = value.Kind switch
            {
                AtomicKind.Add => "Atomics.add",
                AtomicKind.And => "Atomics.and",
                AtomicKind.Or => "Atomics.or",
                AtomicKind.Xor => "Atomics.xor",
                AtomicKind.Exchange => "Atomics.exchange",
                AtomicKind.Min => "Atomics.min_",  // placeholder
                AtomicKind.Max => "Atomics.max_",  // placeholder
                _ => "Atomics.add"
            };

            // For typed arrays backed by SharedArrayBuffer
            AppendLine($"{target} = {atomicOp}({address}_view, {address}_idx, {val});");
        }

        public virtual void GenerateCode(AtomicCAS value)
        {
            var target = Load(value);
            var address = Load(value.Target);
            var cmp = Load(value.Value);
            var val = Load(value.CompareValue);
            Declare(target);
            AppendLine($"{target} = Atomics.compareExchange({address}_view, {address}_idx, {cmp}, {val});");
        }

        public virtual void GenerateCode(MemoryBarrier value)
        {
            // No-op for single worker thread (barriers are between workers)
            AppendLine("// memory barrier (no-op in single worker)");
        }

        public virtual void GenerateCode(global::ILGPU.IR.Values.Barrier value)
        {
            // Barriers require inter-worker synchronization - deferred to Phase 2
            AppendLine("// barrier (not yet implemented for workers)");
        }

        // Function Calls
        public virtual void GenerateCode(MethodCall value)
        {
            var target = Load(value);
            Declare(target);

            var method = value.Target;
            var functionName = $"fn_{method.Id}";

            var args = new StringBuilder();
            for (int i = 0; i < value.Nodes.Length; i++)
            {
                if (i > 0) args.Append(", ");
                args.Append(Load(value.Nodes[i]).Name);
            }

            AppendLine($"{target} = {functionName}({args});");
        }

        // Debug/IO
        public virtual void GenerateCode(DebugAssertOperation value)
        {
            var condition = Load(value.Condition.Resolve());
            AppendLine($"// debug assert: {condition}");
        }

        public virtual void GenerateCode(WriteToOutput value)
        {
            AppendLine("// writeToOutput (not supported in workers)");
        }

        // Predicate
        public virtual void GenerateCode(global::ILGPU.IR.Values.Predicate value)
        {
            var target = Load(value);
            var condition = Load(value.Condition);
            var trueVal = Load(value.TrueValue);
            var falseVal = Load(value.FalseValue);
            Declare(target);
            AppendLine($"{target} = {condition} ? {trueVal} : {falseVal};");
        }

        // Alignment (no-op in JS)
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

        // Dynamic memory length
        public virtual void GenerateCode(DynamicMemoryLengthValue value)
        {
            var target = Load(value);
            Declare(target);
            AppendLine($"{target} = 0; // dynamicMemoryLength (not supported)");
        }

        // Predicate barrier
        public virtual void GenerateCode(PredicateBarrier value)
        {
            AppendLine("// predicateBarrier (not yet implemented for workers)");
        }

        // Broadcast
        public virtual void GenerateCode(Broadcast value)
        {
            var target = Load(value);
            var source = Load(value.Variable.Resolve());
            Declare(target);
            AppendLine($"{target} = {source}; // broadcast (single worker)");
        }

        // Warp shuffle
        public virtual void GenerateCode(WarpShuffle value)
        {
            var target = Load(value);
            var source = Load(value.Variable.Resolve());
            Declare(target);
            AppendLine($"{target} = {source}; // warpShuffle (single worker)");
        }

        // Sub-warp shuffle
        public virtual void GenerateCode(SubWarpShuffle value)
        {
            var target = Load(value);
            var source = Load(value.Variable.Resolve());
            Declare(target);
            AppendLine($"{target} = {source}; // subWarpShuffle (single worker)");
        }

        // Language emit value
        public virtual void GenerateCode(LanguageEmitValue value)
        {
            AppendLine("// languageEmitValue (not supported in workers)");
        }

        #endregion
    }
}
