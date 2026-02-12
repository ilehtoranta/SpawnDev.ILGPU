// ---------------------------------------------------------------------------------------
//                                 SpawnDev.ILGPU.WebGL
//                        Copyright (c) 2024 SpawnDev Project
//
// File: GLSLCodeGenerator.cs
//
// Base GLSL ES 3.0 code generator implementing IBackendCodeGenerator for WebGL backend.
// Uses Transform Feedback to emulate compute shaders via vertex shader output.
// ---------------------------------------------------------------------------------------

using global::ILGPU;
using global::ILGPU.Backends;
using global::ILGPU.Backends.EntryPoints;
using global::ILGPU.IR;
using global::ILGPU.IR.Analyses;
using global::ILGPU.IR.Types;
using global::ILGPU.IR.Values;
using System.Text;

namespace SpawnDev.ILGPU.WebGL.Backend
{
    /// <summary>
    /// Base class for GLSL ES 3.0 code generation. Generates GLSL vertex shader source
    /// from ILGPU IR values by implementing the IBackendCodeGenerator interface.
    /// Transform Feedback captures output varyings to emulate compute shader buffers.
    /// </summary>
    public abstract partial class GLSLCodeGenerator : IBackendCodeGenerator<StringBuilder>
    {
        #region Nested Types

        /// <summary>
        /// Generation arguments for GLSL code generator construction.
        /// </summary>
        public readonly struct GeneratorArgs
        {
            public GeneratorArgs(
                WebGLBackend backend,
                GLSLTypeGenerator typeGenerator,
                EntryPoint entryPoint,
                AllocaKindInformation sharedAllocations,
                AllocaKindInformation dynamicSharedAllocations)
            {
                Backend = backend;
                TypeGenerator = typeGenerator;
                EntryPoint = entryPoint;
                SharedAllocations = sharedAllocations;
                DynamicSharedAllocations = dynamicSharedAllocations;
                OutputVaryings = new List<OutputVaryingInfo>();
                ParameterBindings = new List<KernelParameterBinding>();
            }

            /// <summary>The parent backend.</summary>
            public WebGLBackend Backend { get; }
            /// <summary>The type generator.</summary>
            public GLSLTypeGenerator TypeGenerator { get; }
            /// <summary>The kernel entry point.</summary>
            public EntryPoint EntryPoint { get; }
            /// <summary>Shared memory allocations.</summary>
            public AllocaKindInformation SharedAllocations { get; }
            /// <summary>Dynamic shared memory allocations.</summary>
            public AllocaKindInformation DynamicSharedAllocations { get; }
            /// <summary>Output varying metadata populated by the kernel code generator for TF.</summary>
            public List<OutputVaryingInfo> OutputVaryings { get; }
            /// <summary>Parameter binding metadata populated by the kernel code generator.</summary>
            public List<KernelParameterBinding> ParameterBindings { get; }
        }

        /// <summary>
        /// Represents a variable in GLSL code.
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
        protected bool IsStateMachineActive { get; set; } = false;

        protected GLSLCodeGenerator(in GeneratorArgs args, Method method, Allocas allocas)
        {
            Backend = args.Backend;
            TypeGenerator = args.TypeGenerator;
            Method = method;
            Allocas = allocas;
            Builder = new StringBuilder();
        }

        #endregion

        #region Properties

        public WebGLBackend Backend { get; }
        public GLSLTypeGenerator TypeGenerator { get; }
        public Method Method { get; }
        public Allocas Allocas { get; }
        public StringBuilder Builder { get; protected set; }
        public StringBuilder VariableBuilder { get; } = new StringBuilder();
        public global::ILGPU.IR.Intrinsics.IntrinsicImplementationProvider<GLSLIntrinsic.Handler> ImplementationProvider => Backend.IntrinsicProvider;
        protected int IndentLevel { get; set; } = 0;

        #endregion

        #region IBackendCodeGenerator

        public abstract void GenerateHeader(StringBuilder builder);
        public abstract void GenerateCode();
        public void GenerateConstants(StringBuilder builder) { }
        public void Merge(StringBuilder builder) => builder.Append(Builder);

        #endregion

        #region Variable Management

        protected Variable Allocate(Value value)
        {
            var name = $"v_{varCounter++}";
            var type = TypeGenerator[value.Type];
            var variable = new Variable(name, type);
            valueVariables[value] = variable;
            return variable;
        }

        protected Variable AllocateType(TypeNode type)
        {
            var name = $"v_{varCounter++}";
            var glslType = TypeGenerator[type];
            return new Variable(name, glslType);
        }

        public Variable Load(Value value)
        {
            if (!valueVariables.TryGetValue(value, out var variable))
            {
                variable = Allocate(value);
            }
            return variable;
        }

        public Variable LoadIntrinsicValue(Value value) => Load(value);

        protected void Bind(Value value, Variable variable)
        {
            valueVariables[value] = variable;
        }

        protected readonly HashSet<string> declaredVariables = new();
        protected readonly HashSet<string> booleanVariables = new();

        protected void Declare(Variable variable)
        {
            if (declaredVariables.Contains(variable.Name)) return;
            declaredVariables.Add(variable.Name);

            // Track boolean variables for operator selection (GLSL requires && || for booleans)
            if (variable.Type == "bool")
                booleanVariables.Add(variable.Name);

            if (IsStateMachineActive)
            {
                VariableBuilder.Append("    ");
                VariableBuilder.Append(variable.Type);
                VariableBuilder.Append(" ");
                VariableBuilder.Append(variable.Name);
                VariableBuilder.AppendLine(";");
            }
            else
            {
                AppendIndent();
                Builder.Append(variable.Type);
                Builder.Append(" ");
                Builder.Append(variable.Name);
                Builder.AppendLine(";");
            }
        }

        #endregion

        #region Type Casting Helpers

        /// <summary>
        /// Wraps a variable reference in a GLSL type cast if the source type
        /// differs from the target type. Struct types are excluded from casting.
        /// </summary>
        protected static string CastIfNeeded(Variable source, string targetType)
        {
            if (source.Type == targetType) return source.ToString();
            // Don't cast struct types
            if (targetType.StartsWith("struct_") || source.Type.StartsWith("struct_")) return source.ToString();
            // Don't cast booleans
            if (targetType == "bool" || source.Type == "bool") return source.ToString();
            return $"{targetType}({source})";
        }

        /// <summary>
        /// Wraps a string expression in a GLSL type cast if the inferred type
        /// differs from the target type.
        /// </summary>
        protected static string CastIfNeeded(string expression, string targetType)
        {
            // For raw string expressions we can't infer source type, so wrap if target is numeric
            if (targetType == "int" || targetType == "uint" || targetType == "float")
                return $"{targetType}({expression})";
            return expression;
        }

        /// <summary>
        /// Returns a GLSL default/zero value expression for the given type.
        /// </summary>
        protected static string GetDefaultValue(string glslType)
        {
            return glslType switch
            {
                "int" => "0",
                "uint" => "0u",
                "float" => "0.0",
                "bool" => "false",
                "vec2" => "vec2(0.0)",
                "vec3" => "vec3(0.0)",
                "vec4" => "vec4(0.0)",
                "ivec2" => "ivec2(0)",
                "ivec3" => "ivec3(0)",
                "ivec4" => "ivec4(0)",
                "uvec2" => "uvec2(0u)",
                "uvec3" => "uvec3(0u)",
                "uvec4" => "uvec4(0u)",
                _ => $"{glslType}(0)"
            };
        }


        #endregion

        #region Label Management

        protected string DeclareLabel() => $"L_{Method.Id}_{labelCounter++}";

        protected string GetBlockLabel(BasicBlock block)
        {
            if (!blockLabels.TryGetValue(block, out var label))
            {
                label = DeclareLabel();
                blockLabels[block] = label;
            }
            return label;
        }

        protected void MarkLabel(string label) { }

        #endregion

        #region Code Emission Helpers

        protected void AppendIndent()
        {
            for (int i = 0; i < IndentLevel; i++)
                Builder.Append("    ");
        }

        protected void PushIndent() => IndentLevel++;
        protected void PopIndent() => IndentLevel--;

        protected void AppendLine(string line)
        {
            AppendIndent();
            Builder.AppendLine(line);
        }

        protected void AppendLineRaw(string line) => Builder.AppendLine(line);

        protected void BeginFunctionBody()
        {
            Builder.AppendLine("{");
            PushIndent();
        }

        protected void FinishFunctionBody()
        {
            PopIndent();
            Builder.AppendLine("}");
        }

        #endregion

        #region IR Traversal

        protected void GenerateCodeInternal()
        {
            var blocks = Method.Blocks;
            SetupAllocations(Allocas.LocalAllocations, MemoryAddressSpace.Local);

            bool hasReturnValue = !Method.ReturnType.IsVoidType;
            string returnType = hasReturnValue ? TypeGenerator[Method.ReturnType] : "void";

            if (blocks.Count == 1)
            {
                IsStateMachineActive = false;
                foreach (var valueEntry in blocks.First())
                    GenerateCodeFor(valueEntry.Value);
                return;
            }

            // Multiple blocks: use for(;;)/switch state machine
            IsStateMachineActive = true;

            if (hasReturnValue)
            {
                string zeroVal = returnType == "bool" ? "false" : (returnType.StartsWith("float") || returnType == "vec2" ? "0.0" : "0");
                if (returnType.StartsWith("vec") || returnType.StartsWith("ivec") || returnType.StartsWith("uvec"))
                    zeroVal = $"{returnType}(0)";
                AppendLine($"{returnType} _ilgpu_return_val = {zeroVal};");
            }

            int deferredInsertPosition = Builder.Length;

            AppendLine("int current_block = 0;");
            AppendLine("for (;;) {");
            PushIndent();
            AppendLine("switch (current_block) {");
            PushIndent();

            int blockIndex = 0;
            foreach (var block in blocks)
            {
                AppendLine($"case {blockIndex}: {{");
                PushIndent();
                foreach (var valueEntry in block)
                    GenerateCodeFor(valueEntry.Value);
                AppendLine("break;");
                PopIndent();
                AppendLine("}");
                blockIndex++;
            }

            AppendLine("default: break;");
            PopIndent();
            AppendLine("}"); // end switch

            AppendLine("if (current_block == -1) break;");
            PopIndent();
            AppendLine("}"); // end for

            // Insert deferred variable declarations
            if (VariableBuilder.Length > 0)
                Builder.Insert(deferredInsertPosition, VariableBuilder.ToString());

            if (hasReturnValue)
                AppendLine($"return _ilgpu_return_val;");
            else
                AppendLine("return;");
        }

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

        protected void SetupAllocations(AllocaKindInformation allocas, MemoryAddressSpace addressSpace)
        {
            foreach (var allocaInfo in allocas)
            {
                var variable = Allocate(allocaInfo.Alloca);
                var elementType = TypeGenerator[allocaInfo.ElementType];

                if (allocaInfo.IsArray)
                    AppendLine($"{elementType} {variable.Name}[{allocaInfo.ArraySize}];");
                else
                    AppendLine($"{elementType} {variable.Name};");
            }
        }

        #endregion

        #region Value Visitors - Dispatch

        protected void GenerateCodeFor(Value value)
        {
            if (value.Type.IsVoidType &&
                !(value is TerminatorValue) &&
                !(value is Store) &&
                !(value is MemoryBarrier) &&
                !(value is global::ILGPU.IR.Values.Barrier) &&
                !(value is PredicateBarrier))
                return;

            WebGLBackend.Log($"[GLSL] Generating code for: {value.GetType().FullName} - {value}");

            if (value.GetType().Name.Contains("Throw"))
            {
                GenerateThrow(value);
                return;
            }

            if (ImplementationProvider.TryGetCodeGenerator(value, out var intrinsicCodeGenerator))
            {
                intrinsicCodeGenerator(Backend, this, value);
                return;
            }

            switch (value)
            {
                case global::ILGPU.IR.Values.Parameter p: GenerateCode(p); break;
                case global::ILGPU.IR.Values.MethodCall v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.BinaryArithmeticValue v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.UnaryArithmeticValue v: GenerateUnOp(v); break;
                case global::ILGPU.IR.Values.TernaryArithmeticValue v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.CompareValue v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.ConvertValue v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.Load v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.Store v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.LoadElementAddress v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.LoadFieldAddress v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.Alloca v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.NewView v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.PrimitiveValue v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.NullValue v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.StringValue v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.PhiValue v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.StructureValue v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.GetField v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.SetField v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.GridIndexValue v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.GroupIndexValue v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.GridDimensionValue v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.GroupDimensionValue v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.WarpSizeValue v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.LaneIdxValue v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.ReturnTerminator v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.UnconditionalBranch v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.IfBranch v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.SwitchBranch v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.IntAsPointerCast v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.PointerAsIntCast v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.PointerCast v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.AddressSpaceCast v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.FloatAsIntCast v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.IntAsFloatCast v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.GenericAtomic v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.AtomicCAS v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.MemoryBarrier v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.Barrier v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.PredicateBarrier v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.Broadcast v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.WarpShuffle v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.SubWarpShuffle v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.DebugAssertOperation v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.WriteToOutput v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.Predicate v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.DynamicMemoryLengthValue v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.AlignTo v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.AsAligned v: GenerateCode(v); break;
                case global::ILGPU.IR.Values.LanguageEmitValue v: GenerateCode(v); break;
                default:
                    AppendLine($"// Unhandled value type: {value.GetType().Name}");
                    break;
            }
        }

        #endregion

        #region Value Visitors - Implementation

        public virtual void GenerateCode(Parameter parameter) { }

        public virtual void GenerateCode(BinaryArithmeticValue value)
        {
            var target = Load(value);
            var left = Load(value.Left);
            var right = Load(value.Right);
            Declare(target);

            // Float remainder
            if (value.Kind == BinaryArithmeticKind.Rem && TypeGenerator[value.Left.Type].StartsWith("float"))
            {
                AppendLine($"{target} = {left} - {right} * floor({left} / {right});");
                return;
            }

            if (value.Kind == BinaryArithmeticKind.Min || value.Kind == BinaryArithmeticKind.Max)
            {
                string func = value.Kind == BinaryArithmeticKind.Min ? "min" : "max";
                AppendLine($"{target} = {func}({left}, {right});");
                return;
            }

            // Check if this is a boolean operation. GLSL requires logical operators
            // (&&, ||) for booleans, not bitwise (&, |).
            // The target variable's Type is set from TypeGenerator[value.Type] in Allocate().
            bool isBoolOp = target.Type == "bool";

            string op = value.Kind switch
            {
                BinaryArithmeticKind.Add => "+",
                BinaryArithmeticKind.Sub => "-",
                BinaryArithmeticKind.Mul => "*",
                BinaryArithmeticKind.Div => "/",
                BinaryArithmeticKind.And => isBoolOp ? "&&" : "&",
                BinaryArithmeticKind.Or => isBoolOp ? "||" : "|",
                BinaryArithmeticKind.Xor => "^",
                BinaryArithmeticKind.Shl => "<<",
                BinaryArithmeticKind.Shr => ">>",
                BinaryArithmeticKind.Rem => "%",
                _ => "+"
            };

            // GLSL ES 3.0 requires explicit casts — no implicit int<->float conversion.
            // Cast operands to match the target type when they differ.
            string leftExpr = CastIfNeeded(left, target.Type);
            string rightExpr = CastIfNeeded(right, target.Type);
            AppendLine($"{target} = {leftExpr} {op} {rightExpr};");
        }

        public virtual void GenerateCode(UnaryArithmeticValue value) => GenerateUnOp(value);

        private void GenerateUnOp(UnaryArithmeticValue value)
        {
            var target = Load(value);
            var operand = Load(value.Value);
            Declare(target);

            var operandType = TypeGenerator[value.Value.Type];
            if (Backend.Options.EnableI64Emulation && (operandType == "uvec2") && value.Kind == UnaryArithmeticKind.Neg)
            {
                AppendLine($"{target} = i64_neg({operand});");
                return;
            }

            string result = value.Kind switch
            {
                UnaryArithmeticKind.Neg => $"-{operand}",
                UnaryArithmeticKind.Not => TypeGenerator[value.Value.Type] == "bool" ? $"!{operand}" : $"~{operand}",
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
                UnaryArithmeticKind.RsqrtF => $"inversesqrt({operand})",
                UnaryArithmeticKind.RcpF => $"1.0 / {operand}",
                UnaryArithmeticKind.FloorF => $"floor({operand})",
                UnaryArithmeticKind.CeilingF => $"ceil({operand})",
                _ => "DEBUG_MISSING"
            };

            if (result == "DEBUG_MISSING")
            {
                AppendLine($"// [GLSL] Unhandled UnaryArithmeticKind: {value.Kind}");
                result = $"{operand}";
            }

            // GLSL ES 3.0: cast result to target type if needed
            string castResult = CastIfNeeded(result, target.Type);
            AppendLine($"{target} = {castResult};");
        }

        public virtual void GenerateCode(TernaryArithmeticValue value)
        {
            var target = Load(value);
            var first = Load(value.First);
            var second = Load(value.Second);
            var third = Load(value.Third);
            Declare(target);
            // GLSL ES 3.0 has no fma() — emulate with a*b+c
            AppendLine($"{target} = ({first} * {second} + {third});");
        }

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

        public virtual void GenerateCode(ConvertValue value)
        {
            var target = Load(value);
            var source = Load(value.Value);
            var targetType = TypeGenerator[value.Type];
            var sourceType = TypeGenerator[value.Value.Type];
            Declare(target);
            // Skip redundant casts when types are already the same
            if (targetType == sourceType)
                AppendLine($"{target} = {source};");
            else
                AppendLine($"{target} = {targetType}({source});");
        }

        // Memory Operations — GLSL has no pointers; arrays accessed directly
        public virtual void GenerateCode(global::ILGPU.IR.Values.Load loadVal)
        {
            var target = Load(loadVal);
            var source = Load(loadVal.Source);
            Declare(target);
            AppendLine($"{target} = {source};");
        }

        public virtual void GenerateCode(global::ILGPU.IR.Values.Store storeVal)
        {
            var address = Load(storeVal.Target);
            var val = Load(storeVal.Value);
            // GLSL ES 3.0: cast value to match address type if needed
            string valExpr = CastIfNeeded(val, address.Type);
            AppendLine($"{address} = {valExpr};");
        }

        protected virtual bool IsAtomicPointer(Value ptr) => false;

        public virtual void GenerateCode(LoadElementAddress value)
        {
            var target = Load(value);
            var source = Load(value.Source);
            var offset = Load(value.Offset);
            // In GLSL, array element access is source[offset]
            // We store the expression as a reference for later Load/Store
            Declare(target);
            AppendLine($"// LEA: {target} = {source}[{offset}]");
        }

        public virtual void GenerateCode(global::ILGPU.IR.Values.NewView value)
        {
            var target = Load(value);
            var source = Load(value.Pointer);
            Declare(target);
            AppendLine($"{target} = {source}; // newView");
        }

        public virtual void GenerateCode(LoadFieldAddress value)
        {
            var target = Load(value);
            var source = Load(value.Source);
            Declare(target);
            string fieldName = $"field_{value.FieldSpan.Index}";
            if (IsIndexType(value.Source.Type))
            {
                fieldName = value.FieldSpan.Index switch
                {
                    0 => "x",  1 => "y",  2 => "z",  _ => fieldName
                };
            }
            AppendLine($"// LFA: {target} = {source}.{fieldName}");
        }

        public virtual void GenerateCode(Alloca value) { }

        // Constants
        public virtual void GenerateCode(PrimitiveValue value)
        {
            var target = Load(value);
            var type = TypeGenerator[value.Type];
            Declare(target);

            bool isEmulatedF64 = Backend.Options.EnableF64Emulation && value.BasicValueType == BasicValueType.Float64;
            bool isEmulatedI64 = Backend.Options.EnableI64Emulation && value.BasicValueType == BasicValueType.Int64;

            if (isEmulatedF64)
            {
                double doubleVal = value.Float64Value;
                ulong bits = BitConverter.DoubleToUInt64Bits(doubleVal);
                uint lo = (uint)(bits & 0xFFFFFFFF);
                uint hi = (uint)(bits >> 32);
                AppendLine($"{target} = f64_from_ieee754_bits({lo}u, {hi}u);");
                return;
            }

            if (isEmulatedI64)
            {
                long longVal = value.Int64Value;
                uint lo = (uint)(longVal & 0xFFFFFFFF);
                uint hi = (uint)((ulong)longVal >> 32);
                AppendLine($"{target} = uvec2({lo}u, {hi}u);");
                return;
            }

            string valStr = value.BasicValueType switch
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
                AppendLine($"{target} = {type}({valStr});");
            else
                AppendLine($"{target} = {valStr};");
        }

        private string FormatFloat(float value)
        {
            if (float.IsNaN(value)) return "0.0";
            if (float.IsPositiveInfinity(value)) return "3.402823e+38";
            if (float.IsNegativeInfinity(value)) return "-3.402823e+38";
            var str = value.ToString("G9");
            if (!str.Contains('.') && !str.Contains('e') && !str.Contains('E'))
                str += ".0";
            return str;
        }

        public virtual void GenerateCode(NullValue value)
        {
            var target = Load(value);
            Declare(target);
            AppendLine($"{target} = {target.Type}(0); // null");
        }

        public virtual void GenerateCode(global::ILGPU.IR.Values.Barrier value)
        {
            // WebGL2 vertex shaders do not support barriers
            AppendLine("// barrier not supported in WebGL2 vertex shaders");
        }

        public virtual void GenerateCode(StringValue value)
        {
            AppendLine($"// String: {value.String}");
        }

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
                    0 => "x",  1 => "y",  2 => "z",  _ => fieldName
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
                    0 => "x",  1 => "y",  2 => "z",  _ => fieldName
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

        // Device Constants — mapped from gl_VertexID in kernel generator
        public virtual void GenerateCode(GridIndexValue value)
        {
            var target = Load(value);
            Declare(target);
            AppendLine($"{target} = 0; // GridIndex not supported in WebGL2 TF");
        }

        public virtual void GenerateCode(GroupIndexValue value)
        {
            var target = Load(value);
            Declare(target);
            AppendLine($"{target} = 0; // GroupIndex not supported in WebGL2 TF");
        }

        public virtual void GenerateCode(GridDimensionValue value)
        {
            var target = Load(value);
            Declare(target);
            AppendLine($"{target} = 0; // GridDimension not supported in WebGL2 TF");
        }

        public virtual void GenerateCode(GroupDimensionValue value)
        {
            var target = Load(value);
            Declare(target);
            AppendLine($"{target} = 0; // GroupDimension not supported in WebGL2 TF");
        }

        public virtual void GenerateCode(WarpSizeValue value)
        {
            var target = Load(value);
            Declare(target);
            AppendLine($"{target} = 1; // No warps in WebGL2");
        }

        public virtual void GenerateCode(LaneIdxValue value)
        {
            var target = Load(value);
            Declare(target);
            AppendLine($"{target} = 0; // No lanes in WebGL2");
        }

        // Control Flow
        public virtual void GenerateCode(ReturnTerminator value)
        {
            if (IsStateMachineActive)
            {
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
                if (value.IsVoidReturn)
                    AppendLine("return;");
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
                AppendLine("break;");
                PopIndent();
                AppendLine("}");
            }
            int defaultIdx = GetBlockIndex(branch.DefaultBlock);
            AppendLine("default: {");
            PushIndent();
            EmitPhiAssignments(branch.BasicBlock, branch.DefaultBlock);
            AppendLine($"current_block = {defaultIdx};");
            AppendLine("break;");
            PopIndent();
            AppendLine("}");
            PopIndent();
            AppendLine("}");
            AppendLine("continue;");
        }

        protected virtual void EmitPhiAssignments(BasicBlock sourceBlock, BasicBlock targetBlock)
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
            string? glslFunc = name switch
            {
                var n when n.Contains("Rsqrt") => "inversesqrt",
                var n when n.Contains("Rcp") => "rcp_custom",
                var n when n.Contains("Asin") => "asin",
                var n when n.Contains("Acos") => "acos",
                var n when n.Contains("Atan2") => "atan",
                var n when n.Contains("Atan") => "atan",
                var n when n.Contains("Sinh") => "sinh",
                var n when n.Contains("Cosh") => "cosh",
                var n when n.Contains("Tanh") => "tanh",
                var n when n.Contains("FusedMultiplyAdd") => "fma_custom",
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
                _ => null
            };

            if (glslFunc != null)
            {
                if (glslFunc == "rcp_custom" && methodCall.Count == 1)
                {
                    AppendLine($"{target} = 1.0 / {Load(methodCall[0])};");
                    return;
                }
                if (glslFunc == "fma_custom" && methodCall.Count == 3)
                {
                    var a = Load(methodCall[0]); var b = Load(methodCall[1]); var c = Load(methodCall[2]);
                    AppendLine($"{target} = {a} * {b} + {c};");
                    return;
                }

                var args = new StringBuilder();
                for (int i = 0; i < methodCall.Count; i++)
                {
                    if (i > 0) args.Append(", ");
                    args.Append(Load(methodCall[i]));
                }
                AppendLine($"{target} = {glslFunc}({args});");
                return;
            }

            AppendLine($"// Call: {methodCall.Target.Name} (Unmapped)");
            AppendLine($"{target} = {target.Type}(0);");
        }

        // Casts
        public virtual void GenerateCode(IntAsPointerCast value)
        {
            var target = Load(value); var source = Load(value.Value);
            Declare(target);
            AppendLine($"{target} = {source}; // intAsPtr");
        }

        public virtual void GenerateCode(PointerAsIntCast value)
        {
            var target = Load(value); var source = Load(value.Value);
            Declare(target);
            AppendLine($"{target} = {target.Type}({source}); // ptrAsInt");
        }

        public virtual void GenerateCode(PointerCast value)
        {
            var target = Load(value); var source = Load(value.Value);
            Declare(target);
            AppendLine($"{target} = {source}; // ptrCast");
        }

        public virtual void GenerateCode(AddressSpaceCast value)
        {
            var target = Load(value); var source = Load(value.Value);
            Declare(target);
            AppendLine($"{target} = {source}; // addrSpaceCast");
        }

        public virtual void GenerateCode(FloatAsIntCast value)
        {
            var target = Load(value); var source = Load(value.Value);
            Declare(target);
            AppendLine($"{target} = floatBitsToInt({source});");
        }

        public virtual void GenerateCode(IntAsFloatCast value)
        {
            var target = Load(value); var source = Load(value.Value);
            Declare(target);
            AppendLine($"{target} = intBitsToFloat({source});");
        }

        // Atomics & Barriers — not supported in WebGL2 vertex shaders
        public virtual void GenerateCode(GenericAtomic value)
        {
            var target = Load(value);
            Declare(target);
            AppendLine($"// Atomics not supported in WebGL2 vertex shaders");
            AppendLine($"{target} = {target.Type}(0);");
        }

        public virtual void GenerateCode(AtomicCAS value)
        {
            var target = Load(value);
            Declare(target);
            AppendLine($"// AtomicCAS not supported in WebGL2 vertex shaders");
            AppendLine($"{target} = {target.Type}(0);");
        }

        public virtual void GenerateCode(MemoryBarrier value)
        {
            AppendLine("// memoryBarrier not supported in WebGL2 TF shaders");
        }

        public virtual void GenerateCode(PredicateBarrier value)
        {
            AppendLine("// predicateBarrier not supported in WebGL2 TF shaders");
        }

        // Warp Operations — not supported
        public virtual void GenerateCode(Broadcast value)
        {
            var target = Load(value);
            var source = Load(value.Variable);
            Declare(target);
            AppendLine($"{target} = {source}; // broadcast fallback");
        }

        public virtual void GenerateCode(WarpShuffle value)
        {
            var target = Load(value);
            var source = Load(value.Variable);
            Declare(target);
            AppendLine($"{target} = {source}; // shuffle fallback");
        }

        public virtual void GenerateCode(SubWarpShuffle value)
        {
            var target = Load(value);
            var source = Load(value.Variable);
            Declare(target);
            AppendLine($"{target} = {source}; // subShuffle fallback");
        }

        // Debug/IO
        public virtual void GenerateCode(DebugAssertOperation value) { }
        public virtual void GenerateCode(WriteToOutput value) { }

        // Other
        public virtual void GenerateCode(Predicate value)
        {
            var target = Load(value);
            var cond = Load(value.Condition);
            var trueVal = Load(value.TrueValue);
            var falseVal = Load(value.FalseValue);
            Declare(target);
            AppendLine($"{target} = {cond} ? {trueVal} : {falseVal};");
        }

        public virtual void GenerateCode(DynamicMemoryLengthValue value)
        {
            var target = Load(value);
            Declare(target);
            AppendLine($"{target} = 0; // dynamic memory length (placeholder)");
        }

        public virtual void GenerateCode(AlignTo value)
        {
            var target = Load(value); var source = Load(value.Source);
            Declare(target);
            AppendLine($"{target} = {source}; // alignTo");
        }

        public virtual void GenerateCode(AsAligned value)
        {
            var target = Load(value); var source = Load(value.Source);
            Declare(target);
            AppendLine($"{target} = {source}; // asAligned");
        }

        public virtual void GenerateCode(LanguageEmitValue value) { }

        public virtual void GenerateThrow(Value value)
        {
            AppendLine($"// [GLSL] Throw encountered: {value} (Ignored/Unreachable)");
            if (IsStateMachineActive)
            {
                AppendLine("current_block = -1;");
                AppendLine("break;");
            }
            else
            {
                // For non-void functions, return a typed default value
                // so GLSL can see all code paths return correctly.
                var returnType = TypeGenerator[Method.ReturnType];
                if (returnType == "void")
                    AppendLine("return;");
                else
                    AppendLine($"return {GetDefaultValue(returnType)};");
            }
        }

        #endregion

        #region Math Intrinsics

        public static void GenerateAbs(WebGLBackend backend, GLSLCodeGenerator cg, Value value)
        {
            if (value is MethodCall mc)
            {
                var t = cg.LoadIntrinsicValue(value);
                var o = cg.LoadIntrinsicValue(mc[0].Resolve());
                cg.Declare(t);
                cg.AppendLine($"{t} = abs({o});");
            }
        }

        public static void GenerateSign(WebGLBackend backend, GLSLCodeGenerator cg, Value value)
        {
            if (value is MethodCall mc)
            {
                var t = cg.LoadIntrinsicValue(value);
                var o = cg.LoadIntrinsicValue(mc[0].Resolve());
                cg.Declare(t);
                cg.AppendLine($"{t} = sign({o});");
            }
        }

        public static void GenerateRound(WebGLBackend backend, GLSLCodeGenerator cg, Value value)
        {
            if (value is MethodCall mc)
            {
                var t = cg.LoadIntrinsicValue(value);
                var o = cg.LoadIntrinsicValue(mc[0].Resolve());
                cg.Declare(t);
                cg.AppendLine($"{t} = round({o});");
            }
        }

        public static void GenerateTruncate(WebGLBackend backend, GLSLCodeGenerator cg, Value value)
        {
            if (value is MethodCall mc)
            {
                var t = cg.LoadIntrinsicValue(value);
                var o = cg.LoadIntrinsicValue(mc[0].Resolve());
                cg.Declare(t);
                cg.AppendLine($"{t} = trunc({o});");
            }
        }

        public static void GenerateAtan2(WebGLBackend backend, GLSLCodeGenerator cg, Value value)
        {
            if (value is MethodCall mc)
            {
                var t = cg.LoadIntrinsicValue(value);
                var y = cg.LoadIntrinsicValue(mc[0].Resolve());
                var x = cg.LoadIntrinsicValue(mc[1].Resolve());
                cg.Declare(t);
                cg.AppendLine($"{t} = atan({y}, {x});");
            }
        }

        public static void GenerateMax(WebGLBackend backend, GLSLCodeGenerator cg, Value value)
        {
            if (value is MethodCall mc)
            {
                var t = cg.LoadIntrinsicValue(value);
                var a = cg.LoadIntrinsicValue(mc[0].Resolve());
                var b = cg.LoadIntrinsicValue(mc[1].Resolve());
                cg.Declare(t);
                cg.AppendLine($"{t} = max({a}, {b});");
            }
        }

        public static void GenerateMin(WebGLBackend backend, GLSLCodeGenerator cg, Value value)
        {
            if (value is MethodCall mc)
            {
                var t = cg.LoadIntrinsicValue(value);
                var a = cg.LoadIntrinsicValue(mc[0].Resolve());
                var b = cg.LoadIntrinsicValue(mc[1].Resolve());
                cg.Declare(t);
                cg.AppendLine($"{t} = min({a}, {b});");
            }
        }

        public static void GeneratePow(WebGLBackend backend, GLSLCodeGenerator cg, Value value)
        {
            if (value is MethodCall mc)
            {
                var t = cg.LoadIntrinsicValue(value);
                var b = cg.LoadIntrinsicValue(mc[0].Resolve());
                var e = cg.LoadIntrinsicValue(mc[1].Resolve());
                cg.Declare(t);
                cg.AppendLine($"{t} = pow({b}, {e});");
            }
        }

        public static void GenerateClamp(WebGLBackend backend, GLSLCodeGenerator cg, Value value)
        {
            if (value is MethodCall mc)
            {
                var t = cg.LoadIntrinsicValue(value);
                var v = cg.LoadIntrinsicValue(mc[0].Resolve());
                var mn = cg.LoadIntrinsicValue(mc[1].Resolve());
                var mx = cg.LoadIntrinsicValue(mc[2].Resolve());
                cg.Declare(t);
                cg.AppendLine($"{t} = clamp({v}, {mn}, {mx});");
            }
        }

        public static void GenerateFusedMultiplyAdd(WebGLBackend backend, GLSLCodeGenerator cg, Value value)
        {
            if (value is MethodCall mc)
            {
                var t = cg.LoadIntrinsicValue(value);
                var x = cg.LoadIntrinsicValue(mc[0].Resolve());
                var y = cg.LoadIntrinsicValue(mc[1].Resolve());
                var z = cg.LoadIntrinsicValue(mc[2].Resolve());
                cg.Declare(t);
                cg.AppendLine($"{t} = {x} * {y} + {z};");
            }
        }

        public static void GenerateRsqrt(WebGLBackend backend, GLSLCodeGenerator cg, Value value)
        {
            if (value is MethodCall mc)
            {
                var t = cg.LoadIntrinsicValue(value);
                var o = cg.LoadIntrinsicValue(mc[0].Resolve());
                cg.Declare(t);
                cg.AppendLine($"{t} = inversesqrt({o});");
            }
        }

        public static void GenerateRcp(WebGLBackend backend, GLSLCodeGenerator cg, Value value)
        {
            if (value is MethodCall mc)
            {
                var t = cg.LoadIntrinsicValue(value);
                var o = cg.LoadIntrinsicValue(mc[0].Resolve());
                cg.Declare(t);
                cg.AppendLine($"{t} = 1.0 / {o};");
            }
        }

        #endregion
    }
}
