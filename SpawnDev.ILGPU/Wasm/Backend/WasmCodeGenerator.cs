// ---------------------------------------------------------------------------------------
//                               SpawnDev.ILGPU.Wasm
//                    WebAssembly Compute Backend for Blazor WebAssembly
//
// File: WasmCodeGenerator.cs
//
// Base class for Wasm code generation. Implements IBackendCodeGenerator<StringBuilder>
// to translate ILGPU IR values into WebAssembly binary instructions.
// Each IR value maps to Wasm stack-machine instructions.
// ---------------------------------------------------------------------------------------

using global::ILGPU;
using global::ILGPU.Backends;
using global::ILGPU.Backends.EntryPoints;
using global::ILGPU.IR;
using global::ILGPU.IR.Analyses;
using global::ILGPU.IR.Types;
using global::ILGPU.IR.Values;
using System.Text;

namespace SpawnDev.ILGPU.Wasm.Backend
{
    /// <summary>
    /// Base class for Wasm code generation. Translates ILGPU IR values
    /// into WebAssembly binary instructions using a stack-based model.
    /// Each IR value is assigned a Wasm local variable.
    /// </summary>
    public abstract class WasmCodeGenerator : IBackendCodeGenerator<StringBuilder>
    {
        /// <summary>
        /// Generation arguments for Wasm code generator construction.
        /// </summary>
        public class GeneratorArgs
        {
            public WasmBackend Backend { get; }
            public EntryPoint EntryPoint { get; }
            public AllocaKindInformation SharedAllocations { get; }
            public AllocaKindInformation DynamicSharedAllocations { get; }

            /// <summary>
            /// Shared data: parameter infos populated by the kernel generator, read by CreateKernel.
            /// </summary>
            public List<WasmParamInfo> ParamInfos { get; set; } = new();

            public GeneratorArgs(
                WasmBackend backend,
                EntryPoint entryPoint,
                AllocaKindInformation sharedAllocations,
                AllocaKindInformation dynamicSharedAllocations)
            {
                Backend = backend;
                EntryPoint = entryPoint;
                SharedAllocations = sharedAllocations;
                DynamicSharedAllocations = dynamicSharedAllocations;
            }
        }

        #region Fields

        protected readonly GeneratorArgs _generatorArgs;
        protected readonly Method Method;
        protected readonly Allocas Allocas;

        /// <summary>
        /// The Wasm instruction byte stream for the current function body.
        /// </summary>
        internal readonly List<byte> Code = new();

        /// <summary>
        /// Local variables for the Wasm function (params + locals).
        /// Maps ILGPU Value to Wasm local index.
        /// </summary>
        protected readonly Dictionary<string, uint> _localMap = new();

        /// <summary>
        /// Wasm local variable declarations (type only, separate from params).
        /// </summary>
        internal readonly List<WasmLocal> _locals = new();

        /// <summary>
        /// Next available local index (starts after function params).
        /// </summary>
        internal uint _nextLocalIndex = 0;

        /// <summary>
        /// Number of function parameters.
        /// </summary>
        internal uint _paramCount = 0;

        #endregion

        #region Constructor

        public WasmCodeGenerator(
            in GeneratorArgs args,
            Method method,
            Allocas allocas)
        {
            _generatorArgs = args;
            Method = method;
            Allocas = allocas;
        }

        #endregion

        #region Variable Management

        /// <summary>
        /// Allocates a new Wasm local variable for the given IR value.
        /// Returns the local index.
        /// </summary>
        protected uint AllocateLocal(Value value, byte wasmType = WasmOpCodes.I32)
        {
            var key = GetValueKey(value);
            if (_localMap.TryGetValue(key, out var existing))
                return existing;

            uint index = _nextLocalIndex++;
            _localMap[key] = index;
            // Only add to _locals if this is beyond the parameter range
            if (index >= _paramCount)
            {
                _locals.Add(new WasmLocal { Count = 1, Type = wasmType });
            }
            return index;
        }

        /// <summary>
        /// Gets the local index for an already-allocated value.
        /// </summary>
        protected uint GetLocal(Value value)
        {
            var key = GetValueKey(value);
            if (_localMap.TryGetValue(key, out var index))
                return index;

            // Auto-allocate if not found
            return AllocateLocal(value, GetWasmType(value));
        }

        /// <summary>
        /// Gets a unique key for a value (used for local variable mapping).
        /// </summary>
        protected string GetValueKey(Value value)
        {
            return $"v_{value.Id}";
        }

        /// <summary>
        /// Gets the Wasm type for an ILGPU value.
        /// </summary>
        protected byte GetWasmType(Value value)
        {
            return GetWasmTypeFromIR(value.Type);
        }

        /// <summary>
        /// Maps an ILGPU IR type to a Wasm value type.
        /// </summary>
        protected static byte GetWasmTypeFromIR(TypeNode type)
        {
            if (type is PrimitiveType pt)
            {
                return pt.BasicValueType switch
                {
                    BasicValueType.Int1 => WasmOpCodes.I32,
                    BasicValueType.Int8 => WasmOpCodes.I32,
                    BasicValueType.Int16 => WasmOpCodes.I32,
                    BasicValueType.Int32 => WasmOpCodes.I32,
                    BasicValueType.Int64 => WasmOpCodes.I64,
                    BasicValueType.Float16 => WasmOpCodes.F32,
                    BasicValueType.Float32 => WasmOpCodes.F32,
                    BasicValueType.Float64 => WasmOpCodes.F64,
                    _ => WasmOpCodes.I32,
                };
            }
            // Pointers, views, structs → i32 (memory offset)
            return WasmOpCodes.I32;
        }

        /// <summary>
        /// Gets the element size in bytes for a basic value type.
        /// </summary>
        protected static int GetElementSize(BasicValueType type)
        {
            return type switch
            {
                BasicValueType.Int1 => 1,
                BasicValueType.Int8 => 1,
                BasicValueType.Int16 => 2,
                BasicValueType.Int32 => 4,
                BasicValueType.Int64 => 8,
                BasicValueType.Float16 => 2,
                BasicValueType.Float32 => 4,
                BasicValueType.Float64 => 8,
                _ => 4,
            };
        }

        #endregion

        #region Emit Helpers

        /// <summary>
        /// Pushes the value of a Wasm local onto the stack.
        /// </summary>
        protected void EmitGetLocal(Value value)
        {
            WasmModuleBuilder.EmitLocalGet(Code, GetLocal(value));
        }

        /// <summary>
        /// Pops the stack and stores into a Wasm local.
        /// </summary>
        protected void EmitSetLocal(Value value)
        {
            WasmModuleBuilder.EmitLocalSet(Code, GetLocal(value));
        }

        /// <summary>
        /// Pushes a local by index.
        /// </summary>
        protected void EmitGetLocalByIndex(uint index)
        {
            WasmModuleBuilder.EmitLocalGet(Code, index);
        }

        #endregion

        #region IBackendCodeGenerator<StringBuilder>

        public virtual void GenerateHeader(StringBuilder builder) { }

        public abstract void GenerateCode();

        public virtual void GenerateConstants(StringBuilder builder) { }

        public virtual void Merge(StringBuilder builder)
        {
            builder.AppendLine($"// Wasm function body: {Code.Count} bytes, {_locals.Count} locals");
        }

        #endregion

        #region IBackendCodeGenerator — all required method implementations

        // Arithmetic
        public virtual void GenerateCode(UnaryArithmeticValue value)
        {
            var target = AllocateLocal(value, GetWasmType(value));
            var src = value.Value.Resolve();
            var wasmType = GetWasmType(value);

            EmitGetLocal(src);

            switch (value.Kind)
            {
                case UnaryArithmeticKind.Neg:
                    if (wasmType == WasmOpCodes.F32) Code.Add(WasmOpCodes.F32Neg);
                    else if (wasmType == WasmOpCodes.F64) Code.Add(WasmOpCodes.F64Neg);
                    else
                    {
                        WasmModuleBuilder.EmitLocalSet(Code, target);
                        WasmModuleBuilder.EmitI32Const(Code, 0);
                        WasmModuleBuilder.EmitLocalGet(Code, target);
                        Code.Add(WasmOpCodes.I32Sub);
                    }
                    break;
                case UnaryArithmeticKind.Not:
                    WasmModuleBuilder.EmitI32Const(Code, -1);
                    Code.Add(WasmOpCodes.I32Xor);
                    break;
                case UnaryArithmeticKind.Abs:
                    if (wasmType == WasmOpCodes.F32) Code.Add(WasmOpCodes.F32Abs);
                    else if (wasmType == WasmOpCodes.F64) Code.Add(WasmOpCodes.F64Abs);
                    else
                    {
                        WasmModuleBuilder.EmitLocalTee(Code, target);
                        WasmModuleBuilder.EmitLocalGet(Code, target);
                        WasmModuleBuilder.EmitI32Const(Code, 31);
                        Code.Add(WasmOpCodes.I32ShrS);
                        Code.Add(WasmOpCodes.I32Xor);
                        WasmModuleBuilder.EmitLocalGet(Code, target);
                        WasmModuleBuilder.EmitI32Const(Code, 31);
                        Code.Add(WasmOpCodes.I32ShrS);
                        Code.Add(WasmOpCodes.I32Sub);
                    }
                    break;
                default:
                    break;
            }

            WasmModuleBuilder.EmitLocalSet(Code, target);
        }

        public virtual void GenerateCode(BinaryArithmeticValue value)
        {
            var target = AllocateLocal(value, GetWasmType(value));
            var left = value.Left.Resolve();
            var right = value.Right.Resolve();
            var wasmType = GetWasmType(value);

            EmitGetLocal(left);
            EmitGetLocal(right);

            byte opcode = (wasmType, value.Kind) switch
            {
                (WasmOpCodes.I32, BinaryArithmeticKind.Add) => WasmOpCodes.I32Add,
                (WasmOpCodes.I32, BinaryArithmeticKind.Sub) => WasmOpCodes.I32Sub,
                (WasmOpCodes.I32, BinaryArithmeticKind.Mul) => WasmOpCodes.I32Mul,
                (WasmOpCodes.I32, BinaryArithmeticKind.Div) => WasmOpCodes.I32DivS,
                (WasmOpCodes.I32, BinaryArithmeticKind.Rem) => WasmOpCodes.I32RemS,
                (WasmOpCodes.I32, BinaryArithmeticKind.And) => WasmOpCodes.I32And,
                (WasmOpCodes.I32, BinaryArithmeticKind.Or) => WasmOpCodes.I32Or,
                (WasmOpCodes.I32, BinaryArithmeticKind.Xor) => WasmOpCodes.I32Xor,
                (WasmOpCodes.I32, BinaryArithmeticKind.Shl) => WasmOpCodes.I32Shl,
                (WasmOpCodes.I32, BinaryArithmeticKind.Shr) => WasmOpCodes.I32ShrS,

                (WasmOpCodes.I64, BinaryArithmeticKind.Add) => WasmOpCodes.I64Add,
                (WasmOpCodes.I64, BinaryArithmeticKind.Sub) => WasmOpCodes.I64Sub,
                (WasmOpCodes.I64, BinaryArithmeticKind.Mul) => WasmOpCodes.I64Mul,
                (WasmOpCodes.I64, BinaryArithmeticKind.Div) => WasmOpCodes.I64DivS,
                (WasmOpCodes.I64, BinaryArithmeticKind.And) => WasmOpCodes.I64And,
                (WasmOpCodes.I64, BinaryArithmeticKind.Or) => WasmOpCodes.I64Or,
                (WasmOpCodes.I64, BinaryArithmeticKind.Xor) => WasmOpCodes.I64Xor,
                (WasmOpCodes.I64, BinaryArithmeticKind.Shl) => WasmOpCodes.I64Shl,
                (WasmOpCodes.I64, BinaryArithmeticKind.Shr) => WasmOpCodes.I64ShrS,

                (WasmOpCodes.F32, BinaryArithmeticKind.Add) => WasmOpCodes.F32Add,
                (WasmOpCodes.F32, BinaryArithmeticKind.Sub) => WasmOpCodes.F32Sub,
                (WasmOpCodes.F32, BinaryArithmeticKind.Mul) => WasmOpCodes.F32Mul,
                (WasmOpCodes.F32, BinaryArithmeticKind.Div) => WasmOpCodes.F32Div,
                (WasmOpCodes.F32, BinaryArithmeticKind.Min) => WasmOpCodes.F32Min,
                (WasmOpCodes.F32, BinaryArithmeticKind.Max) => WasmOpCodes.F32Max,

                (WasmOpCodes.F64, BinaryArithmeticKind.Add) => WasmOpCodes.F64Add,
                (WasmOpCodes.F64, BinaryArithmeticKind.Sub) => WasmOpCodes.F64Sub,
                (WasmOpCodes.F64, BinaryArithmeticKind.Mul) => WasmOpCodes.F64Mul,
                (WasmOpCodes.F64, BinaryArithmeticKind.Div) => WasmOpCodes.F64Div,
                (WasmOpCodes.F64, BinaryArithmeticKind.Min) => WasmOpCodes.F64Min,
                (WasmOpCodes.F64, BinaryArithmeticKind.Max) => WasmOpCodes.F64Max,

                _ => WasmOpCodes.I32Add // fallback
            };

            Code.Add(opcode);
            WasmModuleBuilder.EmitLocalSet(Code, target);
        }

        public virtual void GenerateCode(TernaryArithmeticValue value)
        {
            var target = AllocateLocal(value, GetWasmType(value));
            var a = value.First.Resolve();
            var b = value.Second.Resolve();
            var c = value.Third.Resolve();
            var wasmType = GetWasmType(value);

            EmitGetLocal(a);
            EmitGetLocal(b);
            Code.Add(wasmType == WasmOpCodes.F64 ? WasmOpCodes.F64Mul : WasmOpCodes.F32Mul);
            EmitGetLocal(c);
            Code.Add(wasmType == WasmOpCodes.F64 ? WasmOpCodes.F64Add : WasmOpCodes.F32Add);
            WasmModuleBuilder.EmitLocalSet(Code, target);
        }

        // Comparison
        public virtual void GenerateCode(CompareValue value)
        {
            var target = AllocateLocal(value, WasmOpCodes.I32);
            var left = value.Left.Resolve();
            var right = value.Right.Resolve();
            var srcType = GetWasmTypeFromIR(left.Type);

            EmitGetLocal(left);
            EmitGetLocal(right);

            byte opcode = (srcType, value.Kind) switch
            {
                (WasmOpCodes.I32, CompareKind.Equal) => WasmOpCodes.I32Eq,
                (WasmOpCodes.I32, CompareKind.NotEqual) => WasmOpCodes.I32Ne,
                (WasmOpCodes.I32, CompareKind.LessThan) => WasmOpCodes.I32LtS,
                (WasmOpCodes.I32, CompareKind.LessEqual) => WasmOpCodes.I32LeS,
                (WasmOpCodes.I32, CompareKind.GreaterThan) => WasmOpCodes.I32GtS,
                (WasmOpCodes.I32, CompareKind.GreaterEqual) => WasmOpCodes.I32GeS,

                (WasmOpCodes.I64, CompareKind.Equal) => WasmOpCodes.I64Eq,
                (WasmOpCodes.I64, CompareKind.NotEqual) => WasmOpCodes.I64Ne,
                (WasmOpCodes.I64, CompareKind.LessThan) => WasmOpCodes.I64LtS,
                (WasmOpCodes.I64, CompareKind.LessEqual) => WasmOpCodes.I64LeS,
                (WasmOpCodes.I64, CompareKind.GreaterThan) => WasmOpCodes.I64GtS,
                (WasmOpCodes.I64, CompareKind.GreaterEqual) => WasmOpCodes.I64GeS,

                (WasmOpCodes.F32, CompareKind.Equal) => WasmOpCodes.F32Eq,
                (WasmOpCodes.F32, CompareKind.NotEqual) => WasmOpCodes.F32Ne,
                (WasmOpCodes.F32, CompareKind.LessThan) => WasmOpCodes.F32Lt,
                (WasmOpCodes.F32, CompareKind.LessEqual) => WasmOpCodes.F32Le,
                (WasmOpCodes.F32, CompareKind.GreaterThan) => WasmOpCodes.F32Gt,
                (WasmOpCodes.F32, CompareKind.GreaterEqual) => WasmOpCodes.F32Ge,

                (WasmOpCodes.F64, CompareKind.Equal) => WasmOpCodes.F64Eq,
                (WasmOpCodes.F64, CompareKind.NotEqual) => WasmOpCodes.F64Ne,
                (WasmOpCodes.F64, CompareKind.LessThan) => WasmOpCodes.F64Lt,
                (WasmOpCodes.F64, CompareKind.LessEqual) => WasmOpCodes.F64Le,
                (WasmOpCodes.F64, CompareKind.GreaterThan) => WasmOpCodes.F64Gt,
                (WasmOpCodes.F64, CompareKind.GreaterEqual) => WasmOpCodes.F64Ge,

                _ => WasmOpCodes.I32Eq // fallback
            };

            Code.Add(opcode);
            WasmModuleBuilder.EmitLocalSet(Code, target);
        }

        // Conversions
        public virtual void GenerateCode(ConvertValue value)
        {
            var target = AllocateLocal(value, GetWasmType(value));
            var src = value.Value.Resolve();
            var srcType = GetWasmTypeFromIR(src.Type);
            var dstType = GetWasmType(value);

            EmitGetLocal(src);

            byte? opcode = (srcType, dstType) switch
            {
                (WasmOpCodes.I32, WasmOpCodes.I64) => WasmOpCodes.I64ExtendI32S,
                (WasmOpCodes.I32, WasmOpCodes.F32) => WasmOpCodes.F32ConvertI32S,
                (WasmOpCodes.I32, WasmOpCodes.F64) => WasmOpCodes.F64ConvertI32S,
                (WasmOpCodes.I64, WasmOpCodes.I32) => WasmOpCodes.I32WrapI64,
                (WasmOpCodes.I64, WasmOpCodes.F32) => WasmOpCodes.F32ConvertI64S,
                (WasmOpCodes.I64, WasmOpCodes.F64) => WasmOpCodes.F64ConvertI64S,
                (WasmOpCodes.F32, WasmOpCodes.I32) => WasmOpCodes.I32TruncF32S,
                (WasmOpCodes.F32, WasmOpCodes.I64) => WasmOpCodes.I64TruncF32S,
                (WasmOpCodes.F32, WasmOpCodes.F64) => WasmOpCodes.F64PromoteF32,
                (WasmOpCodes.F64, WasmOpCodes.I32) => WasmOpCodes.I32TruncF64S,
                (WasmOpCodes.F64, WasmOpCodes.I64) => WasmOpCodes.I64TruncF64S,
                (WasmOpCodes.F64, WasmOpCodes.F32) => WasmOpCodes.F32DemoteF64,
                _ => null
            };

            if (opcode.HasValue)
                Code.Add(opcode.Value);

            WasmModuleBuilder.EmitLocalSet(Code, target);
        }

        // Casts
        public virtual void GenerateCode(IntAsPointerCast cast)
        {
            var target = AllocateLocal(cast);
            EmitGetLocal(cast.Value.Resolve());
            WasmModuleBuilder.EmitLocalSet(Code, target);
        }

        public virtual void GenerateCode(PointerAsIntCast cast)
        {
            var target = AllocateLocal(cast);
            EmitGetLocal(cast.Value.Resolve());
            WasmModuleBuilder.EmitLocalSet(Code, target);
        }

        public virtual void GenerateCode(PointerCast cast)
        {
            var target = AllocateLocal(cast);
            EmitGetLocal(cast.Value.Resolve());
            WasmModuleBuilder.EmitLocalSet(Code, target);
        }

        public virtual void GenerateCode(AddressSpaceCast value)
        {
            var target = AllocateLocal(value);
            EmitGetLocal(value.Value.Resolve());
            WasmModuleBuilder.EmitLocalSet(Code, target);
        }

        public virtual void GenerateCode(FloatAsIntCast value)
        {
            var target = AllocateLocal(value);
            EmitGetLocal(value.Value.Resolve());
            var srcType = GetWasmTypeFromIR(value.Value.Resolve().Type);
            if (srcType == WasmOpCodes.F32) Code.Add(WasmOpCodes.I32ReinterpretF32);
            else if (srcType == WasmOpCodes.F64) Code.Add(WasmOpCodes.I64ReinterpretF64);
            WasmModuleBuilder.EmitLocalSet(Code, target);
        }

        public virtual void GenerateCode(IntAsFloatCast value)
        {
            var target = AllocateLocal(value);
            EmitGetLocal(value.Value.Resolve());
            var srcType = GetWasmTypeFromIR(value.Value.Resolve().Type);
            if (srcType == WasmOpCodes.I32) Code.Add(WasmOpCodes.F32ReinterpretI32);
            else if (srcType == WasmOpCodes.I64) Code.Add(WasmOpCodes.F64ReinterpretI64);
            WasmModuleBuilder.EmitLocalSet(Code, target);
        }

        // Predicate
        public virtual void GenerateCode(Predicate predicate)
        {
            var target = AllocateLocal(predicate, WasmOpCodes.I32);
            var condition = predicate.Condition.Resolve();
            var trueVal = predicate.TrueValue.Resolve();
            var falseVal = predicate.FalseValue.Resolve();

            EmitGetLocal(trueVal);
            EmitGetLocal(falseVal);
            EmitGetLocal(condition);
            Code.Add(WasmOpCodes.Select);
            WasmModuleBuilder.EmitLocalSet(Code, target);
        }

        // Constants & Values
        public virtual void GenerateCode(PrimitiveValue value)
        {
            var target = AllocateLocal(value, GetWasmType(value));
            var wasmType = GetWasmType(value);

            switch (wasmType)
            {
                case WasmOpCodes.I32:
                    WasmModuleBuilder.EmitI32Const(Code, value.Int32Value);
                    break;
                case WasmOpCodes.I64:
                    WasmModuleBuilder.EmitI64Const(Code, value.Int64Value);
                    break;
                case WasmOpCodes.F32:
                    WasmModuleBuilder.EmitF32Const(Code, value.Float32Value);
                    break;
                case WasmOpCodes.F64:
                    WasmModuleBuilder.EmitF64Const(Code, value.Float64Value);
                    break;
            }

            WasmModuleBuilder.EmitLocalSet(Code, target);
        }

        public virtual void GenerateCode(StringValue value)
        {
            // Strings are not used in compute kernels — no-op
        }

        public virtual void GenerateCode(NullValue value)
        {
            var target = AllocateLocal(value);
            WasmModuleBuilder.EmitI32Const(Code, 0);
            WasmModuleBuilder.EmitLocalSet(Code, target);
        }

        public virtual void GenerateCode(Parameter value)
        {
            // Parameters are handled by the kernel function generator
        }

        public virtual void GenerateCode(PhiValue value)
        {
            AllocateLocal(value, GetWasmType(value));
        }

        // Memory
        public virtual void GenerateCode(Load value)
        {
            var target = AllocateLocal(value, GetWasmType(value));
            EmitGetLocal(value.Source.Resolve());
            WasmModuleBuilder.EmitLocalSet(Code, target);
        }

        public virtual void GenerateCode(Store value) { }

        public virtual void GenerateCode(Alloca value)
        {
            var target = AllocateLocal(value);
            WasmModuleBuilder.EmitI32Const(Code, 0);
            WasmModuleBuilder.EmitLocalSet(Code, target);
        }

        public virtual void GenerateCode(MemoryBarrier barrier) { }

        public virtual void GenerateCode(LoadElementAddress value)
        {
            var target = AllocateLocal(value);
            EmitGetLocal(value.Source.Resolve());
            EmitGetLocal(value.Offset.Resolve());
            int elemSize = 4;
            if (value.Type is AddressSpaceType addrType && addrType.ElementType is PrimitiveType pt)
                elemSize = GetElementSize(pt.BasicValueType);
            WasmModuleBuilder.EmitI32Const(Code, elemSize);
            Code.Add(WasmOpCodes.I32Mul);
            Code.Add(WasmOpCodes.I32Add);
            WasmModuleBuilder.EmitLocalSet(Code, target);
        }

        public virtual void GenerateCode(LoadFieldAddress value)
        {
            var target = AllocateLocal(value);
            EmitGetLocal(value.Source.Resolve());
            WasmModuleBuilder.EmitLocalSet(Code, target);
        }

        public virtual void GenerateCode(AlignTo value)
        {
            var target = AllocateLocal(value);
            EmitGetLocal(value.Source.Resolve());
            WasmModuleBuilder.EmitLocalSet(Code, target);
        }

        public virtual void GenerateCode(AsAligned value)
        {
            var target = AllocateLocal(value);
            EmitGetLocal(value.Source.Resolve());
            WasmModuleBuilder.EmitLocalSet(Code, target);
        }

        // Structs/Fields
        public virtual void GenerateCode(StructureValue value)
        {
            var target = AllocateLocal(value);
            if (value.Count > 0)
                EmitGetLocal(value[0].Resolve());
            else
                WasmModuleBuilder.EmitI32Const(Code, 0);
            WasmModuleBuilder.EmitLocalSet(Code, target);
        }

        public virtual void GenerateCode(GetField value)
        {
            var target = AllocateLocal(value, GetWasmType(value));
            var source = value.ObjectValue.Resolve();
            EmitGetLocal(source);
            WasmModuleBuilder.EmitLocalSet(Code, target);
        }

        public virtual void GenerateCode(SetField value)
        {
            var target = AllocateLocal(value, GetWasmType(value));
            EmitGetLocal(value.Value.Resolve());
            WasmModuleBuilder.EmitLocalSet(Code, target);
        }

        // Device Constants
        public virtual void GenerateCode(GridIndexValue value) { AllocateLocal(value); }
        public virtual void GenerateCode(GroupIndexValue value) { AllocateLocal(value); }
        public virtual void GenerateCode(GridDimensionValue value) { AllocateLocal(value); }
        public virtual void GenerateCode(GroupDimensionValue value) { AllocateLocal(value); }
        public virtual void GenerateCode(WarpSizeValue value)
        {
            var target = AllocateLocal(value);
            WasmModuleBuilder.EmitI32Const(Code, 1); // Wasm: no warp
            WasmModuleBuilder.EmitLocalSet(Code, target);
        }
        public virtual void GenerateCode(LaneIdxValue value)
        {
            var target = AllocateLocal(value);
            WasmModuleBuilder.EmitI32Const(Code, 0); // Wasm: lane 0
            WasmModuleBuilder.EmitLocalSet(Code, target);
        }
        public virtual void GenerateCode(DynamicMemoryLengthValue value)
        {
            var target = AllocateLocal(value);
            WasmModuleBuilder.EmitI32Const(Code, 0);
            WasmModuleBuilder.EmitLocalSet(Code, target);
        }

        // Barriers & Atomics
        public virtual void GenerateCode(PredicateBarrier barrier) { }
        public virtual void GenerateCode(global::ILGPU.IR.Values.Barrier barrier) { }
        public virtual void GenerateCode(Broadcast broadcast)
        {
            var target = AllocateLocal(broadcast, GetWasmType(broadcast));
            EmitGetLocal(broadcast.Variable.Resolve());
            WasmModuleBuilder.EmitLocalSet(Code, target);
        }
        public virtual void GenerateCode(WarpShuffle shuffle)
        {
            var target = AllocateLocal(shuffle, GetWasmType(shuffle));
            EmitGetLocal(shuffle.Variable.Resolve());
            WasmModuleBuilder.EmitLocalSet(Code, target);
        }
        public virtual void GenerateCode(SubWarpShuffle shuffle)
        {
            var target = AllocateLocal(shuffle, GetWasmType(shuffle));
            EmitGetLocal(shuffle.Variable.Resolve());
            WasmModuleBuilder.EmitLocalSet(Code, target);
        }
        public virtual void GenerateCode(GenericAtomic atomic)
        {
            var target = AllocateLocal(atomic, GetWasmType(atomic));
            WasmModuleBuilder.EmitI32Const(Code, 0);
            WasmModuleBuilder.EmitLocalSet(Code, target);
        }
        public virtual void GenerateCode(AtomicCAS atomicCAS)
        {
            var target = AllocateLocal(atomicCAS, GetWasmType(atomicCAS));
            WasmModuleBuilder.EmitI32Const(Code, 0);
            WasmModuleBuilder.EmitLocalSet(Code, target);
        }

        // Debug
        public virtual void GenerateCode(DebugAssertOperation debug) { }
        public virtual void GenerateCode(WriteToOutput writeToOutput) { }

        // Misc
        public virtual void GenerateCode(MethodCall methodCall)
        {
            if (methodCall.Type.IsVoidType) return;
            var target = AllocateLocal(methodCall, GetWasmType(methodCall));
            WasmModuleBuilder.EmitI32Const(Code, 0);
            WasmModuleBuilder.EmitLocalSet(Code, target);
        }

        // Control Flow
        public virtual void GenerateCode(ReturnTerminator returnTerminator)
        {
            Code.Add(WasmOpCodes.Return);
        }

        public virtual void GenerateCode(UnconditionalBranch branch) { }
        public virtual void GenerateCode(IfBranch branch) { }
        public virtual void GenerateCode(SwitchBranch branch) { }
        public virtual void GenerateCode(LanguageEmitValue emit) { }

        #endregion

        #region Dispatch

        /// <summary>
        /// Generates code for a value using the visitor pattern.
        /// </summary>
        protected void GenerateCodeFor(Value value)
        {
            WasmBackend.Log($"[Wasm-IR] Visit: {value.GetType().Name} Type={value.Type} IsVoid={value.Type.IsVoidType}");

            // Skip void values (except terminators, stores, barriers)
            if (value.Type.IsVoidType &&
                !(value is TerminatorValue) &&
                !(value is Store) &&
                !(value is MemoryBarrier) &&
                !(value is global::ILGPU.IR.Values.Barrier) &&
                !(value is PredicateBarrier))
            {
                WasmBackend.Log($"[Wasm-IR] Skipping void value: {value.GetType().Name}");
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
                    GenerateCode(v);
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

                // Memory
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
                case global::ILGPU.IR.Values.MemoryBarrier v:
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
                case global::ILGPU.IR.Values.Barrier v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.PredicateBarrier v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.Broadcast v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.WarpShuffle v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.SubWarpShuffle v:
                    GenerateCode(v);
                    break;

                // Alignment
                case global::ILGPU.IR.Values.AlignTo v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.AsAligned v:
                    GenerateCode(v);
                    break;

                // Misc
                case global::ILGPU.IR.Values.Predicate v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.DebugAssertOperation v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.WriteToOutput v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.DynamicMemoryLengthValue v:
                    GenerateCode(v);
                    break;
                case global::ILGPU.IR.Values.LanguageEmitValue v:
                    GenerateCode(v);
                    break;

                default:
                    // No-op for unhandled values
                    break;
            }
        }

        #endregion
    }
}
