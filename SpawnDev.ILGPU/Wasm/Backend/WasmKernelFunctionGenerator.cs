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
using global::ILGPU.Backends;
using global::ILGPU.IR;
using global::ILGPU.IR.Analyses;
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
        /// Whether this kernel uses views (ArrayView).
        /// </summary>
        private bool _hasViews = false;

        public WasmKernelFunctionGenerator(
            in GeneratorArgs args,
            Method method,
            Allocas allocas)
            : base(args, method, allocas)
        {
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

            // Fixed params: globalIdx (i32), dimX (i32)
            _globalIdxLocal = _nextLocalIndex++;
            _paramCount++;

            _dimXLocal = _nextLocalIndex++;
            _paramCount++;

            // FuncParamTypes tracks Wasm type signatures for the module builder
            FuncParamTypes.Clear();
            FuncParamTypes.Add(WasmOpCodes.I32); // param 0: globalIdx
            FuncParamTypes.Add(WasmOpCodes.I32); // param 1: dimX

            // Iterate IR kernel parameters (skip implicit index at position 0 for grouped kernels)
            int startIdx = entryPoint.IsExplicitlyGrouped ? 0 : 1;

            // Map the implicit index param to globalIdx
            if (!entryPoint.IsExplicitlyGrouped && parameters.Count > 0)
            {
                var indexParam = parameters[0];
                _localMap[GetValueKey(indexParam)] = _globalIdxLocal;
                WasmBackend.Log($"[Wasm-Setup] Index param {GetValueKey(indexParam)} -> local_{_globalIdxLocal}");
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

                    _paramLocals[i] = new[] { offsetLocal, lengthLocal };
                    _localMap[GetValueKey(param)] = offsetLocal;

                    WasmBackend.Log($"[Wasm-Setup]   View: {GetValueKey(param)}=local_{offsetLocal} (length=local_{lengthLocal})");

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

                    _paramInfos.Add(new WasmParamInfo
                    {
                        Index = i,
                        Name = $"param{i}",
                        IsScalar = true,
                        WasmType = wasmType,
                    });
                }
            }

            // Store param infos for the backend
            _generatorArgs.ParamInfos = _paramInfos;

            WasmBackend.Log($"[Wasm-Setup] Final: _nextLocalIndex={_nextLocalIndex}, _paramCount={_paramCount}, FuncParamTypes={FuncParamTypes.Count}");
            WasmBackend.Log($"[Wasm-Setup] _localMap: {string.Join(", ", _localMap.Select(kv => $"{kv.Key}={kv.Value}"))}");
        }

        private bool _parametersInitialized = false;

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
        /// Generates the function body by visiting all blocks.
        /// </summary>
        public override void GenerateCode()
        {
            // CRITICAL: Set up parameter mappings FIRST, before any IR visiting.
            // ILGPU calls GenerateCode() BEFORE GenerateHeader().
            SetupParameters();

            // Visit all basic blocks
            foreach (var block in Method.Blocks)
            {
                foreach (var value in block)
                {
                    GenerateCodeFor(value);
                }

                // Handle terminator
                if (block.Terminator != null)
                {
                    GenerateCodeFor(block.Terminator);
                }
            }
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
            var target = AllocateLocal(value, GetWasmType(value));
            var source = value.Source.Resolve();
            var wasmType = GetWasmType(value);

            // Push the address (byte offset in linear memory)
            EmitGetLocal(source);

            // Emit the appropriate load instruction
            byte loadOp;
            uint align;
            switch (wasmType)
            {
                case WasmOpCodes.I64:
                    loadOp = WasmOpCodes.I64Load;
                    align = 3; // 2^3 = 8 byte alignment
                    break;
                case WasmOpCodes.F32:
                    loadOp = WasmOpCodes.F32Load;
                    align = 2; // 2^2 = 4 byte alignment
                    break;
                case WasmOpCodes.F64:
                    loadOp = WasmOpCodes.F64Load;
                    align = 3;
                    break;
                default: // I32
                    loadOp = WasmOpCodes.I32Load;
                    align = 2;
                    break;
            }

            WasmModuleBuilder.EmitLoad(Code, loadOp, align, 0);
            WasmModuleBuilder.EmitLocalSet(Code, target);
        }

        public override void GenerateCode(Store value)
        {
            var target = value.Target.Resolve();
            var storeValue = value.Value.Resolve();
            var wasmType = GetWasmTypeFromIR(storeValue.Type);

            // Push address, then value
            EmitGetLocal(target);
            EmitGetLocal(storeValue);

            byte storeOp;
            uint align;
            switch (wasmType)
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
            if (value.Type is AddressSpaceType addrType && addrType.ElementType is PrimitiveType pt)
                elemSize = GetElementSize(pt.BasicValueType);

            // addr = source + index * elemSize
            EmitGetLocal(source);
            EmitGetLocal(index);
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

            // Check if source is a Parameter that maps to an ArrayView
            if (source is global::ILGPU.IR.Values.Parameter param && IsViewType(param.Type))
            {
                int paramIdx = -1;
                for (int pi = 0; pi < Method.Parameters.Count; pi++)
                {
                    if (Method.Parameters[pi] == param) { paramIdx = pi; break; }
                }
                if (_paramLocals.TryGetValue(paramIdx, out var locals))
                {
                    switch (fieldIndex)
                    {
                        case 0: // Ptr (byte offset)
                            WasmModuleBuilder.EmitLocalGet(Code, locals[0]);
                            break;
                        case 1: // Index/Offset
                            WasmModuleBuilder.EmitI32Const(Code, 0);
                            break;
                        case 2: // Length
                        case 3:
                            WasmModuleBuilder.EmitLocalGet(Code, locals[1]);
                            break;
                        default:
                            WasmModuleBuilder.EmitI32Const(Code, 0);
                            break;
                    }
                    WasmModuleBuilder.EmitLocalSet(Code, target);
                    return;
                }
            }

            // Fallback: pass through
            EmitGetLocal(source);
            WasmModuleBuilder.EmitLocalSet(Code, target);
        }

        #endregion

        #region Override: Device Constants

        public override void GenerateCode(GroupIndexValue value)
        {
            var target = AllocateLocal(value);
            // For non-grouped kernels: groupIndex = globalIdx (each thread is its own group)
            WasmModuleBuilder.EmitLocalGet(Code, _globalIdxLocal);
            WasmModuleBuilder.EmitLocalSet(Code, target);
        }

        public override void GenerateCode(GridIndexValue value)
        {
            var target = AllocateLocal(value);
            // For non-grouped kernels: gridIndex = 0 (single group)
            WasmModuleBuilder.EmitI32Const(Code, 0);
            WasmModuleBuilder.EmitLocalSet(Code, target);
        }

        public override void GenerateCode(GroupDimensionValue value)
        {
            var target = AllocateLocal(value);
            // For non-grouped kernels: groupDim = dimX (all threads in one group)
            WasmModuleBuilder.EmitLocalGet(Code, _dimXLocal);
            WasmModuleBuilder.EmitLocalSet(Code, target);
        }

        public override void GenerateCode(GridDimensionValue value)
        {
            var target = AllocateLocal(value);
            // For non-grouped kernels: gridDim = 1
            WasmModuleBuilder.EmitI32Const(Code, 1);
            WasmModuleBuilder.EmitLocalSet(Code, target);
        }

        #endregion

        #region Helpers

        /// <summary>
        /// Checks if an IR type is an ArrayView type.
        /// </summary>
        protected bool IsViewType(TypeNode type)
        {
            if (type is StructureType structType)
            {
                var typeName = structType.ToString();
                if (typeName.Contains("ArrayView") || typeName.Contains("View"))
                    return true;
            }
            if (type is AddressSpaceType)
                return true;
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
