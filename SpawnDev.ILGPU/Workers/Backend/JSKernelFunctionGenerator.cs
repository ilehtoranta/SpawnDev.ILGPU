// ---------------------------------------------------------------------------------------
//                               SpawnDev.ILGPU.Workers
//                 Web Worker Compute Library for Blazor WebAssembly
//
// File: JSKernelFunctionGenerator.cs
//
// Generates the JavaScript kernel entry-point function from ILGPU IR.
// ---------------------------------------------------------------------------------------

using global::ILGPU;
using System.Linq;
using global::ILGPU.IR;
using global::ILGPU.IR.Analyses;
using global::ILGPU.IR.Analyses.ControlFlowDirection;
using global::ILGPU.IR.Analyses.TraversalOrders;
using global::ILGPU.IR.Types;
using global::ILGPU.IR.Values;
using System.Text;

namespace SpawnDev.ILGPU.Workers.Backend
{
    /// <summary>
    /// Generates the kernel entry-point function in JavaScript.
    /// Handles parameter binding, index mapping, and structured code generation.
    /// </summary>
    public sealed class JSKernelFunctionGenerator : JSCodeGenerator
    {
        private readonly GeneratorArgs _generatorArgs;

        /// <summary>
        /// Tracks the parameter bindings for this kernel.
        /// Key: parameter index (0-based, skipping implicit index param),
        /// Value: info about how to bind the parameter.
        /// </summary>
        private readonly List<ParameterBinding> _parameterBindings = new();
        /// <summary>
        /// Maps view-wrapper param names (e.g., "param0") to their associated stride param name (e.g., "param0_stride").
        /// Used by GetField to resolve field 1 on ArrayView2D/3D wrapper structs.
        /// </summary>
        private readonly Dictionary<string, string> _viewWrapperStrides = new();
        private readonly Dictionary<string, int> _viewWrapperStrideCount = new();
        /// <summary>
        /// Maps (paramName, fieldIndex) → stride parameter index.
        /// Only Int32 fields after the ViewType are mapped (these are the actual strides).
        /// Int64 fields (extents) are NOT in this map.
        /// </summary>
        private readonly Dictionary<string, Dictionary<int, int>> _viewWrapperFieldToStride = new();

        /// <summary>
        /// Tracks what the implicit index parameter maps to.
        /// </summary>
        private Variable? _indexVariable;

        /// <summary>
        /// Number of parameters (counting the implicit index as 0).
        /// </summary>
        private int _parameterCount;

        /// <summary>
        /// The set of atomic buffer parameters, for which we need Int32Array views.
        /// </summary>
        private readonly HashSet<int> _atomicBufferParams = new();

        /// <summary>
        /// Represents a bound parameter for the kernel.
        /// </summary>
        public record ParameterBinding
        {
            public int Index { get; init; }
            public string Name { get; init; } = "";
            public bool IsView { get; init; }
            public bool IsScalar { get; init; }
            public BasicValueType ScalarType { get; init; }
            public string TypedArrayType { get; init; } = "Int32Array";
            public int ElementSize { get; init; } = 4;
        }

        public JSKernelFunctionGenerator(
            in GeneratorArgs args,
            Method method,
            Allocas allocas)
            : base(args, method, allocas)
        {
            _generatorArgs = args;
            _parameterCount = Method.Parameters.Count;
        }

        /// <summary>
        /// Gets the list of parameter bindings (for the accelerator to set up arguments).
        /// </summary>
        public IReadOnlyList<ParameterBinding> ParameterBindings => _parameterBindings;

        /// <summary>
        /// Generates the kernel function header and parameter bindings.
        ///
        /// The generated JS function looks like:
        /// function kernel(_globalIndex, param0, param1, ...) {
        ///     ...
        /// }
        /// </summary>
        public override void GenerateHeader(StringBuilder builder)
        {
            WorkersBackend.Log("[JS] GenerateHeader: Generating kernel function header");

            // NOTE: SetupParameterBindings() and SetupIndexVariable() have already been
            // called in GenerateCode() (which runs first in the ILGPU pipeline).

            // IMPORTANT: Save the original builder (which holds GenerateCode's output)
            // so that Merge() can correctly append it later.
            var originalBuilder = Builder;
            Builder = builder;

            // Generate the JS function signature using already-populated parameter bindings
            // _dimX/_dimY/_dimZ are total grid dimensions for multi-D index decomposition
            // _groupDimX/_groupDimY/_groupDimZ are group dimensions from KernelConfig
            builder.Append("function kernel(_globalIndex, _dimX, _dimY, _dimZ, _groupDimX, _groupDimY, _groupDimZ");

            foreach (var binding in _parameterBindings)
            {
                builder.Append($", {binding.Name}");
            }

            builder.AppendLine(") {");
            PushIndent();

            // Index decomposition: flat _globalIndex → X/Y/Z components
            AppendLine("// Index decomposition (flat → multi-D)");
            AppendLine("var _indexX = _globalIndex % _dimX;");
            AppendLine("var _indexY = Math.floor(_globalIndex / _dimX) % _dimY;");
            AppendLine("var _indexZ = Math.floor(_globalIndex / (_dimX * _dimY));");

            // Create struct index object for 2D/3D kernels (accessed via GetField .f0/.f1/.f2)
            if (_indexDimensions >= 2)
            {
                AppendLine("var _idx_struct = { f0: _indexX, f1: _indexY, f2: _indexZ };");
            }
            AppendLine("");

            // Device constants: per-dimension group/grid indices and dimensions
            // Group index = position within the group
            AppendLine("var _groupIndexX = _globalIndex % _groupDimX;");
            AppendLine("var _groupIndexY = Math.floor(_globalIndex / _groupDimX) % _groupDimY;");
            AppendLine("var _groupIndexZ = Math.floor(_globalIndex / (_groupDimX * _groupDimY)) % _groupDimZ;");
            // Grid index = which workgroup this thread belongs to (per dimension)
            AppendLine("var _gridIndexX = Math.floor(_indexX / _groupDimX);");
            AppendLine("var _gridIndexY = Math.floor(_indexY / _groupDimY);");
            AppendLine("var _gridIndexZ = Math.floor(_indexZ / _groupDimZ);");
            // Grid dimensions = number of workgroups per dimension
            AppendLine("var _gridDimX = Math.ceil(_dimX / _groupDimX);");
            AppendLine("var _gridDimY = Math.ceil(_dimY / _groupDimY);");
            AppendLine("var _gridDimZ = Math.ceil(_dimZ / _groupDimZ);");
            // For base class compatibility (1D fallbacks)
            AppendLine("var _groupIndex = _groupIndexX;");
            AppendLine("var _groupDim = _groupDimX;");
            AppendLine("var _gridIndex = _gridIndexX;");
            AppendLine("var _gridDim = _gridDimX;");
            AppendLine("");

            // Emit utility functions (reinterpret casts)
            EmitUtilityFunctions();

            // Restore the original builder so Merge() appends the kernel body
            Builder = originalBuilder;
        }

        /// <summary>
        /// Generates the function body code.
        /// </summary>
        public override void GenerateCode()
        {
            WorkersBackend.Log("[JS] GenerateCode: Generating kernel function body");

            // Setup parameter bindings FIRST — GenerateCode runs before GenerateHeader
            // in the ILGPU pipeline, so we need the bindings before processing the IR.
            SetupParameterBindings();
            SetupIndexVariable();

            var blocks = Method.Blocks;

            // Setup local allocations
            SetupAllocations(Allocas.LocalAllocations, MemoryAddressSpace.Local);

            // Setup shared memory allocations (required for SharedMemory.Allocate kernels)
            SetupAllocations(Allocas.SharedAllocations, MemoryAddressSpace.Shared);
            SetupAllocations(Allocas.DynamicSharedAllocations, MemoryAddressSpace.Shared);

            // Single block: no control flow needed
            if (blocks.Count == 1)
            {
                IsStateMachineActive = false;
                var singleBlock = blocks.First();
                foreach (var valueEntry in singleBlock)
                {
                    GenerateCodeFor(valueEntry.Value);
                }
                // The Terminator (branch/return) is stored separately from the block's values
                if (singleBlock.Terminator != null)
                    GenerateCodeFor(singleBlock.Terminator);
            }
            else
            {
                // Multiple blocks: use state machine
                GenerateStateMachineCode(blocks);
            }

            // Close the function body
            PopIndent();
            Builder.AppendLine("}");
        }

        #region Parameter Setup

        /// <summary>
        /// Analyzes the kernel method parameters and creates bindings.
        /// Parameter 0 is the implicit index (skipped in the signature).
        /// </summary>
        private void SetupParameterBindings()
        {
            WorkersBackend.Log($"[JS] SetupParameterBindings: {_parameterCount} total IR params");
            for (int i = 0; i < _parameterCount; i++)
            {
                var param = Method.Parameters[i];
                WorkersBackend.Log($"[JS]   Param {i}: Type={param.Type}, TypeClass={param.Type.GetType().Name}");
                if (param.Type is StructureType st)
                {
                    for (int f = 0; f < st.NumFields; f++)
                        WorkersBackend.Log($"[JS]     Field {f}: {st.Fields[f]} ({st.Fields[f].GetType().Name})");
                }

                if (i == 0)
                {
                    // Implicit index parameter — bound to _globalIndex
                    continue;
                }

                var paramType = param.Type;
                bool isDirectView = IsDirectView(paramType);
                bool isWrapper = IsViewWrapper(paramType);
                bool isView = isDirectView || isWrapper;
                bool isScalar = !isView && paramType is PrimitiveType;

                string paramName = $"param{i - 1}";
                WorkersBackend.Log($"[JS]   -> Binding as: name={paramName}, isView={isView}, isDirectView={isDirectView}, isWrapper={isWrapper}, isScalar={isScalar}");

                // Build binding with all properties at once
                var scalarType = BasicValueType.Int32;
                var typedArrayType = "Int32Array";
                var elementSize = 4;

                if (isScalar && paramType is PrimitiveType pt)
                {
                    scalarType = pt.BasicValueType;
                    var jsInfo = JSTypeGenerator.GetJSType(pt.BasicValueType);
                    elementSize = jsInfo.ElementSize;
                    typedArrayType = jsInfo.TypedArrayName;
                }
                else if (isView)
                {
                    var elemType = GetViewElementType(paramType);
                    var jsInfo = JSTypeGenerator.GetJSType(elemType);
                    typedArrayType = jsInfo.TypedArrayName;
                    elementSize = jsInfo.ElementSize;
                }

                var binding = new ParameterBinding
                {
                    Index = i - 1,
                    Name = paramName,
                    IsView = isView,
                    IsScalar = isScalar,
                    ScalarType = scalarType,
                    TypedArrayType = typedArrayType,
                    ElementSize = elementSize,
                };

                _parameterBindings.Add(binding);

                // Bind the ILGPU parameter to our JS variable
                var variable = new Variable(paramName, isView ? "view" : "scalar");
                Bind(param, variable);

                // For view-wrapper structs (ArrayView2D/3D), the C# marshaling sends
                // stride value(s) after the buffer. ILGPU flattens the wrapper struct:
                // 2D: { ViewType, Int64(extX), Int64(extY), Int32(YStride) }
                // 3D: { ViewType, Int64(extX), Int64(extY), Int64(extZ), Int32(YStride), Int32(ZStride) }
                // We detect stride fields by type: Int32 fields after ViewType are strides.
                if (isWrapper && paramType is StructureType wrapperStruct)
                {
                    WorkersBackend.Log($"[JS]   -> ViewWrapper analysis: NumFields={wrapperStruct.NumFields}");
                    
                    // Find Int32 fields (these are strides). Build field-index → stride-index map.
                    var fieldToStride = new Dictionary<int, int>();
                    int strideIdx = 0;
                    for (int f = 1; f < wrapperStruct.NumFields; f++)
                    {
                        var ft = wrapperStruct.Fields[f];
                        WorkersBackend.Log($"[JS]     Field {f}: {ft} ({ft.GetType().Name})");
                        if (ft is PrimitiveType fieldPrimType && fieldPrimType.BasicValueType == BasicValueType.Int32)
                        {
                            fieldToStride[f] = strideIdx++;
                        }
                    }
                    int strideFieldCount = strideIdx; // 1 for 2D, 2 for 3D
                    WorkersBackend.Log($"[JS]   -> Detected {strideFieldCount} stride fields (Int32)");

                    var strideNames = new List<string>();
                    for (int s = 0; s < strideFieldCount; s++)
                    {
                        string strideName = $"{paramName}_stride{s}";
                        strideNames.Add(strideName);

                        var strideBinding = new ParameterBinding
                        {
                            Index = binding.Index + 1 + s,
                            Name = strideName,
                            IsView = false,
                            IsScalar = true,
                            ScalarType = BasicValueType.Int32,
                            TypedArrayType = "Int32Array",
                            ElementSize = 4,
                        };
                        _parameterBindings.Add(strideBinding);
                        WorkersBackend.Log($"[JS]   -> Added stride binding: {strideName} for wrapper {paramName}");
                    }

                    // Store stride info for GetField resolution
                    _viewWrapperStrides[paramName] = string.Join(",", strideNames);
                    _viewWrapperStrideCount[paramName] = strideFieldCount;
                    _viewWrapperFieldToStride[paramName] = fieldToStride;
                }
            }
        }

        /// <summary>
        /// Sets up the index variable mapping for the implicit kernel index.
        /// For 1D: binds to scalar _globalIndex.
        /// For 2D/3D: binds to a JS object { f0: _indexX, f1: _indexY [, f2: _indexZ] }.
        /// </summary>
        private void SetupIndexVariable()
        {
            if (_parameterCount > 0)
            {
                var indexParam = Method.Parameters[0];
                var indexType = indexParam.Type;

                // Check if the index is a struct (Index2D = {i32,i32}, Index3D = {i32,i32,i32})
                if (indexType is StructureType structType && structType.NumFields >= 2)
                {
                    // Multi-D index: bind to a JS struct object
                    _indexVariable = new Variable("_idx_struct", "struct");
                    _indexDimensions = structType.NumFields;
                    Bind(indexParam, _indexVariable);
                    WorkersBackend.Log($"[JS] SetupIndexVariable: {_indexDimensions}D index (struct with {structType.NumFields} fields)");
                }
                else
                {
                    // Scalar index: bind to the flat _globalIndex directly.
                    // ILGPU auto-grouped kernels receive the full linear index and
                    // compute 2D/3D decomposition in the IR body via modular arithmetic.
                    // Binding to _indexX (which is _globalIndex % _dimX) would break
                    // multi-D kernels where the IR expects the full linear index.
                    _indexVariable = new Variable("_globalIndex", "i32");
                    _indexDimensions = 1;
                    Bind(indexParam, _indexVariable);
                    WorkersBackend.Log("[JS] SetupIndexVariable: scalar index (bound to _globalIndex)");
                }
                AppendLine("// Index mapping");
            }
        }

        /// <summary>
        /// Number of index dimensions (1, 2, or 3).
        /// </summary>
        private int _indexDimensions = 1;


        /// <summary>
        /// Emits helper utility functions for reinterpret casts.
        /// </summary>
        private void EmitUtilityFunctions()
        {
            AppendLine("// Utility: reinterpret cast helpers");
            AppendLine("const _castBuf = new ArrayBuffer(8);");
            AppendLine("const _castF32 = new Float32Array(_castBuf);");
            AppendLine("const _castI32 = new Int32Array(_castBuf);");
            AppendLine("const _castF64 = new Float64Array(_castBuf);");
            AppendLine("function _floatAsInt(f) { _castF32[0] = f; return _castI32[0]; }");
            AppendLine("function _intAsFloat(i) { _castI32[0] = i; return _castF32[0]; }");
            AppendLine("");
        }

        #endregion

        #region Type Helpers

        /// <summary>
        /// Checks if a type represents any ArrayView-like type (direct or wrapper).
        /// Used for parameter classification.
        /// </summary>
        private bool IsViewType(TypeNode type)
        {
            return IsDirectView(type) || IsViewWrapper(type);
        }

        /// <summary>
        /// Checks if a type is a direct ArrayView - represented as ViewType in ILGPU IR.
        /// This is the inner view that maps directly to a typed array in JS.
        /// </summary>
        private bool IsDirectView(TypeNode type)
        {
            return type is ViewType;
        }

        /// <summary>
        /// Checks if a type is an ArrayView2D/3D wrapper struct.
        /// These are StructureType whose first field is a ViewType (the inner ArrayView)
        /// and subsequent fields contain stride info.
        /// </summary>
        private bool IsViewWrapper(TypeNode type)
        {
            if (type is StructureType structType && structType.NumFields >= 2)
            {
                return structType.Fields[0] is ViewType;
            }
            return false;
        }

        /// <summary>
        /// Gets the element type of an ArrayView parameter.
        /// Works for both direct ViewType and wrapper StructureType containing a ViewType.
        /// </summary>
        private BasicValueType GetViewElementType(TypeNode type)
        {
            // Direct view: ViewType has ElementType property
            if (type is ViewType viewType)
            {
                if (viewType.ElementType is PrimitiveType pt)
                    return pt.BasicValueType;
            }

            // Wrapper: first field is ViewType
            if (type is StructureType structType && structType.NumFields > 0)
            {
                if (structType.Fields[0] is ViewType innerView)
                {
                    if (innerView.ElementType is PrimitiveType innerPt)
                        return innerPt.BasicValueType;
                }
            }

            return BasicValueType.Int32; // Default
        }

        #endregion

        #region Code Generation Overrides

        /// <summary>
        /// Override for GetField: handle ArrayView field decomposition.
        /// ArrayView has: field 0 = pointer (buffer ref), field 1 = index/offset, field 2+ = length.
        /// </summary>
        public override void GenerateCode(GetField value)
        {
            var target = Load(value);
            var source = Load(value.ObjectValue);
            int fieldIndex = value.FieldSpan.Index;
            Declare(target);

            var sourceType = value.ObjectValue.Type;

            // Case 1: View-wrapper struct (ArrayView2D/3D)
            // Field 0 = inner ArrayView (the typed array itself)
            // Field 1 = stride struct (from the stride scalar param)
            if (IsViewWrapper(sourceType))
            {
                switch (fieldIndex)
                {
                    case 0:
                        // The inner ArrayView<T> is the typed array itself
                        AppendLine($"{target} = {source}; // view-wrapper.innerView");
                        return;
                    default:
                        // Field N>0 = stride or extent data.
                        // Int32 fields are strides; Int64 fields are extents (bounds checking).
                        // Use field-to-stride map to get the correct stride parameter.
                        if (_viewWrapperFieldToStride.TryGetValue(source.Name, out var ftMap) &&
                            ftMap.TryGetValue(fieldIndex, out var sIdx))
                        {
                            // This is a stride field — map to the specific stride parameter
                            var names = _viewWrapperStrides[source.Name].Split(',');
                            if (sIdx < names.Length)
                            {
                                AppendLine($"{target} = {names[sIdx]}; // view-wrapper.stride[{sIdx}]");
                            }
                            else
                            {
                                AppendLine($"{target} = 0; // view-wrapper.stride (out of range: {sIdx})");
                            }
                        }
                        else if (_viewWrapperStrides.TryGetValue(source.Name, out var strideInfo))
                        {
                            // Non-stride field (Int64 extent) — use first stride as fallback
                            // (these are used for bounds checking and generally fine)
                            var names = strideInfo.Split(',');
                            AppendLine($"{target} = {names[0]}; // view-wrapper.extent (field {fieldIndex})");
                        }
                        else
                        {
                            AppendLine($"{target} = 0; // view-wrapper.stride (fallback)");
                        }
                        return;
                }
            }

            // Case 2: Direct ArrayView<T> (first field = pointer)
            if (IsDirectView(sourceType))
            {
                switch (fieldIndex)
                {
                    case 0:
                        // Field 0: Buffer reference (the typed array itself)
                        AppendLine($"{target} = {source}; // view.ptr");
                        return;
                    case 1:
                        // Field 1: Index/Offset (always 0 for workers)
                        AppendLine($"{target} = 0; // view.offset");
                        return;
                    default:
                        // Field 2+: Length
                        AppendLine($"{target} = {source}.length; // view.length");
                        return;
                }
            }

            // Default struct field access
            AppendLine($"{target} = {source}.f{fieldIndex};");
        }

        /// <summary>
        /// Override for Load: handle pointer dereference.
        /// For struct elements in typed arrays, creates a JS object from DataView reads.
        /// </summary>
        public override void GenerateCode(global::ILGPU.IR.Values.Load loadVal)
        {
            var target = Load(loadVal);
            var source = Load(loadVal.Source);
            Declare(target);

            // Check dictionary for element address (array access) - reliable across all IR opt levels
            if (_elementAddresses.TryGetValue(source.Name, out var elemAddr))
            {
                // Check if the element type is a struct (non-primitive, non-pointer)
                if (_elementStructTypes.TryGetValue(source.Name, out var structType))
                {
                    // Struct element: read fields via DataView for type-correct access
                    int structByteSize = GetStructByteSize(structType);
                    string dvExpr = GetOrCreateDataView($"{elemAddr.Array}");
                    int byteOffset = 0;
                    string expr = BuildStructLoadExpression(structType, dvExpr, $"{elemAddr.Index}", structByteSize, ref byteOffset);
                    AppendLine($"{target} = {expr};");
                    return;
                }

                // Primitive element: simple array access
                AppendLine($"{target} = {elemAddr.Array}[{elemAddr.Index}];");
                return;
            }

            // Fallback: check if source is a field address (struct.field reference)
            if (_fieldAddresses.TryGetValue(source.Name, out var fieldAddr))
            {
                AppendLine($"{target} = {fieldAddr.Struct}.f{fieldAddr.FieldIndex};");
                return;
            }

            AppendLine($"{target} = {source};");
        }

        /// <summary>
        /// Override for Store: handle pointer store.
        /// For struct elements in typed arrays, decomposes JS object into DataView writes.
        /// </summary>
        public override void GenerateCode(global::ILGPU.IR.Values.Store storeVal)
        {
            var address = Load(storeVal.Target);
            var val = Load(storeVal.Value);

            // Check dictionary for element address (array write) - reliable across all IR opt levels
            if (_elementAddresses.TryGetValue(address.Name, out var elemAddr))
            {
                // Check if the element type is a struct (non-primitive, non-pointer)
                if (_elementStructTypes.TryGetValue(address.Name, out var structType))
                {
                    // Struct element: write fields via DataView for type-correct access
                    int structByteSize = GetStructByteSize(structType);
                    string dvExpr = GetOrCreateDataView($"{elemAddr.Array}");
                    int byteOffset = 0;
                    EmitStructStore(structType, dvExpr, $"{elemAddr.Index}", $"{val}", structByteSize, ref byteOffset);
                    return;
                }

                // Primitive element: simple array write
                AppendLine($"{elemAddr.Array}[{elemAddr.Index}] = {val};");
                return;
            }

            // Fallback: check if target is a field address (struct.field reference)
            if (_fieldAddresses.TryGetValue(address.Name, out var fieldAddr))
            {
                AppendLine($"{fieldAddr.Struct}.f{fieldAddr.FieldIndex} = {val};");
                return;
            }

            AppendLine($"{address} = {val};");
        }

        /// <summary>
        /// Tracks created DataView variables for array parameters to avoid re-creating them.
        /// </summary>
        private readonly Dictionary<string, string> _dataViews = new();

        /// <summary>
        /// Gets or creates a DataView variable for the given typed array parameter.
        /// The DataView is created once and cached for reuse.
        /// </summary>
        private string GetOrCreateDataView(string arrayName)
        {
            if (!_dataViews.TryGetValue(arrayName, out var dvName))
            {
                dvName = $"_dv_{arrayName}";
                _dataViews[arrayName] = dvName;
                // Emit the DataView creation (will appear before the current line in the generated JS)
                // Use buffer/byteOffset/byteLength from the typed array
                AppendLine($"var {dvName} = new DataView({arrayName}.buffer, {arrayName}.byteOffset, {arrayName}.byteLength);");
            }
            return dvName;
        }

        /// <summary>
        /// Builds a recursive JS expression for reading a struct from a buffer using DataView.
        /// Uses byte-level access to correctly handle mixed-type structs (e.g., float + int).
        /// For OuterStruct { InnerStruct { float Val }, int ID }:
        ///   => { f0: { f0: _dv.getFloat32(i*8+0, true) }, f1: _dv.getInt32(i*8+4, true) }
        /// </summary>
        private static string BuildStructLoadExpression(StructureType structType, string dvExpr, string indexExpr, int structByteSize, ref int byteOffset)
        {
            var fields = new List<string>();
            for (int i = 0; i < structType.NumFields; i++)
            {
                var fieldType = structType.Fields[i];
                if (fieldType is StructureType nestedStruct)
                {
                    // Recurse for nested struct
                    string nestedExpr = BuildStructLoadExpression(nestedStruct, dvExpr, indexExpr, structByteSize, ref byteOffset);
                    fields.Add($"f{i}: {nestedExpr}");
                }
                else
                {
                    // Primitive field — read via DataView at byte offset
                    var (getter, fieldSize) = GetDataViewAccessor(fieldType);
                    fields.Add($"f{i}: {dvExpr}.{getter}({indexExpr} * {structByteSize} + {byteOffset}, true)");
                    byteOffset += fieldSize;
                }
            }
            return $"{{ {string.Join(", ", fields)} }}";
        }

        /// <summary>
        /// Emits JS statements to write a struct's fields to a buffer using DataView.
        /// Uses byte-level access to correctly handle mixed-type structs.
        /// For OuterStruct { InnerStruct { float Val }, int ID }:
        ///   => _dv.setFloat32(i*8+0, val.f0.f0, true);
        ///      _dv.setInt32(i*8+4, val.f1, true);
        /// </summary>
        private void EmitStructStore(StructureType structType, string dvExpr, string indexExpr, string valueExpr, int structByteSize, ref int byteOffset)
        {
            for (int i = 0; i < structType.NumFields; i++)
            {
                var fieldType = structType.Fields[i];
                string fieldExpr = $"{valueExpr}.f{i}";
                if (fieldType is StructureType nestedStruct)
                {
                    // Recurse for nested struct
                    EmitStructStore(nestedStruct, dvExpr, indexExpr, fieldExpr, structByteSize, ref byteOffset);
                }
                else
                {
                    // Primitive field — write via DataView at byte offset
                    var (setter, fieldSize) = GetDataViewAccessor(fieldType);
                    AppendLine($"{dvExpr}.{setter.Replace("get", "set")}({indexExpr} * {structByteSize} + {byteOffset}, {fieldExpr}, true);");
                    byteOffset += fieldSize;
                }
            }
        }

        /// <summary>
        /// Returns the DataView getter method name and byte size for a primitive type.
        /// </summary>
        private static (string Getter, int ByteSize) GetDataViewAccessor(TypeNode fieldType)
        {
            if (fieldType is PrimitiveType pt)
            {
                return pt.BasicValueType switch
                {
                    BasicValueType.Float32 => ("getFloat32", 4),
                    BasicValueType.Float64 => ("getFloat64", 8),
                    BasicValueType.Int8 => ("getInt8", 1),
                    BasicValueType.Int16 => ("getInt16", 2),
                    BasicValueType.Int32 => ("getInt32", 4),
                    BasicValueType.Int64 => ("getBigInt64", 8),
                    BasicValueType.Int1 => ("getInt32", 4),   // Booleans stored as 32-bit
                    _ => ("getInt32", 4)
                };
            }
            return ("getInt32", 4); // Default fallback
        }

        /// <summary>
        /// Computes the total byte size of a struct type (recursively for nested structs).
        /// </summary>
        private static int GetStructByteSize(StructureType structType)
        {
            int size = 0;
            for (int i = 0; i < structType.NumFields; i++)
            {
                var fieldType = structType.Fields[i];
                if (fieldType is StructureType nestedStruct)
                    size += GetStructByteSize(nestedStruct);
                else
                    size += GetDataViewAccessor(fieldType).ByteSize;
            }
            return size;
        }

        /// <summary>
        /// Tracks struct element types for typed array access.
        /// When a LoadElementAddress points to a struct element (not a view),
        /// we track the struct type so Load/Store can use strided access.
        /// </summary>
        private readonly Dictionary<string, StructureType> _elementStructTypes = new();

        /// <summary>
        /// Override for LoadElementAddress: JS array element access.
        /// Populates _elementAddresses dictionary for reliable Load/Store resolution.
        /// </summary>
        public override void GenerateCode(LoadElementAddress value)
        {
            var target = Load(value);
            var source = Load(value.Source);
            var offset = Load(value.Offset);

            // Bind for variable resolution FIRST — this determines the name used by Load/Store
            var accessVar = new Variable($"{source}[{offset}]", "element_access");
            Bind(value, accessVar);

            // Track in dictionary using the BOUND variable name so Load/Store lookups match
            _elementAddresses[accessVar.Name] = (source, offset);

            // Track struct element type for strided access
            var elemType = GetPointerElementType(value.Source);
            if (elemType is StructureType structType && !IsViewType(elemType))
            {
                _elementStructTypes[accessVar.Name] = structType;
            }

            WorkersBackend.Log($"[JS] LoadElementAddress: {accessVar.Name} => {source}[{offset}]");
        }

        /// <summary>
        /// Gets the element type that a pointer/view source points to.
        /// </summary>
        private TypeNode? GetPointerElementType(Value source)
        {
            var type = source.Type;
            if (type is PointerType ptrType)
                return ptrType.ElementType;
            // ViewType<T> — the element type is directly available
            if (type is ViewType viewType)
                return viewType.ElementType;
            // For AddressSpaceView or similar, check if the resolved value has more info
            if (source.Resolve() is GetField gf && gf.ObjectValue.Type is StructureType st)
            {
                // The pointer field of an ArrayView
                if (st.NumFields > 0 && st.Fields[0] is PointerType viewPtr)
                    return viewPtr.ElementType;
            }
            return null;
        }

        /// <summary>
        /// Counts the total number of primitive fields in a type (recursively for nested structs).
        /// For a flat struct like MyPoint { float X, float Y }, returns 2.
        /// For a nested struct like Outer { Inner inner, int id } where Inner { float val }, returns 2.
        /// </summary>
        private static int GetPrimitiveFieldCount(TypeNode type)
        {
            if (type is PrimitiveType)
                return 1;
            if (type is StructureType structType)
            {
                int count = 0;
                for (int i = 0; i < structType.NumFields; i++)
                {
                    count += GetPrimitiveFieldCount(structType.Fields[i]);
                }
                return count > 0 ? count : 1;
            }
            return 1; // Default: treat as single element
        }

        /// <summary>
        /// Override for GridIndexValue: map to the workgroup index (not global index).
        /// Grid.IdxX is the workgroup index; ILGPU computes GlobalIndex as
        /// Group.IdxX + Grid.IdxX * Group.DimX.
        /// </summary>
        public override void GenerateCode(GridIndexValue value)
        {
            var target = Load(value);
            Declare(target);

            string dimension = value.Dimension switch
            {
                DeviceConstantDimension3D.X => "_gridIndexX",
                DeviceConstantDimension3D.Y => "_gridIndexY",
                DeviceConstantDimension3D.Z => "_gridIndexZ",
                _ => "_gridIndexX"
            };

            AppendLine($"{target} = {dimension};");
        }

        /// <summary>
        /// Override for GridDimensionValue: return number of workgroups per dimension.
        /// </summary>
        public override void GenerateCode(GridDimensionValue value)
        {
            var target = Load(value);
            Declare(target);

            string dimension = value.Dimension switch
            {
                DeviceConstantDimension3D.X => "_gridDimX",
                DeviceConstantDimension3D.Y => "_gridDimY",
                DeviceConstantDimension3D.Z => "_gridDimZ",
                _ => "_gridDimX"
            };

            AppendLine($"{target} = {dimension};");
        }

        /// <summary>
        /// Override for ReturnTerminator: close the kernel.
        /// </summary>
        public override void GenerateCode(ReturnTerminator value)
        {
            if (IsStateMachineActive)
            {
                if (!value.IsVoidReturn)
                {
                    var retVal = Load(value.ReturnValue);
                    AppendLine($"_return_val = {retVal};");
                }
                AppendLine("_block = -1; continue;");
            }
            else
            {
                if (!value.IsVoidReturn)
                {
                    var retVal = Load(value.ReturnValue);
                    AppendLine($"return {retVal};");
                }
                else
                {
                    AppendLine("return;");
                }
            }
        }

        #endregion

        #region State Machine

        /// <summary>
        /// Generates state-machine code for multi-block kernels.
        /// </summary>
        private void GenerateStateMachineCode(BasicBlockCollection<ReversePostOrder, Forwards> blocks)
        {
            IsStateMachineActive = true;

            // Save position for deferred variable declarations
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

            // Insert deferred variable declarations
            if (VariableBuilder.Length > 0)
            {
                Builder.Insert(deferredInsertPosition, VariableBuilder.ToString());
            }
        }

        #endregion
    }
}
