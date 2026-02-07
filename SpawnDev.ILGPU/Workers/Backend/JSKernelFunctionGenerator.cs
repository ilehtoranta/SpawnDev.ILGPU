// ---------------------------------------------------------------------------------------
//                               SpawnDev.ILGPU.Workers
//                 Web Worker Compute Library for Blazor WebAssembly
//
// File: JSKernelFunctionGenerator.cs
//
// Generates the JavaScript kernel entry-point function from ILGPU IR.
// ---------------------------------------------------------------------------------------

using global::ILGPU;
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
            builder.Append("function kernel(_globalIndex");

            foreach (var binding in _parameterBindings)
            {
                builder.Append($", {binding.Name}");
            }

            builder.AppendLine(") {");
            PushIndent();

            // Setup index variable comment (maps _globalIndex to the ILGPU index parameter)
            AppendLine("// Index mapping");

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
            for (int i = 0; i < _parameterCount; i++)
            {
                var param = Method.Parameters[i];

                if (i == 0)
                {
                    // Implicit index parameter — bound to _globalIndex
                    continue;
                }

                var paramType = param.Type;
                bool isView = IsViewType(paramType);
                bool isScalar = !isView && paramType is PrimitiveType;

                string paramName = $"param{i - 1}";

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
            }
        }

        /// <summary>
        /// Sets up the index variable mapping for the implicit kernel index.
        /// </summary>
        private void SetupIndexVariable()
        {
            if (_parameterCount > 0)
            {
                var indexParam = Method.Parameters[0];
                _indexVariable = new Variable("_globalIndex", "i32");
                Bind(indexParam, _indexVariable);
                // For auto-grouped 1D kernels, the index is just _globalIndex
                AppendLine("// Index mapping");
            }
        }

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
        /// Checks if a type represents an ArrayView (view type).
        /// </summary>
        private bool IsViewType(TypeNode type)
        {
            if (type is StructureType structType)
            {
                // ArrayView<T> in ILGPU IR is a structure with a pointer field
                var typeStr = type.ToString();
                if (typeStr.Contains("ArrayView") || typeStr.Contains("View"))
                    return true;

                // Check if it has a pointer field (first field)
                if (structType.NumFields > 0)
                {
                    var firstFieldType = structType.Fields[0];
                    if (firstFieldType is PointerType)
                        return true;
                }
            }
            return false;
        }

        /// <summary>
        /// Gets the element type of an ArrayView parameter.
        /// </summary>
        private BasicValueType GetViewElementType(TypeNode type)
        {
            if (type is StructureType structType && structType.NumFields > 0)
            {
                var firstField = structType.Fields[0];
                if (firstField is PointerType ptrType)
                {
                    if (ptrType.ElementType is PrimitiveType pt)
                        return pt.BasicValueType;
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

            // Check if the source is an ArrayView (view type)
            if (IsViewType(sourceType))
            {
                switch (fieldIndex)
                {
                    case 0:
                        // Field 0: Buffer reference (the typed array itself, param is already the view)
                        AppendLine($"{target} = {source}; // view.ptr");
                        return;
                    case 1:
                        // Field 1: Index/Offset (always 0 for workers, offset handled at binding)
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
        /// In JS, we just read the variable value.
        /// </summary>
        public override void GenerateCode(global::ILGPU.IR.Values.Load loadVal)
        {
            var target = Load(loadVal);
            var source = Load(loadVal.Source);
            Declare(target);

            // Check if source is an element address (array access)
            if (loadVal.Source.Resolve() is LoadElementAddress lea)
            {
                var array = Load(lea.Source);
                var offset = Load(lea.Offset);
                AppendLine($"{target} = {array}[{offset}];");
                return;
            }

            AppendLine($"{target} = {source};");
        }

        /// <summary>
        /// Override for Store: handle pointer store.
        /// </summary>
        public override void GenerateCode(global::ILGPU.IR.Values.Store storeVal)
        {
            var address = Load(storeVal.Target);
            var val = Load(storeVal.Value);

            // Check if target is an element address (array write)
            if (storeVal.Target.Resolve() is LoadElementAddress lea)
            {
                var array = Load(lea.Source);
                var offset = Load(lea.Offset);
                AppendLine($"{array}[{offset}] = {val};");
                return;
            }

            AppendLine($"{address} = {val};");
        }

        /// <summary>
        /// Override for LoadElementAddress: JS array element access.
        /// </summary>
        public override void GenerateCode(LoadElementAddress value)
        {
            var target = Load(value);
            var source = Load(value.Source);
            var offset = Load(value.Offset);

            // For typed arrays, we need to track both the array and the index
            // so Load/Store can use them directly
            AppendIndent();
            Builder.AppendLine($"  // LoadElementAddress: {source}[{offset}]");

            // Bind the target as an accessor that Load/Store can resolve
            // The variable name encodes the array access
            var accessVar = new Variable($"{source}[{offset}]", "element_access");
            Bind(value, accessVar);
        }

        /// <summary>
        /// Override for GridIndexValue: map to the global index parameter.
        /// </summary>
        public override void GenerateCode(GridIndexValue value)
        {
            var target = Load(value);
            Declare(target);

            string dimension = value.Dimension switch
            {
                DeviceConstantDimension3D.X => "_globalIndex",
                DeviceConstantDimension3D.Y => "0", // 1D for now
                DeviceConstantDimension3D.Z => "0",
                _ => "_globalIndex"
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
