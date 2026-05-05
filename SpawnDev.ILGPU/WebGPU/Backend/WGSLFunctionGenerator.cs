// ---------------------------------------------------------------------------------------
//                                 SpawnDev.ILGPU.WebGPU
//                        Copyright (c) 2024 SpawnDev Project
//
// File: WGSLFunctionGenerator.cs
//
// WGSL function generator for helper device functions.
// ---------------------------------------------------------------------------------------

using global::ILGPU;
using global::ILGPU.IR;
using global::ILGPU.IR.Analyses;
using global::ILGPU.IR.Types;
using global::ILGPU.IR.Values;
using System;
using System.Collections.Generic;
using System.Text;

namespace SpawnDev.ILGPU.WebGPU.Backend
{
    /// <summary>
    /// Function generator for helper device functions (non-kernel methods).
    /// </summary>
    internal sealed class WGSLFunctionGenerator : WGSLCodeGenerator
    {
        #region Constants

        /// <summary>
        /// Methods with these flags will be skipped during code generation.
        /// </summary>
        private const MethodFlags MethodFlagsToSkip =
            MethodFlags.External |
            MethodFlags.Intrinsic;

        #endregion

        #region Instance

        private readonly GeneratorArgs _args;

        // Sub-word state for fn-param ArrayView codegen (rc.16 Bug D phase 2,
        // 2026-05-05). Mirrors WGSLKernelFunctionGenerator's _subWord* dictionaries
        // but keyed by helper-local Parameter.Index instead of kernel-binding index.
        // Populated in ScanHelperSubWordParams() called from GenerateCode() before
        // body emission. Consumed by the LEA / Load / Store overrides below.
        private readonly Dictionary<int, int> _subWordParams = new();           // paramIdx -> elemSize (1 byte / 2 short / 2 emu-f16)
        private readonly HashSet<int> _subWordUnsignedParams = new();           // byte / ushort
        private readonly HashSet<int> _subWordFloat16Params = new();            // emulated f16 (no shader-f16)
        private readonly Dictionary<string, int> _subWordLEAVars = new();       // LEA target Variable name -> paramIdx
        private readonly Dictionary<int, string> _paramIndexToLocalVar = new(); // paramIdx -> "v_X" (the helper's `let v_X : ptr<...> = p_Y;` local name)

        // Cross-block FieldAddress inline substitution map. Helper bodies emitted as
        // multi-block switch/case state machines define LoadFieldAddress in one case
        // and use it in others; WGSL `let` is case-scoped, so a use in case N of a
        // `let` declared in case M < N is "unresolved value". The kernel pre-hoists
        // primitives + tracks LEA via _crossBlockPointerExprs; field addresses here
        // get the same treatment via a registered "deref form" expression that the
        // Load / Store overrides substitute into the rhs / lhs.
        // (rc.16 Bug D phase 2 follow-up, 2026-05-05.)
        private readonly Dictionary<string, string> _fieldAddressDerefExpr = new(); // varName -> "(*src).fieldName"

        /// <summary>
        /// Creates a new WGSL function generator.
        /// </summary>
        public WGSLFunctionGenerator(
            in GeneratorArgs args,
            Method method,
            Allocas allocas)
            : base(args, method, allocas)
        {
            _args = args;
        }

        #endregion

        #region Methods

        /// <summary>
        /// Returns the WGSL parameter-position type for a ViewType.
        /// Address space is read from the ViewType: `Shared` → `workgroup`,
        /// `Local` → `function`, `Global`/`Generic` → `storage`. Sub-word element
        /// types (Int8/Int16/Float16-emulated) on storage backing lower to
        /// `ptr&lt;storage, array&lt;atomic&lt;u32&gt;&gt;, read_write&gt;`; full-word lower to
        /// `ptr&lt;{addrSpace}, array&lt;{elem}&gt;, read_write&gt;`. Mirrors the kernel-side
        /// SetupParameterBindings sub-word detection (rc.16 fn-def codegen Bug D
        /// + 2026-05-05 phase 6 — shared-memory address space propagation).
        /// </summary>
        internal static string GetWgslViewParamType(
            ViewType viewType,
            WGSLTypeGenerator typeGenerator,
            bool hasShaderF16)
        {
            bool isSubWord = false;
            if (viewType.ElementType is PrimitiveType prim)
            {
                switch (prim.BasicValueType)
                {
                    case BasicValueType.Int8:
                    case BasicValueType.Int16:
                        isSubWord = true;
                        break;
                    case BasicValueType.Float16:
                        if (!hasShaderF16) isSubWord = true;
                        break;
                }
            }

            // Map ILGPU MemoryAddressSpace to WGSL storage class. Generic falls
            // back to storage (the typical ArrayView<T> kernel-param case).
            string addrSpace = viewType.AddressSpace switch
            {
                MemoryAddressSpace.Shared => "workgroup",
                MemoryAddressSpace.Local => "function",
                _ => "storage",
            };

            // Sub-word packing into atomic<u32> is the storage-buffer pattern
            // (CPU-side packing + atomicLoad/atomicAnd/atomicOr in WGSL). For
            // workgroup-memory sub-word views we'd need a different layout
            // (workgroup vars don't share host-side packing). Until a test
            // surfaces that case, route sub-word through the storage form.
            if (isSubWord && addrSpace == "storage")
                return "ptr<storage, array<atomic<u32>>, read_write>";

            string elemTypeName = typeGenerator[viewType.ElementType];
            // Non-storage address spaces use `read_write` access mode; WGSL
            // does not parameterize ptr<workgroup,T> with an explicit access.
            // Keep the trailing `, read_write` only for storage.
            return addrSpace == "storage"
                ? $"ptr<storage, array<{elemTypeName}>, read_write>"
                : $"ptr<{addrSpace}, array<{elemTypeName}>>";
        }

        /// <summary>
        /// Generates a header stub for the current method.
        /// </summary>
        private void GenerateHeaderStub(StringBuilder builder)
        {
            builder.Append("fn ");
            builder.Append(GetMethodName(Method));
            builder.Append("(");

            // Emit parameters
            bool first = true;
            foreach (var param in Method.Parameters)
            {
                if (!first) builder.Append(", ");
                first = false;

                string paramType;
                if (param.ParameterType is ViewType viewType)
                {
                    // Bug D fix (2026-05-05): ArrayView<T> as a fn-param emits as
                    // a real WGSL pointer to the kernel's storage binding, not the
                    // bare element type. Without this fix, helpers calling
                    // sub-word LEA on a buf param fail because `var v_X : i32 = p_X;`
                    // produces an i32 local that can't be indexed as an array.
                    paramType = GetWgslViewParamType(
                        viewType,
                        TypeGenerator,
                        Backend.HasShaderF16);
                }
                else
                {
                    paramType = TypeGenerator[param.ParameterType];
                }
                builder.Append($"p_{param.Id} : {paramType}");
            }

            builder.Append(")");

            // WGSL grammar: void-returning fns omit the return-type clause entirely.
            // `fn name(...) -> void { ... }` is rejected by the validator with
            // "unresolved type 'void'"; correct form is `fn name(...) { ... }`.
            if (!Method.ReturnType.IsVoidType)
            {
                builder.Append(" -> ");
                builder.Append(TypeGenerator[Method.ReturnType]);
            }
        }

        /// <summary>
        /// Gets the WGSL function name for the given method.
        /// Used by both the function generator (to emit the fn definition) and the
        /// kernel generator (to emit the call site) so the names match.
        /// </summary>
        internal static string GetMethodName(Method method)
        {
            var handleName = method.Handle.Name;
            if (method.HasFlags(MethodFlags.External))
            {
                return handleName;
            }
            else
            {
                // Constructor names in MSIL start with a dot
                return handleName.StartsWith(".")
                    ? handleName.Substring(1) + "_" + method.Id
                    : handleName + "_" + method.Id;
            }
        }

        /// <summary>
        /// Emits module-scope shared memory declarations.
        /// 
        /// CRITICAL: Shared memory Alloca nodes from the kernel's IR are referenced by helper
        /// functions but are NOT in the helper function's Allocas.SharedAllocations. They must
        /// be declared as var&lt;workgroup&gt; at module scope before the function body runs.
        /// We use GeneratorArgs.SharedMemoryVarNames (pre-populated from backendContext) to
        /// emit the declarations and pre-register them in valueVariables.
        /// </summary>
        public override void GenerateHeader(StringBuilder builder)
        {
            // Pre-register shared memory variable names so Load() can find them.
            // NOTE: The actual var<workgroup> declarations are emitted by the
            // KERNEL generator's GenerateHeader() — we must NOT duplicate them here.
            // The Load() interception in the base class handles reference-inequality
            // between the Alloca instance here and the one in the function body.
            foreach (var kvp in _args.SharedMemoryVarNames)
            {
                var allocaNode = kvp.Key;
                var (varName, declaration) = kvp.Value;

                // Register in valueVariables for this specific Alloca instance
                var wgslType = TypeGenerator[allocaNode.Type];
                var sharedVar = new Variable(varName, wgslType);
                valueVariables[allocaNode] = sharedVar;
                declaredVariables.Add(varName);
                if (WebGPUBackend.VerboseLogging) WebGPUBackend.Log($"[DIAG-FUNC-SHARED-REG] Registered (no emit): {varName} for method {Method.Handle.Name}_{Method.Id}");
            }
        }

        /// <summary>
        /// Generates WGSL code for the function body.
        /// </summary>
        public override void GenerateCode()
        {
            if (Method.HasFlags(MethodFlagsToSkip))
                return;

            // Skip code generation for helpers that will be inlined by the kernel generator.
            // The kernel's GenerateCode(MethodCall) handles inlining these methods directly
            // at the call site. Emitting them here as standalone functions would place their
            // code at module scope (outside fn main), causing WGSL validation errors.
            if (_args.HelperMethods.ContainsKey(Method))
            {
                if (WebGPUBackend.VerboseLogging) WebGPUBackend.Log($"[WGSL-FuncGen] Skipping code gen for inlined helper: {Method.Handle.Name}_{Method.Id}");
                return;
            }

            // Pre-scan parameters for sub-word ArrayView types so the LEA / Load /
            // Store overrides know which params need atomic-load + shift / mask
            // emission. Must run before body codegen sees any LoadElementAddress.
            // (rc.16 fn-def-codegen Bug D phase 2, 2026-05-05).
            ScanHelperSubWordParams();

            // Declare function and parameters
            GenerateHeaderStub(Builder);
            Builder.AppendLine(" {");
            PushIndent();

            // Bind parameters to local variables
            foreach (var param in Method.Parameters)
            {
                var variable = Allocate(param);
                // Bug D fix (2026-05-05): ViewType params are emitted as ptr<storage, ...>
                // in the fn signature (see GenerateHeaderStub), so the binding must use
                // `let` (pointers can't be `var`). The Variable's Type comes from
                // TypeGenerator[ViewType] which returns the element type (i32) - we have
                // to override here to keep the binding consistent with the signature.
                if (param.ParameterType is ViewType viewType)
                {
                    string ptrType = GetWgslViewParamType(
                        viewType,
                        TypeGenerator,
                        Backend.HasShaderF16);
                    AppendLine($"let {variable.Name} : {ptrType} = p_{param.Id};");
                    // Remember the helper-local var name for this param so the
                    // sub-word Load / Store overrides can emit `(*v_X)[wordIdx]`.
                    _paramIndexToLocalVar[param.Index] = variable.Name;
                    continue;
                }
                // WGSL: pointer types (ptr<workgroup, ...>, ptr<storage, ...>, ptr<function, ...>)
                // are not constructible and cannot be used with 'var'. Use 'let' instead.
                if (variable.Type.StartsWith("ptr<"))
                    AppendLine($"let {variable.Name} = p_{param.Id};");
                else
                    AppendLine($"var {variable.Name} : {variable.Type} = p_{param.Id};");
            }

            // Generate body
            GenerateCodeInternal();

            PopIndent();
            Builder.AppendLine("}");
        }

        /// <summary>
        /// Pre-scans helper parameters for sub-word ArrayView element types.
        /// Mirrors WGSLKernelFunctionGenerator.PreScanEmulatedParameters but
        /// for non-kernel methods: helpers have no KernelParamOffset and no
        /// EntryPoint.Parameters list, so we read the .NET parameter types
        /// off Method.Source.GetParameters() to detect signed-vs-unsigned.
        /// (rc.16 fn-def-codegen Bug D phase 2, 2026-05-05).
        /// </summary>
        private void ScanHelperSubWordParams()
        {
            System.Reflection.ParameterInfo[]? clrParams = null;
            try
            {
                clrParams = Method.Source?.GetParameters();
            }
            catch
            {
                // Defensive: some IR methods may have no .NET source (synthetic).
                clrParams = null;
            }

            foreach (var param in Method.Parameters)
            {
                if (param.ParameterType is not ViewType viewType) continue;
                if (viewType.ElementType is not PrimitiveType prim) continue;

                Type? clrElement = null;
                if (clrParams != null && param.Index >= 0 && param.Index < clrParams.Length)
                {
                    var clrParamType = clrParams[param.Index].ParameterType;
                    // Strip ByRef wrapper for `ref ArrayView<T>` style params.
                    if (clrParamType.IsByRef) clrParamType = clrParamType.GetElementType()!;
                    if (clrParamType.IsGenericType)
                    {
                        var genArgs = clrParamType.GetGenericArguments();
                        if (genArgs.Length > 0) clrElement = genArgs[0];
                    }
                }

                switch (prim.BasicValueType)
                {
                    case BasicValueType.Int8:
                        _subWordParams[param.Index] = 1;
                        if (clrElement == typeof(byte))
                            _subWordUnsignedParams.Add(param.Index);
                        break;
                    case BasicValueType.Int16:
                        _subWordParams[param.Index] = 2;
                        if (clrElement == typeof(ushort))
                            _subWordUnsignedParams.Add(param.Index);
                        break;
                    case BasicValueType.Float16:
                        if (!Backend.HasShaderF16)
                        {
                            _subWordParams[param.Index] = 2;
                            _subWordFloat16Params.Add(param.Index);
                        }
                        break;
                }
            }
        }

        /// <summary>
        /// LoadElementAddress on a helper's ArrayView parameter. The base class
        /// emits `&amp;source[idx]` which is wrong here because the helper's
        /// `source` is `ptr&lt;storage, array&lt;...&gt;, read_write&gt;` (from the
        /// fn signature) - WGSL requires `(*ptr)[idx]`. For sub-word element
        /// types we instead store the i32 offset and defer the atomic-load + shift +
        /// mask to GenerateCode(Load), mirroring the kernel's sub-word path.
        /// (rc.16 Bug D phase 2, 2026-05-05.)
        /// </summary>
        public override void GenerateCode(LoadElementAddress value)
        {
            if (value.Source.Resolve() is Parameter param
                && param.ParameterType is ViewType)
            {
                var target = Load(value);
                var offset = Load(value.Offset);

                if (_subWordParams.ContainsKey(param.Index))
                {
                    // Sub-word: store the element offset; Load / Store will compute
                    // word index + bit shift from this i32 at the use site.
                    _subWordLEAVars[target.Name] = param.Index;
                    AppendLine($"var {target.Name} : i32 = {offset};");
                    return;
                }

                // Full-word: helper's local source is a ptr, so we must deref before
                // indexing. WGSL: `&((*ptr)[idx])` produces ptr<storage, T>.
                var sourceVar = Load(value.Source);
                AppendLine($"let {target.Name} = &(*{sourceVar})[{offset}];");
                return;
            }

            base.GenerateCode(value);
        }

        /// <summary>
        /// LoadFieldAddress on a struct in a helper body. Mirrors the base class
        /// emit (`let v_X = &amp;(*src).fieldN;`) but ALSO registers the dereffed
        /// form so Load/Store in other switch-case blocks can substitute the
        /// expression directly instead of referencing the case-scoped `let`.
        /// (rc.16 Bug D phase 2 follow-up, 2026-05-05.)
        /// </summary>
        public override void GenerateCode(LoadFieldAddress value)
        {
            var target = Load(value);
            var source = Load(value.Source);

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

            // Build dereffed form (no `&`). Local-alloca sources are var-typed so
            // `source.field` works directly; pointer sources need `(*source).field`.
            string derefExpr = _localAllocaVarNames.Contains(source.Name)
                ? $"{source}.{fieldName}"
                : $"(*{source}).{fieldName}";

            // Register inline-substitution form. Load/Store overrides below check
            // this dictionary FIRST so cross-block uses bypass the case-scoped let.
            _fieldAddressDerefExpr[target.Name] = derefExpr;

            AppendIndent();
            Builder.Append("let ");
            Builder.Append(target.Name);
            Builder.Append(" = &");
            Builder.Append(derefExpr);
            Builder.Append(";");
            Builder.AppendLine();
        }

        /// <summary>
        /// Load on a helper's ArrayView parameter or its sub-word LEA result.
        /// Two cases: (1) Loading the parameter itself ("aliasing" the view) emits
        /// a `let` that copies the local pointer; (2) Loading from a sub-word
        /// LEA target emits the atomic-load + shift + mask + sign-extend chain
        /// using the helper's local pointer name as the binding.
        /// Mirrors WGSLKernelFunctionGenerator.GenerateCode(Load) sub-word path
        /// at lines 4560-4618.
        /// </summary>
        public override void GenerateCode(global::ILGPU.IR.Values.Load loadVal)
        {
            // Case (1): Load(viewParam) returns a pointer alias.
            if (loadVal.Source.Resolve() is Parameter param
                && param.Type is ViewType)
            {
                var target = Load(loadVal);
                var sourceVar = Load(loadVal.Source);
                // The helper's source is already a ptr<storage, ...>; aliasing it
                // is just `let target = sourceVar;`. Downstream code that does
                // `*target` or `(*target)[idx]` works because target IS the ptr.
                AppendLine($"let {target.Name} = {sourceVar};");
                return;
            }

            var source = Load(loadVal.Source);

            // Cross-block field address: substitute the dereffed expression so we
            // never reference a case-scoped `let` from another case.
            if (_fieldAddressDerefExpr.TryGetValue(source.ToString(), out var fieldDeref))
            {
                var target = Load(loadVal);
                Declare(target);
                AppendLine($"{target.Name} = {fieldDeref};");
                return;
            }

            // Case (2): Sub-word LEA result -> extract from packed atomic<u32>.
            if (_subWordLEAVars.TryGetValue(source.ToString(), out var subWordParamIdx))
            {
                var target = Load(loadVal);
                var idx = source.ToString();
                var elemSize = _subWordParams[subWordParamIdx];
                // Helper-local binding form: `(*v_X)` where v_X is the helper's
                // ptr alias for the param. The kernel uses `paramN` (a global
                // var<storage, ...>); here we deref the local ptr.
                string subWordBinding = _paramIndexToLocalVar.TryGetValue(subWordParamIdx, out var local)
                    ? $"(*{local})"
                    : $"param{subWordParamIdx}"; // defensive fallback (shouldn't happen)

                string extractExpr;
                if (elemSize == 1)
                {
                    var wordIdx = $"(u32({idx}) / 4u)";
                    var shift = $"((u32({idx}) % 4u) * 8u)";
                    var rawByte = $"((u32(atomicLoad(&{subWordBinding}[{wordIdx}])) >> {shift}) & 0xFFu)";
                    if (_subWordUnsignedParams.Contains(subWordParamIdx))
                        extractExpr = $"i32({rawByte})";                                    // byte: zero-extend
                    else
                        extractExpr = $"select(i32({rawByte}), (i32({rawByte}) - 256), ({rawByte}) >= 128u)"; // sbyte: sign-extend
                }
                else if (_subWordFloat16Params.Contains(subWordParamIdx))
                {
                    var wordIdx = $"(u32({idx}) / 2u)";
                    var shift = $"((u32({idx}) % 2u) * 16u)";
                    var rawExpr = $"((u32(atomicLoad(&{subWordBinding}[{wordIdx}])) >> {shift}) & 0xFFFFu)";
                    extractExpr = $"_f16_to_f32({rawExpr})";
                }
                else if (_subWordUnsignedParams.Contains(subWordParamIdx))
                {
                    var wordIdx = $"(u32({idx}) / 2u)";
                    var shift = $"((u32({idx}) % 2u) * 16u)";
                    var rawExpr = $"((u32(atomicLoad(&{subWordBinding}[{wordIdx}])) >> {shift}) & 0xFFFFu)";
                    extractExpr = $"i32({rawExpr})"; // ushort: zero-extend
                }
                else
                {
                    var wordIdx = $"(u32({idx}) / 2u)";
                    var shift = $"((u32({idx}) % 2u) * 16u)";
                    var rawExpr = $"((u32(atomicLoad(&{subWordBinding}[{wordIdx}])) >> {shift}) & 0xFFFFu)";
                    extractExpr = $"select(i32({rawExpr}), (i32({rawExpr}) - 65536), ({rawExpr}) >= 32768u)";
                }

                Declare(target);
                AppendLine($"{target.Name} = {extractExpr};");
                return;
            }

            base.GenerateCode(loadVal);
        }

        /// <summary>
        /// Store into a helper's sub-word LEA target. Atomic RMW on the packed
        /// u32 word: clear the target bit-range with atomicAnd, set the new value
        /// with atomicOr. Mirrors WGSLKernelFunctionGenerator.GenerateCode(Store)
        /// sub-word path at lines 4747-4791.
        /// </summary>
        public override void GenerateCode(global::ILGPU.IR.Values.Store storeVal)
        {
            var address = Load(storeVal.Target);

            // Cross-block field address: substitute the dereffed expression on the
            // lhs so we never write through a case-scoped `let` from another case.
            if (_fieldAddressDerefExpr.TryGetValue(address.ToString(), out var fieldDeref))
            {
                var val = Load(storeVal.Value);
                AppendLine($"{fieldDeref} = {val};");
                return;
            }

            if (_subWordLEAVars.TryGetValue(address.ToString(), out var storeParamIdx))
            {
                var val = Load(storeVal.Value);
                var idx = address.ToString();
                var elemSize = _subWordParams[storeParamIdx];
                string subWordBinding = _paramIndexToLocalVar.TryGetValue(storeParamIdx, out var local)
                    ? $"(*{local})"
                    : $"param{storeParamIdx}";

                if (elemSize == 1)
                {
                    var wordIdx = $"(u32({idx}) / 4u)";
                    var shift = $"((u32({idx}) % 4u) * 8u)";
                    AppendLine($"atomicAnd(&{subWordBinding}[{wordIdx}], ~(0xFFu << {shift}));");
                    AppendLine($"atomicOr(&{subWordBinding}[{wordIdx}], ((u32({val}) & 0xFFu) << {shift}));");
                }
                else if (_subWordFloat16Params.Contains(storeParamIdx))
                {
                    var wordIdx = $"(u32({idx}) / 2u)";
                    var shift = $"((u32({idx}) % 2u) * 16u)";
                    AppendLine($"atomicAnd(&{subWordBinding}[{wordIdx}], ~(0xFFFFu << {shift}));");
                    AppendLine($"atomicOr(&{subWordBinding}[{wordIdx}], ((_f32_to_f16({val}) & 0xFFFFu) << {shift}));");
                }
                else // Int16 / UInt16
                {
                    var wordIdx = $"(u32({idx}) / 2u)";
                    var shift = $"((u32({idx}) % 2u) * 16u)";
                    AppendLine($"atomicAnd(&{subWordBinding}[{wordIdx}], ~(0xFFFFu << {shift}));");
                    AppendLine($"atomicOr(&{subWordBinding}[{wordIdx}], ((u32({val}) & 0xFFFFu) << {shift}));");
                }
                return;
            }

            base.GenerateCode(storeVal);
        }

        /// <summary>
        /// AddressSpaceCast on a helper-local alloca. Mirrors
        /// WGSLKernelFunctionGenerator.GenerateCode(AddressSpaceCast): when the
        /// cast source is a function-scope `var` (in _localAllocaVarNames) and
        /// the cast target is a pointer type, register the cast result Variable
        /// with name `&amp;v_source` so call sites that pass it as a `ref`-style
        /// arg pick up the correct pointer expression directly. Without this,
        /// the base class emits `let v_X = v_source;` (a struct copy) and the
        /// downstream fn call gets a value where it expects ptr - WGSL fails
        /// with "type mismatch for argument 1".
        /// (rc.16 Bug D phase 3 follow-up, 2026-05-05.)
        /// </summary>
        public override void GenerateCode(AddressSpaceCast value)
        {
            var source = Load(value.Value);
            if (value.Type is AddressSpaceType
                && _localAllocaVarNames.Contains(source.Name))
            {
                var ptrType = TypeGenerator[value.Type];
                if (ptrType.StartsWith("ptr<"))
                {
                    valueVariables[value] = new Variable($"&{source}", ptrType);
                    return;
                }
            }
            base.GenerateCode(value);
        }

        /// <summary>
        /// MethodCall from inside a helper body. Without this override the base
        /// class falls to "Unmapped fallback" emitting `target = i32(0);` for any
        /// helper-to-helper call. We mirror WGSLKernelFunctionGenerator's
        /// non-inline branch: methods with implementation that aren't intrinsic
        /// or external get a real WGSL fn call to the standalone fn definition.
        /// (rc.16 Bug D phase 3, 2026-05-05.)
        ///
        /// The Inline-into-helper case (target is in HelperMethods) is intentionally
        /// not handled here yet - that requires the kernel's full inline machinery
        /// (200+ lines: alloca setup, structured CFG, varCounter offset, scoped
        /// declaredVariables snapshot). Codec helpers in the rc.16 test set are
        /// all NoInlining so this case does not fire. If a future test mixes Inline
        /// helpers called from NoInlining helpers, extend this branch.
        /// </summary>
        public override void GenerateCode(MethodCall methodCall)
        {
            var targetMethod = methodCall.Target;

            // Inline-into-helper case: not supported (yet). Fall through to base
            // and let the "Unmapped fallback" diagnostic surface; if the IR places
            // an Inline helper inside a NoInlining helper, we want a clear signal,
            // not silently-wrong code.
            if (_args.HelperMethods.ContainsKey(targetMethod))
            {
                base.GenerateCode(methodCall);
                return;
            }

            // Real fn call: target has a body and isn't intrinsic / external,
            // matching the kernel's EmitNonInlinedMethodCall branch.
            if (targetMethod.HasImplementation
                && !targetMethod.HasFlags(MethodFlags.External)
                && !targetMethod.HasFlags(MethodFlags.Intrinsic))
            {
                var fnName = GetMethodName(targetMethod);

                var argStrs = new List<string>(methodCall.Count);
                for (int i = 0; i < methodCall.Count; i++)
                {
                    var argVar = Load(methodCall[i]);
                    argStrs.Add(argVar.ToString());
                }
                string argList = string.Join(", ", argStrs);

                if (methodCall.Type.IsVoidType)
                {
                    AppendLine($"{fnName}({argList});");
                    return;
                }

                var target = Load(methodCall);
                Declare(target);
                AppendLine($"{target} = {fnName}({argList});");
                return;
            }

            base.GenerateCode(methodCall);
        }

        #endregion
    }
}
