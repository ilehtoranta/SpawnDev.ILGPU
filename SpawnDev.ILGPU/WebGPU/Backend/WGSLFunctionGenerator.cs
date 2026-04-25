// ---------------------------------------------------------------------------------------
//                                 SpawnDev.ILGPU.WebGPU
//                        Copyright (c) 2024 SpawnDev Project
//
// File: WGSLFunctionGenerator.cs
//
// WGSL function generator for helper device functions.
// ---------------------------------------------------------------------------------------

using global::ILGPU.IR;
using global::ILGPU.IR.Analyses;
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

                var paramType = TypeGenerator[param.ParameterType];
                builder.Append($"p_{param.Id} : {paramType}");
            }

            builder.Append(") -> ");
            builder.Append(TypeGenerator[Method.ReturnType]);
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

            // Declare function and parameters
            GenerateHeaderStub(Builder);
            Builder.AppendLine(" {");
            PushIndent();

            // Bind parameters to local variables
            foreach (var param in Method.Parameters)
            {
                var variable = Allocate(param);
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

        #endregion
    }
}
