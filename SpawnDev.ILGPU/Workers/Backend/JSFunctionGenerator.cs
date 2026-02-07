// ---------------------------------------------------------------------------------------
//                               SpawnDev.ILGPU.Workers
//                 Web Worker Compute Library for Blazor WebAssembly
//
// File: JSFunctionGenerator.cs
//
// Generates helper JavaScript functions called by the kernel (for user-defined methods).
// ---------------------------------------------------------------------------------------

using global::ILGPU.IR;
using global::ILGPU.IR.Analyses;
using System.Text;

namespace SpawnDev.ILGPU.Workers.Backend
{
    /// <summary>
    /// Generates helper functions (non-kernel methods) in JavaScript.
    /// These are called by the kernel when user code invokes helper methods.
    /// </summary>
    public sealed class JSFunctionGenerator : JSCodeGenerator
    {
        public JSFunctionGenerator(
            in GeneratorArgs args,
            Method method,
            Allocas allocas)
            : base(args, method, allocas)
        {
        }

        /// <summary>
        /// Generates the function header.
        /// </summary>
        public override void GenerateHeader(StringBuilder builder)
        {
            // Save original builder (holds GenerateCode output) for Merge()
            var originalBuilder = Builder;
            Builder = builder;
            var method = Method;

            // Generate function name
            string functionName = $"fn_{method.Id}";

            // Generate parameter list text (bindings already set up in GenerateCode)
            var paramList = new StringBuilder();
            for (int i = 0; i < method.Parameters.Count; i++)
            {
                if (i > 0) paramList.Append(", ");
                paramList.Append($"p_{i}");
            }

            builder.AppendLine($"function {functionName}({paramList}) {{");
            PushIndent();

            // Restore the original builder so Merge() appends the function body
            Builder = originalBuilder;
        }

        /// <summary>
        /// Generates the function body.
        /// </summary>
        public override void GenerateCode()
        {
            // Bind parameters FIRST — GenerateCode runs before GenerateHeader
            // in the ILGPU pipeline, so bindings must exist before processing IR.
            var method = Method;
            for (int i = 0; i < method.Parameters.Count; i++)
            {
                var paramName = $"p_{i}";
                var variable = new Variable(paramName, GetJSType(method.Parameters[i].Type));
                Bind(method.Parameters[i], variable);
            }

            GenerateCodeInternal();

            PopIndent();
            Builder.AppendLine("}");
            Builder.AppendLine();
        }
    }
}
