// ---------------------------------------------------------------------------------------
//                                 SpawnDev.ILGPU.WebGL
//                        Copyright (c) 2024 SpawnDev Project
//
// File: GLSLFunctionGenerator.cs
//
// Generates GLSL ES 3.0 code for helper device functions (non-kernel methods).
// ---------------------------------------------------------------------------------------

using global::ILGPU.IR;
using global::ILGPU.IR.Analyses;
using System.Text;

namespace SpawnDev.ILGPU.WebGL.Backend
{
    /// <summary>
    /// Generates GLSL ES 3.0 code for helper (non-kernel) functions.
    /// These are called by the kernel entry point or by other helpers.
    /// </summary>
    internal sealed class GLSLFunctionGenerator : GLSLCodeGenerator
    {
        public GLSLFunctionGenerator(in GeneratorArgs args, Method method, Allocas allocas)
            : base(args, method, allocas)
        {
        }

        public override void GenerateHeader(StringBuilder builder) { }

        public override void GenerateCode()
        {
            GenerateHeaderStub(Builder);
            Builder.AppendLine(" {");
            IndentLevel = 1;

            GenerateCodeInternal();

            // GLSL ES 3.0 requires all code paths to return a value.
            // Add a fallback return for non-void functions in case the IR
            // ends with a throw (which we translate to a comment/noop)
            // or other unreachable terminator. For struct return types we
            // must emit a constructor with one zero-arg per field; the
            // single-arg form `struct_N(0)` is rejected by GLSL ES with
            // "Number of constructor parameters does not match the number
            // of structure fields" (rc.16 fn-def codegen Bug D, 2026-05-05).
            string returnType = TypeGenerator[Method.ReturnType];
            if (returnType != "void")
            {
                string init;
                if (Method.ReturnType is global::ILGPU.IR.Types.StructureType structType
                    && returnType.StartsWith("struct_"))
                {
                    var sb = new StringBuilder();
                    sb.Append($"{returnType}(");
                    for (int i = 0; i < structType.NumFields; i++)
                    {
                        if (i > 0) sb.Append(", ");
                        sb.Append(GetDefaultValue(TypeGenerator[structType.Fields[i]]));
                    }
                    sb.Append(")");
                    init = sb.ToString();
                }
                else
                {
                    init = GetDefaultValue(returnType);
                }
                Builder.Append("    ");
                Builder.AppendLine($"return {init};");
            }

            IndentLevel = 0;
            Builder.AppendLine("}");
        }

        private void GenerateHeaderStub(StringBuilder builder)
        {
            string returnType = TypeGenerator[Method.ReturnType];

            builder.Append(returnType);
            builder.Append(" ");
            builder.Append(GetMethodName(Method));
            builder.Append("(");

            bool first = true;
            foreach (var param in Method.Parameters)
            {
                if (!first) builder.Append(", ");
                first = false;

                // GLSL has no pointer types: ILGPU IR `Pointer<T>` and
                // `AddressSpaceType` (the lowered shape of `ref T` / `out T`
                // params) both map to the element type via TypeGenerator.
                // For pass-by-value this means the helper would receive a
                // copy and writes wouldn't propagate back. Mark these params
                // with `inout` so the GLSL compiler treats them as
                // bidirectional reference semantics, matching C# `ref`/`out`.
                bool isRefParam = param.ParameterType is global::ILGPU.IR.Types.AddressSpaceType
                               || param.ParameterType is global::ILGPU.IR.Types.PointerType;

                var paramType = TypeGenerator[param.ParameterType];
                if (isRefParam) builder.Append("inout ");
                builder.Append(paramType);
                builder.Append(" ");
                builder.Append($"p_{param.Index}");

                // Bind the parameter's value to a variable
                var variable = new Variable($"p_{param.Index}", paramType);
                Bind(param, variable);
            }

            builder.Append(")");
        }

        public static string GetMethodName(Method method)
        {
            // Generate a unique, valid GLSL function name
            string baseName = method.Name ?? "func";
            // Clean name: replace invalid chars
            baseName = baseName.Replace(".", "_").Replace("<", "_").Replace(">", "_")
                               .Replace(",", "_").Replace(" ", "_").Replace("`", "_");
            return $"fn_{baseName}_{method.Id}";
        }
    }
}
