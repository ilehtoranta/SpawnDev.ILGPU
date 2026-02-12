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

                var paramType = TypeGenerator[param.ParameterType];
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
