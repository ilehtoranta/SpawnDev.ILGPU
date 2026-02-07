// ---------------------------------------------------------------------------------------
//                               SpawnDev.ILGPU.Workers
//                 Web Worker Compute Library for Blazor WebAssembly
//
// File: WorkersIntrinsics.cs
// ---------------------------------------------------------------------------------------

using global::ILGPU.IR.Values;

namespace SpawnDev.ILGPU.Workers.Backend
{
    /// <summary>
    /// Provides JavaScript code generation for ILGPU intrinsic operations.
    /// </summary>
    public static class WorkersIntrinsics
    {
        #region Math Intrinsics

        /// <summary>
        /// Generates JavaScript for Math.Sin.
        /// </summary>
        public static void GenerateSin(JSCodeGenerator generator, MethodCall methodCall)
        {
            var result = generator.LoadVariable(methodCall);
            var arg = generator.LoadVariable(methodCall.Nodes[0]);
            generator.AppendLine($"let {result.Name} = Math.sin({arg.Name});");
        }

        /// <summary>
        /// Generates JavaScript for Math.Cos.
        /// </summary>
        public static void GenerateCos(JSCodeGenerator generator, MethodCall methodCall)
        {
            var result = generator.LoadVariable(methodCall);
            var arg = generator.LoadVariable(methodCall.Nodes[0]);
            generator.AppendLine($"let {result.Name} = Math.cos({arg.Name});");
        }

        /// <summary>
        /// Generates JavaScript for Math.Tan.
        /// </summary>
        public static void GenerateTan(JSCodeGenerator generator, MethodCall methodCall)
        {
            var result = generator.LoadVariable(methodCall);
            var arg = generator.LoadVariable(methodCall.Nodes[0]);
            generator.AppendLine($"let {result.Name} = Math.tan({arg.Name});");
        }

        /// <summary>
        /// Generates JavaScript for Math.Sqrt.
        /// </summary>
        public static void GenerateSqrt(JSCodeGenerator generator, MethodCall methodCall)
        {
            var result = generator.LoadVariable(methodCall);
            var arg = generator.LoadVariable(methodCall.Nodes[0]);
            generator.AppendLine($"let {result.Name} = Math.sqrt({arg.Name});");
        }

        /// <summary>
        /// Generates JavaScript for Math.Abs.
        /// </summary>
        public static void GenerateAbs(JSCodeGenerator generator, MethodCall methodCall)
        {
            var result = generator.LoadVariable(methodCall);
            var arg = generator.LoadVariable(methodCall.Nodes[0]);
            generator.AppendLine($"let {result.Name} = Math.abs({arg.Name});");
        }

        /// <summary>
        /// Generates JavaScript for Math.Min.
        /// </summary>
        public static void GenerateMin(JSCodeGenerator generator, MethodCall methodCall)
        {
            var result = generator.LoadVariable(methodCall);
            var a = generator.LoadVariable(methodCall.Nodes[0]);
            var b = generator.LoadVariable(methodCall.Nodes[1]);
            generator.AppendLine($"let {result.Name} = Math.min({a.Name}, {b.Name});");
        }

        /// <summary>
        /// Generates JavaScript for Math.Max.
        /// </summary>
        public static void GenerateMax(JSCodeGenerator generator, MethodCall methodCall)
        {
            var result = generator.LoadVariable(methodCall);
            var a = generator.LoadVariable(methodCall.Nodes[0]);
            var b = generator.LoadVariable(methodCall.Nodes[1]);
            generator.AppendLine($"let {result.Name} = Math.max({a.Name}, {b.Name});");
        }

        /// <summary>
        /// Generates JavaScript for Math.Pow.
        /// </summary>
        public static void GeneratePow(JSCodeGenerator generator, MethodCall methodCall)
        {
            var result = generator.LoadVariable(methodCall);
            var baseVal = generator.LoadVariable(methodCall.Nodes[0]);
            var exp = generator.LoadVariable(methodCall.Nodes[1]);
            generator.AppendLine($"let {result.Name} = Math.pow({baseVal.Name}, {exp.Name});");
        }

        /// <summary>
        /// Generates JavaScript for Math.Log.
        /// </summary>
        public static void GenerateLog(JSCodeGenerator generator, MethodCall methodCall)
        {
            var result = generator.LoadVariable(methodCall);
            var arg = generator.LoadVariable(methodCall.Nodes[0]);
            generator.AppendLine($"let {result.Name} = Math.log({arg.Name});");
        }

        /// <summary>
        /// Generates JavaScript for Math.Exp.
        /// </summary>
        public static void GenerateExp(JSCodeGenerator generator, MethodCall methodCall)
        {
            var result = generator.LoadVariable(methodCall);
            var arg = generator.LoadVariable(methodCall.Nodes[0]);
            generator.AppendLine($"let {result.Name} = Math.exp({arg.Name});");
        }

        /// <summary>
        /// Generates JavaScript for Math.Atan2.
        /// </summary>
        public static void GenerateAtan2(JSCodeGenerator generator, MethodCall methodCall)
        {
            var result = generator.LoadVariable(methodCall);
            var y = generator.LoadVariable(methodCall.Nodes[0]);
            var x = generator.LoadVariable(methodCall.Nodes[1]);
            generator.AppendLine($"let {result.Name} = Math.atan2({y.Name}, {x.Name});");
        }

        /// <summary>
        /// Generates JavaScript for Math.Floor.
        /// </summary>
        public static void GenerateFloor(JSCodeGenerator generator, MethodCall methodCall)
        {
            var result = generator.LoadVariable(methodCall);
            var arg = generator.LoadVariable(methodCall.Nodes[0]);
            generator.AppendLine($"let {result.Name} = Math.floor({arg.Name});");
        }

        /// <summary>
        /// Generates JavaScript for Math.Ceiling.
        /// </summary>
        public static void GenerateCeiling(JSCodeGenerator generator, MethodCall methodCall)
        {
            var result = generator.LoadVariable(methodCall);
            var arg = generator.LoadVariable(methodCall.Nodes[0]);
            generator.AppendLine($"let {result.Name} = Math.ceil({arg.Name});");
        }

        /// <summary>
        /// Generates JavaScript for FusedMultiplyAdd (a * b + c).
        /// </summary>
        public static void GenerateFMA(JSCodeGenerator generator, MethodCall methodCall)
        {
            var result = generator.LoadVariable(methodCall);
            var a = generator.LoadVariable(methodCall.Nodes[0]);
            var b = generator.LoadVariable(methodCall.Nodes[1]);
            var c = generator.LoadVariable(methodCall.Nodes[2]);
            generator.AppendLine($"let {result.Name} = ({a.Name} * {b.Name} + {c.Name});");
        }

        /// <summary>
        /// Generates JavaScript for reciprocal square root (1/sqrt(x)).
        /// </summary>
        public static void GenerateRsqrt(JSCodeGenerator generator, MethodCall methodCall)
        {
            var result = generator.LoadVariable(methodCall);
            var arg = generator.LoadVariable(methodCall.Nodes[0]);
            generator.AppendLine($"let {result.Name} = (1.0 / Math.sqrt({arg.Name}));");
        }

        /// <summary>
        /// Generates JavaScript for reciprocal (1/x).
        /// </summary>
        public static void GenerateRcp(JSCodeGenerator generator, MethodCall methodCall)
        {
            var result = generator.LoadVariable(methodCall);
            var arg = generator.LoadVariable(methodCall.Nodes[0]);
            generator.AppendLine($"let {result.Name} = (1.0 / {arg.Name});");
        }

        #endregion

        #region Atomic Intrinsics

        /// <summary>
        /// Generates JavaScript for Atomics.add.
        /// </summary>
        public static void GenerateAtomicAdd(JSCodeGenerator generator, MethodCall methodCall)
        {
            var result = generator.LoadVariable(methodCall);
            var target = generator.LoadVariable(methodCall.Nodes[0]);
            var value = generator.LoadVariable(methodCall.Nodes[1]);
            generator.AppendLine($"let {result.Name} = Atomics.add({target.Name}_view, {target.Name}_index, {value.Name});");
        }

        #endregion
    }
}
