// ---------------------------------------------------------------------------------------
//                               SpawnDev.ILGPU.Workers
//                 Web Worker Compute Library for Blazor WebAssembly
//
// File: JSTypeGenerator.cs
// ---------------------------------------------------------------------------------------

using global::ILGPU;
using global::ILGPU.IR.Types;

namespace SpawnDev.ILGPU.Workers.Backend
{
    /// <summary>
    /// Maps ILGPU IR types to JavaScript TypedArray constructors and element types.
    /// </summary>
    public sealed class JSTypeGenerator
    {
        /// <summary>
        /// Information about a JavaScript typed array type mapping.
        /// </summary>
        public readonly struct JSTypeInfo
        {
            /// <summary>
            /// The JavaScript TypedArray constructor name (e.g., "Int32Array").
            /// </summary>
            public string TypedArrayName { get; init; }

            /// <summary>
            /// The element size in bytes.
            /// </summary>
            public int ElementSize { get; init; }

            /// <summary>
            /// The JavaScript type keyword for variable declarations (e.g., "0" for int default, "0.0" for float).
            /// </summary>
            public string DefaultValue { get; init; }

            /// <summary>
            /// Whether this type uses BigInt in JS (Int64/UInt64).
            /// </summary>
            public bool IsBigInt { get; init; }
        }

        /// <summary>
        /// Gets the JavaScript type info for an ILGPU basic value type.
        /// </summary>
        public static JSTypeInfo GetJSType(BasicValueType basicType) => basicType switch
        {
            BasicValueType.Int1 => new JSTypeInfo { TypedArrayName = "Int32Array", ElementSize = 4, DefaultValue = "0", IsBigInt = false },
            BasicValueType.Int8 => new JSTypeInfo { TypedArrayName = "Int8Array", ElementSize = 1, DefaultValue = "0", IsBigInt = false },
            BasicValueType.Int16 => new JSTypeInfo { TypedArrayName = "Int16Array", ElementSize = 2, DefaultValue = "0", IsBigInt = false },
            BasicValueType.Int32 => new JSTypeInfo { TypedArrayName = "Int32Array", ElementSize = 4, DefaultValue = "0", IsBigInt = false },
            BasicValueType.Int64 => new JSTypeInfo { TypedArrayName = "BigInt64Array", ElementSize = 8, DefaultValue = "0n", IsBigInt = true },
            BasicValueType.Float16 => new JSTypeInfo { TypedArrayName = "Float32Array", ElementSize = 4, DefaultValue = "0.0", IsBigInt = false },
            BasicValueType.Float32 => new JSTypeInfo { TypedArrayName = "Float32Array", ElementSize = 4, DefaultValue = "0.0", IsBigInt = false },
            BasicValueType.Float64 => new JSTypeInfo { TypedArrayName = "Float64Array", ElementSize = 8, DefaultValue = "0.0", IsBigInt = false },
            _ => new JSTypeInfo { TypedArrayName = "Int32Array", ElementSize = 4, DefaultValue = "0", IsBigInt = false }
        };

        /// <summary>
        /// Gets the JavaScript type info for an ILGPU IR type node.
        /// </summary>
        public static JSTypeInfo GetJSType(TypeNode type)
        {
            if (type is PrimitiveType primitiveType)
                return GetJSType(primitiveType.BasicValueType);

            // For struct/pointer types, default to i32 (will be expanded later)
            return new JSTypeInfo { TypedArrayName = "Int32Array", ElementSize = 4, DefaultValue = "0", IsBigInt = false };
        }

        /// <summary>
        /// Gets the JavaScript variable type name for a primitive type.
        /// </summary>
        public static string GetJSVariableType(BasicValueType basicType) => basicType switch
        {
            BasicValueType.Float16 or BasicValueType.Float32 or BasicValueType.Float64 => "number",
            BasicValueType.Int64 => "bigint",
            _ => "number"
        };

        /// <summary>
        /// Gets an appropriate JS cast/conversion expression for a type conversion.
        /// </summary>
        public static string GetConversionExpression(BasicValueType sourceType, BasicValueType targetType, string sourceExpr)
        {
            // Float to Int
            if (IsFloatType(sourceType) && IsIntType(targetType))
            {
                if (targetType == BasicValueType.Int64)
                    return $"BigInt(Math.trunc({sourceExpr}))";
                return $"Math.trunc({sourceExpr})";
            }

            // Int to Float
            if (IsIntType(sourceType) && IsFloatType(targetType))
            {
                if (sourceType == BasicValueType.Int64)
                    return $"Number({sourceExpr})";
                return sourceExpr; // JS numbers are already floats
            }

            // BigInt conversions
            if (targetType == BasicValueType.Int64 && sourceType != BasicValueType.Int64)
                return $"BigInt({sourceExpr})";
            if (sourceType == BasicValueType.Int64 && targetType != BasicValueType.Int64)
                return $"Number({sourceExpr})";

            // Int to Int (narrowing)
            if (IsIntType(sourceType) && IsIntType(targetType))
            {
                return targetType switch
                {
                    BasicValueType.Int8 => $"(({sourceExpr}) << 24 >> 24)",
                    BasicValueType.Int16 => $"(({sourceExpr}) << 16 >> 16)",
                    BasicValueType.Int32 => $"(({sourceExpr}) | 0)",
                    _ => sourceExpr
                };
            }

            // Float to Float (fround for f32)
            if (IsFloatType(sourceType) && IsFloatType(targetType))
            {
                if (targetType == BasicValueType.Float32)
                    return $"Math.fround({sourceExpr})";
                return sourceExpr;
            }

            return sourceExpr;
        }

        private static bool IsFloatType(BasicValueType type) =>
            type is BasicValueType.Float16 or BasicValueType.Float32 or BasicValueType.Float64;

        private static bool IsIntType(BasicValueType type) =>
            type is BasicValueType.Int1 or BasicValueType.Int8 or BasicValueType.Int16 or BasicValueType.Int32 or BasicValueType.Int64;
    }
}
