// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                           Copyright (c) 2026 SpawnDev
//
// File: CapturingLambdaHelper.cs
//
// Support for capturing lambda kernel entry points.
// ---------------------------------------------------------------------------------------

using ILGPU.Util;
using System;
using System.Reflection;
using System.Runtime.CompilerServices;

namespace ILGPU.Runtime
{
    /// <summary>
    /// Helper for loading and dispatching capturing lambda kernels.
    /// Handles the mismatch between the user-facing delegate type (which
    /// doesn't include captures) and the compiled kernel (which has a
    /// captures struct parameter).
    /// </summary>
    internal static class CapturingLambdaHelper
    {
        /// <summary>
        /// Loads a capturing lambda kernel and returns a tuple containing
        /// the kernel, the capture target, the captured fields, and the
        /// mapped capture struct type.
        /// </summary>
        private static (Kernel kernel, object captureTarget, FieldInfo[] fields,
            Type captureParamType) LoadCapturingKernel(
            Accelerator accelerator,
            Delegate action)
        {
            var captureTarget = action.Target
                ?? throw new InvalidOperationException(
                    "Capturing lambda has no target.");
            var capturedFields = action.Method.GetCapturedFields();

            // Validate captured field types before compilation
            foreach (var field in capturedFields)
            {
                var ft = field.FieldType;
                // Only simple value types are supported
                if (!ft.IsValueType || !ft.IsPrimitive && !ft.IsEnum
                    && ft.Namespace != "ILGPU")
                {
                    // Allow ILGPU primitive-like structs but reject complex
                    // structs that contain pointers (ArrayView, etc.)
                    if (ft.IsValueType && ft.IsGenericType)
                    {
                        var gtd = ft.GetGenericTypeDefinition();
                        if (gtd.FullName != null && (
                            gtd.FullName.Contains("ArrayView") ||
                            gtd.FullName.Contains("MemoryBuffer")))
                        {
                            throw new NotSupportedException(
                                $"Captured variable '{field.Name}' of type " +
                                $"'{ft.Name}' contains GPU buffer pointers " +
                                $"and cannot be captured in a lambda kernel. " +
                                $"Pass it as an explicit kernel parameter " +
                                $"instead:\n\n" +
                                $"  // Instead of capturing '{field.Name}':\n" +
                                $"  //   var kernel = accelerator" +
                                $".LoadAutoGroupedStreamKernel<Index1D, " +
                                $"ArrayView<T>>(\n" +
                                $"  //       (index, buf) => {{ buf[index] = " +
                                $"{field.Name}[index]; }});\n" +
                                $"  // Pass it as a parameter:\n" +
                                $"  //   var kernel = accelerator" +
                                $".LoadAutoGroupedStreamKernel<Index1D, " +
                                $"ArrayView<T>, ArrayView<T>>(MyKernel);");
                        }
                    }
                }
            }

            // Load the kernel — EntryPointDescription includes captures
            // as the last parameter.
            var kernel = accelerator.LoadAutoGroupedKernel(
                action.Method, out var _);

            // Determine the mapped capture struct type from the launcher.
            // The ArgumentMapper creates a runtime struct for the display
            // class. We need this type to construct the captures value.
            // The last parameter of the launcher method is the captures.
            var launcherParams = kernel.Launcher!.GetParameters();
            var captureParamType = launcherParams[^1].ParameterType;

            return (kernel, captureTarget, capturedFields, captureParamType);
        }

        /// <summary>
        /// Returns the display class instance as the captures argument.
        /// The launcher method expects the display class type. Each
        /// backend's marshalling code handles the conversion:
        /// - CPU: DefaultILBackend reconstructs a display class from captures
        /// - CUDA/OpenCL: ArgumentMapper IL converts to runtime struct
        /// - WebGPU/WebGL/Wasm: marshalling code flattens fields directly
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static object GetCapturesArg(object captureTarget) =>
            captureTarget;

        /// <summary>
        /// Loads an auto-grouped capturing lambda kernel with 1 explicit
        /// parameter and returns a wrapper delegate.
        /// </summary>
        public static Action<TIndex, T1>
            LoadAutoGroupedStreamKernel<TIndex, T1>(
                Accelerator accelerator,
                Delegate action)
            where TIndex : struct, IIndex
            where T1 : struct
        {
            var (kernel, captureTarget, fields, captureType) =
                LoadCapturingKernel(accelerator, action);

            return (TIndex index, T1 param1) =>
            {
                kernel.Launch(
                    accelerator.DefaultStream,
                    index,
                    param1,
                    GetCapturesArg(captureTarget));
            };
        }

        /// <summary>
        /// Loads an auto-grouped capturing lambda kernel with 2 explicit
        /// parameters.
        /// </summary>
        public static Action<TIndex, T1, T2>
            LoadAutoGroupedStreamKernel<TIndex, T1, T2>(
                Accelerator accelerator,
                Delegate action)
            where TIndex : struct, IIndex
            where T1 : struct
            where T2 : struct
        {
            var (kernel, captureTarget, fields, captureType) =
                LoadCapturingKernel(accelerator, action);

            return (TIndex index, T1 param1, T2 param2) =>
            {
                kernel.Launch(
                    accelerator.DefaultStream,
                    index,
                    param1,
                    param2,
                    GetCapturesArg(captureTarget));
            };
        }

        /// <summary>
        /// Loads an auto-grouped capturing lambda kernel with 3 explicit
        /// parameters.
        /// </summary>
        public static Action<TIndex, T1, T2, T3>
            LoadAutoGroupedStreamKernel<TIndex, T1, T2, T3>(
                Accelerator accelerator,
                Delegate action)
            where TIndex : struct, IIndex
            where T1 : struct
            where T2 : struct
            where T3 : struct
        {
            var (kernel, captureTarget, fields, captureType) =
                LoadCapturingKernel(accelerator, action);

            return (TIndex index, T1 param1, T2 param2, T3 param3) =>
            {
                kernel.Launch(
                    accelerator.DefaultStream,
                    index,
                    param1,
                    param2,
                    param3,
                    GetCapturesArg(captureTarget));
            };
        }

        public static Action<TIndex, T1, T2, T3, T4> LoadAutoGroupedStreamKernel<TIndex, T1, T2, T3, T4>(Accelerator accelerator, Delegate action) where TIndex : struct, IIndex where T1 : struct where T2 : struct where T3 : struct where T4 : struct { var (kernel, captureTarget, fields, captureType) = LoadCapturingKernel(accelerator, action); return (TIndex index, T1 p1, T2 p2, T3 p3, T4 p4) => { kernel.Launch(accelerator.DefaultStream, index, p1, p2, p3, p4, GetCapturesArg(captureTarget)); }; }
        public static Action<TIndex, T1, T2, T3, T4, T5> LoadAutoGroupedStreamKernel<TIndex, T1, T2, T3, T4, T5>(Accelerator accelerator, Delegate action) where TIndex : struct, IIndex where T1 : struct where T2 : struct where T3 : struct where T4 : struct where T5 : struct { var (kernel, captureTarget, fields, captureType) = LoadCapturingKernel(accelerator, action); return (TIndex index, T1 p1, T2 p2, T3 p3, T4 p4, T5 p5) => { kernel.Launch(accelerator.DefaultStream, index, p1, p2, p3, p4, p5, GetCapturesArg(captureTarget)); }; }
        public static Action<TIndex, T1, T2, T3, T4, T5, T6> LoadAutoGroupedStreamKernel<TIndex, T1, T2, T3, T4, T5, T6>(Accelerator accelerator, Delegate action) where TIndex : struct, IIndex where T1 : struct where T2 : struct where T3 : struct where T4 : struct where T5 : struct where T6 : struct { var (kernel, captureTarget, fields, captureType) = LoadCapturingKernel(accelerator, action); return (TIndex index, T1 p1, T2 p2, T3 p3, T4 p4, T5 p5, T6 p6) => { kernel.Launch(accelerator.DefaultStream, index, p1, p2, p3, p4, p5, p6, GetCapturesArg(captureTarget)); }; }
        public static Action<TIndex, T1, T2, T3, T4, T5, T6, T7> LoadAutoGroupedStreamKernel<TIndex, T1, T2, T3, T4, T5, T6, T7>(Accelerator accelerator, Delegate action) where TIndex : struct, IIndex where T1 : struct where T2 : struct where T3 : struct where T4 : struct where T5 : struct where T6 : struct where T7 : struct { var (kernel, captureTarget, fields, captureType) = LoadCapturingKernel(accelerator, action); return (TIndex index, T1 p1, T2 p2, T3 p3, T4 p4, T5 p5, T6 p6, T7 p7) => { kernel.Launch(accelerator.DefaultStream, index, p1, p2, p3, p4, p5, p6, p7, GetCapturesArg(captureTarget)); }; }
        public static Action<TIndex, T1, T2, T3, T4, T5, T6, T7, T8> LoadAutoGroupedStreamKernel<TIndex, T1, T2, T3, T4, T5, T6, T7, T8>(Accelerator accelerator, Delegate action) where TIndex : struct, IIndex where T1 : struct where T2 : struct where T3 : struct where T4 : struct where T5 : struct where T6 : struct where T7 : struct where T8 : struct { var (kernel, captureTarget, fields, captureType) = LoadCapturingKernel(accelerator, action); return (TIndex index, T1 p1, T2 p2, T3 p3, T4 p4, T5 p5, T6 p6, T7 p7, T8 p8) => { kernel.Launch(accelerator.DefaultStream, index, p1, p2, p3, p4, p5, p6, p7, p8, GetCapturesArg(captureTarget)); }; }
        public static Action<TIndex, T1, T2, T3, T4, T5, T6, T7, T8, T9> LoadAutoGroupedStreamKernel<TIndex, T1, T2, T3, T4, T5, T6, T7, T8, T9>(Accelerator accelerator, Delegate action) where TIndex : struct, IIndex where T1 : struct where T2 : struct where T3 : struct where T4 : struct where T5 : struct where T6 : struct where T7 : struct where T8 : struct where T9 : struct { var (kernel, captureTarget, fields, captureType) = LoadCapturingKernel(accelerator, action); return (TIndex index, T1 p1, T2 p2, T3 p3, T4 p4, T5 p5, T6 p6, T7 p7, T8 p8, T9 p9) => { kernel.Launch(accelerator.DefaultStream, index, p1, p2, p3, p4, p5, p6, p7, p8, p9, GetCapturesArg(captureTarget)); }; }
        public static Action<TIndex, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10> LoadAutoGroupedStreamKernel<TIndex, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>(Accelerator accelerator, Delegate action) where TIndex : struct, IIndex where T1 : struct where T2 : struct where T3 : struct where T4 : struct where T5 : struct where T6 : struct where T7 : struct where T8 : struct where T9 : struct where T10 : struct { var (kernel, captureTarget, fields, captureType) = LoadCapturingKernel(accelerator, action); return (TIndex index, T1 p1, T2 p2, T3 p3, T4 p4, T5 p5, T6 p6, T7 p7, T8 p8, T9 p9, T10 p10) => { kernel.Launch(accelerator.DefaultStream, index, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, GetCapturesArg(captureTarget)); }; }
        public static Action<TIndex, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11> LoadAutoGroupedStreamKernel<TIndex, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11>(Accelerator accelerator, Delegate action) where TIndex : struct, IIndex where T1 : struct where T2 : struct where T3 : struct where T4 : struct where T5 : struct where T6 : struct where T7 : struct where T8 : struct where T9 : struct where T10 : struct where T11 : struct { var (kernel, captureTarget, fields, captureType) = LoadCapturingKernel(accelerator, action); return (TIndex index, T1 p1, T2 p2, T3 p3, T4 p4, T5 p5, T6 p6, T7 p7, T8 p8, T9 p9, T10 p10, T11 p11) => { kernel.Launch(accelerator.DefaultStream, index, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, GetCapturesArg(captureTarget)); }; }
        public static Action<TIndex, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12> LoadAutoGroupedStreamKernel<TIndex, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12>(Accelerator accelerator, Delegate action) where TIndex : struct, IIndex where T1 : struct where T2 : struct where T3 : struct where T4 : struct where T5 : struct where T6 : struct where T7 : struct where T8 : struct where T9 : struct where T10 : struct where T11 : struct where T12 : struct { var (kernel, captureTarget, fields, captureType) = LoadCapturingKernel(accelerator, action); return (TIndex index, T1 p1, T2 p2, T3 p3, T4 p4, T5 p5, T6 p6, T7 p7, T8 p8, T9 p9, T10 p10, T11 p11, T12 p12) => { kernel.Launch(accelerator.DefaultStream, index, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, GetCapturesArg(captureTarget)); }; }
        public static Action<TIndex, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13> LoadAutoGroupedStreamKernel<TIndex, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13>(Accelerator accelerator, Delegate action) where TIndex : struct, IIndex where T1 : struct where T2 : struct where T3 : struct where T4 : struct where T5 : struct where T6 : struct where T7 : struct where T8 : struct where T9 : struct where T10 : struct where T11 : struct where T12 : struct where T13 : struct { var (kernel, captureTarget, fields, captureType) = LoadCapturingKernel(accelerator, action); return (TIndex index, T1 p1, T2 p2, T3 p3, T4 p4, T5 p5, T6 p6, T7 p7, T8 p8, T9 p9, T10 p10, T11 p11, T12 p12, T13 p13) => { kernel.Launch(accelerator.DefaultStream, index, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, GetCapturesArg(captureTarget)); }; }
        public static Action<TIndex, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14> LoadAutoGroupedStreamKernel<TIndex, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14>(Accelerator accelerator, Delegate action) where TIndex : struct, IIndex where T1 : struct where T2 : struct where T3 : struct where T4 : struct where T5 : struct where T6 : struct where T7 : struct where T8 : struct where T9 : struct where T10 : struct where T11 : struct where T12 : struct where T13 : struct where T14 : struct { var (kernel, captureTarget, fields, captureType) = LoadCapturingKernel(accelerator, action); return (TIndex index, T1 p1, T2 p2, T3 p3, T4 p4, T5 p5, T6 p6, T7 p7, T8 p8, T9 p9, T10 p10, T11 p11, T12 p12, T13 p13, T14 p14) => { kernel.Launch(accelerator.DefaultStream, index, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, GetCapturesArg(captureTarget)); }; }
    }
}
