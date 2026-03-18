// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                           Copyright (c) 2026 SpawnDev
//
// File: DelegateSpecializationHelper.cs
//
// Handles loading and dispatching kernels with DelegateSpecialization
// parameters. Each unique delegate target produces a specialized kernel
// compilation, cached by the target's MethodInfo.
// ---------------------------------------------------------------------------------------

using ILGPU.Util;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Reflection;

namespace ILGPU.Runtime
{
    /// <summary>
    /// Helper for loading and dispatching kernels with
    /// <see cref="DelegateSpecialization{TDelegate}"/> parameters.
    /// Handles compile-time specialization by creating synthetic kernel
    /// methods for each unique delegate target.
    /// </summary>
    internal static class DelegateSpecializationHelper
    {
        // Cache: (originalMethod, targetMethod) → compiled kernel + launcher
        private static readonly ConcurrentDictionary<
            (MethodInfo original, MethodInfo target), Kernel> _kernelCache = new();

        /// <summary>
        /// Gets or creates a specialized kernel for the given delegate target.
        /// </summary>
        private static Kernel GetOrCreateKernel(
            Accelerator accelerator,
            MethodInfo originalMethod,
            MethodInfo targetMethod)
        {
            var key = (originalMethod, targetMethod);
            return _kernelCache.GetOrAdd(key, _ =>
            {
                // Find which parameters are DelegateSpecialization
                var origParams = originalMethod.GetParameters();
                var targets = new Dictionary<int, MethodInfo>();
                for (int i = 0; i < origParams.Length; i++)
                {
                    if (origParams[i].ParameterType
                        .IsDelegateSpecializedType())
                    {
                        targets[i] = targetMethod;
                    }
                }

                // Create the synthetic method with delegate calls inlined
                var syntheticMethod =
                    DelegateSpecializationRewriter.RewriteKernel(
                        originalMethod, targets);

                // Compile through the normal ILGPU pipeline
                return accelerator.LoadAutoGroupedKernel(
                    syntheticMethod);
            });
        }

        /// <summary>
        /// Loads a kernel with 1 explicit param + 1 DelegateSpecialization.
        /// The DelegateSpecialization must be the last type parameter.
        /// </summary>
        public static Action<TIndex, T1>
            LoadAutoGroupedStreamKernel<TIndex, T1, TDel>(
                Accelerator accelerator,
                Delegate action)
            where TIndex : struct, IIndex
            where T1 : struct
            where TDel : struct
        {
            var originalMethod = action.Method;

            return (TIndex index, T1 param1) =>
            {
                // This overload assumes DelegateSpecialization is NOT
                // passed by the user — it's the last generic type param
                // but only T1 is the real kernel param.
                // We need to get the delegate from the call site...
                // Actually, the user DOES pass it as the last arg.
                throw new NotSupportedException(
                    "DelegateSpecialization dispatch requires the " +
                    "specialized overload. This is a placeholder.");
            };
        }

        /// <summary>
        /// Loads a kernel with DelegateSpecialization as the last parameter
        /// and returns a wrapper that handles specialization at dispatch time.
        /// </summary>
        public static Action<TIndex, T1, DelegateSpecialization<TDel>>
            LoadAutoGroupedStreamKernelWithDelegate<TIndex, T1, TDel>(
                Accelerator accelerator,
                MethodInfo originalMethod)
            where TIndex : struct, IIndex
            where T1 : struct
            where TDel : Delegate
        {
            return (TIndex index, T1 param1,
                DelegateSpecialization<TDel> spec) =>
            {
                var target = spec._delegate.Method;
                var kernel = GetOrCreateKernel(
                    accelerator, originalMethod, target);
                // The synthetic kernel has the delegate param removed,
                // so we only pass the remaining params
                kernel.Launch(accelerator.DefaultStream, index, param1);
            };
        }

        /// <summary>
        /// Loads a kernel where the ONLY explicit param is a
        /// DelegateSpecialization (after the index).
        /// </summary>
        public static Action<TIndex, DelegateSpecialization<TDel>>
            LoadAutoGroupedStreamKernelWithDelegate<TIndex, TDel>(
                Accelerator accelerator,
                MethodInfo originalMethod)
            where TIndex : struct, IIndex
            where TDel : Delegate
        {
            return (TIndex index, DelegateSpecialization<TDel> spec) =>
            {
                var target = spec._delegate.Method;
                var kernel = GetOrCreateKernel(
                    accelerator, originalMethod, target);
                kernel.Launch(accelerator.DefaultStream, index);
            };
        }
    }
}
