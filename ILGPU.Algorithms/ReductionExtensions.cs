// ---------------------------------------------------------------------------------------
//                                   ILGPU Algorithms
//                        Copyright (c) 2019-2021 ILGPU Project
//                                    www.ilgpu.net
//
// File: ReductionExtensions.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Algorithms.ScanReduceOperations;
using ILGPU.Runtime;
using System;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

namespace ILGPU.Algorithms
{
    #region Reduction Delegates

    /// <summary>
    /// Represents a reduction using a reduction logic.
    /// </summary>
    /// <typeparam name="T">The underlying type of the reduction.</typeparam>
    /// <typeparam name="TStride">The 1D stride of the source view.</typeparam>
    /// <param name="stream">The accelerator stream.</param>
    /// <param name="input">The input elements to reduce.</param>
    /// <param name="output">The output view to store the reduced value.</param>
    public delegate void Reduction<T, TStride>(
        AcceleratorStream stream,
        ArrayView1D<T, TStride> input,
        ArrayView<T> output)
        where T : unmanaged
        where TStride : struct, IStride1D;

    #endregion

    /// <summary>
    /// Reduction functionality for accelerators.
    /// </summary>
    public static class ReductionExtensions
    {
        #region Reduction Implementation

        /// <summary>
        /// A actual raw reduction implementation.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TStride">The 1D stride of the source view.</typeparam>
        /// <typeparam name="TReduction">The type of the reduction to use.</typeparam>
        internal struct ReductionImplementation<
            T,
            TStride,
            TReduction> : IGridStrideKernelBody
            where T : unmanaged
            where TStride : struct, IStride1D
            where TReduction : struct, IScanReduceOperation<T>
        {
            /// <summary>
            /// Creates a new reduction instance.
            /// </summary>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private static TReduction GetReduction()
            {
                TReduction reduction = default;
                return reduction;
            }

            /// <summary>
            /// Creates a new reduction implementation.
            /// </summary>
            /// <param name="input">The input view.</param>
            /// <param name="output">The output view (1 element min).</param>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public ReductionImplementation(
                ArrayView1D<T, TStride> input,
                ArrayView<T> output)
            {
                Input = input;
                Output = output;
                ReducedValue = GetReduction().Identity;
            }

            /// <summary>
            /// Returns the source view.
            /// </summary>
            public ArrayView1D<T, TStride> Input { get; }

            /// <summary>
            /// Returns the output view.
            /// </summary>
            public ArrayView<T> Output { get; }

            /// <summary>
            /// Stores the current intermediate result of this thread.
            /// </summary>
            public T ReducedValue { get; private set; }

            /// <summary>
            /// Reduces each element in a grid-stride loop.
            /// </summary>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public void Execute(LongIndex1D linearIndex)
            {
                if (linearIndex >= Input.Length)
                    return;

                ReducedValue = GetReduction().Apply(ReducedValue, Input[linearIndex]);
            }

            /// <summary>
            /// Finished a group-wide reduction operation using shuffles, shared memory
            /// and atomic operations.
            /// </summary>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public void Finish()
            {
                // Perform group wide reduction
                ReducedValue = GroupExtensions.Reduce<T, TReduction>(ReducedValue);

                if (Group.IsFirstThread)
                    GetReduction().AtomicApply(ref Output[0], ReducedValue);
            }
        }

        /// <summary>
        /// Creates a new instance of a reduction handler.
        /// </summary>
        /// <typeparam name="T">The underlying type of the reduction.</typeparam>
        /// <typeparam name="TStride">The 1D stride of the source view.</typeparam>
        /// <typeparam name="TReduction">The type of the reduction logic.</typeparam>
        /// <param name="accelerator">The accelerator.</param>
        /// <returns>The created reduction handler.</returns>
        public static Reduction<T, TStride> CreateReduction<T, TStride, TReduction>(
            this Accelerator accelerator)
            where T : unmanaged
            where TStride : struct, IStride1D
            where TReduction : struct, IScanReduceOperation<T>
        {
            var initializer = accelerator.CreateInitializer<T, Stride1D.Dense>();
            var reductionKernel = accelerator.LoadGridStrideKernel<
                ReductionImplementation<T, TStride, TReduction>>();
            return (stream, input, output) =>
            {
                if (!input.IsValid)
                    throw new ArgumentNullException(nameof(input));
                if (input.Length < 1)
                    throw new ArgumentOutOfRangeException(nameof(input));
                if (!output.IsValid)
                    throw new ArgumentNullException(nameof(output));
                if (output.Length < 1)
                    throw new ArgumentOutOfRangeException(nameof(output));

                // Ensure a single element in the ouput view
                output = output.SubView(0, 1);

                TReduction reduction = default;
                initializer(stream, output, reduction.Identity);
                reductionKernel(
                    stream,
                    input.Length,
                    new ReductionImplementation<T, TStride, TReduction>(
                        input,
                        output));
            };
        }

        #endregion

        /// <summary>
        /// Performs a reduction using a reduction logic.
        /// </summary>
        /// <typeparam name="T">The underlying type of the reduction.</typeparam>
        /// <typeparam name="TReduction">The type of the reduction logic.</typeparam>
        /// <param name="accelerator">The accelerator.</param>
        /// <param name="stream">The accelerator stream.</param>
        /// <param name="input">The input elements to reduce.</param>
        /// <param name="output">The output view to store the reduced value.</param>
        /// <remarks>
        /// When <typeparamref name="T"/> is <see cref="Half"/>, the reduction is
        /// routed through a transient f32 buffer. The per-group reduction is computed
        /// in Half, then each workgroup's partial result is converted to f32 for the
        /// cross-workgroup atomic combine (widen-to-f32 pattern). The final f32 is
        /// converted back to Half and written to the user's output slot. This path
        /// sidesteps the lack of hardware Half atomics on every backend while
        /// remaining lossless (f16 is a strict subset of f32 encoding). Unsupported
        /// on backends without f32 atomics (WebGL — no atomics at all in vertex
        /// shaders); those backends throw at kernel compile time.
        /// </remarks>
        public static void Reduce<T, TReduction>(
            this Accelerator accelerator,
            AcceleratorStream stream,
            ArrayView<T> input,
            ArrayView<T> output)
            where T : unmanaged
            where TReduction : struct, IScanReduceOperation<T>
        {
            // Half specialization: widen to f32 for the atomic combine step.
            // Hardware Half atomics don't exist on any backend; rather than introduce
            // a CAS-on-u32 path (which had per-backend codegen issues with
            // Unsafe.As<Half, uint> buffer-pointer reinterpretation), we reduce into
            // an f32 temp and convert the final result back to Half. Dispatch by
            // TReduction type since a typed generic recursive call can't flow the
            // constraint (TReduction is constrained to IScanReduceOperation<T>, which
            // for T=Half means IScanReduceOperation<Half> but the compiler can't
            // see that without a runtime cast).
            //
            // Backend coverage (Phase 4, 2026-04-22):
            //   CPU     : supported (widen-to-f32)
            //   CUDA    : supported (widen-to-f32)
            //   OpenCL  : supported (widen-to-f32)
            //   WebGPU  : NOT SUPPORTED - open codegen issue with the Half-to-float
            //             conversion kernel producing zero-valued f32 intermediate
            //             despite ILGPUReduceFloatTest passing on the same backend.
            //             Under investigation. See Plans/f16-emulation-plan.md Phase 4.
            //   Wasm    : NOT SUPPORTED - related browser-side RangeError, TBD root cause.
            //   WebGL   : NOT SUPPORTED - no vertex-shader atomics at all (existing skip).
            if (typeof(T) == typeof(Half))
            {
                var accType = accelerator.AcceleratorType;
                if (accType == AcceleratorType.WebGPU || accType == AcceleratorType.Wasm)
                {
                    throw new NotSupportedException(
                        $"Reduce<Half, {typeof(TReduction).Name}> on {accType} is an open " +
                        "Phase 4 follow-up item. CPU / CUDA / OpenCL work today. Use " +
                        "GroupExtensions.AllReduce<Half, ...> for single-workgroup cases " +
                        "(supported on WebGPU and Wasm). See Plans/f16-emulation-plan.md.");
                }
                var halfInput = input.Cast<Half>();
                var halfOutput = output.Cast<Half>();
                if (typeof(TReduction) == typeof(AddHalf))
                    ReduceHalfWidenImpl<AddFloat>(accelerator, stream, halfInput, halfOutput);
                else if (typeof(TReduction) == typeof(MaxHalf))
                    ReduceHalfWidenImpl<MaxFloat>(accelerator, stream, halfInput, halfOutput);
                else if (typeof(TReduction) == typeof(MinHalf))
                    ReduceHalfWidenImpl<MinFloat>(accelerator, stream, halfInput, halfOutput);
                else
                    throw new NotSupportedException(
                        $"Reduce<Half, {typeof(TReduction).Name}> is not supported. " +
                        "Half reductions widen to f32 internally and currently support " +
                        "AddHalf, MaxHalf, and MinHalf only.");
                return;
            }

            accelerator.CreateReduction<T, Stride1D.Dense, TReduction>()(
                stream,
                input,
                output);
        }

        private static void ReduceHalfWidenImpl<TFloatReduction>(
            Accelerator accelerator,
            AcceleratorStream stream,
            ArrayView<Half> input,
            ArrayView<Half> output)
            where TFloatReduction : struct, IScanReduceOperation<float>
        {
            // 1. Allocate f32 temp buffers (input-sized + 1-element output).
            using var f32Input = accelerator.Allocate1D<float>(input.Length);
            using var f32Output = accelerator.Allocate1D<float>(1);

            // 2. Convert Half input -> f32 temp via a trivial copy kernel.
            var widenKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<Half>, ArrayView<float>>(
                HalfToFloatCopyKernel);
            widenKernel((Index1D)(int)input.Length, input, f32Input.View);

            // 3. Reduce the f32 view with the widened reduction op.
            accelerator.CreateReduction<float, Stride1D.Dense, TFloatReduction>()(
                stream,
                f32Input.View,
                f32Output.View);

            // 4. Convert final f32 result back to Half in the user's output slot.
            var narrowKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<Half>>(
                FloatToHalfCopyKernel);
            narrowKernel((Index1D)1, f32Output.View, output);
        }

        // Cast operators `(float)halfVal` / `(Half)floatVal` route through ILGPU's standard
        // conversion IR nodes which every backend handles correctly. Do NOT use
        // HalfExtensions.ConvertHalfToFloat/ConvertFloatToHalf here unless every backend
        // registers them as intrinsics - the default lookup-table implementation reads
        // from static .NET arrays that don't exist on GPU and returns zero.

        private static void HalfToFloatCopyKernel(
            Index1D index,
            ArrayView<Half> src,
            ArrayView<float> dst)
        {
            dst[index] = (float)src[index];
        }

        private static void FloatToHalfCopyKernel(
            Index1D index,
            ArrayView<float> src,
            ArrayView<Half> dst)
        {
            dst[index] = (Half)src[index];
        }

        /// <summary>
        /// Performs a reduction using a reduction logic.
        /// </summary>
        /// <typeparam name="T">The underlying type of the reduction.</typeparam>
        /// <typeparam name="TReduction">The type of the reduction logic.</typeparam>
        /// <param name="accelerator">The accelerator.</param>
        /// <param name="stream">The accelerator stream.</param>
        /// <param name="input">The input elements to reduce.</param>
        /// <remarks>
        /// Uses the internal cache to realize a temporary output buffer.
        /// </remarks>
        /// <returns>The reduced value.</returns>
        public static T Reduce<T, TReduction>(
            this Accelerator accelerator,
            AcceleratorStream stream,
            ArrayView<T> input)
            where T : unmanaged
            where TReduction : struct, IScanReduceOperation<T>
        {
            using var output = accelerator.Allocate1D<T>(1);
            accelerator.Reduce<T, TReduction>(stream, input, output.View);
            T result = default;
            output.View.CopyToCPU(stream, ref result, 1);
            return result;
        }

        /// <summary>
        /// Performs a reduction using a reduction logic.
        /// </summary>
        /// <typeparam name="T">The underlying type of the reduction.</typeparam>
        /// <typeparam name="TReduction">The type of the reduction logic.</typeparam>
        /// <param name="accelerator">The accelerator.</param>
        /// <param name="stream">The accelerator stream.</param>
        /// <param name="input">The input elements to reduce.</param>
        /// <remarks>
        /// Uses the internal cache to realize a temporary output buffer.
        /// </remarks>
        /// <returns>The reduced value.</returns>
        public static Task<T> ReduceAsync<T, TReduction>(
            this Accelerator accelerator,
            AcceleratorStream stream,
            ArrayView<T> input)
            where T : unmanaged
            where TReduction : struct, IScanReduceOperation<T> =>
            Task.Run(() => accelerator.Reduce<T, TReduction>(stream, input));

        /// <summary>
        /// Performs a reduction using a reduction logic.
        /// </summary>
        /// <typeparam name="T">The underlying type of the reduction.</typeparam>
        /// <typeparam name="TStride">The 1D stride of the input view.</typeparam>
        /// <typeparam name="TReduction">The type of the reduction logic.</typeparam>
        /// <param name="accelerator">The accelerator.</param>
        /// <param name="stream">The accelerator stream.</param>
        /// <param name="input">The input elements to reduce.</param>
        /// <param name="output">The output view to store the reduced value.</param>
        public static void Reduce<T, TStride, TReduction>(
            this Accelerator accelerator,
            AcceleratorStream stream,
            ArrayView1D<T, TStride> input,
            ArrayView<T> output)
            where T : unmanaged
            where TStride : struct, IStride1D
            where TReduction : struct, IScanReduceOperation<T> =>
            accelerator.CreateReduction<T, TStride, TReduction>()(
                stream,
                input,
                output);

        /// <summary>
        /// Performs a reduction using a reduction logic.
        /// </summary>
        /// <typeparam name="T">The underlying type of the reduction.</typeparam>
        /// <typeparam name="TStride">The 1D stride of the input view.</typeparam>
        /// <typeparam name="TReduction">The type of the reduction logic.</typeparam>
        /// <param name="accelerator">The accelerator.</param>
        /// <param name="stream">The accelerator stream.</param>
        /// <param name="input">The input elements to reduce.</param>
        /// <remarks>
        /// Uses the internal cache to realize a temporary output buffer.
        /// </remarks>
        /// <returns>The reduced value.</returns>
        public static T Reduce<T, TStride, TReduction>(
            this Accelerator accelerator,
            AcceleratorStream stream,
            ArrayView1D<T, TStride> input)
            where T : unmanaged
            where TStride : struct, IStride1D
            where TReduction : struct, IScanReduceOperation<T>
        {
            using var output = accelerator.Allocate1D<T>(1);
            accelerator.Reduce<T, TStride, TReduction>(stream, input, output.View);
            T result = default;
            output.View.CopyToCPU(stream, ref result, 1);
            return result;
        }

        /// <summary>
        /// Performs a reduction using a reduction logic.
        /// </summary>
        /// <typeparam name="T">The underlying type of the reduction.</typeparam>
        /// <typeparam name="TStride">The 1D stride of the input view.</typeparam>
        /// <typeparam name="TReduction">The type of the reduction logic.</typeparam>
        /// <param name="accelerator">The accelerator.</param>
        /// <param name="stream">The accelerator stream.</param>
        /// <param name="input">The input elements to reduce.</param>
        /// <remarks>
        /// Uses the internal cache to realize a temporary output buffer.
        /// </remarks>
        /// <returns>The reduced value.</returns>
        public static Task<T> ReduceAsync<T, TStride, TReduction>(
            this Accelerator accelerator,
            AcceleratorStream stream,
            ArrayView1D<T, TStride> input)
            where T : unmanaged
            where TStride : struct, IStride1D
            where TReduction : struct, IScanReduceOperation<T> =>
            Task.Run(() => accelerator.Reduce<T, TStride, TReduction>(stream, input));
    }
}
