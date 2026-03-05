// ---------------------------------------------------------------------------------------
//                                   ILGPU Algorithms
//                        Copyright (c) 2019-2021 ILGPU Project
//                                    www.ilgpu.net
//
// File: TempViewManager.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Algorithms.Resources;
using ILGPU.Runtime;
using System;
using System.Runtime.CompilerServices;

namespace ILGPU.Algorithms
{
    /// <summary>
    /// Simplifies the subsequent splitting of a temporary memory view
    /// into smaller chunks.
    /// </summary>
    public struct TempViewManager
    {
        #region Instance

        /// <summary>
        /// Constructs a new temp-view manager.
        /// </summary>
        /// <param name="tempView">The source temp view to use.</param>
        /// <param name="paramName">
        /// The associated parameter name (for error messages).
        /// </param>
        public TempViewManager(ArrayView<int> tempView, string paramName)
        {
            if (!tempView.IsValid)
                throw new ArgumentNullException(paramName);

            NumInts = 0;
            TempView = tempView;
            ParamName = paramName;
        }

        #endregion

        #region Properties

        /// <summary>
        /// Returns the total number of ints (32bit integers) that
        /// have been allocated.
        /// </summary>
        public long NumInts { get; private set; }

        /// <summary>
        /// Returns the associated param name (for error messages).
        /// </summary>
        public string ParamName { get; }

        /// <summary>
        /// Returns the underlying temporary array view.
        /// </summary>
        public ArrayView<int> TempView { get; }

        #endregion

        #region Methods

        /// <summary>
        /// Allocates a single element of type <typeparamref name="T"/>.
        /// </summary>
        /// <typeparam name="T">The element type to allocate.</typeparam>
        /// <returns>The allocated variable view.</returns>
        public VariableView<T> Allocate<T>()
            where T : unmanaged =>
            Allocate<T>(1).VariableView(0);

        /// <summary>
        /// Allocates several elements of type <typeparamref name="T"/>.
        /// </summary>
        /// <typeparam name="T">The element type to allocate.</typeparam>
        /// <param name="length">The number of elements to allocate.</param>
        /// <returns>The allocated array view.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ArrayView<T> Allocate<T>(long length)
            where T : unmanaged
        {
            var viewLength = Interop.ComputeRelativeSizeOf<int, T>(length);
            if (NumInts + viewLength > TempView.Length)
                throw new ArgumentOutOfRangeException(ParamName);

            // Ensure correct alignment when allocating types larger than a single int.
            // NB: Structs are assumed to be int-aligned.
            if (typeof(T).IsPrimitive)
            {
                var sizeOfAllocateT = Interop.SizeOf<T>();
                var sizeOfInt = Interop.SizeOf<int>();
                var allocationByteOffset = NumInts * sizeOfInt;
                if (sizeOfAllocateT > sizeOfInt &&
                    allocationByteOffset % sizeOfAllocateT != 0)
                {
                    throw new InvalidOperationException(string.Format(
                        ErrorMessages.TempViewManagerUnalignedAllocation,
                        typeof(T),
                        sizeOfAllocateT,
                        allocationByteOffset));
                }
            }

            var tempView = TempView.SubView(NumInts, viewLength);
            NumInts += viewLength;

            // Pad NumInts to the next 256-byte boundary (64 ints).
            // WebGPU requires GPUBufferBinding.Offset to be a multiple of
            // minStorageBufferOffsetAlignment (256 bytes). Without this padding,
            // TempViewManager sub-views have 4-byte-aligned offsets which violate
            // WebGPU's alignment constraint. Harmless for CUDA/OpenCL (tiny memory waste).
            const long alignmentInInts = 256 / sizeof(int); // 64
            NumInts = (NumInts + alignmentInInts - 1) / alignmentInInts * alignmentInInts;

            return tempView.Cast<T>().SubView(0, length);
        }

        #endregion
    }
}
