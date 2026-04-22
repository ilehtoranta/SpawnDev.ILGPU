// ---------------------------------------------------------------------------------------
//                                   ILGPU Algorithms
//                        Copyright (c) 2019-2023 ILGPU Project
//                                    www.ilgpu.net
//
// File: ScanReduceOperationsHalf.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

// Hand-written Half scan/reduce operations.
// Half is excluded from the T4-generated ScanReduceOperations.cs because
// AtomicNumericTypes intentionally skips Half (hardware typically lacks
// half-precision atomics). These structs support group-level scans and
// reductions.
//
// Phase 4 (2026-04-22): AtomicApply is now implemented via a CAS loop on the
// u32 containing the Half (16-bit Half in the low 16, upper 16 bits treated as
// padding). The approach mirrors Atomic.CompareExchange(ref float, float, float)
// which already does ref Unsafe.As<float, uint>(ref target) + bit reinterpret.
// Requires the Half target to sit in a 4-byte-aligned buffer slot, which is true
// on every backend after the WasmMemoryBuffer ctor was padded to a 4-byte
// minimum (WebGPU already AlignTo4, CUDA/OpenCL align to >= 128, .NET managed
// objects align to >= 8). WebGL is out of scope - WebGL 2.0 vertex shaders have
// no atomic operations period, so accelerator.Reduce is unsupported for every T.

using System.Runtime.CompilerServices;
using ILGPU.AtomicOperations;

#pragma warning disable IDE0004 // Cast is redundant

namespace ILGPU.Algorithms.ScanReduceOperations
{
    #region Atomic plumbing for Half

    /// <summary>
    /// CAS operation for Half. Reinterprets the ref Half as ref uint on the
    /// containing 4-byte-aligned word and delegates to the existing uint CAS
    /// primitive. The low 16 bits of the u32 hold the Half; upper 16 bits are
    /// buffer padding and are preserved across the exchange.
    /// </summary>
    internal readonly struct CompareExchangeHalf : ICompareExchangeOperation<Half>
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Half CompareExchange(ref Half target, Half compare, Half value)
        {
            // Reinterpret the Half ref as a uint ref. This is safe because every
            // backend's Half buffer allocation is padded to at least 4 bytes.
            ref uint word = ref Unsafe.As<Half, uint>(ref target);
            // Preserve the upper 16 bits (padding) so the CAS only touches the
            // Half slot in the low 16 bits. Read the current word once.
            uint currentWord = word;
            uint upperBits = currentWord & 0xFFFF0000u;
            uint compareWord = upperBits | (uint)Interop.FloatAsInt(compare);
            uint newWord = upperBits | (uint)Interop.FloatAsInt(value);
            uint observed = Atomic.CompareExchange(ref word, compareWord, newWord);
            // Return the OLD Half value at target (low 16 bits of observed).
            return Interop.IntAsFloat((ushort)(observed & 0xFFFFu));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool IsSame(Half left, Half right) =>
            Interop.FloatAsInt(left) == Interop.FloatAsInt(right);
    }

    internal readonly struct AddHalfAtomicOp : IAtomicOperation<Half>
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Half Operation(Half current, Half value) =>
            (Half)((float)current + (float)value);
    }

    internal readonly struct MaxHalfAtomicOp : IAtomicOperation<Half>
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Half Operation(Half current, Half value) =>
            (float)current >= (float)value ? current : value;
    }

    internal readonly struct MinHalfAtomicOp : IAtomicOperation<Half>
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Half Operation(Half current, Half value) =>
            (float)current <= (float)value ? current : value;
    }

    #endregion

    /// <summary>
    /// Represents an Add reduction of type Half.
    /// </summary>
    public readonly struct AddHalf : IScanReduceOperation<Half>
    {
        /// <summary>
        /// Returns the associated OpenCL command suffix for the internal code generator
        /// to build the final OpenCL command to use.
        /// </summary>
        public string CLCommand => "add";

        /// <summary>
        /// Returns the identity value (the neutral element of the operation), such that
        /// Apply(Apply(Identity, left), right) == Apply(left, right).
        /// </summary>
        public Half Identity => Half.Zero;

        /// <summary>
        /// Applies the current operation.
        /// </summary>
        public Half Apply(Half first, Half second) =>
            (Half)(first + second);

        /// <summary>
        /// Atomic Add on Half via CAS loop on the containing u32. Required for the
        /// cross-workgroup combine step in <c>Accelerator.Reduce&lt;Half, AddHalf&gt;</c>.
        /// Unsupported on WebGL (no vertex shader atomics).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void AtomicApply(ref Half target, Half value) =>
            Atomic.MakeAtomic(ref target, value, default(AddHalfAtomicOp), default(CompareExchangeHalf));
    }

    /// <summary>
    /// Represents a Max reduction of type Half.
    /// </summary>
    public readonly struct MaxHalf : IScanReduceOperation<Half>
    {
        /// <summary>
        /// Returns the associated OpenCL command suffix for the internal code generator
        /// to build the final OpenCL command to use.
        /// </summary>
        public string CLCommand => "max";

        /// <summary>
        /// Returns the identity value (the neutral element of the operation), such that
        /// Apply(Apply(Identity, left), right) == Apply(left, right).
        /// </summary>
        public Half Identity => Half.MinValue;

        /// <summary>
        /// Applies the current operation.
        /// </summary>
        public Half Apply(Half first, Half second) =>
            (float)first >= (float)second ? first : second;

        /// <inheritdoc cref="AddHalf.AtomicApply(ref Half, Half)"/>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void AtomicApply(ref Half target, Half value) =>
            Atomic.MakeAtomic(ref target, value, default(MaxHalfAtomicOp), default(CompareExchangeHalf));
    }

    /// <summary>
    /// Represents a Min reduction of type Half.
    /// </summary>
    public readonly struct MinHalf : IScanReduceOperation<Half>
    {
        /// <summary>
        /// Returns the associated OpenCL command suffix for the internal code generator
        /// to build the final OpenCL command to use.
        /// </summary>
        public string CLCommand => "min";

        /// <summary>
        /// Returns the identity value (the neutral element of the operation), such that
        /// Apply(Apply(Identity, left), right) == Apply(left, right).
        /// </summary>
        public Half Identity => Half.MaxValue;

        /// <summary>
        /// Applies the current operation.
        /// </summary>
        public Half Apply(Half first, Half second) =>
            (float)first <= (float)second ? first : second;

        /// <inheritdoc cref="AddHalf.AtomicApply(ref Half, Half)"/>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void AtomicApply(ref Half target, Half value) =>
            Atomic.MakeAtomic(ref target, value, default(MinHalfAtomicOp), default(CompareExchangeHalf));
    }
}

#pragma warning restore IDE0004
