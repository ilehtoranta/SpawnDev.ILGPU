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
// reductions where AtomicApply is not invoked by the backend.

using System;

#pragma warning disable IDE0004 // Cast is redundant

namespace ILGPU.Algorithms.ScanReduceOperations
{
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
        /// <param name="first">The first operand.</param>
        /// <param name="second">The second operand.</param>
        /// <returns>The result of the operation.</returns>
        public Half Apply(Half first, Half second) =>
            (Half)(first + second);

        /// <summary>
        /// Performs an atomic operation of the form target = AtomicUpdate(target.Value, value).
        /// </summary>
        /// <param name="target">The target address to update.</param>
        /// <param name="value">The value.</param>
        /// <remarks>
        /// Half-precision atomics are not supported by hardware. This method is not
        /// called by group-level scan/reduce implementations (which use shared memory
        /// instead). Multi-workgroup reductions that require atomics should use a
        /// wider type.
        /// </remarks>
        public void AtomicApply(ref Half target, Half value) =>
            throw new NotSupportedException(
                "Half-precision atomics are not supported. " +
                "Use group-level scan/reduce operations instead.");
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
        /// <param name="first">The first operand.</param>
        /// <param name="second">The second operand.</param>
        /// <returns>The result of the operation.</returns>
        public Half Apply(Half first, Half second) =>
            (float)first >= (float)second ? first : second;

        /// <inheritdoc cref="AddHalf.AtomicApply(ref Half, Half)"/>
        public void AtomicApply(ref Half target, Half value) =>
            throw new NotSupportedException(
                "Half-precision atomics are not supported. " +
                "Use group-level scan/reduce operations instead.");
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
        /// <param name="first">The first operand.</param>
        /// <param name="second">The second operand.</param>
        /// <returns>The result of the operation.</returns>
        public Half Apply(Half first, Half second) =>
            (float)first <= (float)second ? first : second;

        /// <inheritdoc cref="AddHalf.AtomicApply(ref Half, Half)"/>
        public void AtomicApply(ref Half target, Half value) =>
            throw new NotSupportedException(
                "Half-precision atomics are not supported. " +
                "Use group-level scan/reduce operations instead.");
    }
}

#pragma warning restore IDE0004
