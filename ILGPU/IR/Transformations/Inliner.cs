// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2018-2023 ILGPU Project
//                                    www.ilgpu.net
//
// File: Inliner.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Frontend;
using ILGPU.IR.Analyses;
using ILGPU.IR.Values;
using System.Collections.Generic;
using System.Reflection;
using System.Runtime.CompilerServices;

namespace ILGPU.IR.Transformations
{
    /// <summary>
    /// Represents a function inliner.
    /// </summary>
    public sealed class Inliner : OrderedTransformation
    {
        #region Constants

        /// <summary>
        /// The maximum number of IL instructions to inline in Conservative mode.
        /// </summary>
        private const int MaxNumILInstructionsToInline = 32;

        /// <summary>
        /// Hard upper bound for a method's IL instruction count before the inliner
        /// declines to inline in Aggressive / Default mode. Methods explicitly marked
        /// with <c>[MethodImpl(MethodImplOptions.AggressiveInlining)]</c> or living
        /// inside ILGPU's own assembly bypass this cap (their explicit attribute is
        /// the user's signal that the cost is intentional).
        ///
        /// Without this cap, ILGPU's Aggressive (default) mode unconditionally inlines
        /// every method reachable from the kernel entry, regardless of size. For
        /// codecs that compose many medium-sized helpers via a recursive partition
        /// tree (e.g. Tuvok's VP9 EncodeFrameKernel — see
        /// `tuvok-to-geordi-codecs-wasm-monolithic-inlining-2026-05-06.md`), the
        /// resulting kernel function balloons to tens of thousands of locals + KB-MB
        /// of instruction bytes in a single Wasm/WGSL function. wabt's parser rejects
        /// at &gt;50K locals (`function local count exceeds maximum value`), V8's
        /// TurboFan tier-up compile latency dwarfs Playwright's 30s test budget, and
        /// Naga + WebGPU shader-validators choke similarly on the WGSL side.
        ///
        /// 1024 instructions is generous enough to inline almost all "real" C#
        /// helpers (typical 50-200; heavy 500-1000) while preventing 5K+
        /// mega-helpers from being unconditionally inlined.
        /// (2026-05-05 codecs Wasm finding.)
        /// </summary>
        private const int MaxNumILInstructionsAggressiveCap = 1024;

        #endregion

        #region Static

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void SetupInliningAttributes(
            IRContext context,
            Method method,
            DisassembledMethod disassembledMethod)
        {
            // Check whether we can inline this method
            if (!method.HasImplementation)
                return;

            if (method.HasSource)
            {
                var source = method.Source;
                if ((source.MethodImplementationFlags &
                    MethodImplAttributes.NoInlining) ==
                    MethodImplAttributes.NoInlining)
                {
                    return;
                }

                if ((source.MethodImplementationFlags &
                    MethodImplAttributes.AggressiveInlining) ==
                    MethodImplAttributes.AggressiveInlining ||
                    source.Module.Name == Context.FullAssemblyModuleName)
                {
                    // Explicit AggressiveInlining attribute or ILGPU-internal helper
                    // bypasses the size cap. Return immediately so the heuristic
                    // below doesn't re-evaluate (it's idempotent on the flag, but
                    // the early return is cheaper + makes intent clearer).
                    method.AddFlags(MethodFlags.Inline);
                    return;
                }
            }

            // Hard cap (Aggressive / Default mode): decline to inline methods whose
            // IL body exceeds the aggressive cap. Without this, kernels composed of
            // medium-sized helpers via deep call graphs balloon to tens of thousands
            // of locals + KB-MB instruction bytes in a single emitted Wasm/WGSL
            // function — see XML comment on `MaxNumILInstructionsAggressiveCap` for
            // the codec finding that motivated this. Methods opting in explicitly
            // (AggressiveInlining attribute / ILGPU internal) bypassed this above.
            if (disassembledMethod.Instructions.Length > MaxNumILInstructionsAggressiveCap)
                return;

            // Evaluate a simple inlining heuristic
            if (context.Properties.InliningMode != InliningMode.Conservative ||
                disassembledMethod.Instructions.Length <= MaxNumILInstructionsToInline)
            {
                method.AddFlags(MethodFlags.Inline);
            }
        }

        /// <summary>
        /// Tries to inline method calls.
        /// </summary>
        /// <param name="builder">The current method builder.</param>
        /// <param name="currentBlock">The current block (may be modified).</param>
        /// <returns>True, in case of an inlined call.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool InlineCalls(
            Method.Builder builder,
            ref BasicBlock currentBlock)
        {
            foreach (var valueEntry in currentBlock)
            {
                if (!(valueEntry.Value is MethodCall call))
                    continue;

                if (call.Target.HasFlags(MethodFlags.Inline))
                {
                    var blockBuilder = builder[currentBlock];
                    var tempBlock = blockBuilder.SpecializeCall(call);

                    // We can continue our search in the temp block
                    currentBlock = tempBlock.BasicBlock;
                    return true;
                }
            }

            return false;
        }

        #endregion

        #region Instance

        /// <summary>
        /// Constructs a new inliner that inlines all methods marked with
        /// <see cref="MethodFlags.Inline"/> flags.
        /// </summary>
        public Inliner() { }

        #endregion

        #region Methods

        /// <summary>
        /// Applies the inlining transformation.
        /// </summary>
        protected override bool PerformTransformation(
            IRContext context,
            Method.Builder builder,
            Landscape landscape,
            Landscape.Entry current)
        {
            var processed = builder.SourceBlocks.CreateSet();
            var toProcess = new Stack<BasicBlock>();

            bool result = false;
            var currentBlock = builder.EntryBlock;

            while (true)
            {
                if (processed.Add(currentBlock))
                {
                    if (result = InlineCalls(builder, ref currentBlock))
                    {
                        result = true;
                        continue;
                    }

                    var successors = currentBlock.CurrentSuccessors;
                    if (successors.Length > 0)
                    {
                        currentBlock = successors[0];
                        for (int i = 1, e = successors.Length; i < e; ++i)
                            toProcess.Push(successors[i]);
                        continue;
                    }
                }

                if (toProcess.Count < 1)
                    break;
                currentBlock = toProcess.Pop();
            }

            return result;
        }

        #endregion

    }
}
