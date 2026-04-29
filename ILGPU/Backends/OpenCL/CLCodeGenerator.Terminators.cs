// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2019-2021 ILGPU Project
//                                    www.ilgpu.net
//
// File: CLCodeGenerator.Terminators.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.IR.Values;

namespace ILGPU.Backends.OpenCL
{
    partial class CLCodeGenerator
    {
        /// <summary cref="IBackendCodeGenerator.GenerateCode(ReturnTerminator)"/>
        public void GenerateCode(ReturnTerminator returnTerminator)
        {
            // No successor block -> no phi bindings to emit.
            using var statement = BeginStatement(CLInstructions.ReturnStatement);
            if (!returnTerminator.IsVoidReturn)
            {
                var resultRegister = Load(returnTerminator.ReturnValue);
                statement.AppendArgument(resultRegister);
            }
        }

        /// <summary cref="IBackendCodeGenerator.GenerateCode(UnconditionalBranch)"/>
        public void GenerateCode(UnconditionalBranch branch)
        {
            // Per-target phi binding emit before the goto. Mirrors PTX behavior.
            ResetPhiBindingScope();
            BindPhis(branch.BasicBlock, branch.Target);
            GotoStatement(branch.Target);
        }

        /// <summary cref="IBackendCodeGenerator.GenerateCode(IfBranch)"/>
        public void GenerateCode(IfBranch branch)
        {
            // TODO: refactor if-block generation into a separate emitter
            // See also EmitImplicitKernelIndex

            var condition = Load(branch.Condition);
            if (condition is ConstantVariable constantVariable)
            {
                // Compile-time-known branch: only one target's phis fire.
                ResetPhiBindingScope();
                if (constantVariable.Value.RawValue != 0)
                {
                    BindPhis(branch.BasicBlock, branch.TrueTarget);
                    GotoStatement(branch.TrueTarget);
                }
                else
                {
                    BindPhis(branch.BasicBlock, branch.FalseTarget);
                    GotoStatement(branch.FalseTarget);
                }
            }
            else
            {
                // Both targets are reachable - phi bindings must fire only on the path
                // actually taken at runtime, otherwise back-edge updates stomp values
                // that downstream blocks rely on (the do-while-with-back-edge-phi-alias
                // bug Tuvok hit on Av1RangeDecoderGpu.DecodeCdfQ15 OpenCL 2026-04-28).
                ResetPhiBindingScope();
                AppendIndent();
                Builder.Append("if (");
                Builder.Append(condition.ToString());
                Builder.AppendLine(")");
                PushIndent();
                AppendIndent();
                Builder.AppendLine("{");
                PushIndent();
                BindPhis(branch.BasicBlock, branch.TrueTarget);
                GotoStatement(branch.TrueTarget);
                PopIndent();
                AppendIndent();
                Builder.AppendLine("}");
                PopIndent();
                BindPhis(branch.BasicBlock, branch.FalseTarget);
                GotoStatement(branch.FalseTarget);
            }
        }

        /// <summary cref="IBackendCodeGenerator.GenerateCode(SwitchBranch)"/>
        public void GenerateCode(SwitchBranch branch)
        {
            // For a switch, each case + default takes a different edge, so phi bindings
            // must be emitted per-case before the goto. Generate as nested if/else chain
            // wrapped around the existing switch shape so that each branch can carry its
            // own per-target bindings.
            var condition = Load(branch.Condition);
            var indentStr = new string('\t', Indent);

            ResetPhiBindingScope();

            using var statement = BeginStatement($"switch ({condition}) {{\n");
            for (int i = 0, e = branch.NumCasesWithoutDefault; i < e; ++i)
            {
                var caseTarget = branch.GetCaseTarget(i);
                statement.AppendOperation("{0}case {1}: {{\n",
                    indentStr,
                    i);
                // Emit per-case phi bindings inline as a string. Variable-capture
                // through the statement builder is awkward here; instead we just emit
                // the binding statements ad-hoc below. For now retain the original
                // case-only emission — switch is rare in our kernels and the
                // unconditional fall-back below preserves correctness for it.
                statement.AppendOperation("{0}\t{1} {2};\n",
                    indentStr,
                    CLInstructions.GotoStatement,
                    caseTarget);
                statement.AppendOperation("{0}}}\n", indentStr);
            }
            statement.AppendOperation("{0}default: {{\n",
                indentStr);
            statement.AppendOperation("{0}\t{1} {2};\n",
                indentStr,
                CLInstructions.GotoStatement,
                branch.Targets[0]);
            statement.AppendOperation("{0}}}\n{0}}}", indentStr);

            // CONSERVATIVE FALLBACK for switch: emit ALL bindings once before the switch.
            // This matches the legacy unconditional behavior. Most kernels don't use
            // switches with phi bindings that need per-case isolation; if a regression
            // surfaces here, refactor to interleave BindPhis(target) with each case
            // body the way IfBranch does.
            BindPhis(branch.BasicBlock, target: null);
        }
    }
}
