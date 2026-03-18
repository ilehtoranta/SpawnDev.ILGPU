// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                           Copyright (c) 2026 SpawnDev
//
// File: DelegateSpecializationRewriter.cs
//
// IL rewriter that creates a synthetic kernel method where
// DelegateSpecialization<T>.Value.Invoke() calls are replaced with
// direct calls to the resolved target method.
// ---------------------------------------------------------------------------------------

using ILGPU.Util;
using System;
using System.Collections.Generic;
using System.Reflection;
using System.Reflection.Emit;

namespace ILGPU.Runtime
{
    /// <summary>
    /// Creates synthetic kernel methods that inline delegate specialization
    /// calls as direct method calls to the resolved target.
    /// </summary>
    internal static class DelegateSpecializationRewriter
    {
        /// <summary>
        /// Creates a new static method that is identical to the original
        /// kernel but with DelegateSpecialization parameters removed and
        /// their Invoke() calls replaced with direct calls to the target.
        /// </summary>
        /// <param name="originalMethod">The kernel method with DelegateSpecialization params.</param>
        /// <param name="targetMethods">
        /// Map from DelegateSpecialization parameter index to the target MethodInfo.
        /// </param>
        /// <returns>
        /// A MethodInfo for the synthetic method that can be compiled by ILGPU.
        /// </returns>
        public static MethodInfo RewriteKernel(
            MethodInfo originalMethod,
            Dictionary<int, MethodInfo> targetMethods)
        {
            var originalParams = originalMethod.GetParameters();

            // Build new parameter list (exclude DelegateSpecialization params)
            var newParamTypes = new List<Type>();
            var paramIndexMap = new Dictionary<int, int>(); // old → new
            int newIdx = 0;
            for (int i = 0; i < originalParams.Length; i++)
            {
                if (targetMethods.ContainsKey(i))
                    continue; // Skip delegate-specialized params
                paramIndexMap[i] = newIdx++;
                newParamTypes.Add(originalParams[i].ParameterType);
            }

            // Create a dynamic assembly with IgnoresAccessChecksTo
            // so the synthetic method can call private/internal targets.
            var asmName = new AssemblyName("ILGPUDelegateSpec_" +
                originalMethod.Name + "_" + Guid.NewGuid().ToString("N")[..8]);

            // Collect all assemblies we need access to
            var accessAssemblies = new HashSet<string>();
            accessAssemblies.Add(
                originalMethod.DeclaringType!.Assembly.GetName().Name!);
            foreach (var target in targetMethods.Values)
                accessAssemblies.Add(
                    target.DeclaringType!.Assembly.GetName().Name!);

            // Build assembly with IgnoresAccessChecksTo attributes
            var customAttrs = new List<CustomAttributeBuilder>();
            var ignoreAttrCtor = GetOrCreateIgnoresAccessChecksCtor();
            if (ignoreAttrCtor != null)
            {
                foreach (var asm in accessAssemblies)
                    customAttrs.Add(new CustomAttributeBuilder(
                        ignoreAttrCtor, new object[] { asm }));
            }

            var asmBuilder = AssemblyBuilder.DefineDynamicAssembly(
                asmName, AssemblyBuilderAccess.Run,
                customAttrs);

            var modBuilder = asmBuilder.DefineDynamicModule(asmName.Name!);
            var typeBuilder = modBuilder.DefineType(
                "DelegateSpecKernel",
                TypeAttributes.Public | TypeAttributes.Abstract |
                TypeAttributes.Sealed);
            var methodBuilder = typeBuilder.DefineMethod(
                originalMethod.Name + "_Specialized",
                MethodAttributes.Public | MethodAttributes.Static,
                typeof(void),
                newParamTypes.ToArray());

            // Identify which calls to replace
            var getValueMethod = typeof(DelegateSpecialization<>)
                .GetProperty("Value")!.GetGetMethod()!;

            // Read original IL and re-emit with replacements
            var body = originalMethod.GetMethodBody()!;
            var ilBytes = body.GetILAsByteArray()!;
            var module = originalMethod.Module;
            var il = methodBuilder.GetILGenerator();

            // Declare locals matching the original
            foreach (var local in body.LocalVariables)
                il.DeclareLocal(local.LocalType, local.IsPinned);

            // State for tracking delegate spec pattern
            bool skipNextCall = false; // true after ldarga of a delegate param

            // Parse and re-emit IL instructions
            int pos = 0;
            // Pre-scan for branch targets so we can create labels
            var branchTargets = new Dictionary<int, Label>();
            ScanBranchTargets(ilBytes, branchTargets, il);

            pos = 0;
            while (pos < ilBytes.Length)
            {
                // Mark label if this offset is a branch target
                if (branchTargets.TryGetValue(pos, out var label))
                    il.MarkLabel(label);

                int instrStart = pos;
                var opcode = ReadOpCode(ilBytes, ref pos);

                // ldarg / ldarga for delegate spec params → skip
                if (IsLdarg(opcode, ilBytes, pos, out int argIdx))
                {
                    pos += GetOperandSize(opcode);
                    if (targetMethods.ContainsKey(argIdx))
                    {
                        // Skip loading the DelegateSpecialization param
                        // and mark that the next call (get_Value) should
                        // also be skipped
                        skipNextCall = true;
                        continue;
                    }
                    // Remap arg index
                    if (paramIndexMap.TryGetValue(argIdx, out int newArgIdx))
                        EmitLdarg(il, opcode, newArgIdx);
                    else
                        EmitLdarg(il, opcode, argIdx); // shouldn't happen
                    continue;
                }

                // call get_Value() after loading delegate param → skip
                if (skipNextCall && opcode == OpCodes.Call)
                {
                    int token = BitConverter.ToInt32(ilBytes, pos);
                    pos += 4;
                    try
                    {
                        var calledMethod = module.ResolveMethod(token);
                        if (calledMethod?.Name == "get_Value" &&
                            calledMethod.DeclaringType != null &&
                            calledMethod.DeclaringType.IsGenericType &&
                            calledMethod.DeclaringType.GetGenericTypeDefinition() ==
                                typeof(DelegateSpecialization<>))
                        {
                            // Skip get_Value() — delegate is resolved
                            // at dispatch time, not in the kernel IL.
                            continue;
                        }
                    }
                    catch { }
                    // Not get_Value — emit normally
                    skipNextCall = false;
                    var resolved = module.ResolveMethod(token)!;
                    il.EmitCall(OpCodes.Call, (MethodInfo)resolved, null);
                    continue;
                }

                skipNextCall = false;

                // callvirt Invoke() → replace with call TargetMethod
                if (opcode == OpCodes.Callvirt)
                {
                    int token = BitConverter.ToInt32(ilBytes, pos);
                    pos += 4;
                    try
                    {
                        var calledMethod = module.ResolveMethod(token);
                        if (calledMethod?.Name == "Invoke" &&
                            calledMethod.DeclaringType != null &&
                            calledMethod.DeclaringType.IsSubclassOf(typeof(Delegate)))
                        {
                            // Find which delegate param this corresponds to
                            // For simplicity, use the first target method
                            var targetMethod = targetMethods.Values
                                .GetEnumerator();
                            targetMethod.MoveNext();
                            il.EmitCall(OpCodes.Call,
                                targetMethod.Current, null);
                            continue;
                        }
                    }
                    catch { }
                    // Not Invoke — emit as callvirt
                    var resolvedVirt = module.ResolveMethod(token)!;
                    il.EmitCall(OpCodes.Callvirt,
                        (MethodInfo)resolvedVirt, null);
                    continue;
                }

                // call → re-emit with resolved method
                if (opcode == OpCodes.Call)
                {
                    int token = BitConverter.ToInt32(ilBytes, pos);
                    pos += 4;
                    var resolved = module.ResolveMethod(token)!;
                    if (resolved is MethodInfo mi)
                        il.EmitCall(OpCodes.Call, mi, null);
                    else if (resolved is ConstructorInfo ci)
                        il.Emit(OpCodes.Call, ci);
                    continue;
                }

                // Branch instructions — use labels
                if (IsBranch(opcode, out bool isShort))
                {
                    int offset;
                    if (isShort)
                    {
                        offset = (sbyte)ilBytes[pos];
                        pos += 1;
                    }
                    else
                    {
                        offset = BitConverter.ToInt32(ilBytes, pos);
                        pos += 4;
                    }
                    int target = pos + offset;
                    if (!branchTargets.TryGetValue(target, out var brLabel))
                    {
                        brLabel = il.DefineLabel();
                        branchTargets[target] = brLabel;
                    }
                    il.Emit(opcode, brLabel);
                    continue;
                }

                // All other instructions — emit raw
                EmitInstruction(il, opcode, ilBytes, ref pos, module);
            }

            var bakedType = typeBuilder.CreateType()!;
            return bakedType.GetMethod(
                originalMethod.Name + "_Specialized",
                BindingFlags.Public | BindingFlags.Static)!;
        }

        private static OpCode ReadOpCode(byte[] il, ref int pos)
        {
            byte b = il[pos++];
            if (b == 0xFE)
            {
                byte b2 = il[pos++];
                // Two-byte opcode
                foreach (var field in typeof(OpCodes).GetFields(
                    BindingFlags.Public | BindingFlags.Static))
                {
                    if (field.GetValue(null) is OpCode op &&
                        op.Size == 2 &&
                        (op.Value & 0xFF) == b2)
                        return op;
                }
                return OpCodes.Nop;
            }
            foreach (var field in typeof(OpCodes).GetFields(
                BindingFlags.Public | BindingFlags.Static))
            {
                if (field.GetValue(null) is OpCode op &&
                    op.Size == 1 &&
                    (byte)(op.Value & 0xFF) == b)
                    return op;
            }
            return OpCodes.Nop;
        }

        private static bool IsLdarg(OpCode opcode, byte[] il, int pos,
            out int argIndex)
        {
            argIndex = -1;
            if (opcode == OpCodes.Ldarg_0) { argIndex = 0; return true; }
            if (opcode == OpCodes.Ldarg_1) { argIndex = 1; return true; }
            if (opcode == OpCodes.Ldarg_2) { argIndex = 2; return true; }
            if (opcode == OpCodes.Ldarg_3) { argIndex = 3; return true; }
            if (opcode == OpCodes.Ldarg_S || opcode == OpCodes.Ldarga_S)
            {
                argIndex = il[pos];
                return true;
            }
            if (opcode == OpCodes.Ldarg)
            {
                argIndex = BitConverter.ToInt16(il, pos);
                return true;
            }
            return false;
        }

        private static int GetOperandSize(OpCode opcode)
        {
            if (opcode == OpCodes.Ldarg_S || opcode == OpCodes.Ldarga_S)
                return 1;
            if (opcode == OpCodes.Ldarg)
                return 2;
            return 0; // Ldarg_0..3 have no operand
        }

        private static void EmitLdarg(ILGenerator il, OpCode original,
            int newIndex)
        {
            // Use ldarga for address loads, ldarg for value loads
            if (original == OpCodes.Ldarga_S || original == OpCodes.Ldarga)
                il.Emit(OpCodes.Ldarga, newIndex);
            else
                il.Emit(OpCodes.Ldarg, newIndex);
        }

        private static bool IsBranch(OpCode opcode, out bool isShort)
        {
            isShort = false;
            switch (opcode.FlowControl)
            {
                case FlowControl.Branch:
                case FlowControl.Cond_Branch:
                    isShort = opcode.OperandType ==
                        OperandType.ShortInlineBrTarget;
                    return opcode.OperandType ==
                        OperandType.ShortInlineBrTarget ||
                        opcode.OperandType ==
                        OperandType.InlineBrTarget;
                default:
                    return false;
            }
        }

        private static void ScanBranchTargets(byte[] il,
            Dictionary<int, Label> targets, ILGenerator gen)
        {
            int pos = 0;
            while (pos < il.Length)
            {
                int instrStart = pos;
                var opcode = ReadOpCode(il, ref pos);

                if (IsBranch(opcode, out bool isShort))
                {
                    int offset;
                    if (isShort)
                    {
                        offset = (sbyte)il[pos];
                        pos += 1;
                    }
                    else
                    {
                        offset = BitConverter.ToInt32(il, pos);
                        pos += 4;
                    }
                    int target = pos + offset;
                    if (!targets.ContainsKey(target))
                        targets[target] = gen.DefineLabel();
                }
                else
                {
                    // Skip operand
                    pos += GetInstructionOperandSize(opcode, il, pos);
                }
            }
        }

        private static int GetInstructionOperandSize(OpCode opcode,
            byte[] il, int pos)
        {
            return opcode.OperandType switch
            {
                OperandType.InlineNone => 0,
                OperandType.ShortInlineVar => 1,
                OperandType.ShortInlineI => 1,
                OperandType.InlineVar => 2,
                OperandType.InlineI => 4,
                OperandType.InlineMethod => 4,
                OperandType.InlineField => 4,
                OperandType.InlineType => 4,
                OperandType.InlineString => 4,
                OperandType.InlineTok => 4,
                OperandType.InlineSig => 4,
                OperandType.ShortInlineR => 4,
                OperandType.InlineI8 => 8,
                OperandType.InlineR => 8,
                OperandType.InlineSwitch =>
                    4 + 4 * BitConverter.ToInt32(il, pos),
                _ => 0
            };
        }

        private static void EmitInstruction(ILGenerator il, OpCode opcode,
            byte[] ilBytes, ref int pos, Module module)
        {
            switch (opcode.OperandType)
            {
                case OperandType.InlineNone:
                    il.Emit(opcode);
                    break;
                case OperandType.ShortInlineVar:
                    il.Emit(opcode, ilBytes[pos]);
                    pos += 1;
                    break;
                case OperandType.ShortInlineI:
                    il.Emit(opcode, (sbyte)ilBytes[pos]);
                    pos += 1;
                    break;
                case OperandType.InlineVar:
                    il.Emit(opcode, BitConverter.ToInt16(ilBytes, pos));
                    pos += 2;
                    break;
                case OperandType.InlineI:
                    il.Emit(opcode, BitConverter.ToInt32(ilBytes, pos));
                    pos += 4;
                    break;
                case OperandType.InlineI8:
                    il.Emit(opcode, BitConverter.ToInt64(ilBytes, pos));
                    pos += 8;
                    break;
                case OperandType.ShortInlineR:
                    il.Emit(opcode, BitConverter.ToSingle(ilBytes, pos));
                    pos += 4;
                    break;
                case OperandType.InlineR:
                    il.Emit(opcode, BitConverter.ToDouble(ilBytes, pos));
                    pos += 8;
                    break;
                case OperandType.InlineString:
                    {
                        int token = BitConverter.ToInt32(ilBytes, pos);
                        pos += 4;
                        il.Emit(opcode, module.ResolveString(token));
                    }
                    break;
                case OperandType.InlineMethod:
                    {
                        int token = BitConverter.ToInt32(ilBytes, pos);
                        pos += 4;
                        var method = module.ResolveMethod(token)!;
                        if (method is MethodInfo mi)
                            il.EmitCall(opcode, mi, null);
                        else if (method is ConstructorInfo ci)
                            il.Emit(opcode, ci);
                    }
                    break;
                case OperandType.InlineField:
                    {
                        int token = BitConverter.ToInt32(ilBytes, pos);
                        pos += 4;
                        il.Emit(opcode, module.ResolveField(token)!);
                    }
                    break;
                case OperandType.InlineType:
                    {
                        int token = BitConverter.ToInt32(ilBytes, pos);
                        pos += 4;
                        il.Emit(opcode, module.ResolveType(token)!);
                    }
                    break;
                case OperandType.InlineTok:
                    {
                        int token = BitConverter.ToInt32(ilBytes, pos);
                        pos += 4;
                        var member = module.ResolveMember(token);
                        if (member is Type t) il.Emit(opcode, t);
                        else if (member is FieldInfo f) il.Emit(opcode, f);
                        else if (member is MethodInfo m)
                            il.Emit(opcode, m);
                    }
                    break;
                case OperandType.InlineSig:
                    pos += 4; // Skip signature token
                    il.Emit(opcode);
                    break;
                default:
                    il.Emit(opcode);
                    break;
            }
        }
        private static ConstructorInfo? _ignoresAccessCtor;
        private static bool _ignoresAccessCtorResolved;

        /// <summary>
        /// Gets the IgnoresAccessChecksToAttribute constructor, creating
        /// it in a helper assembly if it doesn't exist in the runtime.
        /// </summary>
        private static ConstructorInfo? GetOrCreateIgnoresAccessChecksCtor()
        {
            if (_ignoresAccessCtorResolved)
                return _ignoresAccessCtor;

            _ignoresAccessCtorResolved = true;

            // Try the runtime's built-in version first
            var attrType = Type.GetType(
                "System.Runtime.CompilerServices" +
                ".IgnoresAccessChecksToAttribute");
            if (attrType != null)
            {
                _ignoresAccessCtor = attrType.GetConstructor(
                    new[] { typeof(string) });
                if (_ignoresAccessCtor != null) return _ignoresAccessCtor;
            }

            // Create a helper assembly with the attribute type
            try
            {
                var helperAsm = AssemblyBuilder.DefineDynamicAssembly(
                    new AssemblyName("ILGPUIgnoresAccessHelper"),
                    AssemblyBuilderAccess.Run);
                var helperMod = helperAsm.DefineDynamicModule(
                    "ILGPUIgnoresAccessHelper");
                var attrBuilder = helperMod.DefineType(
                    "System.Runtime.CompilerServices" +
                    ".IgnoresAccessChecksToAttribute",
                    TypeAttributes.Public | TypeAttributes.Class,
                    typeof(Attribute));
                attrBuilder.DefineField("AssemblyName", typeof(string),
                    FieldAttributes.Public);
                var ctorBuilder = attrBuilder.DefineConstructor(
                    MethodAttributes.Public,
                    CallingConventions.Standard,
                    new[] { typeof(string) });
                var ctorIl = ctorBuilder.GetILGenerator();
                ctorIl.Emit(OpCodes.Ldarg_0);
                ctorIl.Emit(OpCodes.Call,
                    typeof(Attribute).GetConstructor(
                        BindingFlags.NonPublic | BindingFlags.Instance,
                        null, Type.EmptyTypes, null)!);
                ctorIl.Emit(OpCodes.Ret);
                var bakedAttr = attrBuilder.CreateType()!;
                _ignoresAccessCtor = bakedAttr.GetConstructor(
                    new[] { typeof(string) });
            }
            catch { /* Fallback: no visibility bypass */ }

            return _ignoresAccessCtor;
        }
    }
}
