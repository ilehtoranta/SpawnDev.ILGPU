// ---------------------------------------------------------------------------------------
//                                   ILGPU Algorithms
//                        Copyright (c) 2019-2024 ILGPU Project
//                                    www.ilgpu.net
//
// File: PTXMemory.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Backends;
using ILGPU.Backends.PTX;
using ILGPU.IR;
using ILGPU.IR.Intrinsics;
using ILGPU.IR.Types;
using ILGPU.IR.Values;
using ILGPU.Util;
using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace ILGPU.Algorithms.PTX
{
    /// <summary>
    /// A 2-wide float vector used by PTX vector-memory intrinsics.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public readonly struct Float2
    {
        public readonly float X;
        public readonly float Y;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Float2(float x, float y)
        {
            X = x;
            Y = y;
        }
    }

    /// <summary>
    /// A 4-wide float vector used by PTX vector-memory intrinsics.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public readonly struct Float4
    {
        public readonly float X;
        public readonly float Y;
        public readonly float Z;
        public readonly float W;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Float4(float x, float y, float z, float w)
        {
            X = x;
            Y = y;
            Z = z;
            W = w;
        }
    }

    /// <summary>
    /// PTX-only explicit vector memory operations.
    /// </summary>
    public static class PTXMemory
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Float2 LoadF32x2(ArrayView<float> source, int index) =>
            LoadF32x2(ref source[index]);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Float4 LoadF32x4(ArrayView<float> source, int index) =>
            LoadF32x4(ref source[index]);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void StoreF32x2(ArrayView<float> target, int index, Float2 value) =>
            StoreF32x2(ref target[index], value);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void StoreF32x4(ArrayView<float> target, int index, Float4 value) =>
            StoreF32x4(ref target[index], value);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void StoreF32x2(
            ArrayView<float> target,
            int index,
            float x,
            float y) =>
            StoreF32x2(ref target[index], x, y);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void StoreF32x4(
            ArrayView<float> target,
            int index,
            float x,
            float y,
            float z,
            float w) =>
            StoreF32x4(ref target[index], x, y, z, w);

        [IntrinsicImplementation]
        public static Float2 LoadF32x2(ref float source) =>
            throw new NotImplementedException();

        [IntrinsicImplementation]
        public static Float4 LoadF32x4(ref float source) =>
            throw new NotImplementedException();

        [IntrinsicImplementation]
        public static void StoreF32x2(ref float target, Float2 value) =>
            throw new NotImplementedException();

        [IntrinsicImplementation]
        public static void StoreF32x4(ref float target, Float4 value) =>
            throw new NotImplementedException();

        [IntrinsicImplementation]
        public static void StoreF32x2(ref float target, float x, float y) =>
            throw new NotImplementedException();

        [IntrinsicImplementation]
        public static void StoreF32x4(
            ref float target,
            float x,
            float y,
            float z,
            float w) =>
            throw new NotImplementedException();

        public static void GenerateLoadF32x2(
            PTXBackend backend,
            PTXCodeGenerator codeGenerator,
            Value value) =>
            GenerateLoad(codeGenerator, value, 2);

        public static void GenerateLoadF32x4(
            PTXBackend backend,
            PTXCodeGenerator codeGenerator,
            Value value) =>
            GenerateLoad(codeGenerator, value, 4);

        public static void GenerateStoreF32x2(
            PTXBackend backend,
            PTXCodeGenerator codeGenerator,
            Value value) =>
            GenerateStore(codeGenerator, value, 2);

        public static void GenerateStoreF32x4(
            PTXBackend backend,
            PTXCodeGenerator codeGenerator,
            Value value) =>
            GenerateStore(codeGenerator, value, 4);

        public static void GenerateStoreF32x2Scalars(
            PTXBackend backend,
            PTXCodeGenerator codeGenerator,
            Value value) =>
            GenerateStoreScalars(codeGenerator, value, 2);

        public static void GenerateStoreF32x4Scalars(
            PTXBackend backend,
            PTXCodeGenerator codeGenerator,
            Value value) =>
            GenerateStoreScalars(codeGenerator, value, 4);

        private static void GenerateLoad(
            PTXCodeGenerator codeGenerator,
            Value value,
            int vectorLength)
        {
            var methodCall = value.AsNotNullCast<MethodCall>();
            var source = methodCall[0].Resolve();
            var sourceType = source.Type.AsNotNullCast<PointerType>();
            var address = codeGenerator.LoadHardware(source);
            var target = codeGenerator
                .Allocate(methodCall)
                .AsNotNullCast<RegisterAllocator<PTXRegisterKind>.CompoundRegister>();
            var targetRegisters = target.SliceAs<RegisterAllocator<PTXRegisterKind>.PrimitiveRegister>(
                0,
                vectorLength);

            using var command = codeGenerator.BeginCommand(PTXInstructions.LoadOperation);
            command.AppendAddressSpace(sourceType.AddressSpace);
            command.AppendVectorSuffix(vectorLength);
            command.AppendSuffix(BasicValueType.Float32);
            command.AppendVectorArgument(targetRegisters);
            command.AppendArgumentValue(address, 0);
        }

        private static void GenerateStore(
            PTXCodeGenerator codeGenerator,
            Value value,
            int vectorLength)
        {
            var methodCall = value.AsNotNullCast<MethodCall>();
            var target = methodCall[0].Resolve();
            var targetType = target.Type.AsNotNullCast<PointerType>();
            var address = codeGenerator.LoadHardware(target);
            var source = codeGenerator
                .Load(methodCall[1].Resolve())
                .AsNotNullCast<RegisterAllocator<PTXRegisterKind>.CompoundRegister>();
            var sourceRegisters = source.SliceAs<RegisterAllocator<PTXRegisterKind>.PrimitiveRegister>(
                0,
                vectorLength);

            using var command = codeGenerator.BeginCommand(PTXInstructions.StoreOperation);
            command.AppendAddressSpace(targetType.AddressSpace);
            command.AppendVectorSuffix(vectorLength);
            command.AppendSuffix(BasicValueType.Float32);
            command.AppendArgumentValue(address, 0);
            command.AppendVectorArgument(sourceRegisters);
        }

        private static void GenerateStoreScalars(
            PTXCodeGenerator codeGenerator,
            Value value,
            int vectorLength)
        {
            var methodCall = value.AsNotNullCast<MethodCall>();
            var target = methodCall[0].Resolve();
            var targetType = target.Type.AsNotNullCast<PointerType>();
            var address = codeGenerator.LoadHardware(target);
            var sourceRegisters = new RegisterAllocator<PTXRegisterKind>.PrimitiveRegister[vectorLength];
            for (int i = 0; i < vectorLength; i++)
            {
                sourceRegisters[i] = codeGenerator.LoadPrimitive(
                    methodCall[i + 1].Resolve());
            }

            using var command = codeGenerator.BeginCommand(PTXInstructions.StoreOperation);
            command.AppendAddressSpace(targetType.AddressSpace);
            command.AppendVectorSuffix(vectorLength);
            command.AppendSuffix(BasicValueType.Float32);
            command.AppendArgumentValue(address, 0);
            command.AppendVectorArgument(sourceRegisters);
        }
    }
}
