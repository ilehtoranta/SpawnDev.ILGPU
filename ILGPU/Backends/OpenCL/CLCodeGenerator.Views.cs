// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2019-2023 ILGPU Project
//                                    www.ilgpu.net
//
// File: CLCodeGenerator.Views.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.IR.Types;
using ILGPU.IR.Values;
using ILGPU.Util;

namespace ILGPU.Backends.OpenCL
{
    partial class CLCodeGenerator
    {
        /// <summary cref="IBackendCodeGenerator.GenerateCode(LoadElementAddress)"/>
        public void GenerateCode(LoadElementAddress value)
        {
            var elementIndex = LoadAs<PrimitiveVariable>(value.Offset);
            var source = Load(value.Source);

            // Float16 emulation: when cl_khr_fp16 is unavailable, don't compute &source[idx]
            // (wrong stride - uses float size). Instead, store base pointer + element index
            // for vload_half/vstore_half in the Load/Store handlers.
            if (value.Type is PointerType ptrType
                && ptrType.ElementType is PrimitiveType ptElem
                && ptElem.BasicValueType == BasicValueType.Float16
                && !TypeGenerator.Capabilities.Float16)
            {
                var target = AllocatePointerType(ptrType);
                // Still emit the &source[idx] for the variable binding (won't be dereferenced)
                using (var statement = BeginStatement(target))
                {
                    statement.AppendCommand(CLInstructions.AddressOfOperation);
                    statement.Append(source);
                    statement.AppendIndexer(elementIndex);
                }
                Bind(value, target);
                // Track for vload_half/vstore_half: base pointer + element index
                _f16EmulatedLEAs[target.ToString()] = (source, elementIndex);
                return;
            }

            var target2 = AllocatePointerType(value.Type.AsNotNullCast<PointerType>());

            using (var statement = BeginStatement(target2))
            {
                statement.AppendCommand(CLInstructions.AddressOfOperation);
                statement.Append(source);
                statement.AppendIndexer(elementIndex);
            }

            Bind(value, target2);
        }

        /// <summary cref="IBackendCodeGenerator.GenerateCode(AddressSpaceCast)"/>
        public void GenerateCode(AddressSpaceCast value)
        {
            var targetType = value.TargetType.AsNotNullCast<AddressSpaceType>();
            var source = Load(value.Value);
            var target = Allocate(value);

            bool isOperation = CLInstructions.TryGetAddressSpaceCast(
                value.TargetAddressSpace,
                out string? operation);

            void GeneratePointerCast(StatementEmitter statement)
            {
                if (isOperation)
                {
                    // There is a specific cast operation
                    statement.AppendCommand(operation.AsNotNull());
                    statement.BeginArguments();
                }
                else
                {
                    statement.AppendPointerCast(TypeGenerator[targetType.ElementType]);
                }
                statement.Append(source);
            }

            using (var statement = BeginStatement(target))
            {
                GeneratePointerCast(statement);
                if (isOperation)
                    statement.EndArguments();
            }
        }
    }
}
