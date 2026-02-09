// ---------------------------------------------------------------------------------------
//                               SpawnDev.ILGPU.Wasm
//                    WebAssembly Compute Backend for Blazor WebAssembly
//
// File: WasmOpCodes.cs
// ---------------------------------------------------------------------------------------

namespace SpawnDev.ILGPU.Wasm.Backend
{
    /// <summary>
    /// WebAssembly opcode constants for binary encoding.
    /// </summary>
    public static class WasmOpCodes
    {
        // === Value Types ===
        public const byte I32 = 0x7F;
        public const byte I64 = 0x7E;
        public const byte F32 = 0x7D;
        public const byte F64 = 0x7C;
        public const byte FuncRef = 0x70;
        public const byte Void = 0x40;  // block type: void

        // === Control Flow ===
        public const byte Unreachable = 0x00;
        public const byte Nop = 0x01;
        public const byte Block = 0x02;
        public const byte Loop = 0x03;
        public const byte If = 0x04;
        public const byte Else = 0x05;
        public const byte End = 0x0B;
        public const byte Br = 0x0C;
        public const byte BrIf = 0x0D;
        public const byte BrTable = 0x0E;
        public const byte Return = 0x0F;
        public const byte Call = 0x10;
        public const byte CallIndirect = 0x11;

        // === Parametric ===
        public const byte Drop = 0x1A;
        public const byte Select = 0x1B;

        // === Variable ===
        public const byte LocalGet = 0x20;
        public const byte LocalSet = 0x21;
        public const byte LocalTee = 0x22;
        public const byte GlobalGet = 0x23;
        public const byte GlobalSet = 0x24;

        // === Memory (i32) ===
        public const byte I32Load = 0x28;
        public const byte I64Load = 0x29;
        public const byte F32Load = 0x2A;
        public const byte F64Load = 0x2B;
        public const byte I32Load8S = 0x2C;
        public const byte I32Load8U = 0x2D;
        public const byte I32Load16S = 0x2E;
        public const byte I32Load16U = 0x2F;
        public const byte I32Store = 0x36;
        public const byte I64Store = 0x37;
        public const byte F32Store = 0x38;
        public const byte F64Store = 0x39;
        public const byte I32Store8 = 0x3A;
        public const byte I32Store16 = 0x3B;

        // === Constants ===
        public const byte I32Const = 0x41;
        public const byte I64Const = 0x42;
        public const byte F32Const = 0x43;
        public const byte F64Const = 0x44;

        // === i32 Comparison ===
        public const byte I32Eqz = 0x45;
        public const byte I32Eq = 0x46;
        public const byte I32Ne = 0x47;
        public const byte I32LtS = 0x48;
        public const byte I32LtU = 0x49;
        public const byte I32GtS = 0x4A;
        public const byte I32GtU = 0x4B;
        public const byte I32LeS = 0x4C;
        public const byte I32LeU = 0x4D;
        public const byte I32GeS = 0x4E;
        public const byte I32GeU = 0x4F;

        // === i64 Comparison ===
        public const byte I64Eqz = 0x50;
        public const byte I64Eq = 0x51;
        public const byte I64Ne = 0x52;
        public const byte I64LtS = 0x53;
        public const byte I64LtU = 0x54;
        public const byte I64GtS = 0x55;
        public const byte I64GtU = 0x56;
        public const byte I64LeS = 0x57;
        public const byte I64LeU = 0x58;
        public const byte I64GeS = 0x59;
        public const byte I64GeU = 0x5A;

        // === f32 Comparison ===
        public const byte F32Eq = 0x5B;
        public const byte F32Ne = 0x5C;
        public const byte F32Lt = 0x5D;
        public const byte F32Gt = 0x5E;
        public const byte F32Le = 0x5F;
        public const byte F32Ge = 0x60;

        // === f64 Comparison ===
        public const byte F64Eq = 0x61;
        public const byte F64Ne = 0x62;
        public const byte F64Lt = 0x63;
        public const byte F64Gt = 0x64;
        public const byte F64Le = 0x65;
        public const byte F64Ge = 0x66;

        // === i32 Arithmetic ===
        public const byte I32Clz = 0x67;
        public const byte I32Ctz = 0x68;
        public const byte I32Popcnt = 0x69;
        public const byte I32Add = 0x6A;
        public const byte I32Sub = 0x6B;
        public const byte I32Mul = 0x6C;
        public const byte I32DivS = 0x6D;
        public const byte I32DivU = 0x6E;
        public const byte I32RemS = 0x6F;
        public const byte I32RemU = 0x70;
        public const byte I32And = 0x71;
        public const byte I32Or = 0x72;
        public const byte I32Xor = 0x73;
        public const byte I32Shl = 0x74;
        public const byte I32ShrS = 0x75;
        public const byte I32ShrU = 0x76;
        public const byte I32Rotl = 0x77;
        public const byte I32Rotr = 0x78;

        // === i64 Arithmetic ===
        public const byte I64Clz = 0x79;
        public const byte I64Ctz = 0x7A;
        public const byte I64Popcnt = 0x7B;
        public const byte I64Add = 0x7C;
        public const byte I64Sub = 0x7D;
        public const byte I64Mul = 0x7E;
        public const byte I64DivS = 0x7F;
        public const byte I64DivU = 0x80;
        public const byte I64RemS = 0x81;
        public const byte I64RemU = 0x82;
        public const byte I64And = 0x83;
        public const byte I64Or = 0x84;
        public const byte I64Xor = 0x85;
        public const byte I64Shl = 0x86;
        public const byte I64ShrS = 0x87;
        public const byte I64ShrU = 0x88;
        public const byte I64Rotl = 0x89;
        public const byte I64Rotr = 0x8A;

        // === f32 Arithmetic ===
        public const byte F32Abs = 0x8B;
        public const byte F32Neg = 0x8C;
        public const byte F32Ceil = 0x8D;
        public const byte F32Floor = 0x8E;
        public const byte F32Trunc = 0x8F;
        public const byte F32Nearest = 0x90;
        public const byte F32Sqrt = 0x91;
        public const byte F32Add = 0x92;
        public const byte F32Sub = 0x93;
        public const byte F32Mul = 0x94;
        public const byte F32Div = 0x95;
        public const byte F32Min = 0x96;
        public const byte F32Max = 0x97;
        public const byte F32Copysign = 0x98;

        // === f64 Arithmetic ===
        public const byte F64Abs = 0x99;
        public const byte F64Neg = 0x9A;
        public const byte F64Ceil = 0x9B;
        public const byte F64Floor = 0x9C;
        public const byte F64Sqrt = 0x9F;
        public const byte F64Add = 0xA0;
        public const byte F64Sub = 0xA1;
        public const byte F64Mul = 0xA2;
        public const byte F64Div = 0xA3;
        public const byte F64Min = 0xA4;
        public const byte F64Max = 0xA5;

        // === Conversions ===
        public const byte I32WrapI64 = 0xA7;
        public const byte I32TruncF32S = 0xA8;
        public const byte I32TruncF32U = 0xA9;
        public const byte I32TruncF64S = 0xAA;
        public const byte I64ExtendI32S = 0xAC;
        public const byte I64ExtendI32U = 0xAD;
        public const byte I64TruncF32S = 0xAE;
        public const byte I64TruncF64S = 0xB0;
        public const byte F32ConvertI32S = 0xB2;
        public const byte F32ConvertI32U = 0xB3;
        public const byte F32ConvertI64S = 0xB4;
        public const byte F32DemoteF64 = 0xB6;
        public const byte F64ConvertI32S = 0xB7;
        public const byte F64ConvertI32U = 0xB8;
        public const byte F64ConvertI64S = 0xB9;
        public const byte F64PromoteF32 = 0xBB;
        public const byte I32ReinterpretF32 = 0xBC;
        public const byte I64ReinterpretF64 = 0xBD;
        public const byte F32ReinterpretI32 = 0xBE;
        public const byte F64ReinterpretI64 = 0xBF;

        // === Sign Extension ===
        public const byte I32Extend8S = 0xC0;
        public const byte I32Extend16S = 0xC1;

        // === Atomic Instructions (0xFE prefix) ===
        public const byte AtomicPrefix = 0xFE;
        // After the prefix byte:
        public const byte MemoryAtomicNotify = 0x00;
        public const byte MemoryAtomicWait32 = 0x01;
        public const byte MemoryAtomicWait64 = 0x02;
        public const byte AtomicFence = 0x03;
        public const byte I32AtomicLoad = 0x10;
        public const byte I64AtomicLoad = 0x11;
        public const byte I32AtomicStore = 0x17;
        public const byte I64AtomicStore = 0x18;
        public const byte I32AtomicRmwAdd = 0x1E;
        public const byte I64AtomicRmwAdd = 0x1F;
        public const byte I32AtomicRmwSub = 0x20;
        public const byte I64AtomicRmwSub = 0x21;
        public const byte I32AtomicRmwAnd = 0x22;
        public const byte I64AtomicRmwAnd = 0x23;
        public const byte I32AtomicRmwOr = 0x24;
        public const byte I64AtomicRmwOr = 0x25;
        public const byte I32AtomicRmwXor = 0x26;
        public const byte I64AtomicRmwXor = 0x27;
        public const byte I32AtomicRmwXchg = 0x28;
        public const byte I64AtomicRmwXchg = 0x29;
        public const byte I32AtomicRmwCmpxchg = 0x48;
        public const byte I64AtomicRmwCmpxchg = 0x49;

        // === Section IDs ===
        public const byte SectionCustom = 0;
        public const byte SectionType = 1;
        public const byte SectionImport = 2;
        public const byte SectionFunction = 3;
        public const byte SectionTable = 4;
        public const byte SectionMemory = 5;
        public const byte SectionGlobal = 6;
        public const byte SectionExport = 7;
        public const byte SectionStart = 8;
        public const byte SectionElement = 9;
        public const byte SectionCode = 10;
        public const byte SectionData = 11;

        // === Import/Export Kinds ===
        public const byte ExternalFunc = 0x00;
        public const byte ExternalTable = 0x01;
        public const byte ExternalMemory = 0x02;
        public const byte ExternalGlobal = 0x03;

        // === Memory Limits Flags ===
        public const byte LimitsNoMax = 0x00;
        public const byte LimitsWithMax = 0x01;
        public const byte LimitsSharedNoMax = 0x02;
        public const byte LimitsSharedWithMax = 0x03;

        // === Function Type Tag ===
        public const byte FuncTypeTag = 0x60;
    }
}
