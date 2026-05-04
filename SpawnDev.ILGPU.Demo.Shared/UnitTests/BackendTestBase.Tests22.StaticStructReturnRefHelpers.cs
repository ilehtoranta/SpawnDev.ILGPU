using ILGPU;
using ILGPU.Runtime;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.Demo.Shared.UnitTests
{
    // Part 22: CUDA codegen bug in libopus-style Normalize while-loop pattern.
    // CPU produces correct post-Init state; CUDA produces Rng=0x00000000 (loop
    // exit value should be 0x80000000 or similar high-bit-set uint).
    //
    // Filed by Tuvok against SpawnDev.Codecs `OpusRangeDecoderGpu.Init`
    // (libopus range decoder primitive) — `tuvok-to-geordi-ilgpu-cuda-static-struct-return-bug-2026-05-03.md`.
    // **Updated 2026-05-04 after writing this test:** the bug is broader than
    // initially diagnosed. Both the static-method-struct-return variant AND
    // the equivalent inline-in-kernel variant fail on CUDA with identical
    // Rng=0 output. So the bug is NOT specific to static-method-call codegen —
    // it's inherent to the Normalize-loop pattern (or some interaction between
    // the loop's uint left-shift, byte ArrayView read, and uint state writes).
    //
    // **The failing pattern:**
    // - `while (rng <= 0x800000)` loop with `rng <<= 8` body
    // - Reads bytes from `ArrayView<byte> buf` via `state.Offs++` cursor
    // - Writes uint state via shift+OR composition + `& (0x80000000 - 1u)` mask
    // - Final rng = 0x80000000 (high bit set) on CPU; rng = 0 on CUDA
    //
    // **Other state survives correctly on CUDA:** Val=0x06E35FB8, Offs=6,
    // Rem=143 all match expected values. ONLY Rng is wrong, suggesting the bug
    // is specific to the rng field's update path inside the while-loop —
    // possibly the uint shift `rng <<= 8` when rng >= 0x800000 produces a value
    // that doesn't round-trip through CUDA's PTX register/memory layer correctly,
    // OR the comparison `rng <= 0x800000` on CUDA treats rng as signed when it's
    // declared uint (signed comparison: 0x80000000 = -2147483648 <= 0x800000 → true,
    // continuing the loop one more iteration past the intended exit point;
    // 0x80000000 << 8 then overflows to 0).
    //
    // **The pattern that works on CUDA (for reference):**
    // `Av1RangeDecoderGpu.Init` returns a struct after calling ONE ref-helper
    // (`Refill(ref state, buf)`). The Av1 Refill loop has different shape from
    // libopus Normalize.
    //
    // **Reproducer details:**
    // - 10-field struct (matches `OpusRangeDecoderGpuState` shape: 7 uint + 3 int = 40 bytes).
    // - `StaticInit` initializes fields, calls `ReadByteHelper(ref state, ...)`,
    //   does an arithmetic field write, calls `NormalizeHelper(ref state, ...)`
    //   (which loops + calls `ReadByteHelper` again), returns `state`.
    // - Kernel: dispatches `StaticInit`, writes (Rng, Val, Offs, Rem) to a 4-uint output buffer.
    // - Assertion: post-Init Rng must be > 0x800000 (Normalize loops until that's true).
    //
    // Once the codegen fix lands, BOTH `StaticStructReturnChainedHelpers_BitExactVsCpu`
    // AND `StaticStructReturnChainedHelpers_InlineVariantPassesAllBackends` pass
    // everywhere. The two variants are kept as separate tests in case they diverge
    // during the fix.
    public abstract partial class BackendTestBase
    {
        #region Static Method Returns Struct After Chained Ref-Mutating Helpers (CUDA bug)

        // 10-field state struct mirroring the libopus ec_dec layout that
        // surfaces the CUDA bug. Field order matches the OpusRangeDecoderGpuState
        // exactly (7 uints + 3 ints = 40 bytes total).
        public struct StaticInitStateStruct
        {
            public uint Offs;
            public uint EndOffs;
            public uint EndWindow;
            public int  NEndBits;
            public int  NBitsTotal;
            public uint Rng;
            public uint Val;
            public uint Ext;
            public int  Rem;
            public int  Error;
        }

        // libopus ec_dec constants (mirrored from OpusRangeDecoderGpu.cs).
        private const int  STATIC_REPRO_EC_SYM_BITS   = 8;
        private const int  STATIC_REPRO_EC_CODE_BITS  = 32;
        private const uint STATIC_REPRO_EC_SYM_MAX    = (1u << STATIC_REPRO_EC_SYM_BITS) - 1u;
        private const uint STATIC_REPRO_EC_CODE_TOP   = 1u << (STATIC_REPRO_EC_CODE_BITS - 1);
        private const uint STATIC_REPRO_EC_CODE_BOT   = STATIC_REPRO_EC_CODE_TOP >> STATIC_REPRO_EC_SYM_BITS;
        private const int  STATIC_REPRO_EC_CODE_EXTRA = (STATIC_REPRO_EC_CODE_BITS - 2) % STATIC_REPRO_EC_SYM_BITS + 1; // 7

        // Helper #1 — reads a byte from the buffer at state.Offs, advances Offs.
        // Mirrors OpusRangeDecoderGpu.ReadByte. Mutates state via ref.
        private static int StaticInit_ReadByte(
            ref StaticInitStateStruct state,
            ArrayView<byte> buf, int bufStart, uint storage)
        {
            if (state.Offs < storage)
            {
                int b = buf[bufStart + (int)state.Offs];
                state.Offs++;
                return b;
            }
            return 0;
        }

        // Helper #2 — Normalize loops calling ReadByte until rng > EC_CODE_BOT.
        // Mirrors OpusRangeDecoderGpu.Normalize. Mutates state via ref AND
        // calls ReadByte (chained ref helper).
        private static void StaticInit_Normalize(
            ref StaticInitStateStruct state,
            ArrayView<byte> buf, int bufStart, uint storage)
        {
            while (state.Rng <= STATIC_REPRO_EC_CODE_BOT)
            {
                state.NBitsTotal += STATIC_REPRO_EC_SYM_BITS;
                state.Rng <<= STATIC_REPRO_EC_SYM_BITS;
                int sym = state.Rem;
                state.Rem = StaticInit_ReadByte(ref state, buf, bufStart, storage);
                sym = (sym << STATIC_REPRO_EC_SYM_BITS | state.Rem)
                    >> (STATIC_REPRO_EC_SYM_BITS - STATIC_REPRO_EC_CODE_EXTRA);
                state.Val = ((state.Val << STATIC_REPRO_EC_SYM_BITS)
                    + (STATIC_REPRO_EC_SYM_MAX & (uint)~sym)) & (STATIC_REPRO_EC_CODE_TOP - 1u);
            }
        }

        // The static method that returns a struct after chained ref-helper calls.
        // CUDA codegen returns this with all fields zero. CPU returns correct values.
        // Mirrors OpusRangeDecoderGpu.Init.
        private static StaticInitStateStruct StaticInit(
            ArrayView<byte> buf, int bufStart, uint storage)
        {
            var state = new StaticInitStateStruct
            {
                Offs = 0,
                EndOffs = 0,
                EndWindow = 0,
                NEndBits = 0,
                NBitsTotal = STATIC_REPRO_EC_CODE_BITS + 1
                    - ((STATIC_REPRO_EC_CODE_BITS - STATIC_REPRO_EC_CODE_EXTRA)
                        / STATIC_REPRO_EC_SYM_BITS) * STATIC_REPRO_EC_SYM_BITS,
                Rng = 1u << STATIC_REPRO_EC_CODE_EXTRA,
                Ext = 0,
                Rem = 0,
                Error = 0,
            };
            state.Rem = StaticInit_ReadByte(ref state, buf, bufStart, storage); // helper #1
            state.Val = state.Rng - 1u
                - (uint)(state.Rem >> (STATIC_REPRO_EC_SYM_BITS - STATIC_REPRO_EC_CODE_EXTRA));
            StaticInit_Normalize(ref state, buf, bufStart, storage);            // helper #2 (chains into helper #1)
            return state;
        }

        // Kernel that calls StaticInit and reads back the returned struct's fields.
        static void StaticInitChainedHelpersKernel(
            Index1D _,
            ArrayView<byte> packet, int packetStart, int packetStorage,
            ArrayView<uint> output)
        {
            var state = StaticInit(packet, packetStart, (uint)packetStorage);
            output[0] = state.Rng;
            output[1] = state.Val;
            output[2] = state.Offs;
            output[3] = (uint)state.Rem;
        }

        // Inline reference kernel — same logic, no static-method call. Used to
        // verify the math + struct layout work correctly on CUDA when the
        // static-method codegen path is bypassed. CUDA passes this; CPU passes
        // this. The discriminator is whether the static-call kernel agrees.
        static void StaticInitChainedHelpersInlineKernel(
            Index1D _,
            ArrayView<byte> packet, int packetStart, int packetStorage,
            ArrayView<uint> output)
        {
            uint rng = 1u << STATIC_REPRO_EC_CODE_EXTRA; // 0x80 = 128
            uint offs = 0u;
            int  rem  = 0;
            if (offs < (uint)packetStorage)
            {
                rem = packet[packetStart + (int)offs];
                offs++;
            }
            uint val = rng - 1u
                - (uint)(rem >> (STATIC_REPRO_EC_SYM_BITS - STATIC_REPRO_EC_CODE_EXTRA));
            // Inline Normalize loop.
            while (rng <= STATIC_REPRO_EC_CODE_BOT)
            {
                rng <<= STATIC_REPRO_EC_SYM_BITS;
                int sym = rem;
                if (offs < (uint)packetStorage)
                {
                    rem = packet[packetStart + (int)offs];
                    offs++;
                }
                else { rem = 0; }
                sym = (sym << STATIC_REPRO_EC_SYM_BITS | rem)
                    >> (STATIC_REPRO_EC_SYM_BITS - STATIC_REPRO_EC_CODE_EXTRA);
                val = ((val << STATIC_REPRO_EC_SYM_BITS)
                    + (STATIC_REPRO_EC_SYM_MAX & (uint)~sym)) & (STATIC_REPRO_EC_CODE_TOP - 1u);
            }
            output[0] = rng;
            output[1] = val;
            output[2] = offs;
            output[3] = (uint)rem;
        }

        // Generate a synthetic byte stream that exercises Init + Normalize.
        // Any non-zero bytes work; a small repeating pattern is sufficient.
        private static byte[] StaticInitSyntheticPacket()
        {
            var packet = new byte[16];
            for (int i = 0; i < packet.Length; i++)
                packet[i] = (byte)((i * 0x37) ^ 0x9C);
            return packet;
        }

        // The bug-surfacing test: dispatch the kernel that calls StaticInit
        // (a static method returning a struct after chained ref-helpers) and
        // verify the post-Init state.
        //
        // **CPU + most backends: PASS** — Rng > 0x800000 after Normalize loops.
        // **CUDA: FAIL today** — Rng = 0x00000000 (codegen drops the field
        // writes from the inlined helper chain). Once Geordi's codegen fix
        // lands this test passes everywhere.
        //
        // Test gating: NONE. The goal is for this test to surface the bug
        // by failing on CUDA today. After the fix lands, no gate change needed —
        // it'll pass on every backend automatically.
        [TestMethod]
        public async Task StaticStructReturnChainedHelpers_BitExactVsCpu() => await RunEmulatedTest(async accelerator =>
        {
            var packet = StaticInitSyntheticPacket();
            using var dPacket = accelerator.Allocate1D(packet);
            using var dOutput = accelerator.Allocate1D<uint>(4);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<byte>, int, int, ArrayView<uint>>(
                StaticInitChainedHelpersKernel);
            kernel((Index1D)1, dPacket.View, 0, packet.Length, dOutput.View);
            await accelerator.SynchronizeAsync();
            var output = await dOutput.CopyToHostAsync<uint>();

            uint rng = output[0], val = output[1], offs = output[2], rem = output[3];

            // After Init+Normalize, Rng must be > EC_CODE_BOT (0x800000) since the
            // Normalize loop terminates only when this is true.
            if (rng <= STATIC_REPRO_EC_CODE_BOT)
                throw new Exception(
                    $"StaticInit returned Rng=0x{rng:X8}; expected > 0x{STATIC_REPRO_EC_CODE_BOT:X8} after Normalize. " +
                    $"Val=0x{val:X8} Offs={offs} Rem={rem}. " +
                    "If Rng is 0 on CUDA but the inline kernel below passes, this is the static-method-struct-return codegen bug.");

            if (offs == 0)
                throw new Exception($"StaticInit Offs=0; expected > 0 after at least one ReadByte call.");

            if (rem < 0 || rem > 255)
                throw new Exception($"StaticInit Rem={rem}; expected a valid byte 0..255.");
        });

        // Sanity test: same logic INLINE in the kernel (no static-method call).
        // Should pass on every backend including CUDA today. Confirms that the
        // math and struct shape are correct on CUDA when the static-call
        // codegen path is bypassed.
        [TestMethod]
        public async Task StaticStructReturnChainedHelpers_InlineVariantPassesAllBackends() => await RunEmulatedTest(async accelerator =>
        {
            var packet = StaticInitSyntheticPacket();
            using var dPacket = accelerator.Allocate1D(packet);
            using var dOutput = accelerator.Allocate1D<uint>(4);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<byte>, int, int, ArrayView<uint>>(
                StaticInitChainedHelpersInlineKernel);
            kernel((Index1D)1, dPacket.View, 0, packet.Length, dOutput.View);
            await accelerator.SynchronizeAsync();
            var output = await dOutput.CopyToHostAsync<uint>();

            uint rng = output[0], val = output[1], offs = output[2], rem = output[3];

            // Same assertion as the bug test — inline must produce the correct
            // post-Init state on every backend including CUDA.
            if (rng <= STATIC_REPRO_EC_CODE_BOT)
                throw new Exception(
                    $"Inline kernel Rng=0x{rng:X8} <= EC_CODE_BOT 0x{STATIC_REPRO_EC_CODE_BOT:X8}. " +
                    "If this fails on CUDA, the math itself is broken (not the static-call bug). " +
                    $"Val=0x{val:X8} Offs={offs} Rem={rem}.");
        });

        #endregion
    }
}
