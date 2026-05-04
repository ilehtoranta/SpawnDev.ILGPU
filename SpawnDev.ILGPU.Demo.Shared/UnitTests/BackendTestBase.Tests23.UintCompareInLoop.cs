using ILGPU;
using ILGPU.Runtime;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.Demo.Shared.UnitTests
{
    // Part 23: Bisection tests for the Tests22 cross-backend bug
    // (`StaticStructReturnChainedHelpers_*` fail on every backend except CPU
    // with Rng=0 instead of >0x800000 after the libopus Normalize loop pattern).
    //
    // Goal: isolate the failing operation. The full Normalize loop combines
    //   1. uint compare via `<=` against a uint constant
    //   2. uint left-shift inside the loop body
    //   3. struct field reads/writes (`state.Rng`)
    //   4. ArrayView<byte> reads via the buffer cursor
    //   5. integer arithmetic on shifted values
    //
    // If a SUBSET of these fails on the same backends as Tests22, the bug is in
    // that subset. If all sub-tests pass, the bug is in their interaction.
    public abstract partial class BackendTestBase
    {
        #region Bisection — uint compare + shift + loop

        // Test A: Bare uint compare. No loop, no shift. If this fails the bug is in Compare codegen.
        static void Tests23_UintCompareKernel(Index1D _, ArrayView<uint> output)
        {
            uint a = 0x80000000u;
            uint b = 0x00800000u;
            output[0] = (a <= b) ? 1u : 0u;          // unsigned: a > b → 0
            output[1] = (b <= a) ? 1u : 0u;          // unsigned: b < a → 1
            output[2] = (a <  b) ? 1u : 0u;          // unsigned: a > b → 0
            output[3] = (b <  a) ? 1u : 0u;          // unsigned: b < a → 1
        }

        [TestMethod]
        public async Task Tests23_BareUintCompare() => await RunEmulatedTest(async accelerator =>
        {
            using var dOut = accelerator.Allocate1D<uint>(4);
            var k = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<uint>>(Tests23_UintCompareKernel);
            k((Index1D)1, dOut.View);
            await accelerator.SynchronizeAsync();
            var r = await dOut.CopyToHostAsync<uint>();
            if (r[0] != 0u) throw new Exception($"BareUintCompare a<=b expected 0 got {r[0]} (high-bit uint compared against small uint as signed?)");
            if (r[1] != 1u) throw new Exception($"BareUintCompare b<=a expected 1 got {r[1]}");
            if (r[2] != 0u) throw new Exception($"BareUintCompare a<b expected 0 got {r[2]}");
            if (r[3] != 1u) throw new Exception($"BareUintCompare b<a expected 1 got {r[3]}");
        });

        // Test B: uint left-shift bare. No loop, no compare. Produces the same
        // "shift past sign bit" pattern that the Normalize loop exercises.
        // Output index map (used by both BareUintShift + the constant-write
        // diagnostic Tests23_HighBitConstWrite below):
        //   [0]=v0  [1]=v1  [2]=v2  [3]=v3  [4]=v4
        //   [5]=0x80000000u const-write (NO SHIFT)
        //   [6]=0xFFFFFFFFu const-write (NO SHIFT, all bits)
        //   [7]=0x7FFFFFFFu const-write (NO SHIFT, all bits except sign)
        static void Tests23_UintShiftKernel(Index1D _, ArrayView<uint> output)
        {
            uint v0 = 0x80u;
            uint v1 = v0 << 8;          // 0x8000
            uint v2 = v1 << 8;          // 0x800000
            uint v3 = v2 << 8;          // 0x80000000 — high bit set
            uint v4 = v3 << 8;          // 0 (high byte shifts off)
            output[0] = v0;
            output[1] = v1;
            output[2] = v2;
            output[3] = v3;
            output[4] = v4;
            // Diagnostic: write the same high-bit-set value DIRECTLY, no shift,
            // so the test isolates whether the failure is in the shift codegen
            // or in the TF readback / output-write path.
            output[5] = 0x80000000u;
            output[6] = 0xFFFFFFFFu;
            output[7] = 0x7FFFFFFFu;
        }

        [TestMethod]
        public async Task Tests23_BareUintShift() => await RunEmulatedTest(async accelerator =>
        {
            using var dOut = accelerator.Allocate1D<uint>(8);
            var k = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<uint>>(Tests23_UintShiftKernel);
            k((Index1D)1, dOut.View);
            await accelerator.SynchronizeAsync();
            var r = await dOut.CopyToHostAsync<uint>();
            if (r[0] != 0x80u) throw new Exception($"v0 expected 0x80 got 0x{r[0]:X8}");
            if (r[1] != 0x8000u) throw new Exception($"v1 expected 0x8000 got 0x{r[1]:X8}");
            if (r[2] != 0x800000u) throw new Exception($"v2 expected 0x800000 got 0x{r[2]:X8}");
            if (r[3] != 0x80000000u)
                throw new Exception($"v3 expected 0x80000000 got 0x{r[3]:X8}. Const-write diagnostic: " +
                    $"out[5]=0x{r[5]:X8} (expect 0x80000000), out[6]=0x{r[6]:X8} (expect 0xFFFFFFFF), out[7]=0x{r[7]:X8} (expect 0x7FFFFFFF). " +
                    "If [5] also wrong → TF readback / output-write path is wrong, not the shift. If [5] right → shift codegen is wrong.");
            if (r[4] != 0u) throw new Exception($"v4 expected 0 got 0x{r[4]:X8}");
            if (r[5] != 0x80000000u) throw new Exception($"const-write [5]: expected 0x80000000 got 0x{r[5]:X8}");
            if (r[6] != 0xFFFFFFFFu) throw new Exception($"const-write [6]: expected 0xFFFFFFFF got 0x{r[6]:X8}");
            if (r[7] != 0x7FFFFFFFu) throw new Exception($"const-write [7]: expected 0x7FFFFFFF got 0x{r[7]:X8}");
        });

        // Test C: the Normalize-shape loop with a LOCAL uint variable (no struct).
        // If C passes but Tests22 fails, the bug is specific to struct field
        // access or SetField persistence inside the loop, not the compare/shift itself.
        static void Tests23_LocalLoopKernel(Index1D _, ArrayView<uint> output)
        {
            uint rng = 0x80u;
            int iter = 0;
            // EC_CODE_BOT = 0x00800000
            while (rng <= 0x00800000u && iter < 8)
            {
                rng <<= 8;
                iter++;
            }
            output[0] = rng;             // expected: 0x80000000 after exactly 3 iterations (0x80→0x8000→0x800000→0x80000000)
            output[1] = (uint)iter;      // expected: 3
        }

        [TestMethod]
        public async Task Tests23_LocalLoop_NoStruct() => await RunEmulatedTest(async accelerator =>
        {
            using var dOut = accelerator.Allocate1D<uint>(2);
            var k = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<uint>>(Tests23_LocalLoopKernel);
            k((Index1D)1, dOut.View);
            await accelerator.SynchronizeAsync();
            var r = await dOut.CopyToHostAsync<uint>();
            if (r[0] != 0x80000000u)
                throw new Exception($"Local-loop rng expected 0x80000000 got 0x{r[0]:X8} (iter={r[1]}). " +
                    "If iter=8, the loop ran to its safety cap → compare is broken (still TRUE after rng=0x80000000). " +
                    "If iter<3 → compare exited early. If iter=3 but rng=0 → shift broke at 0x800000<<8.");
            if (r[1] != 3u)
                throw new Exception($"Local-loop iter expected 3 got {r[1]} (rng=0x{r[0]:X8})");
        });

        // Test D: struct-of-uint with the same loop. Bisects whether struct field
        // SetField semantics inside a loop are the bug.
        public struct Tests23_StateStruct
        {
            public uint Rng;
            public uint Other;
            public int Iter;
        }
        static void Tests23_StructLoopKernel(Index1D _, ArrayView<uint> output)
        {
            Tests23_StateStruct s = default;
            s.Rng = 0x80u;
            s.Iter = 0;
            while (s.Rng <= 0x00800000u && s.Iter < 8)
            {
                s.Rng <<= 8;
                s.Iter++;
                s.Other += s.Rng;
            }
            output[0] = s.Rng;             // expected 0x80000000
            output[1] = (uint)s.Iter;      // expected 3
            output[2] = s.Other;           // expected 0x8000 + 0x800000 + 0x80000000 = 0x80808000
        }

        [TestMethod]
        public async Task Tests23_StructLoop() => await RunEmulatedTest(async accelerator =>
        {
            using var dOut = accelerator.Allocate1D<uint>(3);
            var k = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<uint>>(Tests23_StructLoopKernel);
            k((Index1D)1, dOut.View);
            await accelerator.SynchronizeAsync();
            var r = await dOut.CopyToHostAsync<uint>();
            if (r[0] != 0x80000000u)
                throw new Exception($"Struct-loop Rng expected 0x80000000 got 0x{r[0]:X8} (Iter={r[1]} Other=0x{r[2]:X8})");
            if (r[1] != 3u)
                throw new Exception($"Struct-loop Iter expected 3 got {r[1]} (Rng=0x{r[0]:X8})");
            if (r[2] != 0x80808000u)
                throw new Exception($"Struct-loop Other expected 0x80808000 got 0x{r[2]:X8}");
        });

        // Test E: Tuvok-shape kernel WITHOUT the byte ArrayView read inside the loop.
        // Just iterates the shift+compare loop with const-expression constants, multiple
        // uint locals, the sym/val auxiliary computations.
        // Mimics StaticInitChainedHelpersInlineKernel's body shape but without packet reads.
        private const int  TESTS23E_SYM_BITS = 8;
        private const int  TESTS23E_CODE_BITS = 32;
        private const uint TESTS23E_CODE_TOP = 1u << (TESTS23E_CODE_BITS - 1);  // 0x80000000
        private const uint TESTS23E_CODE_BOT = TESTS23E_CODE_TOP >> TESTS23E_SYM_BITS;  // 0x00800000
        private const int  TESTS23E_CODE_EXTRA = (TESTS23E_CODE_BITS - 2) % TESTS23E_SYM_BITS + 1;  // 7
        private const uint TESTS23E_SYM_MAX = (1u << TESTS23E_SYM_BITS) - 1u;  // 0xFF

        static void Tests23_NormalizeShapeNoBufferKernel(Index1D _, ArrayView<uint> output)
        {
            uint rng = 1u << TESTS23E_CODE_EXTRA;  // 0x80
            int  rem  = 0x9C;
            uint val = rng - 1u - (uint)(rem >> (TESTS23E_SYM_BITS - TESTS23E_CODE_EXTRA));
            int  iter = 0;
            // Synthetic Rem stream — replaces ArrayView<byte> reads with deterministic values.
            // Same byte pattern as Tests22's synthetic packet so the math matches.
            uint synthSeed = 0u;
            while (rng <= TESTS23E_CODE_BOT && iter < 8)
            {
                rng <<= TESTS23E_SYM_BITS;
                int sym = rem;
                synthSeed = (synthSeed + 0x37u) & 0xFFu;
                rem = (int)(synthSeed ^ 0x9Cu);
                sym = (sym << TESTS23E_SYM_BITS | rem) >> (TESTS23E_SYM_BITS - TESTS23E_CODE_EXTRA);
                val = ((val << TESTS23E_SYM_BITS) + (TESTS23E_SYM_MAX & (uint)~sym)) & (TESTS23E_CODE_TOP - 1u);
                iter++;
            }
            output[0] = rng;
            output[1] = val;
            output[2] = (uint)iter;
            output[3] = (uint)rem;
        }

        [TestMethod]
        public async Task Tests23_NormalizeShape_NoBuffer() => await RunEmulatedTest(async accelerator =>
        {
            using var dOut = accelerator.Allocate1D<uint>(4);
            var k = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<uint>>(Tests23_NormalizeShapeNoBufferKernel);
            k((Index1D)1, dOut.View);
            await accelerator.SynchronizeAsync();
            var r = await dOut.CopyToHostAsync<uint>();
            // Expected unsigned-compare exit: 3 iterations (0x80→0x8000→0x800000→0x80000000), Rng=0x80000000.
            // If iter==8 (safety cap) and rng==0, the compare is signed — bug.
            // If iter<3, compare exited too early.
            if (r[0] != 0x80000000u)
                throw new Exception($"NormalizeShape_NoBuffer Rng expected 0x80000000 got 0x{r[0]:X8} (iter={r[2]} val=0x{r[1]:X8} rem=0x{r[3]:X2}). " +
                    "iter==8 + rng==0 → signed-compare bug fires here. iter==3 + rng==0 → shift broke.");
            if (r[2] != 3u)
                throw new Exception($"NormalizeShape_NoBuffer iter expected 3 got {r[2]} (rng=0x{r[0]:X8})");
        });

        // Test F: ADD the byte ArrayView read inside the loop (Tuvok's exact pattern).
        // Test E without the buffer pass; F WITH the buffer. If F fails on
        // CUDA/OpenCL/WebGPU/Wasm but E doesn't, the byte ArrayView read interaction
        // is the trigger.
        static void Tests23_NormalizeShapeWithBufferKernel(
            Index1D _,
            ArrayView<byte> packet, int packetStart, int packetStorage,
            ArrayView<uint> output)
        {
            uint rng = 1u << TESTS23E_CODE_EXTRA;
            uint offs = 0u;
            int  rem = 0;
            if (offs < (uint)packetStorage)
            {
                rem = packet[packetStart + (int)offs];
                offs++;
            }
            uint val = rng - 1u - (uint)(rem >> (TESTS23E_SYM_BITS - TESTS23E_CODE_EXTRA));
            int  iter = 0;
            while (rng <= TESTS23E_CODE_BOT && iter < 8)
            {
                rng <<= TESTS23E_SYM_BITS;
                int sym = rem;
                if (offs < (uint)packetStorage)
                {
                    rem = packet[packetStart + (int)offs];
                    offs++;
                }
                else { rem = 0; }
                sym = (sym << TESTS23E_SYM_BITS | rem) >> (TESTS23E_SYM_BITS - TESTS23E_CODE_EXTRA);
                val = ((val << TESTS23E_SYM_BITS) + (TESTS23E_SYM_MAX & (uint)~sym)) & (TESTS23E_CODE_TOP - 1u);
                iter++;
            }
            output[0] = rng;
            output[1] = val;
            output[2] = offs;
            output[3] = (uint)rem;
            output[4] = (uint)iter;
        }

        [TestMethod]
        public async Task Tests23_NormalizeShape_WithBuffer() => await RunEmulatedTest(async accelerator =>
        {
            var packet = new byte[16];
            for (int i = 0; i < packet.Length; i++)
                packet[i] = (byte)((i * 0x37) ^ 0x9C);
            using var dPacket = accelerator.Allocate1D(packet);
            using var dOut = accelerator.Allocate1D<uint>(5);
            var k = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, int, int, ArrayView<uint>>(Tests23_NormalizeShapeWithBufferKernel);
            k((Index1D)1, dPacket.View, 0, packet.Length, dOut.View);
            await accelerator.SynchronizeAsync();
            var r = await dOut.CopyToHostAsync<uint>();
            // Expected behavior: 3 iterations, Rng=0x80000000, Offs=4 (1 init + 3 iter).
            if (r[0] != 0x80000000u)
                throw new Exception($"NormalizeShape_WithBuffer Rng expected 0x80000000 got 0x{r[0]:X8} " +
                    $"(val=0x{r[1]:X8} offs={r[2]} rem=0x{r[3]:X2} iter={r[4]}). " +
                    $"If iter==8 → loop ran to safety cap, rng update broken or compare wrong. " +
                    $"If iter<3 → exited early.");
            if (r[4] != 3u)
                throw new Exception($"NormalizeShape_WithBuffer iter expected 3 got {r[4]} (rng=0x{r[0]:X8} offs={r[2]})");
        });

        // Test H: ArrayView<long> kernel param backed by a Cast of an int buffer.
        // Surfaced 2026-05-04 by Tuvok's `Vp9FrameEntropyKernel` Wasm OOB — the trap
        // showed `V4=4 bytes wide` for an `ArrayView<long>` param where 8 bytes were
        // needed. Root cause: WasmAccelerator was using `wasmBuf.ElementSize` (4 for
        // the int-allocated parent buffer) instead of the view's element size (8 for
        // long) when computing the view's byte range; the wasm memory copy region
        // was undersized → kernel reads/writes past the end → OOB.
        //
        // This test allocates a buffer as `MemoryBuffer1D<int>`, casts to long view,
        // writes one long value to the view, and reads it back. Pre-fix: Wasm OOB
        // because the byte range reserved is `length*4` instead of `length*8`.
        // Post-fix: passes on every backend.
        static void Tests23_LongViewOverIntBufferKernel(
            Index1D _,
            ArrayView<long> outLen,
            long valueToStore)
        {
            outLen[0] = valueToStore;
        }

        [TestMethod]
        public async Task Tests23_LongViewOverIntBuffer() => await RunEmulatedTest(async accelerator =>
        {
            // Allocate a MemoryBuffer1D<int> with 2 ints (= 8 bytes).
            // Cast to ArrayView<long> — gives a 1-element long view backed by the
            // int allocation. WasmAccelerator must use the VIEW's element size
            // (8 bytes per long), not the buffer's (4 bytes per int), to size the
            // wasm memory copy region.
            using var intBuf = accelerator.Allocate1D<int>(2);
            var longView = intBuf.View.AsContiguous().Cast<long>();
            const long testVal = 0x1122334455667788L;
            var k = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<long>, long>(
                Tests23_LongViewOverIntBufferKernel);
            k((Index1D)1, longView, testVal);
            await accelerator.SynchronizeAsync();

            // Read the parent int buffer back; reinterpret 2 ints as 1 long.
            var ints = await intBuf.CopyToHostAsync<int>();
            long got = ((long)(uint)ints[0]) | (((long)ints[1]) << 32);
            if (got != testVal)
                throw new Exception($"Cast<long> over int buffer: expected 0x{testVal:X16} got 0x{got:X16} (ints=[0x{ints[0]:X8}, 0x{ints[1]:X8}]). " +
                    "If only the LOW 32 bits are correct, WasmAccelerator's view-byte-length used buffer.ElementSize (4) instead of view's actual element size (8) — task #16 follow-up fix.");
        });

        // Test H: NATIVE long buffer (NOT Cast). Tuvok's Vp9 trap surfaced 2026-05-04
        // shows V4 (the kernel's `ArrayView<long> outLen` param) reports byteLen=4
        // even though the underlying buffer is `Allocate1D<long>(1)` directly. Different
        // path from Test E (Cast<long> over int buffer); this isolates the natively-typed
        // long allocation. If the wasm view-byte-length compute uses
        // `wasmBuf.ElementSize` AND that buffer reports 4 bytes per element instead of 8
        // for native long, OR the reflection-based VIEW-element-size override doesn't
        // fire for non-Cast top-level params, this test exposes it.
        static void Tests23_NativeLongBufferKernel(
            Index1D _,
            ArrayView<long> outLen,
            long valueToStore)
        {
            outLen[0] = valueToStore;
        }

        [TestMethod]
        public async Task Tests23_NativeLongBuffer() => await RunEmulatedTest(async accelerator =>
        {
            // Native MemoryBuffer1D<long>(1) - 8 bytes total, kernel param ArrayView<long>.
            // Pre-fix: only LOW 32 bits round-trip on Wasm if buffer.ElementSize=4 path
            // is taken. Post-fix: full 64-bit round-trip on every backend.
            using var longBuf = accelerator.Allocate1D<long>(1);
            const long testVal = 0x1122334455667788L;
            var k = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<long>, long>(
                Tests23_NativeLongBufferKernel);
            k((Index1D)1, longBuf.View, testVal);
            await accelerator.SynchronizeAsync();

            var got = await longBuf.CopyToHostAsync<long>();
            if (got[0] != testVal)
                throw new Exception($"Native ArrayView<long> over Allocate1D<long>(1): expected 0x{testVal:X16} got 0x{got[0]:X16}. " +
                    "If only LOW 32 bits are correct, the view-byte-length compute is using a 4-byte element size for the native long buffer.");
        });

        // Test G: SAME as F but WITHOUT the compound `iter < 8` safety guard.
        // Tuvok's Tests22 inline has no safety cap — loop runs purely on the
        // `rng <= EC_CODE_BOT` condition. This isolates whether the compound
        // && condition was hiding the actual codegen bug.
        // Safety: the loop SHOULD exit in 3 iterations on every backend.
        // If a backend gets stuck in an infinite loop, the test framework's
        // accelerator-level timeout (or kernel watchdog) catches it.
        static void Tests23_NormalizeShapeBareConditionKernel(
            Index1D _,
            ArrayView<byte> packet, int packetStart, int packetStorage,
            ArrayView<uint> output)
        {
            uint rng = 1u << TESTS23E_CODE_EXTRA;
            uint offs = 0u;
            int  rem = 0;
            if (offs < (uint)packetStorage)
            {
                rem = packet[packetStart + (int)offs];
                offs++;
            }
            uint val = rng - 1u - (uint)(rem >> (TESTS23E_SYM_BITS - TESTS23E_CODE_EXTRA));
            int  iter = 0;
            // No safety cap — bare `while (rng <= ...)` matches Tuvok's exact loop shape.
            while (rng <= TESTS23E_CODE_BOT)
            {
                rng <<= TESTS23E_SYM_BITS;
                int sym = rem;
                if (offs < (uint)packetStorage)
                {
                    rem = packet[packetStart + (int)offs];
                    offs++;
                }
                else { rem = 0; }
                sym = (sym << TESTS23E_SYM_BITS | rem) >> (TESTS23E_SYM_BITS - TESTS23E_CODE_EXTRA);
                val = ((val << TESTS23E_SYM_BITS) + (TESTS23E_SYM_MAX & (uint)~sym)) & (TESTS23E_CODE_TOP - 1u);
                iter++;
            }
            output[0] = rng;
            output[1] = val;
            output[2] = offs;
            output[3] = (uint)rem;
            output[4] = (uint)iter;
        }

        [TestMethod]
        public async Task Tests23_NormalizeShape_BareCondition() => await RunEmulatedTest(async accelerator =>
        {
            var packet = new byte[16];
            for (int i = 0; i < packet.Length; i++)
                packet[i] = (byte)((i * 0x37) ^ 0x9C);
            using var dPacket = accelerator.Allocate1D(packet);
            using var dOut = accelerator.Allocate1D<uint>(5);
            var k = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, int, int, ArrayView<uint>>(Tests23_NormalizeShapeBareConditionKernel);
            k((Index1D)1, dPacket.View, 0, packet.Length, dOut.View);
            await accelerator.SynchronizeAsync();
            var r = await dOut.CopyToHostAsync<uint>();
            if (r[0] != 0x80000000u)
                throw new Exception($"NormalizeShape_BareCondition Rng expected 0x80000000 got 0x{r[0]:X8} " +
                    $"(val=0x{r[1]:X8} offs={r[2]} rem=0x{r[3]:X2} iter={r[4]}).");
            if (r[4] != 3u)
                throw new Exception($"NormalizeShape_BareCondition iter expected 3 got {r[4]} (rng=0x{r[0]:X8} offs={r[2]})");
        });

        #endregion
    }
}
