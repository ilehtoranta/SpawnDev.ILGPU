using ILGPU;
using ILGPU.Runtime;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.Demo.Shared.UnitTests
{
    // Tests19: GreedyMerge construct isolation tests
    // Each test exercises ONE construct that the VoxelEngine GreedyMerge kernel relies on,
    // comparing GPU output against CPU reference. If any fail on browser backends, the issue
    // is ILGPU codegen. If all pass, the bug is in the VoxelEngine kernel logic.
    public abstract partial class BackendTestBase
    {
        #region Part 19 Kernel Definitions

        /// <summary>
        /// Test 1: Nested while loop with conditional break (i32 version).
        /// Mimics the GreedyMerge scan loop: extract set bits from a mask, stop at threshold.
        /// </summary>
        static void NestedWhileBreakKernel(Index1D index, ArrayView<int> masks, ArrayView<int> thresholds, ArrayView<int> output, ArrayView<int> counts)
        {
            int mask = masks[index];
            int threshold = thresholds[index];
            int baseOffset = index * 32; // max 32 bits per int
            int count = 0;

            while (mask != 0)
            {
                // Find lowest set bit (trailing zeros on i32)
                int bit = 0;
                int v = mask;
                while ((v & 1) == 0)
                {
                    bit++;
                    v >>= 1;
                }

                if (bit > threshold) break;

                output[baseOffset + count] = bit;
                count++;
                mask &= ~(1 << bit); // clear consumed bit
            }

            counts[index] = count;
        }

        /// <summary>
        /// Test 2: TrailingZeros(long) - manual bit scan on emulated i64.
        /// Exact same algorithm as GreedyMergeKernels.TrailingZeros.
        /// </summary>
        static void TrailingZerosLongKernel(Index1D index, ArrayView<long> input, ArrayView<int> output)
        {
            long value = input[index];
            if (value == 0)
            {
                output[index] = 64;
                return;
            }
            int count = 0;
            long v = value;
            while ((v & 1) == 0)
            {
                count++;
                v >>= 1;
            }
            output[index] = count;
        }

        /// <summary>
        /// Test 3: Atomic.And on i64.
        /// Each thread clears its own bit from a shared i64 value.
        /// Exact pattern from GreedyMerge face mask consumption.
        /// </summary>
        static void AtomicAndLongKernel(Index1D index, ArrayView<long> buffer)
        {
            // Each thread clears bit 'index' from the shared value
            Atomic.And(ref buffer[0], ~(1L << index));
        }

        /// <summary>
        /// Test 4: Complex i64 bit packing - shifts, ORs, masks.
        /// Mimics PackedQuad.Pack() exactly: combine multiple small ints into one long.
        /// </summary>
        static void BitPackLongKernel(Index1D index, ArrayView<int> xs, ArrayView<int> ys, ArrayView<int> zs,
            ArrayView<int> widths, ArrayView<int> heights, ArrayView<int> faces, ArrayView<int> blockTypes,
            ArrayView<long> output)
        {
            int x = xs[index];
            int y = ys[index];
            int z = zs[index];
            int width = widths[index];
            int height = heights[index];
            int face = faces[index];
            int blockType = blockTypes[index];

            // Exact same packing as PackedQuad.Pack
            long packed = (long)(x & 0xF)
                | ((long)(y & 0xF) << 4)
                | ((long)(z & 0xF) << 8)
                | ((long)((width - 1) & 0xF) << 12)
                | ((long)((height - 1) & 0xF) << 16)
                | ((long)(face & 0x7) << 20)
                | ((long)(blockType & 0xFFF) << 23);

            output[index] = packed;
        }

        /// <summary>
        /// Test 5: Nested while + TrailingZeros(long) combined.
        /// i64 version of Test 1 - scan bits in a long mask using TrailingZeros(long).
        /// This is the exact core loop from MergePerpendicularPlane.
        /// </summary>
        static void NestedWhileTrailingZerosLongKernel(Index1D index, ArrayView<long> masks, ArrayView<int> output, ArrayView<int> counts)
        {
            long mask = masks[index];
            int baseOffset = index * 64; // max 64 bits per long
            int count = 0;

            while (mask != 0)
            {
                // TrailingZeros(long) inline
                int bit = 0;
                long v = mask;
                if (v == 0)
                {
                    bit = 64;
                }
                else
                {
                    while ((v & 1) == 0)
                    {
                        bit++;
                        v >>= 1;
                    }
                }

                if (bit >= 64) break;

                // Find run length of consecutive set bits starting at 'bit'
                int runLen = 1;
                while (bit + runLen < 64 && (mask & (1L << (bit + runLen))) != 0)
                {
                    runLen++;
                }

                // Clear consumed bits
                for (int b = 0; b < runLen; b++)
                {
                    mask &= ~(1L << (bit + b));
                }

                // Output: encode (bit position, run length) as bit * 100 + runLen
                output[baseOffset + count] = bit * 100 + runLen;
                count++;
            }

            counts[index] = count;
        }

        /// <summary>
        /// Test 6: Multiple inlined static method calls.
        /// Two static methods (MethodA, MethodB) both write output via atomic counter.
        /// Mimics MergeXZPlane + MergePerpendicularPlane being called from the main kernel.
        /// </summary>
        static void InlinedMethodCallKernel(Index1D index, ArrayView<int> input, ArrayView<int> output, ArrayView<int> counter)
        {
            int val = input[index];
            InlinedMethodA(val, output, counter);
            InlinedMethodB(val, output, counter);
        }

        private static void InlinedMethodA(int val, ArrayView<int> output, ArrayView<int> counter)
        {
            // Square the value and emit via atomic counter
            int result = val * val;
            int slot = Atomic.Add(ref counter[0], 1);
            if (slot < output.IntLength)
            {
                output[slot] = result;
            }
        }

        private static void InlinedMethodB(int val, ArrayView<int> output, ArrayView<int> counter)
        {
            // Cube the value and emit via atomic counter
            int result = val * val * val;
            int slot = Atomic.Add(ref counter[0], 1);
            if (slot < output.IntLength)
            {
                output[slot] = result;
            }
        }

        /// <summary>
        /// Test 7: Full combined pattern - all constructs together.
        /// Minimal GreedyMerge reproduction: nested while + TrailingZeros(long) + Atomic.And(long)
        /// + bit packing into i64 + static method call. 4x4 grid, 16 columns, known bit patterns.
        /// </summary>
        static void FullCombinedPatternKernel(Index1D index,
            ArrayView<long> faceMasks,
            ArrayView<int> blockTypes,
            ArrayView<long> outputQuads,
            ArrayView<int> quadCounter,
            int chunkSize)
        {
            // Each thread processes one column, scanning all set bits in its face mask
            long mask = faceMasks[index];
            int col = index;
            int x = col % chunkSize;
            int z = col / chunkSize;

            while (mask != 0)
            {
                // TrailingZeros(long) inline
                int y = 0;
                long v = mask;
                if (v == 0) { y = 64; }
                else
                {
                    while ((v & 1) == 0)
                    {
                        y++;
                        v >>= 1;
                    }
                }
                if (y >= 64) break;

                int blockType = blockTypes[index * 64 + y];

                // Find run of consecutive same-type bits
                int h = 1;
                while (y + h < 64 && (mask & (1L << (y + h))) != 0)
                {
                    if (blockTypes[index * 64 + y + h] != blockType) break;
                    h++;
                }

                // Clear consumed bits via Atomic.And(long)
                for (int dy = 0; dy < h; dy++)
                {
                    Atomic.And(ref faceMasks[index], ~(1L << (y + dy)));
                }
                // Also clear local copy
                for (int dy = 0; dy < h; dy++)
                {
                    mask &= ~(1L << (y + dy));
                }

                // Pack quad and emit
                EmitPackedQuad(x, y, z, 1, h, 4, blockType, outputQuads, quadCounter);
            }
        }

        private static void EmitPackedQuad(int x, int y, int z, int w, int h, int face, int blockType,
            ArrayView<long> outputQuads, ArrayView<int> quadCounter)
        {
            long packed = (long)(x & 0xF)
                | ((long)(y & 0xF) << 4)
                | ((long)(z & 0xF) << 8)
                | ((long)((w - 1) & 0xF) << 12)
                | ((long)((h - 1) & 0xF) << 16)
                | ((long)(face & 0x7) << 20)
                | ((long)(blockType & 0xFFF) << 23);

            int slot = Atomic.Add(ref quadCounter[0], 1);
            if (slot < outputQuads.IntLength)
            {
                outputQuads[slot] = packed;
            }
            else
            {
                Atomic.Add(ref quadCounter[0], -1);
            }
        }

        #endregion

        #region Part 19 Tests

        /// <summary>
        /// Test 1: Nested while loop with conditional break (i32).
        /// Scans bits from mask, stops at threshold. Tests the core GreedyMerge scan pattern
        /// using i32 to isolate control flow from i64 emulation.
        /// </summary>
        [TestMethod]
        public async Task NestedWhileBreakI32Test() => await RunTest(async accelerator =>
        {
            // 8 test cases with varied masks and thresholds
            int[] maskData = new int[]
            {
                0b10110101,   // bits at 0,2,4,5,7
                0b11111111,   // bits at 0-7
                0b10000001,   // bits at 0 and 7
                0b00001000,   // bit at 3
                0b11111111_11111111, // 16 bits set
                0b01010101_01010101, // every other bit, 0-14
                0,            // no bits
                unchecked((int)0x80000001), // bits at 0 and 31
            };
            int[] thresholdData = new int[] { 5, 10, 31, 31, 8, 31, 31, 31 };
            int count = maskData.Length;

            using var gpuMasks = accelerator.Allocate1D(maskData);
            using var gpuThresholds = accelerator.Allocate1D(thresholdData);
            using var gpuOutput = accelerator.Allocate1D<int>(count * 32);
            using var gpuCounts = accelerator.Allocate1D<int>(count);
            gpuOutput.MemSetToZero();
            gpuCounts.MemSetToZero();

            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>>(NestedWhileBreakKernel);
            kernel((Index1D)count, gpuMasks.View, gpuThresholds.View, gpuOutput.View, gpuCounts.View);
            await accelerator.SynchronizeAsync();

            var resultOutput = await gpuOutput.CopyToHostAsync<int>();
            var resultCounts = await gpuCounts.CopyToHostAsync<int>();

            // CPU reference
            for (int i = 0; i < count; i++)
            {
                int mask = maskData[i];
                int threshold = thresholdData[i];
                var expectedBits = new List<int>();

                while (mask != 0)
                {
                    int bit = 0;
                    int tv = mask;
                    while ((tv & 1) == 0) { bit++; tv >>= 1; }
                    if (bit > threshold) break;
                    expectedBits.Add(bit);
                    mask &= ~(1 << bit);
                }

                if (resultCounts[i] != expectedBits.Count)
                    throw new Exception($"NestedWhileBreakI32 case {i}: count mismatch. Expected {expectedBits.Count}, got {resultCounts[i]}. Mask=0x{maskData[i]:X8}, threshold={threshold}");

                int baseOffset = i * 32;
                for (int j = 0; j < expectedBits.Count; j++)
                {
                    if (resultOutput[baseOffset + j] != expectedBits[j])
                        throw new Exception($"NestedWhileBreakI32 case {i} bit {j}: Expected {expectedBits[j]}, got {resultOutput[baseOffset + j]}");
                }
            }
        });

        /// <summary>
        /// Test 2: TrailingZeros(long) - manual bit scan on emulated i64.
        /// Tests the exact algorithm from GreedyMergeKernels.TrailingZeros with known bit patterns.
        /// </summary>
        [TestMethod]
        public async Task TrailingZerosLongTest() => await RunEmulatedTest(async accelerator =>
        {
            long[] inputData = new long[]
            {
                1L,                     // bit 0 -> 0
                2L,                     // bit 1 -> 1
                4L,                     // bit 2 -> 2
                0x10L,                  // bit 4 -> 4
                0x80L,                  // bit 7 -> 7
                0x8000L,               // bit 15 -> 15
                0x80000000L,           // bit 31 -> 31
                0x100000000L,          // bit 32 -> 32
                0x800000000L,          // bit 35 -> 35
                0x4000000000000000L,   // bit 62 -> 62
                long.MinValue,         // bit 63 -> 63
                0L,                    // no bits -> 64
                0b10110100L,           // bits at 2,4,5,7 -> 2
                0x0F00000000000000L,   // bits at 56-59 -> 56
                unchecked((long)0xFFFFFFFFFFFFFFFF), // all bits -> 0
                0x1000000000L,         // bit 36 -> 36
            };
            int count = inputData.Length;
            int[] expectedResults = new int[]
            {
                0, 1, 2, 4, 7, 15, 31, 32, 35, 62, 63, 64, 2, 56, 0, 36
            };

            using var gpuInput = accelerator.Allocate1D(inputData);
            using var gpuOutput = accelerator.Allocate1D<int>(count);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<long>, ArrayView<int>>(TrailingZerosLongKernel);
            kernel((Index1D)count, gpuInput.View, gpuOutput.View);
            await accelerator.SynchronizeAsync();

            var result = await gpuOutput.CopyToHostAsync<int>();

            for (int i = 0; i < count; i++)
            {
                if (result[i] != expectedResults[i])
                    throw new Exception($"TrailingZerosLong case {i}: Expected {expectedResults[i]}, got {result[i]}. Input=0x{inputData[i]:X16}");
            }
        });

        /// <summary>
        /// Test 3: Atomic.And on i64.
        /// 16 threads each clear their own bit from a value that starts with all 16 lower bits set.
        /// Verifies the exact pattern used in GreedyMerge to mark consumed faces.
        /// </summary>
        [TestMethod]
        public async Task AtomicAndLongTest() => await RunEmulatedTest(async accelerator =>
        {
            int threadCount = 16;
            long initialValue = (1L << threadCount) - 1; // lower 16 bits set = 0xFFFF

            using var gpuBuffer = accelerator.Allocate1D(new long[] { initialValue });

            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<long>>(AtomicAndLongKernel);
            kernel((Index1D)threadCount, gpuBuffer.View);
            await accelerator.SynchronizeAsync();

            var result = await gpuBuffer.CopyToHostAsync<long>();

            // All 16 lower bits should be cleared
            if (result[0] != 0)
                throw new Exception($"AtomicAndLong: Expected 0, got 0x{result[0]:X16}. All 16 bits should have been cleared.");

            // Now test with partial clearing: only clear bits 0-7
            long initial2 = unchecked((long)0xFFFFFFFFFFFFFFFF); // all 64 bits set
            using var gpuBuffer2 = accelerator.Allocate1D(new long[] { initial2 });

            kernel((Index1D)8, gpuBuffer2.View);
            await accelerator.SynchronizeAsync();

            var result2 = await gpuBuffer2.CopyToHostAsync<long>();
            long expected2 = initial2 & ~0xFFL; // bits 0-7 cleared, rest intact
            if (result2[0] != expected2)
                throw new Exception($"AtomicAndLong partial: Expected 0x{expected2:X16}, got 0x{result2[0]:X16}. Bits 0-7 should be cleared, rest intact.");
        });

        /// <summary>
        /// Test 4: Complex i64 bit packing - shifts, ORs, masks.
        /// GPU packs values using the exact PackedQuad.Pack algorithm, CPU unpacks and verifies.
        /// Real values covering the full range of each field.
        /// </summary>
        [TestMethod]
        public async Task BitPackLongTest() => await RunEmulatedTest(async accelerator =>
        {
            // Test data covering various field values
            int[] xs =         { 0,  15, 7,  3,  12, 8,  0,  15 };
            int[] ys =         { 0,  15, 5,  11, 0,  7,  14, 3  };
            int[] zs =         { 0,  15, 9,  2,  6,  13, 1,  10 };
            int[] widths =     { 1,  16, 8,  4,  12, 3,  7,  16 }; // stored as width-1
            int[] heights =    { 1,  16, 5,  10, 2,  15, 9,  6  }; // stored as height-1
            int[] faces =      { 0,  5,  2,  4,  1,  3,  5,  0  };
            int[] blockTypes = { 0,  4095, 1, 2048, 100, 3999, 512, 777 };
            int count = xs.Length;

            using var gpuXs = accelerator.Allocate1D(xs);
            using var gpuYs = accelerator.Allocate1D(ys);
            using var gpuZs = accelerator.Allocate1D(zs);
            using var gpuWidths = accelerator.Allocate1D(widths);
            using var gpuHeights = accelerator.Allocate1D(heights);
            using var gpuFaces = accelerator.Allocate1D(faces);
            using var gpuBlockTypes = accelerator.Allocate1D(blockTypes);
            using var gpuOutput = accelerator.Allocate1D<long>(count);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>,
                ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<long>>(BitPackLongKernel);
            kernel((Index1D)count, gpuXs.View, gpuYs.View, gpuZs.View,
                gpuWidths.View, gpuHeights.View, gpuFaces.View, gpuBlockTypes.View, gpuOutput.View);
            await accelerator.SynchronizeAsync();

            var result = await gpuOutput.CopyToHostAsync<long>();

            // CPU reference: pack + unpack + verify each field
            for (int i = 0; i < count; i++)
            {
                long gpuPacked = result[i];

                // CPU pack for comparison
                long cpuPacked = (long)(xs[i] & 0xF)
                    | ((long)(ys[i] & 0xF) << 4)
                    | ((long)(zs[i] & 0xF) << 8)
                    | ((long)((widths[i] - 1) & 0xF) << 12)
                    | ((long)((heights[i] - 1) & 0xF) << 16)
                    | ((long)(faces[i] & 0x7) << 20)
                    | ((long)(blockTypes[i] & 0xFFF) << 23);

                if (gpuPacked != cpuPacked)
                    throw new Exception($"BitPackLong case {i}: packed mismatch. CPU=0x{cpuPacked:X16}, GPU=0x{gpuPacked:X16}");

                // Also unpack the GPU result and verify each field
                int ux = (int)(gpuPacked & 0xF);
                int uy = (int)((gpuPacked >> 4) & 0xF);
                int uz = (int)((gpuPacked >> 8) & 0xF);
                int uw = (int)((gpuPacked >> 12) & 0xF) + 1;
                int uh = (int)((gpuPacked >> 16) & 0xF) + 1;
                int uf = (int)((gpuPacked >> 20) & 0x7);
                int ubt = (int)((gpuPacked >> 23) & 0xFFF);

                if (ux != xs[i]) throw new Exception($"BitPackLong case {i}: x unpack. Expected {xs[i]}, got {ux}");
                if (uy != ys[i]) throw new Exception($"BitPackLong case {i}: y unpack. Expected {ys[i]}, got {uy}");
                if (uz != zs[i]) throw new Exception($"BitPackLong case {i}: z unpack. Expected {zs[i]}, got {uz}");
                if (uw != widths[i]) throw new Exception($"BitPackLong case {i}: width unpack. Expected {widths[i]}, got {uw}");
                if (uh != heights[i]) throw new Exception($"BitPackLong case {i}: height unpack. Expected {heights[i]}, got {uh}");
                if (uf != faces[i]) throw new Exception($"BitPackLong case {i}: face unpack. Expected {faces[i]}, got {uf}");
                if (ubt != blockTypes[i]) throw new Exception($"BitPackLong case {i}: blockType unpack. Expected {blockTypes[i]}, got {ubt}");
            }
        });

        /// <summary>
        /// Test 5: Nested while + TrailingZeros(long) combined.
        /// Scans a long mask using TrailingZeros to find each bit, then finds consecutive runs.
        /// This is the exact loop from MergePerpendicularPlane: scan bits, find runs, clear them.
        /// If Tests 1 and 2 pass but this fails, the bug is i64 + control flow interaction.
        /// </summary>
        [TestMethod]
        public async Task NestedWhileTrailingZerosLongTest() => await RunEmulatedTest(async accelerator =>
        {
            long[] maskData = new long[]
            {
                0b111L,                          // 3 consecutive bits at 0 -> (0, runLen=3)
                0b1110100L,                      // bits 2, 4, 5, 6 -> (2,1), (4,3)
                0x00FF000000000000L,             // 8 consecutive bits at 48 -> (48, 8)
                0b10101010L,                     // bits 1,3,5,7 each isolated -> (1,1),(3,1),(5,1),(7,1)
                1L | (1L << 31) | (1L << 32) | (1L << 33), // bit 0 alone, bits 31-33 run of 3
                unchecked((long)0xFFFFFFFFFFFFFFFF), // all 64 bits: one run (0, 64)
                0L,                              // empty -> no output
                (1L << 63),                      // bit 63 only -> (63, 1)
            };
            int count = maskData.Length;

            using var gpuMasks = accelerator.Allocate1D(maskData);
            using var gpuOutput = accelerator.Allocate1D<int>(count * 64);
            using var gpuCounts = accelerator.Allocate1D<int>(count);
            gpuOutput.MemSetToZero();
            gpuCounts.MemSetToZero();

            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<long>, ArrayView<int>, ArrayView<int>>(NestedWhileTrailingZerosLongKernel);
            kernel((Index1D)count, gpuMasks.View, gpuOutput.View, gpuCounts.View);
            await accelerator.SynchronizeAsync();

            var resultOutput = await gpuOutput.CopyToHostAsync<int>();
            var resultCounts = await gpuCounts.CopyToHostAsync<int>();

            // CPU reference
            for (int i = 0; i < count; i++)
            {
                long mask = maskData[i];
                var expectedRuns = new List<int>(); // encoded as bit*100 + runLen

                while (mask != 0)
                {
                    int bit = 0;
                    long tv = mask;
                    if (tv == 0) { bit = 64; }
                    else { while ((tv & 1) == 0) { bit++; tv >>= 1; } }
                    if (bit >= 64) break;

                    int runLen = 1;
                    while (bit + runLen < 64 && (mask & (1L << (bit + runLen))) != 0)
                    {
                        runLen++;
                    }
                    for (int b = 0; b < runLen; b++)
                    {
                        mask &= ~(1L << (bit + b));
                    }
                    expectedRuns.Add(bit * 100 + runLen);
                }

                if (resultCounts[i] != expectedRuns.Count)
                    throw new Exception($"NestedWhileTrailingZerosLong case {i}: count mismatch. Expected {expectedRuns.Count}, got {resultCounts[i]}. Mask=0x{maskData[i]:X16}");

                int baseOffset = i * 64;
                for (int j = 0; j < expectedRuns.Count; j++)
                {
                    if (resultOutput[baseOffset + j] != expectedRuns[j])
                    {
                        int expBit = expectedRuns[j] / 100;
                        int expRun = expectedRuns[j] % 100;
                        int gotBit = resultOutput[baseOffset + j] / 100;
                        int gotRun = resultOutput[baseOffset + j] % 100;
                        throw new Exception($"NestedWhileTrailingZerosLong case {i} run {j}: Expected bit={expBit} runLen={expRun}, got bit={gotBit} runLen={gotRun}");
                    }
                }
            }
        });

        /// <summary>
        /// Test 6: Multiple inlined static method calls with atomic counter output.
        /// Two static methods (A: square, B: cube) each emit results via atomic counter.
        /// Verifies method inlining doesn't break output or corrupt control flow.
        /// </summary>
        [TestMethod]
        public async Task InlinedStaticMethodCallsTest() => await RunTest(async accelerator =>
        {
            int[] inputData = { 2, 3, 5, 7, 11, 13, 4, 6 };
            int count = inputData.Length;

            using var gpuInput = accelerator.Allocate1D(inputData);
            using var gpuOutput = accelerator.Allocate1D<int>(count * 2); // 2 outputs per thread
            using var gpuCounter = accelerator.Allocate1D<int>(1);
            gpuOutput.MemSetToZero();
            gpuCounter.MemSetToZero();

            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>(InlinedMethodCallKernel);
            kernel((Index1D)count, gpuInput.View, gpuOutput.View, gpuCounter.View);
            await accelerator.SynchronizeAsync();

            var resultOutput = await gpuOutput.CopyToHostAsync<int>();
            var resultCounter = await gpuCounter.CopyToHostAsync<int>();

            // Should have exactly count*2 outputs (one square + one cube per thread)
            if (resultCounter[0] != count * 2)
                throw new Exception($"InlinedStaticMethodCalls: counter mismatch. Expected {count * 2}, got {resultCounter[0]}");

            // Collect all outputs and verify all expected values are present
            var outputSet = new HashSet<int>();
            for (int i = 0; i < count * 2; i++)
            {
                outputSet.Add(resultOutput[i]);
            }

            for (int i = 0; i < count; i++)
            {
                int val = inputData[i];
                int square = val * val;
                int cube = val * val * val;

                if (!outputSet.Contains(square))
                    throw new Exception($"InlinedStaticMethodCalls: missing square of {val} ({square}) in output");
                if (!outputSet.Contains(cube))
                    throw new Exception($"InlinedStaticMethodCalls: missing cube of {val} ({cube}) in output");
            }
        });

        /// <summary>
        /// Test 7: Full combined pattern - all constructs together.
        /// Minimal GreedyMerge reproduction: nested while + TrailingZeros(long) + Atomic.And(long)
        /// + bit packing into i64 + static method call. 4x4 grid, 16 columns, known bit patterns.
        /// If Tests 1-6 pass but this fails, the bug is a codegen interaction between constructs.
        /// </summary>
        [TestMethod]
        public async Task FullCombinedPatternTest() => await RunEmulatedTest(async accelerator =>
        {
            int chunkSize = 4;
            int columns = chunkSize * chunkSize; // 16 columns

            // Each column has a face mask (64 bits, each bit = a Y layer)
            // and 64 block types (one per possible Y layer)
            long[] faceMaskData = new long[columns];
            int[] blockTypeData = new int[columns * 64];

            // Set up known patterns:
            // Column 0 (x=0,z=0): bits 0,1,2 set, all type 100 -> should produce 1 quad (y=0, h=3)
            faceMaskData[0] = 0b111L;
            blockTypeData[0 * 64 + 0] = 100;
            blockTypeData[0 * 64 + 1] = 100;
            blockTypeData[0 * 64 + 2] = 100;

            // Column 1 (x=1,z=0): bits 5,6 set, type 200 -> 1 quad (y=5, h=2)
            faceMaskData[1] = (1L << 5) | (1L << 6);
            blockTypeData[1 * 64 + 5] = 200;
            blockTypeData[1 * 64 + 6] = 200;

            // Column 5 (x=1,z=1): bits 0,1 type 300 + bit 3 type 400 -> 2 quads
            faceMaskData[5] = 0b1011L;
            blockTypeData[5 * 64 + 0] = 300;
            blockTypeData[5 * 64 + 1] = 300;
            blockTypeData[5 * 64 + 3] = 400;

            // Column 10 (x=2,z=2): bit 31 and bit 32 set, both type 500 -> 1 quad spanning the i32 boundary
            faceMaskData[10] = (1L << 31) | (1L << 32);
            blockTypeData[10 * 64 + 31] = 500;
            blockTypeData[10 * 64 + 32] = 500;

            // Column 15 (x=3,z=3): bit 63 only, type 4095 (max) -> 1 quad at highest bit
            faceMaskData[15] = (1L << 63);
            blockTypeData[15 * 64 + 63] = 4095;

            // Rest of columns: mask = 0, no quads

            int expectedQuadCount = 6; // 1 + 1 + 2 + 1 + 1
            int maxQuads = 64;

            using var gpuFaceMasks = accelerator.Allocate1D(faceMaskData);
            using var gpuBlockTypes = accelerator.Allocate1D(blockTypeData);
            using var gpuOutputQuads = accelerator.Allocate1D<long>(maxQuads);
            using var gpuQuadCounter = accelerator.Allocate1D<int>(1);
            gpuOutputQuads.MemSetToZero();
            gpuQuadCounter.MemSetToZero();

            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<long>, ArrayView<int>, ArrayView<long>, ArrayView<int>, int>(FullCombinedPatternKernel);
            kernel((Index1D)columns, gpuFaceMasks.View, gpuBlockTypes.View, gpuOutputQuads.View, gpuQuadCounter.View, chunkSize);
            await accelerator.SynchronizeAsync();

            var resultQuads = await gpuOutputQuads.CopyToHostAsync<long>();
            var resultCounter = await gpuQuadCounter.CopyToHostAsync<int>();

            if (resultCounter[0] != expectedQuadCount)
                throw new Exception($"FullCombinedPattern: quad count mismatch. Expected {expectedQuadCount}, got {resultCounter[0]}");

            // Unpack all quads and verify each one matches expected output
            // Build expected set: (x, y, z, w, h, face, blockType)
            var expected = new HashSet<string>
            {
                "x=0,y=0,z=0,w=1,h=3,f=4,bt=100",   // column 0
                "x=1,y=5,z=0,w=1,h=2,f=4,bt=200",   // column 1
                "x=1,y=0,z=1,w=1,h=2,f=4,bt=300",   // column 5 first run
                "x=1,y=3,z=1,w=1,h=1,f=4,bt=400",   // column 5 second run
                "x=2,y=15,z=2,w=1,h=2,f=4,bt=500",  // column 10 (y=31 masked to 4 bits = 15)
                "x=3,y=15,z=3,w=1,h=1,f=4,bt=4095", // column 15 (y=63 masked to 4 bits = 15)
            };

            var actual = new HashSet<string>();
            for (int i = 0; i < resultCounter[0]; i++)
            {
                long packed = resultQuads[i];
                int ux = (int)(packed & 0xF);
                int uy = (int)((packed >> 4) & 0xF);
                int uz = (int)((packed >> 8) & 0xF);
                int uw = (int)((packed >> 12) & 0xF) + 1;
                int uh = (int)((packed >> 16) & 0xF) + 1;
                int uf = (int)((packed >> 20) & 0x7);
                int ubt = (int)((packed >> 23) & 0xFFF);
                actual.Add($"x={ux},y={uy},z={uz},w={uw},h={uh},f={uf},bt={ubt}");
            }

            foreach (var exp in expected)
            {
                if (!actual.Contains(exp))
                    throw new Exception($"FullCombinedPattern: missing expected quad: {exp}. Got: [{string.Join("], [", actual)}]");
            }

            foreach (var act in actual)
            {
                if (!expected.Contains(act))
                    throw new Exception($"FullCombinedPattern: unexpected quad: {act}. Expected: [{string.Join("], [", expected)}]");
            }
        });

        #endregion

        #region Part 19b: i64 AtomicAdd CAS Loop Test

        /// <summary>
        /// Kernel: multiple threads concurrently Atomic.Add to the same i64 value.
        /// Tests the CAS-lo + atomicAdd-hi lock-free implementation on WebGPU.
        /// Each thread adds its index+1 to a shared accumulator.
        /// Expected result: sum of 1..N = N*(N+1)/2.
        /// </summary>
        static void AtomicAddLongConcurrentKernel(Index1D index, ArrayView<long> accumulator)
        {
            Atomic.Add(ref accumulator[0], (long)((int)index + 1));
        }

        /// <summary>
        /// Tests concurrent i64 Atomic.Add with many threads writing to the same location.
        /// Verifies the CAS loop produces the correct sum (no lost updates).
        /// </summary>
        [TestMethod]
        public async Task AtomicAddLongConcurrentTest() => await RunEmulatedTest(async accelerator =>
        {
            int threadCount = 64;
            long expectedSum = (long)threadCount * (threadCount + 1) / 2; // sum of 1..N

            using var gpuAccum = accelerator.Allocate1D(new long[] { 0L });

            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<long>>(AtomicAddLongConcurrentKernel);
            kernel((Index1D)threadCount, gpuAccum.View);
            await accelerator.SynchronizeAsync();

            var result = await gpuAccum.CopyToHostAsync<long>();

            if (result[0] != expectedSum)
                throw new Exception($"AtomicAddLongConcurrent: Expected {expectedSum}, got {result[0]}. " +
                    $"Lost updates from {threadCount} concurrent writers.");
        });

        /// <summary>
        /// Kernel: Atomic.Add with large values that cause carry from lo to hi half.
        /// Each thread adds 0x80000000 (2^31), so every 2 adds cause a carry.
        /// Tests that the CAS loop handles carry correctly under contention.
        /// </summary>
        static void AtomicAddLongCarryKernel(Index1D index, ArrayView<long> accumulator)
        {
            Atomic.Add(ref accumulator[0], 0x80000000L); // 2^31 per thread
        }

        /// <summary>
        /// Tests i64 Atomic.Add with values that force carry between lo and hi halves.
        /// 32 threads each add 2^31 = 0x80000000. Every 2 threads overflow the lo half.
        /// Expected: 32 * 0x80000000 = 0x10_0000_0000 = 68719476736.
        /// </summary>
        [TestMethod]
        public async Task AtomicAddLongCarryTest() => await RunEmulatedTest(async accelerator =>
        {
            int threadCount = 32;
            long addValue = 0x80000000L; // 2^31
            long expectedSum = (long)threadCount * addValue; // 32 * 2^31 = 2^36

            using var gpuAccum = accelerator.Allocate1D(new long[] { 0L });

            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<long>>(AtomicAddLongCarryKernel);
            kernel((Index1D)threadCount, gpuAccum.View);
            await accelerator.SynchronizeAsync();

            var result = await gpuAccum.CopyToHostAsync<long>();

            if (result[0] != expectedSum)
                throw new Exception($"AtomicAddLongCarry: Expected {expectedSum} (0x{expectedSum:X}), " +
                    $"got {result[0]} (0x{result[0]:X}). Carry between lo/hi halves is broken.");
        });

        #endregion
    }
}
