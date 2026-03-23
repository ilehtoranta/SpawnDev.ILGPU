using ILGPU;
using ILGPU.Runtime;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.Demo.Shared.UnitTests
{
    // Part 14: Math intrinsic + int buffer codegen regression tests.
    //
    // Bug 1: (int)MathF.Floor(x) optimization strips Floor
    //   MathF.Floor(-0.25) = -1.0, (int)(-1.0) = -1. Correct.
    //   Without Floor: (int)(-0.25) = 0. Wrong.
    //   The ILGPU compiler optimizes (int)MathF.Floor(x) → (int)x, which is
    //   incorrect for negative values with fractional parts.
    //
    // Bug 2: Int buffer bitwise operations on WebGPU
    //   Packed RGBA int buffers with bitwise shift and mask should correctly
    //   extract channel values on all backends.
    //
    // Discovered via ImageTransformKernel in SpawnDev.ILGPU.ML.
    public abstract partial class BackendTestBase
    {
        // ═══════════════════════════════════════════════════════════
        //  Bug 1: MathF.Floor before int cast
        // ═══════════════════════════════════════════════════════════

        /// <summary>
        /// Kernel that floors a float and casts to int. Tests that floor(-0.25) = -1, not 0.
        /// Uses the combined expression (int)MathF.Floor(x) that triggers the optimization bug.
        /// </summary>
        private static void FloorToIntKernel(Index1D idx,
            ArrayView1D<float, Stride1D.Dense> input,
            ArrayView1D<int, Stride1D.Dense> output)
        {
            output[idx] = (int)MathF.Floor(input[idx]);
        }

        /// <summary>
        /// Same as above but with two-statement pattern to see if it bypasses the optimization.
        /// </summary>
        private static void FloorToIntTwoStepKernel(Index1D idx,
            ArrayView1D<float, Stride1D.Dense> input,
            ArrayView1D<int, Stride1D.Dense> output)
        {
            float floored = MathF.Floor(input[idx]);
            output[idx] = (int)floored;
        }

        [TestMethod]
        public async Task FloorToInt_NegativeFractional_ReturnsFloor() => await RunTest(async accelerator =>
        {
            // Test values: positive, negative fractional, negative whole, zero, edge cases
            var input = new float[] { 1.7f, -0.25f, -1.0f, 0.0f, -0.99f, 2.0f, -3.5f, 0.5f };
            var expected = new int[] { 1, -1, -1, 0, -1, 2, -4, 0 };

            using var inBuf = accelerator.Allocate1D(input);
            using var outBuf = accelerator.Allocate1D<int>(input.Length);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>>(FloorToIntKernel);
            kernel(input.Length, inBuf.View, outBuf.View);
            await accelerator.SynchronizeAsync();

            var actual = await outBuf.CopyToHostAsync<int>();

            for (int i = 0; i < expected.Length; i++)
            {
                if (actual[i] != expected[i])
                    throw new Exception($"FloorToInt[{i}]: input={input[i]}, expected={expected[i]}, actual={actual[i]}. " +
                        $"MathF.Floor optimization bug: (int)MathF.Floor({input[i]}) should be {expected[i]}, got {actual[i]}");
            }
        });

        [TestMethod]
        public async Task FloorToInt_TwoStep_NegativeFractional() => await RunTest(async accelerator =>
        {
            var input = new float[] { 1.7f, -0.25f, -1.0f, 0.0f, -0.99f, 2.0f, -3.5f, 0.5f };
            var expected = new int[] { 1, -1, -1, 0, -1, 2, -4, 0 };

            using var inBuf = accelerator.Allocate1D(input);
            using var outBuf = accelerator.Allocate1D<int>(input.Length);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>>(FloorToIntTwoStepKernel);
            kernel(input.Length, inBuf.View, outBuf.View);
            await accelerator.SynchronizeAsync();

            var actual = await outBuf.CopyToHostAsync<int>();

            for (int i = 0; i < expected.Length; i++)
            {
                if (actual[i] != expected[i])
                    throw new Exception($"FloorToInt_TwoStep[{i}]: input={input[i]}, expected={expected[i]}, actual={actual[i]}");
            }
        });

        // ═══════════════════════════════════════════════════════════
        //  Bug 2: Int buffer bitwise shift + mask
        // ═══════════════════════════════════════════════════════════

        /// <summary>
        /// Kernel that reads packed RGBA ints and extracts channels via shift + mask.
        /// Tests that bitwise ops on i32 work correctly on all backends.
        /// output[idx*4+0] = R, [idx*4+1] = G, [idx*4+2] = B, [idx*4+3] = A
        /// </summary>
        private static void UnpackRGBAKernel(Index1D idx,
            ArrayView1D<int, Stride1D.Dense> packed,
            ArrayView1D<int, Stride1D.Dense> unpacked)
        {
            int p = packed[idx];
            unpacked[idx * 4 + 0] = p & 0xFF;           // R
            unpacked[idx * 4 + 1] = (p >> 8) & 0xFF;    // G
            unpacked[idx * 4 + 2] = (p >> 16) & 0xFF;   // B
            unpacked[idx * 4 + 3] = (p >> 24) & 0xFF;   // A
        }

        /// <summary>
        /// Kernel that packs RGBA channels into a single int via shift + or.
        /// </summary>
        private static void PackRGBAKernel(Index1D idx,
            ArrayView1D<int, Stride1D.Dense> channels,
            ArrayView1D<int, Stride1D.Dense> packed)
        {
            int r = channels[idx * 4 + 0] & 0xFF;
            int g = channels[idx * 4 + 1] & 0xFF;
            int b = channels[idx * 4 + 2] & 0xFF;
            int a = channels[idx * 4 + 3] & 0xFF;
            packed[idx] = r | (g << 8) | (b << 16) | (a << 24);
        }

        [TestMethod]
        public async Task IntBuffer_UnpackRGBA_CorrectChannels() => await RunTest(async accelerator =>
        {
            // Pack RGBA: R=255, G=0, B=0, A=255 (red) = 0xFF0000FF
            // Pack RGBA: R=0, G=255, B=0, A=255 (green) = 0xFF00FF00
            // Pack RGBA: R=128, G=64, B=32, A=200 = 0xC8204080
            var packed = new int[]
            {
                unchecked((int)0xFF0000FF),  // red
                unchecked((int)0xFF00FF00),  // green
                unchecked((int)0xC8204080),  // custom
            };
            var expected = new int[]
            {
                255, 0, 0, 255,     // red: R=255, G=0, B=0, A=255
                0, 255, 0, 255,     // green: R=0, G=255, B=0, A=255
                128, 64, 32, 200,   // custom: R=128, G=64, B=32, A=200
            };

            using var packedBuf = accelerator.Allocate1D(packed);
            using var unpackedBuf = accelerator.Allocate1D<int>(packed.Length * 4);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<int, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>>(UnpackRGBAKernel);
            kernel(packed.Length, packedBuf.View, unpackedBuf.View);
            await accelerator.SynchronizeAsync();

            var actual = await unpackedBuf.CopyToHostAsync<int>();

            for (int i = 0; i < expected.Length; i++)
            {
                if (actual[i] != expected[i])
                {
                    int pixelIdx = i / 4;
                    string channel = (i % 4) switch { 0 => "R", 1 => "G", 2 => "B", _ => "A" };
                    throw new Exception($"UnpackRGBA pixel[{pixelIdx}].{channel}: expected={expected[i]}, actual={actual[i]}. " +
                        $"Packed value=0x{((uint)packed[pixelIdx]):X8}");
                }
            }
        });

        [TestMethod]
        public async Task IntBuffer_PackUnpackRoundtrip() => await RunTest(async accelerator =>
        {
            // Channel values to pack: 3 pixels × 4 channels
            var channels = new int[]
            {
                255, 128, 64, 200,
                0, 255, 0, 255,
                100, 150, 200, 250,
            };

            using var channelBuf = accelerator.Allocate1D(channels);
            using var packedBuf = accelerator.Allocate1D<int>(3);
            using var unpackedBuf = accelerator.Allocate1D<int>(12);

            var packKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<int, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>>(PackRGBAKernel);
            var unpackKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<int, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>>(UnpackRGBAKernel);

            // Pack channels → packed ints → unpack back to channels
            packKernel(3, channelBuf.View, packedBuf.View);
            unpackKernel(3, packedBuf.View, unpackedBuf.View);
            await accelerator.SynchronizeAsync();

            var actual = await unpackedBuf.CopyToHostAsync<int>();

            for (int i = 0; i < channels.Length; i++)
            {
                if (actual[i] != channels[i])
                {
                    int pixelIdx = i / 4;
                    string channel = (i % 4) switch { 0 => "R", 1 => "G", 2 => "B", _ => "A" };
                    throw new Exception($"PackUnpackRoundtrip pixel[{pixelIdx}].{channel}: expected={channels[i]}, actual={actual[i]}");
                }
            }
        });
    }
}
