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

        // ═══════════════════════════════════════════════════════════
        //  Bug 4: Double accumulation (Dekker f64 on WebGPU/WebGL)
        // ═══════════════════════════════════════════════════════════

        /// <summary>
        /// Kernel that accumulates float products using double precision.
        /// Tests that (double)f32 * (double)f32 is NOT optimized away to f32*f32.
        /// </summary>
        private static void DoubleAccumKernel(Index1D idx,
            ArrayView1D<float, Stride1D.Dense> a,
            ArrayView1D<float, Stride1D.Dense> b,
            ArrayView1D<float, Stride1D.Dense> output,
            int K)
        {
            double sum = 0.0;
            for (int k = 0; k < K; k++)
                sum += (double)a[idx * K + k] * (double)b[k];
            output[idx] = (float)sum;
        }

        [TestMethod]
        public async Task DoubleAccum_MatchesCpuDouble() => await RunTest(async accelerator =>
        {
            // Create data where float accumulation loses precision but double doesn't
            int M = 4, K = 128;
            var rng = new Random(42);
            var aData = new float[M * K];
            var bData = new float[K];
            for (int i = 0; i < M * K; i++) aData[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < K; i++) bData[i] = (float)(rng.NextDouble() * 2 - 1);

            // CPU double reference
            var expected = new float[M];
            for (int m = 0; m < M; m++)
            {
                double sum = 0.0;
                for (int k = 0; k < K; k++)
                    sum += (double)aData[m * K + k] * (double)bData[k];
                expected[m] = (float)sum;
            }

            using var aBuf = accelerator.Allocate1D(aData);
            using var bBuf = accelerator.Allocate1D(bData);
            using var outBuf = accelerator.Allocate1D<float>(M);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                int>(DoubleAccumKernel);
            kernel(M, aBuf.View, bBuf.View, outBuf.View, K);
            await accelerator.SynchronizeAsync();

            var actual = await outBuf.CopyToHostAsync<float>();

            for (int i = 0; i < M; i++)
            {
                float err = MathF.Abs(actual[i] - expected[i]);
                if (err > 1e-6f)
                    throw new Exception($"DoubleAccum[{i}]: expected={expected[i]}, actual={actual[i]}, err={err}. " +
                        $"If err >> 1e-6, double accumulation was optimized to float.");
            }
        });

        // ═══════════════════════════════════════════════════════════
        //  Bug 4b: Double accumulation in triple-nested loop with continue
        //  (mimics Conv2D pattern that strips double on WebGL)
        // ═══════════════════════════════════════════════════════════

        private static void DoubleAccumTripleLoopKernel(Index1D idx,
            ArrayView1D<float, Stride1D.Dense> input,
            ArrayView1D<float, Stride1D.Dense> weight,
            ArrayView1D<float, Stride1D.Dense> output,
            ArrayView1D<int, Stride1D.Dense> p)
        {
            int inC = p[0]; int inH = p[1]; int inW = p[2]; int kH = p[3]; int kW = p[4];
            double sum = 0.0;
            for (int ic = 0; ic < inC; ic++)
            {
                for (int ky = 0; ky < kH; ky++)
                {
                    int iy = idx + ky;
                    if (iy < 0 || iy >= inH) continue;
                    for (int kx = 0; kx < kW; kx++)
                    {
                        int ix = kx;
                        if (ix < 0 || ix >= inW) continue;
                        sum += (double)input[ic * inH * inW + iy * inW + ix] * (double)weight[ic * kH * kW + ky * kW + kx];
                    }
                }
            }
            output[idx] = (float)sum;
        }

        [TestMethod]
        public async Task DoubleAccum_TripleLoop_MatchesCpuDouble() => await RunTest(async accelerator =>
        {
            int inC = 4, inH = 8, inW = 8, kH = 3, kW = 3;
            var rng = new Random(99);
            var inputData = new float[inC * inH * inW];
            var weightData = new float[inC * kH * kW];
            for (int i = 0; i < inputData.Length; i++) inputData[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < weightData.Length; i++) weightData[i] = (float)(rng.NextDouble() * 2 - 1);

            int outH = inH - kH + 1;
            var expected = new float[outH];
            for (int y = 0; y < outH; y++)
            {
                double sum = 0.0;
                for (int ic = 0; ic < inC; ic++)
                    for (int ky = 0; ky < kH; ky++)
                    {
                        int iy = y + ky;
                        if (iy < 0 || iy >= inH) continue;
                        for (int kx = 0; kx < kW; kx++)
                        {
                            int ix = kx;
                            if (ix < 0 || ix >= inW) continue;
                            sum += (double)inputData[ic * inH * inW + iy * inW + ix] * (double)weightData[ic * kH * kW + ky * kW + kx];
                        }
                    }
                expected[y] = (float)sum;
            }

            using var inBuf = accelerator.Allocate1D(inputData);
            using var wBuf = accelerator.Allocate1D(weightData);
            using var outBuf = accelerator.Allocate1D<float>(outH);
            using var paramsBuf = accelerator.Allocate1D(new int[] { inC, inH, inW, kH, kW, 0, 0, 0 });

            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>>(DoubleAccumTripleLoopKernel);
            kernel(outH, inBuf.View, wBuf.View, outBuf.View, paramsBuf.View);
            await accelerator.SynchronizeAsync();

            var actual = await outBuf.CopyToHostAsync<float>();

            for (int i = 0; i < outH; i++)
            {
                float err = MathF.Abs(actual[i] - expected[i]);
                if (err > 1e-5f)
                    throw new Exception($"DoubleAccumTripleLoop[{i}]: expected={expected[i]}, actual={actual[i]}, err={err}");
            }
        });

        [TestMethod]
        public async Task DoubleAccum_MADThreshold() => await RunTest(async accelerator =>
        {
            int[] inCValues = { 4, 6, 8, 10, 12, 16 };
            int inH = 8, inW = 8, kH = 3, kW = 3;
            var rng = new Random(200);

            int maxInC = inCValues.Max();
            var inputData = new float[maxInC * inH * inW];
            var weightData = new float[maxInC * kH * kW];
            for (int i = 0; i < inputData.Length; i++) inputData[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < weightData.Length; i++) weightData[i] = (float)(rng.NextDouble() * 2 - 1);

            using var inBuf = accelerator.Allocate1D(inputData);
            using var wBuf = accelerator.Allocate1D(weightData);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>>(DoubleAccumTripleLoopKernel);

            var sb = new System.Text.StringBuilder();
            sb.AppendLine("[MADThreshold] Results:");

            foreach (int inC in inCValues)
            {
                int mads = inC * kH * kW;
                int outH = inH - kH + 1;

                var expected = new float[outH];
                for (int y = 0; y < outH; y++)
                {
                    double sum = 0.0;
                    for (int ic = 0; ic < inC; ic++)
                        for (int ky = 0; ky < kH; ky++)
                        {
                            int iy = y + ky;
                            if (iy < 0 || iy >= inH) continue;
                            for (int kx = 0; kx < kW; kx++)
                            {
                                int ix = kx;
                                if (ix < 0 || ix >= inW) continue;
                                sum += (double)inputData[ic * inH * inW + iy * inW + ix]
                                     * (double)weightData[ic * kH * kW + ky * kW + kx];
                            }
                        }
                    expected[y] = (float)sum;
                }

                using var outBuf = accelerator.Allocate1D<float>(outH);
                using var paramsBuf = accelerator.Allocate1D(new int[] { inC, inH, inW, kH, kW, 0, 0, 0 });
                kernel(outH, inBuf.View, wBuf.View, outBuf.View, paramsBuf.View);
                await accelerator.SynchronizeAsync();

                var actual = await outBuf.CopyToHostAsync<float>();

                float maxErr = 0f;
                for (int i = 0; i < outH; i++)
                {
                    float err = MathF.Abs(actual[i] - expected[i]);
                    if (err > maxErr) maxErr = err;
                }

                sb.AppendLine($"  inC={inC,3} MADs={mads,4} maxErr={maxErr:E3}");
            }

            // Results available in sb if needed for debugging
        });

        // Exact copy of ML Conv2DImpl to test in standalone infrastructure
        private static void MLConv2DExactKernel(Index1D idx,
            ArrayView1D<float, Stride1D.Dense> input,
            ArrayView1D<float, Stride1D.Dense> weight,
            ArrayView1D<float, Stride1D.Dense> bias,
            ArrayView1D<float, Stride1D.Dense> output,
            ArrayView1D<int, Stride1D.Dense> p)
        {
            int inC = p[0]; int inH = p[1]; int inW = p[2];
            int outC = p[3]; int kH = p[4]; int kW = p[5];
            int stride = p[6]; int padding = p[7];
            int outH = (inH + 2 * padding - kH) / stride + 1;
            int outW = (inW + 2 * padding - kW) / stride + 1;
            int ox = idx % outW;
            int rem = idx / outW;
            int oy = rem % outH;
            int oc = rem / outH;
            // Always read bias unconditionally — ANGLE's HLSL optimizer changes FP
            // evaluation of the accumulation loop when a conditional branch precedes
            // it, degrading Dekker f64 emulation to float precision (~0.009 error).
            // Callers must always provide a valid bias buffer (zero-filled if no bias).
            double sum = (double)bias[oc];
            for (int ic = 0; ic < inC; ic++)
            {
                int icBase = ic * inH * inW;
                int wcBase = oc * inC * kH * kW + ic * kH * kW;
                for (int ky = 0; ky < kH; ky++)
                {
                    int iy = oy * stride + ky - padding;
                    if (iy < 0 || iy >= inH) continue;
                    for (int kx = 0; kx < kW; kx++)
                    {
                        int ix = ox * stride + kx - padding;
                        if (ix < 0 || ix >= inW) continue;
                        sum += (double)input[icBase + iy * inW + ix] * (double)weight[wcBase + ky * kW + kx];
                    }
                }
            }
            output[idx] = (float)sum;
        }

        [TestMethod]
        public async Task DoubleAccum_MLConv2DExact() => await RunTest(async accelerator =>
        {
            int inC = 8, inH = 10, inW = 10, outC = 4, kH = 3, kW = 3, stride = 1, padding = 0;
            int outH = (inH + 2 * padding - kH) / stride + 1;
            int outW = (inW + 2 * padding - kW) / stride + 1;
            int totalOut = outC * outH * outW;

            var rng110 = new Random(110);
            var inputData = new float[inC * inH * inW];
            for (int i = 0; i < inputData.Length; i++) inputData[i] = (float)(rng110.NextDouble() * 2 - 1) * 0.5f;

            var rng111 = new Random(111);
            var weightData = new float[outC * inC * kH * kW];
            for (int i = 0; i < weightData.Length; i++) weightData[i] = (float)(rng111.NextDouble() * 2 - 1) * 0.1f;

            var rng112 = new Random(112);
            var biasData = new float[outC];
            for (int i = 0; i < biasData.Length; i++) biasData[i] = (float)(rng112.NextDouble() * 2 - 1) * 0.01f;

            var expected = new float[totalOut];
            for (int oc_ = 0; oc_ < outC; oc_++)
                for (int oy_ = 0; oy_ < outH; oy_++)
                    for (int ox_ = 0; ox_ < outW; ox_++)
                    {
                        float sum = biasData.Length > 0 ? biasData[oc_] : 0f;
                        for (int ic = 0; ic < inC; ic++)
                            for (int ky = 0; ky < kH; ky++)
                                for (int kx = 0; kx < kW; kx++)
                                {
                                    int iy = oy_ * stride + ky - padding;
                                    int ix = ox_ * stride + kx - padding;
                                    if (iy >= 0 && iy < inH && ix >= 0 && ix < inW)
                                        sum += inputData[ic * inH * inW + iy * inW + ix]
                                             * weightData[oc_ * inC * kH * kW + ic * kH * kW + ky * kW + kx];
                                }
                        expected[oc_ * outH * outW + oy_ * outW + ox_] = sum;
                    }

            using var inBuf = accelerator.Allocate1D(inputData);
            using var wBuf = accelerator.Allocate1D(weightData);
            using var bBuf = accelerator.Allocate1D(biasData);
            using var outBuf = accelerator.Allocate1D<float>(totalOut);
            using var paramsBuf = accelerator.Allocate1D(new int[] { inC, inH, inW, outC, kH, kW, stride, padding });

            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>>(MLConv2DExactKernel);
            kernel(totalOut, inBuf.View, wBuf.View, bBuf.View, outBuf.View, paramsBuf.View);
            await accelerator.SynchronizeAsync();

            var actual = await outBuf.CopyToHostAsync<float>();

            float maxErr = 0f;
            int maxIdx = 0;
            for (int i = 0; i < totalOut; i++)
            {
                float err = MathF.Abs(actual[i] - expected[i]);
                if (err > maxErr) { maxErr = err; maxIdx = i; }
            }

            if (maxErr > 1e-3f)
                throw new Exception($"[MLConv2DExact] maxErr={maxErr:E3} at [{maxIdx}] expected={expected[maxIdx]:F6} actual={actual[maxIdx]:F6}");
        });

        // ═══════════════════════════════════════════════════════════
        private static void BinarySearchC_MultiOCKernel(Index1D idx,
            ArrayView1D<float, Stride1D.Dense> input,
            ArrayView1D<float, Stride1D.Dense> weight,
            ArrayView1D<float, Stride1D.Dense> output,
            ArrayView1D<int, Stride1D.Dense> p)
        {
            int inC = p[0]; int inH = p[1]; int inW = p[2];
            int outC = p[3]; int kH = p[4]; int kW = p[5];
            int outH = inH - kH + 1;
            int outW = inW - kW + 1;
            int ox = idx % outW;
            int rem = idx / outW;
            int oy = rem % outH;
            int oc = rem / outH;
            double sum = 0.0;
            for (int ic = 0; ic < inC; ic++)
            {
                int icBase = ic * inH * inW;
                int wcBase = oc * inC * kH * kW + ic * kH * kW;
                for (int ky = 0; ky < kH; ky++)
                {
                    int iy = oy + ky;
                    if (iy < 0 || iy >= inH) continue;
                    for (int kx = 0; kx < kW; kx++)
                    {
                        int ix = ox + kx;
                        if (ix < 0 || ix >= inW) continue;
                        sum += (double)input[icBase + iy * inW + ix] * (double)weight[wcBase + ky * kW + kx];
                    }
                }
            }
            output[idx] = (float)sum;
        }

        [TestMethod]
        public async Task DoubleAccum_BinarySearchC() => await RunTest(async accelerator =>
        {
            int inC = 8, inH = 10, inW = 10, outC = 4, kH = 3, kW = 3;
            int outH = inH - kH + 1;
            int outW = inW - kW + 1;
            int totalOut = outC * outH * outW;
            var rng = new Random(110);
            var inputData = new float[inC * inH * inW];
            for (int i = 0; i < inputData.Length; i++) inputData[i] = (float)(rng.NextDouble() * 2 - 1) * 0.5f;
            rng = new Random(111);
            var weightData = new float[outC * inC * kH * kW];
            for (int i = 0; i < weightData.Length; i++) weightData[i] = (float)(rng.NextDouble() * 2 - 1) * 0.1f;

            var expected = new float[totalOut];
            for (int oc_ = 0; oc_ < outC; oc_++)
                for (int oy_ = 0; oy_ < outH; oy_++)
                    for (int ox_ = 0; ox_ < outW; ox_++)
                    {
                        float sum = 0f;
                        for (int ic = 0; ic < inC; ic++)
                            for (int ky = 0; ky < kH; ky++)
                                for (int kx = 0; kx < kW; kx++)
                                {
                                    int iy = oy_ + ky;
                                    int ix = ox_ + kx;
                                    if (iy >= 0 && iy < inH && ix >= 0 && ix < inW)
                                        sum += inputData[ic * inH * inW + iy * inW + ix]
                                             * weightData[oc_ * inC * kH * kW + ic * kH * kW + ky * kW + kx];
                                }
                        expected[oc_ * outH * outW + oy_ * outW + ox_] = sum;
                    }

            using var inBuf = accelerator.Allocate1D(inputData);
            using var wBuf = accelerator.Allocate1D(weightData);
            using var outBuf = accelerator.Allocate1D<float>(totalOut);
            using var paramsBuf = accelerator.Allocate1D(new int[] { inC, inH, inW, outC, kH, kW, 0, 0 });
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>(BinarySearchC_MultiOCKernel);
            kernel(totalOut, inBuf.View, wBuf.View, outBuf.View, paramsBuf.View);
            await accelerator.SynchronizeAsync();
            var actual = await outBuf.CopyToHostAsync<float>();

            float maxErr = 0f; int maxIdx = 0;
            for (int i = 0; i < totalOut; i++) { float err = MathF.Abs(actual[i] - expected[i]); if (err > maxErr) { maxErr = err; maxIdx = i; } }
            if (maxErr > 1e-3f)
                throw new Exception($"[BinarySearchC] maxErr={maxErr:E3} at [{maxIdx}] expected={expected[maxIdx]:F6} actual={actual[maxIdx]:F6}");
        });

        // Variant C+Bias: same as C but with bias buffer (5 params like ML Conv2D)
        private static void BinarySearchCBias_Kernel(Index1D idx,
            ArrayView1D<float, Stride1D.Dense> input,
            ArrayView1D<float, Stride1D.Dense> weight,
            ArrayView1D<float, Stride1D.Dense> bias,
            ArrayView1D<float, Stride1D.Dense> output,
            ArrayView1D<int, Stride1D.Dense> p)
        {
            int inC = p[0]; int inH = p[1]; int inW = p[2];
            int outC = p[3]; int kH = p[4]; int kW = p[5];
            int outH = inH - kH + 1;
            int outW = inW - kW + 1;
            int ox = idx % outW;
            int rem = idx / outW;
            int oy = rem % outH;
            int oc = rem / outH;
            // Always read bias unconditionally — same ANGLE workaround as MLConv2DExactKernel
            double sum = (double)bias[oc];
            for (int ic = 0; ic < inC; ic++)
            {
                int icBase = ic * inH * inW;
                int wcBase = oc * inC * kH * kW + ic * kH * kW;
                for (int ky = 0; ky < kH; ky++)
                {
                    int iy = oy + ky;
                    if (iy < 0 || iy >= inH) continue;
                    for (int kx = 0; kx < kW; kx++)
                    {
                        int ix = ox + kx;
                        if (ix < 0 || ix >= inW) continue;
                        sum += (double)input[icBase + iy * inW + ix] * (double)weight[wcBase + ky * kW + kx];
                    }
                }
            }
            output[idx] = (float)sum;
        }

        [TestMethod]
        public async Task DoubleAccum_BinarySearchCBias() => await RunTest(async accelerator =>
        {
            int inC = 8, inH = 10, inW = 10, outC = 4, kH = 3, kW = 3;
            int outH = inH - kH + 1;
            int outW = inW - kW + 1;
            int totalOut = outC * outH * outW;
            var rng = new Random(110);
            var inputData = new float[inC * inH * inW];
            for (int i = 0; i < inputData.Length; i++) inputData[i] = (float)(rng.NextDouble() * 2 - 1) * 0.5f;
            rng = new Random(111);
            var weightData = new float[outC * inC * kH * kW];
            for (int i = 0; i < weightData.Length; i++) weightData[i] = (float)(rng.NextDouble() * 2 - 1) * 0.1f;
            rng = new Random(112);
            var biasData = new float[outC];
            for (int i = 0; i < biasData.Length; i++) biasData[i] = (float)(rng.NextDouble() * 2 - 1) * 0.01f;

            var expected = new float[totalOut];
            for (int oc_ = 0; oc_ < outC; oc_++)
                for (int oy_ = 0; oy_ < outH; oy_++)
                    for (int ox_ = 0; ox_ < outW; ox_++)
                    {
                        float sum = biasData[oc_];
                        for (int ic = 0; ic < inC; ic++)
                            for (int ky = 0; ky < kH; ky++)
                                for (int kx = 0; kx < kW; kx++)
                                {
                                    int iy = oy_ + ky;
                                    int ix = ox_ + kx;
                                    if (iy >= 0 && iy < inH && ix >= 0 && ix < inW)
                                        sum += inputData[ic * inH * inW + iy * inW + ix]
                                             * weightData[oc_ * inC * kH * kW + ic * kH * kW + ky * kW + kx];
                                }
                        expected[oc_ * outH * outW + oy_ * outW + ox_] = sum;
                    }

            using var inBuf = accelerator.Allocate1D(inputData);
            using var wBuf = accelerator.Allocate1D(weightData);
            using var bBuf = accelerator.Allocate1D(biasData);
            using var outBuf = accelerator.Allocate1D<float>(totalOut);
            using var paramsBuf = accelerator.Allocate1D(new int[] { inC, inH, inW, outC, kH, kW, 0, 0 });
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>>(BinarySearchCBias_Kernel);
            kernel(totalOut, inBuf.View, wBuf.View, bBuf.View, outBuf.View, paramsBuf.View);
            await accelerator.SynchronizeAsync();
            var actual = await outBuf.CopyToHostAsync<float>();

            float maxErr = 0f; int maxIdx = 0;
            for (int i = 0; i < totalOut; i++) { float err = MathF.Abs(actual[i] - expected[i]); if (err > maxErr) { maxErr = err; maxIdx = i; } }
            if (maxErr > 1e-3f)
                throw new Exception($"[BinarySearchCBias] maxErr={maxErr:E3} at [{maxIdx}] expected={expected[maxIdx]:F6} actual={actual[maxIdx]:F6}");
        });

        // Test E: 5 buffers, NO bias.Length branch — always reads bias[oc]
        private static void BinarySearchE_NoBranchKernel(Index1D idx,
            ArrayView1D<float, Stride1D.Dense> input,
            ArrayView1D<float, Stride1D.Dense> weight,
            ArrayView1D<float, Stride1D.Dense> bias,
            ArrayView1D<float, Stride1D.Dense> output,
            ArrayView1D<int, Stride1D.Dense> p)
        {
            int inC = p[0]; int inH = p[1]; int inW = p[2];
            int outC = p[3]; int kH = p[4]; int kW = p[5];
            int outH = inH - kH + 1;
            int outW = inW - kW + 1;
            int ox = idx % outW;
            int rem = idx / outW;
            int oy = rem % outH;
            int oc = rem / outH;
            double sum = (double)bias[oc];  // ALWAYS read bias, NO branch
            for (int ic = 0; ic < inC; ic++)
            {
                int icBase = ic * inH * inW;
                int wcBase = oc * inC * kH * kW + ic * kH * kW;
                for (int ky = 0; ky < kH; ky++)
                {
                    int iy = oy + ky;
                    if (iy < 0 || iy >= inH) continue;
                    for (int kx = 0; kx < kW; kx++)
                    {
                        int ix = ox + kx;
                        if (ix < 0 || ix >= inW) continue;
                        sum += (double)input[icBase + iy * inW + ix] * (double)weight[wcBase + ky * kW + kx];
                    }
                }
            }
            output[idx] = (float)sum;
        }

        [TestMethod]
        public async Task DoubleAccum_BinarySearchE() => await RunTest(async accelerator =>
        {
            int inC = 8, inH = 10, inW = 10, outC = 4, kH = 3, kW = 3;
            int outH = inH - kH + 1;
            int outW = inW - kW + 1;
            int totalOut = outC * outH * outW;
            var rng = new Random(110);
            var inputData = new float[inC * inH * inW];
            for (int i = 0; i < inputData.Length; i++) inputData[i] = (float)(rng.NextDouble() * 2 - 1) * 0.5f;
            rng = new Random(111);
            var weightData = new float[outC * inC * kH * kW];
            for (int i = 0; i < weightData.Length; i++) weightData[i] = (float)(rng.NextDouble() * 2 - 1) * 0.1f;
            rng = new Random(112);
            var biasData = new float[outC];
            for (int i = 0; i < biasData.Length; i++) biasData[i] = (float)(rng.NextDouble() * 2 - 1) * 0.01f;

            var expected = new float[totalOut];
            for (int oc_ = 0; oc_ < outC; oc_++)
                for (int oy_ = 0; oy_ < outH; oy_++)
                    for (int ox_ = 0; ox_ < outW; ox_++)
                    {
                        float sum = biasData[oc_];
                        for (int ic = 0; ic < inC; ic++)
                            for (int ky = 0; ky < kH; ky++)
                                for (int kx = 0; kx < kW; kx++)
                                {
                                    int iy = oy_ + ky;
                                    int ix = ox_ + kx;
                                    if (iy >= 0 && iy < inH && ix >= 0 && ix < inW)
                                        sum += inputData[ic * inH * inW + iy * inW + ix]
                                             * weightData[oc_ * inC * kH * kW + ic * kH * kW + ky * kW + kx];
                                }
                        expected[oc_ * outH * outW + oy_ * outW + ox_] = sum;
                    }

            using var inBuf = accelerator.Allocate1D(inputData);
            using var wBuf = accelerator.Allocate1D(weightData);
            using var bBuf = accelerator.Allocate1D(biasData);
            using var outBuf = accelerator.Allocate1D<float>(totalOut);
            using var paramsBuf = accelerator.Allocate1D(new int[] { inC, inH, inW, outC, kH, kW, 0, 0 });
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>>(BinarySearchE_NoBranchKernel);
            kernel(totalOut, inBuf.View, wBuf.View, bBuf.View, outBuf.View, paramsBuf.View);
            await accelerator.SynchronizeAsync();
            var actual = await outBuf.CopyToHostAsync<float>();

            float maxErr = 0f; int maxIdx = 0;
            for (int i = 0; i < totalOut; i++) { float err = MathF.Abs(actual[i] - expected[i]); if (err > maxErr) { maxErr = err; maxIdx = i; } }
            if (maxErr > 1e-3f)
                throw new Exception($"[BinarySearchE] maxErr={maxErr:E3} at [{maxIdx}] expected={expected[maxIdx]:F6} actual={actual[maxIdx]:F6}");
        });

        //  Bug 3: GPU→GPU buffer copy (peer-to-peer)
        // ═══════════════════════════════════════════════════════════

        [TestMethod]
        public async Task BufferCopy_GPUToGPU_PreservesData() => await RunTest(async accelerator =>
        {
            var data = new float[] { 1.5f, -2.7f, 3.14f, 0f, 100f, -0.001f };

            using var srcBuf = accelerator.Allocate1D(data);
            using var dstBuf = accelerator.Allocate1D<float>(data.Length);

            // GPU→GPU copy via CopyFrom (this threw NotSupportedException on WebGL before fix)
            dstBuf.View.SubView(0, data.Length).CopyFrom(srcBuf.View.SubView(0, data.Length));
            await accelerator.SynchronizeAsync();

            var actual = await dstBuf.CopyToHostAsync<float>();

            for (int i = 0; i < data.Length; i++)
            {
                if (MathF.Abs(actual[i] - data[i]) > 1e-6f)
                    throw new Exception($"BufferCopy[{i}]: expected={data[i]}, actual={actual[i]}");
            }
        });
    }
}
