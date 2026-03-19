using ILGPU;
using ILGPU.Runtime;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.Demo.Shared.UnitTests
{
    // Part 12: Loop break + bool PHI codegen regression tests.
    //
    // These tests reproduce a WGSL codegen bug where a bool variable set to true
    // inside a conditional break (nested if-else chain within a for loop) produces
    // the wrong value after the loop exits normally. The PHI merge at the loop exit
    // always takes the break-path value instead of the normal-exit value.
    //
    // The bug was discovered via PadKernel in SpawnDev.ILGPU.ML — it caused all
    // Pad operations to produce zeros on WebGPU, breaking the style transfer pipeline.
    //
    // Key pattern that triggers the bug (on WebGPU/WebGL):
    //   bool flag = false;
    //   for (int d = 0; d < bufferReadBound; d++) {
    //     if (outerCondition) {
    //       if (innerCondition) { flag = true; break; }
    //       else { /* other work */ }
    //     }
    //     // more work using d
    //   }
    //   output = flag ? A : B;   // flag is ALWAYS true on WebGPU (should be false when no break)
    //
    // Simple break patterns (without the nested if-else) work correctly.
    public abstract partial class BackendTestBase
    {
        private static void AssertCloseF(float[] expected, float[] actual, float tol, string label)
        {
            if (expected.Length != actual.Length)
                throw new Exception($"{label}: length mismatch {expected.Length} vs {actual.Length}");
            for (int i = 0; i < expected.Length; i++)
            {
                float err = MathF.Abs(expected[i] - actual[i]);
                if (err > tol)
                    throw new Exception($"{label}: mismatch at [{i}] expected={expected[i]} actual={actual[i]} err={err}");
            }
        }

        // ═══════════════════════════════════════════════════════════
        //  Test: Simple break with bool — PASSES (baseline)
        //  Pattern: for (...) { if (cond) { flag = true; break; } }
        // ═══════════════════════════════════════════════════════════

        private static void SimpleBreakBoolKernel(Index1D idx,
            ArrayView1D<float, Stride1D.Dense> output,
            ArrayView1D<int, Stride1D.Dense> p)
        {
            int count = p[0];
            float sum = 0f;
            bool broken = false;
            for (int i = 0; i < count; i++)
            {
                int val = p[1 + i];
                if (val > 25)
                {
                    broken = true;
                    break;
                }
                sum += val;
            }
            output[idx] = broken ? -sum : sum;
        }

        [TestMethod]
        public async Task SimpleBreakBoolTest() => await RunTest(async accelerator =>
        {
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>>(SimpleBreakBoolKernel);

            // count=4, values=[10, 20, 30, 40]. Break at 30 (>25). sum=10+20=30, broken=true → -30
            var paramsData = new int[] { 4, 10, 20, 30, 40 };
            using var paramsBuf = accelerator.Allocate1D(paramsData);
            using var outBuf = accelerator.Allocate1D<float>(2);
            kernel(2, outBuf.View, paramsBuf.View);
            await accelerator.SynchronizeAsync();
            var actual = await outBuf.CopyToHostAsync<float>();
            var expected = new float[] { -30, -30 };
            AssertCloseF(expected, actual, 1e-6f, "SimpleBreakBool");
        });

        [TestMethod]
        public async Task SimpleBreakBoolNoBreakTest() => await RunTest(async accelerator =>
        {
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>>(SimpleBreakBoolKernel);

            // count=3, values=[5, 10, 15]. No break (none >25). sum=30, broken=false → 30
            var paramsData = new int[] { 3, 5, 10, 15 };
            using var paramsBuf = accelerator.Allocate1D(paramsData);
            using var outBuf = accelerator.Allocate1D<float>(2);
            kernel(2, outBuf.View, paramsBuf.View);
            await accelerator.SynchronizeAsync();
            var actual = await outBuf.CopyToHostAsync<float>();
            var expected = new float[] { 30, 30 };
            AssertCloseF(expected, actual, 1e-6f, "SimpleBreakBoolNoBreak");
        });

        // ═══════════════════════════════════════════════════════════
        //  Test: Nested if-else break with bool — FAILS on WebGPU
        //  Pattern: for (...) { if (outer) { if (inner) { flag=true; break; } else { work } } more_work }
        //  This is the exact pattern from PadKernel that triggers the bug.
        // ═══════════════════════════════════════════════════════════

        private static void NestedBreakBoolKernel(Index1D idx,
            ArrayView1D<float, Stride1D.Dense> output,
            ArrayView1D<int, Stride1D.Dense> p)
        {
            // p[0] = count (loop bound from buffer)
            // p[1] = mode (0 = break on negative, 1 = clamp, 2 = reflect)
            // p[2..2+count-1] = values
            int count = p[0];
            int mode = p[1];
            float sum = 0f;
            bool flagged = false;

            for (int i = 0; i < count; i++)
            {
                int val = p[2 + i];

                if (val < 0) // outer condition: value is negative
                {
                    if (mode == 0) // inner condition: mode selects behavior
                    {
                        flagged = true;
                        break;
                    }
                    else if (mode == 1) // clamp to 0
                    {
                        val = 0;
                    }
                    else // negate (reflect)
                    {
                        val = -val;
                    }
                }

                sum += val;
            }

            output[idx] = flagged ? -999f : sum;
        }

        /// <summary>
        /// Nested break with mode=0 (break path taken). flag should be true → output = -999.
        /// </summary>
        [TestMethod]
        public async Task NestedBreakBool_BreakTaken() => await RunTest(async accelerator =>
        {
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>>(NestedBreakBoolKernel);

            // count=4, mode=0, values=[10, 20, -5, 40]. Hits -5 with mode=0 → flag=true, break
            // sum before break = 10+20 = 30, but flag=true → output = -999
            var paramsData = new int[] { 4, 0, 10, 20, -5, 40 };
            using var paramsBuf = accelerator.Allocate1D(paramsData);
            using var outBuf = accelerator.Allocate1D<float>(2);
            kernel(2, outBuf.View, paramsBuf.View);
            await accelerator.SynchronizeAsync();
            var actual = await outBuf.CopyToHostAsync<float>();
            var expected = new float[] { -999, -999 };
            AssertCloseF(expected, actual, 1e-6f, "NestedBreakBool_BreakTaken");
        });

        /// <summary>
        /// Nested break with mode=0 but NO negative values → loop exits normally.
        /// flag should be false → output = sum = 10+20+30 = 60.
        /// THIS IS THE BUG: on WebGPU, flag is incorrectly true → output = -999.
        /// </summary>
        [TestMethod]
        public async Task NestedBreakBool_NormalExit() => await RunTest(async accelerator =>
        {
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>>(NestedBreakBoolKernel);

            // count=3, mode=0, values=[10, 20, 30]. No negatives → no break.
            // flag=false, sum=60 → output = 60
            var paramsData = new int[] { 3, 0, 10, 20, 30 };
            using var paramsBuf = accelerator.Allocate1D(paramsData);
            using var outBuf = accelerator.Allocate1D<float>(2);
            kernel(2, outBuf.View, paramsBuf.View);
            await accelerator.SynchronizeAsync();
            var actual = await outBuf.CopyToHostAsync<float>();
            var expected = new float[] { 60, 60 };
            AssertCloseF(expected, actual, 1e-6f, "NestedBreakBool_NormalExit");
        });

        /// <summary>
        /// Nested break with mode=1 (clamp, no break). Negative values get clamped to 0.
        /// flag stays false → output = 10 + 0 + 30 = 40.
        /// </summary>
        [TestMethod]
        public async Task NestedBreakBool_ClampMode() => await RunTest(async accelerator =>
        {
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>>(NestedBreakBoolKernel);

            // count=3, mode=1, values=[10, -5, 30]. -5 clamped to 0 → sum=40
            var paramsData = new int[] { 3, 1, 10, -5, 30 };
            using var paramsBuf = accelerator.Allocate1D(paramsData);
            using var outBuf = accelerator.Allocate1D<float>(2);
            kernel(2, outBuf.View, paramsBuf.View);
            await accelerator.SynchronizeAsync();
            var actual = await outBuf.CopyToHostAsync<float>();
            var expected = new float[] { 40, 40 };
            AssertCloseF(expected, actual, 1e-6f, "NestedBreakBool_ClampMode");
        });

        /// <summary>
        /// Nested break with mode=2 (reflect, no break). Negative values get negated.
        /// flag stays false → output = 10 + 5 + 30 = 45.
        /// </summary>
        [TestMethod]
        public async Task NestedBreakBool_ReflectMode() => await RunTest(async accelerator =>
        {
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<int, Stride1D.Dense>>(NestedBreakBoolKernel);

            // count=3, mode=2, values=[10, -5, 30]. -5 reflected to 5 → sum=45
            var paramsData = new int[] { 3, 2, 10, -5, 30 };
            using var paramsBuf = accelerator.Allocate1D(paramsData);
            using var outBuf = accelerator.Allocate1D<float>(2);
            kernel(2, outBuf.View, paramsBuf.View);
            await accelerator.SynchronizeAsync();
            var actual = await outBuf.CopyToHostAsync<float>();
            var expected = new float[] { 45, 45 };
            AssertCloseF(expected, actual, 1e-6f, "NestedBreakBool_ReflectMode");
        });

        // NestedBreakBool_DumpWGSL removed — diagnostic test that intentionally throws.
        // The WGSL dump was used to diagnose bug #3/#4 which are now fixed.

        // ═══════════════════════════════════════════════════════════
        //  Test: Sequential loops — loop 1 computes a value, loop 2 uses it.
        //  This is the pattern used by LayerNorm and RMSNorm.
        //  Isolates whether the GLSL codegen handles passing values
        //  between sequential for-loops in the same kernel.
        // ═══════════════════════════════════════════════════════════

        private static void SequentialLoopsKernel(Index1D row,
            ArrayView1D<float, Stride1D.Dense> input,
            ArrayView1D<float, Stride1D.Dense> output,
            int C)
        {
            int offset = row * C;
            // Loop 1: compute mean
            float sum = 0f;
            for (int i = 0; i < C; i++)
                sum += input[offset + i];
            float mean = sum / C;

            // Loop 2: compute variance using mean from loop 1
            float varSum = 0f;
            for (int i = 0; i < C; i++)
            {
                float d = input[offset + i] - mean;
                varSum += d * d;
            }
            float invStd = 1f / MathF.Sqrt(varSum / C + 1e-6f);

            // Loop 3: normalize
            for (int i = 0; i < C; i++)
                output[offset + i] = (input[offset + i] - mean) * invStd;
        }

        [TestMethod]
        public async Task SequentialLoops_SmallTest() => await RunTest(async accelerator =>
        {
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                int>(SequentialLoopsKernel);

            int rows = 2, C = 4;
            var input = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };

            // CPU reference
            var expected = new float[rows * C];
            for (int r = 0; r < rows; r++)
            {
                float s = 0;
                for (int i = 0; i < C; i++) s += input[r * C + i];
                float m = s / C;
                float vs = 0;
                for (int i = 0; i < C; i++) { float d = input[r * C + i] - m; vs += d * d; }
                float inv = 1f / MathF.Sqrt(vs / C + 1e-6f);
                for (int i = 0; i < C; i++) expected[r * C + i] = (input[r * C + i] - m) * inv;
            }

            using var inBuf = accelerator.Allocate1D(input);
            using var outBuf = accelerator.Allocate1D<float>(rows * C);
            kernel(rows, inBuf.View, outBuf.View, C);
            await accelerator.SynchronizeAsync();
            var actual = await outBuf.CopyToHostAsync<float>();
            AssertCloseF(expected, actual, 1e-3f, "SequentialLoops_Small");
        });

        // SequentialLoops_DumpGLSL removed — diagnostic test used to debug WebGL TF limitation.

        [TestMethod]
        public async Task SequentialLoops_LargerTest() => await RunTest(async accelerator =>
        {
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                int>(SequentialLoopsKernel);

            int rows = 4, C = 64;
            var rng = new Random(42);
            var input = new float[rows * C];
            for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

            var expected = new float[rows * C];
            for (int r = 0; r < rows; r++)
            {
                float s = 0;
                for (int i = 0; i < C; i++) s += input[r * C + i];
                float m = s / C;
                float vs = 0;
                for (int i = 0; i < C; i++) { float d = input[r * C + i] - m; vs += d * d; }
                float inv = 1f / MathF.Sqrt(vs / C + 1e-6f);
                for (int i = 0; i < C; i++) expected[r * C + i] = (input[r * C + i] - m) * inv;
            }

            using var inBuf = accelerator.Allocate1D(input);
            using var outBuf = accelerator.Allocate1D<float>(rows * C);
            kernel(rows, inBuf.View, outBuf.View, C);
            await accelerator.SynchronizeAsync();
            var actual = await outBuf.CopyToHostAsync<float>();
            AssertCloseF(expected, actual, 1e-2f, "SequentialLoops_Larger");
        });
    }
}
