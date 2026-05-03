using ILGPU;
using ILGPU.Runtime;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.Demo.Shared.UnitTests
{
    // Part 3: Long/Double precision tests, comparison/logic tests, GPU patterns, and misc tests
    public abstract partial class BackendTestBase
    {
        #region Long Integer Tests

        [TestMethod]
        public async Task LongIntegerTest() => await RunEmulatedTest(async accelerator =>
        {
            var data = new long[] { 1L, 100L, -50L, long.MaxValue / 2, 0L, -1L };
            int len = data.Length;
            using var buf = accelerator.Allocate1D(data);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<long>>(LongIntegerKernel);
            kernel((Index1D)len, buf.View);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<long>();
            for (int i = 0; i < len; i++) { long expected = data[i] * 2 + 1; if (result[i] != expected) throw new Exception($"Long integer failed at {i}. Expected {expected}, got {result[i]}"); }
        });

        [TestMethod]
        public async Task LongArithmeticTest() => await RunEmulatedTest(async accelerator =>
        {
            var a = new long[] { 100L, -50L, 1000000L, 0L }; var b = new long[] { 50L, 50L, -500000L, 0L };
            int len = a.Length;
            using var bufA = accelerator.Allocate1D(a); using var bufB = accelerator.Allocate1D(b); using var bufOut = accelerator.Allocate1D<long>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<long>, ArrayView<long>, ArrayView<long>>(LongArithmeticKernel);
            kernel((Index1D)len, bufA.View, bufB.View, bufOut.View);
            await accelerator.SynchronizeAsync();
            var result = await bufOut.CopyToHostAsync<long>();
            for (int i = 0; i < len; i++) { long expected = (a[i] + b[i]) - (a[i] * 2); if (result[i] != expected) throw new Exception($"Long arithmetic failed at {i}"); }
        });

        [TestMethod]
        public async Task LongBitwiseTest() => await RunEmulatedTest(async accelerator =>
        {
            var input = new long[] { 0x123456789ABCDEF0L, -1L, 0L, long.MaxValue };
            int len = input.Length;
            using var bufIn = accelerator.Allocate1D(input); using var bufOut = accelerator.Allocate1D<long>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<long>, ArrayView<long>>(LongBitwiseKernel);
            kernel((Index1D)len, bufIn.View, bufOut.View);
            await accelerator.SynchronizeAsync();
            var result = await bufOut.CopyToHostAsync<long>();
            for (int i = 0; i < len; i++) { long val = input[i]; long mask = 0x0F0F0F0F0F0F0F0FL; long expected = (val & mask) | ((val >> 4) & mask); if (result[i] != expected) throw new Exception($"Long bitwise failed at {i}"); }
        });

        [TestMethod]
        public async Task LongComparisonTest() => await RunEmulatedTest(async accelerator =>
        {
            int len = 64;
            var a = new long[len]; var b = new long[len];
            for (int i = 0; i < len; i++) { a[i] = (i * 7 - 100); b[i] = (i * 3 - 50); }
            using var bufA = accelerator.Allocate1D(a); using var bufB = accelerator.Allocate1D(b); using var bufOut = accelerator.Allocate1D<long>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<long>, ArrayView<long>, ArrayView<long>>(LongComparisonKernel);
            kernel((Index1D)len, bufA.View, bufB.View, bufOut.View);
            await accelerator.SynchronizeAsync();
            var result = await bufOut.CopyToHostAsync<long>();
            for (int i = 0; i < len; i++) { long expected = Math.Max(a[i], b[i]); if (result[i] != expected) throw new Exception($"Long comparison failed at {i}. Expected {expected}, got {result[i]}"); }
        });

        [TestMethod]
        public async Task LongEdgeCasesTest() => await RunEmulatedTest(async accelerator =>
        {
            var data = new long[] { long.MaxValue, long.MinValue, 0L, -1L, 1L, long.MaxValue / 2, long.MinValue / 2, 42L };
            int len = data.Length;
            using var buf = accelerator.Allocate1D(data);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<long>>(LongEdgeCasesKernel);
            kernel((Index1D)len, buf.View);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<long>();
            for (int i = 0; i < len; i++) if (result[i] != data[i]) throw new Exception($"Long edge case failed at {i}");
        });

        [TestMethod]
        public async Task LongMultiBufferTest() => await RunEmulatedTest(async accelerator =>
        {
            var input = new long[] { 10L, 20L, 30L, 40L };
            int len = input.Length;
            using var bufIn = accelerator.Allocate1D(input); using var bufOut = accelerator.Allocate1D<long>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<long>, ArrayView<long>>(LongMultiBufferKernel);
            kernel((Index1D)len, bufIn.View, bufOut.View);
            await accelerator.SynchronizeAsync();
            var result = await bufOut.CopyToHostAsync<long>();
            for (int i = 0; i < len; i++) { long expected = input[i] * 3 + 7; if (result[i] != expected) throw new Exception($"Long multi-buffer failed at {i}"); }
        });

        [TestMethod]
        public async Task LongNegationTest() => await RunEmulatedTest(async accelerator =>
        {
            var data = new long[] { 1L, -1L, 100L, -100L, 0L, long.MaxValue, long.MinValue + 1, 42L };
            int len = data.Length;
            using var buf = accelerator.Allocate1D(data);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<long>>(LongNegationKernel);
            kernel((Index1D)len, buf.View);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<long>();
            for (int i = 0; i < len; i++) { long expected = -data[i]; if (result[i] != expected) throw new Exception($"Long negation failed at {i}"); }
        });

        [TestMethod]
        public async Task LongShiftTest() => await RunEmulatedTest(async accelerator =>
        {
            var input = new long[] { 1L, 0xFFL, 0x1000L, -1L };
            int len = input.Length;
            using var bufIn = accelerator.Allocate1D(input); using var bufOut = accelerator.Allocate1D<long>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<long>, ArrayView<long>>(LongShiftKernel);
            kernel((Index1D)len, bufIn.View, bufOut.View);
            await accelerator.SynchronizeAsync();
            var result = await bufOut.CopyToHostAsync<long>();
            for (int i = 0; i < len; i++) { long expected = (input[i] << 8) >> 4; if (result[i] != expected) throw new Exception($"Long shift failed at {i}"); }
        });

        [TestMethod]
        public async Task LongSignedCompareTest() => await RunEmulatedTest(async accelerator =>
        {
            var a = new long[] { -10L, 10L, 0L, long.MinValue }; var b = new long[] { 10L, -10L, 0L, long.MaxValue };
            int len = a.Length;
            using var bufA = accelerator.Allocate1D(a); using var bufB = accelerator.Allocate1D(b); using var bufOut = accelerator.Allocate1D<long>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<long>, ArrayView<long>, ArrayView<long>>(LongSignedCompareKernel);
            kernel((Index1D)len, bufA.View, bufB.View, bufOut.View);
            await accelerator.SynchronizeAsync();
            var result = await bufOut.CopyToHostAsync<long>();
            long[] expected = { 1L, 0L, 0L, 1L };
            for (int i = 0; i < len; i++) if (result[i] != expected[i]) throw new Exception($"Long signed compare failed at {i}");
        });

        [TestMethod]
        public async Task LongChainedOpsTest() => await RunEmulatedTest(async accelerator =>
        {
            var input = new long[] { 1L, 5L, 10L, 100L };
            int len = input.Length;
            using var bufIn = accelerator.Allocate1D(input); using var bufOut = accelerator.Allocate1D<long>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<long>, ArrayView<long>>(LongChainedOpsKernel);
            kernel((Index1D)len, bufIn.View, bufOut.View);
            await accelerator.SynchronizeAsync();
            var result = await bufOut.CopyToHostAsync<long>();
            for (int i = 0; i < len; i++) { long expected = ((input[i] + 10L) * 3L - 5L); if (result[i] != expected) throw new Exception($"Long chained ops failed at {i}"); }
        });

        [TestMethod]
        public async Task LongNegativeValuesTest() => await RunEmulatedTest(async accelerator =>
        {
            var data = new long[] { -1L, -100L, -1000L, -10000L };
            int len = data.Length;
            using var buf = accelerator.Allocate1D(data);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<long>>(LongNegativeValuesKernel);
            kernel((Index1D)len, buf.View);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<long>();
            for (int i = 0; i < len; i++) { long expected = data[i] * -1L + data[i]; if (result[i] != expected) throw new Exception($"Long negative values failed at {i}"); }
        });

        #endregion

        #region Double Precision Tests

        [TestMethod]
        public async Task DoublePrecisionTest() => await RunEmulatedTest(async accelerator =>
        {
            var input = new double[] { 1.0, 2.5, -3.7, 0.0, 100.5, -50.25 };
            int len = input.Length;
            using var bufIn = accelerator.Allocate1D(input); using var bufOut = accelerator.Allocate1D<double>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>, ArrayView<double>>(DoublePrecisionKernel);
            kernel((Index1D)len, bufIn.View, bufOut.View);
            await accelerator.SynchronizeAsync();
            var result = await bufOut.CopyToHostAsync<double>();
            for (int i = 0; i < len; i++) { double expected = input[i] * 2.0 + 1.0; double tol = Math.Max(Math.Abs(expected * 0.01), 1e-6); if (Math.Abs(result[i] - expected) > tol) throw new Exception($"Double precision failed at {i}"); }
        });

        [TestMethod]
        public async Task DoubleMathTest() => await RunEmulatedTest(async accelerator =>
        {
            var input = new double[] { 1.0, 4.0, 9.0, 16.0, 25.0, 100.0 };
            int len = input.Length;
            using var bufIn = accelerator.Allocate1D(input); using var bufOut = accelerator.Allocate1D<double>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>, ArrayView<double>>(DoubleMathKernel);
            kernel((Index1D)len, bufIn.View, bufOut.View);
            await accelerator.SynchronizeAsync();
            var result = await bufOut.CopyToHostAsync<double>();
            for (int i = 0; i < len; i++) { double expected = Math.Sqrt(input[i]) * 2.0; double tol = Math.Max(Math.Abs(expected * 0.01), 1e-6); if (Math.Abs(result[i] - expected) > tol) throw new Exception($"Double math failed at {i}"); }
        });

        [TestMethod]
        public async Task DoubleEdgeCasesTest() => await RunEmulatedTest(async accelerator =>
        {
            var data = new double[] { 1.0e10, -1.0e10, 0.0, -0.0, 1e-10, -1e-10 };
            int len = data.Length;
            using var buf = accelerator.Allocate1D(data);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>>(DoubleEdgeCasesKernel);
            kernel((Index1D)len, buf.View);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<double>();
            for (int i = 0; i < len; i++) { double tol = Math.Max(Math.Abs(data[i] * 0.01), 1e-6); if (Math.Abs(result[i] - data[i]) > tol) throw new Exception($"Double edge case failed at {i}. Expected {data[i]}, got {result[i]}"); }
        });

        [TestMethod]
        public async Task DoubleMultiBufferTest() => await RunEmulatedTest(async accelerator =>
        {
            var input = new double[] { 1.5, 2.5, 3.5, 4.5 };
            int len = input.Length;
            using var bufIn = accelerator.Allocate1D(input); using var bufOut = accelerator.Allocate1D<double>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>, ArrayView<double>>(DoubleMultiBufferKernel);
            kernel((Index1D)len, bufIn.View, bufOut.View);
            await accelerator.SynchronizeAsync();
            var result = await bufOut.CopyToHostAsync<double>();
            for (int i = 0; i < len; i++) { double expected = input[i] * 3.0 + 0.5; if (Math.Abs(result[i] - expected) > 0.01) throw new Exception($"Double multi-buffer failed at {i}"); }
        });

        [TestMethod]
        public async Task DoubleNegationTest() => await RunEmulatedTest(async accelerator =>
        {
            var data = new double[] { 1.5, -2.5, 0.0, 100.0, -100.0, 0.001 };
            int len = data.Length;
            using var buf = accelerator.Allocate1D(data);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>>(DoubleNegationKernel);
            kernel((Index1D)len, buf.View);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<double>();
            for (int i = 0; i < len; i++) if (Math.Abs(result[i] - (-data[i])) > Math.Max(Math.Abs(data[i] * 0.01), 1e-6)) throw new Exception($"Double negation failed at {i}. Expected {-data[i]}, got {result[i]}");
        });

        [TestMethod]
        public async Task DoubleDivisionTest() => await RunEmulatedTest(async accelerator =>
        {
            var num = new double[] { 10.0, -10.0, 100.0, 1.0 }; var div = new double[] { 3.0, 3.0, 7.0, 3.0 };
            int len = num.Length;
            using var bufN = accelerator.Allocate1D(num); using var bufD = accelerator.Allocate1D(div); using var bufOut = accelerator.Allocate1D<double>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>(DoubleDivisionKernel);
            kernel((Index1D)len, bufN.View, bufD.View, bufOut.View);
            await accelerator.SynchronizeAsync();
            var result = await bufOut.CopyToHostAsync<double>();
            for (int i = 0; i < len; i++) { double expected = num[i] / div[i]; if (Math.Abs(result[i] - expected) > 0.01) throw new Exception($"Double division failed at {i}"); }
        });

        [TestMethod]
        public async Task DoubleMinMaxTest() => await RunEmulatedTest(async accelerator =>
        {
            var a = new double[] { 1.5, -1.5, 0.0, 100.0, -100.0, 1e10 }; var b = new double[] { 2.5, -0.5, 0.0, -100.0, 100.0, 1e5 };
            int len = a.Length;
            using var bufA = accelerator.Allocate1D(a); using var bufB = accelerator.Allocate1D(b); using var bufOut = accelerator.Allocate1D<double>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>(DoubleMinMaxKernel);
            kernel((Index1D)len, bufA.View, bufB.View, bufOut.View);
            await accelerator.SynchronizeAsync();
            var result = await bufOut.CopyToHostAsync<double>();
            for (int i = 0; i < len; i++) { double max = a[i] > b[i] ? a[i] : b[i]; double min = a[i] < b[i] ? a[i] : b[i]; double expected = max - min; double tol = Math.Max(Math.Abs(expected * 0.01), 1e-6); if (Math.Abs(result[i] - expected) > tol) throw new Exception($"Double min/max failed at {i}"); }
        });

        #endregion

        #region Half Precision Tests (Float16)

        /// <summary>
        /// Verifies Half buffer allocation, GPU copy, and readback.
        /// Skips when Float16 is not supported (e.g. OpenCL without cl_khr_fp16).
        /// </summary>
        [TestMethod]
        public async Task HalfBufferRoundTripTest() => await RunTest(async accelerator =>
        {
            if (!accelerator.Capabilities.Float16)
                throw new UnsupportedTestException("Float16 not supported on this device");
            var data = new[] {
                (global::ILGPU.Half)(float)1.5f, (global::ILGPU.Half)(float)(-2.25f),
                (global::ILGPU.Half)(float)0.0f, (global::ILGPU.Half)(float)100.0f,
                (global::ILGPU.Half)(float)0.00390625f, (global::ILGPU.Half)(float)(-0.5f)
            };
            int len = data.Length;
            using var buf = accelerator.Allocate1D(data);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<global::ILGPU.Half>>(HalfPassthroughKernel);
            kernel((Index1D)len, buf.View);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<global::ILGPU.Half>();
            for (int i = 0; i < len; i++)
            {
                float expected = (float)data[i];
                if (MathF.Abs((float)result[i] - expected) > 0.01f)
                    throw new Exception($"Half buffer round-trip failed at [{i}]: expected={expected}, got={(float)result[i]}");
            }
        });

        /// <summary>
        /// Verifies Half arithmetic in a kernel (multiply, add).
        /// </summary>
        [TestMethod]
        public async Task HalfArithmeticTest() => await RunTest(async accelerator =>
        {
            if (!accelerator.Capabilities.Float16)
                throw new UnsupportedTestException("Float16 not supported on this device");
            var input = new[] {
                (global::ILGPU.Half)(float)1.0f, (global::ILGPU.Half)(float)2.5f,
                (global::ILGPU.Half)(float)(-3.0f), (global::ILGPU.Half)(float)0.0f,
                (global::ILGPU.Half)(float)10.0f
            };
            int len = input.Length;
            using var bufIn = accelerator.Allocate1D(input);
            using var bufOut = accelerator.Allocate1D<global::ILGPU.Half>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<global::ILGPU.Half>, ArrayView<global::ILGPU.Half>>(HalfArithmeticKernel);
            kernel((Index1D)len, bufIn.View, bufOut.View);
            await accelerator.SynchronizeAsync();
            var result = await bufOut.CopyToHostAsync<global::ILGPU.Half>();
            for (int i = 0; i < len; i++)
            {
                float expected = (float)input[i] * 2f + 1f;
                if (MathF.Abs((float)result[i] - expected) > 0.05f)
                    throw new Exception($"Half arithmetic failed at [{i}]: expected={expected}, got={(float)result[i]}");
            }
        });

        /// <summary>
        /// Verifies Half Min/Max in a kernel.
        /// </summary>
        [TestMethod]
        public async Task HalfMinMaxTest() => await RunTest(async accelerator =>
        {
            if (!accelerator.Capabilities.Float16)
                throw new UnsupportedTestException("Float16 not supported on this device");
            var a = new[] {
                (global::ILGPU.Half)(float)1.5f, (global::ILGPU.Half)(float)(-1.5f),
                (global::ILGPU.Half)(float)0.0f, (global::ILGPU.Half)(float)100.0f
            };
            var b = new[] {
                (global::ILGPU.Half)(float)2.5f, (global::ILGPU.Half)(float)(-0.5f),
                (global::ILGPU.Half)(float)0.0f, (global::ILGPU.Half)(float)(-100.0f)
            };
            int len = a.Length;
            using var bufA = accelerator.Allocate1D(a);
            using var bufB = accelerator.Allocate1D(b);
            using var bufOut = accelerator.Allocate1D<global::ILGPU.Half>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<global::ILGPU.Half>, ArrayView<global::ILGPU.Half>, ArrayView<global::ILGPU.Half>>(HalfMinMaxKernel);
            kernel((Index1D)len, bufA.View, bufB.View, bufOut.View);
            await accelerator.SynchronizeAsync();
            var result = await bufOut.CopyToHostAsync<global::ILGPU.Half>();
            for (int i = 0; i < len; i++)
            {
                float va = (float)a[i], vb = (float)b[i];
                float expected = (Math.Max(va, vb) - Math.Min(va, vb));
                if (MathF.Abs((float)result[i] - expected) > 0.05f)
                    throw new Exception($"Half MinMax failed at [{i}]: expected={expected}, got={(float)result[i]}");
            }
        });

        /// <summary>
        /// Verifies Half edge cases: zero, negative zero, subnormals (Epsilon), MaxValue, MinValue.
        /// NaN and Infinity are excluded because GPU passthrough of NaN bit patterns
        /// is not guaranteed across all backends.
        /// </summary>
        [TestMethod]
        public async Task HalfEdgeCasesTest() => await RunTest(async accelerator =>
        {
            if (!accelerator.Capabilities.Float16)
                throw new UnsupportedTestException("Float16 not supported on this device");
            var data = new global::ILGPU.Half[]
            {
                global::ILGPU.Half.Zero,
                (global::ILGPU.Half)(-0.0f),
                global::ILGPU.Half.Epsilon,
                global::ILGPU.Half.MaxValue,
                global::ILGPU.Half.MinValue,
                (global::ILGPU.Half)0.00006103515625f, // smallest normal Half
                (global::ILGPU.Half)(-0.00006103515625f),
                (global::ILGPU.Half)65504.0f, // max finite Half
            };
            int len = data.Length;
            using var buf = accelerator.Allocate1D(data);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<global::ILGPU.Half>>(HalfEdgeCasesKernel);
            kernel((Index1D)len, buf.View);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<global::ILGPU.Half>();
            for (int i = 0; i < len; i++)
            {
                float expected = (float)data[i];
                float actual = (float)result[i];
                if (MathF.Abs(actual - expected) > 0.01f)
                    throw new Exception($"Half edge case failed at [{i}]: expected={expected}, got={actual}");
            }
        });

        /// <summary>
        /// Verifies Half values can be combined with int values in a mixed-type kernel.
        /// </summary>
        [TestMethod]
        public async Task HalfMixedTypeTest() => await RunTest(async accelerator =>
        {
            if (!accelerator.Capabilities.Float16)
                throw new UnsupportedTestException("Float16 not supported on this device");
            var halfData = new global::ILGPU.Half[]
            {
                (global::ILGPU.Half)1.5f, (global::ILGPU.Half)2.25f,
                (global::ILGPU.Half)(-3.0f), (global::ILGPU.Half)0.0f
            };
            var intData = new int[] { 10, 20, 30, 40 };
            int len = halfData.Length;
            using var bufHalf = accelerator.Allocate1D(halfData);
            using var bufInt = accelerator.Allocate1D(intData);
            using var bufOut = accelerator.Allocate1D<float>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<global::ILGPU.Half>, ArrayView<int>, ArrayView<float>>(HalfMixedTypeKernel);
            kernel((Index1D)len, bufHalf.View, bufInt.View, bufOut.View);
            await accelerator.SynchronizeAsync();
            var result = await bufOut.CopyToHostAsync<float>();
            for (int i = 0; i < len; i++)
            {
                float expected = (float)halfData[i] + (float)intData[i];
                if (MathF.Abs(result[i] - expected) > 0.1f)
                    throw new Exception($"Half mixed type failed at [{i}]: expected={expected}, got={result[i]}");
            }
        });

        #endregion

        #region Comparison, Logic, and Misc Tests

        [TestMethod]
        public async Task MixedTypeBuffersTest() => await RunTest(async accelerator =>
        {
            int len = 16; var intData = new int[len]; var floatData = new float[len];
            for (int i = 0; i < len; i++) { intData[i] = i; floatData[i] = i * 0.5f; }
            using var bufInt = accelerator.Allocate1D(intData); using var bufFloat = accelerator.Allocate1D(floatData); using var bufResult = accelerator.Allocate1D<float>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<float>, ArrayView<float>>(MixedTypeKernel);
            kernel((Index1D)len, bufInt.View, bufFloat.View, bufResult.View);
            await accelerator.SynchronizeAsync();
            var result = await bufResult.CopyToHostAsync<float>();
            for (int i = 0; i < len; i++) { float expected = intData[i] + floatData[i]; if (MathF.Abs(result[i] - expected) > 0.001f) throw new Exception($"Mixed type failed at {i}"); }
        });

        [TestMethod]
        public async Task EmptyBufferTest() => await RunTest(async accelerator =>
        {
            var data = new int[] { 42 };
            using var buf = accelerator.Allocate1D(data);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(EmptyBufferKernel);
            kernel((Index1D)1, buf.View);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<int>();
            if (result[0] != 84) throw new Exception($"Single element buffer failed. Expected 84, got {result[0]}");
        });

        [TestMethod]
        public async Task LargeDispatchTest() => await RunTest(async accelerator =>
        {
            int len = 65536; var data = new int[len]; for (int i = 0; i < len; i++) data[i] = i;
            using var buf = accelerator.Allocate1D(data);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(LargeDispatchKernel);
            kernel((Index1D)len, buf.View);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<int>();
            int[] checkIndices = { 0, 1000, 32768, 65535 };
            foreach (int i in checkIndices) if (result[i] != i + 1) throw new Exception($"Large dispatch failed at {i}");
        });

        [TestMethod]
        public async Task ComparisonOperatorsTest() => await RunTest(async accelerator =>
        {
            var a = new int[] { 5, 5, 3, 7, 5, 5 }; var b = new int[] { 3, 7, 5, 5, 5, 6 };
            int len = 6;
            using var bufA = accelerator.Allocate1D(a); using var bufB = accelerator.Allocate1D(b); using var bufOut = accelerator.Allocate1D<int>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>(ComparisonKernel);
            kernel((Index1D)len, bufA.View, bufB.View, bufOut.View);
            await accelerator.SynchronizeAsync();
            var result = await bufOut.CopyToHostAsync<int>();
            int[] expected = { 1, 1, 0, 0, 1, 1 };
            for (int i = 0; i < len; i++) if (result[i] != expected[i]) throw new Exception($"Comparison failed at {i}");
        });

        [TestMethod]
        public async Task ShortCircuitTest() => await RunTest(async accelerator =>
        {
            var a = new int[] { 0, 1, 0, 1 }; var b = new int[] { 0, 0, 1, 1 };
            int len = 4;
            using var bufA = accelerator.Allocate1D(a); using var bufB = accelerator.Allocate1D(b); using var bufOut = accelerator.Allocate1D<int>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>(ShortCircuitKernel);
            kernel((Index1D)len, bufA.View, bufB.View, bufOut.View);
            await accelerator.SynchronizeAsync();
            var result = await bufOut.CopyToHostAsync<int>();
            int[] expected = { 0, 0, 1, 1 };
            for (int i = 0; i < len; i++) if (result[i] != expected[i]) throw new Exception($"Short circuit failed at {i}");
        });

        [TestMethod]
        public async Task NaNInfinityTest() => await RunTest(async accelerator =>
        {
            var data = new float[] { 1.0f, 1.0f, 0.0f, 0.0f };
            int len = 4;
            using var buf = accelerator.Allocate1D(data); using var bufOut = accelerator.Allocate1D<int>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<int>>(NaNInfinityKernel);
            kernel((Index1D)len, buf.View, bufOut.View);
            await accelerator.SynchronizeAsync();
            var result = await bufOut.CopyToHostAsync<int>();
            if (result[0] != 1) throw new Exception($"Inf > 0 failed");
            if (result[1] != 1) throw new Exception($"-Inf < 0 failed");
            if (result[2] != 0) throw new Exception($"NaN == NaN should be false");
            if (result[3] != 1) throw new Exception($"IsNaN(0/0) should be true");
        });

        [TestMethod]
        public async Task ReductionTest() => await RunTest(async accelerator =>
        {
            int len = 64; var data = new int[len]; for (int i = 0; i < len; i++) data[i] = 1;
            using var bufData = accelerator.Allocate1D(data); using var bufSum = accelerator.Allocate1D(new int[1]);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>>(ReductionKernel);
            kernel((Index1D)len, bufData.View, bufSum.View);
            await accelerator.SynchronizeAsync();
            var result = await bufSum.CopyToHostAsync<int>();
            if (result[0] != 64) throw new Exception($"Reduction failed. Expected 64, got {result[0]}");
        });

        [TestMethod]
        public async Task ScatterGatherTest() => await RunTest(async accelerator =>
        {
            var data = new int[] { 10, 20, 30, 40, 50, 60, 70, 80 }; var indices = new int[] { 7, 5, 3, 1, 6, 4, 2, 0 };
            int len = 8;
            using var bufData = accelerator.Allocate1D(data); using var bufIndices = accelerator.Allocate1D(indices); using var bufOut = accelerator.Allocate1D<int>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>(GatherKernel);
            kernel((Index1D)len, bufData.View, bufIndices.View, bufOut.View);
            await accelerator.SynchronizeAsync();
            var result = await bufOut.CopyToHostAsync<int>();
            int[] expected = { 80, 60, 40, 20, 70, 50, 30, 10 };
            for (int i = 0; i < len; i++) if (result[i] != expected[i]) throw new Exception($"Gather failed at {i}");
        });

        // Mirrors SpawnDev.ILGPU.ML's GatherAxis0FloatImpl pattern (the embedding-lookup
        // kernel used by DistilGPT-2 / DistilBERT / RMBG transformer pipelines). Data
        // observed all-zero outputs from this kernel on WebGL only (see
        // _DevComms\SpawnDev.ILGPU\data-to-geordi-webgl-gather-is-the-bug-2026-05-03.md);
        // CPU + CUDA + OpenCL produce real values. The kernel exercises:
        //   - Index1D / and % integer divide-modulo
        //   - float-buffer-indexed read of an int-valued payload
        //   - explicit (int) cast of a float index
        //   - `data.Length` used in a runtime bounds check
        //   - ternary fallthrough returning data[srcIdx] vs 0f
        //
        // Test data: 4 rows × 3 columns of float weights, indexed by
        // [3, 0, 2, 1] (in float form). Each output row should be the
        // corresponding source row. If WebGL emits the bounds check or the
        // gather wrong, the output will be zero rows instead of the expected
        // permuted rows.
        [TestMethod]
        public async Task GatherAxis0FloatLikeMlTest() => await RunTest(async accelerator =>
        {
            const int dataRows = 4;
            const int innerSize = 3;
            var data = new float[dataRows * innerSize] {
                10.0f, 11.0f, 12.0f,   // row 0
                20.0f, 21.0f, 22.0f,   // row 1
                30.0f, 31.0f, 32.0f,   // row 2
                40.0f, 41.0f, 42.0f,   // row 3
            };
            var indices = new float[] { 3.0f, 0.0f, 2.0f, 1.0f };
            int numIdx = indices.Length;
            int total = numIdx * innerSize;

            using var bufData = accelerator.Allocate1D(data);
            using var bufIndices = accelerator.Allocate1D(indices);
            using var bufOut = accelerator.Allocate1D<float>(total);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                int>(GatherAxis0FloatLikeMlKernel);
            kernel((Index1D)total, bufData.View, bufIndices.View, bufOut.View, innerSize);
            await accelerator.SynchronizeAsync();
            var result = await bufOut.CopyToHostAsync<float>();
            float[] expected = {
                40.0f, 41.0f, 42.0f,   // out row 0 ← src row 3
                10.0f, 11.0f, 12.0f,   // out row 1 ← src row 0
                30.0f, 31.0f, 32.0f,   // out row 2 ← src row 2
                20.0f, 21.0f, 22.0f,   // out row 3 ← src row 1
            };
            for (int i = 0; i < total; i++)
            {
                if (Math.Abs(result[i] - expected[i]) > 1e-5f)
                    throw new Exception($"GatherAxis0FloatLikeMl failed at i={i}: got {result[i]}, expected {expected[i]}");
            }
        });

        [TestMethod]
        public async Task DeepNestingTest() => await RunTest(async accelerator =>
        {
            int len = 8;
            using var buf = accelerator.Allocate1D<int>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(DeepNestingKernel);
            kernel((Index1D)len, buf.View);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<int>();
            for (int i = 0; i < len; i++) if (result[i] != 5) throw new Exception($"Deep nesting failed at {i}");
        });

        [TestMethod]
        public async Task ZeroIterationLoopTest() => await RunTest(async accelerator =>
        {
            var data = new int[] { 100, 100, 100, 100 };
            using var buf = accelerator.Allocate1D(data);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(ZeroLoopKernel);
            kernel((Index1D)4, buf.View);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<int>();
            for (int i = 0; i < 4; i++) if (result[i] != 100) throw new Exception($"Zero loop failed at {i}");
        });

        [TestMethod]
        public async Task MultipleOutputsTest() => await RunTest(async accelerator =>
        {
            int len = 8; var input = new int[len]; for (int i = 0; i < len; i++) input[i] = i;
            using var bufIn = accelerator.Allocate1D(input); using var bufSum = accelerator.Allocate1D<int>(len); using var bufProd = accelerator.Allocate1D<int>(len); using var bufSq = accelerator.Allocate1D<int>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>>(MultiOutputKernel);
            kernel((Index1D)len, bufIn.View, bufSum.View, bufProd.View, bufSq.View);
            await accelerator.SynchronizeAsync();
            var s = await bufSum.CopyToHostAsync<int>(); var p = await bufProd.CopyToHostAsync<int>(); var q = await bufSq.CopyToHostAsync<int>();
            for (int i = 0; i < len; i++) { if (s[i] != i + 10) throw new Exception($"Sum failed at {i}"); if (p[i] != i * 2) throw new Exception($"Product failed at {i}"); if (q[i] != i * i) throw new Exception($"Square failed at {i}"); }
        });

        [TestMethod]
        public async Task MatrixMultiplyTest() => await RunTest(async accelerator =>
        {
            int size = 4; var a = new float[size * size]; var b = new float[size * size];
            for (int i = 0; i < size; i++) for (int j = 0; j < size; j++) { a[i * size + j] = (i == j) ? 1.0f : 0.0f; b[i * size + j] = i * size + j; }
            using var bufA = accelerator.Allocate1D(a); using var bufB = accelerator.Allocate1D(b); using var bufC = accelerator.Allocate1D<float>(size * size);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int>(MatMulKernel);
            kernel((Index1D)(size * size), bufA.View, bufB.View, bufC.View, size);
            await accelerator.SynchronizeAsync();
            var result = await bufC.CopyToHostAsync<float>();
            for (int i = 0; i < size * size; i++) if (MathF.Abs(result[i] - b[i]) > 0.001f) throw new Exception($"Matrix multiply failed at {i}");
        });

        [TestMethod]
        public async Task KernelReuseTest() => await RunTest(async accelerator =>
        {
            int len = 64;
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, int>(AddConstantKernel);
            var d1 = new int[len]; for (int i = 0; i < len; i++) d1[i] = i;
            using var b1 = accelerator.Allocate1D(d1);
            kernel((Index1D)len, b1.View, 100); await accelerator.SynchronizeAsync();
            var r1 = await b1.CopyToHostAsync<int>();
            var d2 = new int[len]; for (int i = 0; i < len; i++) d2[i] = i * 2;
            using var b2 = accelerator.Allocate1D(d2);
            kernel((Index1D)len, b2.View, 50); await accelerator.SynchronizeAsync();
            var r2 = await b2.CopyToHostAsync<int>();
            for (int i = 0; i < len; i++) { if (r1[i] != i + 100) throw new Exception($"First invocation failed at {i}"); if (r2[i] != i * 2 + 50) throw new Exception($"Second invocation failed at {i}"); }
        });

        [TestMethod]
        public async Task ChainedKernelsTest() => await RunTest(async accelerator =>
        {
            int len = 64; var data = new int[len]; for (int i = 0; i < len; i++) data[i] = i;
            using var buf = accelerator.Allocate1D(data);
            var dk = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(DoubleValueKernel);
            var ak = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(AddTenKernel);
            dk((Index1D)len, buf.View); await accelerator.SynchronizeAsync();
            ak((Index1D)len, buf.View); await accelerator.SynchronizeAsync();
            dk((Index1D)len, buf.View); await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<int>();
            for (int i = 0; i < len; i++) { int expected = 4 * i + 20; if (result[i] != expected) throw new Exception($"Chained kernel failed at {i}"); }
        });

        [TestMethod]
        public async Task BoundaryConditionsTest() => await RunTest(async accelerator =>
        {
            int len = 100;
            using var buf = accelerator.Allocate1D<int>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, int>(BoundaryKernel);
            kernel((Index1D)len, buf.View, len);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<int>();
            for (int i = 0; i < len; i++) { int expected = (i == 0) ? -1 : (i == len - 1) ? 1 : 0; if (result[i] != expected) throw new Exception($"Boundary test failed at {i}"); }
        });

        [TestMethod]
        public async Task FloatSpecialOpsTest() => await RunTest(async accelerator =>
        {
            var data = new float[] { -3.7f, -0.5f, 0.0f, 0.5f, 1.5f, 3.7f };
            int len = data.Length;
            using var buf = accelerator.Allocate1D(data); using var bufOut = accelerator.Allocate1D<float>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(FloatSpecialOpsKernel);
            kernel((Index1D)len, buf.View, bufOut.View);
            await accelerator.SynchronizeAsync();
            var result = await bufOut.CopyToHostAsync<float>();
            float[] expected = { 0.0f, 0.0f, 0.0f, 0.5f, 1.0f, 1.0f };
            for (int i = 0; i < len; i++) if (MathF.Abs(result[i] - expected[i]) > 0.001f) throw new Exception($"Saturate failed at {i}");
        });

        [TestMethod]
        public async Task ModuloOperationsTest() => await RunTest(async accelerator =>
        {
            var data = new int[] { 10, -10, 7, -7, 15, -15, 3, -3 };
            int len = data.Length;
            using var buf = accelerator.Allocate1D(data);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(ModuloKernel);
            kernel((Index1D)len, buf.View);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<int>();
            int[] expected = { 2, -2, 3, -3, 3, -3, 3, -3 };
            for (int i = 0; i < len; i++) if (result[i] != expected[i]) throw new Exception($"Modulo failed at {i}");
        });

        [TestMethod]
        public async Task MillionElementsTest() => await RunTest(async accelerator =>
        {
            int len = 1024 * 1024;
            using var buf = accelerator.Allocate1D<int>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(SimpleSetKernel);
            kernel((Index1D)len, buf.View);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<int>();
            int[] ci = { 0, 100, 1000, 10000, 100000, 500000, 999999, len - 1 };
            foreach (int i in ci) if (result[i] != i) throw new Exception($"Large dispatch failed at {i}");
        });

        [TestMethod]
        public async Task StencilOperationTest() => await RunTest(async accelerator =>
        {
            int len = 64; var input = new float[len]; for (int i = 0; i < len; i++) input[i] = i;
            using var bufIn = accelerator.Allocate1D(input); using var bufOut = accelerator.Allocate1D<float>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, int>(StencilKernel);
            kernel((Index1D)len, bufIn.View, bufOut.View, len);
            await accelerator.SynchronizeAsync();
            var result = await bufOut.CopyToHostAsync<float>();
            for (int i = 1; i < len - 1; i++) { float expected = (input[i - 1] + input[i] + input[i + 1]) / 3.0f; if (MathF.Abs(result[i] - expected) > 0.001f) throw new Exception($"Stencil failed at {i}"); }
        });

        [TestMethod]
        public async Task ParallelSumTest() => await RunTest(async accelerator =>
        {
            int len = 64; var data = new int[len]; int expectedSum = 0;
            for (int i = 0; i < len; i++) { data[i] = i + 1; expectedSum += data[i]; }
            using var buf = accelerator.Allocate1D(data); using var bufResult = accelerator.Allocate1D<int>(1);
            bufResult.MemSetToZero();
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, int>(ParallelSumKernel);
            kernel((Index1D)len, buf.View, bufResult.View, len);
            await accelerator.SynchronizeAsync();
            var result = await bufResult.CopyToHostAsync<int>();
            if (result[0] != expectedSum) throw new Exception($"Parallel sum failed. Expected {expectedSum}, got {result[0]}");
        });

        [TestMethod]
        public async Task TypeConversionTest() => await RunTest(async accelerator =>
        {
            var floatData = new float[] { 1.5f, 2.7f, -3.2f, 0.0f, 100.9f, -50.1f, 0.4f, 0.6f };
            int len = floatData.Length;
            using var bufF = accelerator.Allocate1D(floatData); using var bufI = accelerator.Allocate1D<int>(len); using var bufBack = accelerator.Allocate1D<float>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<float>>(TypeConversionKernel);
            kernel((Index1D)len, bufF.View, bufI.View, bufBack.View);
            await accelerator.SynchronizeAsync();
            var intResult = await bufI.CopyToHostAsync<int>();
            int[] expectedInt = { 1, 2, -3, 0, 100, -50, 0, 0 };
            for (int i = 0; i < len; i++) if (intResult[i] != expectedInt[i]) throw new Exception($"Float->Int conversion failed at {i}");
        });

        [TestMethod]
        public async Task ShiftOperationsTest() => await RunTest(async accelerator =>
        {
            var data = new int[] { 1, 2, 4, 8, 16, 255, 1024, -8 };
            int len = data.Length;
            using var bufL = accelerator.Allocate1D(data); using var bufR = accelerator.Allocate1D(data);
            var lk = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(LeftShiftKernel);
            var rk = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(RightShiftKernel);
            lk((Index1D)len, bufL.View); rk((Index1D)len, bufR.View);
            await accelerator.SynchronizeAsync();
            var lr = await bufL.CopyToHostAsync<int>(); var rr = await bufR.CopyToHostAsync<int>();
            int[] el = { 4, 8, 16, 32, 64, 1020, 4096, -32 }; int[] er = { 0, 1, 2, 4, 8, 127, 512, -4 };
            for (int i = 0; i < len; i++) { if (lr[i] != el[i]) throw new Exception($"Left shift failed at {i}: got {lr[i]}, expected {el[i]}"); if (rr[i] != er[i]) throw new Exception($"Right shift failed at {i}: got {rr[i]}, expected {er[i]}"); }
        });

        [TestMethod]
        public async Task NestedStructTest() => await RunTest(async accelerator =>
        {
            int len = 4; var data = new NestedOuterStruct[len];
            for (int i = 0; i < len; i++) data[i] = new NestedOuterStruct { Inner = new NestedInnerStruct { A = i, B = i * 2 }, Value = i * 1.5f };
            using var buf = accelerator.Allocate1D(data); using var bufResult = accelerator.Allocate1D<int>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<NestedOuterStruct>, ArrayView<int>>(NestedStructKernel);
            kernel((Index1D)len, buf.View, bufResult.View);
            await accelerator.SynchronizeAsync();
            var result = await bufResult.CopyToHostAsync<int>();
            for (int i = 0; i < len; i++) { int expected = i + i * 2 + (int)(i * 1.5f); if (result[i] != expected) throw new Exception($"Nested struct failed at {i}"); }
        });

        [TestMethod]
        public async Task BufferCopyTest() => await RunTest(async accelerator =>
        {
            int len = 64; var source = new int[len]; for (int i = 0; i < len; i++) source[i] = i * 3;
            using var bufSrc = accelerator.Allocate1D(source); using var bufDst = accelerator.Allocate1D<int>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>>(CopyKernel);
            kernel((Index1D)len, bufSrc.View, bufDst.View);
            await accelerator.SynchronizeAsync();
            var result = await bufDst.CopyToHostAsync<int>();
            for (int i = 0; i < len; i++) if (result[i] != source[i]) throw new Exception($"Buffer copy failed at {i}");
        });

        [TestMethod]
        public async Task UnaryOperationsTest() => await RunTest(async accelerator =>
        {
            var intData = new int[] { 0, 1, -1, 5, -10, 127, -128, 255 }; var floatData = new float[] { 0.0f, 1.0f, -1.0f, 3.14f, -2.5f, 100.0f, -0.5f, 0.001f };
            int len = intData.Length;
            using var bufInt = accelerator.Allocate1D(intData); using var bufFloat = accelerator.Allocate1D(floatData);
            var ik = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(IntUnaryKernel);
            var fk = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>>(FloatUnaryKernel);
            ik((Index1D)len, bufInt.View); fk((Index1D)len, bufFloat.View);
            await accelerator.SynchronizeAsync();
            var ir = await bufInt.CopyToHostAsync<int>(); var fr = await bufFloat.CopyToHostAsync<float>();
            for (int i = 0; i < len; i++) { if (ir[i] != ~(-intData[i])) throw new Exception($"Int unary failed at {i}: got {ir[i]}, expected {~(-intData[i])}"); if (MathF.Abs(fr[i] - (-floatData[i])) > 0.001f) throw new Exception($"Float unary failed at {i}: got {fr[i]}, expected {-floatData[i]}"); }
        });

        [TestMethod]
        public async Task SmoothstepTest() => await RunTest(async accelerator =>
        {
            int len = 11; var input = new float[len]; for (int i = 0; i < len; i++) input[i] = i * 0.1f;
            using var buf = accelerator.Allocate1D(input);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>>(SmoothstepKernel);
            kernel((Index1D)len, buf.View);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<float>();
            for (int i = 0; i < len; i++) { float t = input[i]; float expected = t * t * (3.0f - 2.0f * t); if (MathF.Abs(result[i] - expected) > 0.001f) throw new Exception($"Smoothstep failed at {i}"); }
        });

        [TestMethod]
        public async Task LerpTest() => await RunTest(async accelerator =>
        {
            int len = 11; var tValues = new float[len]; for (int i = 0; i < len; i++) tValues[i] = i * 0.1f;
            using var bufT = accelerator.Allocate1D(tValues); using var bufOut = accelerator.Allocate1D<float>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, float, float>(LerpKernel);
            kernel((Index1D)len, bufT.View, bufOut.View, 10.0f, 50.0f);
            await accelerator.SynchronizeAsync();
            var result = await bufOut.CopyToHostAsync<float>();
            for (int i = 0; i < len; i++) { float expected = 10.0f + tValues[i] * (50.0f - 10.0f); if (MathF.Abs(result[i] - expected) > 0.001f) throw new Exception($"Lerp failed at {i}"); }
        });

        [TestMethod]
        public async Task MixedTypesTest() => await RunTest(async accelerator =>
        {
            int len = 8; var intData = new int[] { 1, 2, 3, 4, 5, 6, 7, 8 }; var uintData = new uint[] { 10, 20, 30, 40, 50, 60, 70, 80 }; var floatData = new float[] { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f };
            using var bufInt = accelerator.Allocate1D(intData); using var bufUint = accelerator.Allocate1D(uintData); using var bufFloat = accelerator.Allocate1D(floatData); using var bufResult = accelerator.Allocate1D<float>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<uint>, ArrayView<float>, ArrayView<float>>(MixedTypesKernel);
            kernel((Index1D)len, bufInt.View, bufUint.View, bufFloat.View, bufResult.View);
            await accelerator.SynchronizeAsync();
            var result = await bufResult.CopyToHostAsync<float>();
            for (int i = 0; i < len; i++) { float expected = intData[i] + uintData[i] + floatData[i]; if (MathF.Abs(result[i] - expected) > 0.001f) throw new Exception($"Mixed types failed at {i}"); }
        });

        [TestMethod]
        public async Task ComplexExpressionTest() => await RunTest(async accelerator =>
        {
            int len = 16; var data = new float[len]; for (int i = 0; i < len; i++) data[i] = i + 1;
            using var buf = accelerator.Allocate1D(data);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>>(ComplexExpressionKernel);
            kernel((Index1D)len, buf.View);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<float>();
            for (int i = 0; i < len; i++) { float x = i + 1; float expected = ((x * 2 + 3) / 4 - 1) * 5 + x; if (MathF.Abs(result[i] - expected) > 0.01f) throw new Exception($"Complex expression failed at {i}"); }
        });

        [TestMethod]
        public async Task LongLargeDatasetTest() => await RunEmulatedTest(async accelerator =>
        {
            int len = 1024;
            var data = new long[len]; for (int i = 0; i < len; i++) data[i] = i;
            using var buf = accelerator.Allocate1D(data);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<long>>(LongLargeDatasetKernel);
            kernel((Index1D)len, buf.View);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<long>();
            for (int i = 0; i < len; i += 100) { long expected = i * 2L + 1L; if (result[i] != expected) throw new Exception($"Long large dataset failed at {i}. Expected {expected}, got {result[i]}"); }
        });

        [TestMethod]
        public async Task DoubleChainedOpsTest() => await RunEmulatedTest(async accelerator =>
        {
            var input = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 10.0 };
            int len = input.Length;
            using var bufIn = accelerator.Allocate1D(input); using var bufOut = accelerator.Allocate1D<double>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>, ArrayView<double>>(DoubleChainedOpsKernel);
            kernel((Index1D)len, bufIn.View, bufOut.View);
            await accelerator.SynchronizeAsync();
            var result = await bufOut.CopyToHostAsync<double>();
            for (int i = 0; i < len; i++) { double expected = (input[i] * 2.5 + 1.0) / 3.0; double tol = Math.Max(Math.Abs(expected * 0.01), 1e-6); if (Math.Abs(result[i] - expected) > tol) throw new Exception($"Double chained ops failed at {i}"); }
        });

        [TestMethod]
        public async Task DoubleLargeDatasetTest() => await RunEmulatedTest(async accelerator =>
        {
            int len = 1024;
            var data = new double[len]; for (int i = 0; i < len; i++) data[i] = i * 0.5;
            using var buf = accelerator.Allocate1D(data);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>>(DoubleLargeDatasetKernel);
            kernel((Index1D)len, buf.View);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<double>();
            for (int i = 0; i < len; i += 100) { double expected = i * 0.5 * 2.0 + 1.0; double tol = Math.Max(Math.Abs(expected * 0.01), 1e-6); if (Math.Abs(result[i] - expected) > tol) throw new Exception($"Double large dataset failed at {i}"); }
        });

        [TestMethod]
        public async Task DoublePrecisionVerifyTest() => await RunEmulatedTest(async accelerator =>
        {
            var data = new double[] { 0.1, 0.2, 0.3, 0.1 + 0.2, 1.0 / 3.0, 2.0 / 3.0 };
            int len = data.Length;
            using var buf = accelerator.Allocate1D(data);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>>(DoublePrecisionVerifyKernel);
            kernel((Index1D)len, buf.View);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<double>();
            for (int i = 0; i < len; i++) { double expected = data[i]; double tolerance = Math.Max(Math.Abs(expected * 0.02), 1e-5); if (Math.Abs(result[i] - expected) > tolerance) throw new Exception($"Double precision verify failed at {i}. Expected {expected}, got {result[i]}"); }
        });

        [TestMethod]
        public async Task DynamicSharedF64Test() => await RunEmulatedTest(async accelerator =>
        {
            int len = 64;
            var data = new double[len]; for (int i = 0; i < len; i++) data[i] = i * 1.5;
            using var buf = accelerator.Allocate1D(data);
            var kernel = accelerator.LoadStreamKernel<Index1D, ArrayView<double>>(DynamicSharedF64Kernel);
            kernel(new KernelConfig(1, len, SharedMemoryConfig.RequestDynamic<double>(len)), (Index1D)len, buf.View);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<double>();
            for (int i = 0; i < len; i++) { double expected = (63 - i) * 1.5; double tol = Math.Max(Math.Abs(expected * 0.01), 1e-6); if (Math.Abs(result[i] - expected) > tol) throw new Exception($"Dynamic shared F64 failed at {i}"); }
        });

        [TestMethod]
        public async Task BroadcastTest() => await RunTest(async accelerator =>
        {
            int len = 64;
            var data = new int[len];
            for (int i = 0; i < len; i++) data[i] = i + 10; // [10, 11, 12, ..., 73]
            using var buf = accelerator.Allocate1D(data);

            // Load stream kernel (explicit KernelConfig) and launch with 1 grid, 64-thread group
            var kernel = accelerator.LoadStreamKernel<Index1D, ArrayView<int>>(BroadcastKernel);
            kernel(new KernelConfig(1, len), (Index1D)len, buf.View);
            await accelerator.SynchronizeAsync();

            var result = await buf.CopyToHostAsync<int>();
            // Group.Broadcast(val, 0) should broadcast thread 0's value (10) to all threads
            for (int i = 0; i < len; i++)
            {
                if (result[i] != 10)
                    throw new Exception($"BroadcastTest failed at [{i}]: Expected 10, got {result[i]}");
            }
        });

        [TestMethod]
        public async Task SubgroupShuffleTest() => await RunTest(async accelerator =>
        {
            // Dynamically check if subgroups are available
            RequireFeature(accelerator, "subgroup_shuffle", "Subgroup/Warp shuffle operations require 'subgroup_shuffle' feature");

            // Use 32 threads. Note: subgroup_size varies by GPU (4, 8, 16, 32, 64).
            // Warp.Shuffle(val, 0) reads from lane 0 of each thread's own subgroup,
            // NOT global thread 0.
            int len = 32;
            var data = new int[len];
            for (int i = 0; i < len; i++) data[i] = i + 10; // [10, 11, 12, ..., 41]
            using var buf = accelerator.Allocate1D(data);
            using var warpSizeBuf = accelerator.Allocate1D<int>(1);

            var kernel = accelerator.LoadStreamKernel<Index1D, ArrayView<int>, ArrayView<int>>(SubgroupShuffleKernel);
            kernel(new KernelConfig(1, len), (Index1D)len, buf.View, warpSizeBuf.View);
            await accelerator.SynchronizeAsync();

            var warpSizeResult = await warpSizeBuf.CopyToHostAsync<int>();
            int warpSize = warpSizeResult[0];
            var result = await buf.CopyToHostAsync<int>();

            // Each thread should have lane 0 of its own subgroup's value
            for (int i = 0; i < len; i++)
            {
                int subgroupBase = (i / warpSize) * warpSize;
                int expected = data[subgroupBase]; // lane 0's original value
                if (result[i] != expected)
                    throw new Exception($"SubgroupShuffleTest failed at [{i}]: Expected {expected} (warpSize={warpSize}, subgroupBase={subgroupBase}), got {result[i]}");
            }
        });

        /// <summary>
        /// Tests a kernel with 12 scalar parameters. With auto-grouping overhead (~7 additional
        /// bindings), this generates ~19 storage buffer bindings — well above WebGPU's default
        /// limit of 8. Validates that the adapter's max limit is properly requested.
        /// </summary>
        [TestMethod]
        public async Task ManyScalarKernelTest() => await RunTest(async accelerator =>
        {
            int len = 64;
            using var buf = accelerator.Allocate1D<int>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<int>,
                int, int, int, int, int, int, int, int, int, int, int, int>(ManyScalarKernel);
            // Sum of 1+2+3+4+5+6+7+8+9+10+11+12 = 78
            kernel((Index1D)len, buf.View, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<int>();
            for (int i = 0; i < len; i++)
                if (result[i] != 78)
                    throw new Exception($"Many scalar kernel failed at {i}. Expected 78, got {result[i]}");
        });

        /// <summary>
        /// Tests the colormap kernel which uses multiple Math.Min(Math.Max(...)) calls,
        /// float→uint casts, bitwise operations, and ArrayView.Length — the pattern
        /// that originally triggered the GetViewLength code generation bug.
        /// </summary>
        [TestMethod]
        public async Task ColormapKernelTest() => await RunTest(async accelerator =>
        {
            int length = 64;
            var depthData = Enumerable.Range(0, length).Select(i => (float)i).ToArray();
            float minVal = 0f;
            float invRange = 1f / 63f;

            using var depthBuf = accelerator.Allocate1D(depthData);
            using var colorBuf = accelerator.Allocate1D<uint>(length);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<uint>, float, float>(ColormapKernel);
            kernel((Index1D)length, depthBuf.View, colorBuf.View, minVal, invRange);
            await accelerator.SynchronizeAsync();

            var result = await colorBuf.CopyToHostAsync<uint>();

            for (int i = 0; i < length; i++)
            {
                uint pixel = result[i];
                uint a = (pixel >> 24) & 0xFF;
                if (a != 255)
                    throw new Exception($"Colormap alpha failed at {i}. Expected 255, got {a}");
            }

            // First pixel (t=0) should be dark
            uint r0 = result[0] & 0xFF;
            if (r0 > 100)
                throw new Exception($"First pixel should be dark, but R={r0}");

            // Last pixel (t=1) should be bright
            uint rLast = result[length - 1] & 0xFF;
            if (rLast < 200)
                throw new Exception($"Last pixel should be bright, but R={rLast}");
        });

        #endregion
    }
}
