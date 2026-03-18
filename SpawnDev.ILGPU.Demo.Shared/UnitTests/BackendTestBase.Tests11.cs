using ILGPU;
using ILGPU.Runtime;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.Demo.Shared.UnitTests
{
    // Part 11: Lambda / Closure tests
    public abstract partial class BackendTestBase
    {
        /// <summary>
        /// Tests that a capturing lambda can be compiled and dispatched.
        /// Captures a single int value and uses it in the kernel body.
        /// </summary>
        [TestMethod]
        public async Task CapturingLambdaSingleIntTest() => await RunTest(async accelerator =>
        {
            int len = 64;
            int capturedValue = 42;
            using var buf = accelerator.Allocate1D<int>(len);

            // This lambda captures 'capturedValue' — the C# compiler generates
            // a display class with an int field. ILGPU must treat it as a struct
            // parameter and pass the captured value at dispatch time.
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(
                (Index1D index, ArrayView<int> dataView) =>
                {
                    dataView[index] = index + capturedValue;
                });

            kernel((Index1D)len, buf.View);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<int>();
            for (int i = 0; i < len; i++)
            {
                int expected = i + 42;
                if (result[i] != expected)
                    throw new Exception($"CapturingLambda failed at {i}. Expected {expected}, got {result[i]}");
            }
        });

        /// <summary>
        /// Tests capturing multiple values of different types.
        /// </summary>
        [TestMethod]
        public async Task CapturingLambdaMultiFieldTest() => await RunTest(async accelerator =>
        {
            int len = 64;
            int intCapture = 10;
            float floatCapture = 0.5f;
            using var buf = accelerator.Allocate1D<float>(len);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>>(
                (Index1D index, ArrayView<float> dataView) =>
                {
                    dataView[index] = index * floatCapture + intCapture;
                });

            kernel((Index1D)len, buf.View);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<float>();
            for (int i = 0; i < len; i++)
            {
                float expected = i * 0.5f + 10;
                if (MathF.Abs(result[i] - expected) > 0.001f)
                    throw new Exception($"CapturingLambdaMulti failed at {i}. Expected {expected}, got {result[i]}");
            }
        });

        /// <summary>
        /// Tests that capturing an ArrayView throws NotSupportedException
        /// (ArrayView captures are not yet supported — pass as explicit param).
        /// </summary>
        [TestMethod]
        public async Task CapturingLambdaArrayViewCaptureRejectTest() => await RunTest(async accelerator =>
        {
            int len = 64;
            using var lookupBuf = accelerator.Allocate1D<int>(len);
            var lookupView = lookupBuf.View;

            bool threw = false;
            try
            {
                var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(
                    (Index1D index, ArrayView<int> dataView) =>
                    {
                        dataView[index] = lookupView[index] + 1;
                    });
            }
            catch (NotSupportedException ex) when (ex.Message.Contains("ArrayView"))
            {
                threw = true;
            }

            if (!threw)
                throw new Exception("Expected NotSupportedException for ArrayView capture");

            await Task.CompletedTask;
        });
    }
}
