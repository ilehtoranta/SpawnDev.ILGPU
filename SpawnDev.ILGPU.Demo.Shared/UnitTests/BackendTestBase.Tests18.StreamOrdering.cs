using ILGPU;
using ILGPU.Runtime;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.Demo.Shared.UnitTests
{
    // Tests18: Stream ordering and implicit synchronization
    // Verifies that chained kernel dispatches on the same stream produce correct results
    // and that CopyToHostAsync implicitly synchronizes pending work before readback.
    public abstract partial class BackendTestBase
    {
        // Kernel A: writes index * 2 to each element
        static void StreamOrderKernelA(Index1D index, ArrayView<int> output)
        {
            output[index] = index * 2;
        }

        // Kernel B: reads from input, writes input[i] + 10 to output
        static void StreamOrderKernelB(Index1D index, ArrayView<int> input, ArrayView<int> output)
        {
            output[index] = input[index] + 10;
        }

        // Kernel C: reads from input, writes input[i] * 3 to output
        static void StreamOrderKernelC(Index1D index, ArrayView<int> input, ArrayView<int> output)
        {
            output[index] = input[index] * 3;
        }

        // In-place kernel: doubles each element
        static void StreamOrderInPlaceDouble(Index1D index, ArrayView<int> data)
        {
            data[index] = data[index] * 2;
        }

        // In-place kernel: adds 1 to each element
        static void StreamOrderInPlaceIncrement(Index1D index, ArrayView<int> data)
        {
            data[index] = data[index] + 1;
        }

        /// <summary>
        /// Chains 3 kernel dispatches: A writes buffer, B reads A's output and writes buffer2,
        /// C reads B's output and writes buffer3. No explicit SynchronizeAsync between dispatches.
        /// Verifies the full pipeline produces correct results.
        /// Expected: buf3[i] = (i * 2 + 10) * 3
        /// </summary>
        [TestMethod]
        public async Task ChainedDispatch3KernelTest() => await RunTest(async accelerator =>
        {
            int len = 1024;
            using var buf1 = accelerator.Allocate1D<int>(len);
            using var buf2 = accelerator.Allocate1D<int>(len);
            using var buf3 = accelerator.Allocate1D<int>(len);

            var kernelA = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(StreamOrderKernelA);
            var kernelB = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>>(StreamOrderKernelB);
            var kernelC = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>>(StreamOrderKernelC);

            // Chain 3 dispatches - NO SynchronizeAsync between them
            kernelA((Index1D)len, buf1.View);
            kernelB((Index1D)len, buf1.View, buf2.View);
            kernelC((Index1D)len, buf2.View, buf3.View);

            // Only sync + readback at the end
            await accelerator.SynchronizeAsync();
            var result = await buf3.CopyToHostAsync<int>();

            for (int i = 0; i < len; i++)
            {
                int expected = (i * 2 + 10) * 3;
                if (result[i] != expected)
                    throw new Exception($"ChainedDispatch3Kernel failed at {i}. Expected {expected}, got {result[i]}");
            }
        });

        /// <summary>
        /// Chains 5 in-place operations on the same buffer without explicit sync between them.
        /// This is the pattern most likely to break if stream ordering is wrong.
        /// Expected: buf[i] = ((((i * 2) + 1) * 2) + 1) * 2
        /// </summary>
        [TestMethod]
        public async Task ChainedInPlaceDispatch5Test() => await RunTest(async accelerator =>
        {
            int len = 512;
            using var buf = accelerator.Allocate1D<int>(len);

            var writeKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(StreamOrderKernelA);
            var doubleKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(StreamOrderInPlaceDouble);
            var incKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(StreamOrderInPlaceIncrement);

            // Initial write: buf[i] = i * 2
            writeKernel((Index1D)len, buf.View);
            // Double: buf[i] = i * 4
            doubleKernel((Index1D)len, buf.View);
            // Increment: buf[i] = i * 4 + 1
            incKernel((Index1D)len, buf.View);
            // Double: buf[i] = (i * 4 + 1) * 2
            doubleKernel((Index1D)len, buf.View);
            // Increment: buf[i] = (i * 4 + 1) * 2 + 1
            incKernel((Index1D)len, buf.View);

            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<int>();

            for (int i = 0; i < len; i++)
            {
                int expected = (i * 4 + 1) * 2 + 1;
                if (result[i] != expected)
                    throw new Exception($"ChainedInPlace5 failed at {i}. Expected {expected}, got {result[i]}");
            }
        });

        /// <summary>
        /// Tests CopyToHostAsync implicit synchronization - dispatches a kernel and immediately
        /// calls CopyToHostAsync WITHOUT an explicit SynchronizeAsync() first.
        /// If implicit sync is working, the readback should see the completed kernel output.
        /// </summary>
        [TestMethod]
        public async Task ImplicitSyncOnReadbackTest() => await RunTest(async accelerator =>
        {
            int len = 256;
            using var buf = accelerator.Allocate1D<int>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(StreamOrderKernelA);

            // Dispatch kernel, then immediately readback - NO SynchronizeAsync
            kernel((Index1D)len, buf.View);
            var result = await buf.CopyToHostAsync<int>();

            for (int i = 0; i < len; i++)
            {
                int expected = i * 2;
                if (result[i] != expected)
                    throw new Exception($"ImplicitSyncOnReadback failed at {i}. Expected {expected}, got {result[i]}");
            }
        });

        /// <summary>
        /// Tests CopyToHostAsync implicit sync with chained dispatches.
        /// Dispatches 3 kernels, then calls CopyToHostAsync on the final buffer
        /// WITHOUT any SynchronizeAsync. All 3 dispatches must complete before readback.
        /// </summary>
        [TestMethod]
        public async Task ImplicitSyncChainedReadbackTest() => await RunTest(async accelerator =>
        {
            int len = 512;
            using var buf1 = accelerator.Allocate1D<int>(len);
            using var buf2 = accelerator.Allocate1D<int>(len);
            using var buf3 = accelerator.Allocate1D<int>(len);

            var kernelA = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(StreamOrderKernelA);
            var kernelB = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>>(StreamOrderKernelB);
            var kernelC = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>>(StreamOrderKernelC);

            // Chain 3 dispatches, then readback WITHOUT SynchronizeAsync
            kernelA((Index1D)len, buf1.View);
            kernelB((Index1D)len, buf1.View, buf2.View);
            kernelC((Index1D)len, buf2.View, buf3.View);

            // Direct readback - implicit sync must handle this
            var result = await buf3.CopyToHostAsync<int>();

            for (int i = 0; i < len; i++)
            {
                int expected = (i * 2 + 10) * 3;
                if (result[i] != expected)
                    throw new Exception($"ImplicitSyncChainedReadback failed at {i}. Expected {expected}, got {result[i]}");
            }
        });

        /// <summary>
        /// Tests mid-pipeline readback: dispatch A, readback A's output,
        /// dispatch B using A's buffer, readback B's output.
        /// Verifies that readback between dispatches doesn't corrupt the pipeline.
        /// </summary>
        [TestMethod]
        public async Task MidPipelineReadbackTest() => await RunTest(async accelerator =>
        {
            int len = 256;
            using var buf1 = accelerator.Allocate1D<int>(len);
            using var buf2 = accelerator.Allocate1D<int>(len);

            var kernelA = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(StreamOrderKernelA);
            var kernelB = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>>(StreamOrderKernelB);

            // Dispatch A
            kernelA((Index1D)len, buf1.View);

            // Readback A's output mid-pipeline
            await accelerator.SynchronizeAsync();
            var resultA = await buf1.CopyToHostAsync<int>();
            for (int i = 0; i < len; i++)
            {
                int expected = i * 2;
                if (resultA[i] != expected)
                    throw new Exception($"MidPipelineReadback stage A failed at {i}. Expected {expected}, got {resultA[i]}");
            }

            // Dispatch B reading from same buffer A wrote to
            kernelB((Index1D)len, buf1.View, buf2.View);

            // Readback B's output
            await accelerator.SynchronizeAsync();
            var resultB = await buf2.CopyToHostAsync<int>();
            for (int i = 0; i < len; i++)
            {
                int expected = i * 2 + 10;
                if (resultB[i] != expected)
                    throw new Exception($"MidPipelineReadback stage B failed at {i}. Expected {expected}, got {resultB[i]}");
            }
        });
    }
}
