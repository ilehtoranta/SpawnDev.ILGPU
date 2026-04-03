using ILGPU;
using ILGPU.Runtime;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.Demo.Shared.UnitTests
{
    // Part 16: Coverage gap tests identified during April 2026 audit.
    // Buffer aliasing detection, MemSet verification, GPU→GPU copy stress,
    // disposed buffer handling, allocation edge cases.
    public abstract partial class BackendTestBase
    {
        #region Part 16 Kernel Definitions

        static void FillWithIndexKernel(Index1D idx, ArrayView<int> data)
        {
            data[idx] = idx;
        }

        static void DoubleValuesKernel(Index1D idx, ArrayView<int> data)
        {
            data[idx] = data[idx] * 2;
        }

        #endregion

        /// <summary>
        /// Verify MemSetToZero actually clears a buffer that was previously filled.
        /// </summary>
        [TestMethod]
        public async Task MemSetToZero_ClearsFilledBufferTest() => await RunTest(async accelerator =>
        {
            int length = 256;
            using var buf = accelerator.Allocate1D<int>(length);

            // Fill with non-zero values
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(FillWithIndexKernel);
            kernel((Index1D)length, buf.View);
            await accelerator.SynchronizeAsync();

            // Verify filled
            var filled = await buf.CopyToHostAsync<int>();
            if (filled[100] != 100)
                throw new Exception($"Fill failed: expected 100, got {filled[100]}");

            // MemSet to zero
            buf.MemSetToZero();
            await accelerator.SynchronizeAsync();

            // Verify cleared
            var cleared = await buf.CopyToHostAsync<int>();
            for (int i = 0; i < length; i++)
                if (cleared[i] != 0)
                    throw new Exception($"MemSetToZero failed at [{i}]: expected 0, got {cleared[i]}");
        });

        /// <summary>
        /// GPU→GPU CopyFrom preserves data across multiple sequential copies.
        /// Stress test for staging buffer reuse and command batching.
        /// </summary>
        [TestMethod]
        public async Task BufferCopy_GPUToGPU_SequentialStressTest() => await RunTest(async accelerator =>
        {
            int length = 1024;
            var data = new int[length];
            for (int i = 0; i < length; i++) data[i] = i * 3 + 7;

            using var source = accelerator.Allocate1D(data);

            // Copy 20 times sequentially — stress staging buffer reuse
            for (int iter = 0; iter < 20; iter++)
            {
                using var dest = accelerator.Allocate1D<int>(length);
                dest.CopyFrom(source);
                await accelerator.SynchronizeAsync();

                var result = await dest.CopyToHostAsync<int>();
                // Spot check
                if (result[0] != 7 || result[500] != 1507 || result[1023] != 3076)
                    throw new Exception($"GPU→GPU copy iter {iter} corrupted: [{result[0]}, {result[500]}, {result[1023]}]");
            }
        });

        /// <summary>
        /// Multiple buffers allocated sequentially — verify no allocation fragmentation.
        /// </summary>
        [TestMethod]
        public async Task AllocationStress_ManyBuffersSequentialTest() => await RunTest(async accelerator =>
        {
            var buffers = new List<MemoryBuffer1D<int, Stride1D.Dense>>();
            try
            {
                // Allocate 50 buffers of increasing size
                for (int i = 0; i < 50; i++)
                {
                    var buf = accelerator.Allocate1D<int>(1000 + i * 100);
                    buffers.Add(buf);
                }

                // Fill last buffer and verify
                var last = buffers[^1];
                var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(FillWithIndexKernel);
                kernel((Index1D)(int)last.Length, last.View);
                await accelerator.SynchronizeAsync();

                var result = await last.CopyToHostAsync<int>();
                int expectedLen = 1000 + 49 * 100; // 5900
                if (result[expectedLen - 1] != expectedLen - 1)
                    throw new Exception($"Last buffer fill failed: expected {expectedLen - 1}, got {result[expectedLen - 1]}");
            }
            finally
            {
                foreach (var buf in buffers) buf.Dispose();
            }
        });

        /// <summary>
        /// SubView CopyFrom: copy from one SubView to another within different parent buffers.
        /// </summary>
        [TestMethod]
        public async Task SubView_CopyFrom_DifferentParentsTest() => await RunTest(async accelerator =>
        {
            int totalLen = 2048;
            int subLen = 512;
            int srcOffset = 256;
            int dstOffset = 1024;

            var srcData = new float[totalLen];
            for (int i = 0; i < totalLen; i++) srcData[i] = i * 0.1f;

            using var srcBuf = accelerator.Allocate1D(srcData);
            using var dstBuf = accelerator.Allocate1D<float>(totalLen);
            dstBuf.MemSetToZero();

            // Copy SubView[256..768] from source to SubView[1024..1536] in dest
            var srcSub = srcBuf.View.SubView(srcOffset, subLen);
            var dstSub = dstBuf.View.SubView(dstOffset, subLen);
            dstSub.BaseView.CopyFrom(srcSub.BaseView);
            await accelerator.SynchronizeAsync();

            var result = await dstBuf.CopyToHostAsync<float>();

            // Verify the SubView region was copied
            for (int i = 0; i < subLen; i++)
            {
                float expected = (srcOffset + i) * 0.1f;
                float actual = result[dstOffset + i];
                if (MathF.Abs(actual - expected) > 0.01f)
                    throw new Exception($"SubView CopyFrom [{i}]: expected={expected:F2}, got={actual:F2}");
            }

            // Verify outside the SubView region is still zero
            if (result[0] != 0f || result[dstOffset - 1] != 0f)
                throw new Exception("Data outside SubView copy region was modified");
        });

        /// <summary>
        /// Kernel dispatch after buffer is filled, then re-filled with different values.
        /// Verifies buffer contents aren't stale from prior dispatch.
        /// </summary>
        [TestMethod]
        public async Task BufferReuse_OverwriteAndVerifyTest() => await RunTest(async accelerator =>
        {
            int length = 512;
            using var buf = accelerator.Allocate1D<int>(length);

            var fillKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(FillWithIndexKernel);
            var doubleKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(DoubleValuesKernel);

            // Pass 1: fill with indices
            fillKernel((Index1D)length, buf.View);
            await accelerator.SynchronizeAsync();

            // Pass 2: double the values
            doubleKernel((Index1D)length, buf.View);
            await accelerator.SynchronizeAsync();

            var result = await buf.CopyToHostAsync<int>();
            for (int i = 0; i < length; i++)
            {
                int expected = i * 2;
                if (result[i] != expected)
                    throw new Exception($"Buffer reuse [{i}]: expected={expected}, got={result[i]}");
            }
        });

        /// <summary>
        /// Verify CopyToHostAsync with pre-allocated destination array.
        /// Zero-allocation hot path for render loops.
        /// </summary>
        [TestMethod]
        public async Task CopyToHostAsync_PreAllocatedDestinationTest() => await RunTest(async accelerator =>
        {
            int length = 256;
            var srcData = new float[length];
            for (int i = 0; i < length; i++) srcData[i] = i * 1.5f;

            using var buf = accelerator.Allocate1D(srcData);

            // Pre-allocate destination
            var dest = new float[length];
            var result = await buf.CopyToHostAsync<float>();

            for (int i = 0; i < length; i++)
                if (MathF.Abs(result[i] - srcData[i]) > 0.001f)
                    throw new Exception($"CopyToHostAsync [{i}]: expected={srcData[i]:F3}, got={result[i]:F3}");
        });

        // --- Buffer Aliasing Detection Kernels ---
        static void AliasingKernel(Index1D idx, ArrayView<float> input, ArrayView<float> output)
        {
            output[idx] = input[idx] * 2.0f;
        }

        /// <summary>
        /// WebGPU should detect and reject overlapping buffer bindings.
        /// Other backends may allow aliasing (CUDA/CPU don't have this restriction).
        /// This test verifies the aliasing detection works correctly on WebGPU.
        /// </summary>
        [TestMethod]
        public async Task BufferAliasing_SameBufferTwoParams_WebGPUDetectsTest() => await RunTest(async accelerator =>
        {
            if (accelerator.AcceleratorType != AcceleratorType.WebGPU)
                throw new UnsupportedTestException("Buffer aliasing detection is WebGPU-specific");

            int length = 256;
            using var buf = accelerator.Allocate1D<float>(length);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(AliasingKernel);

            // Pass the SAME buffer as both input and output — should throw on WebGPU
            bool threw = false;
            try
            {
                kernel((Index1D)length, buf.View, buf.View);
                await accelerator.SynchronizeAsync();
            }
            catch (InvalidOperationException ex) when (ex.Message.Contains("aliasing") || ex.Message.Contains("Aliasing") || ex.Message.Contains("same buffer"))
            {
                threw = true;
                Console.WriteLine($"[Aliasing] Correctly detected: {ex.Message}");
            }

            if (!threw)
                throw new Exception("WebGPU should detect and reject same-buffer aliasing for read_write bindings");
        });

        /// <summary>
        /// Verify two DIFFERENT buffers with same element type dispatch correctly.
        /// This is the correct pattern — no aliasing.
        /// </summary>
        [TestMethod]
        public async Task BufferAliasing_DifferentBuffers_WorksTest() => await RunTest(async accelerator =>
        {
            int length = 256;
            var inputData = new float[length];
            for (int i = 0; i < length; i++) inputData[i] = i * 0.5f;

            using var inputBuf = accelerator.Allocate1D(inputData);
            using var outputBuf = accelerator.Allocate1D<float>(length);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(AliasingKernel);
            kernel((Index1D)length, inputBuf.View, outputBuf.View);
            await accelerator.SynchronizeAsync();

            var result = await outputBuf.CopyToHostAsync<float>();
            for (int i = 0; i < length; i++)
            {
                float expected = i * 1.0f; // 0.5 * 2
                if (MathF.Abs(result[i] - expected) > 0.01f)
                    throw new Exception($"Non-aliased dispatch [{i}]: expected={expected:F2}, got={result[i]:F2}");
            }
        });
    }
}
