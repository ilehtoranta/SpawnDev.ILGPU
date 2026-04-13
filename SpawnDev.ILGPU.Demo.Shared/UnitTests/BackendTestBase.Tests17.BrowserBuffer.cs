using ILGPU;
using ILGPU.Runtime;
using SpawnDev.BlazorJS.JSObjects;
using SpawnDev.ILGPU.WebGPU.Backend;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.Demo.Shared.UnitTests
{
    // Part 17: Browser buffer operations - CopyFromJS, public Buffer property,
    // GPU-to-GPU buffer access for zero-copy rendering pipelines.
    public abstract partial class BackendTestBase
    {
        /// <summary>
        /// Verify CopyFromJS with TypedArray writes data correctly to GPU buffer.
        /// Writes via JS, reads back via CopyToHostAsync, compares with expected values.
        /// </summary>
        [TestMethod]
        public async Task CopyFromJS_TypedArray_WritesCorrectDataTest() => await RunTest(async accelerator =>
        {
            if (accelerator.AcceleratorType is not (AcceleratorType.WebGPU or AcceleratorType.WebGL or AcceleratorType.Wasm))
                throw new UnsupportedTestException("CopyFromJS only available on browser backends");

            var testData = new float[] { 1.0f, 2.5f, 3.14f, 42.0f, -7.7f, 0f, 100f, 255f };
            using var buffer = accelerator.Allocate1D<float>(testData.Length);

            // Write via JS TypedArray
            var browserBuffer = buffer.Buffer as IBrowserMemoryBuffer;
            if (browserBuffer == null)
                throw new Exception("Buffer does not implement IBrowserMemoryBuffer");

            using var jsArray = new Float32Array(testData);
            browserBuffer.CopyFromJS(jsArray);

            // Read back and verify
            var result = await buffer.CopyToHostAsync();
            for (int i = 0; i < testData.Length; i++)
            {
                if (Math.Abs(result[i] - testData[i]) > 0.001f)
                    throw new Exception($"Mismatch at [{i}]: expected {testData[i]}, got {result[i]}");
            }
        });

        /// <summary>
        /// Verify CopyFromJS with ArrayBuffer writes data correctly.
        /// </summary>
        [TestMethod]
        public async Task CopyFromJS_ArrayBuffer_WritesCorrectDataTest() => await RunTest(async accelerator =>
        {
            if (accelerator.AcceleratorType is not (AcceleratorType.WebGPU or AcceleratorType.WebGL or AcceleratorType.Wasm))
                throw new UnsupportedTestException("CopyFromJS only available on browser backends");

            var testData = new byte[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
            using var buffer = accelerator.Allocate1D<byte>(testData.Length);

            var browserBuffer = buffer.Buffer as IBrowserMemoryBuffer;
            if (browserBuffer == null)
                throw new Exception("Buffer does not implement IBrowserMemoryBuffer");

            using var jsArray = new Uint8Array(testData);
            using var arrayBuffer = jsArray.Buffer;
            browserBuffer.CopyFromJS(arrayBuffer);

            var result = await buffer.CopyToHostAsync();
            for (int i = 0; i < testData.Length; i++)
            {
                if (result[i] != testData[i])
                    throw new Exception($"Mismatch at [{i}]: expected {testData[i]}, got {result[i]}");
            }
        });

        /// <summary>
        /// Verify CopyFromJS data can be processed by a kernel and produces correct results.
        /// End-to-end: JS data -> GPU buffer -> kernel -> GPU buffer -> CPU readback.
        /// </summary>
        [TestMethod]
        public async Task CopyFromJS_KernelProcessing_EndToEndTest() => await RunTest(async accelerator =>
        {
            if (accelerator.AcceleratorType is not (AcceleratorType.WebGPU or AcceleratorType.WebGL or AcceleratorType.Wasm))
                throw new UnsupportedTestException("CopyFromJS only available on browser backends");

            var inputData = new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f };
            using var input = accelerator.Allocate1D<float>(inputData.Length);
            using var output = accelerator.Allocate1D<float>(inputData.Length);

            // Load input via CopyFromJS
            var browserBuffer = input.Buffer as IBrowserMemoryBuffer;
            if (browserBuffer == null)
                throw new Exception("Buffer does not implement IBrowserMemoryBuffer");

            using var jsArray = new Float32Array(inputData);
            browserBuffer.CopyFromJS(jsArray);

            // Run kernel that doubles values
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(
                (Index1D idx, ArrayView<float> src, ArrayView<float> dst) => { dst[idx] = src[idx] * 2f; });
            kernel(inputData.Length, input.View, output.View);
            await accelerator.SynchronizeAsync();

            // Verify
            var result = await output.CopyToHostAsync();
            for (int i = 0; i < inputData.Length; i++)
            {
                float expected = inputData[i] * 2f;
                if (Math.Abs(result[i] - expected) > 0.001f)
                    throw new Exception($"Mismatch at [{i}]: expected {expected}, got {result[i]}");
            }
        });

        /// <summary>
        /// Verify MemoryBuffer.Buffer property is publicly accessible and returns
        /// the correct backend-specific buffer type.
        /// </summary>
        [TestMethod]
        public async Task Buffer_Property_IsAccessibleTest() => await RunTest(async accelerator =>
        {
            using var buffer = accelerator.Allocate1D<float>(256);

            // This would have failed when Buffer was protected
            var memBuffer = buffer.Buffer;
            if (memBuffer == null)
                throw new Exception("Buffer property returned null");

            // On browser backends, should be a WebGPUMemoryBuffer, WebGLMemoryBuffer, or WasmMemoryBuffer
            if (accelerator.AcceleratorType == AcceleratorType.WebGPU)
            {
                if (memBuffer is not WebGPUMemoryBuffer webGpuMem)
                    throw new Exception($"Expected WebGPUMemoryBuffer, got {memBuffer.GetType().Name}");
                if (webGpuMem.NativeBuffer == null)
                    throw new Exception("WebGPUMemoryBuffer.NativeBuffer is null");
                if (webGpuMem.NativeBuffer.NativeBuffer == null)
                    throw new Exception("WebGPUBuffer.NativeBuffer (GPUBuffer) is null");
            }

            await Task.CompletedTask;
        });

        // ==================== Sub-Word Buffer Access Tests ====================

        static void Int16DoubleKernel(Index1D idx, ArrayView<short> src, ArrayView<int> dst)
        {
            dst[idx] = (int)src[idx] * 2;
        }

        /// <summary>
        /// Basic round-trip: CopyFromCPU short[] -> CopyToHostAsync short[] (no kernel).
        /// Verifies buffer I/O works for short type.
        /// </summary>
        [TestMethod]
        public async Task Int16_RoundTrip_NoKernel_Test() => await RunTest(async accelerator =>
        {
            var testData = new short[] { 1, -2, 300, -400, 32767, -32768, 0, 42 };
            using var buffer = accelerator.Allocate1D<short>(testData.Length);
            buffer.CopyFromCPU(testData);
            await accelerator.SynchronizeAsync();
            var result = await buffer.CopyToHostAsync();
            for (int i = 0; i < testData.Length; i++)
            {
                if (result[i] != testData[i])
                    throw new Exception($"Short round-trip mismatch at [{i}]: expected {testData[i]}, got {result[i]}");
            }
        });

        static void Int16WriteKernel(Index1D idx, ArrayView<short> dst, ArrayView<int> src)
        {
            dst[idx] = (short)src[idx];
        }

        /// <summary>
        /// Verify ArrayView&lt;short&gt; buffer read works correctly on all backends.
        /// Writes short[] via CopyFromCPU, kernel reads as short, outputs to int buffer.
        /// </summary>
        [TestMethod]
        public async Task Int16_BufferRead_Test() => await RunTest(async accelerator =>
        {
            var testData = new short[] { 1, -2, 300, -400, 32767, -32768, 0, 42 };
            using var input = accelerator.Allocate1D<short>(testData.Length);
            using var output = accelerator.Allocate1D<int>(testData.Length);

            input.CopyFromCPU(testData);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<short>, ArrayView<int>>(Int16DoubleKernel);
            kernel(testData.Length, input.View, output.View);
            await accelerator.SynchronizeAsync();

            var result = await output.CopyToHostAsync();
            for (int i = 0; i < testData.Length; i++)
            {
                int expected = (int)testData[i] * 2;
                if (result[i] != expected)
                    throw new Exception($"Int16 read mismatch at [{i}]: expected {expected}, got {result[i]}. Input was {testData[i]}");
            }
        });

        /// <summary>
        /// Verify ArrayView&lt;short&gt; buffer write works correctly.
        /// Kernel writes short values to a short buffer, read back and verify.
        /// </summary>
        [TestMethod]
        public async Task Int16_BufferWrite_Test() => await RunTest(async accelerator =>
        {
            var srcData = new int[] { 1, -2, 300, -400, 32767, -32768, 0, 42 };
            using var src = accelerator.Allocate1D<int>(srcData.Length);
            using var dst = accelerator.Allocate1D<short>(srcData.Length);

            src.CopyFromCPU(srcData);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<short>, ArrayView<int>>(Int16WriteKernel);
            kernel(srcData.Length, dst.View, src.View);
            await accelerator.SynchronizeAsync();

            var result = await dst.CopyToHostAsync();
            for (int i = 0; i < srcData.Length; i++)
            {
                short expected = (short)srcData[i];
                if (result[i] != expected)
                    throw new Exception($"Int16 write mismatch at [{i}]: expected {expected}, got {result[i]}");
            }
        });

        /// <summary>
        /// Verify ArrayView&lt;Half&gt; buffer read works correctly.
        /// Writes Half[] via CopyFromCPU, kernel reads as Half, converts to float.
        /// </summary>
        // Named static method avoids lambda issues with ILGPU Frontend's Half detection
        static void Float16ReadKernel(Index1D idx, ArrayView<global::ILGPU.Half> src, ArrayView<float> dst)
        {
            dst[idx] = (float)src[idx]; // implicit Half -> float via IR Convert node
        }

        [TestMethod]
        public async Task Float16_BufferRead_Test() => await RunTest(async accelerator =>
        {
            var testFloats = new float[] { 1.0f, -2.5f, 0.0f, 3.14f, 100.0f, -0.5f, 0.001f, 65504.0f };
            var testData = testFloats.Select(f => (global::ILGPU.Half)f).ToArray();
            using var input = accelerator.Allocate1D<global::ILGPU.Half>(testData.Length);
            using var output = accelerator.Allocate1D<float>(testData.Length);

            input.CopyFromCPU(testData);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<global::ILGPU.Half>, ArrayView<float>>(Float16ReadKernel);
            kernel(testData.Length, input.View, output.View);
            await accelerator.SynchronizeAsync();

            var result = await output.CopyToHostAsync();
            for (int i = 0; i < testData.Length; i++)
            {
                float expected = (float)testData[i];
                if (MathF.Abs(result[i] - expected) > 0.01f)
                    throw new Exception($"Float16 read mismatch at [{i}]: expected {expected}, got {result[i]}");
            }
        });

        static void Int16EndToEndKernel(Index1D idx, ArrayView<short> input, ArrayView<short> output)
        {
            output[idx] = (short)(input[idx] + (short)10);
        }

        /// <summary>
        /// End-to-end: write short data, kernel reads AND writes short buffers.
        /// Tests both sub-word Load and sub-word Store in one kernel.
        /// </summary>
        [TestMethod]
        public async Task Int16_EndToEnd_ReadWrite_Test() => await RunTest(async accelerator =>
        {
            var testData = new short[] { 0, 1, -1, 100, -100, 32757, -32758, 42 };
            using var input = accelerator.Allocate1D<short>(testData.Length);
            using var output = accelerator.Allocate1D<short>(testData.Length);

            input.CopyFromCPU(testData);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<short>, ArrayView<short>>(Int16EndToEndKernel);
            kernel(testData.Length, input.View, output.View);
            await accelerator.SynchronizeAsync();

            var result = await output.CopyToHostAsync();
            for (int i = 0; i < testData.Length; i++)
            {
                short expected = (short)(testData[i] + 10);
                if (result[i] != expected)
                    throw new Exception($"Int16 end-to-end mismatch at [{i}]: expected {expected}, got {result[i]}. Input was {testData[i]}");
            }
        });
    }
}
