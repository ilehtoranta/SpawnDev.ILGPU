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
    }
}
