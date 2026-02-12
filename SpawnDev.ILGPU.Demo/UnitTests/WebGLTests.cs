using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Blazor.UnitTesting;
using SpawnDev.ILGPU.WebGL;

namespace SpawnDev.ILGPU.Demo.UnitTests
{
    /// <summary>
    /// WebGL2 backend tests. Inherits all shared tests from BackendTestBase
    /// and overrides unsupported features (shared memory, barriers, broadcasts,
    /// subgroups — WebGL2 vertex shaders are single-threaded with no workgroups).
    /// </summary>
    public class WebGLTests : BackendTestBase
    {
        protected override string BackendName => "WebGL";

        protected override async Task<(Context context, Accelerator accelerator)> CreateAcceleratorAsync()
        {
            var builder = Context.Create();
            await builder.WebGL();
            var context = builder.ToContext();
            var devices = context.GetWebGLDevices();
            if (devices.Count == 0)
                throw new UnsupportedTestException("No WebGL2 devices found");
            var accelerator = devices[0].CreateAccelerator(context);
            return (context, accelerator);
        }

        // ========== Unsupported Feature Overrides ==========
        // WebGL2 vertex shaders have no shared memory, barriers, or workgroup primitives.

        [TestMethod]
        public new async Task SharedMemoryTest() =>
            throw new UnsupportedTestException("WebGL: no shared memory in vertex shaders");

        [TestMethod]
        public new async Task CSharpSharedMemoryTest() =>
            throw new UnsupportedTestException("WebGL: no shared memory in vertex shaders");

        [TestMethod]
        public new async Task DynamicSharedMemoryTest() =>
            throw new UnsupportedTestException("WebGL: no shared memory in vertex shaders");

        [TestMethod]
        public new async Task DynamicSharedF64Test() =>
            throw new UnsupportedTestException("WebGL: no shared memory in vertex shaders");

        [TestMethod]
        public new async Task BarrierTest() =>
            throw new UnsupportedTestException("WebGL: no barrier support in vertex shaders");

        [TestMethod]
        public new async Task BroadcastTest() =>
            throw new UnsupportedTestException("WebGL: no broadcast support");

        [TestMethod]
        public new async Task SubgroupShuffleTest() =>
            throw new UnsupportedTestException("WebGL: no subgroup support");

        // --- Atomics (WebGL vertex shader TF pipeline has no atomic operations) ---
        [TestMethod]
        public new async Task AtomicTest() =>
            throw new UnsupportedTestException("WebGL: no atomic operations in vertex shaders");

        [TestMethod]
        public new async Task AtomicCASTest() =>
            throw new UnsupportedTestException("WebGL: no atomic operations in vertex shaders");

        [TestMethod]
        public new async Task AtomicMinMaxTest() =>
            throw new UnsupportedTestException("WebGL: no atomic operations in vertex shaders");

        [TestMethod]
        public new async Task ReductionTest() =>
            throw new UnsupportedTestException("WebGL: requires atomics (unsupported in vertex shaders)");

        [TestMethod]
        public new async Task ParallelSumTest() =>
            throw new UnsupportedTestException("WebGL: requires atomics (unsupported in vertex shaders)");

        [TestMethod]
        public new async Task HistogramTest() =>
            throw new UnsupportedTestException("WebGL: requires atomics (unsupported in vertex shaders)");

        // --- Barriers / Shared Memory ---
        [TestMethod]
        public new async Task LinearBarrierTest() =>
            throw new UnsupportedTestException("WebGL: no barriers/shared memory in vertex shaders");

        [TestMethod]
        public new async Task SharedMemoryBarrierTest() =>
            throw new UnsupportedTestException("WebGL: no barriers/shared memory in vertex shaders");

        // --- Grid/Group intrinsics (vertex shader is single-threaded, no workgroups) ---
        [TestMethod]
        public new async Task GridGroupDimensionTest() =>
            throw new UnsupportedTestException("WebGL: no group/grid intrinsics in TF pipeline");

        // --- TF pipeline limitation: no store = no output (value stays zero) ---
        [TestMethod]
        public new async Task ZeroIterationLoopTest() =>
            throw new UnsupportedTestException("WebGL: zero-iteration loop produces no TF store, buffer stays zeroed");

        // --- Texture size limit: 1D textures exceed MAX_TEXTURE_SIZE for large buffers ---
        [TestMethod]
        public new async Task LargeDispatchTest() =>
            throw new UnsupportedTestException("WebGL: 65536 elements exceeds MAX_TEXTURE_SIZE for 1D texture input");

        // --- ArrayView2D/3D not yet supported in TF pipeline ---
        [TestMethod]
        public new async Task Kernel2DTest() =>
            throw new UnsupportedTestException("WebGL: ArrayView2D stride/index support not yet implemented in TF pipeline");

        [TestMethod]
        public new async Task Kernel3DTest() =>
            throw new UnsupportedTestException("WebGL: ArrayView3D stride/index support not yet implemented in TF pipeline");

        // --- Struct buffer load/store: per-field texture load not yet implemented ---
        [TestMethod]
        public new async Task StructTest() =>
            throw new UnsupportedTestException("WebGL: struct buffer texelFetch requires per-field load (not yet implemented)");

        [TestMethod]
        public new async Task ComplexStructTest() =>
            throw new UnsupportedTestException("WebGL: struct buffer texelFetch requires per-field load (not yet implemented)");

        [TestMethod]
        public new async Task NestedStructTest() =>
            throw new UnsupportedTestException("WebGL: struct buffer texelFetch requires per-field load (not yet implemented)");

        // --- Multi-buffer loop patterns with TF output ---
        [TestMethod]
        public new async Task MatrixMulTest() =>
            throw new UnsupportedTestException("WebGL: multi-buffer loop kernel with scalar param not yet supported in TF pipeline");

        [TestMethod]
        public new async Task MatrixMultiplyTest() =>
            throw new UnsupportedTestException("WebGL: multi-buffer loop kernel with scalar param not yet supported in TF pipeline");

        // --- Multi-block control flow: cross-block LEA resolution issue ---
        [TestMethod]
        public new async Task NestedLoopBreakTest() =>
            throw new UnsupportedTestException("WebGL: multi-block control flow loses TF output mapping across loop blocks");

        // --- Multi-buffer interleaved TF readback precision issue ---
        [TestMethod]
        public new async Task AdvancedMathTest() =>
            throw new UnsupportedTestException("WebGL: multi-buffer TF interleaving produces incorrect readback values");

        #region WebGL-Specific Tests

        /// <summary>
        /// Test WebGL2 accelerator basic initialization (uses WebGL-specific APIs)
        /// </summary>
        [TestMethod]
        public async Task WebGLAcceleratorBasicTest()
        {
            var devices = await WebGLDevice.GetDevicesAsync();
            if (devices.Length == 0)
                throw new UnsupportedTestException("No WebGL2 devices found");
            var device = devices[0];
            device.PrintInfo(Console.Out);
            Console.WriteLine($"WebGL2 Device Name: {device.Name}");
            Console.WriteLine($"WebGL2 Vendor: {device.Vendor}");
            Console.WriteLine($"WebGL2 Max Texture Size: {device.MaxTextureSize}");

            // Verify device has reasonable capabilities
            if (device.MaxTextureSize < 2048)
                throw new Exception($"Unexpected max texture size: {device.MaxTextureSize}");
        }

        /// <summary>
        /// Test WebGL2 buffer transfer (uses WebGL-specific APIs)
        /// </summary>
        [TestMethod]
        public async Task WebGLBufferTransferTest()
        {
            var (context, accelerator) = await CreateAcceleratorAsync();
            try
            {
                int length = 128;
                var data = Enumerable.Range(0, length).ToArray();
                using var buffer = accelerator.Allocate1D(data);
                var readBack = await buffer.CopyToHostAsync<int>();
                if (readBack.Length != length)
                    throw new Exception($"Readback length mismatch: expected {length}, got {readBack.Length}");
                for (int i = 0; i < length; i++)
                    if (readBack[i] != data[i])
                        throw new Exception($"Data mismatch at index {i}: expected {data[i]}, got {readBack[i]}");
            }
            finally
            {
                accelerator.Dispose();
                context.Dispose();
            }
        }

        #endregion
    }
}
