using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Blazor.UnitTesting;
using SpawnDev.ILGPU.WebGL;
using SpawnDev.ILGPU.WebGL.Backend;

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

        protected override async Task<(Context context, Accelerator accelerator)> CreateEmulatedAcceleratorAsync()
        {
            var builder = Context.Create();
            await builder.WebGL();
            var context = builder.ToContext();
            var devices = context.GetWebGLDevices();
            if (devices.Count == 0)
                throw new UnsupportedTestException("No WebGL2 devices found");
            var accelerator = devices[0].CreateAccelerator(context, new WebGLBackendOptions
            {
                EnableF64Emulation = true,
                EnableI64Emulation = true
            });
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

        // LargeDispatchTest: re-enabled — 2D texture tiling now supports buffers > MAX_TEXTURE_SIZE

        // ArrayView2D/3D tests: re-enabled — stride uniform passing now implemented in WebGLAccelerator

        // Struct buffer tests: re-enabled — per-field texelFetch load and per-field TF output now implemented

        // NestedLoopBreakTest: re-enabled — structured control flow (while/break/continue) now handles this correctly

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
