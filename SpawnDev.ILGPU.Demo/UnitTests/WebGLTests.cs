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
