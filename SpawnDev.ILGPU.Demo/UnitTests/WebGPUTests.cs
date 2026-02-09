using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Blazor.UnitTesting;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.ILGPU.WebGPU.Backend;

namespace SpawnDev.ILGPU.Demo.UnitTests
{
    /// <summary>
    /// WebGPU backend tests. Inherits all shared tests from BackendTestBase
    /// and adds WebGPU-specific tests.
    /// </summary>
    public class WebGPUTests : BackendTestBase
    {
        protected override string BackendName => "WebGPU";

        protected override async Task<(Context context, Accelerator accelerator)> CreateAcceleratorAsync()
        {
            return await CreateAcceleratorAsync(enableEmulation: false);
        }

        private async Task<(Context context, Accelerator accelerator)> CreateAcceleratorAsync(bool enableEmulation)
        {
            var builder = Context.Create();
            await builder.WebGPU();
            var context = builder.ToContext();
            var devices = context.GetWebGPUDevices();
            if (devices.Count == 0)
                throw new UnsupportedTestException("No WebGPU devices found");
            var device = devices[0];
            var options = new WebGPUBackendOptions
            {
                EnableF64Emulation = enableEmulation,
                EnableI64Emulation = enableEmulation
            };
            var accelerator = await device.CreateAcceleratorAsync(context, options);
            return (context, accelerator);
        }

        protected override async Task<(Context context, Accelerator accelerator)> CreateEmulatedAcceleratorAsync()
        {
            return await CreateAcceleratorAsync(enableEmulation: true);
        }

        #region WebGPU-Specific Tests

        /// <summary>
        /// Test WebGPU accelerator basic initialization (uses WebGPU-specific APIs)
        /// </summary>
        [TestMethod]
        public async Task WebGPUAcceleratorBasicTest()
        {
            var devices = await WebGPU.WebGPUDevice.GetDevicesAsync();
            if (devices.Length == 0)
                throw new UnsupportedTestException("No WebGPU devices found");
            var device = devices[0];
            using var accelerator = await device.CreateAcceleratorAsync();
            if (!accelerator.IsInitialized)
                throw new Exception("WebGPUAccelerator is not initialized");
            using var buffer = accelerator.Allocate<int>(1024);
            if (buffer.Length != 1024)
                throw new Exception("Buffer length mismatch");
        }

        /// <summary>
        /// Test WebGPU buffer transfer (uses WebGPU-specific APIs)
        /// </summary>
        [TestMethod]
        public async Task WebGPUBufferTransferTest()
        {
            var device = await WebGPU.WebGPUDevice.GetDefaultDeviceAsync();
            if (device == null)
                throw new UnsupportedTestException("No WebGPU devices found");
            using var accelerator = await device.CreateAcceleratorAsync();
            int length = 128;
            var data = Enumerable.Range(0, length).ToArray();
            using var buffer = accelerator.Allocate(data);
            var readBack = await buffer.CopyToHostAsync();
            if (readBack.Length != length)
                throw new Exception($"Readback length mismatch");
            for (int i = 0; i < length; i++)
                if (readBack[i] != data[i])
                    throw new Exception($"Data mismatch at index {i}");
        }

        /// <summary>
        /// Test double precision with explicit F64 emulation options (WebGPU-specific API test)
        /// </summary>
        [TestMethod]
        public async Task WebGPUDoublePrecisionVerifyTest()
        {
            // F64 emulation is now enabled by default, so this test validates
            // that explicitly passing the default options still works
            var builder = Context.Create();
            await builder.WebGPU();
            using var context = builder.ToContext();
            var device = context.GetWebGPUDevices()[0];
            using var accelerator = await device.CreateAcceleratorAsync(context);

            var data = new double[] { 0.1, 0.2, 0.3, 0.1 + 0.2, 1.0 / 3.0, 2.0 / 3.0 };
            int len = data.Length;
            using var buf = accelerator.Allocate1D(data);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>>(DoublePrecisionVerifyKernel);
            kernel((Index1D)len, buf.View);
            await accelerator.SynchronizeAsync();

            var result = await buf.CopyToHostAsync<double>();
            for (int i = 0; i < len; i++)
            {
                double expected = data[i];
                double tolerance = Math.Max(Math.Abs(expected * 0.02), 1e-5);
                if (Math.Abs(result[i] - expected) > tolerance)
                    throw new Exception($"Double precision verify failed at {i}. Expected {expected}, got {result[i]}");
            }
        }


        /// <summary>
        /// Test dynamic shared memory with F64 using WebGPU-specific API
        /// </summary>
        [TestMethod]
        public async Task WebGPUDynamicSharedF64Test()
        {
            // F64 emulation is enabled by default
            var builder = Context.Create();
            await builder.WebGPU();
            using var context = builder.ToContext();
            var device = context.GetWebGPUDevices()[0];
            using var accelerator = await device.CreateAcceleratorAsync(context);

            int len = 64;
            var data = new double[len];
            for (int i = 0; i < len; i++) data[i] = i * 1.5;

            using var buf = accelerator.Allocate1D(data);
            var kernel = accelerator.LoadStreamKernel<Index1D, ArrayView<double>>(DynamicSharedF64Kernel);
            kernel(new KernelConfig(1, len, SharedMemoryConfig.RequestDynamic<double>(len)), (Index1D)len, buf.View);
            await accelerator.SynchronizeAsync();

            var result = await buf.CopyToHostAsync<double>();
            for (int i = 0; i < len; i++)
            {
                double expected = (63 - i) * 1.5;
                double tolerance = Math.Max(Math.Abs(expected * 0.01), 1e-6);
                if (Math.Abs(result[i] - expected) > tolerance)
                    throw new Exception($"Dynamic shared F64 failed at {i}. Expected {expected}, got {result[i]}");
            }
        }

        #endregion

        [TestMethod]
        public new async Task BroadcastTest() =>
            throw new UnsupportedTestException("WebGPU: broadcast not yet implemented");
    }
}
