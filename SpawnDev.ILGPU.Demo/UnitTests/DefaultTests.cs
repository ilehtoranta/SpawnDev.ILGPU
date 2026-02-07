using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Blazor.UnitTesting;

namespace SpawnDev.ILGPU.Demo.UnitTests
{
    /// <summary>
    /// Tests for device enumeration and preferred (default) accelerator selection.
    /// Validates that AllAcceleratorsAsync registers all WASM backends
    /// and that CreatePreferredAcceleratorAsync picks: WebGPU > Workers > CPU.
    /// </summary>
    public class DefaultTests
    {
        #region Device Enumeration

        /// <summary>
        /// Verifies that AllAcceleratorsAsync registers WebGPU, Workers, and CPU devices.
        /// </summary>
        [TestMethod]
        public async Task DeviceEnumerationTest()
        {
            var builder = Context.Create();
            await builder.AllAcceleratorsAsync();
            using var context = builder.ToContext();

            var devices = context.GetAllDeviceInfo();
            Console.WriteLine($"Registered devices ({devices.Count}):");
            foreach (var (name, type) in devices)
            {
                Console.WriteLine($"  [{type}] {name}");
            }

            // Should have at least CPU
            if (devices.Count == 0)
                throw new Exception("No devices registered");

            // Verify we have Workers
            bool hasWorkers = devices.Any(d => d.Type == AcceleratorType.Workers);
            Console.WriteLine($"Workers available: {hasWorkers}");

            // Verify we have WebGPU (expected in Chrome/Edge)
            bool hasWebGPU = devices.Any(d => d.Type == AcceleratorType.WebGPU);
            Console.WriteLine($"WebGPU available: {hasWebGPU}");

            // Verify we have CPU
            bool hasCPU = devices.Any(d => d.Type == AcceleratorType.CPU);
            Console.WriteLine($"CPU available: {hasCPU}");
        }

        #endregion

        #region Preferred Accelerator

        /// <summary>
        /// Verifies that CreatePreferredAcceleratorAsync selects the best backend.
        /// Expected priority: WebGPU > Workers > CPU.
        /// </summary>
        [TestMethod]
        public async Task PreferredAcceleratorTest()
        {
            var builder = Context.Create();
            await builder.AllAcceleratorsAsync();
            using var context = builder.ToContext();

            using var accelerator = await context.CreatePreferredAcceleratorAsync();

            Console.WriteLine($"Preferred accelerator: {accelerator.AcceleratorType} - {accelerator.Name}");

            // Should have selected *something*
            if (accelerator == null)
                throw new Exception("CreatePreferredAcceleratorAsync returned null");

            Console.WriteLine($"AcceleratorType: {accelerator.AcceleratorType}");
            Console.WriteLine($"Name: {accelerator.Name}");
            Console.WriteLine($"MemorySize: {accelerator.MemorySize}");
            Console.WriteLine($"MaxNumThreadsPerGroup: {accelerator.MaxNumThreadsPerGroup}");
        }

        #endregion

        #region Default Kernel Execution

        static void AddKernel(Index1D index, ArrayView<int> output, ArrayView<int> a, ArrayView<int> b)
        {
            output[index] = a[index] + b[index];
        }

        /// <summary>
        /// Runs a simple add kernel on the preferred (default) accelerator.
        /// This validates the full pipeline: discovery → creation → kernel execution.
        /// </summary>
        [TestMethod]
        public async Task DefaultKernelExecutionTest()
        {
            var builder = Context.Create();
            await builder.AllAcceleratorsAsync();
            using var context = builder.ToContext();

            using var accelerator = await context.CreatePreferredAcceleratorAsync();
            Console.WriteLine($"Running kernel on: {accelerator.AcceleratorType} - {accelerator.Name}");

            const int length = 256;
            using var bufA = accelerator.Allocate1D<int>(length);
            using var bufB = accelerator.Allocate1D<int>(length);
            using var bufOut = accelerator.Allocate1D<int>(length);

            // Initialize
            var dataA = Enumerable.Range(0, length).ToArray();
            var dataB = Enumerable.Range(100, length).ToArray();
            bufA.View.CopyFromCPU(dataA);
            bufB.View.CopyFromCPU(dataB);

            // Load and run kernel
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>(AddKernel);
            kernel((int)bufOut.Length, bufOut.View, bufA.View, bufB.View);

            await accelerator.SynchronizeAsync();

            // Read results
            var result = await bufOut.CopyToHostAsync<int>();

            // Verify
            for (int i = 0; i < length; i++)
            {
                int expected = dataA[i] + dataB[i];
                if (result[i] != expected)
                    throw new Exception($"Mismatch at index {i}: expected {expected}, got {result[i]}");
            }

            Console.WriteLine($"Default kernel execution: All {length} results correct on {accelerator.AcceleratorType}");
        }

        #endregion
    }
}
