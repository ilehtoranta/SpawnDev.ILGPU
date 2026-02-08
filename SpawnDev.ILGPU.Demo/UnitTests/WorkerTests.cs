using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Blazor.UnitTesting;
using SpawnDev.ILGPU.Workers;

namespace SpawnDev.ILGPU.Demo.UnitTests
{
    /// <summary>
    /// Workers backend tests. Inherits all shared tests from BackendTestBase.
    /// </summary>
    public class WorkerTests : BackendTestBase
    {
        protected override string BackendName => "Workers";

        protected override Task<(Context context, Accelerator accelerator)> CreateAcceleratorAsync()
        {
            var builder = Context.Create().Workers();
            var context = builder.ToContext();
            var accelerator = context.CreateWorkersAccelerator();
            return Task.FromResult<(Context, Accelerator)>((context, accelerator));
        }

        // Subgroup tests require hardware subgroup/warp operations
        // which are not available in the Workers backend
        [TestMethod]
        public new async Task BroadcastTest() => await RunTest(async accelerator =>
        {
            int len = 64;
            var data = new int[len];
            for (int i = 0; i < len; i++) data[i] = i + 10; // [10, 11, 12, ..., 73]
            using var buf = accelerator.Allocate1D(data);

            // Load stream kernel (explicit KernelConfig) and launch with 1 grid, 64-thread group
            var kernel = accelerator.LoadStreamKernel<Index1D, ArrayView<int>>(BroadcastKernel);
            kernel(new KernelConfig(1, len), (Index1D)len, buf.View);
            await accelerator.SynchronizeAsync();

            var result = await buf.CopyToHostAsync<int>();
            // Group.Broadcast(val, 0) should broadcast thread 0's value (10) to all threads
            for (int i = 0; i < len; i++)
            {
                if (result[i] != 10)
                    throw new Exception($"BroadcastTest failed at [{i}]: Expected 10, got {result[i]}");
            }
        });

        [TestMethod]
        public new async Task SubgroupShuffleTest() =>
            throw new UnsupportedTestException("Workers: subgroups not supported");
    }
}
