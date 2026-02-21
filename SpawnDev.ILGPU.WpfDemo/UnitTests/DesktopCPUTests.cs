using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using SpawnDev.ILGPU.Demo.Shared.UnitTests;

namespace SpawnDev.ILGPU.WpfDemo.UnitTests
{
    /// <summary>
    /// Desktop CPU backend tests. Uses the same CPUDevice configuration
    /// as the Blazor demo (4 threads/warp × 16 warps = 64 MaxThreadsPerGroup)
    /// to match shared memory and barrier test expectations.
    /// </summary>
    public class DesktopCPUTests : BackendTestBase
    {
        protected override string BackendName => "CPU";

        protected override Task<(Context context, Accelerator accelerator)> CreateAcceleratorAsync()
        {
            // Match the Blazor demo's CPU configuration exactly
            var device = new CPUDevice(
                numThreadsPerWarp: 4,
                numWarpsPerMultiprocessor: 16,
                numMultiprocessors: 1);
            var builder = Context.Create().CPU(device);
            var context = builder.ToContext();
            var accelerator = context.CreateCPUAccelerator(0);
            return Task.FromResult<(Context, Accelerator)>((context, accelerator));
        }
    }
}
