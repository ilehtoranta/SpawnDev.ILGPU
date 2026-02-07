using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using SpawnDev.Blazor.UnitTesting;

namespace SpawnDev.ILGPU.Demo.UnitTests
{
    /// <summary>
    /// CPU backend tests. Inherits all shared tests from BackendTestBase.
    /// Serves as the reference implementation for correctness.
    /// </summary>
    public class CPUTests : BackendTestBase
    {
        protected override string BackendName => "CPU";

        protected override Task<(Context context, Accelerator accelerator)> CreateAcceleratorAsync()
        {
            var builder = Context.Create().CPU();
            var context = builder.ToContext();
            var accelerator = context.CreateCPUAccelerator(0);
            return Task.FromResult<(Context, Accelerator)>((context, accelerator));
        }
    }
}
