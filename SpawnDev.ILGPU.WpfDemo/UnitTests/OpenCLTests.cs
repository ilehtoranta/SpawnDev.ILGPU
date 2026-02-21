using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.OpenCL;
using SpawnDev.UnitTesting;
using SpawnDev.ILGPU.Demo.Shared.UnitTests;

namespace SpawnDev.ILGPU.WpfDemo.UnitTests
{
    /// <summary>
    /// OpenCL backend tests. Inherits all shared tests from BackendTestBase.
    /// Uses the same AllAccelerators() pattern as the WPF demo pages.
    /// </summary>
    public class OpenCLTests : BackendTestBase
    {
        protected override string BackendName => "OpenCL";

        protected override Task<(Context context, Accelerator accelerator)> CreateAcceleratorAsync()
        {
            var context = Context.Create(builder => builder.AllAccelerators());
            var clDevices = context.GetCLDevices();
            if (clDevices.Count == 0)
            {
                context.Dispose();
                throw new UnsupportedTestException("No OpenCL devices found");
            }
            var accelerator = clDevices[0].CreateAccelerator(context);
            return Task.FromResult<(Context, Accelerator)>((context, accelerator));
        }
    }
}
