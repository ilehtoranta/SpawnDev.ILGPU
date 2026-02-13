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

    }
}
