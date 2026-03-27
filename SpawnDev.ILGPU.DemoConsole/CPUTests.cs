using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using SpawnDev.ILGPU.Demo.Shared.UnitTests;

public class CPUTests : BackendTestBase
{
    protected override string BackendName => "CPU";

    protected override Task<(Context context, Accelerator accelerator)> CreateAcceleratorAsync()
    {
        // Use the Nvidia preset (warp=32, warps=32) to support group sizes up to 1024.
        // The default preset (warp=4, warps=4) only supports groups of 16.
        var context = Context.Create(builder => builder
            .CPU(CPUDevice.Nvidia));
        var accelerator = context.GetCPUDevice(0).CreateCPUAccelerator(context);
        return Task.FromResult<(Context, Accelerator)>((context, accelerator));
    }

}
