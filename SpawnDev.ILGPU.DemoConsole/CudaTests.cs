using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using SpawnDev.BlazorJS.Cryptography;
using SpawnDev.UnitTesting;
using SpawnDev.ILGPU.Demo.Shared.UnitTests;

public class CudaTests : BackendTestBase
{
    public CudaTests(IPortableCrypto crypto, SpawnDev.WebTorrent.WebTorrentClient webTorrentClient, Func<SpawnDev.WebTorrent.Ed25519Signer> signerFactory) : base(crypto, webTorrentClient, signerFactory) { }
    protected override string BackendName => "CUDA";

    protected override Task<(Context context, Accelerator accelerator)> CreateAcceleratorAsync()
    {
        var context = Context.Create(builder => builder.AllAccelerators().EnableAlgorithms());
        var cudaDevices = context.GetCudaDevices();
        if (cudaDevices.Count == 0)
        {
            context.Dispose();
            throw new UnsupportedTestException("No CUDA devices found");
        }
        var accelerator = cudaDevices[0].CreateAccelerator(context);
        return Task.FromResult<(Context, Accelerator)>((context, accelerator));
    }

}
