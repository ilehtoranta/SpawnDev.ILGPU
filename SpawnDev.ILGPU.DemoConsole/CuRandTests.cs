using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using SpawnDev.UnitTesting;
using System;
using System.Linq;
using System.Threading.Tasks;

/// <summary>
/// cuRand tests covering new distributions, configuration, and quasi-random generators. CUDA-only.
/// </summary>
public class CuRandTests : IDisposable
{
    private Context? _context;
    private CudaAccelerator? _accelerator;

    private async Task EnsureInitialized()
    {
        if (_accelerator != null) return;
        _context = Context.Create(builder => builder.AllAccelerators().EnableAlgorithms());
        var cudaDevices = _context.GetCudaDevices();
        if (cudaDevices.Count == 0)
        {
            _context.Dispose();
            _context = null;
            throw new UnsupportedTestException("No CUDA devices found");
        }
        _accelerator = (CudaAccelerator)cudaDevices[0].CreateAccelerator(_context);
        await Task.CompletedTask;
    }

    public void Dispose()
    {
        _accelerator?.Dispose();
        _context?.Dispose();
    }

    [TestMethod]
    public async Task CuRandUniformFloatTest()
    {
        await EnsureInitialized();
        using var rand = CuRand.CreateGPU(_accelerator!, CuRandRngType.CURAND_RNG_PSEUDO_DEFAULT);
        using var buf = _accelerator!.Allocate1D<float>(1024);

        rand.FillUniform(_accelerator.DefaultStream, buf.View);
        _accelerator.Synchronize();

        var result = buf.GetAsArray1D();
        // All values should be in [0, 1)
        for (int i = 0; i < result.Length; i++)
            if (result[i] < 0f || result[i] >= 1f)
                throw new Exception($"Uniform float out of range at {i}: {result[i]}");
    }

    [TestMethod]
    public async Task CuRandNormalFloatTest()
    {
        await EnsureInitialized();
        using var rand = CuRand.CreateGPU(_accelerator!, CuRandRngType.CURAND_RNG_PSEUDO_DEFAULT);
        using var buf = _accelerator!.Allocate1D<float>(4096);

        rand.FillNormal(_accelerator.DefaultStream, buf.View, 0.0f, 1.0f);
        _accelerator.Synchronize();

        var result = buf.GetAsArray1D();
        double mean = result.Select(x => (double)x).Average();
        // Mean of normal(0,1) with 4096 samples should be close to 0
        if (Math.Abs(mean) > 0.2)
            throw new Exception($"Normal distribution mean too far from 0: {mean:F4}");
    }

    [TestMethod]
    public async Task CuRandLogNormalFloatTest()
    {
        await EnsureInitialized();
        using var rand = CuRand.CreateGPU(_accelerator!, CuRandRngType.CURAND_RNG_PSEUDO_DEFAULT);
        using var buf = _accelerator!.Allocate1D<float>(4096);

        rand.FillLogNormal(_accelerator.DefaultStream, buf.View, 0.0f, 1.0f);
        _accelerator.Synchronize();

        var result = buf.GetAsArray1D();
        // Log-normal values should all be positive
        for (int i = 0; i < result.Length; i++)
            if (result[i] <= 0f)
                throw new Exception($"Log-normal value not positive at {i}: {result[i]}");
    }

    [TestMethod]
    public async Task CuRandLogNormalDoubleTest()
    {
        await EnsureInitialized();
        using var rand = CuRand.CreateGPU(_accelerator!, CuRandRngType.CURAND_RNG_PSEUDO_DEFAULT);
        using var buf = _accelerator!.Allocate1D<double>(4096);

        rand.FillLogNormal(_accelerator.DefaultStream, buf.View, 0.0, 1.0);
        _accelerator.Synchronize();

        var result = buf.GetAsArray1D();
        for (int i = 0; i < result.Length; i++)
            if (result[i] <= 0.0)
                throw new Exception($"Log-normal double value not positive at {i}: {result[i]}");
    }

    [TestMethod]
    public async Task CuRandPoissonTest()
    {
        await EnsureInitialized();
        using var rand = CuRand.CreateGPU(_accelerator!, CuRandRngType.CURAND_RNG_PSEUDO_DEFAULT);
        using var buf = _accelerator!.Allocate1D<uint>(4096);

        double lambda = 5.0;
        rand.FillPoisson(_accelerator.DefaultStream, buf.View, lambda);
        _accelerator.Synchronize();

        var result = buf.GetAsArray1D();
        double mean = result.Select(x => (double)x).Average();
        // Poisson mean should be close to lambda
        if (Math.Abs(mean - lambda) > 1.0)
            throw new Exception($"Poisson distribution mean too far from lambda={lambda}: {mean:F4}");
    }

    [TestMethod]
    public async Task CuRandSetOffsetTest()
    {
        await EnsureInitialized();
        using var rand = CuRand.CreateGPU(_accelerator!, CuRandRngType.CURAND_RNG_PSEUDO_DEFAULT);
        rand.SetSeed(42L);
        using var buf1 = _accelerator!.Allocate1D<float>(256);
        using var buf2 = _accelerator.Allocate1D<float>(256);

        // Generate first sequence
        rand.SetOffset(0);
        rand.FillUniform(_accelerator.DefaultStream, buf1.View);
        _accelerator.Synchronize();

        // Reset to same offset — should reproduce
        rand.SetSeed(42L);
        rand.SetOffset(0);
        rand.FillUniform(_accelerator.DefaultStream, buf2.View);
        _accelerator.Synchronize();

        var r1 = buf1.GetAsArray1D();
        var r2 = buf2.GetAsArray1D();
        for (int i = 0; i < r1.Length; i++)
            if (r1[i] != r2[i])
                throw new Exception($"Offset reproducibility failed at {i}: {r1[i]} != {r2[i]}");
    }

    [TestMethod]
    public async Task CuRandSetOrderingTest()
    {
        await EnsureInitialized();
        using var rand = CuRand.CreateGPU(_accelerator!, CuRandRngType.CURAND_RNG_PSEUDO_DEFAULT);
        // Setting ordering should not throw
        rand.SetOrdering(CuRandOrdering.CURAND_ORDERING_PSEUDO_DEFAULT);

        using var buf = _accelerator!.Allocate1D<float>(256);
        rand.FillUniform(_accelerator.DefaultStream, buf.View);
        _accelerator.Synchronize();

        var result = buf.GetAsArray1D();
        if (result.Length != 256)
            throw new Exception($"Expected 256 values, got {result.Length}");
    }

    [TestMethod]
    public async Task CuRandQuasiRandomTest()
    {
        await EnsureInitialized();
        using var rand = CuRand.CreateGPU(_accelerator!, CuRandRngType.CURAND_RNG_QUASI_SOBOL32);
        rand.SetQuasiRandomDimensions(3);

        using var buf = _accelerator!.Allocate1D<float>(3072);
        rand.FillUniform(_accelerator.DefaultStream, buf.View);
        _accelerator.Synchronize();

        var result = buf.GetAsArray1D();
        // Sobol quasi-random should produce values in [0, 1)
        for (int i = 0; i < result.Length; i++)
            if (result[i] < 0f || result[i] >= 1f)
                throw new Exception($"Quasi-random value out of range at {i}: {result[i]}");
    }

    [TestMethod]
    public async Task CuRandCPULogNormalTest()
    {
        await EnsureInitialized();
        using var rand = CuRand.CreateCPU(_context!, CuRandRngType.CURAND_RNG_PSEUDO_DEFAULT);

        var span = new float[2048];
        rand.FillLogNormal(span.AsSpan(), 0.0f, 1.0f);

        for (int i = 0; i < span.Length; i++)
            if (span[i] <= 0f)
                throw new Exception($"CPU log-normal value not positive at {i}: {span[i]}");
    }

    [TestMethod]
    public async Task CuRandCPUPoissonTest()
    {
        await EnsureInitialized();
        using var rand = CuRand.CreateCPU(_context!, CuRandRngType.CURAND_RNG_PSEUDO_DEFAULT);

        var span = new uint[2048];
        double lambda = 3.0;
        rand.FillPoisson(span.AsSpan(), lambda);

        double mean = span.Select(x => (double)x).Average();
        if (Math.Abs(mean - lambda) > 1.0)
            throw new Exception($"CPU Poisson mean too far from lambda={lambda}: {mean:F4}");

        await Task.CompletedTask;
    }
}
