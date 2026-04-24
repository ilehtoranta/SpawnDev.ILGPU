using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.P2P.IntegrationTests;

/// <summary>
/// Byte-identical mirror of <c>SpawnDev.ILGPU.Demo.Shared.P2PDemoKernels</c>
/// so the RealWebRtc demo-path tests can dispatch the exact kernel bodies the
/// <c>/compute</c> demo page uses without dragging Blazor-only projects into
/// the integration-test surface. If either copy changes, bring the other into
/// sync - the signatures (<c>ArrayView1D&lt;T, Stride1D.Dense&gt;</c>) are
/// load-bearing: they exercise a different kernel-serializer path from the
/// <c>ArrayView&lt;T&gt;</c> kernels used by <c>P2PTestKernels</c>.
/// </summary>
public static class P2PDemoKernelsMirror
{
    /// <summary>In-place doubler. Demo benchmark uses this.</summary>
    public static void MultiplyBy2(Index1D idx, ArrayView1D<float, Stride1D.Dense> data)
    {
        data[idx] = data[idx] * 2.0f;
    }

    /// <summary>Mandelbrot iteration counts. Demo /compute Mandelbrot page uses this.</summary>
    public static void MandelbrotChunk(Index1D idx,
        ArrayView1D<int, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> realCoords,
        ArrayView1D<float, Stride1D.Dense> imagCoords)
    {
        float cr = realCoords[idx];
        float ci = imagCoords[idx];
        float zr = 0f, zi = 0f;
        int iter = 0;
        const int maxIter = 255;

        while (zr * zr + zi * zi <= 4.0f && iter < maxIter)
        {
            float tmp = zr * zr - zi * zi + cr;
            zi = 2.0f * zr * zi + ci;
            zr = tmp;
            iter++;
        }
        output[idx] = iter;
    }
}
