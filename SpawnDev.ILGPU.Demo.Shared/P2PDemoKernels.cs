using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.Demo.Shared;

/// <summary>
/// Demo kernels for P2P distributed compute demonstrations.
/// These kernels are registered with P2PKernelSerializer so they can be
/// dispatched to remote peers via the P2P accelerator.
/// </summary>
public static class P2PDemoKernels
{
    /// <summary>
    /// Simple multiply-by-2 — demonstrates basic distributed element-wise compute.
    /// </summary>
    public static void MultiplyBy2(Index1D idx, ArrayView1D<float, Stride1D.Dense> data)
    {
        data[idx] = data[idx] * 2.0f;
    }

    /// <summary>
    /// Vector addition — demonstrates multi-buffer distributed compute.
    /// </summary>
    public static void VectorAdd(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> a,
        ArrayView1D<float, Stride1D.Dense> b,
        ArrayView1D<float, Stride1D.Dense> result)
    {
        result[idx] = a[idx] + b[idx];
    }

    /// <summary>
    /// Mandelbrot iteration count — compute-heavy, great for benchmarking.
    /// Each element represents a pixel; the value is the iteration count (0-255).
    /// </summary>
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

    /// <summary>
    /// Mandelbrot writing packed RGBA <see cref="uint"/> pixels for direct
    /// presentation via SpawnDev.ILGPU's <c>CanvasRendererFactory</c>. Peer tinting
    /// is baked into the kernel — the worker delivers display-ready pixel data so
    /// the coordinator just stitches strips into the render buffer and presents
    /// (no coord-side CPU pixel-conversion loop).
    /// </summary>
    /// <remarks>
    /// Pixel layout per <c>SpawnDev.ILGPU/Docs/canvas-rendering.md</c>: little-endian
    /// RGBA — R in byte 0, G in byte 1, B in byte 2, A in byte 3. The kernel writes
    /// <c>(0xFFu &lt;&lt; 24) | (b &lt;&lt; 16) | (g &lt;&lt; 8) | r</c>.
    /// In-set pixels (iter == maxIter) are opaque black 0xFF000000. Out-of-set
    /// pixels scale the peer's tint by iter/maxIter.
    ///
    /// Tinting parameters (<paramref name="tintR"/>, <paramref name="tintG"/>,
    /// <paramref name="tintB"/>) come from the consumer (coord) per peer; all three
    /// are 0-255. The coord chooses which peer gets which tint when dispatching.
    /// </remarks>
    public static void MandelbrotChunkPacked(Index1D idx,
        ArrayView1D<uint, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> realCoords,
        ArrayView1D<float, Stride1D.Dense> imagCoords,
        int tintR, int tintG, int tintB, int maxIter)
    {
        float cr = realCoords[idx];
        float ci = imagCoords[idx];
        float zr = 0f, zi = 0f;
        int iter = 0;

        while (zr * zr + zi * zi <= 4.0f && iter < maxIter)
        {
            float tmp = zr * zr - zi * zi + cr;
            zi = 2.0f * zr * zi + ci;
            zr = tmp;
            iter++;
        }

        uint color;
        if (iter >= maxIter)
        {
            color = 0xFF000000u; // opaque black for in-set pixels
        }
        else
        {
            float t = (float)iter / (float)maxIter;
            uint r = (uint)((int)(tintR * t)) & 0xFFu;
            uint g = (uint)((int)(tintG * t)) & 0xFFu;
            uint b = (uint)((int)(tintB * t)) & 0xFFu;
            color = (0xFFu << 24) | (b << 16) | (g << 8) | r;
        }
        output[idx] = color;
    }

    /// <summary>
    /// N-body force computation — O(n²) gravity sim, the classic cluster demo.
    /// Computes forces on particles[startIdx..startIdx+count] from ALL particles.
    /// </summary>
    public static void NBodyForces(Index1D idx,
        ArrayView1D<float, Stride1D.Dense> posX,
        ArrayView1D<float, Stride1D.Dense> posY,
        ArrayView1D<float, Stride1D.Dense> forceX,
        ArrayView1D<float, Stride1D.Dense> forceY)
    {
        int n = (int)posX.Length;
        float fx = 0f, fy = 0f;
        float px = posX[idx], py = posY[idx];
        const float softening = 0.01f;
        const float G = 1.0f;

        for (int j = 0; j < n; j++)
        {
            if (j == idx) continue;
            float dx = posX[j] - px;
            float dy = posY[j] - py;
            float distSq = dx * dx + dy * dy + softening;
            float invDist = 1.0f / MathF.Sqrt(distSq);
            float invDist3 = invDist * invDist * invDist;
            fx += G * dx * invDist3;
            fy += G * dy * invDist3;
        }
        forceX[idx] = fx;
        forceY[idx] = fy;
    }
}
