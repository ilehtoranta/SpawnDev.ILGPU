// ---------------------------------------------------------------------------------------
//                        SpawnDev.ILGPU.Demo.Shared
//             Shared ILGPU Kernels for Blazor WASM & WPF Demos
//
// File: FractalKernels.cs
// ---------------------------------------------------------------------------------------

using global::ILGPU;
using global::ILGPU.Runtime;

namespace SpawnDev.ILGPU.Demo.Shared.Kernels
{
    /// <summary>
    /// Pure ILGPU fractal kernels — backend-agnostic.
    /// Works on CUDA, OpenCL, CPU, WebGPU, WebGL, and Wasm.
    /// </summary>
    public static class FractalKernels
    {
        /// <summary>
        /// Supported fractal types.
        /// </summary>
        public enum FractalType
        {
            Mandelbrot = 0,
            Julia = 1,
            BurningShip = 2,
            Tricorn = 3,
            Phoenix = 4,
        }

        /// <summary>
        /// Available color schemes.
        /// </summary>
        public enum ColorScheme
        {
            Classic = 0,
            Smooth = 1,
            Fire = 2,
            Ocean = 3,
            Neon = 4,
        }

        /// <summary>
        /// Multi-fractal kernel supporting 5 fractal types and 5 color schemes.
        /// Output is packed ABGR (0xAA_BB_GG_RR) for direct WriteableBitmap / ImageData use.
        /// 
        /// Parameters are packed to minimize binding count for WebGPU/WebGL compatibility:
        ///   packedSize = width * 65536 + height
        ///   packedConfig = fractalType * 256 + colorScheme
        ///   extra1 = juliaReal (Julia) or phoenixP (Phoenix)
        ///   extra2 = juliaImag (Julia) or phoenixQ (Phoenix)
        /// </summary>
        public static void Render(
            Index2D index,
            ArrayView2D<uint, Stride2D.DenseX> output,
            int packedSize, int maxIter, int packedConfig,
            double centerX, double centerY, double zoom, double extra1, double extra2)
        {
            int width = packedSize / 65536;
            int height = packedSize - width * 65536;
            int fractalType = packedConfig / 256;
            int colorScheme = packedConfig - fractalType * 256;
            double juliaReal = extra1;
            double juliaImag = extra2;
            double paramP = extra1;
            double paramQ = extra2;

            int x = index.X;
            int y = index.Y;

            if (x >= width || y >= height)
                return;

            double scale = 4.0 / zoom;
            double real = centerX + (x - width * 0.5) * scale / width;
            double imag = centerY + (y - height * 0.5) * scale / height;

            double zr = 0.0;
            double zi = 0.0;
            double cr = real;
            double ci = imag;
            double prevZr = 0.0;
            double prevZi = 0.0;
            int iterations = 0;

            if (fractalType == 1)
            {
                zr = real;
                zi = imag;
                cr = juliaReal;
                ci = juliaImag;
            }
            else if (fractalType == 4)
            {
                zr = real;
                zi = imag;
            }

            while (iterations < maxIter)
            {
                double zr2 = zr * zr;
                double zi2 = zi * zi;
                if (zr2 + zi2 >= 4.0)
                    break;

                double newZr;
                double newZi;

                if (fractalType == 2)
                {
                    double azr = zr;
                    double azi = zi;
                    if (azr < 0.0) azr = -azr;
                    if (azi < 0.0) azi = -azi;
                    newZr = azr * azr - azi * azi + cr;
                    newZi = 2.0 * azr * azi + ci;
                }
                else if (fractalType == 3)
                {
                    newZr = zr2 - zi2 + cr;
                    newZi = -2.0 * zr * zi + ci;
                }
                else if (fractalType == 4)
                {
                    newZr = zr2 - zi2 + paramP + paramQ * prevZr;
                    newZi = 2.0 * zr * zi + paramQ * prevZi;
                    prevZr = zr;
                    prevZi = zi;
                }
                else
                {
                    newZr = zr2 - zi2 + cr;
                    newZi = 2.0 * zr * zi + ci;
                }

                zr = newZr;
                zi = newZi;
                iterations++;
            }

            // Coloring
            uint color;
            if (iterations >= maxIter)
            {
                color = 0xFF000000u;
            }
            else
            {
                double t = (double)iterations / (double)maxIter;
                uint rv;
                uint gv;
                uint bv;

                if (colorScheme == 1)
                {
                    rv = (uint)(9.0 * (1.0 - t) * t * t * t * 255.0);
                    gv = (uint)(15.0 * (1.0 - t) * (1.0 - t) * t * t * 255.0);
                    bv = (uint)(8.5 * (1.0 - t) * (1.0 - t) * (1.0 - t) * t * 255.0);
                }
                else if (colorScheme == 2)
                {
                    rv = (uint)(t * 255.0);
                    gv = (uint)(t * t * 200.0);
                    bv = (uint)(t * t * t * 100.0);
                }
                else if (colorScheme == 3)
                {
                    rv = (uint)(t * t * 100.0);
                    gv = (uint)(t * 200.0);
                    bv = (uint)((0.5 + 0.5 * t) * 255.0);
                }
                else if (colorScheme == 4)
                {
                    rv = (uint)((0.5 + 0.5 * t) * 255.0);
                    gv = (uint)(t * t * 255.0);
                    bv = (uint)((1.0 - t) * 255.0);
                }
                else
                {
                    rv = (uint)(iterations * 10) % 255u;
                    gv = (uint)(iterations * 5) % 255u;
                    bv = (uint)(iterations * 20) % 255u;
                }

                if (rv > 255u) rv = 255u;
                if (gv > 255u) gv = 255u;
                if (bv > 255u) bv = 255u;

                color = (0xFFu << 24) | (bv << 16) | (gv << 8) | rv;
            }

            output[index] = color;
        }
    }
}
