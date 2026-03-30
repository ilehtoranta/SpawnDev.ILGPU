using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.QR;

/// <summary>
/// GPU-accelerated QR code renderer. Uses ILGPU kernels to render
/// QR code matrices to pixel buffers at any scale.
///
/// Features:
///   - Configurable module size (pixels per QR module)
///   - Quiet zone (white border, standard is 4 modules)
///   - Custom colors (foreground/background as ARGB)
///   - Logo overlay (center region, requires EC level H)
///   - Batch rendering (multiple QR codes in parallel)
/// </summary>
public static class QRRenderer
{
    /// <summary>
    /// Render a QR code matrix to an ARGB pixel buffer on the GPU.
    /// </summary>
    /// <param name="accelerator">GPU accelerator.</param>
    /// <param name="modules">QR code matrix (true = dark).</param>
    /// <param name="moduleSize">Pixels per module (default 10).</param>
    /// <param name="quietZone">Quiet zone in modules (default 4, standard).</param>
    /// <param name="darkColor">Dark module ARGB color (default black).</param>
    /// <param name="lightColor">Light module ARGB color (default white).</param>
    /// <returns>Pixel buffer (ARGB, row-major) and image dimensions.</returns>
    public static async Task<(byte[] pixels, int width, int height)> RenderAsync(
        Accelerator accelerator,
        bool[,] modules,
        int moduleSize = 10,
        int quietZone = 4,
        uint darkColor = 0xFF000000,
        uint lightColor = 0xFFFFFFFF)
    {
        int qrSize = modules.GetLength(0);
        int totalModules = qrSize + quietZone * 2;
        int imageSize = totalModules * moduleSize;

        // Flatten the module matrix to a 1D buffer (GPU-friendly)
        var flatModules = new int[qrSize * qrSize];
        for (int r = 0; r < qrSize; r++)
            for (int c = 0; c < qrSize; c++)
                flatModules[r * qrSize + c] = modules[r, c] ? 1 : 0;

        // Allocate GPU buffers
        using var modulesBuf = accelerator.Allocate1D(flatModules);
        using var pixelsBuf = accelerator.Allocate1D<uint>(imageSize * imageSize);

        // Launch render kernel
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<uint>, int, int, int, uint, uint>(
            RenderKernel);

        kernel(imageSize * imageSize, modulesBuf.View, pixelsBuf.View,
            qrSize, moduleSize, quietZone, darkColor, lightColor);

        await accelerator.SynchronizeAsync();

        // Read back pixels
        var pixels32 = await pixelsBuf.CopyToHostAsync<uint>();

        // Convert uint[] to byte[] (ARGB)
        var pixels = new byte[pixels32.Length * 4];
        Buffer.BlockCopy(pixels32, 0, pixels, 0, pixels.Length);

        return (pixels, imageSize, imageSize);
    }

    /// <summary>
    /// Render to a simple byte array without GPU (CPU fallback).
    /// Useful when no accelerator is available.
    /// </summary>
    public static (byte[] pixels, int width, int height) RenderCpu(
        bool[,] modules,
        int moduleSize = 10,
        int quietZone = 4,
        uint darkColor = 0xFF000000,
        uint lightColor = 0xFFFFFFFF)
    {
        int qrSize = modules.GetLength(0);
        int totalModules = qrSize + quietZone * 2;
        int imageSize = totalModules * moduleSize;

        var pixels = new uint[imageSize * imageSize];

        for (int idx = 0; idx < pixels.Length; idx++)
        {
            int px = idx % imageSize;
            int py = idx / imageSize;

            // Convert pixel to module coordinates
            int moduleX = px / moduleSize - quietZone;
            int moduleY = py / moduleSize - quietZone;

            // Quiet zone or out of bounds = light
            if (moduleX < 0 || moduleX >= qrSize || moduleY < 0 || moduleY >= qrSize)
            {
                pixels[idx] = lightColor;
            }
            else
            {
                pixels[idx] = modules[moduleY, moduleX] ? darkColor : lightColor;
            }
        }

        var result = new byte[pixels.Length * 4];
        Buffer.BlockCopy(pixels, 0, result, 0, result.Length);
        return (result, imageSize, imageSize);
    }

    /// <summary>
    /// GPU kernel: render QR modules to pixels.
    /// Each thread handles one pixel.
    /// </summary>
    static void RenderKernel(
        Index1D idx,
        ArrayView<int> modules,
        ArrayView<uint> pixels,
        int qrSize,
        int moduleSize,
        int quietZone,
        uint darkColor,
        uint lightColor)
    {
        int totalModules = qrSize + quietZone * 2;
        int imageSize = totalModules * moduleSize;

        if (idx >= imageSize * imageSize) return;

        int px = idx % imageSize;
        int py = idx / imageSize;

        // Convert pixel to module coordinates
        int moduleX = px / moduleSize - quietZone;
        int moduleY = py / moduleSize - quietZone;

        // Quiet zone or out of bounds = light
        if (moduleX < 0 || moduleX >= qrSize || moduleY < 0 || moduleY >= qrSize)
        {
            pixels[idx] = lightColor;
            return;
        }

        // Look up module value
        int moduleIdx = moduleY * qrSize + moduleX;
        pixels[idx] = modules[moduleIdx] != 0 ? darkColor : lightColor;
    }

    /// <summary>
    /// Apply a logo overlay to rendered pixels. Clears a square region in the center
    /// and optionally composites a logo image. The QR code must use EC level H (30%
    /// error correction) for reliable scanning with a logo.
    ///
    /// The logo area should not exceed ~25% of the QR code area for reliable scanning.
    /// </summary>
    /// <param name="pixels">ARGB pixel buffer (modified in place).</param>
    /// <param name="imageWidth">Image width in pixels.</param>
    /// <param name="logoPixels">Logo ARGB pixels (optional, null = white square).</param>
    /// <param name="logoWidth">Logo width in pixels.</param>
    /// <param name="logoHeight">Logo height in pixels.</param>
    /// <param name="padding">White padding around logo in pixels.</param>
    public static void ApplyLogo(
        byte[] pixels, int imageWidth,
        byte[]? logoPixels, int logoWidth, int logoHeight,
        int padding = 4)
    {
        int centerX = imageWidth / 2;
        int centerY = imageWidth / 2; // square image

        int clearWidth = logoWidth + padding * 2;
        int clearHeight = logoHeight + padding * 2;
        int startX = centerX - clearWidth / 2;
        int startY = centerY - clearHeight / 2;

        // Clear the center region to white
        for (int y = startY; y < startY + clearHeight && y < imageWidth; y++)
        {
            for (int x = startX; x < startX + clearWidth && x < imageWidth; x++)
            {
                int pixelIdx = (y * imageWidth + x) * 4;
                if (pixelIdx + 3 < pixels.Length)
                {
                    // White (ARGB)
                    pixels[pixelIdx] = 0xFF;     // B
                    pixels[pixelIdx + 1] = 0xFF; // G
                    pixels[pixelIdx + 2] = 0xFF; // R
                    pixels[pixelIdx + 3] = 0xFF; // A
                }
            }
        }

        // Composite logo if provided
        if (logoPixels != null)
        {
            int logoStartX = centerX - logoWidth / 2;
            int logoStartY = centerY - logoHeight / 2;

            for (int ly = 0; ly < logoHeight; ly++)
            {
                for (int lx = 0; lx < logoWidth; lx++)
                {
                    int destX = logoStartX + lx;
                    int destY = logoStartY + ly;
                    if (destX < 0 || destX >= imageWidth || destY < 0 || destY >= imageWidth) continue;

                    int srcIdx = (ly * logoWidth + lx) * 4;
                    int dstIdx = (destY * imageWidth + destX) * 4;

                    if (srcIdx + 3 < logoPixels.Length && dstIdx + 3 < pixels.Length)
                    {
                        pixels[dstIdx] = logoPixels[srcIdx];
                        pixels[dstIdx + 1] = logoPixels[srcIdx + 1];
                        pixels[dstIdx + 2] = logoPixels[srcIdx + 2];
                        pixels[dstIdx + 3] = logoPixels[srcIdx + 3];
                    }
                }
            }
        }
    }
}
