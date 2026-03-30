using ILGPU.Runtime;
using static SpawnDev.ILGPU.QR.QRTables;

namespace SpawnDev.ILGPU.QR;

/// <summary>
/// GPU-accelerated QR code generation for SpawnDev.ILGPU.
/// Supports all 40 QR versions, 4 error correction levels,
/// GPU-rendered output with optional logo overlay.
///
/// Usage:
///   // Simple — CPU encode + CPU render
///   var (pixels, w, h) = QRCode.Generate("https://hub.spawndev.com", moduleSize: 10);
///
///   // GPU-accelerated render
///   var (pixels, w, h) = await QRCode.GenerateAsync(accelerator, "https://hub.spawndev.com");
///
///   // With logo overlay (requires EC level H)
///   var (pixels, w, h) = await QRCode.GenerateWithLogoAsync(
///       accelerator, "https://hub.spawndev.com", logoPixels, logoW, logoH);
/// </summary>
public static class QRCode
{
    /// <summary>
    /// Generate a QR code and render to pixels (CPU).
    /// </summary>
    /// <param name="text">Text/URL to encode.</param>
    /// <param name="ecLevel">Error correction level (default M, use H for logo).</param>
    /// <param name="moduleSize">Pixels per QR module.</param>
    /// <param name="quietZone">White border in modules (standard: 4).</param>
    /// <param name="darkColor">Dark module ARGB color.</param>
    /// <param name="lightColor">Light module ARGB color.</param>
    /// <returns>ARGB pixel buffer and dimensions.</returns>
    public static (byte[] pixels, int width, int height) Generate(
        string text,
        ECLevel ecLevel = ECLevel.M,
        int moduleSize = 10,
        int quietZone = 4,
        uint darkColor = 0xFF000000,
        uint lightColor = 0xFFFFFFFF)
    {
        var modules = QREncoder.Encode(text, ecLevel);
        return QRRenderer.RenderCpu(modules, moduleSize, quietZone, darkColor, lightColor);
    }

    /// <summary>
    /// Generate a QR code and render to pixels (GPU-accelerated).
    /// </summary>
    public static async Task<(byte[] pixels, int width, int height)> GenerateAsync(
        Accelerator accelerator,
        string text,
        ECLevel ecLevel = ECLevel.M,
        int moduleSize = 10,
        int quietZone = 4,
        uint darkColor = 0xFF000000,
        uint lightColor = 0xFFFFFFFF)
    {
        var modules = QREncoder.Encode(text, ecLevel);
        return await QRRenderer.RenderAsync(accelerator, modules, moduleSize, quietZone, darkColor, lightColor);
    }

    /// <summary>
    /// Generate a QR code with logo overlay (GPU-accelerated).
    /// Automatically uses EC level H for maximum error correction (30% redundancy),
    /// allowing the center to be obscured by the logo while remaining scannable.
    /// </summary>
    /// <param name="accelerator">GPU accelerator.</param>
    /// <param name="text">Text/URL to encode.</param>
    /// <param name="logoPixels">Logo ARGB pixels.</param>
    /// <param name="logoWidth">Logo width in pixels.</param>
    /// <param name="logoHeight">Logo height in pixels.</param>
    /// <param name="moduleSize">Pixels per QR module.</param>
    /// <param name="quietZone">White border in modules.</param>
    /// <param name="logoPadding">White padding around logo in pixels.</param>
    /// <param name="darkColor">Dark module ARGB color.</param>
    /// <param name="lightColor">Light module ARGB color.</param>
    /// <returns>ARGB pixel buffer and dimensions.</returns>
    public static async Task<(byte[] pixels, int width, int height)> GenerateWithLogoAsync(
        Accelerator accelerator,
        string text,
        byte[] logoPixels,
        int logoWidth,
        int logoHeight,
        int moduleSize = 10,
        int quietZone = 4,
        int logoPadding = 4,
        uint darkColor = 0xFF000000,
        uint lightColor = 0xFFFFFFFF)
    {
        // Force EC level H for logo overlay (30% error correction)
        var modules = QREncoder.Encode(text, ECLevel.H);
        var (pixels, w, h) = await QRRenderer.RenderAsync(
            accelerator, modules, moduleSize, quietZone, darkColor, lightColor);

        QRRenderer.ApplyLogo(pixels, w, logoPixels, logoWidth, logoHeight, logoPadding);

        return (pixels, w, h);
    }

    /// <summary>
    /// Generate a QR code with logo overlay (CPU fallback).
    /// </summary>
    public static (byte[] pixels, int width, int height) GenerateWithLogo(
        string text,
        byte[] logoPixels,
        int logoWidth,
        int logoHeight,
        int moduleSize = 10,
        int quietZone = 4,
        int logoPadding = 4,
        uint darkColor = 0xFF000000,
        uint lightColor = 0xFFFFFFFF)
    {
        var modules = QREncoder.Encode(text, ECLevel.H);
        var (pixels, w, h) = QRRenderer.RenderCpu(modules, moduleSize, quietZone, darkColor, lightColor);
        QRRenderer.ApplyLogo(pixels, w, logoPixels, logoWidth, logoHeight, logoPadding);
        return (pixels, w, h);
    }

    /// <summary>
    /// Decode a QR code from RGBA pixel data (CPU).
    /// </summary>
    /// <param name="pixels">RGBA pixel buffer (4 bytes per pixel).</param>
    /// <param name="width">Image width.</param>
    /// <param name="height">Image height.</param>
    /// <returns>Decoded text, or null if no QR code found.</returns>
    public static string? Decode(byte[] pixels, int width, int height)
    {
        return QRDecoder.Decode(pixels, width, height);
    }

    /// <summary>
    /// Decode a QR code from RGBA pixel data (GPU-accelerated).
    /// </summary>
    public static async Task<string?> DecodeAsync(Accelerator accelerator, byte[] pixels, int width, int height)
    {
        return await QRDecoder.DecodeAsync(accelerator, pixels, width, height);
    }

    /// <summary>
    /// Encode text to a QR code module matrix (no rendering).
    /// Returns true = dark, false = light.
    /// </summary>
    public static bool[,] EncodeMatrix(string text, ECLevel ecLevel = ECLevel.M)
    {
        return QREncoder.Encode(text, ecLevel);
    }

    /// <summary>
    /// Get the QR code module count for the given text and EC level.
    /// </summary>
    public static int GetModuleCount(string text, ECLevel ecLevel = ECLevel.M)
    {
        return QREncoder.GetSize(text, ecLevel);
    }
}
