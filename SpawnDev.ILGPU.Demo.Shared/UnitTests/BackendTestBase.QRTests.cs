using SpawnDev.ILGPU.QR;
using SpawnDev.UnitTesting;
using static SpawnDev.ILGPU.QR.QRTables;

namespace SpawnDev.ILGPU.Demo.Shared.UnitTests;

/// <summary>
/// QR code unit tests — encoding, rendering, GPU acceleration.
/// Tests verify structure, not visual appearance — a QR decoder
/// is the ultimate verification (planned).
/// </summary>
public abstract partial class BackendTestBase
{
    // ═══════════════════════════════════════════════════════════
    //  QR Code — Galois Field
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public Task QR_GaloisField_ExpLogRoundTrip()
    {
        // Every non-zero value should round-trip through exp/log
        for (int i = 1; i < 256; i++)
        {
            int exp = GaloisField.Exp[GaloisField.Log[(byte)i]];
            if (exp != i)
                throw new Exception($"GF(256) exp/log round-trip failed: {i} → log={GaloisField.Log[(byte)i]} → exp={exp}");
        }

        // Verify primitive polynomial: 2^8 should XOR with 285 to give 29
        if (GaloisField.Exp[8] != 29)
            throw new Exception($"GF(256) 2^8 should be 29, got {GaloisField.Exp[8]}");

        Console.WriteLine("[QR] GaloisField exp/log round-trip ✓");
        return Task.CompletedTask;
    }

    [TestMethod]
    public Task QR_GaloisField_Multiply()
    {
        // Known multiplications in GF(256)
        // 2 * 2 = 4
        if (GaloisField.Multiply(2, 2) != 4)
            throw new Exception($"GF(256) 2*2 should be 4, got {GaloisField.Multiply(2, 2)}");

        // Anything * 0 = 0
        if (GaloisField.Multiply(0, 137) != 0)
            throw new Exception("GF(256) 0*x should be 0");

        // Anything * 1 = itself
        if (GaloisField.Multiply(1, 42) != 42)
            throw new Exception("GF(256) 1*x should be x");

        // Multiply and divide should round-trip
        byte a = 37, b = 213;
        byte product = GaloisField.Multiply(a, b);
        byte divided = GaloisField.Divide(product, b);
        if (divided != a)
            throw new Exception($"GF(256) multiply/divide round-trip failed: {a}*{b}={product}, {product}/{b}={divided}");

        Console.WriteLine("[QR] GaloisField multiply/divide ✓");
        return Task.CompletedTask;
    }

    [TestMethod]
    public Task QR_GaloisField_ReedSolomon()
    {
        // Test EC generation for a known case: Version 1-M has 16 data codewords, 10 EC codewords
        // We can verify the EC codewords are non-zero and the correct length
        var data = new byte[16];
        for (int i = 0; i < 16; i++) data[i] = (byte)(i + 32); // some test data

        var ec = GaloisField.ComputeEC(data, 10);

        if (ec.Length != 10)
            throw new Exception($"EC should be 10 codewords, got {ec.Length}");

        // EC should not be all zeros (extremely unlikely for non-zero data)
        bool allZero = true;
        foreach (var b in ec) if (b != 0) { allZero = false; break; }
        if (allZero)
            throw new Exception("EC codewords should not be all zeros");

        Console.WriteLine($"[QR] Reed-Solomon EC: {ec.Length} codewords generated ✓");
        return Task.CompletedTask;
    }

    // ═══════════════════════════════════════════════════════════
    //  QR Code — Encoder
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public Task QR_Encode_Version1()
    {
        // "HELLO" in byte mode fits in Version 1-M (16 data codewords)
        var matrix = QREncoder.Encode("HELLO", ECLevel.M);

        int size = matrix.GetLength(0);
        int expectedSize = ModuleCount(1); // 21

        if (size != expectedSize)
            throw new Exception($"Version 1 should be {expectedSize}x{expectedSize}, got {size}x{size}");

        // Verify finder patterns exist (top-left corner should be dark)
        if (!matrix[0, 0])
            throw new Exception("Top-left finder pattern corner should be dark");
        if (!matrix[0, 6])
            throw new Exception("Top-left finder pattern right edge should be dark");
        if (!matrix[6, 0])
            throw new Exception("Top-left finder pattern bottom edge should be dark");

        // Top-right finder
        if (!matrix[0, size - 1])
            throw new Exception("Top-right finder pattern corner should be dark");

        // Bottom-left finder
        if (!matrix[size - 1, 0])
            throw new Exception("Bottom-left finder pattern corner should be dark");

        Console.WriteLine($"[QR] Encode Version 1 ({size}x{size}): finder patterns verified ✓");
        return Task.CompletedTask;
    }

    [TestMethod]
    public Task QR_Encode_URL()
    {
        // A typical URL
        var matrix = QREncoder.Encode("https://hub.spawndev.com/compute", ECLevel.M);
        int size = matrix.GetLength(0);

        // URL is ~32 bytes → should be Version 3 or 4 in M mode
        if (size < 25 || size > 45)
            throw new Exception($"URL QR should be reasonable size, got {size}x{size}");

        Console.WriteLine($"[QR] Encode URL: {size}x{size} ✓");
        return Task.CompletedTask;
    }

    [TestMethod]
    public Task QR_Encode_ECLevelH_LargerThanM()
    {
        // Same text at H should produce a larger (or same) QR than M
        var matrixM = QREncoder.Encode("SpawnDev.ILGPU P2P Compute", ECLevel.M);
        var matrixH = QREncoder.Encode("SpawnDev.ILGPU P2P Compute", ECLevel.H);

        int sizeM = matrixM.GetLength(0);
        int sizeH = matrixH.GetLength(0);

        // H has more EC overhead, so it needs a higher version (larger matrix)
        if (sizeH < sizeM)
            throw new Exception($"EC level H should be >= M in size: H={sizeH}, M={sizeM}");

        Console.WriteLine($"[QR] EC level comparison: M={sizeM}x{sizeM}, H={sizeH}x{sizeH} ✓");
        return Task.CompletedTask;
    }

    [TestMethod]
    public Task QR_Encode_AllECLevels()
    {
        // Verify all 4 EC levels produce valid matrices
        foreach (var level in new[] { ECLevel.L, ECLevel.M, ECLevel.Q, ECLevel.H })
        {
            var matrix = QREncoder.Encode("Test", level);
            int size = matrix.GetLength(0);
            if (size < 21) // minimum is Version 1 = 21x21
                throw new Exception($"EC level {level} produced too-small matrix: {size}");
        }

        Console.WriteLine("[QR] All 4 EC levels encode successfully ✓");
        return Task.CompletedTask;
    }

    // ═══════════════════════════════════════════════════════════
    //  QR Code — CPU Renderer
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public Task QR_Render_CPU()
    {
        var (pixels, w, h) = QRCode.Generate("https://spawndev.com", moduleSize: 4, quietZone: 2);

        if (w != h)
            throw new Exception($"QR image should be square: {w}x{h}");
        if (w <= 0)
            throw new Exception("Image width should be positive");
        if (pixels.Length != w * h * 4)
            throw new Exception($"Pixel buffer size mismatch: {pixels.Length} vs expected {w * h * 4}");

        // Check corners are white (quiet zone)
        uint topLeft = BitConverter.ToUInt32(pixels, 0);
        if (topLeft != 0xFFFFFFFF)
            throw new Exception($"Top-left pixel should be white (quiet zone), got 0x{topLeft:X8}");

        Console.WriteLine($"[QR] CPU render: {w}x{h}, {pixels.Length} bytes ✓");
        return Task.CompletedTask;
    }

    // ═══════════════════════════════════════════════════════════
    //  QR Code — GPU Renderer
    // ═══════════════════════════════════════════════════════════

    [TestMethod]
    public async Task QR_Render_GPU() => await RunTest(async accelerator =>
    {
        var (pixels, w, h) = await QRCode.GenerateAsync(
            accelerator, "https://spawndev.com", moduleSize: 4, quietZone: 2);

        if (w != h)
            throw new Exception($"QR image should be square: {w}x{h}");
        if (pixels.Length != w * h * 4)
            throw new Exception($"Pixel buffer size mismatch: {pixels.Length} vs expected {w * h * 4}");

        // Verify quiet zone corner is white
        uint topLeft = BitConverter.ToUInt32(pixels, 0);
        if (topLeft != 0xFFFFFFFF)
            throw new Exception($"Top-left pixel should be white (quiet zone), got 0x{topLeft:X8}");

        Console.WriteLine($"[QR] GPU render: {w}x{h}, {pixels.Length} bytes ✓");
    });

    [TestMethod]
    public async Task QR_Render_GPU_WithLogo() => await RunTest(async accelerator =>
    {
        // Create a small red square as the logo
        int logoSize = 20;
        var logo = new byte[logoSize * logoSize * 4];
        for (int i = 0; i < logo.Length; i += 4)
        {
            logo[i] = 0x00;     // B
            logo[i + 1] = 0x00; // G
            logo[i + 2] = 0xFF; // R
            logo[i + 3] = 0xFF; // A
        }

        var (pixels, w, h) = await QRCode.GenerateWithLogoAsync(
            accelerator, "https://hub.spawndev.com/compute",
            logo, logoSize, logoSize, moduleSize: 6);

        if (w != h)
            throw new Exception($"QR image should be square: {w}x{h}");

        // Verify center pixel is red (the logo)
        int centerIdx = (h / 2 * w + w / 2) * 4;
        if (centerIdx + 3 < pixels.Length)
        {
            byte b = pixels[centerIdx], g = pixels[centerIdx + 1], r = pixels[centerIdx + 2];
            if (r != 0xFF || g != 0x00 || b != 0x00)
                throw new Exception($"Center pixel should be red (logo), got R={r} G={g} B={b}");
        }

        Console.WriteLine($"[QR] GPU render with logo: {w}x{h} ✓");
    });

    [TestMethod]
    public async Task QR_Render_GPU_CPUMatch() => await RunTest(async accelerator =>
    {
        // GPU and CPU renders should produce identical output
        string text = "MATCH TEST";
        int moduleSize = 5;
        int quietZone = 3;

        var (cpuPixels, cpuW, cpuH) = QRCode.Generate(text, moduleSize: moduleSize, quietZone: quietZone);
        var (gpuPixels, gpuW, gpuH) = await QRCode.GenerateAsync(
            accelerator, text, moduleSize: moduleSize, quietZone: quietZone);

        if (cpuW != gpuW || cpuH != gpuH)
            throw new Exception($"Size mismatch: CPU={cpuW}x{cpuH}, GPU={gpuW}x{gpuH}");

        int mismatches = 0;
        for (int i = 0; i < cpuPixels.Length && i < gpuPixels.Length; i++)
        {
            if (cpuPixels[i] != gpuPixels[i])
                mismatches++;
        }

        if (mismatches > 0)
            throw new Exception($"GPU/CPU mismatch: {mismatches} bytes differ out of {cpuPixels.Length}");

        Console.WriteLine($"[QR] GPU/CPU render match: {cpuW}x{cpuH}, {cpuPixels.Length} bytes identical ✓");
    });

    [TestMethod]
    public Task QR_Decode_RoundTrip()
    {
        string original = "https://hub.spawndev.com";

        // Encode → render → decode
        var (pixels, w, h) = QRCode.Generate(original, ECLevel.M, moduleSize: 8, quietZone: 4);

        // The renderer outputs ARGB as uint (little-endian: BGRA in bytes)
        // The decoder expects RGBA. Swap B↔R.
        for (int i = 0; i < pixels.Length; i += 4)
            (pixels[i], pixels[i + 2]) = (pixels[i + 2], pixels[i]);

        var decoded = QRDecoder.Decode(pixels, w, h);

        if (decoded == null)
            throw new Exception("QR decode returned null — could not read the QR code");
        if (decoded != original)
            throw new Exception($"QR round-trip mismatch: encoded '{original}', decoded '{decoded}'");

        Console.WriteLine($"[QR] Round-trip: '{original}' → encode → render → decode → '{decoded}' ✓");
        return Task.CompletedTask;
    }

    [TestMethod]
    public Task QR_Decode_RoundTrip_WithLogo()
    {
        string original = "https://spawndev.com/p2p";

        // Create a small logo (white square)
        int logoSize = 16;
        var logo = new byte[logoSize * logoSize * 4];
        for (int i = 0; i < logo.Length; i++) logo[i] = 0xFF; // white

        var (pixels, w, h) = QRCode.GenerateWithLogo(
            original, logo, logoSize, logoSize, moduleSize: 8, quietZone: 4);

        // Swap B↔R for decoder
        for (int i = 0; i < pixels.Length; i += 4)
            (pixels[i], pixels[i + 2]) = (pixels[i + 2], pixels[i]);

        var decoded = QRDecoder.Decode(pixels, w, h);

        if (decoded == null)
            throw new Exception("QR decode with logo returned null — logo may be too large or EC level insufficient");
        if (decoded != original)
            throw new Exception($"QR logo round-trip mismatch: encoded '{original}', decoded '{decoded}'");

        Console.WriteLine($"[QR] Logo round-trip: '{original}' → encode(H) → logo → decode → '{decoded}' ✓");
        return Task.CompletedTask;
    }

    [TestMethod]
    public Task QR_Encode_LongURL()
    {
        // Test with a realistic P2P join link
        string url = "https://hub.spawndev.com/compute/join?compute=a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2&n=Demo+Compute+Swarm";
        var matrix = QREncoder.Encode(url, ECLevel.H);
        int size = matrix.GetLength(0);

        if (size < 21)
            throw new Exception($"Long URL QR too small: {size}");

        Console.WriteLine($"[QR] Long URL encode: {url.Length} chars → {size}x{size} (EC=H) ✓");
        return Task.CompletedTask;
    }
}
