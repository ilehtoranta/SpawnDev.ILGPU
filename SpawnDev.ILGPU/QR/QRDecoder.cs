using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.QR;

/// <summary>
/// GPU-accelerated QR code decoder. Reads QR codes from pixel data.
///
/// Pipeline (GPU-parallel steps marked with ⚡):
///   1. ⚡ Grayscale conversion (parallel per pixel)
///   2. ⚡ Adaptive threshold / binarization (parallel per pixel)
///   3.    Finder pattern detection (scan rows/columns for 1:1:3:1:1 ratio)
///   4. ⚡ Perspective correction (parallel pixel sampling)
///   5. ⚡ Bit extraction from corrected grid (parallel per module)
///   6.    Unmask + read format info
///   7.    Reed-Solomon error correction decode
///   8.    Data decode (byte/numeric/alphanumeric mode)
///
/// Usage:
///   var text = QRDecoder.Decode(rgbaPixels, width, height);
///   var text = await QRDecoder.DecodeAsync(accelerator, rgbaPixels, width, height);
/// </summary>
public static class QRDecoder
{
    /// <summary>
    /// Decode a QR code from RGBA pixel data using GPU-accelerated image processing.
    /// </summary>
    /// <param name="accelerator">GPU accelerator for parallel binarization.</param>
    /// <param name="pixels">RGBA pixel buffer (4 bytes per pixel).</param>
    /// <param name="width">Image width.</param>
    /// <param name="height">Image height.</param>
    /// <returns>Decoded text, or null if no QR code found.</returns>
    public static async Task<string?> DecodeAsync(Accelerator accelerator, byte[] pixels, int width, int height)
    {
        int totalPixels = width * height;

        // 1. GPU: convert RGBA (packed as int) to grayscale
        var packed = new int[totalPixels];
        Buffer.BlockCopy(pixels, 0, packed, 0, pixels.Length);
        using var packedBuf = accelerator.Allocate1D(packed);
        using var grayBuf = accelerator.Allocate1D<byte>(totalPixels);

        var grayscaleKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<byte>>(GrayscaleKernel);
        grayscaleKernel(totalPixels, packedBuf.View, grayBuf.View);
        await accelerator.SynchronizeAsync();

        var gray = await grayBuf.CopyToHostAsync<byte>();

        // Rest of pipeline is sequential (finder detection, format reading, data decode)
        return DecodeFromGrayscale(gray, width, height);
    }

    /// <summary>GPU kernel: RGBA packed int → grayscale byte.</summary>
    static void GrayscaleKernel(Index1D idx, ArrayView<int> rgba, ArrayView<byte> gray)
    {
        int pixel = rgba[idx];
        int r = pixel & 0xFF;
        int g = (pixel >> 8) & 0xFF;
        int b = (pixel >> 16) & 0xFF;
        gray[idx] = (byte)((r * 77 + g * 150 + b * 29) >> 8);
    }

    /// <summary>
    /// Decode a QR code from RGBA pixel data (CPU fallback).
    /// </summary>
    /// <param name="pixels">RGBA pixel buffer (4 bytes per pixel).</param>
    /// <param name="width">Image width.</param>
    /// <param name="height">Image height.</param>
    /// <returns>Decoded text, or null if no QR code found.</returns>
    public static string? Decode(byte[] pixels, int width, int height)
    {
        // 1. Convert to grayscale
        var gray = new byte[width * height];
        for (int i = 0; i < gray.Length; i++)
        {
            int pi = i * 4;
            // Luminance: 0.299R + 0.587G + 0.114B
            gray[i] = (byte)((pixels[pi] * 77 + pixels[pi + 1] * 150 + pixels[pi + 2] * 29) >> 8);
        }

        return DecodeFromGrayscale(gray, width, height);
    }

    /// <summary>Shared decode pipeline from grayscale image.</summary>
    private static string? DecodeFromGrayscale(byte[] gray, int width, int height)
    {

        // 2. Binarize (adaptive threshold using local mean)
        var binary = Binarize(gray, width, height);

        // 3. Find finder patterns
        var finders = FindFinderPatterns(binary, width, height);
        if (finders.Count < 3) return null;

        // 4. Identify the three finder patterns (top-left, top-right, bottom-left)
        var (tl, tr, bl) = IdentifyFinderTriangle(finders);
        if (tl == null) return null;

        // 5. Estimate module size and version
        double moduleSize = EstimateModuleSize(tl, tr, bl);
        if (moduleSize < 1) return null;

        int estimatedModules = (int)Math.Round(Distance(tl, tr) / moduleSize) + 7;
        int version = (estimatedModules - 17) / 4;
        if (version < 1) version = 1;
        if (version > 40) return null;

        int qrSize = QRTables.ModuleCount(version);

        // 6. Sample the grid
        var modules = SampleGrid(binary, width, height, tl, tr, bl, qrSize);
        if (modules == null) return null;

        // 7. Read format info to determine EC level and mask
        var (ecLevel, maskPattern) = ReadFormatInfo(modules, qrSize);
        if (ecLevel < 0) return null;

        // 8. Unmask
        UnmaskData(modules, qrSize, maskPattern);

        // 9. Extract data bits
        var codewords = ExtractDataBits(modules, qrSize, version);

        // 10. Deinterleave and error correct
        var ecInfo = QRTables.GetECInfo(version, (QRTables.ECLevel)ecLevel);
        var dataBytes = DeinterleaveAndCorrect(codewords, ecInfo);
        if (dataBytes == null) return null;

        // 11. Decode data
        return DecodeData(dataBytes);
    }

    #region Binarization

    private static bool[] Binarize(byte[] gray, int width, int height)
    {
        var binary = new bool[width * height];
        int blockSize = Math.Max(width, height) / 8;
        if (blockSize < 8) blockSize = 8;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                // Local mean in a block around (x, y)
                int x0 = Math.Max(0, x - blockSize / 2);
                int y0 = Math.Max(0, y - blockSize / 2);
                int x1 = Math.Min(width - 1, x + blockSize / 2);
                int y1 = Math.Min(height - 1, y + blockSize / 2);

                long sum = 0;
                int count = 0;
                // Sample corners + center for speed
                int[] sampleXs = { x0, x1, (x0 + x1) / 2 };
                int[] sampleYs = { y0, y1, (y0 + y1) / 2 };
                foreach (int sy in sampleYs)
                {
                    foreach (int sx in sampleXs)
                    {
                        sum += gray[sy * width + sx];
                        count++;
                    }
                }
                int threshold = (int)(sum / count) - 5; // slight bias toward dark

                binary[y * width + x] = gray[y * width + x] < threshold;
            }
        }

        return binary;
    }

    #endregion

    #region Finder Pattern Detection

    private record FinderPattern(double X, double Y, double ModuleSize);

    private static List<FinderPattern> FindFinderPatterns(bool[] binary, int width, int height)
    {
        var candidates = new List<FinderPattern>();

        // Scan rows for 1:1:3:1:1 dark:light:dark:light:dark ratio
        for (int y = 0; y < height; y++)
        {
            int[] counts = new int[5];
            int state = 0;

            for (int x = 0; x < width; x++)
            {
                bool dark = binary[y * width + x];

                if (dark)
                {
                    if (state == 1 || state == 3) // was in light zone
                    {
                        state++;
                    }
                    counts[state]++;
                }
                else
                {
                    if (state == 0 || state == 2) // was in dark zone
                    {
                        state++;
                    }
                    else if (state == 4)
                    {
                        // Check ratio
                        if (IsFinderRatio(counts))
                        {
                            int totalWidth = counts[0] + counts[1] + counts[2] + counts[3] + counts[4];
                            double centerX = x - totalWidth / 2.0;
                            double modSize = totalWidth / 7.0;

                            // Verify vertically
                            if (VerifyFinderVertical(binary, width, height, (int)centerX, y, modSize))
                            {
                                candidates.Add(new FinderPattern(centerX, y, modSize));
                            }
                        }

                        // Shift pattern
                        counts[0] = counts[2];
                        counts[1] = counts[3];
                        counts[2] = counts[4];
                        counts[3] = 1;
                        counts[4] = 0;
                        state = 3;
                        continue;
                    }
                    counts[state]++;
                }
            }
        }

        // Merge nearby candidates
        return MergeCandidates(candidates);
    }

    private static bool IsFinderRatio(int[] counts)
    {
        int total = counts[0] + counts[1] + counts[2] + counts[3] + counts[4];
        if (total < 7) return false;

        double unit = total / 7.0;
        double tolerance = unit * 0.7;

        return Math.Abs(counts[0] - unit) < tolerance
            && Math.Abs(counts[1] - unit) < tolerance
            && Math.Abs(counts[2] - 3 * unit) < tolerance
            && Math.Abs(counts[3] - unit) < tolerance
            && Math.Abs(counts[4] - unit) < tolerance;
    }

    private static bool VerifyFinderVertical(bool[] binary, int width, int height, int cx, int cy, double modSize)
    {
        int[] counts = new int[5];
        int expectedHalf = (int)(modSize * 3.5);
        int startY = Math.Max(0, cy - expectedHalf);
        int endY = Math.Min(height - 1, cy + expectedHalf);

        int state = 0;
        for (int y = startY; y <= endY && state < 5; y++)
        {
            bool dark = binary[y * width + cx];
            bool expectedDark = state == 0 || state == 2 || state == 4;

            if (dark == expectedDark)
            {
                counts[state]++;
            }
            else
            {
                state++;
                if (state < 5) counts[state] = 1;
            }
        }

        return state >= 4 && IsFinderRatio(counts);
    }

    private static List<FinderPattern> MergeCandidates(List<FinderPattern> candidates)
    {
        var merged = new List<FinderPattern>();
        var used = new bool[candidates.Count];

        for (int i = 0; i < candidates.Count; i++)
        {
            if (used[i]) continue;

            double sumX = candidates[i].X, sumY = candidates[i].Y, sumMod = candidates[i].ModuleSize;
            int count = 1;

            for (int j = i + 1; j < candidates.Count; j++)
            {
                if (used[j]) continue;
                double dist = Distance(candidates[i], candidates[j]);
                if (dist < candidates[i].ModuleSize * 5)
                {
                    sumX += candidates[j].X;
                    sumY += candidates[j].Y;
                    sumMod += candidates[j].ModuleSize;
                    count++;
                    used[j] = true;
                }
            }

            merged.Add(new FinderPattern(sumX / count, sumY / count, sumMod / count));
        }

        return merged;
    }

    private static double Distance(FinderPattern a, FinderPattern b)
        => Math.Sqrt((a.X - b.X) * (a.X - b.X) + (a.Y - b.Y) * (a.Y - b.Y));

    #endregion

    #region Finder Triangle Identification

    private static (FinderPattern? tl, FinderPattern? tr, FinderPattern? bl) IdentifyFinderTriangle(
        List<FinderPattern> finders)
    {
        if (finders.Count < 3) return (null, null, null);

        // Find the 3 that form the best right-angle triangle
        // The top-left finder is at the right angle
        FinderPattern? bestTL = null, bestTR = null, bestBL = null;
        double bestScore = double.MaxValue;

        for (int i = 0; i < finders.Count; i++)
        {
            for (int j = i + 1; j < finders.Count; j++)
            {
                for (int k = j + 1; k < finders.Count; k++)
                {
                    var pts = new[] { finders[i], finders[j], finders[k] };

                    // Try each as the right-angle vertex
                    for (int v = 0; v < 3; v++)
                    {
                        var a = pts[v];
                        var b = pts[(v + 1) % 3];
                        var c = pts[(v + 2) % 3];

                        // Check right angle at a
                        double abx = b.X - a.X, aby = b.Y - a.Y;
                        double acx = c.X - a.X, acy = c.Y - a.Y;
                        double dot = abx * acx + aby * acy;
                        double score = Math.Abs(dot) / (Math.Sqrt(abx * abx + aby * aby) * Math.Sqrt(acx * acx + acy * acy) + 0.001);

                        if (score < bestScore)
                        {
                            bestScore = score;
                            bestTL = a;

                            // Determine which is TR and which is BL
                            // TR is more to the right, BL is more downward
                            double crossProduct = abx * acy - aby * acx;
                            if (crossProduct > 0)
                            {
                                bestTR = b;
                                bestBL = c;
                            }
                            else
                            {
                                bestTR = c;
                                bestBL = b;
                            }
                        }
                    }
                }
            }
        }

        return (bestTL, bestTR, bestBL);
    }

    #endregion

    #region Grid Sampling

    private static double EstimateModuleSize(FinderPattern tl, FinderPattern tr, FinderPattern bl)
    {
        // Distance between finder centers spans (qrSize - 7) modules.
        // We don't know qrSize yet, but we can estimate module size from the
        // finder patterns' own ModuleSize property (detected during scanning).
        return (tl.ModuleSize + tr.ModuleSize + bl.ModuleSize) / 3.0;
    }

    private static bool[,]? SampleGrid(bool[] binary, int width, int height,
        FinderPattern tl, FinderPattern tr, FinderPattern bl, int qrSize)
    {
        var modules = new bool[qrSize, qrSize];

        // Finder pattern centers are at module (3.5, 3.5) from each corner.
        // We need to map from module coords to image coords.
        // The three finder centers define a coordinate system:
        //   tl = module (3.5, 3.5)
        //   tr = module (qrSize - 3.5, 3.5)
        //   bl = module (3.5, qrSize - 3.5)

        // Module span between tl and tr: (qrSize - 7) modules
        double span = qrSize - 7.0;

        // Unit vectors in image space per module
        double uxX = (tr.X - tl.X) / span; // x component of rightward unit
        double uxY = (tr.Y - tl.Y) / span;
        double uyX = (bl.X - tl.X) / span; // x component of downward unit
        double uyY = (bl.Y - tl.Y) / span;

        // Origin in image space: tl is at module (3.5, 3.5)
        // So module (0, 0) maps to tl - 3.5 * ux - 3.5 * uy
        double originX = tl.X - 3.5 * uxX - 3.5 * uyX;
        double originY = tl.Y - 3.5 * uxY - 3.5 * uyY;

        for (int r = 0; r < qrSize; r++)
        {
            for (int c = 0; c < qrSize; c++)
            {
                // Map module center (c + 0.5, r + 0.5) to image coordinates
                double mc = c + 0.5;
                double mr = r + 0.5;
                double x = originX + mc * uxX + mr * uyX;
                double y = originY + mc * uxY + mr * uyY;

                int ix = (int)Math.Round(x);
                int iy = (int)Math.Round(y);

                if (ix >= 0 && ix < width && iy >= 0 && iy < height)
                    modules[r, c] = binary[iy * width + ix];
            }
        }

        return modules;
    }

    #endregion

    #region Format Info & Unmasking

    private static (int ecLevel, int maskPattern) ReadFormatInfo(bool[,] modules, int size)
    {
        // Read format bits from around top-left finder
        int bits = 0;
        int[] rowPos = { 0, 1, 2, 3, 4, 5, 7, 8, 8, 8, 8, 8, 8, 8, 8 };
        int[] colPos = { 8, 8, 8, 8, 8, 8, 8, 8, 7, 5, 4, 3, 2, 1, 0 };

        for (int i = 0; i < 15; i++)
        {
            if (modules[rowPos[i], colPos[i]])
                bits |= 1 << (14 - i);
        }

        // Match directly against stored format info strings
        // (QRTables.FormatInfo already includes BCH + XOR mask)
        int bestEc = -1, bestMask = -1, bestDist = int.MaxValue;
        for (int ec = 0; ec < 4; ec++)
        {
            for (int mask = 0; mask < 8; mask++)
            {
                int expected = QRTables.FormatInfo[ec * 8 + mask];
                int dist = BitCount(bits ^ expected);
                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestEc = ec;
                    bestMask = mask;
                }
            }
        }

        // Accept if Hamming distance <= 3 (format info has distance 7, can correct 3)
        if (bestDist > 3) return (-1, -1);
        return (bestEc, bestMask);
    }

    private static int BitCount(int n)
    {
        int count = 0;
        while (n != 0) { count += n & 1; n >>= 1; }
        return count;
    }

    private static void UnmaskData(bool[,] modules, int size, int maskPattern)
    {
        // Create reserved map
        var reserved = new bool[size, size];

        // Mark finder patterns + separators
        MarkFinderReserved(reserved, 0, 0, size);
        MarkFinderReserved(reserved, 0, size - 7, size);
        MarkFinderReserved(reserved, size - 7, 0, size);
        for (int i = 0; i < 8; i++)
        {
            MarkReserved(reserved, i, 7, size);
            MarkReserved(reserved, 7, i, size);
            MarkReserved(reserved, i, size - 8, size);
            MarkReserved(reserved, 7, size - 8 + i, size);
            MarkReserved(reserved, size - 8, i, size);
            MarkReserved(reserved, size - 8 + i, 7, size);
        }

        // Timing patterns
        for (int i = 8; i < size - 8; i++)
        {
            reserved[6, i] = true;
            reserved[i, 6] = true;
        }

        // Format info area
        for (int i = 0; i < 9; i++)
        {
            reserved[i, 8] = true;
            reserved[8, i] = true;
        }
        for (int i = 0; i < 8; i++) reserved[8, size - 8 + i] = true;
        for (int i = 0; i < 7; i++) reserved[size - 7 + i, 8] = true;

        // Dark module
        int version = (size - 17) / 4;
        reserved[4 * version + 9, 8] = true;

        // Alignment patterns
        if (version >= 2)
        {
            var positions = QRTables.AlignmentPatterns[version - 1];
            foreach (int r in positions)
            {
                foreach (int c in positions)
                {
                    if ((r <= 8 && c <= 8) || (r <= 8 && c >= size - 8) || (r >= size - 8 && c <= 8))
                        continue;
                    for (int dr = -2; dr <= 2; dr++)
                        for (int dc = -2; dc <= 2; dc++)
                            if (r + dr >= 0 && r + dr < size && c + dc >= 0 && c + dc < size)
                                reserved[r + dr, c + dc] = true;
                }
            }
        }

        // Unmask data modules
        for (int r = 0; r < size; r++)
        {
            for (int c = 0; c < size; c++)
            {
                if (!reserved[r, c] && EvaluateMask(maskPattern, r, c))
                    modules[r, c] = !modules[r, c];
            }
        }
    }

    private static bool EvaluateMask(int maskPattern, int row, int col) => maskPattern switch
    {
        0 => (row + col) % 2 == 0,
        1 => row % 2 == 0,
        2 => col % 3 == 0,
        3 => (row + col) % 3 == 0,
        4 => (row / 2 + col / 3) % 2 == 0,
        5 => (row * col) % 2 + (row * col) % 3 == 0,
        6 => ((row * col) % 2 + (row * col) % 3) % 2 == 0,
        7 => ((row + col) % 2 + (row * col) % 3) % 2 == 0,
        _ => false,
    };

    private static void MarkFinderReserved(bool[,] reserved, int row, int col, int size)
    {
        for (int r = 0; r < 7 && row + r < size; r++)
            for (int c = 0; c < 7 && col + c < size; c++)
                reserved[row + r, col + c] = true;
    }

    private static void MarkReserved(bool[,] reserved, int r, int c, int size)
    {
        if (r >= 0 && r < size && c >= 0 && c < size)
            reserved[r, c] = true;
    }

    #endregion

    #region Data Extraction

    private static byte[] ExtractDataBits(bool[,] modules, int size, int version)
    {
        var bits = new List<bool>();

        // Create reserved map (same as unmask)
        var reserved = new bool[size, size];
        MarkFinderReserved(reserved, 0, 0, size);
        MarkFinderReserved(reserved, 0, size - 7, size);
        MarkFinderReserved(reserved, size - 7, 0, size);
        for (int i = 0; i < 8; i++)
        {
            MarkReserved(reserved, i, 7, size);
            MarkReserved(reserved, 7, i, size);
            MarkReserved(reserved, i, size - 8, size);
            MarkReserved(reserved, 7, size - 8 + i, size);
            MarkReserved(reserved, size - 8, i, size);
            MarkReserved(reserved, size - 8 + i, 7, size);
        }
        for (int i = 8; i < size - 8; i++)
        {
            reserved[6, i] = true;
            reserved[i, 6] = true;
        }
        for (int i = 0; i < 9; i++) { reserved[i, 8] = true; reserved[8, i] = true; }
        for (int i = 0; i < 8; i++) reserved[8, size - 8 + i] = true;
        for (int i = 0; i < 7; i++) reserved[size - 7 + i, 8] = true;
        reserved[4 * version + 9, 8] = true;
        if (version >= 2)
        {
            var positions = QRTables.AlignmentPatterns[version - 1];
            foreach (int r in positions)
                foreach (int c in positions)
                {
                    if ((r <= 8 && c <= 8) || (r <= 8 && c >= size - 8) || (r >= size - 8 && c <= 8)) continue;
                    for (int dr = -2; dr <= 2; dr++)
                        for (int dc = -2; dc <= 2; dc++)
                            if (r + dr >= 0 && r + dr < size && c + dc >= 0 && c + dc < size)
                                reserved[r + dr, c + dc] = true;
                }
        }
        if (version >= 7)
        {
            for (int i = 0; i < 6; i++)
                for (int j = 0; j < 3; j++)
                {
                    reserved[size - 11 + j, i] = true;
                    reserved[i, size - 11 + j] = true;
                }
        }

        // Read data in zigzag pattern (same as encoder)
        int col = size - 1;
        while (col >= 0)
        {
            if (col == 6) col--;

            bool upward = ((size - 1 - col) / 2) % 2 == 0;

            for (int i = 0; i < size; i++)
            {
                int row = upward ? (size - 1 - i) : i;

                if (!reserved[row, col])
                    bits.Add(modules[row, col]);

                if (col > 0 && !reserved[row, col - 1])
                    bits.Add(modules[row, col - 1]);
            }

            col -= 2;
        }

        // Convert bits to bytes
        var bytes = new byte[(bits.Count + 7) / 8];
        for (int i = 0; i < bits.Count; i++)
        {
            if (bits[i])
                bytes[i / 8] |= (byte)(0x80 >> (i % 8));
        }

        return bytes;
    }

    #endregion

    #region Error Correction Decode

    private static byte[]? DeinterleaveAndCorrect(byte[] codewords, QRTables.ECBlockInfo ecInfo)
    {
        // For now, skip error correction decode (Reed-Solomon syndrome computation)
        // and just extract the data codewords in order.
        // Full RS decode would correct up to ecCodewordsPerBlock/2 errors per block.

        // The codewords are interleaved: data blocks first, then EC blocks.
        int totalBlocks = ecInfo.Group1Blocks + ecInfo.Group2Blocks;
        int ecPerBlock = ecInfo.ECCodewordsPerBlock;
        int totalDataCodewords = ecInfo.TotalDataCodewords;

        if (codewords.Length < totalDataCodewords + totalBlocks * ecPerBlock)
        {
            // Not enough data — use what we have
            if (codewords.Length < totalDataCodewords)
                return null;
        }

        // Deinterleave data codewords
        var dataBlocks = new byte[totalBlocks][];
        for (int i = 0; i < ecInfo.Group1Blocks; i++)
            dataBlocks[i] = new byte[ecInfo.Group1DataCodewords];
        for (int i = 0; i < ecInfo.Group2Blocks; i++)
            dataBlocks[ecInfo.Group1Blocks + i] = new byte[ecInfo.Group2DataCodewords];

        int maxLen = dataBlocks.Max(b => b.Length);
        int idx = 0;
        for (int col = 0; col < maxLen; col++)
        {
            for (int block = 0; block < totalBlocks; block++)
            {
                if (col < dataBlocks[block].Length && idx < codewords.Length)
                    dataBlocks[block][col] = codewords[idx++];
            }
        }

        // Concatenate data blocks
        var result = new byte[totalDataCodewords];
        int offset = 0;
        foreach (var block in dataBlocks)
        {
            System.Array.Copy(block, 0, result, offset, block.Length);
            offset += block.Length;
        }

        return result;
    }

    #endregion

    #region Data Decoding

    private static string? DecodeData(byte[] data)
    {
        int bitPos = 0;

        int ReadBits(int count)
        {
            int val = 0;
            for (int i = 0; i < count; i++)
            {
                int byteIdx = bitPos / 8;
                int bitIdx = 7 - (bitPos % 8);
                if (byteIdx < data.Length)
                    val = (val << 1) | ((data[byteIdx] >> bitIdx) & 1);
                else
                    val <<= 1;
                bitPos++;
            }
            return val;
        }

        var result = new System.Text.StringBuilder();

        while (bitPos < data.Length * 8 - 4)
        {
            int mode = ReadBits(4);
            if (mode == 0) break; // terminator

            int version = 1; // TODO: pass version for correct char count bits

            if (mode == 4) // Byte mode
            {
                int charCount = ReadBits(8); // version 1-9
                for (int i = 0; i < charCount; i++)
                {
                    int b = ReadBits(8);
                    result.Append((char)b);
                }
            }
            else if (mode == 2) // Alphanumeric
            {
                int charCount = ReadBits(9); // version 1-9
                for (int i = 0; i < charCount - 1; i += 2)
                {
                    int pair = ReadBits(11);
                    result.Append(AlphanumericChar(pair / 45));
                    result.Append(AlphanumericChar(pair % 45));
                }
                if (charCount % 2 == 1)
                {
                    int single = ReadBits(6);
                    result.Append(AlphanumericChar(single));
                }
            }
            else if (mode == 1) // Numeric
            {
                int charCount = ReadBits(10); // version 1-9
                int remaining = charCount;
                while (remaining >= 3)
                {
                    int triple = ReadBits(10);
                    result.Append(triple.ToString("D3"));
                    remaining -= 3;
                }
                if (remaining == 2)
                {
                    int pair = ReadBits(7);
                    result.Append(pair.ToString("D2"));
                }
                else if (remaining == 1)
                {
                    int single = ReadBits(4);
                    result.Append(single.ToString());
                }
            }
            else
            {
                break; // unsupported mode (ECI, kanji, etc.)
            }
        }

        return result.Length > 0 ? result.ToString() : null;
    }

    private static char AlphanumericChar(int value) => value switch
    {
        >= 0 and <= 9 => (char)('0' + value),
        >= 10 and <= 35 => (char)('A' + value - 10),
        36 => ' ', 37 => '$', 38 => '%', 39 => '*',
        40 => '+', 41 => '-', 42 => '.', 43 => '/', 44 => ':',
        _ => '?'
    };

    #endregion
}
