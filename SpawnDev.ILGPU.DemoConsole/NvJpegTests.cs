using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using SpawnDev.UnitTesting;
using System;
using System.Linq;
using System.Threading.Tasks;

/// <summary>
/// NvJpeg encoding and decoding tests. CUDA-only.
/// </summary>
public class NvJpegTests : IDisposable
{
    private Context? _context;
    private CudaAccelerator? _accelerator;
    private NvJpeg? _nvjpeg;
    private NvJpegLibrary? _library;

    private async Task EnsureInitialized()
    {
        if (_accelerator != null) return;

        _context = Context.Create(builder => builder.AllAccelerators());
        var cudaDevices = _context.GetCudaDevices();
        if (cudaDevices.Count == 0)
        {
            _context.Dispose();
            _context = null;
            throw new UnsupportedTestException("No CUDA devices found");
        }
        _accelerator = (CudaAccelerator)cudaDevices[0].CreateAccelerator(_context);
        _nvjpeg = new NvJpeg();
        _library = _nvjpeg.CreateSimple();
        await Task.CompletedTask;
    }

    public void Dispose()
    {
        _library?.Dispose();
        _accelerator?.Dispose();
        _context?.Dispose();
    }

    [TestMethod]
    public async Task NvJpegVersionTest()
    {
        await EnsureInitialized();
        var major = _nvjpeg!.MajorVersion;
        var minor = _nvjpeg.MinorVersion;
        var patch = _nvjpeg.PatchVersion;
        if (major < 0)
            throw new Exception($"Invalid NvJpeg major version: {major}");
    }

    [TestMethod]
    public async Task NvJpegEncodeDecodeRoundtripTest()
    {
        await EnsureInitialized();

        int width = 64;
        int height = 64;
        int numComponents = 3;

        // Create a test image: solid red (R=255, G=0, B=0) in planar RGB
        var redChannel = new byte[width * height];
        var greenChannel = new byte[width * height];
        var blueChannel = new byte[width * height];
        Array.Fill(redChannel, (byte)255);
        Array.Fill(greenChannel, (byte)0);
        Array.Fill(blueChannel, (byte)0);

        // Allocate GPU buffers for source image
        var sourceImage = NvJpegImage.Create(_accelerator!, width, height, numComponents);
        sourceImage.Channel[0]!.View.CopyFromCPU(redChannel);
        sourceImage.Channel[1]!.View.CopyFromCPU(greenChannel);
        sourceImage.Channel[2]!.View.CopyFromCPU(blueChannel);

        // Create encoder state and params
        using var encoderState = _library!.CreateEncoderState();
        using var encoderParams = _library.CreateEncoderParams();

        // Set quality and subsampling
        NvJpegException.ThrowIfFailed(
            _library.EncoderParamsSetQuality(encoderParams, 95));
        NvJpegException.ThrowIfFailed(
            _library.EncoderParamsSetSamplingFactors(encoderParams,
                NvJpegChromaSubsampling.NVJPEG_CSS_444));

        // Encode the image
        NvJpegException.ThrowIfFailed(
            _library.EncodeImage(
                encoderState,
                encoderParams,
                sourceImage,
                NvJpegInputFormat.NVJPEG_INPUT_RGB,
                width,
                height));

        // Retrieve bitstream size (first call with empty span)
        NvJpegException.ThrowIfFailed(
            _library.EncodeRetrieveBitstream(
                encoderState,
                Span<byte>.Empty,
                out ulong bitstreamSize));

        if (bitstreamSize == 0)
            throw new Exception("Encoded JPEG bitstream size is 0");

        // Retrieve actual bitstream
        var jpegBytes = new byte[bitstreamSize];
        NvJpegException.ThrowIfFailed(
            _library.EncodeRetrieveBitstream(
                encoderState,
                jpegBytes.AsSpan(),
                out ulong actualSize));

        // Verify JPEG header (SOI marker: FF D8)
        if (jpegBytes[0] != 0xFF || jpegBytes[1] != 0xD8)
            throw new Exception($"Invalid JPEG header: {jpegBytes[0]:X2} {jpegBytes[1]:X2}");

        // Get image info from encoded JPEG
        var infoStatus = _library.GetImageInfo(
            jpegBytes.AsSpan(),
            out int decodedComponents,
            out NvJpegChromaSubsampling decodedSubsampling,
            out int[] decodedWidths,
            out int[] decodedHeights);
        NvJpegException.ThrowIfFailed(infoStatus);

        if (decodedWidths[0] != width || decodedHeights[0] != height)
            throw new Exception(
                $"Decoded dimensions mismatch: expected {width}x{height}, " +
                $"got {decodedWidths[0]}x{decodedHeights[0]}");

        // Decode the JPEG back
        var decodedImage = NvJpegImage.Create(_accelerator!, width, height, numComponents);
        using var state = _library.CreateState();
        NvJpegException.ThrowIfFailed(
            _library.Decode(
                state,
                jpegBytes.AsSpan(),
                NvJpegOutputFormat.NVJPEG_OUTPUT_RGB,
                decodedImage));

        _accelerator!.Synchronize();

        // Read back decoded channels
        var decodedRed = new byte[width * height];
        var decodedGreen = new byte[width * height];
        var decodedBlue = new byte[width * height];
        decodedImage.Channel[0]!.View.CopyToCPU(decodedRed);
        decodedImage.Channel[1]!.View.CopyToCPU(decodedGreen);
        decodedImage.Channel[2]!.View.CopyToCPU(decodedBlue);

        // JPEG is lossy — allow tolerance of 10 for quality=95
        int tolerance = 10;
        for (int i = 0; i < width * height; i++)
        {
            if (Math.Abs(decodedRed[i] - 255) > tolerance)
                throw new Exception($"Red channel mismatch at pixel {i}: expected ~255, got {decodedRed[i]}");
            if (Math.Abs(decodedGreen[i] - 0) > tolerance)
                throw new Exception($"Green channel mismatch at pixel {i}: expected ~0, got {decodedGreen[i]}");
            if (Math.Abs(decodedBlue[i] - 0) > tolerance)
                throw new Exception($"Blue channel mismatch at pixel {i}: expected ~0, got {decodedBlue[i]}");
        }

        // Dispose source and decoded image buffers
        for (int i = 0; i < numComponents; i++)
        {
            sourceImage.Channel[i]?.Dispose();
            decodedImage.Channel[i]?.Dispose();
        }

        await Task.CompletedTask;
    }

    [TestMethod]
    public async Task NvJpegEncodeYUVRoundtripTest()
    {
        await EnsureInitialized();

        int width = 32;
        int height = 32;
        int numComponents = 3;

        // Create a test image: solid green (R=0, G=255, B=0) in planar RGB
        var redChannel = new byte[width * height];
        var greenChannel = new byte[width * height];
        var blueChannel = new byte[width * height];
        Array.Fill(redChannel, (byte)0);
        Array.Fill(greenChannel, (byte)255);
        Array.Fill(blueChannel, (byte)0);

        var sourceImage = NvJpegImage.Create(_accelerator!, width, height, numComponents);
        sourceImage.Channel[0]!.View.CopyFromCPU(redChannel);
        sourceImage.Channel[1]!.View.CopyFromCPU(greenChannel);
        sourceImage.Channel[2]!.View.CopyFromCPU(blueChannel);

        using var encoderState = _library!.CreateEncoderState();
        using var encoderParams = _library.CreateEncoderParams();

        NvJpegException.ThrowIfFailed(
            _library.EncoderParamsSetQuality(encoderParams, 90));
        NvJpegException.ThrowIfFailed(
            _library.EncoderParamsSetSamplingFactors(encoderParams,
                NvJpegChromaSubsampling.NVJPEG_CSS_444));

        // Encode using EncodeImage (RGB input)
        NvJpegException.ThrowIfFailed(
            _library.EncodeImage(
                encoderState,
                encoderParams,
                sourceImage,
                NvJpegInputFormat.NVJPEG_INPUT_RGB,
                width,
                height));

        // Retrieve bitstream
        NvJpegException.ThrowIfFailed(
            _library.EncodeRetrieveBitstream(
                encoderState,
                Span<byte>.Empty,
                out ulong bitstreamSize));

        var jpegBytes = new byte[bitstreamSize];
        NvJpegException.ThrowIfFailed(
            _library.EncodeRetrieveBitstream(
                encoderState,
                jpegBytes.AsSpan(),
                out _));

        // Decode and verify
        var decodedImage = NvJpegImage.Create(_accelerator!, width, height, numComponents);
        using var decodeState = _library.CreateState();
        NvJpegException.ThrowIfFailed(
            _library.Decode(
                decodeState,
                jpegBytes.AsSpan(),
                NvJpegOutputFormat.NVJPEG_OUTPUT_RGB,
                decodedImage));

        _accelerator!.Synchronize();

        var decodedGreen = new byte[width * height];
        decodedImage.Channel[1]!.View.CopyToCPU(decodedGreen);

        // Green channel should be close to 255
        int tolerance = 15;
        double avgGreen = decodedGreen.Select(b => (double)b).Average();
        if (Math.Abs(avgGreen - 255.0) > tolerance)
            throw new Exception($"Green channel average mismatch: expected ~255, got {avgGreen:F1}");

        for (int i = 0; i < numComponents; i++)
        {
            sourceImage.Channel[i]?.Dispose();
            decodedImage.Channel[i]?.Dispose();
        }

        await Task.CompletedTask;
    }

    [TestMethod]
    public async Task NvJpegEncodeWithOptimizedHuffmanTest()
    {
        await EnsureInitialized();

        int width = 64;
        int height = 64;
        int numComponents = 3;

        // Create a gradient image for better Huffman test
        var redChannel = new byte[width * height];
        var greenChannel = new byte[width * height];
        var blueChannel = new byte[width * height];
        for (int i = 0; i < width * height; i++)
        {
            redChannel[i] = (byte)(i % 256);
            greenChannel[i] = (byte)((i * 2) % 256);
            blueChannel[i] = (byte)((i * 3) % 256);
        }

        var sourceImage = NvJpegImage.Create(_accelerator!, width, height, numComponents);
        sourceImage.Channel[0]!.View.CopyFromCPU(redChannel);
        sourceImage.Channel[1]!.View.CopyFromCPU(greenChannel);
        sourceImage.Channel[2]!.View.CopyFromCPU(blueChannel);

        // Encode WITHOUT optimized Huffman
        using var state1 = _library!.CreateEncoderState();
        using var params1 = _library.CreateEncoderParams();
        NvJpegException.ThrowIfFailed(_library.EncoderParamsSetQuality(params1, 85));
        NvJpegException.ThrowIfFailed(_library.EncoderParamsSetOptimizedHuffman(params1, 0));
        NvJpegException.ThrowIfFailed(
            _library.EncodeImage(state1, params1, sourceImage,
                NvJpegInputFormat.NVJPEG_INPUT_RGB, width, height));

        NvJpegException.ThrowIfFailed(
            _library.EncodeRetrieveBitstream(state1, Span<byte>.Empty, out ulong size1));
        var bytes1 = new byte[size1];
        NvJpegException.ThrowIfFailed(
            _library.EncodeRetrieveBitstream(state1, bytes1.AsSpan(), out _));

        // Encode WITH optimized Huffman
        using var state2 = _library.CreateEncoderState();
        using var params2 = _library.CreateEncoderParams();
        NvJpegException.ThrowIfFailed(_library.EncoderParamsSetQuality(params2, 85));
        NvJpegException.ThrowIfFailed(_library.EncoderParamsSetOptimizedHuffman(params2, 1));
        NvJpegException.ThrowIfFailed(
            _library.EncodeImage(state2, params2, sourceImage,
                NvJpegInputFormat.NVJPEG_INPUT_RGB, width, height));

        NvJpegException.ThrowIfFailed(
            _library.EncodeRetrieveBitstream(state2, Span<byte>.Empty, out ulong size2));
        var bytes2 = new byte[size2];
        NvJpegException.ThrowIfFailed(
            _library.EncodeRetrieveBitstream(state2, bytes2.AsSpan(), out _));

        // Optimized Huffman should produce equal or smaller output
        if (size2 > size1)
            throw new Exception(
                $"Optimized Huffman produced larger output: {size2} > {size1}");

        // Both should be valid JPEGs
        if (bytes1[0] != 0xFF || bytes1[1] != 0xD8)
            throw new Exception("Non-optimized: invalid JPEG header");
        if (bytes2[0] != 0xFF || bytes2[1] != 0xD8)
            throw new Exception("Optimized: invalid JPEG header");

        for (int i = 0; i < numComponents; i++)
            sourceImage.Channel[i]?.Dispose();

        await Task.CompletedTask;
    }

    [TestMethod]
    public async Task NvJpegEncodeBaselineDCTTest()
    {
        await EnsureInitialized();

        int width = 32;
        int height = 32;
        int numComponents = 3;

        var redChannel = new byte[width * height];
        var greenChannel = new byte[width * height];
        var blueChannel = new byte[width * height];
        Array.Fill(redChannel, (byte)128);
        Array.Fill(greenChannel, (byte)64);
        Array.Fill(blueChannel, (byte)192);

        var sourceImage = NvJpegImage.Create(_accelerator!, width, height, numComponents);
        sourceImage.Channel[0]!.View.CopyFromCPU(redChannel);
        sourceImage.Channel[1]!.View.CopyFromCPU(greenChannel);
        sourceImage.Channel[2]!.View.CopyFromCPU(blueChannel);

        using var encoderState = _library!.CreateEncoderState();
        using var encoderParams = _library.CreateEncoderParams();
        NvJpegException.ThrowIfFailed(
            _library.EncoderParamsSetQuality(encoderParams, 90));
        NvJpegException.ThrowIfFailed(
            _library.EncoderParamsSetEncoding(encoderParams,
                NvJpegJpegEncoding.NVJPEG_ENCODING_BASELINE_DCT));

        NvJpegException.ThrowIfFailed(
            _library.EncodeImage(encoderState, encoderParams, sourceImage,
                NvJpegInputFormat.NVJPEG_INPUT_RGB, width, height));

        NvJpegException.ThrowIfFailed(
            _library.EncodeRetrieveBitstream(encoderState, Span<byte>.Empty, out ulong size));

        if (size == 0)
            throw new Exception("Baseline DCT encoding produced zero-length output");

        var jpegBytes = new byte[size];
        NvJpegException.ThrowIfFailed(
            _library.EncodeRetrieveBitstream(encoderState, jpegBytes.AsSpan(), out _));

        // Verify JPEG SOI marker
        if (jpegBytes[0] != 0xFF || jpegBytes[1] != 0xD8)
            throw new Exception($"Invalid JPEG header: {jpegBytes[0]:X2} {jpegBytes[1]:X2}");

        // Decode and verify pixel values
        var decodedImage = NvJpegImage.Create(_accelerator!, width, height, numComponents);
        using var decodeState = _library.CreateState();
        NvJpegException.ThrowIfFailed(
            _library.Decode(decodeState, jpegBytes.AsSpan(),
                NvJpegOutputFormat.NVJPEG_OUTPUT_RGB, decodedImage));
        _accelerator!.Synchronize();

        var decodedRed = new byte[width * height];
        decodedImage.Channel[0]!.View.CopyToCPU(decodedRed);

        int tolerance = 10;
        double avgRed = decodedRed.Select(b => (double)b).Average();
        if (Math.Abs(avgRed - 128.0) > tolerance)
            throw new Exception($"Baseline DCT red channel avg: expected ~128, got {avgRed:F1}");

        for (int i = 0; i < numComponents; i++)
        {
            sourceImage.Channel[i]?.Dispose();
            decodedImage.Channel[i]?.Dispose();
        }

        await Task.CompletedTask;
    }
}
