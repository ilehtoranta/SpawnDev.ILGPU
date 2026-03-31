# GPU-Accelerated QR Codes

**SpawnDev.ILGPU.QR** — a zero-dependency QR code encoder, decoder, and GPU renderer built into SpawnDev.ILGPU. Generates standard QR codes that any mobile device can scan, with optional logo overlay and GPU-accelerated rendering.

To our knowledge, this is the first GPU-accelerated QR code library written entirely in C#, and the first to run across 7 compute backends — WebGPU, WebGL, Wasm, CUDA, OpenCL, CPU, and P2P — from a single codebase. Every component (Reed-Solomon error correction, Galois Field GF(256) arithmetic, encoder, decoder, renderer) is pure C# with zero external dependencies. Browser and desktop, one library.

## Features

- **Full QR spec** — All 40 versions, 4 error correction levels (L/M/Q/H), byte mode encoding
- **GPU-accelerated rendering** — ILGPU kernel renders QR modules to pixels on any backend (WebGPU, WebGL, Wasm, CUDA, OpenCL, CPU)
- **Logo overlay** — Embed an image in the center of the QR code using EC level H (30% error correction)
- **Decoder** — Read QR codes from pixel data with GPU-accelerated grayscale conversion
- **Round-trip verified** — Encode → render → decode = exact match across all backends
- **Zero dependencies** — Pure C#, no external libraries. Reed-Solomon, Galois Field, everything built-in
- **Cross-platform** — Works in browser (Blazor WASM) and desktop (Console, WPF, ASP.NET)

## Quick Start

### Generate a QR Code (CPU)

```csharp
using SpawnDev.ILGPU.QR;

// Simple: text → pixels
var (pixels, width, height) = QRCode.Generate(
    "https://hub.spawndev.com",
    ecLevel: QRTables.ECLevel.M,
    moduleSize: 10,
    quietZone: 4);

// pixels is ARGB byte array (4 bytes per pixel, row-major)
// width x height image, ready for display
```

### Generate a QR Code (GPU-Accelerated)

```csharp
// GPU render — data stays on the accelerator
var (pixels, width, height) = await QRCode.GenerateAsync(
    accelerator,
    "https://hub.spawndev.com",
    ecLevel: QRTables.ECLevel.M,
    moduleSize: 10);
```

### Generate with Logo Overlay

```csharp
// Automatically uses EC level H (30% error correction)
// so the QR code remains scannable with the center obscured
var (pixels, width, height) = await QRCode.GenerateWithLogoAsync(
    accelerator,
    "https://hub.spawndev.com",
    logoPixels,    // ARGB byte array
    logoWidth,
    logoHeight,
    moduleSize: 8,
    logoPadding: 4);
```

### Decode a QR Code (CPU)

```csharp
// From RGBA pixel data (e.g., camera frame, rendered image)
string? decoded = QRCode.Decode(rgbaPixels, width, height);
if (decoded != null)
    Console.WriteLine($"QR code says: {decoded}");
```

### Decode a QR Code (GPU-Accelerated)

```csharp
// GPU grayscale conversion, then CPU decode pipeline
string? decoded = await QRCode.DecodeAsync(accelerator, rgbaPixels, width, height);
```

### Encode to Matrix Only (No Rendering)

```csharp
// Get the raw module matrix (true = dark, false = light)
bool[,] modules = QRCode.EncodeMatrix("Hello World", QRTables.ECLevel.Q);
int size = modules.GetLength(0); // e.g., 21 for Version 1
```

## Architecture

### Encoding Pipeline

```
Text/URL
  → UTF-8 bytes
  → Mode indicator (0100 = byte mode) + character count
  → Pad to data capacity
  → Split into EC blocks (per version/level table)
  → Reed-Solomon error correction per block (GF(256))
  → Interleave data + EC codewords
  → Build QR matrix:
      - Finder patterns (3 corners)
      - Timing patterns (row 6, col 6)
      - Alignment patterns (version 2+)
      - Dark module
      - Format info (EC level + mask)
      - Version info (version 7+)
  → Place data bits in zigzag pattern
  → Evaluate all 8 mask patterns (penalty scoring)
  → Apply best mask
  → Write format/version info
  → Final QR matrix (bool[,])
```

### Rendering Pipeline

```
QR matrix (bool[,])
  → GPU kernel: each thread maps one pixel to a module
     - Pixel → module coordinate (accounting for quiet zone + module size)
     - Module → dark/light color
  → ARGB pixel buffer on GPU
  → Optional: logo overlay (clear center + composite)
  → CopyToHostAsync or display directly
```

### Decoding Pipeline

```
RGBA pixels
  → GPU kernel: grayscale conversion (luminance)
  → Adaptive binarization (local threshold)
  → Finder pattern detection (1:1:3:1:1 ratio scan)
  → Triangle identification (top-left at right angle)
  → Version estimation (distance between finders / module size)
  → Affine grid sampling (finder centers → module coordinates)
  → Format info reading (15-bit BCH, Hamming distance match)
  → Unmask data modules
  → Zigzag data bit extraction
  → Deinterleave EC blocks
  → Data decoding (byte/numeric/alphanumeric modes)
```

## Components

### GaloisField

Finite field GF(256) arithmetic for Reed-Solomon error correction.

```csharp
// Primitive polynomial: x^8 + x^4 + x^3 + x^2 + 1 (285)
byte product = GaloisField.Multiply(a, b);    // O(1) via log/antilog tables
byte quotient = GaloisField.Divide(a, b);
byte[] ec = GaloisField.ComputeEC(data, ecCount);  // Reed-Solomon EC codewords
byte[] generator = GaloisField.BuildGeneratorPolynomial(ecCount);
```

### QRTables

All QR specification tables:

- **EC block info** for all 40 versions x 4 levels (data codewords, EC per block, block counts)
- **Alignment pattern positions** for all 40 versions
- **Format info strings** (32 pre-computed 15-bit BCH values)
- **Version info strings** (34 pre-computed 18-bit values for versions 7-40)
- **Character count indicator** bit lengths per version range and mode

```csharp
var ecInfo = QRTables.GetECInfo(version: 5, QRTables.ECLevel.H);
// ecInfo.TotalDataCodewords = 46
// ecInfo.ECCodewordsPerBlock = 22
// ecInfo.Group1Blocks = 2, Group1DataCodewords = 11
// ecInfo.Group2Blocks = 2, Group2DataCodewords = 12

int size = QRTables.ModuleCount(version: 5); // 37 (4*5 + 17)
```

### QREncoder

Full encoding pipeline from text to QR module matrix.

```csharp
bool[,] matrix = QREncoder.Encode("https://spawndev.com", QRTables.ECLevel.M);
int size = QREncoder.GetSize("https://spawndev.com"); // module count
```

### QRRenderer

GPU kernel + CPU fallback for pixel rendering.

```csharp
// GPU render
var (pixels, w, h) = await QRRenderer.RenderAsync(
    accelerator, modules, moduleSize: 10, quietZone: 4,
    darkColor: 0xFF000000, lightColor: 0xFFFFFFFF);

// CPU render
var (pixels, w, h) = QRRenderer.RenderCpu(
    modules, moduleSize: 10, quietZone: 4);

// Logo overlay (modifies pixels in place)
QRRenderer.ApplyLogo(pixels, imageWidth, logoPixels, logoWidth, logoHeight, padding: 4);
```

### QRDecoder

Decode QR codes from pixel data.

```csharp
// CPU decode
string? text = QRDecoder.Decode(rgbaPixels, width, height);

// GPU-accelerated decode (grayscale kernel on GPU)
string? text = await QRDecoder.DecodeAsync(accelerator, rgbaPixels, width, height);
```

## Error Correction Levels

| Level | Recovery | Best For |
|-------|----------|----------|
| **L** (Low) | ~7% | Clean digital display |
| **M** (Medium) | ~15% | General use (default) |
| **Q** (Quartile) | ~25% | Moderate damage tolerance |
| **H** (High) | ~30% | Logo overlay, harsh environments |

Use **EC level H** when adding a logo — the 30% redundancy allows the center to be obscured while the QR code remains scannable by any standard reader.

## GPU Kernel Details

### Render Kernel

Each GPU thread handles one pixel:

```csharp
static void RenderKernel(
    Index1D idx,
    ArrayView<int> modules,     // flattened QR matrix
    ArrayView<uint> pixels,     // output ARGB
    int qrSize, int moduleSize, int quietZone,
    uint darkColor, uint lightColor)
{
    int px = idx % imageSize;
    int py = idx / imageSize;
    int moduleX = px / moduleSize - quietZone;
    int moduleY = py / moduleSize - quietZone;

    // Out of bounds = quiet zone (light)
    if (moduleX < 0 || moduleX >= qrSize || moduleY < 0 || moduleY >= qrSize)
        pixels[idx] = lightColor;
    else
        pixels[idx] = modules[moduleY * qrSize + moduleX] != 0 ? darkColor : lightColor;
}
```

For a Version 5 QR code at moduleSize=10 with quietZone=4: the image is 450x450 = 202,500 pixels, each computed independently in parallel.

### Grayscale Kernel (Decoder)

```csharp
static void GrayscaleKernel(Index1D idx, ArrayView<int> rgba, ArrayView<byte> gray)
{
    int pixel = rgba[idx];
    int r = pixel & 0xFF;
    int g = (pixel >> 8) & 0xFF;
    int b = (pixel >> 16) & 0xFF;
    gray[idx] = (byte)((r * 77 + g * 150 + b * 29) >> 8);  // ITU-R BT.601 luminance
}
```

## Customization

### Colors

```csharp
// Custom colors (ARGB uint)
var (pixels, w, h) = QRCode.Generate(text,
    darkColor: 0xFF1E3A5F,    // dark navy
    lightColor: 0xFFF8F9FA);  // light gray
```

### Module Size

```csharp
// Small QR code (4px per module — good for thumbnails)
var (pixels, w, h) = QRCode.Generate(text, moduleSize: 4);

// Large QR code (20px per module — good for print)
var (pixels, w, h) = QRCode.Generate(text, moduleSize: 20);
```

### Quiet Zone

The quiet zone (white border) is required by the QR spec for reliable scanning. The standard is 4 modules. You can reduce it for tight layouts, but scanners may have difficulty.

```csharp
// Standard quiet zone
var (pixels, w, h) = QRCode.Generate(text, quietZone: 4);

// Minimal quiet zone (may reduce scan reliability)
var (pixels, w, h) = QRCode.Generate(text, quietZone: 1);
```

## Real-World Usage in SpawnDev.ILGPU

The QR library powers the P2P Compute Swarm demo:

- **Join link QR code** — generated when a swarm is created, displayed in the dashboard
- **Camera QR scanner** — captures camera frames, runs `QRDecoder.Decode` at 2Hz, auto-joins when a swarm link is detected
- **Mobile-friendly** — standard QR codes scannable by any phone camera app

```csharp
// In Blazor component: generate QR code for join link
var (pixels, w, h) = QRCode.Generate(compute.JoinLink, QRTables.ECLevel.H, moduleSize: 6);

// Render to canvas → data URL → <img> element
using var canvas = new OffscreenCanvas(w, h);
using var ctx = canvas.Get2DContext();
ctx.PutImageBytes(pixels, w, h);
using var blob = await canvas.ConvertToBlob();
var dataUrl = await blob.ToDataURLAsync();
```

## Test Coverage

14 tests across all 7 backends:

| Test | What It Verifies |
|------|-----------------|
| `QR_GaloisField_ExpLogRoundTrip` | GF(256) log/antilog table consistency |
| `QR_GaloisField_Multiply` | Multiplication, division, round-trip |
| `QR_GaloisField_ReedSolomon` | EC codeword generation |
| `QR_Encode_Version1` | Minimum QR matrix, finder patterns |
| `QR_Encode_URL` | Realistic URL encoding |
| `QR_Encode_ECLevelH_LargerThanM` | Higher EC = larger version |
| `QR_Encode_AllECLevels` | All 4 levels produce valid matrices |
| `QR_Render_CPU` | CPU pixel rendering, quiet zone |
| `QR_Render_GPU` | GPU kernel rendering |
| `QR_Render_GPU_WithLogo` | Logo overlay, center pixel verification |
| `QR_Render_GPU_CPUMatch` | GPU and CPU produce identical output |
| `QR_Decode_RoundTrip` | Encode → render → decode = exact match |
| `QR_Decode_RoundTrip_WithLogo` | Round-trip with logo (EC level H) |
| `QR_Encode_LongURL` | Realistic P2P join link encoding |
