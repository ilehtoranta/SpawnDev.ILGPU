using System.Diagnostics;
using global::ILGPU;
using global::ILGPU.Runtime;

namespace SpawnDev.ILGPU.Demo.Shared.Benchmarks;

/// <summary>
/// Defines a single benchmark test.
/// </summary>
public record BenchmarkDef(string Id, string Name, string Icon, string Description);

/// <summary>
/// Configures workload sizes for a given backend.
/// </summary>
public record BackendProfile(string Name, int DefaultN, int MandelbrotSize, int PrimesN);

/// <summary>
/// Result of a single benchmark run for one backend.
/// </summary>
public record BenchmarkResult
{
    public string BackendName { get; init; } = "";
    public string BenchmarkName { get; init; } = "";

    /// <summary>Trimmed median of measured runs (ms).</summary>
    public double? MedianMs { get; init; }

    /// <summary>Formatted throughput string (e.g., "1.23 Melem/s").</summary>
    public string? Throughput { get; init; }

    /// <summary>Raw throughput in elements/sec for chart rendering.</summary>
    public double ThroughputElemPerSec { get; init; }

    /// <summary>Number of elements processed per kernel invocation.</summary>
    public int ElementCount { get; init; }

    /// <summary>Total warmup time in ms (includes shader compilation, JIT).</summary>
    public double WarmupMs { get; init; }

    /// <summary>All raw measured times in ms (before trimming).</summary>
    public double[]? AllTimesMs { get; init; }

    public bool Failed { get; init; }
    public string? Error { get; init; }
}

/// <summary>
/// Standard benchmark definitions — identical across all platforms.
/// </summary>
public static class BenchmarkDefs
{
    public static readonly BenchmarkDef[] All =
    [
        new("mandelbrot", "Mandelbrot", "🎨", "512×512 fractal, 200 iterations — compute-bound"),
        new("vecadd", "Vector Add", "➕", "a[i] = b[i] + c[i] — raw memory throughput"),
        new("vecscale", "Vector Scale", "✖", "a[i] = a[i] * scalar — scalar argument marshaling"),
        new("saxpy", "SAXPY", "📐", "y[i] = α·x[i] + y[i] — classic GPU benchmark"),
        new("primes", "Prime Count", "🔢", "Count primes up to N via trial division — integer ALU-heavy"),
    ];

    /// <summary>
    /// Standard workload sizes for cross-platform comparison.
    /// Both browser and desktop backends use these same values.
    /// </summary>
    public static readonly BackendProfile StandardProfile = new("Standard", 1_000_000, 512, 1_000_000);
}
