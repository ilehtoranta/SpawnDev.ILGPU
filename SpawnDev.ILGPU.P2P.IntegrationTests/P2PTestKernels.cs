using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.P2P.IntegrationTests;

/// <summary>
/// GPU kernels used by all P2P integration tests.
/// Every test that dispatches a kernel uses one of these
/// and verifies results mathematically - no sampling, no approximation.
/// </summary>
public static class P2PTestKernels
{
    /// <summary>
    /// result[i] = a[i] + b[i]
    /// Standard test: a[i]=i, b[i]=i*2, expected result[i]=i*3
    /// </summary>
    public static void VectorAdd(Index1D index,
        ArrayView<float> a, ArrayView<float> b, ArrayView<float> result)
    {
        result[index] = a[index] + b[index];
    }

    /// <summary>
    /// result[i] = input[i] * scalar
    /// Tests scalar parameter serialization across WebRTC.
    /// </summary>
    public static void VectorScale(Index1D index,
        ArrayView<float> input, ArrayView<float> result, float scalar)
    {
        result[index] = input[index] * scalar;
    }

    /// <summary>
    /// result[i] = input[i]
    /// Pure data integrity test - any bit flip causes failure.
    /// </summary>
    public static void Identity(Index1D index,
        ArrayView<int> input, ArrayView<int> result)
    {
        result[index] = input[index];
    }

    /// <summary>
    /// buf[i] = (float)i
    /// No input buffer - generates data on the worker.
    /// Used as pipeline stage 1 (generate) before stage 2 (transform).
    /// </summary>
    public static void FillSequence(Index1D index, ArrayView<float> buf)
    {
        buf[index] = (float)(int)index;
    }

    /// <summary>
    /// result[i] = input[i] * 2
    /// Simple integer kernel for non-float type testing.
    /// </summary>
    public static void IntDoubler(Index1D index,
        ArrayView<int> input, ArrayView<int> result)
    {
        result[index] = input[index] * 2;
    }
}
