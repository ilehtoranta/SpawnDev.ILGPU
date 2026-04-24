using System.Security.Cryptography;

namespace SpawnDev.ILGPU.DemoConsole.P2PTests;

/// <summary>
/// Verification utilities for P2P integration tests.
/// Every dispatch test must verify ALL elements, not a sample.
/// </summary>
public static class DataIntegrityHelper
{
    /// <summary>
    /// Verify all float elements match expected values within tolerance.
    /// Returns (violations, firstMismatchIndex, firstExpected, firstActual).
    /// </summary>
    public static (int violations, int firstIndex, float firstExpected, float firstActual)
        VerifyFloats(float[] actual, float[] expected, float tolerance = 0.001f)
    {
        if (actual.Length != expected.Length)
            return (Math.Abs(actual.Length - expected.Length), -1, expected.Length, actual.Length);

        int violations = 0;
        int firstIndex = -1;
        float firstExpected = 0, firstActual = 0;

        for (int i = 0; i < expected.Length; i++)
        {
            if (Math.Abs(actual[i] - expected[i]) > tolerance)
            {
                if (violations == 0)
                {
                    firstIndex = i;
                    firstExpected = expected[i];
                    firstActual = actual[i];
                }
                violations++;
            }
        }

        return (violations, firstIndex, firstExpected, firstActual);
    }

    /// <summary>
    /// Verify all int elements match exactly.
    /// </summary>
    public static (int violations, int firstIndex, int firstExpected, int firstActual)
        VerifyInts(int[] actual, int[] expected)
    {
        if (actual.Length != expected.Length)
            return (Math.Abs(actual.Length - expected.Length), -1, expected.Length, actual.Length);

        int violations = 0;
        int firstIndex = -1;
        int firstExpected = 0, firstActual = 0;

        for (int i = 0; i < expected.Length; i++)
        {
            if (actual[i] != expected[i])
            {
                if (violations == 0)
                {
                    firstIndex = i;
                    firstExpected = expected[i];
                    firstActual = actual[i];
                }
                violations++;
            }
        }

        return (violations, firstIndex, firstExpected, firstActual);
    }

    /// <summary>
    /// Compute SHA256 of byte array for data integrity verification.
    /// </summary>
    public static string ComputeSha256(byte[] data)
    {
        var hash = SHA256.HashData(data);
        return Convert.ToHexString(hash).ToLowerInvariant();
    }

    /// <summary>
    /// Generate VectorAdd test data: a[i]=i, b[i]=i*2, expected[i]=i*3.
    /// Returns (aBytes, bBytes, expectedFloats).
    /// </summary>
    public static (byte[] aBytes, byte[] bBytes, float[] expected) GenerateVectorAddData(int n)
    {
        var a = new float[n];
        var b = new float[n];
        var expected = new float[n];
        for (int i = 0; i < n; i++)
        {
            a[i] = i;
            b[i] = i * 2;
            expected[i] = i * 3;
        }
        var aBytes = new byte[n * 4];
        var bBytes = new byte[n * 4];
        Buffer.BlockCopy(a, 0, aBytes, 0, n * 4);
        Buffer.BlockCopy(b, 0, bBytes, 0, n * 4);
        return (aBytes, bBytes, expected);
    }

    /// <summary>
    /// Convert byte[] to float[] via BlockCopy.
    /// </summary>
    public static float[] BytesToFloats(byte[] data)
    {
        var floats = new float[data.Length / 4];
        Buffer.BlockCopy(data, 0, floats, 0, data.Length);
        return floats;
    }

    /// <summary>
    /// Convert byte[] to int[] via BlockCopy.
    /// </summary>
    public static int[] BytesToInts(byte[] data)
    {
        var ints = new int[data.Length / 4];
        Buffer.BlockCopy(data, 0, ints, 0, data.Length);
        return ints;
    }
}
