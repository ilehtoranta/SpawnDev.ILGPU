using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.Demo.Shared.UnitTests
{
    /// <summary>
    /// Tests for GpuMatrix4x4 and early-return variable scoping.
    /// </summary>
    public abstract partial class BackendTestBase
    {
        #region Early Return Kernels

        /// <summary>
        /// Kernel with early return — the simplest case that triggers the
        /// "unresolved value" WGSL bug when variables are declared inside if blocks.
        /// </summary>
        static void EarlyReturnKernel(Index1D index, ArrayView<int> output, int threshold)
        {
            if (index >= threshold)
            {
                output[index] = -1;
                return;
            }
            // Variables computed after early return point
            int doubled = index * 2;
            int result = doubled + 10;
            output[index] = result;
        }

        /// <summary>
        /// Kernel with multiple early returns setting different variables.
        /// Tests that variables from different branches are all properly scoped.
        /// </summary>
        static void MultipleEarlyReturnKernel(Index1D index, ArrayView<int> output)
        {
            if (index == 0)
            {
                output[index] = 100;
                return;
            }
            if (index == 1)
            {
                output[index] = 200;
                return;
            }
            if (index == 2)
            {
                output[index] = 300;
                return;
            }
            // Default path
            int val = index * 10;
            output[index] = val;
        }

        /// <summary>
        /// Kernel with early return where variables are computed before the guard
        /// and used after the merge point.
        /// </summary>
        static void EarlyReturnWithComputeKernel(Index1D index, ArrayView<float> input, ArrayView<float> output)
        {
            float val = input[index];
            float squared = val * val;

            // Early return for negative values
            if (val < 0f)
            {
                output[index] = -1f;
                return;
            }

            // Use computed values after the early return
            float result = squared + val + 1.0f;
            output[index] = result;
        }

        #endregion

        #region GpuMatrix4x4 Kernels

        /// <summary>
        /// Tests GpuMatrix4x4 as a kernel struct parameter.
        /// Transforms 3D points using the matrix.
        /// </summary>
        static void GpuMatrix4x4TransformKernel(
            Index1D index,
            ArrayView<float> positions,  // x,y,z interleaved
            ArrayView<float> results,    // rx,ry,rz interleaved
            GpuMatrix4x4 matrix)
        {
            float x = positions[index * 3 + 0];
            float y = positions[index * 3 + 1];
            float z = positions[index * 3 + 2];

            GpuMatrix4x4.TransformPoint(matrix, x, y, z, out float rx, out float ry, out float rz);

            results[index * 3 + 0] = rx;
            results[index * 3 + 1] = ry;
            results[index * 3 + 2] = rz;
        }

        #endregion

        #region Early Return Tests

        [TestMethod]
        public async Task EarlyReturnTest()
        {
            await RunTest(async accelerator =>
            {
                int size = 8;
                int threshold = 4;
                var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, int>(EarlyReturnKernel);

                using var buffer = accelerator.Allocate1D<int>(size);
                kernel(size, buffer.View, threshold);
                await accelerator.SynchronizeAsync();

                var result = await buffer.CopyToHostAsync<int>();

                // Indices 0-3: doubled + 10
                for (int i = 0; i < threshold; i++)
                {
                    int expected = i * 2 + 10;
                    if (result[i] != expected)
                        throw new Exception($"[{BackendName}] EarlyReturn: index {i} expected {expected}, got {result[i]}");
                }
                // Indices 4-7: -1 (early return)
                for (int i = threshold; i < size; i++)
                {
                    if (result[i] != -1)
                        throw new Exception($"[{BackendName}] EarlyReturn: index {i} expected -1, got {result[i]}");
                }
            });
        }

        [TestMethod]
        public async Task MultipleEarlyReturnTest()
        {
            await RunTest(async accelerator =>
            {
                int size = 8;
                var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(MultipleEarlyReturnKernel);

                using var buffer = accelerator.Allocate1D<int>(size);
                kernel(size, buffer.View);
                await accelerator.SynchronizeAsync();

                var result = await buffer.CopyToHostAsync<int>();

                int[] expected = { 100, 200, 300, 30, 40, 50, 60, 70 };
                for (int i = 0; i < size; i++)
                {
                    if (result[i] != expected[i])
                        throw new Exception($"[{BackendName}] MultipleEarlyReturn: index {i} expected {expected[i]}, got {result[i]}");
                }
            });
        }

        [TestMethod]
        public async Task EarlyReturnWithComputeTest()
        {
            await RunTest(async accelerator =>
            {
                int size = 4;
                var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(EarlyReturnWithComputeKernel);

                float[] inputData = { 2f, -1f, 3f, 0f };
                using var inputBuf = accelerator.Allocate1D(inputData);
                using var outputBuf = accelerator.Allocate1D<float>(size);
                kernel(size, inputBuf.View, outputBuf.View);
                await accelerator.SynchronizeAsync();

                var result = await outputBuf.CopyToHostAsync<float>();

                // index 0: val=2, squared=4, result = 4+2+1 = 7
                // index 1: val=-1 (negative), result = -1
                // index 2: val=3, squared=9, result = 9+3+1 = 13
                // index 3: val=0, squared=0, result = 0+0+1 = 1
                float[] expected = { 7f, -1f, 13f, 1f };
                for (int i = 0; i < size; i++)
                {
                    if (MathF.Abs(result[i] - expected[i]) > 0.01f)
                        throw new Exception($"[{BackendName}] EarlyReturnWithCompute: index {i} expected {expected[i]}, got {result[i]}");
                }
            });
        }

        #endregion

        #region GpuMatrix4x4 Tests

        [TestMethod]
        public async Task GpuMatrix4x4IdentityTest()
        {
            await RunTest(async accelerator =>
            {
                int numPoints = 4;
                var kernel = accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<float>, ArrayView<float>, GpuMatrix4x4>(GpuMatrix4x4TransformKernel);

                float[] positions = {
                    1f, 0f, 0f,
                    0f, 1f, 0f,
                    0f, 0f, 1f,
                    3f, 4f, 5f
                };

                var identity = GpuMatrix4x4.FromMatrix4x4(System.Numerics.Matrix4x4.Identity);

                using var posBuf = accelerator.Allocate1D(positions);
                using var resBuf = accelerator.Allocate1D<float>(numPoints * 3);
                kernel(numPoints, posBuf.View, resBuf.View, identity);
                await accelerator.SynchronizeAsync();

                var result = await resBuf.CopyToHostAsync<float>();

                // Identity should not change points
                for (int i = 0; i < positions.Length; i++)
                {
                    if (MathF.Abs(result[i] - positions[i]) > 0.001f)
                        throw new Exception($"[{BackendName}] GpuMatrix4x4Identity: index {i} expected {positions[i]}, got {result[i]}");
                }
            });
        }

        [TestMethod]
        public async Task GpuMatrix4x4TranslationTest()
        {
            await RunTest(async accelerator =>
            {
                int numPoints = 2;
                var kernel = accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<float>, ArrayView<float>, GpuMatrix4x4>(GpuMatrix4x4TransformKernel);

                float[] positions = {
                    0f, 0f, 0f,
                    1f, 2f, 3f
                };

                // Translation by (10, 20, 30)
                var translation = System.Numerics.Matrix4x4.CreateTranslation(10f, 20f, 30f);
                var gpuMatrix = GpuMatrix4x4.FromMatrix4x4(translation);

                using var posBuf = accelerator.Allocate1D(positions);
                using var resBuf = accelerator.Allocate1D<float>(numPoints * 3);
                kernel(numPoints, posBuf.View, resBuf.View, gpuMatrix);
                await accelerator.SynchronizeAsync();

                var result = await resBuf.CopyToHostAsync<float>();

                // Point (0,0,0) + (10,20,30) = (10,20,30)
                float[] expected = { 10f, 20f, 30f, 11f, 22f, 33f };
                for (int i = 0; i < expected.Length; i++)
                {
                    if (MathF.Abs(result[i] - expected[i]) > 0.001f)
                        throw new Exception($"[{BackendName}] GpuMatrix4x4Translation: index {i} expected {expected[i]}, got {result[i]}");
                }
            });
        }

        [TestMethod]
        public async Task GpuMatrix4x4LookAtTest()
        {
            await RunTest(async accelerator =>
            {
                int numPoints = 1;
                var kernel = accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<float>, ArrayView<float>, GpuMatrix4x4>(GpuMatrix4x4TransformKernel);

                // Point at origin
                float[] positions = { 0f, 0f, 0f };

                // Camera at (0,0,5) looking at origin
                var viewMatrix = System.Numerics.Matrix4x4.CreateLookAt(
                    new System.Numerics.Vector3(0, 0, 5),
                    System.Numerics.Vector3.Zero,
                    System.Numerics.Vector3.UnitY);
                var gpuMatrix = GpuMatrix4x4.FromMatrix4x4(viewMatrix);

                using var posBuf = accelerator.Allocate1D(positions);
                using var resBuf = accelerator.Allocate1D<float>(numPoints * 3);
                kernel(numPoints, posBuf.View, resBuf.View, gpuMatrix);
                await accelerator.SynchronizeAsync();

                var result = await resBuf.CopyToHostAsync<float>();

                // Origin in camera space of camera at (0,0,5) looking at origin:
                // x=0, y=0, z=-5 (point is 5 units in front of camera)
                if (MathF.Abs(result[0]) > 0.01f)
                    throw new Exception($"[{BackendName}] GpuMatrix4x4LookAt: x expected 0, got {result[0]}");
                if (MathF.Abs(result[1]) > 0.01f)
                    throw new Exception($"[{BackendName}] GpuMatrix4x4LookAt: y expected 0, got {result[1]}");
                if (MathF.Abs(result[2] - (-5f)) > 0.01f)
                    throw new Exception($"[{BackendName}] GpuMatrix4x4LookAt: z expected -5, got {result[2]}");
            });
        }

        #endregion
    }
}
