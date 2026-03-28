using ILGPU;
using ILGPU.Runtime;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.Demo.Shared.UnitTests
{
    // Part 11b: DelegateSpecialization tests (Feature 2)
    public abstract partial class BackendTestBase
    {
        // Target methods for delegate specialization
        static int Negate(int x) => -x;
        static int DoubleIt(int x) => x * 2;

        // Kernel that uses DelegateSpecialization
        static void MapKernel(
            Index1D index,
            ArrayView<int> buf,
            DelegateSpecialization<Func<int, int>> transform)
        {
            buf[index] = transform.Value(buf[index]);
        }

        /// <summary>
        /// Tests basic DelegateSpecialization with a static method target.
        /// </summary>
        [TestMethod]
        public async Task DelegateSpecializationBasicTest() => await RunTest(async accelerator =>
        {
            int len = 64;
            using var buf = accelerator.Allocate1D<int>(len);

            // Initialize buffer with 1..64
            var initKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, int>(MyKernel);
            initKernel((Index1D)len, buf.View, 1); // buf[i] = i + 1
            await accelerator.SynchronizeAsync();

            // Apply Negate via DelegateSpecialization
            var mapKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<int>, DelegateSpecialization<Func<int, int>>>(MapKernel);
            mapKernel((Index1D)len, buf.View,
                new DelegateSpecialization<Func<int, int>>(Negate));
            await accelerator.SynchronizeAsync();

            var result = await buf.CopyToHostAsync<int>();
            for (int i = 0; i < len; i++)
            {
                int expected = -(i + 1);
                if (result[i] != expected)
                    throw new Exception($"DelegateSpec failed at {i}. Expected {expected}, got {result[i]}");
            }
        });
        /// <summary>
        /// Tests that different delegate targets produce correct results
        /// (validates kernel recompilation/caching per target).
        /// </summary>
        [TestMethod]
        public async Task DelegateSpecializationCacheTest() => await RunTest(async accelerator =>
        {
            int len = 64;
            using var buf = accelerator.Allocate1D<int>(len);

            // Initialize with 1..64
            var initKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, int>(MyKernel);
            initKernel((Index1D)len, buf.View, 1);
            await accelerator.SynchronizeAsync();

            var mapKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<int>, DelegateSpecialization<Func<int, int>>>(MapKernel);

            // Apply Negate
            mapKernel((Index1D)len, buf.View,
                new DelegateSpecialization<Func<int, int>>(Negate));
            await accelerator.SynchronizeAsync();
            var result1 = await buf.CopyToHostAsync<int>();
            for (int i = 0; i < len; i++)
                if (result1[i] != -(i + 1))
                    throw new Exception($"Negate failed at {i}. Expected {-(i + 1)}, got {result1[i]}");

            // Now apply DoubleIt to the negated values
            mapKernel((Index1D)len, buf.View,
                new DelegateSpecialization<Func<int, int>>(DoubleIt));
            await accelerator.SynchronizeAsync();
            var result2 = await buf.CopyToHostAsync<int>();
            for (int i = 0; i < len; i++)
            {
                int expected = -(i + 1) * 2;
                if (result2[i] != expected)
                    throw new Exception($"DoubleIt failed at {i}. Expected {expected}, got {result2[i]}");
            }
        });

        /// <summary>
        /// Tests DelegateSpecialization with a static helper method
        /// (equivalent to a non-capturing lambda).
        /// </summary>
        static int AddHundred(int x) => x + 100;

        [TestMethod]
        public async Task DelegateSpecializationStaticHelperTest() => await RunTest(async accelerator =>
        {
            int len = 64;
            using var buf = accelerator.Allocate1D<int>(len);

            var initKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, int>(MyKernel);
            initKernel((Index1D)len, buf.View, 0);
            await accelerator.SynchronizeAsync();

            var mapKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<int>, DelegateSpecialization<Func<int, int>>>(MapKernel);

            mapKernel((Index1D)len, buf.View,
                new DelegateSpecialization<Func<int, int>>(AddHundred));
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<int>();
            for (int i = 0; i < len; i++)
            {
                int expected = i + 100;
                if (result[i] != expected)
                    throw new Exception($"StaticHelper spec failed at {i}. Expected {expected}, got {result[i]}");
            }
        });

        /// <summary>
        /// Tests that a lambda target throws NotSupportedException
        /// (only static methods are supported as DelegateSpecialization targets).
        /// </summary>
        [TestMethod]
        public async Task DelegateSpecializationLambdaRejectTest() => await RunTest(async accelerator =>
        {
            bool threw = false;
            try
            {
                // Non-capturing lambdas are instance methods on <>c
                var spec = new DelegateSpecialization<Func<int, int>>(x => x + 1);
            }
            catch (NotSupportedException)
            {
                threw = true;
            }
            if (!threw)
                throw new Exception("Expected NotSupportedException for lambda delegate target");
            await Task.CompletedTask;
        });

        // ═══════════════════════════════════════════════════════════
        //  Multi-argument delegate specialization (2-arg Func)
        // ═══════════════════════════════════════════════════════════

        static float FloatAdd(float a, float b) => a + b;
        static float FloatMul(float a, float b) => a * b;

        static void BinaryMapKernel(
            Index1D index,
            ArrayView<float> a,
            ArrayView<float> b,
            ArrayView<float> output,
            DelegateSpecialization<Func<float, float, float>> op)
        {
            output[index] = op.Value(a[index], b[index]);
        }

        /// <summary>
        /// Tests DelegateSpecialization with a 2-argument Func (binary operation).
        /// This is the pattern needed for general N-D broadcast kernels.
        /// </summary>
        [TestMethod]
        public async Task DelegateSpecialization_BinaryFunc_Add() => await RunTest(async accelerator =>
        {
            int len = 64;
            var aData = new float[len];
            var bData = new float[len];
            for (int i = 0; i < len; i++) { aData[i] = i + 1; bData[i] = 100; }

            using var aBuf = accelerator.Allocate1D(aData);
            using var bBuf = accelerator.Allocate1D(bData);
            using var outBuf = accelerator.Allocate1D<float>(len);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
                DelegateSpecialization<Func<float, float, float>>>(BinaryMapKernel);

            kernel((Index1D)len, aBuf.View, bBuf.View, outBuf.View,
                new DelegateSpecialization<Func<float, float, float>>(FloatAdd));
            await accelerator.SynchronizeAsync();

            var result = await outBuf.CopyToHostAsync<float>();
            for (int i = 0; i < len; i++)
            {
                float expected = (i + 1) + 100;
                if (MathF.Abs(result[i] - expected) > 0.001f)
                    throw new Exception($"BinaryFunc Add failed at {i}. Expected {expected}, got {result[i]}");
            }
        });

        [TestMethod]
        public async Task DelegateSpecialization_BinaryFunc_Mul() => await RunTest(async accelerator =>
        {
            int len = 64;
            var aData = new float[len];
            var bData = new float[len];
            for (int i = 0; i < len; i++) { aData[i] = i + 1; bData[i] = 2; }

            using var aBuf = accelerator.Allocate1D(aData);
            using var bBuf = accelerator.Allocate1D(bData);
            using var outBuf = accelerator.Allocate1D<float>(len);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
                DelegateSpecialization<Func<float, float, float>>>(BinaryMapKernel);

            kernel((Index1D)len, aBuf.View, bBuf.View, outBuf.View,
                new DelegateSpecialization<Func<float, float, float>>(FloatMul));
            await accelerator.SynchronizeAsync();

            var result = await outBuf.CopyToHostAsync<float>();
            for (int i = 0; i < len; i++)
            {
                float expected = (i + 1) * 2;
                if (MathF.Abs(result[i] - expected) > 0.001f)
                    throw new Exception($"BinaryFunc Mul failed at {i}. Expected {expected}, got {result[i]}");
            }
        });

        // ═══════════════════════════════════════════════════════════
        //  Complex kernel: DelegateSpecialization + ArrayView1D + int strides
        //  This is the pattern used by N-D broadcast kernels in SpawnDev.ILGPU.ML.
        //  The C# compiler may store the delegate in a local variable when the
        //  kernel has many typed parameters — the IL rewriter must handle this.
        // ═══════════════════════════════════════════════════════════

        static float StridedAdd(float a, float b) => a + b;
        static float StridedDiv(float a, float b) => b != 0f ? a / b : 0f;

        /// <summary>
        /// Complex kernel with 5 ArrayView params + DelegateSpecialization.
        /// Reproduces the broadcast binary kernel pattern from SpawnDev.ILGPU.ML.
        /// </summary>
        static void StridedBroadcastKernel(Index1D idx,
            ArrayView1D<float, Stride1D.Dense> a,
            ArrayView1D<float, Stride1D.Dense> b,
            ArrayView1D<float, Stride1D.Dense> output,
            ArrayView1D<int, Stride1D.Dense> strides,
            DelegateSpecialization<Func<float, float, float>> op)
        {
            // Use strides to compute broadcast indices (simplified: just use direct index)
            int rank = strides[0];
            int aIdx = idx;
            int bIdx = idx;
            if (rank > 0 && strides.Length > 1)
            {
                // Simple stride: a uses direct index, b uses modular index
                bIdx = idx % strides[1];
            }
            output[idx] = op.Value(a[aIdx], b[bIdx]);
        }

        [TestMethod]
        public async Task DelegateSpecialization_ComplexKernel_StridedBroadcast() => await RunTest(async accelerator =>
        {
            int len = 64;
            var aData = new float[len];
            var bData = new float[len];
            for (int i = 0; i < len; i++) { aData[i] = i + 1; bData[i] = 100; }

            using var aBuf = accelerator.Allocate1D(aData);
            using var bBuf = accelerator.Allocate1D(bData);
            using var outBuf = accelerator.Allocate1D<float>(len);
            // strides: [rank=0] (no broadcast, just direct index)
            using var stridesBuf = accelerator.Allocate1D(new int[] { 0 });

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
                DelegateSpecialization<Func<float, float, float>>>(StridedBroadcastKernel);

            // Test with Add
            kernel((Index1D)len, aBuf.View, bBuf.View, outBuf.View, stridesBuf.View,
                new DelegateSpecialization<Func<float, float, float>>(StridedAdd));
            await accelerator.SynchronizeAsync();

            var result = await outBuf.CopyToHostAsync<float>();
            for (int i = 0; i < len; i++)
            {
                float expected = (i + 1) + 100;
                if (MathF.Abs(result[i] - expected) > 0.001f)
                    throw new Exception($"StridedBroadcast Add failed at {i}. Expected {expected}, got {result[i]}");
            }

            // Test with Div (reuses same kernel, different specialization)
            for (int i = 0; i < len; i++) bData[i] = 2;
            aBuf.View.CopyFromCPU(aData);
            bBuf.View.CopyFromCPU(bData);

            kernel((Index1D)len, aBuf.View, bBuf.View, outBuf.View, stridesBuf.View,
                new DelegateSpecialization<Func<float, float, float>>(StridedDiv));
            await accelerator.SynchronizeAsync();

            result = await outBuf.CopyToHostAsync<float>();
            for (int i = 0; i < len; i++)
            {
                float expected = (i + 1) / 2f;
                if (MathF.Abs(result[i] - expected) > 0.001f)
                    throw new Exception($"StridedBroadcast Div failed at {i}. Expected {expected}, got {result[i]}");
            }
        });

        // ═══════════════════════════════════════════════════════════
        //  Exact reproduction of SpawnDev.ILGPU.ML BroadcastBinaryKernel.
        //  This is the pattern that crashed on WebGPU due to the
        //  DelegateSpecializationRewriter skipNextStore bug.
        //  The for loop + stride computation + op.Value() is the key pattern.
        // ═══════════════════════════════════════════════════════════

        static float BroadcastSub(float a, float b) => a - b;

        /// <summary>
        /// Mirrors the exact BroadcastBinaryKernel from SpawnDev.ILGPU.ML.
        /// For loop over rank, stride-based index computation, op.Value() at the end.
        /// This was the kernel that crashed on WebGPU.
        /// </summary>
        static void FullBroadcastBinaryKernel(Index1D idx,
            ArrayView1D<float, Stride1D.Dense> a,
            ArrayView1D<float, Stride1D.Dense> b,
            ArrayView1D<float, Stride1D.Dense> output,
            ArrayView1D<int, Stride1D.Dense> strides,
            DelegateSpecialization<Func<float, float, float>> op)
        {
            // strides layout: [rank, aStrides[0..rank], bStrides[0..rank], outStrides[0..rank]]
            int rank = strides[0];
            int aIdx = 0, bIdx = 0, remaining = idx;
            for (int d = 0; d < rank; d++)
            {
                int outStride = strides[1 + 2 * rank + d];
                int coord = outStride > 0 ? remaining / outStride : 0;
                remaining = outStride > 0 ? remaining % outStride : remaining;
                aIdx += coord * strides[1 + d];
                bIdx += coord * strides[1 + rank + d];
            }
            output[idx] = op.Value(a[aIdx], b[bIdx]);
        }

        [TestMethod]
        public async Task DelegateSpecialization_FullBroadcast_Sub() => await RunTest(async accelerator =>
        {
            // Test: [4,3] - [4,1] = per-row scalar broadcast
            // a = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
            // b = [[10],[20],[30],[40]]
            // result = [[-9,-8,-7],[-16,-15,-14],[-23,-22,-21],[-30,-29,-28]]
            int rows = 4, cols = 3;
            int totalA = rows * cols;
            int totalB = rows; // broadcast: b has 1 col per row
            var aData = new float[totalA];
            var bData = new float[totalB];
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                    aData[r * cols + c] = r * cols + c + 1;
                bData[r] = (r + 1) * 10;
            }

            using var aBuf = accelerator.Allocate1D(aData);
            using var bBuf = accelerator.Allocate1D(bData);
            using var outBuf = accelerator.Allocate1D<float>(totalA);

            // strides: [rank=2, aStrides=[cols,1], bStrides=[1,0], outStrides=[cols,1]]
            var stridesData = new int[]
            {
                2,          // rank
                cols, 1,    // aStrides: [3, 1]
                1, 0,       // bStrides: [1, 0] — b broadcasts along cols
                cols, 1,    // outStrides: [3, 1]
            };
            using var stridesBuf = accelerator.Allocate1D(stridesData);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
                DelegateSpecialization<Func<float, float, float>>>(FullBroadcastBinaryKernel);

            kernel((Index1D)totalA, aBuf.View, bBuf.View, outBuf.View, stridesBuf.View,
                new DelegateSpecialization<Func<float, float, float>>(BroadcastSub));
            await accelerator.SynchronizeAsync();

            var result = await outBuf.CopyToHostAsync<float>();
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    int i = r * cols + c;
                    float expected = aData[i] - bData[r];
                    if (MathF.Abs(result[i] - expected) > 0.001f)
                        throw new Exception(
                            $"FullBroadcast Sub failed at [{r},{c}]. Expected {expected}, got {result[i]}");
                }
            }
        });
    }
}
