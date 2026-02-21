using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Blazor.UnitTesting;

namespace SpawnDev.ILGPU.Demo.UnitTests
{
    // Part 4: Tests inspired by ILGPU reference samples — expanded atomics, XMath, explicit grouping,
    //         prefix sum, dot product, while loops, multi-buffer, float rem, IsNaN/IsInf, and more.
    public abstract partial class BackendTestBase
    {
        #region Part 4 Kernel Definitions

        // --- Atomic And/Or/Xor (from SimpleAtomics sample) ---
        static void AtomicAndOrXorKernel(Index1D index, ArrayView<int> data, int constant)
        {
            Atomic.And(ref data[0], constant);
            Atomic.Or(ref data[1], constant);
            Atomic.Xor(ref data[2], constant);
        }

        // --- XMath Functions (from AlgorithmsMath sample) ---
        static void XMathFunctionsKernel(Index1D index, ArrayView<float> output, float c)
        {
            // XMath functions compile down to IR arithmetic nodes
            float sinhVal = MathF.Sinh(c + index);
            float atanVal = MathF.Atan(c);
            float expVal = MathF.Exp(c * 0.1f);
            float logVal = MathF.Log(1.0f + index);
            output[index] = sinhVal + atanVal + expVal + logVal;
        }

        // --- IntrinsicMath.Clamp (from SimpleMath sample) ---
        static void IntrinsicMathClampKernel(Index1D index, ArrayView<float> output)
        {
            float val = index * 3.0f - 10.0f; // range: -10, -7, -4, -1, 2, 5, 8, 11
            output[index] = IntrinsicMath.Clamp(val, 0.0f, 10.0f);
        }

        // --- Explicitly Grouped Kernel (from ExplicitlyGroupedKernels sample) ---
        static void ExplicitGroupKernel(ArrayView<int> data, int constant)
        {
            var globalIndex = Grid.GlobalIndex.X;
            if (globalIndex < data.Length)
                data[globalIndex] = globalIndex + constant;
        }

        // --- Prefix Sum using Shared Memory ---
        static void PrefixSumKernel(ArrayView<int> data)
        {
            var tid = Group.IdxX;
            var shared = SharedMemory.Allocate<int>(64);
            shared[tid] = data[tid];
            Group.Barrier();

            // Inclusive prefix sum (Hillis-Steele)
            for (int offset = 1; offset < 64; offset *= 2)
            {
                int val = 0;
                if (tid >= offset)
                    val = shared[tid - offset];
                Group.Barrier();
                if (tid >= offset)
                    shared[tid] += val;
                Group.Barrier();
            }

            data[tid] = shared[tid];
        }

        // --- Dot Product with Shared Memory + Atomic ---
        static void DotProductKernel(
            ArrayView<float> a,
            ArrayView<float> b,
            ArrayView<float> result)
        {
            var tid = Group.IdxX;
            var shared = SharedMemory.Allocate<float>(64);

            shared[tid] = a[tid] * b[tid];
            Group.Barrier();

            // Tree reduction within the group
            for (int stride = 32; stride > 0; stride >>= 1)
            {
                if (tid < stride)
                    shared[tid] += shared[tid + stride];
                Group.Barrier();
            }

            if (tid == 0)
                Atomic.Add(ref result[0], shared[0]);
        }

        // --- While Loop (Collatz sequence length) ---
        static void WhileLoopKernel(Index1D index, ArrayView<int> output)
        {
            int n = index + 1; // start from 1
            int count = 0;
            while (n > 1)
            {
                if (n % 2 == 0)
                    n = n / 2;
                else
                    n = 3 * n + 1;
                count++;
            }
            output[index] = count;
        }

        // --- Multi-Buffer Input ---
        static void MultiBufferInputKernel(
            Index1D index,
            ArrayView<float> a,
            ArrayView<float> b,
            ArrayView<float> c,
            ArrayView<float> output)
        {
            output[index] = a[index] + b[index] * c[index];
        }

        // --- Conditional Expression ---
        static void ConditionalExpressionKernel(Index1D index, ArrayView<int> output)
        {
            int val = index;
            // Nested ternary expressions
            int result = val < 4 ? val * 10 :
                         val < 8 ? val * 20 :
                         val < 12 ? val * 30 : val * 40;
            output[index] = result;
        }

        // --- Array Fill Pattern ---
        static void ArrayFillPatternKernel(Index1D index, ArrayView<int> output, int stride, int value)
        {
            int pos = index * stride;
            if (pos < output.Length)
                output[pos] = value;
        }

        // --- Float Rem (verifies Wasm Rem fix) ---
        static void FloatRemKernel(Index1D index, ArrayView<float> output)
        {
            float val = index * 1.5f; // 0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5
            output[index] = val % 4.0f;  // 0, 1.5, 3.0, 0.5, 2.0, 3.5, 1.0, 2.5
        }

        // --- IsNaN/IsInf Test (verifies Wasm fixes) ---
        static void IsNaNIsInfKernel(Index1D index, ArrayView<float> input, ArrayView<int> output)
        {
            float val = input[index];
            int result = 0;
            // Use manual comparison patterns that compile to IsNaNF/IsInfF IR
            if (val != val) result |= 1;  // IsNaN
            if (val * 0.0f != 0.0f) result |= 2; // Another NaN check (NaN * 0 = NaN)
            output[index] = result;
        }

        // --- Bitwise Rotate ---
        static void BitwiseRotateKernel(Index1D index, ArrayView<int> output)
        {
            int val = 0x12345678;
            int shift = index % 32;
            // Rotate left: (val << shift) | (val >>> (32 - shift))
            int rotated = (val << shift) | ((int)((uint)val >> (32 - shift)));
            output[index] = rotated;
        }

        // --- Map kernel for Map-Reduce test ---
        static void MapSquareKernel(Index1D index, ArrayView<float> input, ArrayView<float> output)
        {
            output[index] = input[index] * input[index];
        }

        // --- Reduce kernel for Map-Reduce test ---
        static void ReduceSumKernel(ArrayView<float> data, ArrayView<float> result)
        {
            var tid = Group.IdxX;
            var shared = SharedMemory.Allocate<float>(64);
            shared[tid] = data[tid];
            Group.Barrier();

            for (int stride = 32; stride > 0; stride >>= 1)
            {
                if (tid < stride)
                    shared[tid] += shared[tid + stride];
                Group.Barrier();
            }

            if (tid == 0)
                result[0] = shared[0];
        }

        // --- Saturating Arithmetic (Min/Max clamping) ---
        static void SaturatingArithmeticKernel(Index1D index, ArrayView<float> output)
        {
            float val = (index - 8.0f) * 0.5f; // range: -4, -3.5, -3, ..., 3.5
            // Clamp to [-2, 2] using Math.Min(Math.Max(...))
            float clamped = Math.Min(Math.Max(val, -2.0f), 2.0f);
            // Normalize to [0, 1]
            float normalized = (clamped + 2.0f) * 0.25f;
            output[index] = normalized;
        }

        #endregion

        #region Part 4 Tests

        [TestMethod]
        public async Task AtomicAndOrXorTest() => await RunTest(async accelerator =>
        {
            using var buf = accelerator.Allocate1D<int>(3);
            var initData = new int[] { -1, 0, 0 }; // And identity, Or identity, Xor identity
            buf.CopyFromCPU(initData);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, int>(AtomicAndOrXorKernel);
            // Launch 8 threads with constant = 0b1010 (10)
            kernel((Index1D)8, buf.View, 0b1010);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<int>();

            // And: -1 & 10 & 10 & ... = 10 (all threads And with same value)
            if (result[0] != 0b1010)
                throw new Exception($"Atomic.And failed. Expected {0b1010}, got {result[0]}");
            // Or: 0 | 10 | 10 | ... = 10
            if (result[1] != 0b1010)
                throw new Exception($"Atomic.Or failed. Expected {0b1010}, got {result[1]}");
            // Xor: 0 ^ 10 ^ 10 ^ ... = 0 (even number of XORs cancel out)
            if (result[2] != 0)
                throw new Exception($"Atomic.Xor failed. Expected 0, got {result[2]}");
        });

        [TestMethod]
        public async Task XMathFunctionsTest() => await RunTest(async accelerator =>
        {
            int len = 8;
            float c = 0.1f;
            using var buf = accelerator.Allocate1D<float>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, float>(XMathFunctionsKernel);
            kernel((Index1D)len, buf.View, c);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<float>();

            for (int i = 0; i < len; i++)
            {
                float expected = MathF.Sinh(c + i) + MathF.Atan(c) + MathF.Exp(c * 0.1f) + MathF.Log(1.0f + i);
                if (MathF.Abs(result[i] - expected) > 0.01f * MathF.Abs(expected) + 0.01f)
                    throw new Exception($"XMath failed at {i}. Expected {expected}, got {result[i]}");
            }
        });

        [TestMethod]
        public async Task IntrinsicMathClampTest() => await RunTest(async accelerator =>
        {
            int len = 8;
            using var buf = accelerator.Allocate1D<float>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>>(IntrinsicMathClampKernel);
            kernel((Index1D)len, buf.View);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<float>();

            float[] expected = { 0, 0, 0, 0, 2, 5, 8, 10 };
            for (int i = 0; i < len; i++)
                if (MathF.Abs(result[i] - expected[i]) > 0.01f)
                    throw new Exception($"IntrinsicMath.Clamp failed at {i}. Expected {expected[i]}, got {result[i]}");
        });

        [TestMethod]
        public async Task ExplicitGroupKernelTest() => await RunTest(async accelerator =>
        {
            int len = 64;
            using var buf = accelerator.Allocate1D<int>(len);
            buf.MemSetToZero();

            var kernel = accelerator.LoadStreamKernel<ArrayView<int>, int>(ExplicitGroupKernel);
            // 1 group of 64 threads
            kernel(new KernelConfig(1, 64), buf.View, 100);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<int>();

            for (int i = 0; i < len; i++)
                if (result[i] != i + 100)
                    throw new Exception($"Explicit group kernel failed at {i}. Expected {i + 100}, got {result[i]}");
        });

        [TestMethod]
        public async Task PrefixSumTest() => await RunTest(async accelerator =>
        {
            int len = 64;
            var data = new int[len];
            for (int i = 0; i < len; i++) data[i] = 1; // all ones → prefix sum = 1, 2, 3, ..., 64
            using var buf = accelerator.Allocate1D(data);

            var kernel = accelerator.LoadStreamKernel<ArrayView<int>>(PrefixSumKernel);
            kernel(new KernelConfig(1, 64), buf.View);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<int>();

            for (int i = 0; i < len; i++)
            {
                int expected = i + 1;
                if (result[i] != expected)
                    throw new Exception($"Prefix sum failed at {i}. Expected {expected}, got {result[i]}");
            }
        });

        [TestMethod]
        public async Task DotProductTest() => await RunTest(async accelerator =>
        {
            int len = 64;
            var aData = new float[len];
            var bData = new float[len];
            for (int i = 0; i < len; i++) { aData[i] = 1.0f; bData[i] = 2.0f; }

            using var bufA = accelerator.Allocate1D(aData);
            using var bufB = accelerator.Allocate1D(bData);
            using var bufR = accelerator.Allocate1D<float>(1);
            bufR.MemSetToZero();

            var kernel = accelerator.LoadStreamKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>>(DotProductKernel);
            kernel(new KernelConfig(1, 64), bufA.View, bufB.View, bufR.View);
            await accelerator.SynchronizeAsync();
            var result = await bufR.CopyToHostAsync<float>();

            float expected = 128.0f; // 64 * 1 * 2
            if (MathF.Abs(result[0] - expected) > 0.01f)
                throw new Exception($"Dot product failed. Expected {expected}, got {result[0]}");
        });

        [TestMethod]
        public async Task WhileLoopTest() => await RunTest(async accelerator =>
        {
            int len = 16;
            using var buf = accelerator.Allocate1D<int>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(WhileLoopKernel);
            kernel((Index1D)len, buf.View);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<int>();

            // Compute expected Collatz sequence lengths on CPU
            for (int i = 0; i < len; i++)
            {
                int n = i + 1;
                int count = 0;
                while (n > 1) { n = (n % 2 == 0) ? n / 2 : 3 * n + 1; count++; }
                if (result[i] != count)
                    throw new Exception($"While loop (Collatz) failed at {i}. Expected {count}, got {result[i]}");
            }
        });

        [TestMethod]
        public async Task MultiBufferInputTest() => await RunTest(async accelerator =>
        {
            int len = 32;
            var aData = new float[len]; var bData = new float[len]; var cData = new float[len];
            for (int i = 0; i < len; i++) { aData[i] = i; bData[i] = 2.0f; cData[i] = 0.5f; }

            using var bufA = accelerator.Allocate1D(aData);
            using var bufB = accelerator.Allocate1D(bData);
            using var bufC = accelerator.Allocate1D(cData);
            using var bufOut = accelerator.Allocate1D<float>(len);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>>(MultiBufferInputKernel);
            kernel((Index1D)len, bufA.View, bufB.View, bufC.View, bufOut.View);
            await accelerator.SynchronizeAsync();
            var result = await bufOut.CopyToHostAsync<float>();

            for (int i = 0; i < len; i++)
            {
                float expected = i + 2.0f * 0.5f; // a + b * c
                if (MathF.Abs(result[i] - expected) > 0.01f)
                    throw new Exception($"Multi-buffer failed at {i}. Expected {expected}, got {result[i]}");
            }
        });

        [TestMethod]
        public async Task ConditionalExpressionTest() => await RunTest(async accelerator =>
        {
            int len = 16;
            using var buf = accelerator.Allocate1D<int>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(ConditionalExpressionKernel);
            kernel((Index1D)len, buf.View);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<int>();

            for (int i = 0; i < len; i++)
            {
                int expected = i < 4 ? i * 10 :
                               i < 8 ? i * 20 :
                               i < 12 ? i * 30 : i * 40;
                if (result[i] != expected)
                    throw new Exception($"Conditional expression failed at {i}. Expected {expected}, got {result[i]}");
            }
        });

        [TestMethod]
        public async Task FloatRemTest() => await RunTest(async accelerator =>
        {
            int len = 8;
            using var buf = accelerator.Allocate1D<float>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>>(FloatRemKernel);
            kernel((Index1D)len, buf.View);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<float>();

            float[] expected = { 0f, 1.5f, 3.0f, 0.5f, 2.0f, 3.5f, 1.0f, 2.5f };
            for (int i = 0; i < len; i++)
                if (MathF.Abs(result[i] - expected[i]) > 0.01f)
                    throw new Exception($"Float Rem failed at {i}. Expected {expected[i]}, got {result[i]}");
        });

        [TestMethod]
        public async Task IsNaNIsInfTest() => await RunTest(async accelerator =>
        {
            var input = new float[]
            {
                0.0f,                        // normal
                float.NaN,                   // NaN
                float.PositiveInfinity,      // +Inf
                float.NegativeInfinity,      // -Inf
                1.0f,                        // normal
                -1.0f,                       // normal
                float.NaN,                   // NaN
                42.0f                        // normal
            };

            using var bufIn = accelerator.Allocate1D(input);
            using var bufOut = accelerator.Allocate1D<int>(input.Length);
            bufOut.MemSetToZero();

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<int>>(IsNaNIsInfKernel);
            kernel((Index1D)input.Length, bufIn.View, bufOut.View);
            await accelerator.SynchronizeAsync();
            var result = await bufOut.CopyToHostAsync<int>();

            // Bit 0 = IsNaN (val != val), Bit 1 = NaN via multiply (val * 0 != 0)
            int[] expected = { 0, 3, 2, 2, 0, 0, 3, 0 };
            // Note: Inf * 0 = NaN, so Inf values should have bit 1 set
            for (int i = 0; i < input.Length; i++)
            {
                if (result[i] != expected[i])
                    throw new Exception($"IsNaN/IsInf failed at {i} (input={input[i]}). Expected {expected[i]}, got {result[i]}");
            }
        });

        [TestMethod]
        public async Task BitwiseRotateTest() => await RunTest(async accelerator =>
        {
            int len = 16;
            using var buf = accelerator.Allocate1D<int>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(BitwiseRotateKernel);
            kernel((Index1D)len, buf.View);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<int>();

            int val = 0x12345678;
            for (int i = 0; i < len; i++)
            {
                int shift = i % 32;
                int expected = (val << shift) | (int)((uint)val >> (32 - shift));
                if (result[i] != expected)
                    throw new Exception($"Bitwise rotate failed at {i}. Expected 0x{expected:X8}, got 0x{result[i]:X8}");
            }
        });

        [TestMethod]
        public async Task MapReduceTest() => await RunTest(async accelerator =>
        {
            int len = 64;
            var data = new float[len];
            for (int i = 0; i < len; i++) data[i] = i + 1; // 1..64

            using var bufIn = accelerator.Allocate1D(data);
            using var bufMapped = accelerator.Allocate1D<float>(len);
            using var bufResult = accelerator.Allocate1D<float>(1);
            bufResult.MemSetToZero();

            // Step 1: Map (square each element)
            var mapKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>>(MapSquareKernel);
            mapKernel((Index1D)len, bufIn.View, bufMapped.View);
            await accelerator.SynchronizeAsync();

            // Step 2: Reduce (sum all mapped elements)
            var reduceKernel = accelerator.LoadStreamKernel<ArrayView<float>, ArrayView<float>>(ReduceSumKernel);
            reduceKernel(new KernelConfig(1, 64), bufMapped.View, bufResult.View);
            await accelerator.SynchronizeAsync();

            var result = await bufResult.CopyToHostAsync<float>();

            // Expected: sum of squares 1^2 + 2^2 + ... + 64^2 = 64*65*129/6 = 89440
            float expected = 0;
            for (int i = 1; i <= len; i++) expected += i * i;
            if (MathF.Abs(result[0] - expected) > 1.0f)
                throw new Exception($"Map-Reduce failed. Expected {expected}, got {result[0]}");
        });

        [TestMethod]
        public async Task SaturatingArithmeticTest() => await RunTest(async accelerator =>
        {
            int len = 16;
            using var buf = accelerator.Allocate1D<float>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>>(SaturatingArithmeticKernel);
            kernel((Index1D)len, buf.View);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<float>();

            for (int i = 0; i < len; i++)
            {
                float val = (i - 8.0f) * 0.5f;
                float clamped = MathF.Min(MathF.Max(val, -2.0f), 2.0f);
                float expected = (clamped + 2.0f) * 0.25f;
                if (MathF.Abs(result[i] - expected) > 0.01f)
                    throw new Exception($"Saturating arithmetic failed at {i}. Expected {expected}, got {result[i]}");
            }
        });

        [TestMethod]
        public async Task ArrayFillPatternTest() => await RunTest(async accelerator =>
        {
            int count = 16;
            int stride = 2;
            int totalSize = count * stride;
            using var buf = accelerator.Allocate1D<int>(totalSize);
            buf.MemSetToZero();

            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, int, int>(ArrayFillPatternKernel);
            kernel((Index1D)count, buf.View, stride, 42);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<int>();

            for (int i = 0; i < totalSize; i++)
            {
                int expected = (i % stride == 0) ? 42 : 0;
                if (result[i] != expected)
                    throw new Exception($"Array fill pattern failed at {i}. Expected {expected}, got {result[i]}");
            }
        });

        #endregion
    }
}
