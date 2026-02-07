using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using SpawnDev.Blazor.UnitTesting;

namespace SpawnDev.ILGPU.Demo.UnitTests
{
    /// <summary>
    /// Unit tests for the ILGPU CPU backend.
    /// These serve as the reference implementation — all other backends
    /// should produce results consistent with these tests.
    /// </summary>
    public class CPUTests
    {
        #region Helper

        /// <summary>
        /// Creates an ILGPU Context + CPU Accelerator.
        /// </summary>
        private static (Context context, Accelerator accelerator) CreateCPUAccelerator()
        {
            var builder = Context.Create().CPU();
            var context = builder.ToContext();
            var accelerator = context.CreateCPUAccelerator(0);
            return (context, accelerator);
        }

        #endregion

        #region Basic Tests

        [TestMethod]
        public async Task CPUAcceleratorBasicTest()
        {
            var (context, accelerator) = CreateCPUAccelerator();
            try
            {
                if (accelerator == null)
                    throw new Exception("CPUAccelerator creation failed");

                Console.WriteLine($"CPU Accelerator: {accelerator.Name}");
                Console.WriteLine($"  MaxNumThreadsPerGroup: {accelerator.MaxNumThreadsPerGroup}");
                Console.WriteLine($"  MemorySize: {accelerator.MemorySize}");

                // Allocate a buffer and verify its size
                using var buffer = accelerator.Allocate1D<int>(1024);
                if (buffer.Length != 1024)
                    throw new Exception($"Buffer length mismatch. Expected 1024, got {buffer.Length}");
            }
            finally
            {
                accelerator.Dispose();
                context.Dispose();
            }
        }

        [TestMethod]
        public async Task CPUBufferTransferTest()
        {
            var (context, accelerator) = CreateCPUAccelerator();
            try
            {
                int length = 128;
                var data = Enumerable.Range(0, length).ToArray();

                using var buffer = accelerator.Allocate1D(data);

                // Read back via unified extension
                var readBack = await buffer.CopyToHostAsync<int>();

                if (readBack.Length != length)
                    throw new Exception($"Readback length mismatch. Expected {length}, got {readBack.Length}");

                for (int i = 0; i < length; i++)
                {
                    if (readBack[i] != data[i])
                        throw new Exception($"Data mismatch at index {i}. Expected {data[i]}, got {readBack[i]}");
                }
            }
            finally
            {
                accelerator.Dispose();
                context.Dispose();
            }
        }

        #endregion

        #region Kernel Tests

        /// <summary>
        /// Simple 1D kernel: output[index] = index + constant
        /// </summary>
        static void MyKernel(Index1D index, ArrayView<int> output, int constant)
        {
            output[index] = index + constant;
        }

        [TestMethod]
        public async Task CPUKernelTest()
        {
            var (context, accelerator) = CreateCPUAccelerator();
            try
            {
                int length = 256;
                int constant = 42;
                using var buffer = accelerator.Allocate1D<int>(length);

                var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, int>(MyKernel);
                kernel(length, buffer.View, constant);

                await accelerator.SynchronizeAsync();
                var results = await buffer.CopyToHostAsync<int>();

                for (int i = 0; i < length; i++)
                {
                    int expected = i + constant;
                    if (results[i] != expected)
                        throw new Exception($"Mismatch at index {i}: expected {expected}, got {results[i]}");
                }
            }
            finally
            {
                accelerator.Dispose();
                context.Dispose();
            }
        }

        /// <summary>
        /// Vector add kernel: C[i] = A[i] + B[i]
        /// </summary>
        static void VectorAddKernel(Index1D index, ArrayView<float> a, ArrayView<float> b, ArrayView<float> c)
        {
            c[index] = a[index] + b[index];
        }

        [TestMethod]
        public async Task CPUVectorAddKernelTest()
        {
            var (context, accelerator) = CreateCPUAccelerator();
            try
            {
                int length = 256;
                var dataA = Enumerable.Range(0, length).Select(i => (float)i).ToArray();
                var dataB = Enumerable.Range(0, length).Select(i => (float)i * 2f).ToArray();

                using var bufA = accelerator.Allocate1D(dataA);
                using var bufB = accelerator.Allocate1D(dataB);
                using var bufC = accelerator.Allocate1D<float>(length);

                var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(VectorAddKernel);
                kernel(length, bufA.View, bufB.View, bufC.View);

                await accelerator.SynchronizeAsync();
                var results = await bufC.CopyToHostAsync<float>();

                for (int i = 0; i < length; i++)
                {
                    float expected = dataA[i] + dataB[i];
                    if (Math.Abs(results[i] - expected) > 0.001f)
                        throw new Exception($"Mismatch at index {i}: expected {expected}, got {results[i]}");
                }
            }
            finally
            {
                accelerator.Dispose();
                context.Dispose();
            }
        }

        /// <summary>
        /// Float kernel: output[i] = i * 0.5f + constant
        /// </summary>
        static void FloatKernel(Index1D index, ArrayView<float> output, float constant)
        {
            output[index] = index * 0.5f + constant;
        }

        [TestMethod]
        public async Task CPUKernelFloatTest()
        {
            var (context, accelerator) = CreateCPUAccelerator();
            try
            {
                int length = 256;
                float constant = 3.14f;
                using var buffer = accelerator.Allocate1D<float>(length);

                var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, float>(FloatKernel);
                kernel(length, buffer.View, constant);

                await accelerator.SynchronizeAsync();
                var results = await buffer.CopyToHostAsync<float>();

                for (int i = 0; i < length; i++)
                {
                    float expected = i * 0.5f + constant;
                    if (Math.Abs(results[i] - expected) > 0.001f)
                        throw new Exception($"Mismatch at index {i}: expected {expected}, got {results[i]}");
                }
            }
            finally
            {
                accelerator.Dispose();
                context.Dispose();
            }
        }

        /// <summary>
        /// Multi-scalar kernel: output[i] = i + a + b
        /// </summary>
        static void MultiScalarKernel(Index1D index, ArrayView<int> output, int a, int b)
        {
            output[index] = index + a + b;
        }

        [TestMethod]
        public async Task CPUMultiScalarKernelTest()
        {
            var (context, accelerator) = CreateCPUAccelerator();
            try
            {
                int length = 256;
                int a = 10, b = 20;
                using var buffer = accelerator.Allocate1D<int>(length);

                var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, int, int>(MultiScalarKernel);
                kernel(length, buffer.View, a, b);

                await accelerator.SynchronizeAsync();
                var results = await buffer.CopyToHostAsync<int>();

                for (int i = 0; i < length; i++)
                {
                    int expected = i + a + b;
                    if (results[i] != expected)
                        throw new Exception($"Mismatch at index {i}: expected {expected}, got {results[i]}");
                }
            }
            finally
            {
                accelerator.Dispose();
                context.Dispose();
            }
        }

        #endregion

        #region Math Tests

        /// <summary>
        /// Math kernel: output[i] = abs(-i) * 2 + min(i, 10)
        /// </summary>
        static void MathKernel(Index1D index, ArrayView<int> output)
        {
            int val = -index;
            int absVal = Math.Abs(val);
            int minVal = Math.Min(index, 10);
            output[index] = absVal * 2 + minVal;
        }

        [TestMethod]
        public async Task CPUMathKernelTest()
        {
            var (context, accelerator) = CreateCPUAccelerator();
            try
            {
                int length = 256;
                using var buffer = accelerator.Allocate1D<int>(length);

                var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(MathKernel);
                kernel(length, buffer.View);

                await accelerator.SynchronizeAsync();
                var results = await buffer.CopyToHostAsync<int>();

                for (int i = 0; i < length; i++)
                {
                    int expected = Math.Abs(-i) * 2 + Math.Min(i, 10);
                    if (results[i] != expected)
                        throw new Exception($"Mismatch at index {i}: expected {expected}, got {results[i]}");
                }
            }
            finally
            {
                accelerator.Dispose();
                context.Dispose();
            }
        }

        /// <summary>
        /// Intrinsic math: sin, cos, tan, sqrt, pow, log, exp, abs
        /// </summary>
        static void IntrinsicMathKernel(Index1D index, ArrayView<float> output)
        {
            float val = (index + 1) * 0.1f;
            float result = 0f;
            result += (float)Math.Sin(val);
            result += (float)Math.Cos(val);
            result += (float)Math.Sqrt(val);
            result += (float)Math.Pow(val, 2.0);
            result += (float)Math.Abs(-val);
            output[index] = result;
        }

        [TestMethod]
        public async Task CPUIntrinsicMathTest()
        {
            var (context, accelerator) = CreateCPUAccelerator();
            try
            {
                int length = 64;
                using var buffer = accelerator.Allocate1D<float>(length);

                var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>>(IntrinsicMathKernel);
                kernel(length, buffer.View);

                await accelerator.SynchronizeAsync();
                var results = await buffer.CopyToHostAsync<float>();

                for (int i = 0; i < length; i++)
                {
                    float val = (i + 1) * 0.1f;
                    float expected = (float)(Math.Sin(val) + Math.Cos(val) + Math.Sqrt(val) + Math.Pow(val, 2.0) + Math.Abs(-val));
                    if (Math.Abs(results[i] - expected) > 0.01f)
                        throw new Exception($"Mismatch at index {i}: expected {expected}, got {results[i]}");
                }
            }
            finally
            {
                accelerator.Dispose();
                context.Dispose();
            }
        }

        /// <summary>
        /// Integer math: abs, min, max, negation
        /// </summary>
        static void IntMathKernel(Index1D index, ArrayView<int> output)
        {
            int i = index;
            int a = Math.Abs(i - 50);
            int b = Math.Min(i, 30);
            int c = Math.Max(i - 10, 0);
            int d = -i;
            output[index] = a + b + c + d;
        }

        [TestMethod]
        public async Task CPUIntMathTest()
        {
            var (context, accelerator) = CreateCPUAccelerator();
            try
            {
                int length = 64;
                using var buffer = accelerator.Allocate1D<int>(length);

                var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(IntMathKernel);
                kernel(length, buffer.View);

                await accelerator.SynchronizeAsync();
                var results = await buffer.CopyToHostAsync<int>();

                for (int i = 0; i < length; i++)
                {
                    int expected = Math.Abs(i - 50) + Math.Min(i, 30) + Math.Max(i - 10, 0) + (-i);
                    if (results[i] != expected)
                        throw new Exception($"Mismatch at index {i}: expected {expected}, got {results[i]}");
                }
            }
            finally
            {
                accelerator.Dispose();
                context.Dispose();
            }
        }

        /// <summary>
        /// Modulo operation: output[i] = i % 7
        /// </summary>
        static void ModuloKernel(Index1D index, ArrayView<int> output) => output[index] = index % 7;

        [TestMethod]
        public async Task CPUModuloOperationsTest()
        {
            var (context, accelerator) = CreateCPUAccelerator();
            try
            {
                int length = 64;
                using var buffer = accelerator.Allocate1D<int>(length);

                var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(ModuloKernel);
                kernel(length, buffer.View);

                await accelerator.SynchronizeAsync();
                var results = await buffer.CopyToHostAsync<int>();

                for (int i = 0; i < length; i++)
                {
                    if (results[i] != i % 7)
                        throw new Exception($"Modulo mismatch at {i}: expected {i % 7}, got {results[i]}");
                }
            }
            finally
            {
                accelerator.Dispose();
                context.Dispose();
            }
        }

        /// <summary>
        /// NaN and Infinity detection
        /// </summary>
        static void NaNInfinityKernel(Index1D index, ArrayView<int> output)
        {
            float nan = float.NaN;
            float inf = float.PositiveInfinity;
            output[index] = (float.IsNaN(nan) ? 1 : 0) + (float.IsInfinity(inf) ? 10 : 0);
        }

        [TestMethod]
        public async Task CPUNaNInfinityTest()
        {
            var (context, accelerator) = CreateCPUAccelerator();
            try
            {
                int length = 16;
                using var buffer = accelerator.Allocate1D<int>(length);

                var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(NaNInfinityKernel);
                kernel(length, buffer.View);

                await accelerator.SynchronizeAsync();
                var results = await buffer.CopyToHostAsync<int>();

                for (int i = 0; i < length; i++)
                {
                    if (results[i] != 11) // 1 (NaN) + 10 (Inf)
                        throw new Exception($"NaN/Inf detection failed at {i}: expected 11, got {results[i]}");
                }
            }
            finally
            {
                accelerator.Dispose();
                context.Dispose();
            }
        }

        #endregion

        #region Control Flow Tests

        /// <summary>
        /// Control flow kernel: tests if/else
        /// </summary>
        static void ControlFlowKernel(Index1D index, ArrayView<int> output)
        {
            if (index % 2 == 0)
                output[index] = index * 2;
            else
                output[index] = index * 3;
        }

        [TestMethod]
        public async Task CPUControlFlowTest()
        {
            var (context, accelerator) = CreateCPUAccelerator();
            try
            {
                int length = 128;
                using var buffer = accelerator.Allocate1D<int>(length);

                var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(ControlFlowKernel);
                kernel(length, buffer.View);

                await accelerator.SynchronizeAsync();
                var results = await buffer.CopyToHostAsync<int>();

                for (int i = 0; i < length; i++)
                {
                    int expected = i % 2 == 0 ? i * 2 : i * 3;
                    if (results[i] != expected)
                        throw new Exception($"Control flow mismatch at {i}: expected {expected}, got {results[i]}");
                }
            }
            finally
            {
                accelerator.Dispose();
                context.Dispose();
            }
        }

        /// <summary>
        /// Nested control flow: for loops + if/else
        /// </summary>
        static void NestedControlFlowKernel(Index1D index, ArrayView<int> output)
        {
            int sum = 0;
            for (int j = 0; j < 10; j++)
            {
                if (j % 2 == 0)
                    sum += j;
                else
                    sum -= 1;
            }
            output[index] = sum + index;
        }

        [TestMethod]
        public async Task CPUNestedControlFlowTest()
        {
            var (context, accelerator) = CreateCPUAccelerator();
            try
            {
                int length = 64;
                using var buffer = accelerator.Allocate1D<int>(length);

                var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(NestedControlFlowKernel);
                kernel(length, buffer.View);

                await accelerator.SynchronizeAsync();
                var results = await buffer.CopyToHostAsync<int>();

                // Compute expected: sum of even j (0+2+4+6+8=20) minus 5 odd j's = 20 - 5 = 15
                int baseExpected = 15;
                for (int i = 0; i < length; i++)
                {
                    int expected = baseExpected + i;
                    if (results[i] != expected)
                        throw new Exception($"Nested flow mismatch at {i}: expected {expected}, got {results[i]}");
                }
            }
            finally
            {
                accelerator.Dispose();
                context.Dispose();
            }
        }

        /// <summary>
        /// Select (ternary) operation
        /// </summary>
        static void SelectKernel(Index1D index, ArrayView<int> output)
        {
            int val = index > 32 ? 100 : -100;
            output[index] = val + index;
        }

        [TestMethod]
        public async Task CPUSelectOperationTest()
        {
            var (context, accelerator) = CreateCPUAccelerator();
            try
            {
                int length = 64;
                using var buffer = accelerator.Allocate1D<int>(length);

                var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(SelectKernel);
                kernel(length, buffer.View);

                await accelerator.SynchronizeAsync();
                var results = await buffer.CopyToHostAsync<int>();

                for (int i = 0; i < length; i++)
                {
                    int expected = (i > 32 ? 100 : -100) + i;
                    if (results[i] != expected)
                        throw new Exception($"Select mismatch at {i}: expected {expected}, got {results[i]}");
                }
            }
            finally
            {
                accelerator.Dispose();
                context.Dispose();
            }
        }

        #endregion

        #region Bitwise Tests

        static void BitwiseKernel(Index1D index, ArrayView<int> output)
        {
            int val = index;
            output[index] = (val & 0xF) | (val << 4) ^ 0xFF;
        }

        [TestMethod]
        public async Task CPUBitwiseTest()
        {
            var (context, accelerator) = CreateCPUAccelerator();
            try
            {
                int length = 64;
                using var buffer = accelerator.Allocate1D<int>(length);

                var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(BitwiseKernel);
                kernel(length, buffer.View);

                await accelerator.SynchronizeAsync();
                var results = await buffer.CopyToHostAsync<int>();

                for (int i = 0; i < length; i++)
                {
                    int expected = (i & 0xF) | (i << 4) ^ 0xFF;
                    if (results[i] != expected)
                        throw new Exception($"Bitwise mismatch at {i}: expected {expected}, got {results[i]}");
                }
            }
            finally
            {
                accelerator.Dispose();
                context.Dispose();
            }
        }

        #endregion

        #region Type Conversion Tests

        static void ConversionKernel(Index1D index, ArrayView<float> output)
        {
            int intVal = index * 3;
            float floatVal = (float)intVal;
            uint uintVal = (uint)index;
            output[index] = floatVal + (float)uintVal;
        }

        [TestMethod]
        public async Task CPUConversionTest()
        {
            var (context, accelerator) = CreateCPUAccelerator();
            try
            {
                int length = 64;
                using var buffer = accelerator.Allocate1D<float>(length);

                var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>>(ConversionKernel);
                kernel(length, buffer.View);

                await accelerator.SynchronizeAsync();
                var results = await buffer.CopyToHostAsync<float>();

                for (int i = 0; i < length; i++)
                {
                    float expected = (float)(i * 3) + (float)(uint)i;
                    if (Math.Abs(results[i] - expected) > 0.001f)
                        throw new Exception($"Conversion mismatch at {i}: expected {expected}, got {results[i]}");
                }
            }
            finally
            {
                accelerator.Dispose();
                context.Dispose();
            }
        }

        #endregion

        #region Struct Tests

        public struct MyPoint
        {
            public float X;
            public float Y;
        }

        static void StructKernel(Index1D index, ArrayView<MyPoint> output)
        {
            output[index] = new MyPoint { X = index * 1.0f, Y = index * 2.0f };
        }

        [TestMethod]
        public async Task CPUStructKernelTest()
        {
            var (context, accelerator) = CreateCPUAccelerator();
            try
            {
                int length = 64;
                using var buffer = accelerator.Allocate1D<MyPoint>(length);

                var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<MyPoint>>(StructKernel);
                kernel(length, buffer.View);

                await accelerator.SynchronizeAsync();
                var results = await buffer.CopyToHostAsync<MyPoint>();

                for (int i = 0; i < length; i++)
                {
                    if (Math.Abs(results[i].X - i * 1.0f) > 0.001f || Math.Abs(results[i].Y - i * 2.0f) > 0.001f)
                        throw new Exception($"Struct mismatch at {i}: expected ({i * 1.0f}, {i * 2.0f}), got ({results[i].X}, {results[i].Y})");
                }
            }
            finally
            {
                accelerator.Dispose();
                context.Dispose();
            }
        }

        #endregion

        #region Function Call Tests

        static int MyAdd(int a, int b) => a + b;

        static void FunctionCallKernel(Index1D index, ArrayView<int> output)
        {
            output[index] = MyAdd(index, 100);
        }

        [TestMethod]
        public async Task CPUFunctionCallTest()
        {
            var (context, accelerator) = CreateCPUAccelerator();
            try
            {
                int length = 64;
                using var buffer = accelerator.Allocate1D<int>(length);

                var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(FunctionCallKernel);
                kernel(length, buffer.View);

                await accelerator.SynchronizeAsync();
                var results = await buffer.CopyToHostAsync<int>();

                for (int i = 0; i < length; i++)
                {
                    int expected = i + 100;
                    if (results[i] != expected)
                        throw new Exception($"Function call mismatch at {i}: expected {expected}, got {results[i]}");
                }
            }
            finally
            {
                accelerator.Dispose();
                context.Dispose();
            }
        }

        #endregion

        #region Comparison Tests

        static void ComparisonKernel(Index1D index, ArrayView<int> output)
        {
            int i = index;
            int result = 0;
            if (i > 10) result += 1;
            if (i < 50) result += 2;
            if (i >= 20) result += 4;
            if (i <= 40) result += 8;
            if (i == 30) result += 16;
            if (i != 25) result += 32;
            output[index] = result;
        }

        [TestMethod]
        public async Task CPUComparisonOperatorsTest()
        {
            var (context, accelerator) = CreateCPUAccelerator();
            try
            {
                int length = 64;
                using var buffer = accelerator.Allocate1D<int>(length);

                var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(ComparisonKernel);
                kernel(length, buffer.View);

                await accelerator.SynchronizeAsync();
                var results = await buffer.CopyToHostAsync<int>();

                for (int i = 0; i < length; i++)
                {
                    int expected = 0;
                    if (i > 10) expected += 1;
                    if (i < 50) expected += 2;
                    if (i >= 20) expected += 4;
                    if (i <= 40) expected += 8;
                    if (i == 30) expected += 16;
                    if (i != 25) expected += 32;
                    if (results[i] != expected)
                        throw new Exception($"Comparison mismatch at {i}: expected {expected}, got {results[i]}");
                }
            }
            finally
            {
                accelerator.Dispose();
                context.Dispose();
            }
        }

        #endregion

        #region GPU Pattern Tests

        /// <summary>
        /// Reduction: sum elements in blocks
        /// </summary>
        static void ReductionKernel(Index1D index, ArrayView<int> input, ArrayView<int> output, int blockSize)
        {
            int start = index * blockSize;
            int sum = 0;
            for (int i = 0; i < blockSize; i++)
            {
                sum += input[start + i];
            }
            output[index] = sum;
        }

        [TestMethod]
        public async Task CPUReductionTest()
        {
            var (context, accelerator) = CreateCPUAccelerator();
            try
            {
                int blockSize = 8;
                int numBlocks = 32;
                int totalLength = blockSize * numBlocks;
                var data = Enumerable.Range(0, totalLength).ToArray();

                using var input = accelerator.Allocate1D(data);
                using var output = accelerator.Allocate1D<int>(numBlocks);

                var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, int>(ReductionKernel);
                kernel(numBlocks, input.View, output.View, blockSize);

                await accelerator.SynchronizeAsync();
                var results = await output.CopyToHostAsync<int>();

                for (int b = 0; b < numBlocks; b++)
                {
                    int expected = 0;
                    for (int i = 0; i < blockSize; i++)
                        expected += b * blockSize + i;
                    if (results[b] != expected)
                        throw new Exception($"Reduction mismatch at block {b}: expected {expected}, got {results[b]}");
                }
            }
            finally
            {
                accelerator.Dispose();
                context.Dispose();
            }
        }

        /// <summary>
        /// Matrix multiply: C[row,col] = sum(A[row,k] * B[k,col])
        /// Uses 1D indexing into flat arrays.
        /// </summary>
        static void MatMulKernel(Index1D index, ArrayView<float> a, ArrayView<float> b, ArrayView<float> c, int n)
        {
            int row = index / n;
            int col = index % n;
            float sum = 0;
            for (int k = 0; k < n; k++)
            {
                sum += a[row * n + k] * b[k * n + col];
            }
            c[index] = sum;
        }

        [TestMethod]
        public async Task CPUMatrixMultiplyTest()
        {
            var (context, accelerator) = CreateCPUAccelerator();
            try
            {
                int n = 8;
                int total = n * n;
                var dataA = new float[total];
                var dataB = new float[total];
                for (int i = 0; i < total; i++)
                {
                    dataA[i] = i;
                    dataB[i] = i * 0.1f;
                }

                using var bufA = accelerator.Allocate1D(dataA);
                using var bufB = accelerator.Allocate1D(dataB);
                using var bufC = accelerator.Allocate1D<float>(total);

                var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int>(MatMulKernel);
                kernel(total, bufA.View, bufB.View, bufC.View, n);

                await accelerator.SynchronizeAsync();
                var results = await bufC.CopyToHostAsync<float>();

                // Verify a few entries
                for (int row = 0; row < n; row++)
                {
                    for (int col = 0; col < n; col++)
                    {
                        float expected = 0;
                        for (int k = 0; k < n; k++)
                            expected += dataA[row * n + k] * dataB[k * n + col];
                        float actual = results[row * n + col];
                        if (Math.Abs(actual - expected) > 0.1f)
                            throw new Exception($"MatMul mismatch at ({row},{col}): expected {expected}, got {actual}");
                    }
                }
            }
            finally
            {
                accelerator.Dispose();
                context.Dispose();
            }
        }

        /// <summary>
        /// Lerp: result = a + t * (b - a)
        /// </summary>
        static void LerpKernel(Index1D index, ArrayView<float> output, float a, float b)
        {
            float t = index / 63.0f;
            output[index] = a + t * (b - a);
        }

        [TestMethod]
        public async Task CPULerpTest()
        {
            var (context, accelerator) = CreateCPUAccelerator();
            try
            {
                int length = 64;
                float a = 10.0f, b = 50.0f;
                using var buffer = accelerator.Allocate1D<float>(length);

                var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, float, float>(LerpKernel);
                kernel(length, buffer.View, a, b);

                await accelerator.SynchronizeAsync();
                var results = await buffer.CopyToHostAsync<float>();

                for (int i = 0; i < length; i++)
                {
                    float t = i / 63.0f;
                    float expected = a + t * (b - a);
                    if (Math.Abs(results[i] - expected) > 0.01f)
                        throw new Exception($"Lerp mismatch at {i}: expected {expected}, got {results[i]}");
                }
            }
            finally
            {
                accelerator.Dispose();
                context.Dispose();
            }
        }

        /// <summary>
        /// Smoothstep: smooth Hermite interpolation
        /// </summary>
        static void SmoothstepKernel(Index1D index, ArrayView<float> output)
        {
            float t = index / 63.0f;
            float x = Math.Min(Math.Max(t, 0.0f), 1.0f);
            output[index] = x * x * (3.0f - 2.0f * x);
        }

        [TestMethod]
        public async Task CPUSmoothstepTest()
        {
            var (context, accelerator) = CreateCPUAccelerator();
            try
            {
                int length = 64;
                using var buffer = accelerator.Allocate1D<float>(length);

                var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>>(SmoothstepKernel);
                kernel(length, buffer.View);

                await accelerator.SynchronizeAsync();
                var results = await buffer.CopyToHostAsync<float>();

                for (int i = 0; i < length; i++)
                {
                    float t = i / 63.0f;
                    float x = Math.Min(Math.Max(t, 0.0f), 1.0f);
                    float expected = x * x * (3.0f - 2.0f * x);
                    if (Math.Abs(results[i] - expected) > 0.001f)
                        throw new Exception($"Smoothstep mismatch at {i}: expected {expected}, got {results[i]}");
                }
            }
            finally
            {
                accelerator.Dispose();
                context.Dispose();
            }
        }

        /// <summary>
        /// Complex expression: tests operator precedence
        /// </summary>
        static void ComplexExpressionKernel(Index1D index, ArrayView<int> output)
        {
            int a = index * 3 + 7;
            int b = (index + 2) * (index - 1);
            int c = a > b ? a - b : b - a;
            output[index] = c + index;
        }

        [TestMethod]
        public async Task CPUComplexExpressionTest()
        {
            var (context, accelerator) = CreateCPUAccelerator();
            try
            {
                int length = 64;
                using var buffer = accelerator.Allocate1D<int>(length);

                var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(ComplexExpressionKernel);
                kernel(length, buffer.View);

                await accelerator.SynchronizeAsync();
                var results = await buffer.CopyToHostAsync<int>();

                for (int i = 0; i < length; i++)
                {
                    int a = i * 3 + 7;
                    int b = (i + 2) * (i - 1);
                    int expected = (a > b ? a - b : b - a) + i;
                    if (results[i] != expected)
                        throw new Exception($"ComplexExpr mismatch at {i}: expected {expected}, got {results[i]}");
                }
            }
            finally
            {
                accelerator.Dispose();
                context.Dispose();
            }
        }

        #endregion

        #region Double Precision Tests

        /// <summary>
        /// Double precision: native 64-bit (no emulation needed on CPU)
        /// </summary>
        static void DoubleKernel(Index1D index, ArrayView<double> output)
        {
            double val = index * 0.001;
            output[index] = Math.Sin(val) + Math.Cos(val);
        }

        [TestMethod]
        public async Task CPUDoublePrecisionTest()
        {
            var (context, accelerator) = CreateCPUAccelerator();
            try
            {
                int length = 64;
                using var buffer = accelerator.Allocate1D<double>(length);

                var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>>(DoubleKernel);
                kernel(length, buffer.View);

                await accelerator.SynchronizeAsync();
                var results = await buffer.CopyToHostAsync<double>();

                for (int i = 0; i < length; i++)
                {
                    double val = i * 0.001;
                    double expected = Math.Sin(val) + Math.Cos(val);
                    if (Math.Abs(results[i] - expected) > 1e-10)
                        throw new Exception($"Double mismatch at {i}: expected {expected}, got {results[i]}");
                }
            }
            finally
            {
                accelerator.Dispose();
                context.Dispose();
            }
        }

        /// <summary>
        /// Long (int64): native on CPU
        /// </summary>
        static void LongKernel(Index1D index, ArrayView<long> output)
        {
            long val = (long)(int)index * 1000000L + 999999L;
            output[index] = val;
        }

        [TestMethod]
        public async Task CPULongPrecisionTest()
        {
            var (context, accelerator) = CreateCPUAccelerator();
            try
            {
                int length = 64;
                using var buffer = accelerator.Allocate1D<long>(length);

                var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<long>>(LongKernel);
                kernel(length, buffer.View);

                await accelerator.SynchronizeAsync();
                var results = await buffer.CopyToHostAsync<long>();

                for (int i = 0; i < length; i++)
                {
                    long expected = (long)i * 1000000L + 999999L;
                    if (results[i] != expected)
                        throw new Exception($"Long mismatch at {i}: expected {expected}, got {results[i]}");
                }
            }
            finally
            {
                accelerator.Dispose();
                context.Dispose();
            }
        }

        #endregion
    }
}
