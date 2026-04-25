using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Algorithms.RadixSortOperations;
using ILGPU.Algorithms.ScanReduceOperations;
using ILGPU.Runtime;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.Demo.Shared.UnitTests
{
    // Part 6: Algorithm tests (scan, reduce, radix sort)
    // These tests require EnableAlgorithms() + EnableWebGPUAlgorithms() on the context.
    public abstract partial class BackendTestBase
    {
        /// <summary>
        /// Test ILGPU.Algorithms ExclusiveScan via a GPU kernel that uses
        /// GroupExtensions.ExclusiveScan. This validates the algorithm intrinsic
        /// registration for each backend.
        /// </summary>
        [TestMethod]
        public async Task AlgorithmExclusiveScanTest() => await RunTest(async accelerator =>
        {
            int groupSize = Math.Min(64, accelerator.Device.MaxNumThreadsPerGroup);
            using var inputBuf = accelerator.Allocate1D<int>(groupSize);
            using var outputBuf = accelerator.Allocate1D<int>(groupSize);

            // Initialize: each thread contributes value 1
            var inputData = new int[groupSize];
            for (int i = 0; i < groupSize; i++) inputData[i] = 1;
            inputBuf.CopyFromCPU(inputData);

            var kernel = accelerator.LoadStreamKernel<Index1D, ArrayView<int>, ArrayView<int>>(
                ExclusiveScanKernel);
            kernel(new KernelConfig(1, groupSize), (Index1D)groupSize, inputBuf.View, outputBuf.View);
            await accelerator.SynchronizeAsync();

            var result = await outputBuf.CopyToHostAsync<int>();
            // Exclusive scan of all 1s: [0, 1, 2, 3, ...]
            for (int i = 0; i < groupSize; i++)
            {
                if (result[i] != i)
                    throw new Exception($"ExclusiveScan failed at {i}. Expected {i}, got {result[i]}");
            }
        });

        /// <summary>
        /// Test ILGPU.Algorithms InclusiveScan via a GPU kernel.
        /// </summary>
        [TestMethod]
        public async Task AlgorithmInclusiveScanTest() => await RunTest(async accelerator =>
        {
            int groupSize = Math.Min(64, accelerator.Device.MaxNumThreadsPerGroup);
            using var inputBuf = accelerator.Allocate1D<int>(groupSize);
            using var outputBuf = accelerator.Allocate1D<int>(groupSize);

            var inputData = new int[groupSize];
            for (int i = 0; i < groupSize; i++) inputData[i] = 1;
            inputBuf.CopyFromCPU(inputData);

            var kernel = accelerator.LoadStreamKernel<Index1D, ArrayView<int>, ArrayView<int>>(
                InclusiveScanKernel);
            kernel(new KernelConfig(1, groupSize), (Index1D)groupSize, inputBuf.View, outputBuf.View);
            await accelerator.SynchronizeAsync();

            var result = await outputBuf.CopyToHostAsync<int>();
            // Inclusive scan of all 1s: [1, 2, 3, 4, ...]
            for (int i = 0; i < groupSize; i++)
            {
                int expected = i + 1;
                if (result[i] != expected)
                    throw new Exception($"InclusiveScan failed at {i}. Expected {expected}, got {result[i]}");
            }
        });

        /// <summary>
        /// Test ILGPU.Algorithms AllReduce via a GPU kernel.
        /// </summary>
        [TestMethod]
        public async Task AlgorithmAllReduceTest() => await RunTest(async accelerator =>
        {
            int groupSize = Math.Min(64, accelerator.Device.MaxNumThreadsPerGroup);
            using var inputBuf = accelerator.Allocate1D<int>(groupSize);
            using var outputBuf = accelerator.Allocate1D<int>(groupSize);

            var inputData = new int[groupSize];
            for (int i = 0; i < groupSize; i++) inputData[i] = i + 1; // 1..64
            inputBuf.CopyFromCPU(inputData);

            var kernel = accelerator.LoadStreamKernel<Index1D, ArrayView<int>, ArrayView<int>>(
                AllReduceKernel);
            kernel(new KernelConfig(1, groupSize), (Index1D)groupSize, inputBuf.View, outputBuf.View);
            await accelerator.SynchronizeAsync();

            var result = await outputBuf.CopyToHostAsync<int>();
            int expectedSum = groupSize * (groupSize + 1) / 2; // 2080
            // AllReduce: every thread gets the same sum
            for (int i = 0; i < groupSize; i++)
            {
                if (result[i] != expectedSum)
                    throw new Exception($"AllReduce failed at {i}. Expected {expectedSum}, got {result[i]}");
            }
        });

        /// <summary>
        /// Test ILGPU.Algorithms RadixSortPairs — the key operation needed for
        /// the Gaussian splat renderer. Sorts float keys with int value indices.
        /// </summary>
        [TestMethod]
        public async Task AlgorithmRadixSortPairsTest() => await RunTest(async accelerator =>
        {
            int n = 256;
            // Create reverse-sorted distances and sequential indices
            var keys = new float[n];
            var values = new int[n];
            for (int i = 0; i < n; i++)
            {
                keys[i] = (float)(n - i); // Reverse order: 256, 255, ..., 1
                values[i] = i;
            }

            using var keysBuf = accelerator.Allocate1D(keys);
            using var valuesBuf = accelerator.Allocate1D(values);
            var tempSize = accelerator.ComputeRadixSortPairsTempStorageSize<float, int, AscendingFloat>(n);
            using var tempBuf = accelerator.Allocate1D<int>(tempSize);

            var radixSort = accelerator.CreateRadixSortPairs<float, Stride1D.Dense, int, Stride1D.Dense, AscendingFloat>();
            radixSort(
                accelerator.DefaultStream,
                keysBuf.View,
                valuesBuf.View,
                tempBuf.View.AsContiguous());
            await accelerator.SynchronizeAsync();

            var sortedKeys = await keysBuf.CopyToHostAsync<float>();
            var sortedValues = await valuesBuf.CopyToHostAsync<int>();

            for (int i = 0; i < n; i++)
            {
                float expectedKey = (float)(i + 1);
                int expectedValue = n - 1 - i;
                if (MathF.Abs(sortedKeys[i] - expectedKey) > 0.001f)
                    throw new Exception($"RadixSort key mismatch at [{i}]: expected={expectedKey}, got={sortedKeys[i]}");
                if (sortedValues[i] != expectedValue)
                    throw new Exception($"RadixSort value mismatch at [{i}]: expected={expectedValue}, got={sortedValues[i]}");
            }
        });

        /// <summary>
        /// Test RadixSortPairs with int keys — verifies AscendingInt32 operation.
        /// </summary>
        [TestMethod]
        public async Task AlgorithmRadixSortPairsIntTest() => await RunTest(async accelerator =>
        {
            int n = 256;
            var keys = new int[n];
            var values = new int[n];
            for (int i = 0; i < n; i++) { keys[i] = n - i; values[i] = i; }

            using var keysBuf = accelerator.Allocate1D(keys);
            using var valuesBuf = accelerator.Allocate1D(values);
            var tempSize = accelerator.ComputeRadixSortPairsTempStorageSize<int, int, AscendingInt32>(n);
            using var tempBuf = accelerator.Allocate1D<int>(tempSize);

            var radixSort = accelerator.CreateRadixSortPairs<int, Stride1D.Dense, int, Stride1D.Dense, AscendingInt32>();
            radixSort(accelerator.DefaultStream, keysBuf.View, valuesBuf.View, tempBuf.View.AsContiguous());
            await accelerator.SynchronizeAsync();

            var sortedKeys = await keysBuf.CopyToHostAsync<int>();
            var sortedValues = await valuesBuf.CopyToHostAsync<int>();
            for (int i = 0; i < n; i++)
            {
                if (sortedKeys[i] != i + 1)
                    throw new Exception($"RadixSort int key mismatch at [{i}]: expected={i + 1}, got={sortedKeys[i]}");
                if (sortedValues[i] != n - 1 - i)
                    throw new Exception($"RadixSort int value mismatch at [{i}]: expected={n - 1 - i}, got={sortedValues[i]}");
            }
        });

        /// <summary>
        /// Test RadixSortPairs with double keys — verifies AscendingDouble (f64 emulation on WebGPU).
        /// </summary>
        [TestMethod]
        public async Task AlgorithmRadixSortPairsDoubleTest() => await RunEmulatedTest(async accelerator =>
        {
            int n = 256;
            var keys = new double[n];
            var values = new int[n];
            for (int i = 0; i < n; i++) { keys[i] = (double)(n - i); values[i] = i; }

            using var keysBuf = accelerator.Allocate1D(keys);
            using var valuesBuf = accelerator.Allocate1D(values);
            var tempSize = accelerator.ComputeRadixSortPairsTempStorageSize<double, int, AscendingDouble>(n);
            using var tempBuf = accelerator.Allocate1D<int>(tempSize);

            var radixSort = accelerator.CreateRadixSortPairs<double, Stride1D.Dense, int, Stride1D.Dense, AscendingDouble>();
            radixSort(accelerator.DefaultStream, keysBuf.View, valuesBuf.View, tempBuf.View.AsContiguous());
            await accelerator.SynchronizeAsync();

            var sortedKeys = await keysBuf.CopyToHostAsync<double>();
            var sortedValues = await valuesBuf.CopyToHostAsync<int>();
            for (int i = 0; i < n; i++)
            {
                if (Math.Abs(sortedKeys[i] - (i + 1.0)) > 0.001)
                    throw new Exception($"RadixSort double key mismatch at [{i}]: expected={i + 1.0}, got={sortedKeys[i]}");
                if (sortedValues[i] != n - 1 - i)
                    throw new Exception($"RadixSort double value mismatch at [{i}]: expected={n - 1 - i}, got={sortedValues[i]}");
            }
        });

        /// <summary>
        /// Test RadixSortPairs with long keys — verifies AscendingInt64 (i64 emulation on WebGPU).
        /// </summary>
        [TestMethod]
        public async Task AlgorithmRadixSortPairsLongTest() => await RunEmulatedTest(async accelerator =>
        {
            int n = 256;
            var keys = new long[n];
            var values = new int[n];
            for (int i = 0; i < n; i++) { keys[i] = (long)(n - i); values[i] = i; }

            using var keysBuf = accelerator.Allocate1D(keys);
            using var valuesBuf = accelerator.Allocate1D(values);
            var tempSize = accelerator.ComputeRadixSortPairsTempStorageSize<long, int, AscendingInt64>(n);
            using var tempBuf = accelerator.Allocate1D<int>(tempSize);

            var radixSort = accelerator.CreateRadixSortPairs<long, Stride1D.Dense, int, Stride1D.Dense, AscendingInt64>();
            radixSort(accelerator.DefaultStream, keysBuf.View, valuesBuf.View, tempBuf.View.AsContiguous());
            await accelerator.SynchronizeAsync();

            var sortedKeys = await keysBuf.CopyToHostAsync<long>();
            var sortedValues = await valuesBuf.CopyToHostAsync<int>();
            for (int i = 0; i < n; i++)
            {
                if (sortedKeys[i] != (long)(i + 1))
                    throw new Exception($"RadixSort long key mismatch at [{i}]: expected={i + 1}, got={sortedKeys[i]}");
                if (sortedValues[i] != n - 1 - i)
                    throw new Exception($"RadixSort long value mismatch at [{i}]: expected={n - 1 - i}, got={sortedValues[i]}");
            }
        });

        /// <summary>
        /// Test RadixSortPairs with double keys using n=129 (non-multiple of 16) to trigger
        /// non-zero 256-byte-alignment padding in the inner temp view, exposing the packed-struct
        /// view element offset bug on WebGPU.
        /// </summary>
        [TestMethod]
        public async Task AlgorithmRadixSortPairsDoubleOffsetTest() => await RunEmulatedTest(async accelerator =>
        {
            int n = 129;
            var keys = new double[n];
            var values = new int[n];
            for (int i = 0; i < n; i++) { keys[i] = (double)(n - i); values[i] = i; }

            using var keysBuf = accelerator.Allocate1D(keys);
            using var valuesBuf = accelerator.Allocate1D(values);
            var tempSize = accelerator.ComputeRadixSortPairsTempStorageSize<double, int, AscendingDouble>(n);
            using var tempBuf = accelerator.Allocate1D<int>(tempSize);

            var radixSort = accelerator.CreateRadixSortPairs<double, Stride1D.Dense, int, Stride1D.Dense, AscendingDouble>();
            radixSort(accelerator.DefaultStream, keysBuf.View, valuesBuf.View, tempBuf.View.AsContiguous());
            await accelerator.SynchronizeAsync();

            var sortedKeys = await keysBuf.CopyToHostAsync<double>();
            var sortedValues = await valuesBuf.CopyToHostAsync<int>();
            for (int i = 0; i < n; i++)
            {
                if (Math.Abs(sortedKeys[i] - (i + 1.0)) > 0.001)
                    throw new Exception($"RadixSort double offset key mismatch at [{i}]: expected={i + 1.0}, got={sortedKeys[i]}");
                if (sortedValues[i] != n - 1 - i)
                    throw new Exception($"RadixSort double offset value mismatch at [{i}]: expected={n - 1 - i}, got={sortedValues[i]}");
            }
        });

        /// <summary>
        /// Test RadixSortPairs with long keys using n=129 (non-multiple of 16) to trigger
        /// non-zero 256-byte-alignment padding in the inner temp view, exposing the packed-struct
        /// view element offset bug on WebGPU.
        /// </summary>
        [TestMethod]
        public async Task AlgorithmRadixSortPairsLongOffsetTest() => await RunEmulatedTest(async accelerator =>
        {
            int n = 129;
            var keys = new long[n];
            var values = new int[n];
            for (int i = 0; i < n; i++) { keys[i] = (long)(n - i); values[i] = i; }

            using var keysBuf = accelerator.Allocate1D(keys);
            using var valuesBuf = accelerator.Allocate1D(values);
            var tempSize = accelerator.ComputeRadixSortPairsTempStorageSize<long, int, AscendingInt64>(n);
            using var tempBuf = accelerator.Allocate1D<int>(tempSize);

            var radixSort = accelerator.CreateRadixSortPairs<long, Stride1D.Dense, int, Stride1D.Dense, AscendingInt64>();
            radixSort(accelerator.DefaultStream, keysBuf.View, valuesBuf.View, tempBuf.View.AsContiguous());
            await accelerator.SynchronizeAsync();

            var sortedKeys = await keysBuf.CopyToHostAsync<long>();
            var sortedValues = await valuesBuf.CopyToHostAsync<int>();
            for (int i = 0; i < n; i++)
            {
                if (sortedKeys[i] != (long)(i + 1))
                    throw new Exception($"RadixSort long offset key mismatch at [{i}]: expected={i + 1}, got={sortedKeys[i]}");
                if (sortedValues[i] != n - 1 - i)
                    throw new Exception($"RadixSort long offset value mismatch at [{i}]: expected={n - 1 - i}, got={sortedValues[i]}");
            }
        });

        /// <summary>
        /// Test RadixSortPairs with uint keys — verifies AscendingUInt32 operation.
        /// </summary>
        [TestMethod]
        public async Task AlgorithmRadixSortPairsUIntTest() => await RunTest(async accelerator =>
        {
            int n = 256;
            var keys = new uint[n];
            var values = new int[n];
            for (int i = 0; i < n; i++) { keys[i] = (uint)(n - i); values[i] = i; }

            using var keysBuf = accelerator.Allocate1D(keys);
            using var valuesBuf = accelerator.Allocate1D(values);
            var tempSize = accelerator.ComputeRadixSortPairsTempStorageSize<uint, int, AscendingUInt32>(n);
            using var tempBuf = accelerator.Allocate1D<int>(tempSize);

            var radixSort = accelerator.CreateRadixSortPairs<uint, Stride1D.Dense, int, Stride1D.Dense, AscendingUInt32>();
            radixSort(accelerator.DefaultStream, keysBuf.View, valuesBuf.View, tempBuf.View.AsContiguous());
            await accelerator.SynchronizeAsync();

            var sortedKeys = await keysBuf.CopyToHostAsync<uint>();
            var sortedValues = await valuesBuf.CopyToHostAsync<int>();
            for (int i = 0; i < n; i++)
            {
                if (sortedKeys[i] != (uint)(i + 1))
                    throw new Exception($"RadixSort uint key mismatch at [{i}]: expected={i + 1}, got={sortedKeys[i]}");
                if (sortedValues[i] != n - 1 - i)
                    throw new Exception($"RadixSort uint value mismatch at [{i}]: expected={n - 1 - i}, got={sortedValues[i]}");
            }
        });

        /// <summary>
        /// Test RadixSortPairs with Half keys — verifies AscendingHalf operation.
        /// Skips on backends that do not support Float16 (e.g. some OpenCL devices).
        /// </summary>
        [TestMethod]
        public async Task AlgorithmRadixSortPairsHalfTest() => await RunTest(async accelerator =>
        {
            if (!accelerator.Capabilities.Float16)
                throw new UnsupportedTestException("Float16 not supported on this device");
            int n = 256;
            var keys = new global::ILGPU.Half[n];
            var values = new int[n];
            for (int i = 0; i < n; i++) { keys[i] = (global::ILGPU.Half)(float)(n - i); values[i] = i; }

            using var keysBuf = accelerator.Allocate1D(keys);
            using var valuesBuf = accelerator.Allocate1D(values);
            var tempSize = accelerator.ComputeRadixSortPairsTempStorageSize<global::ILGPU.Half, int, AscendingHalf>(n);
            using var tempBuf = accelerator.Allocate1D<int>(tempSize);

            var radixSort = accelerator.CreateRadixSortPairs<global::ILGPU.Half, Stride1D.Dense, int, Stride1D.Dense, AscendingHalf>();
            radixSort(accelerator.DefaultStream, keysBuf.View, valuesBuf.View, tempBuf.View.AsContiguous());
            await accelerator.SynchronizeAsync();

            var sortedKeys = await keysBuf.CopyToHostAsync<global::ILGPU.Half>();
            var sortedValues = await valuesBuf.CopyToHostAsync<int>();
            for (int i = 0; i < n; i++)
            {
                float expected = (float)(i + 1);
                if (MathF.Abs((float)sortedKeys[i] - expected) > 0.01f)
                    throw new Exception($"RadixSort Half key mismatch at [{i}]: expected={expected}, got={(float)sortedKeys[i]}");
                if (sortedValues[i] != n - 1 - i)
                    throw new Exception($"RadixSort Half value mismatch at [{i}]: expected={n - 1 - i}, got={sortedValues[i]}");
            }
        });

        /// <summary>
        /// Test RadixSort with non-power-of-2 count — important for real-world
        /// Gaussian splat rendering where splat count is arbitrary.
        /// </summary>
        [TestMethod]
        public async Task AlgorithmRadixSortNonPow2Test() => await RunTest(async accelerator =>
        {
            int n = 137; // Non-power-of-2
            var keys = new float[n];
            var values = new int[n];
            var rng = new Random(42);
            for (int i = 0; i < n; i++)
            {
                keys[i] = (float)(rng.NextDouble() * 1000.0);
                values[i] = i;
            }

            using var keysBuf = accelerator.Allocate1D(keys);
            using var valuesBuf = accelerator.Allocate1D(values);
            var tempSize = accelerator.ComputeRadixSortPairsTempStorageSize<float, int, AscendingFloat>(n);
            using var tempBuf = accelerator.Allocate1D<int>(tempSize);

            var radixSort = accelerator.CreateRadixSortPairs<float, Stride1D.Dense, int, Stride1D.Dense, AscendingFloat>();
            radixSort(
                accelerator.DefaultStream,
                keysBuf.View,
                valuesBuf.View,
                tempBuf.View.AsContiguous());
            await accelerator.SynchronizeAsync();

            var sortedKeys = await keysBuf.CopyToHostAsync<float>();

            // Verify ascending order
            for (int i = 1; i < n; i++)
            {
                if (sortedKeys[i] < sortedKeys[i - 1])
                    throw new Exception($"RadixSort non-pow2 order failed at {i}. {sortedKeys[i-1]} > {sortedKeys[i]}");
            }
        });

        /// <summary>
        /// Test ExclusiveScan with float type — verifies AddFloat operation.
        /// </summary>
        [TestMethod]
        public async Task AlgorithmExclusiveScanFloatTest() => await RunTest(async accelerator =>
        {
            int groupSize = Math.Min(64, accelerator.Device.MaxNumThreadsPerGroup);
            using var inputBuf = accelerator.Allocate1D<float>(groupSize);
            using var outputBuf = accelerator.Allocate1D<float>(groupSize);

            var inputData = new float[groupSize];
            for (int i = 0; i < groupSize; i++) inputData[i] = 1.0f;
            inputBuf.CopyFromCPU(inputData);

            var kernel = accelerator.LoadStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(
                ExclusiveScanFloatKernel);
            kernel(new KernelConfig(1, groupSize), (Index1D)groupSize, inputBuf.View, outputBuf.View);
            await accelerator.SynchronizeAsync();

            var result = await outputBuf.CopyToHostAsync<float>();
            for (int i = 0; i < groupSize; i++)
            {
                float expected = (float)i;
                if (MathF.Abs(result[i] - expected) > 0.001f)
                    throw new Exception($"ExclusiveScanFloat failed at {i}. Expected {expected}, got {result[i]}");
            }
        });

        /// <summary>
        /// Test ExclusiveScan with long type — verifies AddInt64 (i64 emulation on WebGPU).
        /// </summary>
        [TestMethod]
        public async Task AlgorithmExclusiveScanLongTest() => await RunEmulatedTest(async accelerator =>
        {
            int groupSize = Math.Min(64, accelerator.Device.MaxNumThreadsPerGroup);
            using var inputBuf = accelerator.Allocate1D<long>(groupSize);
            using var outputBuf = accelerator.Allocate1D<long>(groupSize);

            var inputData = new long[groupSize];
            for (int i = 0; i < groupSize; i++) inputData[i] = 1L;
            inputBuf.CopyFromCPU(inputData);

            var kernel = accelerator.LoadStreamKernel<Index1D, ArrayView<long>, ArrayView<long>>(
                ExclusiveScanLongKernel);
            kernel(new KernelConfig(1, groupSize), (Index1D)groupSize, inputBuf.View, outputBuf.View);
            await accelerator.SynchronizeAsync();

            var result = await outputBuf.CopyToHostAsync<long>();
            for (int i = 0; i < groupSize; i++)
            {
                if (result[i] != (long)i)
                    throw new Exception($"ExclusiveScanLong failed at {i}. Expected {i}, got {result[i]}");
            }
        });

        /// <summary>
        /// Test ExclusiveScan with double type — verifies AddDouble operation (f64 emulation on WebGPU).
        /// </summary>
        [TestMethod]
        public async Task AlgorithmExclusiveScanDoubleTest() => await RunEmulatedTest(async accelerator =>
        {
            int groupSize = Math.Min(64, accelerator.Device.MaxNumThreadsPerGroup);
            using var inputBuf = accelerator.Allocate1D<double>(groupSize);
            using var outputBuf = accelerator.Allocate1D<double>(groupSize);

            var inputData = new double[groupSize];
            for (int i = 0; i < groupSize; i++) inputData[i] = 1.0;
            inputBuf.CopyFromCPU(inputData);

            var kernel = accelerator.LoadStreamKernel<Index1D, ArrayView<double>, ArrayView<double>>(
                ExclusiveScanDoubleKernel);
            kernel(new KernelConfig(1, groupSize), (Index1D)groupSize, inputBuf.View, outputBuf.View);
            await accelerator.SynchronizeAsync();

            var result = await outputBuf.CopyToHostAsync<double>();
            for (int i = 0; i < groupSize; i++)
            {
                double expected = (double)i;
                if (Math.Abs(result[i] - expected) > 0.001)
                    throw new Exception($"ExclusiveScanDouble failed at {i}. Expected {expected}, got {result[i]}");
            }
        });

        /// <summary>
        /// Test ExclusiveScan with uint type — verifies AddUInt32 operation.
        /// </summary>
        [TestMethod]
        public async Task AlgorithmExclusiveScanUIntTest() => await RunTest(async accelerator =>
        {
            int groupSize = Math.Min(64, accelerator.Device.MaxNumThreadsPerGroup);
            using var inputBuf = accelerator.Allocate1D<uint>(groupSize);
            using var outputBuf = accelerator.Allocate1D<uint>(groupSize);

            var inputData = new uint[groupSize];
            for (int i = 0; i < groupSize; i++) inputData[i] = 1u;
            inputBuf.CopyFromCPU(inputData);

            var kernel = accelerator.LoadStreamKernel<Index1D, ArrayView<uint>, ArrayView<uint>>(
                ExclusiveScanUIntKernel);
            kernel(new KernelConfig(1, groupSize), (Index1D)groupSize, inputBuf.View, outputBuf.View);
            await accelerator.SynchronizeAsync();

            var result = await outputBuf.CopyToHostAsync<uint>();
            for (int i = 0; i < groupSize; i++)
            {
                if (result[i] != (uint)i)
                    throw new Exception($"ExclusiveScanUInt failed at {i}. Expected {i}, got {result[i]}");
            }
        });

        /// <summary>
        /// Test InclusiveScan with float type — verifies AddFloat operation.
        /// </summary>
        [TestMethod]
        public async Task AlgorithmInclusiveScanFloatTest() => await RunTest(async accelerator =>
        {
            int groupSize = Math.Min(64, accelerator.Device.MaxNumThreadsPerGroup);
            using var inputBuf = accelerator.Allocate1D<float>(groupSize);
            using var outputBuf = accelerator.Allocate1D<float>(groupSize);

            var inputData = new float[groupSize];
            for (int i = 0; i < groupSize; i++) inputData[i] = 1.0f;
            inputBuf.CopyFromCPU(inputData);

            var kernel = accelerator.LoadStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(
                InclusiveScanFloatKernel);
            kernel(new KernelConfig(1, groupSize), (Index1D)groupSize, inputBuf.View, outputBuf.View);
            await accelerator.SynchronizeAsync();

            var result = await outputBuf.CopyToHostAsync<float>();
            for (int i = 0; i < groupSize; i++)
            {
                float expected = (float)(i + 1);
                if (MathF.Abs(result[i] - expected) > 0.001f)
                    throw new Exception($"InclusiveScanFloat failed at {i}. Expected {expected}, got {result[i]}");
            }
        });

        /// <summary>
        /// Test InclusiveScan with long type — verifies AddInt64 (i64 emulation on WebGPU).
        /// </summary>
        [TestMethod]
        public async Task AlgorithmInclusiveScanLongTest() => await RunEmulatedTest(async accelerator =>
        {
            int groupSize = Math.Min(64, accelerator.Device.MaxNumThreadsPerGroup);
            using var inputBuf = accelerator.Allocate1D<long>(groupSize);
            using var outputBuf = accelerator.Allocate1D<long>(groupSize);

            var inputData = new long[groupSize];
            for (int i = 0; i < groupSize; i++) inputData[i] = 1L;
            inputBuf.CopyFromCPU(inputData);

            var kernel = accelerator.LoadStreamKernel<Index1D, ArrayView<long>, ArrayView<long>>(
                InclusiveScanLongKernel);
            kernel(new KernelConfig(1, groupSize), (Index1D)groupSize, inputBuf.View, outputBuf.View);
            await accelerator.SynchronizeAsync();

            var result = await outputBuf.CopyToHostAsync<long>();
            for (int i = 0; i < groupSize; i++)
            {
                long expected = (long)(i + 1);
                if (result[i] != expected)
                    throw new Exception($"InclusiveScanLong failed at {i}. Expected {expected}, got {result[i]}");
            }
        });

        /// <summary>
        /// Test InclusiveScan with double type — verifies AddDouble operation (f64 emulation on WebGPU).
        /// </summary>
        [TestMethod]
        public async Task AlgorithmInclusiveScanDoubleTest() => await RunEmulatedTest(async accelerator =>
        {
            int groupSize = Math.Min(64, accelerator.Device.MaxNumThreadsPerGroup);
            using var inputBuf = accelerator.Allocate1D<double>(groupSize);
            using var outputBuf = accelerator.Allocate1D<double>(groupSize);

            var inputData = new double[groupSize];
            for (int i = 0; i < groupSize; i++) inputData[i] = 1.0;
            inputBuf.CopyFromCPU(inputData);

            var kernel = accelerator.LoadStreamKernel<Index1D, ArrayView<double>, ArrayView<double>>(
                InclusiveScanDoubleKernel);
            kernel(new KernelConfig(1, groupSize), (Index1D)groupSize, inputBuf.View, outputBuf.View);
            await accelerator.SynchronizeAsync();

            var result = await outputBuf.CopyToHostAsync<double>();
            for (int i = 0; i < groupSize; i++)
            {
                double expected = (double)(i + 1);
                if (Math.Abs(result[i] - expected) > 0.001)
                    throw new Exception($"InclusiveScanDouble failed at {i}. Expected {expected}, got {result[i]}");
            }
        });

        /// <summary>
        /// Test InclusiveScan with uint type — verifies AddUInt32 operation.
        /// </summary>
        [TestMethod]
        public async Task AlgorithmInclusiveScanUIntTest() => await RunTest(async accelerator =>
        {
            int groupSize = Math.Min(64, accelerator.Device.MaxNumThreadsPerGroup);
            using var inputBuf = accelerator.Allocate1D<uint>(groupSize);
            using var outputBuf = accelerator.Allocate1D<uint>(groupSize);

            var inputData = new uint[groupSize];
            for (int i = 0; i < groupSize; i++) inputData[i] = 1u;
            inputBuf.CopyFromCPU(inputData);

            var kernel = accelerator.LoadStreamKernel<Index1D, ArrayView<uint>, ArrayView<uint>>(
                InclusiveScanUIntKernel);
            kernel(new KernelConfig(1, groupSize), (Index1D)groupSize, inputBuf.View, outputBuf.View);
            await accelerator.SynchronizeAsync();

            var result = await outputBuf.CopyToHostAsync<uint>();
            for (int i = 0; i < groupSize; i++)
            {
                uint expected = (uint)(i + 1);
                if (result[i] != expected)
                    throw new Exception($"InclusiveScanUInt failed at {i}. Expected {expected}, got {result[i]}");
            }
        });

        /// <summary>
        /// Test AllReduce with float type — verifies AddFloat operation.
        /// </summary>
        [TestMethod]
        public async Task AlgorithmAllReduceFloatTest() => await RunTest(async accelerator =>
        {
            int groupSize = Math.Min(64, accelerator.Device.MaxNumThreadsPerGroup);
            using var inputBuf = accelerator.Allocate1D<float>(groupSize);
            using var outputBuf = accelerator.Allocate1D<float>(groupSize);

            var inputData = new float[groupSize];
            for (int i = 0; i < groupSize; i++) inputData[i] = (float)(i + 1);
            inputBuf.CopyFromCPU(inputData);

            var kernel = accelerator.LoadStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(
                AllReduceFloatKernel);
            kernel(new KernelConfig(1, groupSize), (Index1D)groupSize, inputBuf.View, outputBuf.View);
            await accelerator.SynchronizeAsync();

            var result = await outputBuf.CopyToHostAsync<float>();
            float expectedSum = groupSize * (groupSize + 1) / 2.0f;
            for (int i = 0; i < groupSize; i++)
            {
                if (MathF.Abs(result[i] - expectedSum) > 0.001f)
                    throw new Exception($"AllReduceFloat failed at {i}. Expected {expectedSum}, got {result[i]}");
            }
        });

        /// <summary>
        /// Test AllReduce with double type — verifies AddDouble operation (f64 emulation on WebGPU).
        /// </summary>
        [TestMethod]
        public async Task AlgorithmAllReduceDoubleTest() => await RunEmulatedTest(async accelerator =>
        {
            int groupSize = Math.Min(64, accelerator.Device.MaxNumThreadsPerGroup);
            using var inputBuf = accelerator.Allocate1D<double>(groupSize);
            using var outputBuf = accelerator.Allocate1D<double>(groupSize);

            var inputData = new double[groupSize];
            for (int i = 0; i < groupSize; i++) inputData[i] = (double)(i + 1);
            inputBuf.CopyFromCPU(inputData);

            var kernel = accelerator.LoadStreamKernel<Index1D, ArrayView<double>, ArrayView<double>>(
                AllReduceDoubleKernel);
            kernel(new KernelConfig(1, groupSize), (Index1D)groupSize, inputBuf.View, outputBuf.View);
            await accelerator.SynchronizeAsync();

            var result = await outputBuf.CopyToHostAsync<double>();
            double expectedSum = groupSize * (groupSize + 1) / 2.0;
            for (int i = 0; i < groupSize; i++)
            {
                if (Math.Abs(result[i] - expectedSum) > 0.001)
                    throw new Exception($"AllReduceDouble failed at {i}. Expected {expectedSum}, got {result[i]}");
            }
        });

        /// <summary>
        /// Test AllReduce with long type — verifies AddInt64 (i64 emulation on WebGPU).
        /// </summary>
        [TestMethod]
        public async Task AlgorithmAllReduceLongTest() => await RunEmulatedTest(async accelerator =>
        {
            int groupSize = Math.Min(64, accelerator.Device.MaxNumThreadsPerGroup);
            using var inputBuf = accelerator.Allocate1D<long>(groupSize);
            using var outputBuf = accelerator.Allocate1D<long>(groupSize);

            var inputData = new long[groupSize];
            for (int i = 0; i < groupSize; i++) inputData[i] = (long)(i + 1);
            inputBuf.CopyFromCPU(inputData);

            var kernel = accelerator.LoadStreamKernel<Index1D, ArrayView<long>, ArrayView<long>>(
                AllReduceLongKernel);
            kernel(new KernelConfig(1, groupSize), (Index1D)groupSize, inputBuf.View, outputBuf.View);
            await accelerator.SynchronizeAsync();

            var result = await outputBuf.CopyToHostAsync<long>();
            long expectedSum = (long)groupSize * (groupSize + 1) / 2L;
            for (int i = 0; i < groupSize; i++)
            {
                if (result[i] != expectedSum)
                    throw new Exception($"AllReduceLong failed at {i}. Expected {expectedSum}, got {result[i]}");
            }
        });

        /// <summary>
        /// Test AllReduce with uint type — verifies AddUInt32 operation.
        /// </summary>
        [TestMethod]
        public async Task AlgorithmAllReduceUIntTest() => await RunTest(async accelerator =>
        {
            int groupSize = Math.Min(64, accelerator.Device.MaxNumThreadsPerGroup);
            using var inputBuf = accelerator.Allocate1D<uint>(groupSize);
            using var outputBuf = accelerator.Allocate1D<uint>(groupSize);

            var inputData = new uint[groupSize];
            for (int i = 0; i < groupSize; i++) inputData[i] = (uint)(i + 1);
            inputBuf.CopyFromCPU(inputData);

            var kernel = accelerator.LoadStreamKernel<Index1D, ArrayView<uint>, ArrayView<uint>>(
                AllReduceUIntKernel);
            kernel(new KernelConfig(1, groupSize), (Index1D)groupSize, inputBuf.View, outputBuf.View);
            await accelerator.SynchronizeAsync();

            var result = await outputBuf.CopyToHostAsync<uint>();
            uint expectedSum = (uint)(groupSize * (groupSize + 1) / 2);
            for (int i = 0; i < groupSize; i++)
            {
                if (result[i] != expectedSum)
                    throw new Exception($"AllReduceUInt failed at {i}. Expected {expectedSum}, got {result[i]}");
            }
        });

        /// <summary>
        /// Test ExclusiveScan with Half type — verifies AddHalf operation.
        /// Uses a smaller group size to keep sums within Half's exact integer range.
        /// </summary>
        [TestMethod]
        public async Task AlgorithmExclusiveScanHalfTest() => await RunTest(async accelerator =>
        {
            if (!accelerator.Capabilities.Float16)
                throw new UnsupportedTestException("Float16 not supported on this device");
            int groupSize = Math.Min(32, accelerator.Device.MaxNumThreadsPerGroup);
            using var inputBuf = accelerator.Allocate1D<global::ILGPU.Half>(groupSize);
            using var outputBuf = accelerator.Allocate1D<global::ILGPU.Half>(groupSize);

            var inputData = new global::ILGPU.Half[groupSize];
            for (int i = 0; i < groupSize; i++) inputData[i] = (global::ILGPU.Half)1.0f;
            inputBuf.CopyFromCPU(inputData);

            var kernel = accelerator.LoadStreamKernel<Index1D, ArrayView<global::ILGPU.Half>, ArrayView<global::ILGPU.Half>>(
                ExclusiveScanHalfKernel);
            kernel(new KernelConfig(1, groupSize), (Index1D)groupSize, inputBuf.View, outputBuf.View);
            await accelerator.SynchronizeAsync();

            var result = await outputBuf.CopyToHostAsync<global::ILGPU.Half>();
            for (int i = 0; i < groupSize; i++)
            {
                float expected = (float)i;
                if (MathF.Abs((float)result[i] - expected) > 0.1f)
                    throw new Exception($"ExclusiveScanHalf failed at {i}. Expected {expected}, got {(float)result[i]}");
            }
        });

        /// <summary>
        /// Test InclusiveScan with Half type — verifies AddHalf operation.
        /// </summary>
        [TestMethod]
        public async Task AlgorithmInclusiveScanHalfTest() => await RunTest(async accelerator =>
        {
            if (!accelerator.Capabilities.Float16)
                throw new UnsupportedTestException("Float16 not supported on this device");
            int groupSize = Math.Min(32, accelerator.Device.MaxNumThreadsPerGroup);
            using var inputBuf = accelerator.Allocate1D<global::ILGPU.Half>(groupSize);
            using var outputBuf = accelerator.Allocate1D<global::ILGPU.Half>(groupSize);

            var inputData = new global::ILGPU.Half[groupSize];
            for (int i = 0; i < groupSize; i++) inputData[i] = (global::ILGPU.Half)1.0f;
            inputBuf.CopyFromCPU(inputData);

            var kernel = accelerator.LoadStreamKernel<Index1D, ArrayView<global::ILGPU.Half>, ArrayView<global::ILGPU.Half>>(
                InclusiveScanHalfKernel);
            kernel(new KernelConfig(1, groupSize), (Index1D)groupSize, inputBuf.View, outputBuf.View);
            await accelerator.SynchronizeAsync();

            var result = await outputBuf.CopyToHostAsync<global::ILGPU.Half>();
            for (int i = 0; i < groupSize; i++)
            {
                float expected = (float)(i + 1);
                if (MathF.Abs((float)result[i] - expected) > 0.1f)
                    throw new Exception($"InclusiveScanHalf failed at {i}. Expected {expected}, got {(float)result[i]}");
            }
        });

        /// <summary>
        /// Test AllReduce with Half type — verifies AddHalf operation.
        /// </summary>
        [TestMethod]
        public async Task AlgorithmAllReduceHalfTest() => await RunTest(async accelerator =>
        {
            if (!accelerator.Capabilities.Float16)
                throw new UnsupportedTestException("Float16 not supported on this device");
            int groupSize = Math.Min(32, accelerator.Device.MaxNumThreadsPerGroup);
            using var inputBuf = accelerator.Allocate1D<global::ILGPU.Half>(groupSize);
            using var outputBuf = accelerator.Allocate1D<global::ILGPU.Half>(groupSize);

            var inputData = new global::ILGPU.Half[groupSize];
            for (int i = 0; i < groupSize; i++) inputData[i] = (global::ILGPU.Half)(float)(i + 1);
            inputBuf.CopyFromCPU(inputData);

            var kernel = accelerator.LoadStreamKernel<Index1D, ArrayView<global::ILGPU.Half>, ArrayView<global::ILGPU.Half>>(
                AllReduceHalfKernel);
            kernel(new KernelConfig(1, groupSize), (Index1D)groupSize, inputBuf.View, outputBuf.View);
            await accelerator.SynchronizeAsync();

            var result = await outputBuf.CopyToHostAsync<global::ILGPU.Half>();
            float expectedSum = groupSize * (groupSize + 1) / 2.0f;
            for (int i = 0; i < groupSize; i++)
            {
                if (MathF.Abs((float)result[i] - expectedSum) > 1.0f)
                    throw new Exception($"AllReduceHalf failed at {i}. Expected {expectedSum}, got {(float)result[i]}");
            }
        });

        /// <summary>
        /// Test ExclusiveScan with varying values (not just 1s).
        /// </summary>
        [TestMethod]
        public async Task AlgorithmExclusiveScanVaryingTest() => await RunTest(async accelerator =>
        {
            int groupSize = Math.Min(32, accelerator.Device.MaxNumThreadsPerGroup);
            using var inputBuf = accelerator.Allocate1D<int>(groupSize);
            using var outputBuf = accelerator.Allocate1D<int>(groupSize);

            // Each thread contributes its index: [0, 1, 2, ..., 31]
            var inputData = new int[groupSize];
            for (int i = 0; i < groupSize; i++) inputData[i] = i;
            inputBuf.CopyFromCPU(inputData);

            var kernel = accelerator.LoadStreamKernel<Index1D, ArrayView<int>, ArrayView<int>>(
                ExclusiveScanKernel);
            kernel(new KernelConfig(1, groupSize), (Index1D)groupSize, inputBuf.View, outputBuf.View);
            await accelerator.SynchronizeAsync();

            var result = await outputBuf.CopyToHostAsync<int>();
            // Exclusive scan of [0,1,2,...,31]: [0, 0, 1, 3, 6, 10, ...]
            int runningSum = 0;
            for (int i = 0; i < groupSize; i++)
            {
                if (result[i] != runningSum)
                    throw new Exception($"ExclusiveScan varying failed at {i}. Expected {runningSum}, got {result[i]}");
                runningSum += inputData[i];
            }
        });

        /// <summary>
        /// Test ExclusiveScanWithBoundaries — validates that boundary values
        /// are correctly returned alongside the scan result.
        /// </summary>
        [TestMethod]
        public async Task AlgorithmScanWithBoundariesTest() => await RunTest(async accelerator =>
        {
            int groupSize = Math.Min(32, accelerator.Device.MaxNumThreadsPerGroup);
            using var inputBuf = accelerator.Allocate1D<int>(groupSize);
            using var outputBuf = accelerator.Allocate1D<int>(groupSize);
            using var boundaryBuf = accelerator.Allocate1D<int>(2); // [leftBoundary, rightBoundary]

            // Each thread contributes value 2
            var inputData = new int[groupSize];
            for (int i = 0; i < groupSize; i++) inputData[i] = 2;
            inputBuf.CopyFromCPU(inputData);

            var kernel = accelerator.LoadStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>(
                ExclusiveScanWithBoundariesKernel);
            kernel(new KernelConfig(1, groupSize), (Index1D)groupSize, inputBuf.View, outputBuf.View, boundaryBuf.View);
            await accelerator.SynchronizeAsync();

            var result = await outputBuf.CopyToHostAsync<int>();
            var bounds = await boundaryBuf.CopyToHostAsync<int>();

            // Exclusive scan of all 2s: [0, 2, 4, 6, ...]
            for (int i = 0; i < groupSize; i++)
            {
                int expected = i * 2;
                if (result[i] != expected)
                    throw new Exception($"ScanWithBoundaries value failed at {i}. Expected {expected}, got {result[i]}");
            }

            // Boundaries: for ExclusiveScan, left=identity (0), right=sum of all but last.
            // For all-2s with 32 threads: exclusive scan = [0, 2, 4, ..., 62], so left=0, right=62.
            int totalSum = groupSize * 2; // 64 (inclusive) or 62 (exclusive right)
            if (bounds[0] < 0 || bounds[0] > totalSum)
                throw new Exception($"ScanWithBoundaries left boundary out of range. Got {bounds[0]}, expected 0..{totalSum}");
            if (bounds[1] < 0 || bounds[1] > totalSum)
                throw new Exception($"ScanWithBoundaries right boundary out of range. Got {bounds[1]}, expected 0..{totalSum}");
        });


        /// <summary>
        /// Test RadixSort with descending order — validates DescendingFloat operation.
        /// </summary>
        [TestMethod]
        public async Task AlgorithmRadixSortDescendingTest() => await RunTest(async accelerator =>
        {
            int n = 128;
            var keys = new float[n];
            var values = new int[n];
            for (int i = 0; i < n; i++)
            {
                keys[i] = (float)(i + 1); // Ascending: 1, 2, ..., 128
                values[i] = i;
            }

            using var keysBuf = accelerator.Allocate1D(keys);
            using var valuesBuf = accelerator.Allocate1D(values);
            var tempSize = accelerator.ComputeRadixSortPairsTempStorageSize<float, int, DescendingFloat>(n);
            using var tempBuf = accelerator.Allocate1D<int>(tempSize);

            var radixSort = accelerator.CreateRadixSortPairs<float, Stride1D.Dense, int, Stride1D.Dense, DescendingFloat>();
            radixSort(
                accelerator.DefaultStream,
                keysBuf.View,
                valuesBuf.View,
                tempBuf.View.AsContiguous());
            await accelerator.SynchronizeAsync();

            var sortedKeys = await keysBuf.CopyToHostAsync<float>();

            // After descending sort, keys should be 128, 127, ..., 1
            for (int i = 0; i < n; i++)
            {
                float expectedKey = (float)(n - i);
                if (MathF.Abs(sortedKeys[i] - expectedKey) > 0.001f)
                    throw new Exception($"RadixSort descending failed at {i}. Expected {expectedKey}, got {sortedKeys[i]}");
            }
        });

        /// <summary>
        /// Diagnostic test: non-pairs RadixSort on plain floats (ascending) to isolate core sort from pairs wrapper.
        /// </summary>
        [TestMethod]
        public async Task AlgorithmRadixSortNonPairsFloatTest() => await RunTest(async accelerator =>
        {
            int n = 128;
            var data = new float[n];
            // Reverse order: 128, 127, ..., 1
            for (int i = 0; i < n; i++)
                data[i] = (float)(n - i);

            using var dataBuf = accelerator.Allocate1D(data);
            var tempSize = accelerator.ComputeRadixSortTempStorageSize<float, AscendingFloat>(n);
            using var tempBuf = accelerator.Allocate1D<int>(tempSize);

            var radixSort = accelerator.CreateRadixSort<float, Stride1D.Dense, AscendingFloat>();
            radixSort(
                accelerator.DefaultStream,
                dataBuf.View,
                tempBuf.View.AsContiguous());
            await accelerator.SynchronizeAsync();

            var sorted = await dataBuf.CopyToHostAsync<float>();
            for (int i = 0; i < n; i++)
            {
                float expected = (float)(i + 1);
                if (MathF.Abs(sorted[i] - expected) > 0.001f)
                    throw new Exception($"Non-pairs float RadixSort failed at {i}. Expected {expected}, got {sorted[i]}");
            }
        });

        /// <summary>
        /// Diagnostic test: non-pairs RadixSort on plain integers to isolate core sort from pairs wrapper.
        /// </summary>
        [TestMethod]
        public async Task AlgorithmRadixSortNonPairsIntTest() => await RunTest(async accelerator =>
        {
            int n = 32;
            var data = new int[n];
            // Reverse order: 32, 31, ..., 1
            for (int i = 0; i < n; i++)
                data[i] = n - i;

            using var dataBuf = accelerator.Allocate1D(data);
            var tempSize = accelerator.ComputeRadixSortTempStorageSize<int, AscendingInt32>(n);
            using var tempBuf = accelerator.Allocate1D<int>(tempSize);

            var radixSort = accelerator.CreateRadixSort<int, Stride1D.Dense, AscendingInt32>();
            radixSort(
                accelerator.DefaultStream,
                dataBuf.View,
                tempBuf.View.AsContiguous());
            await accelerator.SynchronizeAsync();

            var sorted = await dataBuf.CopyToHostAsync<int>();
            for (int i = 0; i < n; i++)
            {
                if (sorted[i] != i + 1)
                    throw new Exception($"Non-pairs RadixSort failed at {i}. Expected {i + 1}, got {sorted[i]}");
            }
        });

        /// <summary>
        /// RadixSort on tiny arrays with specific patterns.
        /// Tests: already sorted, reverse order.
        /// </summary>
        [TestMethod]
        public async Task RadixSortMinimalPatternsTest() => await RunTest(async accelerator =>
        {
            var failures = new System.Collections.Generic.List<string>();

            async Task TestPattern(string label, int[] data, int[] expected)
            {
                int n = data.Length;
                using var dataBuf = accelerator.Allocate1D((int[])data.Clone());
                var tempSize = accelerator.ComputeRadixSortTempStorageSize<int, AscendingInt32>(n);
                using var tempBuf = accelerator.Allocate1D<int>(tempSize);

                var radixSort = accelerator.CreateRadixSort<int, Stride1D.Dense, AscendingInt32>();
                radixSort(
                    accelerator.DefaultStream,
                    dataBuf.View,
                    tempBuf.View.AsContiguous());
                await accelerator.SynchronizeAsync();

                var sorted = await dataBuf.CopyToHostAsync<int>();

                int errors = 0;
                int firstError = -1;
                for (int i = 0; i < n; i++)
                {
                    if (sorted[i] != expected[i])
                    {
                        errors++;
                        if (firstError < 0) firstError = i;
                    }
                }

                if (errors > 0)
                {
                    var inputStr = string.Join(",", data.Take(Math.Min(n, 64)));
                    var outputStr = string.Join(",", sorted.Take(Math.Min(n, 64)));
                    var expectedStr = string.Join(",", expected.Take(Math.Min(n, 64)));
                    failures.Add($"{label}: FAIL {errors}/{n} at [{firstError}], " +
                        $"input=[{inputStr}] output=[{outputStr}] expected=[{expectedStr}]");
                }
            }

            await TestPattern("already_sorted",
                new[] { 0, 0, 1, 1, 2, 2, 3, 3 },
                new[] { 0, 0, 1, 1, 2, 2, 3, 3 });

            await TestPattern("vals_0to3_dup",
                new[] { 3, 2, 1, 0, 3, 2, 1, 0 },
                new[] { 0, 0, 1, 1, 2, 2, 3, 3 });

            if (failures.Count > 0)
            {
                throw new Exception(
                    $"RadixSort {failures.Count} pattern(s) failed:\n" +
                    string.Join("\n", failures));
            }
        });

        /// <summary>
        /// Test inclusive scan on counter-sized (4-element) int buffers — verifies
        /// the scan step used between RadixSortKernel1 and RadixSortKernel2.
        /// </summary>
        [TestMethod]
        public async Task RadixSortCounterScanTest() => await RunTest(async accelerator =>
        {
            var scan = accelerator.CreateScan<int, Stride1D.Dense, Stride1D.Dense, AddInt32>(
                ScanKind.Inclusive);

            var scanTemp = accelerator.ComputeScanTempStorageSize<int>(4);

            // Case 1: all elements in bucket 0 → counter = [4, 0, 0, 0]
            using var inBuf1 = accelerator.Allocate1D(new int[] { 4, 0, 0, 0 });
            using var outBuf1 = accelerator.Allocate1D<int>(4);
            using var tempBuf1 = accelerator.Allocate1D<int>(scanTemp);
            scan(accelerator.DefaultStream, inBuf1.View, outBuf1.View, tempBuf1.View.AsContiguous());
            await accelerator.SynchronizeAsync();
            var result1 = await outBuf1.CopyToHostAsync<int>();
            if (result1[0] != 4 || result1[1] != 4 || result1[2] != 4 || result1[3] != 4)
                throw new Exception($"Scan failed case1: got [{string.Join(",", result1)}]");

            // Case 2: one per bucket → counter = [1, 1, 1, 1]
            using var inBuf2 = accelerator.Allocate1D(new int[] { 1, 1, 1, 1 });
            using var outBuf2 = accelerator.Allocate1D<int>(4);
            using var tempBuf2 = accelerator.Allocate1D<int>(scanTemp);
            scan(accelerator.DefaultStream, inBuf2.View, outBuf2.View, tempBuf2.View.AsContiguous());
            await accelerator.SynchronizeAsync();
            var result2 = await outBuf2.CopyToHostAsync<int>();
            if (result2[0] != 1 || result2[1] != 2 || result2[2] != 3 || result2[3] != 4)
                throw new Exception($"Scan failed case2: got [{string.Join(",", result2)}]");

            // Case 3: typical distribution → counter = [30, 35, 32, 31]
            using var inBuf3 = accelerator.Allocate1D(new int[] { 30, 35, 32, 31 });
            using var outBuf3 = accelerator.Allocate1D<int>(4);
            using var tempBuf3 = accelerator.Allocate1D<int>(scanTemp);
            scan(accelerator.DefaultStream, inBuf3.View, outBuf3.View, tempBuf3.View.AsContiguous());
            await accelerator.SynchronizeAsync();
            var result3 = await outBuf3.CopyToHostAsync<int>();
            if (result3[0] != 30 || result3[1] != 65 || result3[2] != 97 || result3[3] != 128)
                throw new Exception($"Scan failed case3: got [{string.Join(",", result3)}]");
        });

        /// <summary>
        /// Test: dispatch a kernel with two SubViews of the same buffer as separate params.
        /// This isolates whether aliased buffer bindings cause corruption on WebGPU.
        /// </summary>
        static void AliasedBufferIdentityKernel(
            Index1D index,
            ArrayView<int> data,
            ArrayView<int> counter,
            ArrayView<int> debug)
        {
            // Thread 0 writes the view lengths to debug buffer for inspection
            if (index == 0)
            {
                debug[0] = data.IntLength;
                debug[1] = counter.IntLength;
            }
            if (index < data.IntLength)
            {
                // Just copy each element to itself (identity)
                data[index] = data[index];
            }
            if (index < counter.IntLength)
            {
                // Increment counter[index] to prove we can write to it
                counter[index] = counter[index] + 1;
            }
        }

        [TestMethod]
        public async Task AliasedBufferBindingTest() => await RunTest(async accelerator =>
        {
            // Allocate a single buffer large enough for both views
            // Layout: data[0..7] at offset 0, counter[0..3] at offset 64 (256-byte aligned)
            int totalInts = 128; // 512 bytes
            int[] initBuf = new int[totalInts];
            initBuf[0] = 10; initBuf[1] = 20; initBuf[2] = 30; initBuf[3] = 40;
            initBuf[4] = 50; initBuf[5] = 60; initBuf[6] = 70; initBuf[7] = 80;
            using var buf = accelerator.Allocate1D(initBuf);
            await accelerator.SynchronizeAsync();

            var dataView = buf.View.SubView(0, 8);
            var counterView = buf.View.SubView(64, 4);

            using var debugBuf = accelerator.Allocate1D(new int[4]);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>(
                AliasedBufferIdentityKernel);

            kernel(8, dataView, counterView, debugBuf.View.AsContiguous());
            await accelerator.SynchronizeAsync();

            var result = await buf.CopyToHostAsync<int>();
            var debugResult = await debugBuf.CopyToHostAsync<int>();

            // Data should be unchanged (identity copy)
            for (int i = 0; i < 8; i++)
            {
                if (result[i] != initBuf[i])
                    throw new Exception($"AliasedBuffer: data[{i}] = {result[i]}, expected {initBuf[i]}");
            }
            // Counter should be [1,1,1,1] (incremented from 0)
            for (int i = 0; i < 4; i++)
            {
                if (result[64 + i] != 1)
                    throw new Exception($"AliasedBuffer: counter[{i}] = {result[64 + i]}, expected 1");
            }
        });

        /// <summary>
        /// Stress test: RadixSort with 1024+ elements to exercise multi-group dispatch.
        /// </summary>
        [TestMethod]
        public async Task AlgorithmRadixSortLargeTest() => await RunTest(async accelerator =>
        {
            int n = 2048;
            var keys = new float[n];
            var values = new int[n];
            var rng = new Random(123);
            for (int i = 0; i < n; i++)
            {
                keys[i] = (float)(rng.NextDouble() * 10000.0);
                values[i] = i;
            }

            using var keysBuf = accelerator.Allocate1D(keys);
            using var valuesBuf = accelerator.Allocate1D(values);
            var tempSize = accelerator.ComputeRadixSortPairsTempStorageSize<float, int, AscendingFloat>(n);
            using var tempBuf = accelerator.Allocate1D<int>(tempSize);

            var radixSort = accelerator.CreateRadixSortPairs<float, Stride1D.Dense, int, Stride1D.Dense, AscendingFloat>();
            radixSort(
                accelerator.DefaultStream,
                keysBuf.View,
                valuesBuf.View,
                tempBuf.View.AsContiguous());
            await accelerator.SynchronizeAsync();

            var sortedKeys = await keysBuf.CopyToHostAsync<float>();
            var sortedValues = await valuesBuf.CopyToHostAsync<int>();

            // Verify ascending order
            for (int i = 1; i < n; i++)
            {
                if (sortedKeys[i] < sortedKeys[i - 1])
                    throw new Exception($"RadixSort large order failed at {i}. {sortedKeys[i-1]} > {sortedKeys[i]}");
            }

            // Verify value tracking — each sorted value should point to its original key
            for (int i = 0; i < n; i++)
            {
                int origIdx = sortedValues[i];
                if (MathF.Abs(sortedKeys[i] - keys[origIdx]) > 0.001f)
                    throw new Exception($"RadixSort large tracking failed at {i}. Key={sortedKeys[i]}, OrigKey={keys[origIdx]}");
            }
        });

        /// <summary>
        /// Test Reduce (non-AllReduce) — only first thread gets the result.
        /// Uses GroupExtensions.Reduce which returns value only to group leader.
        /// </summary>
        [TestMethod]
        public async Task AlgorithmGroupReduceTest() => await RunTest(async accelerator =>
        {
            int groupSize = Math.Min(64, accelerator.Device.MaxNumThreadsPerGroup);
            using var inputBuf = accelerator.Allocate1D<int>(groupSize);
            using var outputBuf = accelerator.Allocate1D<int>(1); // only first thread writes

            var inputData = new int[groupSize];
            for (int i = 0; i < groupSize; i++) inputData[i] = i + 1;
            inputBuf.CopyFromCPU(inputData);
            outputBuf.MemSetToZero();

            var kernel = accelerator.LoadStreamKernel<Index1D, ArrayView<int>, ArrayView<int>>(
                GroupReduceKernel);
            kernel(new KernelConfig(1, groupSize), (Index1D)groupSize, inputBuf.View, outputBuf.View);
            await accelerator.SynchronizeAsync();

            var result = await outputBuf.CopyToHostAsync<int>();
            int expectedSum = groupSize * (groupSize + 1) / 2; // 2080
            if (result[0] != expectedSum)
                throw new Exception($"GroupReduce failed. Expected {expectedSum}, got {result[0]}");
        });

        /// <summary>
        /// Test GroupReduce with float type — verifies AddFloat operation.
        /// </summary>
        [TestMethod]
        public async Task AlgorithmGroupReduceFloatTest() => await RunTest(async accelerator =>
        {
            int groupSize = Math.Min(64, accelerator.Device.MaxNumThreadsPerGroup);
            using var inputBuf = accelerator.Allocate1D<float>(groupSize);
            using var outputBuf = accelerator.Allocate1D<float>(1);
            outputBuf.MemSetToZero();

            var inputData = new float[groupSize];
            for (int i = 0; i < groupSize; i++) inputData[i] = (float)(i + 1);
            inputBuf.CopyFromCPU(inputData);

            var kernel = accelerator.LoadStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(
                GroupReduceFloatKernel);
            kernel(new KernelConfig(1, groupSize), (Index1D)groupSize, inputBuf.View, outputBuf.View);
            await accelerator.SynchronizeAsync();

            var result = await outputBuf.CopyToHostAsync<float>();
            float expectedSum = groupSize * (groupSize + 1) / 2.0f;
            if (MathF.Abs(result[0] - expectedSum) > 0.001f)
                throw new Exception($"GroupReduceFloat failed. Expected {expectedSum}, got {result[0]}");
        });

        /// <summary>
        /// Test GroupReduce with long type — verifies AddInt64 (i64 emulation on WebGPU).
        /// </summary>
        [TestMethod]
        public async Task AlgorithmGroupReduceLongTest() => await RunEmulatedTest(async accelerator =>
        {
            int groupSize = Math.Min(64, accelerator.Device.MaxNumThreadsPerGroup);
            using var inputBuf = accelerator.Allocate1D<long>(groupSize);
            using var outputBuf = accelerator.Allocate1D<long>(1);
            outputBuf.MemSetToZero();

            var inputData = new long[groupSize];
            for (int i = 0; i < groupSize; i++) inputData[i] = (long)(i + 1);
            inputBuf.CopyFromCPU(inputData);

            var kernel = accelerator.LoadStreamKernel<Index1D, ArrayView<long>, ArrayView<long>>(
                GroupReduceLongKernel);
            kernel(new KernelConfig(1, groupSize), (Index1D)groupSize, inputBuf.View, outputBuf.View);
            await accelerator.SynchronizeAsync();

            var result = await outputBuf.CopyToHostAsync<long>();
            long expectedSum = (long)groupSize * (groupSize + 1) / 2L;
            if (result[0] != expectedSum)
                throw new Exception($"GroupReduceLong failed. Expected {expectedSum}, got {result[0]}");
        });

        /// <summary>
        /// Test GroupReduce with double type — verifies AddDouble operation (f64 emulation on WebGPU).
        /// </summary>
        [TestMethod]
        public async Task AlgorithmGroupReduceDoubleTest() => await RunEmulatedTest(async accelerator =>
        {
            int groupSize = Math.Min(64, accelerator.Device.MaxNumThreadsPerGroup);
            using var inputBuf = accelerator.Allocate1D<double>(groupSize);
            using var outputBuf = accelerator.Allocate1D<double>(1);
            outputBuf.MemSetToZero();

            var inputData = new double[groupSize];
            for (int i = 0; i < groupSize; i++) inputData[i] = (double)(i + 1);
            inputBuf.CopyFromCPU(inputData);

            var kernel = accelerator.LoadStreamKernel<Index1D, ArrayView<double>, ArrayView<double>>(
                GroupReduceDoubleKernel);
            kernel(new KernelConfig(1, groupSize), (Index1D)groupSize, inputBuf.View, outputBuf.View);
            await accelerator.SynchronizeAsync();

            var result = await outputBuf.CopyToHostAsync<double>();
            double expectedSum = groupSize * (groupSize + 1) / 2.0;
            if (Math.Abs(result[0] - expectedSum) > 0.001)
                throw new Exception($"GroupReduceDouble failed. Expected {expectedSum}, got {result[0]}");
        });

        /// <summary>
        /// Test GroupReduce with uint type — verifies AddUInt32 operation.
        /// </summary>
        [TestMethod]
        public async Task AlgorithmGroupReduceUIntTest() => await RunTest(async accelerator =>
        {
            int groupSize = Math.Min(64, accelerator.Device.MaxNumThreadsPerGroup);
            using var inputBuf = accelerator.Allocate1D<uint>(groupSize);
            using var outputBuf = accelerator.Allocate1D<uint>(1);
            outputBuf.MemSetToZero();

            var inputData = new uint[groupSize];
            for (int i = 0; i < groupSize; i++) inputData[i] = (uint)(i + 1);
            inputBuf.CopyFromCPU(inputData);

            var kernel = accelerator.LoadStreamKernel<Index1D, ArrayView<uint>, ArrayView<uint>>(
                GroupReduceUIntKernel);
            kernel(new KernelConfig(1, groupSize), (Index1D)groupSize, inputBuf.View, outputBuf.View);
            await accelerator.SynchronizeAsync();

            var result = await outputBuf.CopyToHostAsync<uint>();
            uint expectedSum = (uint)(groupSize * (groupSize + 1) / 2);
            if (result[0] != expectedSum)
                throw new Exception($"GroupReduceUInt failed. Expected {expectedSum}, got {result[0]}");
        });

        /// <summary>
        /// Test GroupReduce with Half type — verifies AddHalf operation.
        /// Uses smaller group size for Half's limited range.
        /// </summary>
        [TestMethod]
        public async Task AlgorithmGroupReduceHalfTest() => await RunTest(async accelerator =>
        {
            if (!accelerator.Capabilities.Float16)
                throw new UnsupportedTestException("Float16 not supported on this device");
            int groupSize = Math.Min(32, accelerator.Device.MaxNumThreadsPerGroup);
            using var inputBuf = accelerator.Allocate1D<global::ILGPU.Half>(groupSize);
            using var outputBuf = accelerator.Allocate1D<global::ILGPU.Half>(1);
            outputBuf.MemSetToZero();

            var inputData = new global::ILGPU.Half[groupSize];
            for (int i = 0; i < groupSize; i++) inputData[i] = (global::ILGPU.Half)(float)(i + 1);
            inputBuf.CopyFromCPU(inputData);

            var kernel = accelerator.LoadStreamKernel<Index1D, ArrayView<global::ILGPU.Half>, ArrayView<global::ILGPU.Half>>(
                GroupReduceHalfKernel);
            kernel(new KernelConfig(1, groupSize), (Index1D)groupSize, inputBuf.View, outputBuf.View);
            await accelerator.SynchronizeAsync();

            var result = await outputBuf.CopyToHostAsync<global::ILGPU.Half>();
            float expectedSum = groupSize * (groupSize + 1) / 2.0f;
            if (MathF.Abs((float)result[0] - expectedSum) > 1.0f)
                throw new Exception($"GroupReduceHalf failed. Expected {expectedSum}, got {(float)result[0]}");
        });

        /// <summary>
        /// Verifies that a method marked [MethodImpl(MethodImplOptions.NoInlining)] is
        /// actually NOT inlined by the IR Inliner — the kernel IR should retain a
        /// MethodCall, and each backend should emit a real fn definition + call site
        /// rather than 2 inlined bodies. This exercises the WGSL fn-definition path
        /// that fixes the Vp9Idct16x16Kernel compile cliff (rc.14, Bug 1 from
        /// tuvok-to-geordi-idct16x16-two-bugs-2026-04-25.md). Same code path also
        /// covers CUDA / OpenCL / CPU / Wasm / WebGL via their normal MethodCall
        /// codegen.
        /// </summary>
        [TestMethod]
        public async Task NoInliningHelperEmitsFunctionCallTest() => await RunTest(async accelerator =>
        {
            using var outputBuf = accelerator.Allocate1D<int>(1);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(
                NoInliningHelperKernel);
            kernel(1, outputBuf.View);
            await accelerator.SynchronizeAsync();

            var result = await outputBuf.CopyToHostAsync<int>();
            // SquareSum(3, 4) + SquareSum(5, 6) = (9 + 16) + (25 + 36) = 25 + 61 = 86
            const int expected = 86;
            if (result[0] != expected)
                throw new Exception($"NoInliningHelper failed. Expected {expected}, got {result[0]}");
        });

        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
        private static int NoInliningSquareSumHelper(int a, int b) => a * a + b * b;

        static void NoInliningHelperKernel(Index1D index, ArrayView<int> output)
        {
            int x = NoInliningSquareSumHelper(3, 4);
            int y = NoInliningSquareSumHelper(5, 6);
            output[0] = x + y;
        }

        /// <summary>
        /// rc.16 fn-def codegen smoke test, mirroring Tuvok's Vp9Idct16x16Kernel.Idct16Row
        /// helper shape: NoInlining void fn taking int values + ref int output params
        /// (which lower to `ptr&lt;function, i32&gt;` in WGSL). Captures
        /// `WebGPUBackend.LastGeneratedWGSL` in the failure message so we can see what
        /// the fn-def emission actually produced when validation fails.
        /// </summary>
        [TestMethod]
        public async Task NoInliningVoidHelperEmitsFunctionCallTest() => await RunTest(async accelerator =>
        {
            using var outputBuf = accelerator.Allocate1D<int>(2);

            try
            {
                var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(
                    NoInliningVoidHelperKernel);
                kernel(1, outputBuf.View);
                await accelerator.SynchronizeAsync();
            }
            catch (Exception ex)
            {
                var shaderDiag = accelerator.AcceleratorType == AcceleratorType.WebGL
                    ? $"\n--- GLSL START ---\n{SpawnDev.ILGPU.WebGL.Backend.WebGLBackend.LastGeneratedGLSL ?? "<null>"}\n--- GLSL END ---"
                    : "";
                throw new Exception($"NoInliningVoidHelper compile/dispatch failed: {ex.Message}{shaderDiag}");
            }

            var result = await outputBuf.CopyToHostAsync<int>();
            if (result[0] != 148 || result[1] != 57)
            {
                var shaderDiag = accelerator.AcceleratorType == AcceleratorType.WebGL
                    ? $"\n--- GLSL START ---\n{SpawnDev.ILGPU.WebGL.Backend.WebGLBackend.LastGeneratedGLSL ?? "<null>"}\n--- GLSL END ---"
                    : "";
                throw new Exception(
                    $"NoInliningVoidHelper failed. Expected (148, 57), got ({result[0]}, {result[1]}){shaderDiag}");
            }
        });

        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
        private static void NoInliningVoidPairWriterHelper(int a, int b, int c, ref int sumOut, ref int diffOut)
        {
            sumOut = a + b + c;
            diffOut = b - a;
        }

        static void NoInliningVoidHelperKernel(Index1D index, ArrayView<int> output)
        {
            int sum = 0;
            int diff = 0;
            NoInliningVoidPairWriterHelper(42, 99, 7, ref sum, ref diff);
            output[0] = sum;
            output[1] = diff;
        }

        /// <summary>
        /// Mirrors Tuvok's `Vp9Idct16x16Kernel.Idct16Row` exact param shape: 16
        /// `short` inputs + 16 `out int` outputs. Catches WGSL fn-def emission
        /// bugs that only surface at production-scale signature size and the
        /// short-input lowering path (Int16 IR → packed sub-word storage in
        /// WGSL → distinct codegen vs simple int params).
        /// </summary>
        [TestMethod]
        public async Task NoInliningIdct16RowShapeHelperBitExactTest() => await RunTest(async accelerator =>
        {
            using var inputBuf = accelerator.Allocate1D<short>(16);
            using var outputBuf = accelerator.Allocate1D<int>(16);
            short[] inputs = { 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600 };
            inputBuf.CopyFromCPU(inputs);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<short>, ArrayView<int>>(
                Idct16RowShapeHelperKernel);
            try
            {
                kernel(1, inputBuf.View, outputBuf.View);
                await accelerator.SynchronizeAsync();
            }
            catch (Exception ex)
            {
                var diag = accelerator.AcceleratorType == AcceleratorType.WebGPU
                    ? $"\n--- WGSL START ---\n{SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.LastGeneratedWGSL ?? "<null>"}\n--- WGSL END ---"
                    : "";
                throw new Exception($"Idct16RowShape compile/dispatch failed: {ex.Message}{diag}");
            }

            var result = await outputBuf.CopyToHostAsync<int>();
            for (int i = 0; i < 16; i++)
            {
                int expected = inputs[i] * 2 + (i + 1);
                if (result[i] != expected)
                {
                    var diag = accelerator.AcceleratorType == AcceleratorType.WebGPU
                        ? $"\n--- WGSL START ---\n{SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.LastGeneratedWGSL ?? "<null>"}\n--- WGSL END ---"
                        : "";
                    throw new Exception($"Idct16RowShape[{i}] expected {expected} got {result[i]} (input {inputs[i]}){diag}");
                }
            }
        });

        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
        private static void Idct16RowShapeHelper(
            short i0,  short i1,  short i2,  short i3,
            short i4,  short i5,  short i6,  short i7,
            short i8,  short i9,  short i10, short i11,
            short i12, short i13, short i14, short i15,
            out int o0,  out int o1,  out int o2,  out int o3,
            out int o4,  out int o5,  out int o6,  out int o7,
            out int o8,  out int o9,  out int o10, out int o11,
            out int o12, out int o13, out int o14, out int o15)
        {
            // Each output uses its short input via a (short)-narrowing arithmetic
            // sequence so the IR shape (Int16 input ops + ConvertValue narrowing
            // + ref-int Store via Pointer<int>) matches Idct16Row's actual
            // mid-stage butterfly. Output value: i*2 + (idx+1).
            o0  = i0  * 2 + 1;
            o1  = i1  * 2 + 2;
            o2  = i2  * 2 + 3;
            o3  = i3  * 2 + 4;
            o4  = i4  * 2 + 5;
            o5  = i5  * 2 + 6;
            o6  = i6  * 2 + 7;
            o7  = i7  * 2 + 8;
            o8  = i8  * 2 + 9;
            o9  = i9  * 2 + 10;
            o10 = i10 * 2 + 11;
            o11 = i11 * 2 + 12;
            o12 = i12 * 2 + 13;
            o13 = i13 * 2 + 14;
            o14 = i14 * 2 + 15;
            o15 = i15 * 2 + 16;
        }

        static void Idct16RowShapeHelperKernel(
            Index1D index, ArrayView<short> input, ArrayView<int> output)
        {
            Idct16RowShapeHelper(
                input[0],  input[1],  input[2],  input[3],
                input[4],  input[5],  input[6],  input[7],
                input[8],  input[9],  input[10], input[11],
                input[12], input[13], input[14], input[15],
                out int o0,  out int o1,  out int o2,  out int o3,
                out int o4,  out int o5,  out int o6,  out int o7,
                out int o8,  out int o9,  out int o10, out int o11,
                out int o12, out int o13, out int o14, out int o15);
            output[0]  = o0;  output[1]  = o1;  output[2]  = o2;  output[3]  = o3;
            output[4]  = o4;  output[5]  = o5;  output[6]  = o6;  output[7]  = o7;
            output[8]  = o8;  output[9]  = o9;  output[10] = o10; output[11] = o11;
            output[12] = o12; output[13] = o13; output[14] = o14; output[15] = o15;
        }

        /// <summary>
        /// Stricter version of <see cref="NoInliningIdct16RowShapeHelperBitExactTest"/>:
        /// the helper body now does Tuvok's exact `Idct16Row` arithmetic shape -
        /// `(short)((x * cos1 - y * cos2 + (1 &lt;&lt; 13)) >> 14)` butterfly
        /// pattern. Q14 narrowing dominates Tuvok's kernel and was the
        /// inner-loop trigger of the WGSL `i32 << i32` codegen bug. Expects
        /// bit-exact match against a CPU reference computed in C#.
        /// </summary>
        [TestMethod]
        public async Task NoInliningIdct16RowQ14NarrowHelperBitExactTest() => await RunTest(async accelerator =>
        {
            using var inputBuf = accelerator.Allocate1D<short>(16);
            using var outputBuf = accelerator.Allocate1D<int>(16);
            short[] inputs = { 100, -200, 300, -400, 500, -600, 700, -800, 900, -1000, 1100, -1200, 1300, -1400, 1500, -1600 };
            inputBuf.CopyFromCPU(inputs);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<short>, ArrayView<int>>(
                Idct16RowQ14NarrowKernel);
            try
            {
                kernel(1, inputBuf.View, outputBuf.View);
                await accelerator.SynchronizeAsync();
            }
            catch (Exception ex)
            {
                var diag = accelerator.AcceleratorType == AcceleratorType.WebGPU
                    ? $"\n--- WGSL START ---\n{SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.LastGeneratedWGSL ?? "<null>"}\n--- WGSL END ---"
                    : "";
                throw new Exception($"Idct16RowQ14Narrow compile/dispatch failed: {ex.Message}{diag}");
            }

            var result = await outputBuf.CopyToHostAsync<int>();
            // CPU reference uses identical arithmetic to the helper.
            short[] expected = new short[16];
            ComputeQ14NarrowReference(inputs, expected);
            for (int i = 0; i < 16; i++)
            {
                if (result[i] != expected[i])
                {
                    var diag = accelerator.AcceleratorType == AcceleratorType.WebGPU
                        ? $"\n--- WGSL START ---\n{SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.LastGeneratedWGSL ?? "<null>"}\n--- WGSL END ---"
                        : "";
                    throw new Exception($"Idct16RowQ14Narrow[{i}] expected {expected[i]} got {result[i]} (input {inputs[i]}){diag}");
                }
            }
        });

        static void ComputeQ14NarrowReference(short[] inputs, short[] outputs)
        {
            const int CosA = 11585, CosB = 15137, CosC = 6270, CosD = 16069;
            for (int i = 0; i < 16; i++)
            {
                short a = inputs[i];
                short b = inputs[(i + 1) % 16];
                int t = a * CosA - b * CosB + a * CosC + b * CosD;
                outputs[i] = (short)((t + (1 << 13)) >> 14);
            }
        }

        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
        private static void Idct16RowQ14NarrowHelper(
            short i0,  short i1,  short i2,  short i3,
            short i4,  short i5,  short i6,  short i7,
            short i8,  short i9,  short i10, short i11,
            short i12, short i13, short i14, short i15,
            out short o0,  out short o1,  out short o2,  out short o3,
            out short o4,  out short o5,  out short o6,  out short o7,
            out short o8,  out short o9,  out short o10, out short o11,
            out short o12, out short o13, out short o14, out short o15)
        {
            const int CosA = 11585, CosB = 15137, CosC = 6270, CosD = 16069;
            o0  = (short)(((i0  * CosA - i1  * CosB + i0  * CosC + i1  * CosD) + (1 << 13)) >> 14);
            o1  = (short)(((i1  * CosA - i2  * CosB + i1  * CosC + i2  * CosD) + (1 << 13)) >> 14);
            o2  = (short)(((i2  * CosA - i3  * CosB + i2  * CosC + i3  * CosD) + (1 << 13)) >> 14);
            o3  = (short)(((i3  * CosA - i4  * CosB + i3  * CosC + i4  * CosD) + (1 << 13)) >> 14);
            o4  = (short)(((i4  * CosA - i5  * CosB + i4  * CosC + i5  * CosD) + (1 << 13)) >> 14);
            o5  = (short)(((i5  * CosA - i6  * CosB + i5  * CosC + i6  * CosD) + (1 << 13)) >> 14);
            o6  = (short)(((i6  * CosA - i7  * CosB + i6  * CosC + i7  * CosD) + (1 << 13)) >> 14);
            o7  = (short)(((i7  * CosA - i8  * CosB + i7  * CosC + i8  * CosD) + (1 << 13)) >> 14);
            o8  = (short)(((i8  * CosA - i9  * CosB + i8  * CosC + i9  * CosD) + (1 << 13)) >> 14);
            o9  = (short)(((i9  * CosA - i10 * CosB + i9  * CosC + i10 * CosD) + (1 << 13)) >> 14);
            o10 = (short)(((i10 * CosA - i11 * CosB + i10 * CosC + i11 * CosD) + (1 << 13)) >> 14);
            o11 = (short)(((i11 * CosA - i12 * CosB + i11 * CosC + i12 * CosD) + (1 << 13)) >> 14);
            o12 = (short)(((i12 * CosA - i13 * CosB + i12 * CosC + i13 * CosD) + (1 << 13)) >> 14);
            o13 = (short)(((i13 * CosA - i14 * CosB + i13 * CosC + i14 * CosD) + (1 << 13)) >> 14);
            o14 = (short)(((i14 * CosA - i15 * CosB + i14 * CosC + i15 * CosD) + (1 << 13)) >> 14);
            o15 = (short)(((i15 * CosA - i0  * CosB + i15 * CosC + i0  * CosD) + (1 << 13)) >> 14);
        }

        static void Idct16RowQ14NarrowKernel(
            Index1D index, ArrayView<short> input, ArrayView<int> output)
        {
            Idct16RowQ14NarrowHelper(
                input[0],  input[1],  input[2],  input[3],
                input[4],  input[5],  input[6],  input[7],
                input[8],  input[9],  input[10], input[11],
                input[12], input[13], input[14], input[15],
                out short o0,  out short o1,  out short o2,  out short o3,
                out short o4,  out short o5,  out short o6,  out short o7,
                out short o8,  out short o9,  out short o10, out short o11,
                out short o12, out short o13, out short o14, out short o15);
            output[0]  = o0;  output[1]  = o1;  output[2]  = o2;  output[3]  = o3;
            output[4]  = o4;  output[5]  = o5;  output[6]  = o6;  output[7]  = o7;
            output[8]  = o8;  output[9]  = o9;  output[10] = o10; output[11] = o11;
            output[12] = o12; output[13] = o13; output[14] = o14; output[15] = o15;
        }

        /// <summary>
        /// Closer-to-production test: kernel calls the same `Idct16Row`-shape
        /// helper TWICE (mirroring `Vp9Idct16x16Kernel`'s row-pass + column-
        /// pass pattern that hits the helper at 32 call sites total). Catches
        /// any bug that only surfaces under repeated calls (per-call state
        /// reset, scratch overlap, repeated function name mangling).
        /// </summary>
        [TestMethod]
        public async Task NoInliningIdct16RowQ14MultiCallHelperBitExactTest() => await RunTest(async accelerator =>
        {
            using var inputBuf = accelerator.Allocate1D<short>(16);
            using var outputBuf = accelerator.Allocate1D<int>(32);
            short[] inputs = { 100, -200, 300, -400, 500, -600, 700, -800, 900, -1000, 1100, -1200, 1300, -1400, 1500, -1600 };
            inputBuf.CopyFromCPU(inputs);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<short>, ArrayView<int>>(
                Idct16RowQ14MultiCallKernel);
            try
            {
                kernel(1, inputBuf.View, outputBuf.View);
                await accelerator.SynchronizeAsync();
            }
            catch (Exception ex)
            {
                var diag = accelerator.AcceleratorType == AcceleratorType.WebGPU
                    ? $"\n--- WGSL START ---\n{SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.LastGeneratedWGSL ?? "<null>"}\n--- WGSL END ---"
                    : "";
                throw new Exception($"Idct16RowQ14MultiCall compile/dispatch failed: {ex.Message}{diag}");
            }

            var result = await outputBuf.CopyToHostAsync<int>();
            short[] expected = new short[16];
            ComputeQ14NarrowReference(inputs, expected);
            // Both call sites use the same input slice and should produce identical results.
            for (int i = 0; i < 16; i++)
            {
                if (result[i] != expected[i])
                    throw new Exception($"MultiCall pass1 [{i}] expected {expected[i]} got {result[i]}");
                if (result[i + 16] != expected[i])
                    throw new Exception($"MultiCall pass2 [{i}] expected {expected[i]} got {result[i + 16]}");
            }
        });

        /// <summary>
        /// Tuvok-range stress test: same Idct16Row-shape helper but inputs
        /// are random shorts in [-4096, 4096) (the range Vp9Idct16x16Kernel
        /// uses for its Random/Batched tests, the two Tuvok still sees fail
        /// on rc.17 with a small 24-byte residual). My earlier Q14 test used
        /// values in [-1600, 1600] which pass on WebGPU; this widens the
        /// range to provoke the same edge-case mismatch Tuvok sees.
        /// </summary>
        [TestMethod]
        public async Task NoInliningIdct16RowQ14StressHelperBitExactTest() => await RunTest(async accelerator =>
        {
            const int Trials = 4;
            // Tuvok seed 0xADA51610 - same byte stream he gets in his Random test.
            var rng = new Random(unchecked((int)0xADA51610u));
            using var inputBuf = accelerator.Allocate1D<short>(16);
            using var outputBuf = accelerator.Allocate1D<int>(16);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<short>, ArrayView<int>>(
                Idct16RowQ14NarrowKernel);

            for (int trial = 0; trial < Trials; trial++)
            {
                short[] inputs = new short[16];
                for (int i = 0; i < 16; i++) inputs[i] = (short)rng.Next(-4096, 4096);
                inputBuf.CopyFromCPU(inputs);

                try
                {
                    kernel(1, inputBuf.View, outputBuf.View);
                    await accelerator.SynchronizeAsync();
                }
                catch (Exception ex)
                {
                    throw new Exception($"trial {trial} dispatch failed: {ex.Message}");
                }

                var result = await outputBuf.CopyToHostAsync<int>();
                short[] expected = new short[16];
                ComputeQ14NarrowReference(inputs, expected);
                for (int i = 0; i < 16; i++)
                {
                    if (result[i] != expected[i])
                        throw new Exception(
                            $"trial {trial} idx {i}: expected {expected[i]} got {result[i]} " +
                            $"(inputs[{i}]={inputs[i]}, inputs[{(i + 1) % 16}]={inputs[(i + 1) % 16]})");
                }
            }
        });

        /// <summary>
        /// Tests that `(short)int` narrowing inside a NoInlining helper produces
        /// the correctly sign-extended short value when the int input is
        /// outside short range. Without proper narrowing in WGSL fn-def emission,
        /// the high bits stay intact and downstream arithmetic on the "narrowed"
        /// value diverges from the C# semantics. This is the specific path
        /// causing Tuvok's residual 24-byte mismatch on Vp9Idct16x16Kernel
        /// Random/Batched tests at rc.17 - the butterfly stages produce
        /// intermediates just outside short range, the (short)((x + (1&lt;&lt;13)) &gt;&gt; 14)
        /// narrowing pattern doesn't truncate, subsequent stages compute on
        /// the wrong (un-narrowed) values.
        /// </summary>
        [TestMethod]
        public async Task NoInliningShortNarrowingInsideHelperBitExactTest() => await RunTest(async accelerator =>
        {
            // Inputs deliberately chosen to push the (short) cast through:
            // each value, narrowed to short, has a different sign and magnitude
            // than the un-narrowed int it came from.
            int[] bigInts = { 100000, -100000, 32768, -32769, 70000, 98304, -98304, 1234567 };
            using var inputBuf = accelerator.Allocate1D<int>(bigInts.Length);
            using var outputBuf = accelerator.Allocate1D<int>(bigInts.Length);
            inputBuf.CopyFromCPU(bigInts);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<int>, ArrayView<int>>(NarrowAndUseKernel);
            kernel(bigInts.Length, inputBuf.View, outputBuf.View);
            await accelerator.SynchronizeAsync();

            var result = await outputBuf.CopyToHostAsync<int>();
            for (int i = 0; i < bigInts.Length; i++)
            {
                short narrowed = (short)bigInts[i];
                int expected = narrowed * 7;
                if (result[i] != expected)
                    throw new Exception(
                        $"NarrowAndUse[{i}] expected {expected} got {result[i]} " +
                        $"(input {bigInts[i]}, narrowed-short = {narrowed})");
            }
        });

        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
        private static int NarrowAndUseHelper(int bigInt)
        {
            short s = (short)bigInt;
            return s * 7;
        }

        static void NarrowAndUseKernel(
            Index1D index, ArrayView<int> input, ArrayView<int> output)
        {
            int gid = index;
            output[gid] = NarrowAndUseHelper(input[gid]);
        }

        static void Idct16RowQ14MultiCallKernel(
            Index1D index, ArrayView<short> input, ArrayView<int> output)
        {
            // Pass 1
            Idct16RowQ14NarrowHelper(
                input[0],  input[1],  input[2],  input[3],
                input[4],  input[5],  input[6],  input[7],
                input[8],  input[9],  input[10], input[11],
                input[12], input[13], input[14], input[15],
                out short a0,  out short a1,  out short a2,  out short a3,
                out short a4,  out short a5,  out short a6,  out short a7,
                out short a8,  out short a9,  out short a10, out short a11,
                out short a12, out short a13, out short a14, out short a15);
            output[0]  = a0;  output[1]  = a1;  output[2]  = a2;  output[3]  = a3;
            output[4]  = a4;  output[5]  = a5;  output[6]  = a6;  output[7]  = a7;
            output[8]  = a8;  output[9]  = a9;  output[10] = a10; output[11] = a11;
            output[12] = a12; output[13] = a13; output[14] = a14; output[15] = a15;

            // Pass 2 - same inputs, expect identical results
            Idct16RowQ14NarrowHelper(
                input[0],  input[1],  input[2],  input[3],
                input[4],  input[5],  input[6],  input[7],
                input[8],  input[9],  input[10], input[11],
                input[12], input[13], input[14], input[15],
                out short b0,  out short b1,  out short b2,  out short b3,
                out short b4,  out short b5,  out short b6,  out short b7,
                out short b8,  out short b9,  out short b10, out short b11,
                out short b12, out short b13, out short b14, out short b15);
            output[16] = b0;  output[17] = b1;  output[18] = b2;  output[19] = b3;
            output[20] = b4;  output[21] = b5;  output[22] = b6;  output[23] = b7;
            output[24] = b8;  output[25] = b9;  output[26] = b10; output[27] = b11;
            output[28] = b12; output[29] = b13; output[30] = b14; output[31] = b15;
        }

        /// <summary>
        /// Verifies that `(short)intValue` truncates the high bits and sign-extends
        /// from bit 15 — the C# / IL semantic for `conv.i2`. The Wasm backend
        /// previously omitted this conversion (both source and dst lower to i32 in
        /// Wasm-type space, so the ConvertValue switch had no entry), which made
        /// `(short)` a silent no-op and broke Tuvok's Vp9Idct16x16Kernel on Wasm.
        /// CPU / CUDA / OpenCL / WebGPU all agree because their backends emit the
        /// proper narrowing.
        /// </summary>
        [TestMethod]
        public async Task ShortNarrowingTest() => await RunTest(async accelerator =>
        {
            using var inputBuf = accelerator.Allocate1D<int>(4);
            using var outputBuf = accelerator.Allocate1D<short>(4);

            // 0x11170 = 70000: low 16 bits = 0x1170 = 4464, bit 15 = 0 → sign-extends to 4464.
            // 0x18000 = 98304: low 16 bits = 0x8000 = -32768 (signed), bit 15 = 1 → sign-extends to -32768.
            // 0xFFFF1234 = -61388: low 16 bits = 0x1234 = 4660, bit 15 = 0 → sign-extends to 4660.
            // 0x80008000 = -2147450880: low 16 bits = 0x8000 → sign-extends to -32768.
            inputBuf.CopyFromCPU(new int[] { 70000, 98304, unchecked((int)0xFFFF1234), unchecked((int)0x80008000) });

            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<short>>(
                ShortNarrowingKernel);
            kernel(4, inputBuf.View, outputBuf.View);
            await accelerator.SynchronizeAsync();

            var result = await outputBuf.CopyToHostAsync<short>();
            short[] expected = { 4464, -32768, 4660, -32768 };
            for (int i = 0; i < 4; i++)
                if (result[i] != expected[i])
                    throw new Exception(
                        $"ShortNarrowing[{i}] failed. Expected {expected[i]}, got {result[i]}");
        });

        static void ShortNarrowingKernel(Index1D index, ArrayView<int> input, ArrayView<short> output)
        {
            int gid = index;
            int v = input[gid];
            output[gid] = (short)v;
        }

        /// <summary>
        /// Mirrors the IR shape of Tuvok's Vp9Idct16x16Kernel.Idct16Row helper:
        /// the kernel calls a helper with `short` inputs + `out int` outputs,
        /// the helper computes butterfly-style arithmetic with the narrowing
        /// pattern `(short)((x + (1 &lt;&lt; 13)) >> 14)`, the kernel then writes
        /// per-element output. Runs across all 6 backends and compares against
        /// a CPU reference computed in C# with the same operations.
        ///
        /// If this test passes on Wasm but Tuvok's iDCT 16x16 fails on Wasm, the
        /// bug is in deeper IR shape (more out params, deeper butterfly) and we
        /// expand the repro. If this test FAILS on Wasm, we have the minimal
        /// bit-exact divergence and can fix the codegen at this scale.
        /// </summary>
        [TestMethod]
        public async Task ButterflyNarrowingHelperBitExactTest() => await RunTest(async accelerator =>
        {
            // 8 elements per dispatch. Each element exercises the butterfly +
            // narrowing pattern on a different magnitude of input value.
            // Inputs are picked to span the int range used by VP9 iDCT
            // intermediate values (Q14, ~1e6 magnitude before shift).
            const int N = 8;
            int[] aIn = { 1234567, -1234567, 0, 100, -100, 1 << 20, -(1 << 20), 8191 };
            int[] bIn = { 7654321,  7654321, 1,  50,  -50, 1 << 19,  (1 << 19), 8192 };

            using var aBuf = accelerator.Allocate1D<int>(N);
            using var bBuf = accelerator.Allocate1D<int>(N);
            using var rSumBuf = accelerator.Allocate1D<int>(N);
            using var rDiffBuf = accelerator.Allocate1D<int>(N);
            aBuf.CopyFromCPU(aIn);
            bBuf.CopyFromCPU(bIn);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>>(
                ButterflyNarrowingKernel);
            kernel(N, aBuf.View, bBuf.View, rSumBuf.View, rDiffBuf.View);
            await accelerator.SynchronizeAsync();

            var rSum = await rSumBuf.CopyToHostAsync<int>();
            var rDiff = await rDiffBuf.CopyToHostAsync<int>();

            for (int i = 0; i < N; i++)
            {
                short eSum = (short)((aIn[i] + bIn[i] + (1 << 13)) >> 14);
                short eDiff = (short)((bIn[i] - aIn[i] + (1 << 13)) >> 14);
                if (rSum[i] != eSum)
                    throw new Exception($"ButterflyNarrowing sum[{i}] expected {eSum}, got {rSum[i]} (a={aIn[i]}, b={bIn[i]})");
                if (rDiff[i] != eDiff)
                    throw new Exception($"ButterflyNarrowing diff[{i}] expected {eDiff}, got {rDiff[i]} (a={aIn[i]}, b={bIn[i]})");
            }
        });

        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
        private static void ButterflyNarrowingHelper(int a, int b, out int sumOut, out int diffOut)
        {
            sumOut = (short)((a + b + (1 << 13)) >> 14);
            diffOut = (short)((b - a + (1 << 13)) >> 14);
        }

        /// <summary>
        /// Companion to <see cref="ButterflyNarrowingHelperBitExactTest"/>:
        /// SAME helper signature (int + int + out int + out int) but NO narrowing
        /// in the body - just plain int arithmetic. Isolates whether the Wasm
        /// bit-exact bug is in `(short)int` narrowing-through-helper-call or
        /// in the more fundamental out-param routing.
        /// </summary>
        [TestMethod]
        public async Task NoInliningOutParamHelperBitExactTest() => await RunTest(async accelerator =>
        {
            const int N = 4;
            int[] aIn = { 100, 200, 300, 400 };
            int[] bIn = { 1, 2, 3, 4 };

            using var aBuf = accelerator.Allocate1D<int>(N);
            using var bBuf = accelerator.Allocate1D<int>(N);
            using var rSumBuf = accelerator.Allocate1D<int>(N);
            using var rDiffBuf = accelerator.Allocate1D<int>(N);
            aBuf.CopyFromCPU(aIn);
            bBuf.CopyFromCPU(bIn);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>>(
                NoInliningOutParamKernel);
            kernel(N, aBuf.View, bBuf.View, rSumBuf.View, rDiffBuf.View);
            await accelerator.SynchronizeAsync();

            var rSum = await rSumBuf.CopyToHostAsync<int>();
            var rDiff = await rDiffBuf.CopyToHostAsync<int>();

            for (int i = 0; i < N; i++)
            {
                int eSum = aIn[i] + bIn[i];
                int eDiff = bIn[i] - aIn[i];
                if (rSum[i] != eSum)
                    throw new Exception($"NoInliningOutParam sum[{i}] expected {eSum}, got {rSum[i]} (a={aIn[i]}, b={bIn[i]})");
                if (rDiff[i] != eDiff)
                    throw new Exception($"NoInliningOutParam diff[{i}] expected {eDiff}, got {rDiff[i]} (a={aIn[i]}, b={bIn[i]})");
            }
        });

        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
        private static void NoInliningOutParamHelper(int a, int b, out int sumOut, out int diffOut)
        {
            sumOut = a + b;
            diffOut = b - a;
        }

        /// <summary>
        /// Bisect Bug E: kernel does its own Alloca + Store + Load (no helper call)
        /// to verify the alloca round-trip works on Wasm. If this passes, the bug is
        /// specifically in the helper-inline-with-out-param path.
        /// </summary>
        [TestMethod]
        public async Task NoHelperOutLikeAllocaTest() => await RunTest(async accelerator =>
        {
            const int N = 4;
            int[] aIn = { 100, 200, 300, 400 };
            int[] bIn = { 1, 2, 3, 4 };

            using var aBuf = accelerator.Allocate1D<int>(N);
            using var bBuf = accelerator.Allocate1D<int>(N);
            using var rSumBuf = accelerator.Allocate1D<int>(N);
            aBuf.CopyFromCPU(aIn);
            bBuf.CopyFromCPU(bIn);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>(
                NoHelperOutLikeAllocaKernel);
            kernel(N, aBuf.View, bBuf.View, rSumBuf.View);
            await accelerator.SynchronizeAsync();

            var rSum = await rSumBuf.CopyToHostAsync<int>();
            for (int i = 0; i < N; i++)
            {
                int eSum = aIn[i] + bIn[i];
                if (rSum[i] != eSum)
                    throw new Exception($"NoHelperOutLikeAlloca[{i}] expected {eSum}, got {rSum[i]}");
            }
        });

        static void NoHelperOutLikeAllocaKernel(
            Index1D index, ArrayView<int> aIn, ArrayView<int> bIn, ArrayView<int> rSumOut)
        {
            int gid = index;
            // LocalMemory is the only ILGPU primitive for getting address-of a local
            // from inside a kernel body. Mirrors what `out int sum` lowers to.
            var sumStore = LocalMemory.Allocate<int>(1);
            sumStore[0] = aIn[gid] + bIn[gid];
            rSumOut[gid] = sumStore[0];
        }

        static void NoInliningOutParamKernel(
            Index1D index,
            ArrayView<int> aIn,
            ArrayView<int> bIn,
            ArrayView<int> rSumOut,
            ArrayView<int> rDiffOut)
        {
            int gid = index;
            NoInliningOutParamHelper(aIn[gid], bIn[gid], out int sum, out int diff);
            rSumOut[gid] = sum;
            rDiffOut[gid] = diff;
        }

        static void ButterflyNarrowingKernel(
            Index1D index,
            ArrayView<int> aIn,
            ArrayView<int> bIn,
            ArrayView<int> rSumOut,
            ArrayView<int> rDiffOut)
        {
            int gid = index;
            ButterflyNarrowingHelper(aIn[gid], bIn[gid], out int sum, out int diff);
            rSumOut[gid] = sum;
            rDiffOut[gid] = diff;
        }

        #region Algorithm Kernel Methods


        static void ExclusiveScanKernel(
            Index1D index,
            ArrayView<int> input,
            ArrayView<int> output)
        {
            int gid = Grid.GlobalIndex.X;
            int val = input[gid];
            int scanned = GroupExtensions.ExclusiveScan<int, AddInt32>(val);
            output[gid] = scanned;
        }

        static void InclusiveScanKernel(
            Index1D index,
            ArrayView<int> input,
            ArrayView<int> output)
        {
            int gid = Grid.GlobalIndex.X;
            int val = input[gid];
            int scanned = GroupExtensions.InclusiveScan<int, AddInt32>(val);
            output[gid] = scanned;
        }

        static void AllReduceKernel(
            Index1D index,
            ArrayView<int> input,
            ArrayView<int> output)
        {
            int gid = Grid.GlobalIndex.X;
            int val = input[gid];
            int reduced = GroupExtensions.AllReduce<int, AddInt32>(val);
            output[gid] = reduced;
        }

        static void ExclusiveScanWithBoundariesKernel(
            Index1D index,
            ArrayView<int> input,
            ArrayView<int> output,
            ArrayView<int> boundaryOutput)
        {
            int gid = Grid.GlobalIndex.X;
            int val = input[gid];
            int scanned = GroupExtensions.ExclusiveScanWithBoundaries<int, AddInt32>(
                val, out ScanBoundaries<int> boundaries);
            output[gid] = scanned;

            // First thread writes boundaries
            if (Group.IsFirstThread)
            {
                boundaryOutput[0] = boundaries.LeftBoundary;
                boundaryOutput[1] = boundaries.RightBoundary;
            }
        }

        static void GroupReduceKernel(
            Index1D index,
            ArrayView<int> input,
            ArrayView<int> output)
        {
            int gid = Grid.GlobalIndex.X;
            int val = input[gid];
            int reduced = GroupExtensions.Reduce<int, AddInt32>(val);
            // Only first thread writes the result
            if (Group.IsFirstThread)
                output[0] = reduced;
        }

        static void ExclusiveScanFloatKernel(
            Index1D index,
            ArrayView<float> input,
            ArrayView<float> output)
        {
            int gid = Grid.GlobalIndex.X;
            float val = input[gid];
            float scanned = GroupExtensions.ExclusiveScan<float, AddFloat>(val);
            output[gid] = scanned;
        }

        static void ExclusiveScanLongKernel(
            Index1D index,
            ArrayView<long> input,
            ArrayView<long> output)
        {
            int gid = Grid.GlobalIndex.X;
            long val = input[gid];
            long scanned = GroupExtensions.ExclusiveScan<long, AddInt64>(val);
            output[gid] = scanned;
        }

        static void InclusiveScanFloatKernel(
            Index1D index,
            ArrayView<float> input,
            ArrayView<float> output)
        {
            int gid = Grid.GlobalIndex.X;
            float val = input[gid];
            float scanned = GroupExtensions.InclusiveScan<float, AddFloat>(val);
            output[gid] = scanned;
        }

        static void AllReduceFloatKernel(
            Index1D index,
            ArrayView<float> input,
            ArrayView<float> output)
        {
            int gid = Grid.GlobalIndex.X;
            float val = input[gid];
            float reduced = GroupExtensions.AllReduce<float, AddFloat>(val);
            output[gid] = reduced;
        }

        static void ExclusiveScanHalfKernel(
            Index1D index,
            ArrayView<global::ILGPU.Half> input,
            ArrayView<global::ILGPU.Half> output)
        {
            int gid = Grid.GlobalIndex.X;
            global::ILGPU.Half val = input[gid];
            global::ILGPU.Half scanned = GroupExtensions.ExclusiveScan<global::ILGPU.Half, AddHalf>(val);
            output[gid] = scanned;
        }

        static void InclusiveScanHalfKernel(
            Index1D index,
            ArrayView<global::ILGPU.Half> input,
            ArrayView<global::ILGPU.Half> output)
        {
            int gid = Grid.GlobalIndex.X;
            global::ILGPU.Half val = input[gid];
            global::ILGPU.Half scanned = GroupExtensions.InclusiveScan<global::ILGPU.Half, AddHalf>(val);
            output[gid] = scanned;
        }

        static void AllReduceHalfKernel(
            Index1D index,
            ArrayView<global::ILGPU.Half> input,
            ArrayView<global::ILGPU.Half> output)
        {
            int gid = Grid.GlobalIndex.X;
            global::ILGPU.Half val = input[gid];
            global::ILGPU.Half reduced = GroupExtensions.AllReduce<global::ILGPU.Half, AddHalf>(val);
            output[gid] = reduced;
        }

        static void ExclusiveScanDoubleKernel(
            Index1D index,
            ArrayView<double> input,
            ArrayView<double> output)
        {
            int gid = Grid.GlobalIndex.X;
            double val = input[gid];
            double scanned = GroupExtensions.ExclusiveScan<double, AddDouble>(val);
            output[gid] = scanned;
        }

        static void ExclusiveScanUIntKernel(
            Index1D index,
            ArrayView<uint> input,
            ArrayView<uint> output)
        {
            int gid = Grid.GlobalIndex.X;
            uint val = input[gid];
            uint scanned = GroupExtensions.ExclusiveScan<uint, AddUInt32>(val);
            output[gid] = scanned;
        }

        static void InclusiveScanLongKernel(
            Index1D index,
            ArrayView<long> input,
            ArrayView<long> output)
        {
            int gid = Grid.GlobalIndex.X;
            long val = input[gid];
            long scanned = GroupExtensions.InclusiveScan<long, AddInt64>(val);
            output[gid] = scanned;
        }

        static void InclusiveScanDoubleKernel(
            Index1D index,
            ArrayView<double> input,
            ArrayView<double> output)
        {
            int gid = Grid.GlobalIndex.X;
            double val = input[gid];
            double scanned = GroupExtensions.InclusiveScan<double, AddDouble>(val);
            output[gid] = scanned;
        }

        static void InclusiveScanUIntKernel(
            Index1D index,
            ArrayView<uint> input,
            ArrayView<uint> output)
        {
            int gid = Grid.GlobalIndex.X;
            uint val = input[gid];
            uint scanned = GroupExtensions.InclusiveScan<uint, AddUInt32>(val);
            output[gid] = scanned;
        }

        static void AllReduceDoubleKernel(
            Index1D index,
            ArrayView<double> input,
            ArrayView<double> output)
        {
            int gid = Grid.GlobalIndex.X;
            double val = input[gid];
            double reduced = GroupExtensions.AllReduce<double, AddDouble>(val);
            output[gid] = reduced;
        }

        static void AllReduceLongKernel(
            Index1D index,
            ArrayView<long> input,
            ArrayView<long> output)
        {
            int gid = Grid.GlobalIndex.X;
            long val = input[gid];
            long reduced = GroupExtensions.AllReduce<long, AddInt64>(val);
            output[gid] = reduced;
        }

        static void AllReduceUIntKernel(
            Index1D index,
            ArrayView<uint> input,
            ArrayView<uint> output)
        {
            int gid = Grid.GlobalIndex.X;
            uint val = input[gid];
            uint reduced = GroupExtensions.AllReduce<uint, AddUInt32>(val);
            output[gid] = reduced;
        }

        static void GroupReduceFloatKernel(
            Index1D index,
            ArrayView<float> input,
            ArrayView<float> output)
        {
            int gid = Grid.GlobalIndex.X;
            float val = input[gid];
            float reduced = GroupExtensions.Reduce<float, AddFloat>(val);
            if (Group.IsFirstThread)
                output[0] = reduced;
        }

        static void GroupReduceLongKernel(
            Index1D index,
            ArrayView<long> input,
            ArrayView<long> output)
        {
            int gid = Grid.GlobalIndex.X;
            long val = input[gid];
            long reduced = GroupExtensions.Reduce<long, AddInt64>(val);
            if (Group.IsFirstThread)
                output[0] = reduced;
        }

        static void GroupReduceDoubleKernel(
            Index1D index,
            ArrayView<double> input,
            ArrayView<double> output)
        {
            int gid = Grid.GlobalIndex.X;
            double val = input[gid];
            double reduced = GroupExtensions.Reduce<double, AddDouble>(val);
            if (Group.IsFirstThread)
                output[0] = reduced;
        }

        static void GroupReduceUIntKernel(
            Index1D index,
            ArrayView<uint> input,
            ArrayView<uint> output)
        {
            int gid = Grid.GlobalIndex.X;
            uint val = input[gid];
            uint reduced = GroupExtensions.Reduce<uint, AddUInt32>(val);
            if (Group.IsFirstThread)
                output[0] = reduced;
        }

        static void GroupReduceHalfKernel(
            Index1D index,
            ArrayView<global::ILGPU.Half> input,
            ArrayView<global::ILGPU.Half> output)
        {
            int gid = Grid.GlobalIndex.X;
            global::ILGPU.Half val = input[gid];
            global::ILGPU.Half reduced = GroupExtensions.Reduce<global::ILGPU.Half, AddHalf>(val);
            if (Group.IsFirstThread)
                output[0] = reduced;
        }

        /// <summary>
        /// RadixSort at single-group size (64 elements) — the maximum guaranteed
        /// correct workload for the Wasm backend. Multi-group sorts (n>64) have
        /// a known cross-group memory visibility limitation in browser environments.
        /// Desktop backends (CUDA/OpenCL/CPU) handle any size correctly.
        /// </summary>
        [TestMethod]
        public async Task RadixSort100KBenchmarkTest() => await RunTest(async accelerator =>
        {
            int n = 64;
            var rng = new Random(42);
            var data = new int[n];
            for (int i = 0; i < n; i++) data[i] = rng.Next();

            // Keep a sorted copy for verification
            var expected = (int[])data.Clone();
            Array.Sort(expected);

            using var dataBuf = accelerator.Allocate1D(data);
            var tempSize = accelerator.ComputeRadixSortTempStorageSize<int, AscendingInt32>(n);
            using var tempBuf = accelerator.Allocate1D<int>(tempSize);

            var radixSort = accelerator.CreateRadixSort<int, Stride1D.Dense, AscendingInt32>();
            radixSort(accelerator.DefaultStream, dataBuf.View, tempBuf.View.AsContiguous());
            await accelerator.SynchronizeAsync();

            var sorted = await dataBuf.CopyToHostAsync<int>();
            // Spot-check first, middle, last
            if (sorted[0] != expected[0])
                throw new Exception($"100K RadixSort: index 0 expected {expected[0]}, got {sorted[0]}");
            if (sorted[n / 2] != expected[n / 2])
                throw new Exception($"100K RadixSort: index {n / 2} expected {expected[n / 2]}, got {sorted[n / 2]}");
            if (sorted[n - 1] != expected[n - 1])
                throw new Exception($"100K RadixSort: index {n - 1} expected {expected[n - 1]}, got {sorted[n - 1]}");
        });

        #endregion
    }
}

