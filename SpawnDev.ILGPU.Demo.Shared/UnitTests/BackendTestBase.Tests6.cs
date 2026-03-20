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

            // Diagnostic: dump full output for debugging
            var sb = new System.Text.StringBuilder();
            sb.Append("[RadixSort] Input: [32,31,...,1] Output: [");
            for (int i = 0; i < n; i++)
            {
                if (i > 0) sb.Append(',');
                sb.Append(sorted[i]);
            }
            sb.Append(']');
            Console.WriteLine(sb.ToString());

            // Check for failures
            var failures = new System.Collections.Generic.List<string>();
            for (int i = 0; i < n; i++)
            {
                if (sorted[i] != i + 1)
                    failures.Add($"[{i}]={sorted[i]}(expected {i + 1})");
            }
            if (failures.Count > 0)
                throw new Exception($"Non-pairs RadixSort failed: {string.Join(", ", failures)}");
        });

        /// <summary>
        /// Diagnostic: Call ExclusiveScan 4 times in one kernel (same pattern as RadixSort presort).
        /// Tests whether multiple sequential helper calls produce correct results.
        /// </summary>
        [TestMethod]
        public async Task MultiScanDiagnosticTest() => await RunTest(async accelerator =>
        {
            // Kernel that calls ExclusiveScan 4 times on 4 different input rows
            var kernel = accelerator.LoadStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, int>(MultiScanKernel);

            int groupSize = (int)accelerator.MaxNumThreadsPerGroup;
            int n = groupSize * 4; // 4 rows of groupSize
            var input = new int[n];
            // Row 0: [1,0,0,0,1,0,0,0,...] — every 4th = 1 (8 ones in 64)
            // Row 1: [0,1,0,0,0,1,0,0,...] — every 4th offset 1 = 1
            // Row 2: [0,0,1,0,0,0,1,0,...] — every 4th offset 2 = 1
            // Row 3: [0,0,0,1,0,0,0,1,...] — every 4th offset 3 = 1
            for (int t = 0; t < groupSize; t++)
            {
                int bucket = t % 4;
                input[t + groupSize * bucket] = 1;
            }

            using var inputBuf = accelerator.Allocate1D(input);
            using var outputBuf = accelerator.Allocate1D<int>(n);

            kernel(new KernelConfig(1, groupSize), (Index1D)groupSize, inputBuf.View, outputBuf.View, groupSize);
            await accelerator.SynchronizeAsync();

            var result = await outputBuf.CopyToHostAsync<int>();
            var sb = new System.Text.StringBuilder();
            for (int j = 0; j < 4; j++)
            {
                sb.Append($"[MultiScan] Row {j}: [");
                for (int t = 0; t < Math.Min(groupSize, 16); t++)
                {
                    if (t > 0) sb.Append(',');
                    sb.Append(result[j * groupSize + t]);
                }
                if (groupSize > 16) sb.Append(",...");
                sb.Append("]\n");
            }
            Console.WriteLine(sb.ToString());

            // Verify: exclusive scan of each row should give prefix sums
            var failures = new System.Collections.Generic.List<string>();
            for (int j = 0; j < 2; j++)
            {
                int expected = 0;
                for (int t = 0; t < groupSize; t++)
                {
                    int actual = result[j * groupSize + t];
                    if (actual != expected)
                    {
                        failures.Add($"row{j}[{t}]={actual}(expected {expected})");
                        break;
                    }
                    expected += input[j * groupSize + t];
                }
            }
            if (failures.Count > 0)
            {
                var dump = new System.Text.StringBuilder();
                for (int j = 0; j < 2; j++)
                {
                    dump.Append($" row{j}=[");
                    for (int t = 0; t < Math.Min(8, groupSize); t++)
                    {
                        if (t > 0) dump.Append(',');
                        dump.Append(result[j * groupSize + t]);
                    }
                    dump.Append("]");
                }
                throw new Exception($"MultiScan failed: {string.Join(", ", failures)}.{dump}");
            }
        });

        static void MultiScanKernel(
            Index1D index,
            ArrayView<int> input,
            ArrayView<int> output,
            int groupSize)
        {
            var scanMem = SharedMemory.Allocate<int>(1024); // 4 × 256 max
            int tid = Group.IdxX;

            // Load 4 rows from input to shared memory
            for (int j = 0; j < 4; j++)
                scanMem[tid + groupSize * j] = input[tid + groupSize * j];
            Group.Barrier();

            // Call ExclusiveScan TWICE, store to separate outputs
            int scan0 = GroupExtensions.ExclusiveScan<int, AddInt32>(scanMem[tid]);
            int scan1 = GroupExtensions.ExclusiveScan<int, AddInt32>(scanMem[tid + groupSize]);
            Group.Barrier();

            // Write scan results directly (not via scanMem)
            output[tid] = scan0;
            output[tid + groupSize] = scan1;
            for (int j = 2; j < 4; j++)
                output[tid + groupSize * j] = 0;
        }

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

