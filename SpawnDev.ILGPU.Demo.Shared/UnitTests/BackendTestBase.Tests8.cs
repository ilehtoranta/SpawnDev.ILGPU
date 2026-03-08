using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Algorithms.RadixSortOperations;
using ILGPU.Algorithms.ScanReduceOperations;
using ILGPU.Runtime;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.Demo.Shared.UnitTests
{
    // Part 8: Shared memory aliasing regression tests.
    //
    // These tests verify that multiple SharedMemory.Allocate<T>(N) calls within a
    // single kernel produce independent allocations, even when they share the same
    // element type and array size.
    //
    // Background: The WGSL code generator resolves shared memory alloca nodes by
    // (elementType, arraySize). When two allocations share the same key, they can
    // be aliased to the same var<workgroup> variable, silently corrupting data.
    // This was the root cause of the radix sort producing 80% duplicate indices:
    // RadixSortKernel1's 1024-int histogram and ExclusiveScan's 1024-int workspace
    // were aliased, causing the scan to overwrite histogram data.
    public abstract partial class BackendTestBase
    {
        // ═══════════════════════════════════════════════════════════
        //  Test 1: Two SharedMemory allocations of the same type and size
        //  in one kernel must be independent.
        // ═══════════════════════════════════════════════════════════

        /// <summary>
        /// Regression test for shared memory aliasing: two SharedMemory.Allocate&lt;int&gt;(128)
        /// calls in the same kernel must map to separate var&lt;workgroup&gt; variables.
        /// If aliased, writes to shared2 would corrupt shared1 values.
        /// </summary>
        [TestMethod]
        public async Task SharedMemoryDualSameSizeTest() => await RunTest(async accelerator =>
        {
            int groupSize = Math.Min(64, accelerator.Device.MaxNumThreadsPerGroup);
            int outputLen = groupSize * 2; // two values per thread
            using var outputBuf = accelerator.Allocate1D<int>(outputLen);

            var kernel = accelerator.LoadStreamKernel<ArrayView<int>, int>(
                SharedMemoryDualSameSizeKernel);
            kernel(new KernelConfig(1, groupSize), outputBuf.View, groupSize);
            await accelerator.SynchronizeAsync();

            var result = await outputBuf.CopyToHostAsync<int>();
            for (int tid = 0; tid < groupSize; tid++)
            {
                int val1 = result[tid * 2];
                int val2 = result[tid * 2 + 1];
                int expected1 = tid + 1;        // shared1 stores tid+1
                int expected2 = tid * 10 + 5;   // shared2 stores tid*10+5

                if (val1 != expected1)
                    throw new Exception(
                        $"SharedMemory aliasing detected! shared1[{tid}] = {val1}, expected {expected1}. " +
                        $"Two SharedMemory.Allocate<int>(128) calls were likely mapped to the same workgroup variable.");
                if (val2 != expected2)
                    throw new Exception(
                        $"SharedMemory aliasing detected! shared2[{tid}] = {val2}, expected {expected2}. " +
                        $"Two SharedMemory.Allocate<int>(128) calls were likely mapped to the same workgroup variable.");
            }
        });

        // ═══════════════════════════════════════════════════════════
        //  Test 2: ExclusiveScan + own SharedMemory in the same kernel.
        //  This mimics the exact RadixSortKernel1 scenario that was broken.
        // ═══════════════════════════════════════════════════════════

        /// <summary>
        /// Regression test for the radix sort shared memory aliasing bug.
        /// A kernel that has its own SharedMemory.Allocate AND calls
        /// GroupExtensions.ExclusiveScan (which internally allocates shared memory).
        /// If the scan's internal workspace aliases with the kernel's buffer,
        /// the scan would overwrite the buffer's contents.
        /// </summary>
        [TestMethod]
        public async Task ExclusiveScanWithSharedMemoryTest() => await RunTest(async accelerator =>
        {
            int groupSize = Math.Min(64, accelerator.Device.MaxNumThreadsPerGroup);
            int outputLen = groupSize * 2; // [scanResult, bufferValue] per thread
            using var outputBuf = accelerator.Allocate1D<int>(outputLen);

            var kernel = accelerator.LoadStreamKernel<ArrayView<int>, int>(
                ExclusiveScanWithSharedMemoryKernel);
            kernel(new KernelConfig(1, groupSize), outputBuf.View, groupSize);
            await accelerator.SynchronizeAsync();

            var result = await outputBuf.CopyToHostAsync<int>();
            for (int tid = 0; tid < groupSize; tid++)
            {
                int scanResult = result[tid * 2];
                int bufferValue = result[tid * 2 + 1];

                // ExclusiveScan of all-1s: [0, 1, 2, 3, ...]
                int expectedScan = tid;
                // Buffer wrote tid+1 before the scan and should be unmodified after
                int expectedBuffer = tid + 1;

                if (scanResult != expectedScan)
                    throw new Exception(
                        $"ExclusiveScan result wrong at thread {tid}: got {scanResult}, expected {expectedScan}");
                if (bufferValue != expectedBuffer)
                    throw new Exception(
                        $"SharedMemory aliasing detected! Buffer[{tid}] = {bufferValue}, expected {expectedBuffer}. " +
                        $"ExclusiveScan's internal shared memory overwrote the kernel's buffer. " +
                        $"This is the radix sort aliasing bug.");
            }
        });

        // ═══════════════════════════════════════════════════════════
        //  Test 3: Large RadixSortPairs verifying index integrity.
        //  Catches shared memory aliasing in the sort itself via
        //  duplicate/missing index detection.
        // ═══════════════════════════════════════════════════════════

        /// <summary>
        /// Stress test: RadixSortPairs with 16K random int keys, verifying that
        /// every original index appears exactly once after sorting (no duplicates,
        /// no missing). This is the definitive test for radix sort correctness —
        /// the shared memory aliasing bug caused ~80% duplicate indices.
        /// </summary>
        [TestMethod]
        public async Task RadixSortPairsIndexIntegrityTest() => await RunTest(async accelerator =>
        {
            int n = 16384;
            var keys = new int[n];
            var values = new int[n];
            var rng = new Random(42);
            for (int i = 0; i < n; i++)
            {
                keys[i] = rng.Next(0, 100000);
                values[i] = i; // identity permutation
            }

            using var keysBuf = accelerator.Allocate1D(keys);
            using var valuesBuf = accelerator.Allocate1D(values);
            var tempSize = accelerator.ComputeRadixSortPairsTempStorageSize<int, int, AscendingInt32>(n);
            using var tempBuf = accelerator.Allocate1D<int>(tempSize);

            var radixSort = accelerator.CreateRadixSortPairs<
                int, Stride1D.Dense, int, Stride1D.Dense, AscendingInt32>();
            radixSort(
                accelerator.DefaultStream,
                keysBuf.View,
                valuesBuf.View,
                tempBuf.View.AsContiguous());
            await accelerator.SynchronizeAsync();

            var sortedKeys = await keysBuf.CopyToHostAsync<int>();
            var sortedValues = await valuesBuf.CopyToHostAsync<int>();

            // Check 1: keys are in ascending order
            int orderViolations = 0;
            for (int i = 1; i < n; i++)
            {
                if (sortedKeys[i] < sortedKeys[i - 1])
                {
                    if (++orderViolations <= 5)
                        Console.WriteLine($"  Sort order violation at [{i}]: {sortedKeys[i - 1]} > {sortedKeys[i]}");
                }
            }
            if (orderViolations > 0)
                throw new Exception($"RadixSort order violated {orderViolations} times out of {n} elements");

            // Check 2: every original index appears exactly once (no duplicates, no missing)
            var seen = new HashSet<int>(n);
            int dupes = 0, outOfRange = 0;
            foreach (var v in sortedValues)
            {
                if (v < 0 || v >= n) outOfRange++;
                else if (!seen.Add(v)) dupes++;
            }
            int missing = n - seen.Count;

            if (dupes > 0 || missing > 0 || outOfRange > 0)
                throw new Exception(
                    $"RadixSort index integrity failure: {dupes} duplicates, {missing} missing, " +
                    $"{outOfRange} out-of-range out of {n} elements. " +
                    $"This typically indicates shared memory aliasing in the sort kernels.");
        });

        /// <summary>
        /// Same as RadixSortPairsIndexIntegrityTest but with DescendingInt32 sort
        /// operation to verify descending sort also preserves index integrity.
        /// </summary>
        [TestMethod]
        public async Task RadixSortPairsDescendingIndexIntegrityTest() => await RunTest(async accelerator =>
        {
            int n = 16384;
            var keys = new int[n];
            var values = new int[n];
            var rng = new Random(99);
            for (int i = 0; i < n; i++)
            {
                keys[i] = rng.Next(0, 100000);
                values[i] = i;
            }

            using var keysBuf = accelerator.Allocate1D(keys);
            using var valuesBuf = accelerator.Allocate1D(values);
            var tempSize = accelerator.ComputeRadixSortPairsTempStorageSize<int, int, DescendingInt32>(n);
            using var tempBuf = accelerator.Allocate1D<int>(tempSize);

            var radixSort = accelerator.CreateRadixSortPairs<
                int, Stride1D.Dense, int, Stride1D.Dense, DescendingInt32>();
            radixSort(
                accelerator.DefaultStream,
                keysBuf.View,
                valuesBuf.View,
                tempBuf.View.AsContiguous());
            await accelerator.SynchronizeAsync();

            var sortedKeys = await keysBuf.CopyToHostAsync<int>();
            var sortedValues = await valuesBuf.CopyToHostAsync<int>();

            // Check 1: keys are in descending order
            int orderViolations = 0;
            for (int i = 1; i < n; i++)
            {
                if (sortedKeys[i] > sortedKeys[i - 1])
                {
                    if (++orderViolations <= 5)
                        Console.WriteLine($"  Descending order violation at [{i}]: {sortedKeys[i - 1]} < {sortedKeys[i]}");
                }
            }
            if (orderViolations > 0)
                throw new Exception($"RadixSort descending order violated {orderViolations} times out of {n} elements");

            // Check 2: every original index appears exactly once
            var seen = new HashSet<int>(n);
            int dupes = 0, outOfRange = 0;
            foreach (var v in sortedValues)
            {
                if (v < 0 || v >= n) outOfRange++;
                else if (!seen.Add(v)) dupes++;
            }
            int missing = n - seen.Count;

            if (dupes > 0 || missing > 0 || outOfRange > 0)
                throw new Exception(
                    $"RadixSort descending index integrity failure: {dupes} duplicates, {missing} missing, " +
                    $"{outOfRange} out-of-range out of {n} elements.");
        });

        #region Tests8 Kernel Methods

        /// <summary>
        /// Kernel with two SharedMemory.Allocate&lt;int&gt;(128) calls — same type, same size.
        /// Writes different values to each, then reads both back to verify independence.
        /// Uses size 128 which is deliberately a common allocation size.
        /// </summary>
        static void SharedMemoryDualSameSizeKernel(
            ArrayView<int> output,
            int groupSize)
        {
            // Two shared memory allocations with IDENTICAL type and size.
            // The WGSL code generator must assign these to separate var<workgroup> variables.
            var shared1 = SharedMemory.Allocate<int>(128);
            var shared2 = SharedMemory.Allocate<int>(128);

            int tid = Group.IdxX;

            // Write distinct values to each allocation
            shared1[tid] = tid + 1;           // 1, 2, 3, ...
            shared2[tid] = tid * 10 + 5;      // 5, 15, 25, ...
            Group.Barrier();

            // Read back — if aliased, shared1 values would be overwritten by shared2
            int val1 = shared1[tid];
            int val2 = shared2[tid];

            // Output interleaved: [shared1[0], shared2[0], shared1[1], shared2[1], ...]
            output[tid * 2] = val1;
            output[tid * 2 + 1] = val2;
        }

        /// <summary>
        /// Kernel that uses its own SharedMemory.Allocate AND calls ExclusiveScan.
        /// This mimics RadixSortKernel1's pattern: a histogram buffer in shared memory
        /// plus ExclusiveScan (which internally allocates its own shared memory workspace).
        /// If the scan's workspace aliases with the kernel's buffer, the buffer's
        /// contents get corrupted by the scan.
        /// </summary>
        static void ExclusiveScanWithSharedMemoryKernel(
            ArrayView<int> output,
            int groupSize)
        {
            // Allocate a buffer in shared memory (like RadixSortKernel1's histogram)
            var buffer = SharedMemory.Allocate<int>(128);

            int tid = Group.IdxX;

            // Write known values to our buffer BEFORE the scan
            buffer[tid] = tid + 1; // 1, 2, 3, ...
            Group.Barrier();

            // Perform an ExclusiveScan — this internally allocates shared memory
            // for its workspace. If that workspace aliases with our buffer above,
            // the scan will overwrite our values.
            int scanInput = 1; // each thread contributes 1
            int scanResult = GroupExtensions.ExclusiveScan<int, AddInt32>(scanInput);

            // After scan completes, re-read our buffer.
            // If aliased, these values will be wrong (overwritten by scan workspace).
            Group.Barrier();
            int bufferValue = buffer[tid];

            // Output interleaved: [scanResult, bufferValue] per thread
            output[tid * 2] = scanResult;
            output[tid * 2 + 1] = bufferValue;
        }

        #endregion
    }
}
