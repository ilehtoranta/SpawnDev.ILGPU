using ILGPU;
using ILGPU.Runtime;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.Demo.Shared.UnitTests
{
    // Part 10: SharedMemoryResolver stress tests.
    //
    // These tests exercise shared memory allocation patterns that stress the
    // SharedMemoryResolver's matching logic, including:
    // - Single shared allocation
    // - Multiple shared allocations of different sizes/types
    // - Multiple shared allocations of the same element type
    // - Shared memory used in explicitly-grouped kernels with barriers
    //
    // The resolver must correctly map IR Alloca nodes to WGSL var<workgroup>
    // declarations even when object identity mismatches occur between the
    // backend context and the kernel/helper method's own Allocas.
    public abstract partial class BackendTestBase
    {
        #region Shared Memory Kernels

        /// <summary>
        /// Single shared allocation: classic parallel reduction pattern.
        /// Tests the simplest SharedMemoryResolver case (one alloca → shared_0).
        /// </summary>
        static void SingleSharedMemKernel(
            ArrayView<int> input,
            ArrayView<int> output,
            int length)
        {
            var sharedMem = SharedMemory.Allocate<int>(256);
            var lidx = Group.IdxX;
            var gidx = Grid.IdxX * Group.DimX + lidx;

            sharedMem[lidx] = gidx < length ? input[gidx] : 0;
            Group.Barrier();

            // Simple sum reduction in shared memory
            for (int stride = Group.DimX / 2; stride > 0; stride >>= 1)
            {
                if (lidx < stride)
                    sharedMem[lidx] += sharedMem[lidx + stride];
                Group.Barrier();
            }

            if (lidx == 0)
                output[Grid.IdxX] = sharedMem[0];
        }

        /// <summary>
        /// Two shared allocations of different sizes and element types.
        /// Tests SharedMemoryResolver's strict matching (element type + size).
        /// shared_0 = array&lt;i32, 256&gt;, shared_1 = array&lt;f32, 128&gt;.
        /// </summary>
        static void DualSharedMemKernel(
            ArrayView<int> intInput,
            ArrayView<float> floatInput,
            ArrayView<int> intOutput,
            ArrayView<float> floatOutput,
            int length)
        {
            var sharedInt = SharedMemory.Allocate<int>(256);
            var sharedFloat = SharedMemory.Allocate<float>(128);
            var lidx = Group.IdxX;
            var gidx = Grid.IdxX * Group.DimX + lidx;

            // Load into shared memory
            sharedInt[lidx] = gidx < length ? intInput[gidx] : 0;
            if (lidx < 128)
                sharedFloat[lidx] = gidx < length ? floatInput[gidx] : 0f;
            Group.Barrier();

            // Write back with reversed thread mapping to verify correct shared var assignment
            if (lidx == 0)
            {
                intOutput[Grid.IdxX] = sharedInt[0] + sharedInt[Group.DimX - 1];
                floatOutput[Grid.IdxX] = sharedFloat[0] + sharedFloat[127];
            }
        }

        /// <summary>
        /// Two shared allocations of the SAME element type but different sizes.
        /// This is the hardest case for the resolver — it must match by size
        /// to avoid swapping the two allocations.
        /// shared_0 = array&lt;i32, 64&gt;, shared_1 = array&lt;i32, 256&gt;.
        /// </summary>
        static void SameTypeDualSharedMemKernel(
            ArrayView<int> input,
            ArrayView<int> output,
            int length)
        {
            var sharedSmall = SharedMemory.Allocate<int>(64);   // shared_0
            var sharedLarge = SharedMemory.Allocate<int>(256);  // shared_1
            var lidx = Group.IdxX;
            var gidx = Grid.IdxX * Group.DimX + lidx;

            // Write to both: small gets thread index, large gets input value
            if (lidx < 64)
                sharedSmall[lidx] = lidx;
            sharedLarge[lidx] = gidx < length ? input[gidx] : 0;
            Group.Barrier();

            // Output: sum from large + value from small (verifies correct mapping)
            // Thread 0 outputs: sharedLarge[0] + sharedSmall[0] (= input[gidx] + 0)
            // Thread 1 outputs: sharedLarge[1] + sharedSmall[1] (= input[gidx+1] + 1)
            if (lidx < 64)
                output[gidx] = sharedLarge[lidx] + sharedSmall[lidx];
        }

        /// <summary>
        /// Shared memory with workgroup barrier used in a tile-scan pattern.
        /// This exercises the uniformity analysis alongside shared memory.
        /// </summary>
        static void SharedMemTileScanKernel(
            ArrayView<int> input,
            ArrayView<int> output,
            int length)
        {
            var sharedMem = SharedMemory.Allocate<int>(256);
            var lidx = Group.IdxX;
            var gidx = Grid.IdxX * Group.DimX + lidx;

            // Load
            sharedMem[lidx] = gidx < length ? input[gidx] : 0;
            Group.Barrier();

            // Hillis-Steele inclusive scan in shared memory
            for (int offset = 1; offset < Group.DimX; offset <<= 1)
            {
                int temp = lidx >= offset ? sharedMem[lidx - offset] : 0;
                Group.Barrier();
                sharedMem[lidx] += temp;
                Group.Barrier();
            }

            // Write scanned values
            if (gidx < length)
                output[gidx] = sharedMem[lidx];
        }

        #endregion

        #region Shared Memory Tests

        /// <summary>
        /// Tests single shared memory allocation (reduction pattern).
        /// Verifies SharedMemoryResolver handles the simplest case.
        /// </summary>
        [TestMethod]
        public async Task SharedMemSingleAllocTest() => await RunTest(async accelerator =>
        {
            int n = 256;
            var input = new int[n];
            for (int i = 0; i < n; i++) input[i] = 1;

            using var inputBuf = accelerator.Allocate1D(input);
            using var outputBuf = accelerator.Allocate1D<int>(1);

            var kernel = accelerator.LoadStreamKernel<
                ArrayView<int>, ArrayView<int>, int>(SingleSharedMemKernel);
            kernel(new KernelConfig(1, 256), inputBuf.View, outputBuf.View, n);
            await accelerator.SynchronizeAsync();

            var result = await outputBuf.CopyToHostAsync<int>();
            if (result[0] != n)
                throw new Exception($"SharedMemSingleAlloc: expected {n}, got {result[0]}");
        });

        /// <summary>
        /// Tests dual shared memory with different element types (int vs float).
        /// Verifies strict type+size matching in SharedMemoryResolver.
        /// </summary>
        [TestMethod]
        public async Task SharedMemDualDiffTypeTest() => await RunTest(async accelerator =>
        {
            int n = 256;
            var intInput = new int[n];
            var floatInput = new float[n];
            for (int i = 0; i < n; i++)
            {
                intInput[i] = i;
                floatInput[i] = i * 1.5f;
            }

            using var intInputBuf = accelerator.Allocate1D(intInput);
            using var floatInputBuf = accelerator.Allocate1D(floatInput);
            using var intOutputBuf = accelerator.Allocate1D<int>(1);
            using var floatOutputBuf = accelerator.Allocate1D<float>(1);

            var kernel = accelerator.LoadStreamKernel<
                ArrayView<int>, ArrayView<float>, ArrayView<int>, ArrayView<float>, int>(
                DualSharedMemKernel);
            kernel(new KernelConfig(1, 256),
                intInputBuf.View, floatInputBuf.View,
                intOutputBuf.View, floatOutputBuf.View, n);
            await accelerator.SynchronizeAsync();

            var intResult = await intOutputBuf.CopyToHostAsync<int>();
            var floatResult = await floatOutputBuf.CopyToHostAsync<float>();

            // Expected: intOutput[0] = intInput[0] + intInput[255] = 0 + 255 = 255
            int expectedInt = 0 + 255;
            if (intResult[0] != expectedInt)
                throw new Exception($"SharedMemDualDiffType int: expected {expectedInt}, got {intResult[0]}");

            // Expected: floatOutput[0] = floatInput[0] + floatInput[127] = 0 + 190.5 = 190.5
            float expectedFloat = 0f + 127f * 1.5f;
            if (MathF.Abs(floatResult[0] - expectedFloat) > 0.01f)
                throw new Exception($"SharedMemDualDiffType float: expected {expectedFloat}, got {floatResult[0]}");
        });

        /// <summary>
        /// Tests two shared memory allocations of the same element type but different sizes.
        /// This is the critical stress test for SharedMemoryResolver's disambiguation logic.
        /// If the resolver swaps shared_0 and shared_1, the sizes won't match and either:
        /// - OOB access (writing to index 200 in a 64-element array), or
        /// - Wrong values (reading from the wrong shared buffer)
        /// </summary>
        [TestMethod]
        public async Task SharedMemSameTypeDiffSizeTest() => await RunTest(async accelerator =>
        {
            int n = 256;
            var input = new int[n];
            for (int i = 0; i < n; i++) input[i] = 100 + i;

            using var inputBuf = accelerator.Allocate1D(input);
            using var outputBuf = accelerator.Allocate1D<int>(n);

            var kernel = accelerator.LoadStreamKernel<
                ArrayView<int>, ArrayView<int>, int>(SameTypeDualSharedMemKernel);
            kernel(new KernelConfig(1, 256), inputBuf.View, outputBuf.View, n);
            await accelerator.SynchronizeAsync();

            var result = await outputBuf.CopyToHostAsync<int>();
            // For threads 0..63: result[i] = input[i] + i = (100 + i) + i = 100 + 2*i
            int errors = 0;
            for (int i = 0; i < 64; i++)
            {
                int expected = (100 + i) + i;
                if (result[i] != expected)
                {
                    if (errors++ < 5)
                        Console.WriteLine($"  SharedMemSameTypeDiffSize[{i}]: expected {expected}, got {result[i]}");
                }
            }
            if (errors > 0)
                throw new Exception($"SharedMemSameTypeDiffSize: {errors}/64 mismatches");
        });

        /// <summary>
        /// Tests shared memory with Hillis-Steele scan pattern.
        /// Exercises shared memory + multiple barriers + cross-thread communication.
        /// </summary>
        [TestMethod]
        public async Task SharedMemTileScanTest() => await RunTest(async accelerator =>
        {
            int n = 256;
            var input = new int[n];
            for (int i = 0; i < n; i++) input[i] = 1;

            using var inputBuf = accelerator.Allocate1D(input);
            using var outputBuf = accelerator.Allocate1D<int>(n);

            var kernel = accelerator.LoadStreamKernel<
                ArrayView<int>, ArrayView<int>, int>(SharedMemTileScanKernel);
            kernel(new KernelConfig(1, 256), inputBuf.View, outputBuf.View, n);
            await accelerator.SynchronizeAsync();

            var result = await outputBuf.CopyToHostAsync<int>();
            // Inclusive scan of all 1s: result[i] = i + 1
            int errors = 0;
            for (int i = 0; i < n; i++)
            {
                int expected = i + 1;
                if (result[i] != expected)
                {
                    if (errors++ < 5)
                        Console.WriteLine($"  SharedMemTileScan[{i}]: expected {expected}, got {result[i]}");
                }
            }
            if (errors > 0)
                throw new Exception($"SharedMemTileScan: {errors}/{n} mismatches");
        });

        /// <summary>
        /// Tests shared memory with multi-group dispatch.
        /// Each group independently reduces its tile, verifying that shared memory
        /// is correctly isolated per workgroup.
        /// </summary>
        [TestMethod]
        public async Task SharedMemMultiGroupTest() => await RunTest(async accelerator =>
        {
            int groupSize = 256;
            int numGroups = 4;
            int n = groupSize * numGroups;
            var input = new int[n];
            for (int i = 0; i < n; i++) input[i] = 1;

            using var inputBuf = accelerator.Allocate1D(input);
            using var outputBuf = accelerator.Allocate1D<int>(numGroups);

            var kernel = accelerator.LoadStreamKernel<
                ArrayView<int>, ArrayView<int>, int>(SingleSharedMemKernel);
            kernel(new KernelConfig(numGroups, groupSize), inputBuf.View, outputBuf.View, n);
            await accelerator.SynchronizeAsync();

            var result = await outputBuf.CopyToHostAsync<int>();
            // Each group sums 256 ones → 256
            int errors = 0;
            for (int g = 0; g < numGroups; g++)
            {
                if (result[g] != groupSize)
                {
                    if (errors++ < 5)
                        Console.WriteLine($"  SharedMemMultiGroup[{g}]: expected {groupSize}, got {result[g]}");
                }
            }
            if (errors > 0)
                throw new Exception($"SharedMemMultiGroup: {errors}/{numGroups} group reduction mismatches");
        });

        #endregion
    }
}
