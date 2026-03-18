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
    }
}
