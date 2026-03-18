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
    }
}
