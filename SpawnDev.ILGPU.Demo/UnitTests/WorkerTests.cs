using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Blazor.UnitTesting;
using SpawnDev.ILGPU.Workers;

namespace SpawnDev.ILGPU.Demo.UnitTests
{
    /// <summary>
    /// Workers backend tests. Inherits all shared tests from BackendTestBase.
    /// </summary>
    public class WorkerTests : BackendTestBase
    {
        protected override string BackendName => "Workers";

        protected override Task<(Context context, Accelerator accelerator)> CreateAcceleratorAsync()
        {
            var builder = Context.Create().Workers();
            var context = builder.ToContext();
            var accelerator = context.CreateWorkersAccelerator();
            return Task.FromResult<(Context, Accelerator)>((context, accelerator));
        }

        // Shared memory & barrier tests require inter-worker synchronization
        // which the current architecture doesn't support (each work item = 1 worker)
        [TestMethod]
        public new async Task SharedMemoryTest() =>
            throw new UnsupportedTestException("Workers: shared memory requires inter-worker sync");

        [TestMethod]
        public new async Task CSharpSharedMemoryTest() =>
            throw new UnsupportedTestException("Workers: shared memory requires inter-worker sync");

        [TestMethod]
        public new async Task DynamicSharedMemoryTest() =>
            throw new UnsupportedTestException("Workers: shared memory requires inter-worker sync");

        [TestMethod]
        public new async Task SharedMemoryBarrierTest() =>
            throw new UnsupportedTestException("Workers: barriers require inter-worker sync");

        [TestMethod]
        public new async Task LinearBarrierTest() =>
            throw new UnsupportedTestException("Workers: barriers require inter-worker sync");

        // Atomic tests require SharedArrayBuffer-backed typed arrays
        // which only work with cross-origin isolation
        [TestMethod]
        public new async Task AtomicTest() =>
            throw new UnsupportedTestException("Workers: atomics require SharedArrayBuffer");

        [TestMethod]
        public new async Task AtomicCASTest() =>
            throw new UnsupportedTestException("Workers: atomics require SharedArrayBuffer");

        [TestMethod]
        public new async Task AtomicMinMaxTest() =>
            throw new UnsupportedTestException("Workers: atomics require SharedArrayBuffer");

        [TestMethod]
        public new async Task HistogramTest() =>
            throw new UnsupportedTestException("Workers: atomics require SharedArrayBuffer");

        [TestMethod]
        public new async Task ReductionTest() =>
            throw new UnsupportedTestException("Workers: atomics require SharedArrayBuffer");

        [TestMethod]
        public new async Task ParallelSumTest() =>
            throw new UnsupportedTestException("Workers: atomics require SharedArrayBuffer");

        [TestMethod]
        public new async Task BroadcastTest() =>
            throw new UnsupportedTestException("Workers: subgroups not supported");

        [TestMethod]
        public new async Task SubgroupShuffleTest() =>
            throw new UnsupportedTestException("Workers: subgroups not supported");
    }
}
