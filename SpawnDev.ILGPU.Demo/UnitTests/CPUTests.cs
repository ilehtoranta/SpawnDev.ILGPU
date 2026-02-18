using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using SpawnDev.Blazor.UnitTesting;

namespace SpawnDev.ILGPU.Demo.UnitTests
{
    /// <summary>
    /// CPU backend tests. Inherits all shared tests from BackendTestBase.
    /// Serves as the reference implementation for correctness.
    /// </summary>
    public class CPUTests : BackendTestBase
    {
        protected override string BackendName => "CPU";

        protected override Task<(Context context, Accelerator accelerator)> CreateAcceleratorAsync()
        {
            // Use a custom CPUDevice with MaxNumThreadsPerGroup = 64
            // (4 threads/warp × 16 warps = 64) to match the group size
            // used by shared memory and barrier tests.
            // In WASM, the WasmProcessor executes all threads sequentially
            // on the calling thread, so no real OS threads are spawned.
            var device = new CPUDevice(
                numThreadsPerWarp: 4,
                numWarpsPerMultiprocessor: 16,
                numMultiprocessors: 1);
            var builder = Context.Create().CPU(device);
            var context = builder.ToContext();
            var accelerator = context.CreateCPUAccelerator(0);
            return Task.FromResult<(Context, Accelerator)>((context, accelerator));
        }

        // ====================================================================
        // WASM Limitation: Barrier/SharedMemory tests are unsupported.
        //
        // In WASM, the WasmProcessor runs all "threads" sequentially on a
        // single thread. Group.Barrier() is a no-op, so thread 0 reads from
        // shared memory before threads 1..N have written to it.
        // Proper barrier support would require lockstep/coroutine execution
        // or Web Workers — see implementation_plan.md for details.
        // ====================================================================

        [TestMethod]
        public new async Task SharedMemoryTest() =>
            throw new UnsupportedTestException("CPU barriers unsupported in WASM (single-threaded)");

        [TestMethod]
        public new async Task CSharpSharedMemoryTest() =>
            throw new UnsupportedTestException("CPU barriers unsupported in WASM (single-threaded)");

        [TestMethod]
        public new async Task DynamicSharedMemoryTest() =>
            throw new UnsupportedTestException("CPU barriers unsupported in WASM (single-threaded)");

        [TestMethod]
        public new async Task DynamicSharedF64Test() =>
            throw new UnsupportedTestException("CPU barriers unsupported in WASM (single-threaded)");

        [TestMethod]
        public new async Task SharedMemoryBarrierTest() =>
            throw new UnsupportedTestException("CPU barriers unsupported in WASM (single-threaded)");

        [TestMethod]
        public new async Task LinearBarrierTest() =>
            throw new UnsupportedTestException("CPU barriers unsupported in WASM (single-threaded)");

        [TestMethod]
        public new async Task AtomicTest() =>
            throw new UnsupportedTestException("CPU atomics crash WASM runtime (assertion failure in CPURuntimeGroupContext)");

        [TestMethod]
        public new async Task AtomicCASTest() =>
            throw new UnsupportedTestException("CPU atomics crash WASM runtime");

        [TestMethod]
        public new async Task AtomicMinMaxTest() =>
            throw new UnsupportedTestException("CPU atomics crash WASM runtime");

        [TestMethod]
        public new async Task HistogramTest() =>
            throw new UnsupportedTestException("CPU atomics crash WASM runtime");

        [TestMethod]
        public new async Task ReductionTest() =>
            throw new UnsupportedTestException("CPU atomics crash WASM runtime");

        [TestMethod]
        public new async Task ParallelSumTest() =>
            throw new UnsupportedTestException("CPU atomics crash WASM runtime");

        [TestMethod]
        public new async Task BroadcastTest() =>
            throw new UnsupportedTestException("Subgroups not supported on CPU in WASM");

        [TestMethod]
        public new async Task SubgroupShuffleTest() =>
            throw new UnsupportedTestException("Subgroups not supported on CPU in WASM");

        [TestMethod]
        public new async Task ReduceMinMaxTest() =>
            throw new UnsupportedTestException("CPU: float atomics and warp shuffle not supported in WASM");

        [TestMethod]
        public new async Task ILGPUReduceTest() =>
            throw new UnsupportedTestException("CPU: GroupExtensions.Reduce requires warp shuffles + barriers, unsupported in WASM");

    }
}
