using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Blazor.UnitTesting;
using SpawnDev.ILGPU.Wasm;

namespace SpawnDev.ILGPU.Demo.UnitTests
{
    /// <summary>
    /// Wasm backend tests. Inherits all shared tests from BackendTestBase.
    /// </summary>
    public class WasmTests : BackendTestBase
    {
        protected override string BackendName => "Wasm";

        protected override async Task<(Context context, Accelerator accelerator)> CreateAcceleratorAsync()
        {
            var builder = Context.Create().Wasm();
            var context = builder.ToContext();
            var accelerator = await context.CreateWasmAcceleratorAsync();
            return (context, accelerator);
        }

        // Override tests that are unsupported in Phase 1
        // (barriers, atomics, subgroups)

        [TestMethod]
        public new async Task SharedMemoryTest() =>
            throw new UnsupportedTestException("Wasm Phase 1: barriers not yet implemented");

        [TestMethod]
        public new async Task CSharpSharedMemoryTest() =>
            throw new UnsupportedTestException("Wasm Phase 1: barriers not yet implemented");

        [TestMethod]
        public new async Task DynamicSharedMemoryTest() =>
            throw new UnsupportedTestException("Wasm Phase 1: barriers not yet implemented");

        [TestMethod]
        public new async Task SharedMemoryBarrierTest() =>
            throw new UnsupportedTestException("Wasm Phase 1: barriers not yet implemented");

        [TestMethod]
        public new async Task LinearBarrierTest() =>
            throw new UnsupportedTestException("Wasm Phase 1: barriers not yet implemented");

        [TestMethod]
        public new async Task BroadcastTest() =>
            throw new UnsupportedTestException("Wasm Phase 1: broadcast not yet implemented");

        [TestMethod]
        public new async Task SubgroupShuffleTest() =>
            throw new UnsupportedTestException("Subgroups not supported in browser environment");

        [TestMethod]
        public new async Task AtomicAddTest() =>
            throw new UnsupportedTestException("Wasm Phase 1: atomics not yet implemented");

        [TestMethod]
        public new async Task AtomicMinMaxTest() =>
            throw new UnsupportedTestException("Wasm Phase 1: atomics not yet implemented");

        [TestMethod]
        public new async Task AtomicAndOrXorTest() =>
            throw new UnsupportedTestException("Wasm Phase 1: atomics not yet implemented");

        [TestMethod]
        public new async Task AtomicCompareExchangeTest() =>
            throw new UnsupportedTestException("Wasm Phase 1: atomics not yet implemented");

        [TestMethod]
        public new async Task AtomicAddFloatTest() =>
            throw new UnsupportedTestException("Wasm Phase 1: atomics not yet implemented");

        [TestMethod]
        public new async Task AtomicAddLongTest() =>
            throw new UnsupportedTestException("Wasm Phase 1: atomics not yet implemented");

        [TestMethod]
        public new async Task ParallelSumTest() =>
            throw new UnsupportedTestException("Wasm Phase 1: atomics not yet implemented (uses Atomic.Add)");
    }
}
