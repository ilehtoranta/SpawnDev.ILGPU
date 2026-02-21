using ILGPU;
using ILGPU.Runtime;
using SpawnDev.UnitTesting;
using SpawnDev.ILGPU.Demo.Shared.UnitTests;
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

        // Override tests that are unsupported
        // (subgroups — browser limitations)

        [TestMethod]
        public new async Task SubgroupShuffleTest() =>
            throw new UnsupportedTestException("Subgroups not supported in browser environment");

        [TestMethod]
        public new async Task ILGPUReduceTest() =>
            throw new UnsupportedTestException("Wasm: GroupExtensions.Reduce requires warp shuffles, unsupported in browser environment");

        [TestMethod]
        public new async Task ILGPUReduceFloatTest() =>
            throw new UnsupportedTestException("Wasm: GroupExtensions.Reduce unsupported in browser environment");

        [TestMethod]
        public new async Task ILGPUReduceDoubleTest() =>
            throw new UnsupportedTestException("Wasm: GroupExtensions.Reduce unsupported in browser environment");

        [TestMethod]
        public new async Task ILGPUReduceLongTest() =>
            throw new UnsupportedTestException("Wasm: GroupExtensions.Reduce unsupported in browser environment");

        [TestMethod]
        public new async Task ILGPUReduceUIntTest() =>
            throw new UnsupportedTestException("Wasm: GroupExtensions.Reduce unsupported in browser environment");

        [TestMethod]
        public new async Task ILGPUReduceULongTest() =>
            throw new UnsupportedTestException("Wasm: GroupExtensions.Reduce unsupported in browser environment");

        // --- Part 4: Wasm limitations ---
        [TestMethod]
        public new async Task AtomicAndOrXorTest() =>
            throw new UnsupportedTestException("Wasm: atomic RMW alignment unsupported for And/Or/Xor");

        [TestMethod]
        public new async Task ExplicitGroupKernelTest() =>
            throw new UnsupportedTestException("Wasm: Grid.GlobalIndex not supported (no workgroup semantics)");

        // --- Shared memory + barrier tests ---
        // PrefixSumTest, MapReduceTest, DotProductTest: Now working after fixing the
        // KernelConfig/IndexType parameter offset bug and improving barrier robustness
        // (per-thread local sense tracking + atomic.fence).

    }
}
