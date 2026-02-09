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

        // Override tests that are unsupported
        // (broadcast, subgroups — browser limitations)

        [TestMethod]
        public new async Task BroadcastTest() =>
            throw new UnsupportedTestException("Wasm: broadcast not yet implemented");

        [TestMethod]
        public new async Task SubgroupShuffleTest() =>
            throw new UnsupportedTestException("Subgroups not supported in browser environment");

    }
}
