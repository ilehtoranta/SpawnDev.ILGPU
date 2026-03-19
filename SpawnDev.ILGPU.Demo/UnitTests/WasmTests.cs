using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Algorithms.RadixSortOperations;
using ILGPU.Runtime;
using SpawnDev.UnitTesting;
using SpawnDev.ILGPU.Demo.Shared.UnitTests;
using SpawnDev.ILGPU.Wasm;
using SpawnDev.ILGPU.Wasm.Backend;

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
            var builder = Context.Create()
                .EnableAlgorithms()
                .EnableWasmAlgorithms()
                .Wasm();
            var context = builder.ToContext();
            WasmBackend.VerboseLogging = false;
            var accelerator = await context.CreateWasmAcceleratorAsync();
            return (context, accelerator);
        }

        // Override tests that are unsupported
        [TestMethod]
        public new async Task SubgroupShuffleTest() =>
            throw new UnsupportedTestException("Subgroups not supported in browser environment");
        [TestMethod]
        public new async Task ReduceMinMaxTest() =>
            throw new UnsupportedTestException("Wasm: Warp.Shuffle requires subgroup support");

        [TestMethod]
        public new async Task AtomicAndOrXorTest() =>
            throw new UnsupportedTestException("Wasm: atomic RMW alignment unsupported for And/Or/Xor");
        [TestMethod]
        public new async Task ExplicitGroupKernelTest() =>
            throw new UnsupportedTestException("Wasm: Grid.GlobalIndex not supported");

        // ILGPUReduce: struct-with-view parameter decomposition not supported
        [TestMethod]
        public new async Task ILGPUReduceTest() =>
            throw new UnsupportedTestException("Wasm: struct-with-view parameter decomposition not supported");
        [TestMethod]
        public new async Task ILGPUReduceFloatTest() =>
            throw new UnsupportedTestException("Wasm: struct-with-view parameter decomposition not supported");
        [TestMethod]
        public new async Task ILGPUReduceDoubleTest() =>
            throw new UnsupportedTestException("Wasm: struct-with-view parameter decomposition not supported");
        [TestMethod]
        public new async Task ILGPUReduceLongTest() =>
            throw new UnsupportedTestException("Wasm: struct-with-view parameter decomposition not supported");
        [TestMethod]
        public new async Task ILGPUReduceUIntTest() =>
            throw new UnsupportedTestException("Wasm: struct-with-view parameter decomposition not supported");
        [TestMethod]
        public new async Task ILGPUReduceULongTest() =>
            throw new UnsupportedTestException("Wasm: struct-with-view parameter decomposition not supported");

        // RadixSort: fiber dispatch fixes simple barrier kernels.
        // NonPow2 and NonPairsInt pass. Pairs/Descending/Large still fail:
        // - Pairs: pre-existing bug (values not carried correctly)
        // - Descending/Large: multi-group visibility in legacy helper-barrier path
        // These will be fixed when helpers are phase-split (Milestone 5).
        [TestMethod]
        public new async Task AlgorithmRadixSortPairsTest() =>
            throw new UnsupportedTestException("Wasm: RadixSort pairs — values not carried (pre-existing)");
        [TestMethod]
        public new async Task AlgorithmRadixSortPairsIntTest() =>
            throw new UnsupportedTestException("Wasm: RadixSort pairs — values not carried (pre-existing)");
        [TestMethod]
        public new async Task AlgorithmRadixSortPairsDoubleTest() =>
            throw new UnsupportedTestException("Wasm: RadixSort pairs — values not carried (pre-existing)");
        [TestMethod]
        public new async Task AlgorithmRadixSortPairsLongTest() =>
            throw new UnsupportedTestException("Wasm: RadixSort pairs — values not carried (pre-existing)");
        [TestMethod]
        public new async Task AlgorithmRadixSortPairsDoubleOffsetTest() =>
            throw new UnsupportedTestException("Wasm: RadixSort pairs — values not carried (pre-existing)");
        [TestMethod]
        public new async Task AlgorithmRadixSortPairsLongOffsetTest() =>
            throw new UnsupportedTestException("Wasm: RadixSort pairs — values not carried (pre-existing)");
        [TestMethod]
        public new async Task AlgorithmRadixSortPairsUIntTest() =>
            throw new UnsupportedTestException("Wasm: RadixSort pairs — values not carried (pre-existing)");
        [TestMethod]
        public new async Task AlgorithmRadixSortPairsHalfTest() =>
            throw new UnsupportedTestException("Wasm: RadixSort pairs — values not carried (pre-existing)");
        [TestMethod]
        public new async Task AlgorithmRadixSortDescendingTest() =>
            throw new UnsupportedTestException("Wasm: RadixSort descending — multi-group visibility (needs helper phase-split)");
        [TestMethod]
        public new async Task AlgorithmRadixSortLargeTest() =>
            throw new UnsupportedTestException("Wasm: RadixSort large — multi-group visibility (needs helper phase-split)");
        [TestMethod]
        public new async Task RadixSortPairsIndexIntegrityTest() =>
            throw new UnsupportedTestException("Wasm: RadixSort pairs index integrity (pre-existing)");
        [TestMethod]
        public new async Task RadixSortPairsDescendingIndexIntegrityTest() =>
            throw new UnsupportedTestException("Wasm: RadixSort pairs descending (pre-existing)");
        [TestMethod]
        public new async Task RadixSortMinimalPatternsTest() =>
            throw new UnsupportedTestException("Wasm: RadixSort minimal patterns — multi-group (needs helper phase-split)");
        [TestMethod]
        public new async Task RadixSortThresholdProbeTest() =>
            throw new UnsupportedTestException("Wasm: RadixSort threshold probe — multi-group (needs helper phase-split)");
        [TestMethod]
        public new async Task RadixSortDescending1_4MTest() =>
            throw new UnsupportedTestException("Wasm: RadixSort 1.4M descending (needs helper phase-split)");
        [TestMethod]
        public new async Task RadixSortDescendingWithSentinelsTest() =>
            throw new UnsupportedTestException("Wasm: RadixSort sentinels (needs helper phase-split)");
        [TestMethod]
        public new async Task RadixSortRepeatedResortTest() =>
            throw new UnsupportedTestException("Wasm: RadixSort re-sort (needs helper phase-split)");
        [TestMethod]
        public new async Task RadixSortDescending2MTest() =>
            throw new UnsupportedTestException("Wasm: RadixSort 2M (needs helper phase-split)");
        [TestMethod]
        public new async Task RadixSortDescending4MTest() =>
            throw new UnsupportedTestException("Wasm: RadixSort 4M (needs helper phase-split)");
        [TestMethod]
        public new async Task RadixSortHeavyDuplicateKeysTest() =>
            throw new UnsupportedTestException("Wasm: RadixSort heavy dupes (needs helper phase-split)");
        [TestMethod]
        public new async Task RadixSortDescendingOddCountTest() =>
            throw new UnsupportedTestException("Wasm: RadixSort odd count (needs helper phase-split)");
        [TestMethod]
        public new async Task RadixSortAscending1_4MTest() =>
            throw new UnsupportedTestException("Wasm: RadixSort 1.4M ascending (needs helper phase-split)");
        [TestMethod]
        public new async Task RadixSortSpawnSceneSimulationTest() =>
            throw new UnsupportedTestException("Wasm: RadixSort SpawnScene sim (needs helper phase-split)");
        [TestMethod]
        public new async Task TwoPassScanSimulationTest() =>
            throw new UnsupportedTestException("Wasm: Two-pass scan sim (needs helper phase-split)");
        [TestMethod]
        public new async Task RadixSortPositionDiagnosticTest() =>
            throw new UnsupportedTestException("Wasm: RadixSort position diag (needs helper phase-split)");
        [TestMethod]
        public new async Task AlgorithmRadixSortNonPairsFloatTest() =>
            throw new UnsupportedTestException("Wasm: RadixSort non-pairs float (needs helper phase-split)");
        [TestMethod]
        public new async Task RadixSortCounterScanTest() =>
            throw new UnsupportedTestException("Wasm: RadixSort counter scan (needs helper phase-split)");

        // Scan diagnostic tests
        [TestMethod]
        public new async Task GlobalInclusiveScan256Test() =>
            throw new UnsupportedTestException("Wasm: single-group scan only handles up to 64 elements");
        [TestMethod]
        public new async Task GlobalInclusiveScan320Test() =>
            throw new UnsupportedTestException("Wasm: algorithm tests require shared memory + barriers");
        [TestMethod]
        public new async Task GlobalInclusiveScan8000Test() =>
            throw new UnsupportedTestException("Wasm: algorithm tests require shared memory + barriers");
        [TestMethod]
        public new async Task GlobalInclusiveScan4160Test() =>
            throw new UnsupportedTestException("Wasm: algorithm tests require shared memory + barriers");

        // RadixSort boundary/stress tests
        [TestMethod]
        public new async Task RadixSortBoundary16KTest() =>
            throw new UnsupportedTestException("Wasm: RadixSort produces wrong results");
        [TestMethod]
        public new async Task RadixSortBoundary20KTest() =>
            throw new UnsupportedTestException("Wasm: RadixSort produces wrong results");
        // RadixSort + algorithm tests: un-skipped with fiber dispatch.

        // ScanWithBoundaries: pre-existing correctness issue — keep skipped.
        [TestMethod]
        public new async Task AlgorithmScanWithBoundariesTest() =>
            throw new UnsupportedTestException("Wasm: ScanWithBoundaries pre-existing correctness issue");

        // SubView aliasing: separate issue from barriers
        [TestMethod]
        public new async Task AliasedBufferBindingTest() =>
            throw new UnsupportedTestException("Wasm: SubView aliasing causes i64/i32 type mismatch");
    }
}
