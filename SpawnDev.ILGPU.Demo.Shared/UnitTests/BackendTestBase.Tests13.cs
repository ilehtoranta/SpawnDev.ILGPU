using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Algorithms.RadixSortOperations;
using ILGPU.Algorithms.ScanReduceOperations;
using ILGPU.Runtime;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.Demo.Shared.UnitTests
{
    public abstract partial class BackendTestBase
    {
        [TestMethod]
        public async Task WasmBinaryDump_RadixSort() => await RunTest(async accelerator =>
        {
            // This test is Wasm-specific (references WasmBackend.LastWasmBinary)
            if (accelerator.AcceleratorType != AcceleratorType.Wasm)
                throw new UnsupportedTestException("Wasm-specific diagnostic test");
            int n = 32;
            var data = new int[n];
            for (int i = 0; i < n; i++) data[i] = n - i;
            using var dataBuf = accelerator.Allocate1D(data);
            var tempSize = accelerator.ComputeRadixSortTempStorageSize<int, AscendingInt32>(n);
            using var tempBuf = accelerator.Allocate1D<int>(tempSize);

            try
            {
                var sort = accelerator.CreateRadixSort<int, Stride1D.Dense, AscendingInt32>();
                sort(accelerator.DefaultStream, dataBuf.View, tempBuf.View.AsContiguous());
                await accelerator.SynchronizeAsync();
            }
            catch (Exception ex)
            {
                var binary = SpawnDev.ILGPU.Wasm.Backend.WasmBackend.LastWasmBinary;
                if (binary != null)
                {
                    // Output just the size so we know the total
                    throw new Exception($"WASM_SIZE:{binary.Length} ERR:{ex.Message}");
                }
                throw;
            }
        });

        // Diagnostic: compile InclusiveScan kernel and dump binary on failure
        static void ScanDumpKernel(Index1D index, ArrayView<uint> input, ArrayView<uint> output)
        {
            int gid = Grid.GlobalIndex.X;
            uint val = input[gid];
            uint scanned = GroupExtensions.InclusiveScan<uint, AddUInt32>(val);
            output[gid] = scanned;
        }

        [TestMethod]
        public async Task WasmBinaryDump_InclusiveScan() => await RunTest(async accelerator =>
        {
            // This test is Wasm-specific (references WasmBackend.LastWasmBinary)
            if (accelerator.AcceleratorType != AcceleratorType.Wasm)
                throw new UnsupportedTestException("Wasm-specific diagnostic test");
            int groupSize = Math.Min(64, accelerator.Device.MaxNumThreadsPerGroup);
            using var inputBuf = accelerator.Allocate1D<uint>(groupSize);
            using var outputBuf = accelerator.Allocate1D<uint>(groupSize);
            var inputData = new uint[groupSize];
            for (int i = 0; i < groupSize; i++) inputData[i] = 1u;
            inputBuf.CopyFromCPU(inputData);

            // Capture the binary after compilation but before dispatch
            var kernel = accelerator.LoadStreamKernel<Index1D, ArrayView<uint>, ArrayView<uint>>(ScanDumpKernel);
            var binary = SpawnDev.ILGPU.Wasm.Backend.WasmBackend.LastWasmBinary;
            // Binary available via ShaderDebugService dump if needed for debugging
            try
            {
                kernel(new KernelConfig(1, groupSize), (Index1D)groupSize, inputBuf.View, outputBuf.View);
                await accelerator.SynchronizeAsync();
            }
            catch (Exception ex)
            {
                throw new Exception($"WASM_SIZE:{binary?.Length ?? 0} ERR:{ex.Message}");
            }
        });
    }
}
