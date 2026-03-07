using ILGPU;
using ILGPU.Runtime;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.Demo.Shared.UnitTests
{
    // Part 7: Large-dispatch tests — verifies correct Index1D indexing when element count
    // exceeds WebGPU's maxComputeWorkgroupsPerDimension limit (65535).
    //
    // Fix: WebGPUAccelerator now uses a 2D workgroup dispatch when totalWG > 65535,
    // and WGSLKernelFunctionGenerator computes the linear index as:
    //   (group_id.x + group_id.y * num_workgroups.x) * workgroup_size.x + local_index
    // which is identical to the 1D formula when workY=1 (group_id.y=0).
    public abstract partial class BackendTestBase
    {
        // ─── Kernels ───────────────────────────────────────────────────────────────

        /// <summary>
        /// Writes the global linear thread index to each output slot.
        /// Used to verify the 2D-dispatch linear index mapping is correct.
        /// OOB guard is required: 2D dispatch may allocate more threads than elements.
        /// </summary>
        static void WriteGlobalIndexKernel(Index1D index, ArrayView<int> output, int count)
        {
            if (index >= count) return;
            output[index] = index;
        }

        /// <summary>
        /// Counts elements whose index is even via Atomic.Add.
        /// Mirrors the DepthToGaussianKernel atomic-compaction pattern that originally
        /// failed with "Dispatch workgroup count X (229954) exceeds 65535" on 5K images.
        /// </summary>
        static void CountEvenIndicesKernel(Index1D index, ArrayView<int> counter, int count)
        {
            if (index >= count) return;
            if ((int)index % 2 == 0) Atomic.Add(ref counter[0], 1);
        }

        // ─── Tests ────────────────────────────────────────────────────────────────

        /// <summary>
        /// Dispatches a kernel with 5,000,000 elements — more than WebGPU's
        /// maxComputeWorkgroupsPerDimension limit of 65535 × 64 = 4,194,240.
        ///
        /// The WebGPU backend responds by using a 2D dispatch:
        ///   wgX = 65535, wgY = ceil(78125 / 65535) = 2
        ///
        /// The test verifies that the linear index formula is correct at three boundaries:
        ///   - index 0 (first element)
        ///   - index 4,194,239 (last element of workgroup-row 0)
        ///   - index 4,194,240 (first element of workgroup-row 1, the critical boundary)
        ///   - index 4,999,999 (last element)
        ///
        /// If the WGSL index formula only uses group_id.x (the bug), every thread
        /// in workgroup-row 1 maps to the same indices as row 0, producing wrong values
        /// and corrupting indices above 4,194,239.
        /// </summary>
        [TestMethod]
        public async Task LargeDispatch_Index1D_CorrectLinearIndexAcrossWorkgroupRows() => await RunTest(async accelerator =>
        {
            // 5,000,000 elements → 78,125 workgroups → exceeds 65535 → 2D dispatch
            const int count = 5_000_000;
            using var buf = accelerator.Allocate1D<int>(count);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<int>, int>(WriteGlobalIndexKernel);
            kernel((Index1D)count, buf.View, count);
            await accelerator.SynchronizeAsync();

            var result = await buf.CopyToHostAsync<int>();

            // Boundary between workgroup-row 0 and row 1 (65535 * 64 = 4,194,240)
            const int rowBoundary = 65535 * 64; // = 4,194,240

            void Check(int idx, string label)
            {
                if (result[idx] != idx)
                    throw new Exception(
                        $"LargeDispatch index wrong at {label} (idx={idx}). " +
                        $"Expected {idx}, got {result[idx]}");
            }

            Check(0,                "first element");
            Check(rowBoundary - 1, "last element of workgroup-row 0");
            Check(rowBoundary,     "first element of workgroup-row 1");  // critical boundary
            Check(count - 1,       "last element");
        });

        /// <summary>
        /// Verifies that Atomic.Add over 5,000,000 elements produces the correct count
        /// when the dispatch exceeds 65535 workgroups.
        ///
        /// Counts even-indexed elements: expected = 5,000,000 / 2 = 2,500,000.
        ///
        /// This is the exact atomic-compaction pattern used by DepthToGaussianKernel
        /// (Generate from Depth) that produced 0 valid splats instead of ~7M when
        /// the dispatch failed on a 5K full-resolution image.
        ///
        /// Verifies both:
        ///   - All elements in the large range are visited (no elements skipped due to
        ///     index aliasing from missing group_id.y in the WGSL formula)
        ///   - No elements are double-counted (no index aliasing in row 1+)
        /// </summary>
        [TestMethod]
        public async Task LargeDispatch_AtomicCompaction_CorrectCountAcrossWorkgroupRows() => await RunTest(async accelerator =>
        {
            const int count = 5_000_000;
            using var counter = accelerator.Allocate1D<int>(1);
            counter.CopyFromCPU(new int[] { 0 });

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<int>, int>(CountEvenIndicesKernel);
            kernel((Index1D)count, counter.View, count);
            await accelerator.SynchronizeAsync();

            var result = await counter.CopyToHostAsync<int>();

            int expected = count / 2; // 2,500,000
            if (result[0] != expected)
                throw new Exception(
                    $"LargeDispatch atomic compaction wrong. " +
                    $"Expected {expected}, got {result[0]}. " +
                    $"If got 0: dispatch failed entirely (missing 2D fix). " +
                    $"If got ~{expected / 2}: only row 0 processed (group_id.y not applied).");
        });
    }
}
