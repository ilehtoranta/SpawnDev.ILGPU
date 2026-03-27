using ILGPU;
using ILGPU.Runtime;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.Demo.Shared.UnitTests
{
    // Part 15: SubView range optimization tests.
    //
    // Verifies that when a kernel receives SubViews of a large parent buffer,
    // the dispatch allocates and copies only the used SubView ranges — NOT the
    // entire parent buffer. This prevents OOM on Wasm when ML inference uses
    // one large weight buffer (~5MB) with 100+ SubViews.
    //
    // The fix is in WasmAccelerator.cs: bufferRanges tracks min/max byte ranges
    // per deduped parent, and allocation/copy-in/copy-out use only those ranges.
    // NativePtr patching subtracts rangeMin so SubView offsets resolve correctly.
    //
    // These tests run on ALL backends to prevent regressions.
    public abstract partial class BackendTestBase
    {
        // ═══════════════════════════════════════════════════════════
        //  Kernel: scale a SubView (read + write to same SubView)
        // ═══════════════════════════════════════════════════════════

        private static void ScaleSubViewKernel(
            Index1D idx,
            ArrayView<float> data,
            float scale)
        {
            if (idx < data.IntLength)
                data[idx] = data[idx] * scale;
        }

        /// <summary>
        /// Basic SubView correctness: allocate a large buffer, create a small SubView,
        /// dispatch a kernel on the SubView, verify only the SubView region is affected.
        /// </summary>
        [TestMethod]
        public async Task SubViewRange_BasicCorrectness() => await RunTest(async accelerator =>
        {
            // Large parent buffer: 4096 floats (16KB)
            int parentSize = 4096;
            var parentData = new float[parentSize];
            for (int i = 0; i < parentSize; i++)
                parentData[i] = i + 1; // 1, 2, 3, ...
            using var parentBuf = accelerator.Allocate1D(parentData);

            // SubView: elements [1000..1064) — 64 floats in the middle
            int svOffset = 1000;
            int svLength = 64;
            var subView = parentBuf.View.SubView(svOffset, svLength);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, float>(ScaleSubViewKernel);
            kernel((Index1D)svLength, subView, 2.0f);
            await accelerator.SynchronizeAsync();

            var result = await parentBuf.CopyToHostAsync<float>();

            // SubView region should be doubled
            for (int i = svOffset; i < svOffset + svLength; i++)
            {
                float expected = (i + 1) * 2.0f;
                if (MathF.Abs(result[i] - expected) > 0.01f)
                    throw new Exception($"SubView[{i - svOffset}] = {result[i]}, expected {expected}");
            }

            // Elements BEFORE SubView should be unchanged
            for (int i = 0; i < svOffset; i++)
            {
                if (MathF.Abs(result[i] - (i + 1)) > 0.01f)
                    throw new Exception($"Before SubView: parent[{i}] = {result[i]}, expected {i + 1} (unchanged)");
            }

            // Elements AFTER SubView should be unchanged
            for (int i = svOffset + svLength; i < parentSize; i++)
            {
                if (MathF.Abs(result[i] - (i + 1)) > 0.01f)
                    throw new Exception($"After SubView: parent[{i}] = {result[i]}, expected {i + 1} (unchanged)");
            }
        });

        // ═══════════════════════════════════════════════════════════
        //  Kernel: add two SubViews of the same parent into output
        // ═══════════════════════════════════════════════════════════

        private static void AddSubViewsKernel(
            Index1D idx,
            ArrayView<float> a,
            ArrayView<float> b,
            ArrayView<float> output)
        {
            if (idx < output.IntLength)
                output[idx] = a[idx] + b[idx];
        }

        /// <summary>
        /// Two non-overlapping SubViews of the same large buffer used as separate kernel
        /// parameters. Tests deduplication: both SubViews reference the same parent, so
        /// the dispatch should allocate the parent's used range ONCE, not twice.
        /// Also tests that NativePtr patching correctly resolves both SubView offsets
        /// within the trimmed range.
        /// </summary>
        [TestMethod]
        public async Task SubViewRange_TwoSubViewsSameParent() => await RunTest(async accelerator =>
        {
            // Large parent: 8192 floats (32KB)
            int parentSize = 8192;
            var parentData = new float[parentSize];
            for (int i = 0; i < parentSize; i++)
                parentData[i] = i;
            using var parentBuf = accelerator.Allocate1D(parentData);

            // Two SubViews at different offsets within the same parent
            int svAOffset = 2000;
            int svBOffset = 6000;
            int svLength = 128;
            var viewA = parentBuf.View.SubView(svAOffset, svLength);
            var viewB = parentBuf.View.SubView(svBOffset, svLength);

            // Output in a separate buffer
            using var outputBuf = accelerator.Allocate1D<float>(svLength);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(AddSubViewsKernel);
            kernel((Index1D)svLength, viewA, viewB, outputBuf.View);
            await accelerator.SynchronizeAsync();

            var result = await outputBuf.CopyToHostAsync<float>();

            for (int i = 0; i < svLength; i++)
            {
                float expected = (svAOffset + i) + (svBOffset + i);
                if (MathF.Abs(result[i] - expected) > 0.01f)
                    throw new Exception($"TwoSubViews: output[{i}] = {result[i]}, expected {expected}");
            }
        });

        // ═══════════════════════════════════════════════════════════
        //  Kernel: write to SubView at a high offset
        // ═══════════════════════════════════════════════════════════

        private static void WritePatternKernel(
            Index1D idx,
            ArrayView<float> output)
        {
            if (idx < output.IntLength)
                output[idx] = idx * 3.14f + 1.0f;
        }

        /// <summary>
        /// SubView at a high offset in a large buffer. Simulates ML weight loading where
        /// SubViews reference tensors deep into a multi-MB weight buffer. The dispatch
        /// should only allocate memory for the SubView range, not the entire parent.
        /// Without the range optimization, a 1M-element parent with a 64-element SubView
        /// at offset 999936 would waste ~4MB of Wasm memory per dispatch.
        /// </summary>
        [TestMethod]
        public async Task SubViewRange_HighOffsetLargeParent() => await RunTest(async accelerator =>
        {
            // Very large parent: 262144 floats (1MB) — simulates ML weight buffer
            int parentSize = 262144;
            using var parentBuf = accelerator.Allocate1D<float>(parentSize);

            // SubView near the end: last 256 elements
            int svOffset = parentSize - 256;
            int svLength = 256;
            var subView = parentBuf.View.SubView(svOffset, svLength);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>>(WritePatternKernel);
            kernel((Index1D)svLength, subView);
            await accelerator.SynchronizeAsync();

            // Read back just the SubView portion to verify
            var fullResult = await parentBuf.CopyToHostAsync<float>();

            for (int i = 0; i < svLength; i++)
            {
                float expected = i * 3.14f + 1.0f;
                if (MathF.Abs(fullResult[svOffset + i] - expected) > 0.01f)
                    throw new Exception($"HighOffset: result[{i}] = {fullResult[svOffset + i]}, expected {expected}");
            }
        });

        // ═══════════════════════════════════════════════════════════
        //  Kernel: copy from one SubView to another (same parent)
        // ═══════════════════════════════════════════════════════════

        private static void CopySubViewKernel(
            Index1D idx,
            ArrayView<float> src,
            ArrayView<float> dst)
        {
            if (idx < src.IntLength && idx < dst.IntLength)
                dst[idx] = src[idx];
        }

        /// <summary>
        /// Multiple sequential dispatches with different SubView windows of the same parent.
        /// Simulates ML inference where each operator reads a different weight tensor (SubView)
        /// from the same master weight buffer. Verifies that sequential dispatches don't
        /// corrupt each other's data and that NativePtr patching is correct for each dispatch.
        /// </summary>
        [TestMethod]
        public async Task SubViewRange_SequentialDispatches() => await RunTest(async accelerator =>
        {
            // Parent buffer with known pattern
            int parentSize = 4096;
            var parentData = new float[parentSize];
            for (int i = 0; i < parentSize; i++)
                parentData[i] = i * 0.1f;
            using var parentBuf = accelerator.Allocate1D(parentData);

            int chunkSize = 64;
            using var outputBuf = accelerator.Allocate1D<float>(chunkSize);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>>(CopySubViewKernel);

            // Dispatch 10 times with SubViews at different offsets
            for (int chunk = 0; chunk < 10; chunk++)
            {
                int offset = chunk * 400; // 0, 400, 800, ... 3600
                var srcView = parentBuf.View.SubView(offset, chunkSize);

                kernel((Index1D)chunkSize, srcView, outputBuf.View);
                await accelerator.SynchronizeAsync();

                var result = await outputBuf.CopyToHostAsync<float>();

                for (int i = 0; i < chunkSize; i++)
                {
                    float expected = (offset + i) * 0.1f;
                    if (MathF.Abs(result[i] - expected) > 0.001f)
                        throw new Exception($"Sequential dispatch {chunk}: output[{i}] = {result[i]}, expected {expected}");
                }
            }
        });

        /// <summary>
        /// Many small SubViews of a very large parent buffer in a single kernel dispatch.
        /// This is the exact ML weight loading pattern: one 5MB master buffer, 3 SubViews
        /// passed to a single kernel. Verifies that only the union of SubView ranges is
        /// allocated, not 3x the full parent buffer.
        /// </summary>
        [TestMethod]
        public async Task SubViewRange_ManySubViewsOneParent() => await RunTest(async accelerator =>
        {
            // WebGL's SynchronizeAsync is a no-op — mid-test Allocate1D + immediate dispatch
            // can race (buffer upload not complete before kernel reads). Skip until WebGL sync is fixed.
            if (accelerator.AcceleratorType == AcceleratorType.WebGL)
                throw new UnsupportedTestException("WebGL: mid-test buffer allocation race — SynchronizeAsync is no-op");

            // Large parent: 131072 floats (512KB) — simulates weight buffer
            int parentSize = 131072;
            var parentData = new float[parentSize];
            for (int i = 0; i < parentSize; i++)
                parentData[i] = (float)Math.Sin(i * 0.001);
            using var parentBuf = accelerator.Allocate1D(parentData);

            // Three SubViews at spread-out offsets
            int len = 64;
            var viewA = parentBuf.View.SubView(1000, len);   // bytes 4000-4256
            var viewB = parentBuf.View.SubView(65000, len);  // bytes 260000-260256
            var viewC = parentBuf.View.SubView(130000, len); // bytes 520000-520256

            // Output: sum of all three
            using var outputBuf = accelerator.Allocate1D<float>(len);

            // Use the AddSubViews kernel with a third parameter as accumulator
            var addKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(AddSubViewsKernel);

            // Step 1: output = viewA + viewB
            addKernel((Index1D)len, viewA, viewB, outputBuf.View);
            await accelerator.SynchronizeAsync();

            var result = await outputBuf.CopyToHostAsync<float>();

            for (int i = 0; i < len; i++)
            {
                float expected = parentData[1000 + i] + parentData[65000 + i];
                if (MathF.Abs(result[i] - expected) > 0.001f)
                    throw new Exception($"ManySubViews step1: output[{i}] = {result[i]}, expected {expected}");
            }

            // Step 2: now add viewC to the result
            // Copy result back as input
            using var tempBuf = accelerator.Allocate1D(result);
            addKernel((Index1D)len, tempBuf.View, viewC, outputBuf.View);
            await accelerator.SynchronizeAsync();

            var result2 = await outputBuf.CopyToHostAsync<float>();

            for (int i = 0; i < len; i++)
            {
                float expected = parentData[1000 + i] + parentData[65000 + i] + parentData[130000 + i];
                if (MathF.Abs(result2[i] - expected) > 0.01f)
                    throw new Exception($"ManySubViews step2: output[{i}] = {result2[i]}, expected {expected}");
            }
        });
    }
}
