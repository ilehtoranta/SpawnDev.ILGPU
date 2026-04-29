using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.Demo.Shared.UnitTests
{
    // Part 20: ArrayView<T>.CopyToHostAsync partial readback tests.
    //
    // Verifies that the new ArrayView<T>.CopyToHostAsync() extension performs
    // a real per-backend partial readback - it must NOT pull the full backing
    // buffer to host and slice on the CPU. The byte range read off the device
    // is exactly [view.IndexInBytes, view.IndexInBytes + view.Length * sizeof(T)).
    //
    // Concrete consumer pattern (Tuvok's AV1 YUV plane separation):
    //
    //     var y = await dRecon.View.SubView(0,            yLen).CopyToHostAsync();
    //     var u = await dRecon.View.SubView(yLen,         uvLen).CopyToHostAsync();
    //     var v = await dRecon.View.SubView(yLen + uvLen, uvLen).CopyToHostAsync();
    //
    // These tests exercise the extension across all 6 backends and verify the
    // returned array's contents match the SubView's slice exactly.
    public abstract partial class BackendTestBase
    {
        // ═══════════════════════════════════════════════════════════
        //  Kernel: write index+1 to each element of a buffer
        // ═══════════════════════════════════════════════════════════

        private static void WriteIndexPlusOneByteKernel(
            Index1D idx,
            ArrayView<byte> data)
        {
            if (idx < data.IntLength)
                data[idx] = (byte)((idx.X + 1) & 0xFF);
        }

        /// <summary>
        /// Reads back a SubView of a single buffer using the new
        /// ArrayView&lt;T&gt;.CopyToHostAsync() extension. Verifies that the returned
        /// array length equals the SubView length and that each element matches
        /// the SubView's slice of the parent buffer.
        /// </summary>
        [TestMethod]
        public async Task ArrayViewCopyToHost_SubView_ByteSlice() => await RunTest(async accelerator =>
        {
            // Parent: 4096 bytes of i+1 mod 256.
            int parentLen = 4096;
            using var parent = accelerator.Allocate1D<byte>(parentLen);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<byte>>(WriteIndexPlusOneByteKernel);
            kernel((Index1D)parentLen, parent.View);
            await accelerator.SynchronizeAsync();

            // SubView: bytes [1000..1064)
            int svOffset = 1000;
            int svLen = 64;
            var slice = await parent.View.SubView(svOffset, svLen).CopyToHostAsync();

            if (slice.Length != svLen)
                throw new Exception($"Slice length: got {slice.Length}, expected {svLen}");

            for (int i = 0; i < svLen; i++)
            {
                byte expected = (byte)(((svOffset + i) + 1) & 0xFF);
                if (slice[i] != expected)
                    throw new Exception($"slice[{i}] = {slice[i]}, expected {expected} (parent index {svOffset + i})");
            }
        });

        // ═══════════════════════════════════════════════════════════
        //  Kernel: write index*scale to int buffer
        // ═══════════════════════════════════════════════════════════

        private static void WriteIndexTimesScaleIntKernel(
            Index1D idx,
            ArrayView<int> data,
            int scale)
        {
            if (idx < data.IntLength)
                data[idx] = idx.X * scale;
        }

        /// <summary>
        /// Three non-overlapping SubViews of the same parent buffer, each read back
        /// with ArrayView&lt;T&gt;.CopyToHostAsync(). Mirrors Tuvok's AV1 YUV plane
        /// separation: one buffer, three slices, each plane copied independently.
        /// </summary>
        [TestMethod]
        public async Task ArrayViewCopyToHost_ThreeSlicesSameParent_AV1Pattern() => await RunTest(async accelerator =>
        {
            // Parent: 3072 ints. Treat as Y=0..2047, U=2048..2559, V=2560..3071
            // (Y is 4x larger than U/V like a 4:2:0 chroma layout).
            int yLen = 2048;
            int uvLen = 512;
            int parentLen = yLen + 2 * uvLen;
            using var parent = accelerator.Allocate1D<int>(parentLen);

            int scale = 7;
            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<int>, int>(WriteIndexTimesScaleIntKernel);
            kernel((Index1D)parentLen, parent.View, scale);
            await accelerator.SynchronizeAsync();

            var y = await parent.View.SubView(0,             yLen ).CopyToHostAsync();
            var u = await parent.View.SubView(yLen,          uvLen).CopyToHostAsync();
            var v = await parent.View.SubView(yLen + uvLen,  uvLen).CopyToHostAsync();

            if (y.Length != yLen)  throw new Exception($"Y length: got {y.Length}, expected {yLen}");
            if (u.Length != uvLen) throw new Exception($"U length: got {u.Length}, expected {uvLen}");
            if (v.Length != uvLen) throw new Exception($"V length: got {v.Length}, expected {uvLen}");

            for (int i = 0; i < yLen; i++)
            {
                int expected = i * scale;
                if (y[i] != expected) throw new Exception($"Y[{i}] = {y[i]}, expected {expected}");
            }
            for (int i = 0; i < uvLen; i++)
            {
                int expected = (yLen + i) * scale;
                if (u[i] != expected) throw new Exception($"U[{i}] = {u[i]}, expected {expected}");
            }
            for (int i = 0; i < uvLen; i++)
            {
                int expected = (yLen + uvLen + i) * scale;
                if (v[i] != expected) throw new Exception($"V[{i}] = {v[i]}, expected {expected}");
            }
        });

        /// <summary>
        /// Whole-view readback through the extension - SubView covers the entire
        /// parent buffer. Verifies the extension matches the full-buffer
        /// CopyToHostAsync result element-for-element.
        /// </summary>
        [TestMethod]
        public async Task ArrayViewCopyToHost_FullView_MatchesBufferReadback() => await RunTest(async accelerator =>
        {
            int len = 1024;
            using var buf = accelerator.Allocate1D<int>(len);

            int scale = 13;
            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<int>, int>(WriteIndexTimesScaleIntKernel);
            kernel((Index1D)len, buf.View, scale);
            await accelerator.SynchronizeAsync();

            var viaView = await buf.View.CopyToHostAsync();
            var viaBuffer = await buf.CopyToHostAsync<int>();

            if (viaView.Length != viaBuffer.Length)
                throw new Exception($"Length: view={viaView.Length}, buffer={viaBuffer.Length}");
            for (int i = 0; i < viaView.Length; i++)
            {
                if (viaView[i] != viaBuffer[i])
                    throw new Exception($"viaView[{i}]={viaView[i]} != viaBuffer[{i}]={viaBuffer[i]}");
            }
        });

        /// <summary>
        /// SubView of a SubView - the byte-range arithmetic must compose correctly so
        /// the readback only touches the inner slice's bytes.
        /// </summary>
        [TestMethod]
        public async Task ArrayViewCopyToHost_NestedSubView() => await RunTest(async accelerator =>
        {
            int parentLen = 2048;
            using var parent = accelerator.Allocate1D<int>(parentLen);

            int scale = 5;
            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<int>, int>(WriteIndexTimesScaleIntKernel);
            kernel((Index1D)parentLen, parent.View, scale);
            await accelerator.SynchronizeAsync();

            // Outer slice: parent[500..1500). Inner slice: outer[100..200) -> parent[600..700).
            int outerOffset = 500;
            int outerLen = 1000;
            int innerOffset = 100;
            int innerLen = 100;

            var outer = parent.View.SubView(outerOffset, outerLen);
            var inner = outer.SubView(innerOffset, innerLen);

            var slice = await inner.CopyToHostAsync();
            if (slice.Length != innerLen)
                throw new Exception($"Inner length: got {slice.Length}, expected {innerLen}");

            int expectedParentBase = outerOffset + innerOffset;
            for (int i = 0; i < innerLen; i++)
            {
                int expected = (expectedParentBase + i) * scale;
                if (slice[i] != expected)
                    throw new Exception($"inner[{i}] = {slice[i]}, expected {expected} (parent index {expectedParentBase + i})");
            }
        });
    }
}
