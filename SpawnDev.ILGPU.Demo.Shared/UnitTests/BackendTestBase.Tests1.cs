using ILGPU;
using ILGPU.Runtime;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.Demo.Shared.UnitTests
{
    // Part 1: Core tests (basic, float, 2D, 3D, vector, struct, math, control flow, atomics, intrinsics, shared memory)
    public abstract partial class BackendTestBase
    {
        [TestMethod]
        public async Task KernelTest() => await RunTest(async accelerator =>
        {
            int len = 64;
            using var buf = accelerator.Allocate1D<int>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, int>(MyKernel);
            kernel((Index1D)len, buf.View, 42);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<int>();
            for (int i = 0; i < len; i++)
                if (result[i] != i + 42) throw new Exception($"Kernel failed at {i}. Expected {i + 42}, got {result[i]}");
        });

        [TestMethod]
        public async Task KernelFloatTest() => await RunTest(async accelerator =>
        {
            int len = 64;
            using var buf = accelerator.Allocate1D<float>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, float>(FloatKernel);
            kernel((Index1D)len, buf.View, 0.5f);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<float>();
            for (int i = 0; i < len; i++)
            {
                float expected = i * 2.0f + 0.5f;
                if (MathF.Abs(result[i] - expected) > 0.001f)
                    throw new Exception($"Float kernel failed at {i}. Expected {expected}, got {result[i]}");
            }
        });

        [TestMethod]
        public async Task MultiScalarKernelTest() => await RunTest(async accelerator =>
        {
            int len = 64;
            using var buf = accelerator.Allocate1D<int>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, int, int>(MultiScalarKernel);
            kernel((Index1D)len, buf.View, 10, 20);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<int>();
            for (int i = 0; i < len; i++)
                if (result[i] != i + 30) throw new Exception($"Multi-scalar failed at {i}. Expected {i + 30}, got {result[i]}");
        });

        [TestMethod]
        public async Task Kernel2DTest() => await RunTest(async accelerator =>
        {
            int width = 8, height = 8;
            using var buf = accelerator.Allocate2DDenseX<float>(new Index2D(width, height));
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>>(Kernel2D);
            kernel(buf.IntExtent, buf.View);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<float>();
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                {
                    float expected = x + y * 100.0f;
                    int idx = y * width + x;
                    if (MathF.Abs(result[idx] - expected) > 0.001f)
                        throw new Exception($"2D kernel failed at ({x},{y}). Expected {expected}, got {result[idx]}");
                }
        });

        [TestMethod]
        public async Task Kernel3DTest() => await RunTest(async accelerator =>
        {
            int w = 4, h = 4, d = 4;
            using var buf = accelerator.Allocate3DDenseXY<float>(new Index3D(w, h, d));
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView3D<float, Stride3D.DenseXY>>(Kernel3D);
            kernel(buf.IntExtent, buf.View);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<float>();
            for (int z = 0; z < d; z++)
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                    {
                        float expected = x + y * 100.0f + z * 1000.0f;
                        int idx = z * (w * h) + y * w + x;
                        if (MathF.Abs(result[idx] - expected) > 0.001f)
                            throw new Exception($"3D kernel failed at ({x},{y},{z}). Expected {expected}, got {result[idx]}");
                    }
        });

        [TestMethod]
        public async Task VectorAddTest() => await RunTest(async accelerator =>
        {
            int len = 64;
            var aData = new float[len]; var bData = new float[len];
            for (int i = 0; i < len; i++) { aData[i] = i; bData[i] = i * 0.5f; }
            using var bufA = accelerator.Allocate1D(aData);
            using var bufB = accelerator.Allocate1D(bData);
            using var bufC = accelerator.Allocate1D<float>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(VectorAddKernel);
            kernel((Index1D)len, bufA.View, bufB.View, bufC.View);
            await accelerator.SynchronizeAsync();
            var result = await bufC.CopyToHostAsync<float>();
            for (int i = 0; i < len; i++)
            {
                float expected = i + i * 0.5f;
                if (MathF.Abs(result[i] - expected) > 0.001f)
                    throw new Exception($"Vector add failed at {i}. Expected {expected}, got {result[i]}");
            }
        });

        [TestMethod]
        public async Task StructTest() => await RunTest(async accelerator =>
        {
            int len = 32;
            var points = new MyPoint[len];
            for (int i = 0; i < len; i++) points[i] = new MyPoint { X = i, Y = i * 2.0f };
            using var buf = accelerator.Allocate1D(points);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<MyPoint>>(StructKernel);
            kernel((Index1D)len, buf.View);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<MyPoint>();
            for (int i = 0; i < len; i++)
            {
                if (MathF.Abs(result[i].X - (i + 1.0f)) > 0.001f)
                    throw new Exception($"Struct X failed at {i}. Expected {i + 1.0f}, got {result[i].X}");
                if (MathF.Abs(result[i].Y - (i * 4.0f)) > 0.001f)
                    throw new Exception($"Struct Y failed at {i}. Expected {i * 4.0f}, got {result[i].Y}");
            }
        });

        [TestMethod]
        public async Task MathTest() => await RunTest(async accelerator =>
        {
            int len = 8;
            var input = new float[] { 0.0f, 1.0f, 2.0f, 3.14f, -1.0f, 0.5f, 10.0f, 100.0f };
            using var bufIn = accelerator.Allocate1D(input);
            using var bufOut = accelerator.Allocate1D<float>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(MathKernel);
            kernel((Index1D)len, bufIn.View, bufOut.View);
            await accelerator.SynchronizeAsync();
            var result = await bufOut.CopyToHostAsync<float>();
            for (int i = 0; i < len; i++)
            {
                float expected = MathF.Sin(input[i]) + MathF.Cos(input[i]) + MathF.Sqrt(MathF.Abs(input[i]));
                if (MathF.Abs(result[i] - expected) > 0.01f)
                    throw new Exception($"Math failed at {i}. Expected {expected}, got {result[i]}");
            }
        });

        [TestMethod]
        public async Task ControlFlowTest() => await RunTest(async accelerator =>
        {
            int len = 16;
            var data = new int[len];
            for (int i = 0; i < len; i++) data[i] = i;
            using var buf = accelerator.Allocate1D(data);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(ControlFlowKernel);
            kernel((Index1D)len, buf.View);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<int>();
            for (int i = 0; i < len; i++)
            {
                int expected = (data[i] % 2 == 0) ? 10 : -1;
                if (result[i] != expected) throw new Exception($"Control flow failed at {i}. Expected {expected}, got {result[i]}");
            }
        });

        [TestMethod]
        public async Task AtomicTest() => await RunTest(async accelerator =>
        {
            int len = 64;
            using var buf = accelerator.Allocate1D<int>(len);
            using var atomicBuf = accelerator.Allocate1D<Index1D>(1);
            atomicBuf.MemSetToZero();
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<Index1D>>(AtomicKernel);
            kernel((Index1D)len, buf.View, atomicBuf.View);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<int>();
            var atomicResult = await atomicBuf.CopyToHostAsync<Index1D>();
            int expectedSum = (len * (len + 1)) / 2;
            if ((int)atomicResult[0] != expectedSum)
                throw new Exception($"Atomic sum failed. Expected {expectedSum}, got {atomicResult[0]}");
            for (int i = 0; i < len; i++)
                if (result[i] != i + 1) throw new Exception($"Data failed at {i}");
        });

        [TestMethod]
        public async Task IntrinsicMathTest() => await RunTest(async accelerator =>
        {
            int len = 8;
            using var buf = accelerator.Allocate1D<float>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>>(IntrinsicMathKernel);
            kernel((Index1D)len, buf.View);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<float>();
            if (MathF.Abs(result[0] - MathF.Atan2(1.0f, 1.0f)) > 0.01f)
                throw new Exception($"Atan2 failed. Expected {MathF.Atan2(1.0f, 1.0f)}, got {result[0]}");
            if (MathF.Abs(result[1] - 10.0f) > 0.01f)
                throw new Exception($"FMA failed. Expected 10, got {result[1]}");
            if (MathF.Abs(result[5] - 5.0f) > 0.01f)
                throw new Exception($"Clamp failed. Expected 5, got {result[5]}");
        });

        [TestMethod]
        public async Task ConversionTest() => await RunTest(async accelerator =>
        {
            int len = 8;
            var data = new float[] { 1.9f, 2.1f, -3.7f, 0.0f, 100.5f, -50.5f, 0.1f, 0.9f };
            using var buf = accelerator.Allocate1D(data);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>>(ConversionKernel);
            kernel((Index1D)len, buf.View);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<float>();
            float[] expected = { 1.0f, 2.0f, -3.0f, 0.0f, 100.0f, -50.0f, 0.0f, 0.0f };
            for (int i = 0; i < len; i++)
                if (MathF.Abs(result[i] - expected[i]) > 0.001f)
                    throw new Exception($"Conversion failed at {i}. Expected {expected[i]}, got {result[i]}");
        });

        [TestMethod]
        public async Task SharedMemoryTest() => await RunTest(async accelerator =>
        {
            int len = 64;
            var data = new int[len];
            for (int i = 0; i < len; i++) data[i] = i * 10;
            using var buf = accelerator.Allocate1D(data);
            var kernel = accelerator.LoadStreamKernel<Index1D, ArrayView<int>>(SharedMemoryKernel);
            kernel(new KernelConfig(1, len), (Index1D)len, buf.View);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<int>();
            for (int i = 0; i < len; i++)
            {
                int expected = data[(i + 1) % 64];
                if (result[i] != expected)
                    throw new Exception($"Shared memory failed at {i}. Expected {expected}, got {result[i]}");
            }
        });

        [TestMethod]
        public async Task NestedControlFlowTest() => await RunTest(async accelerator =>
        {
            int len = 8;
            using var buf = accelerator.Allocate1D<int>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(NestedControlFlowKernel);
            kernel((Index1D)len, buf.View);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<int>();
            int expected = 19; // 3 iterations of inner(0+1+2=3)*3=9, plus 10 for j==1 = 19
            for (int i = 0; i < len; i++)
                if (result[i] != expected)
                    throw new Exception($"Nested control flow failed at {i}. Expected {expected}, got {result[i]}");
        });

        [TestMethod]
        public async Task FunctionCallTest() => await RunTest(async accelerator =>
        {
            int len = 16;
            using var buf = accelerator.Allocate1D<int>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(FunctionCallKernel);
            kernel((Index1D)len, buf.View);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<int>();
            for (int i = 0; i < len; i++)
                if (result[i] != i + 100)
                    throw new Exception($"Function call failed at {i}. Expected {i + 100}, got {result[i]}");
        });

        [TestMethod]
        public async Task CSharpSharedMemoryTest() => await RunTest(async accelerator =>
        {
            int len = 64;
            var data = new int[len];
            for (int i = 0; i < len; i++) data[i] = i * 3;
            using var buf = accelerator.Allocate1D(data);
            var kernel = accelerator.LoadStreamKernel<Index1D, ArrayView<int>>(CSharpSharedMemoryKernel);
            kernel(new KernelConfig(1, len), (Index1D)len, buf.View);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<int>();
            for (int i = 0; i < len; i++)
            {
                int expected = data[63 - i];
                if (result[i] != expected)
                    throw new Exception($"C# shared memory failed at {i}. Expected {expected}, got {result[i]}");
            }
        });

        [TestMethod]
        public async Task ComplexStructTest() => await RunTest(async accelerator =>
        {
            int len = 16;
            var data = new OuterStruct[len];
            for (int i = 0; i < len; i++) data[i] = new OuterStruct { Inner = new InnerStruct { Val = i * 0.5f }, ID = i };
            using var buf = accelerator.Allocate1D(data);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<OuterStruct>>(ComplexStructKernel);
            kernel((Index1D)len, buf.View);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<OuterStruct>();
            for (int i = 0; i < len; i++)
            {
                if (MathF.Abs(result[i].Inner.Val - (i * 0.5f + 1.0f)) > 0.01f)
                    throw new Exception($"Complex struct Val failed at {i}");
                if (result[i].ID != i * 2)
                    throw new Exception($"Complex struct ID failed at {i}");
            }
        });

        [TestMethod]
        public async Task AtomicCASTest() => await RunTest(async accelerator =>
        {
            int len = 16;
            var data = new int[len];
            for (int i = 0; i < len; i++) data[i] = i;
            using var buf = accelerator.Allocate1D(data);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(AtomicCASKernel);
            kernel((Index1D)len, buf.View);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<int>();
            for (int i = 0; i < len; i++)
                if (result[i] != i + 100) throw new Exception($"CAS failed at {i}. Expected {i + 100}, got {result[i]}");
        });

        [TestMethod]
        public async Task FMATest() => await RunTest(async accelerator =>
        {
            int len = 8;
            using var buf = accelerator.Allocate1D<float>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>>(FMAKernel);
            kernel((Index1D)len, buf.View);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<float>();
            for (int i = 0; i < len; i++)
            {
                float expected = MathF.FusedMultiplyAdd(i, 2.0f, 0.5f);
                if (MathF.Abs(result[i] - expected) > 0.01f)
                    throw new Exception($"FMA failed at {i}. Expected {expected}, got {result[i]}");
            }
        });

        [TestMethod]
        public async Task StructScalarArgTest() => await RunTest(async accelerator =>
        {
            int len = 32;
            using var buf = accelerator.Allocate1D<float>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ScalarStruct>(StructScalarArgKernel);
            var s = new ScalarStruct { X = 3.0f, Y = 7.0f };
            kernel((Index1D)len, buf.View, s);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<float>();
            for (int i = 0; i < len; i++)
            {
                float expected = 10.0f; // 3.0 + 7.0
                if (MathF.Abs(result[i] - expected) > 0.001f)
                    throw new Exception($"Struct scalar arg failed at {i}. Expected {expected}, got {result[i]}");
            }
        });

        [TestMethod]
        public async Task NestedStructScalarArgTest() => await RunTest(async accelerator =>
        {
            int len = 32;
            using var buf = accelerator.Allocate1D<int>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, NestedOuterStruct>(NestedStructScalarArgKernel);
            var s = new NestedOuterStruct { Inner = new NestedInnerStruct { A = 3, B = 5 }, Value = 2.0f };
            kernel((Index1D)len, buf.View, s);
            await accelerator.SynchronizeAsync();
            var result = await buf.CopyToHostAsync<int>();
            for (int i = 0; i < len; i++)
            {
                int expected = 10; // 3 + 5 + 2
                if (result[i] != expected)
                    throw new Exception($"Nested struct scalar arg failed at {i}. Expected {expected}, got {result[i]}");
            }
        });

        [TestMethod]
        public async Task CopyToHostPartialReadback() => await RunTest(async accelerator =>
        {
            int len = 64;
            using var buf = accelerator.Allocate1D<int>(len);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, int>(MyKernel);
            kernel((Index1D)len, buf.View, 0); // buf[i] = i + 0 = i
            await accelerator.SynchronizeAsync();

            // Full readback
            var full = await buf.CopyToHostAsync<int>();
            if (full.Length != 64) throw new Exception($"Full readback should be 64, got {full.Length}");

            // Partial readback: offset=10, count=5
            var partial = await buf.CopyToHostAsync<int>(10, 5);
            if (partial.Length != 5) throw new Exception($"Partial should be 5, got {partial.Length}");
            for (int i = 0; i < 5; i++)
            {
                if (partial[i] != i + 10)
                    throw new Exception($"Partial[{i}] expected {i + 10}, got {partial[i]}");
            }

            // Partial readback: first 3 elements
            var first3 = await buf.CopyToHostAsync<int>(0, 3);
            if (first3.Length != 3) throw new Exception($"First3 should be 3, got {first3.Length}");
            if (first3[0] != 0 || first3[1] != 1 || first3[2] != 2)
                throw new Exception($"First3 values wrong: {first3[0]},{first3[1]},{first3[2]}");

            // Partial readback: last element
            var last = await buf.CopyToHostAsync<int>(63, 1);
            if (last.Length != 1) throw new Exception($"Last should be 1, got {last.Length}");
            if (last[0] != 63) throw new Exception($"Last[0] expected 63, got {last[0]}");

            // Full range (offset=0, count=len) should equal full readback
            var fullRange = await buf.CopyToHostAsync<int>(0, 64);
            if (fullRange.Length != 64) throw new Exception($"FullRange should be 64, got {fullRange.Length}");
            for (int i = 0; i < 64; i++)
            {
                if (fullRange[i] != full[i])
                    throw new Exception($"FullRange[{i}] mismatch: {fullRange[i]} != {full[i]}");
            }
        });
    }
}
