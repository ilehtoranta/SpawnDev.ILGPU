using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Blazor.UnitTesting;
using SpawnDev.ILGPU.WebGPU;

namespace SpawnDev.ILGPU.Demo.UnitTests
{
    /// <summary>
    /// Abstract base class containing all shared kernel tests.
    /// Each backend (WebGPU, Workers, CPU) inherits and overrides CreateAcceleratorAsync().
    /// </summary>
    public abstract partial class BackendTestBase
    {
        /// <summary>Creates the backend-specific accelerator. Caller is responsible for disposing both.</summary>
        protected abstract Task<(Context context, Accelerator accelerator)> CreateAcceleratorAsync();

        /// <summary>Creates accelerator with 64-bit emulation enabled (for Long/Double tests). Default falls back to standard.</summary>
        protected virtual Task<(Context context, Accelerator accelerator)> CreateEmulatedAcceleratorAsync()
            => CreateAcceleratorAsync();

        /// <summary>Backend display name for error messages.</summary>
        protected abstract string BackendName { get; }

        // Helper to run a test body with proper resource cleanup
        protected async Task RunTest(Func<Accelerator, Task> testBody)
        {
            var (context, accelerator) = await CreateAcceleratorAsync();
            try { await testBody(accelerator); }
            finally { accelerator.Dispose(); context.Dispose(); }
        }

        // Helper to run a test body that requires 64-bit emulation
        protected async Task RunEmulatedTest(Func<Accelerator, Task> testBody)
        {
            var (context, accelerator) = await CreateEmulatedAcceleratorAsync();
            try { await testBody(accelerator); }
            finally { accelerator.Dispose(); context.Dispose(); }
        }

        /// <summary>
        /// Checks that the accelerator supports the specified WebGPU feature.
        /// Throws UnsupportedTestException if the feature is not available.
        /// </summary>
        protected static void RequireFeature(Accelerator accelerator, string featureName, string? reason = null)
        {
            if (accelerator is WebGPUAccelerator webGpuAccelerator)
            {
                if (!webGpuAccelerator.EnabledFeatures.Contains(featureName))
                    throw new UnsupportedTestException(reason ?? $"WebGPU feature '{featureName}' not supported");
            }
        }

        #region Structs

        struct MyPoint
        {
            public float X;
            public float Y;
        }

        struct InnerStruct
        {
            public float Val;
        }

        struct OuterStruct
        {
            public InnerStruct Inner;
            public int ID;
        }

        public struct NestedInnerStruct
        {
            public int A;
            public int B;
        }

        public struct NestedOuterStruct
        {
            public NestedInnerStruct Inner;
            public float Value;
        }

        /// <summary>
        /// Simple struct for testing struct scalar kernel arguments.
        /// </summary>
        struct ScalarStruct
        {
            public float X;
            public float Y;
        }

        #endregion

        #region Kernel Methods

        static void MyKernel(Index1D index, ArrayView<int> dataView, int constant)
        {
            dataView[index] = index + constant;
        }

        static void FloatKernel(Index1D index, ArrayView<float> dataView, float constant)
        {
            dataView[index] = index * 2.0f + constant;
        }

        static void MultiScalarKernel(Index1D index, ArrayView<int> dataView, int c1, int c2)
        {
            dataView[index] = index + c1 + c2;
        }

        static void Kernel2D(Index2D index, ArrayView2D<float, Stride2D.DenseX> dataView)
        {
            dataView[index] = index.X + index.Y * 100.0f;
        }

        static void Kernel3D(Index3D index, ArrayView3D<float, Stride3D.DenseXY> dataView)
        {
            dataView[index] = index.X + index.Y * 100.0f + index.Z * 1000.0f;
        }

        static void VectorAddKernel(Index1D index, ArrayView<float> a, ArrayView<float> b, ArrayView<float> c)
        {
            c[index] = a[index] + b[index];
        }

        static void StructKernel(Index1D index, ArrayView<MyPoint> data)
        {
            var p = data[index];
            p.X += 1.0f;
            p.Y *= 2.0f;
            data[index] = p;
        }

        static void MathKernel(Index1D index, ArrayView<float> input, ArrayView<float> output)
        {
            float val = input[index];
            output[index] = MathF.Sin(val) + MathF.Cos(val) + MathF.Sqrt(MathF.Abs(val));
        }

        static void ControlFlowKernel(Index1D index, ArrayView<int> data)
        {
            int val = data[index];
            int ret = 0;
            if (val % 2 == 0)
            {
                for (int i = 0; i < 5; i++) ret += i;
            }
            else
            {
                ret = -1;
            }
            data[index] = ret;
        }

        static void AtomicKernel(Index1D index, ArrayView<int> data, ArrayView<Index1D> atomicData)
        {
            data[index] = index + 1;
            Atomic.Add(ref atomicData[0], (Index1D)(index + 1));
        }

        static void IntrinsicMathKernel(Index1D index, ArrayView<float> data)
        {
            if (index == 0) data[index] = MathF.Atan2(1.0f, 1.0f);
            else if (index == 1) data[index] = MathF.FusedMultiplyAdd(2.0f, 3.0f, 4.0f);
            else if (index == 2) data[index] = 5.5f % 2.0f;
            else if (index == 5) data[index] = Math.Min(Math.Max(10.0f, 0.0f), 5.0f);
            else if (index == 7) data[index] = IntrinsicMathHelper(0.5f);
        }

        static float IntrinsicMathHelper(float val) { return val; }

        static void ConversionKernel(Index1D index, ArrayView<float> data)
        {
            float val = data[index];
            int intVal = (int)val;
            float floatVal = (float)intVal;
            data[index] = floatVal;
        }

        static void SharedMemoryKernel(Index1D index, ArrayView<int> data)
        {
            var shared = SharedMemory.Allocate<int>(64);
            shared[index] = data[index];
            Group.Barrier();
            int neighbor = (index + 1) % 64;
            data[index] = shared[neighbor];
        }

        static void NestedControlFlowKernel(Index1D index, ArrayView<int> data)
        {
            int sum = 0;
            for (int j = 0; j < 3; j++)
            {
                for (int k = 0; k < 3; k++) { sum += k; }
                if (j == 1) sum += 10;
            }
            data[index] = sum;
        }

        static int MyAdd(int a, int b) { return a + b; }

        static void FunctionCallKernel(Index1D index, ArrayView<int> data)
        {
            data[index] = MyAdd(index, 100);
        }

        static void CSharpSharedMemoryKernel(Index1D index, ArrayView<int> data)
        {
            var sharedMem = SharedMemory.Allocate<int>(64);
            sharedMem[index] = data[index];
            Group.Barrier();
            int reversedIndex = 63 - index;
            int val = sharedMem[reversedIndex];
            data[index] = val;
        }

        static void ComplexStructKernel(Index1D index, ArrayView<OuterStruct> data)
        {
            var item = data[index];
            item.Inner.Val += 1.0f;
            item.ID *= 2;
            data[index] = item;
        }

        static void AtomicCASKernel(Index1D index, ArrayView<int> data)
        {
            Atomic.CompareExchange(ref data[index], index, index + 100);
        }

        static void FMAKernel(Index1D index, ArrayView<float> data)
        {
            float a = (float)(int)index;
            float b = 2.0f;
            float c = 0.5f;
            data[index] = MathF.FusedMultiplyAdd(a, b, c);
        }

        static void DynamicSharedKernel(Index1D index, ArrayView<int> data)
        {
            var shared = SharedMemory.GetDynamic<int>();
            shared[index] = index;
            Group.Barrier();
            int rev = 63 - index;
            data[index] = shared[rev];
        }

        protected static void DynamicSharedF64Kernel(Index1D index, ArrayView<double> data)
        {
            var shared = SharedMemory.GetDynamic<double>();
            shared[index] = data[index];
            Group.Barrier();
            int rev = 63 - index;
            data[index] = shared[rev];
        }

        static void IntMathKernel(Index1D index, ArrayView<int> input, ArrayView<int> output)
        {
            int val = input[index];
            if (index == 0 || index == 1) output[index] = Math.Abs(val);
            else if (index == 2) output[index] = Math.Min(val, 15);
            else if (index == 3) output[index] = Math.Max(val, 15);
            else if (index == 4) output[index] = Math.Min(Math.Max(val, 1), 5);
            else if (index == 5) output[index] = Math.Min(Math.Max(val, -200), -50);
            else output[index] = val;
        }

        static void MatrixMulKernel(Index2D index, ArrayView<float> a, ArrayView<float> b, ArrayView<float> c, int size)
        {
            int row = index.Y; int col = index.X;
            if (row >= size || col >= size) return;
            float sum = 0.0f;
            for (int k = 0; k < size; k++) sum += a[row * size + k] * b[k * size + col];
            c[row * size + col] = sum;
        }

        static void SpecializedIntrinsicsKernel(Index1D index, ArrayView<float> input, ArrayView<float> output)
        {
            float val = input[index];
            if (index == 0) output[index] = global::ILGPU.Algorithms.XMath.Rsqrt(val);
            else if (index == 1 || index == 2) output[index] = MathF.Floor(val) + MathF.Ceiling(val);
            else if (index == 4) output[index] = global::ILGPU.Algorithms.XMath.Rcp(val);
            else output[index] = 0.0f;
        }

        static void BitManipulationKernel(Index1D index, ArrayView<int> input, ArrayView<int> output)
        {
            int val = input[index];
            if (index == 0) output[index] = System.Numerics.BitOperations.PopCount((uint)val);
            else if (index == 1) output[index] = System.Numerics.BitOperations.TrailingZeroCount(val);
            else if (index == 2) output[index] = System.Numerics.BitOperations.LeadingZeroCount((uint)val);
            else if (index == 3) output[index] = System.Numerics.BitOperations.PopCount((uint)val);
        }

        static void HistogramKernel(Index1D index, ArrayView<int> data, ArrayView<int> bins)
        {
            int binIdx = data[index];
            Atomic.Add(ref bins[binIdx], 1);
        }

        static void NestedLoopBreakKernel(Index1D index, ArrayView<int> output)
        {
            int acc = 0;
            for (int j = 0; j < 10; j++) { if (j == 5) break; acc++; }
            for (int k = 0; k < 5; k++) { if (k == 2) continue; acc++; }
            output[index] = acc;
        }

        static void HyperbolicKernel(Index1D index, ArrayView<float> input, ArrayView<float> output)
        {
            float val = input[index];
            if (index == 0) output[index] = MathF.Sinh(val);
            else if (index == 1) output[index] = MathF.Cosh(val);
            else if (index == 2) output[index] = MathF.Tanh(val);
        }

        static void SharedMemoryBarrierKernel(Index1D index, ArrayView<int> output)
        {
            int tid = Group.IdxX;
            var shared = SharedMemory.Allocate<int>(64);
            if (tid < 32) { shared[tid] = tid * 2; }
            Group.Barrier();
            if (tid >= 32) { int val = shared[tid - 32]; int gid = Grid.GlobalIndex.X; output[gid] = val; }
        }

        static void SelectKernel(Index1D index, ArrayView<int> input, ArrayView<int> output)
        {
            int val = input[index];
            output[index] = (val > 0) ? 1 : -1;
        }

        static void LinearBarrierKernel(Index1D index, ArrayView<int> output)
        {
            int tid = Group.IdxX;
            int gid = Grid.GlobalIndex.X;
            var shared = SharedMemory.Allocate<int>(64);
            shared[tid] = gid;
            Group.Barrier();
            int neighbor = (tid + 1) % 64;
            int val = shared[neighbor];
            output[gid] = val;
        }

        static void AdvancedMathKernel(Index1D index, ArrayView<float> input, ArrayView<float> output)
        {
            float val = input[index];
            output[index] = MathF.Tan(val) + MathF.Exp(val) + MathF.Log(MathF.Abs(val) + 1.0f) + MathF.Pow(val, 2.0f) + MathF.Min(val, 2.0f) + MathF.Max(val, 3.0f);
        }

        static void BitwiseKernel(Index1D index, ArrayView<int> data)
        {
            int val = data[index];
            data[index] = (val << 1) + (val >> 1) + (val & 1) + (val | 1) + (val ^ 1) + (~val);
        }

        static void InverseTrigKernel(Index1D index, ArrayView<float> input, ArrayView<float> output)
        {
            float val = input[index];
            if (index == 0) output[index] = MathF.Asin(val);
            else if (index == 1) output[index] = MathF.Acos(val);
            else if (index == 2) output[index] = MathF.Atan(val);
        }

        static void LargeBufferKernel(Index1D index, ArrayView<int> data) { data[index] = data[index] * 2; }
        static void SequentialKernel1(Index1D index, ArrayView<int> data) { data[index] = data[index] * 2; }
        static void SequentialKernel2(Index1D index, ArrayView<int> data) { data[index] = data[index] + 10; }

        static void UnsignedIntKernel(Index1D index, ArrayView<uint> data)
        {
            uint val = data[index];
            if (index == 0) data[index] = val / 3;
            else if (index == 1) data[index] = val % 4;
            else if (index == 2) data[index] = val + 1;
            else if (index == 3) data[index] = val + 100;
        }

        static void AtomicMinMaxKernel(Index1D index, ArrayView<int> minData, ArrayView<int> maxData)
        {
            Atomic.Min(ref minData[0], (int)index);
            Atomic.Max(ref maxData[0], (int)index);
        }

        static void BufferReuseKernel(Index1D index, ArrayView<int> data) { data[index] = data[index] + 1; }

        static void GridGroupDimensionKernel(Index1D index, ArrayView<int> output)
        {
            int globalId = Grid.GlobalIndex.X;
            int localId = Group.IdxX;
            int groupId = Grid.IdxX;
            int groupDim = Group.DimX;
            int baseIdx = globalId * 4;
            output[baseIdx] = globalId;
            output[baseIdx + 1] = localId;
            output[baseIdx + 2] = groupId;
            output[baseIdx + 3] = groupDim;
        }

        // Double/Long kernels
        static void DoublePrecisionKernel(Index1D index, ArrayView<double> input, ArrayView<double> output) { output[index] = input[index] * 2.0 + 1.0; }
        static void LongIntegerKernel(Index1D index, ArrayView<long> data) { data[index] = data[index] * 2 + 1; }
        static void LongArithmeticKernel(Index1D index, ArrayView<long> a, ArrayView<long> b, ArrayView<long> output) { output[index] = (a[index] + b[index]) - (a[index] * 2); }
        static void LongBitwiseKernel(Index1D index, ArrayView<long> input, ArrayView<long> output) { long val = input[index]; long mask = 0x0F0F0F0F0F0F0F0FL; output[index] = (val & mask) | ((val >> 4) & mask); }
        static void LongComparisonKernel(Index1D index, ArrayView<long> a, ArrayView<long> b, ArrayView<long> output) { output[index] = a[index] > b[index] ? a[index] : b[index]; }
        static void LongEdgeCasesKernel(Index1D index, ArrayView<long> data) { data[index] = data[index] + 1L - 1L; }
        static void DoubleMathKernel(Index1D index, ArrayView<double> input, ArrayView<double> output) { output[index] = Math.Sqrt(input[index]) * 2.0; }
        static void DoubleEdgeCasesKernel(Index1D index, ArrayView<double> data) { data[index] = data[index] * 1.0; }
        static void LongMultiBufferKernel(Index1D index, ArrayView<long> input, ArrayView<long> output) { output[index] = input[index] * 3 + 7; }
        static void DoubleMultiBufferKernel(Index1D index, ArrayView<double> input, ArrayView<double> output) { output[index] = input[index] * 3.0 + 0.5; }
        static void LongNegationKernel(Index1D index, ArrayView<long> data) { data[index] = -data[index]; }
        static void LongShiftKernel(Index1D index, ArrayView<long> input, ArrayView<long> output) { output[index] = (input[index] << 8) >> 4; }
        static void LongSignedCompareKernel(Index1D index, ArrayView<long> a, ArrayView<long> b, ArrayView<long> output) { output[index] = (a[index] < b[index]) ? 1L : 0L; }
        static void LongChainedOpsKernel(Index1D index, ArrayView<long> input, ArrayView<long> output) { output[index] = ((input[index] + 10L) * 3L - 5L); }
        static void LongLargeDatasetKernel(Index1D index, ArrayView<long> data) { data[index] = data[index] * 2L + 1L; }
        static void LongNegativeValuesKernel(Index1D index, ArrayView<long> data) { data[index] = data[index] * -1L + data[index]; }
        static void DoubleNegationKernel(Index1D index, ArrayView<double> data) { data[index] = -data[index]; }
        static void DoubleDivisionKernel(Index1D index, ArrayView<double> numerator, ArrayView<double> divisor, ArrayView<double> output) { output[index] = numerator[index] / divisor[index]; }
        static void DoubleChainedOpsKernel(Index1D index, ArrayView<double> input, ArrayView<double> output) { output[index] = (input[index] * 2.5 + 1.0) / 3.0; }
        static void DoubleLargeDatasetKernel(Index1D index, ArrayView<double> data) { data[index] = data[index] * 2.0 + 1.0; }
        static void DoubleMinMaxKernel(Index1D index, ArrayView<double> a, ArrayView<double> b, ArrayView<double> output) { double va = a[index], vb = b[index]; output[index] = (va > vb ? va : vb) - (va < vb ? va : vb); }
        protected static void DoublePrecisionVerifyKernel(Index1D index, ArrayView<double> data) { data[index] = data[index] * 10.0 / 10.0; }

        // More kernels
        static void MixedTypeKernel(Index1D index, ArrayView<int> intData, ArrayView<float> floatData, ArrayView<float> result) { result[index] = (float)intData[index] + floatData[index]; }
        static void EmptyBufferKernel(Index1D index, ArrayView<int> data) { data[index] = data[index] * 2; }
        static void LargeDispatchKernel(Index1D index, ArrayView<int> data) { data[index] = data[index] + 1; }
        static void ComparisonKernel(Index1D index, ArrayView<int> a, ArrayView<int> b, ArrayView<int> output)
        {
            int va = a[index], vb = b[index];
            if (index == 0) output[index] = (va > vb) ? 1 : 0;
            else if (index == 1) output[index] = (va < vb) ? 1 : 0;
            else if (index == 2) output[index] = (va >= vb) ? 1 : 0;
            else if (index == 3) output[index] = (va <= vb) ? 1 : 0;
            else if (index == 4) output[index] = (va == vb) ? 1 : 0;
            else if (index == 5) output[index] = (va != vb) ? 1 : 0;
        }

        static void ShortCircuitKernel(Index1D index, ArrayView<int> a, ArrayView<int> b, ArrayView<int> output)
        {
            bool valA = a[index] != 0, valB = b[index] != 0;
            bool result;
            if (index < 2) result = valA && valB;
            else result = valA || valB;
            output[index] = result ? 1 : 0;
        }

        static void NaNInfinityKernel(Index1D index, ArrayView<float> input, ArrayView<int> output)
        {
            if (index == 0) { float inf = 1.0f / 0.0f; output[index] = (inf > 0.0f) ? 1 : 0; }
            else if (index == 1) { float negInf = -1.0f / 0.0f; output[index] = (negInf < 0.0f) ? 1 : 0; }
            else if (index == 2) { float nan = 0.0f / 0.0f; output[index] = (nan == nan) ? 1 : 0; }
            else { float nan = 0.0f / 0.0f; output[index] = (nan != nan) ? 1 : 0; }
        }

        static void ReductionKernel(Index1D index, ArrayView<int> data, ArrayView<int> sum) { Atomic.Add(ref sum[0], data[index]); }
        static void GatherKernel(Index1D index, ArrayView<int> data, ArrayView<int> indices, ArrayView<int> output) { output[index] = data[indices[index]]; }

        static void DeepNestingKernel(Index1D index, ArrayView<int> data)
        {
            int count = 0;
            if (true) { count++; if (true) { count++; if (true) { count++; if (true) { count++; if (true) { count++; } } } } }
            data[index] = count;
        }

        static void ZeroLoopKernel(Index1D index, ArrayView<int> data) { for (int i = 0; i < 0; i++) data[index] = 0; }

        static void MultiOutputKernel(Index1D index, ArrayView<int> input, ArrayView<int> sum, ArrayView<int> product, ArrayView<int> square)
        {
            int val = input[index]; sum[index] = val + 10; product[index] = val * 2; square[index] = val * val;
        }

        static void MatMulKernel(Index1D index, ArrayView<float> a, ArrayView<float> b, ArrayView<float> c, int size)
        {
            int row = index / size, col = index % size;
            float s = 0.0f;
            for (int k = 0; k < size; k++) s += a[row * size + k] * b[k * size + col];
            c[index] = s;
        }

        static void AddConstantKernel(Index1D index, ArrayView<int> data, int constant) { data[index] += constant; }
        static void DoubleValueKernel(Index1D index, ArrayView<int> data) { data[index] *= 2; }
        static void AddTenKernel(Index1D index, ArrayView<int> data) { data[index] += 10; }

        static void BoundaryKernel(Index1D index, ArrayView<int> data, int length)
        {
            if (index == 0) data[index] = -1;
            else if (index == length - 1) data[index] = 1;
            else data[index] = 0;
        }

        static void FloatSpecialOpsKernel(Index1D index, ArrayView<float> input, ArrayView<float> output)
        {
            output[index] = Math.Min(Math.Max(input[index], 0.0f), 1.0f);
        }

        static void ModuloKernel(Index1D index, ArrayView<int> data) { data[index] = data[index] % 4; }
        static void SimpleSetKernel(Index1D index, ArrayView<int> data) { data[index] = index; }

        static void StencilKernel(Index1D index, ArrayView<float> input, ArrayView<float> output, int length)
        {
            int i = (int)index; int lastIndex = length - 1;
            if (i == 0) output[i] = input[i];
            else if (i == lastIndex) output[i] = input[i];
            else output[i] = (input[i - 1] + input[i] + input[i + 1]) / 3.0f;
        }

        static void ParallelSumKernel(Index1D index, ArrayView<int> data, ArrayView<int> result, int length) { Atomic.Add(ref result[0], data[index]); }

        static void TypeConversionKernel(Index1D index, ArrayView<float> floatIn, ArrayView<int> intOut, ArrayView<float> floatOut)
        {
            int truncated = (int)floatIn[index]; intOut[index] = truncated; floatOut[index] = (float)truncated;
        }

        static void LeftShiftKernel(Index1D index, ArrayView<int> data) { data[index] = data[index] << 2; }
        static void RightShiftKernel(Index1D index, ArrayView<int> data) { data[index] = data[index] >> 1; }

        static void NestedStructKernel(Index1D index, ArrayView<NestedOuterStruct> structs, ArrayView<int> result)
        {
            var s = structs[index]; result[index] = s.Inner.A + s.Inner.B + (int)s.Value;
        }

        static void CopyKernel(Index1D index, ArrayView<int> src, ArrayView<int> dst) { dst[index] = src[index]; }

        static void IntUnaryKernel(Index1D index, ArrayView<int> data) { data[index] = ~(-data[index]); }
        static void FloatUnaryKernel(Index1D index, ArrayView<float> data) { data[index] = -data[index]; }

        static void SmoothstepKernel(Index1D index, ArrayView<float> data) { float t = data[index]; data[index] = t * t * (3.0f - 2.0f * t); }

        static void LerpKernel(Index1D index, ArrayView<float> t, ArrayView<float> output, float a, float b)
        {
            output[index] = a + t[index] * (b - a);
        }

        static void MixedTypesKernel(Index1D index, ArrayView<int> ints, ArrayView<uint> uints, ArrayView<float> floats, ArrayView<float> result)
        {
            result[index] = ints[index] + uints[index] + floats[index];
        }

        static void ComplexExpressionKernel(Index1D index, ArrayView<float> data)
        {
            float x = data[index]; data[index] = ((x * 2 + 3) / 4 - 1) * 5 + x;
        }

        protected static void BroadcastKernel(Index1D index, ArrayView<int> data)
        {
            int val = data[index];
            int broadcasted = Group.Broadcast(val, 0);
            data[index] = broadcasted;
        }

        protected static void SubgroupShuffleKernel(Index1D index, ArrayView<int> data)
        {
            int val = data[index];
            // Each thread reads from lane 0 of its subgroup via warp shuffle
            int shuffled = Warp.Shuffle(val, 0);
            data[index] = shuffled;
        }

        /// <summary>
        /// Kernel that takes a struct as a direct scalar argument (not in an ArrayView).
        /// Writes the sum of the struct's fields to each element of the output buffer.
        /// </summary>
        static void StructScalarArgKernel(Index1D index, ArrayView<float> output, ScalarStruct s)
        {
            output[index] = s.X + s.Y;
        }

        /// <summary>
        /// Kernel that takes a nested struct as a direct scalar argument.
        /// Writes Inner.A + Inner.B + (int)Value to each element of the output buffer.
        /// </summary>
        static void NestedStructScalarArgKernel(Index1D index, ArrayView<int> output, NestedOuterStruct s)
        {
            output[index] = s.Inner.A + s.Inner.B + (int)s.Value;
        }

        #endregion


    }
}
