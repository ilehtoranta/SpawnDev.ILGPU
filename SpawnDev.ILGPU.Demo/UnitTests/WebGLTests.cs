using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Blazor.UnitTesting;
using SpawnDev.ILGPU.WebGL;
using SpawnDev.ILGPU.WebGL.Backend;

namespace SpawnDev.ILGPU.Demo.UnitTests
{
    /// <summary>
    /// WebGL2 backend tests. Inherits all shared tests from BackendTestBase
    /// and overrides unsupported features (shared memory, barriers, broadcasts,
    /// subgroups — WebGL2 vertex shaders are single-threaded with no workgroups).
    /// </summary>
    public class WebGLTests : BackendTestBase
    {
        protected override string BackendName => "WebGL";

        protected override async Task<(Context context, Accelerator accelerator)> CreateAcceleratorAsync()
        {
            var builder = Context.Create();
            await builder.WebGL();
            var context = builder.ToContext();
            var devices = context.GetWebGLDevices();
            if (devices.Count == 0)
                throw new UnsupportedTestException("No WebGL2 devices found");
            var accelerator = devices[0].CreateAccelerator(context);
            return (context, accelerator);
        }

        protected override async Task<(Context context, Accelerator accelerator)> CreateEmulatedAcceleratorAsync()
        {
            var builder = Context.Create();
            await builder.WebGL();
            var context = builder.ToContext();
            var devices = context.GetWebGLDevices();
            if (devices.Count == 0)
                throw new UnsupportedTestException("No WebGL2 devices found");
            var accelerator = devices[0].CreateAccelerator(context, new WebGLBackendOptions
            {
                EnableF64Emulation = true,
                EnableI64Emulation = true
            });
            return (context, accelerator);
        }

        // ========== Unsupported Feature Overrides ==========
        // WebGL2 vertex shaders have no shared memory, barriers, or workgroup primitives.

        [TestMethod]
        public new async Task SharedMemoryTest() =>
            throw new UnsupportedTestException("WebGL: no shared memory in vertex shaders");

        [TestMethod]
        public new async Task CSharpSharedMemoryTest() =>
            throw new UnsupportedTestException("WebGL: no shared memory in vertex shaders");

        [TestMethod]
        public new async Task DynamicSharedMemoryTest() =>
            throw new UnsupportedTestException("WebGL: no shared memory in vertex shaders");

        [TestMethod]
        public new async Task DynamicSharedF64Test() =>
            throw new UnsupportedTestException("WebGL: no shared memory in vertex shaders");

        [TestMethod]
        public new async Task BarrierTest() =>
            throw new UnsupportedTestException("WebGL: no barrier support in vertex shaders");

        [TestMethod]
        public new async Task BroadcastTest() =>
            throw new UnsupportedTestException("WebGL: no broadcast support");

        [TestMethod]
        public new async Task SubgroupShuffleTest() =>
            throw new UnsupportedTestException("WebGL: no subgroup support");

        [TestMethod]
        public new async Task ReduceMinMaxTest() =>
            throw new UnsupportedTestException("WebGL: no float atomics or warp shuffle in vertex shaders");

        [TestMethod]
        public new async Task ILGPUReduceTest() =>
            throw new UnsupportedTestException("WebGL: GroupExtensions.Reduce requires shared memory + barriers + atomics, unsupported in vertex shaders");

        [TestMethod]
        public new async Task ILGPUReduceFloatTest() =>
            throw new UnsupportedTestException("WebGL: GroupExtensions.Reduce unsupported in vertex shaders");

        [TestMethod]
        public new async Task ILGPUReduceDoubleTest() =>
            throw new UnsupportedTestException("WebGL: GroupExtensions.Reduce unsupported in vertex shaders");

        [TestMethod]
        public new async Task ILGPUReduceLongTest() =>
            throw new UnsupportedTestException("WebGL: GroupExtensions.Reduce unsupported in vertex shaders");

        [TestMethod]
        public new async Task ILGPUReduceUIntTest() =>
            throw new UnsupportedTestException("WebGL: GroupExtensions.Reduce unsupported in vertex shaders");

        [TestMethod]
        public new async Task ILGPUReduceULongTest() =>
            throw new UnsupportedTestException("WebGL: GroupExtensions.Reduce unsupported in vertex shaders");

        // --- Atomics (WebGL vertex shader TF pipeline has no atomic operations) ---
        [TestMethod]
        public new async Task AtomicTest() =>
            throw new UnsupportedTestException("WebGL: no atomic operations in vertex shaders");

        [TestMethod]
        public new async Task AtomicCASTest() =>
            throw new UnsupportedTestException("WebGL: no atomic operations in vertex shaders");

        [TestMethod]
        public new async Task AtomicMinMaxTest() =>
            throw new UnsupportedTestException("WebGL: no atomic operations in vertex shaders");

        [TestMethod]
        public new async Task ReductionTest() =>
            throw new UnsupportedTestException("WebGL: requires atomics (unsupported in vertex shaders)");

        [TestMethod]
        public new async Task ParallelSumTest() =>
            throw new UnsupportedTestException("WebGL: requires atomics (unsupported in vertex shaders)");

        [TestMethod]
        public new async Task HistogramTest() =>
            throw new UnsupportedTestException("WebGL: requires atomics (unsupported in vertex shaders)");

        // --- Barriers / Shared Memory ---
        [TestMethod]
        public new async Task LinearBarrierTest() =>
            throw new UnsupportedTestException("WebGL: no barriers/shared memory in vertex shaders");

        [TestMethod]
        public new async Task SharedMemoryBarrierTest() =>
            throw new UnsupportedTestException("WebGL: no barriers/shared memory in vertex shaders");

        // --- Grid/Group intrinsics (vertex shader is single-threaded, no workgroups) ---
        [TestMethod]
        public new async Task GridGroupDimensionTest() =>
            throw new UnsupportedTestException("WebGL: no group/grid intrinsics in TF pipeline");

        // --- TF pipeline limitation: no store = no output (value stays zero) ---
        [TestMethod]
        public new async Task ZeroIterationLoopTest() =>
            throw new UnsupportedTestException("WebGL: zero-iteration loop produces no TF store, buffer stays zeroed");

        // LargeDispatchTest: re-enabled — 2D texture tiling now supports buffers > MAX_TEXTURE_SIZE

        // ArrayView2D/3D tests: re-enabled — stride uniform passing now implemented in WebGLAccelerator

        // Struct buffer tests: re-enabled — per-field texelFetch load and per-field TF output now implemented

        // NestedLoopBreakTest: re-enabled — structured control flow (while/break/continue) now handles this correctly

        #region WebGL-Specific Tests

        /// <summary>
        /// Test WebGL2 accelerator basic initialization (uses WebGL-specific APIs)
        /// </summary>
        [TestMethod]
        public async Task WebGLAcceleratorBasicTest()
        {
            var devices = await WebGLDevice.GetDevicesAsync();
            if (devices.Length == 0)
                throw new UnsupportedTestException("No WebGL2 devices found");
            var device = devices[0];
            device.PrintInfo(Console.Out);
            Console.WriteLine($"WebGL2 Device Name: {device.Name}");
            Console.WriteLine($"WebGL2 Vendor: {device.Vendor}");
            Console.WriteLine($"WebGL2 Max Texture Size: {device.MaxTextureSize}");

            // Verify device has reasonable capabilities
            if (device.MaxTextureSize < 2048)
                throw new Exception($"Unexpected max texture size: {device.MaxTextureSize}");
        }

        /// <summary>
        /// Test WebGL2 buffer transfer (uses WebGL-specific APIs)
        /// </summary>
        [TestMethod]
        public async Task WebGLBufferTransferTest()
        {
            var (context, accelerator) = await CreateAcceleratorAsync();
            try
            {
                int length = 128;
                var data = Enumerable.Range(0, length).ToArray();
                using var buffer = accelerator.Allocate1D(data);
                var readBack = await buffer.CopyToHostAsync<int>();
                if (readBack.Length != length)
                    throw new Exception($"Readback length mismatch: expected {length}, got {readBack.Length}");
                for (int i = 0; i < length; i++)
                    if (readBack[i] != data[i])
                        throw new Exception($"Data mismatch at index {i}: expected {data[i]}, got {readBack[i]}");
            }
            finally
            {
                accelerator.Dispose();
                context.Dispose();
            }
        }

        #endregion

        #region Ozaki F64 Tests

        /// <summary>
        /// Creates an accelerator with Ozaki f64 emulation (vec4) enabled.
        /// </summary>
        private async Task<(Context context, Accelerator accelerator)> CreateOzakiAcceleratorAsync()
        {
            var builder = Context.Create();
            await builder.WebGL();
            var context = builder.ToContext();
            var devices = context.GetWebGLDevices();
            if (devices.Count == 0)
                throw new UnsupportedTestException("No WebGL2 devices found");
            var accelerator = devices[0].CreateAccelerator(context, new WebGLBackendOptions
            {
                EnableF64Emulation = true,
                UseOzakiF64Emulation = true,
                EnableI64Emulation = true
            });
            return (context, accelerator);
        }

        /// <summary>
        /// Tests basic Ozaki f64 arithmetic (add, mul) on WebGL
        /// </summary>
        [TestMethod]
        public async Task WebGLOzakiDoublePrecisionTest()
        {
            var (context, accelerator) = await CreateOzakiAcceleratorAsync();
            try
            {
                var input = new double[] { 1.0, 2.5, -3.7, 0.0, 100.5, -50.25 };
                int len = input.Length;
                using var bufIn = accelerator.Allocate1D(input);
                using var bufOut = accelerator.Allocate1D<double>(len);
                var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>, ArrayView<double>>(DoublePrecisionKernel);
                kernel((Index1D)len, bufIn.View, bufOut.View);
                await accelerator.SynchronizeAsync();
                var result = await bufOut.CopyToHostAsync<double>();
                for (int i = 0; i < len; i++)
                {
                    double expected = input[i] * 2.0 + 1.0;
                    double tol = Math.Max(Math.Abs(expected * 0.01), 1e-6);
                    if (Math.Abs(result[i] - expected) > tol)
                        throw new Exception($"Ozaki double precision failed at {i}. Expected {expected}, got {result[i]}");
                }
            }
            finally
            {
                accelerator.Dispose();
                context.Dispose();
            }
        }

        /// <summary>
        /// Tests Ozaki f64 division on WebGL
        /// </summary>
        [TestMethod]
        public async Task WebGLOzakiDoubleDivisionTest()
        {
            var (context, accelerator) = await CreateOzakiAcceleratorAsync();
            try
            {
                var num = new double[] { 10.0, -10.0, 100.0, 1.0 };
                var div = new double[] { 3.0, 3.0, 7.0, 3.0 };
                int len = num.Length;
                using var bufN = accelerator.Allocate1D(num);
                using var bufD = accelerator.Allocate1D(div);
                using var bufOut = accelerator.Allocate1D<double>(len);
                var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>(DoubleDivisionKernel);
                kernel((Index1D)len, bufN.View, bufD.View, bufOut.View);
                await accelerator.SynchronizeAsync();
                var result = await bufOut.CopyToHostAsync<double>();
                for (int i = 0; i < len; i++)
                {
                    double expected = num[i] / div[i];
                    if (Math.Abs(result[i] - expected) > 0.01)
                        throw new Exception($"Ozaki double division failed at {i}. Expected {expected}, got {result[i]}");
                }
            }
            finally
            {
                accelerator.Dispose();
                context.Dispose();
            }
        }

        /// <summary>
        /// Tests Ozaki f64 identity (multiply then divide by 10 — precision test)
        /// </summary>
        [TestMethod]
        public async Task WebGLOzakiDoublePrecisionVerifyTest()
        {
            var (context, accelerator) = await CreateOzakiAcceleratorAsync();
            try
            {
                var data = new double[] { 0.1, 0.2, 0.3, 0.1 + 0.2, 1.0 / 3.0, 2.0 / 3.0 };
                int len = data.Length;
                using var buf = accelerator.Allocate1D(data);
                var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>>(DoublePrecisionVerifyKernel);
                kernel((Index1D)len, buf.View);
                await accelerator.SynchronizeAsync();
                var result = await buf.CopyToHostAsync<double>();
                for (int i = 0; i < len; i++)
                {
                    double expected = data[i];
                    double tolerance = Math.Max(Math.Abs(expected * 0.02), 1e-5);
                    if (Math.Abs(result[i] - expected) > tolerance)
                        throw new Exception($"Ozaki double precision verify failed at {i}. Expected {expected}, got {result[i]}");
                }
            }
            finally
            {
                accelerator.Dispose();
                context.Dispose();
            }
        }

        /// <summary>
        /// Tests Ozaki f64 chained operations (mul, add, div)
        /// </summary>
        [TestMethod]
        public async Task WebGLOzakiDoubleChainedOpsTest()
        {
            var (context, accelerator) = await CreateOzakiAcceleratorAsync();
            try
            {
                var input = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 10.0 };
                int len = input.Length;
                using var bufIn = accelerator.Allocate1D(input);
                using var bufOut = accelerator.Allocate1D<double>(len);
                var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>, ArrayView<double>>(DoubleChainedOpsKernel);
                kernel((Index1D)len, bufIn.View, bufOut.View);
                await accelerator.SynchronizeAsync();
                var result = await bufOut.CopyToHostAsync<double>();
                for (int i = 0; i < len; i++)
                {
                    double expected = (input[i] * 2.5 + 1.0) / 3.0;
                    double tol = Math.Max(Math.Abs(expected * 0.01), 1e-6);
                    if (Math.Abs(result[i] - expected) > tol)
                        throw new Exception($"Ozaki double chained ops failed at {i}. Expected {expected}, got {result[i]}");
                }
            }
            finally
            {
                accelerator.Dispose();
                context.Dispose();
            }
        }

        /// <summary>
        /// Tests Ozaki f64 min/max comparisons
        /// </summary>
        [TestMethod]
        public async Task WebGLOzakiDoubleMinMaxTest()
        {
            var (context, accelerator) = await CreateOzakiAcceleratorAsync();
            try
            {
                var a = new double[] { 1.5, -1.5, 0.0, 100.0, -100.0, 1e10 };
                var b = new double[] { 2.5, -0.5, 0.0, -100.0, 100.0, 1e5 };
                int len = a.Length;
                using var bufA = accelerator.Allocate1D(a);
                using var bufB = accelerator.Allocate1D(b);
                using var bufOut = accelerator.Allocate1D<double>(len);
                var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>(DoubleMinMaxKernel);
                kernel((Index1D)len, bufA.View, bufB.View, bufOut.View);
                await accelerator.SynchronizeAsync();
                var result = await bufOut.CopyToHostAsync<double>();
                for (int i = 0; i < len; i++)
                {
                    double max = a[i] > b[i] ? a[i] : b[i];
                    double min = a[i] < b[i] ? a[i] : b[i];
                    double expected = max - min;
                    double tol = Math.Max(Math.Abs(expected * 0.01), 1e-6);
                    if (Math.Abs(result[i] - expected) > tol)
                        throw new Exception($"Ozaki double min/max failed at {i}. Expected {expected}, got {result[i]}");
                }
            }
            finally
            {
                accelerator.Dispose();
                context.Dispose();
            }
        }

        #endregion

        #region Diagnostic Tests

        /// <summary>
        /// Diagnostic test: generates GLSL for a kernel with bounds checking and
        /// dumps it to window.glslDebug WITHOUT crashing the page. 
        /// Checks whether the INT_MIN fix is applied.
        /// </summary>
        [TestMethod]
        public async Task WebGLIntMinGLSLDiagnosticTest()
        {
            var (context, accelerator) = await CreateAcceleratorAsync();
            try
            {
                // This kernel uses ArrayView bounds checking, which generates int.MinValue comparisons
                Action<Index1D, ArrayView<int>>? kernel = null;
                string? glslSource = null;
                try
                {
                    kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(IntMinDiagKernel);
                }
                catch (Exception ex)
                {
                    // Shader compile may fail — that's OK, we just want the GLSL
                    Console.WriteLine($"[DIAG] Kernel load error (expected if fix not applied): {ex.Message}");
                }

                // The accelerator sets window.glslDebug during kernel load
                try
                {
                    glslSource = SpawnDev.BlazorJS.BlazorJSRuntime.JS.Get<string>("glslDebug");
                }
                catch { }

                if (string.IsNullOrEmpty(glslSource))
                {
                    Console.WriteLine("[DIAG] No GLSL source captured. The backend may not have reached CreateKernel.");
                    throw new Exception("GLSL source not captured - backend CreateKernel may not have been called");
                }

                bool hasBadLiteral = glslSource.Contains("int(-2147483648)");
                bool hasFix = glslSource.Contains("-2147483647 - 1");

                Console.WriteLine($"[DIAG] GLSL length: {glslSource.Length}");
                Console.WriteLine($"[DIAG] Has bad INT_MIN literal: {hasBadLiteral}");
                Console.WriteLine($"[DIAG] Has INT_MIN fix: {hasFix}");
                Console.WriteLine($"[DIAG] First 300 chars: {glslSource.Substring(0, Math.Min(300, glslSource.Length))}");

                if (hasBadLiteral)
                    throw new Exception($"GLSL still contains int(-2147483648)! Fix not applied. GLSL length={glslSource.Length}");

                if (!hasFix && glslSource.Contains("2147483647"))
                    Console.WriteLine("[DIAG] Fix applied but using different format");
            }
            finally
            {
                accelerator.Dispose();
                context.Dispose();
            }
        }

        /// <summary>
        /// Simple kernel that accesses ArrayView (generates bounds check with INT_MIN/INT_MAX).
        /// </summary>
        protected static void IntMinDiagKernel(Index1D index, ArrayView<int> data)
        {
            if (index < data.Length)
                data[index] = index;
        }

        #endregion
    }
}
