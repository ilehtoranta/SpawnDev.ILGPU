using ILGPU;
using ILGPU.Runtime;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.Demo.Shared.UnitTests
{
    // Part 21: Coalesce-binding regression tests.
    //
    // When a kernel takes a body struct with many ArrayView fields of the same element
    // type (Tuvok's VorbisPacketDecodeStaticInputs has 38 ArrayView<int> + 2 ArrayView<double>),
    // the WebGPU codegen would emit one storage-buffer binding per field and exceed
    // maxStorageBuffersPerShaderStage (Chrome default = 10). The codegen now coalesces
    // same-element-type body-struct ArrayView fields into a single shared binding;
    // the runtime concatenates each member's GPU buffer data into one coalesced buffer
    // at dispatch time and routes per-field offsets through _scalar_params.
    //
    // These tests verify the coalesce path works end-to-end: kernel reads from each
    // member field at index i, the result must match the CPU reference computed from
    // the original per-field arrays. Independent buffers (NOT SubViews of one parent)
    // mirror Tuvok's actual case.
    public abstract partial class BackendTestBase
    {
        #region Coalesce Body-Struct Bindings

        // 12 ArrayView<int> fields — well above Chrome's maxStorageBuffersPerShaderStage = 10.
        // The codegen detects the binding-count overflow and groups all 12 into one binding.
        public struct ManyIntViewsStruct
        {
            public ArrayView<int> V0;
            public ArrayView<int> V1;
            public ArrayView<int> V2;
            public ArrayView<int> V3;
            public ArrayView<int> V4;
            public ArrayView<int> V5;
            public ArrayView<int> V6;
            public ArrayView<int> V7;
            public ArrayView<int> V8;
            public ArrayView<int> V9;
            public ArrayView<int> V10;
            public ArrayView<int> V11;
        }

        static void ManyIntViewsKernel(Index1D idx, ArrayView<int> output, ManyIntViewsStruct s)
        {
            output[idx] = s.V0[idx]
                        + s.V1[idx]
                        + s.V2[idx]
                        + s.V3[idx]
                        + s.V4[idx]
                        + s.V5[idx]
                        + s.V6[idx]
                        + s.V7[idx]
                        + s.V8[idx]
                        + s.V9[idx]
                        + s.V10[idx]
                        + s.V11[idx];
        }

        // Diagnostic kernel: writes EACH field's [0] value to a separate output slot.
        // Reveals which field reads return the right buffer's data. If output[f] = refData[f][0]
        // for every f, the body-struct decomp is fine. If any output[f] is wrong, the kernel's
        // GetField for that f reads from a wrong offset / wrong buffer pointer.
        // Used to bisect the Wasm body-struct decomp many-field bug (task #16).
        static void ManyIntViewsDiagnosticKernel(Index1D _, ArrayView<int> output, ManyIntViewsStruct s)
        {
            output[0] = s.V0[0];
            output[1] = s.V1[0];
            output[2] = s.V2[0];
            output[3] = s.V3[0];
            output[4] = s.V4[0];
            output[5] = s.V5[0];
            output[6] = s.V6[0];
            output[7] = s.V7[0];
            output[8] = s.V8[0];
            output[9] = s.V9[0];
            output[10] = s.V10[0];
            output[11] = s.V11[0];
        }

        [TestMethod]
        public async Task BodyStruct_12ArrayViewInt_PerFieldDiagnostic() => await RunEmulatedTest(async accelerator =>
        {
            const int len = 64; // small per-buffer length, only need [0]
            var refData = new int[12][];
            var bufs = new MemoryBuffer1D<int, Stride1D.Dense>[12];
            try
            {
                for (int f = 0; f < 12; f++)
                {
                    refData[f] = new int[len];
                    for (int i = 0; i < len; i++)
                        refData[f][i] = (f + 1) * 100000 + i;
                    bufs[f] = accelerator.Allocate1D(refData[f]);
                }
                var s = new ManyIntViewsStruct
                {
                    V0 = bufs[0].View, V1 = bufs[1].View, V2 = bufs[2].View, V3 = bufs[3].View,
                    V4 = bufs[4].View, V5 = bufs[5].View, V6 = bufs[6].View, V7 = bufs[7].View,
                    V8 = bufs[8].View, V9 = bufs[9].View, V10 = bufs[10].View, V11 = bufs[11].View,
                };
                using var output = accelerator.Allocate1D<int>(12);
                var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ManyIntViewsStruct>(ManyIntViewsDiagnosticKernel);
                kernel((Index1D)1, output.View, s);
                await accelerator.SynchronizeAsync();
                var result = await output.CopyToHostAsync<int>();

                // Each output[f] should equal (f+1)*100000.
                var sb = new System.Text.StringBuilder();
                bool anyWrong = false;
                for (int f = 0; f < 12; f++)
                {
                    int expected = (f + 1) * 100000;
                    bool ok = result[f] == expected;
                    if (!ok) anyWrong = true;
                    sb.Append($"f{f}:{(ok ? "OK" : $"WRONG got={result[f]} expect={expected}")}; ");
                }
                if (anyWrong)
                {
                    // Wasm-only: pull the IR-layout-vs-CLR-serialization diagnostic
                    // accumulator. These strings record exactly what the host wrote
                    // into struct memory and at what offsets, so a mismatch between
                    // what the kernel reads and what the host wrote becomes visible.
                    string wasmDiag = "";
                    try
                    {
                        // Look up WasmAccelerator by name across all loaded assemblies.
                        // Avoids a project-time reference to SpawnDev.ILGPU.Wasm internals.
                        System.Type? accelType = null;
                        foreach (var a in System.AppDomain.CurrentDomain.GetAssemblies())
                        {
                            var t = a.GetType("SpawnDev.ILGPU.Wasm.WasmAccelerator");
                            if (t != null) { accelType = t; break; }
                        }
                        if (accelType != null)
                        {
                            var idxField = accelType.GetField("_lastImplicitIndexDebug",
                                System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static);
                            var serField = accelType.GetField("_lastStructSerialDebug",
                                System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static);
                            wasmDiag = " | implDbg=" + (idxField?.GetValue(null) ?? "")
                                     + " | serDbg=" + (serField?.GetValue(null) ?? "");
                        }
                    }
                    catch { /* not Wasm — debug accumulators aren't available */ }
                    throw new Exception("Per-field diagnostic: " + sb.ToString() + wasmDiag);
                }
            }
            finally
            {
                for (int f = 0; f < 12; f++) bufs[f]?.Dispose();
            }
        });

        [TestMethod]
        public async Task BodyStruct_12ArrayViewInt_CoalesceTest() => await RunEmulatedTest(async accelerator =>
        {
            const int len = 256;
            var refData = new int[12][];
            var bufs = new MemoryBuffer1D<int, Stride1D.Dense>[12];
            try
            {
                for (int f = 0; f < 12; f++)
                {
                    refData[f] = new int[len];
                    for (int i = 0; i < len; i++)
                        refData[f][i] = (f + 1) * 100000 + i;
                    bufs[f] = accelerator.Allocate1D(refData[f]);
                }
                var s = new ManyIntViewsStruct
                {
                    V0 = bufs[0].View, V1 = bufs[1].View, V2 = bufs[2].View, V3 = bufs[3].View,
                    V4 = bufs[4].View, V5 = bufs[5].View, V6 = bufs[6].View, V7 = bufs[7].View,
                    V8 = bufs[8].View, V9 = bufs[9].View, V10 = bufs[10].View, V11 = bufs[11].View,
                };

                using var output = accelerator.Allocate1D<int>(len);
                var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ManyIntViewsStruct>(ManyIntViewsKernel);
                kernel((Index1D)len, output.View, s);
                await accelerator.SynchronizeAsync();
                var result = await output.CopyToHostAsync<int>();

                for (int i = 0; i < len; i++)
                {
                    int expected = 0;
                    for (int f = 0; f < 12; f++) expected += refData[f][i];
                    if (result[i] != expected)
                        throw new Exception($"Coalesce 12-int field mismatch at i={i}: expected {expected}, got {result[i]}");
                }
            }
            finally
            {
                for (int f = 0; f < 12; f++) bufs[f]?.Dispose();
            }
        });

        // Mixed group sizes: 11 ArrayView<int> fields + 1 ArrayView<float>. The two
        // types coalesce into separate bindings; only the int group needs coalescing
        // for binding-count compliance, but both groups should still work.
        public struct MixedIntFloatStruct
        {
            public ArrayView<int>   I0;
            public ArrayView<int>   I1;
            public ArrayView<int>   I2;
            public ArrayView<int>   I3;
            public ArrayView<int>   I4;
            public ArrayView<int>   I5;
            public ArrayView<int>   I6;
            public ArrayView<int>   I7;
            public ArrayView<int>   I8;
            public ArrayView<int>   I9;
            public ArrayView<int>   I10;
            public ArrayView<float> F0;
        }

        static void MixedIntFloatKernel(Index1D idx, ArrayView<float> output, MixedIntFloatStruct s)
        {
            int isum = s.I0[idx] + s.I1[idx] + s.I2[idx] + s.I3[idx] + s.I4[idx] + s.I5[idx]
                     + s.I6[idx] + s.I7[idx] + s.I8[idx] + s.I9[idx] + s.I10[idx];
            output[idx] = isum + s.F0[idx];
        }

        [TestMethod]
        public async Task BodyStruct_MixedIntFloatCoalesceTest() => await RunEmulatedTest(async accelerator =>
        {
            const int len = 128;
            var iData = new int[11][];
            var iBufs = new MemoryBuffer1D<int, Stride1D.Dense>[11];
            MemoryBuffer1D<float, Stride1D.Dense>? fBuf = null;
            try
            {
                for (int f = 0; f < 11; f++)
                {
                    iData[f] = new int[len];
                    for (int i = 0; i < len; i++)
                        iData[f][i] = (f + 1) * 1000 + i;
                    iBufs[f] = accelerator.Allocate1D(iData[f]);
                }
                var fData = new float[len];
                for (int i = 0; i < len; i++) fData[i] = i * 0.5f - 7.25f;
                fBuf = accelerator.Allocate1D(fData);

                var s = new MixedIntFloatStruct
                {
                    I0 = iBufs[0].View, I1 = iBufs[1].View, I2 = iBufs[2].View, I3 = iBufs[3].View,
                    I4 = iBufs[4].View, I5 = iBufs[5].View, I6 = iBufs[6].View, I7 = iBufs[7].View,
                    I8 = iBufs[8].View, I9 = iBufs[9].View, I10 = iBufs[10].View,
                    F0 = fBuf.View,
                };

                using var output = accelerator.Allocate1D<float>(len);
                var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, MixedIntFloatStruct>(MixedIntFloatKernel);
                kernel((Index1D)len, output.View, s);
                await accelerator.SynchronizeAsync();
                var result = await output.CopyToHostAsync<float>();

                for (int i = 0; i < len; i++)
                {
                    int isum = 0;
                    for (int f = 0; f < 11; f++) isum += iData[f][i];
                    float expected = isum + fData[i];
                    float got = result[i];
                    if (Math.Abs(got - expected) > 1e-3f)
                        throw new Exception($"Mixed int/float coalesce mismatch at i={i}: expected {expected}, got {got}");
                }
            }
            finally
            {
                for (int f = 0; f < 11; f++) iBufs[f]?.Dispose();
                fBuf?.Dispose();
            }
        });

        // Variable-length members in the same group. Tuvok's case has different
        // lengths per field — verify the runtime handles that correctly via the
        // ViewOffsetSlot routing of each field's coalesce-relative offset.
        public struct VariableLengthIntStruct
        {
            public ArrayView<int> Short32;
            public ArrayView<int> Med128;
            public ArrayView<int> Long512;
            public ArrayView<int> Tiny8;
            public ArrayView<int> Med96;
            public ArrayView<int> Med256;
            public ArrayView<int> Long384;
            public ArrayView<int> Long768;
            public ArrayView<int> Tiny16;
            public ArrayView<int> Med192;
            public ArrayView<int> Long640;
            public ArrayView<int> Tiny4;
        }

        // Each field stores a constant in its element 0 — kernel sums those constants
        // into output[idx]. Probes the field-by-field offset routing because each
        // field starts at a different point in the coalesced buffer.
        static void VariableLengthIntKernel(Index1D idx, ArrayView<int> output, VariableLengthIntStruct s)
        {
            output[idx] = s.Short32[0]
                        + s.Med128[0]
                        + s.Long512[0]
                        + s.Tiny8[0]
                        + s.Med96[0]
                        + s.Med256[0]
                        + s.Long384[0]
                        + s.Long768[0]
                        + s.Tiny16[0]
                        + s.Med192[0]
                        + s.Long640[0]
                        + s.Tiny4[0];
        }

        [TestMethod]
        public async Task BodyStruct_VariableLengthCoalesceTest() => await RunEmulatedTest(async accelerator =>
        {
            int[] lengths = { 32, 128, 512, 8, 96, 256, 384, 768, 16, 192, 640, 4 };
            int[] firstElementValues = { 11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121, 132 };
            var bufs = new MemoryBuffer1D<int, Stride1D.Dense>[12];
            const int outputLen = 64;
            try
            {
                for (int f = 0; f < 12; f++)
                {
                    var data = new int[lengths[f]];
                    data[0] = firstElementValues[f];
                    for (int i = 1; i < data.Length; i++) data[i] = -1; // sentinel — kernel only reads [0]
                    bufs[f] = accelerator.Allocate1D(data);
                }
                var s = new VariableLengthIntStruct
                {
                    Short32 = bufs[0].View, Med128 = bufs[1].View, Long512 = bufs[2].View,
                    Tiny8 = bufs[3].View, Med96 = bufs[4].View, Med256 = bufs[5].View,
                    Long384 = bufs[6].View, Long768 = bufs[7].View, Tiny16 = bufs[8].View,
                    Med192 = bufs[9].View, Long640 = bufs[10].View, Tiny4 = bufs[11].View,
                };

                using var output = accelerator.Allocate1D<int>(outputLen);
                var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, VariableLengthIntStruct>(VariableLengthIntKernel);
                kernel((Index1D)outputLen, output.View, s);
                await accelerator.SynchronizeAsync();
                var result = await output.CopyToHostAsync<int>();

                int expected = 0;
                foreach (var v in firstElementValues) expected += v;
                for (int i = 0; i < outputLen; i++)
                    if (result[i] != expected)
                        throw new Exception($"VariableLength coalesce mismatch at i={i}: expected {expected}, got {result[i]}");
            }
            finally
            {
                for (int f = 0; f < 12; f++) bufs[f]?.Dispose();
            }
        });

        #endregion
    }
}
