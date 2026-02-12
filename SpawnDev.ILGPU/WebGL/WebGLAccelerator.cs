// ---------------------------------------------------------------------------------------
//                               SpawnDev.ILGPU.WebGL
//                 WebGL2 Compute Library for Blazor WebAssembly
//
// File: WebGLAccelerator.cs
// ---------------------------------------------------------------------------------------

using global::ILGPU;
using global::ILGPU.Backends;
using global::ILGPU.Runtime;
using SpawnDev.BlazorJS;
using SpawnDev.BlazorJS.JSObjects;
using GL = SpawnDev.BlazorJS.JSObjects.GL;
using SpawnDev.ILGPU.WebGL.Backend;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.InteropServices;
using Array = System.Array;

namespace SpawnDev.ILGPU.WebGL
{
    /// <summary>
    /// WebGL2 accelerator implementation for ILGPU.
    /// Provides kernel compilation and execution capabilities using WebGL2 Transform Feedback.
    /// </summary>
    public class WebGLAccelerator : KernelAccelerator<WebGLCompiledKernel, WebGLKernel>
    {
        /// <summary>
        /// Gets the WebGL backend used for kernel compilation.
        /// </summary>
        public WebGLBackend Backend { get; private set; } = null!;

        /// <summary>
        /// Gets the WebGL2 rendering context.
        /// </summary>
        public WebGL2RenderingContext GLContext { get; private set; } = null!;

        /// <summary>
        /// Gets the OffscreenCanvas used for the WebGL2 context.
        /// </summary>
        public OffscreenCanvas Canvas { get; private set; } = null!;

        /// <summary>
        /// Method info for the static RunKernel method used by kernel launchers.
        /// </summary>
        public static readonly MethodInfo RunKernelMethod = typeof(WebGLAccelerator).GetMethod(
            nameof(RunKernel),
            BindingFlags.Public | BindingFlags.Static)!;

        /// <summary>
        /// Controls verbose logging output.
        /// </summary>
        public static bool VerboseLogging { get; set; } = false;

        internal static void Log(string message)
        {
            if (VerboseLogging) Console.WriteLine(message);
        }

        #region Construction

        private WebGLAccelerator(Context context, Device device) : base(context, device) { }

        /// <summary>
        /// Creates a new WebGL2 accelerator.
        /// </summary>
        public static WebGLAccelerator Create(Context context, WebGLILGPUDevice device, WebGLBackendOptions? options)
        {
            var accelerator = new WebGLAccelerator(context, device);
            var (canvas, gl) = device.NativeDevice.CreateContext();
            accelerator.Canvas = canvas;
            accelerator.GLContext = gl;
            accelerator.Backend = new WebGLBackend(context, options ?? WebGLBackendOptions.Default);
            accelerator.Init(accelerator.Backend);
            accelerator.DefaultStream = accelerator.CreateStreamInternal();

            Console.WriteLine($"[WebGL] Accelerator created: {device.NativeDevice.Name}");
            Console.WriteLine($"[WebGL] Max TF Components: {device.NativeDevice.MaxTransformFeedbackInterleavedComponents}");

            return accelerator;
        }

        #endregion

        #region Kernel Management

        /// <inheritdoc/>
        protected override WebGLKernel CreateKernel(WebGLCompiledKernel compiledKernel)
        {
            return new WebGLKernel(this, compiledKernel, null);
        }

        /// <inheritdoc/>
        protected override WebGLKernel CreateKernel(WebGLCompiledKernel compiledKernel, MethodInfo launcher)
        {
            return new WebGLKernel(this, compiledKernel, launcher);
        }

        /// <inheritdoc/>
        protected override MethodInfo GenerateKernelLauncherMethod(WebGLCompiledKernel kernel, int customGroupSize)
        {
            var parameters = kernel.EntryPoint.Parameters;
            var indexType = kernel.EntryPoint.KernelIndexType;
            var argTypes = new List<Type> { typeof(Kernel), typeof(AcceleratorStream), indexType };
            for (int i = 0; i < parameters.Count; i++) argTypes.Add(parameters[i]);

            var dynamicMethod = new DynamicMethod("WebGLLauncher", typeof(void), argTypes.ToArray(), typeof(WebGLAccelerator).Module);
            var ilGenerator = dynamicMethod.GetILGenerator();
            var argsLocal = ilGenerator.DeclareLocal(typeof(object[]));

            ilGenerator.Emit(OpCodes.Ldc_I4, parameters.Count);
            ilGenerator.Emit(OpCodes.Newarr, typeof(object));
            ilGenerator.Emit(OpCodes.Stloc, argsLocal);

            for (int i = 0; i < parameters.Count; i++)
            {
                ilGenerator.Emit(OpCodes.Ldloc, argsLocal);
                ilGenerator.Emit(OpCodes.Ldc_I4, i);
                ilGenerator.Emit(OpCodes.Ldarg, i + 3);
                var paramType = parameters[i];
                if (paramType.IsValueType) ilGenerator.Emit(OpCodes.Box, paramType);
                ilGenerator.Emit(OpCodes.Stelem_Ref);
            }

            ilGenerator.Emit(OpCodes.Ldarg_0);
            ilGenerator.Emit(OpCodes.Ldarg_1);
            ilGenerator.Emit(OpCodes.Ldarg_2);
            if (indexType.IsValueType) ilGenerator.Emit(OpCodes.Box, indexType);

            ilGenerator.Emit(OpCodes.Ldloc, argsLocal);
            ilGenerator.EmitCall(OpCodes.Call, RunKernelMethod, null);
            ilGenerator.Emit(OpCodes.Ret);

            return dynamicMethod;
        }

        #endregion

        #region Kernel Execution

        /// <summary>
        /// Executes a WebGL2 kernel with the specified parameters via Transform Feedback.
        /// </summary>
        public static void RunKernel(Kernel kernel, AcceleratorStream stream, object dimension, object[] args)
        {
            var webGlAccel = (WebGLAccelerator)kernel.Accelerator;
            var webGlKernel = (WebGLKernel)kernel;
            var compiledKernel = webGlKernel.CompiledKernel;
            var gl = webGlAccel.GLContext;

            Log("\n[WebGL-Debug] ---- GENERATED GLSL ----");
            Log(compiledKernel.GLSLSource);
            Log("[WebGL-Debug] ------------------------\n");

            // Determine dispatch size (total vertices = total work items)
            int totalVertices = 1;
            int dimX = 1, dimY = 1, dimZ = 1;

            if (dimension is KernelConfig config)
            {
                dimX = config.GridDim.X * config.GroupDim.X;
                dimY = config.GridDim.Y * config.GroupDim.Y;
                dimZ = config.GridDim.Z * config.GroupDim.Z;
                totalVertices = dimX * dimY * dimZ;
            }
            else if (dimension is Index1D i1) { dimX = i1.X; totalVertices = dimX; }
            else if (dimension is Index2D i2) { dimX = i2.X; dimY = i2.Y; totalVertices = dimX * dimY; }
            else if (dimension is Index3D i3) { dimX = i3.X; dimY = i3.Y; dimZ = i3.Z; totalVertices = dimX * dimY * dimZ; }
            else if (dimension is LongIndex1D l1) { dimX = (int)l1.X; totalVertices = dimX; }
            else if (dimension is LongIndex2D l2) { dimX = (int)l2.X; dimY = (int)l2.Y; totalVertices = dimX * dimY; }
            else if (dimension is LongIndex3D l3) { dimX = (int)l3.X; dimY = (int)l3.Y; dimZ = (int)l3.Z; totalVertices = dimX * dimY * dimZ; }

            Log($"[WebGL-Debug] Dispatch: {totalVertices} vertices (dim={dimX}x{dimY}x{dimZ})");

            // ---- Step 1: Compile shader program ----
            var vertexShaderSource = compiledKernel.GLSLSource;
            Console.WriteLine($"[WebGL-SHADER-SRC]\n{vertexShaderSource}");

            // Minimal fragment shader (required but output is discarded via RASTERIZER_DISCARD)
            var fragmentShaderSource = "#version 300 es\nprecision mediump float;\nvoid main() {}\n";

            using var vertShader = CompileShader(gl, GL.VERTEX_SHADER, vertexShaderSource);
            using var fragShader = CompileShader(gl, GL.FRAGMENT_SHADER, fragmentShaderSource);
            using var program = gl.CreateProgram();
            gl.AttachShader(program, vertShader);
            gl.AttachShader(program, fragShader);

            // ---- Step 2: Set up Transform Feedback varyings ----
            // The GLSL kernel generator emits `out` varyings (e.g. `tf_output_0`)
            // Collect the varying names from the compiled kernel's output metadata
            var varyingNames = compiledKernel.OutputVaryings
                .Select(o => o.VaryingName)
                .ToArray();

            if (varyingNames.Length > 0)
            {
                gl.TransformFeedbackVaryings(program, varyingNames, GL.INTERLEAVED_ATTRIBS);
            }

            gl.LinkProgram(program);
            var linkStatus = gl.GetProgramParameter<bool>(program, GL.LINK_STATUS);
            if (!linkStatus)
            {
                var log = gl.GetProgramInfoLog(program);
                throw new InvalidOperationException($"[WebGL] Program link failed:\n{log}");
            }

            gl.UseProgram(program);

            // ---- Step 3: Upload dimension uniforms ----
            var dimWidthLoc = gl.GetUniformLocation(program, "u_dimWidth");
            if (dimWidthLoc != null) gl.Uniform1i(dimWidthLoc, dimX);

            var dimHeightLoc = gl.GetUniformLocation(program, "u_dimHeight");
            if (dimHeightLoc != null) gl.Uniform1i(dimHeightLoc, dimY);

            // ---- Step 4: Bind input parameters ----
            var disposables = new List<IDisposable>();
            int textureUnit = 0;

            // Use fully qualified name to avoid collision with SpawnDev.BlazorJS.JSObjects.WebGLBuffer
            var glDisposables = new List<IDisposable>();

            try
            {
                // The GLSL generator uses Method.Parameters indices (0 = implicit index,
                // 1+ = data params), but args[] and EntryPoint.Parameters use 0-based
                // indexing without the implicit index. We add an offset of 1 so uniform
                // names (u_param1, u_param2, ...) match the generated GLSL.
                const int glslParamOffset = 1; // KernelParamOffset in the GLSL generator

                for (int pIdx = 0; pIdx < args.Length; pIdx++)
                {
                    int glslParamIndex = pIdx + glslParamOffset;

                    var arg = args[pIdx];
                    IArrayView? arrayView = arg as IArrayView;

                    if (arrayView == null && arg != null)
                    {
                        var baseViewProp = arg.GetType().GetProperty("BaseView");
                        if (baseViewProp != null)
                            arrayView = baseViewProp.GetValue(arg) as IArrayView;
                    }

                    if (arrayView != null)
                    {
                        // Buffer argument → upload via texture (TBO emulation)
                        var contiguous = arrayView as IContiguousArrayView;
                        if (contiguous == null)
                        {
                            var baseViewProp = arrayView.GetType().GetProperty("BaseView");
                            contiguous = (baseViewProp != null ? baseViewProp.GetValue(arrayView) : arrayView) as IContiguousArrayView;
                        }

                        if (contiguous == null)
                            throw new Exception($"Argument {pIdx} is not a contiguous buffer");

                        var memBuffer = contiguous.Buffer as WebGLMemoryBuffer;
                        if (memBuffer?.BackingArray == null)
                            throw new Exception($"Argument {pIdx} has no backing array");

                        // Create a GL texture from the backing data for TBO access
                        var texUnit = textureUnit++;
                        var uniformName = $"u_param{glslParamIndex}";
                        var uniformLoc = gl.GetUniformLocation(program, uniformName);

                        if (uniformLoc != null)
                        {
                            // Upload data as a 1D texture for texelFetch access
                            var elementSize = contiguous.ElementSize;
                            var length = (int)contiguous.Length;
                            var byteLength = length * elementSize;

                            // Read array data
                            var byteData = memBuffer.BackingArray.Read<byte>((int)contiguous.Index, byteLength);

                            // Determine texture format from the parameter binding's GLSL type
                            var binding = compiledKernel.ParameterBindings
                                .FirstOrDefault(b => b.ParamIndex == glslParamIndex && b.Kind == KernelParamKind.Buffer);
                            string bufferGlslType = binding.GlslType ?? "float";

                            var texelCount = (byteLength + 3) / 4; // ceil(bytes / 4)

                            // 2D texture tiling: when texel count exceeds MAX_TEXTURE_SIZE,
                            // tile data into rows of maxTexSize width
                            int maxTexSize = gl.GetParameter<int>(GL.MAX_TEXTURE_SIZE);
                            int tileWidth, tileHeight;
                            if (texelCount > maxTexSize)
                            {
                                tileWidth = maxTexSize;
                                tileHeight = (texelCount + tileWidth - 1) / tileWidth;
                            }
                            else
                            {
                                tileWidth = texelCount;
                                tileHeight = 1;
                            }

                            // Pad data to fill the full 2D texture (tileWidth * tileHeight texels)
                            int totalTexels = tileWidth * tileHeight;
                            var paddedData = new byte[totalTexels * 4];
                            System.Buffer.BlockCopy(byteData, 0, paddedData, 0, byteLength);


                            gl.ActiveTexture(GL.TEXTURE0 + (uint)texUnit);
                            var tex = gl.CreateTexture();
                            glDisposables.Add(tex);
                            gl.BindTexture(GL.TEXTURE_2D, tex);
                            gl.TexParameteri(GL.TEXTURE_2D, GL.TEXTURE_MIN_FILTER, GL.NEAREST);
                            gl.TexParameteri(GL.TEXTURE_2D, GL.TEXTURE_MAG_FILTER, GL.NEAREST);
                            gl.TexParameteri(GL.TEXTURE_2D, GL.TEXTURE_WRAP_S, GL.CLAMP_TO_EDGE);
                            gl.TexParameteri(GL.TEXTURE_2D, GL.TEXTURE_WRAP_T, GL.CLAMP_TO_EDGE);

                            using var uint8Array = new Uint8Array(paddedData);
                            using var arrayBuffer = uint8Array.Buffer;

                            if (bufferGlslType == "int")
                            {
                                // Integer buffer: use R32I with isampler2D
                                using var int32Array = new Int32Array(arrayBuffer);

                                gl.JSRef!.CallVoid("texImage2D",
                                    GL.TEXTURE_2D, 0, GL.R32I, tileWidth, tileHeight, 0,
                                    GL.RED_INTEGER, GL.INT, int32Array);

                                Log($"[WebGL-Debug] Arg {pIdx}: Bound as texture unit {texUnit}, {tileWidth}x{tileHeight} R32I (int)");
                            }
                            else if (bufferGlslType == "uint")
                            {
                                // Unsigned integer buffer: use R32UI with usampler2D
                                using var uint32Array = new Uint32Array(arrayBuffer);

                                gl.JSRef!.CallVoid("texImage2D",
                                    GL.TEXTURE_2D, 0, GL.R32UI, tileWidth, tileHeight, 0,
                                    GL.RED_INTEGER, GL.UNSIGNED_INT, uint32Array);

                                Log($"[WebGL-Debug] Arg {pIdx}: Bound as texture unit {texUnit}, {tileWidth}x{tileHeight} R32UI (uint)");
                            }
                            else
                            {
                                // Float buffer: use R32F with sampler2D (default)
                                using var float32Array = new Float32Array(arrayBuffer);

                                gl.TexImage2D(GL.TEXTURE_2D, 0, GL.R32F, tileWidth, tileHeight, 0, GL.RED, GL.FLOAT, float32Array);

                                Log($"[WebGL-Debug] Arg {pIdx}: Bound as texture unit {texUnit}, {tileWidth}x{tileHeight} R32F (float)");
                            }
                            gl.Uniform1i(uniformLoc, texUnit);

                            // Pass tile width uniform for 2D coordinate computation in shader
                            var tileWLoc = gl.GetUniformLocation(program, $"u_param{glslParamIndex}_tileW");
                            if (tileWLoc != null)
                                gl.Uniform1i(tileWLoc, tileWidth);

                        }
                        // ---- Stride uniforms for ArrayView2D/3D ----
                        // If this buffer arg has stride metadata, pass it as a uniform array.
                        // Extract dimension info from the argument to compute strides.
                        if (arg != null)
                        {
                            var argType = arg.GetType();
                            var strideLoc = gl.GetUniformLocation(program, $"u_param{glslParamIndex}_stride[0]");
                            if (strideLoc != null)
                            {
                                // This is a multi-dim view — extract Extent dimensions
                                int[] dims = ExtractViewDimensions(arg, argType);
                                if (dims.Length >= 2)
                                {
                                    using var strideArray = new Int32Array(dims);
                                    gl.Uniform1iv(strideLoc, strideArray);
                                    Console.WriteLine($"[WebGL-STRIDE-DEBUG] param{glslParamIndex}: stride=[{string.Join(", ", dims)}]");
                                }
                            }
                        }
                    }
                    else
                    {
                        // Scalar argument → uniform
                        // Emulated 64-bit types use separate _lo/_hi uniforms, so handle them
                        // before the base uniform lookup (which would fail for _lo/_hi names).
                        if (arg is double dVal && webGlAccel.Backend.Options.EnableF64Emulation)
                        {
                            // f64 emulation: pass as 2 separate uint uniforms (lo, hi)
                            var bits = BitConverter.DoubleToUInt64Bits(dVal);
                            var lo = (uint)(bits & 0xFFFFFFFF);
                            var hi = (uint)(bits >> 32);
                            var loLoc = gl.GetUniformLocation(program, $"u_param{glslParamIndex}_lo");
                            var hiLoc = gl.GetUniformLocation(program, $"u_param{glslParamIndex}_hi");
                            if (loLoc != null) gl.Uniform1ui(loLoc, lo);
                            if (hiLoc != null) gl.Uniform1ui(hiLoc, hi);
                            Console.WriteLine($"[WebGL-UNI-DEBUG] param{glslParamIndex}: f64 emulated value={dVal}, lo=0x{lo:X8} (loc={loLoc != null}), hi=0x{hi:X8} (loc={hiLoc != null})");
                        }
                        else if (arg is long lVal && webGlAccel.Backend.Options.EnableI64Emulation)
                        {
                            // i64 emulation: pass as 2 separate uint uniforms (lo, hi)
                            var bits = (ulong)lVal;
                            var lo = (uint)(bits & 0xFFFFFFFF);
                            var hi = (uint)(bits >> 32);
                            var loLoc = gl.GetUniformLocation(program, $"u_param{glslParamIndex}_lo");
                            var hiLoc = gl.GetUniformLocation(program, $"u_param{glslParamIndex}_hi");
                            if (loLoc != null) gl.Uniform1ui(loLoc, lo);
                            if (hiLoc != null) gl.Uniform1ui(hiLoc, hi);
                            Console.WriteLine($"[WebGL-UNI-DEBUG] param{glslParamIndex}: i64 emulated value={lVal}, lo=0x{lo:X8}, hi=0x{hi:X8}");
                        }
                        else
                        {
                            // Standard (non-emulated) scalar uniform
                            var uniformName = $"u_param{glslParamIndex}";
                            var uniformLoc = gl.GetUniformLocation(program, uniformName);
                            if (uniformLoc != null)
                            {
                                if (arg is int iVal) gl.Uniform1i(uniformLoc, iVal);
                                else if (arg is uint uiVal) gl.Uniform1ui(uniformLoc, uiVal);
                                else if (arg is float fVal) gl.Uniform1f(uniformLoc, fVal);
                                else if (arg is double dValFallback) gl.Uniform1f(uniformLoc, (float)dValFallback);
                                else if (arg is bool blVal) gl.Uniform1i(uniformLoc, blVal ? 1 : 0);
                                else if (arg is byte bVal) gl.Uniform1i(uniformLoc, bVal);
                                else if (arg is long lValFallback) gl.Uniform1i(uniformLoc, (int)lValFallback);
                                else if (arg is ulong ulVal) gl.Uniform1ui(uniformLoc, (uint)ulVal);
                                else throw new NotSupportedException($"Unsupported scalar argument type: {arg?.GetType()}");

                                Console.WriteLine($"[WebGL-UNI-DEBUG] param{glslParamIndex}: scalar value={arg} (type={arg?.GetType().Name})");
                                Log($"[WebGL-Debug] Arg {pIdx}: Bound as uniform, value={arg}");
                            }
                        }
                    }
                }

                // Check for GL errors after parameter binding
                {
                    var bindErr = gl.GetError();
                    if (bindErr != 0)
                        Console.WriteLine($"[WebGL-GL-ERROR] After param binding: error={bindErr}");
                }

                // ---- Step 5: Set up Transform Feedback output buffer ----
                SpawnDev.BlazorJS.JSObjects.WebGLBuffer? tfBuffer = null;
                WebGLTransformFeedback? transformFeedback = null;

                if (varyingNames.Length > 0)
                {
                    // Calculate TF output size (total floats * 4 bytes)
                    int tfFloatCount = totalVertices * varyingNames.Length;
                    int tfByteSize = tfFloatCount * 4;

                    tfBuffer = gl.CreateBuffer();
                    glDisposables.Add(tfBuffer);
                    gl.BindBuffer(GL.TRANSFORM_FEEDBACK_BUFFER, tfBuffer);
                    gl.BufferData(GL.TRANSFORM_FEEDBACK_BUFFER, tfByteSize, GL.DYNAMIC_READ);

                    transformFeedback = gl.CreateTransformFeedback();
                    glDisposables.Add(transformFeedback);
                    gl.BindTransformFeedback(GL.TRANSFORM_FEEDBACK, transformFeedback);
                    gl.BindBufferBase(GL.TRANSFORM_FEEDBACK_BUFFER, 0, tfBuffer);
                }

                // ---- Step 6: Dispatch via drawArrays with RASTERIZER_DISCARD ----
                // WebGL2 requires a VAO bound for DrawArrays to produce output
                var vao = gl.CreateVertexArray();
                glDisposables.Add(vao);
                gl.BindVertexArray(vao);

                // ANGLE D3D11 workaround: The D3D11 backend requires at least one
                // vertex attribute enabled for dynamic vertex executable compilation.
                // Without this, DrawArrays fails with "Error compiling dynamic vertex
                // executable" even though we only use gl_VertexID.
                var dummyVtxBuf = gl.CreateBuffer();
                glDisposables.Add(dummyVtxBuf);
                gl.BindBuffer(GL.ARRAY_BUFFER, dummyVtxBuf);
                gl.BufferData(GL.ARRAY_BUFFER, totalVertices * 4, GL.STATIC_DRAW);
                gl.EnableVertexAttribArray(0);
                gl.VertexAttribPointer(0, 1, (int)GL.FLOAT, false, 0, 0);

                gl.Enable(GL.RASTERIZER_DISCARD);

                if (transformFeedback != null)
                {
                    gl.BeginTransformFeedback(GL.POINTS);
                }

                gl.DrawArrays(GL.POINTS, 0, totalVertices);

                // Check for GL errors after draw
                var glErr = gl.GetError();
                if (glErr != 0)
                    Console.WriteLine($"[WebGL-GL-ERROR] After DrawArrays: error={glErr}");

                if (transformFeedback != null)
                {
                    gl.EndTransformFeedback();
                    glErr = gl.GetError();
                    if (glErr != 0)
                        Console.WriteLine($"[WebGL-GL-ERROR] After EndTransformFeedback: error={glErr}");
                }

                gl.Disable(GL.RASTERIZER_DISCARD);

                // ---- Step 7: Read back Transform Feedback results ----
                if (tfBuffer != null && varyingNames.Length > 0)
                {
                    // Read TF output and write back to output parameter backing arrays
                    gl.BindBuffer(GL.TRANSFORM_FEEDBACK_BUFFER, tfBuffer);

                    int tfFloatCount = totalVertices * varyingNames.Length;
                    int tfByteSize = tfFloatCount * 4;
                    using var readback = new Float32Array(tfFloatCount);
                    gl.GetBufferSubData(GL.TRANSFORM_FEEDBACK_BUFFER, 0, readback);

                    // Copy output data back to the appropriate ILGPU buffers
                    var outputVaryings = compiledKernel.OutputVaryings;
                    var readbackBytes = readback.ReadBytes();

                    // Debug: dump first few readback values
                    {
                        int numFloats = Math.Min(readbackBytes.Length / 4, 8);
                        var vals = new float[numFloats];
                        for (int f = 0; f < numFloats; f++)
                            vals[f] = BitConverter.ToSingle(readbackBytes, f * 4);
                        Console.WriteLine($"[WebGL-TF-DEBUG] TF readback: {readbackBytes.Length} bytes ({readbackBytes.Length / 4} floats), varyingCount={varyingNames.Length}, vertices={totalVertices}");
                        Console.WriteLine($"[WebGL-TF-DEBUG] First {numFloats} float values: [{string.Join(", ", vals)}]");
                        Console.WriteLine($"[WebGL-TF-DEBUG] OutputVaryings count={outputVaryings.Count}: [{string.Join(", ", outputVaryings.Select(o => $"param[{o.ParamIndex}]→{o.VaryingName}(outputIdx={o.OutputIndex})"))}]");
                    }

                    for (int outIdx = 0; outIdx < outputVaryings.Count; outIdx++)
                    {
                        var outputInfo = outputVaryings[outIdx];

                        // Skip "hi" emulated varyings — they are read together with their "lo" counterpart
                        if (outputInfo.IsEmulated && outputInfo.EmulatedSuffix == "hi")
                            continue;

                        // ParamIndex stores the Method.Parameters index (1-based);
                        // convert to args[] index (0-based) by subtracting the offset
                        var argsIdx = outputInfo.ParamIndex - glslParamOffset;

                        if (argsIdx >= 0 && argsIdx < args.Length)
                        {
                            var arg = args[argsIdx];
                            IArrayView? arrayView = arg as IArrayView;
                            if (arrayView == null && arg != null)
                            {
                                var baseViewProp = arg.GetType().GetProperty("BaseView");
                                if (baseViewProp != null)
                                    arrayView = baseViewProp.GetValue(arg) as IArrayView;
                            }

                            if (arrayView != null)
                            {
                                var contiguous = arrayView as IContiguousArrayView;
                                if (contiguous == null)
                                {
                                    var baseViewProp = arrayView.GetType().GetProperty("BaseView");
                                    contiguous = (baseViewProp != null ? baseViewProp.GetValue(arrayView) : arrayView) as IContiguousArrayView;
                                }

                                if (contiguous?.Buffer is WebGLMemoryBuffer memBuffer && memBuffer.BackingArray != null)
                                {
                                    // INTERLEAVED_ATTRIBS: data is packed per-vertex as
                                    // [varying0_v0, varying1_v0, ...varyingN_v0, varying0_v1, varying1_v1, ...]
                                    // Stride between same-varying values = varyingCount * 4 bytes
                                    int varyingCount = varyingNames.Length;
                                    int strideBytes = varyingCount * 4;  // bytes between same varying in adjacent vertices

                                    if (outputInfo.IsEmulated && outputInfo.EmulatedSuffix == "lo")
                                    {
                                        // Emulated 64-bit: read lo and hi varyings, interleave into 8-byte pairs
                                        int hiOutIdx = outIdx + 1; // hi varying is always right after lo
                                        int elementCount = Math.Min(totalVertices, (int)(contiguous.LengthInBytes / 8));
                                        if (elementCount > 0)
                                        {
                                            var slice = new byte[elementCount * 8];
                                            for (int v = 0; v < elementCount; v++)
                                            {
                                                // Read lo word from interleaved position
                                                int loByteOffset = v * strideBytes + outIdx * 4;
                                                // Read hi word from interleaved position
                                                int hiByteOffset = v * strideBytes + hiOutIdx * 4;
                                                if (loByteOffset + 4 <= readbackBytes.Length && hiByteOffset + 4 <= readbackBytes.Length)
                                                {
                                                    // Pack lo (4 bytes) + hi (4 bytes) into 8-byte slot
                                                    System.Buffer.BlockCopy(readbackBytes, loByteOffset, slice, v * 8, 4);
                                                    System.Buffer.BlockCopy(readbackBytes, hiByteOffset, slice, v * 8 + 4, 4);
                                                }
                                            }
                                            memBuffer.BackingArray.Write(slice, (int)contiguous.Index);
                                        }
                                    }
                                    else if (outputInfo.FieldIndex >= 0)
                                    {
                                        // Struct buffer TF readback: field varyings are interleaved per vertex
                                        // Skip non-first fields — they are read together with field 0
                                        if (outputInfo.FieldIndex > 0) continue;

                                        // Find all field varyings for this struct buffer param
                                        var structFieldVaryings = outputVaryings
                                            .Where(o => o.ParamIndex == outputInfo.ParamIndex && o.FieldIndex >= 0)
                                            .OrderBy(o => o.FieldIndex).ToList();
                                        int fieldCount = structFieldVaryings.Count;
                                        int structElementSize = fieldCount * 4; // 4 bytes per field
                                        int elementCount = Math.Min(totalVertices, (int)(contiguous.LengthInBytes / structElementSize));
                                        if (elementCount > 0)
                                        {
                                            var slice = new byte[elementCount * structElementSize];
                                            for (int v = 0; v < elementCount; v++)
                                            {
                                                for (int fi = 0; fi < fieldCount; fi++)
                                                {
                                                    // Find the outIdx (position in varyingNames) for this field varying
                                                    var fieldVarying = structFieldVaryings[fi];
                                                    int fieldOutIdx = fieldVarying.OutputIndex;
                                                    int srcByteOffset = v * strideBytes + fieldOutIdx * 4;
                                                    int dstByteOffset = v * structElementSize + fi * 4;
                                                    if (srcByteOffset + 4 <= readbackBytes.Length)
                                                    {
                                                        System.Buffer.BlockCopy(readbackBytes, srcByteOffset, slice, dstByteOffset, 4);
                                                    }
                                                }
                                            }
                                            memBuffer.BackingArray.Write(slice, (int)contiguous.Index);
                                        }
                                    }
                                    else
                                    {
                                        // Standard single-slot TF readback
                                        int elementCount = Math.Min(totalVertices, (int)(contiguous.LengthInBytes / 4));
                                        if (elementCount > 0)
                                        {
                                            var slice = new byte[elementCount * 4];
                                            for (int v = 0; v < elementCount; v++)
                                            {
                                                // Source offset in interleaved buffer: vertex v, varying outIdx
                                                int srcByteOffset = v * strideBytes + outIdx * 4;
                                                if (srcByteOffset + 4 <= readbackBytes.Length)
                                                {
                                                    System.Buffer.BlockCopy(readbackBytes, srcByteOffset, slice, v * 4, 4);
                                                }
                                            }
                                            memBuffer.BackingArray.Write(slice, (int)contiguous.Index);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Unbind TF
                if (transformFeedback != null)
                {
                    gl.BindTransformFeedback((uint)GL.TRANSFORM_FEEDBACK, null);
                }

                Log($"[WebGL-Debug] Kernel dispatch complete");
            }
            catch (Exception ex)
            {
                Log($"[WebGL] Error running kernel: {ex}");
                throw;
            }
            finally
            {
                gl.UseProgram(null);
                foreach (var d in glDisposables)
                    d.Dispose();
            }
        }

        /// <summary>
        /// Compiles a WebGL shader of the specified type.
        /// </summary>
        private static WebGLShader CompileShader(WebGL2RenderingContext gl, uint type, string source)
        {
            var shader = gl.CreateShader(type);
            gl.ShaderSource(shader, source);
            gl.CompileShader(shader);

            var success = gl.GetShaderParameter<bool>(shader, GL.COMPILE_STATUS);
            if (!success)
            {
                var log = gl.GetShaderInfoLog(shader);
                gl.DeleteShader(shader);
                throw new InvalidOperationException($"[WebGL] Shader compilation failed:\n{log}\n\nSource:\n{source}");
            }

            return shader;
        }

        #endregion

        #region Abstract Method Implementations

        protected override MemoryBuffer AllocateRawInternal(long length, int elementSize) =>
            new WebGLMemoryBuffer(this, length, elementSize);

        protected override AcceleratorStream CreateStreamInternal() => new WebGLStream(this);

        protected override void SynchronizeInternal()
        {
            // WebGL2 rendering is synchronous on the main thread.
            // glFinish() ensures all GL commands complete.
            GLContext?.Finish();
        }

        protected override void OnBind() { }
        protected override void OnUnbind() { }

        protected override void DisposeAccelerator_SyncRoot(bool disposing)
        {
            if (disposing)
            {
                GLContext?.Dispose();
                Canvas?.Dispose();
            }
        }

        public override TExtension CreateExtension<TExtension, TExtensionProvider>(TExtensionProvider provider) => default;
        protected override PageLockScope<T> CreatePageLockFromPinnedInternal<T>(IntPtr ptr, long numElements) => throw new NotSupportedException();
        protected override int EstimateGroupSizeInternal(Kernel kernel, int dynamicSharedMemorySize, int maxGridSize, out int groupSize) { groupSize = 1; return 1; }
        protected override int EstimateGroupSizeInternal(Kernel kernel, Func<int, int> computeSharedMemorySize, int maxGridSize, out int groupSize) { groupSize = 1; return 1; }
        protected override int EstimateMaxActiveGroupsPerMultiprocessorInternal(Kernel kernel, int groupSize, int dynamicSharedMemorySize) => 1;
        protected override void EnablePeerAccessInternal(Accelerator other) { }
        protected override void DisablePeerAccessInternal(Accelerator other) { }
        protected override bool CanAccessPeerInternal(Accelerator other) => false;

        /// <summary>
        /// Extracts [width, height] or [width, height, depth] dimensions from an ArrayView2D/3D argument.
        /// Used for setting stride uniforms in multi-dimensional view parameters.
        /// </summary>
        private static int[] ExtractViewDimensions(object arg, Type argType)
        {
            var flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic;

            // Try Extent property first (gives the logical dimensions)
            var extentProp = argType.GetProperty("Extent", flags) ?? argType.GetProperty("IntExtent", flags);
            if (extentProp != null)
            {
                try
                {
                    var extent = extentProp.GetValue(arg);
                    if (extent != null)
                    {
                        var extType = extent.GetType();
                        int x = -1, y = -1, z = -1;

                        var fX = extType.GetField("X", flags);
                        if (fX != null) x = Convert.ToInt32(fX.GetValue(extent));
                        else { var pX = extType.GetProperty("X", flags); if (pX != null) x = Convert.ToInt32(pX.GetValue(extent)); }

                        var fY = extType.GetField("Y", flags);
                        if (fY != null) y = Convert.ToInt32(fY.GetValue(extent));
                        else { var pY = extType.GetProperty("Y", flags); if (pY != null) y = Convert.ToInt32(pY.GetValue(extent)); }

                        var fZ = extType.GetField("Z", flags);
                        if (fZ != null) z = Convert.ToInt32(fZ.GetValue(extent));
                        else { var pZ = extType.GetProperty("Z", flags); if (pZ != null) z = Convert.ToInt32(pZ.GetValue(extent)); }

                        if (x > 0 && y > 0 && z > 0) return new[] { x, y, z };
                        if (x > 0 && y > 0) return new[] { x, y };
                    }
                }
                catch { }
            }

            // Fallback: scan non-primitive fields/properties looking for struct with X, Y, Z
            foreach (var field in argType.GetFields(flags))
            {
                if (field.FieldType.IsPrimitive || field.FieldType.IsPointer) continue;
                try
                {
                    var val = field.GetValue(arg);
                    if (val == null) continue;
                    var t = val.GetType();
                    var xF = t.GetField("X", flags);
                    var yF = t.GetField("Y", flags);
                    if (xF != null && yF != null)
                    {
                        int x = Convert.ToInt32(xF.GetValue(val));
                        int y = Convert.ToInt32(yF.GetValue(val));
                        if (x > 0 && y > 0)
                        {
                            var zF = t.GetField("Z", flags);
                            if (zF != null)
                            {
                                int z = Convert.ToInt32(zF.GetValue(val));
                                if (z > 0) return new[] { x, y, z };
                            }
                            return new[] { x, y };
                        }
                    }
                }
                catch { }
            }

            return Array.Empty<int>();
        }

        private class WebGLStream : AcceleratorStream
        {
            public WebGLStream(Accelerator acc) : base(acc) { }
            protected override void DisposeAcceleratorObject(bool disposing) { }
            public override void Synchronize() { }
            protected override global::ILGPU.Runtime.ProfilingMarker AddProfilingMarkerInternal() => throw new NotSupportedException();
        }

        #endregion
    }

    /// <summary>
    /// Represents a compiled WebGL2 kernel ready for execution.
    /// </summary>
    public class WebGLKernel : Kernel
    {
        public WebGLKernel(Accelerator accelerator, CompiledKernel compiledKernel, MethodInfo launcher)
            : base(accelerator, compiledKernel, launcher) { }

        /// <summary>
        /// Gets the WebGL-specific compiled kernel.
        /// </summary>
        public new WebGLCompiledKernel CompiledKernel => (WebGLCompiledKernel)base.CompiledKernel;

        /// <inheritdoc/>
        protected override void DisposeAcceleratorObject(bool disposing) { }
    }
}
