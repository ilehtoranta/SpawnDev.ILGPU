// ---------------------------------------------------------------------------------------
//                               SpawnDev.ILGPU.WebGL
//                 WebGL2 Compute Library for Blazor WebAssembly
//
// File: WebGLAccelerator.cs
// ---------------------------------------------------------------------------------------

using global::ILGPU;
using global::ILGPU.Backends;
using global::ILGPU.Runtime;
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
                            // Upload data as R32F texture for texelFetch
                            var elementSize = contiguous.ElementSize;
                            var length = (int)contiguous.Length;
                            var byteLength = length * elementSize;

                            // Read array data
                            var byteData = memBuffer.BackingArray.Read<byte>((int)contiguous.Index, byteLength);

                            // Create float texture (R32F, 1D texture laid out as width x 1)
                            var texWidth = (byteLength + 3) / 4; // ceil(bytes / 4) for R32F
                            var paddedData = new byte[texWidth * 4];
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
                            using var float32Array = new Float32Array(arrayBuffer);
                            gl.TexImage2D(GL.TEXTURE_2D, 0, GL.R32F, texWidth, 1, 0, GL.RED, GL.FLOAT, float32Array);
                            gl.Uniform1i(uniformLoc, texUnit);

                            Log($"[WebGL-Debug] Arg {pIdx}: Bound as texture unit {texUnit}, {texWidth}x1 R32F");
                        }
                    }
                    else
                    {
                        // Scalar argument → uniform
                        var uniformName = $"u_param{glslParamIndex}";
                        var uniformLoc = gl.GetUniformLocation(program, uniformName);
                        if (uniformLoc != null)
                        {
                            if (arg is int iVal) gl.Uniform1i(uniformLoc, iVal);
                            else if (arg is uint uiVal) gl.Uniform1ui(uniformLoc, uiVal);
                            else if (arg is float fVal) gl.Uniform1f(uniformLoc, fVal);
                            else if (arg is double dVal)
                            {
                                if (webGlAccel.Backend.Options.EnableF64Emulation)
                                {
                                    // f64 emulation: pass as 2 uints (low, high)
                                    var bits = BitConverter.DoubleToUInt64Bits(dVal);
                                    var lo = (uint)(bits & 0xFFFFFFFF);
                                    var hi = (uint)(bits >> 32);
                                    var uLoc = gl.GetUniformLocation(program, $"u_param{glslParamIndex}_emu");
                                    if (uLoc != null)
                                    {
                                        gl.Uniform2ui(uLoc, lo, hi);
                                    }
                                    else
                                    {
                                        gl.Uniform1f(uniformLoc, (float)dVal);
                                    }
                                }
                                else
                                {
                                    gl.Uniform1f(uniformLoc, (float)dVal);
                                }
                            }
                            else if (arg is bool blVal) gl.Uniform1i(uniformLoc, blVal ? 1 : 0);
                            else if (arg is byte bVal) gl.Uniform1i(uniformLoc, bVal);
                            else if (arg is long lVal) gl.Uniform1i(uniformLoc, (int)lVal);
                            else if (arg is ulong ulVal) gl.Uniform1ui(uniformLoc, (uint)ulVal);
                            else throw new NotSupportedException($"Unsupported scalar argument type: {arg?.GetType()}");

                            Log($"[WebGL-Debug] Arg {pIdx}: Bound as uniform, value={arg}");
                        }
                    }
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

                gl.Enable(GL.RASTERIZER_DISCARD);

                if (transformFeedback != null)
                {
                    gl.BeginTransformFeedback(GL.POINTS);
                }

                gl.DrawArrays(GL.POINTS, 0, totalVertices);

                if (transformFeedback != null)
                {
                    gl.EndTransformFeedback();
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
                    if (readbackBytes.Length >= 4)
                    {
                        var firstInt = BitConverter.ToInt32(readbackBytes, 0);
                        var firstFloat = BitConverter.ToSingle(readbackBytes, 0);
                        Log($"[WebGL-Debug] TF readback: {readbackBytes.Length} bytes, first 4 bytes as int={firstInt}, as float={firstFloat}");
                        if (readbackBytes.Length >= 16)
                        {
                            var secondInt = BitConverter.ToInt32(readbackBytes, 4);
                            var thirdInt = BitConverter.ToInt32(readbackBytes, 8);
                            var fourthInt = BitConverter.ToInt32(readbackBytes, 12);
                            Log($"[WebGL-Debug] TF readback values [0..3]: {firstInt}, {secondInt}, {thirdInt}, {fourthInt}");
                        }
                    }

                    for (int outIdx = 0; outIdx < outputVaryings.Count; outIdx++)
                    {
                        var outputInfo = outputVaryings[outIdx];
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
                                    // Calculate the offset in the TF buffer for this output
                                    int srcOffset = outIdx * totalVertices * 4;
                                    int copyLen = Math.Min(totalVertices * 4, readbackBytes.Length - srcOffset);
                                    copyLen = Math.Min(copyLen, (int)contiguous.LengthInBytes);

                                    if (copyLen > 0)
                                    {
                                        var slice = new byte[copyLen];
                                        System.Buffer.BlockCopy(readbackBytes, srcOffset, slice, 0, copyLen);
                                        memBuffer.BackingArray.Write(slice, (int)contiguous.Index);
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
