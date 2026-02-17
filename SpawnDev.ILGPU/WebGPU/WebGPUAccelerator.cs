using global::ILGPU;
using global::ILGPU.Backends;
using global::ILGPU.Runtime;
using SpawnDev.BlazorJS.JSObjects;
using SpawnDev.ILGPU.WebGPU.Backend;
using System.Collections.Concurrent;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.InteropServices;
using Array = System.Array;

namespace SpawnDev.ILGPU.WebGPU
{
    /// <summary>
    /// WebGPU accelerator implementation for ILGPU.
    /// Provides kernel compilation and execution capabilities using WebGPU compute shaders.
    /// </summary>
    public class WebGPUAccelerator : KernelAccelerator<WebGPUCompiledKernel, WebGPUKernel>
    {
        /// <summary>
        /// Gets the native WebGPU accelerator for low-level GPU access.
        /// </summary>
        public WebGPUNativeAccelerator NativeAccelerator { get; private set; } = null!;

        /// <summary>
        /// Gets the WebGPU backend used for kernel compilation.
        /// </summary>
        public WebGPUBackend Backend { get; private set; } = null!;

        /// <summary>
        /// Gets the set of WebGPU features enabled on this device.
        /// </summary>
        public HashSet<string> EnabledFeatures => NativeAccelerator?.EnabledFeatures ?? new HashSet<string>();

        /// <summary>
        /// Method info for the static RunKernel method used by kernel launchers.
        /// </summary>
        public static readonly MethodInfo RunKernelMethod = typeof(WebGPUAccelerator).GetMethod(
            nameof(RunKernel),
            BindingFlags.Public | BindingFlags.Static)!;

        #region Caching Infrastructure

        // Reflection cache: caches PropertyInfo/FieldInfo for dimension extraction per type
        private static readonly ConcurrentDictionary<Type, ReflectionMetadataCache> _reflectionCache = new();

        // Buffer pool for scalar arguments (per-device pools would require instance field)
        [ThreadStatic]
        private static List<GPUBuffer>? _scalarBufferPool;

        // Reusable lists to avoid per-dispatch allocations
        [ThreadStatic]
        private static List<GPUBuffer>? _reusableScalarReturnList;
        [ThreadStatic]
        private static List<GPUBindGroupEntry>? _reusableEntryList;

        private class ReflectionMetadataCache
        {
            public PropertyInfo? BaseViewProperty { get; set; }
            public PropertyInfo? IntLengthProperty { get; set; }
            public PropertyInfo? LengthProperty { get; set; }
            public PropertyInfo? WidthProperty { get; set; }
            public FieldInfo? XField { get; set; }
            public FieldInfo? YField { get; set; }
            public FieldInfo? ZField { get; set; }
            public PropertyInfo? XProperty { get; set; }
            public PropertyInfo? YProperty { get; set; }
            public PropertyInfo? ZProperty { get; set; }
        }

        private static ReflectionMetadataCache GetOrCreateReflectionCache(Type type)
        {
            if (!WebGPUBackend.EnableReflectionCaching)
                return BuildReflectionCache(type);

            return _reflectionCache.GetOrAdd(type, BuildReflectionCache);
        }

        private static ReflectionMetadataCache BuildReflectionCache(Type type)
        {
            var flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
            return new ReflectionMetadataCache
            {
                BaseViewProperty = type.GetProperty("BaseView", flags),
                IntLengthProperty = type.GetProperty("IntLength", flags),
                LengthProperty = type.GetProperty("Length", flags),
                WidthProperty = type.GetProperty("Width", flags),
                XField = type.GetField("X", flags),
                YField = type.GetField("Y", flags),
                ZField = type.GetField("Z", flags),
                XProperty = type.GetProperty("X", flags),
                YProperty = type.GetProperty("Y", flags),
                ZProperty = type.GetProperty("Z", flags)
            };
        }

        private static GPUBuffer GetPooledScalarBuffer(GPUDevice device)
        {
            if (!WebGPUBackend.EnableBufferPooling)
                return CreateScalarBuffer(device);

            _scalarBufferPool ??= new List<GPUBuffer>();

            // Try to find a reusable buffer
            if (_scalarBufferPool.Count > 0)
            {
                var buffer = _scalarBufferPool[_scalarBufferPool.Count - 1];
                _scalarBufferPool.RemoveAt(_scalarBufferPool.Count - 1);
                return buffer;
            }

            return CreateScalarBuffer(device);
        }

        private static void ReturnPooledScalarBuffer(GPUBuffer buffer)
        {
            if (!WebGPUBackend.EnableBufferPooling)
            {
                buffer.Destroy();
                buffer.Dispose();
                return;
            }

            _scalarBufferPool ??= new List<GPUBuffer>();
            // Limit pool size to prevent memory bloat (increased for batching headroom)
            if (_scalarBufferPool.Count < 64)
            {
                _scalarBufferPool.Add(buffer);
            }
            else
            {
                buffer.Destroy();
                buffer.Dispose();
            }
        }

        private static GPUBuffer CreateScalarBuffer(GPUDevice device)
        {
            return device.CreateBuffer(new GPUBufferDescriptor
            {
                Label = "PooledScalar",
                Size = 256,
                Usage = GPUBufferUsage.Storage | GPUBufferUsage.CopyDst,
                MappedAtCreation = false
            });
        }

        #endregion

        private WebGPUAccelerator(Context context, Device device) : base(context, device) { }

        /// <summary>
        /// Creates a new WebGPU accelerator asynchronously with default options.
        /// </summary>
        /// <param name="context">The ILGPU context.</param>
        /// <param name="device">The WebGPU device to use.</param>
        /// <returns>A task that represents the async creation of the accelerator.</returns>
        public static Task<WebGPUAccelerator> CreateAsync(Context context, WebGPUILGPUDevice device)
            => CreateAsync(context, device, null);

        /// <summary>
        /// Creates a new WebGPU accelerator asynchronously with the specified options.
        /// </summary>
        /// <param name="context">The ILGPU context.</param>
        /// <param name="device">The WebGPU device to use.</param>
        /// <param name="options">The backend configuration options (null for defaults).</param>
        /// <returns>A task that represents the async creation of the accelerator.</returns>
        public static async Task<WebGPUAccelerator> CreateAsync(Context context, WebGPUILGPUDevice device, WebGPUBackendOptions? options)
        {
            var accelerator = new WebGPUAccelerator(context, device);
            accelerator.NativeAccelerator = await device.NativeDevice.CreateAcceleratorAsync();
            accelerator.Backend = new WebGPUBackend(context, options ?? WebGPUBackendOptions.Default, accelerator.NativeAccelerator.EnabledFeatures);
            accelerator.Init(accelerator.Backend);
            accelerator.DefaultStream = accelerator.CreateStreamInternal();
            // Wire flush callback so WebGPUBuffer readback operations auto-flush pending dispatches
            accelerator.NativeAccelerator.FlushPendingCommands = () => accelerator.FlushPendingCommands();

            // Always log detected features (important for diagnostics)
            var features = accelerator.NativeAccelerator.EnabledFeatures;
            if (features.Count > 0)
                WebGPUBackend.Log($"[WebGPU] Enabled features ({features.Count}): {string.Join(", ", features)}");
            else
                WebGPUBackend.Log("[WebGPU] No optional features detected");

            return accelerator;
        }

        /// <summary>
        /// Creates a new WebGPU accelerator from an externally-provided GPUDevice.
        /// This is used when sharing a device with another library (e.g., ONNX Runtime Web).
        /// The external device is used directly — no adapter probing or device creation.
        /// </summary>
        /// <param name="context">The ILGPU context.</param>
        /// <param name="externalDevice">An existing GPUDevice (e.g., from ort.env.webgpu.device).</param>
        /// <param name="options">The backend configuration options (null for defaults).</param>
        /// <returns>A new WebGPU accelerator using the external device.</returns>
        public static WebGPUAccelerator CreateFromExternalDevice(Context context, GPUDevice externalDevice, WebGPUBackendOptions? options = null)
        {
            var ilgpuDevice = new WebGPUILGPUDevice(externalDevice);
            var accelerator = new WebGPUAccelerator(context, ilgpuDevice);
            accelerator.NativeAccelerator = WebGPUNativeAccelerator.CreateFromExternalDevice(externalDevice);
            accelerator.Backend = new WebGPUBackend(context, options ?? WebGPUBackendOptions.Default, accelerator.NativeAccelerator.EnabledFeatures);
            accelerator.Init(accelerator.Backend);
            accelerator.DefaultStream = accelerator.CreateStreamInternal();
            accelerator.NativeAccelerator.FlushPendingCommands = () => accelerator.FlushPendingCommands();

            var features = accelerator.NativeAccelerator.EnabledFeatures;
            if (features.Count > 0)
                WebGPUBackend.Log($"[WebGPU] Enabled features ({features.Count}): {string.Join(", ", features)}");
            else
                WebGPUBackend.Log("[WebGPU] No optional features detected");

            WebGPUBackend.Log("[WebGPU] Accelerator created from external GPUDevice");
            return accelerator;
        }

        /// <inheritdoc/>
        protected override WebGPUKernel CreateKernel(WebGPUCompiledKernel compiledKernel)
        {
            return new WebGPUKernel(this, compiledKernel, null);
        }

        /// <inheritdoc/>
        protected override WebGPUKernel CreateKernel(WebGPUCompiledKernel compiledKernel, MethodInfo launcher)
        {
            return new WebGPUKernel(this, compiledKernel, launcher);
        }

        /// <inheritdoc/>
        protected override MethodInfo GenerateKernelLauncherMethod(WebGPUCompiledKernel kernel, int customGroupSize)
        {
            var parameters = kernel.EntryPoint.Parameters;
            var indexType = kernel.EntryPoint.KernelIndexType;
            var argTypes = new List<Type> { typeof(Kernel), typeof(AcceleratorStream), indexType };
            for (int i = 0; i < parameters.Count; i++) argTypes.Add(parameters[i]);

            var dynamicMethod = new DynamicMethod("WebGPULauncher", typeof(void), argTypes.ToArray(), typeof(WebGPUAccelerator).Module);
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

        // Helper to robustly extract dimensions (X, Y, Z) using Duck Typing
        private static int[] ExtractDimensionsFromView(object view, Type viewType)
        {
            try
            {
                var flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;

                int[] GetXYZ(object d)
                {
                    if (d == null) return Array.Empty<int>();
                    var t = d.GetType();
                    int x = -1, y = -1, z = -1;

                    // Try to get X
                    var fX = t.GetField("X", flags);
                    if (fX != null) x = Convert.ToInt32(fX.GetValue(d));
                    else
                    {
                        var pX = t.GetProperty("X", flags);
                        if (pX != null) x = Convert.ToInt32(pX.GetValue(d));
                    }

                    // Try to get Y
                    var fY = t.GetField("Y", flags);
                    if (fY != null) y = Convert.ToInt32(fY.GetValue(d));
                    else
                    {
                        var pY = t.GetProperty("Y", flags);
                        if (pY != null) y = Convert.ToInt32(pY.GetValue(d));
                    }

                    // Try to get Z
                    var fZ = t.GetField("Z", flags);
                    if (fZ != null) z = Convert.ToInt32(fZ.GetValue(d));
                    else
                    {
                        var pZ = t.GetProperty("Z", flags);
                        if (pZ != null) z = Convert.ToInt32(pZ.GetValue(d));
                    }

                    if (x >= 0)
                    {
                        if (y >= 0)
                        {
                            if (z >= 0) return new int[] { x, y, z };
                            return new int[] { x, y };
                        }
                        return new int[] { x };
                    }
                    return Array.Empty<int>();
                }

                foreach (var field in viewType.GetFields(flags))
                {
                    if (field.FieldType.IsPrimitive || field.FieldType.IsPointer) continue;
                    try
                    {
                        var val = field.GetValue(view);
                        var res = GetXYZ(val);
                        if (res.Length > 0 && res[0] > 0) return res;
                    }
                    catch { }
                }

                // DIRECT PROPERTY CHECK (Fallback for 1D ArrayView/Base)
                var pIntLength = viewType.GetProperty("IntLength", flags);
                if (pIntLength != null)
                {
                    try
                    {
                        var val = (int)pIntLength.GetValue(view);
                        return new int[] { val };
                    }
                    catch { }
                }

                // Fallback to Length (Long)
                var pLength = viewType.GetProperty("Length", flags);
                if (pLength != null && (pLength.PropertyType == typeof(int) || pLength.PropertyType == typeof(long)))
                {
                    try
                    {
                        var val = Convert.ToInt32(pLength.GetValue(view));
                        return new int[] { val };
                    }
                    catch { }
                }

                foreach (var prop in viewType.GetProperties(flags))
                {
                    if (prop.GetIndexParameters().Length > 0) continue;
                    if (prop.PropertyType.IsPrimitive || prop.PropertyType.IsPointer) continue;
                    try
                    {
                        var val = prop.GetValue(view);
                        var res = GetXYZ(val);
                        if (res.Length > 0 && res[0] > 0) return res;
                    }
                    catch { }
                }

                var directWidth = viewType.GetProperty("Width", flags);
                if (directWidth != null)
                {
                    int x = Convert.ToInt32(directWidth.GetValue(view));
                    if (x > 0) return new int[] { x, 0 };
                }
            }
            catch { }
            return Array.Empty<int>();
        }

        /// <summary>
        /// Executes a WebGPU kernel with the specified parameters.
        /// </summary>
        /// <param name="kernel">The kernel to execute.</param>
        /// <param name="stream">The accelerator stream (not used in WebGPU).</param>
        /// <param name="dimension">The launch dimensions.</param>
        /// <param name="args">The kernel arguments.</param>
        public static void RunKernel(Kernel kernel, AcceleratorStream stream, object dimension, object[] args)
        {
            var webGpuAccel = (WebGPUAccelerator)kernel.Accelerator;
            var nativeAccel = webGpuAccel.NativeAccelerator;
            var webGpuKernel = (WebGPUKernel)kernel;
            var compiledKernel = webGpuKernel.CompiledKernel;

            // ---- DEBUG LOGGING: WGSL SOURCE ----
            WebGPUBackend.Log("\n[WebGPU-Debug] ---- GENERATED WGSL ----");
            WebGPUBackend.Log(compiledKernel.WGSLSource);
            WebGPUBackend.Log("[WebGPU-Debug] ------------------------\n");
            // ------------------------------------

            // Build override constants for dynamic shared memory
            Dictionary<string, object>? overrideConstants = null;
            if (compiledKernel.HasDynamicSharedMemory && dimension is KernelConfig kernelConfig && kernelConfig.UsesDynamicSharedMemory)
            {
                overrideConstants = new Dictionary<string, object>();
                var sharedMemConfig = kernelConfig.SharedMemoryConfig;
                foreach (var overrideInfo in compiledKernel.DynamicSharedOverrides)
                {
                    // SharedMemoryConfig provides total bytes = NumElements * ElementSize
                    // The WGSL override sizes the array in elements of the declared type,
                    // so we need: total bytes / element size of the WGSL type
                    int numElements = sharedMemConfig.NumElements;
                    if (sharedMemConfig.ElementSize != overrideInfo.ElementSize && overrideInfo.ElementSize > 0)
                    {
                        // The element types may differ between C# request and WGSL declaration.
                        // Convert from bytes to elements of the WGSL type.
                        numElements = (sharedMemConfig.NumElements * sharedMemConfig.ElementSize) / overrideInfo.ElementSize;
                    }
                    overrideConstants[overrideInfo.ConstantName] = (double)numElements;
                    WebGPUBackend.Log($"[WebGPU-Debug] Dynamic shared memory override: {overrideInfo.ConstantName} = {numElements}");
                }
            }

            var shader = nativeAccel.GetOrCreateComputeShader(compiledKernel.WGSLSource, "main", overrideConstants);
            var device = nativeAccel.NativeDevice!;

            // Track scalar buffers for pool return (reuse list to avoid per-frame allocation)
            _reusableScalarReturnList ??= new List<GPUBuffer>();
            _reusableScalarReturnList.Clear();
            var scalarBuffersToReturn = _reusableScalarReturnList;

            try
            {
                int currentBindingIndex = 0;
                _reusableEntryList ??= new List<GPUBindGroupEntry>();
                _reusableEntryList.Clear();
                var entries = _reusableEntryList;

                // Build a lookup of packed scalar params by their args-array index.
                // ScalarPackingManifest stores IR param.Index (0-based, includes implicit index param).
                // The WGSL generator (KernelParamOffset) always skips param 0 if it's an Index type,
                // regardless of KernelConfig grouping. We must replicate this logic here.
                //
                // For auto-grouped kernels: KernelIndexParameterOffset=1, args excludes Index
                // For stream kernels with Index: KernelIndexParameterOffset=0, args INCLUDES Index
                // But the WGSL generator skips Index param in both cases (KernelParamOffset=1).
                //
                // So we need a "runtime skip" count to align args[] with WGSL bindings.
                int kernelParamOffset = compiledKernel.EntryPoint.KernelIndexParameterOffset;
                var paramTypes = compiledKernel.EntryPoint.Parameters;

                // Detect if args[0] is an Index type that should be skipped for bindings.
                // This happens with explicitly grouped stream kernels where KernelIndexParameterOffset=0
                // but the WGSL generator still skips the Index param.
                int runtimeIndexSkip = 0;
                if (kernelParamOffset == 0 && paramTypes.Count > 0)
                {
                    var first = paramTypes[0];
                    if (first == typeof(Index1D) || first == typeof(Index2D) || first == typeof(Index3D) ||
                        first == typeof(LongIndex1D) || first == typeof(LongIndex2D) || first == typeof(LongIndex3D))
                        runtimeIndexSkip = 1;
                }

                // The effective offset for mapping IR param indices to args[] indices:
                // effectiveOffset = kernelParamOffset + runtimeIndexSkip
                // For auto-grouped: effectiveOffset = 1 + 0 = 1
                // For stream with Index: effectiveOffset = 0 + 1 = 1
                // For stream without Index: effectiveOffset = 0 + 0 = 0
                int effectiveOffset = kernelParamOffset + runtimeIndexSkip;

                var packedScalarLookup = new Dictionary<int, ScalarPackingEntry>();
                if (compiledKernel.HasScalarPacking)
                {
                    foreach (var entry in compiledKernel.ScalarPackingManifest)
                    {
                        // Map from IR param index to args array index
                        int argsIndex = entry.ParamIndex - effectiveOffset;
                        if (argsIndex >= 0)
                            packedScalarLookup[argsIndex] = entry;
                    }
                }

                // --- Phase 1: Emit bindings for non-packed params (views, structs, atomics) ---
                // Note: args[] and paramTypes both exclude KernelIndexParameterOffset params.
                // But they may include the Index type param for explicitly grouped kernels.
                for (int i = 0; i < args.Length; i++)
                {
                    // Skip the Index type param at args[0] for explicitly grouped kernels
                    if (i < runtimeIndexSkip)
                        continue;

                    // Skip packed scalars — they'll be handled in Phase 2
                    if (packedScalarLookup.ContainsKey(i))
                        continue;

                    var arg = args[i];
                    IArrayView? arrayView = arg as IArrayView;
                    int[] dims = Array.Empty<int>();

                    if (arg != null)
                    {
                        var argType = arg.GetType();
                        var argCache = GetOrCreateReflectionCache(argType);
                        if (argType.Name.Contains("ArrayView"))
                        {
                            dims = ExtractDimensionsFromView(arg, argType);
                        }

                        if (arrayView == null && argCache.BaseViewProperty != null)
                        {
                            arrayView = argCache.BaseViewProperty.GetValue(arg) as IArrayView;
                        }
                    }

                    GPUBufferBinding? resource = null;

                    if (arrayView != null)
                    {
                        var contiguous = arrayView as IContiguousArrayView;
                        if (contiguous == null)
                        {
                            var viewCache = GetOrCreateReflectionCache(arrayView.GetType());
                            contiguous = (viewCache.BaseViewProperty != null ? viewCache.BaseViewProperty.GetValue(arrayView) : arrayView) as IContiguousArrayView;
                        }

                        if (contiguous == null) throw new Exception($"Argument {i} is not a contiguous WebGPU buffer");

                        var nativeBuffer = contiguous.Buffer as WebGPUMemoryBuffer;
                        var gpuBuffer = nativeBuffer!.NativeBuffer.NativeBuffer!;

                        WebGPUBackend.Log($"[WebGPU-Debug] Arg {i}: Binding Buffer. Size={contiguous.LengthInBytes}, Offset={contiguous.IndexInBytes}");

                        resource = new GPUBufferBinding
                        {
                            Buffer = gpuBuffer,
                            Offset = (ulong)((long)contiguous.IndexInBytes),
                            Size = (ulong)((long)contiguous.LengthInBytes)
                        };
                    }
                    else
                    {
                        // Non-packed, non-view param (struct scalar with own binding)
                        var size = 256;
                        var uBuffer = GetPooledScalarBuffer(device);
                        scalarBuffersToReturn.Add(uBuffer);

                        WebGPUBackend.Log($"[WebGPU-Debug] Arg {i}: Binding Struct Scalar. Value={arg}");

                        if (arg != null && arg.GetType().IsValueType)
                        {
                            int structSize = Marshal.SizeOf(arg.GetType());
                            byte[] bytes = new byte[structSize];
                            var handle = GCHandle.Alloc(bytes, GCHandleType.Pinned);
                            try { Marshal.StructureToPtr(arg, handle.AddrOfPinnedObject(), false); }
                            finally { handle.Free(); }
                            device.Queue.WriteBuffer(uBuffer, 0, bytes);
                            WebGPUBackend.Log($"[WebGPU-Debug] Arg {i}: Struct scalar {arg.GetType().Name}, Size={structSize} bytes");
                        }
                        else throw new NotSupportedException($"Unsupported non-packed non-view argument type: {arg?.GetType()}");

                        resource = new GPUBufferBinding { Buffer = uBuffer, Offset = 0, Size = (ulong)size };
                    }

                    entries.Add(new GPUBindGroupEntry { Binding = (uint)currentBindingIndex, Resource = resource! });
                    currentBindingIndex++;

                    if (dims.Length > 1)
                    {
                        WebGPUBackend.Log($"[WebGPU-Debug] Arg {i}: Binding Stride Buffer. Values=[{string.Join(", ", dims)}]");

                        var strideSize = 256;
                        var strideBuffer = GetPooledScalarBuffer(device);
                        scalarBuffersToReturn.Add(strideBuffer);

                        var strideData = new int[dims.Length];
                        Array.Copy(dims, strideData, dims.Length);
                        var byteData = new byte[dims.Length * 4];
                        Buffer.BlockCopy(strideData, 0, byteData, 0, byteData.Length);

                        device.Queue.WriteBuffer(strideBuffer, 0, byteData);

                        entries.Add(new GPUBindGroupEntry { Binding = (uint)currentBindingIndex, Resource = new GPUBufferBinding { Buffer = strideBuffer, Offset = 0, Size = (ulong)strideSize } });
                        currentBindingIndex++;
                    }
                }

                // --- Phase 2: Pack all scalar args into single buffer ---
                if (compiledKernel.HasScalarPacking)
                {
                    var manifest = compiledKernel.ScalarPackingManifest;

                    // Calculate total packed buffer size (round up to 4-byte alignment, min 256 for WebGPU)
                    int totalSlots = 0;
                    foreach (var entry in manifest)
                        totalSlots = Math.Max(totalSlots, entry.ByteOffset / 4 + entry.SlotCount);
                    int totalBytes = Math.Max(totalSlots * 4, 4);

                    var packedData = new byte[totalBytes];

                    foreach (var entry in manifest)
                    {
                        // Map from IR param index to args array index (using effectiveOffset, same as Phase 1)
                        int argsIdx = entry.ParamIndex - effectiveOffset;
                        if (argsIdx < 0 || argsIdx >= args.Length)
                            throw new InvalidOperationException($"[SCALAR-PACK] ParamIndex {entry.ParamIndex} (argsIdx={argsIdx}) is out of bounds for args (Length={args.Length}, offset={effectiveOffset})");
                        var arg = args[argsIdx];
                        int byteOffset = entry.ByteOffset;

                        WebGPUBackend.Log($"[WebGPU-Debug] Packing scalar param {entry.ParamIndex} (args[{argsIdx}]) at byte offset {byteOffset}: {arg}");

                        if (arg is int iVal)
                            BitConverter.GetBytes(iVal).CopyTo(packedData, byteOffset);
                        else if (arg is float fVal)
                            BitConverter.GetBytes(fVal).CopyTo(packedData, byteOffset);
                        else if (arg is uint uiVal)
                            BitConverter.GetBytes(uiVal).CopyTo(packedData, byteOffset);
                        else if (arg is long lVal)
                            BitConverter.GetBytes((int)lVal).CopyTo(packedData, byteOffset);
                        else if (arg is ulong ulVal)
                            BitConverter.GetBytes((uint)ulVal).CopyTo(packedData, byteOffset);
                        else if (arg is double dVal)
                        {
                            if (webGpuAccel.Backend.Options.EnableF64Emulation)
                            {
                                // Write full 64-bit IEEE-754 as 2 u32 values
                                BitConverter.GetBytes(dVal).CopyTo(packedData, byteOffset);
                            }
                            else
                            {
                                BitConverter.GetBytes((float)dVal).CopyTo(packedData, byteOffset);
                            }
                        }
                        else if (arg is byte bVal)
                            BitConverter.GetBytes((uint)bVal).CopyTo(packedData, byteOffset);
                        else if (arg is bool blVal)
                            BitConverter.GetBytes(blVal ? 1u : 0u).CopyTo(packedData, byteOffset);
                        else if (arg != null)
                            throw new NotSupportedException($"Unsupported packed scalar type: {arg.GetType()}");
                    }

                    var packedBuffer = GetPooledScalarBuffer(device);
                    scalarBuffersToReturn.Add(packedBuffer);
                    device.Queue.WriteBuffer(packedBuffer, 0, packedData);

                    entries.Add(new GPUBindGroupEntry
                    {
                        Binding = (uint)currentBindingIndex,
                        Resource = new GPUBufferBinding { Buffer = packedBuffer, Offset = 0, Size = 256 }
                    });
                    currentBindingIndex++;

                    WebGPUBackend.Log($"[WebGPU-Debug] Packed {manifest.Count} scalars into 1 buffer ({totalBytes} bytes used, binding {currentBindingIndex - 1})");
                }

                var bindGroupDesc = new GPUBindGroupDescriptor
                {
                    Layout = shader.Pipeline!.GetBindGroupLayout(0),
                    Entries = entries.ToArray()
                };

                var bindGroup = device.CreateBindGroup(bindGroupDesc);

                uint workX = 1, workY = 1, workZ = 1;

                // Handle KernelConfig for explicit launches
                if (dimension is KernelConfig config)
                {
                    workX = (uint)config.GridDim.X;
                    workY = (uint)config.GridDim.Y;
                    workZ = (uint)config.GridDim.Z;
                }
                else if (dimension is Index1D i1) workX = (uint)Math.Ceiling(i1.X / 64.0);
                else if (dimension is Index2D i2) { workX = (uint)Math.Ceiling(i2.X / 8.0); workY = (uint)Math.Ceiling(i2.Y / 8.0); }
                else if (dimension is Index3D i3) { workX = (uint)Math.Ceiling(i3.X / 4.0); workY = (uint)Math.Ceiling(i3.Y / 4.0); workZ = (uint)Math.Ceiling(i3.Z / 4.0); }
                else if (dimension is LongIndex1D l1) workX = (uint)Math.Ceiling(l1.X / 64.0);
                else if (dimension is LongIndex2D l2) { workX = (uint)Math.Ceiling(l2.X / 8.0); workY = (uint)Math.Ceiling(l2.Y / 8.0); }
                else if (dimension is LongIndex3D l3) { workX = (uint)Math.Ceiling(l3.X / 4.0); workY = (uint)Math.Ceiling(l3.Y / 4.0); workZ = (uint)Math.Ceiling(l3.Z / 4.0); }

                WebGPUBackend.Log($"[WebGPU-Debug] Dispatching: ({workX}, {workY}, {workZ})");

                // Use the stream's shared encoder for batched submission
                var webGpuStream = stream as WebGPUStream ?? (WebGPUStream)webGpuAccel.DefaultStream;
                var encoder = webGpuStream.GetOrCreateEncoder();
                using var pass = encoder.BeginComputePass();
                pass.SetPipeline(shader.Pipeline);
                pass.SetBindGroup(0, bindGroup);
                pass.DispatchWorkgroups(workX, workY, workZ);
                pass.End();
                webGpuStream.IncrementPassCount();

                // Defer resource cleanup until the batch is flushed
                webGpuStream.DeferBindGroupDisposal(bindGroup);
                foreach (var buffer in scalarBuffersToReturn)
                    webGpuStream.DeferScalarReturn(buffer);
                scalarBuffersToReturn.Clear();
            }
            catch (Exception ex)
            {
                WebGPUBackend.Log($"[WebGPU] Error running kernel: {ex}");
                throw;
            }
            finally
            {
                // Return any scalar buffers not deferred (error path)
                foreach (var buffer in scalarBuffersToReturn)
                {
                    ReturnPooledScalarBuffer(buffer);
                }
            }
        }

        protected override MemoryBuffer AllocateRawInternal(long length, int elementSize) => new WebGPUMemoryBuffer(this, length, elementSize);
        protected override AcceleratorStream CreateStreamInternal() => new WebGPUStream(this);
        protected override void SynchronizeInternal()
        {
            // Flush any pending batched commands on the default stream
            ((WebGPUStream)DefaultStream).Flush();
        }
        protected override void OnBind() { }
        protected override void OnUnbind() { }
        protected override void DisposeAccelerator_SyncRoot(bool disposing) { if (disposing) NativeAccelerator.Dispose(); }
        public override TExtension CreateExtension<TExtension, TExtensionProvider>(TExtensionProvider provider) => default;
        protected override PageLockScope<T> CreatePageLockFromPinnedInternal<T>(IntPtr ptr, long numElements) => throw new NotSupportedException();
        protected override int EstimateGroupSizeInternal(Kernel kernel, int dynamicSharedMemorySize, int maxGridSize, out int groupSize) { groupSize = 64; return 64; }
        protected override int EstimateGroupSizeInternal(Kernel kernel, Func<int, int> computeSharedMemorySize, int maxGridSize, out int groupSize) { groupSize = 64; return 64; }
        protected override int EstimateMaxActiveGroupsPerMultiprocessorInternal(Kernel kernel, int groupSize, int dynamicSharedMemorySize) => 1;
        protected override void EnablePeerAccessInternal(Accelerator other) { }
        protected override void DisablePeerAccessInternal(Accelerator other) { }
        protected override bool CanAccessPeerInternal(Accelerator other) => false;

        /// <summary>
        /// Flush any pending batched kernel dispatches to the GPU.
        /// Call this after kernel sequences before consuming results externally.
        /// </summary>
        public void FlushPendingCommands() => ((WebGPUStream)DefaultStream).Flush();

        /// <summary>
        /// WebGPU stream that batches multiple kernel dispatches into a single command buffer.
        /// Compute passes are accumulated in a shared GPUCommandEncoder and submitted together
        /// when Synchronize() or Flush() is called, matching CUDA's streaming dispatch model.
        /// </summary>
        private class WebGPUStream : AcceleratorStream
        {
            private readonly WebGPUAccelerator _webGpuAccelerator;
            private GPUCommandEncoder? _encoder;
            private readonly List<GPUBindGroup> _pendingBindGroups = new();
            private readonly List<GPUBuffer> _pendingScalarBuffers = new();
            private int _pendingPassCount;

            public WebGPUStream(Accelerator acc) : base(acc)
            {
                _webGpuAccelerator = (WebGPUAccelerator)acc;
            }

            /// <summary>
            /// Gets or creates the shared command encoder for this batch.
            /// </summary>
            internal GPUCommandEncoder GetOrCreateEncoder()
            {
                if (_encoder == null)
                {
                    var device = _webGpuAccelerator.NativeAccelerator.NativeDevice!;
                    _encoder = device.CreateCommandEncoder();
                    _pendingPassCount = 0;
                }
                return _encoder;
            }

            /// <summary>
            /// Defer a bind group's disposal until the batch is flushed.
            /// </summary>
            internal void DeferBindGroupDisposal(GPUBindGroup bg) => _pendingBindGroups.Add(bg);

            /// <summary>
            /// Defer a scalar buffer's return to pool until the batch is flushed.
            /// </summary>
            internal void DeferScalarReturn(GPUBuffer buf) => _pendingScalarBuffers.Add(buf);

            /// <summary>
            /// Flush accumulated compute passes: Finish the command encoder and submit to GPU queue.
            /// After submission, dispose deferred bind groups and return scalar buffers to pool.
            /// </summary>
            public void Flush()
            {
                if (_encoder == null) return;

                WebGPUBackend.Log($"[WebGPU] Flushing batch: {_pendingPassCount} compute passes");

                using var cmd = _encoder.Finish();
                _webGpuAccelerator.NativeAccelerator.Queue!.Submit(new[] { cmd });
                _encoder.Dispose();
                _encoder = null;

                // Clean up deferred resources
                foreach (var bg in _pendingBindGroups)
                    bg.Dispose();
                _pendingBindGroups.Clear();

                foreach (var buf in _pendingScalarBuffers)
                    ReturnPooledScalarBuffer(buf);
                _pendingScalarBuffers.Clear();

                _pendingPassCount = 0;
            }

            /// <summary>
            /// Track that a compute pass was appended to the current batch.
            /// </summary>
            internal void IncrementPassCount() => _pendingPassCount++;

            public override void Synchronize() => Flush();

            protected override void DisposeAcceleratorObject(bool disposing)
            {
                if (disposing) Flush();
            }

            protected override global::ILGPU.Runtime.ProfilingMarker AddProfilingMarkerInternal() => throw new NotSupportedException();
        }
    }

    /// <summary>
    /// Represents a compiled WebGPU kernel ready for execution.
    /// </summary>
    public class WebGPUKernel : Kernel
    {
        /// <summary>
        /// Creates a new WebGPU kernel instance.
        /// </summary>
        /// <param name="accelerator">The parent accelerator.</param>
        /// <param name="compiledKernel">The compiled kernel.</param>
        /// <param name="launcher">The launcher method info.</param>
        public WebGPUKernel(Accelerator accelerator, CompiledKernel compiledKernel, MethodInfo launcher) : base(accelerator, compiledKernel, launcher) { }

        /// <summary>
        /// Gets the WebGPU-specific compiled kernel.
        /// </summary>
        public new WebGPUCompiledKernel CompiledKernel => (WebGPUCompiledKernel)base.CompiledKernel;

        /// <inheritdoc/>
        protected override void DisposeAcceleratorObject(bool disposing) { }
    }
}