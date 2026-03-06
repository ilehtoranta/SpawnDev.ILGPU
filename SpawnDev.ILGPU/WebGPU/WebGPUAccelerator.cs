using global::ILGPU;
using global::ILGPU.Backends;
using global::ILGPU.Runtime;
using SpawnDev.BlazorJS.JSObjects;
using SpawnDev.ILGPU.WebGPU.Backend;
using System.Collections.Concurrent;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
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
        [ThreadStatic]
        private static List<object?>? _reusableExpandedArgs;
        [ThreadStatic]
        private static Dictionary<int, List<int>>? _reusableBodyStructMap;
        [ThreadStatic]
        private static HashSet<int>? _reusableBodyStructScalarSet;
        [ThreadStatic]
        private static Dictionary<int, ScalarPackingEntry>? _reusablePackedScalarLookup;

        // Cache for CopyStructToBytes<T> MethodInfo per type (avoids repeated MakeGenericMethod)
        private static readonly ConcurrentDictionary<Type, MethodInfo> _copyStructMethodCache = new();

        /// <summary>
        /// Copies the raw bytes of a value-type struct into the destination byte array.
        /// Uses MemoryMarshal.AsBytes to reinterpret the struct as bytes — no GCHandle pinning required.
        /// This works for all value types including closed generic structs.
        /// </summary>
        private static void CopyStructToBytes<T>(T value, byte[] dest) where T : struct
        {
            var span = System.Runtime.InteropServices.MemoryMarshal.AsBytes(
                System.Runtime.InteropServices.MemoryMarshal.CreateReadOnlySpan(ref value, 1));
            span.CopyTo(dest);
        }

        private static MethodInfo GetCopyStructMethod(Type type)
        {
            return _copyStructMethodCache.GetOrAdd(type, t =>
                typeof(WebGPUAccelerator)
                    .GetMethod(nameof(CopyStructToBytes), BindingFlags.NonPublic | BindingFlags.Static)!
                    .MakeGenericMethod(t));
        }

        /// <summary>
        /// Checks if a value-type struct contains any pointer-like fields (IntPtr, UIntPtr, or pointer types)
        /// at any nesting level. Such structs cannot be safely copied with MemoryMarshal.AsBytes in Blazor WASM
        /// and must be decomposed into their constituent fields.
        /// This catches IGridStrideKernelBody implementations like InitializerImplementation and
        /// ReductionImplementation which contain ArrayView fields (which internally hold an IntPtr).
        /// </summary>
        private static readonly ConcurrentDictionary<Type, bool> _containsPointerCache = new();
        private static bool ContainsPointerFields(Type type)
        {
            if (!type.IsValueType || type.IsPrimitive) return false;
            if (type == typeof(IntPtr) || type == typeof(UIntPtr)) return true;
            return _containsPointerCache.GetOrAdd(type, t =>
            {
                var flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
                foreach (var field in t.GetFields(flags))
                {
                    var ft = field.FieldType;
                    // MemoryMarshal.AsBytes fails for structs containing:
                    // 1. Raw pointer types (T*)
                    // 2. IntPtr / UIntPtr
                    // 3. Reference types (class instances, e.g. MemoryBuffer inside ArrayView<T>)
                    if (ft.IsPointer || ft == typeof(IntPtr) || ft == typeof(UIntPtr)) return true;
                    if (!ft.IsValueType) return true; // reference type field (class)
                    if (!ft.IsPrimitive && ContainsPointerFields(ft)) return true;
                }
                return false;
            });
        }

        /// <summary>
        /// Flattens a body struct into its constituent field values (in declaration order).
        /// This mirrors what the ILGPU compiler does when it inlines struct parameters into
        /// separate IR parameters.
        /// </summary>
        private static List<object?> FlattenStructFields(object structValue)
        {
            var result = new List<object?>();
            var flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
            var type = structValue.GetType();
            foreach (var field in type.GetFields(flags))
            {
                var fieldVal = field.GetValue(structValue);
                var ft = field.FieldType;
                // Recursively flatten nested structs that also contain pointers
                // but stop at IArrayView — those are leaf nodes handled by the view binding path
                if (fieldVal != null && ft.IsValueType && !ft.IsPrimitive
                    && !typeof(IArrayView).IsAssignableFrom(ft)
                    && ContainsPointerFields(ft))
                {
                    result.AddRange(FlattenStructFields(fieldVal));
                }
                else
                {
                    result.Add(fieldVal);
                }
            }
            return result;
        }

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
            accelerator.Backend.DefaultMaxWorkgroupSize = accelerator.MaxNumThreadsPerGroup;
            WebGPUBackend.Log($"[WebGPU-Init] DefaultMaxWorkgroupSize set to {accelerator.Backend.DefaultMaxWorkgroupSize} (MaxNumThreadsPerGroup={accelerator.MaxNumThreadsPerGroup})");
            accelerator.Init(accelerator.Backend);
            accelerator.DefaultStream = accelerator.CreateStreamInternal();
            // Wire flush callback so WebGPUBuffer readback operations auto-flush pending dispatches
            accelerator.NativeAccelerator.FlushPendingCommands = () => accelerator.FlushPendingCommands();

            // Update device capabilities from actual enabled features (Float16, Float64, Int64)
            if (device.Capabilities is WebGPUCapabilityContext webCaps)
                webCaps.UpdateFromEnabledFeatures(accelerator.NativeAccelerator.EnabledFeatures);

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
            accelerator.Backend.DefaultMaxWorkgroupSize = accelerator.MaxNumThreadsPerGroup;
            accelerator.Init(accelerator.Backend);
            accelerator.DefaultStream = accelerator.CreateStreamInternal();
            accelerator.NativeAccelerator.FlushPendingCommands = () => accelerator.FlushPendingCommands();

            // Update device capabilities from actual enabled features (Float16, Float64, Int64)
            if (ilgpuDevice.Capabilities is WebGPUCapabilityContext webCaps)
                webCaps.UpdateFromEnabledFeatures(accelerator.NativeAccelerator.EnabledFeatures);

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

        // Cache for dimension extraction: per-type function that extracts int[] dimensions from a view
        private static readonly ConcurrentDictionary<Type, Func<object, int[]>> _dimensionExtractorCache = new();

        // Helper to robustly extract dimensions (X, Y, Z) using Duck Typing, with caching
        private static int[] ExtractDimensionsFromView(object view, Type viewType)
        {
            var extractor = _dimensionExtractorCache.GetOrAdd(viewType, BuildDimensionExtractor);
            return extractor(view);
        }

        private static Func<object, int[]> BuildDimensionExtractor(Type viewType)
        {
            var flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;

            // Strategy 1: Find a sub-struct field with X/Y/Z members (e.g., Extent, Index)
            foreach (var field in viewType.GetFields(flags))
            {
                if (field.FieldType.IsPrimitive || field.FieldType.IsPointer) continue;
                var xyzAccessor = BuildXYZAccessor(field.FieldType);
                if (xyzAccessor != null)
                {
                    var capturedField = field;
                    var capturedAccessor = xyzAccessor;
                    return (obj) =>
                    {
                        try
                        {
                            var val = capturedField.GetValue(obj);
                            if (val != null)
                            {
                                var res = capturedAccessor(val);
                                if (res.Length > 0 && res[0] > 0) return res;
                            }
                        }
                        catch { }
                        return Array.Empty<int>();
                    };
                }
            }

            // Strategy 2: IntLength property (1D ArrayView)
            var pIntLength = viewType.GetProperty("IntLength", flags);
            if (pIntLength != null)
                return (obj) => { try { return new int[] { (int)pIntLength.GetValue(obj)! }; } catch { return Array.Empty<int>(); } };

            // Strategy 3: Length property
            var pLength = viewType.GetProperty("Length", flags);
            if (pLength != null && (pLength.PropertyType == typeof(int) || pLength.PropertyType == typeof(long)))
                return (obj) => { try { return new int[] { Convert.ToInt32(pLength.GetValue(obj)) }; } catch { return Array.Empty<int>(); } };

            // Strategy 4: Properties with X/Y/Z sub-structs
            foreach (var prop in viewType.GetProperties(flags))
            {
                if (prop.GetIndexParameters().Length > 0) continue;
                if (prop.PropertyType.IsPrimitive || prop.PropertyType.IsPointer) continue;
                var xyzAccessor = BuildXYZAccessor(prop.PropertyType);
                if (xyzAccessor != null)
                {
                    var capturedProp = prop;
                    var capturedAccessor = xyzAccessor;
                    return (obj) =>
                    {
                        try
                        {
                            var val = capturedProp.GetValue(obj);
                            if (val != null)
                            {
                                var res = capturedAccessor(val);
                                if (res.Length > 0 && res[0] > 0) return res;
                            }
                        }
                        catch { }
                        return Array.Empty<int>();
                    };
                }
            }

            // Strategy 5: Direct Width property
            var directWidth = viewType.GetProperty("Width", flags);
            if (directWidth != null)
                return (obj) => { try { int x = Convert.ToInt32(directWidth.GetValue(obj)); return x > 0 ? new int[] { x, 0 } : Array.Empty<int>(); } catch { return Array.Empty<int>(); } };

            // No dimension info found
            return (_) => Array.Empty<int>();
        }

        // Build a function that extracts X/Y/Z from a sub-struct type. Returns null if type has no X/Y/Z.
        private static Func<object, int[]>? BuildXYZAccessor(Type type)
        {
            var flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;

            // Resolve X accessor
            var fX = type.GetField("X", flags);
            var pX = fX == null ? type.GetProperty("X", flags) : null;
            if (fX == null && pX == null) return null;

            // Resolve Y accessor
            var fY = type.GetField("Y", flags);
            var pY = fY == null ? type.GetProperty("Y", flags) : null;

            // Resolve Z accessor
            var fZ = type.GetField("Z", flags);
            var pZ = fZ == null ? type.GetProperty("Z", flags) : null;

            return (obj) =>
            {
                int x = fX != null ? Convert.ToInt32(fX.GetValue(obj)) : (pX != null ? Convert.ToInt32(pX.GetValue(obj)) : -1);
                if (x < 0) return Array.Empty<int>();

                int y = fY != null ? Convert.ToInt32(fY.GetValue(obj)) : (pY != null ? Convert.ToInt32(pY.GetValue(obj)) : -1);
                if (y < 0) return new int[] { x };

                int z = fZ != null ? Convert.ToInt32(fZ.GetValue(obj)) : (pZ != null ? Convert.ToInt32(pZ.GetValue(obj)) : -1);
                return z >= 0 ? new int[] { x, y, z } : new int[] { x, y };
            };
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
            if (WebGPUBackend.VerboseLogging)
            {
                WebGPUBackend.Log("\n[WebGPU-Debug] ---- GENERATED WGSL ----");
                WebGPUBackend.Log(compiledKernel.WGSLSource);
                WebGPUBackend.Log("[WebGPU-Debug] ------------------------\n");
            }
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
                    if (WebGPUBackend.VerboseLogging)
                        WebGPUBackend.Log($"[WebGPU-Debug] Dynamic shared memory override: {overrideInfo.ConstantName} = {numElements}");
                }
            }

            // For explicitly grouped kernels (KernelConfig), the dispatch's GroupDim
            // specifies the number of threads per workgroup. WebGPU bakes this into the
            // shader as @workgroup_size at compile time, but the ILGPU algorithm kernels
            // (RadixSort, Scan, etc.) choose GroupDim at runtime based on device limits.
            // If the compiled @workgroup_size doesn't match the dispatch's GroupDim, patch
            // the WGSL source so @workgroup_size and the const workgroup_size variable agree
            // with the actual dispatch dimensions. The shader cache handles deduplication.
            string wgslSource = compiledKernel.WGSLSource;
            if (dimension is KernelConfig kcWg)
            {
                int reqX = kcWg.GroupDim.X;
                int reqY = kcWg.GroupDim.Y;
                int reqZ = kcWg.GroupDim.Z;

                var wgMatch = Regex.Match(
                    wgslSource, @"@workgroup_size\((\d+)(?:\s*,\s*(\d+))?(?:\s*,\s*(\d+))?\)");
                if (wgMatch.Success)
                {
                    int compiledX = int.Parse(wgMatch.Groups[1].Value);
                    int compiledY = wgMatch.Groups[2].Success ? int.Parse(wgMatch.Groups[2].Value) : 1;
                    int compiledZ = wgMatch.Groups[3].Success ? int.Parse(wgMatch.Groups[3].Value) : 1;

                    if (compiledX != reqX || compiledY != reqY || compiledZ != reqZ)
                    {
                        if (WebGPUBackend.VerboseLogging)
                            WebGPUBackend.Log($"[WebGPU] @workgroup_size mismatch: compiled=({compiledX},{compiledY},{compiledZ}), dispatch=({reqX},{reqY},{reqZ}). Patching WGSL.");

                        wgslSource = wgslSource.Replace(
                            wgMatch.Value,
                            $"@workgroup_size({reqX}, {reqY}, {reqZ})");

                        var constMatch = Regex.Match(
                            wgslSource, @"const workgroup_size : vec3<u32> = vec3<u32>\(\d+u, \d+u, \d+u\);");
                        if (constMatch.Success)
                        {
                            wgslSource = wgslSource.Replace(
                                constMatch.Value,
                                $"const workgroup_size : vec3<u32> = vec3<u32>({reqX}u, {reqY}u, {reqZ}u);");
                        }
                    }
                }
            }

            var shader = nativeAccel.GetOrCreateComputeShader(wgslSource, "main", overrideConstants);
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

                // Track element offsets per binding index for view offset packing.
                // When binding at offset=0 for sub-views, the element offset needs
                // to be packed into the scalar buffer so the WGSL can read it.
                var viewElementOffsets = new Dictionary<int, int>();
                // For packed-struct views, the element COUNT is also sent to the GPU since
                // arrayLength() returns CPU-allocation-size/4, not the logical element count.
                var viewElementCounts = new Dictionary<int, int>();

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

                // Detect if args[0] is a short Index type that should be skipped for bindings.
                // This happens with explicitly grouped stream kernels where KernelIndexParameterOffset=0
                // but the WGSL generator still skips the Index param (mapped to global_id.x).
                // Note: LongIndex types are NOT skipped — they are data counts for grid-stride kernels.
                int runtimeIndexSkip = 0;
                if (kernelParamOffset == 0 && paramTypes.Count > 0)
                {
                    var first = paramTypes[0];
                    if (first == typeof(Index1D) || first == typeof(Index2D) || first == typeof(Index3D))
                        runtimeIndexSkip = 1;
                }

                // The effective offset for mapping IR param indices to args[] indices:
                // effectiveOffset = kernelParamOffset + runtimeIndexSkip
                // For auto-grouped: effectiveOffset = 1 + 0 = 1
                // For stream with Index: effectiveOffset = 0 + 1 = 1
                // For stream without Index: effectiveOffset = 0 + 0 = 0
                int effectiveOffset = kernelParamOffset + runtimeIndexSkip;

                // --- PRE-EXPAND: Flatten body structs in args[] ---
                // IGridStrideKernelBody implementations (e.g. InitializerImplementation, ReductionImplementation)
                // contain IArrayView fields. The ILGPU compiler inlines these structs into separate IR
                // parameters, so the WGSL generator creates one binding per field. We must match this
                // at runtime by expanding the args array to replace each body struct with its fields.
                //
                // We also build argsToEffectiveOffset: for each original args[i], the starting index
                // in effectiveArgs where args[i]'s contribution begins. This lets Phase 2 (scalar packing)
                // correctly look up scalar values from effectiveArgs using IR param indices.
                _reusableExpandedArgs ??= new List<object?>();
                _reusableExpandedArgs.Clear();
                var expandedArgs = _reusableExpandedArgs;
                // argsToEffectiveOffset[i] = index in effectiveArgs where args[i] starts
                var argsToEffectiveOffset = new int[args.Length];
                for (int i = 0; i < args.Length; i++)
                {
                    argsToEffectiveOffset[i] = expandedArgs.Count;
                    var a = args[i];
                    if (a != null && a.GetType().IsValueType && !(a is IArrayView) && ContainsPointerFields(a.GetType()))
                    {
                        if (WebGPUBackend.VerboseLogging)
                            WebGPUBackend.Log($"[WebGPU-Debug] Pre-expand: Flattening body struct {a.GetType().Name} at args[{i}]");
                        expandedArgs.AddRange(FlattenStructFields(a));
                    }
                    else
                    {
                        expandedArgs.Add(a);
                    }
                }
                // Use expandedArgs list directly as indexable span to avoid ToArray()
                var effectiveArgsCount = expandedArgs.Count;

                // Build packed scalar lookup mapping effectiveArgs index → ScalarPackingEntry.
                //
                // For regular params: effectiveArgsIdx = paramIndex - kernelParamOffset
                // For body struct scalar fields: ScalarPackingManifest uses synthetic ParamIndex
                //   = bodyStructParamIndex * 1000 + irFieldIndex
                // We need to find the effectiveArgs index for the non-view backing field.
                //
                // Strategy: track which effectiveArgs indices are non-view backing fields of body structs.
                // bodyStructScalarEffectiveIdx[argsIdx][nthNonViewField] = effectiveArgsIdx
                _reusableBodyStructMap ??= new Dictionary<int, List<int>>();
                _reusableBodyStructMap.Clear();
                var bodyStructScalarEffectiveIdxMap = _reusableBodyStructMap;
                for (int i = 0; i < args.Length; i++)
                {
                    var a = args[i];
                    if (a != null && a.GetType().IsValueType && !(a is IArrayView) && ContainsPointerFields(a.GetType()))
                    {
                        // This is a body struct — track which effectiveArgs entries are non-view fields
                        var nonViewIndices = new List<int>();
                        int baseIdx = argsToEffectiveOffset[i];
                        var fields = a.GetType().GetFields(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                        int fieldEffIdx = baseIdx;
                        foreach (var field in fields)
                        {
                            var fieldVal = field.GetValue(a);
                            if (fieldVal is IArrayView)
                                fieldEffIdx++; // View field → buffer binding, not scalar
                            else
                            {
                                nonViewIndices.Add(fieldEffIdx); // Non-view field → scalar
                                fieldEffIdx++;
                            }
                        }
                        bodyStructScalarEffectiveIdxMap[i] = nonViewIndices;
                    }
                }

                _reusableBodyStructScalarSet ??= new HashSet<int>();
                _reusableBodyStructScalarSet.Clear();
                var allBodyStructScalarEffectiveIdxSet = _reusableBodyStructScalarSet;
                foreach (var kvp in bodyStructScalarEffectiveIdxMap)
                    foreach (var idx in kvp.Value)
                        allBodyStructScalarEffectiveIdxSet.Add(idx);

                _reusablePackedScalarLookup ??= new Dictionary<int, ScalarPackingEntry>();
                _reusablePackedScalarLookup.Clear();
                var packedScalarLookup = _reusablePackedScalarLookup;
                if (compiledKernel.HasScalarPacking)
                {
                    // Pre-compute total manifest entry count per body-struct param index.
                    // The WGSL generator may add scalar manifest entries for IR-level fields that
                    // live *inside* an IArrayView (e.g. Extent from ArrayView1D<long, Dense>).
                    // Those do NOT have corresponding C# runtime args — they are derived from
                    // arrayLength() in the WGSL itself. We must skip the leading "IR-only" entries
                    // and only map the trailing entries (actual user scalar fields) to runtime slots.
                    //
                    // IMPORTANT: The WGSL generator encodes the body-struct param index as
                    //   (param.Index + 1) * 1000 + fieldIndex
                    // so that grid-stride kernels with param.Index = 0 still produce synthetic
                    // values >= 1000 (distinguishable from real param indices).
                    // Decoding: bodyStructIRParamIdx = (entry.ParamIndex / 1000) - 1
                    var manifestCountPerBodyStructParam = new Dictionary<int, int>();
                    foreach (var e in compiledKernel.ScalarPackingManifest)
                    {
                        if (e.ParamIndex >= 1000 && !e.IsViewOffset && !e.IsViewCount)
                        {
                            int bsIdx = (e.ParamIndex / 1000) - 1; // true IR param index
                            manifestCountPerBodyStructParam.TryGetValue(bsIdx, out int c);
                            manifestCountPerBodyStructParam[bsIdx] = c + 1;
                        }
                    }

                    foreach (var entry in compiledKernel.ScalarPackingManifest)
                    {
                        if (entry.ParamIndex >= 1000)
                        {
                            // View offset/count entries for body struct fields are handled in a
                            // separate phase — don't try to map them to C# struct fields.
                            if (entry.IsViewOffset || entry.IsViewCount) continue;

                            // Synthetic param index: body struct scalar field
                            // Decoded: bodyStructIRParamIdx = (entry.ParamIndex / 1000) - 1
                            int bodyStructIRParamIdx = (entry.ParamIndex / 1000) - 1;
                            int bodyStructArgsIdx = bodyStructIRParamIdx - kernelParamOffset;

                            // Count how many non-view-offset scalar manifest entries for this
                            // body struct come before this one (by ByteOffset order).
                            int nthManifestEntry = 0;
                            foreach (var e2 in compiledKernel.ScalarPackingManifest)
                            {
                                if (e2.ParamIndex >= 1000 && !e2.IsViewOffset && !e2.IsViewCount && (e2.ParamIndex / 1000) - 1 == bodyStructIRParamIdx)
                                {
                                    if (e2.ByteOffset < entry.ByteOffset)
                                        nthManifestEntry++;
                                }
                            }

                            if (bodyStructScalarEffectiveIdxMap.TryGetValue(bodyStructArgsIdx, out var nonViewIdxList))
                            {
                                // Total manifest entries may exceed C# non-view fields because the WGSL
                                // generator also emits manifest entries for IR-level fields inside IArrayView
                                // (e.g., the Extent i32 from ArrayView1D<long, Dense>). These leading
                                // entries don't correspond to any runtime arg — skip them.
                                int totalManifest = manifestCountPerBodyStructParam.TryGetValue(bodyStructIRParamIdx, out int tc) ? tc : 0;
                                int extraIROnlyEntries = totalManifest - nonViewIdxList.Count;
                                int nthCSharpScalarField = nthManifestEntry - extraIROnlyEntries;

                                if (nthCSharpScalarField >= 0 && nthCSharpScalarField < nonViewIdxList.Count)
                                {
                                    int effectiveArgsIdx = nonViewIdxList[nthCSharpScalarField];
                                    packedScalarLookup[effectiveArgsIdx] = entry;
                                    if (WebGPUBackend.VerboseLogging)
                                        WebGPUBackend.Log($"[WebGPU-Debug] Body struct scalar: ParamIndex={entry.ParamIndex}, effectiveArgsIdx={effectiveArgsIdx}, nthManifest={nthManifestEntry}, nthCSharp={nthCSharpScalarField}");
                                }
                                else
                                {
                                    if (WebGPUBackend.VerboseLogging)
                                        WebGPUBackend.Log($"[WebGPU-Debug] Body struct scalar: ParamIndex={entry.ParamIndex}, SKIPPED (IR-only, nthManifest={nthManifestEntry}, extra={extraIROnlyEntries})");
                                }
                            }
                        }
                        else
                        {
                            // Regular param: map IR param index to effectiveArgs index
                            // Skip IsViewOffset/IsViewCount entries — they're packed separately after the scalar loop
                            if (entry.IsViewOffset || entry.IsViewCount) continue;
                            int effectiveArgsIdx = entry.ParamIndex - kernelParamOffset;
                            if (effectiveArgsIdx >= 0 && effectiveArgsIdx < effectiveArgsCount)
                                packedScalarLookup[effectiveArgsIdx] = entry;
                        }
                    }
                }

                // --- Phase 1: Emit bindings for non-packed params (views, structs, atomics) ---
                // effectiveArgs[] has body structs pre-expanded into their constituent fields.
                // Skip the index param (effectiveArgs[0..runtimeIndexSkip-1]) and packed scalars.
                for (int i = 0; i < effectiveArgsCount; i++)
                {
                    // Skip the Index type param at effectiveArgs[0] for explicitly grouped kernels
                    if (i < runtimeIndexSkip)
                        continue;

                    // Skip packed scalars — they'll be handled in Phase 2
                    if (packedScalarLookup.ContainsKey(i))
                        continue;

                    // Skip body-struct non-view scalar fields that have no manifest entry.
                    // These are shader-local variables (e.g., ReducedValue in ReductionImplementation)
                    // that the WGSL handles internally — they don't correspond to GPU buffer bindings.
                    // Without this check, they'd create spurious bindings and shift the scalar
                    // pack buffer to the wrong binding index.
                    if (allBodyStructScalarEffectiveIdxSet.Contains(i))
                        continue;

                    var arg = expandedArgs[i];

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

                        if (WebGPUBackend.VerboseLogging)
                            WebGPUBackend.Log($"[WebGPU-Debug] Arg {i}: Binding Buffer. Size={contiguous.LengthInBytes}, Offset={contiguous.IndexInBytes}");

                        // WebGPU requires storage buffer binding offsets to be aligned
                        // to minStorageBufferOffsetAlignment (256 bytes). ILGPU sub-views
                        // may have arbitrary element-aligned offsets (e.g., 4612 bytes).
                        //
                        // FIX: Round the byte offset DOWN to the nearest 256-byte boundary.
                        // The remainder (padding) becomes an element offset packed into
                        // _scalar_params so the WGSL reads the correct slice.
                        //
                        // This avoids aliasing: two sub-views of the same buffer at
                        // different 256-aligned positions get NON-OVERLAPPING binding ranges,
                        // satisfying WebGPU's writable storage buffer aliasing rules.
                        ulong rawOffset = (ulong)((long)contiguous.IndexInBytes);
                        ulong alignedOffset = rawOffset & ~255UL; // round DOWN to 256
                        ulong padding = rawOffset - alignedOffset;
                        ulong bindingSize = (ulong)WebGPUAlignment.AlignTo4((long)(padding + (ulong)contiguous.LengthInBytes));
                        // Clamp to actual GPU buffer size (which was allocated with AlignTo4)
                        ulong bufferSize = (ulong)WebGPUAlignment.AlignTo4(nativeBuffer.LengthInBytes);
                        if (alignedOffset + bindingSize > bufferSize)
                            bindingSize = bufferSize - alignedOffset;
                        
                        resource = new GPUBufferBinding
                        {
                            Buffer = gpuBuffer,
                            Offset = alignedOffset,
                            Size = bindingSize
                        };

                        // Record the u32 offset for this binding so Phase 2 can pack it.
                        // We store padding/4 (u32 units) rather than padding/elementSize (element units)
                        // because the WGSL formula is: base_idx = u32Offset + i * packed_stride
                        // This correctly handles packed structs where CPU element size != GPU packed size.
                        // For regular 4-byte views: u32Offset == elementOffset (no change in behavior).
                        // For emu_f64/i64 (8-byte CPU, 2 u32s): formulas are mathematically equivalent.
                        int elementOffset = (int)(padding / 4UL);
                        viewElementOffsets[currentBindingIndex] = elementOffset;
                        // Record the logical element count for packed-struct views.
                        // contiguous.Length gives the exact count regardless of CPU vs GPU element size.
                        viewElementCounts[currentBindingIndex] = (int)contiguous.Length;
                        
                        if (WebGPUBackend.VerboseLogging)
                            WebGPUBackend.Log($"[WebGPU-Debug] Arg {i}: rawOffset={rawOffset}, alignedOffset={alignedOffset}, padding={padding}, elemOffset={elementOffset}, bindingSize={bindingSize}");
                    }
                    else
                    {
                        // Non-packed, non-view param (struct scalar with own binding)
                        var size = 256;
                        var uBuffer = GetPooledScalarBuffer(device);
                        scalarBuffersToReturn.Add(uBuffer);

                        if (WebGPUBackend.VerboseLogging)
                            WebGPUBackend.Log($"[WebGPU-Debug] Arg {i}: Binding Struct Scalar. Value={arg}");

                        if (arg != null && arg.GetType().IsValueType)
                        {
                            Type argType = arg.GetType();

                            // Use Interop.SizeOf(Type) which handles generic structs via Unsafe.SizeOf<T>
                            // (Marshal.SizeOf fails on generic types like ReductionImplementation<T,S,R>).
                            int structSize = global::ILGPU.Interop.SizeOf(argType);
                            int paddedSize = (int)WebGPUAlignment.AlignTo4(structSize);
                            byte[] bytes = new byte[paddedSize];

                            // GCHandle.Alloc(Pinned) fails for boxed generic structs in Blazor WASM
                            // (ArgumentException_NotIsomorphic). Use our CopyStructToBytes<T> helper
                            // which uses MemoryMarshal.AsBytes — no GC pinning required.
                            GetCopyStructMethod(argType).Invoke(null, new object[] { arg, bytes });

                            device.Queue.WriteBuffer(uBuffer, 0, bytes);
                            if (WebGPUBackend.VerboseLogging)
                                WebGPUBackend.Log($"[WebGPU-Debug] Arg {i}: Struct scalar {argType.Name}, Size={structSize} bytes");
                        }
                        else throw new NotSupportedException($"Unsupported non-packed non-view argument type: {arg?.GetType()}");

                        resource = new GPUBufferBinding { Buffer = uBuffer, Offset = 0, Size = (ulong)size };
                    }

                    entries.Add(new GPUBindGroupEntry { Binding = (uint)currentBindingIndex, Resource = resource! });
                    currentBindingIndex++;

                    if (dims.Length > 1)
                    {
                        if (WebGPUBackend.VerboseLogging)
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

                    // Use packedScalarLookup (effectiveArgsIdx → entry) which correctly handles
                    // both regular params and body struct scalar fields (synthetic param indices).
                    foreach (var kvp in packedScalarLookup)
                    {
                        int effectiveArgsIdx = kvp.Key;
                        var entry = kvp.Value;
                        var arg = expandedArgs[effectiveArgsIdx];
                        int byteOffset = entry.ByteOffset;

                        if (WebGPUBackend.VerboseLogging)
                            WebGPUBackend.Log($"[WebGPU-Debug] Packing scalar param {entry.ParamIndex} (effectiveArgs[{effectiveArgsIdx}]) at byte offset {byteOffset}: {arg}");

                        // Unwrap SpecializedValue<T> to extract the inner T value
                        if (arg != null && arg.GetType().IsGenericType && 
                            arg.GetType().Name.StartsWith("SpecializedValue"))
                        {
                            arg = arg.GetType().GetProperty("Value")!.GetValue(arg);
                        }

                        if (arg is int iVal)
                            BitConverter.GetBytes(iVal).CopyTo(packedData, byteOffset);
                        else if (arg is float fVal)
                            BitConverter.GetBytes(fVal).CopyTo(packedData, byteOffset);
                        else if (arg is uint uiVal)
                            BitConverter.GetBytes(uiVal).CopyTo(packedData, byteOffset);
                        else if (arg is long lVal)
                        {
                            // emu_i64: pack as two u32 values (low word, high word)
                            BitConverter.GetBytes((uint)(lVal & 0xFFFFFFFFL)).CopyTo(packedData, byteOffset);
                            if (byteOffset + 4 < packedData.Length)
                                BitConverter.GetBytes((uint)((lVal >> 32) & 0xFFFFFFFFL)).CopyTo(packedData, byteOffset + 4);
                        }
                        else if (arg is ulong ulVal)
                        {
                            // emu_u64: pack as two u32 values (low word, high word)
                            BitConverter.GetBytes((uint)(ulVal & 0xFFFFFFFFL)).CopyTo(packedData, byteOffset);
                            if (byteOffset + 4 < packedData.Length)
                                BitConverter.GetBytes((uint)((ulVal >> 32) & 0xFFFFFFFFL)).CopyTo(packedData, byteOffset + 4);
                        }
                        else if (arg is LongIndex1D li1)
                        {
                            // LongIndex1D wraps a long — pack as emu_i64 (two u32 values)
                            long rawVal = li1.X;
                            BitConverter.GetBytes((uint)(rawVal & 0xFFFFFFFFL)).CopyTo(packedData, byteOffset);
                            if (byteOffset + 4 < packedData.Length)
                                BitConverter.GetBytes((uint)((rawVal >> 32) & 0xFFFFFFFFL)).CopyTo(packedData, byteOffset + 4);
                        }
                        else if (arg is LongIndex2D li2)
                        {
                            // LongIndex2D: pack X then Y as two emu_i64 pairs
                            BitConverter.GetBytes((uint)(li2.X & 0xFFFFFFFFL)).CopyTo(packedData, byteOffset);
                            if (byteOffset + 4 < packedData.Length)
                                BitConverter.GetBytes((uint)((li2.X >> 32) & 0xFFFFFFFFL)).CopyTo(packedData, byteOffset + 4);
                            if (byteOffset + 8 < packedData.Length)
                                BitConverter.GetBytes((uint)(li2.Y & 0xFFFFFFFFL)).CopyTo(packedData, byteOffset + 8);
                            if (byteOffset + 12 < packedData.Length)
                                BitConverter.GetBytes((uint)((li2.Y >> 32) & 0xFFFFFFFFL)).CopyTo(packedData, byteOffset + 12);
                        }
                        else if (arg is LongIndex3D li3)
                        {
                            // LongIndex3D: pack X, Y, Z as three emu_i64 pairs
                            BitConverter.GetBytes((uint)(li3.X & 0xFFFFFFFFL)).CopyTo(packedData, byteOffset);
                            if (byteOffset + 4 < packedData.Length)
                                BitConverter.GetBytes((uint)((li3.X >> 32) & 0xFFFFFFFFL)).CopyTo(packedData, byteOffset + 4);
                            if (byteOffset + 8 < packedData.Length)
                                BitConverter.GetBytes((uint)(li3.Y & 0xFFFFFFFFL)).CopyTo(packedData, byteOffset + 8);
                            if (byteOffset + 12 < packedData.Length)
                                BitConverter.GetBytes((uint)((li3.Y >> 32) & 0xFFFFFFFFL)).CopyTo(packedData, byteOffset + 12);
                            if (byteOffset + 16 < packedData.Length)
                                BitConverter.GetBytes((uint)(li3.Z & 0xFFFFFFFFL)).CopyTo(packedData, byteOffset + 16);
                            if (byteOffset + 20 < packedData.Length)
                                BitConverter.GetBytes((uint)((li3.Z >> 32) & 0xFFFFFFFFL)).CopyTo(packedData, byteOffset + 20);
                        }
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
                        else if (arg is global::ILGPU.Half hVal)
                        {
                            // Half occupies 2 bytes but is packed into a u32 slot.
                            // Place the raw f16 bits in the low 16 bits so the WGSL
                            // bitcast<vec2<f16>>(u32).x pattern extracts the correct value.
                            ushort rawBits = global::ILGPU.Interop.FloatAsInt(hVal);
                            BitConverter.GetBytes(rawBits).CopyTo(packedData, byteOffset);
                        }
                        else if (arg is byte bVal)
                            BitConverter.GetBytes((uint)bVal).CopyTo(packedData, byteOffset);
                        else if (arg is bool blVal)
                            BitConverter.GetBytes(blVal ? 1u : 0u).CopyTo(packedData, byteOffset);
                        else if (arg is Index1D idx1)
                            BitConverter.GetBytes(idx1.X).CopyTo(packedData, byteOffset);
                        else if (arg is Index2D idx2)
                        {
                            BitConverter.GetBytes(idx2.X).CopyTo(packedData, byteOffset);
                            if (byteOffset + 4 < packedData.Length)
                                BitConverter.GetBytes(idx2.Y).CopyTo(packedData, byteOffset + 4);
                        }
                        else if (arg is Index3D idx3)
                        {
                            BitConverter.GetBytes(idx3.X).CopyTo(packedData, byteOffset);
                            if (byteOffset + 4 < packedData.Length)
                                BitConverter.GetBytes(idx3.Y).CopyTo(packedData, byteOffset + 4);
                            if (byteOffset + 8 < packedData.Length)
                                BitConverter.GetBytes(idx3.Z).CopyTo(packedData, byteOffset + 8);
                        }
                        else if (arg != null && arg.GetType().IsValueType)
                        {
                            // Fallback for any other value type: serialize struct bytes directly
                            int structSize = global::ILGPU.Interop.SizeOf(arg.GetType());
                            byte[] bytes = new byte[structSize];
                            GetCopyStructMethod(arg.GetType()).Invoke(null, new object[] { arg, bytes });
                            int copyLen = Math.Min(structSize, packedData.Length - byteOffset);
                            Array.Copy(bytes, 0, packedData, byteOffset, copyLen);
                            if (WebGPUBackend.VerboseLogging)
                                WebGPUBackend.Log($"[WebGPU-Debug] Packed fallback struct {arg.GetType().Name}, {structSize} bytes at offset {byteOffset}");
                        }
                        else if (arg != null)
                            throw new NotSupportedException($"Unsupported packed scalar type: {arg.GetType()}");

                    }

                    // --- Pack view element offsets and counts ---
                    // For each IsViewOffset entry in the manifest, pack the element offset.
                    // For each IsViewCount entry, pack the true element count for packed-struct views.
                    foreach (var entry in manifest)
                    {
                        if (entry.IsViewOffset)
                        {
                            int byteOffset = entry.ByteOffset;
                            if (byteOffset + 4 > packedData.Length)
                            {
                                // Extend packedData if needed (view offsets may extend beyond user scalars)
                                var newData = new byte[byteOffset + 4];
                                Array.Copy(packedData, newData, packedData.Length);
                                packedData = newData;
                            }
                            int elemOffset = viewElementOffsets.TryGetValue(entry.ViewBindingIndex, out int eo) ? eo : 0;
                            BitConverter.GetBytes(elemOffset).CopyTo(packedData, byteOffset);
                            if (WebGPUBackend.VerboseLogging)
                                WebGPUBackend.Log($"[WebGPU-Debug] Packed view offset: binding={entry.ViewBindingIndex}, elemOffset={elemOffset}, byteOffset={byteOffset}");
                        }
                        else if (entry.IsViewCount)
                        {
                            int byteOffset = entry.ByteOffset;
                            if (byteOffset + 4 > packedData.Length)
                            {
                                var newData = new byte[byteOffset + 4];
                                Array.Copy(packedData, newData, packedData.Length);
                                packedData = newData;
                            }
                            int elemCount = viewElementCounts.TryGetValue(entry.ViewCountBindingIndex, out int ec) ? ec : 0;
                            BitConverter.GetBytes(elemCount).CopyTo(packedData, byteOffset);
                            if (WebGPUBackend.VerboseLogging)
                                WebGPUBackend.Log($"[WebGPU-Debug] Packed view count: binding={entry.ViewCountBindingIndex}, elemCount={elemCount}, byteOffset={byteOffset}");
                        }
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

                    if (WebGPUBackend.VerboseLogging)
                        WebGPUBackend.Log($"[WebGPU-Debug] Packed {packedScalarLookup.Count} scalars into 1 buffer ({totalBytes} bytes used, binding {currentBindingIndex - 1})");
                }


                // Validate/fix entry count to match WGSL layout (avoids "binding index N not present" errors)
                int expectedCount = compiledKernel.ExpectedBindingCount;
                // When WGSL declares only 1 binding (the _scalar_params buffer) but the runtime
                // built 2 entries [view, scalar], the view was folded into _scalar_params at
                // compile time. Only apply this workaround when expectedCount confirms the
                // layout truly has 1 binding — otherwise we'd drop legitimate view bindings.
                bool applyScalarOnlyWorkaround = entries.Count == 2
                    && compiledKernel.HasScalarPacking
                    && expectedCount == 1;
                if (applyScalarOnlyWorkaround)
                {
                    // Layout expects 1 binding (scalar-only) but runtime built 2 [view, scalar].
                    // This happens when WGSL packs a view's offset into _scalar_params but omits
                    // the view buffer (e.g. certain body-struct or length-only cases). Use only
                    // the scalar buffer at binding 0.
                    var scalarEntry = entries[1]; // Phase 2 adds scalar buffer after Phase 1's views
                    entries.Clear();
                    entries.Add(new GPUBindGroupEntry { Binding = 0, Resource = scalarEntry.Resource });
                    if (WebGPUBackend.VerboseLogging)
                        WebGPUBackend.Log($"[WebGPU-BindGroup] Workaround: using scalar-only bindings for kernel '{compiledKernel.Name}'");
                }
                else if (expectedCount > 0 && entries.Count > expectedCount)
                {
                    // The WGSL binding count is the source of truth. The runtime may create
                    // extra entries (e.g. Phase 2 scalar buffer for ArrayView extents that the
                    // WGSL handles via arrayLength()). Trim to match the WGSL layout.
                    if (WebGPUBackend.VerboseLogging)
                    {
                        var kernelName = compiledKernel.Name ?? "unknown";
                        Console.WriteLine($"[WebGPU-BindGroup] Trimming entries from {entries.Count} to {expectedCount} for kernel '{kernelName}' (HasScalarPacking={compiledKernel.HasScalarPacking})");
                    }
                    while (entries.Count > expectedCount)
                        entries.RemoveAt(entries.Count - 1);
                }
                else if (expectedCount > 0 && entries.Count < expectedCount)
                {
                    var kernelName = compiledKernel.Name ?? "unknown";
                    throw new InvalidOperationException(
                        $"[WebGPU] Bind group mismatch for kernel '{kernelName}': " +
                        $"layout expects {expectedCount} binding(s), but only {entries.Count} entries were built.");
                }

                if (WebGPUBackend.VerboseLogging)
                {
                    WebGPUBackend.Log($"[WebGPU-BindGroup] Creating bind group with {entries.Count} entries (expected={expectedCount}) for kernel: {compiledKernel.Name}");
                    for (int ei = 0; ei < entries.Count; ei++)
                        WebGPUBackend.Log($"  entries[{ei}]: binding={entries[ei].Binding}");
                }

                var bindGroupDesc = new GPUBindGroupDescriptor
                {
                    Layout = shader.BindGroupLayout!,
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

                if (WebGPUBackend.VerboseLogging)
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
            // Surface any GPU validation errors that occurred during dispatch
            NativeAccelerator.ThrowIfGpuErrors();
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
        /// Wraps an externally-owned <see cref="GPUBuffer"/> as an ILGPU <see cref="ArrayView{T}"/>.
        /// The buffer is NOT owned — it will not be destroyed when the returned view is no longer used.
        /// <para>
        /// Use this to pass an external GPU buffer (e.g. from ONNX Runtime Web) directly to ILGPU kernels
        /// without copying data. Both the external buffer and this accelerator must share the same GPUDevice.
        /// </para>
        /// </summary>
        /// <typeparam name="T">The element type to interpret the buffer as.</typeparam>
        /// <param name="externalBuffer">The externally-owned GPU buffer to wrap.</param>
        /// <param name="elementCount">Number of elements of type <typeparamref name="T"/> in the buffer.</param>
        /// <returns>
        /// An <see cref="ArrayView{T}"/> backed by the external buffer.
        /// The caller must ensure <paramref name="externalBuffer"/> remains valid for the duration of any kernel that uses this view.
        /// Dispose the returned <see cref="ExternalWebGPUMemoryBuffer"/> when done to release the non-owning wrapper.
        /// </returns>
        public (ArrayView<T> View, ExternalWebGPUMemoryBuffer Buffer) WrapExternalBuffer<T>(GPUBuffer externalBuffer, int elementCount)
            where T : unmanaged
        {
            int elementSize = System.Runtime.InteropServices.Marshal.SizeOf<T>();
            var memBuffer = new ExternalWebGPUMemoryBuffer(this, externalBuffer, elementCount, elementSize);
            // Build a typed ArrayView<T> over the full buffer (byte buffer reinterpreted as T)
            var rawView = memBuffer.AsRawArrayView();
            var typedView = rawView.Cast<T>();
            return (typedView, memBuffer);
        }

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