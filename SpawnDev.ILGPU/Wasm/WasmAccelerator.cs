// ---------------------------------------------------------------------------------------
//                               SpawnDev.ILGPU.Wasm
//                    WebAssembly Compute Backend for Blazor WebAssembly
//
// File: WasmAccelerator.cs
//
// Accelerator implementation that dispatches compiled Wasm kernels to Web Workers.
// Each worker instantiates the Wasm module with shared memory and runs the kernel.
// ---------------------------------------------------------------------------------------

using global::ILGPU;
using global::ILGPU.Backends;
using global::ILGPU.Runtime;
using SpawnDev.BlazorJS;
using SpawnDev.BlazorJS.JSObjects;
using SpawnDev.ILGPU.Wasm.Backend;
using System.Collections.Concurrent;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;

namespace SpawnDev.ILGPU.Wasm
{
    /// <summary>
    /// Wasm accelerator implementation for ILGPU.
    /// Compiles kernels to WebAssembly binary and executes them on the main thread (Phase 1).
    /// </summary>
    public class WasmAccelerator : KernelAccelerator<WasmCompiledKernel, WasmKernel>
    {
        // Pending async work (kernel dispatches)
        internal readonly List<Task> _pendingWork = new();
        public static string? _lastImplicitIndexDebug;

        // Worker count for parallel dispatch
        private int _workerCount = 4;

        /// <summary>
        /// Reusable worker pool — lazily initialized on first dispatch.
        /// Workers are created once and reused across kernel dispatches.
        /// </summary>
        private WorkerPool? _workerPool;

        /// <summary>
        /// Tracks which pool workers have already received and cached
        /// the compiled Wasm module, so we don't re-send wasmBytes.
        /// </summary>
        private readonly HashSet<int> _initializedWorkers = new();

        /// <summary>
        /// Reference to the last wasmBytes sent to workers.
        /// When a different kernel is dispatched, we clear _initializedWorkers.
        /// </summary>
        private byte[]? _lastWasmBytes;

        /// <summary>
        /// Cached WebAssembly.Memory to avoid per-frame allocation.
        /// SharedArrayBuffer-backed memories reserve large virtual address space,
        /// so creating a new one every frame causes OOM.
        /// </summary>
        private JSObject? _cachedWasmMemory;
        private SharedArrayBuffer? _cachedMemoryBuffer;
        private int _cachedWasmPages;

        /// <summary>
        /// Counts how many RunKernelAsync dispatches are currently active.
        /// Incremented BEFORE the async task starts so the count is visible
        /// during the synchronous phase of each task.
        /// </summary>
        internal int _activeDispatchCount;

        // --- Reflection caches for hot-path stride extraction and struct marshaling ---
        private static readonly ConcurrentDictionary<Type, StrideReflectionCache> _strideCache = new();
        private static readonly ConcurrentDictionary<Type, MethodInfo> _unsafeWriteCache = new();

        private sealed class StrideReflectionCache
        {
            public PropertyInfo? StrideProp;
            public PropertyInfo? YStrideProp;
            public PropertyInfo? XStrideProp;
            public PropertyInfo? ZStrideProp;
        }

        private static StrideReflectionCache GetOrCreateStrideCache(Type argType, Type strideType)
        {
            // Two-level cache: argType → StrideProp, strideType → YStride/XStride/ZStride
            var argCache = _strideCache.GetOrAdd(argType, t => new StrideReflectionCache
            {
                StrideProp = t.GetProperty("Stride"),
            });
            if (argCache.StrideProp != null && argCache.YStrideProp == null && argCache.XStrideProp == null)
            {
                // Populate stride sub-properties from the stride object's type (done once per stride type)
                argCache.YStrideProp = strideType.GetProperty("YStride");
                argCache.XStrideProp = strideType.GetProperty("XStride");
                argCache.ZStrideProp = strideType.GetProperty("ZStride");
            }
            return argCache;
        }

        // Backend instance
        public WasmBackend Backend { get; private set; } = null!;

        /// <summary>
        /// Method info for the static RunKernel method used by dynamic kernel launchers.
        /// </summary>
        public static readonly MethodInfo RunKernelMethod = typeof(WasmAccelerator).GetMethod(
            nameof(RunKernel),
            BindingFlags.Public | BindingFlags.Static)!;

        #region Construction

        private WasmAccelerator(Context context, Device device) : base(context, device) { }

        /// <summary>
        /// Creates a new Wasm accelerator.
        /// </summary>
        public static async Task<WasmAccelerator> Create(Context context)
        {
            return await Create(context, new WasmBackendOptions());
        }

        /// <summary>
        /// Creates a new Wasm accelerator with options.
        /// </summary>
        public static async Task<WasmAccelerator> Create(Context context, WasmBackendOptions options)
        {
            var device = new WasmILGPUDevice();
            var accelerator = new WasmAccelerator(context, device);
            accelerator._workerCount = options?.WorkerCount ?? 4;
            var backend = new WasmBackend(context, options ?? new WasmBackendOptions());
            accelerator.Backend = backend;
            accelerator.Init(backend);
            accelerator.DefaultStream = accelerator.CreateStreamInternal();
            return accelerator;
        }

        #endregion

        #region Kernel Management

        /// <inheritdoc/>
        protected override WasmKernel CreateKernel(WasmCompiledKernel compiledKernel)
        {
            return new WasmKernel(this, compiledKernel, default!);
        }

        /// <inheritdoc/>
        protected override WasmKernel CreateKernel(WasmCompiledKernel compiledKernel, MethodInfo launcher)
        {
            return new WasmKernel(this, compiledKernel, launcher);
        }

        /// <inheritdoc/>
        protected override MethodInfo GenerateKernelLauncherMethod(
            WasmCompiledKernel wasmKernel, int customGroupSize)
        {
            // Build method signature: (Kernel, AcceleratorStream, indexType, param0, param1, ...)
            // This matches the pattern that Kernel.CreateLauncherDelegate binds:
            //   Delegate.CreateDelegate(typeof(TDelegate), kernel, launcher)
            // where 'kernel' becomes the first arg.
            var parameters = wasmKernel.EntryPoint.Parameters;
            var indexType = wasmKernel.EntryPoint.KernelIndexType;
            var argTypes = new List<Type> { typeof(Kernel), typeof(AcceleratorStream), indexType };
            for (int i = 0; i < parameters.Count; i++) argTypes.Add(parameters[i]);

            var dynamicMethod = new DynamicMethod(
                "WasmLauncher",
                typeof(void),
                argTypes.ToArray(),
                typeof(WasmAccelerator).Module);

            var ilGenerator = dynamicMethod.GetILGenerator();
            var argsLocal = ilGenerator.DeclareLocal(typeof(object[]));

            // Create object[] for kernel arguments
            ilGenerator.Emit(OpCodes.Ldc_I4, parameters.Count);
            ilGenerator.Emit(OpCodes.Newarr, typeof(object));
            ilGenerator.Emit(OpCodes.Stloc, argsLocal);

            // Pack each argument
            for (int i = 0; i < parameters.Count; i++)
            {
                ilGenerator.Emit(OpCodes.Ldloc, argsLocal);
                ilGenerator.Emit(OpCodes.Ldc_I4, i);
                ilGenerator.Emit(OpCodes.Ldarg, i + 3); // Skip Kernel, AcceleratorStream, dimension
                var paramType = parameters[i];
                if (paramType.IsValueType) ilGenerator.Emit(OpCodes.Box, paramType);
                ilGenerator.Emit(OpCodes.Stelem_Ref);
            }

            // Call RunKernel(kernel, stream, dimension, args)
            ilGenerator.Emit(OpCodes.Ldarg_0); // kernel
            ilGenerator.Emit(OpCodes.Ldarg_1); // stream
            ilGenerator.Emit(OpCodes.Ldarg_2); // dimension (index)
            if (indexType.IsValueType) ilGenerator.Emit(OpCodes.Box, indexType);
            ilGenerator.Emit(OpCodes.Ldloc, argsLocal);
            ilGenerator.EmitCall(OpCodes.Call, RunKernelMethod, null);
            ilGenerator.Emit(OpCodes.Ret);

            return dynamicMethod;
        }

        #endregion

        #region Kernel Execution

        /// <summary>
        /// Executes a Wasm kernel. Called by the dynamic launcher.
        /// </summary>
        /// <summary>
        /// Recent dispatch log for diagnostics. Captures last N kernel dispatches.
        /// </summary>
        public static string _dispatchLog = "";
        public static int _dispatchCount = 0;

        public static void RunKernel(Kernel kernel, AcceleratorStream stream, object dimension, object[] args)
        {
            var wasmAccel = (WasmAccelerator)kernel.Accelerator;
            var wasmKernel = (WasmKernel)kernel;
            var compiledKernel = (WasmCompiledKernel)wasmKernel.CompiledKernel;

            int dispNum = ++_dispatchCount;

            if (WasmBackend.VerboseLogging) WasmBackend.Log($"[Wasm-Debug] RunKernel called with {args.Length} args");

            // Increment active dispatch count BEFORE starting the async task,
            // so the count is visible during the task's synchronous execution phase.
            wasmAccel._activeDispatchCount++;
            var task = wasmAccel.RunKernelAsync(compiledKernel, dimension, args, dispNum);
            wasmAccel._pendingWork.Add(task);
        }

        private async Task RunKernelAsync(
            WasmCompiledKernel compiledKernel,
            object dimension,
            object[] args,
            int dispNum = 0)
        {
            // Serialize kernel execution: wait for all previous dispatches to complete
            // before starting a new one. Prevents data races in multi-kernel algorithms.
            if (_pendingWork.Count > 0)
            {
                var pending = _pendingWork.ToArray();
                _pendingWork.Clear();
                await Task.WhenAll(pending);
            }

            try
            {
                var js = BlazorJSRuntime.JS;
                var (gridDimX, gridDimY, gridDimZ) = GetGridDimensions(dimension);
                int totalItems = gridDimX * gridDimY * gridDimZ;

                // Extract actual group size for barrier kernels
                int groupSize = GetGroupSize(dimension, compiledKernel);
                int numGroups = compiledKernel.HasBarriers && groupSize > 1
                    ? Math.Max(1, (totalItems + groupSize - 1) / groupSize)
                    : totalItems;

                // Extract dynamic shared memory size from KernelConfig
                int dynamicSharedElements = 0;
                if (dimension is KernelConfig kConfig)
                    dynamicSharedElements = kConfig.SharedMemoryConfig.NumElements;

                if (WasmBackend.VerboseLogging) WasmBackend.Log($"[Wasm] Dispatching kernel: {totalItems} items ({gridDimX}x{gridDimY}x{gridDimZ}), groupSize={groupSize}, numGroups={numGroups}, hasBarriers={compiledKernel.HasBarriers}, dynamicSharedElements={dynamicSharedElements}");

                // Determine isView flags from compiled kernel metadata
                var paramInfos = compiledKernel.ParamInfos;

                // Skip the implicit extent argument (args[0]) when it's an Index or LongIndex value.
                // These are automatically added by ILGPU's kernel launcher and correspond to the
                // kernel's implicit index parameter (which is mapped to globalIdx in the Wasm func).
                // Skip the implicit extent argument (args[0]) only when it's an Index value.
                // Index types represent the kernel's implicit extent parameter which is
                // mapped to _globalIdxLocal in the kernel. LongIndex types are NOT skipped
                // because they represent data like paddedNumDataElements in GridStrideLoopKernel
                // which is used as the loop bound (different from globalIdx).
                bool hasImplicitIndex = args.Length > 0
                    && (args[0] is Index1D || args[0] is Index2D || args[0] is Index3D);
                if (_dispatchCount <= 4)
                    _lastImplicitIndexDebug = $"[D{_dispatchCount}] hasImpl={hasImplicitIndex}, dim={dimension}, argCnt={args.Length}, piCnt={paramInfos.Count}";
                else
                    _lastImplicitIndexDebug = $"[D{_dispatchCount}]";
                int argOffset = hasImplicitIndex ? 1 : 0;

                // Collect UNIQUE buffers (dedup SubViews of the same buffer) and
                // track per-view SubView byte offsets within the buffer.
                var uniqueBuffers = new Dictionary<WasmMemoryBuffer, int>(); // buffer → index
                var bufferInfos = new List<(WasmMemoryBuffer buffer, int byteOffset)>();
                var viewBufferIdx = new List<int>();   // per-view: which buffer in bufferInfos
                var viewSubOffsets = new List<int>();   // per-view: SubView byte offset
                var wasmArgs = new List<(bool isBuffer, WasmMemoryBuffer? buffer, int length, int stride, int stride2, object? value)>();

                for (int i = argOffset; i < args.Length; i++)
                {
                    int paramIdx = i - argOffset;
                    bool isView = paramIdx < paramInfos.Count && paramInfos[paramIdx].IsView;

                    if (isView && args[i] is IArrayView iav)
                    {
                        // ArrayView<T> is a value struct that implements IArrayView.
                        // Extract the underlying MemoryBuffer via .Buffer property.
                        var wasmBuf = iav.Buffer as WasmMemoryBuffer;
                        if (wasmBuf != null)
                        {
                            // Deduplicate: add each unique buffer only once
                            if (!uniqueBuffers.TryGetValue(wasmBuf, out int bufIdx))
                            {
                                bufIdx = bufferInfos.Count;
                                uniqueBuffers[wasmBuf] = bufIdx;
                                bufferInfos.Add((wasmBuf, 0));
                            }
                            viewBufferIdx.Add(bufIdx);

                            // Compute SubView byte offset within the buffer
                            int subViewByteOffset = 0;
                            try
                            {
                                var viewType = args[i].GetType();
                                var baseProp = viewType.GetProperty("BaseView");
                                object viewObj = baseProp != null ? baseProp.GetValue(args[i])! : args[i];
                                var indexProp = viewObj.GetType().GetProperty("Index",
                                    BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                                if (indexProp != null)
                                {
                                    var idx = indexProp.GetValue(viewObj);
                                    if (idx is long longIdx)
                                        subViewByteOffset = (int)(longIdx * iav.Buffer.ElementSize);
                                }
                            }
                            catch { }
                            viewSubOffsets.Add(subViewByteOffset);

                            // Log buffer identity for dispatch debugging
                            if (dispNum >= 2 && dispNum <= 20)
                                _dispatchLog += $"|D{dispNum}V{i}:buf={wasmBuf.GetHashCode()%1000},sub={subViewByteOffset},len={iav.Length}";

                            // Extract stride via cached reflection for multi-dimensional views
                            int stride = 1;
                            int stride2 = 0;
                            var argType = args[i].GetType();
                            var argCache = _strideCache.GetOrAdd(argType, t => new StrideReflectionCache
                            {
                                StrideProp = t.GetProperty("Stride"),
                            });
                            if (argCache.StrideProp != null)
                            {
                                var strideObj = argCache.StrideProp.GetValue(args[i]);
                                if (strideObj != null)
                                {
                                    var strideType = strideObj.GetType();
                                    // Lazily populate stride sub-properties (once per type)
                                    if (argCache.YStrideProp == null && argCache.XStrideProp == null)
                                    {
                                        argCache.YStrideProp = strideType.GetProperty("YStride");
                                        argCache.XStrideProp = strideType.GetProperty("XStride");
                                        argCache.ZStrideProp = strideType.GetProperty("ZStride");
                                    }

                                    if (argCache.YStrideProp != null)
                                        stride = (int)argCache.YStrideProp.GetValue(strideObj)!;
                                    else if (argCache.XStrideProp != null)
                                        stride = (int)argCache.XStrideProp.GetValue(strideObj)!;

                                    if (argCache.ZStrideProp != null)
                                        stride2 = (int)argCache.ZStrideProp.GetValue(strideObj)!;
                                }
                            }
                            if (WasmBackend.VerboseLogging) WasmBackend.Log($"[Wasm] View arg[{i}]: length={iav.Length}, stride={stride}, stride2={stride2}");
                            wasmArgs.Add((true, wasmBuf, (int)iav.Length, stride, stride2, null));
                        }
                        else
                        {
                            wasmArgs.Add((false, null, 0, 0, 0, 0));
                        }
                    }
                    else
                    {
                        // Check if this is a struct with embedded views — decompose it
                        bool decomposed = false;
                        if (args[i] != null && ((args[i].GetType().IsValueType
                            && !args[i].GetType().IsPrimitive && !args[i].GetType().IsEnum)
                            || args[i].GetType().IsDefined(typeof(System.Runtime.CompilerServices.CompilerGeneratedAttribute), false)))
                        {
                            var structFields = args[i].GetType().GetFields(
                                BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                            bool hasViews = false;
                            foreach (var sf in structFields)
                            {
                                try { if (sf.GetValue(args[i]) is IArrayView) { hasViews = true; break; } }
                                catch { }
                            }

                            if (hasViews)
                            {
                                // Struct with embedded views: extract their buffers for
                                // copy-in/copy-out. The struct itself is serialized to scratch
                                // (NOT decomposed — ILGPU keeps it as a single param).
                                // NativePtr patching ensures the view pointer in the serialized
                                // bytes points to the correct Wasm memory offset.
                                ExtractBuffersFromStruct(args[i], uniqueBuffers, bufferInfos);
                            }
                        }

                        if (!decomposed)
                        {
                            // Unwrap LongIndex types to their underlying long value
                            object arg = args[i];
                            if (arg is LongIndex1D li1) arg = li1.X;
                            else if (arg is LongIndex2D li2) arg = li2.X; // TODO: handle Y
                            else if (arg is LongIndex3D li3) arg = li3.X; // TODO: handle Y,Z
                            wasmArgs.Add((false, null, 0, 0, 0, arg));
                        }
                    }
                }

                // Calculate total memory needed for all buffers
                int totalMemoryBytes = 0;
                var bufferOffsets = new List<int>();

                foreach (var (buf, _) in bufferInfos)
                {
                    totalMemoryBytes = (totalMemoryBytes + 7) & ~7; // 8-byte align
                    bufferOffsets.Add(totalMemoryBytes);
                    totalMemoryBytes += (int)buf.LengthInBytes;
                }
                // Scratch memory for struct construction (after all buffers).
                // For barrier kernels, each worker needs its own scratch to avoid races.
                int scratchPerThread = Math.Max(compiledKernel.ScratchPerThread, 64); // min 64 bytes
                int scratchBase = (totalMemoryBytes + 7) & ~7;
                int scratchSize = compiledKernel.HasBarriers
                    ? scratchPerThread * groupSize  // per-thread scratch for barrier kernels
                    : 4096;                          // shared scratch for non-barrier kernels
                int afterScratch = scratchBase + scratchSize;

                // Shared memory region (for barrier kernels)
                int sharedMemBase = (afterScratch + 7) & ~7; // 8-byte align
                int sharedMemSize = compiledKernel.SharedMemorySize;
                // Add dynamic shared memory bytes (element count * element size)
                int dynamicSharedBytes = dynamicSharedElements * Math.Max(compiledKernel.DynamicSharedElementSize, 1);
                if (dynamicSharedElements > 0)
                {
                    sharedMemSize += dynamicSharedBytes;
                    if (WasmBackend.VerboseLogging) WasmBackend.Log($"[Wasm] Dynamic shared memory: {dynamicSharedElements} elements x {compiledKernel.DynamicSharedElementSize} bytes = {dynamicSharedBytes} bytes, total shared={sharedMemSize}");
                }
                int afterShared = sharedMemBase + sharedMemSize;

                // Barrier slots region: each barrier = 8 bytes (arrival counter i32 + sense flag i32)
                int barrierBase = (afterShared + 3) & ~3; // 4-byte align
                int barrierSize = compiledKernel.BarrierCount * 8;
                // Add 16 bytes for two inter-worker barriers (phase + group).
                // Each barrier = 2 × i32 (counter + generation).
                // Phase barrier: syncs workers between phases within a group.
                // Group barrier: syncs workers between group iterations.
                int fenceSlot = barrierBase + barrierSize;
                int totalWithBarriers = fenceSlot + 16;

                if (WasmBackend.VerboseLogging) WasmBackend.Log($"[Wasm] Memory layout: buffers={totalMemoryBytes}, scratch={scratchBase}, sharedMem={sharedMemBase}({sharedMemSize}), barrier={barrierBase}({barrierSize}), hasBarriers={compiledKernel.HasBarriers}, groupSize={groupSize}");

                // Round up to Wasm page size (64KB)
                int wasmPages = Math.Max(1, (totalWithBarriers + 65535) / 65536);

                // Determine whether we can reuse the cached memory.
                // If there are other pending kernel dispatches running concurrently,
                // they share the same SharedArrayBuffer and we'd corrupt their data.
                // In that case, create a dedicated per-dispatch memory.
                bool hasConcurrentWork = _activeDispatchCount > 1;
                JSObject wasmMemory;
                SharedArrayBuffer memoryBuffer;
                JSObject? disposeWasmMemory = null;   // track per-dispatch memory for cleanup
                SharedArrayBuffer? disposeBuffer = null;

                if (!hasConcurrentWork && _cachedWasmMemory != null && wasmPages <= _cachedWasmPages)
                {
                    // Reuse cached memory — fast path for render loops (same kernel, no overlap)
                    wasmMemory = _cachedWasmMemory;
                    memoryBuffer = _cachedMemoryBuffer!;
                }
                else if (!hasConcurrentWork)
                {
                    // No concurrent work, but need bigger memory — update cache
                    _cachedMemoryBuffer?.Dispose();
                    _cachedWasmMemory?.Dispose();
                    _cachedWasmPages = wasmPages;
                    _cachedWasmMemory = js.Call<JSObject>(
                        "eval",
                        $"new WebAssembly.Memory({{ initial: {wasmPages}, maximum: 2048, shared: true }})");
                    _cachedMemoryBuffer = _cachedWasmMemory.JSRef!.Get<SharedArrayBuffer>("buffer");
                    _initializedWorkers.Clear();
                    wasmMemory = _cachedWasmMemory;
                    memoryBuffer = _cachedMemoryBuffer;
                }
                else
                {
                    // Concurrent dispatches — create isolated per-dispatch memory
                    disposeWasmMemory = js.Call<JSObject>(
                        "eval",
                        $"new WebAssembly.Memory({{ initial: {wasmPages}, maximum: 2048, shared: true }})");
                    disposeBuffer = disposeWasmMemory.JSRef!.Get<SharedArrayBuffer>("buffer");
                    wasmMemory = disposeWasmMemory;
                    memoryBuffer = disposeBuffer;
                }

                // Zero out shared memory and barrier regions to prevent stale data.
                // Zero scratch + shared + barrier regions to prevent stale data from
                // previous dispatches affecting the current dispatch (phase state, shared
                // memory, barrier counters). Buffer regions are overwritten by copy-in.
                int zeroStart = scratchBase;
                int zeroEnd = totalWithBarriers;
                if (zeroEnd > zeroStart)
                {
                    using var zeroView = new Uint8Array(memoryBuffer, zeroStart, zeroEnd - zeroStart);
                    zeroView.JSRef!.CallVoid("fill", 0);
                }

                // Copy buffer data into the Wasm linear memory at computed offsets
                for (int i = 0; i < bufferInfos.Count; i++)
                {
                    var (buf, _) = bufferInfos[i];
                    int offset = bufferOffsets[i];

                    using var srcView = new Uint8Array(buf.SharedBuffer);
                    using var dstView = new Uint8Array(memoryBuffer, offset, (int)buf.LengthInBytes);
                    dstView.JSRef!.CallVoid("set", srcView);
                }

                // Debug: check buf.SharedBuffer and Wasm memory after copy-in
                if (dispNum >= 6 && dispNum <= 8 && bufferInfos.Count > 0)
                {
                    // What's in the buffer's SharedArrayBuffer?
                    using var sabView = new Uint8Array(bufferInfos[0].buffer.SharedBuffer, 0, 4);
                    int sabVal = BitConverter.ToInt32(sabView.ReadBytes());
                    // What's in Wasm memory after copy-in?
                    using var wmView = new Uint8Array(memoryBuffer, bufferOffsets[0], 4);
                    int wmVal = BitConverter.ToInt32(wmView.ReadBytes());
                    _dispatchLog += $"|D{dispNum}pre:sab={sabVal},wm={wmVal}";
                }
                _lastImplicitIndexDebug += $" | bufInfoCnt={bufferInfos.Count} bufOffCnt={bufferOffsets.Count}";
                // CRITICAL: Set each buffer's NativePtr to its Wasm memory offset.
                // When struct parameters contain ArrayViews, the struct serialization
                // (Unsafe.Write) copies the NativePtr value into the byte array. The
                // kernel reads this as the buffer's address. With NativePtr=0, the kernel
                // would write to address 0 instead of the buffer's actual Wasm position.
                // We restore NativePtr to 0 after serialization.
                for (int i = 0; i < bufferInfos.Count; i++)
                {
                    var (buf, _) = bufferInfos[i];
                    buf.NativePtr = (IntPtr)bufferOffsets[i];
                }

                // Build flat argument list
                // Track struct scalar args that need to be written into scratch memory
                var structScratchWrites = new List<(int scratchOffset, byte[] bytes)>();
                int scratchCursor = 0; // offset within scratch region

                var flatArgs = new List<string>();
                int viewIndex = 0; // tracks views for SubView offset lookup
                int wasmArgIdx = 0; // tracks current wasmArgs index for IR param lookup
                foreach (var (isBuffer, buffer, length, stride, stride2, value) in wasmArgs)
                {
                    if (isBuffer)
                    {
                        // Compute the kernel's byte address for this view:
                        // = buffer's Wasm memory base + SubView byte offset within buffer
                        int bufIdx = viewBufferIdx[viewIndex];
                        int viewOffset = bufferOffsets[bufIdx] + viewSubOffsets[viewIndex];
                        flatArgs.Add(viewOffset.ToString());
                        flatArgs.Add(length.ToString());
                        if (stride != -1) // -1 = skip (struct-embedded views)
                            flatArgs.Add(stride.ToString());
                        if (stride2 != -1)
                            flatArgs.Add(stride2.ToString());
                        viewIndex++;
                    }
                    else
                    {
                        if (value is float fv) flatArgs.Add(fv.ToString("G9"));
                        else if (value is double dv) flatArgs.Add(dv.ToString("G17"));
                        else if (value is long lv) flatArgs.Add($"{lv}n"); // BigInt for i64
                        else if (value is ulong ulv) flatArgs.Add($"{ulv}n"); // BigInt for i64
                        else if (value != null && !value.GetType().IsPrimitive && !value.GetType().IsEnum
                            && (value.GetType().IsValueType
                                || value.GetType().IsDefined(typeof(System.Runtime.CompilerServices.CompilerGeneratedAttribute), false)))
                        {
                            // Check if the IR treats this as a scalar (e.g., SpecializedValue<int>
                            // is a struct wrapping an int, but the IR lowers it to PrimitiveType).
                            // In that case, unwrap to the inner value instead of serializing to scratch.
                            int irIdx = wasmArgIdx + argOffset;
                            var irParam = (irIdx < paramInfos.Count) ? paramInfos[irIdx] : null;
                            if (irParam != null && irParam.IsScalar && irParam.StructFields == null && irParam.StructSize == 0)
                            {
                                // IR expects a scalar, but CLR has a wrapper struct. Extract first field.
                                var innerFields = value.GetType().GetFields(
                                    BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                                object innerVal = value;
                                // Unwrap nested wrappers until we get a primitive
                                while (innerFields.Length > 0 && !innerVal.GetType().IsPrimitive)
                                {
                                    innerVal = innerFields[0].GetValue(innerVal);
                                    if (innerVal == null) break;
                                    innerFields = innerVal.GetType().GetFields(
                                        BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                                }
                                if (innerVal != null && innerVal.GetType().IsPrimitive)
                                {
                                    if (innerVal is float fvi) flatArgs.Add(fvi.ToString("G9"));
                                    else if (innerVal is double dvi) flatArgs.Add(dvi.ToString("G17"));
                                    else if (innerVal is long lvi) flatArgs.Add($"{lvi}n");
                                    else if (innerVal is ulong ulvi) flatArgs.Add($"{ulvi}n");
                                    else flatArgs.Add(innerVal.ToString() ?? "0");
                                    wasmArgIdx++;
                                    continue;
                                }
                            }

                            // Look up the IR struct layout from the compiled kernel's param info.
                            // This tells us the exact byte offset and type of each leaf field
                            // as the kernel expects them in scratch memory.
                            int irParamIdx = wasmArgIdx + argOffset; // adjust for implicit index skip
                            var irLayout = (irParamIdx < paramInfos.Count) ? paramInfos[irParamIdx].StructFields : null;
                            int irStructSize = (irParamIdx < paramInfos.Count) ? paramInfos[irParamIdx].StructSize : 0;
                            var piEntry = (irParamIdx < paramInfos.Count) ? paramInfos[irParamIdx] : null;
                            _lastImplicitIndexDebug += $" | STRUCT: idx={wasmArgIdx},irIdx={irParamIdx},piCnt={paramInfos.Count},hasLayout={irLayout != null},irSize={irStructSize},piIsView={piEntry?.IsView},piIsScalar={piEntry?.IsScalar},piName={piEntry?.Name},irType={piEntry?.IRTypeName}";

                            byte[] bytes;
                            int structSize;
                            if (irLayout != null && irLayout.Count > 0 && irStructSize > 0)
                            {
                                // Manual serialization using IR layout.
                                // Flatten the CLR struct depth-first to get primitive values
                                // in the same order as the IR's flattened fields.
                                structSize = irStructSize;
                                bytes = new byte[structSize];
                                var flatValues = new List<object>();
                                FlattenCLRStruct(value, flatValues);

                                _lastImplicitIndexDebug += $" | IRLayout: fields={irLayout.Count}, size={structSize}, clrFlat={flatValues.Count}";
                                for (int fi = 0; fi < irLayout.Count && fi < flatValues.Count; fi++)
                                {
                                    var field = irLayout[fi];
                                    var fieldVal = flatValues[fi];
                                    _lastImplicitIndexDebug += $" | f{fi}:off={field.Offset},t=0x{field.WasmType:X2},vp={field.IsViewPtr},v={fieldVal?.GetType().Name}";

                                    if (field.IsViewPtr)
                                    {
                                        // View pointer: write the Wasm buffer offset
                                        if (fieldVal is IArrayView iavInner)
                                        {
                                            var wasmBuf = iavInner.Buffer as WasmMemoryBuffer;
                                            if (wasmBuf != null && uniqueBuffers.TryGetValue(wasmBuf, out int bufIdxInner))
                                            {
                                                int wasmOffset = bufferOffsets[bufIdxInner];
                                                // Get SubView byte offset from view's Index
                                                int subOffset = 0;
                                                try
                                                {
                                                    var viewType = fieldVal.GetType();
                                                    var baseProp = viewType.GetProperty("BaseView");
                                                    object viewObj = baseProp != null ? baseProp.GetValue(fieldVal)! : fieldVal;
                                                    var indexProp = viewObj.GetType().GetProperty("Index",
                                                        BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                                                    if (indexProp?.GetValue(viewObj) is long idx)
                                                        subOffset = (int)(idx * iavInner.Buffer.ElementSize);
                                                }
                                                catch { }
                                                BitConverter.TryWriteBytes(bytes.AsSpan(field.Offset), wasmOffset + subOffset);
                                            }
                                        }
                                        else
                                        {
                                            // Fallback: write 0
                                            BitConverter.TryWriteBytes(bytes.AsSpan(field.Offset), 0);
                                        }
                                    }
                                    else
                                    {
                                        // Primitive value: write at the IR offset
                                        WritePrimitiveToBytes(bytes, field.Offset, fieldVal, field.WasmType, field.Size);
                                    }
                                }
                            }
                            else if (value.GetType().IsValueType)
                            {
                                // Fallback: use Unsafe.Write (CLR layout).
                                // This works for VALUE TYPE structs WITHOUT views.
                                structSize = global::ILGPU.Interop.SizeOf(value.GetType());
                                bytes = new byte[structSize];
                                var handle = System.Runtime.InteropServices.GCHandle.Alloc(bytes, System.Runtime.InteropServices.GCHandleType.Pinned);
                                try
                                {
                                    if (value.GetType().IsGenericType)
                                    {
                                        unsafe
                                        {
                                            byte* ptr = (byte*)handle.AddrOfPinnedObject();
                                            var unsafeWriteMethod = _unsafeWriteCache.GetOrAdd(value.GetType(), t =>
                                                typeof(Unsafe)
                                                    .GetMethod("Write", BindingFlags.Public | BindingFlags.Static)!
                                                    .MakeGenericMethod(t));
                                            unsafeWriteMethod.Invoke(null, new object[] { (IntPtr)ptr, value });
                                        }
                                    }
                                    else
                                    {
                                        System.Runtime.InteropServices.Marshal.StructureToPtr(value, handle.AddrOfPinnedObject(), false);
                                    }
                                }
                                finally { handle.Free(); }
                            }
                            else
                            {
                                // Reference type (display class): flatten
                                // fields via reflection and serialize each
                                // primitive to scratch.
                                var captureValues = new List<object>();
                                FlattenCLRStruct(value, captureValues);
                                structSize = 0;
                                foreach (var cv in captureValues)
                                {
                                    if (cv is long || cv is double) structSize += 8;
                                    else structSize += 4;
                                }
                                bytes = new byte[structSize];
                                int off = 0;
                                foreach (var cv in captureValues)
                                {
                                    WritePrimitiveToBytes(bytes, off, cv,
                                        0x7F /* i32 default */, 4);
                                    if (cv is long || cv is double) off += 8;
                                    else off += 4;
                                }
                            }

                            // Align to 4 bytes within scratch
                            scratchCursor = (scratchCursor + 3) & ~3;
                            int absoluteOffset = scratchBase + scratchCursor;
                            structScratchWrites.Add((absoluteOffset, bytes));
                            flatArgs.Add(absoluteOffset.ToString());
                            scratchCursor += structSize;
                            if (WasmBackend.VerboseLogging) WasmBackend.Log($"[Wasm] Struct scalar arg: type={value.GetType().Name}, size={structSize}, scratchOffset={absoluteOffset}");
                        }
                        else flatArgs.Add(value?.ToString() ?? "0");
                    }
                    wasmArgIdx++;
                }

                // Write struct scalar args into scratch memory region
                if (structScratchWrites.Count > 0)
                {
                    foreach (var (scratchOffset, bytes) in structScratchWrites)
                    {
                        using var dstView = new Uint8Array(memoryBuffer, scratchOffset, bytes.Length);
                        dstView.JSRef!.CallVoid("set", (object)bytes);
                    }
                }

                // Restore NativePtr to 0 (cleanup from the pre-serialization patching)
                for (int i = 0; i < bufferInfos.Count; i++)
                    bufferInfos[i].buffer.NativePtr = IntPtr.Zero;

                // Log first few dispatches' flat args for debugging
                if (dispNum >= 2 && dispNum <= 6)
                {
                    string flatArgStr = "";
                    for (int fi = 0; fi < flatArgs.Count; fi++)
                        flatArgStr += (fi > 0 ? "," : "") + flatArgs[fi];
                    _dispatchLog += $"|D{dispNum}:items={totalItems},gs={groupSize},ng={numGroups},bar={compiledKernel.HasBarriers},flat=[{flatArgStr}]";
                }

                // Dispatch to workers
                await DispatchToWorkers(
                    totalItems, gridDimX, gridDimY, scratchBase, scratchPerThread,
                    sharedMemBase, barrierBase, fenceSlot, compiledKernel,
                    groupSize, numGroups,
                    flatArgs, compiledKernel.WasmBinary,
                    wasmMemory, memoryBuffer, bufferOffsets, bufferInfos,
                    dynamicSharedElements, dispNum);

                // Clean up per-dispatch memory (only created for concurrent dispatches)
                disposeBuffer?.Dispose();
                disposeWasmMemory?.Dispose();
            }
            catch (Exception ex)
            {
                _dispatchLog += $"|ERR_D{dispNum}:{ex.Message}";
                WasmBackend.Log($"[Wasm] Kernel execution error: {ex}");
                throw;
            }
            finally
            {
                // Always decrement active dispatch count
                _activeDispatchCount--;
            }
        }

        /// <summary>
        /// Dispatches the Wasm kernel across multiple Web Workers for true Wasm multithreading.
        /// For barrier kernels: distributes groups across workers. Each worker runs all threads within its groups.
        /// For non-barrier kernels: distributes items across workers with flat range dispatch.
        /// </summary>
        private async Task DispatchToWorkers(
            int totalItems,
            int gridDimX,
            int gridDimY,
            int scratchBase,
            int scratchPerThread,
            int sharedMemBase,
            int barrierBase,
            int fenceSlot,
            WasmCompiledKernel compiledKernel,
            int groupSize,
            int numGroups,
            List<string> flatArgs,
            byte[] wasmBytes,
            JSObject wasmMemory,
            SharedArrayBuffer memoryBuffer,
            List<int> bufferOffsets,
            List<(WasmMemoryBuffer buffer, int byteOffset)> bufferInfos,
            int dynamicSharedElements = 0,
            int dispNum = 0)
        {
            bool hasBarriers = compiledKernel.HasBarriers;

            int workerCount;
            int phaseCount = compiledKernel.PhaseCount;
            if (hasBarriers)
            {
                // Fiber dispatch: use hardwareConcurrency workers, each running
                // groupSize/workerCount fibers sequentially per phase.
                // All barrier kernels (including those with helper barriers) use this path.
                // Single worker for now — cross-worker Atomics.wait barrier
                // deadlocks in browser workers. The fiber dispatch is correct
                // but the inter-worker sync needs investigation.
                // TODO: Fix cross-worker barrier to use multiple workers.
                workerCount = 1;
            }
            else
            {
                // For non-barrier kernels, don't spawn more workers than items
                workerCount = _workerCount;
                if (workerCount > totalItems) workerCount = Math.Max(1, totalItems);
            }

            if (WasmBackend.VerboseLogging) WasmBackend.Log($"[Wasm] Dispatch: workers={workerCount}, items={totalItems}, barriers={hasBarriers}, gs={groupSize}, ng={numGroups}, phases={phaseCount}");

            // Dump barrier kernel binary for debugging (console capture by PlaywrightMultiTest)
            if (hasBarriers)
            {
                Console.WriteLine($"[Wasm_DUMP_START] dispatch={_dispatchCount} size={wasmBytes.Length} phases={phaseCount} spt={scratchPerThread} shm={sharedMemBase} bar={barrierBase}");
                var b64 = Convert.ToBase64String(wasmBytes);
                for (int ci = 0; ci < b64.Length; ci += 1000)
                    Console.WriteLine($"[Wasm_DUMP] {b64.Substring(ci, Math.Min(1000, b64.Length - ci))}");
                Console.WriteLine("[Wasm_DUMP_END]");
            }

            // Build the worker script
            string argStr = string.Join(", ", flatArgs);
            var workerScript = BuildWasmWorkerScript(
                gridDimX, gridDimY, scratchBase, scratchPerThread,
                sharedMemBase, barrierBase, fenceSlot,
                groupSize, numGroups, hasBarriers, argStr,
                dynamicSharedElements, workerCount, phaseCount);

            // Lazily initialize the worker pool, or grow it if needed
            if (_workerPool == null)
                _workerPool = new WorkerPool(workerCount, useAsync: true);
            else
                _workerPool.EnsureSize(workerCount, useAsync: true);

            var workers = _workerPool.Acquire(workerCount);
            // If pool didn't have enough idle workers, grow and re-acquire
            if (workers.Count < workerCount)
            {
                var shortfall = workerCount - workers.Count;
                _workerPool.EnsureSize(_workerPool.Size + shortfall, useAsync: true);
                var extra = _workerPool.Acquire(shortfall);
                workers.AddRange(extra);
            }

            // If the kernel changed since last dispatch, invalidate worker caches
            if (!ReferenceEquals(wasmBytes, _lastWasmBytes))
            {
                _initializedWorkers.Clear();
                _lastWasmBytes = wasmBytes;
            }
            var tasks = new List<Task>();

            if (hasBarriers)
            {
                // Barrier dispatch: each worker gets its threadId (0..groupSize-1)
                // All workers process the same groups sequentially
                for (int w = 0; w < workerCount; w++)
                {
                    var worker = workers[w];
                    var tcs = new TaskCompletionSource();
                    int workerIdx = w;

                    Action<MessageEvent>? msgHandler = null;
                    Action<Event>? errHandler = null;

                    msgHandler = new Action<MessageEvent>((msg) =>
                    {
                        worker.OnMessage -= msgHandler!;
                        worker.OnError -= errHandler!;
                        _workerPool?.Return(worker);

                        // Capture diagnostic from worker 0
                        if (workerIdx == 0 && _dispatchCount <= 6)
                        {
                            try
                            {
                                var d0 = msg.JSRef!.Get<int?>("data.diag.0");
                                var d1 = msg.JSRef!.Get<int?>("data.diag.1");
                                _dispatchLog += $"|W0mem=[{d0},{d1}]";
                            }
                            catch { }
                        }

                        var done = msg.JSRef!.Get<bool>("data.done");
                        if (!done)
                        {
                            var errorMsg = msg.JSRef!.Get<string?>("data.error") ?? "Unknown worker error";
                            tcs.TrySetException(new Exception($"[Wasm] Worker {workerIdx} error: {errorMsg}"));
                            return;
                        }
                        tcs.TrySetResult();
                    });

                    errHandler = new Action<Event>((err) =>
                    {
                        worker.OnMessage -= msgHandler!;
                        worker.OnError -= errHandler!;
                        _workerPool?.Return(worker);
                        tcs.TrySetException(new Exception($"[Wasm] Worker {workerIdx} error during kernel execution"));
                    });

                    worker.OnMessage += msgHandler;
                    worker.OnError += errHandler;

                    // Send fiber range to worker.
                    // Fiber dispatch: each worker handles a contiguous range of threads.
                    int fibersPerWorker = (groupSize + workerCount - 1) / workerCount;
                    int threadStart = w * fibersPerWorker;
                    int threadEnd = Math.Min(threadStart + fibersPerWorker, groupSize);

                    var workerId = worker.JSRef!.GetHashCode();
                    if (_initializedWorkers.Add(workerId))
                    {
                        worker.PostMessage(new
                        {
                            script = workerScript,
                            wasmBytes = wasmBytes,
                            memory = wasmMemory,
                            threadStart,
                            threadEnd,
                        });
                    }
                    else
                    {
                        worker.PostMessage(new
                        {
                            script = workerScript,
                            memory = wasmMemory,
                            threadStart,
                            threadEnd,
                        });
                    }

                    tasks.Add(tcs.Task);
                }
            }
            else
            {
                // Non-barrier: flat item distribution
                int itemsPerWorker = totalItems / workerCount;
                int remainder = totalItems % workerCount;

                for (int w = 0; w < workerCount; w++)
                {
                    var worker = workers[w];
                    int startIdx = w * itemsPerWorker + Math.Min(w, remainder);
                    int endIdx = startIdx + itemsPerWorker + (w < remainder ? 1 : 0);

                    var tcs = new TaskCompletionSource();
                    int workerIdx = w;

                    Action<MessageEvent>? msgHandler = null;
                    Action<Event>? errHandler = null;

                    msgHandler = new Action<MessageEvent>((msg) =>
                    {
                        worker.OnMessage -= msgHandler!;
                        worker.OnError -= errHandler!;
                        _workerPool?.Return(worker);

                        var done = msg.JSRef!.Get<bool>("data.done");
                        if (!done)
                        {
                            var errorMsg = msg.JSRef!.Get<string?>("data.error") ?? "Unknown worker error";
                            tcs.TrySetException(new Exception($"[Wasm] Worker {workerIdx} error: {errorMsg}"));
                            return;
                        }
                        tcs.TrySetResult();
                    });

                    errHandler = new Action<Event>((err) =>
                    {
                        worker.OnMessage -= msgHandler!;
                        worker.OnError -= errHandler!;
                        _workerPool?.Return(worker);
                        tcs.TrySetException(new Exception($"[Wasm] Worker {workerIdx} error during kernel execution"));
                    });

                    worker.OnMessage += msgHandler;
                    worker.OnError += errHandler;

                    // Only include wasmBytes on first dispatch to this worker;
                    // the bootstrap caches the compiled module.
                    // Always include memory since each dispatch creates a new WebAssembly.Memory.
                    var workerId = worker.JSRef!.GetHashCode();
                    if (_initializedWorkers.Add(workerId))
                    {
                        worker.PostMessage(new
                        {
                            script = workerScript,
                            wasmBytes = wasmBytes,
                            memory = wasmMemory,
                            startIdx = startIdx,
                            endIdx = endIdx,
                        });
                    }
                    else
                    {
                        worker.PostMessage(new
                        {
                            script = workerScript,
                            memory = wasmMemory,
                            startIdx = startIdx,
                            endIdx = endIdx,
                        });
                    }

                    tasks.Add(tcs.Task);
                }
            }

            // Wait for all workers to complete
            await Task.WhenAll(tasks);

            // Debug: dump first 4 bytes of each buffer in Wasm memory after kernel
            if (dispNum >= 2 && dispNum <= 6)
            {
                for (int bi = 0; bi < bufferInfos.Count; bi++)
                {
                    int off = bufferOffsets[bi];
                    int bufLen = (int)bufferInfos[bi].buffer.LengthInBytes;
                    int dumpLen = Math.Min(bufLen, 16); // first 4 ints
                    using var rawView = new Uint8Array(memoryBuffer, off, dumpLen);
                    var rawBytes = rawView.ReadBytes();
                    string vals = "";
                    for (int j = 0; j + 3 < rawBytes.Length; j += 4)
                        vals += (j > 0 ? "," : "") + BitConverter.ToInt32(rawBytes, j);
                    _dispatchLog += $"|D{dispNum}B{bi}=[{vals}]";
                }
            }

            // Copy results back from Wasm linear memory to individual buffers
            for (int i = 0; i < bufferInfos.Count; i++)
            {
                var (buf, _) = bufferInfos[i];
                int offset = bufferOffsets[i];

                using var srcView = new Uint8Array(memoryBuffer, offset, (int)buf.LengthInBytes);
                using var dstView = new Uint8Array(buf.SharedBuffer);
                dstView.JSRef!.CallVoid("set", srcView);
                // Read first 4 bytes from Wasm memory at this offset for debugging
                var debugSrc = new Uint8Array(memoryBuffer, offset, 4);
                var debugBytes = debugSrc.ReadBytes();
                debugSrc.Dispose();
                int debugVal = BitConverter.ToInt32(debugBytes);
                _lastImplicitIndexDebug += $" | wasmMem[{offset}]={debugVal}";
            }
            _lastImplicitIndexDebug += $" | copyOutCount={bufferInfos.Count}";
        }

        /// <summary>
        /// Builds the JS script that runs inside each Wasm worker.
        /// For barrier kernels: worker receives { wasmBytes, memory, threadId }
        ///   and iterates over groups, calling kernel(globalIdx, dimX, dimY, scratchBase, groupDimX, threadIdX, sharedMemBase, barrierBase, ...args)
        /// For non-barrier kernels: worker receives { wasmBytes, memory, startIdx, endIdx }
        ///   and iterates over its assigned items with the same kernel signature (groupDimX=dimX, threadIdX=globalIdx).
        /// </summary>
        private static string BuildWasmWorkerScript(
            int gridDimX, int gridDimY, int scratchBase, int scratchPerThread,
            int sharedMemBase, int barrierBase, int fenceSlot,
            int groupSize, int numGroups, bool hasBarriers,
            string argStr,
            int dynamicSharedLength = 0,
            int workerCount = 1,
            int phaseCount = 1)
        {
            // Produces an async function body string that is sent as the 'script' field
            // in the message to the pool worker's async bootstrap.
            // The bootstrap caches WebAssembly.compile/instantiate — the script body
            // just uses the pre-cached instance from d._instance.
            var sb = new System.Text.StringBuilder();
            sb.AppendLine("    const kernel = d._instance.exports.kernel;");
            sb.AppendLine();

            if (hasBarriers)
            {
                // Fiber dispatch: each worker runs multiple fibers (threads) per phase.
                // N workers × M fibers each. The kernel returns i32: 0=done, 1=yielded.
                // Workers loop until all fibers return 0 (complete).
                sb.AppendLine("    const threadStart = d.threadStart;");
                sb.AppendLine("    const threadEnd = d.threadEnd;");
                sb.AppendLine($"    const groupSize = {groupSize};");
                sb.AppendLine($"    const numGroups = {numGroups};");
                sb.AppendLine($"    const workerCount = {workerCount};");
                sb.AppendLine($"    const scratchPerThread = {scratchPerThread};");
                sb.AppendLine($"    const _phaseBarrier = new Int32Array(d.memory.buffer, {fenceSlot}, 2);");
                sb.AppendLine($"    const _groupBarrier = new Int32Array(d.memory.buffer, {fenceSlot} + 8, 2);");
                sb.AppendLine();
                sb.AppendLine("    for (let g = 0; g < numGroups; g++) {");
                sb.AppendLine("      let phase = 0;");
                sb.AppendLine("      let _phaseTrace = '';");
                sb.AppendLine("      while (true) {");
                sb.AppendLine("        let anyYielded = false;");
                sb.AppendLine("        let _yieldCount = 0;");
                sb.AppendLine("        for (let tid = threadStart; tid < threadEnd; tid++) {");
                sb.AppendLine("          const globalIdx = g * groupSize + tid;");
                sb.AppendLine($"          const myScratch = {scratchBase} + tid * scratchPerThread;");
                sb.AppendLine("          let r;");
                sb.AppendLine("          try {");
                sb.Append($"            r = kernel(globalIdx, {gridDimX}, {gridDimY}, myScratch, {groupSize}, tid, {sharedMemBase}, {barrierBase}, {dynamicSharedLength}, phase");
                if (argStr.Length > 0)
                {
                    sb.Append(", ");
                    sb.Append(argStr);
                }
                sb.AppendLine(");");
                sb.AppendLine("          } catch(e) { self.postMessage({ done: false, error: 'Kernel trap: ' + e.message + ' g=' + g + ' tid=' + tid + ' phase=' + phase + ' spt=' + scratchPerThread + ' trace:' + _phaseTrace }); return; }");
                sb.AppendLine("          if (r === 1) { anyYielded = true; _yieldCount++; } else if (phase >= 10) { _phaseTrace += 'DONE:t' + tid + ' '; }");
                sb.AppendLine("        }");
                sb.AppendLine("        if (phase >= 10) _phaseTrace += 'p' + phase + ':' + _yieldCount + '/' + (threadEnd-threadStart) + ' ';");
                sb.AppendLine("        if (!anyYielded) break;");
                sb.AppendLine("        if (phase >= 50) { self.postMessage({ done: false, error: 'Phase limit 50. trace: ' + _phaseTrace }); return; }");
                // Cross-worker barrier between phases
                // Phase barrier: only needed with multiple workers
                sb.AppendLine("        if (workerCount > 1) {");
                sb.AppendLine("          const arrived = Atomics.add(_phaseBarrier, 0, 1) + 1;");
                sb.AppendLine("          if (arrived === workerCount) {");
                sb.AppendLine("            Atomics.store(_phaseBarrier, 0, 0);");
                sb.AppendLine("            Atomics.add(_phaseBarrier, 1, 1);");
                sb.AppendLine("            Atomics.notify(_phaseBarrier, 1, workerCount);");
                sb.AppendLine("          } else {");
                sb.AppendLine("            const gen = Atomics.load(_phaseBarrier, 1);");
                sb.AppendLine("            while (Atomics.load(_phaseBarrier, 1) === gen) {");
                sb.AppendLine("              Atomics.wait(_phaseBarrier, 1, gen, 1);");
                sb.AppendLine("            }");
                sb.AppendLine("          }");
                sb.AppendLine("        }");
                sb.AppendLine("        phase++;");
                sb.AppendLine("      }");
                // Cross-group barrier
                sb.AppendLine("      const gArrived = Atomics.add(_groupBarrier, 0, 1) + 1;");
                sb.AppendLine("      if (gArrived === workerCount) {");
                sb.AppendLine("        Atomics.store(_groupBarrier, 0, 0);");
                sb.AppendLine("        Atomics.add(_groupBarrier, 1, 1);");
                sb.AppendLine("        Atomics.notify(_groupBarrier, 1, workerCount);");
                sb.AppendLine("      } else {");
                sb.AppendLine("        const gGen = Atomics.load(_groupBarrier, 1);");
                sb.AppendLine("        while (Atomics.load(_groupBarrier, 1) === gGen) {");
                sb.AppendLine("          Atomics.wait(_groupBarrier, 1, gGen, 1);");
                sb.AppendLine("        }");
                sb.AppendLine("      }");
                sb.AppendLine("    }");
            }
            else
            {
                // Non-barrier kernel: flat item dispatch
                sb.AppendLine("    const startIdx = d.startIdx;");
                sb.AppendLine("    const endIdx = d.endIdx;");
                sb.AppendLine();
                sb.AppendLine("    for (let i = startIdx; i < endIdx; i++) {");
                // For non-barrier kernels: pass groupSize so Grid.IdxX/Y can decompose correctly
                sb.Append($"      kernel(i, {gridDimX}, {gridDimY}, {scratchBase}, {groupSize}, i % {groupSize}, 0, 0, 0, 0");
                if (argStr.Length > 0)
                {
                    sb.Append(", ");
                    sb.Append(argStr);
                }
                sb.AppendLine(");");
                sb.AppendLine("    }");
            }

            sb.AppendLine();
            sb.AppendLine("    self.postMessage({ done: true });");
            return sb.ToString();
        }

        /// <summary>
        /// Patches ArrayView pointer fields inside a serialized struct to use the
        /// correct Wasm memory offsets. Without this, the serialized NativePtr (0)
        /// causes kernels to write to address 0 instead of the buffer's actual location.
        /// </summary>
        /// <summary>
        /// Recursively scans a struct for IArrayView fields and adds their
        /// buffers to the buffer collection (for copy-in/copy-out and NativePtr patching).
        /// </summary>
        private static void ExtractBuffersFromStruct(
            object structValue,
            Dictionary<WasmMemoryBuffer, int> uniqueBuffers,
            List<(WasmMemoryBuffer buffer, int byteOffset)> bufferInfos)
        {
            try
            {
                var type = structValue.GetType();
                var fields = type.GetFields(
                    BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                _lastImplicitIndexDebug += $" | ExtractBuf: type={type.Name}, fields={fields.Length}";
                foreach (var field in fields)
                {
                    object? val;
                    try { val = field.GetValue(structValue); }
                    catch (Exception ex) { WasmBackend.Log($"[Wasm] ExtractBuffers: field '{field.Name}' get failed: {ex.Message}"); continue; }
                    if (val == null) continue;
                    _lastImplicitIndexDebug += $" | field='{field.Name}' type={val.GetType().Name} isView={val is IArrayView}";

                    if (val is IArrayView iav)
                    {
                        var wasmBuf = iav.Buffer as WasmMemoryBuffer;
                        _lastImplicitIndexDebug += $" | buf={wasmBuf?.GetType().Name ?? "null"} len={iav.Length} inDict={wasmBuf != null && uniqueBuffers.ContainsKey(wasmBuf)}";
                        if (wasmBuf != null && !uniqueBuffers.ContainsKey(wasmBuf))
                        {
                            int bufIdx = bufferInfos.Count;
                            uniqueBuffers[wasmBuf] = bufIdx;
                            bufferInfos.Add((wasmBuf, 0));
                            _lastImplicitIndexDebug += $" | ADDED buf#{bufIdx}";
                        }
                    }
                    else if (val.GetType().IsValueType && !val.GetType().IsPrimitive && !val.GetType().IsEnum)
                    {
                        ExtractBuffersFromStruct(val, uniqueBuffers, bufferInfos);
                    }
                }
            }
            catch { }
        }

        /// <summary>
        /// Patches ArrayView pointer fields inside a serialized struct.
        /// Scans the struct's fields for IArrayView instances and writes the correct
        /// Wasm buffer offset at the view's NativePtr position in the serialized bytes.
        /// </summary>
        private static void PatchViewPointersInStruct(
            object structValue,
            byte[] bytes,
            List<(WasmMemoryBuffer buffer, int byteOffset)> bufferInfos,
            List<int> bufferOffsets,
            List<int> viewSubOffsets,
            List<int> viewBufferIdx)
        {
            try
            {
                // Find ALL IArrayView fields in the struct (including nested)
                var viewFields = new List<(IArrayView view, int bytePos)>();
                FindViewFieldsInStruct(structValue, structValue.GetType(), 0, viewFields);

                foreach (var (view, bytePos) in viewFields)
                {
                    var wasmBuf = view.Buffer as WasmMemoryBuffer;
                    if (wasmBuf == null || bytePos + 4 > bytes.Length) continue;

                    // Find this buffer's Wasm offset
                    for (int bi = 0; bi < bufferInfos.Count; bi++)
                    {
                        if (bufferInfos[bi].buffer == wasmBuf)
                        {
                            int wasmOffset = bufferOffsets[bi];

                            // Get SubView byte offset
                            int subOffset = 0;
                            try
                            {
                                var baseProp = view.GetType().GetProperty("BaseView");
                                object viewObj = baseProp != null ? baseProp.GetValue(view)! : view;
                                var indexProp = viewObj.GetType().GetProperty("Index",
                                    BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                                if (indexProp?.GetValue(viewObj) is long idx)
                                    subOffset = (int)(idx * view.Buffer.ElementSize);
                            }
                            catch { }

                            int correctPtr = wasmOffset + subOffset;
                            BitConverter.TryWriteBytes(bytes.AsSpan(bytePos), correctPtr);
                            break;
                        }
                    }
                }
            }
            catch { }
        }

        private static void FindViewFieldsInStruct(
            object obj, Type type, int baseOffset,
            List<(IArrayView view, int bytePos)> results)
        {
            // Use Unsafe.SizeOf via ILGPU's Interop to get the struct size
            int runningOffset = baseOffset;
            var fields = type.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);

            foreach (var field in fields)
            {
                object? val;
                try { val = field.GetValue(obj); }
                catch { runningOffset += IntPtr.Size; continue; }
                if (val == null) { runningOffset += IntPtr.Size; continue; }

                if (val is IArrayView iav)
                {
                    // The NativePtr is at the start of the ArrayView's memory layout
                    results.Add((iav, runningOffset));
                    runningOffset += global::ILGPU.Interop.SizeOf(field.FieldType);
                }
                else if (val.GetType().IsValueType && !val.GetType().IsPrimitive && !val.GetType().IsEnum)
                {
                    // Recurse into nested struct
                    FindViewFieldsInStruct(val, field.FieldType, runningOffset, results);
                    runningOffset += global::ILGPU.Interop.SizeOf(field.FieldType);
                }
                else
                {
                    try { runningOffset += System.Runtime.InteropServices.Marshal.SizeOf(field.FieldType); }
                    catch { runningOffset += IntPtr.Size; }
                }
            }
        }

        /// <summary>
        /// Recursively flattens a CLR struct into a list of leaf values (depth-first).
        /// The order matches ILGPU's IR StructureType flattening, since both process
        /// fields in declaration order.
        ///
        /// For IArrayView fields: adds the view itself (for view pointer patching).
        /// For primitive fields: adds the boxed primitive value.
        /// For nested structs: recurses depth-first.
        /// </summary>
        private static void FlattenCLRStruct(object structValue, List<object> result)
        {
            var type = structValue.GetType();
            var fields = type.GetFields(
                BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);

            foreach (var field in fields)
            {
                object? val;
                try { val = field.GetValue(structValue); }
                catch { result.Add(0); continue; }
                if (val == null) { result.Add(0); continue; }

                if (val is IArrayView)
                {
                    // ILGPU's IR represents ArrayView as a single AddressSpaceType (just
                    // the pointer). The view's Index and Length are NOT separate fields in
                    // the IR struct. So we add only the view itself for ptr patching.
                    //
                    // For ArrayView1D<T, TStride>, the IR also includes Extent and Stride
                    // as separate fields (from the wrapper struct, not from ArrayView<T>).
                    // We add those by recursing into the ArrayView1D's non-BaseView fields.
                    var viewType = val.GetType();
                    var baseProp = viewType.GetProperty("BaseView");

                    // The view pointer (maps to AddressSpaceType in IR)
                    object baseView = baseProp != null ? baseProp.GetValue(val)! : val;
                    result.Add(baseView is IArrayView ? baseView : val);

                    // If ArrayView1D: add Extent and Stride (NOT Index/Length — those
                    // are internal to ArrayView<T> and not in the IR struct)
                    if (baseProp != null)
                    {
                        // Extent (LongIndex1D → flattens to Int64)
                        var extentProp = viewType.GetProperty("Extent");
                        if (extentProp != null)
                        {
                            var extent = extentProp.GetValue(val);
                            if (extent != null)
                                FlattenCLRStruct(extent, result);
                        }

                        // Stride (e.g., Stride1D.Dense → flattens to its internal fields)
                        var strideProp = viewType.GetProperty("Stride");
                        if (strideProp != null)
                        {
                            var stride = strideProp.GetValue(val);
                            if (stride != null)
                            {
                                if (stride.GetType().IsPrimitive)
                                {
                                    result.Add(stride);
                                }
                                else
                                {
                                    int beforeStride = result.Count;
                                    FlattenCLRStruct(stride, result);
                                    // Empty structs (like Dense) have no instance fields.
                                    // ILGPU adds an Int8 padding field for them.
                                    if (result.Count == beforeStride)
                                        result.Add((byte)0);
                                }
                            }
                        }
                    }
                }
                else if (val.GetType().IsPrimitive || val.GetType().IsEnum)
                {
                    result.Add(val);
                }
                else if (val.GetType().IsValueType)
                {
                    // Nested struct: recurse
                    int beforeCount = result.Count;
                    FlattenCLRStruct(val, result);
                    // If the struct had no fields (like Stride1D.Dense which has only
                    // computed properties), ILGPU adds an Int8 padding field.
                    // Emit a default 0 to match.
                    if (result.Count == beforeCount)
                        result.Add((byte)0);
                }
                else
                {
                    result.Add(val);
                }
            }
        }

        /// <summary>
        /// Writes a primitive value to a byte array at the specified offset.
        /// </summary>
        private static void WritePrimitiveToBytes(byte[] bytes, int offset, object value, byte wasmType, int size)
        {
            if (offset + size > bytes.Length) return;
            try
            {
                switch (wasmType)
                {
                    case WasmOpCodes.I32:
                        int i32Val = value switch
                        {
                            int i => i,
                            uint u => (int)u,
                            short s => s,
                            ushort us => us,
                            byte b => b,
                            sbyte sb => sb,
                            bool bl => bl ? 1 : 0,
                            _ => Convert.ToInt32(value)
                        };
                        BitConverter.TryWriteBytes(bytes.AsSpan(offset), i32Val);
                        break;
                    case WasmOpCodes.I64:
                        long i64Val = value switch
                        {
                            long l => l,
                            ulong ul => (long)ul,
                            int i => i,
                            uint u => u,
                            _ => Convert.ToInt64(value)
                        };
                        BitConverter.TryWriteBytes(bytes.AsSpan(offset), i64Val);
                        break;
                    case WasmOpCodes.F32:
                        float f32Val = value switch
                        {
                            float f => f,
                            _ => Convert.ToSingle(value)
                        };
                        BitConverter.TryWriteBytes(bytes.AsSpan(offset), f32Val);
                        break;
                    case WasmOpCodes.F64:
                        double f64Val = value switch
                        {
                            double d => d,
                            float f => f,
                            _ => Convert.ToDouble(value)
                        };
                        BitConverter.TryWriteBytes(bytes.AsSpan(offset), f64Val);
                        break;
                }
            }
            catch { }
        }

        #endregion

        #region Memory Management

        /// <inheritdoc/>
        protected override MemoryBuffer AllocateRawInternal(long length, int elementSize)
        {
            return new WasmMemoryBuffer(this, length, elementSize);
        }

        protected override void SynchronizeInternal()
        {
            // No-op: synchronous blocking would deadlock in single-threaded Blazor WASM.
            // Buffer synchronization is handled by the serialized dispatch in RunKernelAsync.
        }

        /// <summary>
        /// Asynchronously waits for all pending kernel dispatches to complete.
        /// </summary>
        public async Task SynchronizeAsync()
        {
            if (_pendingWork.Count > 0)
            {
                await Task.WhenAll(_pendingWork);
                _pendingWork.Clear();
            }
        }

        #endregion

        #region Helpers

        private static (int x, int y, int z) GetGridDimensions(object dimension)
        {
            return dimension switch
            {
                Index1D i1 => (i1.X, 1, 1),
                Index2D i2 => (i2.X, i2.Y, 1),
                Index3D i3 => (i3.X, i3.Y, i3.Z),
                LongIndex1D l1 => ((int)l1.X, 1, 1),
                LongIndex2D l2 => ((int)l2.X, (int)l2.Y, 1),
                LongIndex3D l3 => ((int)l3.X, (int)l3.Y, (int)l3.Z),
                KernelConfig config => (
                    config.GridDim.X * config.GroupDim.X,
                    config.GridDim.Y * config.GroupDim.Y,
                    config.GridDim.Z * config.GroupDim.Z),
                _ => throw new NotSupportedException($"Unsupported dimension type: {dimension.GetType()}")
            };
        }

        private (int x, int y, int z) GetGroupDimensions(KernelConfig config)
        {
            var groupDim = config.GroupDim;
            return (Math.Max(groupDim.X, 1), Math.Max(groupDim.Y, 1), Math.Max(groupDim.Z, 1));
        }

        /// <summary>
        /// Extracts the group size from the dimension object.
        /// For KernelConfig, returns GroupDim product.
        /// For auto-grouped kernels (Index1D/2D/3D), returns the estimated group size.
        /// </summary>
        private int GetGroupSize(object dimension, WasmCompiledKernel compiledKernel)
        {
            if (dimension is KernelConfig config)
            {
                return config.GroupDim.X * config.GroupDim.Y * config.GroupDim.Z;
            }
            // For auto-grouped kernels, the group size is determined by EstimateGroupSizeInternal
            // which returns 64 for the Wasm backend.
            return compiledKernel.HasBarriers ? 64 : 1;
        }

        #endregion

        #region Required Overrides (stubs)

        protected override int EstimateGroupSizeInternal(
            Kernel kernel, int dynamicSharedMemorySizeInBytes,
            int maxGroupSize,
            out int groupSize)
        {
            groupSize = 64;
            return 64;
        }

        protected override int EstimateGroupSizeInternal(
            Kernel kernel, Func<int, int> computeSharedMemorySize,
            int maxGroupSize, out int groupSize)
        {
            groupSize = 64;
            return 64;
        }

        protected override int EstimateMaxActiveGroupsPerMultiprocessorInternal(
            Kernel kernel, int groupSize, int dynamicSharedMemorySizeInBytes) => 1;

        protected override void EnablePeerAccessInternal(Accelerator otherAccelerator) { }
        protected override void DisablePeerAccessInternal(Accelerator otherAccelerator) { }
        protected override bool CanAccessPeerInternal(Accelerator otherAccelerator) => false;

        public override TExtension CreateExtension<TExtension, TExtensionProvider>(TExtensionProvider provider) =>
            default!;

        protected override AcceleratorStream CreateStreamInternal() =>
            new WasmStream(this);

        protected override PageLockScope<T> CreatePageLockFromPinnedInternal<T>(IntPtr ptr, long numElements) =>
            throw new NotSupportedException("Page locking is not supported in Wasm backend.");

        protected override void OnBind() { }
        protected override void OnUnbind() { }

        protected override void DisposeAccelerator_SyncRoot(bool disposing)
        {
            if (disposing)
            {
                _pendingWork.Clear();
                _initializedWorkers.Clear();
                _activeDispatchCount = 0;
                _cachedMemoryBuffer?.Dispose();
                _cachedMemoryBuffer = null;
                _cachedWasmMemory?.Dispose();
                _cachedWasmMemory = null;
                _cachedWasmPages = 0;
                _workerPool?.Dispose();
                _workerPool = null;
            }
        }

        #endregion
    }

    /// <summary>
    /// Wasm-specific kernel wrapper.
    /// </summary>
    public class WasmKernel : Kernel
    {
        public WasmKernel(Accelerator accelerator, CompiledKernel compiledKernel, MethodInfo launcher)
            : base(accelerator, compiledKernel, launcher) { }

        public new WasmCompiledKernel CompiledKernel => (WasmCompiledKernel)base.CompiledKernel;

        protected override void DisposeAcceleratorObject(bool disposing) { }
    }
}
