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
using global::ILGPU.Backends.EntryPoints;
using global::ILGPU.Resources;
using global::ILGPU.Runtime;
using SpawnDev.BlazorJS;
using SpawnDev.BlazorJS.JSObjects;
using SpawnDev.BlazorJS.Toolbox;
using SpawnDev.ILGPU.Wasm.Backend;
using System.Reflection;
using System.Reflection.Emit;

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

        // Worker count for parallel dispatch
        private int _workerCount = 4;

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
        public static void RunKernel(Kernel kernel, AcceleratorStream stream, object dimension, object[] args)
        {
            var wasmAccel = (WasmAccelerator)kernel.Accelerator;
            var wasmKernel = (WasmKernel)kernel;
            var compiledKernel = (WasmCompiledKernel)wasmKernel.CompiledKernel;

            WasmBackend.Log($"[Wasm-Debug] RunKernel called with {args.Length} args");

            var task = wasmAccel.RunKernelAsync(compiledKernel, dimension, args);
            wasmAccel._pendingWork.Add(task);
        }

        private async Task RunKernelAsync(
            WasmCompiledKernel compiledKernel,
            object dimension,
            object[] args)
        {
            try
            {
                var js = BlazorJSRuntime.JS;
                var (gridDimX, gridDimY, gridDimZ) = GetGridDimensions(dimension);
                int totalItems = gridDimX * gridDimY * gridDimZ;

                WasmBackend.Log($"[Wasm] Dispatching kernel: {totalItems} items ({gridDimX}x{gridDimY}x{gridDimZ})");

                // Determine isView flags from compiled kernel metadata
                var paramInfos = compiledKernel.ParamInfos;

                // Collect all SharedArrayBuffers from buffer arguments
                var bufferInfos = new List<(WasmMemoryBuffer buffer, int byteOffset)>();
                var wasmArgs = new List<(bool isBuffer, WasmMemoryBuffer? buffer, int length, int stride, int stride2, object? value)>();

                for (int i = 0; i < args.Length; i++)
                {
                    bool isView = i < paramInfos.Count && paramInfos[i].IsView;

                    if (isView && args[i] is IArrayView iav)
                    {
                        // ArrayView<T> is a value struct that implements IArrayView.
                        // Extract the underlying MemoryBuffer via .Buffer property.
                        var wasmBuf = iav.Buffer as WasmMemoryBuffer;
                        if (wasmBuf != null)
                        {
                            bufferInfos.Add((wasmBuf, wasmBuf.ByteOffset));

                            // Extract stride via reflection for multi-dimensional views
                            int stride = 1;
                            int stride2 = 0;
                            var argType = args[i].GetType();
                            var strideProp = argType.GetProperty("Stride");
                            if (strideProp != null)
                            {
                                var strideObj = strideProp.GetValue(args[i]);
                                if (strideObj != null)
                                {
                                    // Try YStride (for Stride2D.DenseX, Stride3D.DenseXY)
                                    var yStrideProp = strideObj.GetType().GetProperty("YStride");
                                    if (yStrideProp != null)
                                    {
                                        stride = (int)yStrideProp.GetValue(strideObj)!;
                                    }
                                    else
                                    {
                                        // Try XStride (for Stride2D.DenseY)
                                        var xStrideProp = strideObj.GetType().GetProperty("XStride");
                                        if (xStrideProp != null)
                                            stride = (int)xStrideProp.GetValue(strideObj)!;
                                    }

                                    // Try ZStride (for Stride3D.DenseXY)
                                    var zStrideProp = strideObj.GetType().GetProperty("ZStride");
                                    if (zStrideProp != null)
                                    {
                                        stride2 = (int)zStrideProp.GetValue(strideObj)!;
                                    }
                                }
                            }
                            WasmBackend.Log($"[Wasm] View arg[{i}]: length={iav.Length}, stride={stride}, stride2={stride2}");
                            wasmArgs.Add((true, wasmBuf, (int)iav.Length, stride, stride2, null));
                        }
                        else
                        {
                            wasmArgs.Add((false, null, 0, 0, 0, 0));
                        }
                    }
                    else
                    {
                        wasmArgs.Add((false, null, 0, 0, 0, args[i]));
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
                // Scratch memory for struct construction (after all buffers)
                int scratchBase = (totalMemoryBytes + 7) & ~7; // 8-byte align
                int scratchSize = 4096; // 4KB for temporary structs
                int totalWithScratch = scratchBase + scratchSize;

                // Round up to Wasm page size (64KB)
                int wasmPages = Math.Max(1, (totalWithScratch + 65535) / 65536);

                // Create shared Wasm memory
                using var wasmMemory = js.Call<JSObject>(
                    "eval",
                    $"new WebAssembly.Memory({{ initial: {wasmPages}, maximum: 65536, shared: true }})");

                // Get the SharedArrayBuffer backing the Wasm memory
                using var memoryBuffer = wasmMemory.JSRef!.Get<SharedArrayBuffer>("buffer");

                // Copy buffer data into the Wasm linear memory at computed offsets
                for (int i = 0; i < bufferInfos.Count; i++)
                {
                    var (buf, _) = bufferInfos[i];
                    int offset = bufferOffsets[i];

                    using var srcView = new Uint8Array(buf.SharedBuffer);
                    using var dstView = new Uint8Array(memoryBuffer, offset, (int)buf.LengthInBytes);
                    dstView.JSRef!.CallVoid("set", srcView);
                }

                // Build flat argument list
                var flatArgs = new List<string>();
                int bufferIndex = 0;
                foreach (var (isBuffer, buffer, length, stride, stride2, value) in wasmArgs)
                {
                    if (isBuffer)
                    {
                        flatArgs.Add(bufferOffsets[bufferIndex].ToString());
                        flatArgs.Add(length.ToString());
                        flatArgs.Add(stride.ToString());
                        flatArgs.Add(stride2.ToString());
                        bufferIndex++;
                    }
                    else
                    {
                        if (value is float fv) flatArgs.Add(fv.ToString("G9"));
                        else if (value is double dv) flatArgs.Add(dv.ToString("G17"));
                        else flatArgs.Add(value?.ToString() ?? "0");
                    }
                }

                // Dispatch to workers
                await DispatchToWorkers(
                    totalItems, gridDimX, gridDimY, scratchBase,
                    flatArgs, compiledKernel.WasmBinary,
                    wasmMemory, memoryBuffer, bufferOffsets, bufferInfos);
            }
            catch (Exception ex)
            {
                WasmBackend.Log($"[Wasm] Kernel execution error: {ex}");
                throw;
            }
        }

        /// <summary>
        /// Dispatches the Wasm kernel across multiple Web Workers for true Wasm multithreading.
        /// Each worker instantiates the same Wasm module with shared memory and runs its range.
        /// </summary>
        private async Task DispatchToWorkers(
            int totalItems,
            int gridDimX,
            int gridDimY,
            int scratchBase,
            List<string> flatArgs,
            byte[] wasmBytes,
            JSObject wasmMemory,
            SharedArrayBuffer memoryBuffer,
            List<int> bufferOffsets,
            List<(WasmMemoryBuffer buffer, int byteOffset)> bufferInfos)
        {
            int workerCount = _workerCount;
            // Don't spawn more workers than work items
            if (workerCount > totalItems) workerCount = Math.Max(1, totalItems);

            WasmBackend.Log($"[Wasm] Dispatching to {workerCount} worker(s), {totalItems} items");

            // Build the worker script
            string argStr = string.Join(", ", flatArgs);
            var workerScript = BuildWasmWorkerScript(gridDimX, gridDimY, scratchBase, argStr);

            // Create workers
            var workers = QuickWorker.CreateWorkersFromJS(workerScript, workerCount);
            var tasks = new List<Task>();

            // Distribute items evenly across workers
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
                    worker.Terminate();
                    worker.Dispose();

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
                    worker.Terminate();
                    worker.Dispose();
                    tcs.TrySetException(new Exception($"[Wasm] Worker {workerIdx} error during kernel execution"));
                });

                worker.OnMessage += msgHandler;
                worker.OnError += errHandler;

                // Send the Wasm binary, shared memory, and work range to the worker
                worker.PostMessage(new
                {
                    wasmBytes = wasmBytes,
                    memory = wasmMemory,
                    startIdx = startIdx,
                    endIdx = endIdx,
                });

                tasks.Add(tcs.Task);
            }

            // Wait for all workers to complete
            await Task.WhenAll(tasks);

            // Copy results back from Wasm linear memory to individual buffers
            for (int i = 0; i < bufferInfos.Count; i++)
            {
                var (buf, _) = bufferInfos[i];
                int offset = bufferOffsets[i];

                using var srcView = new Uint8Array(memoryBuffer, offset, (int)buf.LengthInBytes);
                using var dstView = new Uint8Array(buf.SharedBuffer);
                dstView.JSRef!.CallVoid("set", srcView);
            }
        }

        /// <summary>
        /// Builds the JS script that runs inside each Wasm worker.
        /// The worker receives: { wasmBytes, memory, startIdx, endIdx }
        /// It instantiates the Wasm module with the shared memory and runs the kernel loop.
        /// </summary>
        private static string BuildWasmWorkerScript(int gridDimX, int gridDimY, int scratchBase, string argStr)
        {
            var sb = new System.Text.StringBuilder();
            sb.AppendLine("self.onmessage = async function(e) {");
            sb.AppendLine("  try {");
            sb.AppendLine("    const d = e.data;");
            sb.AppendLine("    const wasmBuf = new Uint8Array(d.wasmBytes).buffer;");
            sb.AppendLine("    const memory = d.memory;");
            sb.AppendLine("    const startIdx = d.startIdx;");
            sb.AppendLine("    const endIdx = d.endIdx;");
            sb.AppendLine();
            sb.AppendLine("    const module = await WebAssembly.compile(wasmBuf);");
            sb.AppendLine("    const instance = await WebAssembly.instantiate(module, {");
            sb.AppendLine("      env: { memory: memory },");
            sb.AppendLine("      Math: {");
            sb.AppendLine("        sin: Math.sin, cos: Math.cos, tan: Math.tan,");
            sb.AppendLine("        asin: Math.asin, acos: Math.acos, atan: Math.atan,");
            sb.AppendLine("        sinh: Math.sinh, cosh: Math.cosh, tanh: Math.tanh,");
            sb.AppendLine("        exp: Math.exp, log: Math.log, log2: Math.log2,");
            sb.AppendLine("        log10: Math.log10, round: Math.round,");
            sb.AppendLine("        truncate: Math.trunc, sign: Math.sign,");
            sb.AppendLine("        exp2: (x) => Math.pow(2, x),");
            sb.AppendLine("        sqrt: Math.sqrt, abs: Math.abs,");
            sb.AppendLine("        ceil: Math.ceil, floor: Math.floor,");
            sb.AppendLine("        pow: Math.pow, atan2: Math.atan2");
            sb.AppendLine("      }");
            sb.AppendLine("    });");
            sb.AppendLine("    const kernel = instance.exports.kernel;");
            sb.AppendLine();
            sb.AppendLine("    // Run kernel for assigned range");
            sb.AppendLine("    for (let i = startIdx; i < endIdx; i++) {");
            sb.Append($"      kernel(i, {gridDimX}, {gridDimY}, {scratchBase}");
            if (argStr.Length > 0)
            {
                sb.Append(", ");
                sb.Append(argStr);
            }
            sb.AppendLine(");");
            sb.AppendLine("    }");
            sb.AppendLine();
            sb.AppendLine("    self.postMessage({ done: true });");
            sb.AppendLine("  } catch(ex) {");
            sb.AppendLine("    self.postMessage({ done: false, error: (ex && ex.message) ? ex.message : String(ex) });");
            sb.AppendLine("  }");
            sb.AppendLine("};");
            return sb.ToString();
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
            WasmBackend.Log("[Wasm] WARNING: SynchronizeInternal() called on main thread. Use SynchronizeAsync() instead.");
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
