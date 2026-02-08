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
                var (gridDimX, gridDimY, gridDimZ) = GetGridDimensions(dimension);
                int totalItems = gridDimX * gridDimY * gridDimZ;

                WasmBackend.Log($"[Wasm] Dispatching kernel: {totalItems} items ({gridDimX}x{gridDimY}x{gridDimZ})");

                // Determine isView flags from compiled kernel metadata
                var paramInfos = compiledKernel.ParamInfos;

                // Collect all SharedArrayBuffers from buffer arguments
                var bufferInfos = new List<(WasmMemoryBuffer buffer, int byteOffset)>();
                var wasmArgs = new List<(bool isBuffer, WasmMemoryBuffer? buffer, int length, object? value)>();

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
                            wasmArgs.Add((true, wasmBuf, (int)iav.Length, null));
                        }
                        else
                        {
                            wasmArgs.Add((false, null, 0, 0));
                        }
                    }
                    else
                    {
                        wasmArgs.Add((false, null, 0, args[i]));
                    }
                }

                // Build and execute
                var wasmBase64 = Convert.ToBase64String(compiledKernel.WasmBinary);
                await DispatchToMainThread(wasmBase64, totalItems, wasmArgs, bufferInfos);
            }
            catch (Exception ex)
            {
                WasmBackend.Log($"[Wasm] Kernel execution error: {ex}");
                throw;
            }
        }

        /// <summary>
        /// Dispatches the Wasm kernel on the main thread (Phase 1).
        /// </summary>
        private async Task DispatchToMainThread(
            string wasmBase64,
            int totalItems,
            List<(bool isBuffer, WasmMemoryBuffer? buffer, int length, object? value)> wasmArgs,
            List<(WasmMemoryBuffer buffer, int byteOffset)> bufferInfos)
        {
            var js = BlazorJSRuntime.JS;

            // Calculate total memory needed
            int totalMemoryBytes = 0;
            var bufferOffsets = new List<int>();

            foreach (var (buf, _) in bufferInfos)
            {
                totalMemoryBytes = (totalMemoryBytes + 7) & ~7; // 8-byte align
                bufferOffsets.Add(totalMemoryBytes);
                totalMemoryBytes += (int)buf.LengthInBytes;
            }

            // Round up to Wasm page size (64KB)
            int wasmPages = Math.Max(1, (totalMemoryBytes + 65535) / 65536);

            // Create Wasm memory
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

            // Build worker args: [bufferOffset, bufferLen] or [scalarValue]
            var flatArgs = new List<string>();
            int bufferIndex = 0;
            foreach (var (isBuffer, buffer, length, value) in wasmArgs)
            {
                if (isBuffer)
                {
                    flatArgs.Add(bufferOffsets[bufferIndex].ToString());
                    flatArgs.Add(length.ToString());
                    bufferIndex++;
                }
                else
                {
                    if (value is float fv) flatArgs.Add(fv.ToString("G9"));
                    else if (value is double dv) flatArgs.Add(dv.ToString("G17"));
                    else flatArgs.Add(value?.ToString() ?? "0");
                }
            }

            // Decode wasm binary from base64
            byte[] wasmBytes = Convert.FromBase64String(wasmBase64);

            // Build the JS execution script
            string argStr = string.Join(", ", flatArgs);

            string script = $@"
(async function() {{
    try {{
        const memory = globalThis.__wasm_memory__;
        const wasmBytesArr = globalThis.__wasm_bytes__;
        console.log('[Wasm-JS] Memory:', memory, 'ByteLength:', memory.buffer.byteLength);
        console.log('[Wasm-JS] WasmBytes length:', wasmBytesArr.length);
        console.log('[Wasm-JS] Hex:', Array.from(wasmBytesArr).map(b => b.toString(16).padStart(2,'0')).join(' '));
        const wasmBuf = wasmBytesArr.buffer.slice(wasmBytesArr.byteOffset, wasmBytesArr.byteOffset + wasmBytesArr.byteLength);
        console.log('[Wasm-JS] Compiling Wasm module from', wasmBuf.byteLength, 'bytes');
        const module = await WebAssembly.compile(wasmBuf);
        console.log('[Wasm-JS] Module compiled. Exports:', WebAssembly.Module.exports(module));
        console.log('[Wasm-JS] Imports:', WebAssembly.Module.imports(module));
        const instance = await WebAssembly.instantiate(module, {{ env: {{ memory: memory }} }});
        const kernel = instance.exports.kernel;
        console.log('[Wasm-JS] Kernel function:', kernel, 'length:', kernel.length);
        const totalItems = {totalItems};
        const view32 = new Int32Array(memory.buffer);
        console.log('[Wasm-JS] First 8 i32s before exec:', Array.from(view32.slice(0, 8)));
        console.log('[Wasm-JS] Calling kernel with totalItems=', totalItems, 'args: [{argStr}]');
        for (let i = 0; i < totalItems; i++) {{
            kernel(i, totalItems{(argStr.Length > 0 ? ", " + argStr : "")});
        }}
        console.log('[Wasm-JS] First 8 i32s after exec:', Array.from(view32.slice(0, 8)));
        console.log('[Wasm-JS] Kernel execution complete');
    }} catch(e) {{
        console.error('[Wasm-JS] ERROR:', e.message, e.stack);
        throw e;
    }}
}})()";

            WasmBackend.Log($"[Wasm] Executing script:\n{script}");

            // Store memory and wasm bytes on globalThis
            using var wasmArray = new Uint8Array(wasmBytes.Length);
            wasmArray.JSRef!.CallVoid("set", wasmBytes);
            js.Set("__wasm_memory__", wasmMemory);
            js.Set("__wasm_bytes__", wasmArray);

            // Execute
            using var promise = js.Call<JSObject>("eval", script);
            // Await the promise by creating a Task from it
            var tcs = new TaskCompletionSource();
            using var onResolve = Callback.Create(() => tcs.SetResult());
            using var onReject = Callback.Create((string err) => tcs.SetException(new Exception($"Wasm kernel error: {err}")));
            promise.JSRef!.CallVoid("then", onResolve, onReject);
            await tcs.Task;

            // Cleanup
            js.CallVoid("eval", "delete globalThis.__wasm_memory__; delete globalThis.__wasm_bytes__;");

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
