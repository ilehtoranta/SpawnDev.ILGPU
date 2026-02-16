// ---------------------------------------------------------------------------------------
//                               SpawnDev.ILGPU.Workers
//                 Web Worker Compute Library for Blazor WebAssembly
//
// File: WorkersAccelerator.cs
//
// Main accelerator for the Workers backend. Compiles and dispatches kernels
// across a pool of Web Workers using SharedArrayBuffer for memory sharing.
// ---------------------------------------------------------------------------------------

using global::ILGPU;
using global::ILGPU.Backends;
using global::ILGPU.Runtime;
using SpawnDev.BlazorJS;
using SpawnDev.BlazorJS.JSObjects;
using SpawnDev.ILGPU.Workers.Backend;
using System.Reflection;
using System.Reflection.Emit;

namespace SpawnDev.ILGPU.Workers
{
    /// <summary>
    /// Workers accelerator implementation for ILGPU.
    /// Compiles kernels to JavaScript and executes them across Web Workers.
    /// </summary>
    public class WorkersAccelerator : KernelAccelerator<WorkersCompiledKernel, WorkersKernel>
    {
        /// <summary>
        /// Gets the Workers backend used for kernel compilation (IR → JS).
        /// </summary>
        public WorkersBackend Backend { get; private set; } = null!;

        /// <summary>
        /// Gets the Workers device.
        /// </summary>
        public WorkersILGPUDevice WorkersDevice { get; private set; } = null!;

        /// <summary>
        /// Gets the worker pool size (number of concurrent Web Workers).
        /// When SharedArrayBuffer is available, kernels are distributed across this many workers.
        /// When not available, a single worker is used to avoid blocking the UI.
        /// </summary>
        public int WorkerCount { get; private set; }

        /// <summary>
        /// Gets whether multi-worker dispatch is available (requires SharedArrayBuffer / Cross-Origin Isolation).
        /// </summary>
        public bool UseMultiWorker => WorkerCount > 1 && WorkersMemoryBuffer.IsCrossOriginIsolated;

        /// <summary>
        /// Gets the list of pending work tasks from kernel dispatches.
        /// SynchronizeAsync awaits all of these to ensure all workers have completed.
        /// </summary>
        internal List<Task> PendingWorkTasks { get; } = new List<Task>();

        /// <summary>
        /// Reusable worker pool — lazily initialized on first dispatch.
        /// Workers are created once and reused across kernel dispatches.
        /// </summary>
        private WorkerPool? _workerPool;

        /// <summary>
        /// Method info for the static RunKernel method used by dynamic kernel launchers.
        /// </summary>
        public static readonly MethodInfo RunKernelMethod = typeof(WorkersAccelerator).GetMethod(
            nameof(RunKernel),
            BindingFlags.Public | BindingFlags.Static)!;

        #region Construction

        private WorkersAccelerator(Context context, Device device) : base(context, device) { }

        /// <summary>
        /// Creates a new Workers accelerator with default options.
        /// </summary>
        public static WorkersAccelerator Create(Context context, WorkersILGPUDevice device)
            => Create(context, device, null);

        /// <summary>
        /// Creates a new Workers accelerator with the specified options.
        /// </summary>
        public static WorkersAccelerator Create(
            Context context,
            WorkersILGPUDevice device,
            WorkersBackendOptions? options)
        {
            var opts = options ?? WorkersBackendOptions.Default;
            var accelerator = new WorkersAccelerator(context, device);
            accelerator.WorkersDevice = device;
            accelerator.WorkerCount = opts.MaxWorkerCount > 0
                ? opts.MaxWorkerCount
                : device.HardwareConcurrency;
            accelerator.Backend = new WorkersBackend(context, opts);
            accelerator.Init(accelerator.Backend);
            accelerator.DefaultStream = accelerator.CreateStreamInternal();

            WorkersBackend.Log($"[Workers] Accelerator created: {device.Name}, WorkerCount={accelerator.WorkerCount}");
            return accelerator;
        }

        #endregion

        #region Kernel Management

        /// <inheritdoc/>
        protected override WorkersKernel CreateKernel(WorkersCompiledKernel compiledKernel)
        {
            return new WorkersKernel(this, compiledKernel, null!);
        }

        /// <inheritdoc/>
        protected override WorkersKernel CreateKernel(WorkersCompiledKernel compiledKernel, MethodInfo launcher)
        {
            return new WorkersKernel(this, compiledKernel, launcher);
        }

        /// <inheritdoc/>
        protected override MethodInfo GenerateKernelLauncherMethod(
            WorkersCompiledKernel kernel,
            int customGroupSize)
        {
            var parameters = kernel.EntryPoint.Parameters;
            var indexType = kernel.EntryPoint.KernelIndexType;
            var argTypes = new List<Type> { typeof(Kernel), typeof(AcceleratorStream), indexType };
            for (int i = 0; i < parameters.Count; i++) argTypes.Add(parameters[i]);

            var dynamicMethod = new DynamicMethod(
                "WorkersLauncher",
                typeof(void),
                argTypes.ToArray(),
                typeof(WorkersAccelerator).Module);

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
        /// Executes a Workers kernel. Called by the dynamic launcher.
        /// Dispatches work to Web Workers (fire-and-forget).
        /// - With SharedArrayBuffer: N workers in parallel, zero-copy memory sharing.
        /// - Without SharedArrayBuffer: 1 worker with ArrayBuffer copy to avoid blocking UI.
        /// Completion is tracked via PendingWork, awaited by SynchronizeAsync.
        /// </summary>
        public static void RunKernel(Kernel kernel, AcceleratorStream stream, object dimension, object[] args)
        {
            var workersAccel = (WorkersAccelerator)kernel.Accelerator;
            var workersKernel = (WorkersKernel)kernel;
            var compiledKernel = workersKernel.CompiledKernel;

            WorkersBackend.Log("\n[Workers-Debug] ---- GENERATED JS ----");
            WorkersBackend.Log(compiledKernel.JSSource);
            WorkersBackend.Log("[Workers-Debug] -------------------------\n");

            // Extract grid dimensions for multi-D index decomposition
            var (dimX, dimY, dimZ) = GetGridDimensions(dimension);
            int totalItems = dimX * dimY * dimZ;
            var (groupDimX, groupDimY, groupDimZ) = GetGroupDimensions(dimension);
            WorkersBackend.Log($"[Workers-Debug] Grid dimensions: ({dimX}, {dimY}, {dimZ}), Total: {totalItems}, GroupDim: ({groupDimX}, {groupDimY}, {groupDimZ})");

            // Extract dynamic shared memory size from KernelConfig
            int dynamicSharedElements = 0;
            if (dimension is KernelConfig kConfig)
                dynamicSharedElements = kConfig.SharedMemoryConfig.NumElements;

            // Marshal arguments for JS execution
            var jsArgs = MarshalArguments(compiledKernel, args);

            // Build or reuse cached script body + hash
            bool canUseShared = WorkersMemoryBuffer.IsCrossOriginIsolated;
            bool isShared = canUseShared && workersAccel.WorkerCount > 1;
            bool hasBarriers = compiledKernel.JSSource.Contains("// __HAS_BARRIERS__");
            var scriptKey = $"{isShared}|{dimX}|{dimY}|{dimZ}|{groupDimX}|{groupDimY}|{groupDimZ}|{hasBarriers}|{dynamicSharedElements}";

            string scriptBody;
            string scriptHash;
            if (workersKernel.CachedScriptKey == scriptKey && workersKernel.CachedScriptBody != null)
            {
                scriptBody = workersKernel.CachedScriptBody;
                scriptHash = workersKernel.CachedScriptHash!;
            }
            else
            {
                scriptBody = BuildWorkerScript(compiledKernel.JSSource, jsArgs, isShared, dimX, dimY, dimZ, groupDimX, groupDimY, groupDimZ, hasBarriers, dynamicSharedElements);
                scriptHash = $"wk_{compiledKernel.JSSource.GetHashCode():x8}_{scriptKey.GetHashCode():x8}";
                workersKernel.CachedScriptBody = scriptBody;
                workersKernel.CachedScriptHash = scriptHash;
                workersKernel.CachedScriptKey = scriptKey;
            }

            // Dispatch to workers — fire and forget
            workersAccel.PendingWorkTasks.Add(DispatchToWorkers(
                compiledKernel.JSSource, totalItems, dimX, dimY, dimZ, groupDimX, groupDimY, groupDimZ, jsArgs, workersAccel, dynamicSharedElements, scriptBody, scriptHash));
        }

        /// <summary>
        /// Dispatches kernel work to Web Workers.
        /// With SharedArrayBuffer: splits work across N workers (zero-copy parallel).
        /// Without SharedArrayBuffer: sends all work to 1 worker with zero-copy transferred ArrayBuffers.
        /// For barrier kernels: distributes groups across workers. Each worker processes
        /// all its groups using generators (intra-group sync), then uses Atomics to sync
        /// across workers at each barrier step.
        /// </summary>
        private static Task DispatchToWorkers(
            string jsSource,
            int totalItems,
            int dimX, int dimY, int dimZ,
            int groupDimX, int groupDimY, int groupDimZ,
            List<object?> jsArgs,
            WorkersAccelerator accelerator,
            int dynamicSharedElements = 0,
            string? prebuiltScript = null,
            string? scriptHash = null)
        {
            int workerCount = accelerator.WorkerCount;
            bool canUseShared = WorkersMemoryBuffer.IsCrossOriginIsolated;
            bool isShared = canUseShared && workerCount > 1;
            if (!canUseShared) workerCount = 1;

            // Detect if this kernel uses barriers
            bool hasBarriers = jsSource.Contains("// __HAS_BARRIERS__");

            // For barrier kernels: allocate worker-level barrier SAB
            SharedArrayBuffer? barrierSab = null;
            int groupSize = groupDimX * groupDimY * groupDimZ;
            int numGroups = groupSize > 0 ? totalItems / groupSize : 0;

            if (hasBarriers && isShared && groupSize > 0)
            {
                // Parse barrier count from metadata
                int barrierCount = 1;
                var barrierCountMatch = System.Text.RegularExpressions.Regex.Match(
                    jsSource, @"// __BARRIER_COUNT__\s+(\d+)");
                if (barrierCountMatch.Success)
                    barrierCount = int.Parse(barrierCountMatch.Groups[1].Value);

                // Worker-level barrier: barrierCount barriers, each needs 2 i32 slots (arrival + sense)
                int totalSlots = barrierCount * 2;
                int barrierBytes = totalSlots * 4;
                barrierSab = new SharedArrayBuffer(barrierBytes);

                // Clamp worker count to numGroups (no point having more workers than groups)
                workerCount = Math.Min(workerCount, numGroups);
                if (workerCount < 1) workerCount = 1;

                WorkersBackend.Log($"[Workers] Barrier kernel: {numGroups} groups x {groupSize} threads, {barrierCount} barriers, {workerCount} workers");
            }

            WorkersBackend.Log($"[Workers] Dispatching kernel: {totalItems} items across {workerCount} worker(s), SharedArrayBuffer={isShared}, HasBarriers={hasBarriers}");

            // Use pre-built script from cache if available, otherwise build it
            var scriptBody = prebuiltScript ?? BuildWorkerScript(jsSource, jsArgs, isShared, dimX, dimY, dimZ, groupDimX, groupDimY, groupDimZ, hasBarriers, dynamicSharedElements);

            // Lazily initialize the worker pool, or grow it if needed
            if (accelerator._workerPool == null)
                accelerator._workerPool = new WorkerPool(workerCount, useAsync: false);
            else
                accelerator._workerPool.EnsureSize(workerCount, useAsync: false);

            var workers = accelerator._workerPool.Acquire(workerCount);
            // If pool didn't have enough idle workers, fall back to creating temporary ones
            if (workers.Count < workerCount)
            {
                var shortfall = workerCount - workers.Count;
                accelerator._workerPool.EnsureSize(accelerator._workerPool.Size + shortfall, useAsync: false);
                var extra = accelerator._workerPool.Acquire(shortfall);
                workers.AddRange(extra);
            }
            var tasks = new List<Task>();

            if (hasBarriers && isShared && groupSize > 0)
            {
                // Group distribution: each worker gets a contiguous range of groups
                int groupsPerWorker = numGroups / workerCount;
                int remainder = numGroups % workerCount;

                for (int w = 0; w < workerCount; w++)
                {
                    var worker = workers[w];
                    int startGroup = w * groupsPerWorker + Math.Min(w, remainder);
                    int endGroup = startGroup + groupsPerWorker + (w < remainder ? 1 : 0);

                    DispatchSingleWorker(worker, w, jsArgs, startGroup, endGroup, isShared, tasks, accelerator._workerPool,
                        scriptBody, barrierSab, groupSize, workerCount, scriptHash);
                }
            }
            else
            {
                // Flat distribution: items are distributed evenly
                int itemsPerWorker = totalItems / workerCount;
                int remainder = totalItems % workerCount;

                for (int w = 0; w < workerCount; w++)
                {
                    var worker = workers[w];
                    int startIdx = w * itemsPerWorker + Math.Min(w, remainder);
                    int endIdx = startIdx + itemsPerWorker + (w < remainder ? 1 : 0);

                    DispatchSingleWorker(worker, w, jsArgs, startIdx, endIdx, isShared, tasks, accelerator._workerPool,
                        scriptBody, scriptHash: scriptHash);
                }
            }

            var allTask = Task.WhenAll(tasks);
            if (barrierSab != null)
            {
                allTask = allTask.ContinueWith(_ => barrierSab.Dispose(), TaskScheduler.Default);
            }
            return allTask;
        }

        /// <summary>
        /// Dispatches a single worker with the given start/end range.
        /// For barrier kernels: startIdx/endIdx are group indices, plus groupSize and workerCount.
        /// For non-barrier: startIdx/endIdx are flat item indices.
        /// </summary>
        private static void DispatchSingleWorker(
            Worker worker, int workerIndex, List<object?> jsArgs,
            int startIdx, int endIdx, bool isShared, List<Task> tasks,
            WorkerPool pool, string scriptBody,
            SharedArrayBuffer? barrierSab = null, int groupSize = 0, int workerCount = 0,
            string? scriptHash = null)
        {
            var tcs = new TaskCompletionSource();
            Action<MessageEvent>? msgHandler = null;
            Action<Event>? errHandler = null;
            int w = workerIndex;

            msgHandler = new Action<MessageEvent>((msg) =>
            {
                worker.OnMessage -= msgHandler!;
                worker.OnError -= errHandler!;
                pool.Return(worker);

                var done = msg.JSRef!.Get<bool>("data.done");
                if (!done)
                {
                    var errorMsg = msg.JSRef!.Get<string?>("data.error") ?? "Unknown worker script error";
                    tcs.TrySetException(new Exception($"[Workers] Worker {w} JS error: {errorMsg}"));
                    return;
                }

                if (!isShared)
                {
                    ReceiveTransferredBuffers(msg, jsArgs);
                }

                tcs.TrySetResult();
            });

            errHandler = new Action<Event>((err) =>
            {
                worker.OnMessage -= msgHandler!;
                worker.OnError -= errHandler!;
                pool.Return(worker);
                tcs.TrySetException(new Exception($"[Workers] Worker {w} error during kernel execution"));
            });

            worker.OnMessage += msgHandler;
            worker.OnError += errHandler;

            var (messageArgs, transferList) = BuildWorkerMessage(jsArgs, startIdx, endIdx, isShared,
                scriptBody, barrierSab, groupSize, workerCount, scriptHash);
            if (transferList != null)
            {
                worker.PostMessage(messageArgs, transferList);
            }
            else
            {
                worker.PostMessage(messageArgs);
            }

            tasks.Add(tcs.Task);
        }

        /// <summary>
        /// Builds the JS script that runs inside each worker.
        /// The worker receives a message with: { startIdx, endIdx, args: [...] }
        /// where each arg is either { sab, arrayType, byteOffset, elementCount } for buffers
        /// or { value } for scalars.
        /// For barrier kernels: each worker processes a range of groups using
        /// generator/yield lockstep (intra-group), then uses Atomics for
        /// worker-level synchronization at each barrier step.
        /// </summary>
        private static string BuildWorkerScript(string jsSource, List<object?> jsArgs, bool isShared, int dimX, int dimY, int dimZ, int groupDimX, int groupDimY, int groupDimZ, bool hasBarriers, int dynamicSharedElements = 0)
        {
            // Produces a function body string that is sent as the 'script' field
            // in the message to the pool worker's universal bootstrap.
            // The bootstrap calls: new Function('d', scriptBody)(data)
            var sb = new System.Text.StringBuilder();
            sb.AppendLine("  var startIdx = d.startIdx;");
            sb.AppendLine("  var endIdx = d.endIdx;");
            sb.AppendLine();

            // Emit kernel function
            // For barrier kernels: strip the shared memory tagged block from jsSource
            // (it will be emitted inside the group loop instead)
            var cleanedSource = jsSource;
            if (hasBarriers)
            {
                var startMarker = "// __SHARED_DECLS_START__";
                var endMarker = "// __SHARED_DECLS_END__";
                int sIdx = cleanedSource.IndexOf(startMarker);
                int eIdx = cleanedSource.IndexOf(endMarker);
                if (sIdx >= 0 && eIdx > sIdx)
                {
                    cleanedSource = cleanedSource.Substring(0, sIdx) +
                        cleanedSource.Substring(eIdx + endMarker.Length);
                }
            }
            sb.AppendLine("  // Kernel function");
            sb.AppendLine("  " + cleanedSource.Replace("\n", "\n  "));
            sb.AppendLine();
            sb.AppendLine();

            // Create typed array views from args
            sb.AppendLine("  // Create parameter views");
            var callArgs = new List<string>();
            int argIdx = 0;
            for (int a = 0; a < jsArgs.Count; a++)
            {
                var arg = jsArgs[a];
                if (arg is BufferArg bufArg)
                {
                    var varName = $"_p{argIdx}";
                    var arrayType = GetTypedArrayName(bufArg.ElementType, bufArg.ElementSize);
                    sb.AppendLine($"  var {varName} = new {arrayType}(d.args[{argIdx}].buffer, d.args[{argIdx}].byteOffset, d.args[{argIdx}].elementCount);");
                    callArgs.Add(varName);
                }
                else
                {

                    callArgs.Add($"d.args[{argIdx}].value");
                }
                argIdx++;
            }

            sb.AppendLine();
            sb.AppendLine("  // Execute kernel for assigned range");
            sb.AppendLine("  try {");

            if (hasBarriers)
            {
                // Worker-level barrier execution:
                // 1. startIdx/endIdx are GROUP indices (not item indices)
                // 2. For each group, create generators for all threads, step in lockstep
                // 3. After all groups complete one yield step → Atomics worker-level barrier
                // 4. Repeat until all generators are done
                int groupSize = groupDimX * groupDimY * groupDimZ;

                // Extract shared memory declarations from the tagged block
                string sharedDecls = "";
                var sharedStartMarker = "// __SHARED_DECLS_START__";
                var sharedEndMarker = "// __SHARED_DECLS_END__";
                int sharedStart = jsSource.IndexOf(sharedStartMarker);
                int sharedEnd = jsSource.IndexOf(sharedEndMarker);
                if (sharedStart >= 0 && sharedEnd > sharedStart)
                {
                    sharedDecls = jsSource.Substring(
                        sharedStart + sharedStartMarker.Length,
                        sharedEnd - sharedStart - sharedStartMarker.Length).Trim();

                    if (dynamicSharedElements > 0)
                        sharedDecls = sharedDecls.Replace("__DYNSHARED_SIZE__", dynamicSharedElements.ToString());
                }

                sb.AppendLine($"    var _groupSize = {groupSize};");
                sb.AppendLine($"    var _barrierBuf = d.barrierBuf;");
                sb.AppendLine($"    var _workerCount = d.workerCount;");
                sb.AppendLine();

                // Build arrays of generators for all groups assigned to this worker
                sb.AppendLine($"    // Create generators for all assigned groups");
                sb.AppendLine($"    var _groupGens = [];");
                sb.AppendLine($"    for (var _g = startIdx; _g < endIdx; _g++) {{");

                // Emit shared memory declarations inside group loop
                if (!string.IsNullOrEmpty(sharedDecls))
                {
                    sb.AppendLine($"      // Shared memory (reset per group)");
                    foreach (var line in sharedDecls.Split('\n', StringSplitOptions.RemoveEmptyEntries))
                    {
                        sb.AppendLine($"      {line.Trim()}");
                    }
                }

                sb.AppendLine($"      var _gens = [];");
                sb.AppendLine($"      for (var _li = 0; _li < _groupSize; _li++) {{");

                // Build kernel call — globalIndex = group * groupSize + localIndex
                sb.Append($"        _gens.push(kernel(_g * _groupSize + _li, {dimX}, {dimY}, {dimZ}, {groupDimX}, {groupDimY}, {groupDimZ}");
                foreach (var name in callArgs)
                {
                    sb.Append(", ");
                    sb.Append(name);
                }
                sb.AppendLine("));");
                sb.AppendLine($"      }}");
                sb.AppendLine($"      _groupGens.push(_gens);");
                sb.AppendLine($"    }}");
                sb.AppendLine();

                // Step all groups' generators in lockstep, with worker-level atomics sync
                sb.AppendLine($"    // Step generators: intra-group lockstep + inter-worker atomics");
                sb.AppendLine($"    var _barrierIdx = 0;");
                sb.AppendLine($"    var _allDone = false;");
                sb.AppendLine($"    while (!_allDone) {{");
                sb.AppendLine($"      _allDone = true;");
                // Step each group's generators one step
                sb.AppendLine($"      for (var _gi = 0; _gi < _groupGens.length; _gi++) {{");
                sb.AppendLine($"        var _gens = _groupGens[_gi];");
                sb.AppendLine($"        for (var _ti = 0; _ti < _gens.length; _ti++) {{");
                sb.AppendLine($"          if (!_gens[_ti].next().done) _allDone = false;");
                sb.AppendLine($"        }}");
                sb.AppendLine($"      }}");
                sb.AppendLine();
                // Worker-level atomics barrier (sense-reversing)
                sb.AppendLine($"      if (!_allDone && _workerCount > 1) {{");
                sb.AppendLine($"        // Worker-level sync: wait for all workers to reach this barrier step");
                sb.AppendLine($"        var _bSlot = _barrierIdx * 2;");
                sb.AppendLine($"        var _arrived = Atomics.add(_barrierBuf, _bSlot, 1) + 1;");
                sb.AppendLine($"        if (_arrived === _workerCount) {{");
                sb.AppendLine($"          // Last worker: reset counter and flip sense");
                sb.AppendLine($"          Atomics.store(_barrierBuf, _bSlot, 0);");
                sb.AppendLine($"          Atomics.store(_barrierBuf, _bSlot + 1, 1 - Atomics.load(_barrierBuf, _bSlot + 1));");
                sb.AppendLine($"          Atomics.notify(_barrierBuf, _bSlot + 1);");
                sb.AppendLine($"        }} else {{");
                sb.AppendLine($"          // Wait for last worker to flip sense");
                sb.AppendLine($"          var _sense = Atomics.load(_barrierBuf, _bSlot + 1);");
                sb.AppendLine($"          Atomics.wait(_barrierBuf, _bSlot + 1, _sense);");
                sb.AppendLine($"        }}");
                sb.AppendLine($"        _barrierIdx++;");
                sb.AppendLine($"      }}");
                sb.AppendLine($"    }}");
            }
            else
            {
                // Standard flat loop (no barriers)
                sb.Append($"    for (var _i = startIdx; _i < endIdx; _i++) {{ kernel(_i, {dimX}, {dimY}, {dimZ}, {groupDimX}, {groupDimY}, {groupDimZ}");
                foreach (var name in callArgs)
                {
                    sb.Append(", ");
                    sb.Append(name);
                }
                sb.AppendLine("); }");
            }

            if (!isShared)
            {
                sb.AppendLine("    // Transfer modified buffers back (zero-copy)");
                sb.AppendLine("    var resultBuffers = [];");
                sb.AppendLine("    var transferList = [];");
                argIdx = 0;
                for (int a = 0; a < jsArgs.Count; a++)
                {
                    if (jsArgs[a] is BufferArg)
                    {
                        sb.AppendLine($"    resultBuffers.push({{ index: {argIdx}, buffer: _p{argIdx}.buffer }});");
                        sb.AppendLine($"    transferList.push(_p{argIdx}.buffer);");
                    }
                    argIdx++;
                }
                sb.AppendLine("    self.postMessage({ done: true, buffers: resultBuffers }, transferList);");
            }
            else
            {
                sb.AppendLine("    self.postMessage({ done: true });");
            }

            sb.AppendLine("  } catch(ex) {");
            sb.AppendLine("    self.postMessage({ done: false, error: (ex && ex.message) ? ex.message : String(ex) });");
            sb.AppendLine("  }");
            return sb.ToString();
        }

        /// <summary>
        /// Builds the message object sent to each worker via PostMessage.
        /// Returns both the message and a transfer list for non-shared ArrayBuffers.
        /// For barrier kernels: includes barrierBuf (Int32Array on SAB), groupSize, workerCount.
        /// </summary>
        private static (object message, object[]? transferList) BuildWorkerMessage(
            List<object?> jsArgs, int startIdx, int endIdx, bool isShared,
            string scriptBody,
            SharedArrayBuffer? barrierSab = null, int groupSize = 0, int workerCount = 0,
            string? scriptHash = null)
        {
            var args = new List<object>();
            var transfers = isShared ? null : new List<object>();
            for (int a = 0; a < jsArgs.Count; a++)
            {
                var arg = jsArgs[a];
                if (arg is BufferArg bufArg)
                {
                    var underlyingBuffer = bufArg.MemoryBuffer.UnderlyingBuffer;
                    // Pass the underlying buffer (SharedArrayBuffer or ArrayBuffer)
                    args.Add(new
                    {
                        buffer = underlyingBuffer,
                        byteOffset = bufArg.ByteOffset,
                        elementCount = bufArg.LengthInBytes / bufArg.ElementSize,
                        arrayType = GetTypedArrayName(bufArg.ElementType, bufArg.ElementSize)
                    });
                    // For non-shared mode, add buffers to transfer list (zero-copy)
                    transfers?.Add(underlyingBuffer);
                }
                else
                {
                    args.Add(new { value = arg });
                }
            }

            if (barrierSab != null)
            {
                var barrierView = new Int32Array(barrierSab);
                var message = new { script = scriptBody, scriptHash, startIdx, endIdx, args = args.ToArray(), barrierBuf = barrierView, workerCount };
                return (message, transfers?.ToArray());
            }
            else
            {
                var message = new { script = scriptBody, scriptHash, startIdx, endIdx, args = args.ToArray() };
                return (message, transfers?.ToArray());
            }
        }

        /// <summary>
        /// Receives transferred ArrayBuffers back from a worker and replaces the
        /// underlying buffer references in WorkersMemoryBuffer (zero-copy).
        /// Used only in non-SharedArrayBuffer mode.
        /// </summary>
        private static void ReceiveTransferredBuffers(MessageEvent msg, List<object?> jsArgs)
        {
            try
            {
                // The worker posts back { done: true, buffers: [{ index, buffer }, ...] }
                using var data = msg.GetData<JSObject>();
                using var buffersArr = data.JSRef!.Get<Array<JSObject>?>("buffers");
                if (buffersArr == null) return;

                for (int i = 0; i < buffersArr.Length; i++)
                {
                    using var bufObj = buffersArr[i];
                    var argIndex = bufObj.JSRef!.Get<int>("index");
                    // Get the transferred ArrayBuffer (do NOT dispose — we're taking ownership)
                    var transferredBuffer = bufObj.JSRef!.Get<ArrayBuffer>("buffer");

                    // Replace the underlying ArrayBuffer in the WorkersMemoryBuffer
                    if (argIndex < jsArgs.Count && jsArgs[argIndex] is BufferArg bufArg)
                    {
                        bufArg.MemoryBuffer.ReplaceArrayBuffer(transferredBuffer);
                    }
                }
            }
            catch (Exception ex)
            {
                WorkersBackend.Log($"[Workers] Error receiving transferred buffers: {ex.Message}");
            }
        }

        /// <summary>
        /// Extracts grid dimensions from the dimension object.
        /// Returns (dimX, dimY, dimZ) - for 1D, dimY=dimZ=1.
        /// </summary>
        private static (int dimX, int dimY, int dimZ) GetGridDimensions(object dimension)
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

        /// <summary>
        /// Extracts group dimensions from a dimension/KernelConfig.
        /// For auto-grouped kernels (Index types), group dim defaults to the total dim (single group).
        /// For stream kernels (KernelConfig), returns the explicit group dimensions.
        /// </summary>
        private static (int groupDimX, int groupDimY, int groupDimZ) GetGroupDimensions(object dimension)
        {
            if (dimension is KernelConfig config)
                return (config.GroupDim.X, config.GroupDim.Y, config.GroupDim.Z);
            // Auto-grouped kernels: each work item is its own "group" — group dim = total dim
            var (dimX, dimY, dimZ) = GetGridDimensions(dimension);
            return (dimX, dimY, dimZ);
        }

        /// <summary>
        /// Marshals kernel arguments from C#/ILGPU objects to JS-compatible values.
        /// </summary>
        private static List<object?> MarshalArguments(WorkersCompiledKernel compiledKernel, object[] args)
        {
            var jsArgs = new List<object?>();
            var parameters = compiledKernel.EntryPoint.Parameters;

            for (int i = 0; i < args.Length; i++)
            {
                var paramType = parameters[i];

                // Skip implicit index parameter
                if (i == 0 && (paramType == typeof(Index1D) || paramType == typeof(Index2D) ||
                               paramType == typeof(Index3D) || paramType == typeof(LongIndex1D) ||
                               paramType == typeof(LongIndex2D) || paramType == typeof(LongIndex3D)))
                    continue;

                var arg = args[i];

                // Check if it's an ArrayView (buffer)
                IArrayView? arrayView = arg as IArrayView;
                if (arrayView == null && arg != null)
                {
                    var argType = arg.GetType();
                    var baseViewProp = argType.GetProperty("BaseView");
                    if (baseViewProp != null)
                        arrayView = baseViewProp.GetValue(arg) as IArrayView;
                }

                if (arrayView != null)
                {
                    // Get the contiguous view to access the underlying buffer
                    var contiguous = arrayView as IContiguousArrayView;
                    if (contiguous == null)
                    {
                        var baseViewProp = arrayView.GetType().GetProperty("BaseView");
                        contiguous = (baseViewProp != null ? baseViewProp.GetValue(arrayView) : arrayView) as IContiguousArrayView;
                    }

                    if (contiguous == null)
                        throw new InvalidOperationException($"Argument {i} is not a contiguous buffer");

                    var memBuffer = contiguous.Buffer as WorkersMemoryBuffer;
                    if (memBuffer == null)
                        throw new InvalidOperationException($"Argument {i} buffer is not a WorkersMemoryBuffer");

                    // Determine the element type from the view's generic arguments
                    Type elementType = typeof(int); // default
                    int elementSize = contiguous.ElementSize;
                    var viewType = arrayView.GetType();
                    if (viewType.IsGenericType)
                    {
                        elementType = viewType.GetGenericArguments()[0];

                        // For struct element types (non-primitive), we need to treat the buffer
                        // as a flat array of the underlying primitive type. The JS kernel codegen
                        // uses strided access to read/write struct fields from the flat array.
                        if (!elementType.IsPrimitive && elementType.IsValueType)
                        {
                            // Determine the primitive type used in the struct fields
                            var fields = elementType.GetFields(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance);
                            if (fields.Length > 0)
                            {
                                // Use the first field's type as the primitive element type
                                // (assumes homogeneous struct or at least same-size fields)
                                var primitiveFieldType = GetLeafFieldType(fields[0].FieldType);
                                elementType = primitiveFieldType;
                                elementSize = System.Runtime.InteropServices.Marshal.SizeOf(primitiveFieldType);
                            }
                        }
                    }

                    // Pass the shared buffer info for worker access
                    jsArgs.Add(new BufferArg
                    {
                        MemoryBuffer = memBuffer,
                        ByteOffset = (int)contiguous.IndexInBytes,
                        LengthInBytes = (int)contiguous.LengthInBytes,
                        ElementSize = elementSize,
                        ElementType = elementType
                    });

                    WorkersBackend.Log($"[Workers-Debug] Arg {i}: Buffer<{elementType.Name}>, Offset={contiguous.IndexInBytes}, Size={contiguous.LengthInBytes}");

                    // For multi-D views (ArrayView2D/3D), ILGPU IR decomposes the view
                    // into a flat ArrayView + stride struct. We need to add the stride as
                    // additional scalar argument(s) to match the IR parameter count.
                    if (arg != null)
                    {
                        var argType = arg.GetType();
                        var argTypeName = argType.Name;

                        // ArrayView2D: add YStride (= width for DenseX)
                        if (argTypeName.StartsWith("ArrayView2D"))
                        {
                            // Try to get the stride from the view's Stride property
                            var strideProp = argType.GetProperty("Stride");
                            if (strideProp != null)
                            {
                                var strideObj = strideProp.GetValue(arg);
                                if (strideObj != null)
                                {
                                    // Stride2D.DenseX has a YStride field
                                    var yStrideProp = strideObj.GetType().GetProperty("YStride");
                                    if (yStrideProp != null)
                                    {
                                        var yStride = (int)yStrideProp.GetValue(strideObj)!;
                                        jsArgs.Add(yStride);
                                        WorkersBackend.Log($"[Workers-Debug] Arg {i}: ArrayView2D stride YStride={yStride}");
                                    }
                                    else
                                    {
                                        // Fallback: try XStride for other stride types
                                        var xStrideProp = strideObj.GetType().GetProperty("XStride");
                                        if (xStrideProp != null)
                                        {
                                            var xStride = (int)xStrideProp.GetValue(strideObj)!;
                                            jsArgs.Add(xStride);
                                            WorkersBackend.Log($"[Workers-Debug] Arg {i}: ArrayView2D stride XStride={xStride}");
                                        }
                                    }
                                }
                            }
                        }
                        // ArrayView3D: add YStride and ZStride
                        else if (argTypeName.StartsWith("ArrayView3D"))
                        {
                            var strideProp = argType.GetProperty("Stride");
                            if (strideProp != null)
                            {
                                var strideObj = strideProp.GetValue(arg);
                                if (strideObj != null)
                                {
                                    var yStrideProp = strideObj.GetType().GetProperty("YStride");
                                    var zStrideProp = strideObj.GetType().GetProperty("ZStride");
                                    if (yStrideProp != null)
                                    {
                                        jsArgs.Add((int)yStrideProp.GetValue(strideObj)!);
                                        WorkersBackend.Log($"[Workers-Debug] Arg {i}: ArrayView3D stride YStride={yStrideProp.GetValue(strideObj)}");
                                    }
                                    if (zStrideProp != null)
                                    {
                                        jsArgs.Add((int)zStrideProp.GetValue(strideObj)!);
                                        WorkersBackend.Log($"[Workers-Debug] Arg {i}: ArrayView3D stride ZStride={zStrideProp.GetValue(strideObj)}");
                                    }
                                }
                            }
                        }
                    }
                }
                else if (arg != null && arg.GetType().IsValueType && !arg.GetType().IsPrimitive && !arg.GetType().IsEnum)
                {
                    // Struct scalar: convert to JS-compatible object with f0, f1, f2... keys
                    // matching the ILGPU IR's GetField access pattern (param.f0, param.f0.f1, etc.)
                    jsArgs.Add(ConvertStructToJSObject(arg));
                    WorkersBackend.Log($"[Workers-Debug] Arg {i}: Struct, converted {arg.GetType().Name} to f{{N}} object");
                }
                else
                {
                    // Primitive scalar argument — pass directly
                    jsArgs.Add(arg);
                    WorkersBackend.Log($"[Workers-Debug] Arg {i}: Scalar, Value={arg}");
                }
            }

            return jsArgs;
        }

        /// <summary>
        /// Recursively flattens a C# struct value into a single-level Dictionary
        /// with sequential f0, f1, f2... keys for all leaf primitive fields.
        /// ILGPU IR flattens nested structs: NestedOuterStruct { NestedInnerStruct { A, B }, Value }
        /// becomes 3 flat fields: f0=A, f1=B, f2=Value (NOT f0={f0:A,f1:B}, f1=Value).
        /// </summary>
        private static object ConvertStructToJSObject(object structValue)
        {
            var result = new Dictionary<string, object?>();
            int fieldCounter = 0;
            FlattenStructFields(structValue, result, ref fieldCounter);
            return result;
        }

        /// <summary>
        /// Recursively extracts all leaf primitive fields from a struct and adds them
        /// to the result dictionary with sequential f{N} keys.
        /// </summary>
        private static void FlattenStructFields(object structValue, Dictionary<string, object?> result, ref int fieldCounter)
        {
            var fields = structValue.GetType().GetFields(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance);
            foreach (var field in fields)
            {
                var fieldVal = field.GetValue(structValue);
                if (fieldVal != null && fieldVal.GetType().IsValueType && !fieldVal.GetType().IsPrimitive && !fieldVal.GetType().IsEnum)
                {
                    // Nested struct — recurse (flatten into same level)
                    FlattenStructFields(fieldVal, result, ref fieldCounter);
                }
                else
                {
                    // Leaf primitive — add with sequential index
                    result[$"f{fieldCounter}"] = fieldVal;
                    fieldCounter++;
                }
            }
        }

        /// <summary>
        /// MVP: Executes the kernel on the main thread using JS eval.
        /// This is a synchronous fallback — future versions will dispatch across Web Workers.
        ///
        /// Strategy:
        /// 1. Set SharedArrayBuffers on globalThis as temporary globals
        /// 2. Build a single IIFE script that creates typed array views + runs the loop
        /// 3. Eval the script
        /// 4. Clean up the temporary globals
        /// </summary>
        private static void ExecuteKernelOnMainThread(
            string jsSource,
            int totalItems,
            List<object?> jsArgs,
            WorkersAccelerator accelerator)
        {
            WorkersBackend.Log($"[Workers] Executing kernel on main thread: {totalItems} items");

            // Phase 1: Set buffer objects on globalThis so the eval'd script can access them
            // (SharedArrayBuffer when COI available, regular ArrayBuffer otherwise)
            var sabGlobals = new List<string>();
            for (int a = 0; a < jsArgs.Count; a++)
            {
                if (jsArgs[a] is BufferArg bufArg)
                {
                    var globalName = $"_wk_sab{a}";
                    BlazorJSRuntime.JS.Set(globalName, bufArg.MemoryBuffer.UnderlyingBuffer);
                    sabGlobals.Add(globalName);
                }
            }

            try
            {
                // Phase 2: Build a single IIFE script
                var sb = new System.Text.StringBuilder();
                sb.AppendLine("(function() {");

                // Emit the kernel function definition
                sb.AppendLine(jsSource);
                sb.AppendLine();

                // Create typed array views for buffer args, embed scalar literals inline
                var callArgs = new List<string>();
                for (int a = 0; a < jsArgs.Count; a++)
                {
                    var arg = jsArgs[a];
                    if (arg is BufferArg bufArg)
                    {
                        var varName = $"_p{a}";
                        var arrayType = GetTypedArrayName(bufArg.ElementType, bufArg.ElementSize);
                        var elementCount = bufArg.LengthInBytes / bufArg.ElementSize;
                        sb.AppendLine($"  var {varName} = new {arrayType}(globalThis._wk_sab{a}, {bufArg.ByteOffset}, {elementCount});");
                        callArgs.Add(varName);
                    }
                    else
                    {
                        // Scalar: embed directly as a literal in the function call
                        callArgs.Add(FormatScalarLiteral(arg));
                    }
                }

                // Emit the execution loop: call kernel(_i, arg0, arg1, ...) for each work item
                sb.Append("  for (var _i = 0; _i < ");
                sb.Append(totalItems);
                sb.Append("; _i++) { kernel(_i");
                foreach (var name in callArgs)
                {
                    sb.Append(", ");
                    sb.Append(name);
                }
                sb.AppendLine("); }");
                sb.AppendLine("})();");

                var script = sb.ToString();
                WorkersBackend.Log($"[Workers-Debug] Execution script:\n{script}");

                // Phase 3: Execute via eval
                BlazorJSRuntime.JS.CallVoid("eval", script);
            }
            catch (Exception ex)
            {
                WorkersBackend.Log($"[Workers] Error executing kernel: {ex}");
                throw;
            }
            finally
            {
                // Phase 4: Clean up temporary globals
                foreach (var globalName in sabGlobals)
                {
                    try { BlazorJSRuntime.JS.CallVoid("eval", $"delete globalThis.{globalName}"); }
                    catch { /* ignore cleanup errors */ }
                }
            }
        }

        /// <summary>
        /// Recursively gets the leaf (primitive) field type from a struct type.
        /// For MyPoint { float X, float Y }, returns typeof(float).
        /// For nested structs, drills into the first field.
        /// </summary>
        private static Type GetLeafFieldType(Type type)
        {
            if (type.IsPrimitive) return type;
            if (type.IsValueType && !type.IsEnum)
            {
                var fields = type.GetFields(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance);
                if (fields.Length > 0)
                    return GetLeafFieldType(fields[0].FieldType);
            }
            return type;
        }

        /// <summary>
        /// Gets the JS TypedArray constructor name for a given element type.
        /// Uses the actual C# type to distinguish int vs float at the same byte size.
        /// </summary>
        private static string GetTypedArrayName(Type elementType, int fallbackElementSize)
        {
            if (elementType == typeof(float)) return "Float32Array";
            if (elementType == typeof(double)) return "Float64Array";
            if (elementType == typeof(int) || elementType == typeof(uint)) return "Int32Array";
            if (elementType == typeof(short) || elementType == typeof(ushort)) return "Int16Array";
            if (elementType == typeof(byte) || elementType == typeof(sbyte)) return "Uint8Array";
            if (elementType == typeof(long)) return "BigInt64Array";
            if (elementType == typeof(ulong)) return "BigUint64Array";

            // Fallback by size
            return fallbackElementSize switch
            {
                1 => "Uint8Array",
                2 => "Int16Array",
                4 => "Int32Array",
                8 => "Float64Array",
                _ => "Int32Array"
            };
        }

        /// <summary>
        /// Formats a scalar value as a JS literal.
        /// </summary>
        private static string FormatScalarLiteral(object? value)
        {
            return value switch
            {
                null => "0",
                int i => i.ToString(),
                uint u => u.ToString(),
                float f => f.ToString("G9"),
                double d => d.ToString("G17"),
                long l => l.ToString(),
                ulong ul => ul.ToString(),
                byte b => b.ToString(),
                bool bl => bl ? "1" : "0",
                short s => s.ToString(),
                ushort us => us.ToString(),
                _ => value.ToString() ?? "0"
            };
        }

        /// <summary>
        /// Internal class to hold buffer argument info during marshaling.
        /// </summary>
        private class BufferArg
        {
            public WorkersMemoryBuffer MemoryBuffer { get; set; } = null!;
            public int ByteOffset { get; set; }
            public int LengthInBytes { get; set; }
            public int ElementSize { get; set; }
            public Type ElementType { get; set; } = typeof(int);
        }

        /// <summary>
        /// Cached metadata from reflection for a kernel parameter.
        /// Used to avoid repeated reflection in MarshalArguments.
        /// </summary>
        internal class MarshalMetadata
        {
            public bool IsBuffer { get; set; }
            public Type? ElementType { get; set; }
            public int ElementSize { get; set; }
            public bool IsMultiDView { get; set; }
            public string? ViewTypeName { get; set; }
        }

        #endregion

        #region Accelerator Infrastructure

        protected override MemoryBuffer AllocateRawInternal(long length, int elementSize) =>
            new WorkersMemoryBuffer(this, length, elementSize);

        protected override AcceleratorStream CreateStreamInternal() => new WorkersStream(this);

        protected override void SynchronizeInternal()
        {
            // Synchronous synchronization is not reliable in Blazor WASM when using Web Workers.
            // Use SynchronizeAsync() instead.
            if (PendingWorkTasks.Count > 0)
            {
                WorkersBackend.Log("[Workers Warning] SynchronizeInternal called with pending async work. Use 'await accelerator.SynchronizeAsync()' instead.");
            }
        }

        protected override void OnBind() { }
        protected override void OnUnbind() { }

        protected override void DisposeAccelerator_SyncRoot(bool disposing)
        {
            if (disposing)
            {
                PendingWorkTasks.Clear();
                _workerPool?.Dispose();
                _workerPool = null;
            }
        }

        public override TExtension CreateExtension<TExtension, TExtensionProvider>(TExtensionProvider provider) => default!;

        protected override PageLockScope<T> CreatePageLockFromPinnedInternal<T>(IntPtr ptr, long numElements) =>
            throw new NotSupportedException("Page locking is not supported in Workers backend.");

        protected override int EstimateGroupSizeInternal(Kernel kernel, int dynamicSharedMemorySize, int maxGridSize, out int groupSize)
        {
            groupSize = WorkerCount;
            return WorkerCount;
        }

        protected override int EstimateGroupSizeInternal(Kernel kernel, Func<int, int> computeSharedMemorySize, int maxGridSize, out int groupSize)
        {
            groupSize = WorkerCount;
            return WorkerCount;
        }

        protected override int EstimateMaxActiveGroupsPerMultiprocessorInternal(Kernel kernel, int groupSize, int dynamicSharedMemorySize) => 1;

        protected override void EnablePeerAccessInternal(Accelerator other) { }
        protected override void DisablePeerAccessInternal(Accelerator other) { }
        protected override bool CanAccessPeerInternal(Accelerator other) => false;

        #endregion

        #region Nested Types

        /// <summary>
        /// Minimal accelerator stream for Workers.
        /// </summary>
        private class WorkersStream : AcceleratorStream
        {
            public WorkersStream(Accelerator acc) : base(acc) { }
            protected override void DisposeAcceleratorObject(bool disposing) { }
            public override void Synchronize() { }
            protected override global::ILGPU.Runtime.ProfilingMarker AddProfilingMarkerInternal() =>
                throw new NotSupportedException();
        }

        #endregion
    }

    /// <summary>
    /// Represents a compiled Workers kernel ready for execution.
    /// </summary>
    public class WorkersKernel : Kernel
    {
        /// <summary>
        /// Creates a new Workers kernel instance.
        /// </summary>
        public WorkersKernel(Accelerator accelerator, CompiledKernel compiledKernel, MethodInfo launcher)
            : base(accelerator, compiledKernel, launcher) { }

        /// <summary>
        /// Gets the Workers-specific compiled kernel.
        /// </summary>
        public new WorkersCompiledKernel CompiledKernel => (WorkersCompiledKernel)base.CompiledKernel;

        /// <summary>
        /// Cached worker script body, keyed by (isShared, dims, hasBarriers, dynShared).
        /// Avoids rebuilding the StringBuilder script on every dispatch for the same kernel.
        /// </summary>
        internal string? CachedScriptBody { get; set; }

        /// <summary>
        /// Hash of the cached script body, sent to workers for function cache lookup.
        /// </summary>
        internal string? CachedScriptHash { get; set; }

        /// <summary>
        /// Cache key for the current cached script (encodes dimensions, shared mode, etc.)
        /// </summary>
        internal string? CachedScriptKey { get; set; }

        /// <summary>
        /// Cached argument metadata from reflection (Types, element sizes, stride info).
        /// Avoids repeating reflection on every dispatch for the same kernel.
        /// </summary>
        internal WorkersAccelerator.MarshalMetadata[]? CachedMarshalMeta { get; set; }

        /// <inheritdoc/>
        protected override void DisposeAcceleratorObject(bool disposing) { }
    }
}
