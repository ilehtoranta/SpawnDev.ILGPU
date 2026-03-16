// ---------------------------------------------------------------------------------------
//                               SpawnDev.ILGPU.Wasm
//                    WebAssembly Compute Backend for Blazor WebAssembly
//
// File: WasmBackend.cs
//
// The ILGPU backend that compiles IR to WebAssembly binary modules.
// Mirrors WorkersBackend but emits Wasm instead of JavaScript.
// ---------------------------------------------------------------------------------------

using global::ILGPU;
using global::ILGPU.Backends;
using global::ILGPU.Backends.EntryPoints;
using global::ILGPU.IR;
using global::ILGPU.IR.Analyses;
using global::ILGPU.IR.Intrinsics;
using global::ILGPU.IR.Values;
using global::ILGPU.Runtime;
using System.IO;
using System.Reflection;
using System.Text;

namespace SpawnDev.ILGPU.Wasm.Backend
{
    /// <summary>
    /// WebAssembly backend for ILGPU.
    /// Compiles ILGPU IR to WebAssembly binary modules for execution in Web Workers.
    /// </summary>
    public class WasmBackend : CodeGeneratorBackend<
        WasmIntrinsicHandler,
        WasmCodeGenerator.GeneratorArgs,
        WasmCodeGenerator,
        StringBuilder>
    {
        #region Static

        /// <summary>
        /// Backend type ID for Wasm (custom enum value).
        /// </summary>
        public static readonly BackendType BackendTypeWasm = BackendType.Wasm;

        /// <summary>
        /// Controls verbose debug logging.
        /// </summary>
        public static bool VerboseLogging { get; set; } = false;

        /// <summary>
        /// When set, dumps generated Wasm binaries to this directory. Desktop only.
        /// </summary>
        public static string? WasmDumpPath { get; set; }

        /// <summary>Diagnostic: info about all compiled kernels.</summary>
        public static readonly List<string> AllKernelInfos = new();

        /// <summary>Diagnostic: last compiled Wasm binary (for inspection).</summary>
        public static byte[]? LastWasmBinary { get; set; }

        /// <summary>Diagnostic: all compiled Wasm binaries (for capturing multi-kernel compilations like RadixSort).</summary>
        public static List<byte[]> AllWasmBinaries = new();

        /// <summary>
        /// Circular buffer of recent log messages for diagnostics.
        /// Always captures the last N messages regardless of VerboseLogging.
        /// </summary>
        public static readonly List<string> RecentLogs = new();
        private static readonly int MaxRecentLogs = 500;

        /// <summary>
        /// Writes a message to the console and captures to RecentLogs.
        /// Caller MUST check <see cref="VerboseLogging"/> BEFORE constructing the message string
        /// to avoid allocating interpolated strings when logging is disabled.
        /// </summary>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        public static void Log(string message)
        {
            Console.WriteLine(message);
            RecentLogs.Add(message);
            if (RecentLogs.Count > MaxRecentLogs)
                RecentLogs.RemoveAt(0);
        }

        #endregion

        #region Constructor

        public WasmBackend(Context context)
            : this(context, new WasmBackendOptions())
        {
        }

        public WasmBackend(Context context, WasmBackendOptions options)
            : base(
                  context,
                  new WasmCapabilityContext(),
                  BackendTypeWasm,
                  new WasmArgumentMapper(context))
        {
            Options = options ?? new WasmBackendOptions();

            InitIntrinsicProvider();
            RegisterMathIntrinsics();
            RegisterScanIntrinsics();

            InitializeKernelTransformers(builder =>
            {
                // No Wasm-specific transformers needed for Phase 1
            });

            // Hard reference for bundling
            _ = typeof(global::ILGPU.Algorithms.XMath);
        }

        #endregion

        #region Properties

        public WasmBackendOptions Options { get; }

        public new WasmArgumentMapper ArgumentMapper =>
            (WasmArgumentMapper)base.ArgumentMapper;

        /// <summary>
        /// The kernel function generator for the current compilation (set during CreateKernelCodeGenerator).
        /// </summary>
        internal WasmKernelFunctionGenerator? KernelGenerator { get; set; }

        #endregion

        #region Methods

        private static IntrinsicImplementationManager GetIntrinsicManager(Context context)
        {
            var prop = typeof(Context).GetProperty(
                "IntrinsicManager",
                BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public);
            if (prop == null)
                throw new InvalidOperationException("Could not find IntrinsicManager property on Context.");

            var manager = (IntrinsicImplementationManager)prop.GetValue(context)!;
            FixIntrinsicManager(manager);
            return manager;
        }

        private static void FixIntrinsicManager(IntrinsicImplementationManager manager)
        {
            try
            {
                var mgrType = typeof(IntrinsicImplementationManager);
                var containersField = mgrType.GetField("containers", BindingFlags.Instance | BindingFlags.NonPublic);
                if (containersField == null) return;

                var containers = (Array)containersField.GetValue(manager)!;
                int wasmIndex = (int)BackendTypeWasm;

                if (wasmIndex >= containers.Length || containers.GetValue(wasmIndex) == null)
                {
                    var containerType = mgrType.GetNestedType("BackendContainer", BindingFlags.NonPublic)!;
                    var createMethod = containerType.GetMethod("Create", BindingFlags.Static | BindingFlags.Public)!;

                    if (wasmIndex >= containers.Length)
                    {
                        if (VerboseLogging) Log($"Wasm: Resizing IntrinsicManager containers from {containers.Length} to {wasmIndex + 1}");
                        var newContainers = Array.CreateInstance(containerType, wasmIndex + 1);
                        Array.Copy(containers, newContainers, containers.Length);
                        containers = newContainers;
                        containersField.SetValue(manager, containers);
                    }

                    var newContainer = createMethod.Invoke(null, null);
                    containers.SetValue(newContainer, wasmIndex);
                    if (VerboseLogging) Log("Wasm: Initialized BackendContainer.");
                }
                else
                {
                    var containerType = mgrType.GetNestedType("BackendContainer", BindingFlags.NonPublic)!;
                    var container = containers.GetValue(wasmIndex);
                    var matchersField = containerType.GetField("matchers", BindingFlags.Instance | BindingFlags.NonPublic)!;
                    var matchers = matchersField.GetValue(container!);
                    if (matchers == null)
                    {
                        if (VerboseLogging) Log("Wasm: BackendContainer found but uninitialized. Re-initializing.");
                        var createMethod = containerType.GetMethod("Create", BindingFlags.Static | BindingFlags.Public)!;
                        var newContainer = createMethod.Invoke(null, null);
                        containers.SetValue(newContainer, wasmIndex);
                    }
                }
            }
            catch (Exception ex)
            {
                if (VerboseLogging) Log($"Wasm: Error fixing IntrinsicManager: {ex}");
            }
        }

        private void RegisterRedirect(MethodInfo original, MethodInfo target)
        {
            if (original == null || target == null) return;
            if (VerboseLogging) Log($"Wasm: Redirecting {original.DeclaringType?.Name}.{original.Name} -> {target.DeclaringType?.Name}.{target.Name}");
            GetIntrinsicManager(Context).RegisterMethod(
                original,
                new global::ILGPU.Backends.Wasm.WasmIntrinsic(
                    target,
                    IntrinsicImplementationMode.Redirect));
        }

        private void RegisterScanIntrinsics()
        {
            var manager = GetIntrinsicManager(Context);
            var groupExtType = typeof(global::ILGPU.Algorithms.GroupExtensions);
            var wasmGroupType = typeof(SpawnDev.ILGPU.Wasm.Algorithms.WasmGroupExtensions);

            void RegScan(string name)
            {
                try
                {
                    var src = groupExtType.GetMethod(name, BindingFlags.Public | BindingFlags.Static);
                    if (src == null) return;
                    manager.RegisterMethod(src, new global::ILGPU.Backends.Wasm.WasmIntrinsic(
                        wasmGroupType, name, IntrinsicImplementationMode.Redirect));
                    if (VerboseLogging) Log($"Wasm: Scan intrinsic {name} registered");
                }
                catch (Exception ex)
                {
                    if (VerboseLogging) Log($"Wasm: Scan intrinsic {name} FAILED: {ex.Message}");
                }
            }

            RegScan("Reduce");
            RegScan("AllReduce");
            RegScan("ExclusiveScan");
            RegScan("InclusiveScan");
            RegScan("ExclusiveScanWithBoundaries");
            RegScan("InclusiveScanWithBoundaries");
            RegScan("ExclusiveScanNextIteration");
            RegScan("InclusiveScanNextIteration");
        }

        private void RegisterMathIntrinsics()
        {
            var t = typeof(WasmIntrinsics);

            void RegAll(Type type, string name)
            {
                var methods = type.GetMethods(BindingFlags.Public | BindingFlags.Static)
                    .Where(m => m.Name == name);

                foreach (var m in methods)
                {
                    MethodInfo target = m;
                    if (m.IsGenericMethod)
                    {
                        var gArgs = m.GetGenericArguments();
                        if (gArgs.Length == 1)
                        {
                            try { target = m.MakeGenericMethod(typeof(float)); } catch { continue; }
                        }
                        else continue;
                    }

                    var pTypes = target.GetParameters().Select(p => p.ParameterType).ToArray();
                    var wrapper = t.GetMethod(
                        name,
                        BindingFlags.Public | BindingFlags.Static,
                        null, pTypes, null);

                    if (wrapper != null)
                    {
                        if (VerboseLogging) Log($"Wasm: Mapping {type.Name}.{name}({string.Join(",", pTypes.Select(pt => pt.Name))}) to {t.Name}.{name}");
                        RegisterRedirect(target, wrapper);
                    }
                }
            }

            // Unary — redirect Math.Round/Truncate/Sign to throw-free wrappers
            RegAll(typeof(Math), "Abs");
            RegAll(typeof(MathF), "Abs");
            RegAll(typeof(Math), "Sign");
            RegAll(typeof(MathF), "Sign");
            RegAll(typeof(Math), "Round");
            RegAll(typeof(MathF), "Round");
            RegAll(typeof(Math), "Truncate");
            RegAll(typeof(MathF), "Truncate");

            // Binary
            RegAll(typeof(Math), "Atan2");
            RegAll(typeof(MathF), "Atan2");
            RegAll(typeof(Math), "Max");
            RegAll(typeof(MathF), "Max");
            RegAll(typeof(Math), "Min");
            RegAll(typeof(MathF), "Min");
            RegAll(typeof(Math), "Pow");
            RegAll(typeof(MathF), "Pow");

            // Ternary
            RegAll(typeof(Math), "Clamp");
            RegAll(typeof(MathF), "Clamp");
            RegAll(typeof(Math), "FusedMultiplyAdd");
            RegAll(typeof(MathF), "FusedMultiplyAdd");

            // IntrinsicMath (targets of RemappedIntrinsics)
            RegAll(typeof(IntrinsicMath), "Abs");
            RegAll(typeof(IntrinsicMath), "Min");
            RegAll(typeof(IntrinsicMath), "Max");

            // XMath Rsqrt/Rcp
            try
            {
                var xmathType = Type.GetType("ILGPU.Algorithms.XMath, ILGPU.Algorithms");
                if (xmathType != null)
                {
                    RegAll(xmathType, "Rsqrt");
                    RegAll(xmathType, "Rcp");
                    if (VerboseLogging) Log("Wasm: Registered XMath intrinsics (Rsqrt, Rcp)");
                }
            }
            catch (Exception ex)
            {
                if (VerboseLogging) Log($"Wasm: Error registering XMath intrinsics: {ex.Message}");
            }
        }

        protected override EntryPoint CreateEntryPoint(
            in EntryPointDescription entry,
            in BackendContext backendContext,
            in KernelSpecialization specialization) =>
            new EntryPoint(
                entry,
                backendContext.SharedMemorySpecification,
                specialization);

        protected override StringBuilder CreateKernelBuilder(
            EntryPoint entryPoint,
            in BackendContext backendContext,
            in KernelSpecialization specialization,
            out WasmCodeGenerator.GeneratorArgs data)
        {
            var builder = new StringBuilder();

            builder.AppendLine("//");
            builder.Append("// Generated by SpawnDev.ILGPU.Wasm v");
            builder.AppendLine(Context.Version);
            builder.AppendLine("//");
            builder.AppendLine();

            data = new WasmCodeGenerator.GeneratorArgs(
                this,
                entryPoint,
                backendContext.SharedAllocations,
                backendContext.DynamicSharedAllocations);

            return builder;
        }

        protected override WasmCodeGenerator CreateFunctionCodeGenerator(
            Method method,
            Allocas allocas,
            WasmCodeGenerator.GeneratorArgs data)
        {
            // Store helper methods so the kernel generator can inline them
            data.HelperMethods[method] = allocas;
            return new WasmFunctionGenerator(data, method, allocas);
        }

        /// <summary>
        /// Math function names that will be imported into every Wasm module.
        /// Order matters — the function index assignment must match CreateKernel.
        /// </summary>
        internal static readonly string[] UnaryMathFuncs = { "sin", "cos", "tan", "asin", "acos", "atan",
                                         "sinh", "cosh", "tanh", "exp", "log", "log2",
                                         "log10", "round", "truncate", "sign", "exp2",
                                         "sqrt", "abs", "ceil", "floor" };

        internal static readonly string[] BinaryMathFuncs = { "pow", "atan2" };

        protected override WasmCodeGenerator CreateKernelCodeGenerator(
            in AllocaKindInformation sharedAllocations,
            Method method,
            Allocas allocas,
            WasmCodeGenerator.GeneratorArgs data)
        {
            var gen = new WasmKernelFunctionGenerator(data, method, allocas);

            // Pre-populate math imports with deterministic indices.
            // These MUST match the import order in CreateKernel exactly.
            // Import function indices start at 0.
            var mathImports = new Dictionary<string, uint>();
            uint funcIdx = 0;
            foreach (var name in UnaryMathFuncs)
                mathImports[name] = funcIdx++;
            foreach (var name in BinaryMathFuncs)
                mathImports[name] = funcIdx++;
            gen.MathImports = mathImports;

            // NOTE: Function index assignment for multi-block helpers is done in
            // WasmKernelFunctionGenerator.AssignHelperFunctionIndices(), called at the
            // start of GenerateCode(). This is because CreateKernelCodeGenerator runs
            // BEFORE CreateFunctionCodeGenerator (ILGPU compilation order), so
            // data.HelperMethods is empty at this point.

            KernelGenerator = gen;
            return gen;
        }

        protected override CompiledKernel CreateKernel(
            EntryPoint entryPoint,
            CompiledKernel.KernelInfo? kernelInfo,
            StringBuilder builder,
            WasmCodeGenerator.GeneratorArgs data)
        {
            var kernelGen = KernelGenerator!;

            // Build the Wasm module
            var moduleBuilder = new WasmModuleBuilder();

            // Import shared memory
            // Max 2048 pages = 128MB. Larger values (65536=4GB, 16384=1GB) cause RangeError
            // on some browsers when creating SharedArrayBuffer-backed WebAssembly.Memory.
            // 128MB provides 60x+ headroom over realistic kernel workloads.
            moduleBuilder.ImportSharedMemory("env", "memory", 1, 2048);

            // Import math functions from JavaScript Math object
            var mathImports = new Dictionary<string, uint>();

            // Add unary math type: (f64) -> f64
            int unaryTypeIdx = moduleBuilder.AddFuncType(
                new byte[] { WasmOpCodes.F64 },
                new byte[] { WasmOpCodes.F64 });

            // Add binary math type: (f64, f64) -> f64
            int binaryTypeIdx = moduleBuilder.AddFuncType(
                new byte[] { WasmOpCodes.F64, WasmOpCodes.F64 },
                new byte[] { WasmOpCodes.F64 });

            foreach (var name in UnaryMathFuncs)
            {
                int idx = moduleBuilder.ImportFunction("Math", name, unaryTypeIdx);
                mathImports[name] = (uint)idx;
            }

            foreach (var name in BinaryMathFuncs)
            {
                int idx = moduleBuilder.ImportFunction("Math", name, binaryTypeIdx);
                mathImports[name] = (uint)idx;
            }

            // Pass math imports to the code generator
            kernelGen.MathImports = mathImports;

            // Add function type for the kernel (void return)
            var paramTypes = kernelGen.GetParamTypes();
            int typeIdx = moduleBuilder.AddFuncType(paramTypes, Array.Empty<byte>());

            // Add kernel function (index = importFuncCount + 0)
            int funcIdx = moduleBuilder.AddFunction(typeIdx);

            // Export as "kernel"
            moduleBuilder.ExportFunction("kernel", funcIdx);

            // Set kernel function body (defined function index 0)
            moduleBuilder.SetFunctionBody(0, kernelGen._locals, kernelGen.Code.ToArray());

            // Generate helper function bodies for multi-block helpers
            int definedFuncIndex = 1; // 0 = kernel
            int maxSharedMemorySize = data.SharedMemorySize;

            foreach (var helperMethod in data.HelperFunctionOrder)
            {
                var helperAllocas = data.HelperMethods[helperMethod];
                var helperGen = new WasmKernelFunctionGenerator(data, helperMethod, helperAllocas);
                var result = helperGen.GenerateAsHelper(
                    kernelGen.SharedAllocaOffsets,
                    kernelGen.SharedAllocaMetadata,
                    kernelGen.SharedMemorySizeValue,
                    mathImports);

                // Add helper function type
                int helperTypeIdx = moduleBuilder.AddFuncType(result.ParamTypes, result.ResultTypes);

                // Add helper function (index must match pre-assigned index)
                int helperFuncIdx = moduleBuilder.AddFunction(helperTypeIdx);
                int expectedIdx = data.HelperFunctionIndices[helperMethod];
                if (helperFuncIdx != expectedIdx)
                {
                    if (VerboseLogging) Log($"Wasm: WARNING: Helper '{helperMethod.Name}' funcIdx mismatch: got {helperFuncIdx}, expected {expectedIdx}");
                }

                // Set helper function body
                moduleBuilder.SetFunctionBody(definedFuncIndex, result.Locals, result.Code);
                definedFuncIndex++;

                // Track max shared memory (helpers may allocate Broadcast slots)
                if (result.SharedMemorySize > maxSharedMemorySize)
                    maxSharedMemorySize = result.SharedMemorySize;

                if (VerboseLogging) Log($"Wasm: Helper '{helperMethod.Name}' generated: funcIdx={helperFuncIdx}, params={result.ParamTypes.Length}, locals={result.Locals.Count}, code={result.Code.Length}b, barriers={result.BarrierCount}, sharedMem={result.SharedMemorySize}");
            }

            // Update shared memory size to account for helper Broadcast slots
            data.SharedMemorySize = maxSharedMemorySize;

            // Emit binary
            var wasmBinary = moduleBuilder.Emit();

            // Dump Wasm binary to file for debugging (desktop only)
            if (WasmDumpPath != null && !OperatingSystem.IsBrowser())
            {
                try
                {
                    Directory.CreateDirectory(WasmDumpPath);
                    var name = $"kernel_{wasmBinary.Length}";
                    File.WriteAllBytes(Path.Combine(WasmDumpPath, $"{name}.wasm"), wasmBinary);
                }
                catch { }
            }

            // Record compilation info for diagnostics
            var info = $"Kernel params={paramTypes.Length} (userParams={data.ParamInfos.Count}), locals={kernelGen._locals.Count}, code={kernelGen.Code.Count}b, helpers={data.HelperFunctionOrder.Count}, sharedMem={data.SharedMemorySize}, barriers={data.BarrierCount}, hasBarriers={data.HasBarriers}, dynSharedElemSize={data.DynamicSharedElementSize}";
            if (VerboseLogging)
            {
                Log($"--- GENERATED WASM BINARY ({wasmBinary.Length} bytes) ---");
                Log(info);
                Log("---");
            }
            AllKernelInfos.Add(info);
            LastWasmBinary = wasmBinary;
            AllWasmBinaries.Add(wasmBinary);

            return new WasmCompiledKernel(
                Context,
                entryPoint,
                wasmBinary,
                data.ParamInfos.Count,
                data.ParamInfos,
                data.SharedMemorySize,
                data.BarrierCount,
                data.HasBarriers,
                data.DynamicSharedElementSize,
                data.ScratchPerThread);
        }

        #endregion
    }

    /// <summary>
    /// Intrinsic handler delegate type for the Wasm backend.
    /// </summary>
    public delegate void WasmIntrinsicHandler(
        WasmBackend backend,
        WasmCodeGenerator codeGenerator,
        Value value);

    /// <summary>
    /// Backend options for Wasm.
    /// </summary>
    public class WasmBackendOptions
    {
        /// <summary>
        /// Number of Web Workers to use for parallel dispatch.
        /// Defaults to navigator.hardwareConcurrency.
        /// </summary>
        public int WorkerCount { get; set; } = WasmILGPUDevice.GetHardwareConcurrency();
    }

    /// <summary>
    /// Argument mapper for Wasm kernel parameters.
    /// </summary>
    public class WasmArgumentMapper : ArgumentMapper
    {
        public WasmArgumentMapper(Context context) : base(context) { }

        protected override Type MapViewType(Type viewType, Type elementType)
        {
            return viewType;
        }

        protected override void MapViewInstance<TILEmitter, TSource, TTarget>(
            in TILEmitter emitter,
            Type viewType,
            in TSource source,
            in TTarget target)
        {
            // View mapping handled separately for Wasm
        }
    }


    /// <summary>
    /// Capability context for Wasm backend.
    /// </summary>
    public class WasmCapabilityContext : CapabilityContext
    {
        public WasmCapabilityContext() : base()
        {
        }
    }
}
