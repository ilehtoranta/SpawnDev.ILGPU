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
using global::ILGPU.Runtime;
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
        /// Logs a message if VerboseLogging is enabled.
        /// </summary>
        public static void Log(string message)
        {
            if (VerboseLogging)
                Console.WriteLine(message);
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

            InitializeKernelTransformers(builder =>
            {
                // No Wasm-specific transformers needed for Phase 1
            });

            // Hard reference for bundling
            // XMath reference removed - not available in core ILGPU
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
                        Log($"Wasm: Resizing IntrinsicManager containers from {containers.Length} to {wasmIndex + 1}");
                        var newContainers = Array.CreateInstance(containerType, wasmIndex + 1);
                        Array.Copy(containers, newContainers, containers.Length);
                        containers = newContainers;
                        containersField.SetValue(manager, containers);
                    }

                    var newContainer = createMethod.Invoke(null, null);
                    containers.SetValue(newContainer, wasmIndex);
                    Log("Wasm: Initialized BackendContainer.");
                }
                else
                {
                    var containerType = mgrType.GetNestedType("BackendContainer", BindingFlags.NonPublic)!;
                    var container = containers.GetValue(wasmIndex);
                    var matchersField = containerType.GetField("matchers", BindingFlags.Instance | BindingFlags.NonPublic)!;
                    var matchers = matchersField.GetValue(container!);
                    if (matchers == null)
                    {
                        Log("Wasm: BackendContainer found but uninitialized. Re-initializing.");
                        var createMethod = containerType.GetMethod("Create", BindingFlags.Static | BindingFlags.Public)!;
                        var newContainer = createMethod.Invoke(null, null);
                        containers.SetValue(newContainer, wasmIndex);
                    }
                }
            }
            catch (Exception ex)
            {
                Log($"Wasm: Error fixing IntrinsicManager: {ex}");
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
            WasmCodeGenerator.GeneratorArgs data) =>
            new WasmFunctionGenerator(data, method, allocas);

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
            moduleBuilder.ImportSharedMemory("env", "memory", 1, 65536);

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

            // Add function type for the kernel
            var paramTypes = kernelGen.GetParamTypes();
            int typeIdx = moduleBuilder.AddFuncType(paramTypes, Array.Empty<byte>());

            // Add function (index = importFuncCount + 0)
            int funcIdx = moduleBuilder.AddFunction(typeIdx);

            // Export as "kernel"
            moduleBuilder.ExportFunction("kernel", funcIdx);

            // Set function body
            moduleBuilder.SetFunctionBody(0, kernelGen._locals, kernelGen.Code.ToArray());

            // Emit binary
            var wasmBinary = moduleBuilder.Emit();

            Log($"--- GENERATED WASM BINARY ({wasmBinary.Length} bytes) ---");
            Log($"Params: {paramTypes.Length}, Locals: {kernelGen._locals.Count}, Code: {kernelGen.Code.Count} bytes");
            Log("---");

            return new WasmCompiledKernel(
                Context,
                entryPoint,
                wasmBinary,
                data.ParamInfos.Count,
                data.ParamInfos,
                data.SharedMemorySize,
                data.BarrierCount,
                data.HasBarriers,
                data.DynamicSharedElementSize);
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
