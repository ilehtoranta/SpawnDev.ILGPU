// ---------------------------------------------------------------------------------------
//                               SpawnDev.ILGPU.Workers
//                 Web Worker Compute Library for Blazor WebAssembly
//
// File: WorkersBackend.cs
//
// CodeGeneratorBackend that compiles ILGPU IR to JavaScript for Web Worker execution.
// ---------------------------------------------------------------------------------------

using global::ILGPU;
using global::ILGPU.Backends;
using global::ILGPU.Backends.EntryPoints;
using global::ILGPU.IR;
using global::ILGPU.IR.Analyses;
using global::ILGPU.IR.Intrinsics;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using System.Reflection;
using System.Text;

namespace SpawnDev.ILGPU.Workers.Backend
{
    /// <summary>
    /// Web Workers/JavaScript backend for ILGPU.
    /// Compiles ILGPU IR to JavaScript source code for execution in Web Workers.
    /// </summary>
    public class WorkersBackend : CodeGeneratorBackend<
        WorkersIntrinsicHandler,
        JSCodeGenerator.GeneratorArgs,
        JSCodeGenerator,
        StringBuilder>
    {
        /// <summary>
        /// The backend type constant for Workers.
        /// </summary>
        public const BackendType BackendTypeWorkers = BackendType.Workers;

        /// <summary>
        /// Controls whether verbose logging is enabled.
        /// </summary>
        public static bool VerboseLogging { get; set; } = false;

        /// <summary>
        /// Logs a message to the console if VerboseLogging is enabled.
        /// </summary>
        public static void Log(string message)
        {
            if (VerboseLogging)
                Console.WriteLine(message);
        }

        #region Instance

        /// <summary>
        /// Creates a new Workers backend with default options.
        /// </summary>
        public WorkersBackend(Context context)
            : this(context, new WorkersBackendOptions())
        {
        }

        /// <summary>
        /// Creates a new Workers backend with the specified options.
        /// </summary>
        public WorkersBackend(Context context, WorkersBackendOptions options)
            : base(
                  context,
                  new WorkersCapabilityContext(),
                  BackendTypeWorkers,
                  new WorkersArgumentMapper(context))
        {
            Options = options ?? new WorkersBackendOptions();

            InitIntrinsicProvider();

            InitializeKernelTransformers(builder =>
            {
                // No Workers-specific transformers needed for MVP
            });

            // Hard reference for bundling
            _ = typeof(XMath);
        }

        #endregion

        #region Properties

        /// <summary>
        /// The configuration options for this backend instance.
        /// </summary>
        public WorkersBackendOptions Options { get; }

        /// <summary>
        /// Returns the associated argument mapper.
        /// </summary>
        public new WorkersArgumentMapper ArgumentMapper =>
            base.ArgumentMapper as WorkersArgumentMapper;

        #endregion

        #region Methods

        /// <summary>
        /// Gets the intrinsic manager via reflection (it is internal to ILGPU).
        /// </summary>
        private static IntrinsicImplementationManager GetIntrinsicManager(Context context)
        {
            var prop = typeof(Context).GetProperty(
                "IntrinsicManager",
                BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public);
            if (prop == null)
                throw new InvalidOperationException("Could not find IntrinsicManager property on Context.");

            var manager = (IntrinsicImplementationManager)prop.GetValue(context);
            FixIntrinsicManager(manager);
            return manager;
        }

        /// <summary>
        /// Ensures the IntrinsicManager has a container initialized for the Workers backend type.
        /// </summary>
        private static void FixIntrinsicManager(IntrinsicImplementationManager manager)
        {
            try
            {
                var mgrType = typeof(IntrinsicImplementationManager);
                var containersField = mgrType.GetField("containers", BindingFlags.Instance | BindingFlags.NonPublic);
                if (containersField == null) return;

                var containers = (Array)containersField.GetValue(manager);
                int workersIndex = (int)BackendTypeWorkers;

                if (workersIndex >= containers.Length || containers.GetValue(workersIndex) == null)
                {
                    var containerType = mgrType.GetNestedType("BackendContainer", BindingFlags.NonPublic);
                    var createMethod = containerType.GetMethod("Create", BindingFlags.Static | BindingFlags.Public);

                    if (workersIndex >= containers.Length)
                    {
                        Log($"Workers: Resizing IntrinsicManager containers from {containers.Length} to {workersIndex + 1}");
                        var newContainers = Array.CreateInstance(containerType, workersIndex + 1);
                        Array.Copy(containers, newContainers, containers.Length);
                        containers = newContainers;
                        containersField.SetValue(manager, containers);
                    }

                    var newContainer = createMethod.Invoke(null, null);
                    containers.SetValue(newContainer, workersIndex);
                    Log("Workers: Initialized BackendContainer for Workers.");
                }
                else
                {
                    var containerType = mgrType.GetNestedType("BackendContainer", BindingFlags.NonPublic);
                    var container = containers.GetValue(workersIndex);
                    var matchersField = containerType.GetField("matchers", BindingFlags.Instance | BindingFlags.NonPublic);
                    var matchers = matchersField.GetValue(container);
                    if (matchers == null)
                    {
                        Log("Workers: BackendContainer found but uninitialized. Re-initializing.");
                        var createMethod = containerType.GetMethod("Create", BindingFlags.Static | BindingFlags.Public);
                        var newContainer = createMethod.Invoke(null, null);
                        containers.SetValue(newContainer, workersIndex);
                    }
                }
            }
            catch (Exception ex)
            {
                Log($"Workers: Error fixing IntrinsicManager: {ex}");
            }
        }

        /// <summary>
        /// Creates a new entry point.
        /// </summary>
        protected override EntryPoint CreateEntryPoint(
            in EntryPointDescription entry,
            in BackendContext backendContext,
            in KernelSpecialization specialization) =>
            new EntryPoint(
                entry,
                backendContext.SharedMemorySpecification,
                specialization);

        /// <summary>
        /// Creates the main kernel builder and initializes generator args.
        /// </summary>
        protected override StringBuilder CreateKernelBuilder(
            EntryPoint entryPoint,
            in BackendContext backendContext,
            in KernelSpecialization specialization,
            out JSCodeGenerator.GeneratorArgs data)
        {
            var builder = new StringBuilder();

            builder.AppendLine("//");
            builder.Append("// Generated by SpawnDev.ILGPU.Workers v");
            builder.AppendLine(Context.Version);
            builder.AppendLine("//");
            builder.AppendLine();

            var typeGenerator = new JSTypeGenerator();

            data = new JSCodeGenerator.GeneratorArgs(
                this,
                typeGenerator,
                entryPoint,
                backendContext.SharedAllocations,
                backendContext.DynamicSharedAllocations);

            return builder;
        }

        /// <summary>
        /// Creates a function-code generator for helper methods.
        /// </summary>
        protected override JSCodeGenerator CreateFunctionCodeGenerator(
            Method method,
            Allocas allocas,
            JSCodeGenerator.GeneratorArgs data) =>
            new JSFunctionGenerator(data, method, allocas);

        /// <summary>
        /// Creates a kernel-code generator for the main entry point.
        /// </summary>
        protected override JSCodeGenerator CreateKernelCodeGenerator(
            in AllocaKindInformation sharedAllocations,
            Method method,
            Allocas allocas,
            JSCodeGenerator.GeneratorArgs data) =>
            new JSKernelFunctionGenerator(data, method, allocas);

        /// <summary>
        /// Creates the final compiled kernel.
        /// </summary>
        protected override CompiledKernel CreateKernel(
            EntryPoint entryPoint,
            CompiledKernel.KernelInfo? kernelInfo,
            StringBuilder builder,
            JSCodeGenerator.GeneratorArgs data)
        {
            var jsSource = builder.ToString();
            Log("--- GENERATED JAVASCRIPT ---");
            Log(jsSource);
            Log("----------------------------");
            return new WorkersCompiledKernel(
                Context,
                entryPoint,
                jsSource,
                0 /* binding count determined at runtime */);
        }

        #endregion
    }

    /// <summary>
    /// Intrinsic handler delegate type for the Workers backend.
    /// </summary>
    /// <param name="backend">The backend instance.</param>
    /// <param name="codeGenerator">The code generator instance.</param>
    /// <param name="value">The value to generate code for.</param>
    public delegate void WorkersIntrinsicHandler(
        WorkersBackend backend,
        JSCodeGenerator codeGenerator,
        Value value);


    /// <summary>
    /// Argument mapper for Workers kernel parameters.
    /// </summary>
    public class WorkersArgumentMapper : ArgumentMapper
    {
        public WorkersArgumentMapper(Context context) : base(context) { }

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
            // View mapping handled separately for Workers
        }
    }

    /// <summary>
    /// Capability context for Workers backend.
    /// </summary>
    public class WorkersCapabilityContext : CapabilityContext
    {
        public WorkersCapabilityContext() : base()
        {
        }
    }
}
