// ---------------------------------------------------------------------------------------
//                               SpawnDev.ILGPU.Workers
//                 Web Worker Compute Library for Blazor WebAssembly
//
// File: WorkersILGPUDevice.cs
// ---------------------------------------------------------------------------------------

using global::ILGPU;
using global::ILGPU.Runtime;
using SpawnDev.ILGPU.Workers.Backend;

namespace SpawnDev.ILGPU.Workers
{
    /// <summary>
    /// Represents an ILGPU Device implementation for Web Workers.
    /// Unlike WebGPU (which probes GPU adapters), Workers devices are always available
    /// in Blazor WASM since Web Workers are universally supported.
    /// </summary>
    [DeviceType(WorkersILGPUDevice.WorkersAcceleratorType)]
    public sealed class WorkersILGPUDevice : Device
    {
        /// <summary>
        /// The accelerator type constant for Workers devices.
        /// </summary>
        public const AcceleratorType WorkersAcceleratorType = AcceleratorType.Workers;

        #region Static

        /// <summary>
        /// Asynchronously detects the Workers device and registers it.
        /// Workers are always available in Blazor WASM.
        /// </summary>
        public static Task<DeviceRegistry> GetDevicesAsync()
        {
            var registry = new DeviceRegistry();
            GetDevices(device => true, registry);
            return Task.FromResult(registry);
        }

        /// <summary>
        /// Detects workers devices matching the predicate and adds them to the registry.
        /// </summary>
        public static void GetDevices(
            Predicate<WorkersILGPUDevice> predicate,
            DeviceRegistry registry)
        {
            if (predicate is null)
                throw new ArgumentNullException(nameof(predicate));
            if (registry is null)
                throw new ArgumentNullException(nameof(registry));

            // Create a single Workers device
            var device = new WorkersILGPUDevice();
            registry.Register(device, predicate);
        }

        /// <summary>
        /// Gets the number of hardware threads (navigator.hardwareConcurrency).
        /// </summary>
        public static int GetHardwareConcurrency()
        {
            try
            {
                using var navigator = SpawnDev.BlazorJS.BlazorJSRuntime.JS.Get<SpawnDev.BlazorJS.JSObjects.Navigator>("navigator");
                return navigator.HardwareConcurrency ?? 4;
            }
            catch
            {
                return 4; // Conservative fallback
            }
        }

        #endregion

        #region Instance

        /// <summary>
        /// The number of hardware threads reported by the browser.
        /// </summary>
        public int HardwareConcurrency { get; }

        /// <summary>
        /// Constructs a new Workers ILGPU device.
        /// </summary>
        public WorkersILGPUDevice()
        {
            HardwareConcurrency = GetHardwareConcurrency();

            // Set device properties
            Name = $"Web Workers ({HardwareConcurrency} cores)";
            WarpSize = 1; // No warp-level parallelism in JS workers

            // Workers don't have GPU-style limits, but ILGPU requires these
            MaxGroupSize = new Index3D(1024, 1024, 1024);
            MaxNumThreadsPerGroup = 1024;
            NumMultiprocessors = HardwareConcurrency;
            MaxNumThreadsPerMultiprocessor = 1; // One thread per worker
            MaxGridSize = new Index3D(int.MaxValue, 1, 1);

            // Memory: use a large value (JS heap is the limit)
            MemorySize = 2L * 1024 * 1024 * 1024; // 2 GB typical WASM limit
            MaxSharedMemoryPerGroup = 0; // No shared memory in MVP
            MaxConstantMemory = 65536;

            // Capability context
            Capabilities = new WorkersCapabilityContext();
        }

        #endregion

        #region Methods

        /// <summary>
        /// Creates a new Workers accelerator synchronously.
        /// NOTE: Workers support async initialization. Use CreateAcceleratorAsync for best practice.
        /// </summary>
        public override Accelerator CreateAccelerator(Context context)
        {
            // Workers CAN be created synchronously since no GPU probing is needed
            return WorkersAccelerator.Create(context, this, null);
        }

        /// <summary>
        /// Creates a new Workers accelerator asynchronously.
        /// </summary>
        public override async Task<Accelerator> CreateAcceleratorAsync(Context context)
        {
            return await Task.FromResult(WorkersAccelerator.Create(context, this, null));
        }

        /// <summary>
        /// Creates a new Workers accelerator with the specified options.
        /// </summary>
        public WorkersAccelerator CreateAccelerator(Context context, WorkersBackendOptions? options)
        {
            return WorkersAccelerator.Create(context, this, options);
        }

        #endregion
    }

    /// <summary>
    /// Capability context for the Workers backend.
    /// </summary>
    public class WorkersCapabilityContext : CapabilityContext
    {
        // Workers support i32, f32, f64 (JS numbers) but no f16
    }
}
