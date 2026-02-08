// ---------------------------------------------------------------------------------------
//                               SpawnDev.ILGPU.Wasm
//                    WebAssembly Compute Backend for Blazor WebAssembly
//
// File: WasmILGPUDevice.cs
// ---------------------------------------------------------------------------------------

using global::ILGPU;
using global::ILGPU.Runtime;
using SpawnDev.ILGPU.Wasm.Backend;

namespace SpawnDev.ILGPU.Wasm
{
    /// <summary>
    /// Represents an ILGPU Device implementation for the WebAssembly backend.
    /// Always available in Blazor WASM environments.
    /// </summary>
    [DeviceType(AcceleratorType.CPU)]
    public sealed class WasmILGPUDevice : Device
    {
        #region Static

        /// <summary>
        /// Detects Wasm devices and adds them to the registry.
        /// </summary>
        public static void GetDevices(
            Predicate<WasmILGPUDevice> predicate,
            DeviceRegistry registry)
        {
            if (predicate is null) throw new ArgumentNullException(nameof(predicate));
            if (registry is null) throw new ArgumentNullException(nameof(registry));
            var device = new WasmILGPUDevice();
            registry.Register(device, predicate);
        }

        #endregion

        #region Instance

        public WasmILGPUDevice()
        {
            Name = "WebAssembly Compute";
            WarpSize = 1;
            MaxGroupSize = new Index3D(1024, 1024, 1024);
            MaxNumThreadsPerGroup = 1024;
            NumMultiprocessors = 1;
            MaxNumThreadsPerMultiprocessor = 1;
            MaxGridSize = new Index3D(int.MaxValue, 1, 1);
            MemorySize = 2L * 1024 * 1024 * 1024;
            MaxSharedMemoryPerGroup = 65536;
            MaxConstantMemory = 65536;
            Capabilities = new WasmCapabilityContext();
        }

        #endregion

        #region Methods

        public override Accelerator CreateAccelerator(Context context)
        {
            // Must use async; synchronous creation is not safe in WASM
            return WasmAccelerator.Create(context).GetAwaiter().GetResult();
        }

        public override async Task<Accelerator> CreateAcceleratorAsync(Context context)
        {
            return await WasmAccelerator.Create(context);
        }

        public WasmAccelerator CreateAccelerator(Context context, WasmBackendOptions? options)
        {
            return WasmAccelerator.Create(context, options ?? new WasmBackendOptions()).GetAwaiter().GetResult();
        }

        #endregion
    }
}
