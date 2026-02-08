// ---------------------------------------------------------------------------------------
//                               SpawnDev.ILGPU.Wasm
//                    WebAssembly Compute Backend for Blazor WebAssembly
//
// File: WasmContext.cs
// ---------------------------------------------------------------------------------------

using global::ILGPU;
using SpawnDev.ILGPU.Wasm.Backend;

namespace SpawnDev.ILGPU.Wasm
{
    /// <summary>
    /// Extension methods for ILGPU Context to support the Wasm backend.
    /// </summary>
    public static class WasmContextExtensions
    {
        #region Builder

        /// <summary>
        /// Enables the Wasm backend device on the context builder.
        /// </summary>
        public static Context.Builder Wasm(this Context.Builder builder)
        {
            WasmILGPUDevice.GetDevices(device => true, builder.DeviceRegistry);
            return builder;
        }

        #endregion

        #region Context

        /// <summary>
        /// Gets the registered Wasm device.
        /// </summary>
        public static WasmILGPUDevice GetWasmDevice(
            this Context context, int index = 0) =>
            context.GetDevice<WasmILGPUDevice>(index);

        /// <summary>
        /// Creates a new Wasm accelerator.
        /// </summary>
        public static async Task<WasmAccelerator> CreateWasmAcceleratorAsync(
            this Context context)
        {
            return await WasmAccelerator.Create(context);
        }

        /// <summary>
        /// Creates a new Wasm accelerator with options.
        /// </summary>
        public static async Task<WasmAccelerator> CreateWasmAcceleratorAsync(
            this Context context, WasmBackendOptions options)
        {
            return await WasmAccelerator.Create(context, options);
        }

        #endregion
    }
}
