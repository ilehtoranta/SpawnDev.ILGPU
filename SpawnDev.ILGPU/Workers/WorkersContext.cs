// ---------------------------------------------------------------------------------------
//                               SpawnDev.ILGPU.Workers
//                 Web Worker Compute Library for Blazor WebAssembly
//
// File: WorkersContext.cs
//
// Context builder extension methods for the Workers backend.
// ---------------------------------------------------------------------------------------

using global::ILGPU;
using SpawnDev.ILGPU.Workers.Backend;

namespace SpawnDev.ILGPU.Workers
{
    /// <summary>
    /// Workers context extensions for ILGPU Context.Builder.
    /// </summary>
    public static class WorkersContextExtensions
    {
        #region Builder

        /// <summary>
        /// Enables the Workers backend device.
        /// Unlike WebGPU, this is synchronous since no GPU probing is needed.
        /// </summary>
        /// <param name="builder">The context builder instance.</param>
        /// <returns>The builder for chaining.</returns>
        public static Context.Builder Workers(this Context.Builder builder)
        {
            WorkersILGPUDevice.GetDevices(
                device => true,
                builder.DeviceRegistry);
            return builder;
        }

        /// <summary>
        /// Enables the Workers backend device (async version for API symmetry with WebGPU).
        /// </summary>
        /// <param name="builder">The context builder instance.</param>
        public static async Task WorkersAsync(this Context.Builder builder)
        {
            await Task.Run(() => WorkersILGPUDevice.GetDevices(
                device => true,
                builder.DeviceRegistry));
        }

        #endregion

        #region Context

        /// <summary>
        /// Gets the i-th registered Workers device.
        /// </summary>
        /// <param name="context">The ILGPU context.</param>
        /// <param name="workersDeviceIndex">
        /// The relative device index for the Workers device. 0 for the first (usually only) device.
        /// </param>
        /// <returns>The registered Workers device.</returns>
        public static WorkersILGPUDevice GetWorkersDevice(
            this Context context,
            int workersDeviceIndex = 0) =>
            context.GetDevice<WorkersILGPUDevice>(workersDeviceIndex);

        /// <summary>
        /// Gets all registered Workers devices.
        /// </summary>
        /// <param name="context">The ILGPU context.</param>
        /// <returns>All registered Workers devices.</returns>
        public static Context.DeviceCollection<WorkersILGPUDevice> GetWorkersDevices(
            this Context context) =>
            context.GetDevices<WorkersILGPUDevice>();

        /// <summary>
        /// Creates a new Workers accelerator.
        /// </summary>
        /// <param name="context">The ILGPU context.</param>
        /// <param name="workersDeviceIndex">The relative device index.</param>
        /// <returns>The Workers accelerator.</returns>
        public static WorkersAccelerator CreateWorkersAccelerator(
            this Context context,
            int workersDeviceIndex = 0)
            => CreateWorkersAccelerator(context, workersDeviceIndex, null);

        /// <summary>
        /// Creates a new Workers accelerator with the specified options.
        /// </summary>
        /// <param name="context">The ILGPU context.</param>
        /// <param name="workersDeviceIndex">The relative device index.</param>
        /// <param name="options">The backend configuration options (null for defaults).</param>
        /// <returns>The Workers accelerator.</returns>
        public static WorkersAccelerator CreateWorkersAccelerator(
            this Context context,
            int workersDeviceIndex,
            WorkersBackendOptions? options)
        {
            var device = context.GetWorkersDevice(workersDeviceIndex);
            return device.CreateAccelerator(context, options);
        }

        /// <summary>
        /// Creates a new Workers accelerator asynchronously (for API symmetry).
        /// </summary>
        public static async Task<WorkersAccelerator> CreateWorkersAcceleratorAsync(
            this Context context,
            int workersDeviceIndex = 0)
        {
            return await Task.FromResult(context.CreateWorkersAccelerator(workersDeviceIndex));
        }

        #endregion
    }
}
