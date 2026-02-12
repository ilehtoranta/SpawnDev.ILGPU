// ---------------------------------------------------------------------------------------
//                               SpawnDev.ILGPU.WebGL
//                 WebGL2 Compute Library for Blazor WebAssembly
//
// File: WebGLContext.cs
// ---------------------------------------------------------------------------------------

using global::ILGPU;
using SpawnDev.ILGPU.WebGL.Backend;

namespace SpawnDev.ILGPU.WebGL
{
    /// <summary>
    /// WebGL2 context extensions for ILGPU Context.Builder.
    /// </summary>
    public static class WebGLContextExtensions
    {
        #region Builder

        /// <summary>
        /// Asynchronously enables all detected WebGL2 devices.
        /// </summary>
        public static async System.Threading.Tasks.Task WebGL(this Context.Builder builder)
        {
            await WebGLILGPUDevice.GetDevicesAsync(
                device => true,
                builder.DeviceRegistry);
        }

        /// <summary>
        /// Asynchronously enables WebGL2 devices matching the predicate.
        /// </summary>
        public static async System.Threading.Tasks.Task WebGL(
            this Context.Builder builder,
            System.Predicate<WebGLILGPUDevice> predicate)
        {
            await WebGLILGPUDevice.GetDevicesAsync(
                predicate,
                builder.DeviceRegistry);
        }

        #endregion

        #region Context

        /// <summary>
        /// Gets the i-th registered WebGL2 device.
        /// </summary>
        public static WebGLILGPUDevice GetWebGLDevice(
            this Context context,
            int webGlDeviceIndex) =>
            context.GetDevice<WebGLILGPUDevice>(webGlDeviceIndex);

        /// <summary>
        /// Gets all registered WebGL2 devices.
        /// </summary>
        public static Context.DeviceCollection<WebGLILGPUDevice> GetWebGLDevices(
            this Context context) =>
            context.GetDevices<WebGLILGPUDevice>();

        /// <summary>
        /// Creates a new WebGL2 accelerator.
        /// </summary>
        public static WebGLAccelerator CreateWebGLAccelerator(
            this Context context,
            int webGlDeviceIndex)
            => CreateWebGLAccelerator(context, webGlDeviceIndex, null);

        /// <summary>
        /// Creates a new WebGL2 accelerator with the specified options.
        /// </summary>
        public static WebGLAccelerator CreateWebGLAccelerator(
            this Context context,
            int webGlDeviceIndex,
            WebGLBackendOptions? options)
        {
            var device = context.GetWebGLDevice(webGlDeviceIndex);
            return device.CreateAccelerator(context, options);
        }

        /// <summary>
        /// Creates a new WebGL2 accelerator asynchronously (for API compatibility).
        /// </summary>
        public static Task<WebGLAccelerator> CreateWebGLAcceleratorAsync(
            this Context context,
            int webGlDeviceIndex,
            WebGLBackendOptions? options = null)
        {
            return Task.FromResult(CreateWebGLAccelerator(context, webGlDeviceIndex, options));
        }

        #endregion
    }
}
