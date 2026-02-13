// ---------------------------------------------------------------------------------------
//                               SpawnDev.ILGPU
//                 Unified Context Extensions for Blazor WebAssembly
//
// File: SpawnDevContextExtensions.cs
//
// Provides AllAcceleratorsAsync() and CreatePreferredAcceleratorAsync()
// for easy device discovery and accelerator creation in WASM.
// ---------------------------------------------------------------------------------------

using global::ILGPU;
using global::ILGPU.Runtime;
using SpawnDev.ILGPU.Wasm;
using SpawnDev.ILGPU.WebGL;
using SpawnDev.ILGPU.WebGL.Backend;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.ILGPU.WebGPU.Backend;
using SpawnDev.ILGPU.Workers;
using System.Runtime.InteropServices;

namespace SpawnDev.ILGPU
{
    /// <summary>
    /// Unified context extensions for Blazor WebAssembly.
    /// Provides async device probing for all WASM-compatible backends
    /// (WebGPU, Wasm, Workers, CPU).
    /// </summary>
    public static class SpawnDevContextExtensions
    {
        #region Builder Extensions

        /// <summary>
        /// Enables all supported WASM accelerators: CPU, Workers, and WebGPU.
        /// WebGPU requires async GPU probing, so this method is async.
        /// If WebGPU is not available, it is silently skipped.
        /// </summary>
        /// <param name="builder">The context builder instance.</param>
        /// <returns>The builder for chaining.</returns>
        public static async Task<Context.Builder> AllAcceleratorsAsync(
            this Context.Builder builder)
        {
            // Synchronous backends first
            builder.AllAccelerators(); // CPU, OpenCL, Cuda (latter two will fail silently in WASM)
            builder.Workers();         // Always available in WASM
            builder.Wasm();            // Always available in WASM

            // WebGPU requires async probing — may not be available
            try
            {
                await builder.WebGPU();
            }
            catch
            {
                // WebGPU not available in this environment
            }

            // WebGL2 requires async probing — may not be available
            try
            {
                await builder.WebGL();
            }
            catch
            {
                // WebGL2 not available in this environment
            }

            return builder;
        }



        #endregion

        #region Preferred Accelerator

        /// <summary>
        /// Creates the preferred accelerator for WASM environments.
        /// Priority: WebGPU (GPU compute) > Wasm (native Wasm) > Workers (multi-threaded JS) > CPU (fallback).
        /// </summary>
        /// <param name="context">The ILGPU context (must have devices registered).</param>
        /// <returns>The best available accelerator.</returns>
        public static async Task<Accelerator> CreatePreferredAcceleratorAsync(
            this Context context)
        {
            // Try WebGPU first (true GPU compute)
            var webGpuDevices = context.GetDevices<WebGPUILGPUDevice>();
            if (webGpuDevices.Count > 0)
            {
                return await webGpuDevices[0].CreateAcceleratorAsync(context, null);
            }

            // Try WebGL2 (GPU compute via Transform Feedback)
            var webGlDevices = context.GetDevices<WebGLILGPUDevice>();
            if (webGlDevices.Count > 0)
            {
                return webGlDevices[0].CreateAccelerator(context);
            }

            // Try Wasm (near-native WebAssembly compute)
            var wasmDevices = context.GetDevices<WasmILGPUDevice>();
            if (wasmDevices.Count > 0)
            {
                return await WasmAccelerator.Create(context);
            }

            // Try Workers (multi-threaded JS)
            var workersDevices = context.GetDevices<WorkersILGPUDevice>();
            if (workersDevices.Count > 0)
            {
                return workersDevices[0].CreateAccelerator(context);
            }

            // Fall back to CPU
            return context.GetPreferredDevice(preferCPU: true).CreateAccelerator(context);
        }

        /// <summary>
        /// Gets information about all registered devices suitable for display.
        /// </summary>
        /// <param name="context">The ILGPU context.</param>
        /// <returns>
        /// A list of tuples containing (Name, AcceleratorType) for each registered device.
        /// </returns>
        public static List<(string Name, AcceleratorType Type)> GetAllDeviceInfo(
            this Context context)
        {
            var result = new List<(string, AcceleratorType)>();
            foreach (var device in context.Devices)
            {
                result.Add((device.Name, device.AcceleratorType));
            }
            return result;
        }

        #endregion

        #region Unified Buffer Readback

        /// <summary>
        /// Copies data from any ILGPU buffer (WebGPU, Workers, or CPU) back to the host.
        /// Automatically detects the underlying buffer type and uses the appropriate method.
        /// Use this instead of backend-specific CopyToHostAsync to avoid ambiguity.
        /// </summary>
        /// <typeparam name="T">The element type of the buffer.</typeparam>
        /// <param name="buffer">The MemoryBuffer1D to read from.</param>
        /// <returns>An array containing the buffer data.</returns>
        public static async Task<T[]> CopyToHostAsync<T>(
            this MemoryBuffer1D<T, Stride1D.Dense> buffer) where T : unmanaged
        {
            return await CopyToHostAsync<T>((MemoryBuffer)buffer);
        }

        /// <summary>
        /// Copies data from any ILGPU buffer (WebGPU, Workers, or CPU) back to the host.
        /// Automatically detects the underlying buffer type and uses the appropriate method.
        /// Use this instead of backend-specific CopyToHostAsync to avoid ambiguity.
        /// </summary>
        /// <typeparam name="T">The element type of the buffer.</typeparam>
        /// <param name="buffer">The MemoryBuffer to read from.</param>
        /// <returns>An array containing the buffer data.</returns>
        public static async Task<T[]> CopyToHostAsync<T>(
            this MemoryBuffer buffer) where T : unmanaged
        {
            var iView = (IArrayView)buffer;

            // Check for WebGPU buffer
            if (iView.Buffer is WebGPUMemoryBuffer webGpuBuffer)
            {
                var byteData = await webGpuBuffer.NativeBuffer.CopyToHostAsync();
                var result = new T[buffer.Length];
                MemoryMarshal.Cast<byte, T>(byteData).CopyTo(new Span<T>(result));
                return result;
            }

            // Check for WebGL2 buffer
            if (iView.Buffer is WebGLMemoryBuffer webGlBuffer)
            {
                var byteData = webGlBuffer.BackingArray!.ReadBytes();
                var result = new T[buffer.Length];
                MemoryMarshal.Cast<byte, T>(byteData).CopyTo(new Span<T>(result));
                return result;
            }

            // Check for Workers buffer
            if (iView.Buffer is WorkersMemoryBuffer workersBuffer)
            {
                var uint8View = workersBuffer.Uint8View
                    ?? throw new ObjectDisposedException(nameof(WorkersMemoryBuffer));
                var byteData = uint8View.ReadBytes();
                var result = new T[buffer.Length];
                MemoryMarshal.Cast<byte, T>(byteData).CopyTo(new Span<T>(result));
                return result;
            }

            // Check for Wasm buffer
            if (iView.Buffer is WasmMemoryBuffer wasmBuffer)
            {
                var byteData = wasmBuffer.TypedArrayView.ReadBytes();
                var result = new T[buffer.Length];
                MemoryMarshal.Cast<byte, T>(byteData).CopyTo(new Span<T>(result));
                return result;
            }

            // CPU buffer — use standard ILGPU synchronous copy
            var cpuResult = new T[buffer.Length];
            buffer.AsArrayView<T>(0, buffer.Length).CopyToCPU(cpuResult);
            return cpuResult;
        }

        #endregion

        #region Unified Synchronization

        /// <summary>
        /// Asynchronously waits for all submitted work to complete.
        /// Works with any ILGPU Accelerator — dispatches to the correct
        /// backend-specific implementation (WebGPU, Workers, or CPU).
        /// </summary>
        /// <param name="accelerator">The ILGPU accelerator.</param>
        /// <returns>A task that completes when all work is done.</returns>
        public static async Task SynchronizeAsync(this global::ILGPU.Runtime.Accelerator accelerator)
        {
            if (accelerator is WebGPUAccelerator webGpuAccelerator)
            {
                await WebGPUAcceleratorExtensions.SynchronizeAsync(webGpuAccelerator);
            }
            else if (accelerator is WebGLAccelerator webGlAccelerator)
            {
                // WebGL2 now uses async worker dispatch — must await pending tasks
                await WebGLAcceleratorExtensions.SynchronizeAsync(webGlAccelerator);
            }
            else if (accelerator is WorkersAccelerator workersAccelerator)
            {
                await WorkersAcceleratorExtensions.SynchronizeAsync(workersAccelerator);
            }
            else if (accelerator is WasmAccelerator wasmAccelerator)
            {
                await wasmAccelerator.SynchronizeAsync();
            }
            else
            {
                // For CPU or other accelerators, use synchronous method
                accelerator.Synchronize();
            }
        }

        #endregion
    }
}
