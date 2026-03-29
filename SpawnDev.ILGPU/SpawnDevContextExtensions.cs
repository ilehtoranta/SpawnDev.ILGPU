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
using ILGPU.Runtime.CPU;
using SpawnDev.BlazorJS.JSObjects;
using SpawnDev.ILGPU.Wasm;
using SpawnDev.ILGPU.Wasm.Algorithms;
using SpawnDev.ILGPU.WebGL;
using SpawnDev.ILGPU.WebGL.Backend;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.ILGPU.WebGPU.Algorithms;
using SpawnDev.ILGPU.WebGPU.Backend;

using System.Runtime.InteropServices;

namespace SpawnDev.ILGPU
{
    /// <summary>
    /// Unified context extensions for Blazor WebAssembly.
    /// Provides async device probing for all WASM-compatible backends
    /// (WebGPU, WebGL, Wasm, CPU).
    /// </summary>
    public static class SpawnDevContextExtensions
    {
        #region Builder Extensions

        /// <summary>
        /// Enables all supported WASM accelerators: CPU, WebGL, Wasm, and WebGPU.
        /// WebGPU requires async GPU probing, so this method is async.
        /// If WebGPU is not available, it is silently skipped.
        /// </summary>
        /// <param name="builder">The context builder instance.</param>
        /// <returns>The builder for chaining.</returns>
        public static async Task<Context.Builder> AllAcceleratorsAsync(
            this Context.Builder builder)
        {
            // Enable algorithms by default — users shouldn't need to call this manually
            builder.EnableAlgorithms();

            // Synchronous backends first (CPU, OpenCL, Cuda — latter two fail silently in WASM)
            builder.AllAccelerators();

            // Browser backends — only available in Blazor WebAssembly
            if (OperatingSystem.IsBrowser())
            {
                // Wasm backend — always available in WASM
                try
                {
                    builder.Wasm();
                    builder.EnableWasmAlgorithms();
                }
                catch
                {
                    // Wasm registration failed
                }

                // WebGPU requires async probing — may not be available
                try
                {
                    await builder.WebGPU();
                    builder.EnableWebGPUAlgorithms();
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
            }

            return builder;
        }



        #endregion

        #region Preferred Accelerator

        /// <summary>
        /// Creates the preferred accelerator.
        /// Browser priority: WebGPU > WebGL > Wasm > CPU.
        /// Desktop priority: Cuda > OpenCL > CPU (via GetPreferredDevice).
        /// </summary>
        /// <param name="context">The ILGPU context (must have devices registered).</param>
        /// <returns>The best available accelerator.</returns>
        public static async Task<Accelerator> CreatePreferredAcceleratorAsync(
            this Context context)
        {
            if (OperatingSystem.IsBrowser())
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
            }

            // Desktop: Cuda > OpenCL > CPU  |  Browser fallback: CPU
            return context.GetPreferredDevice(preferCPU: false).CreateAccelerator(context);
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
        /// Copies data from any ILGPU buffer (WebGPU, WebGL, Wasm, or CPU) back to the host.
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
        /// Copies a range of data from any ILGPU buffer back to the host.
        /// Works on all backends (WebGPU, WebGL, Wasm, CUDA, OpenCL, CPU).
        /// For small reads (≤64 elements), the overhead of reading the full buffer is negligible.
        /// </summary>
        /// <typeparam name="T">The element type of the buffer.</typeparam>
        /// <param name="buffer">The MemoryBuffer1D to read from.</param>
        /// <param name="offset">Start offset in elements.</param>
        /// <param name="count">Number of elements to read.</param>
        /// <returns>An array containing the requested range.</returns>
        public static async Task<T[]> CopyToHostAsync<T>(
            this MemoryBuffer1D<T, Stride1D.Dense> buffer, long offset, long count) where T : unmanaged
        {
            var all = await CopyToHostAsync<T>(buffer);
            if (offset == 0 && count == all.Length) return all;
            var result = new T[count];
            System.Array.Copy(all, offset, result, 0, count);
            return result;
        }

        /// <summary>
        /// Copies data from any ILGPU buffer (WebGPU, WebGL, Workers, or CPU) back to the host.
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

            // Check for WebGL2 buffer — must request readback from GL worker first
            if (iView.Buffer is WebGLMemoryBuffer webGlBuffer)
            {
                var accel = (WebGLAccelerator)buffer.Accelerator;
                using var readback = await accel.ReadbackAndGetUint8ArrayAsync(webGlBuffer);
                var byteData = readback.ReadBytes();
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

        /// <summary>
        /// Copies data from any ILGPU buffer (WebGPU, WebGL, Wasm, or CPU) back to the host as a Uint8Array.
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="sourceByteOffset"></param>
        /// <param name="copyBytes"></param>
        /// <returns></returns>
        public static async Task<Uint8Array> CopyToHostUint8ArrayAsync(this MemoryBuffer buffer,long sourceByteOffset = 0, long? copyBytes = null)
        {
            var iView = (IArrayView)buffer;

            // Check for WebGPU buffer
            if (iView.Buffer is WebGPUMemoryBuffer webGpuBuffer)
            {
                var result = await webGpuBuffer.NativeBuffer.CopyToHostUint8ArrayAsync(sourceByteOffset, copyBytes);
                return result;
            }

            // Check for WebGL2 buffer — request readback from GL worker
            if (iView.Buffer is WebGLMemoryBuffer webGlBuffer)
            {
                var accel = (WebGLAccelerator)buffer.Accelerator;
                return await accel.ReadbackAndGetUint8ArrayAsync(webGlBuffer, sourceByteOffset, copyBytes);
            }



            // Check for Wasm buffer
            if (iView.Buffer is WasmMemoryBuffer wasmBuffer)
            {
                using var uint8Array = new Uint8Array(wasmBuffer.SharedBuffer);
                return copyBytes == null ? uint8Array.SubArray(sourceByteOffset) : uint8Array.SubArray(sourceByteOffset, copyBytes.Value + sourceByteOffset);
            }

            // Check for CPU buffer
            if (iView.Buffer is CPUMemoryBuffer)
            {
                // CPU buffer — use standard ILGPU synchronous copy
                var cpuResult = await CopyToHostAsync<byte>(buffer);
                using var uint8Array = new Uint8Array(cpuResult);
                return copyBytes == null ? uint8Array.SubArray(sourceByteOffset) : uint8Array.SubArray(sourceByteOffset, copyBytes.Value + sourceByteOffset);
            }

            throw new NotSupportedException();
        }

        /// <summary>
        /// Copies data from the buffer back to the host as a TypedArray asynchronously.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="buffer"></param>
        /// <returns></returns>
        public static async Task<T> CopyToHostTypeArrayAsync<T>(this MemoryBuffer buffer) where T : TypedArray
        {
            var iView = (IArrayView)buffer;

            // Check for WebGPU buffer
            if (iView.Buffer is WebGPUMemoryBuffer webGpuBuffer)
            {
                var result = await webGpuBuffer.NativeBuffer.CopyToHostUint8ArrayAsync();
                return result.ReCast<T>();
            }

            // Check for WebGL2 buffer — request readback from GL worker
            if (iView.Buffer is WebGLMemoryBuffer webGlBuffer)
            {
                var accel = (WebGLAccelerator)buffer.Accelerator;
                using var readback = await accel.ReadbackAndGetUint8ArrayAsync(webGlBuffer);
                return readback.ReCast<T>();
            }



            // Check for Wasm buffer
            if (iView.Buffer is WasmMemoryBuffer wasmBuffer)
            {
                return new Uint8Array(wasmBuffer.SharedBuffer).ReCast<T>();
            }

            // CPU buffer — use standard ILGPU synchronous copy
            var cpuResult = await CopyToHostAsync<byte>(buffer);
            return new Uint8Array(cpuResult).ReCast<T>();
        }

        #endregion

        #region Unified Synchronization

        /// <summary>
        /// Asynchronously waits for all submitted work to complete.
        /// Works with any ILGPU Accelerator — dispatches to the correct
        /// backend-specific implementation (WebGPU, Wasm, or CPU).
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
