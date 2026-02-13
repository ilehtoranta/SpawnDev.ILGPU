//using global::ILGPU;
//using global::ILGPU.Runtime;
//using SpawnDev.ILGPU.WebGPU.Backend;
//using System.Runtime.InteropServices;

//namespace SpawnDev.ILGPU.WebGPU
//{
//    /// <summary>
//    /// Extension methods for ILGPU buffer types to support async readback in WebGPU.
//    /// </summary>
//    public static class WebGPUBufferExtensions
//    {
//        /// <summary>
//        /// Copies data from the GPU buffer back to the host asynchronously.
//        /// Allocates and returns a new T[] array.
//        /// For rendering hot paths, use the overload that accepts a destination array.
//        /// </summary>
//        public static async Task<T[]> CopyToHostAsync<T>(this MemoryBuffer1D<T, Stride1D.Dense> buffer) where T : unmanaged
//        {
//            var result = new T[buffer.Length];
//            await CopyToHostAsync(buffer, result);
//            return result;
//        }

//        /// <summary>
//        /// Copies GPU data into a caller-provided array, reusing a cached staging buffer.
//        /// Zero-allocation hot path for rendering loops.
//        /// </summary>
//        public static async Task CopyToHostAsync<T>(this MemoryBuffer1D<T, Stride1D.Dense> buffer, T[] destination) where T : unmanaged
//        {
//            var internalBuffer = GetWebGPUBuffer((MemoryBuffer)buffer);
//            var nativeBuffer = internalBuffer.NativeBuffer;
//            var elementSize = Marshal.SizeOf<T>();
//            var byteCount = buffer.Length * elementSize;
//            // Read bytes into cached staging buffer, then use Read<T> to fill destination
//            await nativeBuffer.CopyToHostAsync(destination, 0, buffer.Length, elementSize);
//        }

//        /// <summary>
//        /// Copies data from the GPU buffer back to the host asynchronously.
//        /// Allocates and returns a new T[] array (flattened for multi-dimensional buffers).
//        /// For rendering hot paths, use the overload that accepts a destination array.
//        /// </summary>
//        public static async Task<T[]> CopyToHostAsync<T>(this MemoryBuffer buffer) where T : unmanaged
//        {
//            var result = new T[buffer.Length];
//            await CopyToHostAsync(buffer, result);
//            return result;
//        }

//        /// <summary>
//        /// Copies GPU data into a caller-provided array, reusing a cached staging buffer.
//        /// Zero-allocation hot path for rendering loops.
//        /// </summary>
//        public static async Task CopyToHostAsync<T>(this MemoryBuffer buffer, T[] destination) where T : unmanaged
//        {
//            var internalBuffer = GetWebGPUBuffer(buffer);
//            var nativeBuffer = internalBuffer.NativeBuffer;
//            var elementSize = Marshal.SizeOf<T>();
//            await nativeBuffer.CopyToHostAsync(destination, 0, buffer.Length, elementSize);
//        }

//        /// <summary>
//        /// Gets the underlying WebGPUMemoryBuffer from an ILGPU MemoryBuffer.
//        /// </summary>
//        private static WebGPUMemoryBuffer GetWebGPUBuffer(MemoryBuffer buffer)
//        {
//            var iView = (IArrayView)buffer;
//            if (iView.Buffer is WebGPUMemoryBuffer webGpuBuffer)
//                return webGpuBuffer;
//            throw new InvalidOperationException("CopyToHostAsync is only supported for WebGPU-backed buffers. The buffer must be allocated on a WebGPUAccelerator.");
//        }
//    }
//}
