// ---------------------------------------------------------------------------------------
//                               SpawnDev.ILGPU.Wasm
//                    WebAssembly Compute Backend for Blazor WebAssembly
//
// File: WasmCompiledKernel.cs
// ---------------------------------------------------------------------------------------

using global::ILGPU;
using global::ILGPU.Backends;
using global::ILGPU.Backends.EntryPoints;

namespace SpawnDev.ILGPU.Wasm.Backend
{
    /// <summary>
    /// Represents a compiled Wasm kernel containing the binary module bytes.
    /// </summary>
    public sealed class WasmCompiledKernel : CompiledKernel
    {
        /// <summary>
        /// Gets the compiled WebAssembly binary module.
        /// </summary>
        public byte[] WasmBinary { get; }

        /// <summary>
        /// Gets the number of parameter bindings expected by the kernel.
        /// </summary>
        public int BindingCount { get; }

        /// <summary>
        /// Gets the parameter metadata for marshaling arguments.
        /// </summary>
        public List<WasmParamInfo> ParamInfos { get; }

        /// <summary>
        /// Total bytes needed for shared memory allocations within a workgroup.
        /// </summary>
        public int SharedMemorySize { get; }

        /// <summary>
        /// Number of barrier synchronization points in the kernel.
        /// Each barrier uses 8 bytes (2 × i32: arrival counter + sense flag).
        /// </summary>
        public int BarrierCount { get; }

        /// <summary>
        /// Whether this kernel uses shared memory or barriers.
        /// </summary>
        public bool HasBarriers { get; }

        /// <summary>
        /// Number of elements requested for dynamic shared memory (0 if none).
        /// </summary>
        public int DynamicSharedElementSize { get; }

        /// <summary>
        /// Creates a new compiled Wasm kernel.
        /// </summary>
        public WasmCompiledKernel(
            Context context,
            EntryPoint entryPoint,
            byte[] wasmBinary,
            int bindingCount,
            List<WasmParamInfo> paramInfos,
            int sharedMemorySize = 0,
            int barrierCount = 0,
            bool hasBarriers = false,
            int dynamicSharedElementSize = 0)
            : base(context, entryPoint, null)
        {
            WasmBinary = wasmBinary;
            BindingCount = bindingCount;
            ParamInfos = paramInfos;
            SharedMemorySize = sharedMemorySize;
            BarrierCount = barrierCount;
            HasBarriers = hasBarriers;
            DynamicSharedElementSize = dynamicSharedElementSize;
        }
    }

    /// <summary>
    /// Describes a kernel parameter for marshaling.
    /// </summary>
    public class WasmParamInfo
    {
        public int Index { get; set; }
        public string Name { get; set; } = "";
        public bool IsView { get; set; }
        public bool IsScalar { get; set; }
        public byte WasmType { get; set; } = WasmOpCodes.I32;
        public int ElementSize { get; set; } = 4;
    }
}
