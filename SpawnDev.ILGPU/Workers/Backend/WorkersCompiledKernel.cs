// ---------------------------------------------------------------------------------------
//                               SpawnDev.ILGPU.Workers
//                 Web Worker Compute Library for Blazor WebAssembly
//
// File: WorkersCompiledKernel.cs
// ---------------------------------------------------------------------------------------

using global::ILGPU;
using global::ILGPU.Backends;
using global::ILGPU.Backends.EntryPoints;

namespace SpawnDev.ILGPU.Workers.Backend
{
    /// <summary>
    /// Represents a compiled Workers/JavaScript kernel.
    /// </summary>
    public sealed class WorkersCompiledKernel : CompiledKernel
    {
        /// <summary>
        /// Gets the generated JavaScript source code for this kernel.
        /// </summary>
        public string JSSource { get; }

        /// <summary>
        /// Gets the number of bindings (parameters) expected by this kernel.
        /// </summary>
        public int BindingCount { get; }

        /// <summary>
        /// Creates a new compiled Workers kernel.
        /// </summary>
        /// <param name="context">The ILGPU context.</param>
        /// <param name="entryPoint">The kernel entry point.</param>
        /// <param name="jsSource">The generated JavaScript source code.</param>
        /// <param name="bindingCount">The number of parameter bindings.</param>
        public WorkersCompiledKernel(
            Context context,
            EntryPoint entryPoint,
            string jsSource,
            int bindingCount)
            : base(context, entryPoint, null)
        {
            JSSource = jsSource;
            BindingCount = bindingCount;
        }
    }
}
