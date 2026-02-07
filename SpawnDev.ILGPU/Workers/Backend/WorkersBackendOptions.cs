// ---------------------------------------------------------------------------------------
//                               SpawnDev.ILGPU.Workers
//                 Web Worker Compute Library for Blazor WebAssembly
//
// File: WorkersBackendOptions.cs
// ---------------------------------------------------------------------------------------

namespace SpawnDev.ILGPU.Workers.Backend
{
    /// <summary>
    /// Configuration options for the Workers backend.
    /// </summary>
    public class WorkersBackendOptions
    {
        /// <summary>
        /// Default options instance.
        /// </summary>
        public static readonly WorkersBackendOptions Default = new();

        /// <summary>
        /// Gets or sets the maximum number of Web Workers to spawn.
        /// 0 means auto-detect from navigator.hardwareConcurrency.
        /// </summary>
        public int MaxWorkerCount { get; set; } = 0;

        /// <summary>
        /// Gets or sets the default workgroup size (items per worker batch).
        /// Defaults to 64.
        /// </summary>
        public int WorkgroupSize { get; set; } = 64;

        /// <summary>
        /// Gets or sets whether verbose logging is enabled.
        /// </summary>
        public bool VerboseLogging { get; set; } = false;
    }
}
