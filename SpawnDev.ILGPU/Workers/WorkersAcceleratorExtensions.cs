// ---------------------------------------------------------------------------------------
//                               SpawnDev.ILGPU.Workers
//                 Web Worker Compute Library for Blazor WebAssembly
//
// File: WorkersAcceleratorExtensions.cs
//
// Extension methods for Workers accelerator synchronization.
// ---------------------------------------------------------------------------------------

namespace SpawnDev.ILGPU.Workers
{
    /// <summary>
    /// Extension methods for WorkersAccelerator.
    /// </summary>
    public static class WorkersAcceleratorExtensions
    {
        /// <summary>
        /// Asynchronously waits for all submitted worker dispatch to complete.
        /// This awaits the PendingWork task set by the most recent RunKernel call.
        /// </summary>
        /// <param name="accelerator">The Workers accelerator.</param>
        /// <returns>A task that completes when all workers are done.</returns>
        public static async Task SynchronizeAsync(this WorkersAccelerator accelerator)
        {
            if (accelerator.PendingWorkTasks.Count > 0)
            {
                await Task.WhenAll(accelerator.PendingWorkTasks);
                accelerator.PendingWorkTasks.Clear();
            }
        }
    }
}
