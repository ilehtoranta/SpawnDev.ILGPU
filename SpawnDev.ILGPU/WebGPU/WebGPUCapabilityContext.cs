using ILGPU.Runtime;

namespace SpawnDev.ILGPU.WebGPU
{
    /// <summary>
    /// Represents the capabilities of the WebGPU backend.
    /// Capabilities are initialized conservatively and updated when the accelerator
    /// is created with the actual enabled device features.
    /// </summary>
    public sealed class WebGPUCapabilityContext : CapabilityContext
    {
        /// <summary>
        /// Creates a new WebGPU capability context.
        /// Defaults are conservative; call UpdateFromEnabledFeatures after device creation
        /// to reflect the actual enabled features (shader-f16, shader-float64, shader-int64).
        /// </summary>
        public WebGPUCapabilityContext()
        {
            // Conservative defaults until UpdateFromEnabledFeatures is called
            Float16 = false;
            Float64 = false;
            Int64 = false;
        }

        /// <summary>
        /// Updates capabilities based on the WebGPU device's actually enabled features.
        /// Called by WebGPUAccelerator after device creation.
        /// </summary>
        /// <param name="enabledFeatures">The set of features successfully enabled on the device (e.g. from adapter.requestDevice).</param>
        internal void UpdateFromEnabledFeatures(HashSet<string> enabledFeatures)
        {
            if (enabledFeatures == null) return;
            Float16 = enabledFeatures.Contains("shader-f16");
            Float64 = enabledFeatures.Contains("shader-float64");
            Int64 = enabledFeatures.Contains("shader-int64");
        }

        /// <summary>
        /// Supports Float64 (double) data type.
        /// </summary>
        public bool Float64 { get; internal set; }

        /// <summary>
        /// Supports Int64 (long) data type.
        /// </summary>
        public bool Int64 { get; internal set; }
    }
}
