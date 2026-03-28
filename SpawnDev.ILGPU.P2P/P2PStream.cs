using ILGPU.Runtime;

namespace SpawnDev.ILGPU.P2P;

/// <summary>
/// P2P accelerator stream — queues operations for remote peer execution.
/// </summary>
internal sealed class P2PStream : AcceleratorStream
{
    internal P2PStream(Accelerator accelerator) : base(accelerator) { }

    /// <inheritdoc/>
    public override void Synchronize() { }

    /// <inheritdoc/>
    protected override ProfilingMarker AddProfilingMarkerInternal() =>
        throw new NotSupportedException("Profiling markers not supported for P2P backend.");

    /// <inheritdoc/>
    protected override void DisposeAcceleratorObject(bool disposing) { }
}
