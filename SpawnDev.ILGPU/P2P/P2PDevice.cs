using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.P2P;

/// <summary>
/// Represents a P2P accelerator device — an aggregate of remote peer GPUs
/// connected via SpawnDev.WebTorrent WebRTC data channels.
/// </summary>
[DeviceType(AcceleratorType.P2P)]
public sealed class P2PDevice : Device
{
    /// <summary>
    /// Detects P2P devices and adds them to the registry.
    /// Always available — peers connect after creation.
    /// </summary>
    public static void GetDevices(
        Predicate<P2PDevice> predicate,
        DeviceRegistry registry)
    {
        if (predicate is null) throw new ArgumentNullException(nameof(predicate));
        if (registry is null) throw new ArgumentNullException(nameof(registry));
        var device = new P2PDevice();
        registry.Register(device, predicate);
    }

    public P2PDevice()
    {
        Name = "P2P Distributed Accelerator";
        WarpSize = 32;
        MaxGroupSize = new Index3D(256, 256, 256);
        MaxNumThreadsPerGroup = 256;
        NumMultiprocessors = 1; // updated when peers connect
        MaxNumThreadsPerMultiprocessor = 256;
        MaxGridSize = new Index3D(int.MaxValue, 65536, 65536);
        MemorySize = 0; // updated when peers connect
        MaxSharedMemoryPerGroup = 0;
        MaxConstantMemory = 0;
        Capabilities = new P2PCapabilityContext();
    }

    /// <inheritdoc/>
    public override Accelerator CreateAccelerator(Context context)
    {
        return new P2PAccelerator(context, this);
    }
}

/// <summary>
/// Capability context for the P2P backend.
/// </summary>
public sealed class P2PCapabilityContext : CapabilityContext
{
}
