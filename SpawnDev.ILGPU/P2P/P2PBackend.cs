using ILGPU;
using ILGPU.Backends;
using ILGPU.Backends.EntryPoints;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.P2P;

/// <summary>
/// P2P backend — wraps kernel entry point metadata for serialization to remote peers.
/// Peers compile locally with their own backends. IR is the common language.
/// </summary>
public class P2PBackend : Backend
{
    public P2PBackend(Context context)
        : base(context, new P2PCapabilityContext(), BackendType.Wasm, new P2PArgumentMapper(context))
    {
        // BackendType.Wasm is a placeholder — P2P doesn't have its own type in the enum.
        // We just need a valid Backend to satisfy the framework.
    }

    /// <inheritdoc/>
    protected override CompiledKernel Compile(
        EntryPoint entryPoint,
        in BackendContext backendContext,
        in KernelSpecialization specialization)
    {
        // P2P doesn't compile to a shader — it wraps the entry point
        // for serialization to remote peers who compile locally.
        return new P2PCompiledKernel(Context, entryPoint, null);
    }
}

/// <summary>
/// Compiled kernel for P2P — holds entry point metadata for serialization.
/// </summary>
public class P2PCompiledKernel : CompiledKernel
{
    public P2PCompiledKernel(Context context, EntryPoint entryPoint, KernelInfo? info)
        : base(context, entryPoint, info)
    {
    }
}

/// <summary>
/// Argument mapper for P2P kernel parameters.
/// </summary>
public class P2PArgumentMapper : ArgumentMapper
{
    public P2PArgumentMapper(Context context) : base(context) { }

    protected override Type MapViewType(Type viewType, Type elementType) => viewType;

    protected override void MapViewInstance<TILEmitter, TSource, TTarget>(
        in TILEmitter emitter, Type viewType, in TSource source, in TTarget target) { }
}
