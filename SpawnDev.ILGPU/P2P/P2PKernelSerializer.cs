using System.Reflection;

namespace SpawnDev.ILGPU.P2P;

/// <summary>
/// Serializes kernel references for P2P dispatch.
///
/// Key insight: we don't serialize IL or IR across the wire.
/// Both coordinator and worker run the same app (same Blazor WASM bundle
/// or same .NET assemblies). We just send the method reference (type + name)
/// and the worker compiles locally with its own best backend.
///
/// This means:
///   - Zero overhead for kernel "transfer" — just a string
///   - Worker picks its own backend (WebGPU, CUDA, etc.)
///   - Same NuGet package = same kernel code on both sides
///   - No IR serialization format to maintain
/// </summary>
public static class P2PKernelSerializer
{
    /// <summary>
    /// Serialize a kernel method reference to a dispatch request.
    /// </summary>
    public static KernelDispatchRequest CreateDispatch(
        MethodInfo kernelMethod,
        long gridDimX, long gridDimY = 1, long gridDimZ = 1,
        int groupDimX = 256, int groupDimY = 1, int groupDimZ = 1)
    {
        return new KernelDispatchRequest
        {
            KernelType = kernelMethod.DeclaringType?.AssemblyQualifiedName ?? "",
            KernelMethod = kernelMethod.Name,
            GridDimX = gridDimX,
            GridDimY = gridDimY,
            GridDimZ = gridDimZ,
            GroupDimX = groupDimX,
            GroupDimY = groupDimY,
            GroupDimZ = groupDimZ,
        };
    }

    /// <summary>
    /// Resolve a kernel method on the worker side from the dispatch request.
    /// Looks up the type and method in loaded assemblies.
    /// </summary>
    public static MethodInfo? ResolveKernel(KernelDispatchRequest request)
    {
        var type = Type.GetType(request.KernelType);
        if (type == null)
        {
            // Try all loaded assemblies
            foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
            {
                type = assembly.GetType(request.KernelType);
                if (type != null) break;
            }
        }
        if (type == null) return null;

        return type.GetMethod(request.KernelMethod,
            BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static);
    }

    /// <summary>
    /// Check if this worker can execute the requested kernel
    /// (i.e., the type and method exist in our loaded assemblies).
    /// </summary>
    public static bool CanExecute(KernelDispatchRequest request)
    {
        return ResolveKernel(request) != null;
    }
}
