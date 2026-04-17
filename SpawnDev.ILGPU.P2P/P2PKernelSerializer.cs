using System.Reflection;

namespace SpawnDev.ILGPU.P2P;

/// <summary>
/// Serializes kernel references for P2P dispatch with security restrictions.
///
/// SECURITY: Only methods on explicitly registered types can be resolved.
/// This prevents arbitrary code execution via malicious dispatch requests.
/// Both coordinator and worker must register the same kernel types.
/// </summary>
public static class P2PKernelSerializer
{
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<Type, bool> _allowedTypes = new();
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, MethodInfo?> _resolveCache = new();

    /// <summary>
    /// Register a type whose static methods can be dispatched via P2P.
    /// Both coordinator and worker must register the same types.
    /// </summary>
    public static void RegisterKernelType(Type type)
    {
        _allowedTypes[type] = true;
        // Clear resolve cache - a previously-rejected type may now be allowed
        _resolveCache.Clear();
    }

    /// <summary>
    /// Register multiple kernel types at once.
    /// </summary>
    public static void RegisterKernelTypes(params Type[] types)
    {
        foreach (var type in types)
            _allowedTypes[type] = true;
    }

    /// <summary>
    /// Check if a type is registered for P2P dispatch.
    /// </summary>
    public static bool IsTypeAllowed(Type type) => _allowedTypes.ContainsKey(type);

    /// <summary>
    /// Clear the allowlist (allows all types again). For testing only.
    /// </summary>
    public static void ClearAllowlist()
    {
        _allowedTypes.Clear();
        _resolveCache.Clear();
    }

    /// <summary>
    /// Number of registered kernel types.
    /// </summary>
    public static int AllowedTypeCount => _allowedTypes.Count;

    /// <summary>
    /// Serialize a kernel method reference to a dispatch request.
    /// The method's declaring type must be registered via RegisterKernelType.
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
    /// SECURITY: Only resolves methods on registered types.
    /// Returns null if the type is not in the allowlist.
    /// </summary>
    public static MethodInfo? ResolveKernel(KernelDispatchRequest request)
    {
        var cacheKey = $"{request.KernelType}::{request.KernelMethod}";

        // Check cache first - only successful resolutions are cached
        if (_resolveCache.TryGetValue(cacheKey, out var cached))
            return cached;

        var type = Type.GetType(request.KernelType);
        if (type == null)
        {
            foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
            {
                type = assembly.GetType(request.KernelType);
                if (type != null) break;
            }
        }
        if (type == null) return null;

        // SECURITY: Only allow registered types. Fail-closed - if no types
        // are registered, reject ALL dispatches. Never allow arbitrary code execution.
        if (!_allowedTypes.ContainsKey(type))
            return null;

        // Include NonPublic so private/internal static kernel methods (the C# default when
        // no access modifier is specified) resolve correctly. Type allowlist is the
        // security boundary, not method visibility.
        var method = type.GetMethod(request.KernelMethod,
            BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static);
        if (method != null)
            _resolveCache[cacheKey] = method;
        return method;
    }

    /// <summary>
    /// Check if this worker can execute the requested kernel.
    /// </summary>
    public static bool CanExecute(KernelDispatchRequest request)
    {
        return ResolveKernel(request) != null;
    }
}
