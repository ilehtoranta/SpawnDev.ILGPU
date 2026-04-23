using System.Collections.Concurrent;
using System.Linq.Expressions;
using System.Reflection;
using System.Text.Json;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.ILGPU.P2P;

/// <summary>
/// Runtime kernel launcher for P2P workers.
/// Uses reflection to call ILGPU's typed LoadAutoGroupedStreamKernel,
/// then DynamicInvoke to dispatch with runtime-resolved arguments.
///
/// This bridges the gap between P2P's runtime dispatch (kernel method known
/// only from deserialization) and ILGPU's compile-time typed API.
///
/// Flow:
///   1. Resolve kernel MethodInfo (P2PKernelSerializer)
///   2. LoadAndCache — reflection-invoke LoadAutoGroupedStreamKernel on local accelerator
///   3. Launch — allocate typed GPU buffers, DynamicInvoke the cached launcher, readback
/// </summary>
public class P2PKernelLauncher
{
    private readonly Accelerator _accelerator;
    private readonly ConcurrentDictionary<string, CachedLauncher> _cache = new();

    /// <summary>Number of cached kernel launchers.</summary>
    public int CachedCount => _cache.Count;

    public P2PKernelLauncher(Accelerator accelerator)
    {
        _accelerator = accelerator;
    }

    /// <summary>
    /// Load and cache a kernel launcher from a kernel method.
    /// Uses reflection to call the appropriate LoadAutoGroupedStreamKernel overload.
    /// </summary>
    public CachedLauncher LoadAndCache(MethodInfo kernelMethod)
    {
        var cacheKey = $"{kernelMethod.DeclaringType?.FullName}.{kernelMethod.Name}";
        return _cache.GetOrAdd(cacheKey, _ => BuildLauncher(kernelMethod));
    }

    /// <summary>
    /// Execute a kernel with the given buffer data.
    /// Allocates GPU buffers, launches the kernel, reads back results.
    /// Uses async sync/readback for cross-backend compatibility (WebGPU, WebGL, Wasm, CPU, CUDA).
    /// </summary>
    /// <param name="kernelMethod">The kernel method to execute.</param>
    /// <param name="gridDim">Total work items (Index1D extent).</param>
    /// <param name="bufferBindings">Buffer data keyed by parameter index.</param>
    /// <param name="scalarParams">
    /// Optional JSON-encoded scalar parameter values (from <see cref="KernelDispatchRequest.ScalarParams"/>).
    /// Format: <c>{"paramIndex": jsonValue, ...}</c> where paramIndex matches the kernel signature.
    /// When null or empty, scalar parameters fall back to <c>default(T)</c>.
    /// </param>
    /// <returns>Modified buffer data keyed by parameter index.</returns>
    public async Task<Dictionary<int, byte[]>> ExecuteAsync(
        MethodInfo kernelMethod,
        long gridDim,
        Dictionary<int, BufferData> bufferBindings,
        byte[]? scalarParams = null)
    {
        ArgumentNullException.ThrowIfNull(kernelMethod);
        ArgumentNullException.ThrowIfNull(bufferBindings);
        var launcher = LoadAndCache(kernelMethod);
        var paramInfos = kernelMethod.GetParameters();
        var args = new object[paramInfos.Length];
        var allocatedBuffers = new List<(int paramIdx, MemoryBuffer buffer, Type elementType, long count)>();

        // Decode scalar parameters once. JSON keys are parameter indices as strings.
        Dictionary<int, JsonElement>? scalars = null;
        if (scalarParams != null && scalarParams.Length > 0)
        {
            var raw = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(scalarParams);
            if (raw != null && raw.Count > 0)
            {
                scalars = new Dictionary<int, JsonElement>(raw.Count);
                foreach (var kvp in raw)
                {
                    if (int.TryParse(kvp.Key, out var idx))
                        scalars[idx] = kvp.Value;
                }
            }
        }

        try
        {
            // Build kernel arguments
            for (int i = 0; i < paramInfos.Length; i++)
            {
                var paramType = paramInfos[i].ParameterType;

                if (i == 0 && typeof(IIndex).IsAssignableFrom(paramType))
                {
                    // First parameter is the index — set from grid dimension
                    args[i] = CreateIndex(paramType, gridDim);
                    continue;
                }

                // Check if this parameter has a buffer binding
                if (bufferBindings.TryGetValue(i, out var bufData))
                {
                    var (buffer, view, elemType) = AllocateAndFill(paramType, bufData);
                    args[i] = view;
                    allocatedBuffers.Add((i, buffer, elemType, bufData.ElementCount));
                }
                else if (scalars != null && scalars.TryGetValue(i, out var scalarJson))
                {
                    // Scalar parameter transmitted by the coordinator
                    args[i] = DeserializeScalar(scalarJson, paramType);
                }
                else
                {
                    // No value transmitted — fall back to default(T). Kernel signatures with
                    // meaningful scalar parameters should always set them via CreateDispatch's
                    // scalarValues argument; this default is for optional/unused scalars only.
                    args[i] = paramType.IsValueType ? Activator.CreateInstance(paramType)! : null!;
                }
            }

            // Launch the kernel
            launcher.Launcher.DynamicInvoke(args);

            // Async sync — flushes queue and waits for completion on all backends
            await _accelerator.SynchronizeAsync();

            // Read back modified buffers using backend-agnostic async readback
            var results = new Dictionary<int, byte[]>();
            foreach (var (paramIdx, buffer, elemType, count) in allocatedBuffers)
            {
                results[paramIdx] = await ReadBackAsync(buffer, elemType, count);
            }

            return results;
        }
        finally
        {
            // Dispose allocated GPU buffers
            foreach (var (_, buffer, _, _) in allocatedBuffers)
            {
                buffer.Dispose();
            }
        }
    }

    private CachedLauncher BuildLauncher(MethodInfo kernelMethod)
    {
        var paramTypes = kernelMethod.GetParameters().Select(p => p.ParameterType).ToArray();
        var paramCount = paramTypes.Length; // includes TIndex

        // Create the Action<TIndex, T1, T2, ...> delegate from the static kernel method
        var actionType = Expression.GetActionType(paramTypes);
        var kernelDelegate = Delegate.CreateDelegate(actionType, kernelMethod);

        // Find the right LoadAutoGroupedStreamKernel overload via reflection
        // It's an extension method on KernelLoaders with (paramCount) generic type parameters
        // Signature: LoadAutoGroupedStreamKernel<TIndex, T1, ...>(this Accelerator, Action<TIndex, T1, ...>)
        var loaderType = typeof(KernelLoaders);
        var loadMethod = loaderType.GetMethods(BindingFlags.Public | BindingFlags.Static)
            .Where(m => m.Name == "LoadAutoGroupedStreamKernel"
                     && m.GetGenericArguments().Length == paramCount
                     && m.GetParameters().Length == 2) // (Accelerator, Action<...>)
            .FirstOrDefault();

        if (loadMethod == null)
            throw new InvalidOperationException(
                $"No LoadAutoGroupedStreamKernel overload found for {paramCount} type parameters. " +
                $"ILGPU supports up to 14 kernel parameters.");

        // Make the generic method with our actual types
        var genericLoad = loadMethod.MakeGenericMethod(paramTypes);

        // Invoke: returns Action<TIndex, T1, T2, ...> launcher delegate
        var launcher = (Delegate)genericLoad.Invoke(null, new object[] { _accelerator, kernelDelegate })!;

        return new CachedLauncher
        {
            KernelMethod = kernelMethod,
            Launcher = launcher,
            ParameterTypes = paramTypes,
        };
    }

    private static object CreateIndex(Type indexType, long dim)
    {
        if (indexType == typeof(Index1D))
            return new Index1D((int)dim);
        if (indexType == typeof(Index2D))
            return new Index2D((int)dim, 1);
        if (indexType == typeof(Index3D))
            return new Index3D((int)dim, 1, 1);
        if (indexType == typeof(LongIndex1D))
            return new LongIndex1D(dim);
        // Fallback
        return Activator.CreateInstance(indexType, dim)!;
    }

    private (MemoryBuffer buffer, object view, Type elementType) AllocateAndFill(
        Type paramType, BufferData data)
    {
        // Extract element type from ArrayView<T> or ArrayView1D<T, Stride1D.Dense>
        var elemType = ExtractElementType(paramType);
        // Check if kernel uses ArrayView<T> (needs explicit conversion from ArrayView1D)
        bool useBaseView = paramType.IsGenericType &&
            paramType.GetGenericTypeDefinition().Name == "ArrayView`1";

        if (elemType == typeof(float))
            return AllocateTyped<float>(data, 4, useBaseView);
        if (elemType == typeof(int))
            return AllocateTyped<int>(data, 4, useBaseView);
        if (elemType == typeof(double))
            return AllocateTyped<double>(data, 8, useBaseView);
        if (elemType == typeof(byte))
            return AllocateTyped<byte>(data, 1, useBaseView);
        if (elemType == typeof(long))
            return AllocateTyped<long>(data, 8, useBaseView);
        if (elemType == typeof(short))
            return AllocateTyped<short>(data, 2, useBaseView);
        if (elemType == typeof(uint))
            return AllocateTyped<uint>(data, 4, useBaseView);
        if (elemType == typeof(ulong))
            return AllocateTyped<ulong>(data, 8, useBaseView);
        if (elemType == typeof(System.Half))
            return AllocateTyped<System.Half>(data, 2, useBaseView);

        // Fallback: treat as byte buffer
        return AllocateTyped<byte>(data, 1, useBaseView);
    }

    private (MemoryBuffer buffer, object view, Type elementType) AllocateTyped<T>(
        BufferData data, int elemSize, bool useBaseView)
        where T : unmanaged
    {
        var count = data.ElementCount > 0 ? data.ElementCount : data.RawData.Length / elemSize;
        var buffer = _accelerator.Allocate1D<T>(count);

        // Copy data to GPU
        if (data.RawData.Length > 0)
        {
            var hostArray = new T[count];
            Buffer.BlockCopy(data.RawData, 0, hostArray, 0,
                Math.Min(data.RawData.Length, (int)(count * elemSize)));
            buffer.CopyFromCPU(hostArray);
        }

        // Return the right view type for DynamicInvoke compatibility:
        // ArrayView<T> for kernels using the base type (DynamicInvoke won't do implicit conversion)
        // ArrayView1D<T, Stride1D.Dense> for kernels using the strided type
        object view = useBaseView
            ? (object)(ArrayView<T>)buffer.View  // explicit cast uses implicit operator
            : buffer.View;

        return (buffer, view, typeof(T));
    }

    private async Task<byte[]> ReadBackAsync(MemoryBuffer buffer, Type elementType, long count)
    {
        if (elementType == typeof(float))
            return ReadBackFromHost(await buffer.CopyToHostAsync<float>(), count, 4);
        if (elementType == typeof(int))
            return ReadBackFromHost(await buffer.CopyToHostAsync<int>(), count, 4);
        if (elementType == typeof(double))
            return ReadBackFromHost(await buffer.CopyToHostAsync<double>(), count, 8);
        if (elementType == typeof(byte))
            return ReadBackFromHost(await buffer.CopyToHostAsync<byte>(), count, 1);
        if (elementType == typeof(long))
            return ReadBackFromHost(await buffer.CopyToHostAsync<long>(), count, 8);
        if (elementType == typeof(short))
            return ReadBackFromHost(await buffer.CopyToHostAsync<short>(), count, 2);
        if (elementType == typeof(uint))
            return ReadBackFromHost(await buffer.CopyToHostAsync<uint>(), count, 4);
        if (elementType == typeof(ulong))
            return ReadBackFromHost(await buffer.CopyToHostAsync<ulong>(), count, 8);
        if (elementType == typeof(System.Half))
            return ReadBackFromHost(await buffer.CopyToHostAsync<System.Half>(), count, 2);

        return Array.Empty<byte>();
    }

    private static byte[] ReadBackFromHost<T>(T[] hostArray, long count, int elemSize)
        where T : unmanaged
    {
        var bytes = new byte[count * elemSize];
        Buffer.BlockCopy(hostArray, 0, bytes, 0, Math.Min(bytes.Length, hostArray.Length * elemSize));
        return bytes;
    }

    /// <summary>
    /// Deserialize a transmitted scalar JSON value to the kernel's expected parameter type.
    /// Handles every ILGPU primitive explicitly; falls back to JSON-based deserialization
    /// for structs and other complex types.
    /// </summary>
    private static object DeserializeScalar(JsonElement json, Type targetType)
    {
        if (targetType == typeof(float)) return json.GetSingle();
        if (targetType == typeof(double)) return json.GetDouble();
        if (targetType == typeof(int)) return json.GetInt32();
        if (targetType == typeof(uint)) return json.GetUInt32();
        if (targetType == typeof(long)) return json.GetInt64();
        if (targetType == typeof(ulong)) return json.GetUInt64();
        if (targetType == typeof(short)) return json.GetInt16();
        if (targetType == typeof(ushort)) return json.GetUInt16();
        if (targetType == typeof(byte)) return json.GetByte();
        if (targetType == typeof(sbyte)) return json.GetSByte();
        if (targetType == typeof(bool)) return json.GetBoolean();
        // System.Half is transmitted as float (STJ has no native Half support).
        if (targetType == typeof(System.Half)) return (System.Half)json.GetSingle();
        // Struct or other: let STJ handle it.
        var raw = json.GetRawText();
        var value = JsonSerializer.Deserialize(raw, targetType);
        return value ?? Activator.CreateInstance(targetType)!;
    }

    /// <summary>
    /// Extract the element type from an ArrayView-like parameter type.
    /// Handles ArrayView&lt;T&gt;, ArrayView1D&lt;T, Stride1D.Dense&gt;, etc.
    /// </summary>
    private static Type ExtractElementType(Type paramType)
    {
        if (paramType.IsGenericType)
        {
            var genDef = paramType.GetGenericTypeDefinition();
            // ArrayView<T> or ArrayView1D<T, TStride>
            if (genDef.Name.StartsWith("ArrayView"))
                return paramType.GetGenericArguments()[0];
        }
        // Non-generic struct parameter (scalar) — return the type itself
        return paramType;
    }
}

/// <summary>
/// A cached kernel launcher — the compiled+loaded kernel ready for dispatch.
/// </summary>
public class CachedLauncher
{
    /// <summary>The original kernel method.</summary>
    public MethodInfo KernelMethod { get; set; } = null!;

    /// <summary>The typed launcher delegate (Action&lt;TIndex, T1, T2, ...&gt;).</summary>
    public Delegate Launcher { get; set; } = null!;

    /// <summary>Parameter types in order.</summary>
    public Type[] ParameterTypes { get; set; } = Array.Empty<Type>();
}

/// <summary>
/// Buffer data for a kernel parameter — raw bytes + metadata.
/// </summary>
public class BufferData
{
    /// <summary>Raw byte data.</summary>
    public byte[] RawData { get; set; } = Array.Empty<byte>();

    /// <summary>Number of elements.</summary>
    public long ElementCount { get; set; }

    /// <summary>Size of each element in bytes.</summary>
    public int ElementSize { get; set; }
}
