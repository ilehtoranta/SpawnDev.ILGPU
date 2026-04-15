using System.Reflection;

namespace SpawnDev.ILGPU.P2P;

/// <summary>
/// Chains multiple kernel dispatches to execute sequentially on the SAME peer.
/// Intermediate results stay on the peer — only the final result is copied back.
/// This eliminates network round-trips between pipeline stages.
///
/// Usage:
///   var pipeline = new P2PDispatchPipeline(accelerator)
///       .Add(typeof(MyKernels), nameof(MyKernels.Preprocess), 1024,
///            ("input", rawData, 4), ("temp", null, 4))
///       .Add(typeof(MyKernels), nameof(MyKernels.Transform), 1024,
///            ("temp", null, 4), ("output", null, 4))
///       .Add(typeof(MyKernels), nameof(MyKernels.Postprocess), 1024,
///            ("output", null, 4));
///
///   var result = await pipeline.ExecuteAsync();
///   // Only the final kernel's modified buffers are transferred back.
///   // Intermediate "temp" buffer stayed on the peer the entire time.
/// </summary>
public class P2PDispatchPipeline
{
    private readonly P2PAccelerator _accelerator;
    private readonly List<PipelineStage> _stages = new();

    public P2PDispatchPipeline(P2PAccelerator accelerator)
    {
        _accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
    }

    /// <summary>
    /// Add a kernel dispatch stage to the pipeline.
    /// Buffers with the same bufferId across stages share the same GPU memory on the peer.
    /// </summary>
    public P2PDispatchPipeline Add(Type kernelType, string methodName, long gridDimX,
        params (string bufferId, byte[]? data, int elementSize)[] buffers)
    {
        var method = kernelType.GetMethod(methodName,
            BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new ArgumentException($"Kernel method not found: {kernelType.Name}.{methodName}");

        _stages.Add(new PipelineStage
        {
            KernelMethod = method,
            GridDimX = gridDimX,
            Buffers = buffers.Select(b => new PipelineBuffer
            {
                BufferId = b.bufferId,
                Data = b.data,
                ElementSize = b.elementSize,
            }).ToArray(),
        });
        return this;
    }

    /// <summary>
    /// Execute all stages sequentially on the same peer.
    /// Returns the result of the LAST stage.
    /// </summary>
    public async Task<PipelineResult> ExecuteAsync()
    {
        if (_stages.Count == 0)
            throw new InvalidOperationException("Pipeline has no stages.");
        if (_accelerator.Dispatcher == null)
            throw new InvalidOperationException("Dispatcher not set.");

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var stageResults = new List<KernelDispatchResult>();
        // Track all intermediate buffer IDs for cleanup on failure
        var intermediateBufferIds = new HashSet<string>();

        try
        {
            // Execute each stage sequentially - all on the same peer
            // The dispatcher's buffer locality tracking keeps data on the first peer
            for (int i = 0; i < _stages.Count; i++)
            {
                var stage = _stages[i];
                var request = P2PKernelSerializer.CreateDispatch(stage.KernelMethod, stage.GridDimX);

                var bindings = new BufferBinding[stage.Buffers.Length];
                for (int j = 0; j < stage.Buffers.Length; j++)
                {
                    var buf = stage.Buffers[j];
                    bindings[j] = new BufferBinding
                    {
                        ParameterIndex = j + 1,
                        BufferId = buf.BufferId,
                        Length = buf.Data?.Length / buf.ElementSize ?? 0,
                        ElementSize = buf.ElementSize,
                    };
                    // Track intermediate buffers (those without initial data that are
                    // created by the pipeline, not supplied by the caller)
                    if (buf.Data == null)
                        intermediateBufferIds.Add(buf.BufferId);
                }
                request.Buffers = bindings;

                // First stage sends data; subsequent stages reuse buffers on the peer
                var result = await _accelerator.Dispatcher.DispatchAsync(request);
                stageResults.Add(result);

                if (!result.Success)
                {
                    sw.Stop();
                    return new PipelineResult
                    {
                        Success = false,
                        FailedStage = i,
                        Error = $"Stage {i} ({stage.KernelMethod.Name}) failed: {result.Error}",
                        WallTimeMs = sw.Elapsed.TotalMilliseconds,
                        StageResults = stageResults.ToArray(),
                        IntermediateBufferIds = intermediateBufferIds.ToArray(),
                    };
                }
            }

            sw.Stop();
            return new PipelineResult
            {
                Success = true,
                WallTimeMs = sw.Elapsed.TotalMilliseconds,
                StageResults = stageResults.ToArray(),
                TotalStages = _stages.Count,
                IntermediateBufferIds = intermediateBufferIds.ToArray(),
            };
        }
        catch (Exception ex)
        {
            sw.Stop();
            return new PipelineResult
            {
                Success = false,
                FailedStage = stageResults.Count,
                Error = $"Pipeline exception at stage {stageResults.Count}: {ex.Message}",
                WallTimeMs = sw.Elapsed.TotalMilliseconds,
                StageResults = stageResults.ToArray(),
                IntermediateBufferIds = intermediateBufferIds.ToArray(),
            };
        }
    }

    /// <summary>Number of stages in the pipeline.</summary>
    public int StageCount => _stages.Count;

    private record PipelineStage
    {
        public MethodInfo KernelMethod { get; init; } = null!;
        public long GridDimX { get; init; }
        public PipelineBuffer[] Buffers { get; init; } = Array.Empty<PipelineBuffer>();
    }

    private record PipelineBuffer
    {
        public string BufferId { get; init; } = "";
        public byte[]? Data { get; init; }
        public int ElementSize { get; init; }
    }
}

/// <summary>
/// Result of a pipeline execution.
/// </summary>
public record PipelineResult
{
    /// <summary>True if all stages completed successfully.</summary>
    public bool Success { get; init; }

    /// <summary>Index of the failed stage (-1 if all succeeded).</summary>
    public int FailedStage { get; init; } = -1;

    /// <summary>Error message from the failed stage.</summary>
    public string? Error { get; init; }

    /// <summary>Wall clock time for the entire pipeline (ms).</summary>
    public double WallTimeMs { get; init; }

    /// <summary>Total number of stages.</summary>
    public int TotalStages { get; init; }

    /// <summary>Per-stage results.</summary>
    public KernelDispatchResult[] StageResults { get; init; } = Array.Empty<KernelDispatchResult>();

    /// <summary>IDs of intermediate buffers created during pipeline execution.
    /// Callers should clean these up on the worker when the pipeline is done or fails.</summary>
    public string[] IntermediateBufferIds { get; init; } = Array.Empty<string>();

    /// <summary>Sum of all stage durations (ms).</summary>
    public double TotalStageDurationMs => StageResults.Sum(s => s.DurationMs);
}
