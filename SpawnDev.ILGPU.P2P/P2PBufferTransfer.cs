namespace SpawnDev.ILGPU.P2P;

/// <summary>
/// Handles tensor/buffer data transfer between coordinator and workers.
/// Large buffers are chunked for WebRTC data channel compatibility
/// (max ~256KB per message on most browsers).
///
/// Transfer flow:
///   Coordinator: BufferSend header → chunk 0 → chunk 1 → ... → chunk N
///   Worker: receives all chunks → reassembles → stores in local buffer
///   Worker (after kernel): BufferData header → chunk 0 → ... → chunk N
///   Coordinator: receives all chunks → updates local buffer
/// </summary>
public class P2PBufferTransfer
{
    /// <summary>
    /// Maximum chunk size in bytes. WebRTC data channels typically
    /// support ~256KB per message, but we use 64KB for reliability
    /// across all browsers and network conditions.
    /// </summary>
    public int MaxChunkSize { get; set; } = 64 * 1024;

    private readonly System.Collections.Concurrent.ConcurrentDictionary<string, BufferTransferState> _inProgress = new();

    /// <summary>
    /// Fired when a complete buffer has been received.
    /// </summary>
    public event Action<string, byte[]>? OnBufferReceived; // bufferId, data

    /// <summary>
    /// Split a buffer into chunks for transmission.
    /// </summary>
    public BufferChunk[] CreateChunks(string bufferId, byte[] data)
    {
        int totalChunks = (data.Length + MaxChunkSize - 1) / MaxChunkSize;
        var chunks = new BufferChunk[totalChunks];

        for (int i = 0; i < totalChunks; i++)
        {
            int offset = i * MaxChunkSize;
            int length = Math.Min(MaxChunkSize, data.Length - offset);
            var chunkData = new byte[length];
            Array.Copy(data, offset, chunkData, 0, length);

            chunks[i] = new BufferChunk
            {
                BufferId = bufferId,
                ChunkIndex = i,
                TotalChunks = totalChunks,
                TotalBytes = data.Length,
                Data = chunkData,
            };
        }

        return chunks;
    }

    /// <summary>
    /// Process a received chunk. Returns true if the buffer is now complete.
    /// </summary>
    /// <summary>
    /// Maximum allowed total chunks per transfer (prevents OOM from malicious headers).
    /// 1GB / 64KB = 16384 chunks max.
    /// </summary>
    public int MaxTotalChunks { get; set; } = 16384;

    /// <summary>
    /// Maximum allowed total bytes per transfer (prevents OOM).
    /// Default: 1GB.
    /// </summary>
    public long MaxTotalBytes { get; set; } = 1L * 1024 * 1024 * 1024;

    public bool ReceiveChunk(BufferChunk chunk)
    {
        // SECURITY: Bounds check to prevent OOM from malicious chunk headers
        if (chunk.TotalChunks <= 0 || chunk.TotalChunks > MaxTotalChunks)
            return false;
        if (chunk.TotalBytes <= 0 || chunk.TotalBytes > MaxTotalBytes)
            return false;
        if (chunk.ChunkIndex < 0 || chunk.ChunkIndex >= chunk.TotalChunks)
            return false;

        var state = _inProgress.GetOrAdd(chunk.BufferId, _ => new BufferTransferState
        {
            BufferId = chunk.BufferId,
            TotalChunks = chunk.TotalChunks,
            TotalBytes = chunk.TotalBytes,
            ReceivedChunks = new byte[chunk.TotalChunks][],
            ReceivedCount = 0,
        });

        // Store chunk (idempotent — handles retransmits). Lock per-state for chunk assembly.
        lock (state)
        {
            if (state.ReceivedChunks[chunk.ChunkIndex] == null)
            {
                state.ReceivedChunks[chunk.ChunkIndex] = chunk.Data;
                state.ReceivedCount++;
            }

            // Check if complete
            if (state.ReceivedCount >= state.TotalChunks)
            {
                var assembled = Assemble(state);
                _inProgress.TryRemove(chunk.BufferId, out _);
                OnBufferReceived?.Invoke(chunk.BufferId, assembled);
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// Get transfer progress for a buffer (0.0 to 1.0).
    /// </summary>
    public double GetProgress(string bufferId)
    {
        if (_inProgress.TryGetValue(bufferId, out var state))
            return (double)state.ReceivedCount / state.TotalChunks;
        return 0;
    }

    /// <summary>
    /// Cancel an in-progress transfer.
    /// </summary>
    public void CancelTransfer(string bufferId)
    {
        _inProgress.TryRemove(bufferId, out _);
    }

    /// <summary>
    /// Number of in-progress transfers.
    /// </summary>
    public int ActiveTransfers => _inProgress.Count;

    /// <summary>
    /// Maximum time (seconds) a transfer can be in progress before cleanup. Default: 120s.
    /// </summary>
    public int TransferTimeoutSeconds { get; set; } = 120;

    /// <summary>
    /// Clean up stale transfers that have exceeded the timeout.
    /// Returns the number of transfers cleaned up.
    /// </summary>
    public int CleanupStaleTransfers()
    {
        var now = DateTime.UtcNow;
        int cleaned = 0;
        foreach (var kvp in _inProgress)
        {
            if ((now - kvp.Value.StartTime).TotalSeconds > TransferTimeoutSeconds)
            {
                if (_inProgress.TryRemove(kvp.Key, out _))
                    cleaned++;
            }
        }
        return cleaned;
    }

    private byte[] Assemble(BufferTransferState state)
    {
        var result = new byte[state.TotalBytes];
        int pos = 0;
        for (int i = 0; i < state.TotalChunks; i++)
        {
            var chunk = state.ReceivedChunks[i];
            if (chunk != null)
            {
                Array.Copy(chunk, 0, result, pos, chunk.Length);
                pos += chunk.Length;
            }
        }
        return result;
    }
}

/// <summary>
/// A single chunk of buffer data for transmission.
/// </summary>
public class BufferChunk
{
    public string BufferId { get; set; } = "";
    public int ChunkIndex { get; set; }
    public int TotalChunks { get; set; }
    public int TotalBytes { get; set; }
    public byte[] Data { get; set; } = Array.Empty<byte>();
}

/// <summary>
/// Tracks an in-progress buffer transfer.
/// </summary>
internal class BufferTransferState
{
    public string BufferId { get; set; } = "";
    public int TotalChunks { get; set; }
    public int TotalBytes { get; set; }
    public byte[][] ReceivedChunks { get; set; } = Array.Empty<byte[]>();
    public int ReceivedCount { get; set; }
    public DateTime StartTime { get; set; } = DateTime.UtcNow;
}
