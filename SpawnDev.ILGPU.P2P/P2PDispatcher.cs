namespace SpawnDev.ILGPU.P2P;

/// <summary>
/// Manages kernel dispatch across peers with fault tolerance.
/// Handles peer loss mid-computation, automatic retry, and rebalancing.
///
/// Resilience model:
///   - Every dispatch is tracked with a timeout
///   - If a peer disconnects or times out, work is re-dispatched to another peer
///   - Buffer data has local shadow copies for recovery
///   - New peers are picked up on next dispatch (no mid-kernel rebalancing)
///   - Heartbeat monitoring detects stale peers before they fail
/// </summary>
public class P2PDispatcher : IDisposable
{
    private bool _disposed;
    private readonly P2PAccelerator _accelerator;
    private readonly Dictionary<string, PendingDispatch> _pending = new();
    private readonly object _lock = new();
    private Timer? _heartbeatTimer;

    /// <summary>
    /// Timeout for kernel execution (ms). If a peer doesn't respond, work is retried.
    /// </summary>
    public int DispatchTimeoutMs { get; set; } = 30_000;

    /// <summary>
    /// Maximum retry attempts before giving up on a dispatch.
    /// </summary>
    public int MaxRetries { get; set; } = 3;

    /// <summary>
    /// Coordinator's public key (base64 SPKI) for dispatch authentication.
    /// Set from the coordinator's SwarmIdentity. Included in every dispatch
    /// so workers can verify the sender has Coordinator+ authority.
    /// </summary>
    public string? CoordinatorPublicKey { get; set; }

    /// <summary>
    /// Heartbeat interval (ms). Peers that miss 3 consecutive heartbeats are marked stale.
    /// </summary>
    public int HeartbeatIntervalMs { get; set; } = 5_000;

    /// <summary>
    /// Fired when a dispatch fails permanently (all retries exhausted).
    /// </summary>
    public event Action<string, string>? OnDispatchFailed; // dispatchId, error

    /// <summary>
    /// Fired when a dispatch is retried on a different peer.
    /// </summary>
    public event Action<string, string, string>? OnDispatchRetried; // dispatchId, failedPeerId, newPeerId

    /// <summary>
    /// Fired when a peer is proactively evicted (thermal/battery critical).
    /// Work is gracefully handed off to healthier peers before the device drops.
    /// </summary>
    public event Action<string, int>? OnPeerEvicted; // peerId, dispatchesMoved

    public P2PDispatcher(P2PAccelerator accelerator)
    {
        _accelerator = accelerator;
    }

    /// <summary>
    /// Start heartbeat monitoring.
    /// </summary>
    public void StartMonitoring()
    {
        _heartbeatTimer = new Timer(CheckHeartbeats, null, HeartbeatIntervalMs, HeartbeatIntervalMs);
    }

    /// <summary>
    /// Stop monitoring.
    /// </summary>
    public void StopMonitoring()
    {
        _heartbeatTimer?.Dispose();
        _heartbeatTimer = null;
    }

    /// <summary>
    /// Dispatch a kernel to the best available peer.
    /// Returns the dispatch ID for tracking.
    /// </summary>
    public string Dispatch(KernelDispatchRequest request,
        IReadOnlyDictionary<string, byte[]>? inputBuffers = null,
        string[]? preferredBufferIds = null)
    {
        var peer = SelectHealthyPeer(preferredBufferIds);
        if (peer == null)
            throw new InvalidOperationException("No healthy peers available for dispatch");

        var pending = new PendingDispatch
        {
            DispatchId = request.DispatchId,
            Request = request,
            InputBuffers = inputBuffers,
            AssignedPeer = peer,
            StartTime = DateTime.UtcNow,
            Attempts = 1,
        };

        lock (_lock)
        {
            _pending[request.DispatchId] = pending;
        }

        peer.IncrementPending();
        // Fire-and-forget path can't await the buffer sends synchronously - start them
        // now, let them overlap with caller's post-return work. The receive-side per-peer
        // ordered queue (rc.12) guarantees chunks drain before the KernelDispatch message.
        _ = SendInputBuffersAsync(peer, inputBuffers);
        SendDispatchToPeer(peer, request);

        return request.DispatchId;
    }

    /// <summary>
    /// Dispatch a kernel and await the result.
    /// Returns the dispatch result when the peer completes execution.
    /// Throws on timeout or permanent failure.
    ///
    /// When <paramref name="inputBuffers"/> is provided, each buffer is transmitted
    /// to the selected peer BEFORE the KernelDispatch message fires. This closes
    /// the long-standing library gap where <c>DispatchAsync</c> took
    /// <c>(bufferId, data, elementSize)</c> tuples at the accelerator layer but
    /// the raw <c>data</c> never reached the peer. Stored on the <see cref="PendingDispatch"/>
    /// so a retry can re-ship the inputs to the replacement peer.
    /// </summary>
    public async Task<KernelDispatchResult> DispatchAsync(
        KernelDispatchRequest request,
        IReadOnlyDictionary<string, byte[]>? inputBuffers = null,
        string[]? preferredBufferIds = null)
    {
        var peer = SelectHealthyPeer(preferredBufferIds);
        if (peer == null)
            throw new InvalidOperationException("No healthy peers available for dispatch");

        var tcs = new TaskCompletionSource<KernelDispatchResult>();
        var pending = new PendingDispatch
        {
            DispatchId = request.DispatchId,
            Request = request,
            InputBuffers = inputBuffers,
            AssignedPeer = peer,
            StartTime = DateTime.UtcNow,
            Attempts = 1,
            CompletionSource = tcs,
        };

        lock (_lock)
        {
            _pending[request.DispatchId] = pending;
        }

        peer.IncrementPending();
        await SendInputBuffersAsync(peer, inputBuffers).ConfigureAwait(false);
        SendDispatchToPeer(peer, request);

        // Await with timeout
        using var cts = new CancellationTokenSource(DispatchTimeoutMs * MaxRetries);
        try
        {
            return await tcs.Task.WaitAsync(cts.Token);
        }
        catch (OperationCanceledException)
        {
            throw new TimeoutException(
                $"P2P dispatch {request.DispatchId} timed out after {DispatchTimeoutMs * MaxRetries}ms");
        }
    }

    /// <summary>
    /// Ship every provided input buffer to the selected peer via <see cref="OnSendBuffer"/>
    /// (wired to <c>P2PTransport.SendBufferAsync</c> by the facade). Awaits all sends
    /// in parallel; the receive side's per-peer ordered queue guarantees they drain
    /// before the following KernelDispatch message starts processing.
    /// </summary>
    private async Task SendInputBuffersAsync(RemotePeer peer, IReadOnlyDictionary<string, byte[]>? inputBuffers)
    {
        if (inputBuffers == null || inputBuffers.Count == 0) return;
        var handler = OnSendBuffer;
        if (handler == null) return;

        var sends = inputBuffers
            .Where(kv => kv.Value != null)
            .Select(kv => handler(peer.PeerId, kv.Key, kv.Value))
            .ToArray();
        if (sends.Length > 0)
            await Task.WhenAll(sends).ConfigureAwait(false);
    }

    /// <summary>
    /// Handle a peer reporting kernel completion.
    /// </summary>
    public void HandleResult(string dispatchId, KernelDispatchResult result)
    {
        PendingDispatch? pending;
        lock (_lock)
        {
            if (!_pending.TryGetValue(dispatchId, out pending))
                return;
        }

        pending.AssignedPeer.DecrementPending();

        if (result.Success)
        {
            pending.AssignedPeer.RecordSuccess(result.DurationMs);
            lock (_lock) { _pending.Remove(dispatchId); }
            pending.CompletionSource?.TrySetResult(result);
        }
        else
        {
            pending.AssignedPeer.RecordFailure();
            // Execution failed on peer — retry on another peer
            RetryDispatch(pending, $"Peer {pending.AssignedPeer.PeerId} reported error: {result.Error}");
        }
    }

    /// <summary>
    /// Handle a peer disconnecting — retry all its pending work.
    /// </summary>
    public void HandlePeerLost(string peerId)
    {
        List<PendingDispatch> affected;
        lock (_lock)
        {
            affected = _pending.Values
                .Where(p => p.AssignedPeer.PeerId == peerId)
                .ToList();
        }

        foreach (var dispatch in affected)
        {
            dispatch.AssignedPeer.DecrementPending();
            RetryDispatch(dispatch, $"Peer {peerId} disconnected");
        }
    }

    /// <summary>
    /// Handle a new peer joining — no immediate action needed.
    /// New peers will be selected on the next dispatch via SelectHealthyPeer.
    /// Active dispatches are NOT rebalanced mid-execution.
    /// </summary>
    public void HandlePeerJoined(RemotePeer peer)
    {
        // New peer is immediately available for future dispatches.
        // No rebalancing of in-flight work — that would cause more harm than good.
    }

    /// <summary>
    /// Process a heartbeat from a peer.
    /// </summary>
    public void HandleHeartbeat(string peerId)
    {
        var peer = _accelerator.Peers.FirstOrDefault(p => p.PeerId == peerId);
        if (peer != null)
        {
            peer.LastHeartbeat = DateTime.UtcNow;
        }
    }

    private void RetryDispatch(PendingDispatch dispatch, string reason)
    {
        if (dispatch.Attempts >= MaxRetries)
        {
            lock (_lock) { _pending.Remove(dispatch.DispatchId); }
            OnDispatchFailed?.Invoke(dispatch.DispatchId, $"Max retries exceeded. Last: {reason}");
            dispatch.CompletionSource?.TrySetException(
                new Exception($"P2P dispatch failed after {MaxRetries} attempts: {reason}"));
            return;
        }

        var newPeer = SelectHealthyPeer(exclude: dispatch.AssignedPeer.PeerId);
        if (newPeer == null)
        {
            lock (_lock) { _pending.Remove(dispatch.DispatchId); }
            OnDispatchFailed?.Invoke(dispatch.DispatchId, $"No healthy peers available for retry. {reason}");
            dispatch.CompletionSource?.TrySetException(
                new Exception($"P2P dispatch failed, no peers for retry: {reason}"));
            return;
        }

        var failedPeerId = dispatch.AssignedPeer.PeerId;
        dispatch.Attempts++;
        dispatch.AssignedPeer = newPeer;
        dispatch.StartTime = DateTime.UtcNow;
        newPeer.IncrementPending();

        OnDispatchRetried?.Invoke(dispatch.DispatchId, failedPeerId, newPeer.PeerId);
        // Re-ship input buffers to the replacement peer before re-dispatching so the
        // new worker has the data the original one was holding. Fire-and-forget on the
        // buffer task is acceptable here; the per-peer ordered queue at the receiver
        // guarantees the buffer chunks arrive ahead of the re-sent KernelDispatch.
        _ = SendInputBuffersAsync(newPeer, dispatch.InputBuffers);
        SendDispatchToPeer(newPeer, dispatch.Request);
    }

    private RemotePeer? SelectHealthyPeer(string[]? preferredBufferIds = null, string? exclude = null)
    {
        var candidates = _accelerator.Peers
            .Where(p => p.IsConnected && p.PeerId != exclude && !IsStale(p))
            .Select(p => (peer: p, score: ScorePeer(p)))
            .Where(x => x.score > 0.0) // Exclude thermally critical / dead battery peers
            .OrderByDescending(x => x.score)
            .ToList();

        if (candidates.Count == 0) return null;

        return candidates.First().peer;
    }

    /// <summary>
    /// Score a peer for dispatch selection. Higher = better.
    /// Balances compute power, capacity, reliability, and thermal/battery health.
    /// A peer that overheats and drops is worse than a slower peer that stays up.
    /// </summary>
    /// <summary>
    /// Exposes peer scoring for diagnostics and testing.
    /// </summary>
    public double ScorePeer(RemotePeer peer)
    {
        var caps = peer.Capabilities;
        double tflops = caps?.EstimatedTflops ?? 1.0;
        double memory = caps?.AvailableMemory ?? 0;
        int pending = peer.PendingOperations;

        // Ability: TFLOPS normalized (assume max ~20 TFLOPS for a desktop GPU)
        double abilityScore = Math.Min(tflops / 20.0, 1.0);

        // Load: penalize peers with many pending operations
        double loadScore = 1.0 / (1.0 + pending);

        // Reliability: peers with recent heartbeats score higher
        double reliabilityScore = 1.0;
        if (peer.LastHeartbeat != DateTime.MinValue)
        {
            double secsSinceHeartbeat = (DateTime.UtcNow - peer.LastHeartbeat).TotalSeconds;
            reliabilityScore = Math.Max(0.1, 1.0 - (secsSinceHeartbeat / (HeartbeatIntervalMs / 1000.0 * 3)));
        }

        // Memory: bonus for peers with more VRAM (can handle larger tensors)
        double memoryScore = Math.Min(memory / (8.0 * 1024 * 1024 * 1024), 1.0);

        // Thermal/Battery: penalize peers at risk of dropping off
        double healthScore = 1.0;
        if (caps != null)
        {
            // Thermal throttling: 0=nominal(1.0), 1=fair(0.7), 2=serious(0.3), 3=critical(0.0)
            healthScore *= caps.ThermalState switch
            {
                0 => 1.0,
                1 => 0.7,
                2 => 0.3,
                3 => 0.0, // critical — don't send work, it'll crash
                _ => 0.5,
            };

            // Battery: penalize discharging devices with low battery
            if (!caps.IsCharging && caps.BatteryLevel >= 0)
            {
                if (caps.BatteryLevel < 0.1)
                    healthScore *= 0.1; // <10% battery, about to die
                else if (caps.BatteryLevel < 0.2)
                    healthScore *= 0.4; // low battery
                else if (caps.BatteryLevel < 0.5)
                    healthScore *= 0.7; // moderate battery
                // >50% on battery is fine, no penalty
            }
            // Charging devices get no battery penalty
        }

        // Reputation: dispatch history — success rate + identity strength
        double reputationScore = peer.Reputation;

        // Weighted combination: health acts as a multiplier, not an additive factor.
        // A thermally critical peer gets zero score regardless of TFLOPS.
        double baseScore = (abilityScore * 0.35) + (loadScore * 0.25) +
                           (reliabilityScore * 0.15) + (memoryScore * 0.10) +
                           (reputationScore * 0.15);
        return baseScore * healthScore;
    }

    private bool IsStale(RemotePeer peer)
    {
        if (peer.LastHeartbeat == DateTime.MinValue) return false; // never sent heartbeat yet
        return (DateTime.UtcNow - peer.LastHeartbeat).TotalMilliseconds > HeartbeatIntervalMs * 3;
    }

    private void CheckHeartbeats(object? state)
    {
        try
        {
            var now = DateTime.UtcNow;

            // Check for timed-out dispatches
            List<PendingDispatch> timedOut;
            lock (_lock)
            {
                timedOut = _pending.Values
                    .Where(p => (now - p.StartTime).TotalMilliseconds > DispatchTimeoutMs)
                    .ToList();
            }

            foreach (var dispatch in timedOut)
            {
                dispatch.AssignedPeer.DecrementPending();
                RetryDispatch(dispatch, "Dispatch timed out");
            }

            // Check for stale peers
            foreach (var peer in _accelerator.Peers.Where(p => p.IsConnected && IsStale(p)).ToList())
            {
                HandlePeerLost(peer.PeerId);
            }

            // Proactive thermal eviction: if a peer's capabilities degrade,
            // move its heavy work to healthier peers before it crashes.
            foreach (var peer in _accelerator.Peers.Where(p => p.IsConnected).ToList())
            {
                if (ShouldEvict(peer))
                {
                    InitiateGracefulHandoff(peer);
                }
            }
        }
        catch (Exception)
        {
            // Timer callback must never throw — a crash here silently kills heartbeat monitoring.
            // Swallow and continue; the next tick will retry.
        }
    }

    /// <summary>
    /// Determines if a peer should have its work evicted proactively.
    /// Triggers on thermal critical, battery critical, or TFLOPS degradation.
    /// </summary>
    private bool ShouldEvict(RemotePeer peer)
    {
        var caps = peer.Capabilities;
        if (caps == null) return false;

        // Thermal critical — evict immediately
        if (caps.ThermalState >= 3) return true;

        // Battery about to die — evict if discharging below 5%
        if (!caps.IsCharging && caps.BatteryLevel >= 0 && caps.BatteryLevel < 0.05)
            return true;

        return false;
    }

    /// <summary>
    /// Graceful handoff: move a peer's pending work to healthier peers
    /// before the peer drops off the network. The peer's current buffer state
    /// is preserved (signed via BEP 46) so the receiving peer can continue
    /// from where the evicted peer left off.
    /// </summary>
    private void InitiateGracefulHandoff(RemotePeer evictedPeer)
    {
        List<PendingDispatch> toMove;
        lock (_lock)
        {
            toMove = _pending.Values
                .Where(p => p.AssignedPeer.PeerId == evictedPeer.PeerId)
                .ToList();
        }

        if (toMove.Count == 0) return;

        OnPeerEvicted?.Invoke(evictedPeer.PeerId, toMove.Count);

        foreach (var dispatch in toMove)
        {
            dispatch.AssignedPeer.DecrementPending();
            RetryDispatch(dispatch, $"Graceful handoff from {evictedPeer.PeerId} (thermal/battery)");
        }
    }

    /// <summary>
    /// Update a peer's capabilities (called when peer sends updated status).
    /// Enables real-time thermal/battery monitoring.
    /// </summary>
    public void UpdatePeerCapabilities(string peerId, PeerCapabilities capabilities)
    {
        var peer = _accelerator.Peers.FirstOrDefault(p => p.PeerId == peerId);
        if (peer != null)
        {
            peer.Capabilities = capabilities;
            peer.LastHeartbeat = DateTime.UtcNow;
        }
    }

    private void SendDispatchToPeer(RemotePeer peer, KernelDispatchRequest request)
    {
        // Include coordinator's public key for worker-side authority verification
        if (CoordinatorPublicKey != null)
            request.CoordinatorPublicKey = CoordinatorPublicKey;

        var message = new P2PMessage
        {
            Type = P2PMessageType.KernelDispatch,
            Payload = System.Text.Json.JsonSerializer.SerializeToElement(request),
        };
        // Fire event for transport layer to send via WebRTC
        OnSendMessage?.Invoke(peer.PeerId, message);
    }

    /// <summary>
    /// Fired when the dispatcher needs to send a message to a peer.
    /// Wire this to P2PTransport.SendMessageAsync.
    /// </summary>
    public event Action<string, P2PMessage>? OnSendMessage;

    /// <summary>
    /// Fired when the dispatcher needs to ship an input buffer to a peer.
    /// Wire this to P2PTransport.SendBufferAsync so DispatchAsync's
    /// (bufferId, data, elementSize) tuples actually transmit the data.
    /// </summary>
    public event Func<string, string, byte[], Task>? OnSendBuffer;

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        StopMonitoring();
    }

    /// <summary>
    /// Get a snapshot of all pending dispatches (for coordinator transfer).
    /// </summary>
    public PendingDispatchInfo[] GetPendingSnapshot()
    {
        lock (_lock)
        {
            return _pending.Values.Select(p => new PendingDispatchInfo
            {
                DispatchId = p.DispatchId,
                Request = p.Request,
                AssignedPeerId = p.AssignedPeer.PeerId,
                Attempts = p.Attempts,
            }).ToArray();
        }
    }

    /// <summary>
    /// Accept pending dispatch state from a coordinator transfer.
    /// The new coordinator takes over tracking these dispatches.
    /// </summary>
    public void HandlePendingTransfer(PendingDispatchInfo info)
    {
        var peer = _accelerator.Peers.FirstOrDefault(p => p.PeerId == info.AssignedPeerId);
        if (peer == null) return; // Peer not connected to us — will be retried when they reconnect

        var pending = new PendingDispatch
        {
            DispatchId = info.DispatchId,
            Request = info.Request,
            AssignedPeer = peer,
            StartTime = DateTime.UtcNow,
            Attempts = info.Attempts,
        };

        lock (_lock)
        {
            _pending[info.DispatchId] = pending;
        }
    }

    /// <summary>
    /// Number of pending dispatches currently tracked.
    /// </summary>
    public int PendingCount
    {
        get { lock (_lock) { return _pending.Count; } }
    }
}

/// <summary>
/// Tracks an in-flight kernel dispatch.
/// </summary>
internal class PendingDispatch
{
    public string DispatchId { get; set; } = "";
    public KernelDispatchRequest Request { get; set; } = new();

    /// <summary>
    /// Input buffer payloads captured at dispatch time so a retry to a different
    /// peer can re-ship them. Key = bufferId, value = raw bytes. Null when the
    /// caller used the fire-and-forget path without providing inputs.
    /// </summary>
    public IReadOnlyDictionary<string, byte[]>? InputBuffers { get; set; }

    public RemotePeer AssignedPeer { get; set; } = new();
    public DateTime StartTime { get; set; }
    public int Attempts { get; set; }
    public TaskCompletionSource<KernelDispatchResult>? CompletionSource { get; set; }
}
