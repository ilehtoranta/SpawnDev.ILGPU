using SpawnDev.ILGPU.P2P;

namespace SpawnDev.ILGPU.Demo.Shared.Services;

/// <summary>
/// Shared singleton service that holds the active P2P compute swarm.
/// Injected into all demo pages — when a swarm is active, demos can
/// dispatch work across peers instead of running locally.
///
/// Lifecycle managed by the /compute page. Other pages are consumers only.
/// </summary>
public class P2PSwarmService : IAsyncDisposable
{
    private P2PCompute? _compute;

    /// <summary>The active P2P compute instance, or null if no swarm is active.</summary>
    public P2PCompute? Compute => _compute;

    /// <summary>True if a swarm is active and has at least one connected peer.</summary>
    public bool IsActive => _compute != null && _compute.PeerCount > 0;

    /// <summary>True if a swarm exists (even with zero peers — coordinator is active).</summary>
    public bool HasSwarm => _compute != null;

    /// <summary>The P2P accelerator for dispatching distributed work. Null if no swarm.</summary>
    public P2PAccelerator? Accelerator => _compute?.Accelerator;

    /// <summary>Number of connected peers.</summary>
    public int PeerCount => _compute?.PeerCount ?? 0;

    /// <summary>Total TFLOPS across the swarm.</summary>
    public double TotalTflops => _compute?.TotalTflops ?? 0;

    /// <summary>Current role in the swarm.</summary>
    public P2PRole? Role => _compute?.Role;

    /// <summary>Fired when the swarm state changes (created, joined, peer count change, disposed).</summary>
    public event Action? OnStateChanged;

    /// <summary>
    /// Set the active compute instance. Called by the /compute page when creating or joining a swarm.
    /// </summary>
    public void SetCompute(P2PCompute compute)
    {
        _compute = compute;

        // Wire capacity changes to our state changed event
        compute.Coordinator.OnCapacityChanged += () => OnStateChanged?.Invoke();
        compute.Coordinator.OnPeerJoined += _ => OnStateChanged?.Invoke();
        compute.Coordinator.OnPeerLeft += _ => OnStateChanged?.Invoke();
        compute.Coordinator.OnCoordinatorChanged += _ => OnStateChanged?.Invoke();

        OnStateChanged?.Invoke();
    }

    /// <summary>
    /// Clear the active swarm. Called when disposing or leaving.
    /// </summary>
    public async Task ClearAsync()
    {
        var compute = _compute;
        _compute = null;
        OnStateChanged?.Invoke();

        if (compute != null)
            await compute.DisposeAsync();
    }

    public async ValueTask DisposeAsync()
    {
        await ClearAsync();
    }
}
