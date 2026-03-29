namespace SpawnDev.ILGPU.P2P;

/// <summary>
/// Swarm join and access policies. Configured by the owner.
/// Controls who can join, what access level they get, and resource limits.
/// </summary>
public class SwarmPolicy
{
    /// <summary>
    /// Join permission mode — controls how new peers are admitted.
    /// </summary>
    public JoinMode JoinPermission { get; set; } = JoinMode.Open;

    /// <summary>
    /// Maximum number of peers allowed in the swarm. 0 = unlimited.
    /// </summary>
    public int MaxPeers { get; set; } = 0;

    /// <summary>
    /// Maximum TFLOPS the swarm will accept. 0 = unlimited.
    /// Useful for rate-limiting resource consumption.
    /// </summary>
    public double MaxTflops { get; set; } = 0;

    /// <summary>
    /// Whether anonymous peers (no SwarmIdentity) can join.
    /// When false, all peers must present a signed identity.
    /// </summary>
    public bool AllowAnonymous { get; set; } = true;

    /// <summary>
    /// Default role assigned to new peers on join.
    /// In Approval mode, this is the role they get BEFORE owner approves.
    /// </summary>
    public SwarmRole DefaultRole { get; set; } = SwarmRole.Worker;

    /// <summary>
    /// Role assigned after owner manually approves a pending peer.
    /// Only used in Approval mode.
    /// </summary>
    public SwarmRole ApprovedRole { get; set; } = SwarmRole.Worker;

    /// <summary>
    /// Whether to remember previously joined peers and auto-admit them.
    /// When true, a peer that joined before is admitted without re-approval.
    /// </summary>
    public bool RememberPeers { get; set; } = true;
}

/// <summary>
/// Join permission modes for the swarm.
/// </summary>
public enum JoinMode
{
    /// <summary>
    /// Anyone can join immediately with full worker access.
    /// Best for public compute pools and demos.
    /// </summary>
    Open,

    /// <summary>
    /// Anyone can join, but starts with minimal access (can't receive dispatches)
    /// until the owner manually approves them.
    /// Peers see the swarm and can browse, but don't contribute until approved.
    /// </summary>
    Approval,

    /// <summary>
    /// Only previously joined peers (remembered by fingerprint) can rejoin.
    /// New devices must be manually added by the owner via key management.
    /// Best for private/family compute groups.
    /// </summary>
    KnownOnly,

    /// <summary>
    /// Only peers with keys in the KeyRegistry can join.
    /// Maximum security — owner explicitly authorizes every device.
    /// </summary>
    InviteOnly,
}
