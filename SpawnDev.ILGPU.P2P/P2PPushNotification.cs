namespace SpawnDev.ILGPU.P2P;

/// <summary>
/// Client-side interface for P2P swarm push notifications.
/// When the swarm needs more compute power, opted-in users get a
/// Web Push notification recalling them to contribute.
///
/// Flow:
/// 1. User clicks "Enable Notifications" → browser prompts
/// 2. PushSubscription sent to hub.spawndev.com/compute/subscribe
/// 3. When swarm TFLOPS drops below threshold, hub sends push
/// 4. User's device wakes, opens the join link, reconnects
///
/// Requires:
/// - VAPID public key from hub.spawndev.com
/// - Service worker registered (already done by webtorrent-sw.js)
/// - hub.spawndev.com /compute/subscribe endpoint (NOT YET IMPLEMENTED)
/// </summary>
public class P2PPushNotification
{
    /// <summary>
    /// VAPID public key from the push notification server.
    /// Must match the server's private key.
    /// TODO: Retrieve from hub.spawndev.com/compute/vapid-key endpoint.
    /// </summary>
    public string? VapidPublicKey { get; set; }

    /// <summary>
    /// The push subscription endpoint (from browser PushManager.subscribe).
    /// </summary>
    public PushSubscriptionInfo? Subscription { get; set; }

    /// <summary>
    /// True if the user has granted notification permission.
    /// </summary>
    public bool IsEnabled => Subscription != null;

    /// <summary>
    /// Minimum TFLOPS threshold — if swarm drops below this, send push.
    /// </summary>
    public double MinTflopsThreshold { get; set; } = 5.0;

    /// <summary>
    /// Check if the swarm needs help and should send notifications.
    /// Called periodically by the coordinator.
    /// </summary>
    public bool ShouldSendHelpWanted(double currentTflops) =>
        currentTflops < MinTflopsThreshold;
}

/// <summary>
/// Push subscription info to send to the server.
/// Mirrors the Web Push API PushSubscription object.
/// </summary>
public record PushSubscriptionInfo
{
    /// <summary>The push service endpoint URL.</summary>
    public string Endpoint { get; init; } = "";

    /// <summary>ECDH P-256 public key (base64url).</summary>
    public string P256dhKey { get; init; } = "";

    /// <summary>Authentication secret (base64url).</summary>
    public string AuthSecret { get; init; } = "";

    /// <summary>Swarm hash to rejoin when notification is tapped.</summary>
    public string SwarmHash { get; init; } = "";

    /// <summary>Join link URL to open on tap.</summary>
    public string JoinLink { get; init; } = "";
}
