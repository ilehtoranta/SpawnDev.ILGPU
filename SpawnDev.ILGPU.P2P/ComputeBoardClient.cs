using System.Net.Http.Json;
using System.Text;
using System.Text.Json;

namespace SpawnDev.ILGPU.P2P;

/// <summary>
/// Client for the compute request board on hub.spawndev.com.
/// Posts signed "need TFLOPS" requests and browses available swarms.
///
/// Authentication:
///   POST requires SwarmIdentity signature (OwnerFingerprint + PublicKey + Signature)
///   DELETE requires fingerprint query parameter matching the owner
/// </summary>
public class ComputeBoardClient
{
    private readonly HttpClient _http;

    /// <summary>
    /// Base URL of the compute board server.
    /// Default: hub.spawndev.com.
    /// </summary>
    public string BaseUrl { get; set; } = "https://hub.spawndev.com:44365";

    public ComputeBoardClient(HttpClient? httpClient = null)
    {
        _http = httpClient ?? new HttpClient();
    }

    /// <summary>
    /// Post a signed compute request to the board.
    /// </summary>
    public async Task<ComputeBoardRequest?> PostRequestAsync(ComputeBoardRequest request)
    {
        try
        {
            var response = await _http.PostAsJsonAsync($"{BaseUrl}/compute/request", request);
            if (response.IsSuccessStatusCode)
                return await response.Content.ReadFromJsonAsync<ComputeBoardRequest>();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ComputeBoard] POST failed: {ex.Message}");
        }
        return null;
    }

    /// <summary>
    /// Browse active compute requests.
    /// </summary>
    public async Task<List<ComputeBoardRequest>> GetRequestsAsync()
    {
        try
        {
            return await _http.GetFromJsonAsync<List<ComputeBoardRequest>>($"{BaseUrl}/compute/requests") ?? new();
        }
        catch { return new(); }
    }

    /// <summary>
    /// Get aggregate compute stats.
    /// </summary>
    public async Task<ComputeBoardStats?> GetStatsAsync()
    {
        try
        {
            return await _http.GetFromJsonAsync<ComputeBoardStats>($"{BaseUrl}/compute/stats");
        }
        catch { return null; }
    }

    /// <summary>
    /// Remove a compute request. Requires the owner's fingerprint.
    /// </summary>
    public async Task<bool> RemoveRequestAsync(string id, string ownerFingerprint)
    {
        try
        {
            var response = await _http.DeleteAsync(
                $"{BaseUrl}/compute/request/{id}?fingerprint={Uri.EscapeDataString(ownerFingerprint)}");
            return response.IsSuccessStatusCode;
        }
        catch { return false; }
    }

    /// <summary>
    /// Post a signed request for the given P2PCompute instance.
    /// Signs the request payload with the swarm's identity.
    /// </summary>
    public async Task<ComputeBoardRequest?> PostFromComputeAsync(
        P2PCompute compute, string purpose, double tflopsNeeded)
    {
        var request = new ComputeBoardRequest
        {
            SwarmName = "Compute Swarm",
            Purpose = purpose,
            OwnerFingerprint = compute.Identity.Fingerprint,
            PublicKey = Convert.ToBase64String(compute.Identity.PublicKeySpki),
            TflopsNeeded = tflopsNeeded,
            TflopsAvailable = compute.TotalTflops,
            PeerCount = compute.PeerCount,
            MagnetLink = compute.MagnetLink,
            JoinLink = compute.JoinLink,
        };

        // Sign the request payload
        var payload = JsonSerializer.SerializeToUtf8Bytes(new
        {
            request.SwarmName,
            request.Purpose,
            request.OwnerFingerprint,
            request.TflopsNeeded,
        });
        var signature = await compute.Identity.SignAsync(payload);
        request.Signature = Convert.ToBase64String(signature);

        return await PostRequestAsync(request);
    }
}

/// <summary>
/// Client-side DTO for compute board requests.
/// Mirrors SpawnDev.WebTorrent.Server.ComputeRequest.
/// </summary>
public class ComputeBoardRequest
{
    public string Id { get; set; } = "";
    public string SwarmName { get; set; } = "";
    public string Purpose { get; set; } = "";
    public string? OwnerFingerprint { get; set; }
    public string? PublicKey { get; set; }
    public string? Signature { get; set; }
    public double TflopsNeeded { get; set; }
    public double TflopsAvailable { get; set; }
    public int PeerCount { get; set; }
    public string? MagnetLink { get; set; }
    public string? JoinLink { get; set; }
    public DateTimeOffset PostedAt { get; set; }
    public DateTimeOffset ExpiresAt { get; set; }
}

/// <summary>
/// Client-side DTO for compute board stats.
/// </summary>
public class ComputeBoardStats
{
    public int ActiveRequests { get; set; }
    public double TotalTflopsNeeded { get; set; }
    public double TotalTflopsAvailable { get; set; }
    public int UniqueSwarms { get; set; }
}
