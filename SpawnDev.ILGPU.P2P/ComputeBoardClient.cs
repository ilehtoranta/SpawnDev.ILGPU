using System.Net.Http.Json;

namespace SpawnDev.ILGPU.P2P;

/// <summary>
/// Client for the compute request board on hub.spawndev.com.
/// Posts "need TFLOPS" requests and browses available swarms.
///
/// Used by both the demo UI and sovereign AI swarms.
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
    /// Post a compute request to the board.
    /// </summary>
    public async Task<ComputeBoardRequest?> PostRequestAsync(ComputeBoardRequest request)
    {
        try
        {
            var response = await _http.PostAsJsonAsync($"{BaseUrl}/compute/request", request);
            if (response.IsSuccessStatusCode)
                return await response.Content.ReadFromJsonAsync<ComputeBoardRequest>();
        }
        catch { }
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
    /// Remove a compute request.
    /// </summary>
    public async Task<bool> RemoveRequestAsync(string id)
    {
        try
        {
            var response = await _http.DeleteAsync($"{BaseUrl}/compute/request/{id}");
            return response.IsSuccessStatusCode;
        }
        catch { return false; }
    }

    /// <summary>
    /// Post a request for the given P2PCompute instance.
    /// </summary>
    public async Task<ComputeBoardRequest?> PostFromComputeAsync(P2PCompute compute, string purpose, double tflopsNeeded)
    {
        return await PostRequestAsync(new ComputeBoardRequest
        {
            SwarmName = "Compute Swarm",
            Purpose = purpose,
            OwnerFingerprint = compute.Identity.Fingerprint,
            TflopsNeeded = tflopsNeeded,
            TflopsAvailable = compute.TotalTflops,
            PeerCount = compute.PeerCount,
            MagnetLink = compute.MagnetLink,
            JoinLink = compute.JoinLink,
        });
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
