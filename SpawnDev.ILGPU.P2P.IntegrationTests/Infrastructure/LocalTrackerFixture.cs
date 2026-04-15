using System.Diagnostics;
using System.Net.Http;
using NUnit.Framework;

namespace SpawnDev.ILGPU.P2P.IntegrationTests.Infrastructure;

/// <summary>
/// NUnit assembly-level fixture that starts a local SpawnDev.WebTorrent.ServerApp
/// instance for P2P tracker signaling. Eliminates dependency on hub.spawndev.com
/// and gives tests full control of their signaling infrastructure.
///
/// The ServerApp runs on ws://localhost:5561/announce (HTTP) and
/// wss://localhost:5560/announce (HTTPS).
/// </summary>
[SetUpFixture]
public class LocalTrackerFixture
{
    private static Process? _trackerProcess;

    /// <summary>Tracker WebSocket URL for tests to use.</summary>
    public const string TrackerUrl = "ws://localhost:5561/announce";

    /// <summary>Tracker HTTP base URL for health checks.</summary>
    public const string TrackerHttpUrl = "http://localhost:5561";

    /// <summary>
    /// Path to the ServerApp project.
    /// </summary>
    public static string ServerAppPath =>
        Path.GetFullPath(Path.Combine(
            AppContext.BaseDirectory, "..", "..", "..", "..", "..", "..",
            "SpawnDev.WebTorrent", "SpawnDev.WebTorrent", "SpawnDev.WebTorrent.ServerApp"));

    /// <summary>Whether the local tracker is running.</summary>
    public static bool IsAvailable { get; private set; }

    [OneTimeSetUp]
    public async Task StartTracker()
    {
        // Check if ServerApp project exists
        if (!Directory.Exists(ServerAppPath))
        {
            Console.WriteLine($"[LocalTracker] ServerApp not found at {ServerAppPath}. " +
                "Tests will use hub.spawndev.com:44365 as fallback.");
            IsAvailable = false;
            return;
        }

        // Check if already running (another test session or manual start)
        if (await IsTrackerRunning())
        {
            Console.WriteLine("[LocalTracker] Local tracker already running on port 5561.");
            IsAvailable = true;
            return;
        }

        Console.WriteLine("[LocalTracker] Starting local tracker...");
        var psi = new ProcessStartInfo
        {
            FileName = "dotnet",
            Arguments = $"run --project \"{ServerAppPath}\"",
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
        };

        _trackerProcess = new Process { StartInfo = psi };
        _trackerProcess.Start();
        _trackerProcess.BeginOutputReadLine();
        _trackerProcess.BeginErrorReadLine();

        // Wait for tracker to be responsive
        var deadline = DateTime.UtcNow.AddSeconds(30);
        while (DateTime.UtcNow < deadline)
        {
            if (await IsTrackerRunning())
            {
                Console.WriteLine("[LocalTracker] Local tracker started successfully.");
                IsAvailable = true;
                return;
            }
            await Task.Delay(500);
        }

        Console.WriteLine("[LocalTracker] Failed to start local tracker within 30s. " +
            "Tests will use hub.spawndev.com:44365 as fallback.");
        IsAvailable = false;
    }

    [OneTimeTearDown]
    public void StopTracker()
    {
        if (_trackerProcess != null && !_trackerProcess.HasExited)
        {
            try
            {
                _trackerProcess.Kill(entireProcessTree: true);
                Console.WriteLine("[LocalTracker] Local tracker stopped.");
            }
            catch { }
        }
        _trackerProcess?.Dispose();
    }

    /// <summary>
    /// Get the tracker URL to use. Prefers local, falls back to hub.spawndev.com.
    /// </summary>
    public static string GetTrackerUrl()
    {
        return IsAvailable ? TrackerUrl : "wss://hub.spawndev.com:44365/announce";
    }

    private static async Task<bool> IsTrackerRunning()
    {
        try
        {
            using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(3) };
            var response = await http.GetAsync($"{TrackerHttpUrl}/stats");
            return response.IsSuccessStatusCode;
        }
        catch
        {
            return false;
        }
    }
}
