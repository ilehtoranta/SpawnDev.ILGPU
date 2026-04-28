using System.Diagnostics;
using System.Net.Http;

namespace SpawnDev.ILGPU.DemoConsole.P2PTests;

/// <summary>
/// Starts a local SpawnDev.WebTorrent.ServerApp for P2P tracker signaling so tests
/// don't depend on hub.spawndev.com. The ServerApp runs on
/// ws://localhost:5561/announce (HTTP) and wss://localhost:5560/announce (HTTPS).
///
/// Lifecycle: call <see cref="InitAsync"/> once before running any tests that need
/// the tracker (in DemoConsole's <c>Program.cs</c>, before <c>ConsoleRunner.Run</c>).
/// The tracker child process is killed via <c>AppDomain.CurrentDomain.ProcessExit</c>
/// when the test runner exits - so a test failure or PMT subprocess kill does not
/// leave an orphan tracker holding the parent shell's stdout pipe (which previously
/// caused diagnostic bash pipelines to hang for minutes per run; observed 2026-04-27).
/// Static <see cref="IsAvailable"/> and <see cref="GetTrackerUrl"/> expose the run
/// state to test methods.
/// </summary>
public static class LocalTrackerFixture
{
    private static Process? _trackerProcess;
    private static int _exitHandlerInstalled;

    /// <summary>Tracker WebSocket URL for tests to use.</summary>
    public const string TrackerUrl = "ws://localhost:5561/announce";

    /// <summary>Tracker HTTP base URL for health checks.</summary>
    public const string TrackerHttpUrl = "http://localhost:5561";

    /// <summary>
    /// Path to the ServerApp project. Walks up from the DemoConsole publish
    /// directory to find the sibling SpawnDev.WebTorrent repo.
    /// </summary>
    public static string ServerAppPath
    {
        get
        {
            // Walk up from AppContext.BaseDirectory looking for a directory that
            // contains both SpawnDev.ILGPU/ and SpawnDev.WebTorrent/. This is
            // resilient to whether we're in `bin/Release/net10.0/` (build output)
            // or `bin/Release/net10.0/publish/` (PlaywrightMultiTest publish output).
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir != null)
            {
                var wtDir = Path.Combine(dir.FullName, "SpawnDev.WebTorrent");
                if (Directory.Exists(wtDir))
                {
                    var appPath = Path.Combine(wtDir, "SpawnDev.WebTorrent", "SpawnDev.WebTorrent.ServerApp");
                    if (Directory.Exists(appPath)) return appPath;
                }
                dir = dir.Parent;
            }
            return "";
        }
    }

    /// <summary>Whether the local tracker is running (or confirmed already running elsewhere).</summary>
    public static bool IsAvailable { get; private set; }

    /// <summary>
    /// Start the local tracker. Safe to call multiple times; only the first call
    /// launches the child process. If another tracker is already listening on port
    /// 5561 (another test session or manual start), reuses it without spawning a
    /// duplicate.
    /// </summary>
    public static async Task InitAsync()
    {
        var appPath = ServerAppPath;
        if (string.IsNullOrEmpty(appPath) || !Directory.Exists(appPath))
        {
            Console.WriteLine($"[LocalTracker] ServerApp not found via directory walk from {AppContext.BaseDirectory}. " +
                "Tests will use hub.spawndev.com:44365 as fallback.");
            IsAvailable = false;
            return;
        }

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
            Arguments = $"run --project \"{appPath}\"",
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
        };

        _trackerProcess = new Process { StartInfo = psi };
        _trackerProcess.Start();
        _trackerProcess.BeginOutputReadLine();
        _trackerProcess.BeginErrorReadLine();

        // Kill the tracker subprocess when the test runner exits. Without this,
        // PMT (or any parent shell that pipes our stdout) hangs after the test
        // completes because the tracker subprocess inherits a handle on the
        // pipe and keeps the pipeline open. Using Interlocked.Exchange so that
        // a second InitAsync (different test session) does not double-register.
        if (System.Threading.Interlocked.Exchange(ref _exitHandlerInstalled, 1) == 0)
        {
            AppDomain.CurrentDomain.ProcessExit += (_, _) => StopTracker();
            // Ctrl-C / Ctrl-Break also need to clean up the tracker.
            Console.CancelKeyPress += (_, _) => StopTracker();
        }

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

    /// <summary>
    /// Kill the tracker child process and any of its descendants. Also closes the
    /// inherited stdout/stderr pipe handles so a parent shell pipeline can terminate.
    /// Safe to call multiple times (idempotent) and from process-exit handlers.
    /// </summary>
    private static void StopTracker()
    {
        var proc = System.Threading.Interlocked.Exchange(ref _trackerProcess, null);
        if (proc == null) return;
        try
        {
            if (!proc.HasExited)
                proc.Kill(entireProcessTree: true);
        }
        catch { /* best-effort cleanup; never let exit-handler exceptions bubble */ }
        try { proc.Dispose(); }
        catch { }
    }
}
