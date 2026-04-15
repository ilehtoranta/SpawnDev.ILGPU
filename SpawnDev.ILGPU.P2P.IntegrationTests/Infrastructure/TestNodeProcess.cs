using System.Collections.Concurrent;
using System.Diagnostics;

namespace SpawnDev.ILGPU.P2P.IntegrationTests.Infrastructure;

/// <summary>
/// Manages a P2P.TestNode subprocess. Starts the process, parses structured
/// stdout events, and provides async methods to wait for specific events.
///
/// TestNode output protocol:
///   MAGNET:&lt;link&gt;
///   READY
///   PEER_JOINED:&lt;peerId&gt;
///   PEER_LEFT:&lt;peerId&gt;
///   DISPATCH_SENT:&lt;dispatchId&gt;
///   DISPATCH_COMPLETE:&lt;dispatchId&gt;:&lt;success&gt;:&lt;durationMs&gt;
///   RESULT:&lt;bufferId&gt;:&lt;base64data&gt;
///   ERROR:&lt;message&gt;
/// </summary>
public class TestNodeProcess : IAsyncDisposable
{
    private Process? _process;
    private readonly List<string> _allOutput = new();
    private readonly ConcurrentDictionary<string, TaskCompletionSource<string>> _eventWaiters = new();
    private readonly ConcurrentBag<string> _events = new();
    private readonly ConcurrentDictionary<string, string> _results = new();

    /// <summary>All stdout lines captured.</summary>
    public IReadOnlyList<string> AllOutput => _allOutput;

    /// <summary>Structured events received.</summary>
    public IReadOnlyCollection<string> Events => _events;

    /// <summary>The magnet link emitted by the coordinator.</summary>
    public string? MagnetLink { get; private set; }

    /// <summary>Whether the process has exited.</summary>
    public bool HasExited => _process?.HasExited ?? true;

    /// <summary>The process exit code (null if still running).</summary>
    public int? ExitCode => _process?.HasExited == true ? _process.ExitCode : null;

    /// <summary>
    /// Path to the TestNode project directory.
    /// </summary>
    public static string ProjectPath =>
        Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "SpawnDev.ILGPU.P2P.TestNode"));

    /// <summary>
    /// Start a TestNode process with the given arguments.
    /// </summary>
    public async Task StartAsync(string args, int startupTimeoutMs = 30000)
    {
        var psi = new ProcessStartInfo
        {
            FileName = "dotnet",
            Arguments = $"run --project \"{ProjectPath}\" -- {args}",
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
        };

        _process = new Process { StartInfo = psi };
        _process.OutputDataReceived += OnOutputReceived;
        _process.ErrorDataReceived += OnErrorReceived;
        _process.Start();
        _process.BeginOutputReadLine();
        _process.BeginErrorReadLine();

        // Wait for READY signal or timeout
        using var cts = new CancellationTokenSource(startupTimeoutMs);
        try
        {
            await WaitForEventAsync("READY", cts.Token);
        }
        catch (OperationCanceledException)
        {
            throw new TimeoutException(
                $"TestNode did not emit READY within {startupTimeoutMs}ms. Output:\n{string.Join("\n", _allOutput)}");
        }
    }

    /// <summary>
    /// Wait for a specific event tag (e.g., "PEER_JOINED", "DISPATCH_COMPLETE").
    /// Returns the full event line.
    /// </summary>
    public async Task<string> WaitForEventAsync(string eventTag, CancellationToken ct = default)
    {
        // Check if already received
        foreach (var e in _events)
            if (e.StartsWith(eventTag)) return e;

        var tcs = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
        _eventWaiters[eventTag] = tcs;

        using var reg = ct.Register(() => tcs.TrySetCanceled());
        return await tcs.Task;
    }

    /// <summary>
    /// Wait for the magnet link to be emitted.
    /// </summary>
    public async Task<string> WaitForMagnetLinkAsync(int timeoutMs = 30000)
    {
        if (MagnetLink != null) return MagnetLink;

        using var cts = new CancellationTokenSource(timeoutMs);
        var line = await WaitForEventAsync("MAGNET:", cts.Token);
        return line["MAGNET:".Length..];
    }

    /// <summary>
    /// Wait for a dispatch to complete and return (success, durationMs).
    /// </summary>
    public async Task<(bool success, double durationMs)> WaitForDispatchCompleteAsync(
        string? dispatchId = null, int timeoutMs = 15000)
    {
        using var cts = new CancellationTokenSource(timeoutMs);
        var line = await WaitForEventAsync("DISPATCH_COMPLETE:", cts.Token);
        // DISPATCH_COMPLETE:<dispatchId>:<success>:<durationMs>
        var parts = line["DISPATCH_COMPLETE:".Length..].Split(':');
        if (parts.Length >= 3)
        {
            bool success = bool.Parse(parts[1]);
            double duration = double.Parse(parts[2]);
            return (success, duration);
        }
        return (false, 0);
    }

    /// <summary>
    /// Get a result buffer that was output by the worker.
    /// </summary>
    public byte[]? GetResultBuffer(string bufferId)
    {
        return _results.TryGetValue(bufferId, out var base64)
            ? Convert.FromBase64String(base64)
            : null;
    }

    /// <summary>
    /// Kill the process.
    /// </summary>
    public void Kill()
    {
        try
        {
            if (_process != null && !_process.HasExited)
                _process.Kill(entireProcessTree: true);
        }
        catch { }
    }

    private void OnOutputReceived(object sender, DataReceivedEventArgs e)
    {
        if (e.Data == null) return;
        _allOutput.Add(e.Data);

        // Parse structured events
        if (e.Data.StartsWith("MAGNET:"))
        {
            MagnetLink = e.Data["MAGNET:".Length..];
            NotifyWaiters("MAGNET:", e.Data);
        }
        else if (e.Data.StartsWith("READY"))
            NotifyWaiters("READY", e.Data);
        else if (e.Data.StartsWith("PEER_JOINED:"))
            NotifyWaiters("PEER_JOINED", e.Data);
        else if (e.Data.StartsWith("PEER_LEFT:"))
            NotifyWaiters("PEER_LEFT", e.Data);
        else if (e.Data.StartsWith("DISPATCH_SENT:"))
            NotifyWaiters("DISPATCH_SENT", e.Data);
        else if (e.Data.StartsWith("DISPATCH_COMPLETE:"))
            NotifyWaiters("DISPATCH_COMPLETE:", e.Data);
        else if (e.Data.StartsWith("RESULT:"))
        {
            // RESULT:<bufferId>:<base64data>
            var payload = e.Data["RESULT:".Length..];
            var sep = payload.IndexOf(':');
            if (sep > 0)
            {
                var bufferId = payload[..sep];
                var base64 = payload[(sep + 1)..];
                _results[bufferId] = base64;
            }
            NotifyWaiters("RESULT:", e.Data);
        }
        else if (e.Data.StartsWith("ERROR:"))
            NotifyWaiters("ERROR:", e.Data);

        _events.Add(e.Data);
    }

    private void OnErrorReceived(object sender, DataReceivedEventArgs e)
    {
        if (e.Data != null)
            _allOutput.Add($"[stderr] {e.Data}");
    }

    private void NotifyWaiters(string eventTag, string fullLine)
    {
        if (_eventWaiters.TryRemove(eventTag, out var tcs))
            tcs.TrySetResult(fullLine);
    }

    public ValueTask DisposeAsync()
    {
        Kill();
        _process?.Dispose();
        return ValueTask.CompletedTask;
    }
}
