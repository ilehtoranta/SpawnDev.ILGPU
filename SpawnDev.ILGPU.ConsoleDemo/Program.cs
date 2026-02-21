using System.Diagnostics;
using System.Text.Json;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using SpawnDev.UnitTesting;
using SpawnDev.ILGPU.Demo.Shared.UnitTests;
using ILGPU.Algorithms;

// ════════════════════════════════════════════════════════════════════
//  Process-Isolated Test Runner
//
//  Each test runs in its own process to prevent CUDA sticky errors
//  from cascading. When a GPU fault occurs, only that test's process
//  is affected — the orchestrator continues with the next test.
//
//  Usage:
//    dotnet run -- [cuda|opencl|all]           Run all tests (orchestrator)
//    dotnet run -- --run-test ClassName.Method backend   Run single test (worker)
// ════════════════════════════════════════════════════════════════════

// Check if we're in worker mode (running a single test)
var runTestIdx = Array.IndexOf(args, "--run-test");
if (runTestIdx >= 0 && runTestIdx + 2 < args.Length)
{
    return await RunSingleTest(args[runTestIdx + 1], args[runTestIdx + 2]);
}

// ── Diagnostic Mode ───────────────────────────────────────────────
if (args.Length > 0 && args[0].Equals("diag", StringComparison.OrdinalIgnoreCase))
{
    Console.WriteLine("=== OpenCL Diagnostic ===");
    Console.WriteLine($"Runtime Platform: {ILGPU.Backends.Backend.RuntimePlatform}");
    Console.WriteLine($"OS Platform: {ILGPU.Backends.Backend.OSPlatform}");
    Console.WriteLine($"RunningOnNativePlatform: {ILGPU.Backends.Backend.RunningOnNativePlatform}");
    Console.WriteLine($"CLAPI.CurrentAPI type: {ILGPU.Runtime.OpenCL.CLAPI.CurrentAPI.GetType().Name}");
    Console.WriteLine($"CLAPI.CurrentAPI.IsSupported: {ILGPU.Runtime.OpenCL.CLAPI.CurrentAPI.IsSupported}");
    Console.WriteLine();
    
    // Try creating context with verbose output
    try
    {
        Console.WriteLine("Creating ILGPU context with AllAccelerators()...");
        using var context = Context.Create(builder => builder.AllAccelerators().EnableAlgorithms());
        
        Console.WriteLine($"Total devices: {context.Devices.Length}");
        foreach (var device in context.Devices)
        {
            Console.WriteLine($"  Device: {device.Name} (Type: {device.AcceleratorType})");
        }
        
        var clDevices = context.GetCLDevices();
        Console.WriteLine($"OpenCL devices: {clDevices.Count}");
        
        if (clDevices.Count > 0)
        {
            var clDev = clDevices[0];
            Console.WriteLine($"  CL Device: {clDev.Name}");
            Console.WriteLine($"  Platform: {clDev.PlatformName}");
            Console.WriteLine($"  Version: {clDev.DeviceVersion}");
            Console.WriteLine($"  CVersion: {clDev.CVersion}");
            Console.WriteLine($"  CLStdVersion: {clDev.CLStdVersion}");
            Console.WriteLine($"  GenericAddressSpace: {clDev.Capabilities.GenericAddressSpace}");
            Console.WriteLine($"  Vendor: {clDev.Vendor}");
            
            using var writer = new System.IO.StringWriter();
            clDev.PrintInformation(writer);
            Console.WriteLine(writer.ToString());
        }
        
        // Also try with a custom predicate that accepts everything
        Console.WriteLine("\nTrying OpenCL with no filters...");
        using var context2 = Context.Create(builder => builder
            .OpenCL(id => true)
            .EnableAlgorithms());
        var clDevices2 = context2.GetCLDevices();
        Console.WriteLine($"OpenCL devices (no filter): {clDevices2.Count}");
        if (clDevices2.Count > 0)
        {
            var clDev = clDevices2[0];
            Console.WriteLine($"  CL Device: {clDev.Name}");
            Console.WriteLine($"  Version: {clDev.DeviceVersion}");
            Console.WriteLine($"  CVersion: {clDev.CVersion}");
            Console.WriteLine($"  CLStdVersion: {clDev.CLStdVersion}");
            Console.WriteLine($"  GenericAddressSpace: {clDev.Capabilities.GenericAddressSpace}");
        }
    }
    catch (Exception ex)
    {
        Console.WriteLine($"Error: {ex.GetType().Name}: {ex.Message}");
        Console.WriteLine(ex.StackTrace);
    }
    return 0;
}

// ── Orchestrator Mode ──────────────────────────────────────────────
return await RunOrchestrator(args);

// ════════════════════════════════════════════════════════════════════
//  Orchestrator: discovers tests, spawns self per-test, aggregates
// ════════════════════════════════════════════════════════════════════
static async Task<int> RunOrchestrator(string[] args)
{
    Console.WriteLine("=== SpawnDev.ILGPU Console Test Runner (Process-Isolated) ===");
    Console.WriteLine();

    var backendArg = args.FirstOrDefault()?.ToLowerInvariant() ?? "all";
    var testTypes = new List<Type>();

    if (backendArg == "cuda" || backendArg == "all")
        testTypes.Add(typeof(CudaTests));
    if (backendArg == "opencl" || backendArg == "all")
        testTypes.Add(typeof(OpenCLTests));
    if (backendArg == "cpu" || backendArg == "all")
        testTypes.Add(typeof(CPUTests));

    if (testTypes.Count == 0)
    {
        Console.WriteLine($"Unknown backend: {backendArg}");
        Console.WriteLine("Usage: dotnet run -- [cuda|opencl|cpu|all]");
        return 1;
    }

    // Discover tests using UnitTestRunner (just for discovery, not execution)
    var runner = new UnitTestRunner();
    runner.SetTestTypes(testTypes);

    Console.WriteLine($"Running tests for: {string.Join(", ", testTypes.Select(t => t.Name))}");
    Console.WriteLine($"Found {runner.Tests.Count} tests (each in own process)");
    Console.WriteLine(new string('─', 80));

    // Find our own executable path
    var exePath = Environment.ProcessPath ?? Process.GetCurrentProcess().MainModule?.FileName;
    if (exePath == null)
    {
        Console.WriteLine("ERROR: Could not determine executable path for process isolation.");
        return 1;
    }

    var isDll = exePath.EndsWith(".dll", StringComparison.OrdinalIgnoreCase);

    int passed = 0, failed = 0, skipped = 0, crashed = 0;
    double totalDuration = 0;
    var failures = new List<(string test, string error)>();
    var sw = new Stopwatch();

    foreach (var test in runner.Tests)
    {
        var testName = $"{test.TestTypeName}.{test.TestMethodName}";
        var backend = test.TestType == typeof(CudaTests) ? "cuda" :
                      test.TestType == typeof(OpenCLTests) ? "opencl" : "cpu";

        sw.Restart();

        // Spawn self with --run-test
        var psi = new ProcessStartInfo
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };

        if (isDll)
        {
            psi.FileName = "dotnet";
            psi.Arguments = $"\"{exePath}\" --run-test {testName} {backend}";
        }
        else
        {
            psi.FileName = exePath;
            psi.Arguments = $"--run-test {testName} {backend}";
        }

        string stdout = "", stderr = "";
        int exitCode = -1;

        try
        {
            using var proc = Process.Start(psi)!;
            stdout = await proc.StandardOutput.ReadToEndAsync();
            stderr = await proc.StandardError.ReadToEndAsync();

            // 30-second timeout per test
            var completed = proc.WaitForExit(30_000);
            if (!completed)
            {
                try { proc.Kill(entireProcessTree: true); } catch { }
                exitCode = -1;
            }
            else
            {
                exitCode = proc.ExitCode;
            }
        }
        catch (Exception ex)
        {
            stderr = ex.Message;
            exitCode = -1;
        }

        sw.Stop();
        var durationMs = Math.Round(sw.Elapsed.TotalMilliseconds);
        totalDuration += durationMs;

        // Parse result from exit code
        // 0 = pass, 1 = fail, 2 = skip/unsupported, -1 = crash/timeout
        switch (exitCode)
        {
            case 0:
                passed++;
                Console.ForegroundColor = ConsoleColor.Green;
                Console.Write("  [✓] ");
                Console.ResetColor();
                Console.WriteLine($"{testName} ({durationMs}ms)");
                break;

            case 2:
                skipped++;
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.Write("  [⏭] ");
                Console.ResetColor();
                var skipReason = ExtractMessage(stdout, "SKIP:");
                Console.WriteLine($"{testName} ({durationMs}ms) - {skipReason}");
                break;

            case 1:
                failed++;
                var failMsg = ExtractMessage(stdout, "FAIL:");
                failures.Add((testName, failMsg));
                Console.ForegroundColor = ConsoleColor.Red;
                Console.Write("  [✗] ");
                Console.ResetColor();
                Console.Write($"{testName} ({durationMs}ms)");
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($" - {failMsg}");
                Console.ResetColor();
                break;

            default:
                crashed++;
                failed++;
                var crashMsg = !string.IsNullOrEmpty(stderr) ? stderr.Split('\n')[0].Trim() : "Process crashed or timed out";
                failures.Add((testName, $"CRASH: {crashMsg}"));
                Console.ForegroundColor = ConsoleColor.Magenta;
                Console.Write("  [💀] ");
                Console.ResetColor();
                Console.Write($"{testName} ({durationMs}ms)");
                Console.ForegroundColor = ConsoleColor.Magenta;
                Console.WriteLine($" - CRASH: {crashMsg}");
                Console.ResetColor();
                break;
        }
    }

    Console.WriteLine(new string('─', 80));

    // Summary
    Console.ForegroundColor = failed == 0 ? ConsoleColor.Green : ConsoleColor.Red;
    Console.WriteLine($"\n  {(failed == 0 ? "ALL PASSED" : "FAILURES DETECTED")}");
    Console.ResetColor();
    Console.WriteLine($"  Total: {runner.Tests.Count} | Passed: {passed} | Failed: {failed} | Skipped: {skipped} | Crashed: {crashed} | Duration: {totalDuration}ms\n");

    if (failures.Count > 0)
    {
        Console.ForegroundColor = ConsoleColor.Red;
        Console.WriteLine("  Failed tests:");
        Console.ResetColor();
        foreach (var (test, error) in failures)
        {
            Console.WriteLine($"    {test}: {error}");
        }
        Console.WriteLine();
    }

    return failed > 0 ? 1 : 0;
}

static string ExtractMessage(string stdout, string prefix)
{
    var line = stdout.Split('\n').FirstOrDefault(l => l.Contains(prefix));
    if (line != null)
    {
        var idx = line.IndexOf(prefix);
        return line.Substring(idx + prefix.Length).Trim();
    }
    return stdout.Trim().Split('\n').LastOrDefault()?.Trim() ?? "Unknown";
}

// ════════════════════════════════════════════════════════════════════
//  Worker: runs a single test and exits
//  Exit codes: 0=pass, 1=fail, 2=skip/unsupported
// ════════════════════════════════════════════════════════════════════
static async Task<int> RunSingleTest(string testName, string backend)
{
    var parts = testName.Split('.', 2);
    if (parts.Length != 2)
    {
        Console.WriteLine($"FAIL: Invalid test name format: {testName}");
        return 1;
    }

    var className = parts[0];
    var methodName = parts[1];

    // Resolve test type
    Type? testType = backend.ToLowerInvariant() switch
    {
        "cuda" => typeof(CudaTests),
        "opencl" => typeof(OpenCLTests),
        "cpu" => typeof(CPUTests),
        _ => null
    };

    if (testType == null)
    {
        Console.WriteLine($"FAIL: Unknown backend: {backend}");
        return 1;
    }

    // Only run if class name matches
    if (!testType.Name.Equals(className, StringComparison.OrdinalIgnoreCase))
    {
        Console.WriteLine($"FAIL: Class {className} not found for backend {backend}");
        return 1;
    }

    var runner = new UnitTestRunner();
    runner.DefaultTimeoutMs = 25_000; // 25s timeout per test
    runner.SetTestTypes(new[] { testType });

    var test = runner.Tests.FirstOrDefault(t =>
        t.TestMethodName.Equals(methodName, StringComparison.OrdinalIgnoreCase));

    if (test == null)
    {
        Console.WriteLine($"FAIL: Test method {methodName} not found in {className}");
        return 1;
    }

    await runner.RunTest(test);

    switch (test.Result)
    {
        case TestResult.Success:
            Console.WriteLine($"PASS: {testName}");
            if (!string.IsNullOrEmpty(test.ResultText) && test.ResultText != "Success")
                Console.WriteLine($"  Result: {test.ResultText}");
            return 0;

        case TestResult.Unsupported:
            Console.WriteLine($"SKIP: {test.ResultText}");
            return 2;

        case TestResult.Error:
            Console.WriteLine($"FAIL: {test.Error}");
            if (!string.IsNullOrEmpty(test.StackTrace))
                Console.Error.WriteLine(test.StackTrace);
            return 1;

        default:
            Console.WriteLine($"FAIL: Unknown result: {test.Result}");
            return 1;
    }
}

// ════════════════════════════════════════════════════════════════════
//  Test Classes
// ════════════════════════════════════════════════════════════════════

public class CudaTests : BackendTestBase
{
    protected override string BackendName => "CUDA";

    protected override Task<(Context context, Accelerator accelerator)> CreateAcceleratorAsync()
    {
        var context = Context.Create(builder => builder.AllAccelerators().EnableAlgorithms());
        var cudaDevices = context.GetCudaDevices();
        if (cudaDevices.Count == 0)
        {
            context.Dispose();
            throw new UnsupportedTestException("No CUDA devices found");
        }
        var accelerator = cudaDevices[0].CreateAccelerator(context);
        return Task.FromResult<(Context, Accelerator)>((context, accelerator));
    }
}

public class OpenCLTests : BackendTestBase
{
    protected override string BackendName => "OpenCL";

    protected override Task<(Context context, Accelerator accelerator)> CreateAcceleratorAsync()
    {
        var context = Context.Create(builder => builder.AllAccelerators().EnableAlgorithms());
        var clDevices = context.GetCLDevices();
        if (clDevices.Count == 0)
        {
            context.Dispose();
            throw new UnsupportedTestException("No OpenCL devices found");
        }
        var accelerator = clDevices[0].CreateAccelerator(context);
        return Task.FromResult<(Context, Accelerator)>((context, accelerator));
    }
}

public class CPUTests : BackendTestBase
{
    protected override string BackendName => "CPU";

    protected override Task<(Context context, Accelerator accelerator)> CreateAcceleratorAsync()
    {
        // Use the Nvidia preset (warp=32, warps=32) to support group sizes up to 1024.
        // The default preset (warp=4, warps=4) only supports groups of 16.
        var context = Context.Create(builder => builder
            .CPU(CPUDevice.Nvidia)
            .EnableAlgorithms());
        var accelerator = context.GetCPUDevice(0).CreateCPUAccelerator(context);
        return Task.FromResult<(Context, Accelerator)>((context, accelerator));
    }
}
