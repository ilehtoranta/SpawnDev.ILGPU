// Lists running testhost / dotnet.exe processes that look like PMT runs.
// Returns exit code 0 if safe, 1 if conflicts. Uses NtQuerySystemInformation
// via P/Invoke to read process command lines without needing wmic / COM /
// PowerShell (none of which are guaranteed available on Win11).
// Usage: dotnet run D:\users\tj\Projects\SpawnDev.ILGPU\SpawnDev.ILGPU\Scripts\check-testhosts.cs

using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

var keywordsAny = new[] { "PlaywrightMultiTest", "vstest.console", "testhost.dll" };
// Order matters - more specific paths first (Riker's fork before generic ILGPU).
var ownerHints = new[]
{
    ("SpawnDev.ILGPU.RikerWasmFork", "Riker"),
    ("SpawnDev.WebTorrent",  "Riker"),
    ("SpawnDev.RTC",         "Riker"),
    ("SpawnDev.MultiMedia",  "Riker"),
    ("SpawnDev.Codecs",      "Tuvok"),
    ("SpawnDev.PatchStreams","Tuvok"),
    ("SpawnDev.EBML",        "Tuvok"),
    ("SpawnDev.VoxelEngine", "Data"),
    ("SpawnDev.GameUI",      "Data"),
    ("Lost Spawns",          "Data"),
    ("SpawnDev.ILGPU",       "Geordi"),
};

var found = new List<(int Pid, string Name, string Owner, string Cmd)>();

foreach (var p in Process.GetProcesses())
{
    string name;
    try { name = p.ProcessName; } catch { continue; }
    if (!name.Equals("dotnet", StringComparison.OrdinalIgnoreCase) &&
        !name.Equals("testhost", StringComparison.OrdinalIgnoreCase))
        continue;

    string? cmd = null;
    try { cmd = ProcessCmdLine.Get(p.Id); } catch { /* access denied */ }
    if (string.IsNullOrEmpty(cmd)) continue;
    if (!keywordsAny.Any(k => cmd.Contains(k, StringComparison.OrdinalIgnoreCase))) continue;

    var owner = "?";
    foreach (var (path, hint) in ownerHints)
        if (cmd.Contains(path, StringComparison.OrdinalIgnoreCase)) { owner = hint; break; }

    var trimmed = cmd.Length > 140 ? cmd[..140] + "..." : cmd;
    found.Add((p.Id, name, owner, trimmed));
}

if (found.Count == 0)
{
    Console.WriteLine("[check-testhosts] No PMT testhosts running. Safe to start your run.");
    return 0;
}

Console.WriteLine($"[check-testhosts] {found.Count} PMT-related process(es) running:");
foreach (var (pid, name, owner, cmd) in found.OrderBy(f => f.Owner).ThenBy(f => f.Pid))
    Console.WriteLine($"  PID {pid,6} {name,-10} owner={owner,-7} cmd={cmd}");
Console.WriteLine();
Console.WriteLine("[check-testhosts] Coordinate before starting another browser-context test run.");
return 1;

static class ProcessCmdLine
{
    const uint PROCESS_QUERY_LIMITED_INFORMATION = 0x1000;
    const int ProcessBasicInformation = 0;

    [StructLayout(LayoutKind.Sequential)]
    struct PROCESS_BASIC_INFORMATION
    {
        public IntPtr Reserved1;
        public IntPtr PebBaseAddress;
        public IntPtr Reserved2_0;
        public IntPtr Reserved2_1;
        public IntPtr UniqueProcessId;
        public IntPtr Reserved3;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct UNICODE_STRING
    {
        public ushort Length;
        public ushort MaximumLength;
        public IntPtr Buffer;
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    static extern IntPtr OpenProcess(uint dwDesiredAccess, bool bInheritHandle, int dwProcessId);

    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool CloseHandle(IntPtr hObject);

    [DllImport("ntdll.dll")]
    static extern int NtQueryInformationProcess(
        IntPtr handle, int infoClass, ref PROCESS_BASIC_INFORMATION info, int size, out int returned);

    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool ReadProcessMemory(
        IntPtr hProcess, IntPtr addr, IntPtr buffer, IntPtr size, out IntPtr bytesRead);

    public static string Get(int pid)
    {
        var h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | 0x0010 /* VM_READ */, false, pid);
        if (h == IntPtr.Zero) return "";
        try
        {
            var pbi = new PROCESS_BASIC_INFORMATION();
            if (NtQueryInformationProcess(h, ProcessBasicInformation, ref pbi,
                Marshal.SizeOf<PROCESS_BASIC_INFORMATION>(), out _) != 0) return "";
            // PEB layout (x64): ProcessParameters at offset 0x20.
            var ppPtr = ReadPtr(h, pbi.PebBaseAddress + 0x20);
            if (ppPtr == IntPtr.Zero) return "";
            // RTL_USER_PROCESS_PARAMETERS layout (x64): CommandLine at offset 0x70 (UNICODE_STRING).
            var cmdLineUS = ReadStruct<UNICODE_STRING>(h, ppPtr + 0x70);
            if (cmdLineUS.Length == 0 || cmdLineUS.Buffer == IntPtr.Zero) return "";
            var buf = new byte[cmdLineUS.Length];
            var bufHandle = GCHandle.Alloc(buf, GCHandleType.Pinned);
            try
            {
                if (!ReadProcessMemory(h, cmdLineUS.Buffer, bufHandle.AddrOfPinnedObject(),
                    new IntPtr(cmdLineUS.Length), out _)) return "";
                return Encoding.Unicode.GetString(buf, 0, cmdLineUS.Length);
            }
            finally { bufHandle.Free(); }
        }
        finally { CloseHandle(h); }
    }

    static IntPtr ReadPtr(IntPtr h, IntPtr addr)
    {
        var b = new byte[IntPtr.Size];
        var g = GCHandle.Alloc(b, GCHandleType.Pinned);
        try
        {
            if (!ReadProcessMemory(h, addr, g.AddrOfPinnedObject(), new IntPtr(IntPtr.Size), out _))
                return IntPtr.Zero;
            return IntPtr.Size == 8 ? new IntPtr(BitConverter.ToInt64(b, 0))
                                    : new IntPtr(BitConverter.ToInt32(b, 0));
        }
        finally { g.Free(); }
    }

    static T ReadStruct<T>(IntPtr h, IntPtr addr) where T : struct
    {
        var size = Marshal.SizeOf<T>();
        var b = new byte[size];
        var g = GCHandle.Alloc(b, GCHandleType.Pinned);
        try
        {
            ReadProcessMemory(h, addr, g.AddrOfPinnedObject(), new IntPtr(size), out _);
            return Marshal.PtrToStructure<T>(g.AddrOfPinnedObject());
        }
        finally { g.Free(); }
    }
}
