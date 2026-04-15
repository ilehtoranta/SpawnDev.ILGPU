using Microsoft.AspNetCore.Components.Web;
using Microsoft.AspNetCore.Components.WebAssembly.Hosting;

using SpawnDev.BlazorJS;
using SpawnDev.BlazorJS.Cryptography;
using SpawnDev.ILGPU.Demo;
using SpawnDev.ILGPU.Demo.UnitTests;
using SpawnDev.ILGPU.WebGPU.Backend;

// Print build timestamp so we can verify we're running the right build via browser console
Console.WriteLine($"[SpawnDev.ILGPU.Demo] Build: {BuildTimestamp.Value}");

var builder = WebAssemblyHostBuilder.CreateDefault(args);

// Ensure WebGPU verbose logging is disabled during Blazor unit tests
WebGPUBackend.VerboseLogging = false;
builder.Services.AddBlazorJSRuntime();
builder.Services.AddPlatformCrypto();

// P2P: WebTorrent client
builder.Services.AddSingleton(sp =>
    new SpawnDev.WebTorrent.WebTorrentClient());

// P2P: Ed25519 signer - transient (each P2P test gets a fresh instance)
builder.Services.AddTransient<SpawnDev.WebTorrent.Ed25519Signer>();
builder.Services.AddSingleton<Func<SpawnDev.WebTorrent.Ed25519Signer>>(sp =>
    () => sp.GetRequiredService<SpawnDev.WebTorrent.Ed25519Signer>());

// P2P: Shared swarm service — holds the active compute swarm for all pages
builder.Services.AddSingleton<SpawnDev.ILGPU.Demo.Shared.Services.P2PSwarmService>();

builder.Services.AddSingleton<WebGPUTests>();
builder.Services.AddSingleton<WebGPUNoSubgroupsTests>();

builder.Services.AddSingleton<WasmTests>();
builder.Services.AddSingleton<WebGLTests>();
builder.Services.AddSingleton<DefaultTests>();


builder.Services.AddScoped(sp => new HttpClient { BaseAddress = new Uri(builder.HostEnvironment.BaseAddress) });
builder.Services.AddSingleton<SpawnDev.ILGPU.Services.ShaderDebugService>();

builder.RootComponents.Add<App>("#app");
builder.RootComponents.Add<HeadOutlet>("head::after");

await builder.Build().BlazorJSRunAsync();
