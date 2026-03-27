using Microsoft.AspNetCore.Components.Web;
using Microsoft.AspNetCore.Components.WebAssembly.Hosting;

using SpawnDev.BlazorJS;
using SpawnDev.ILGPU.Demo;
using SpawnDev.ILGPU.Demo.UnitTests;
using SpawnDev.ILGPU.WebGPU.Backend;

// Print build timestamp so we can verify we're running the right build via browser console
Console.WriteLine($"[SpawnDev.ILGPU.Demo] Build: {BuildTimestamp.Value}");

var builder = WebAssemblyHostBuilder.CreateDefault(args);

// Ensure WebGPU verbose logging is disabled during Blazor unit tests
WebGPUBackend.VerboseLogging = false;
builder.Services.AddBlazorJSRuntime();
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
