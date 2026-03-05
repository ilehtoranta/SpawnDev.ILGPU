using Microsoft.AspNetCore.Components.Web;
using Microsoft.AspNetCore.Components.WebAssembly.Hosting;

using SpawnDev.BlazorJS;
using SpawnDev.ILGPU.Demo;
using SpawnDev.ILGPU.Demo.UnitTests;
using SpawnDev.ILGPU.WebGPU.Backend;

var builder = WebAssemblyHostBuilder.CreateDefault(args);

// Ensure WebGPU verbose logging is disabled during Blazor unit tests
WebGPUBackend.VerboseLogging = false;
builder.Services.AddBlazorJSRuntime();
builder.Services.AddSingleton<WebGPUTests>();
builder.Services.AddSingleton<WebGPUNoSubgroupsTests>();

builder.Services.AddSingleton<CPUTests>();
builder.Services.AddSingleton<WasmTests>();
builder.Services.AddSingleton<WebGLTests>();
builder.Services.AddSingleton<DefaultTests>();


builder.Services.AddScoped(sp => new HttpClient { BaseAddress = new Uri(builder.HostEnvironment.BaseAddress) });

builder.RootComponents.Add<App>("#app");
builder.RootComponents.Add<HeadOutlet>("head::after");

await builder.Build().BlazorJSRunAsync();
