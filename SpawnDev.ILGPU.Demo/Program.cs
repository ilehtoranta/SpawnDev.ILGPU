using Microsoft.AspNetCore.Components.Web;
using Microsoft.AspNetCore.Components.WebAssembly.Hosting;
using SpawnDev.Blazor.UnitTesting;
using SpawnDev.BlazorJS;
using SpawnDev.ILGPU.Demo;
using SpawnDev.ILGPU.Demo.UnitTests;

var builder = WebAssemblyHostBuilder.CreateDefault(args);
builder.Services.AddBlazorJSRuntime();
builder.Services.AddSingleton<WebGPUTests>();
builder.Services.AddSingleton<WorkerTests>();
builder.Services.AddSingleton<CPUTests>();
builder.Services.AddSingleton<DefaultTests>();

builder.Services.AddSingleton<UnitTestService>();
builder.Services.AddScoped(sp => new HttpClient { BaseAddress = new Uri(builder.HostEnvironment.BaseAddress) });

builder.RootComponents.Add<App>("#app");
builder.RootComponents.Add<HeadOutlet>("head::after");

await builder.Build().BlazorJSRunAsync();
