using Microsoft.Extensions.DependencyInjection;
using SpawnDev.BlazorJS.Cryptography;
using SpawnDev.UnitTesting;

try
{
    var services = new ServiceCollection();
    services.AddPlatformCrypto();
    services.AddSingleton<SpawnDev.WebTorrent.WebTorrentClient>();
    var sp = services.BuildServiceProvider();
    var runner = new UnitTestRunner(sp, true);
    await ConsoleRunner.Run(args, runner);
}
catch (Exception ex)
{
    return 1;
}
return 0;
