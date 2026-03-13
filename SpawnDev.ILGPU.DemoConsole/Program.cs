using SpawnDev.UnitTesting;
using System.Reflection;


var runner = new UnitTestRunner(false);
runner.SetTestAssemblies([Assembly.GetExecutingAssembly()]);
try
{
    await ConsoleRunner.Run(args);
}
catch (Exception ex)
{
    return 1;
}
return 0;
