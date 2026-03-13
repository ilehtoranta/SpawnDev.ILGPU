using Microsoft.Playwright;
using SpawnDev.UnitTesting;
using System.Diagnostics;
using System.Text.Json;

namespace PlaywrightMultiTest
{
    public class ProjectRunner
    {
        public static ProjectRunner Instance => GetRunner().GetAwaiter().GetResult()!;
        private static Task<ProjectRunner>? _projectRunner;
        public List<TestableProject> TestableProjects { get; } = new List<TestableProject>();

        /// <summary>
        /// Returns an initialized ProjectRunner singleton
        /// </summary>
        /// <returns></returns>
        static Task<ProjectRunner> GetRunner() => _projectRunner ??= new Func<Task<ProjectRunner>>(async () =>
        {
            var ret = new ProjectRunner();
            await ret.Init();
            return ret;
        })();

        /// <summary>
        /// Private consturoctor to prevent external instantiation. The runner should only be created through the GetRunner property which ensures proper initialization.
        /// </summary>
        private ProjectRunner() { }

        private static async Task<int> RunDotnetAsync(string args, string workingDir)
        {
            var startInfo = new ProcessStartInfo("dotnet", args)
            {
                WorkingDirectory = workingDir,
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
            };
            using var p = Process.Start(startInfo);
            if (p == null) return -1;
            await p.WaitForExitAsync();
            return p.ExitCode;
        }
        /// <summary>
        /// Async initialization method for the ProjectRunner. This is where you can perform any setup that needs to happen before tests are enumerated, such as reading configuration files, setting up logging, etc.
        /// </summary>
        /// <returns></returns>
        private async Task Init()
        {
            Debug.WriteLine("Init()");

            string[] args = Environment.GetCommandLineArgs();
            var filter = args.LastOrDefault(o => o.StartsWith("--filter="))?.Substring(9);


            var projects = ProjectDiscovery.GetWorkspaceRoot();
            // add tests to _tests list based on the projects found. You can use the ProjectDetails to determine what kind of project it is and how to get the tests from it. For example, if it's a Blazor WASM project, you might want to start a Playwright instance and navigate to the app to get the tests. If it's a console app, you might want to run the exe with a specific argument to get the tests.
            foreach (var project in projects)
            {
                if (project.AppProjectType == ProjectType.BlazorWasm)
                {
                    var testableProject = new TestableBlazorWasm
                    {
                        ProjectDetails = project,
                    };
                    TestableProjects.Add(testableProject);

                    var buildTest = new ProjectTest(testableProject, $"Build {project.Name}");
                    testableProject.Tests.Add(buildTest);

                    var indexPath = Path.Combine(testableProject.ProjectDetails.WwwRoot, "index.html");

                    // build a publish version of the app for testing
                    var pubResult = await RunDotnetAsync($"publish \"{project.CsprojPath}\" -c Release", project.Directory);
                    if (pubResult != 0 || !File.Exists(indexPath))
                    {
                        // build failed
                        buildTest.SetError();
                        continue;
                    }

                    try
                    {
                        var exitCode = Microsoft.Playwright.Program.Main(new[] { "install" });

                        if (exitCode != 0)
                        {
                            throw new Exception($"Playwright browser installation failed with exit code {exitCode}");
                        }

                        // start a static file server to serve the published output
                        var _port = new Random().Next(5000, 9000);
                        var baseUrl = $"https://localhost:{_port}/";
                        testableProject.Server = new StaticFileServer(testableProject.ProjectDetails.WwwRoot, baseUrl);
                        // start https server to serve the Blazor WASM app
                        testableProject.Server.Start();

                        // create a playwright browser, navigate to the app, and enumerate the tests
                        testableProject.Playwright = await Playwright.CreateAsync();
                        // launch browser
                        testableProject.Browser = await testableProject.Playwright.Chromium.LaunchAsync(new BrowserTypeLaunchOptions
                        {
                            Headless = false,
                            Args = new[]
                            {
                                "--enable-unsafe-webgpu",
                                // Force D3D12 (Native Windows WebGPU backend)
                                "--enable-features=Vulkan,WebGPUService,SkiaGraphite", 
                                // DO NOT use --disable-vulkan-surface on Windows; it kills hardware compositing
                                "--ignore-gpu-blocklist",
                                //"--use-angle=d3d12", // Much more stable on Windows than Vulkan
                                "--no-sandbox"
                            }
                            //SlowMo = 500 // Slows down operations by 500ms so you can follow along
                        });
                        // new browser context
                        testableProject.BrowserContext = await testableProject.Browser.NewContextAsync();
                        // new page
                        testableProject.Page = await testableProject.BrowserContext.NewPageAsync();

                        // go to the app's unit tests page. This assumes that your Blazor WASM app has a page that lists the unit tests and is accessible at the root URL. You would need to implement this page in your Blazor WASM app to return the tests you want to run.
                        var testPageUrl = new Uri(new Uri(baseUrl), testableProject.TestPage).ToString();
                        await testableProject.Page.GotoAsync(testPageUrl);

                        // wait for tests to load
                        await testableProject.Page.WaitForSelectorAsync("table.unit-test-ready", new() { Timeout = 30000 });

                        // get the table
                        var table = testableProject.Page.Locator("table.unit-test-view");

                        // get table body
                        var tbody = table.Locator("tbody");

                        // get all rows in the target table body
                        var rows = tbody.Locator("tr");

                        // iterate the rows
                        int rowCount = await rows.CountAsync();

                        // wait for the tests to load. This assumes that your Blazor WASM app will render an element with the id "test-list" that contains the list of tests. You would need to implement this in your Blazor WASM app to return the tests you want to run.
                        // get a list of tests

                        for (int i = 0; i < rowCount; i++)
                        {
                            // get the specific row by index
                            var currentRow = rows.Nth(i);

                            // get test type name
                            var typeName = await currentRow.Locator(".test-type-name").TextContentAsync();

                            // get test method name
                            var methodName = await currentRow.Locator(".test-method-name").TextContentAsync();

                            var rowTest = new ProjectTest(testableProject, typeName!, methodName!, testPageUrl);

                            if (filter != null)
                            {
                                if (rowTest.Name != filter && rowTest.TestTypeName != filter && rowTest.TestMethodName != filter)
                                {
                                    continue;
                                }
                            }

                            testableProject.Tests.Add(rowTest);
                        }

                    }
                    catch (Exception ex)
                    {
                        var nmtt = true;
                    }
                }
                else if (project.AppProjectType == ProjectType.Exe)
                {
                    // enumerate tests by calling the console app. by default it will return a list of the tests in the exe

                    var testableProject = new TestableConsole
                    {
                        ProjectDetails = project,
                    };
                    TestableProjects.Add(testableProject);

                    var buildTest = new ProjectTest(testableProject, $"Build {project.Name}");
                    testableProject.Tests.Add(buildTest);

                    // build a publish version of the app for testing
                    var pubResult = await RunDotnetAsync($"publish \"{project.CsprojPath}\" -c Release", project.Directory);
                    var publishedBinary = project.ExistingPublishBinary;
                    if (pubResult != 0 || string.IsNullOrEmpty(publishedBinary))
                    {
                        // build failed
                        buildTest.SetError();
                        continue;
                    }

                    // get list of tests by running the exe with a specific argument
                    var result = await ProcessRunner.Run(publishedBinary);
                    var testList = result.Text.Split(new[] { '\n', '\r' }, StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
                    foreach (var test in testList)
                    {
                        // get test type name
                        var typeName = test.Split(".")[0];

                        // get test method name
                        var methodName = test.Split(".")[1];

                        var rowTest = new ProjectTest(testableProject, typeName!, methodName!);
                        if (filter != null)
                        {
                            if (rowTest.Name != filter && rowTest.TestTypeName != filter && rowTest.TestMethodName != filter)
                            {
                                continue;
                            }
                        }
                        testableProject.Tests.Add(rowTest);

                        rowTest.TestFunc = async (page) =>
                        {
                            var result = await ProcessRunner.Run(publishedBinary, rowTest.Name);
                            var resultLines = result.Text.Split(new[] { '\n', '\r' }, StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
                            var testResltTest = resultLines.LastOrDefault(o => o.StartsWith("TEST: "))?.Substring(6);
                            var unitTest = testResltTest != null ? JsonSerializer.Deserialize<UnitTest>(testResltTest) : null;
                            if (unitTest == null)
                            {
                                throw new Exception("Test run failed");
                            }
                            var stateMessage = unitTest.ResultText;
                            rowTest.Result = unitTest.Result;

                            if (rowTest.Result == TestResult.Unsupported)
                            {
                                if (string.IsNullOrWhiteSpace(stateMessage))
                                {
                                    stateMessage = "Skipped";
                                }
                            }
                            else if (rowTest.Result == TestResult.Error)
                            {
                                if (string.IsNullOrWhiteSpace(stateMessage))
                                {
                                    stateMessage = "Failed";
                                }
                                rowTest.ResultMessage = stateMessage;
                                throw new Exception(stateMessage);
                            }
                            else
                            {
                                if (string.IsNullOrWhiteSpace(stateMessage))
                                {
                                    stateMessage = "Success";
                                }
                            }

                            rowTest.ResultMessage = stateMessage;
                            rowTest.Result = unitTest.Result;
                            var nmtt = true;
                        };
                    }

                    var nmt11 = true;
                }
            }
            var nmt = true;
        }
        IEnumerable<TestCaseData>? _TestCases;
        public IEnumerable<TestCaseData> TestCases => _TestCases ??= GetPlaywrightTasks();

        /// <summary>
        /// Returns all the tests that are found. This is called before StartUp, so you should not rely on any services or infrastructure being available when this is called. You can return any tests you want to run here, and they will be run by the test runner.
        /// </summary>
        /// <returns></returns>
        public IEnumerable<TestCaseData> GetPlaywrightTasks()
        {
            Debug.WriteLine("GetPlaywrightTasks()");
            foreach (var testableProject in TestableProjects)
            {
                foreach (var test in testableProject.Tests)
                {
                    var testCaseData = new TestCaseData(test).SetName(test.Name).SetCategory(test.TestTypeName ?? test.Name);
                    yield return testCaseData;
                }
            }
            var nmt = true;
        }

        /// <summary>
        /// This is called after tests have been enumerated bu before they are run. You can use this to start up any services or infrastructure needed for the tests.
        /// </summary>
        /// <returns></returns>
        public async Task StartUp()
        {
            Debug.WriteLine("StartUp()");
        }

        /// <summary>
        /// This is called after tests have ran. You can use this to stop any services or infrastructure started in StartUp.
        /// </summary>
        /// <returns></returns>
        public async Task Shutdown()
        {
            Debug.WriteLine("Shutdown()");
            foreach (var testableProject in TestableProjects)
            {
                if (testableProject is TestableBlazorWasm blazorProj)
                {
                    if (blazorProj.Page != null) await blazorProj.Page.CloseAsync();
                    if (blazorProj.BrowserContext != null) await blazorProj.BrowserContext.CloseAsync();
                    if (blazorProj.Browser != null) await blazorProj.Browser.CloseAsync();
                    if (blazorProj.Server != null) await blazorProj.Server.Stop();
                }
                else if (testableProject is TestableConsole consoleProj)
                {
                    // do any cleanup needed for console projects

                }
            }
        }
    }
}