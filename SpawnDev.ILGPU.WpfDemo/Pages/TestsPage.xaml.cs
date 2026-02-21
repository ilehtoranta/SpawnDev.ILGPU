using System.ComponentModel;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using SpawnDev.UnitTesting;
using SpawnDev.ILGPU.WpfDemo.UnitTests;

namespace SpawnDev.ILGPU.WpfDemo.Pages
{
    public partial class TestsPage : UserControl
    {
        private UnitTestRunner _runner = new UnitTestRunner();
        private string _backendFilter = "CUDA";
        private List<TestViewModel> _viewModels = new();

        public TestsPage()
        {
            InitializeComponent();
        }

        private void OnLoaded(object sender, RoutedEventArgs e)
        {
            _runner.TestStatusChanged += OnTestStatusChanged;
            RefreshTestList();
        }

        private Type[] GetTestTypes()
        {
            return _backendFilter switch
            {
                "CUDA" => new[] { typeof(CudaTests) },
                "OpenCL" => new[] { typeof(OpenCLTests) },
                _ => new[] { typeof(CudaTests), typeof(OpenCLTests) }
            };
        }

        private void RefreshTestList()
        {
            _runner.SetTestTypes(GetTestTypes());
            RebuildViewModels();
            UpdateSummary();
        }

        private void RebuildViewModels()
        {
            _viewModels = _runner.Tests.Select((t, i) => new TestViewModel(t, i + 1)).ToList();
            TestList.ItemsSource = _viewModels;
        }

        private void OnBackendFilterChanged(object sender, SelectionChangedEventArgs e)
        {
            if (!IsLoaded) return;
            if (BackendFilter.SelectedItem is ComboBoxItem item && item.Tag is string tag)
            {
                _backendFilter = tag;
                RefreshTestList();
            }
        }

        private async void OnRunAll(object sender, RoutedEventArgs e)
        {
            RunAllBtn.IsEnabled = false;
            CancelBtn.IsEnabled = true;
            BackendFilter.IsEnabled = false;
            try
            {
                await _runner.RunTests();
                // Write results to file for debugging
                var logPath = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "test_results.log");
                var lines = new List<string>();
                foreach (var test in _runner.Tests)
                {
                    var status = test.Result == TestResult.Success ? "PASS" : 
                                 test.Result == TestResult.Unsupported ? "SKIP" : "FAIL";
                    lines.Add($"[{status}] {test.TestTypeName}.{test.TestMethodName} ({test.Duration}ms)");
                    if (test.Result == TestResult.Error)
                        lines.Add($"  ERROR: {test.Error}");
                }
                System.IO.File.WriteAllLines(logPath, lines);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Test runner error: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
            RunAllBtn.IsEnabled = true;
            CancelBtn.IsEnabled = false;
            BackendFilter.IsEnabled = true;
        }

        private void OnCancel(object sender, RoutedEventArgs e)
        {
            _runner.CancelTests();
        }

        private void OnTestStatusChanged()
        {
            Dispatcher.Invoke(() =>
            {
                // Update each viewmodel from the underlying UnitTest
                foreach (var vm in _viewModels) vm.Refresh();
                TestList.Items.Refresh();
                UpdateSummary();
            });
        }

        private void UpdateSummary()
        {
            var tests = _runner.Tests;
            var completed = tests.Where(t => t.State == TestState.Done).ToList();
            var passed = completed.Count(t => t.Result == TestResult.Success);
            var failed = completed.Count(t => t.Result == TestResult.Error);
            var skipped = completed.Count(t => t.Result == TestResult.Unsupported);
            var total = tests.Count;
            var totalDuration = completed.Sum(t => t.Duration);

            if (_runner.State == TestState.Running)
            {
                StatusText.Text = "⏳ Running...";
                StatusText.Foreground = new SolidColorBrush(Color.FromRgb(100, 180, 255));
            }
            else if (_runner.State == TestState.Done)
            {
                StatusText.Text = failed == 0 ? "✅ Done" : "❌ Done";
                StatusText.Foreground = failed == 0
                    ? new SolidColorBrush(Color.FromRgb(80, 200, 120))
                    : new SolidColorBrush(Color.FromRgb(220, 80, 80));
            }
            else
            {
                StatusText.Text = "⏸ Ready";
                StatusText.Foreground = new SolidColorBrush(Color.FromRgb(160, 160, 208));
            }

            SummaryText.Text = $"📊 {total} tests  |  ✅ {passed} passed  |  ❌ {failed} failed  |  ⏭ {skipped} skipped  |  ⏱ {totalDuration:F0}ms";
            FooterText.Text = completed.Count > 0
                ? $"Completed {completed.Count}/{total} tests in {totalDuration:F0}ms"
                : $"{total} tests ready to run";

            Progress.Maximum = total > 0 ? total : 1;
            Progress.Value = completed.Count;
        }

        /// <summary>
        /// ViewModel for displaying a UnitTest in the ListView
        /// </summary>
        public class TestViewModel : INotifyPropertyChanged
        {
            private readonly UnitTest _test;

            public int Index { get; }
            public string ClassName => _test.TestTypeName;
            public string MethodName => _test.TestMethodName;

            public string DurationText => _test.State == TestState.Done
                ? $"{_test.Duration:F0} ms"
                : "-";

            public string ResultText
            {
                get
                {
                    return _test.State switch
                    {
                        TestState.None => "-",
                        TestState.Running => "⏳ running...",
                        TestState.Done => _test.Result switch
                        {
                            TestResult.Success => _test.ResultText,
                            TestResult.Unsupported => $"Skipped: {_test.ResultText}",
                            TestResult.Error => _test.Error,
                            _ => _test.Result.ToString()
                        },
                        _ => ""
                    };
                }
            }

            public Brush ResultColor
            {
                get
                {
                    if (_test.State == TestState.Running)
                        return new SolidColorBrush(Color.FromRgb(100, 180, 255));
                    if (_test.State != TestState.Done)
                        return new SolidColorBrush(Color.FromRgb(136, 136, 184));
                    return _test.Result switch
                    {
                        TestResult.Success => new SolidColorBrush(Color.FromRgb(80, 200, 120)),
                        TestResult.Error => new SolidColorBrush(Color.FromRgb(220, 80, 80)),
                        TestResult.Unsupported => new SolidColorBrush(Color.FromRgb(230, 160, 60)),
                        _ => new SolidColorBrush(Color.FromRgb(136, 136, 184))
                    };
                }
            }

            public TestViewModel(UnitTest test, int index)
            {
                _test = test;
                Index = index;
            }

            public void Refresh()
            {
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(DurationText)));
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(ResultText)));
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(ResultColor)));
            }

            public event PropertyChangedEventHandler? PropertyChanged;
        }
    }
}
