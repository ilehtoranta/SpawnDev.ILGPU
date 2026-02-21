using System.Diagnostics;
using System.Windows;
using System.Windows.Controls;
using SpawnDev.ILGPU.WpfDemo.Pages;

namespace SpawnDev.ILGPU.WpfDemo
{
    public partial class MainWindow : Window
    {
        private HomePage? _homePage;
        private FractalExplorerPage? _fractalPage;
        private RaymarchingPage? _raymarchPage;
        private BoidsPage? _boidsPage;

        public MainWindow()
        {
            InitializeComponent();
            // Show home page on startup
            ShowPage("Home");
        }

        private void OnNavChanged(object sender, RoutedEventArgs e)
        {
            if (sender is RadioButton rb && rb.Tag is string tag)
            {
                ShowPage(tag);
            }
        }

        private void ShowPage(string page)
        {
            switch (page)
            {
                case "Home":
                    _homePage ??= new HomePage();
                    ContentArea.Content = _homePage;
                    break;
                case "Fractal":
                    _fractalPage ??= new FractalExplorerPage();
                    ContentArea.Content = _fractalPage;
                    break;
                case "Raymarch":
                    _raymarchPage ??= new RaymarchingPage();
                    ContentArea.Content = _raymarchPage;
                    break;
                case "Boids":
                    _boidsPage ??= new BoidsPage();
                    ContentArea.Content = _boidsPage;
                    break;
            }
        }

        private void OnGitHubClick(object sender, RoutedEventArgs e)
        {
            Process.Start(new ProcessStartInfo("https://github.com/LostBeard/SpawnDev.ILGPU") { UseShellExecute = true });
        }

        private void OnIlgpuNetClick(object sender, RoutedEventArgs e)
        {
            Process.Start(new ProcessStartInfo("https://ilgpu.net/") { UseShellExecute = true });
        }

        protected override void OnClosed(EventArgs e)
        {
            (_fractalPage as IDisposable)?.Dispose();
            (_raymarchPage as IDisposable)?.Dispose();
            (_boidsPage as IDisposable)?.Dispose();
            base.OnClosed(e);
        }
    }
}
