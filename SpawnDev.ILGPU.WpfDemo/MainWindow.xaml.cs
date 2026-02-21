using System.Diagnostics;
using System.Windows;
using System.Windows.Controls;
using SpawnDev.ILGPU.WpfDemo.Pages;

namespace SpawnDev.ILGPU.WpfDemo
{
    public partial class MainWindow : Window
    {
        private HomePage? _homePage;

        public MainWindow()
        {
            InitializeComponent();
            ShowPage("Home");
        }

        private void OnNavChanged(object sender, RoutedEventArgs e)
        {
            if (sender is RadioButton rb && rb.Tag is string tag)
            {
                ShowPage(tag);
            }
        }

        /// <summary>
        /// Disposes the current GPU page (if any) before showing the next.
        /// GPU pages own render threads and GPU contexts — running multiple
        /// simultaneously causes resource conflicts and crashes.
        /// </summary>
        private void ShowPage(string page)
        {
            // Dispose current GPU page before switching
            if (ContentArea.Content is IDisposable disposable && ContentArea.Content is not HomePage)
            {
                ContentArea.Content = null;
                disposable.Dispose();
            }

            switch (page)
            {
                case "Home":
                    _homePage ??= new HomePage();
                    ContentArea.Content = _homePage;
                    break;
                case "Fractal":
                    ContentArea.Content = new FractalExplorerPage();
                    break;
                case "Raymarch":
                    ContentArea.Content = new RaymarchingPage();
                    break;
                case "Boids":
                    ContentArea.Content = new BoidsPage();
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
            if (ContentArea.Content is IDisposable disposable)
                disposable.Dispose();
            base.OnClosed(e);
        }
    }
}
