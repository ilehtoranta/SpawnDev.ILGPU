using System.IO;
using System.Windows;
using System.Windows.Threading;

namespace SpawnDev.ILGPU.WpfDemo
{
    public partial class App : Application
    {
        protected override void OnStartup(StartupEventArgs e)
        {
            base.OnStartup(e);
            DispatcherUnhandledException += OnDispatcherUnhandledException;
            AppDomain.CurrentDomain.UnhandledException += OnDomainUnhandledException;
            TaskScheduler.UnobservedTaskException += OnUnobservedTaskException;
        }

        private void OnDispatcherUnhandledException(object sender, DispatcherUnhandledExceptionEventArgs e)
        {
            LogCrash("DispatcherUnhandled", e.Exception);
            e.Handled = true; // Prevent app from closing
            MessageBox.Show($"Caught UI exception:\n{e.Exception.GetType().Name}: {e.Exception.Message}\n\n{e.Exception.StackTrace}",
                "Debug: Dispatcher Exception", MessageBoxButton.OK, MessageBoxImage.Warning);
        }

        private void OnDomainUnhandledException(object sender, UnhandledExceptionEventArgs e)
        {
            LogCrash("DomainUnhandled", e.ExceptionObject as Exception);
        }

        private void OnUnobservedTaskException(object? sender, UnobservedTaskExceptionEventArgs e)
        {
            LogCrash("UnobservedTask", e.Exception);
            e.SetObserved();
        }

        private void LogCrash(string source, Exception? ex)
        {
            try
            {
                var logPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "crash.log");
                var msg = $"[{DateTime.Now:HH:mm:ss.fff}] [{source}] {ex?.GetType().Name}: {ex?.Message}\n{ex?.StackTrace}\n---\n";
                File.AppendAllText(logPath, msg);
            }
            catch { }
        }
    }
}
