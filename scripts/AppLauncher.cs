using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

// Workspace launcher: CUDA, model weights and data stay in the adjacent project.
internal static class AppLauncher
{
    [STAThread]
    private static int Main()
    {
        string root = AppDomain.CurrentDomain.BaseDirectory;
        string python = Path.Combine(root, ".venv", "Scripts", "pythonw.exe");
        try
        {
            if (!File.Exists(python))
                throw new FileNotFoundException("The project Python environment is missing. Run scripts\\setup.ps1 first.");
            var start = new ProcessStartInfo(python, "-m adp desktop") {
                WorkingDirectory = root,
                UseShellExecute = false,
                CreateNoWindow = true
            };
            // Resolve this checkout even when launched through a shortcut with another working directory.
            start.EnvironmentVariables["PYTHONPATH"] = Path.Combine(root, "src");
            using (Process process = Process.Start(start))
            {
                process.WaitForExit();
                if (process.ExitCode != 0)
                    throw new Exception("Interaction Lab could not start. Check artifacts\\desktop\\app.log, or rerun scripts\\setup.ps1.");
            }
            return 0;
        }
        catch (Exception error)
        {
            MessageBox.Show(error.Message, "Interaction Lab — startup error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }
    }
}
