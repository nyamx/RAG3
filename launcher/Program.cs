using System.Diagnostics;
using System.Runtime.InteropServices;

internal static class Program
{
    private const string WindowTitle = "RAG3 Haystack";
    private const string LaunchMutex = @"Local\RAG3_UI_LAUNCH";
    private const uint MbIconError = 0x10;
    private const uint MbIconWarning = 0x30;

    [DllImport("user32.dll", CharSet = CharSet.Unicode, ExactSpelling = true)]
    private static extern nint FindWindowW(string? className, string windowName);

    [DllImport("user32.dll")]
    private static extern bool ShowWindow(nint window, int command);

    [DllImport("user32.dll")]
    private static extern bool SetForegroundWindow(nint window);

    [DllImport("user32.dll", CharSet = CharSet.Unicode, ExactSpelling = true)]
    private static extern int MessageBoxW(nint owner, string text, string title, uint type);

    [STAThread]
    private static int Main()
    {
        using var launchMutex = new Mutex(true, LaunchMutex, out bool ownsLaunchMutex);
        if (!ownsLaunchMutex)
        {
            FocusExistingWindow();
            return 0;
        }

        try
        {
            if (FocusExistingWindow())
                return 0;

            string root = AppContext.BaseDirectory;
            string python = Path.Combine(root, ".venv", "Scripts", "pythonw.exe");
            string ui = Path.Combine(root, "scripts", "ingest-ui.py");
            if (!File.Exists(python) || !File.Exists(ui))
            {
                ShowError("RAG3.exe를 RAG3 프로젝트 폴더에 두고 .venv와 scripts 폴더가 있는지 확인하세요.\n\n"
                          + $"Python: {python}\nUI: {ui}");
                return 1;
            }

            var start = new ProcessStartInfo(python)
            {
                WorkingDirectory = root,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            start.ArgumentList.Add(ui);
            using Process? process = Process.Start(start);
            if (process is null)
            {
                ShowError("RAG3 UI 프로세스를 시작하지 못했습니다.");
                return 1;
            }

            // Keep the mutex through startup so repeated double-clicks do not
            // open multiple UI windows while the Python imports are loading.
            DateTime deadline = DateTime.UtcNow.AddMinutes(3);
            while (DateTime.UtcNow < deadline)
            {
                if (WindowExists())
                    return 0;
                if (process.HasExited)
                {
                    MessageBoxW(0,
                        $"RAG3 UI가 창을 열기 전에 종료되었습니다 (종료 코드 {process.ExitCode}).\n"
                        + "Qdrant 실행 상태와 Python 가상환경을 확인하세요.",
                        "RAG3 시작 실패", MbIconError);
                    return 1;
                }
                Thread.Sleep(300);
            }

            MessageBoxW(0,
                "RAG3 UI 시작이 지연되고 있습니다. 잠시 후 창이 열리는지 확인하세요.",
                "RAG3 시작 지연", MbIconWarning);
            return 0;
        }
        catch (Exception ex)
        {
            ShowError(ex.Message);
            return 1;
        }
        finally
        {
            launchMutex.ReleaseMutex();
        }
    }

    private static bool WindowExists() => FindWindowW(null, WindowTitle) != 0;

    private static bool FocusExistingWindow()
    {
        nint window = FindWindowW(null, WindowTitle);
        if (window == 0)
            return false;
        ShowWindow(window, 9); // SW_RESTORE
        SetForegroundWindow(window);
        return true;
    }

    private static void ShowError(string message) =>
        MessageBoxW(0, message, "RAG3 실행 오류", MbIconError);
}
