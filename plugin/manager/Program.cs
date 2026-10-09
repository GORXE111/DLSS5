using System;
using System.Windows.Forms;

[assembly: System.Reflection.AssemblyTitle("DLSS5 Manager")]
[assembly: System.Reflection.AssemblyProduct("DLSS5 Manager")]
[assembly: System.Reflection.AssemblyVersion("0.1.0.0")]

namespace Dlss5Manager
{
    static class Program
    {
        [STAThread]
        static int Main(string[] args)
        {
#if CLI
            return Cli.Run(args);
#else
            if (args.Length > 0) { MessageBox.Show("命令行请用 dlss5.exe"); return 64; }
            Display.Primary();   // 先设 DPI 感知
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new MainForm());
            return 0;
#endif
        }
    }
}
