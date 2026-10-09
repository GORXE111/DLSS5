// 命令行: dlss5.exe <命令> ...   (不带参数运行 DLSS5Manager.exe 则打开图形界面)
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace Dlss5Manager
{
    public static class Cli
    {
        const string Usage = @"DLSS5 管理工具 (命令行)

  dlss5 scan                         列出 Steam / Epic / GOG / 手动添加的游戏
  dlss5 info <游戏>                  检测: 主程序、引擎、图形 API、超分、反作弊、占用的注入名、当前参数
  dlss5 install <游戏> [选项]        安装 (覆盖的文件先备份)
  dlss5 config <游戏> [选项]         修改已安装游戏的参数 (不带选项则显示当前参数)
  dlss5 uninstall <游戏>             卸载并还原被覆盖的文件
  dlss5 add <文件夹>                 把不在游戏库里的游戏加入列表
  dlss5 gpu                          显卡与各预设的模型分辨率
  dlss5 params                       可调参数说明

  <游戏> 可以是文件夹路径，也可以是 scan 列出的游戏名 (部分匹配)。
选项:
  --preset 性能|均衡|画质            按显卡和屏幕分辨率选模型分辨率
  --set 名字=值                      设置任意参数，可重复 (例: --set LocalStructure=0.8)
  --proxy dxgi.dll|winmm.dll|...     指定注入文件名 (默认自动选一个没被占用的)
  --res 2560x1440                    输出分辨率 (预设估算用，默认取主显示器)
  --force                            跳过反作弊/显卡检查 —— 只在确认游戏离线运行、没有反作弊时使用";

        public static int Run(string[] args)
        {
            try { Console.OutputEncoding = new UTF8Encoding(false); } catch { }
            if (args.Length == 0 || args[0] == "-h" || args[0] == "--help" || args[0] == "help") { Console.WriteLine(Usage); return 0; }
            try
            {
                string cmd = args[0].ToLowerInvariant();
                var opt = Options.Parse(args.Skip(1).ToArray());
                switch (cmd)
                {
                    case "scan": return Scan();
                    case "info": return Info(Resolve(opt));
                    case "install": return Install(Resolve(opt), opt);
                    case "config": return Config(Resolve(opt), opt);
                    case "uninstall": Installer.Uninstall(Resolve(opt).ExeDir, Console.WriteLine); return 0;
                    case "add":
                        if (opt.Target == null || !Directory.Exists(opt.Target)) throw new ArgumentException("请给出存在的文件夹");
                        Library.AddManual(Path.GetFullPath(opt.Target));
                        Console.WriteLine("已加入: " + Path.GetFullPath(opt.Target));
                        return 0;
                    case "gpu": return GpuInfo(opt);
                    case "params": return Params();
                    default: Console.WriteLine("未知命令 " + args[0] + "\n\n" + Usage); return 64;
                }
            }
            catch (Exception e) when (e is ArgumentException || e is InvalidOperationException || e is IOException || e is UnauthorizedAccessException)
            {
                Console.WriteLine("[停止] " + e.Message);
                return 1;
            }
        }

        class Options
        {
            public string Target, Preset, Proxy;
            public List<string> Sets = new List<string>();
            public bool Force;
            public int ResW, ResH;

            public static Options Parse(string[] a)
            {
                var o = new Options();
                for (int i = 0; i < a.Length; i++)
                {
                    Func<string> next = () => { if (i + 1 >= a.Length) throw new ArgumentException(a[i] + " 后面缺少值"); return a[++i]; };
                    switch (a[i].ToLowerInvariant())
                    {
                        case "--preset": o.Preset = next(); break;
                        case "--set": o.Sets.Add(next()); break;
                        case "--proxy": o.Proxy = next(); break;
                        case "--force": o.Force = true; break;
                        case "--res":
                            var r = next().Split('x', 'X');
                            if (r.Length != 2 || !int.TryParse(r[0], out o.ResW) || !int.TryParse(r[1], out o.ResH)) throw new ArgumentException("--res 写成 2560x1440");
                            break;
                        default:
                            if (a[i].StartsWith("--")) throw new ArgumentException("未知选项 " + a[i]);
                            o.Target = o.Target == null ? a[i] : o.Target + " " + a[i];
                            break;
                    }
                }
                if (o.Preset != null && !Presets.Names.Contains(o.Preset)) throw new ArgumentException("预设只能是 " + string.Join(" / ", Presets.Names));
                if (o.ResW == 0) { var s = Display.Primary(); o.ResW = s[0]; o.ResH = s[1]; }
                return o;
            }

            // 预设 + --set -> [DlssNr] 的改动 (--set 优先)
            public Dictionary<string, string> Changes()
            {
                var d = new Dictionary<string, string>();
                if (Preset != null) d["WorkingScale"] = Settings.Format(Settings.Find("WorkingScale"), Presets.Scale(Preset, ResW, ResH));
                foreach (var kv in Settings.FromPairs(Sets)) d[kv.Key] = kv.Value;
                return d;
            }
        }

        static Probe Resolve(Options o)
        {
            if (o.Target == null) throw new ArgumentException("请给出游戏文件夹或游戏名");
            if (Directory.Exists(o.Target)) return Probe.Run(o.Target);
            var hits = Library.ScanAll().Where(g => g.Name.IndexOf(o.Target, StringComparison.CurrentCultureIgnoreCase) >= 0).ToList();
            if (hits.Count == 0) throw new ArgumentException("游戏库里没有名字含 \"" + o.Target + "\" 的游戏，也不是存在的文件夹");
            if (hits.Count > 1) throw new ArgumentException("匹配到多个游戏，请写得更具体: " + string.Join(" | ", hits.Select(g => g.Name)));
            Console.WriteLine("游戏: " + hits[0].Name + " (" + hits[0].Source + ")");
            return Probe.Run(hits[0].Root, hits[0].HintExe);
        }

        static int Scan()
        {
            var games = Library.ScanAll();
            foreach (var g in games)
            {
                string mark = File.Exists(Manifest.PathIn(g.Root)) ? " [已安装]" : "";
                Console.WriteLine(string.Format("{0,-6} {1}{2}\n       {3}", g.Source, g.Name, mark, g.Root));
            }
            Console.WriteLine("共 " + games.Count + " 个。info <游戏名> 查看能否安装 (已安装标记只看游戏根目录，以 info 为准)。");
            return 0;
        }

        public static string Describe(Probe p)
        {
            var sb = new StringBuilder();
            sb.AppendLine("主程序    " + (p.Exe ?? "(没找到)"));
            sb.AppendLine("安装位置  " + p.ExeDir);
            sb.AppendLine("引擎      " + p.Engine);
            sb.AppendLine("图形 API  " + (p.Apis.Count > 0 ? string.Join(" / ", p.Apis) : "未能从导入表判断"));
            sb.AppendLine("超分      " + (p.Upscalers.Count > 0 ? string.Join(" / ", p.Upscalers) : "未发现"));
            sb.AppendLine("反作弊    " + (p.AntiCheat.Count > 0 ? "有" : "未发现"));
            sb.AppendLine("状态      " + (p.Installed != null ? "已安装 (注入名 " + p.Installed.proxy + ")" : "未安装"));
            var probs = p.Problems(false);
            foreach (string s in probs) sb.AppendLine("  ✗ " + s);
            foreach (string s in p.Warnings()) sb.AppendLine("  ! " + s);
            if (probs.Count == 0) sb.AppendLine(p.Upscalers.Count > 0 ? "  ✓ 可以安装" : "  ✓ 可以安装 (但见上面关于超分的提醒)");
            return sb.ToString();
        }

        static int Info(Probe p)
        {
            Console.Write(Describe(p));
            if (p.Installed != null) PrintSettings(p.ExeDir);
            return 0;
        }

        static void PrintSettings(string dir)
        {
            var cur = Installer.ReadSettings(dir);
            Console.WriteLine("当前参数 (" + Installer.IniPath(dir) + "):");
            foreach (var s in Settings.All)
            {
                string raw;
                cur.TryGetValue(s.Key, out raw);
                Console.WriteLine(string.Format("  {0,-18}{1,-8} {2}", s.Key, Settings.Format(s, Settings.Parse(s, raw)), s.Label));
            }
        }

        static int Install(Probe p, Options o)
        {
            Console.Write(Describe(p));
            var ch = o.Changes();
            if (!ch.ContainsKey("WorkingScale") && p.Installed == null)   // 首次安装且没指定: 用"均衡"
                ch["WorkingScale"] = Settings.Format(Settings.Find("WorkingScale"), Presets.Scale("均衡", o.ResW, o.ResH));
            Installer.Install(p, ch, o.Proxy, o.Force, Console.WriteLine);
            PrintSettings(p.ExeDir);
            Console.WriteLine("进游戏后在图形设置里打开 DLSS (或 FSR / XeSS)；按 Insert 打开 OptiScaler 菜单可实时调整。");
            return 0;
        }

        static int Config(Probe p, Options o)
        {
            if (p.Installed == null) throw new InvalidOperationException("这个游戏还没安装: " + p.ExeDir);
            var ch = o.Changes();
            if (ch.Count > 0)
            {
                Installer.ApplySettings(p.ExeDir, ch, p);
                Console.WriteLine("已写入 " + ch.Count + " 项。游戏运行中改的话需要重启游戏 (或在 OptiScaler 菜单里调)。");
            }
            PrintSettings(p.ExeDir);
            return 0;
        }

        static int GpuInfo(Options o)
        {
            Console.WriteLine("显卡: " + (Gpu.Name == "" ? "(未识别)" : Gpu.Name) + (Gpu.Supported ? "" : "  —— 不在支持范围 (需要 RTX 30/40/50)"));
            Console.WriteLine(string.Format("输出分辨率 {0}x{1} 下各预设 (估算，DLSS5 本身的耗时):", o.ResW, o.ResH));
            foreach (string n in Presets.Names)
            {
                double s = Presets.Scale(n, o.ResW, o.ResH);
                Console.WriteLine(string.Format(Const.Inv, "  {0}  模型分辨率 {1:0.##} ({2}x{3})  约 {4:0} ms", n, s,
                    (int)(o.ResW * s), (int)(o.ResH * s), Presets.EstimateMs(s, o.ResW, o.ResH)));
            }
            string pp = Installer.PayloadProblem();
            Console.WriteLine(pp == null ? "安装包: 完整" : "安装包: " + pp);
            return 0;
        }

        static int Params()
        {
            foreach (var s in Settings.All)
                Console.WriteLine(string.Format(Const.Inv, "{0,-18}{1} ({2}，默认 {3})\n                  {4}", s.Key, s.Label,
                    s.Kind == "bool" ? "true/false" : s.Min + "~" + s.Max, Settings.Format(s, s.Default), s.Help));
            return 0;
        }
    }

    public static class Display
    {
        [System.Runtime.InteropServices.DllImport("user32.dll")] static extern int GetSystemMetrics(int i);
        [System.Runtime.InteropServices.DllImport("user32.dll")] static extern bool SetProcessDPIAware();

        // 主显示器的物理分辨率
        public static int[] Primary()
        {
            try { SetProcessDPIAware(); } catch { }
            int w = GetSystemMetrics(0), h = GetSystemMetrics(1);
            return w > 0 ? new[] { w, h } : new[] { 1920, 1080 };
        }
    }
}
