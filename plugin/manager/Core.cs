// DLSS5 Manager 核心: 游戏库扫描、游戏检测、显卡/预设、OptiScaler.ini / dlss5fb.ini 读写、安装/卸载。
// 两种注入方式: OptiScaler (游戏自带超分，借用它的深度/运动矢量) 与兜底模式 (自带 dxgi.dll 代理，截取 DX12 画面)。
// 图形界面 (Gui.cs) 与命令行 (Cli.cs) 共用。目标 .NET Framework 4.8 (Windows 10/11 自带)。
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Management;
using System.Text;
using System.Text.RegularExpressions;
using System.Web.Script.Serialization;
using Microsoft.Win32;

namespace Dlss5Manager
{
    public static class Const
    {
        public const string Package = "DLSS5-RTX30";
        public const string BackupDir = ".dlss5_backup";
        public static readonly string[] ProxyNames = { "dxgi.dll", "winmm.dll", "version.dll", "dbghelp.dll", "d3d12.dll", "wininet.dll", "winhttp.dll" };
        public static readonly CultureInfo Inv = CultureInfo.InvariantCulture;
        public static string AppData
        {
            get
            {
                string d = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "DLSS5Manager");
                Directory.CreateDirectory(d);
                return d;
            }
        }
        public static string ExeDir { get { return AppDomain.CurrentDomain.BaseDirectory; } }
        public static string Payload { get { return Path.Combine(ExeDir, "payload"); } }
        public const string FallbackDir = "fallback";   // payload 里兜底模式的文件 (dxgi.dll, dlss5_nvngx.dll)
    }

    // 注入方式
    public static class Modes
    {
        public const string OptiScaler = "optiscaler", Fallback = "fallback";
        public static string Label(string m) { return m == Fallback ? "兜底模式 (截取画面)" : "OptiScaler (游戏自带超分)"; }
        public static string Parse(string s)
        {
            if (string.IsNullOrEmpty(s) || s == "auto") return null;
            s = s.ToLowerInvariant();
            if (s == OptiScaler || s == Fallback) return s;
            throw new ArgumentException("注入方式只能是 auto / optiscaler / fallback");
        }
    }

    // ------------------------------------------------------------------ 游戏库
    public class GameEntry
    {
        public string Name;
        public string Root;     // 游戏安装根目录
        public string Source;   // Steam / Epic / GOG / 手动
        public string HintExe;  // 商店记录的启动程序 (可能为空)
    }

    public static class Library
    {
        static readonly Regex SkipNames = new Regex(@"Redistributable|Steamworks Common|Proton|Steam Linux Runtime|SteamVR|Wallpaper Engine|Dedicated Server|\bSDK\b", RegexOptions.IgnoreCase);

        public static List<GameEntry> ScanAll()
        {
            var all = new List<GameEntry>();
            foreach (var f in new Func<IEnumerable<GameEntry>>[] { Steam, Epic, Gog, Manual })
            {
                try { all.AddRange(f()); } catch { }
            }
            return all.Where(g => g.Root != null && Directory.Exists(g.Root) && !SkipNames.IsMatch(g.Name ?? ""))
                      .GroupBy(g => Path.GetFullPath(g.Root).TrimEnd('\\').ToLowerInvariant())
                      .Select(x => x.First())
                      .OrderBy(g => g.Name, StringComparer.CurrentCultureIgnoreCase).ToList();
        }

        static IEnumerable<GameEntry> Steam()
        {
            string steam = Registry.GetValue(@"HKEY_CURRENT_USER\Software\Valve\Steam", "SteamPath", null) as string;
            if (steam == null) yield break;
            steam = steam.Replace('/', '\\');
            var libs = new List<string> { steam };
            string vdf = Path.Combine(steam, @"steamapps\libraryfolders.vdf");
            if (File.Exists(vdf))
                foreach (Match m in Regex.Matches(File.ReadAllText(vdf), "\"path\"\\s+\"([^\"]+)\""))
                    libs.Add(m.Groups[1].Value.Replace(@"\\", @"\"));
            foreach (string lib in libs.Distinct(StringComparer.OrdinalIgnoreCase))
            {
                string apps = Path.Combine(lib, "steamapps");
                if (!Directory.Exists(apps)) continue;
                foreach (string acf in Directory.GetFiles(apps, "appmanifest_*.acf"))
                {
                    string t = File.ReadAllText(acf);
                    Match n = Regex.Match(t, "\"name\"\\s+\"([^\"]*)\""), d = Regex.Match(t, "\"installdir\"\\s+\"([^\"]*)\"");
                    if (!d.Success) continue;
                    yield return new GameEntry { Name = n.Success ? n.Groups[1].Value : d.Groups[1].Value, Root = Path.Combine(apps, "common", d.Groups[1].Value), Source = "Steam" };
                }
            }
        }

        static IEnumerable<GameEntry> Epic()
        {
            string dir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData), @"Epic\EpicGamesLauncher\Data\Manifests");
            if (!Directory.Exists(dir)) yield break;
            var js = new JavaScriptSerializer();
            foreach (string item in Directory.GetFiles(dir, "*.item"))
            {
                Dictionary<string, object> d;
                try { d = js.Deserialize<Dictionary<string, object>>(File.ReadAllText(item)); } catch { continue; }
                object loc, name, exe, cats;
                if (!d.TryGetValue("InstallLocation", out loc)) continue;
                // 只要游戏: AppCategories 含 "games" (引擎、Bridge 等是 "applications")
                if (d.TryGetValue("AppCategories", out cats) && cats is System.Collections.IEnumerable
                    && !((System.Collections.IEnumerable)cats).Cast<object>().Any(c => string.Equals(c as string, "games", StringComparison.OrdinalIgnoreCase)))
                    continue;
                d.TryGetValue("DisplayName", out name);
                d.TryGetValue("LaunchExecutable", out exe);
                yield return new GameEntry { Name = (name ?? loc).ToString(), Root = loc.ToString(), Source = "Epic", HintExe = exe == null ? null : exe.ToString() };
            }
        }

        static IEnumerable<GameEntry> Gog()
        {
            foreach (string path in new[] { @"SOFTWARE\WOW6432Node\GOG.com\Games", @"SOFTWARE\GOG.com\Games" })
            {
                RegistryKey k = Registry.LocalMachine.OpenSubKey(path);
                if (k == null) continue;
                foreach (string id in k.GetSubKeyNames())
                {
                    RegistryKey g = k.OpenSubKey(id);
                    if (g == null) continue;
                    string root = g.GetValue("path") as string;
                    if (root == null) continue;
                    yield return new GameEntry { Name = (g.GetValue("gameName") as string) ?? id, Root = root, Source = "GOG", HintExe = g.GetValue("exe") as string };
                }
            }
        }

        static string ManualFile { get { return Path.Combine(Const.AppData, "manual_games.txt"); } }

        static IEnumerable<GameEntry> Manual()
        {
            if (!File.Exists(ManualFile)) yield break;
            foreach (string line in File.ReadAllLines(ManualFile, Encoding.UTF8))
                if (line.Trim().Length > 0)
                    yield return new GameEntry { Name = Path.GetFileName(line.Trim().TrimEnd('\\')), Root = line.Trim(), Source = "手动" };
        }

        public static void AddManual(string dir)
        {
            var lines = File.Exists(ManualFile) ? File.ReadAllLines(ManualFile, Encoding.UTF8).ToList() : new List<string>();
            if (!lines.Any(l => string.Equals(l.Trim(), dir, StringComparison.OrdinalIgnoreCase))) lines.Add(dir);
            File.WriteAllLines(ManualFile, lines, Encoding.UTF8);
        }
    }

    // ------------------------------------------------------------------ PE 导入表 (判断图形 API)
    public static class Pe
    {
        // 返回导入与延迟导入的 dll 名 (小写)。只读文件头和导入段，大 exe 也很快。
        public static HashSet<string> Imports(string path)
        {
            var set = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            try
            {
                using (var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
                using (var br = new BinaryReader(fs))
                {
                    fs.Position = 0x3C;
                    int pe = br.ReadInt32();
                    fs.Position = pe + 4;
                    br.ReadUInt16();
                    int nsec = br.ReadUInt16();
                    fs.Position = pe + 20;
                    int optSize = br.ReadUInt16();
                    int opt = pe + 24;
                    fs.Position = opt;
                    bool pe32plus = br.ReadUInt16() == 0x20B;
                    int dirs = opt + (pe32plus ? 112 : 96);
                    var secs = new List<uint[]>();
                    for (int i = 0; i < nsec; i++)
                    {
                        fs.Position = opt + optSize + 40 * i + 8;
                        uint vsz = br.ReadUInt32(), va = br.ReadUInt32(), rsz = br.ReadUInt32(), raw = br.ReadUInt32();
                        secs.Add(new[] { va, Math.Max(vsz, rsz), raw });
                    }
                    Func<uint, long> off = rva =>
                    {
                        foreach (var s in secs) if (rva >= s[0] && rva < s[0] + s[1]) return rva - s[0] + s[2];
                        return -1;
                    };
                    Func<long, string> cstr = o =>
                    {
                        fs.Position = o;
                        var sb = new StringBuilder();
                        for (int c; (c = fs.ReadByte()) > 0 && sb.Length < 260;) sb.Append((char)c);
                        return sb.ToString();
                    };
                    foreach (var entry in new[] { new { Dir = 1, Size = 20, NameAt = 12 }, new { Dir = 13, Size = 32, NameAt = 4 } })
                    {
                        fs.Position = dirs + 8 * entry.Dir;
                        uint rva = br.ReadUInt32();
                        long o = rva == 0 ? -1 : off(rva);
                        for (int i = 0; o >= 0 && i < 4096; i++, o += entry.Size)
                        {
                            fs.Position = o + entry.NameAt;
                            uint nameRva = br.ReadUInt32();
                            if (nameRva == 0) break;
                            long no = off(nameRva);
                            if (no >= 0) set.Add(cstr(no));
                        }
                    }
                }
            }
            catch { }
            return set;
        }
    }

    // ------------------------------------------------------------------ 游戏检测
    public class Probe
    {
        public string Root, ExeDir, Exe, Engine = "未知";
        public List<string> Apis = new List<string>();
        public List<string> Upscalers = new List<string>();
        public List<string> AntiCheat = new List<string>();
        public Dictionary<string, string> ProxyOwners = new Dictionary<string, string>();   // 已存在的代理 dll -> 归属
        public Manifest Installed;
        public bool ApisGuessed;   // 导入表里没有，是从程序里的字符串推测的 (运行时才加载图形库的引擎，如 Godot)
        public bool UnrealEngine { get { return Engine.StartsWith("Unreal"); } }

        // 没有超分、但能跑 DX12 的游戏默认走兜底模式；已安装的以安装记录为准
        public string SuggestedMode { get { return Upscalers.Count == 0 && Apis.Contains("DX12") ? Modes.Fallback : Modes.OptiScaler; } }
        public string Mode { get { return Installed != null ? (Installed.mode ?? Modes.OptiScaler) : SuggestedMode; } }

        static readonly Regex SkipDir = new Regex(@"^(_CommonRedist|Redist|redist|DirectX|vcredist|EasyAntiCheat|BattlEye|Support|Installers?|__Installer|\.dlss5_backup|ThirdParty|Prerequisites|CrashReportClient)$", RegexOptions.IgnoreCase);
        static readonly Regex SkipExe = new Regex(@"^(unins|setup|vc_?redist|dxsetup|UnityCrashHandler|CrashReport|.*crash.*|.*launcher.*|.*helper.*|EasyAntiCheat.*|BEService.*|start_protected_game|.*_BE|.*_EAC|REDprelauncher|QuickSFV|dotnet.*|oalinst|PhysX.*)", RegexOptions.IgnoreCase);
        static readonly Regex AcName = new Regex(@"^(EasyAntiCheat|EAC(?![a-z])|BattlEye|BEService|xigncode|GameGuard|nProtect|vgk|mhyprot|ACE-|AntiCheatExpert|start_protected_game)|anti.?cheat", RegexOptions.IgnoreCase);
        static readonly Dictionary<string, string> UpscalerFiles = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            { "nvngx_dlss.dll", "DLSS" }, { "sl.dlss.dll", "DLSS (Streamline)" }, { "NVUnityPlugin.dll", "DLSS (Unity)" },
            { "libxess.dll", "XeSS" }, { "libxess_dx11.dll", "XeSS" },
            { "amd_fidelityfx_dx12.dll", "FSR" }, { "amd_fidelityfx_vk.dll", "FSR" }, { "amd_fidelityfx_upscaler_dx12.dll", "FSR" },
            { "ffx_fsr2_api_x64.dll", "FSR2" }, { "ffx_fsr2_api_dx12_x64.dll", "FSR2" }, { "ffx_fsr2_api_vk_x64.dll", "FSR2" },
        };

        static IEnumerable<string> Files(string dir, int depth)
        {
            string[] files = new string[0], dirs = new string[0];
            try { files = Directory.GetFiles(dir); dirs = Directory.GetDirectories(dir); } catch { }
            foreach (string f in files) yield return f;
            if (depth <= 0) yield break;
            foreach (string d in dirs)
                if (!SkipDir.IsMatch(Path.GetFileName(d)))
                    foreach (string f in Files(d, depth - 1)) yield return f;
        }

        // root: 游戏根目录或 exe 目录都可以；hintExe: 商店记录的启动程序 (相对 root)
        public static Probe Run(string root, string hintExe = null)
        {
            var p = new Probe { Root = Path.GetFullPath(root) };
            var all = Files(p.Root, 6).ToList();
            var exes = all.Where(f => f.EndsWith(".exe", StringComparison.OrdinalIgnoreCase) && !SkipExe.IsMatch(Path.GetFileNameWithoutExtension(f))).ToList();

            // 主程序: Unreal 的 *-Shipping.exe > 商店记录 > 最大的 exe
            string exe = exes.FirstOrDefault(f => Regex.IsMatch(Path.GetFileName(f), @"-Win64-Shipping\.exe$|-WinGDK-Shipping\.exe$", RegexOptions.IgnoreCase));
            if (exe == null && hintExe != null)
            {
                string h = Path.Combine(p.Root, hintExe);
                if (File.Exists(h) && !SkipExe.IsMatch(Path.GetFileNameWithoutExtension(h))) exe = h;
            }
            if (exe == null) exe = exes.OrderByDescending(f => { try { return new FileInfo(f).Length; } catch { return 0L; } }).FirstOrDefault();
            p.Exe = exe;
            p.ExeDir = exe != null ? Path.GetDirectoryName(exe) : p.Root;

            // 引擎
            if (exe != null && Regex.IsMatch(exe, @"-Shipping\.exe$", RegexOptions.IgnoreCase)) p.Engine = "Unreal Engine";
            else if (File.Exists(Path.Combine(p.ExeDir, "UnityPlayer.dll"))) p.Engine = "Unity";
            else if (all.Any(f => Path.GetFileName(f).Equals("REDprelauncher.exe", StringComparison.OrdinalIgnoreCase))) p.Engine = "REDengine";
            else if (Directory.Exists(Path.Combine(p.Root, "Engine"))) p.Engine = "Unreal Engine";

            // 图形 API: 主程序 (Unity 看 UnityPlayer.dll) 的导入表 + Agility SDK
            var imp = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            if (exe != null) imp.UnionWith(Pe.Imports(exe));
            string up = Path.Combine(p.ExeDir, "UnityPlayer.dll");
            if (File.Exists(up)) imp.UnionWith(Pe.Imports(up));
            if (imp.Contains("d3d12.dll") || File.Exists(Path.Combine(p.ExeDir, @"D3D12\D3D12Core.dll"))) p.Apis.Add("DX12");
            if (imp.Contains("d3d11.dll")) p.Apis.Add("DX11");
            if (imp.Contains("vulkan-1.dll")) p.Apis.Add("Vulkan");
            if (p.Apis.Count == 0 && exe != null)
            {
                var found = FindAscii(exe, "D3D12CreateDevice", "D3D11CreateDevice", "vkCreateInstance");
                if (found.Contains("D3D12CreateDevice")) p.Apis.Add("DX12");
                if (found.Contains("D3D11CreateDevice")) p.Apis.Add("DX11");
                if (found.Contains("vkCreateInstance")) p.Apis.Add("Vulkan");
                p.ApisGuessed = p.Apis.Count > 0;
            }

            // 超分
            foreach (string f in all)
            {
                string kind;
                if (UpscalerFiles.TryGetValue(Path.GetFileName(f), out kind) && !p.Upscalers.Contains(kind) && f.IndexOf(Const.BackupDir, StringComparison.OrdinalIgnoreCase) < 0
                    && !f.StartsWith(p.ExeDir + "\\OptiScaler", StringComparison.OrdinalIgnoreCase))
                    p.Upscalers.Add(kind);
            }
            if (p.UnrealEngine && Directory.Exists(Path.Combine(p.Root, @"Engine\Plugins\Runtime\Nvidia\DLSS")) && !p.Upscalers.Any(u => u.StartsWith("DLSS")))
                p.Upscalers.Add("DLSS");

            // 反作弊: exe 目录与往上 3 层
            var roots = new List<string> { p.ExeDir };
            for (string d = p.ExeDir; roots.Count < 4 && (d = Path.GetDirectoryName(d)) != null;) roots.Add(d);
            foreach (string r in roots)
            {
                IEnumerable<string> entries;
                try { entries = Directory.GetFileSystemEntries(r); } catch { continue; }
                foreach (string e in entries)
                    if (AcName.IsMatch(Path.GetFileName(e)) && !p.AntiCheat.Contains(e)) p.AntiCheat.Add(e);
                if (string.Equals(r.TrimEnd('\\'), p.Root.TrimEnd('\\'), StringComparison.OrdinalIgnoreCase)) break;
            }

            // 已占用的代理 dll 与本工具的安装记录
            p.Installed = Manifest.Load(p.ExeDir);
            foreach (string n in Const.ProxyNames)
            {
                string f = Path.Combine(p.ExeDir, n);
                if (!File.Exists(f)) continue;
                if (p.Installed != null && p.Installed.proxy == n) p.ProxyOwners[n] = "本工具";
                else p.ProxyOwners[n] = Identify(f);
            }
            return p;
        }

        // 文件里出现了哪些 ASCII 串 (分块读，块间重叠，不整个读进内存)
        static HashSet<string> FindAscii(string path, params string[] needles)
        {
            var found = new HashSet<string>();
            int overlap = needles.Max(n => n.Length);
            var buf = new byte[(1 << 22) + overlap];
            var pats = needles.Select(n => Encoding.ASCII.GetBytes(n)).ToArray();
            try
            {
                using (var f = File.OpenRead(path))
                {
                    int keep = 0, read;
                    while (found.Count < needles.Length && (read = f.Read(buf, keep, buf.Length - keep)) > 0)
                    {
                        int len = keep + read;
                        for (int k = 0; k < pats.Length; k++)
                        {
                            if (found.Contains(needles[k])) continue;
                            byte[] p = pats[k];
                            for (int i = Array.IndexOf(buf, p[0], 0, len); i >= 0 && i <= len - p.Length; i = Array.IndexOf(buf, p[0], i + 1, len - i - 1))
                            {
                                int j = 1;
                                while (j < p.Length && buf[i + j] == p[j]) j++;
                                if (j == p.Length) { found.Add(needles[k]); break; }
                            }
                        }
                        keep = Math.Min(overlap, len);
                        Buffer.BlockCopy(buf, len - keep, buf, 0, keep);
                    }
                }
            }
            catch { }
            return found;
        }

        static string Identify(string dll)
        {
            try
            {
                var v = FileVersionInfo.GetVersionInfo(dll);
                string s = (v.OriginalFilename ?? "") + " " + (v.ProductName ?? "") + " " + (v.FileDescription ?? "");
                if (s.IndexOf("OptiScaler", StringComparison.OrdinalIgnoreCase) >= 0) return "OptiScaler (非本工具安装)";
                if (s.IndexOf("ReShade", StringComparison.OrdinalIgnoreCase) >= 0) return "ReShade";
                if (s.IndexOf("Special K", StringComparison.OrdinalIgnoreCase) >= 0) return "Special K";
                if (s.IndexOf("Microsoft", StringComparison.OrdinalIgnoreCase) >= 0) return "游戏自带的系统库";
                return string.IsNullOrWhiteSpace(s) ? "未知 mod" : s.Trim();
            }
            catch { return "未知"; }
        }

        // 能否安装、以及要提醒用户的事
        // mode: null = 按 Mode
        public List<string> Problems(bool force, string mode = null)
        {
            mode = mode ?? Mode;
            var list = new List<string>();
            if (Exe == null) list.Add("没找到游戏主程序 (.exe)");
            if (AntiCheat.Count > 0 && !force) list.Add("发现反作弊组件，注入 dll 可能导致封号，不安装");
            if (ProxyOwners.Values.Any(v => v.StartsWith("OptiScaler"))) list.Add("已有一份别人装的 OptiScaler，请先卸载它，避免两份冲突");
            if (mode == Modes.Fallback)
            {
                if (!Apis.Contains("DX12")) list.Add("兜底模式只支持 DX12 游戏" + (Apis.Count > 0 ? " (这个游戏是 " + string.Join(" / ", Apis) + ")" : ""));
                string owner;
                if (ProxyOwners.TryGetValue("dxgi.dll", out owner) && owner != "本工具")
                    list.Add("兜底模式必须以 dxgi.dll 注入，但它已被占用 (" + owner + ")");
            }
            return list;
        }

        public List<string> Warnings(string mode = null)
        {
            mode = mode ?? Mode;
            var list = new List<string>();
            if (mode == Modes.Fallback)
            {
                list.Add("兜底模式: 从游戏画面直接截取，没有深度、运动矢量和历史帧，UI 也会一起被处理；进游戏后 F10 开关、F11 左右对比");
                if (Engine == "Unity" && Apis.Contains("DX11"))
                    list.Add("Unity 游戏常默认用 DX11，兜底模式需要 DX12: 在 Steam 启动选项里加 -force-d3d12");
                if (Upscalers.Count > 0) list.Add("这个游戏自带超分，用 OptiScaler 方式效果更好 (有深度和运动矢量)");
                if (ApisGuessed && Apis.Count > 1)
                    list.Add("图形 API 是从程序里推测的，游戏可能支持多种: 必须以 DX12 运行兜底模式才生效 (Godot 游戏可加启动参数 --rendering-driver d3d12)；"
                             + "装好后进一次游戏，dlss5fb.log 里出现 tracked 就说明挂上了");
            }
            else
            {
                if (Upscalers.Count == 0)
                    list.Add(UnrealEngine ? "没找到超分的 dll；Unreal 游戏的 FSR/TSR 常常编进了主程序，能否生效要进游戏看 OptiScaler 菜单"
                                          : "没找到 DLSS/FSR/XeSS。OptiScaler 方式要从游戏的超分调用里拿深度和运动矢量，游戏不支持超分时不会生效"
                                            + (Apis.Contains("DX12") ? "；可以改用兜底模式" : ""));
                if (Apis.Count > 0 && !Apis.Contains("DX12") && !Apis.Contains("Vulkan") && Apis.Contains("DX11"))
                    list.Add("DX11 游戏: 会把超分切到 dlss_12 (DX11-on-12)，这是 DX11 下唯一能跑 DLSS5 的方式");
                foreach (var kv in ProxyOwners)
                    if (kv.Value != "本工具") list.Add(kv.Key + " 已被占用 (" + kv.Value + ")，会换用别的注入文件名");
            }
            if (AntiCheat.Count > 0) list.Add("反作弊: " + string.Join(", ", AntiCheat.Select(Path.GetFileName)));
            return list;
        }
    }

    // ------------------------------------------------------------------ 显卡与预设
    public static class Gpu
    {
        static string name;
        public static string Name
        {
            get
            {
                if (name != null) return name;
                name = "";
                try
                {
                    using (var s = new ManagementObjectSearcher("SELECT Name FROM Win32_VideoController"))
                        foreach (ManagementObject o in s.Get())
                        {
                            string n = (o["Name"] ?? "").ToString();
                            if (n.IndexOf("NVIDIA", StringComparison.OrdinalIgnoreCase) >= 0) { name = n; break; }
                            if (name == "") name = n;
                        }
                }
                catch { }
                return name;
            }
        }

        // 相对 RTX 3060 的 DLSS5 吞吐 (张量算力粗估；3060 实测 1080p 41.9 ms)。0 = 不支持
        public static double Speed
        {
            get
            {
                string n = Name;
                var table = new[]
                {
                    new { P = @"RTX 50\d\d", S = 4.0 },
                    new { P = @"RTX 4090", S = 4.0 }, new { P = @"RTX 4080", S = 2.8 }, new { P = @"RTX 4070 Ti", S = 2.1 },
                    new { P = @"RTX 4070", S = 1.7 }, new { P = @"RTX 4060 Ti", S = 1.3 }, new { P = @"RTX 40\d\d", S = 1.1 },
                    new { P = @"RTX 3090", S = 2.3 }, new { P = @"RTX 3080", S = 2.0 }, new { P = @"RTX 3070", S = 1.4 },
                    new { P = @"RTX 3060 Ti", S = 1.2 }, new { P = @"RTX 30\d\d", S = 1.0 }, new { P = @"RTX A\d+", S = 1.0 },
                };
                foreach (var t in table) if (Regex.IsMatch(n, t.P)) return t.S;
                return 0;
            }
        }
        public static bool Supported { get { return Speed > 0; } }
    }

    public static class Presets
    {
        public static readonly string[] Names = { "性能", "均衡", "画质" };
        static readonly double[] BudgetMs = { 10, 14, 20 };   // DLSS5 本身每帧可占的时间 (本机显卡上)

        // RTX 3060 实测 (移植版 kernel，NR-only): 540p 14.3 / 720p 22.6 / 900p 31.8 / 1080p 41.9 ms
        // 拟合 t = 5.1 + 17.8 × 百万像素 (固定开销 + 按像素)；其他显卡按 Gpu.Speed 整体缩放
        const double FixedMs = 5.1, MsPerMpx = 17.8;

        public static double EstimateMs(double scale, int outW, int outH)
        {
            double speed = Gpu.Speed > 0 ? Gpu.Speed : 1.0;
            return (FixedMs + MsPerMpx * outW * (double)outH * scale * scale / 1e6) / speed;
        }

        // 预算内能用的最大 WorkingScale (0.05 步进，0.25~1.0)
        public static double Scale(string preset, int outW, int outH)
        {
            int i = Array.IndexOf(Names, preset);
            if (i < 0) i = 1;
            double speed = Gpu.Speed > 0 ? Gpu.Speed : 1.0;
            double mpx = (BudgetMs[i] * speed - FixedMs) / MsPerMpx;   // 预算内 (本机显卡) 能处理的百万像素
            double s = mpx > 0 ? Math.Sqrt(mpx * 1e6 / (outW * (double)outH)) : 0.25;
            s = Math.Max(0.25, Math.Min(1.0, s));
            return Math.Floor(s * 20 + 1e-9) / 20;
        }
    }

    // ------------------------------------------------------------------ 可调参数 (OptiScaler.ini 的 [DlssNr] 节)
    public class Setting
    {
        public string Key, Label, Help, Kind;   // Kind: bool / float / int / key
        public double Min, Max, Default;
        public string Only;                     // null = 两种注入方式都有；否则只属于该方式
        public bool AppliesTo(string mode) { return Only == null || Only == mode; }
    }

    public static class Settings
    {
        public static readonly Setting[] All =
        {
            new Setting { Key = "Enabled", Label = "启用 DLSS5", Kind = "bool", Default = 1, Help = "关掉后游戏照常运行，只是不跑 DLSS5" },
            new Setting { Key = "WorkingScale", Label = "模型分辨率", Kind = "float", Min = 0.25, Max = 1.0, Default = 0.5,
                Help = "模型工作分辨率占输出的比例，耗时约按平方下降 (3060 上 1080p: 1.0 ≈ 42 ms，0.5 ≈ 14 ms)" },
            new Setting { Key = "Intensity", Label = "效果强度", Kind = "float", Min = 0, Max = 1, Default = 1,
                Help = "DLSSNR.Intensity: 在原画面与模型输出之间线性混合，0 = 原画面" },
            new Setting { Key = "LocalStructure", Label = "局部结构", Kind = "float", Min = 0, Max = 2, Default = 1,
                Help = "DLSSNR.LocalStructureStrength: 细节/纹理的合成强度 (影响最大的一项)" },
            new Setting { Key = "LocalTone", Label = "局部色调", Kind = "float", Min = 0, Max = 2, Default = 1,
                Help = "DLSSNR.LocalToneStrength: 局部明暗的调整强度" },
            new Setting { Key = "SkinStructure", Label = "皮肤结构", Kind = "float", Min = -1, Max = 2, Default = -1,
                Help = "DLSSNR.SkinStructureStrength: 皮肤区域的细节强度；-1 = 跟随局部结构 (模型默认)" },
            new Setting { Key = "AutoMask", Label = "自动遮罩", Kind = "bool", Default = 1,
                Help = "DLSSNR.UseAutoMask: 让模型自己区分皮肤等区域；关掉后皮肤/结构两路输入都置 -1" },
            new Setting { Key = "Style", Label = "风格", Kind = "int", Min = 0, Max = 2, Default = 0,
                Help = "DLSSNR.Style 0/1/2: 1、2 会额外做一次调色后处理" },
            new Setting { Key = "TransferStrength", Label = "合成强度", Kind = "float", Min = 0, Max = 1.5, Default = 1,
                Help = "OptiScaler 的合成: 画面向模型结果移动多少 (按亮度合成)，0 = 超分原输出", Only = Modes.OptiScaler },
            new Setting { Key = "ColourStrength", Label = "颜色强度", Kind = "float", Min = 0, Max = 1, Default = 1,
                Help = "0 = 保持游戏原本的色相，只取模型的明暗；1 = 连颜色一起取", Only = Modes.OptiScaler },
            new Setting { Key = "MaxRatio", Label = "最大增亮倍数", Kind = "float", Min = 1, Max = 4, Default = 2,
                Help = "单个像素最多被提亮到原来的几倍，防止亮光源变成色块", Only = Modes.OptiScaler },
            new Setting { Key = "Temporal", Label = "历史帧 + 光流", Kind = "bool", Default = 1, Only = Modes.Fallback,
                Help = "兜底模式: 用显卡的硬件光流算运动矢量，让模型沿用历史帧 (运动中闪烁减半以上，约 +4 ms)；关掉 = 每帧独立" },
            new Setting { Key = "Compare", Label = "左右对比", Kind = "bool", Default = 0, Only = Modes.Fallback,
                Help = "兜底模式: 左半屏原画面、右半屏 DLSS5 (游戏里按 F11 也能切换)" },
            new Setting { Key = "Stabilize", Label = "输入死区", Kind = "float", Min = 0, Max = 4, Default = 1.5, Only = Modes.Fallback,
                Help = "兜底模式: 像素变化小于这么多 (1/255 为单位) 时沿用上次送进模型的值，滤掉抖动、胶片颗粒造成的闪烁；0 = 关" },
            new Setting { Key = "Smooth", Label = "时间平滑", Kind = "float", Min = 0.05, Max = 1, Default = 0.2, Only = Modes.Fallback,
                Help = "兜底模式: 画面没变的地方，DLSS5 的改动每帧只跟进这么多 (越小越稳)；画面在变的地方立刻跟上，不拖影；1 = 关" },
        };

        public static IEnumerable<Setting> For(string mode) { return All.Where(s => s.AppliesTo(mode)); }

        public static Setting Find(string key)
        {
            return All.FirstOrDefault(s => string.Equals(s.Key, key, StringComparison.OrdinalIgnoreCase));
        }

        public static string Format(Setting s, double v)
        {
            if (s.Kind == "bool") return v != 0 ? "true" : "false";
            if (s.Kind == "int") return ((int)Math.Round(v)).ToString(Const.Inv);
            return Math.Round(v, 3).ToString("0.###", Const.Inv);
        }

        // ini 里的值 -> 数值 ("auto" 或缺省时用默认值)
        public static double Parse(Setting s, string raw)
        {
            if (raw == null || raw.Trim().Equals("auto", StringComparison.OrdinalIgnoreCase)) return s.Default;
            raw = raw.Trim();
            if (s.Kind == "bool") return raw.Equals("true", StringComparison.OrdinalIgnoreCase) || raw == "1" ? 1 : 0;
            double v;
            return double.TryParse(raw, NumberStyles.Float, Const.Inv, out v) ? v : s.Default;
        }

        // "Key=Value" 列表 -> 规范化的 ini 值，未知 key 或越界值抛错
        public static Dictionary<string, string> FromPairs(IEnumerable<string> pairs)
        {
            var d = new Dictionary<string, string>();
            foreach (string kv in pairs)
            {
                int i = kv.IndexOf('=');
                if (i <= 0) throw new ArgumentException("参数要写成 名字=值: " + kv);
                Setting s = Find(kv.Substring(0, i).Trim());
                if (s == null) throw new ArgumentException("未知参数 " + kv.Substring(0, i) + "，可用: " + string.Join(", ", All.Select(x => x.Key)));
                string raw = kv.Substring(i + 1).Trim();
                if (raw.Equals("auto", StringComparison.OrdinalIgnoreCase)) { d[s.Key] = "auto"; continue; }
                double v = Parse(s, raw);
                if (s.Kind != "bool" && (v < s.Min || v > s.Max))
                    throw new ArgumentException(string.Format(Const.Inv, "{0} 的范围是 {1}~{2}", s.Key, s.Min, s.Max));
                d[s.Key] = Format(s, v);
            }
            return d;
        }
    }

    // ------------------------------------------------------------------ ini: 只改指定节里的 key，保留注释和其余内容
    public static class Ini
    {
        public static Dictionary<string, string> Read(string path, string section)
        {
            var d = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            if (!File.Exists(path)) return d;
            string cur = null;
            foreach (string line in File.ReadAllLines(path))
            {
                Match m = Regex.Match(line, @"^\s*\[(\w+)\]\s*$");
                if (m.Success) { cur = m.Groups[1].Value; continue; }
                if (!string.Equals(cur, section, StringComparison.OrdinalIgnoreCase)) continue;
                m = Regex.Match(line, @"^\s*(\w+)\s*=(.*)$");
                if (m.Success) d[m.Groups[1].Value] = m.Groups[2].Value.Trim();
            }
            return d;
        }

        public static void Write(string path, Dictionary<string, Dictionary<string, string>> changes)
        {
            var lines = File.Exists(path) ? File.ReadAllLines(path).ToList() : new List<string>();
            var outp = new List<string>();
            var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            string cur = null;
            Action flush = () =>
            {
                Dictionary<string, string> ch;
                if (cur != null && changes.TryGetValue(cur, out ch))
                    foreach (var kv in ch) if (seen.Add(cur + "." + kv.Key)) outp.Add(kv.Key + "=" + kv.Value);
            };
            foreach (string line in lines)
            {
                Match m = Regex.Match(line, @"^\s*\[(\w+)\]\s*$");
                if (m.Success)
                {
                    flush();
                    cur = m.Groups[1].Value;
                    outp.Add(line);
                    continue;
                }
                Dictionary<string, string> c;
                m = Regex.Match(line, @"^\s*(\w+)\s*=");
                if (m.Success && cur != null && changes.TryGetValue(cur, out c) && c.ContainsKey(m.Groups[1].Value))
                {
                    string k = m.Groups[1].Value;
                    outp.Add(k + "=" + c[k]);
                    seen.Add(cur + "." + k);
                    continue;
                }
                outp.Add(line);
            }
            flush();
            foreach (var sec in changes)
                if (!lines.Any(l => Regex.IsMatch(l, @"^\s*\[" + sec.Key + @"\]\s*$")))
                {
                    outp.Add("[" + sec.Key + "]");
                    foreach (var kv in sec.Value) outp.Add(kv.Key + "=" + kv.Value);
                }
            File.WriteAllText(path, string.Join("\r\n", outp) + "\r\n", new UTF8Encoding(false));
        }
    }

    // ------------------------------------------------------------------ 安装记录 (与旧版 install.ps1 的格式兼容)
    public class Manifest
    {
        public string package = Const.Package;
        public List<string> installed = new List<string>();
        public List<string> backups = new List<string>();
        public string proxy;
        public string mode;     // null (旧记录) / optiscaler / fallback
        public double scale;
        public string gpu;
        public string date;
        public string tool = "DLSS5Manager";

        public static string PathIn(string exeDir) { return Path.Combine(exeDir, Const.Package + ".manifest.json"); }

        public static Manifest Load(string exeDir)
        {
            string p = PathIn(exeDir);
            if (!File.Exists(p)) return null;
            try { return new JavaScriptSerializer().Deserialize<Manifest>(File.ReadAllText(p, Encoding.UTF8)); }
            catch { return new Manifest(); }
        }

        public void Save(string exeDir)
        {
            File.WriteAllText(PathIn(exeDir), new JavaScriptSerializer().Serialize(this), new UTF8Encoding(false));
        }
    }

    public static class Installer
    {
        [System.Runtime.InteropServices.DllImport("kernel32.dll", CharSet = System.Runtime.InteropServices.CharSet.Unicode)]
        static extern bool DeleteFileW(string path);

        static readonly string[] OptiFiles = { "OptiScaler.dll", "OptiScaler.ini", "nvngx.dll_dlssnr.dll", "nvngx_dlssnr.dll" };
        static readonly string[] FallbackFiles = { Const.FallbackDir + @"\dxgi.dll", Const.FallbackDir + @"\dlss5_nvngx.dll", "nvngx_dlssnr.dll" };

        // mode: null = 两种方式的文件都检查
        public static string PayloadProblem(string mode = null)
        {
            if (!Directory.Exists(Const.Payload)) return "找不到 payload 文件夹 (应在本程序旁边): " + Const.Payload;
            var need = mode == Modes.Fallback ? FallbackFiles : mode == Modes.OptiScaler ? OptiFiles : OptiFiles.Concat(FallbackFiles).Distinct();
            foreach (string f in need)
                if (!File.Exists(Path.Combine(Const.Payload, f)))
                    return "payload 缺少 " + f + (f == "nvngx_dlssnr.dll" ? " (NVIDIA 的模型文件，需自行放入，不随工具分发)" : "");
            return null;
        }

        static string InstalledMode(string exeDir)
        {
            Manifest m = Manifest.Load(exeDir);
            return m != null && m.mode == Modes.Fallback ? Modes.Fallback : Modes.OptiScaler;
        }

        public static string IniPath(string exeDir, string mode = null)
        {
            return Path.Combine(exeDir, (mode ?? InstalledMode(exeDir)) == Modes.Fallback ? "dlss5fb.ini" : "OptiScaler.ini");
        }

        static string Section(string mode) { return mode == Modes.Fallback ? "DLSS5" : "DlssNr"; }

        public static Dictionary<string, string> ReadSettings(string exeDir, string mode = null)
        {
            mode = mode ?? InstalledMode(exeDir);
            return Ini.Read(IniPath(exeDir, mode), Section(mode));
        }

        // mode: null = 按安装记录
        public static void ApplySettings(string exeDir, Dictionary<string, string> dlssnr, Probe probe = null, string mode = null)
        {
            mode = mode ?? InstalledMode(exeDir);
            var own = dlssnr.Where(kv => Settings.Find(kv.Key) == null || Settings.Find(kv.Key).AppliesTo(mode)).ToDictionary(kv => kv.Key, kv => kv.Value);
            var changes = new Dictionary<string, Dictionary<string, string>> { { Section(mode), own } };
            // DX11 游戏只能经 dx11on12 的 DLSS 跑 DLSS5
            if (mode == Modes.OptiScaler && probe != null && probe.Apis.Contains("DX11") && !probe.Apis.Contains("DX12") && !probe.Apis.Contains("Vulkan"))
                changes["Upscalers"] = new Dictionary<string, string> { { "Dx11Upscaler", "dlss_12" } };
            Ini.Write(IniPath(exeDir, mode), changes);
        }

        // log: 进度输出 (界面与命令行各自显示)；mode: null = 按检测结果自动选
        public static void Install(Probe p, Dictionary<string, string> dlssnr, string proxy, bool force, Action<string> log, string mode = null)
        {
            mode = mode ?? p.SuggestedMode;
            string pp = PayloadProblem(mode);
            if (pp != null) throw new InvalidOperationException(pp);
            var probs = p.Problems(force, mode);
            if (probs.Count > 0) throw new InvalidOperationException(string.Join("；", probs));
            if (!Gpu.Supported && !force) throw new InvalidOperationException("显卡 " + Gpu.Name + " 不在支持范围 (需要 RTX 30/40/50)");

            string dir = p.ExeDir;
            Dictionary<string, string> keep = null;
            if (p.Installed != null)
            {
                log("检测到已安装，先卸载旧版本 (保留现有参数)...");
                keep = ReadSettings(dir);
                Uninstall(dir, log);
                p.ProxyOwners.Clear();
                foreach (string n in Const.ProxyNames) if (File.Exists(Path.Combine(dir, n))) p.ProxyOwners[n] = "其他";
            }

            if (mode == Modes.Fallback)
            {
                if (!string.IsNullOrEmpty(proxy) && proxy != "auto" && !proxy.Equals("dxgi.dll", StringComparison.OrdinalIgnoreCase))
                    throw new InvalidOperationException("兜底模式只能以 dxgi.dll 注入");
                proxy = "dxgi.dll";
            }
            else if (string.IsNullOrEmpty(proxy) || proxy == "auto")
            {
                proxy = Const.ProxyNames.FirstOrDefault(n => !File.Exists(Path.Combine(dir, n)));
                if (proxy == null) throw new InvalidOperationException("常用的注入文件名都被占用了，请手动指定一个");
            }
            else if (!Const.ProxyNames.Contains(proxy.ToLowerInvariant()))
                throw new InvalidOperationException("不支持的注入文件名 " + proxy + "，可选: " + string.Join(", ", Const.ProxyNames));
            log(mode == Modes.Fallback ? "注入方式: 兜底模式 (dxgi.dll 代理)" : "注入方式: OptiScaler.dll -> " + proxy);

            var m = new Manifest { proxy = proxy, mode = mode, gpu = Gpu.Name, date = DateTime.Now.ToString("s") };
            string backup = Path.Combine(dir, Const.BackupDir);
            string payload = Const.Payload.TrimEnd('\\');
            string fbPrefix = Const.FallbackDir + "\\";
            IEnumerable<string> sources = mode == Modes.Fallback
                ? FallbackFiles.Select(f => Path.Combine(payload, f))
                : Directory.GetFiles(payload, "*", SearchOption.AllDirectories).Where(f => !f.Substring(payload.Length + 1).StartsWith(fbPrefix, StringComparison.OrdinalIgnoreCase));
            foreach (string src in sources)
            {
                string rel = src.Substring(payload.Length + 1);
                if (rel.StartsWith(fbPrefix, StringComparison.OrdinalIgnoreCase)) rel = rel.Substring(fbPrefix.Length);
                else if (rel.Equals("OptiScaler.dll", StringComparison.OrdinalIgnoreCase)) rel = proxy;
                string dst = Path.Combine(dir, rel);
                if (File.Exists(dst))
                {
                    string bak = Path.Combine(backup, rel);
                    Directory.CreateDirectory(Path.GetDirectoryName(bak));
                    if (File.Exists(bak)) File.Delete(bak);
                    File.Move(dst, bak);
                    m.backups.Add(rel);
                }
                Directory.CreateDirectory(Path.GetDirectoryName(dst));
                File.Copy(src, dst, true);
                DeleteFileW(dst + ":Zone.Identifier");   // 去掉"来自网络"标记 (等同 Unblock-File)
                m.installed.Add(rel);
            }
            if (mode == Modes.Fallback) m.installed.AddRange(new[] { "dlss5fb.ini", "dlss5fb.log" });   // 运行时生成，卸载时一并删掉
            m.Save(dir);   // 先写清单: 后面出错也能卸载干净

            var settings = new Dictionary<string, string> { { "Enabled", "true" } };
            if (mode == Modes.OptiScaler) settings["AutoCapture"] = "false";
            if (keep != null) foreach (var kv in keep) if (Settings.Find(kv.Key) != null) settings[kv.Key] = kv.Value;
            foreach (var kv in dlssnr) settings[kv.Key] = kv.Value;
            ApplySettings(dir, settings, p, mode);
            double sc;
            string scs;
            if (settings.TryGetValue("WorkingScale", out scs) && double.TryParse(scs, NumberStyles.Float, Const.Inv, out sc)) { m.scale = sc; m.Save(dir); }
            log(string.Format("安装完成: {0} 个文件，备份了 {1} 个被覆盖的文件到 {2}", m.installed.Count, m.backups.Count, Const.BackupDir));
        }

        public static void Uninstall(string exeDir, Action<string> log)
        {
            Manifest m = Manifest.Load(exeDir);
            if (m == null) throw new InvalidOperationException("这个目录里没有本工具的安装记录: " + exeDir);
            foreach (string rel in m.installed)
            {
                string f = Path.Combine(exeDir, rel);
                if (File.Exists(f)) File.Delete(f);
            }
            string backup = Path.Combine(exeDir, Const.BackupDir);
            foreach (string rel in m.backups)
            {
                string src = Path.Combine(backup, rel), dst = Path.Combine(exeDir, rel);
                if (!File.Exists(src)) continue;
                if (File.Exists(dst)) File.Delete(dst);
                File.Move(src, dst);
            }
            // 清掉安装时建的、现在已空的子目录 (含各级父目录，深的先删)
            var dirs = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (string rel in m.installed)
                for (string d = Path.GetDirectoryName(rel); !string.IsNullOrEmpty(d); d = Path.GetDirectoryName(d)) dirs.Add(d);
            foreach (string d in dirs.OrderByDescending(d => d.Length))
            {
                string full = Path.Combine(exeDir, d);
                try { if (Directory.Exists(full) && !Directory.EnumerateFileSystemEntries(full).Any()) Directory.Delete(full); } catch { }
            }
            try { if (Directory.Exists(backup) && !Directory.EnumerateFiles(backup, "*", SearchOption.AllDirectories).Any()) Directory.Delete(backup, true); } catch { }
            File.Delete(Manifest.PathIn(exeDir));
            log(string.Format("已卸载，还原了 {0} 个被覆盖的文件", m.backups.Count));
        }
    }
}
