// 图形界面: 左侧游戏列表，右侧检测结果 + 注入方式 + 参数 + 安装/保存/卸载
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Dlss5Manager
{
    public class MainForm : Form
    {
        readonly ListView list = new ListView();
        readonly TextBox search = new TextBox();
        readonly Label title = new Label();
        readonly TextBox info = new TextBox();
        readonly TextBox log = new TextBox();
        readonly ComboBox preset = new ComboBox();
        readonly ComboBox modeBox = new ComboBox();
        readonly GroupBox group = new GroupBox();
        readonly Dictionary<string, Control[]> rows = new Dictionary<string, Control[]>();
        readonly Label estimate = new Label();
        readonly Button btnInstall = new Button(), btnSave = new Button(), btnUninstall = new Button(), btnOpen = new Button();
        readonly Dictionary<string, Control> inputs = new Dictionary<string, Control>();
        readonly ToolTip tips = new ToolTip { AutoPopDelay = 20000 };
        readonly ToolStripStatusLabel status = new ToolStripStatusLabel();
        readonly int[] screen = Display.Primary();

        List<GameEntry> games = new List<GameEntry>();
        readonly Dictionary<string, Probe> probes = new Dictionary<string, Probe>(StringComparer.OrdinalIgnoreCase);
        GameEntry current;
        Probe probe;

        public MainForm()
        {
            Text = "DLSS5 管理工具 —— 给游戏装上 DLSS5 神经渲染";
            Font = new Font("Microsoft YaHei UI", 9f);
            AutoScaleMode = AutoScaleMode.Dpi;
            Size = new Size(1180, 780);
            MinimumSize = new Size(900, 600);
            StartPosition = FormStartPosition.CenterScreen;

            // ---------------- 左: 工具条 + 列表
            var left = new TableLayoutPanel { Dock = DockStyle.Fill, RowCount = 2, ColumnCount = 1 };
            left.RowStyles.Add(new RowStyle(SizeType.AutoSize));
            left.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
            var bar = new FlowLayoutPanel { Dock = DockStyle.Fill, AutoSize = true, WrapContents = true };
            var btnScan = new Button { Text = "重新扫描", AutoSize = true };
            var btnAdd = new Button { Text = "添加文件夹…", AutoSize = true };
            search.Width = 180;
            SetCue(search, "搜索游戏");
            bar.Controls.AddRange(new Control[] { btnScan, btnAdd, search });
            list.Dock = DockStyle.Fill;
            list.View = View.Details;
            list.FullRowSelect = true;
            list.HideSelection = false;
            list.MultiSelect = false;
            list.Columns.Add("游戏", 230);
            list.Columns.Add("来源", 60);
            list.Columns.Add("状态", 80);
            left.Controls.Add(bar, 0, 0);
            left.Controls.Add(list, 0, 1);

            // ---------------- 右: 标题 / 检测 / 参数 / 按钮 / 日志
            var right = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 5, Padding = new Padding(6, 0, 0, 0) };
            right.RowStyles.Add(new RowStyle(SizeType.AutoSize));
            right.RowStyles.Add(new RowStyle(SizeType.Absolute, 190));
            right.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
            right.RowStyles.Add(new RowStyle(SizeType.AutoSize));
            right.RowStyles.Add(new RowStyle(SizeType.Absolute, 90));
            title.AutoSize = true;
            title.Font = new Font(Font.FontFamily, 13f, FontStyle.Bold);
            title.Margin = new Padding(0, 4, 0, 6);
            title.Text = "选择左侧的游戏";
            info.Multiline = true; info.ReadOnly = true; info.ScrollBars = ScrollBars.Vertical; info.Dock = DockStyle.Fill;
            info.Font = new Font("Consolas", 9.5f);
            info.BackColor = SystemColors.Window;
            log.Multiline = true; log.ReadOnly = true; log.ScrollBars = ScrollBars.Vertical; log.Dock = DockStyle.Fill;
            log.BackColor = SystemColors.Window;

            group.Dock = DockStyle.Fill;
            var grid = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 3, AutoScroll = true };
            grid.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
            grid.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
            grid.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
            preset.DropDownStyle = ComboBoxStyle.DropDownList;
            preset.Items.AddRange(Presets.Names);
            preset.SelectedIndex = 1;
            preset.Width = 110;
            var btnPreset = new Button { Text = "套用预设", AutoSize = true };
            var presetRow = new FlowLayoutPanel { AutoSize = true, WrapContents = false, Margin = new Padding(0) };
            presetRow.Controls.AddRange(new Control[] { preset, btnPreset });
            grid.Controls.Add(new Label { Text = "预设", AutoSize = true, Anchor = AnchorStyles.Left, Margin = new Padding(3, 8, 12, 3) });
            grid.Controls.Add(presetRow);
            estimate.AutoSize = true; estimate.Anchor = AnchorStyles.Left; estimate.ForeColor = SystemColors.GrayText;
            grid.Controls.Add(estimate);
            foreach (var s in Settings.All) AddSettingRow(grid, s);
            group.Controls.Add(grid);

            var buttons = new FlowLayoutPanel { Dock = DockStyle.Fill, AutoSize = true, Margin = new Padding(0, 6, 0, 6) };
            btnInstall.Text = "安装"; btnSave.Text = "保存参数"; btnUninstall.Text = "卸载"; btnOpen.Text = "打开游戏文件夹";
            foreach (var b in new[] { btnInstall, btnSave, btnUninstall, btnOpen }) { b.AutoSize = true; b.Padding = new Padding(10, 2, 10, 2); b.Enabled = false; }
            modeBox.DropDownStyle = ComboBoxStyle.DropDownList;
            modeBox.Items.AddRange(new object[] { "自动", Modes.Label(Modes.OptiScaler), Modes.Label(Modes.Fallback) });
            modeBox.SelectedIndex = 0;
            modeBox.Width = 190;
            tips.SetToolTip(modeBox, "OptiScaler: 借用游戏自带的 DLSS/FSR/XeSS，有深度和运动矢量，效果最好\n" +
                "兜底模式: 没有超分的 DX12 / DX11 游戏，直接截取画面处理 (没有深度，运动矢量靠光流估计；不变的界面会被保护)");
            buttons.Controls.AddRange(new Control[] { new Label { Text = "注入方式", AutoSize = true, Margin = new Padding(3, 8, 3, 3) }, modeBox,
                btnInstall, btnSave, btnUninstall, btnOpen });

            right.Controls.Add(title, 0, 0);
            right.Controls.Add(info, 0, 1);
            right.Controls.Add(group, 0, 2);
            right.Controls.Add(buttons, 0, 3);
            right.Controls.Add(log, 0, 4);

            var split = new SplitContainer { Dock = DockStyle.Fill, FixedPanel = FixedPanel.Panel1 };
            split.Panel1.Controls.Add(left);
            split.Panel2.Controls.Add(right);
            var strip = new StatusStrip();
            strip.Items.Add(status);
            Controls.Add(split);
            Controls.Add(strip);

            // ---------------- 事件
            btnScan.Click += (s, e) => Rescan();
            btnAdd.Click += (s, e) => AddFolder();
            search.TextChanged += (s, e) => Fill();
            list.SelectedIndexChanged += (s, e) => OnSelect();
            modeBox.SelectedIndexChanged += (s, e) => OnMode();
            btnPreset.Click += (s, e) => SetValue("WorkingScale", Presets.Scale((string)preset.SelectedItem, screen[0], screen[1]));
            btnInstall.Click += (s, e) => DoInstall();
            btnSave.Click += (s, e) => DoSave();
            btnUninstall.Click += (s, e) => DoUninstall();
            btnOpen.Click += (s, e) => { if (probe != null) Process.Start("explorer.exe", "\"" + probe.ExeDir + "\""); };
            Load += (s, e) => split.SplitterDistance = (int)(390 * DeviceDpi / 96.0);   // 布局完成后再设，构造时设会被忽略
            Shown += (s, e) => Rescan();

            string pp = Installer.PayloadProblem();
            status.Text = string.Format("显卡: {0}{1}    屏幕: {2}x{3}    安装包: {4}",
                Gpu.Name == "" ? "(未识别)" : Gpu.Name, Gpu.Supported ? "" : " (不支持，需要 RTX 30/40/50)", screen[0], screen[1], pp ?? "完整");
            LoadDefaults();
            ShowRows(Modes.OptiScaler);
        }

        static void SetCue(TextBox t, string cue)
        {
            t.HandleCreated += (s, e) => SendMessage(t.Handle, 0x1501, (IntPtr)1, cue);
        }
        [System.Runtime.InteropServices.DllImport("user32.dll", CharSet = System.Runtime.InteropServices.CharSet.Unicode)]
        static extern IntPtr SendMessage(IntPtr h, int msg, IntPtr w, string l);

        void AddSettingRow(TableLayoutPanel grid, Setting s)
        {
            var label = new Label { Text = s.Label, AutoSize = true, Anchor = AnchorStyles.Left, Margin = new Padding(3, 8, 12, 3) };
            tips.SetToolTip(label, s.Key + "\n" + s.Help);
            Control input;
            if (s.Kind == "bool")
            {
                var c = new CheckBox { AutoSize = true, Anchor = AnchorStyles.Left };
                c.CheckedChanged += (o, e) => OnChanged(s.Key);
                input = c;
                var help = new Label { Text = s.Help, AutoSize = true, Anchor = AnchorStyles.Left, ForeColor = SystemColors.GrayText };
                grid.Controls.Add(label);
                grid.Controls.Add(c);
                grid.Controls.Add(help);
                rows[s.Key] = new Control[] { label, c, help };
            }
            else
            {
                var n = new NumericUpDown
                {
                    Minimum = (decimal)s.Min, Maximum = (decimal)s.Max, Width = 80,
                    DecimalPlaces = s.Kind == "int" ? 0 : 2, Increment = s.Kind == "int" ? 1 : 0.05m, Anchor = AnchorStyles.Left,
                };
                int steps = s.Kind == "int" ? (int)(s.Max - s.Min) : (int)Math.Round((s.Max - s.Min) / 0.05);
                var t = new TrackBar { Minimum = 0, Maximum = steps, TickFrequency = Math.Max(1, steps / 10), Dock = DockStyle.Fill, AutoSize = false, Height = 28 };
                bool sync = false;
                n.ValueChanged += (o, e) =>
                {
                    if (!sync) { sync = true; t.Value = Math.Max(0, Math.Min(steps, (int)Math.Round(((double)n.Value - s.Min) / ((s.Max - s.Min) / Math.Max(1, steps))))); sync = false; }
                    OnChanged(s.Key);
                };
                t.ValueChanged += (o, e) =>
                {
                    if (!sync) { sync = true; n.Value = (decimal)(s.Min + t.Value * (s.Max - s.Min) / Math.Max(1, steps)); sync = false; OnChanged(s.Key); }
                };
                tips.SetToolTip(t, s.Help);
                tips.SetToolTip(n, s.Help);
                input = n;
                grid.Controls.Add(label);
                grid.Controls.Add(n);
                grid.Controls.Add(t);
                rows[s.Key] = new Control[] { label, n, t };
            }
            inputs[s.Key] = input;
        }

        double GetValue(string key)
        {
            Control c = inputs[key];
            var cb = c as CheckBox;
            return cb != null ? (cb.Checked ? 1 : 0) : (double)((NumericUpDown)c).Value;
        }

        void SetValue(string key, double v)
        {
            Control c = inputs[key];
            var cb = c as CheckBox;
            if (cb != null) { cb.Checked = v != 0; return; }
            var n = (NumericUpDown)c;
            n.Value = Math.Max(n.Minimum, Math.Min(n.Maximum, (decimal)v));
        }

        void OnChanged(string key)
        {
            if (key == "WorkingScale" || key == null)
            {
                double s = GetValue("WorkingScale");
                estimate.Text = string.Format("当前模型分辨率 {0}x{1}，DLSS5 本身约 {2:0} ms/帧 (按 {3}x{4} 输出估算)",
                    (int)(screen[0] * s), (int)(screen[1] * s), Presets.EstimateMs(s, screen[0], screen[1]), screen[0], screen[1]);
            }
        }

        void LoadDefaults()
        {

            foreach (var s in Settings.All) SetValue(s.Key, s.Default);
            SetValue("WorkingScale", Presets.Scale("均衡", screen[0], screen[1]));

            OnChanged(null);
        }

        void LoadFromIni(string dir)
        {

            var cur = Installer.ReadSettings(dir);
            foreach (var s in Settings.All)
            {
                string raw;
                cur.TryGetValue(s.Key, out raw);
                SetValue(s.Key, Settings.Parse(s, raw));
            }

            OnChanged(null);
        }

        Dictionary<string, string> Collect()
        {
            return Settings.For(Mode).ToDictionary(s => s.Key, s => Settings.Format(s, GetValue(s.Key)));
        }

        // 下拉框选中的注入方式；"自动" = 已安装的方式或检测建议
        string Mode
        {
            get
            {
                if (modeBox.SelectedIndex == 1) return Modes.OptiScaler;
                if (modeBox.SelectedIndex == 2) return Modes.Fallback;
                return probe != null ? probe.Mode : Modes.OptiScaler;
            }
        }

        void ShowRows(string mode)
        {
            foreach (var s in Settings.All) foreach (Control c in rows[s.Key]) c.Visible = s.AppliesTo(mode);
            group.Text = mode == Modes.Fallback ? "参数 (写入游戏目录的 dlss5fb.ini；游戏运行中修改约 1 秒内生效)"
                                                : "参数 (写入游戏目录的 OptiScaler.ini；进游戏后按 Insert 也能实时调)";
        }

        void OnMode()
        {
            ShowRows(Mode);
            if (probe == null) return;
            info.Text = Cli.Describe(probe, probe.Installed == null ? (modeBox.SelectedIndex == 0 ? null : Mode) : null)
                .Replace("\n", "\r\n").Replace("\r\r\n", "\r\n");
            btnInstall.Enabled = probe.Problems(false, Mode).Count == 0 && Gpu.Supported;
        }

        // ---------------- 列表
        void Rescan()
        {
            status.Text = status.Text.Split(new[] { "    扫描" }, StringSplitOptions.None)[0] + "    扫描游戏库…";
            Task.Run(() => Library.ScanAll()).ContinueWith(t =>
            {
                games = t.Result;
                probes.Clear();
                Fill();
                ProbeAllInBackground(games);
                status.Text = status.Text.Split(new[] { "    扫描" }, StringSplitOptions.None)[0] + "    扫描到 " + games.Count + " 个游戏";
            }, TaskScheduler.FromCurrentSynchronizationContext());
        }

        void Fill()
        {
            string q = search.Text.Trim();
            list.BeginUpdate();
            list.Items.Clear();
            foreach (var g in games.Where(g => q == "" || g.Name.IndexOf(q, StringComparison.CurrentCultureIgnoreCase) >= 0))
            {
                Probe p;
                string state = probes.TryGetValue(g.Root, out p) ? StateOf(p) : "";
                var it = new ListViewItem(new[] { g.Name, g.Source, state }) { Tag = g };
                list.Items.Add(it);
                if (current != null && current.Root == g.Root) it.Selected = true;
            }
            list.EndUpdate();
        }

        // 逐个检测，填"状态"列 (每个游戏通常不到 1 秒)
        void ProbeAllInBackground(List<GameEntry> batch)
        {
            var ui = TaskScheduler.FromCurrentSynchronizationContext();
            Task.Run(() =>
            {
                foreach (var g in batch)
                {
                    if (games != batch) return;   // 又重新扫描了
                    Probe p;
                    try { p = Probe.Run(g.Root, g.HintExe); } catch { continue; }
                    Task.Factory.StartNew(() =>
                    {
                        if (games != batch || probes.ContainsKey(g.Root)) return;
                        probes[g.Root] = p;
                        foreach (ListViewItem it in list.Items) if (it.Tag == g) it.SubItems[2].Text = StateOf(p);
                    }, System.Threading.CancellationToken.None, TaskCreationOptions.None, ui);
                }
            });
        }

        static string StateOf(Probe p)
        {
            if (p.Installed != null) return "已安装";
            if (p.Problems(false).Count > 0) return p.AntiCheat.Count > 0 ? "有反作弊" : "不可安装";
            return p.Is32Bit ? "32 位" : p.Upscalers.Count > 0 ? "可安装" : p.FallbackCapable ? "可兜底" : "无超分?";
        }

        void AddFolder()
        {
            using (var d = new FolderBrowserDialog { Description = "选择游戏文件夹 (游戏根目录或 exe 所在目录都可以)" })
            {
                if (d.ShowDialog(this) != DialogResult.OK) return;
                Library.AddManual(d.SelectedPath);
                Rescan();
            }
        }

        void OnSelect()
        {
            if (list.SelectedItems.Count == 0) return;
            var g = (GameEntry)list.SelectedItems[0].Tag;
            if (current == g && probe != null) return;
            current = g;
            probe = null;
            title.Text = g.Name;
            info.Text = "检测中…";
            foreach (var b in new[] { btnInstall, btnSave, btnUninstall, btnOpen }) b.Enabled = false;
            Probe cached;
            if (probes.TryGetValue(g.Root, out cached)) { ShowProbe(g, cached); return; }
            Task.Run(() => Probe.Run(g.Root, g.HintExe)).ContinueWith(t =>
            {
                if (current != g) return;
                if (t.IsFaulted) { info.Text = "检测失败: " + t.Exception.InnerException.Message; return; }
                ShowProbe(g, t.Result);
            }, TaskScheduler.FromCurrentSynchronizationContext());
        }

        void ShowProbe(GameEntry g, Probe p)
        {
            probe = p;
            probes[g.Root] = p;
            foreach (ListViewItem it in list.Items) if (it.Tag == g) it.SubItems[2].Text = StateOf(p);
            bool installed = p.Installed != null;
            btnInstall.Text = installed ? "重新安装" : "安装";
            if (modeBox.SelectedIndex != 0) modeBox.SelectedIndex = 0; else OnMode();   // 都会走 OnMode
            btnSave.Enabled = installed;
            btnUninstall.Enabled = installed;
            btnOpen.Enabled = true;
            if (installed) LoadFromIni(p.ExeDir); else LoadDefaults();
        }

        void Log(string s) { log.AppendText(DateTime.Now.ToString("HH:mm:ss  ") + s + "\r\n"); }

        void Act(string what, Action a)
        {
            if (probe == null || current == null) return;
            Cursor = Cursors.WaitCursor;
            try { a(); }
            catch (Exception e) when (e is InvalidOperationException || e is IOException || e is UnauthorizedAccessException || e is ArgumentException)
            {
                Log(what + "失败: " + e.Message);
                MessageBox.Show(this, e.Message, what + "失败", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
            finally { Cursor = Cursors.Default; }
            var g = current;
            ShowProbe(g, Probe.Run(g.Root, g.HintExe));
        }

        void DoInstall()
        {
            string mode = Mode;
            var warn = probe.Warnings(mode).Where(w => !w.StartsWith("反作弊")).ToList();
            if (warn.Count > 0 && MessageBox.Show(this, string.Join("\n\n", warn) + "\n\n继续安装？", "提醒", MessageBoxButtons.OKCancel, MessageBoxIcon.Information) != DialogResult.OK)
                return;
            Act("安装", () =>
            {
                Log("安装到 " + probe.ExeDir);
                Installer.Install(probe, Collect(), "auto", false, Log, mode);
                Log(Cli.Hint(mode));
            });
        }

        void DoSave()
        {
            Act("保存", () =>
            {
                Installer.ApplySettings(probe.ExeDir, Collect(), probe);
                Log("参数已写入 " + Installer.IniPath(probe.ExeDir) + (probe.Mode == Modes.Fallback ? " (游戏运行中约 1 秒内生效)"
                    : " (游戏运行中需重启游戏生效，或在 OptiScaler 菜单里调)"));
            });
        }

        void DoUninstall()
        {
            if (MessageBox.Show(this, "卸载并还原被覆盖的文件？", "卸载", MessageBoxButtons.OKCancel) != DialogResult.OK) return;
            Act("卸载", () => Installer.Uninstall(probe.ExeDir, Log));
        }
    }
}
