<#
DLSS5 神经渲染 (Neural Rendering) —— RTX 30/40 移植版 安装器

用法 (任选其一):
  1. 右键本文件 -> "使用 PowerShell 运行"，按提示选择游戏 exe 所在文件夹
  2. powershell -ExecutionPolicy Bypass -File install.ps1 -GameDir "D:\Games\某游戏\bin\x64"

参数:
  -GameDir    游戏 exe 所在文件夹 (Unreal 引擎游戏是 <项目名>\Binaries\Win64)
  -Proxy      OptiScaler 的注入文件名: auto(默认) / dxgi.dll / winmm.dll / version.dll / dbghelp.dll / d3d12.dll / wininet.dll / winhttp.dll
  -Scale      模型分辨率 0.25~1.0 (默认按显卡自动选；游戏内也可随时在菜单里调)
  -Force      跳过反作弊/显卡检查 (仅限你确认游戏离线运行、没有反作弊时)
  -Uninstall  卸载并还原被覆盖的文件
#>
param(
    [string]$GameDir,
    [string]$Proxy = 'auto',
    [double]$Scale = 0,
    [switch]$Force,
    [switch]$Uninstall
)
$ErrorActionPreference = 'Stop'
$Package = 'DLSS5-RTX30'
$payload = Join-Path $PSScriptRoot 'payload'
$ProxyNames = @('dxgi.dll', 'winmm.dll', 'version.dll', 'dbghelp.dll', 'd3d12.dll', 'wininet.dll', 'winhttp.dll')

function Say([string]$msg, [string]$color = 'Gray') { Write-Host $msg -ForegroundColor $color }
function Fail([string]$msg) { Say "`n[停止] $msg" 'Red'; if ($Host.Name -eq 'ConsoleHost') { Read-Host '按回车退出' | Out-Null }; exit 1 }

# ------------------------------------------------------------------ 选目录
if (-not $GameDir) {
    Add-Type -AssemblyName System.Windows.Forms
    $dlg = New-Object System.Windows.Forms.FolderBrowserDialog
    $dlg.Description = '选择游戏 exe 所在的文件夹 (Unreal 引擎游戏选 <项目名>\Binaries\Win64)'
    if ($dlg.ShowDialog() -ne 'OK') { Fail '没有选择文件夹' }
    $GameDir = $dlg.SelectedPath
}
$GameDir = (Resolve-Path -LiteralPath $GameDir).Path
$manifestPath = Join-Path $GameDir "$Package.manifest.json"
$backupDir = Join-Path $GameDir ".dlss5_backup"

# ------------------------------------------------------------------ 卸载
function Invoke-Uninstall {
    if (-not (Test-Path -LiteralPath $manifestPath)) { Fail "这个文件夹里没有 $Package 的安装记录: $GameDir" }
    $m = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($f in $m.installed) {
        $p = Join-Path $GameDir $f
        if (Test-Path -LiteralPath $p) { Remove-Item -LiteralPath $p -Force }
    }
    foreach ($f in $m.backups) {
        $src = Join-Path $backupDir $f
        if (Test-Path -LiteralPath $src) { Move-Item -LiteralPath $src -Destination (Join-Path $GameDir $f) -Force }
    }
    # 清掉安装时新建、现在已空的子目录
    foreach ($d in ($m.installed | ForEach-Object { Split-Path $_ -Parent } | Where-Object { $_ } | Sort-Object -Unique -Descending)) {
        $p = Join-Path $GameDir $d
        if ((Test-Path -LiteralPath $p) -and -not (Get-ChildItem -LiteralPath $p -Force)) { Remove-Item -LiteralPath $p -Force }
    }
    if ((Test-Path -LiteralPath $backupDir) -and -not (Get-ChildItem -LiteralPath $backupDir -Recurse -File -Force)) {
        Remove-Item -LiteralPath $backupDir -Recurse -Force
    }
    Remove-Item -LiteralPath $manifestPath -Force
    Say "已卸载，并还原了 $(@($m.backups).Count) 个被覆盖的文件。" 'Green'
}
if ($Uninstall) { Invoke-Uninstall; exit 0 }

Say "== $Package 安装 ==" 'Cyan'
Say "目标: $GameDir"
if (-not (Get-ChildItem -LiteralPath $GameDir -Filter *.exe -File)) {
    Say '提醒: 这个文件夹里没有 .exe。请确认选的是游戏主程序所在目录。' 'Yellow'
}
if (Test-Path -LiteralPath (Join-Path $GameDir 'Engine')) {
    Say '提醒: 发现 Engine 文件夹，这可能是 Unreal 游戏根目录；应装到 <项目名>\Binaries\Win64。' 'Yellow'
}
if (Test-Path -LiteralPath $manifestPath) {
    Say '检测到已安装过本插件，先卸载旧版本...' 'Yellow'
    Invoke-Uninstall
}

# ------------------------------------------------------------------ 反作弊
# 游戏根目录可能在 exe 目录的上几层 (Unreal: 根\项目\Binaries\Win64)
$roots = @($GameDir)
$p = $GameDir
foreach ($i in 1..3) { $p = Split-Path $p -Parent; if ($p) { $roots += $p } }
$acHits = @()
foreach ($r in $roots) {
    $acHits += Get-ChildItem -LiteralPath $r -Force -ErrorAction SilentlyContinue | Where-Object {
        $_.Name -match '^(EasyAntiCheat|EAC|BattlEye|BEService|xigncode|GameGuard|nProtect|vgk|mhyprot|ACE-|AntiCheatExpert)' -or
        $_.Name -match 'anti.?cheat'
    } | ForEach-Object { $_.FullName }
}
if ($acHits) {
    Say '发现反作弊组件:' 'Red'
    $acHits | Select-Object -Unique | ForEach-Object { Say "  $_" 'Red' }
    if (-not $Force) {
        Fail '带反作弊的游戏注入 dll 可能导致封号，已拒绝安装。确认只离线玩、反作弊已关闭时才可加 -Force。'
    }
    Say '-Force: 已按你的要求继续。' 'Yellow'
}

# ------------------------------------------------------------------ 游戏是否有可接管的超分
$gameRoot = $roots[[Math]::Min(2, $roots.Count - 1)]
$upscalers = Get-ChildItem -LiteralPath $gameRoot -Recurse -Depth 6 -File -ErrorAction SilentlyContinue -Include `
    'nvngx_dlss.dll', 'sl.dlss.dll', 'libxess.dll', 'amd_fidelityfx_dx12.dll', 'ffx_fsr2_api_x64.dll', 'NVUnityPlugin.dll' |
    Select-Object -First 5
if ($upscalers) {
    Say "游戏自带超分: $(($upscalers | ForEach-Object { $_.Name } | Select-Object -Unique) -join ', ')" 'Green'
} else {
    Say '提醒: 没找到 DLSS/FSR/XeSS 文件。DLSS5 需要从游戏的超分调用里拿深度和运动矢量，不支持超分的游戏里它不会生效。' 'Yellow'
}

# ------------------------------------------------------------------ 显卡 -> 默认模型分辨率
$gpu = (Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'NVIDIA' } | Select-Object -First 1).Name
if (-not $gpu) { $gpu = '(未识别)' }
$autoScale = 0
switch -Regex ($gpu) {
    'RTX 50\d\d'          { $autoScale = 1.0; break }
    'RTX 40(80|90)'       { $autoScale = 0.75; break }
    'RTX 40\d\d'          { $autoScale = 0.6; break }
    'RTX 30(80|90)'       { $autoScale = 0.6; break }
    'RTX 30\d\d|RTX A\d+' { $autoScale = 0.5; break }
}
if ($autoScale -eq 0) {
    if (-not $Force) { Fail "显卡 $gpu 不在支持范围 (需要 RTX 30/40/50)。RTX 20 系还没有移植。" }
    $autoScale = 0.5
}
if ($Scale -le 0) { $Scale = $autoScale }
$Scale = [Math]::Max(0.25, [Math]::Min(1.0, $Scale))
Say ("显卡: {0}  ->  模型分辨率 {1:P0}" -f $gpu, $Scale)

# ------------------------------------------------------------------ 注入文件名
function Test-IsOptiScaler([string]$path) {
    try { return (Get-Item -LiteralPath $path).VersionInfo.OriginalFilename -eq 'OptiScaler.dll' } catch { return $false }
}
foreach ($n in $ProxyNames) {
    if (Test-IsOptiScaler (Join-Path $GameDir $n)) {
        Fail "$n 已经是一份 OptiScaler (不是本插件装的)。请先用它自带的卸载脚本删掉，避免两份冲突。"
    }
}
if ($Proxy -eq 'auto') {
    $Proxy = $ProxyNames | Where-Object { -not (Test-Path -LiteralPath (Join-Path $GameDir $_)) } | Select-Object -First 1
    if (-not $Proxy) { Fail '常用的注入文件名都被占用了，请用 -Proxy 指定一个。' }
} elseif ($ProxyNames -notcontains $Proxy.ToLower()) {
    Fail "不支持的注入文件名 $Proxy，可选: $($ProxyNames -join ', ')"
}
Say "注入方式: OptiScaler.dll -> $Proxy"

# ------------------------------------------------------------------ 拷贝 (覆盖前先备份)
$installed = @()
$backups = @()
$files = Get-ChildItem -LiteralPath $payload -Recurse -File
foreach ($f in $files) {
    $rel = $f.FullName.Substring($payload.Length + 1)
    if ($rel -eq 'OptiScaler.dll') { $rel = $Proxy }
    $dst = Join-Path $GameDir $rel
    if (Test-Path -LiteralPath $dst) {
        $bak = Join-Path $backupDir $rel
        New-Item -ItemType Directory -Force -Path (Split-Path $bak -Parent) | Out-Null
        Move-Item -LiteralPath $dst -Destination $bak -Force
        $backups += $rel
    }
    New-Item -ItemType Directory -Force -Path (Split-Path $dst -Parent) | Out-Null
    Copy-Item -LiteralPath $f.FullName -Destination $dst -Force
    Unblock-File -LiteralPath $dst -ErrorAction SilentlyContinue
    $installed += $rel
}

# 按显卡写入模型分辨率 (只改 [DlssNr] 节里的 WorkingScale)
$ini = Join-Path $GameDir 'OptiScaler.ini'
$text = [IO.File]::ReadAllText($ini)
$text = [regex]::Replace($text, '(?ms)(^\[DlssNr\].*?^WorkingScale=)[^\r\n]*', ('${1}' + $Scale.ToString('0.###', [Globalization.CultureInfo]::InvariantCulture)))
[IO.File]::WriteAllText($ini, $text)

$manifest = [ordered]@{
    package   = $Package
    installed = $installed
    backups   = $backups
    proxy     = $Proxy
    scale     = $Scale
    gpu       = $gpu
    date      = (Get-Date).ToString('s')
}
$manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

Say "`n安装完成: $($installed.Count) 个文件，备份 $($backups.Count) 个被覆盖的文件到 .dlss5_backup" 'Green'
Say @"

进游戏后:
  1. 游戏图形设置里打开 DLSS (或 FSR/XeSS)，DLSS5 跟在超分后面运行
  2. 按 Insert 打开 OptiScaler 菜单 -> "DLSS Neural Rendering"
     - 已默认开启；"Model resolution" 越低越快 (耗时约按平方下降)
     - "Detail strength" 调效果强度，0 = 完全关闭效果
  3. 卸载: 再次运行本脚本并加 -Uninstall，或在游戏目录执行
     powershell -ExecutionPolicy Bypass -File "$PSCommandPath" -GameDir "$GameDir" -Uninstall
"@ 'Cyan'
if ($Host.Name -eq 'ConsoleHost' -and -not $PSBoundParameters.ContainsKey('GameDir')) { Read-Host '按回车退出' | Out-Null }
