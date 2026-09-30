# 安装器自测: 在临时假游戏目录里跑安装/卸载，检查备份还原逐字节一致、反作弊与重复 OptiScaler 会被拒绝。
param([string]$Work = (Join-Path $env:TEMP 'dlss5_install_test'))
$ErrorActionPreference = 'Stop'
$pkg = Join-Path $PSScriptRoot 'dist\DLSS5-RTX30'
$installer = Join-Path $pkg 'install.ps1'
function New-FakeGame([string]$dir) {
    if (Test-Path $dir) { Remove-Item $dir -Recurse -Force }
    New-Item -ItemType Directory -Force $dir | Out-Null
    [IO.File]::WriteAllBytes((Join-Path $dir 'Game.exe'), [byte[]](0x4D, 0x5A))
    [IO.File]::WriteAllText((Join-Path $dir 'dxgi.dll'), 'pretend-reshade')
    [IO.File]::WriteAllText((Join-Path $dir 'OptiScaler.ini'), 'user-own-ini')
    [IO.File]::WriteAllText((Join-Path $dir 'nvngx_dlss.dll'), 'pretend-dlss')
}
function Get-Snapshot([string]$dir) {
    Get-ChildItem $dir -Recurse -File -Force | ForEach-Object {
        '{0}|{1}' -f $_.FullName.Substring($dir.Length), (Get-FileHash $_.FullName -Algorithm SHA1).Hash
    } | Sort-Object
}
function Invoke-Installer([string[]]$InstallerArgs) {
    $out = & powershell -NoProfile -ExecutionPolicy Bypass -File $installer @InstallerArgs 2>&1 | Out-String
    return @{ code = $LASTEXITCODE; out = $out }
}
$pass = 0; $fail = 0
function Check([bool]$ok, [string]$what) {
    if ($ok) { $script:pass++; Write-Host "PASS  $what" -ForegroundColor Green } else { $script:fail++; Write-Host "FAIL  $what" -ForegroundColor Red }
}

# 1. 正常安装 + 卸载还原
$g = Join-Path $Work 'game1'
New-FakeGame $g
$before = Get-Snapshot $g
$r = Invoke-Installer @('-GameDir', $g)
Check ($r.code -eq 0) 'install exit code 0'
Check (Test-Path (Join-Path $g 'winmm.dll')) 'dxgi.dll occupied -> proxy winmm.dll'
Check ((Get-Item (Join-Path $g 'winmm.dll')).VersionInfo.OriginalFilename -eq 'OptiScaler.dll') 'winmm.dll is OptiScaler'
Check ((Get-Content (Join-Path $g 'dxgi.dll') -Raw) -eq 'pretend-reshade') 'existing dxgi.dll untouched'
Check ((Get-Content (Join-Path $g '.dlss5_backup\OptiScaler.ini') -Raw) -eq 'user-own-ini') 'overwritten ini backed up'
Check ((Get-Item (Join-Path $g 'nvngx_dlssnr.dll')).Length -gt 100MB) 'model dll installed'
$ini = Get-Content (Join-Path $g 'OptiScaler.ini') -Raw
Check ($ini -match '(?m)^\[DlssNr\][\s\S]*?^WorkingScale=0\.5\r?$') 'WorkingScale set by GPU (3060 -> 0.5)'
Check ($ini -match '(?m)^\[DlssNr\][\s\S]*?^Enabled=true\r?$') 'NR enabled by default'
$m = Get-Content (Join-Path $g 'DLSS5-RTX30.manifest.json') -Raw | ConvertFrom-Json
Check ($m.proxy -eq 'winmm.dll' -and @($m.backups).Count -eq 1) 'manifest recorded'
$r = Invoke-Installer @('-GameDir', $g, '-Scale', '0.33')
Check ($r.code -eq 0 -and (Get-Content (Join-Path $g 'OptiScaler.ini') -Raw) -match '(?m)^WorkingScale=0\.33\r?$') 'reinstall with -Scale 0.33'
$r = Invoke-Installer @('-GameDir', $g, '-Uninstall')
Check ($r.code -eq 0) 'uninstall exit code 0'
$after = Get-Snapshot $g
Check ((Compare-Object $before $after) -eq $null) 'uninstall restores folder byte-for-byte'

# 2. 反作弊拒装
$g2 = Join-Path $Work 'game2'
New-FakeGame $g2
New-Item -ItemType Directory (Join-Path $g2 'EasyAntiCheat') | Out-Null
$before2 = Get-Snapshot $g2
$r = Invoke-Installer @('-GameDir', $g2)
Check ($r.code -ne 0 -and $r.out -match 'EasyAntiCheat') 'anti-cheat -> refused'
Check ((Compare-Object $before2 (Get-Snapshot $g2)) -eq $null) 'refused install leaves folder untouched'

# 3. 已有别的 OptiScaler
$g3 = Join-Path $Work 'game3'
New-FakeGame $g3
Copy-Item (Join-Path $pkg 'payload\OptiScaler.dll') (Join-Path $g3 'version.dll')
$r = Invoke-Installer @('-GameDir', $g3)
Check ($r.code -ne 0 -and $r.out -match 'version.dll') 'foreign OptiScaler -> refused'

Write-Host "`n$pass passed, $fail failed"
Remove-Item $Work -Recurse -Force
if ($fail) { exit 1 }
