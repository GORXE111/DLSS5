# 兜底模式自测: 用 fbtest 跑各种交换链情况，检查不崩、日志无错误、需要的日志都出现，
# 并在 WorkingScale=1 时把 NR 输出与 nr-lab 的参考 (research/out_f0.ppm，本地文件) 逐字节比较。
#   powershell -ExecutionPolicy Bypass -File plugin\fallback\run_tests.ps1 [-Fullscreen]
# 需要: bin\ 已构建；test\ 里有 nvngx_dlssnr.dll (移植版) 与 pattern.ppm (nr-lab 第 0 帧的合成输入，640x360)
param([switch]$Fullscreen, [int]$Frames = 300)
$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
$test = Join-Path $here 'test'
$ref = Join-Path $here '..\..\research\out_f0.ppm'
Copy-Item (Join-Path $here 'bin\*') $test -Force
foreach ($f in 'nvngx_dlssnr.dll', 'pattern.ppm') {
    if (-not (Test-Path (Join-Path $test $f))) { throw "test\$f 不存在" }
}

function Same-Pixels([string]$a, [string]$b) {
    $x = [IO.File]::ReadAllBytes($a); $y = [IO.File]::ReadAllBytes($b)
    $n = 640 * 360 * 3
    if ($x.Length -lt $n -or $y.Length -lt $n) { return $false }
    $xa = New-Object byte[] $n; $ya = New-Object byte[] $n
    [Array]::Copy($x, $x.Length - $n, $xa, 0, $n); [Array]::Copy($y, $y.Length - $n, $ya, 0, $n)
    return [Linq.Enumerable]::SequenceEqual($xa, $ya)
}

$cases = @(
    @{ name = 'rgba8';        args = @();                            ini = "WorkingScale=1.0`nDumpFrame=5"; dump = $true }
    @{ name = 'bgra8';        args = @('--format', 'bgra8');         ini = "WorkingScale=1.0`nDumpFrame=5"; dump = $true }
    @{ name = 'rgb10';        args = @('--format', 'rgb10');         ini = "WorkingScale=1.0`nDumpFrame=5"; dump = $true }
    @{ name = 'rgba16f (HDR, pass through)'; args = @('--format', 'rgba16f'); ini = ''; expect = @('not supported') ; noProcess = $true }
    @{ name = 'Present1';     args = @('--present1');                ini = ''; expect = @('frame 1 processed') }
    @{ name = 'ResizeBuffers';  args = @('--resize', '100');         ini = ''; expect = @('NR 320x180', 'NR 240x135') }
    @{ name = 'ResizeBuffers1'; args = @('--resize1', '100');        ini = ''; expect = @('NR 320x180', 'NR 240x135') }
    @{ name = 'recreate';     args = @('--recreate', '100');         ini = ''; expect = @('released', 'tracked') ; tracked = 2 }
    @{ name = 'waitable';     args = @('--waitable');                ini = ''; expect = @('frame 1 processed') }
    @{ name = 'two chains';   args = @('--chains', '2');             ini = ''; expect = @('frame 1 processed'); tracked = 2 }
    @{ name = 'scale 0.5';    args = @();                            ini = 'WorkingScale=0.5'; expect = @('NR 320x180') }
)
if ($Fullscreen) { $cases += @{ name = 'fullscreen'; args = @('--fullscreen', '60'); ini = ''; expect = @('frame 1 processed') } }

$pass = 0; $fail = 0
Push-Location $test
try {
    foreach ($c in $cases) {
        Remove-Item dlss5fb_*.ppm, dlss5fb.log -ErrorAction SilentlyContinue
        $ini = if ($c.ini) { $c.ini } else { 'WorkingScale=0.5' }
        "[DLSS5]`n$ini" | Set-Content -Encoding ascii dlss5fb.ini
        $out = & .\fbtest.exe pattern.ppm $Frames @($c.args) 2>&1 | Out-String
        $code = $LASTEXITCODE
        $log = if (Test-Path dlss5fb.log) { Get-Content dlss5fb.log -Raw } else { '' }
        $why = @()
        if ($code -ne 0 -or $out -notmatch 'OK') { $why += "exit $code" }
        if ($log -match 'FAIL') { $why += 'FAIL in log' }
        foreach ($e in @($c.expect)) { if ($e -and $log -notmatch [regex]::Escape($e)) { $why += "log lacks '$e'" } }
        if ($c.tracked -and ([regex]::Matches($log, 'tracked')).Count -lt $c.tracked) { $why += "expected $($c.tracked) tracked swap chains" }
        if ($c.noProcess -and $log -match 'processed') { $why += 'HDR frame was processed' }
        if ($c.dump) {
            if (-not (Test-Path dlss5fb_nr.ppm)) { $why += 'no dump' }
            elseif (Test-Path $ref) { if (-not (Same-Pixels (Join-Path $test dlss5fb_nr.ppm) $ref)) { $why += 'NR output differs from nr-lab' } }
            if ((Test-Path dlss5fb_in.ppm) -and -not (Same-Pixels (Join-Path $test dlss5fb_in.ppm) (Join-Path $test pattern.ppm))) { $why += 'captured input differs from the image' }
        }
        $ms = if ($out -match '([\d.]+) ms/frame') { $Matches[1] } else { '?' }
        if ($why.Count -eq 0) { $pass++; Write-Host ("PASS  {0,-30} {1,7} ms/frame" -f $c.name, $ms) -ForegroundColor Green }
        else { $fail++; Write-Host ("FAIL  {0,-30} {1}" -f $c.name, ($why -join '; ')) -ForegroundColor Red; Write-Host $out }
    }
} finally { Pop-Location }
Write-Host "`n$pass passed, $fail failed"
if ($fail) { exit 1 }
