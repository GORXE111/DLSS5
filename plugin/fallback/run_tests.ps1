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
foreach ($f in 'nvngx_dlssnr.dll', 'pattern.ppm', 'sponza640.ppm') {
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
    @{ name = 'rgba8';        args = @();                            ini = "WorkingScale=1.0`nTemporal=0`nDumpFrame=5"; dump = $true }
    @{ name = 'bgra8';        args = @('--format', 'bgra8');         ini = "WorkingScale=1.0`nTemporal=0`nDumpFrame=5"; dump = $true }
    @{ name = 'rgb10';        args = @('--format', 'rgb10');         ini = "WorkingScale=1.0`nTemporal=0`nDumpFrame=5"; dump = $true }
    @{ name = 'rgba16f (= scRGB HDR)'; args = @('--format', 'rgba16f'); ini = ''; expect = @('picture: scRGB', 'frame 1 processed') }
    @{ name = 'scRGB HDR, highlights x4'; args = @('--hdr', 'scrgb', '--hdr-gain', '4'); ini = ''; expect = @('colour space 1', 'picture: scRGB', 'frame 1 processed') }
    @{ name = 'HDR10 (PQ, Rec.2020)'; args = @('--hdr', 'hdr10'); ini = ''; expect = @('colour space 12', 'picture: HDR10', 'frame 1 processed') }
    @{ name = 'Present1';     args = @('--present1');                ini = ''; expect = @('frame 1 processed') }
    @{ name = 'ResizeBuffers';  args = @('--resize', '100');         ini = ''; expect = @('NR 320x180', 'NR 240x135') }
    @{ name = 'ResizeBuffers1'; args = @('--resize1', '100');        ini = ''; expect = @('NR 320x180', 'NR 240x135') }
    @{ name = 'recreate';     args = @('--recreate', '100');         ini = ''; expect = @('released', 'tracked') ; tracked = 2 }
    @{ name = 'waitable';     args = @('--waitable');                ini = ''; expect = @('frame 1 processed') }
    @{ name = 'two chains';   args = @('--chains', '2');             ini = ''; expect = @('frame 1 processed'); tracked = 2 }
    @{ name = 'scale 0.5';    args = @();                            ini = 'WorkingScale=0.5'; expect = @('NR 320x180') }
    @{ name = 'optical flow (moving)'; args = @('--pan', '4');       ini = 'WorkingScale=0.5'; expect = @('optical flow: 320x180') }
    @{ name = 'scene cut';    args = @('--cut', '100', 'sponza640.ppm'); ini = 'WorkingScale=0.5'; expect = @('scene cut: frame 101') }
    @{ name = 'exact MV = nr-lab mvok'; args = @('--pan', '4');      ini = "WorkingScale=1.0`nStabilize=0`nSmooth=1`nMvConstX=-4`nDumpFrame=1`nDumpCount=4"; mvok = $true }
    # D3D11 "games" (fbtest11): processed on our own D3D12 device through a shared texture. rb = the back buffer read
    # back on the D3D11 side after the last Present must equal what the proxy wrote (blt model keeps its contents).
    @{ name = 'D3D11 blt + readback';    exe = 'fbtest11.exe'; args = @('--model', 'blt', '--readback', 'rb11.ppm'); ini = "WorkingScale=1.0`nTemporal=0`nDumpFrame=$Frames"; dump = $true; rb = $true; expect = @('tracked (D3D11 device') }
    @{ name = 'D3D11 legacy create';     exe = 'fbtest11.exe'; args = @('--legacy', '--readback', 'rb11.ppm');       ini = "WorkingScale=1.0`nTemporal=0`nDumpFrame=$Frames"; dump = $true; rb = $true; expect = @('tracked (D3D11 device') }
    @{ name = 'D3D11 flip';              exe = 'fbtest11.exe'; args = @();                                ini = "WorkingScale=1.0`nTemporal=0`nDumpFrame=5"; dump = $true; expect = @('tracked (D3D11 device') }
    @{ name = 'D3D11 bgra8';             exe = 'fbtest11.exe'; args = @('--format', 'bgra8', '--model', 'blt', '--readback', 'rb11.ppm'); ini = "WorkingScale=1.0`nTemporal=0`nDumpFrame=$Frames"; dump = $true; rb = $true }
    @{ name = 'D3D11 rgba8 sRGB (blt)';  exe = 'fbtest11.exe'; args = @('--format', 'rgba8srgb', '--model', 'blt'); ini = ''; expect = @('shared texture 640x360 fmt=29', 'frame 1 processed') }
    @{ name = 'D3D11 rgb10';             exe = 'fbtest11.exe'; args = @('--format', 'rgb10');             ini = "WorkingScale=1.0`nTemporal=0`nDumpFrame=5"; dump = $true }
    @{ name = 'D3D11 ResizeBuffers';     exe = 'fbtest11.exe'; args = @('--resize', '100');               ini = ''; expect = @('NR 320x180', 'NR 240x135', 'shared texture 480x270') }
    @{ name = 'D3D11 recreate';          exe = 'fbtest11.exe'; args = @('--recreate', '100');             ini = ''; expect = @('released', 'tracked'); tracked = 2 }
    @{ name = 'D3D11 optical flow';      exe = 'fbtest11.exe'; args = @('--pan', '4');                    ini = 'WorkingScale=0.5'; expect = @('optical flow: 320x180') }
)
if ($Fullscreen) { $cases += @{ name = 'fullscreen'; args = @('--fullscreen', '60'); ini = ''; expect = @('frame 1 processed') } }

$pass = 0; $fail = 0
Push-Location $test
try {
    foreach ($c in $cases) {
        Remove-Item dlss5fb_*.ppm, dlss5fb_*.f16, dlss5fb_*.pq, dlss5fb.log, rb11.ppm -ErrorAction SilentlyContinue
        $ini = if ($c.ini) { $c.ini } else { 'WorkingScale=0.5' }
        "[DLSS5]`n$ini" | Set-Content -Encoding ascii dlss5fb.ini
        $exe = if ($c.exe) { $c.exe } else { 'fbtest.exe' }
        $out = & ".\$exe" pattern.ppm $Frames @($c.args) 2>&1 | Out-String
        $code = $LASTEXITCODE
        $log = if (Test-Path dlss5fb.log) { Get-Content dlss5fb.log -Raw } else { '' }
        $why = @()
        if ($code -ne 0 -or $out -notmatch 'OK') { $why += "exit $code" }
        if ($log -match 'FAIL') { $why += 'FAIL in log' }
        foreach ($e in @($c.expect)) { if ($e -and $log -notmatch [regex]::Escape($e)) { $why += "log lacks '$e'" } }
        if ($c.tracked -and ([regex]::Matches($log, 'tracked')).Count -lt $c.tracked) { $why += "expected $($c.tracked) tracked swap chains" }
        if ($c.mvok -and (Test-Path $ref)) {
            foreach ($k in 1, 2, 3) {
                $mv = Join-Path (Split-Path $ref) "out_mvok_f$k.ppm"
                if (-not (Same-Pixels (Join-Path $test "dlss5fb_nr_$k.ppm") $mv)) { $why += "frame $k differs from nr-lab (correct MV)" }
            }
        }
        if ($c.dump) {
            if (-not (Test-Path dlss5fb_nr.ppm)) { $why += 'no dump' }
            elseif (Test-Path $ref) { if (-not (Same-Pixels (Join-Path $test dlss5fb_nr.ppm) $ref)) { $why += 'NR output differs from nr-lab' } }
            if ((Test-Path dlss5fb_in.ppm) -and -not (Same-Pixels (Join-Path $test dlss5fb_in.ppm) (Join-Path $test pattern.ppm))) { $why += 'captured input differs from the image' }
        }
        if ($c.rb) {
            if (-not (Test-Path rb11.ppm) -or -not (Test-Path dlss5fb_out.ppm)) { $why += 'no read-back / output dump' }
            elseif (-not (Same-Pixels (Join-Path $test rb11.ppm) (Join-Path $test dlss5fb_out.ppm))) { $why += 'D3D11 back buffer differs from the proxy output' }
        }
        $ms = if ($out -match '([\d.]+) ms/frame') { $Matches[1] } else { '?' }
        if ($why.Count -eq 0) { $pass++; Write-Host ("PASS  {0,-30} {1,7} ms/frame" -f $c.name, $ms) -ForegroundColor Green }
        else { $fail++; Write-Host ("FAIL  {0,-30} {1}" -f $c.name, ($why -join '; ')) -ForegroundColor Red; Write-Host $out }
    }
} finally { Pop-Location }
Write-Host "`n$pass passed, $fail failed"
if ($fail) { exit 1 }
