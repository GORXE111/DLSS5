# 兜底模式自测: 用 fbtest 跑各种交换链情况，检查不崩、日志无错误、需要的日志都出现，
# 并在 WorkingScale=1 时把 NR 输出与 nr-lab 的参考 (research/out_f0.ppm，本地文件) 逐字节比较。
#   powershell -ExecutionPolicy Bypass -File plugin\fallback\run_tests.ps1 [-Fullscreen]
# 需要: bin\ 已构建；test\ 里有 nvngx_dlssnr.dll (移植版) 与 pattern.ppm (nr-lab 第 0 帧的合成输入，640x360)
param([switch]$Fullscreen, [ValidateRange(151, 10000)][int]$Frames = 300,
      [string]$ReferenceDirectory = (Join-Path $PSScriptRoot '..\..\research'),
      [string]$Results = (Join-Path $PSScriptRoot ('test\results\' + (Get-Date -Format 'yyyyMMdd-HHmmss'))),
      [string]$Python = (Join-Path $PSScriptRoot '..\..\.venv-compile\Scripts\python.exe'))
$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
$test = Join-Path $here 'test'
$ref = Join-Path $ReferenceDirectory 'out_f0.ppm'
# Required evidence is a precondition: absence must never silently turn a comparison into PASS.
foreach ($f in 'out_f0.ppm', 'out_mvok_f1.ppm', 'out_mvok_f2.ppm', 'out_mvok_f3.ppm') {
    if (-not (Test-Path -LiteralPath (Join-Path $ReferenceDirectory $f) -PathType Leaf)) { throw "Missing required reference: $f" }
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "HDR validator Python not found: $Python" }
foreach ($f in 'nvngx_dlssnr.dll', 'pattern.ppm', 'sponza640.ppm') {
    if (-not (Test-Path (Join-Path $test $f))) { throw "test\$f 不存在" }
}
Copy-Item (Join-Path $here 'bin\*') $test -Force
New-Item -ItemType Directory -Path $Results -Force | Out-Null
$Results = (Resolve-Path -LiteralPath $Results).Path

function Same-Pixels([string]$a, [string]$b) {
    if (-not (Test-Path -LiteralPath $a -PathType Leaf) -or -not (Test-Path -LiteralPath $b -PathType Leaf)) { return $false }
    $x = [IO.File]::ReadAllBytes($a); $y = [IO.File]::ReadAllBytes($b)
    $n = 640 * 360 * 3
    if ($x.Length -lt $n -or $y.Length -lt $n) { return $false }
    foreach ($bytes in @($x, $y)) {
        $header = [Text.Encoding]::ASCII.GetString($bytes, 0, [Math]::Min(256, $bytes.Length))
        $match = [regex]::Match($header, '\AP6\s+640\s+360\s+255(?:\r\n|\n| |\t)')
        if (-not $match.Success -or $bytes.Length -ne $match.Length + $n) { return $false }
    }
    $xa = New-Object byte[] $n; $ya = New-Object byte[] $n
    [Array]::Copy($x, $x.Length - $n, $xa, 0, $n); [Array]::Copy($y, $y.Length - $n, $ya, 0, $n)
    return [Linq.Enumerable]::SequenceEqual($xa, $ya)
}

# mean and max absolute difference of two 640x360 PPMs inside [x0,x1) x [y0,y1)
function Region-Diff([string]$a, [string]$b, [int]$x0, [int]$y0, [int]$x1, [int]$y1) {
    $x = [IO.File]::ReadAllBytes($a); $y = [IO.File]::ReadAllBytes($b)
    $n = 640 * 360 * 3; $ox = $x.Length - $n; $oy = $y.Length - $n
    $sum = 0; $max = 0; $cnt = 0
    for ($r = $y0; $r -lt $y1; $r++) {
        for ($i = ($r * 640 + $x0) * 3; $i -lt ($r * 640 + $x1) * 3; $i++) {
            $d = [Math]::Abs([int]$x[$ox + $i] - [int]$y[$oy + $i]); $sum += $d; $cnt++
            if ($d -gt $max) { $max = $d }
        }
    }
    return @{ mean = [Math]::Round($sum / $cnt, 2); max = $max }
}

$cases = @(
    @{ name = 'rgba8';        args = @();                            ini = "WorkingScale=1.0`nTemporal=0`nDumpFrame=5"; dump = $true }
    @{ name = 'bgra8';        args = @('--format', 'bgra8');         ini = "WorkingScale=1.0`nTemporal=0`nDumpFrame=5"; dump = $true }
    @{ name = 'rgb10';        args = @('--format', 'rgb10');         ini = "WorkingScale=1.0`nTemporal=0`nDumpFrame=5"; dump = $true }
    @{ name = 'rgba16f (= scRGB HDR)'; args = @('--format', 'rgba16f'); ini = 'DumpFrame=5'; hdr = 'f16'; highlights = 60; expect = @('picture: scRGB', 'frame 1 processed') }
    @{ name = 'scRGB HDR, highlights x4'; args = @('--hdr', 'scrgb', '--hdr-gain', '4'); ini = 'DumpFrame=5'; hdr = 'f16'; highlights = 400; expect = @('colour space 1', 'picture: scRGB', 'frame 1 processed') }
    @{ name = 'HDR10 (PQ, Rec.2020)'; args = @('--hdr', 'hdr10'); ini = 'DumpFrame=5'; hdr = 'pq'; highlights = 120; expect = @('colour space 12', 'picture: HDR10', 'frame 1 processed') }
    @{ name = 'HDR10 highlights x4'; args = @('--hdr', 'hdr10', '--hdr-gain', '4'); ini = 'DumpFrame=5'; hdr = 'pq'; highlights = 400; expect = @('picture: HDR10', 'frame 1 processed') }
    @{ name = 'Present1';     args = @('--present1');                ini = ''; expect = @('frame 1 processed') }
    @{ name = 'ResizeBuffers';  args = @('--resize', '100');         ini = ''; expect = @('NR 320x180', 'NR 240x135') }
    @{ name = 'ResizeBuffers1'; args = @('--resize1', '100');        ini = ''; expect = @('NR 320x180', 'NR 240x135') }
    @{ name = 'recreate';     args = @('--recreate', '100');         ini = ''; expect = @('released', 'tracked') ; tracked = 2 }
    @{ name = 'waitable';     args = @('--waitable');                ini = ''; expect = @('frame 1 processed') }
    @{ name = 'two chains';   args = @('--chains', '2');             ini = ''; expect = @('frame 1 processed'); tracked = 2 }
    @{ name = 'parallel Present two threads'; args = @('--parallel-present'); ini = ''; expect = @('frame 1 processed'); tracked = 2; processed = 2 }
    @{ name = 'scale 0.5';    args = @();                            ini = 'WorkingScale=0.5'; expect = @('NR 320x180') }
    @{ name = 'optical flow (moving)'; args = @('--pan', '4');       ini = 'WorkingScale=0.5'; expect = @('optical flow: 320x180') }
    @{ name = 'scene cut';    args = @('--cut', '100', 'sponza640.ppm'); ini = 'WorkingScale=0.5'; expect = @('scene cut: frame 101') }
    @{ name = 'exact MV = nr-lab mvok'; args = @('--pan', '4');      ini = "WorkingScale=1.0`nStabilize=0`nSmooth=1`nMvConstX=-4`nDumpFrame=1`nDumpCount=4"; mvok = $true }
    # HUD protection: a static rectangle over a panning picture must come out untouched, the picture around it changed
    @{ name = 'UI protection';           args = @('--pan', '4', '--ui'); ini = "WorkingScale=0.5`nDumpFrame=150"; ui = $true }
    @{ name = 'UI protection off';       args = @('--pan', '4', '--ui'); ini = "WorkingScale=0.5`nUiProtect=0`nDumpFrame=150"; ui = $false; uioff = $true }
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

$pass = 0; $fail = 0; $evidence = @(); $caseIndex = 0
Push-Location $test
try {
    foreach ($c in $cases) {
        $caseIndex++
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
        if ($c.processed -and ([regex]::Matches($log, 'frame 1 processed')).Count -lt $c.processed) { $why += "expected $($c.processed) chains to process frames" }
        if ($c.mvok) {
            foreach ($k in 1, 2, 3) {
                $mv = Join-Path (Split-Path $ref) "out_mvok_f$k.ppm"
                $dump = Join-Path $test "dlss5fb_nr_$k.ppm"
                if (-not (Test-Path -LiteralPath $dump)) { $why += "missing frame $k NR dump" }
                elseif (-not (Same-Pixels $dump $mv)) { $why += "frame $k differs from nr-lab (correct MV)" }
            }
        }
        if ($c.dump) {
            if (-not (Test-Path dlss5fb_nr.ppm)) { $why += 'no dump' }
            elseif (-not (Same-Pixels (Join-Path $test dlss5fb_nr.ppm) $ref)) { $why += 'NR output differs from nr-lab' }
            if (-not (Test-Path dlss5fb_in.ppm)) { $why += 'missing captured input' }
            elseif (-not (Same-Pixels (Join-Path $test dlss5fb_in.ppm) (Join-Path $test pattern.ppm))) { $why += 'captured input differs from the image' }
        }
        $hdrMetrics = $null
        if ($c.hdr) {
            $hdrText = & $Python (Join-Path $here 'check_hdr.py') $test --encoding $c.hdr --highlight-nits $c.highlights 2>&1 | Out-String
            if ($LASTEXITCODE -ne 0) { $why += "HDR numerical check failed: $hdrText" }
            else { $hdrMetrics = $hdrText | ConvertFrom-Json }
        }
        if ($c.ui -or $c.uioff) {
            if (-not (Test-Path dlss5fb_in.ppm) -or -not (Test-Path dlss5fb_out.ppm)) { $why += 'no dump' }
            else {
                # fbtest --ui: HUD at x 80..239, y 45..104 of the 640x360 picture
                $d = Region-Diff (Join-Path $test dlss5fb_in.ppm) (Join-Path $test dlss5fb_out.ppm) 80 45 240 105
                $o = Region-Diff (Join-Path $test dlss5fb_in.ppm) (Join-Path $test dlss5fb_out.ppm) 300 150 460 210
                if ($c.ui -and $d.max -ne 0) { $why += "HUD changed (max $($d.max))" }
                if ($c.uioff -and $d.mean -lt 2) { $why += "HUD not processed with UiProtect=0 (mean $($d.mean))" }
                if ($o.mean -lt 2) { $why += "picture outside the HUD not processed (mean $($o.mean))" }
            }
        }
        if ($c.rb) {
            if (-not (Test-Path rb11.ppm) -or -not (Test-Path dlss5fb_out.ppm)) { $why += 'no read-back / output dump' }
            elseif (-not (Same-Pixels (Join-Path $test rb11.ppm) (Join-Path $test dlss5fb_out.ppm))) { $why += 'D3D11 back buffer differs from the proxy output' }
        }
        $ms = if ($out -match '([\d.]+) ms/frame') { $Matches[1] } else { '?' }
        $caseDir = Join-Path $Results ('{0:D2}' -f $caseIndex)
        New-Item -ItemType Directory -Path $caseDir | Out-Null
        $out | Set-Content -LiteralPath (Join-Path $caseDir 'stdout.txt') -Encoding UTF8
        $log | Set-Content -LiteralPath (Join-Path $caseDir 'dlss5fb.log') -Encoding UTF8
        if ($c.hdr) { Get-ChildItem (Join-Path $test ('dlss5fb_*.' + $c.hdr)) | Copy-Item -Destination $caseDir }
        $evidence += @{ name = $c.name; passed = ($why.Count -eq 0); reasons = @($why); ms_per_frame = $ms; hdr = $hdrMetrics }
        if ($why.Count -eq 0) { $pass++; Write-Host ("PASS  {0,-30} {1,7} ms/frame" -f $c.name, $ms) -ForegroundColor Green }
        else { $fail++; Write-Host ("FAIL  {0,-30} {1}" -f $c.name, ($why -join '; ')) -ForegroundColor Red; Write-Host $out }
    }
} finally { Pop-Location }
Write-Host "`n$pass passed, $fail failed"
@{ passed = $pass; failed = $fail; frames = $Frames; reference = (Get-FileHash -LiteralPath $ref -Algorithm SHA256).Hash; cases = $evidence } | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $Results 'summary.json') -Encoding UTF8
Write-Host "Evidence: $Results"
if ($fail) { exit 1 }
