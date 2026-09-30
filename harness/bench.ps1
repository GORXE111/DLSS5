# sm_86 移植的整链路回归 + 计时。
#   .\bench.ps1 -Variant rt_sm86 -Dll ..\sm86_port\nvngx_dlssnr.dll
# 输出: 360p 校验和 (4 帧) / intensity=0 校验和 / 720p、1080p 每帧耗时
param(
    [string]$Variant = 'rt_sm86',
    [string]$Dll = (Join-Path $PSScriptRoot '..\sm86_port\nvngx_dlssnr.dll'),
    [string[]]$Sizes = @('1280x720', '1920x1080'),
    [int]$Frames = 24
)
$dir = Join-Path $PSScriptRoot $Variant
New-Item -ItemType Directory -Force $dir | Out-Null
foreach ($f in 'nr-lab.exe', 'nvngx.dll') { Copy-Item (Join-Path $PSScriptRoot "rt_sm86\$f") $dir -Force -ErrorAction SilentlyContinue }
foreach ($f in 'nvngx_dlss.dll', 'nvngx_dlssg.dll') { Copy-Item (Join-Path $PSScriptRoot "..\dlss5\$f") $dir -Force }
Copy-Item $Dll (Join-Path $dir 'nvngx_dlssnr.dll') -Force
Set-Location $dir

function Invoke-Lab([string[]]$LabArgs, [string]$Log) {
    $p = Start-Process -FilePath .\nr-lab.exe -ArgumentList $LabArgs -NoNewWindow -PassThru `
        -RedirectStandardOutput $Log -RedirectStandardError 'err.txt'
    if (-not $p.WaitForExit(600000)) { $p.Kill(); throw "nr-lab 超时: $LabArgs" }
    if (Select-String -Path $Log -Pattern 'sm86-crash' -Quiet) { Get-Content $Log | Select-String 'sm86-crash' -Context 0, 6 }
    Get-Content nr-lab-result.json -Raw | ConvertFrom-Json
}

$base = @('--nr-only', '--input', '640x360', '--output', '640x360', '--frames', '4')
$r = Invoke-Lab $base 'o_check.txt'
Copy-Item nr-lab-output-model1-srgb.ppm out_check.ppm -Force
$z = Invoke-Lab ($base + @('--intensity', '0')) 'o_int0.txt'
Copy-Item nr-lab-output-model1-srgb.ppm out_int0.ppm -Force
"{0}: 360p checksum={1} evals={2}  intensity0 checksum={3}" -f $Variant, $r.checksumFnv1a64, $r.evaluationsSucceeded, $z.checksumFnv1a64

foreach ($res in $Sizes) {
    $null = Invoke-Lab @('--nr-only', '--input', $res, '--output', $res, '--frames', "$Frames") 'o_perf.txt'
    $t = Get-Content o_perf.txt | Select-String 'frame (\d+) EvaluateFeature = 0x00000001' |
        ForEach-Object { [datetime]::ParseExact($_.Line.Substring(0, 12), 'HH:mm:ss.fff', $null) }
    $dt = for ($i = 5; $i -lt $t.Count; $i++) { ($t[$i] - $t[$i - 1]).TotalMilliseconds }
    $m = $dt | Measure-Object -Average -Minimum
    "{0}: {1,-10} frames={2} avg={3:N1} ms  min={4:N1} ms" -f $Variant, $res, $t.Count, $m.Average, $m.Minimum
}
