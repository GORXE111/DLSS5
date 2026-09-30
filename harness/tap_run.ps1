param(
    [Parameter(Mandatory = $true)][string]$Spec,   # 例: '2;5:p1:1966080;5:p8:983040' (帧号;序号:地址:字节数;...)
    [Parameter(Mandatory = $true)][string]$Dest    # 结果目录 (追加，不清空)
)
# 运行 nr-lab 一次抓取显存，并把 tap_s*.bin 与本次轨迹 (nr-trace.tsv) 移到 $Dest。
# 单次抓取总量不宜超过 ~2MB (过大时整批读回全 0)。
# 注意: Start-Process 不跟随 Set-Location，必须给绝对路径和 -WorkingDirectory。
$rt = Join-Path $PSScriptRoot 'rt_sm86'
Set-Location $rt
Remove-Item -ErrorAction SilentlyContinue (Join-Path $rt 'tap_s*')
$frame = $Spec.Split(';')[0]
$env:DLSS5_PROF = '1'; $env:DLSS5_TRACE = $frame; $env:DLSS5_TAP = $Spec
$p = Start-Process -FilePath (Join-Path $rt 'nr-lab.exe') -WorkingDirectory $rt `
    -ArgumentList '--nr-only', '--input', '640x360', '--output', '640x360', '--frames', '4' `
    -NoNewWindow -PassThru -RedirectStandardOutput (Join-Path $rt 'tap.txt') -RedirectStandardError (Join-Path $rt 'e.txt')
$null = $p.WaitForExit(600000)
$env:DLSS5_PROF = ''; $env:DLSS5_TAP = ''; $env:DLSS5_TRACE = ''
New-Item -ItemType Directory -Force $Dest | Out-Null
Get-ChildItem (Join-Path $rt 'tap_s*') | ForEach-Object {
    $b = [IO.File]::ReadAllBytes($_.FullName)
    $nz = ($b | Where-Object { $_ -ne 0 }).Count
    "$($_.Name) $($b.Length) nonzero=$nz"
}
Move-Item -Force (Join-Path $rt 'tap_s*') $Dest
Move-Item -Force (Join-Path $rt 'nr-trace.tsv') (Join-Path $Dest 'nr-trace.tsv')
