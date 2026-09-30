# 用 GPU 时间戳比较多个 dll 变体 (不受 CPU 提交/等待与其他程序干扰的平均值影响)。
# 变体交替跑 Rounds 轮，每轮每个变体一次，报告每帧 GPU 跨度的 min / 中位数。
#   .\gpubench.ps1 -Dlls @{ base='..\sm86_port\nvngx_dlssnr.dll'; e2='..\sm86_port_ev2dv1\nvngx_dlssnr.dll' }
param(
    [hashtable]$Dlls,
    [string]$Size = '1920x1080',
    [int]$Frames = 20,
    [int]$Rounds = 3
)
$results = @{}
foreach ($r in 1..$Rounds) {
    foreach ($name in $Dlls.Keys) {
        $dir = Join-Path $PSScriptRoot "gb_$name"
        New-Item -ItemType Directory -Force $dir | Out-Null
        foreach ($f in 'nr-lab.exe', 'nvngx.dll') { Copy-Item (Join-Path $PSScriptRoot $f) $dir -Force }
        foreach ($f in 'nvngx_dlss.dll', 'nvngx_dlssg.dll') { Copy-Item (Join-Path $PSScriptRoot "..\dlss5\$f") $dir -Force }
        if ($r -eq 1) { Copy-Item $Dlls[$name] (Join-Path $dir 'nvngx_dlssnr.dll') -Force }
        Push-Location $dir
        $env:DLSS5_PROF = '1'
        $p = Start-Process -FilePath .\nr-lab.exe -ArgumentList @('--nr-only', '--input', $Size, '--output', $Size, '--frames', "$Frames") `
            -NoNewWindow -PassThru -RedirectStandardOutput gb.txt -RedirectStandardError gb_err.txt
        $null = $p.WaitForExit(600000)
        $env:DLSS5_PROF = ''
        $line = (Get-Content gb.txt | Select-String 'span_min=([\d.]+) span_median=([\d.]+)').Matches
        Pop-Location
        if ($line) {
            if (-not $results[$name]) { $results[$name] = @() }
            $results[$name] += [pscustomobject]@{ min = [double]$line[0].Groups[1].Value; med = [double]$line[0].Groups[2].Value }
        }
    }
}
foreach ($name in ($Dlls.Keys | Sort-Object)) {
    $v = $results[$name]
    "{0,-10} GPU/frame: min {1:N2} ms  median(best round) {2:N2} ms  [{3}]" -f $name, ($v.min | Measure-Object -Minimum).Minimum,
        ($v.med | Measure-Object -Minimum).Minimum, (($v | ForEach-Object { '{0:N1}' -f $_.med }) -join ' ')
}
