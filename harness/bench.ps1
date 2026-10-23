# sm_86 整链路回归。这里断言固定基准的校验和；近似误差验收使用 torch/check*.py。
param(
    [string]$Variant = 'rt_sm86',
    [string]$Dll = (Join-Path $PSScriptRoot '..\sm86_port\nvngx_dlssnr.dll'),
    [string[]]$Sizes = @('1280x720', '1920x1080'),
    [ValidateRange(6, 100000)][int]$Frames = 24,
    [string]$Report
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$dir = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot $Variant))
$dllSource = (Resolve-Path -LiteralPath $Dll).Path
New-Item -ItemType Directory -Force $dir | Out-Null
if (-not $Report) { $Report = Join-Path $dir 'bench-result.json' }
$Report = [IO.Path]::GetFullPath($Report)
$summary = [ordered]@{
    schema = 1; variant = $Variant; referenceKind = 'RTX3060_port'; claim = 'port_regression_only'
    mode = 'fixed_checksum_bit_exact'; dllSha256 = (Get-FileHash -LiteralPath $dllSource -Algorithm SHA256).Hash
    expected = @{ normal = '5B3A70119F21C9EE'; intensity0 = 'D61DE47C7AECF78F' }
    checks = @(); performance = @(); passed = $false; errors = @()
}

function Copy-IfDifferent([string]$Source, [string]$Destination) {
    if ([IO.Path]::GetFullPath($Source) -ne [IO.Path]::GetFullPath($Destination)) {
        Copy-Item -LiteralPath $Source -Destination $Destination -Force
    }
}

function Invoke-Lab([string[]]$LabArgs, [string]$Log, [int]$ExpectedFrames) {
    # Never let a crashed process borrow the previous run's JSON.
    if (Test-Path -LiteralPath 'nr-lab-result.json') { Remove-Item -LiteralPath 'nr-lab-result.json' }
    $p = Start-Process -FilePath .\nr-lab.exe -ArgumentList $LabArgs -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $Log -RedirectStandardError 'err.txt'
    if (-not $p.WaitForExit(600000)) { $p.Kill(); throw "nr-lab 超时: $LabArgs" }
    $p.Refresh()
    if ($p.ExitCode -ne 0) { throw "nr-lab 退出码 $($p.ExitCode): $LabArgs" }
    if (-not (Test-Path -LiteralPath 'nr-lab-result.json')) { throw 'nr-lab 未生成结果 JSON' }
    $result = Get-Content -LiteralPath 'nr-lab-result.json' -Raw | ConvertFrom-Json
    if (-not $result.created -or -not $result.upscalingValidated -or $result.evaluationsSucceeded -ne $ExpectedFrames) {
        throw "nr-lab 未完成全部有效帧: $($result.evaluationsSucceeded)/$ExpectedFrames"
    }
    if (Select-String -LiteralPath $Log -Pattern 'sm86-crash' -Quiet) { throw "nr-lab 捕获到 sm86-crash: $Log" }
    return $result
}

function Assert-Checksum([string]$Name, $Result, [string]$Expected) {
    $passed = $Result.checksumFnv1a64 -eq $Expected
    $summary.checks += @{ name = $Name; mode = 'bit_exact_checksum'; actual = $Result.checksumFnv1a64; expected = $Expected; passed = $passed }
    if (-not $passed) { throw "$Name 校验和不一致: $($Result.checksumFnv1a64), 预期 $Expected。近似吻合不等于此基准通过。" }
}

$pushed = $false
try {
    foreach ($f in 'nr-lab.exe', 'nvngx.dll') {
        Copy-IfDifferent (Join-Path $PSScriptRoot "rt_sm86\$f") (Join-Path $dir $f)
    }
    foreach ($f in 'nvngx_dlss.dll', 'nvngx_dlssg.dll') {
        Copy-IfDifferent (Join-Path $PSScriptRoot "..\dlss5\$f") (Join-Path $dir $f)
    }
    Copy-IfDifferent $dllSource (Join-Path $dir 'nvngx_dlssnr.dll')
    Push-Location $dir
    $pushed = $true
    $base = @('--nr-only', '--input', '640x360', '--output', '640x360', '--frames', '4', '--model', '1')
    $r = Invoke-Lab $base 'o_check.txt' 4
    Assert-Checksum '360p/4frames' $r $summary.expected.normal
    Copy-Item -LiteralPath 'nr-lab-output-model1-srgb.ppm' -Destination 'out_check.ppm' -Force
    $z = Invoke-Lab ($base + @('--intensity', '0')) 'o_int0.txt' 4
    Assert-Checksum '360p/intensity0' $z $summary.expected.intensity0
    Copy-Item -LiteralPath 'nr-lab-output-model1-srgb.ppm' -Destination 'out_int0.ppm' -Force
    "{0}: bit-exact 基准通过 (normal={1}, intensity0={2})" -f $Variant, $r.checksumFnv1a64, $z.checksumFnv1a64
    foreach ($res in $Sizes) {
        if ($res -notmatch '^[1-9][0-9]*x[1-9][0-9]*$') { throw "无效分辨率: $res" }
        $null = Invoke-Lab @('--nr-only', '--input', $res, '--output', $res, '--frames', "$Frames", '--model', '1') 'o_perf.txt' $Frames
        $t = @(Get-Content -LiteralPath 'o_perf.txt' | Select-String 'frame (\d+) EvaluateFeature = 0x00000001' |
            ForEach-Object { [datetime]::ParseExact($_.Line.Substring(0, 12), 'HH:mm:ss.fff', $null) })
        if ($t.Count -ne $Frames) { throw "性能日志帧数错误: $($t.Count)/$Frames" }
        $dt = @(for ($i = 5; $i -lt $t.Count; $i++) {
            $elapsed = ($t[$i] - $t[$i - 1]).TotalMilliseconds
            if ($elapsed -lt 0) { $elapsed += 86400000 }
            $elapsed
        })
        $m = $dt | Measure-Object -Average -Minimum
        $summary.performance += @{ resolution = $res; frames = $Frames; samples = $dt.Count; averageMs = $m.Average; minimumMs = $m.Minimum }
        "{0}: {1,-10} frames={2} avg={3:N1} ms min={4:N1} ms" -f $Variant, $res, $t.Count, $m.Average, $m.Minimum
    }
    $summary.passed = $true
} catch {
    $summary.errors += $_.Exception.Message
    throw
} finally {
    if ($pushed) { Pop-Location }
    $reportParent = Split-Path -Parent $Report
    if ($reportParent) { New-Item -ItemType Directory -Force $reportParent | Out-Null }
    $summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $Report -Encoding utf8
}
