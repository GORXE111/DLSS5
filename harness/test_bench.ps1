# CPU-only failure-path checks against the actual bench functions; launches no application.
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$benchAst = [System.Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot 'bench.ps1'), [ref]$null, [ref]$null)
foreach ($name in 'Invoke-Lab', 'Assert-Checksum') {
    $node = $benchAst.Find({ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name }, $true)
    Invoke-Expression $node.Extent.Text
}
$testDir = Join-Path ([IO.Path]::GetTempPath()) ('dlss5_bench_' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $testDir | Out-Null
$summary = @{ checks = @() }
$script:fakeMode = 'success'
function Start-Process {
    param($FilePath, $ArgumentList, $WindowStyle, [switch]$PassThru, $RedirectStandardOutput, $RedirectStandardError)
    if (Test-Path -LiteralPath 'nr-lab-result.json') { throw 'Stale JSON was not removed before launch' }
    Set-Content -LiteralPath $RedirectStandardOutput -Value 'no crash'
    if ($script:fakeMode -ne 'missing') {
        @{ created = $true; upscalingValidated = $true; evaluationsSucceeded = $(if ($script:fakeMode -eq 'badframes') { 1 } else { 4 }) } |
            ConvertTo-Json | Set-Content -LiteralPath 'nr-lab-result.json'
    }
    $fake = [pscustomobject]@{ ExitCode = $(if ($script:fakeMode -eq 'exitfailure') { 9 } else { 0 }) }
    $fake | Add-Member -MemberType ScriptMethod -Name WaitForExit -Value { param($timeout) return $true }
    $fake | Add-Member -MemberType ScriptMethod -Name Refresh -Value { }
    return $fake
}
function Assert-Throws([scriptblock]$Action, [string]$Message) {
    $thrown = $false
    try { & $Action | Out-Null } catch { $thrown = $true }
    if (-not $thrown) { throw $Message }
}
Push-Location $testDir
try {
    Set-Content -LiteralPath 'nr-lab-result.json' -Value '{"stale":true}'
    $script:fakeMode = 'missing'
    Assert-Throws { Invoke-Lab @('--test') 'out.txt' 4 } 'A stale/missing result passed'
    $script:fakeMode = 'exitfailure'
    Assert-Throws { Invoke-Lab @('--test') 'out.txt' 4 } 'A failed process passed'
    $script:fakeMode = 'badframes'
    Assert-Throws { Invoke-Lab @('--test') 'out.txt' 4 } 'Incomplete evaluation count passed'
    $script:fakeMode = 'success'
    $null = Invoke-Lab @('--test') 'out.txt' 4
    Assert-Throws { Assert-Checksum 'bad' ([pscustomobject]@{ checksumFnv1a64 = 'wrong' }) 'expected' } 'An incorrect checksum passed'
    Assert-Checksum 'good' ([pscustomobject]@{ checksumFnv1a64 = 'expected' }) 'expected'
    if ($summary.checks.Count -ne 2 -or $summary.checks[0].passed -or -not $summary.checks[1].passed) { throw 'Checksum report omitted failure' }
    'PASS: bench stale JSON, exit failure, frame count, checksum failure and success (5 checks)'
} finally {
    Pop-Location
    $resolvedTestDir = [IO.Path]::GetFullPath($testDir)
    $resolvedTempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
    if (-not $resolvedTestDir.StartsWith($resolvedTempRoot, [StringComparison]::OrdinalIgnoreCase)) { throw 'Unexpected test cleanup path' }
    Remove-Item -LiteralPath $resolvedTestDir -Recurse -Force
}
