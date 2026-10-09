# 运动中的闪烁: fbtest 让图片每帧右移 Pan 像素；把每帧输出按已知位移移回去，与上一帧比较 (运动补偿后的帧间差)。
# 同一个场景点在运动中保持稳定 = 不闪。对比几种设置，全分辨率。
#   powershell -ExecutionPolicy Bypass -File motion_flicker.ps1 [-Image sponza640.ppm] [-Pan 4]
param([string]$Image = 'sponza640.ppm', [int]$Pan = 4, [int]$Frames = 12)
$ErrorActionPreference = 'Stop'
$t = Join-Path $PSScriptRoot 'test'
Copy-Item (Join-Path $PSScriptRoot 'bin\*') $t -Force
Push-Location $t
$cases = [ordered]@{
    '每帧重置 (无历史)'             = 'Temporal=0;Stabilize=0;Smooth=1'
    '历史 + 零运动矢量'             = "MvConstX=0.0001;Temporal=1;Stabilize=0;Smooth=1"
    '历史 + 光流 (中等, 无细化)'     = 'FlowPerf=10;FlowRefine=0;Stabilize=0;Smooth=1'
    '历史 + 光流 (中等) + LK 细化'   = 'FlowPerf=10;FlowRefine=2;Stabilize=0;Smooth=1'
    '历史 + 光流 (最好) + LK 细化'   = 'FlowPerf=5;FlowRefine=2;Stabilize=0;Smooth=1'
    '默认设置'                       = ''
    '历史 + 精确运动矢量 (参考)'    = "MvConstX=-$Pan;Stabilize=0;Smooth=1"
}
try {
    foreach ($name in $cases.Keys) {
        "[DLSS5]`nWorkingScale=1.0`nDumpFrame=1`nDumpCount=$Frames`n$($cases[$name] -replace ';', "`n")" | Set-Content -Encoding ascii dlss5fb.ini
        & .\fbtest.exe $Image ($Frames + 4) --pan $Pan | Out-Null
        $r = python (Join-Path $PSScriptRoot 'motion_flicker.py') $t $Pan $Frames
        "{0,-28} {1}" -f $name, $r
    }
} finally { Pop-Location }
