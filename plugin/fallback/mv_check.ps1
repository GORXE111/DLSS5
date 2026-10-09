# 运动矢量 (光流) 校验: fbtest 让图片每帧右移 4 像素 (与 nr-lab --temporal-shift 4 相同)，兜底模式在全分辨率、
# 关闭死区与平滑时，比较 NR 输出:
#   -Image pattern.ppm (nr-lab 的测试图): 与 nr-lab 的"正确运动矢量"(mv_x=-4) / "零运动矢量"参考逐帧比较
#   其他图片: 参考 = 同一序列用精确的常数运动矢量 (MvConstX=-4) 跑出的结果 (这条路径已证明与 nr-lab 逐字节一致)
#   powershell -ExecutionPolicy Bypass -File mv_check.ps1 [-Image sponza640.ppm] [-Extra "FlowPerf=5;FlowGrid=1"]
param([string]$Image = 'pattern.ppm', [string]$Extra = '')
$ErrorActionPreference = 'Stop'
$t = Join-Path $PSScriptRoot 'test'
Copy-Item (Join-Path $PSScriptRoot 'bin\*') $t -Force
Push-Location $t
function Run([string]$more) {
    "[DLSS5]`nWorkingScale=1.0`nStabilize=0`nSmooth=1`nTemporal=1`nDumpFrame=1`nDumpCount=4`n$($more -replace ';', "`n")" |
        Set-Content -Encoding ascii dlss5fb.ini
    & .\fbtest.exe $Image 10 --pan 4 | Out-Null
}
try {
    if ($Image -eq 'pattern.ppm') {
        Run $Extra
        python (Join-Path $PSScriptRoot 'mv_check.py') $t (Join-Path $PSScriptRoot '..\..\research')
    } else {
        Run 'MvConstX=-4'
        foreach ($k in 1, 2, 3) { Copy-Item "dlss5fb_nr_$k.ppm" "ref_nr_$k.ppm" -Force }
        Run $Extra
        python (Join-Path $PSScriptRoot 'mv_check.py') $t --self
    }
} finally { Pop-Location }
