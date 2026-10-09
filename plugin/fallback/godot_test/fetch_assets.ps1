# 取测试场景: Khronos glTF 示例里的 Crytek Sponza (Cryengine Limited License，只在本地用，不进仓库)，
# 拷进 godot_test\sponza\ 并让 Godot 无界面导入一次。
#   powershell -ExecutionPolicy Bypass -File fetch_assets.ps1 -Godot <Godot_v4.x_win64_console.exe>
param([Parameter(Mandatory)][string]$Godot, [string]$Cache)
$ErrorActionPreference = 'Stop'
if (-not $Cache) { $Cache = Join-Path $PSScriptRoot '..\..\..\_dl\gltf' }   # PS 5.1: $PSScriptRoot 在 param 默认值里为空
if (-not (Test-Path $Cache)) {
    git clone --depth 1 --filter=blob:none --sparse https://github.com/KhronosGroup/glTF-Sample-Assets.git $Cache
}
git -C $Cache sparse-checkout set Models/Sponza
$dst = Join-Path $PSScriptRoot 'sponza'
New-Item -ItemType Directory -Force $dst | Out-Null
Copy-Item (Join-Path $Cache 'Models\Sponza\glTF\*') $dst -Force
& $Godot --headless --path $PSScriptRoot --import
Write-Host "done: $dst"
