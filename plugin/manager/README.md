# DLSS5 Manager

给游戏装上 DLSS5 神经渲染的管理工具：扫描游戏库、检测游戏、一键安装/卸载、调参数。两种注入方式：

- **OptiScaler** (游戏自带 DLSS / FSR2+ / XeSS)：注入与画面合成由 OptiScaler 的 DLSS Neural Rendering 分支完成
  (GPL-3，github.com/Dagherbou/OptiScaler_DLSSNR)，本工具写 `OptiScaler.ini` 的 `[DlssNr]` 节。有深度、运动矢量与历史帧，效果最好。
- **兜底模式** (没有超分的 DX12 / DX11 游戏)：本仓库自己的 dxgi.dll 代理 (`plugin/fallback/`)，在 Present 时截取画面跑 DLSS-NR，
  参数在 `dlss5fb.ini`。没有深度；运动矢量来自硬件光流 (有历史帧)；画面运动时保持不变的界面 (HUD) 不处理。

没有超分、但有 DX12 或 DX11 的游戏默认选兜底模式，其余默认 OptiScaler；`--mode` / 界面的"注入方式"可以改。

- `DLSS5Manager.exe`：图形界面；`dlss5.exe`：命令行 (`dlss5 help`)。
- .NET Framework 4.8 (Windows 10/11 自带)，不需要另装运行库。
- 运行时读取程序旁边的 `payload/` (由 `plugin/build_plugin.py` 生成)。**payload 里的 `nvngx_dlssnr.dll` 是 NVIDIA 的程序，
  不在本仓库里，也不能公开分发**；仓库只有工具的源码。

## 构建

```
plugin\manager\build.bat           -> plugin\manager\bin\DLSS5Manager.exe, dlss5.exe
python plugin\build_plugin.py      -> plugin\dist\DLSS5-RTX30\ (含 payload 与两个 exe)
```

## 检测内容 (`Core.cs` 的 `Probe`)

| 项 | 方法 |
|---|---|
| 游戏库 | Steam `libraryfolders.vdf` + `appmanifest_*.acf`；Epic `Manifests/*.item` (只取 AppCategories 含 games)；GOG 注册表；手动添加的文件夹 |
| 主程序 | Unreal 的 `*-Win64-Shipping.exe` > 商店记录的启动程序 > 最大的 exe (排除启动器、崩溃上报、安装程序等) |
| 引擎 | Shipping exe / `Engine` 目录 → Unreal；`UnityPlayer.dll` → Unity；`REDprelauncher.exe` → REDengine |
| 位数 | 主程序的 PE 头: 32 位游戏 → **拒绝安装** (DLSS5 的 dll 只有 64 位) |
| 图形 API | 主程序 (Unity 看 UnityPlayer.dll) 的导入表与延迟导入表：d3d12 / d3d11 / vulkan-1；`D3D12\D3D12Core.dll` (Agility SDK) |
| 超分 | `nvngx_dlss.dll`、`sl.dlss.dll`、`libxess*.dll`、`amd_fidelityfx_*.dll`、`ffx_fsr2_api_*.dll` 等；Unreal 的 DLSS 插件目录 |
| 反作弊 | exe 目录及往上 3 层里 EasyAntiCheat / BattlEye / xigncode / GameGuard / vgk / ACE 等 → **拒绝安装** |
| 注入名 | dxgi / winmm / version / dbghelp / d3d12 / wininet / winhttp 中第一个没被占用的；识别已有的 ReShade、Special K、别人装的 OptiScaler |

OptiScaler 方式下 DX11 游戏只能经 dx11on12 跑 DLSS5，安装时会写 `[Upscalers] Dx11Upscaler=dlss_12`；兜底模式下 DX11 游戏的画面经共享纹理交给代理自建的 DX12 设备处理 (DLSS-NR 的 D3D11 入口是空壳)。

## 预设

模型分辨率 (`WorkingScale`) 按显卡与屏幕分辨率估算：RTX 3060 实测 (移植版 kernel) 540p 14.3 / 720p 22.6 / 900p 31.8 / 1080p 41.9 ms，
拟合 `t = 5.1 ms + 17.8 ms × 百万像素`，其他显卡按相对吞吐缩放；性能 / 均衡 / 画质分别给 DLSS5 本身 10 / 14 / 20 ms。

## 参数

`dlss5 params` 列出全部参数。画面参数的含义来自逆向 (见 `torch/README.md` 与 `tools/notes.jsonl`)：
LocalStructure / LocalTone / SkinStructure / AutoMask / Style 进网络的 pre_block，Intensity 是网络之外的线性混合。

兜底模式没有 OptiScaler 的合成参数 (TransferStrength / ColourStrength / MaxRatio)，另有 Temporal (保留历史帧) 与 Compare (左右对比)。
它的 ini 在游戏运行中约 1 秒内重新读取，改参数不用重启游戏。原理与验证见 `plugin/fallback/README.md`。
