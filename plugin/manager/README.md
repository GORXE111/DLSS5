# DLSS5 Manager

给支持超分 (DLSS / FSR2+ / XeSS) 的游戏装上 DLSS5 神经渲染的管理工具：扫描游戏库、检测游戏、一键安装/卸载、调参数。
注入与画面合成由 OptiScaler 的 DLSS Neural Rendering 分支完成 (GPL-3，github.com/Dagherbou/OptiScaler_DLSSNR)，
本工具负责选游戏、选注入方式、写 `OptiScaler.ini` 的 `[DlssNr]` 节、备份与还原。

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
| 图形 API | 主程序 (Unity 看 UnityPlayer.dll) 的导入表与延迟导入表：d3d12 / d3d11 / vulkan-1；`D3D12\D3D12Core.dll` (Agility SDK) |
| 超分 | `nvngx_dlss.dll`、`sl.dlss.dll`、`libxess*.dll`、`amd_fidelityfx_*.dll`、`ffx_fsr2_api_*.dll` 等；Unreal 的 DLSS 插件目录 |
| 反作弊 | exe 目录及往上 3 层里 EasyAntiCheat / BattlEye / xigncode / GameGuard / vgk / ACE 等 → **拒绝安装** |
| 注入名 | dxgi / winmm / version / dbghelp / d3d12 / wininet / winhttp 中第一个没被占用的；识别已有的 ReShade、Special K、别人装的 OptiScaler |

DX11 游戏只能经 dx11on12 跑 DLSS5，安装时会写 `[Upscalers] Dx11Upscaler=dlss_12`。

## 预设

模型分辨率 (`WorkingScale`) 按显卡与屏幕分辨率估算：RTX 3060 实测 (移植版 kernel) 540p 14.3 / 720p 22.6 / 900p 31.8 / 1080p 41.9 ms，
拟合 `t = 5.1 ms + 17.8 ms × 百万像素`，其他显卡按相对吞吐缩放；性能 / 均衡 / 画质分别给 DLSS5 本身 10 / 14 / 20 ms。

## 参数

`dlss5 params` 列出全部参数。画面参数的含义来自逆向 (见 `torch/README.md` 与 `tools/notes.jsonl`)：
LocalStructure / LocalTone / SkinStructure / AutoMask / Style 进网络的 pre_block，Intensity 是网络之外的线性混合。

## 还没做

- 不带超分的游戏 (兜底模式)：从交换链截取画面、无历史逐帧处理。需要自己的 present hook，不经 OptiScaler。
