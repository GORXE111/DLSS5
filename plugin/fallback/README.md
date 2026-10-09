# DLSS5 兜底模式 (dxgi.dll 代理)

给**没有超分**的 D3D12 游戏用的 DLSS5：不借助 OptiScaler，自己挂在交换链的 Present 上，每帧把游戏画面送进 DLSS-NR。
由 DLSS5 Manager 安装 (`dlss5 install <游戏> --mode fallback`，没有超分的 DX12 游戏会自动选它)。

## 做法

```
游戏 Present ─┬─ 拷贝后缓冲 ──> 缩到 WorkingScale (计算着色器) ──> DLSS-NR (feature 18) ──┐
              │                                                                        │
              └─ 写回后缓冲:  原画面 + 放大(NR 输出 - NR 输入)   (WorkingScale = 1 时直接用 NR 输出) <┘
              ──> 真正的 Present
```

- `dxgi.dll` 代理：20 个导出与系统 dxgi.dll 同名同序号，`CreateDXGIFactory*` 之外都用 jmp 转发 (`exports.asm`)。
  拿到工厂后改它的虚表 (CreateSwapChain / ForHwnd / ForCoreWindow / ForComposition)，从参数里拿到游戏的 D3D12 命令队列，
  再改交换链的虚表 (Present / Present1 / ResizeBuffers / ResizeBuffers1 / SetColorSpace1)。D3D11 的交换链原样放过。
- 所有处理录在自己的命令列表里，提交到游戏的同一个队列，排在游戏这一帧的渲染之后、Present 之前。
- NGX 的调用顺序与 `harness/nr-lab.cpp` 相同：驱动的 `_nvngx.dll` Init → 经调用桥 `dlss5_nvngx.dll`
  (就是 `harness/nvngx-bridge.cpp`；NR 片段要求调用方模块名里含 "nvngx.dll") 调片段的 Init_Ext → CreateFeature(18)。
  不需要 `nvngx_dlss.dll` / `nvngx_dlssg.dll`。
- 降采样用 `round(0.5/scale)²` 个双线性采样的盒滤波 (0.5 时正好是 2x2 平均)；只把模型的**改动量**放大回原分辨率，
  原画面的细节不会因为模型分辨率低而变糊。
- 没有深度 (给全 0，DepthInverted) 和运动矢量 (全 0)。默认每帧 Reset (不用历史)；`Temporal=1` 时沿用历史。
- 只处理 SDR：R8G8B8A8 / B8G8R8A8 / R10G10B10A2 且色彩空间为 sRGB。HDR (scRGB 浮点、HDR10) 原样放过并写日志。

## 文件 (装进游戏 exe 目录)

| 文件 | 来源 |
|---|---|
| `dxgi.dll` | 本目录 `build.bat` |
| `dlss5_nvngx.dll` | `harness/nvngx-bridge.cpp` (Apache-2.0，见 `harness/NOTICE.upstream`) |
| `nvngx_dlssnr.dll` | sm_86 移植版 (NVIDIA 的程序，**不在仓库里，不能公开分发**) |
| `dlss5fb.ini` | 参数，见下 |
| `dlss5fb.log` | 运行日志 |

## 参数 (`dlss5fb.ini` 的 `[DLSS5]` 节，游戏运行中改动约 1 秒内生效)

| 名字 | 默认 | 说明 |
|---|---|---|
| Enabled | 1 | 0 = 原样放过 |
| WorkingScale | 0.5 | 模型分辨率占画面的比例 (0.25~1) |
| Intensity / LocalTone / LocalStructure / SkinStructure / AutoMask / Style | 1 / 1 / 1 / -1 / 1 / 0 | DLSSNR 画面参数，含义见 `torch/README.md` |
| Temporal | 0 | 1 = 保留历史帧 (运动矢量为 0：静止画面更稳，运动时可能拖影) |
| Compare | 0 | 1 = 左半屏原画面，右半屏 DLSS5 |
| ToggleKey / CompareKey | 0x79 / 0x7A | 开关与对比的热键 (默认 F10 / F11，虚拟键码) |
| DumpFrame | -1 | 测试用：第 N 帧把输入/NR 输出/结果存成 `dlss5fb_in/nr/out.ppm` (也可用环境变量 `DLSS5FB_DUMP`) |

## 构建与测试

```
plugin\fallback\build.bat     -> bin\dxgi.dll, bin\dlss5_nvngx.dll, bin\fbtest.exe
                                 (需要 DLSS SDK 头文件 oss\nvidia-dlss\include，不在本仓库里)
fbtest.exe image.ppm [帧数]    最小的 D3D12 "游戏"：每帧把图片拷进后缓冲再 Present；与 dxgi.dll 等放在同一目录
```

RTX 3060 上的验证 (fbtest + nr-lab 的合成测试图)：

| 设定 | 结果 |
|---|---|
| 640x360，WorkingScale 1 | NR 输出与 nr-lab 的 `research/out_f0.ppm` **逐字节一致** |
| 1920x1080，WorkingScale 1 | 与 nr-lab 的 1080p 输出逐字节一致；每帧多 40 ms |
| 1920x1080，WorkingScale 0.5 | 每帧多 12 ms；改动量与全分辨率的相关 0.82、幅度 0.94 |
| 运行中改 Style / Temporal | 约 1 秒内重读 ini 并重建 feature，不中断 |

真实游戏还没测过。
