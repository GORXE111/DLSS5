# DLSS5 兜底模式 (dxgi.dll 代理)

给**没有超分**的 D3D12 / D3D11 游戏用的 DLSS5：不借助 OptiScaler，自己挂在交换链的 Present 上，每帧把游戏画面送进 DLSS-NR。
由 DLSS5 Manager 安装 (`dlss5 install <游戏> --mode fallback`，没有超分的 DX12 / DX11 游戏会自动选它)。

## 做法

```
游戏 Present ─┬─ 拷贝后缓冲 ──> 缩到 WorkingScale (计算着色器，同时写一张灰度图)
              │      ──> 硬件光流 (这一帧灰度 vs 上一帧) ──> 运动矢量 (3x3 中值 + Lucas-Kanade 细化)
              │      ──> DLSS-NR (feature 18，带历史帧) ──> 改动量的时间平滑
              └─ 写回后缓冲:  原画面 + 放大(NR 输出 - NR 输入)
              ──> 真正的 Present
```

- `dxgi.dll` 代理：20 个导出与系统 dxgi.dll 同名同序号，`CreateDXGIFactory*` 之外都用 jmp 转发 (`exports.asm`)。
  拿到工厂后改它的虚表 (CreateSwapChain / ForHwnd / ForCoreWindow / ForComposition)，从参数里拿到游戏的 D3D12 命令队列，
  再改交换链的虚表 (Present / Present1 / ResizeBuffers / ResizeBuffers1 / SetColorSpace1)。
- 所有处理录在自己的命令列表里，提交到游戏的同一个队列，排在游戏这一帧的渲染之后、Present 之前。
- **D3D11 游戏**：`nvngx_dlssnr.dll` 虽然导出了 `NVSDK_NGX_D3D11_*`，但 `D3D11_Init_Ext` 检查完调用方就无条件返回
  0xBAD00001 (不支持；`harness/nr11.cpp` 实测 + 反汇编)，所以不能直接走 D3D11。代理在游戏的适配器上自建一个 D3D12 设备和队列
  (全进程一个，NGX 只认第一个设备)，再建一张两边共享的纹理：D3D11 把后缓冲拷进去 → 上面整条管线原样在这张纹理上跑 →
  D3D11 拷回后缓冲。两个设备用共享 fence (`ID3D11DeviceContext4::Signal/Wait` 与队列的 `Wait/Signal`) 在 GPU 上排队，CPU 不等待。
  翻转 / blt 交换链、`D3D11CreateDeviceAndSwapChain`、sRGB 后缓冲都测过；多重采样的后缓冲不处理。需要 Windows 10 1703+。
- NGX 的调用顺序与 `harness/nr-lab.cpp` 相同：驱动的 `_nvngx.dll` Init → 经调用桥 `dlss5_nvngx.dll`
  (就是 `harness/nvngx-bridge.cpp`；NR 片段要求调用方模块名里含 "nvngx.dll") 调片段的 Init_Ext → CreateFeature(18)。
  不需要 `nvngx_dlss.dll` / `nvngx_dlssg.dll`。
- 降采样用 `round(0.5/scale)²` 个双线性采样的盒滤波 (0.5 时正好是 2x2 平均)；只把模型的**改动量**放大回原分辨率，
  原画面的细节不会因为模型分辨率低而变糊。
- 没有深度 (给全 0，DepthInverted)。**运动矢量来自显卡的硬件光流** (NVIDIA Optical Flow，驱动自带的 `nvofapi64.dll`)：
  每帧把 NR 输入的灰度图交给光流引擎，与上一帧比较 (前向流 = 当前像素在上一帧的位置，正是 DLSS-NR 要的方向，
  已用 nr-lab 的"正确运动矢量"参考验证)。光流引擎异步运行：第一张命令列表提交后 Signal，光流等它，
  游戏队列再等光流的 fence，然后录第二张命令列表 (运动矢量、DLSS-NR、合成)。
  - 光流输出 S10.5 定点 (每像素 32)。光流的 ABGR8 "彩色"输入不可用：D3D12 驱动把 RGBA8 的每个字节当成一个灰度像素
    (水平流大 4 倍)，所以只用灰度。
  - 硬件光流本身有 ±0.4 像素的平滑误差，DLSS-NR 对运动矢量很敏感，所以在其上做 2 轮 Lucas-Kanade 细化 (5x5 窗口)。
  - 销毁顺序要跟 NVIDIA 示例一致 (注销 → 释放纹理 → 销毁会话)，先销毁会话会让 D3D12 驱动随后读到已释放的内存而崩溃。
  - 没有光流 (非 NVIDIA 驱动 / 初始化失败) 时退回每帧重置。
- **场景切换**：光流之后用运动矢量把上一帧灰度图对齐到这一帧再比较，平均差超过 `CutThreshold` (0.08) 就判为切换。
  DLSS-NR 自己的门控要晚一帧才丢掉旧历史 (切换后第一帧有旧场景的残影，与旧画面相关 0.27)，所以切换那一帧直接显示游戏原画面，
  下一帧起立即恢复全部效果。全在 GPU 上判断，不需要 CPU 等 GPU。平移 24 像素/帧也不会误判。
- **颜色**：模型会按画面内容做局部色调调整 (Sponza 各帧: 饱和度 x0.98~1.06，色相转 5~8°，有的帧偏暖、有的偏冷)。
  `ColourStrength=0` 只取它的明暗与细节，保持游戏原本的颜色 (色相转动降到 1°，细节增强保留)。
  曾怀疑模型要的是线性 HDR 输入：实测线性输入 (`Linear=1`，RGBA16F + IsHDR) 让细节增强几乎消失 (x1.003 vs sRGB 的 x1.03~1.07)，
  所以默认送 sRGB 画面。
- **防闪烁**：网络对输入的微小扰动极其敏感，而且有全局注意力：游戏画面里 ±1/255 的去色带抖动、
  或者角落里一个 HUD 数字变化，都会让整幅输出明暗起伏 (静止镜头下帧间变化 1.40/255，见 `tools/notes.jsonl`)。
  正常接入时历史帧会把它平均掉；兜底模式另外加了两层 (有了光流历史帧后仍然有益)：输入死区 (`Stabilize`) 让没变的像素送进模型的值完全不变
  (网络是确定性的，输入不变输出就不变)；改动量的时间平滑 (`Smooth`)，只在局部输入没变的地方生效，所以运动画面不拖影。
  静止镜头 40 帧：1.40 → 0.18/255，超过 8/255 的跳变从 0.9% 降到 0。
- 格式: SDR 的 R8G8B8A8 / B8G8R8A8 / R10G10B10A2 (sRGB 色彩空间)；HDR 的 scRGB (R16G16B16A16 浮点，线性) 与 HDR10
  (R10G10B10A2 + PQ / Rec.2020)。其他组合与 MSAA 原样放过并写日志。
- **HDR**：DLSS-NR 即使开 IsHDR 也把输出截到 [0,1]，线性输入又会让细节效果消失 (nr-lab 实测)，所以送给它的是"看起来像 SDR"的画面:
  后缓冲换成线性 Rec.709、纸白 = 1 (`HdrPaperWhite` 尼特，scRGB 的 1.0 = 80 尼特)，高于 `HdrKnee` (0.8) 的亮度平滑压进 [0.8, 1)，
  再按 sRGB 编码。结果: 原 HDR 值 + 模型在压缩域里的改变量 (低于拐点处 = 模型的原样改变；高光处不被反向放大)，再编回后缓冲格式。
  fbtest (Sponza): HDR 亮度 x1 时与 SDR 结果相差 1.6/255 (拐点以下)；x4 时高光 (最高 4 倍纸白) 原样保留，高光区平均只变 3%。

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
| Temporal | 1 | 1 = 历史帧 + 光流运动矢量；0 = 每帧重置 |
| FlowPerf / FlowGrid / FlowRefine / FlowFilter | 5 / 2 / 2 / 1 | 光流质量 (5 最好、10 中、20 快)、每个矢量覆盖的像素、Lucas-Kanade 轮数、3x3 中值 |
| Stabilize | 1.5 | 输入死区 (1/255 为单位)：像素变化小于它时沿用上次送进模型的值，0 = 关 |
| Smooth | 0.2 | 画面没变的地方 DLSS5 改动每帧跟进的比例 (指数平均)；画面在变的地方立刻跟上；1 = 关 |
| Compare | 0 | 1 = 左半屏原画面，右半屏 DLSS5 |
| ToggleKey / CompareKey | 0x79 / 0x7A | 开关与对比的热键 (默认 F10 / F11，虚拟键码) |
| ColourStrength | 1 | 0 = 保持游戏原本的颜色，只取 DLSS5 的明暗与细节；1 = 连颜色一起取 |
| CutThreshold | 0.08 | 场景切换判定 (运动补偿后的平均灰度差)，0 = 关 |
| HdrPaperWhite | 200 | HDR: 游戏的纸白亮度 (尼特)，对应模型看到的白 |
| HdrKnee | 0.8 | HDR: 高于此亮度 (纸白 = 1) 平滑压缩，模型看到的高光不会被截断 |
| Linear | 0 | 研究用：1 = 送线性光 (RGBA16F + IsHDR) |
| DumpFrame / DumpCount | -1 / 1 | 测试用：从第 N 帧起连续存 `dlss5fb_in/nr/out(_k).ppm` 与光流 `dlss5fb_flow_k.bin` (也可用环境变量 `DLSS5FB_DUMP`) |
| MvConstX / MvScale | 0 / 1 | 研究用：用常数水平运动矢量代替光流 / 缩放运动矢量 |

## 构建与测试

```
plugin\fallback\build.bat     -> bin\dxgi.dll, bin\dlss5_nvngx.dll, bin\fbtest.exe, bin\fbtest11.exe
                                 (需要 DLSS SDK 头文件 oss\nvidia-dlss\include，不在本仓库里)
fbtest.exe image.ppm [帧数] [--pan N ...]   最小的 D3D12 "游戏"：每帧把图片拷进后缓冲再 Present (--pan: 每帧右移 N 像素)
fbtest11.exe image.ppm [帧数] [--model flip|blt] [--legacy] [--readback out.ppm ...]   同样的 D3D11 "游戏"
                              (--readback: 最后一次 Present 之后从 D3D11 这边读回后缓冲，确认处理结果真的拷回来了)
run_tests.ps1                 25 项回归: D3D12 16 项 (格式、HDR scRGB / HDR10、改大小、重建、多交换链、光流、场景切换、
                              精确运动矢量 = nr-lab) + D3D11 9 项 (blt / 翻转 / 老式创建、bgra8 / sRGB / rgb10、改大小、重建、光流)
mv_check.ps1                  运动矢量校验: 与 nr-lab 的正确 / 零运动矢量参考比较
motion_flicker.ps1            运动中的闪烁 (运动补偿后的帧间差)，各模式对比
flicker.py                    静止镜头的帧间闪烁 (DumpCount 存的连续帧)
color_stats.py                DLSS5 对颜色的影响 (饱和度、Lab 彩度、色相、冷暖、细节)
fbtest --cut N image2.ppm     第 N 帧起换成另一张图 (场景切换测试)
fbtest --hdr scrgb|hdr10      HDR 交换链 (--paper-white 尼特，--hdr-gain 倍数造高光)；HDR 的转储是 .f16 / .pq 原始数据
godot_test/                   Godot 4 (DX12) 场景：Crytek Sponza (fetch_assets.ps1 取，不进仓库)
```

RTX 3060 上的验证 (fbtest + nr-lab 的合成测试图)：

| 设定 | 结果 |
|---|---|
| 640x360，WorkingScale 1 | NR 输出与 nr-lab 的 `research/out_f0.ppm` **逐字节一致** |
| 1920x1080，WorkingScale 1 | 与 nr-lab 的 1080p 输出逐字节一致；每帧多 40 ms |
| 1920x1080，WorkingScale 0.5 | 每帧多 12 ms；改动量与全分辨率的相关 0.82、幅度 0.94 |
| 运行中改 Style / Temporal | 约 1 秒内重读 ini 并重建 feature，不中断 |
| 平移图案 + 常数精确运动矢量 (-4) | 第 1-3 帧与 nr-lab 的 `out_mvok_f*.ppm` **逐字节一致** (历史帧路径正确) |

运动中的闪烁 (Sponza 截图每帧平移 4 像素，640x360 全分辨率，运动补偿后的帧间差):

| 模式 | 帧间差 | 跳变 >8/255 |
|---|---|---|
| 每帧重置 (无历史) | 3.30/255 | 11.7% |
| 历史 + 零运动矢量 | 2.66/255 | 6.2% |
| 历史 + 光流 (中等，无细化) | 1.89/255 | 1.5% |
| 历史 + 光流 (中等) + Lucas-Kanade | 1.13/255 | 1.4% |
| **历史 + 光流 (最好) + Lucas-Kanade (默认)** | **0.74/255** | **0.4%** |
| 历史 + 精确运动矢量 (上限) | 0.58/255 | 0% |

Godot Sponza 1080p、模型分辨率 0.45：无 DLSS5 约 123 fps，每帧重置约 50 fps，光流历史约 42 fps (+3.7 ms)。
静止镜头 40 帧帧间差 0.12/255 (输入本身 0.04)。真实游戏还没测过。
