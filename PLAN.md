# 推进计划 (不依赖真实游戏的部分)

更新: 2026-10-09。每个阶段做完: 验收 → 提交推送 → 在本文件勾掉并写一行结果。按顺序推进，前一阶段不过不进下一阶段。

固定约束: 不启动联网游戏、不碰 E:\ZZZ；NVIDIA 的 dll / 权重 / PTX 等不进仓库；测试图片只在本地用。

---

## A. 兜底模式加固 (测试程序里能覆盖的情况)

现在只验证了最简单的路径 (RGBA8、窗口、单交换链、Present)。真实游戏会遇到更多情况。

- [x] A1 `fbtest` 增加选项: 后缓冲格式 BGRA8 / RGB10A2 / RGBA16F，`Present1`，运行中改窗口大小 (ResizeBuffers / ResizeBuffers1)，
      全屏切换，帧延迟等待对象 (FRAME_LATENCY_WAITABLE_OBJECT)，两个交换链同时存在，交换链销毁后重建
- [x] A2 钩子修正: 交换链释放时清掉记录 (现在是泄漏 + 地址复用风险)，设备移除 (DEVICE_REMOVED) 时停用而不是崩，
      GPU 资源 / NGX feature 的释放顺序
- [x] A3 每种情况跑 300 帧以上: 不崩、日志无错误、DumpFrame 的输出与 nr-lab 一致 (格式换算后)

验收: A1 全部情况通过；HDR 格式按设计"跳过并写日志"。

**结果 (2026-10-09)**: `run_tests.ps1` 11/11 通过 (RGBA8 / BGRA8 / RGB10A2 的 NR 输出都与 nr-lab 逐字节一致；RGBA16F 原样放过；Present1、ResizeBuffers / ResizeBuffers1、销毁重建、等待对象、两个交换链、0.5 倍)。新增: 交换链最后一次 Release 时清理、设备移除时停用、改大小时等尺寸稳定 250 ms 再重建 (拖窗口不会反复建 feature)。独占全屏切换没跑 (会让你的显示器切模式)，留着 `-Fullscreen` 选项，进 Godot 测试时一起看。

## B. 真实引擎测试: Godot 4 (DX12)

你本机的 Godot 是源码自编译版，没有编进 DX12；官方发布版带 DX12。

- [x] B1 下载 Godot 4 官方 Windows 版 (GitHub 官方发布，约 100 MB，放 `_dl/godot`，不进仓库)
- [x] B2 写一个测试工程: 带贴图和光照的 3D 场景、相机自动移动、屏幕上有 UI 文字，固定运行 N 秒后退出 (工程本身可以进仓库)
- [x] B3 `--rendering-driver d3d12` 运行，兜底模式挂上: 截取若干帧看效果 (我直接看截图)、对比开关前后的帧率、
      运行中改窗口大小 / 全屏、改 ini 热更新
- [x] B4 记录问题并修复，回到 B3 直到稳定 (独占全屏除外: 会占用你的屏幕，等你在场时用 `--fullscreen-at` 测)

**结果 (2026-10-10)**: 1080p 起跑 140 秒，每 9 秒在 1280x720 / 1600x900 / 1366x768 / 1920x1080 之间改一次窗口 (14 次)，
同时热改 ini (Enabled 0→1、WorkingScale 0.45→0.6→0.45、ColourStrength、Intensity、FlowPerf、Temporal)，每项都在 1 秒内生效，
改大小时 feature 与光流跟着重建；退出码 0，日志无错误。改大小后的输出与输入对齐 (最佳偏移 0,0)，无黑边。
帧率 (模型分辨率 0.45，光流开): 1080p 约 43 fps、900p 约 55、768p 约 62、720p 约 65；关闭时 1600x900 约 260 fps。
场景切换只在镜头 180° 掉头处触发。

进展 (2026-10-09): 官方 Godot 4.7.2 (校验 SHA512) + `godot_test/` (Crytek Sponza，`fetch_assets.ps1` 取，不进仓库；
太阳阴影 + SDFGI + 体积雾 + 走廊漫游 + FPS 文字)。窗口放在屏幕外 (`--position -4000,-4000`) 跑，屏幕上什么都不出现；
官方 `--headless` 不渲染 (64x64 假视口、不建 D3D12 设备)，你的引擎的 `--render-worker` 是编译时 `d3d12=no`，所以暂时用屏幕外窗口。
1080p、模型分辨率 0.45: 无 DLSS5 约 120-170 fps，有 DLSS5 约 49 fps (+12~14 ms)，每帧都处理。
观察: DLSS5 让画面偏冷、饱和度下降 (平均改变 11/255)。管线已与 nr-lab 逐字节核对，这是模型对"已色调映射的 sRGB 画面"的反应；
模型可能期望线性 HDR 场景色 → 列入 D/E 研究 (见 E0)。早先"处理到 120 帧就停"是窗口被别的窗口挡住时 Godot 停止 Present，不是钩子问题。

验收: 连续运行 2 分钟以上不崩；开销与 fbtest 的测量一致 (1080p、模型分辨率 0.5 约 +12 ms)；截图里无明显错误 (错位、色偏、黑边)。
注: Godot 的 FSR2 是编在着色器里的，OptiScaler 截不到，所以 OptiScaler 方式仍需要一个真实游戏来测。

## C. 兜底模式的运动矢量 (光流)

没有运动矢量就不能用历史帧，画面会闪。RTX 30 有硬件光流 (驱动自带 `nvofapi64.dll`，已确认存在)。

- [x] C1 调通 NVIDIA Optical Flow 的 D3D12 接口: 两帧 → 光流图，测 1080p / 540p 的耗时
- [x] C2 确定 DLSS-NR 要的运动矢量格式与方向 (单位、正负、指向上一帧还是下一帧)，用 nr-lab 的已知运动数据核对
      (research/out_mvok_* 是"正确运动矢量"的参考输出)
- [x] C3 接进兜底模式: 光流 → 运动矢量纹理，打开历史帧；画面突变 (切场景、开菜单) 时自动 Reset
- [x] C4 验证: fbtest 的平移图案下，"光流运动矢量"的输出接近"真实运动矢量"的输出；静止/运动画面的帧间闪烁
      比每帧 Reset 的模式小。Godot 场景里再看一次

验收: C4 的两项数字成立；光流开销 ≤ 3 ms (1080p)。

**结果 (2026-10-09)**: 运动中闪烁 3.30 → 0.74/255 (精确矢量的上限 0.58)，跳变 >8/255 从 11.7% → 0.4%；静止 0.12/255。开销 +3.7 ms (略超 3 ms 目标，换来最好质量档；fbtest 里中等档约少 1 ms、闪烁 1.13/255)。细节见 plugin/fallback/README.md。场景切换的自动 Reset 未做 (网络自身的门控处理历史不匹配)。

## D. 模型的 UI 相关输入 (逆向研究)

DLSS-NR 有几个还没研究的可选输入: `UI`、`UIAlpha`、`UICorrection`、`ControlMask`、`Backbuffer`、`BidirectionalDistortionField`。

- [x] D1 用 nr-lab 的可选输入探针 (`--optional-probe` 已有) 加执行轨迹，查清每个输入进了哪个 kernel、在网络前还是网络后
- [x] D2 确定语义: 是否能让模型避开 / 还原 UI 区域，ControlMask 是否是逐像素的效果强度
- [x] D3 结论写进 `tools/notes.jsonl` 与 `torch/` (如果进网络就在 torch 版里复现)；判断兜底模式能不能用 (例如从画面估计 UI 遮罩)

验收: 每个输入都有"用途 + 证据"；给出兜底模式用或不用的结论。

**结果 (2026-10-10)**: nr-lab 加了 `--optional-value / --optional-rgba / --ramp-channels / --bundle / --bundle-backbuffer`
(常数、方块、横向渐变，逐通道取值)，640x360 合成图逐项对比，并对比执行的 CUDA kernel:

| 输入 | 用途 | 证据 |
|---|---|---|
| `ControlMask` (RGBA) | 逐像素画面参数，与全局参数**相乘**: R x Intensity (网络之后混合)，G x LocalTone，B x LocalStructure (进 pre_block 的控制输入)，A 未用。提供即强制 UseAutoMask=0 | 与相应全局参数逐字节一致 (G=0.5 ≡ LocalTone 0.5；G=0.5 且 LocalTone 0.5 ≡ LocalTone 0.25)；R 渐变线性；post_block 换成 `..._control_mask_full_rect` 变体；torch 版加 `control_mask=` 后与 nr-lab 吻合 0.5-1.9/255 |
| `UICorrection` + `UIAlpha` (或 `UI` 的 alpha) + `Backbuffer` | 网络之后的合成: out = Backbuffer + (1 - α)·(NR - Color)；无 Backbuffer 时 = lerp(NR, Color, α)。UI 的 RGB 不用 | 公式与输出逐像素差 0.01/255；α=0.5 时在一半处；多跑 `cg2r_post_process_kernel` (Style 调色用的同一个)；网络输入不变 (其余区域变化 ≤0.07/255) |
| `UI` / `UIAlpha` / `Backbuffer` 不开 UICorrection | 无作用 | 输出与不给逐字节相同 |
| `BidirectionalDistortionField` | 此模型不用 | 每帧都读取参数，但常数/方块/值 8、带历史的 4 帧平移下输出逐字节不变，也没有多出的 kernel |

兜底模式的结论: **不用**。UICorrection 就是兜底模式 Combine 已经在做的"原画面 + 变化量"，还需要游戏给出无 UI 画面和 UI 透明度；
ControlMask 能让某些像素不受影响，但网络照样看到整张画面 (UI 变化引起的全局波动不会消失)，而且会关掉 AutoMask (默认画面变 2.5/255)。
将来如果从画面估计出 HUD 区域，直接在 Combine 里按区域减弱变化量更便宜。OptiScaler 模式下如果游戏提供无 HUD 画面 + UI，
UICorrection 可以直接用 (OptiScaler 那边的事)。

## E. 兜底模式支持 HDR

- [x] E0 同一画面分别以 sRGB (现状) 与线性 HDR (+IsHDR) 送入，比较色彩变化；决定兜底模式送哪种
      → 结果: 线性输入让细节增强几乎消失 (x1.003)，sRGB 才是模型要的。"偏冷"是模型随内容的局部色调调整 (有的帧偏暖)，
        不是错误；加了 ColourStrength (0 = 保持游戏颜色)。另做了场景切换检测 (GPU 上运动补偿差，切换那帧显示原画面)。
- [x] E1 在 nr-lab 里确认 HDR 输入 (scRGB 浮点 + IsHDR 标志、HDR10) 的正确用法和输出范围
- [x] E2 兜底模式按交换链格式与色彩空间选择: 浮点 scRGB 直接送、HDR10 先转线性再送，结果转回
- [x] E3 fbtest 加 HDR 交换链验证

验收: HDR 交换链被处理，亮部 (>1.0) 不被截断。

**结果 (2026-10-10)**: E1 — DLSS-NR 没有真正的 HDR 路径: 开 IsHDR、scRGB 输入 x2/x4/x16 时输出最大值都是 1.0 (截断)，
同样数值下 IsHDR 开/关只差 1.9/255，HDR10 输入与 scRGB 相同 (nr-lab 新增 `--hdr-scale` 与原始 f16 输出)。
E2 — 兜底模式自己把 HDR 变成模型能用的画面: 线性 Rec.709、纸白 = 1 (HdrPaperWhite 默认 200 尼特)，0.8 以上平滑压缩，sRGB 编码；
结果 = 原 HDR 值 + 压缩域里的改变量，再编回 scRGB / HDR10 (宽色域的负值原样保留)。FP16 交换链没设色彩空间时按 scRGB。
E3 — fbtest `--hdr scrgb|hdr10 --hdr-gain`: x1 时与 SDR 结果相差 1.6/255 (拐点以下)、改变量 10.6 vs SDR 10.4/255；
x4 时输入最高 4.0 → 输出 4.0，高光区平均变化 3%，无 NaN；run_tests 16/16 (SDR 的逐字节项不变)。管理器加 HdrPaperWhite。

## F. 移植版提速 (可选，研究性质)

目前 1080p 41.9 ms，离 3060 硬件上限约 2–3 倍。

- [ ] F1 重新做一次逐 kernel 计时，找出现在占比最大的部分
- [ ] F2 只做不改变结果 (校验和不变) 的优化；改变数值的方案先在 torch 版里评估画质损失

验收: 校验和 5B3A70119F21C9EE 不变的前提下有可测的提速，或者给出"不值得继续"的结论。

---

## 需要你的时候

- 有了带 DLSS / FSR2 / XeSS、单机、无反作弊的游戏时: 测 OptiScaler 方式
- B1 下载 Godot 官方版、C1 用到 NVIDIA Optical Flow SDK 头文件时，我会先说一声
