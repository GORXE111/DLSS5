# 十月 Small FP8 在 RTX 3060 上的整网实测

2026-11-06。官方 310.9.16.0 预设 2 的隔离实验版在 RTX 3060（12 GB，驱动 610.74）上成功创建并完成 D3D12 推理。以下结果来自离线合成输入。实验 DLL 使用了 15 个双架构模块和显式的 Ampere 选择器，试验构建可从原始输入复算。[构建复现](october_nr_stage_replay.md) · [实测数据](validation/october_nr_ampere_d3d12.json)

## 整网与权重

| 检查 | 实测 |
| --- | --- |
| 常规捕获 | 基本套件 3/3、遮罩套件 10/10 通过。包含 0–3 四种混合模式、首帧和第 4 帧，以及缺少显式遮罩、关闭自动遮罩两组对照 |
| 保留原链的真实提交轨迹 | 4 帧，每帧 31 个逻辑块、78 次网络提交、38 条跨块指针连接和 20 组窗口偏移。另各有一次缓冲清理与最终拷贝。记录前后 PPM 和 RGBA16F 输出哈希完全相同 |
| GPU 权重回读 | 从首帧 pre 参数 `+232` 所指的 113,907,200 B 对齐缓冲回读。77 条记录的 **113,887,490 B 有效载荷**各自与原 DLL 台账的 SHA‑256 一致 |
| 真实 ViT 重排 | 首帧序号 19 和 60 各读回两端缓冲。两方向各 98,304 B，逐字节与 CPU 依据同次输入预测的输出一致 |

这四帧的两次重排和每帧 8 次注意力调用都使用 **8×12 个 token**。RTX 3060 实际选中的是 chained FP8 attention。注意力参数 `+32` 收到了 8 条 `layer3.layer` 记录的 GPU 地址，对应此前未解释的 16 B。显存里的字节与原件一致。已追踪的普通和 chained FP8 PTX 都未读取该参数，序列化数据的原始语义仍待确定。

原版资源内容、实际上传内容、运行期指针绑定和数值等价是不同的验收项。前三项已在此实验场景逐项核验。独立参考工具也能按新图绑定 31 块和 77 条记录，并通过 CPU 构造器读取检查。它复用了旧算子，整网数值仍待原生张量对照。[参考骨架](../tools/october_nr_small_reference.py) · [静态图和字段台账](october_nr_small_execution.md)

## 遮罩具体改了哪些量

两个十月版 pre 内核的原 PTX 遮罩片段在 3060 上各跑了 450 组有限输入，合计 **6,300 个 FP16 输出逐位一致**。片段输入是已采样的 G/B，保留 272 B 参数块和 `+40` 模式位置。实际 D3D12 推理另通过每帧的有符号 32 位参数 setter/getter 日志，核对了应用请求的值。[片段结果](validation/october_nr_mask_slices.json)

- **G** 乘以基础 LocalTone。**B** 乘以基础 LocalStructure，得到同一个显式强度，分别与有效的自动 Skin、Structure 做平均、取小或取大。缺显式遮罩、关 AutoMask 或自动参数同时失效时，会走各自的分支。
- 当 G=0.25、B=0.75，基础 Tone=0.75、Structure=0.5，有效自动 Skin=0.25、Structure=0.75 时，LocalTone 四种模式都是 0.1875。打开自动遮罩后的模式 1、2、3 分别产生 Skin/Structure `0.3125/0.5625`、`0.25/0.375`、`0.375/0.75`。模式 0 的显式遮罩测试里，StructureGate 为 0.375。
- 内部选择器 `+196` 大于 15 时，三个槽按 `0, Skin, Structure` 写入，否则按 `Skin, Structure, 0` 写入。这是原始字段的阈值，公共 `Style` 到该内部字段的映射另行取证。

完整 640×360 scRGB 场景里，模式 2 对模式 0 的线性 RGB 平均绝对差在首帧为 **0.01094**，第 4 帧为 **0.01066**。第 4 帧按画面从左到右等分，三段平均差分别为 **0.01530、0.01001、0.00666**。本例使用从左到右递增的 B 遮罩。统计单位是原始 scRGB 数值，页面热图使用 8 位显示预览。[逐模式和局部统计](validation/october_nr_ampere_d3d12.json)

## 4K 画面和覆盖限制

3840×2160 输入及同尺寸输出下，强度 0 和 1 的两次首帧捕获均覆盖全画面。两张 8 位 RGB 预览的平均绝对差为 **12.381/255**。本地查看器提供这一对 4K 画面、640×360 强度对照，以及模式 0 对 2 的首帧与第 4 帧，均可拖动滑块、切换热图及按原始像素查看局部。[查看器使用说明](nr_viewer/README.md)

1920×1080 输入直写 3840×2160 的独立 NR 测试虽然返回 Evaluate 成功，覆盖检查只测得 **25%**，右上、左下和右下仍为哨兵像素。后续通过 CUDA 提交确认了写入范围，并验证紧凑 NR 串 SR 与 DXGI 回写两条完整 4K 路径。[4K 接入实测](october_nr_render_pipeline.md)

## 重放入口

先按[构建复现](october_nr_stage_replay.md)生成实验 DLL。将 `$staged` 指向已验证的 DLL，`$harness` 指向用当前 `harness/nr-lab.cpp` 编译的可执行文件。在比较 RTX 50 原版和实验版时，两边使用相同的 lab、桥接 DLL 和辅助 DLL。

```powershell
python -B harness/reference.py capture --kind experimental-sm86 --nr-version 310.9.16.0 --model-preset 2 --suite october-mask --dll $staged --harness $harness --out _dl/october_mask_ampere
python -B harness/reference.py capture --kind official-sm120 --nr-version 310.9.16.0 --model-preset 2 --suite october-mask --dll $original --harness $harness --out _dl/october_mask_blackwell
```

轨迹及显存取样各使用独立的 lab 运行目录，保存 DLL 哈希、参数、输出和日志。普通轨迹只记录 CPU 提交，保持原调用的链和参数。额外的显存取样会插入 CUDA 拷贝，单独执行并和普通输出比对。以下配置使用此次 640×360、预设 2、Min、第 4 帧用例，在当前 lab 参数不变时可复算。先从 `october-mask` 捕获的 manifest 取 `blend2_f3` 的 `args`。显存取样需要由 `harness/tap.ptx` 编译的 `tap.cubin` 放在 lab 同目录。

| 运行 | 环境设置 | 从仓库根目录检查 |
| --- | --- | --- |
| 原链轨迹 | `DLSS5_TRACE_ONLY=1` 和 `DLSS5_TRACE=-2` | `python -B -m tools.align_nr_small_trace PATH/nr-trace.tsv --expect-mask-mode 2 --require-weight-bindings` |
| ViT 首帧缓冲 | `DLSS5_PROF=1`、`DLSS5_TRACE=-2`、`DLSS5_TAP=0;19:p0:98304;19:p1:98304;60:p0:98304;60:p1:98304` | `python -B -m tools.ptx86.check_october_live_repack PATH` |
| 首帧权重堆 | `DLSS5_PROF=1`、`DLSS5_TRACE=-2`、`DLSS5_TAP=0;1:p29:113907200` | `python -B -m tools.ptx86.check_october_live_weights PATH` |

两个取样检查器要求同次运行的 `stdout.log`、`nr-trace.tsv`、`tap_index.tsv` 及索引对应的缓冲。保留原版、实验版和所用本地缓冲以便复查，公开记录只包含元数据和哈希。

## 下一道验收

在 RTX 50 用原版预设 2 采同一批输入。先对齐 pre 的真实张量和首末 ViT 的 QKV、attention、投影，再扩到 31 块的中间张量。每层分别报告逐位相等、误差位置和边界条件。速度应以独立的无取样运行测量。至此才能确认当前 Ampere 计算在什么范围内与原版数值等价。
