# DLSS-NR 310.9.16.0 Small FP8 执行研究

2026-11-06。研究对象是 NVIDIA 官方 RTX Remix 1.6.1 中的 `nvngx_dlssnr.dll`，原件 SHA-256 为 `2bc19bc09a7b89dd44f73e2ac0108fcc586e2154bdd0830f242f4f040e28789f`。以下分别记录静态重建、RTX 3060 实测和仍需原版运行才能确认的项目。公开仓库只保存分析工具、地址、尺寸、哈希和测试结果。[原件及版本调查](october_nr_remix.md)

## 小模型怎样执行

310.9.16.0 的主机工厂把预设 **2** 绑定到 `WEIGHTS_CC_SMALL_FP8`，预设 **1** 绑定到原有的 `WEIGHTS_HT`。官方 Remix 固定提交里的 Vulkan 调用显式选择预设 B，也就是 2。画面风格使用独立的 `Style`，AutoMask 开启时 Remix 另把遮罩混合设为 Min。[固定版本的官方调用方](https://raw.githubusercontent.com/NVIDIAGameWorks/dxvk-remix/0a6cd5b1ee814ec390c358e0c9c32ece8895ddc0/src/dxvk/rtx_render/rtx_ngx_wrapper.cpp)

| 逻辑块 | 原版主机构图中的职责 |
| --- | --- |
| 0 | 融合预处理与第一层 Swin，入口接收颜色、历史帧、运动向量、遮罩及控制参数 |
| 1–8 | 四个编码尺度（1、2、4、8 个注意力头），每级 2 块 |
| 9–10 | 16 头的拆分编码级，随后接入中段 |
| 11–18 | 8 个 ViT1D。块 11 的 FFN 前进行 2D→1D 重排，块 18 的投影后进行 1D→2D 重排。块 12–18 各接收前块的输出槽 1 |
| 19 | 1024→512 通道的桥接，读取 `18:0` 和 `10:1` |
| 20–21 | 16 头的拆分解码级 |
| 22–29 | 四个解码尺度，每级 2 块。上采样块 22、24、26、28 分别另接编码块 8、6、4、2 的输出槽 1 |
| 30 | 后处理，读取 `29:0` 与 `0:1` |

旧大档的五级重复深度是 `[4,4,6,8,8]`，小档是 `[2,2,2,2,2]`。小档各编码级的窗口位移 `(y,x)` 为 `(0,0),(-4,-4)`，各解码级为 `(0,-4),(-4,0)`，两者都包含拆分的 16 头级。按原 DLL 的构造、端口、分支和 PTX 核对到 **14 个阶段、31 个逻辑块和 77 条权重记录**。运行中实际发射的网格、同步分支和发射次数仍需跟踪。[主机构图及地址证据](validation/october_nr_small_host_graph.json)

## 权重掌握到什么程度

| 量 | 核验结果 |
| --- | ---: |
| 完整 PE 资源 | 113,893,343 B |
| 77 条记录的载荷 | 113,887,490 B |
| 依旧版构造器与新版记录匹配的字段 | 113,886,522 B |
| E4M3 / FP16 / FP32 存储槽 | 112,799,744 / 542,625 / 382 |
| FP16 已计入的全零非活动槽 | 416 个 |
| 未归属 | 968 B，含 952 B 全零间隙及 8 条 ViT 注意力记录各 2 B 独立载荷 |

CPU 上用 NumPy 跟踪旧版构造器的实际读取范围，31 块均可构造，77 条记录的字段范围和类型均吻合。按这些类型解码的存储槽全部有限。对可检查的 126 个展开注意力偏置头，按旧版布局归并成每头 225 个二维相对位移值时，0 个头逐值满足约束。偏置共享规则、训练期张量形状和未归属字节的语义因此仍留在账上。NVIDIA 公布的约 1.18 亿是模型参数口径，当前恢复的数字是序列化存储槽数。学习参数总数仍待核实。[权重字段台账](validation/october_nr_small_layout.json) · [NVIDIA 十月公告](https://www.nvidia.com/en-us/geforce/news/dlss-5-faster-ai-model-hogwarts-legacy-rtx-spark-laptops/)

块 11–18 的 `layer3.layer` 各有一条 2 B 非零载荷。已检查普通和 chained 两个 ViT1D FP8 attention 的 PTX 参数读取，均未见这些 2 B 对应的权重输入。主机构造器在 `0x180064dc0` 也没有给 ViT 的 layer3 注册命名权重槽。它们在其他路径中的用途仍未确定。

## 遮罩与宿主

- `DLSSNR.MaskBlendMode` 已从 Evaluate 的取值一路追到主机状态 `+0xa8`，再到 272 B 的 pre 内核参数块的 `+40`，与原 PTX 的读取相符。模式 0 保持显式 ControlMask 与 AutoMask 的旧互斥逻辑。两路都有效时，模式 1、2、3 分别取平均、最小和最大。四种模式的逐像素输出尚待原版验证。[参数传递证据](validation/october_nr_small_host_graph.json)
- 15 个 CUDA 模块的宿主绑定均已定位，含 13 个 RIP 相对地址及对应长度、2 个 DIR64 绝对指针重定位。新版模块 1–7 的长度项为 8 B，8–13 为 4 B，14–15 走独立描述符。架构检查和导出入口地址也有变化，旧版补丁偏移不能沿用。[宿主移植证据](validation/october_nr_port_readiness.json)
- 原版宿主的静态架构分支将 Ampere 导向不支持路径。RTX Remix 1.6.1 公布的驱动基线是 616.92，本机为 610.74。裸 NR DLL 的最低驱动版本仍待单独验证。[官方版本说明](https://docs.omniverse.nvidia.com/kit/docs/rtx_remix/1.6.1-0/docs/changelog/remix-releasenotes.html)

## RTX 3060 已跑出的结果

| 验证 | 实测结果 | 覆盖边界 |
| --- | --- | --- |
| 模块 6 装载 | 实验 sm_86 cubin 装载成功，31 个符号均可解析 | 符号解析本身只证明可装载 |
| 跨 CTA 同步微测 | 同作用域 `acq_rel` 与较强的 `sc` 对照各检查 524,288 个值，均 0 错误。错值注入和不发布两组负对照均被识别 | 自写探针复现 flag 等待、CTA barrier、GPU fence 和读回。原模型的 21 处调用在真实调度下仍待验证 |
| 原十月版两个重排入口 | 用试编译 cubin 分别执行双向重排，4 个尺寸 × 2 个随机种子 × 2 个方向。共 1,441,792 B，和独立 NumPy 索引计算逐字节一致，输入及两端哨兵均完整 | 覆盖 ViT 边界布局，未涉及模型权重、注意力或整网输出 |
| 原版捕获配置 | 25 项 CPU 测试通过，旧版既有 52 例 manifest 可加载。3060 上请求新版原生捕获时，包装器在创建输出和运行原 DLL 之前拒绝该硬件 | RTX 50 成功运行及实际权重资源加载仍需取证 |

[RTX 3060 原始测试结果和交叉校验](validation/october_nr_sm86_primitives.json) · [NVIDIA PTX fence 规范](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar)

## 本地复算与验收

先按[原件调查](october_nr_remix.md)准备固定版本 DLL 及本地提取文件。命令在仓库根目录运行，分析用二进制放在 Git 忽略的目录。实测以下构建能复现两份 PTX 和 cubin 的 SHA-256，GPU 探针也会核验试验 cubin 的身份。[编译输入及产物哈希](validation/october_nr_remix_compile.json)

```powershell
$lab = '_dl/october_nr_20261106'
$oct = "$lab/original/nvngx_dlssnr.dll"
$ptx = "$lab/candidate_extract/fatbin_06.1.sm_120.ptx"
$replay = "$lab/replay_sm86"
New-Item -ItemType Directory $replay
python -B -m tools.ptx86.rewrite $ptx "$replay/module06.sm86.ptx"
python -B -c "from pathlib import Path; import sys; src, out = map(Path, sys.argv[1:]); data = src.read_bytes(); assert data.count(b'fence.acquire.gpu;') == 21; out.write_bytes(data.replace(b'fence.acquire.gpu;', b'fence.acq_rel.gpu;'))" "$replay/module06.sm86.ptx" "$replay/module06.experimental.ptx"
$trial = "$replay/module06.experimental.cubin"
& cuda_tools/bin/ptxas.exe -arch=sm_86 "$replay/module06.experimental.ptx" -o $trial

python -B -m tools.inspect_nr_small_host_graph
python -B -m tools.inspect_nr_small_layout --numpy-constructors
python -B -m tools.inspect_nr_port_host --baseline dlss5/nvngx_dlssnr.dll --candidate $oct
python -B -m tools.ptx86.probe_october_fence --rounds 4096 --october-ptx $ptx --trial-cubin $trial
python -B -m tools.ptx86.probe_october_repack --trial-cubin $trial
```

在 RTX 50 和兼容的 NGX 环境中，先用独立目录捕获原版两个预设。捕获工具会验证 DLL、硬件、请求预设和每例输出，实际加载的权重仍要通过独立的运行追踪确认。

```powershell
python -B harness/reference.py capture --kind official-sm120 --nr-version 310.9.16.0 --model-preset 2 --suite smoke --dll $oct --harness harness/batch3/nr-lab.exe --out "$lab/native-small-smoke"
python -B harness/reference.py capture --kind official-sm120 --nr-version 310.9.16.0 --model-preset 1 --suite smoke --dll $oct --harness harness/batch3/nr-lab.exe --out "$lab/native-large-smoke"
```

1. 原版小档先确认资源身份，再记录首帧、持续帧、Reset、遮罩 0–3 及中段张量。把实际 CUDA 发射顺序、网格和选中的内核与静态图逐项对照。
2. 用新版宿主映射和参数合同单独建立 310.9.16.0 的 sm_86 验证目标。先比 pre、8 个 ViT 块的边界及其同步点，再比整网和时序画面。新版产物需要登记独立的版本与身份。
3. 用同一输入、同一预设、同一帧对比原版与移植结果，分别记录逐值差异、画面热图和速度。现有 4K 查看器可以复用，输入必须来自新模型实测。[同帧查看器](nr_viewer/README.md)

当前已能复核原版小档的逻辑连接、字段级存储、遮罩参数和宿主模块地址，也能在 RTX 3060 上复现两段真实的边界重排。整网逐层等价、四种遮罩的真实画面和完整性能数据仍是验收缺口。
