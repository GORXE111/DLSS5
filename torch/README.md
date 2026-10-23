# DLSS5 PyTorch 参考实现

`nvngx_dlssnr.dll` 310.8.0 (DLSS5 NR) 的独立主网络参考，使用该版本原始权重。更新：2026-10-23。

覆盖主网络和已验证的同尺寸后处理；参考对象是 RTX 3060 的 sm86 移植版，不是 RTX 50 原生输出。默认与精确模式都不保证逐值一致。
采样修正前的历史测量：RTX 3060 上一帧 640x360 约 190 ms、1280x720 约 0.75 s、1920x1080 约 1.6 s (默认快速模式 + CUDA Graph)，峰值显存 1080p 3.4 GB。第三、四批未重新做稳定性能测量。

```python
import sys; sys.path.insert(0, "torch")
from dlss5 import DLSS5

net = DLSS5()                                  # 读 WEIGHTS_HT.bin + weights_map.json，解码全部权重到 GPU (~5 s)
                                               # DLSS5(precise=True): 逐处模拟 f16 舍入、f32 计算 (逐块对照用，慢 1.7 倍)
out = net(color)                               # 重置帧: color (360, 640, 3) in [0,1]
out = net(color, hist=out, mv=mv, frame=1)     # 带历史: hist = 上一帧输出, mv (H, W, 2) 像素位移
```

输入尺寸 (H, W) 的补齐网格按 DLL 的规则 `net.grid_for(H, W)` (从 `CCNetwork::SetResolution` 反汇编得到，见下文)。

DLSSNR 的画面参数 (默认值与 DLL 相同):

```python
from dlss5.net import control_inputs
ctl = control_inputs(tone=1.0, structure=1.0, skin=-1.0, auto_mask=True, style=0)   # LocalTone/LocalStructure/SkinStructure/UseAutoMask/Style
out = net(color, controls=ctl, intensity=1.0)  # intensity: 夹到 [0,1]，lerp(color, NR, t)
# DLSSNR.ControlMask (H, W, 4): R x Intensity, G x LocalTone, B x LocalStructure 逐像素相乘, A 未使用; 要求 auto_mask=False
out = net(color, controls=control_inputs(auto_mask=False), control_mask=mask)
```

```python
run = net.graph(history=True)                  # 可选: CUDA Graph (输入尺寸固定)，输出为静态缓冲
out = run(color, hist=prev, mv=mv, frame=5)
```

```
python check.py           # 端到端: nr-lab 第 0 帧合成输入 -> 与 3060 上 nr-lab 的实际输出比较 + 计时
python check_res.py       # 多分辨率: 540p / 720p / 1080p 与 nr-lab 比较 (需要 research/out_f0_<W>x<H>.ppm)
python check_motion.py    # 运动画面: 图案逐帧平移 + 正确/错误运动矢量，与 nr-lab 逐帧比较
python check_frames.py    # 多帧时域累积: 连跑 4 帧与 nr-lab 逐帧比较 (需要 research/out_f0..3.ppm)
python check_levels.py --capture ../research/batch3/schedule_capture    # 逐级: 从明确的 pre 输出开始比较
python check_blocks.py --capture ../research/batch3/schedule_capture    # 逐块: 只读已校验的精确缓冲
python check_numerics.py --capture ../research/batch3/schedule_capture --levels 4h 8h  # 选项扫描，本批未重跑GPU
python check_controls.py  # 画面参数: 每组 DLSSNR 参数与 nr-lab 比较
python bench.py           # 按块类型计时
python bench_modes.py     # 精确 / 快速 / 快速+Graph 三种模式的耗时与吻合度
python profile_ops.py     # torch.profiler 按算子统计
```

## 历史验收测量

下表为采样与 Reset 修正前的测量，当前第四批结果以本页对应章节和 `research/validation/session_summary.json` 为准。

| 对照 | 相关 | 平均差 |
|---|---|---|
| torch vs nr-lab 实际输出 (第 0 帧) | 0.99931 | 1.9/255 |
| torch vs nr-lab (第 1-3 帧，带历史) | 0.99945-0.99947 | 1.6-1.7/255 |
| torch vs nr-lab (540p / 720p / 1080p，第 0 帧) | 0.9993-0.9994 | 1.8-1.9/255 |
| torch vs nr-lab (运动画面，第 1-3 帧) | 0.9993-0.9995 | 1.6-2.0/255 |
| 网络改动量 (输出 - 输入) vs nr-lab | 0.992-0.995 | |
| torch vs nr-lab (LocalTone/Structure/Skin/AutoMask/Intensity/Style 各设定) | 0.9990-0.9999 | 0.8-2.2/255 |
| torch vs nr-lab (ControlMask: R/G/B 渐变、中央方块、倍数 0.5/1.5) | 0.9992-0.9999 | 0.5-1.9/255 |

Style 1/2 的调色后处理 (`cg2r_post_process_kernel`) 在 `dlss5/style.py`：黑白场 → 冷暖/色调 → 曝光 → 对比度 →
5 段分区 gamma → gamma → HSL 饱和度 → vibrance，最后 `sat(输入 + Intensity·(调色(NR) - 输入))`。预设取自 DLL 常量区
(Style 1: 曝光 -0.1 EV、对比度 -0.25、饱和度 -10%；Style 2: 饱和度 -15%)。与单独运行的 kernel 逐分支对照
(`research/pp_check.py`) 最大差 < 0.001/255。该 kernel 另有一条 OKLab 迁移分支，独立实验 helper 已通过第二批专项；310.8.0 常规 NGX Evaluate 的静态调用点传空迁移 descriptor，不能将强制注参实验视为端到端启用，见下文。

## 2026-10-23 后处理补齐

同尺寸 `ControlMask + Style 1/2` 已补齐；`UICorrection` 支持 `UIAlpha`、`UI.a` 和 `Backbuffer`：

```python
out = net(color, controls=control_inputs(style=1, auto_mask=False), control_mask=mask)
out = net(color, ui_correction=True, ui_alpha=alpha, backbuffer=backbuffer)
```

`ui_alpha` 为 `(H,W)` 或 `(H,W,1)`，优先于 `ui[...,3]`；`ui` 为 `(H,W,4)`，`backbuffer` 为 `(H,W,3/4)`。不开 `ui_correction` 时三种可选输入均被忽略。
同尺寸后处理的关系为：`sat(base + sat(Intensity * Mask.R * (1-sat(alpha))) * (grade(NR)-Color))`，其中未提供的 Mask.R 为 1、alpha 为 0；仅启用 UICorrection 时 Backbuffer 才作为 base。G/B 在网络入口应用，R 在最终合成只应用一次。

新增 `check_postprocess.py` 对 8 个整网组合和 8 个隔离原后处理 kernel 作独立对照。已测整网 MAE 0.280–1.915/255，隔离后处理最大差 0.000289/255。这不消除主网络误差，也不证明 RTX 50 原生等价。

```powershell
python torch/check_postprocess.py --report research/check_postprocess.json
python torch/check_controls.py --report research/validation/check_controls.json
```

当前 `graph()` 只捕获固定参数的基础网络路径，不支持新增的 UI/ControlMask 动态接口。

## OKLab 迁移实验

`dlss5.style.transfer(nr, base, target, grid_size=(H,W), ...)` 独立实现原后处理 kernel 的迁移公式，返回调色/UI/遮罩之前的 RGB；可与 `style.apply()` 组合。它不启动 NGX，也未接入默认网络。`base`、`target` 分别对应原参数块 +224、+256，迁移开关为 +300，网格为 +304/+308；公开 NGX 启用键仍未建立。

第二批静态追踪已确定此 DLL 保留分支的 D3D12 采样器：`base_filter="linear", target_filter="point"`，两者均为 CLAMP。正常 Evaluate 的四组动态轨迹中迁移块均为零。独立 helper 继续允许显式选择过滤方式，已有混合过滤用例覆盖保留分支的组合；不将该研究能力接入默认网络。

```powershell
python torch/check_transfer.py --report research/validation/check_transfer_batch2.json
```

第二批在相同门槛（最大差 0.005/255、MAE 0.0005/255）下通过 34/34 项迁移用例和 17/17 项采样探针，最大差 0.003769/255。第一批的点采样子区域边界和大网格线性边界失败全部保留并修正，另扩展 6 项相邻边界/尺寸/混合过滤用例。独立 TEX 的 X/Y 共 6,156 个边界样本支持 RTX 3060 上归一化坐标截断为 21 位小数的测量模型；这不是跨 GPU 的规范保证。

旧失败报告 `research/validation/check_transfer.json` 保留；新报告关联旧报告及新旧代码 SHA，保存全部输入/输出哈希。禁用、暗部、奇数尺寸、子区域和调色/UI/遮罩组合均有原 kernel 对照。通过数不解释为一般正确率；这些局部结果仍不能证明 NGX 端到端启用或 RTX 50 原生等价。高层静态调用证据见根目录 PLAN。

## 第四批：Reset 与真实采样算术

`hist=None` 的重置帧直接使用当前 Color 作为历史输入值，并忽略显式非零 MV。raw `PreBlock` / `PostBlock` / 引擎外 `Core` 在 History 或 MV 任一缺失时跳过历史分支；公共 `DLSS5(..., mv=None)` 仍把 MV 补为全零，兼容原用法。Graph 的 `history` 状态固定，调用时 History 的存在性必须匹配；重置需使用 `history=False` 的独立图。

原 pre 颜色/历史六列先 `half()`，再分别作 half 减 0.5、乘 0.125，最后转回 float 以拼接噪声与控制列；快速和 precise 模式固定执行同样的入口舍入。`Core.inputs` 复用此入口。该修复关闭了最终集成首次出现的高控制项失败；历史失败 `batch4_controls.json` 和归因记录保留，没有改阈值或添加参数特例。独立 half oracle、原特征探针和时序对照见 `batch4_rounding_review.json` / `batch4_pre_round_fix.json`；部分时序误差略增的事实也保留。

完整原 sm120/sm86 pre 和 post 的采样中心/分数都使用 FFMA；第三批 pre 切片的人工非 FTZ setup 改变了融合条件。旧四组探针保留为 split 诊断，新增四组 `pre_native` 真实上下文；默认 `catmull_rom5(..., arithmetic="fused")`，`split` 仅供历史比较。

`dlss5/ptx_math.py` 用自写 PTX 调用当前 GPU 的 `rcp.approx.ftz.f32`，并保留原 RCP + FFMA 运算顺序。它无需 nvcc 或额外包，按 CUDA context 缓存模块，在当前 Torch stream 发射；Graph 前须预热，`net.graph()` 的预热会完成此步骤。模块保持至进程结束，CUDA 错误不会静默转 CPU。CPU 路径是普通 RN 倒数，只用于非硬件对照，不承诺相同结果。这里不包含 NVIDIA 提取代码或尺寸查表。

```powershell
python torch/check_sampling.py --report research/validation/batch4_sampling_rounded.json
python torch/check_reset.py --capture research/batch4/reset_capture --report research/validation/batch4_reset_fused_rounded.json
```

采样完整套件为 16 组探针（其中 4 组是 legacy split 诊断）、9 组完整 post、20 项整网及 Graph。完整 post 隔离权重/门控以研究采样，不是所有神经状态的全面验收。原门槛保持，旧失败和新增真实上下文的初始失败均保留。`--isolated-only`、`--network-only`、Reset 的 `--cpu-only` 会明确标注未运行完整套件，不能据其局部通过宣称完整通过。

本批最终冻结回归见 `research/validation/batch4_rounded_gpu.json`：上述采样与 Reset 完整通过，另有 71 步/79 比较、二维运动 35 项、分辨率 3 项、控制 18 项、后处理 16 项通过；新源码 35 项 CPU 方法通过。Reset 的旧语义 baseline 6/8 是非门控诊断，不能算作旧实现通过。各项使用已标识的 d442/f454 本机参考，性能未重新测量。最新总索引为 `research/validation/session_summary.json`，历史失败保持原状态。

## 自动验收约定

第三批新增 `check_sampling.py`，将硬件 TEX、抽取的 pre/post 采样指令、完整 post kernel 和整网分别验收。以下第三批记录保留当时的 Torch 倒数近似和失败；第四批已按上文修正完整 pre/post 的融合上下文并加入硬件 RCP。`ops.bilinear` 的 21 位坐标截断、8 位轴/交叉权重仍仅由 RTX 3060 实测支持。

`research/validation/batch3_sampling.json` 及默认 cubin 更新后重跑的 `batch3_sampling_current.json` 均明确为 **passed:false**：12 组隔离采样通过 10 组，两组随机坐标 post 边界失败；7 组完整 post 测试通过；20 组整网检查通过，CUDA Graph 与 eager 逐值一致。完整 post 测试将神经权重置零并指定 blend，目的是隔离历史采样，不代表全部后处理组合。整网十项前后比较的平均 MAE 净降 0.012978/255，但零运动矢量对照的三帧均略增，不能写成所有用例改善。

```powershell
python torch/check_sampling.py --report research/validation/check_sampling.json
```

第三批该命令因两项边界失败返回非零；第四批扩展后的完整套件已通过，旧报告仍为 false，未重写历史。最大差 0.005/255、MAE 0.0005/255 的隔离门槛未放宽。采样模型只由本机测量支持，RTX 50 仍需独立验收。

`check.py`、`check_frames.py`、`check_motion.py`、`check_res.py`、`check_controls.py` 现在支持 `--report`、`--bit-exact`、`--min-correlation` 与 `--max-mae`。默认门槛为相关至少 0.999、MAE 至多 3/255；这是移植版回归门槛，不能解释为一比一。报告保留参考文件哈希、误差、逐值一致率和验收结果。缺资源、非有限输出或指标未达标返回非零。

`check_blocks.py` / `check_levels.py` 默认要求其定义的检查点均有抓取；缺项必须报告并失败。`--allow-missing-taps` 允许部分研究运行，但报告仍标为 incomplete。其覆盖率仅相对于脚本定义的检查点，不是整个网络所有数值分支的覆盖率。

控制/后处理脚本会使用 `harness/rt_sm86` 的共享调试目录，应串行运行；它们拒绝复用旧结果，但不提供多进程隔离。

严格 `--bit-exact` 预期目前不通过；失败应保留，不应通过放宽门槛冒充完成。原生跨硬件对照见根目录 README 的 `harness/reference.py`。

第三批新增 `check_motion2d.py`，只读取完整、带哈希的 motion2d 捕获；19 组捕获对应 32 项浮点/8 位历史数值检查和 3 项输入逐字节证明，35/35 通过，0 跳过。覆盖正负对角、纵向、分数和越界矢量及奇数尺寸；数值比较仍是 sm86 移植版近似门槛。

```powershell
python torch/check_motion2d.py --capture harness/captures/port-batch3-motion2d --report research/validation/check_motion2d_batch3.json
```

第三批发现的 `hist=None` 与显式非零 `mv` 组合缺口已在第四批修复，raw / public / Graph / 引擎外接口的区别见上文。第三批旧套件未覆盖该组合的事实及失败证据保留，不把第四批结果反写成旧套件已验证。

逐步验收改用 `tools/schedule_refs.py` 生成并校验的单一 fresh manifest，不再从多个旧 tap 目录按最小文件大小猜选。640×360 重置帧的 71/71 调度步、79/79 比较通过，0 跳过；其中 77 项为步内输出或辅助分支、2 项为布局 repack。post 只对照最终 sRGB8，其他边界使用明确的 FP8 字节布局。该结果不是所有内核内部状态或所有尺寸全覆盖。

```powershell
python -m tools.schedule_refs capture --out research/my_schedule_capture --harness harness/batch3/nr-lab.exe --expected-port-sha256 f454090a97d18e2eb8e6008d42a5fa4e37c8e4bd251a86aedfff8901fef23382
python torch/check_schedule.py --capture research/my_schedule_capture --harness harness/batch3/nr-lab.exe --report research/my_schedule_check.json
```

捕获目录必须新建，运行 DLL、程序和辅助组件保留哈希身份。`--expected-port-sha256` 显式固定新 NaN 修复版；不指定时仍固定本批逐步参考所用的旧 d442 版本，不能静默替换来源。旧捕获绑定其自身保存的 runtime 和源码快照；明确指定的 harness 身份也须一致。旧 `check_blocks.py` / `check_levels.py` 的历史数字仅对各自当时的抓取成立，当前验证入口使用上述 manifest。

## 精度模式（历史测量）

| 模式 | 360p | 1080p | 第 0 帧相关 | 正确/错误运动矢量的输出之差 (kernel 2.91 / 3.11 / 3.17) |
|---|---|---|---|---|
| 精确 `precise=True` (f32，模拟每处 f16 与 fp8 舍入) | 317 ms | 2.5 s | 0.99926 | 2.74 / 2.99 / 3.11 |
| 快速 (默认: 半精度 autocast，只模拟 fp8 量化) | 187 ms | 1.6 s | 0.99931 | 2.72 / 3.14 / 3.00 |
| (对照) 不模拟任何舍入 | — | — | 0.99929 | 2.02 / 2.20 / 2.34 |

f16 舍入对结果没有可见影响，fp8 量化则不能省: 单帧看不出差别，但去掉后网络对运动矢量的响应弱 25%。
快速模式在吻合度上不输精确模式 (半精度 matmul 以 f16 存结果，本来就接近 kernel)。逐块对照脚本 (check_blocks /
check_levels / check_numerics) 使用精确模式。全分辨率 1h 级 (pre/post) 的 FFN 与注意力按 2^18 token 分块，
1080p 峰值显存从 7 GB 降到 3.4 GB (之前 3060 上 1080p 会因显存换页慢到几十秒一帧)。

逐块 (以 kernel 上一块输出为输入): 1h-8h 每块相关 0.9994-0.9998、逐值一致 66-89%；16h 块 0.99997；ViT 块 0.99965。
逐级出口 (起点为抓取的 pre 输出): 编码 1h→8h 0.9987 → 0.994，16h 入口 0.990，解码 16h→1h 0.985 → 0.996。
研究支持 fp8 舍入、rcp/rsqrt 近似和累加顺序对剩余误差有贡献，尚未严格排除所有结构或边界差异。上述逐块结果不是 71 步自动全覆盖证明。torch 版比 research/net_ref.py (numpy) 更贴近移植 kernel，
差别见下面"容易写错的地方"。

## 结构

```
pre_block   16 路输入 (颜色/重投影历史/3 路噪声/常数/5 路画面参数/恒 0) -> 适配器 16->32 -> 全分辨率 1h swin -> 2x2 平均
编码        1h(32ch) x4 -> 2h(64) x4 -> 4h(128) x6 -> 8h(256) x8，每级末块 2x2 平均 + W->2W 下采样
瓶颈        16h 分组 swin (512ch, 16 头) x8 -> 池化 + 升维 1024 -> ViT-1d x8 (96 token 全局注意力)
            -> block39 (1024->512 上采样 + 跳连) -> 16h x8
解码        8h x8 -> 4h x6 -> 2h x4 -> 1h x4，每级首块: z = c⊙skip + 上采样(low)·W_up, y = s⊙z + FFN(z)
post_block  s1⊙上采样(x) + s2⊙skip(pre) -> 全分辨率 1h swin -> 32->4 输出卷积
            -> cur = clamp(color + 0.25·rgb)，out = cur + clamp(sigmoid(a)·0.7397)·(CatmullRom(历史) - cur)
```

| 文件 | 内容 |
|---|---|
| `dlss5/layout.py` | 权重字节的解码 (FP8 mma 片段序、通道重排、偏置排布)，加载时用一次 |
| `dlss5/ops.py` | GPU 数值原语: f16/fp8 舍入、窗口划分、位运算 exp、纹理采样、噪声 |
| `dlss5/blocks.py` | Swin / SwinDown / SwinUp (1h-8h)、Split16、FinalHead、ViT、Block39 |
| `dlss5/net.py` | PreBlock、PostBlock、按 `data/schedule.json` 执行 71 步的 DLSS5 |
| `dlss5/data/` | `schedule.json` (块类型/记录/窗口平移，由 `research/gen_schedule.py` 从执行轨迹导出)、`exec_order.json`、`adapter_map.json`、`control_map.json` (适配器里的常数与 5 路画面参数，`research/control_map.py`)、`maps_block1.json` |

容易写错的地方 (都已实测):

- 余弦注意力 `q̂ = τ·q/|q|` (τ 只乘 Q)，相对位置偏置作为 QK 累加器初值。
- **所有级别**的 softmax 分子都是位运算近似 exp (PTX 里没有 ex2)，不减行最大值：1h-16h 截断在 [-6,6] 且先归一化再乘 V，
  ViT 截断在 [-3,3] 且先乘 V 再归一化。
- 残差与投影输入的量化按级别不同 (`blocks.Swin.NUMERICS`)：1h 残差用 f16 的 y、投影前 O 量化到 fp8；上采样块的 FFN 输入先量化成 fp8；
  2h-8h 残差读的是存进共享内存的 fp8 y。pre_block 适配器的输入先舍入到 f16。
- 16h 的 FFN 是分组低秩的 (8 组，每组 512->64->256->64)。
- 整网在补齐网格上运行 (360p: 384x640)，补齐行按 `y' = 2(H-1) - y` 镜像取样；重置帧以颜色充当历史。

## 补齐网格规则

`CCNetwork::SetResolution` (sub_18003C580) 先用原始尺寸建网，逐层推导形状，统计每一维里"输出比输入小"的层数 n，
取 m = 2^n：

```
G = max(ceil(x / m) * m, 320)            每维分别算
若 GH % 4mH == 0 且 GW % 4mW == 0: GW += mW
```

6 个下采样层各输出 `ceil4(ceil(x/2))`，所以通常 n = 6、m = 64。2h 级的 outview 层输出的是计算图记录的形状 `ceil8(x)/4`
(计算图在输入端 `constant_pad_nd` 到 8 的倍数)，它比链上的值小时再计一次，m = 128。例如 1080 → 1152，1040 → 1088。
规则与 90 个实测尺寸、15 个盲测尺寸 (含奇数尺寸、4K、21:9) 全部一致；实测数据在 `research/grid_sweep.jsonl`。

## TensorRT 路线 (已试，不如移植版)

`dlss5/export.py` 把网络主体 (pre 适配器之后 -> post 输出卷积) 导出为半精度 ONNX，`trt_check.py` 构建 TensorRT 引擎并与参考比较
(需要 `pip install onnx tensorrt-cu12`；引擎写到 `_dl/trt/`，不入库)。导出时 FP8 量化可去掉 (`--q8 none`) 或换成算术伪量化
(`--q8 arith`，torch 里验证过，ONNX 导出未做完)，位运算 exp 换成 `exp(clamp(L))`。`--canon 1` 把所有通道重排折进权重。

| 540p (960x540)，RTX 3060，TensorRT 11.4 | 网络主体 | 与 torch 参考 |
|---|---|---|
| 导出版在 torch 里跑 (`eager`，arith) | — | 相关 0.9997，平均差 1.1/255，运动矢量响应一致 |
| TensorRT，`--q8 none` | 47.0 ms | 相关 0.9992-0.9996，运动矢量响应弱约 25% (与去掉量化的结论一致) |
| 同上，规范序 (`--canon 1`) | 45.1 ms | 同上 |
| 同上，适配器移出引擎 | 44.3 ms | 同上 |
| (对照) sm_86 移植版整帧 | 14.0 ms | |

结论: 引擎结果正确，但比移植版慢约 3 倍，而且还没算引擎外的入口采样/噪声 (15 ms) 与出口混合 (9 ms，均为未优化的 torch 实现)。
逐层统计: 矩阵乘约 55%、窗口划分/还原等重排约 24%、注意力约 12%。原版 kernel 把一个 swin 块的全部运算融合在片上完成，
TensorRT 拆成近千层、每层读写一遍显存；高分辨率级只有 32 通道，瓶颈是显存带宽而不是算力。1080p 未测出 (导出时内存不足)。
这次 TensorRT 实验没有超过融合移植版；更小网络或定制融合是未来研究方向，现有结果不构成其他优化不可能的证明。

## 数据来源

逆向过程、每条结论的证据与精度记录在 `tools/notes.jsonl`；numpy 逐块参考和探针在 `research/`。
早期的"形状骨架"(打包模型已被证伪) 移到了 `legacy/`，仅作存档。
