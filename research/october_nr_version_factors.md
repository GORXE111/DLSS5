# 十月版 NR 运行时与模型预设的差异

2026-11-06，RTX 3060 12 GB，驱动 610.74。三组 D3D12 重放使用同一 Sponza 原图、同一 DXGI 代理和同一桥接库。每个场景内，4K 后缓冲和 1080p 工作输入均逐字节相同。

| 组 | 运行时 | 预设 | PE 权重资源 |
| --- | --- | ---: | --- |
| A | 310.8.0 Ampere 移植 | 1 | `WEIGHTS_HT` |
| B | 310.9.16.0 Ampere 实验版 | 1 | 与 A 完全相同的 `WEIGHTS_HT` |
| C | 与 B 相同 | 2 | `WEIGHTS_CC_SMALL_FP8` |

两份官方原件和各自的 Ampere DLL 均复核了 HT 资源，147,695,410 B 的完整 PE 资源哈希一致。小档更换了权重，也将序列化逻辑块从 71 改为 31。因此 B→C 衡量整个预设的变化。[四份 DLL 的静态资源核验](validation/october_nr_version_factor_readiness.json)

| 实验场景 | A→B 的 1080p NR / 4K 合成 | B→C 的 1080p NR MAE | B→C 的 4K 合成 MAE |
| --- | --- | ---: | ---: |
| 静止，逐帧重置 | 两份输出均逐字节相同 | 5.744/255 | 5.624/255 |
| 4K 画面每帧平移 4 px，使用时序 | 两份输出均逐字节相同 | 5.558/255 | 5.460/255 |

比较的是同一场景条件下的第 5 次 Present。两组中 A→C 的差异都与 B→C 完全相同。另用同一个 D3D12 lab 做了三组 640×360 scRGB 对照，分别是 Reset 首帧、带运动的第 4 帧和显式遮罩。A/B 三组的 PPM 和原始 RGBA16F 都逐字节一致。带运动的四帧共 624 条提交，入口名、顺序、网格、线程块和共享内存字段均相同。每帧 pre 参数由 264 B 扩至 272 B，新遮罩字段均为 0。上述等价性限定于实测输入和这两份 Ampere 构建。[实测矩阵与文件哈希](validation/october_nr_version_factor_experiment.json)

## 注意力残差定位

同一批首尾 ViT 的真实 Q、K、V 在 3060 上独立送入十月版普通和 chained attention。四次输出共 393,216 B，均与整网抽取的输出逐字节一致。chained 重放预置了就绪计数，用于隔离算术和输出写入。[CUDA 入口重放](validation/october_nr_attention_replay.json)

根据物理 K16 分组修正 CPU 模型后，首个 attention 从 730 个不同值降到 450 个，末个从 1,828 降到 1,779 个，每处均比较 98,304 个值。末个调用的 MAE 从 0.001407 略升至 0.001456，最坏差异仍有 16 个 E4M3 步长。浮点输入全零的 Q 提供了一个可独立验证的例子。在同一个 PV 点，真实分组的输出为 `0xcb`，旧分组为 `0xca`。其第三次 MMA 的数学和为 −28.74609375，单次最近舍入应为 −28.75，3060 实测为 −28.734375。这证明该指令的结果在此输入上与单次最近舍入的数学模型不同。[MMA 指令实测](validation/october_nr_pv_order.json)

第二个全零 Q 点仍有残差。单 warp 探针归一化得到 1.9375，量化为 2.0，完整内核输出为 1.875。下一次应在普通入口分别记录该点的实际分母、PV 半精度累加值和量化前值，以区分差异从哪个阶段开始。CPU 诊断可复算误差坐标及分组、规约、倒数敏感性。[数值诊断汇总](validation/october_nr_version_factor_experiment.json)

后续已完成完整内核的分阶段取样，并将该点的首个差异定位到 PV 第 3 次 MMA。[逐指令实测](october_nr_attention_instruction_trace.md)

```powershell
python -B -m tools.ptx86.check_october_attention _dl/october_nr_20261106/batch8/attention_tap --diagnose
python -B -m tools.ptx86.probe_october_pv_order _dl/october_nr_20261106/batch8/attention_tap
```

## 在 RTX 50 采集同一批张量

新的采集命令依次运行普通、原链追踪和张量抽取。它根据本次轨迹生成取样清单，并要求三次最终输出逐字节一致，再为每个张量写入哈希和封存记录。3060 的实际运行已经重新抓到 73 个边界，并逐一复现了独立手工捕获的 73 个张量哈希。

下面的两个命令分别在 3060 与 RTX 50 运行。从仓库根目录执行，保持 lab、桥接 DLL 和两份辅助 DLL 的 SHA-256 相同。为两张卡分别编译 `harness/tap.ptx`，编译目标为 `sm_86` 和 `sm_120`。正式侧使用官方 310.9.16.0 DLL 的原件，以及从 3060 捕获目录复制的 `tap/seal.json`。若正式侧选择了工具尚未覆盖的链路，采集会保留日志和失败状态。

```powershell
$port = '_dl/october_port'
$native = '_dl/october_native'
python -B -m tools.capture_october_taps --kind experimental-sm86 --dll $staged --harness $harness --bridge $bridge --support $support --tap-cubin $tap86 --out $port --suite october-mask --case blend2_f3 --frame 0 --all-roles
$tap120Sha = (Get-FileHash -Algorithm SHA256 $tap120).Hash.ToLowerInvariant()
python -B -m tools.capture_october_taps --kind official-sm120 --dll $original --harness $harness --bridge $bridge --support $support --tap-cubin $tap120 --tap-sha256 $tap120Sha --out $native --suite october-mask --case blend2_f3 --frame 0 --all-roles --match-seal "$port/tap/seal.json"
```

在同一台分析机器上保留两边的捕获目录及独立记录的封存哈希，再进行比较。

```powershell
$left = Get-Content -Raw "$native/capture.json" | ConvertFrom-Json
$right = Get-Content -Raw "$port/capture.json" | ConvertFrom-Json
$roleArgs = foreach ($role in $right.roles) { '--role'; $role }
python -B -m tools.compare_october_taps compare "$native/tap" "$port/tap" --native-seal $left.capture_seal_sha256 --port-seal $right.capture_seal_sha256 @roleArgs
```

原版 RTX 50 的实际输出仍待采集。当前已能区分同权重运行时更新、整套预设切换以及 3060 上两类注意力计算证据。
