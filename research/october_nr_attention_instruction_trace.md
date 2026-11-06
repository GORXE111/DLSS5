# 十月版注意力的逐指令实测

2026-11-06。使用 RTX 3060 上已封存的首尾 ViT Q/K/V，重放十月版普通注意力入口。实验副本将两个点的分母、倒数、PV 累加和量化前值写入独立缓冲。插桩版的完整正常输出与整网取样逐字节一致。标记试验只改变预期的两个 FP8 字节，确认寄存器与输出位置的映射。[指令和张量证据](validation/october_nr_attention_stages.json)

后续已将验证扩展到八个完整注意力张量，同时修正 QK 的 K16 分组。[整张量重放](october_nr_attention_tensor_replay.md)

## 一个量化边界的差异从哪里开始

在块 11 的 head 28、token 10、通道 29，完整内核和 CPU 的分母均为 8.0625，半精度倒数均为 0.1240234375。差异出现在相乘之前。

| 阶段 | 完整 GPU 内核 | CPU 独立计算 |
| --- | ---: | ---: |
| PV 半精度累加 | 15.609375 | 15.625 |
| 量化前半精度 | 1.935546875 | 1.9375 |
| E4M3 | `0x3f`，即 1.875 | `0x40`，即 2.0 |

独立取整器用 GPU 量化前值可复算实际输出。逐条记录八次 K16 后，使用真实操作数和真实前一累加值对照单次最近舍入，第 3 步是第一个分歧，GPU 为 8.015625，计算模型为 8.0234375。第 4 步继续相差一个半精度步长，影响随后累加。最后两步的 V 操作数来自填充，实测保持累加值。PTX 对这类 `mma` 的累加顺序和舍入细节留给实现。[逐步实测](validation/october_nr_attention_stages.json) · [NVIDIA PTX mma 规范](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma)

把首尾两个 ViT、每个两个点的八步实际 A/B/C 寄存器分别送入独立单 warp，**32 次 MMA 的 2,048 个输出字全部逐位一致**。每个字包含两个 FP16 值。重放使用取样记录中的操作数，结果覆盖这些 sm_86 指令的输入与输出。[独立指令重放](validation/october_nr_measured_mma_replay.json)

## 实际的 V 取数顺序

此前的独立单 warp 实验按交错 K16 组织 V。新实验给每个有效 key 的一个 V 通道赋唯一 FP8 标签，核对真实内核读到的操作数。96 个有效 key 分成六段连续的 16 个，后两段为填充。该映射又与首尾两处注意力、每处两个实际输出位置的 512 条真实 V 操作数行核对。

| 独立 CPU 计算 | 旧 PV 分组的数值差异 | 使用实测 PV 顺序 | 新平均绝对误差 |
| --- | ---: | ---: | ---: |
| 首个 ViT，98,304 个值 | 450 | 364 | 0.001028 |
| 最后一个 ViT，98,304 个值 | 1,779 | 1,697 | 0.001398 |

表中只调整 PV 的分组顺序。CPU 的 MMA 数值模型仍以半精度最近舍入计算，剩余差异继续保留。测试覆盖实测的 96-token 几何和这些捕获输入。

## HT 大模型的中间张量

对旧版 310.8.0 与十月版 310.9.16.0 的相同 HT 预设，补做了同一个 scRGB 运动用例。首帧的 pre 下采样与 skip、首尾 ViT 的 Q/K/V/输出、post 的两个输入，共 12 份张量、20,447,232 B，逐字节一致。采集版与普通运行的最终 RGBA16F 和 PPM 也一致。两版回调记录的比例分别为 1.0 与 0.85。

四帧的 624 次提交顺序相同。十月 pre 参数去掉 `+40` 的遮罩模式和 `+196` 的槽位选择器后，四帧都与旧版 264 B 参数块逐字节一致。本结论覆盖这一个配对用例的已抽取边界及最终输出。[HT 张量对照报告](validation/october_nr_ht_activation_taps.json)

## 复算入口

从仓库根目录运行。实验编译所需的固定 PTX 来源见[十月实验版宿主重建](october_nr_stage_replay.md)。`--work` 使用 Git 忽略区里的新目录，编译物与原始操作数留在本机。

```powershell
$lab = '_dl/october_nr_20261106'
$work = "$lab/attention-stage-replay"
$stageReport = 'research/validation/october_nr_attention_stages.json'
python -B -m tools.ptx86.probe_october_attention_stages --capture "$lab/batch8/attention_tap" --ptx "$lab/sm86_compile_probe/module06.experimental-acq-rel.ptx" --ptxas cuda_tools/bin/ptxas.exe --work $work
$stageSha = (Get-FileHash -Algorithm SHA256 $stageReport).Hash.ToLowerInvariant()
python -B -m tools.ptx86.replay_october_mma_fragments --capture $work --report $stageReport --report-sha256 $stageSha
python -B -m tools.compare_ht_activation_taps "$lab/batch10/ht_old1_tap" "$lab/batch10/ht_oct1_tap"
```

RTX 50 原版的逐层输出仍待采集。现有证据已将这一处 Ampere 与 CPU 的差异定位到实测 PV 指令，后续可在同一张量角色上直接对照原版算术。
