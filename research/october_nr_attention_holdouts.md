# 十月版 Small FP8 的跨尺寸与时序注意力复核

2026-11-06。用 RTX 3060 的 `310.9.16.0` Small FP8 实验移植，在两组独立用例抓取全部八个 ViT 注意力块。两组都是实验程序生成的场景。每组 Q/K/V/输出的 32 份边界哈希均与上一批不同，插桩采集与同配置普通运行的最终画面逐字节一致。[用例、运行时与封存记录](validation/october_nr_attention_holdout_captures.json)

| 实验用例 | 输入画面与帧 | 内部 token 形状 | 每头查询分块数 | FP8 输出字节数 | 独立重放差异 |
| --- | --- | --- | ---: | ---: | ---: |
| `resolution_1920x1080` | 1920×1080，首帧 | 20×32，640 token | 3 | 5,242,880 | 0 |
| `history_b_f3` | 640×360，第 4 帧，场景变体 1 | 8×12，96 token | 1 | 786,432 | 0 |

每个用例同时核对了 PV 累加、FP16 倒数和量化前值。三组中间值各覆盖 **6,029,312** 个 FP16 位置，全部逐位一致。倒数按输出通道广播。输入来自整网的 chained 入口，量化前中间值由普通入口对相同 Q/K/V 隔离插桩。独立实现只用 Q/K/V 生成预测，QK 和 PV 的 K16 由独立的 sm_86 矩阵指令重放。[1080p 中间值](validation/october_nr_attention_640_stages.json) · [1080p 重放](validation/october_nr_attention_640_replay.json) · [时序中间值](validation/october_nr_attention_temporal_f3_stages.json) · [时序重放](validation/october_nr_attention_temporal_f3_replay.json)

1080p 经过十轮 64-key 归约，无 key 尾部填充。每个 CTA 覆盖 256 个查询，最后一个 CTA 覆盖 128 个有效查询。时序用例每行执行两轮归约，其中 32 个 key 为零填充。CPU 数学近似在 1080p 和时序用例分别有 90,380 和 6,208 个 FP8 数值差异。只替换 QK 指令时，两个用例的倒数差异都消失，仍有 81,507 和 2,688 个输出值不同。继续换成 3060 的 PV 指令，剩余差异消失。

## 直接观察 QK 和量化前概率

普通入口的 64 个 half2 概率生成位置读出了 QK logits 与未量化的概率 P16。写入坐标按实际 head、query、key 排列，保留内核真实执行的尾部 key。缓冲的未写区域以 NaN 标记，插桩后的原 FP8 输出与封存输出相同。三组用例共覆盖 24 个注意力调用，logits 和 P16 各比较 **111,149,056** 个 FP16 位置，含 1,572,864 个填充位置。独立 sm_86 重放的 logits 全部吻合，从实测 logits 在主机重算的 P16 也全部吻合。[原 96-token 首帧](validation/october_nr_attention_qk_96.json) · [1080p 首帧](validation/october_nr_attention_qk_640.json) · [96-token 第 4 帧](validation/october_nr_attention_qk_temporal_f3.json)

前一批在块 11、head 7、query 14、key 5 发现的临界值，现在可直接读到。GPU 的 logit 为 `-1.5869140625`，比 CPU 近似的 `-1.5859375` 低一个 FP16 步长。实测 P16 是 `0.0166015625`，编码后落到 E4M3 `0x08`，CPU 近似落到相邻的 `0x09`。这里的 E4M3 概率由捕获的 P16 重新编码。[先前的算术与 K16 顺序](october_nr_attention_tensor_replay.md)

## 为什么主机倒数能吻合

用精确有理数穷举钳位后全部 **30,704** 个有限 FP16 正分母，范围为 `0x0410` 至 `0x7bff`。FP32 近似倒数离精确值最多 1 ULP。穷举中，精确倒数距最近的 FP16 舍入中点至少 `4096/2047` 个 FP32 ULP。将近似值舍入到 FP16 与主机 FP32 除法再舍入到 FP16 的结果逐位相同。这个边界覆盖倒数阶段，分母仍取决于 logits 和归约顺序。[有理数验证结果](validation/october_nr_attention_reciprocal_bound.json) · [NVIDIA 倒数指令规范](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-rcp)

## 复现

在仓库根目录，用已核验的实验运行时准备两组独立捕获。`$base` 指向由[自动采集工具](october_nr_version_factors.md)生成的本机运行目录，其中包含 RTX 3060 实验版 DLL、辅助库和 `tap.cubin`。输出使用 Git 忽略区的新目录。

```powershell
$lab = '_dl/october_nr_20261106'
$base = "$lab/batch9/controlled_cli/tap"
$out = "$lab/attention-holdouts"
$roles = foreach ($i in 11..18) { foreach ($r in 'in0','in1','in2','out') { '--role'; "block$i.attention.$r" } }
$common = @('--kind','experimental-sm86','--dll',"$base/nvngx_dlssnr.dll",'--harness',"$base/nr-lab.exe",'--bridge',"$base/nvngx.dll",'--support',$base,'--tap-cubin',"$base/tap.cubin")
python -B -m tools.capture_october_taps @common --out "$out/hd1080" --suite full --case resolution_1920x1080 --frame 0 @roles
python -B -m tools.capture_october_taps @common --out "$out/temporal_f3" --suite temporal --case history_b_f3 --frame 3 @roles
$probe = @('--ptx',"$lab/sm86_compile_probe/module06.experimental-acq-rel.ptx",'--ptxas','cuda_tools/bin/ptxas.exe','--blocks') + @(11..18)
foreach ($case in 'hd1080','temporal_f3') {
    python -B -m tools.ptx86.dump_october_attention_full --capture "$out/$case/tap" --work "$out/$case/planes" @probe
    python -B -m tools.ptx86.check_october_attention_tensor "$out/$case/tap" --dump "$out/$case/planes"
    python -B -m tools.ptx86.dump_october_attention_qk --capture "$out/$case/tap" --work "$out/$case/qk" @probe
}
python -B -m tools.ptx86.dump_october_attention_qk --capture "$lab/batch8/full_boundary_tap" --work "$out/original96-qk" @probe
python -B -m tools.ptx86.check_october_attention_reciprocal
```

这些结果将移植版注意力的逐位验收扩展到 1080p 首帧与另一场景的时序帧。独立实现仍借助 RTX 3060 执行矩阵指令。4K 原尺寸的 NR 注意力和 RTX 50 原版逐层数值仍待验证。
