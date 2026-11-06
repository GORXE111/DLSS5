# 十月版 Small FP8 的 QKV 归一化与 ViT 串联

2026-11-06。用 RTX 3060 的 `310.9.16.0` Small FP8 实验移植采集两组 640×360 场景，注意力为 96 token。每组包含八个 ViT 块各自的 FFN 输入和扩张输出，共 64 份边界，两组对应张量的内容均不同。首帧用例与上一批共同采集的 48 份张量及最终画面逐字节相同。[捕获记录](validation/october_nr_vit_capture.json)

## Q、K 的 692 个字节差异已消除

对普通 QKV 入口隔离插桩，读出投影半精度、钳位后平方和、逆范数及量化前值。八块的 Q、K 各核对 786,432 个投影和量化前半精度位置，平方和及逆范数各核对 24,576 个 token/head 值。插桩后的完整 Q/K/V 与封存输出逐字节相同。实测中间值和独立重放的矩阵乘结果逐位一致。[Q/K 中间值](validation/october_nr_qkv_norm.json)

| 平方和计算 | Q 的 FP16 平方和差异 | K 的 FP16 平方和差异 |
| --- | ---: | ---: |
| 两个平方各自取整后相加 | 2,864 | 2,929 |
| 先取整高半组平方，再与低半组平方融合乘加 | 0 | 0 |
| 反过来，先取整低半组平方 | 3,585 | 3,647 |

同一固定 cubin 的反汇编显示，归一化路径中存在 `HFMA2`。原来的逐步近似漏掉了这一融合。块 11 的 K 在 head 0、token 16、通道 19 给出了可复算的例子。投影值为 `-16.15625`，GPU 平方和为 `1468`，旧模型得到 `1467`，输出因此分别落入 E4M3 `0xad` 与 `0xae`。修正半精度乘加顺序后，八块 **2,359,296 个 Q/K/V 字节**全部由独立算术复现。[QKV 输出复算](validation/october_nr_qkv_replay_fused.json)

另在 RTX 3060 上直接运行 `rsqrt.approx.ftz.f32` 并转换为 FP16，穷举钳位范围内全部 30,704 个有限正 FP16 平方和。GPU、主机 FP32 计算与精确值的 FP16 舍入全部一致。按比规范更宽的相对误差 `2^-22` 检查舍入边界，也没有可跨越 FP16 中点的输入。[指令实测与数学边界](validation/october_nr_qkv_rsqrt.json) · [NVIDIA rsqrt 规范](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-rsqrt)

## FFN 与块间连接

FFN 扩张使用 `1024→4096` 的 E4M3 权重，点积之后运行半精度分段三次激活。令 `s = clamp(x, -4, 4)`，先用两次 FP16 融合乘加计算斜率与增益，再以原始 `x` 乘增益并量化。八块的 3,145,728 个 FP8 字节逐位吻合。[FFN 扩张](validation/october_nr_ffn_expand_replay.json)

FFN 压缩使用 `4096→1024` 权重，四个连续 K1024 组分别累加。首组以按通道缩放的 FP16 残差为初值，分组结果按顺序做 FP16 加法，最后编码为 E4M3。输出继续进入 QKV，并作为注意力输出投影的残差。八块的 786,432 个 FP8 字节逐位吻合。[FFN 压缩](validation/october_nr_ffn_contract_replay.json)

将 FFN 扩张、压缩、QKV、注意力和输出投影依次串联。每组只提供块 11 在输入重排后的激活和原始权重，上一步生成的输出直接成为下一步输入。块 18 的验证止于输出重排前。每组独立计算的 63 个边界共 8,552,448 B，全部逐字节一致。[首帧串联报告](validation/october_nr_vit_chain_reset.json) · [时序串联报告](validation/october_nr_vit_chain_temporal.json)

| 用例 | 生成边界 | 比较字节 | 差异字节 |
| --- | ---: | ---: | ---: |
| `reset`，首帧 | 63 | 8,552,448 | 0 |
| `history_b_f3`，另一场景的第 4 帧 | 63 | 8,552,448 | 0 |

另从上一批 4K 捕获的块 12 和 18 各取六个 16-token 分块，包含跨 CTA 边界的分块及末尾有效分块。独立串联两层 FFN 后，与 4K QKV 输入对应位置的 196,608 个字节全部一致。[4K FFN 抽样](validation/october_nr_ffn_4k_tiles.json)

## 复现

从仓库根目录运行。`$base` 指向已核验的 RTX 3060 实验版目录，`$dll` 只用于读取固定权重资源。用 `--out` 和 `--work` 指定 Git 忽略区中的新目录。[运行目录准备](october_nr_attention_holdouts.md)

```powershell
$lab = '_dl/october_nr_20261106'
$base = "$lab/batch9/controlled_cli/tap"
$out = "$lab/vit-chain"
$dll = "$lab/original/nvngx_dlssnr.dll"
$cubin = "$lab/sm86_compile_probe/module06.experimental-acq-rel.cubin"
$ptx = "$lab/sm86_compile_probe/module06.experimental-acq-rel.ptx"
$roles = foreach ($i in 11..18) {
    foreach ($r in 'ffn.expand.input','ffn.expand.output','qkv.input','attention.in0','attention.in1','attention.in2','attention.out') { '--role'; "block$i.$r" }
    '--role'; if ($i -eq 18) { 'block18.repack.input' } else { "block$i.out1" }
}
$common = @('--kind','experimental-sm86','--dll',"$base/nvngx_dlssnr.dll",'--harness',"$base/nr-lab.exe",'--bridge',"$base/nvngx.dll",'--support',$base,'--tap-cubin',"$base/tap.cubin")
python -B -m tools.capture_october_taps @common --out "$out/reset" --suite smoke --case reset --frame 0 @roles
$cap = "$out/reset/tap"
python -B -m tools.ptx86.dump_october_qkv_norm $cap "$lab/october_weight_tap" --ptx $ptx --ptxas cuda_tools/bin/ptxas.exe --work "$out/norm" --blocks 11 12 13 14 15 16 17 18
python -B -m tools.ptx86.check_october_qkv_rsqrt
python -B -m tools.ptx86.check_october_qkv $cap "$lab/october_weight_tap" --trial-cubin $cubin --gpu-mma
python -B -m tools.ptx86.probe_october_ffn_expand $cap --original-dll $dll --trial-cubin $cubin
python -B -m tools.ptx86.probe_october_ffn_contract $cap --original-dll $dll --trial-cubin $cubin
python -B -m tools.ptx86.check_october_vit_chain $cap --original-dll $dll
python -B -m tools.capture_october_taps @common --out "$out/temporal" --suite temporal --case history_b_f3 --frame 3 @roles
python -B -m tools.ptx86.check_october_vit_chain "$out/temporal/tap" --original-dll $dll
python -B -m tools.ptx86.check_october_ffn_4k_tiles "$lab/batch13/uhd2160_padded/tap" --original-dll $dll
```

本批串联覆盖实验版 ViT 核心的两组 96-token 输入，4K 验证在本批覆盖全注意力与 FFN 抽样。后续已完成 4K 全 ViT 串联和两次 2D 重排。整网独立数值对照及 RTX 50 原版逐层验收仍待完成。[4K 全链实测](october_nr_vit_uhd_chain.md)
