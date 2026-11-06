# 十月版 Small FP8 的八块注意力张量重放

2026-11-06。RTX Remix `310.9.16.0` Small FP8 的 RTX 3060 实验移植中，八个 ViT 块（11–18）的首帧注意力输出已由独立实现逐字节复现。每块 96 个 token、32 个 head、每 head 32 个输出通道，共 **786,432 个 E4M3 字节**。逐项核对 PV 累加、分母的半精度倒数、量化前结果，各有 786,432 个 FP16 位置，也全部逐位一致。倒数按输出通道广播。[数值报告](validation/october_nr_attention_tensor_replay.json) · [完整插桩报告](validation/october_nr_attention_full_stages.json)

## 把两处矩阵乘法分开替换

四组对照共用相同的封存 Q/K/V 和主机代码，只替换 QK（点积）与 PV（对 V 的加权和）两处的 MMA 实现。CPU 使用数学点积后取整到 FP16。硬件重放将逻辑张量按 NVIDIA 的 lane 布局打包，用独立的 `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16` 重放。指数近似、分母归约、倒数与 E4M3 编码都在主机执行。预测路径只读取 Q/K/V。实测 FP8 和中间值用于核对，替换实测中间值的对照结果在 JSON 中单独记录。[NVIDIA MMA 寄存器布局](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16816-float)

| QK / PV | PV FP16 字差异 | FP16 倒数字差异 | 量化前 FP16 字差异 | 最终 FP8 字节差异 | 最终数值差异 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU / CPU | 278,986 | 58,720 | 285,056 | 7,534 | 7,490 |
| 3060 / CPU | 245,969 | 0 | 228,376 | 3,224 | 3,183 |
| CPU / 3060 | 57,300 | 58,720 | 97,059 | 4,872 | 4,870 |
| 3060 / 3060 | 0 | 0 | 0 | 0 | 0 |

FP8 字节数计入正零和负零，数值数将二者视为相等。只把 QK 换成 3060 指令，FP16 倒数就全部一致，最终 FP8 最大偏差降为一个 E4M3 步长。这定位了分母路径的误差来源。用相同的 QK 结果继续换掉 PV，剩余的量化前差异也全部消失。

## 物理顺序与取证

- QK 的两个 K16 使用连续逻辑通道 `0..15`、`16..31`。第一条 MMA 解包第一个 B 寄存器的全部四字节，第二条解包下一个寄存器。单个寄存器内的高低字节不跨这两个 K16。该映射修正了早期 CPU 对照中的交错假设。
- PV 依次使用 96 个有效 key 的六段连续 K16，再执行两段 V 为零的填充指令。分母按每 64 个 key 分块，用量化前的概率归约，再扣除填充产生的概率质量。[上一批的指令取样](october_nr_attention_instruction_trace.md)
- 封存张量来自移植的 chained attention 边界，追踪序号为 `23, 28, 33, 38, 43, 48, 53, 58`。普通入口 `cc_vit_1d_attention_fp8` 独立执行同一批 Q/K/V，在归一化乘法处读出半精度操作数和结果。每块核验 49,152 个 half2 写入标记、缓冲守卫与输入完整性。插桩输出及用捕获 norm 重新编码的结果均与 chained 边界逐字节相同。[采集工具](../tools/ptx86/dump_october_attention_full.py)

## 复现与验收范围

先按[4K 管线](october_nr_render_pipeline.md)准备已封存的首帧 attention taps，按[实验版重建](october_nr_stage_replay.md)准备对应改写 PTX。下面用 Git 忽略区的新目录保存本地编译件与中间数据。

```powershell
$lab = '_dl/october_nr_20261106'
$capture = "$lab/batch8/full_boundary_tap"
$work = "$lab/attention-full-replay"
python -B -m tools.ptx86.mma_sm86_tiles
python -B -m tools.ptx86.dump_october_attention_full --capture $capture --ptx "$lab/sm86_compile_probe/module06.experimental-acq-rel.ptx" --ptxas cuda_tools/bin/ptxas.exe --work $work --blocks 11 12 13 14 15 16 17 18
python -B -m tools.ptx86.check_october_attention_tensor $capture --dump $work
```

结论覆盖这份 96-token 首帧捕获及 sm_86 移植算术。两条重放路径在同一 RTX 3060 上运行，前者保留改写内核，后者独立重建张量计算。完整 QK logits、量化前概率、其他输入几何及 RTX 50 原版的逐层算术仍待对照。全网原生等价验收仍需用同输入、同预设的 RTX 50 原版张量。
