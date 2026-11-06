# 十月版 Small FP8 的 4K 原尺寸与投影实测

2026-11-06。RTX 3060 的 `310.9.16.0` Small FP8 实验版使用 `nr-lab` 的 standalone NR 路径，以 **3840×2160 输入和输出**处理一帧确定性 sRGB 合成场景。输出哨兵与四象限检查都达到 100%，NR 比例回调为 0.85。普通运行、仅记录提交轨迹和插桩取样的最终 PPM 逐字节一致，独立重复采集也得到相同的边界张量。[4K 采集与图像证据](validation/october_nr_attention_4k_capture.json)

## 2,160 个有效 token 与 2,176 个存储槽

八个注意力块的 ABI 记录了 `36×60` 个逻辑 token、32 个头、每头 32 个通道。每头启动 9 个查询 CTA，每个 CTA 最多覆盖 256 个查询。显存布局按 32 个 token 对齐为 2,176 个槽，每个 Q/K/V/O 缓冲占 **2,228,224 B**。实际地址间隔和 launch 几何与此一致。全部 32 个注意力边界的末尾 16 个 token 都是正零。[封存范围与填充核对](validation/october_nr_attention_4k_capture.json)

独立算术按 64 个 key 流式处理，保持 34 轮归约和 K16 的原顺序。最后一轮加入 16 个零 key，在分母累加完成后扣掉其概率质量。普通内核仍计算填充查询的中间值，随后将对应的 FP8 输出槽清零。直接量化这些中间值会产生 130,976 个非零值，实际对应位置全为正零。八块共 **17,825,792 个 FP8 输出字节**逐位吻合，含 17,694,720 个有效输出和 131,072 个补零字节。PV、倒数和量化前值各核对了 17,825,792 个 FP16 位置，均逐位一致。[4K 三组 FP16 中间值](validation/october_nr_attention_4k_stages.json) · [4K 独立重放](validation/october_nr_attention_4k_replay.json)

## 注意力前后的两个算子

另取一组 640×360、96-token 的捕获。链式提交确认 FFN contract 的输出同时作为 QKV 的输入和输出投影的残差。将两处普通入口、实测输入和对应原始权重记录隔离运行，八块 Q/K/V 的 2,359,296 个 FP8 字节和八块输出投影的 786,432 个字节均与整网对应边界吻合。[QKV 复算](validation/october_nr_qkv_replay.json) · [输出投影重放](validation/october_nr_attention_projection_replay.json)

输出投影的独立 sm_86 算术重放逐位复现了全部 786,432 个字节。每个点积以四组连续 K256 计算，首组用 FP16 缩放残差作初值，随后按实际顺序以 FP16 合并。零注意力对照单独验证了残差通道布局和 E4M3 编码。本批 QKV 重放逐位复现了全部 V，Q 与 K 分别差 372 和 320 个字节。后续插桩确认差异来自平方和的半精度融合乘加，已复现全部 Q、K、V。[Q/K 归一化取证](october_nr_vit_chain.md) · [本批源码版本](https://github.com/GORXE111/DLSS5/tree/a127f55277afea6ad056c5657b4785831f5cda6e/tools/ptx86)

## 4K 图像对照

同一确定性场景分别执行 NR 强度 0 和 1，保持尺寸及其他参数相同。两张 4K 图的 RGB 平均绝对差为 **12.381/255**，最大单通道差为 **38/255**。本机查看器提供可拖动对照、原尺寸局部、绝对差和亮度方向热图。按 `stage: "pair"` 将两份 PPM 交给生成器即可重建页面。[离线查看器](nr_viewer/README.md) · [源图哈希与差异](validation/october_nr_attention_4k_capture.json)

## 复现

从仓库根目录运行。`$base` 指向已核验的 RTX 3060 实验版运行目录。选择八块的 Q/K/V/O、QKV 输入和投影输出，共 48 个边界。`--out` 和 `--work` 使用 Git 忽略区中的新目录。[运行目录准备](october_nr_attention_holdouts.md)

```powershell
$lab = '_dl/october_nr_20261106'
$base = "$lab/batch9/controlled_cli/tap"
$out = "$lab/attention-uhd"
$roles = foreach ($i in 11..18) {
    foreach ($r in 'in0','in1','in2','out') { '--role'; "block$i.attention.$r" }
    '--role'; "block$i.qkv.input"
    '--role'; if ($i -eq 18) { 'block18.repack.input' } else { "block$i.out1" }
}
$common = @('--kind','experimental-sm86','--dll',"$base/nvngx_dlssnr.dll",'--harness',"$base/nr-lab.exe",'--bridge',"$base/nvngx.dll",'--support',$base,'--tap-cubin',"$base/tap.cubin")
python -B -m tools.capture_october_taps @common --out "$out/uhd" --suite october-uhd --case resolution_3840x2160 --frame 0 @roles
python -B -m tools.ptx86.dump_october_attention_full --capture "$out/uhd/tap" --ptx "$lab/sm86_compile_probe/module06.experimental-acq-rel.ptx" --ptxas cuda_tools/bin/ptxas.exe --work "$out/planes" --blocks 11 12 13 14 15 16 17 18
python -B -m tools.ptx86.check_october_attention_tensor "$out/uhd/tap" --dump "$out/planes" --modes gpu_qk_pv
```

两处投影的复现使用同一 `$roles`，将采集命令替换为 `--out "$out/projection96" --suite smoke --case reset`。原件 DLL 仅用于读取权重资源。独立复算使用已核验的 sm_86 cubin 与矩阵指令。

```powershell
$cap96 = "$out/projection96/tap"
$cubin = "$lab/sm86_compile_probe/module06.experimental-acq-rel.cubin"
python -B -m tools.ptx86.check_october_qkv $cap96 "$lab/october_weight_tap" --trial-cubin $cubin --gpu-mma
python -B -m tools.ptx86.probe_october_attention_projection $cap96 --original-dll "$lab/original/nvngx_dlssnr.dll" --trial-cubin $cubin
```

本批的 4K 数据来自一个 sRGB 合成首帧，覆盖实验移植的注意力及写入合同。投影的独立算术验收使用另一个 96-token 用例。RTX 50 原版的逐层数值仍需同输入、同预设的实机对照。
