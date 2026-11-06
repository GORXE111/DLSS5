# 十月版 Small FP8 的 4K ViT 全链复算

2026-11-06。用 RTX 3060 的 `310.9.16.0` Small FP8 实验版运行 3840×2160 sRGB 合成首帧。新采集包含 ViT 两端的 2D 重排、各块 FFN 输入和扩张输出，以及前一批已核对的 Q/K/V、注意力和投影边界。67 个边界共 202,735,616 B，和前一批重叠的 48 个缓冲逐字节一致。普通运行、仅记录轨迹、边界取样的 4K 输出逐字节一致。[采集和封存记录](validation/october_nr_vit_uhd_capture.json)

## 从第一个 2D 输入连续算到最后一个 2D 输出

独立复算读入块 11 的 2D 激活和原始权重。CPU 执行输入重排，接着逐块计算 FFN 扩张、FFN 压缩、QKV、注意力和输出投影。每一块的计算结果直接成为下一块输入，最后再由 CPU 重排回 2D。矩阵乘使用独立的 RTX 3060 FP16 MMA，归一化、激活与量化沿用已经实测的算法。各层实测值用于计算差异和核对哈希。[全链报告](validation/october_nr_vit_uhd_chain.json)

**66 个生成后的校验边界，共 200,523,776 B，全部逐字节一致。** 其中有效数据为 199,065,600 B，填充为 1,458,176 B，捕获与预测的填充均为正零。FFN 输入行记录的是输入重排或上一个块输出在下一次提交时的同一份内容，统计按实际校验边界计数。

| 校验位置 | 边界数 | 比较字节 | 差异字节 |
| --- | ---: | ---: | ---: |
| 两端 2D 重排 | 2 | 4,440,064 | 0 |
| FFN 输入、扩张输出、压缩输出 | 24 | 106,954,752 | 0 |
| Q、K、V | 24 | 53,477,376 | 0 |
| 注意力输出 | 8 | 17,825,792 | 0 |
| 输出投影 | 8 | 17,825,792 | 0 |

ViT 使用 `36×60` 的 2,160 个逻辑 token，并按 32 对齐存为 2,176 个槽。FFN、QKV 和输出投影按完整 16-token 分块处理，独立模型明确写出尾部的正零。注意力按 64 个 key 流式累加，保持同样的 FP16 运算顺序。输入重排写出 16 个全零槽，输出重排只还原 2,160 个有效 token。两端重排各自的 2,228,224 B 和 2,211,840 B 已单独核对。[重排报告](validation/october_nr_vit_uhd_repack.json)

## 96-token 用例的编码器末端和解码器入口

块 10 的末端投影读取上一步池化后的 `8×12×512` TIN2D 张量，逐位置投影为 `8×12×1024`，再由输入重排送入 ViT。使用原始 E4M3 权重独立复算，16 次 K32 各按两次 K16 执行，最后量化。隔离入口和独立算术的 98,304 个输出字节全部吻合。[编码器末端报告](validation/october_nr_encoder_final_head.json)

块 19 同时读取 ViT 2D 输出 `block18.out0` 和编码器跳连 `block10.out1`。固定记录提供 `1024→512` 的 FP8 投影与 512 个 FP16 跳连系数。复算将有效的 6×10 投影按最近邻扩为 12×20，然后把按通道缩放的跳连与投影做一次 FP16 融合乘加，最后编码为 E4M3。隔离内核、独立 sm_86 算术和零 ViT 输入的 CPU 跳连对照，各自的 122,880 个 FP8 字节全部逐位吻合。将乘与加分开取整会产生 371 个字节差异，实际 `HFMA2` 消除了这些差异。[解码器入口报告](validation/october_nr_decoder_input_replay.json)

使用与已验证 ViT 串联相同的 `reset` 用例，另取编码器和解码器边界做交叉核对。编码器预测经 CPU 重排后吻合 ViT 串联的初始哈希，ViT 串联末尾的 1D 输出经 CPU 重排后吻合解码器输入哈希。`reset` 的解码器入口 122,880 B 也逐位复现。三份封存的最终画面哈希一致。[接口哈希](validation/october_nr_vit_96_edges.json) · [解码器复测](validation/october_nr_decoder_input_reset.json)

## 复现

从仓库根目录运行，使用已核验的 RTX 3060 实验运行目录。原件 DLL 只用于读取权重资源。`--out` 指向 Git 忽略区的新目录。[运行目录准备](october_nr_attention_holdouts.md)

```powershell
$lab = '_dl/october_nr_20261106'
$base = "$lab/batch9/controlled_cli/tap"
$out = "$lab/vit-uhd"
$roles = @('--role','block11.repack.input','--role','block11.repack.output','--role','block18.out0')
$roles += @(foreach ($i in 11..18) {
    foreach ($r in 'ffn.expand.input','ffn.expand.output','qkv.input','attention.in0','attention.in1','attention.in2','attention.out') { '--role'; "block$i.$r" }
    '--role'; if ($i -eq 18) { 'block18.repack.input' } else { "block$i.out1" }
})
$common = @('--kind','experimental-sm86','--dll',"$base/nvngx_dlssnr.dll",'--harness',"$base/nr-lab.exe",'--bridge',"$base/nvngx.dll",'--support',$base,'--tap-cubin',"$base/tap.cubin")
python -B -m tools.capture_october_taps @common --out $out --suite october-uhd --case resolution_3840x2160 --frame 0 @roles
python -B -m tools.ptx86.check_october_vit_repack "$out/tap"
python -B -m tools.ptx86.check_october_vit_chain_uhd "$out/tap" --original-dll "$lab/original/nvngx_dlssnr.dll"
python -B -m tools.capture_october_taps @common --out "$out/encoder96" --suite smoke --case reset --frame 0 --role block10.final_head.input --role block10.out0 --role block11.repack.input --role block11.repack.output
python -B -m tools.ptx86.probe_october_encoder_final_head "$out/encoder96/tap" --original-dll "$lab/original/nvngx_dlssnr.dll" --module-cubin "$lab/sm86_remaining_modules/module05.sm86.cubin"
python -B -m tools.ptx86.probe_october_decoder_input "$lab/batch8/full_boundary_tap" --original-dll "$lab/original/nvngx_dlssnr.dll" --trial-cubin "$lab/sm86_remaining_modules/module07.sm86.cubin"
python -B -m tools.capture_october_taps @common --out "$out/decoder96" --suite smoke --case reset --frame 0 --role block10.out1 --role block18.repack.input --role block18.out0 --role block19.out1
python -B -m tools.ptx86.probe_october_decoder_input "$out/decoder96/tap" --original-dll "$lab/original/nvngx_dlssnr.dll" --trial-cubin "$lab/sm86_remaining_modules/module07.sm86.cubin"
```

4K 串联覆盖实验移植的 ViT 和两次重排。编码器末端、解码器入口的算术结果覆盖各自的 96-token 捕获。更上游的编码器池化从量化前 FP16 寄存器取值，编码器其余阶段和解码器后续阶段仍待独立验收。RTX 50 原版的逐层对照也仍待完成。
