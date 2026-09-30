# DLSS5 PyTorch 参考实现

`nvngx_dlssnr.dll` 310.8.0 (DLSS5 NR) 的完整前向，逐块由逆向得到，并与 kernel 的真实数据核对过。
在 GPU 上一帧约 250 ms（RTX 3060，640x360），数值上复刻 kernel 的 FP8/f16 舍入位置。

```python
import sys; sys.path.insert(0, "torch")
from dlss5 import DLSS5

net = DLSS5()                                  # 读 WEIGHTS_HT.bin + weights_map.json，解码全部权重到 GPU (~5 s)
out = net(color)                               # 重置帧: color (360, 640, 3) in [0,1]
out = net(color, hist=out, mv=mv, frame=1)     # 带历史: hist = 上一帧输出, mv (360, 640, 2) 像素位移
```

```
python check.py          # 端到端: nr-lab 第 0 帧合成输入 -> 与 3060 上 nr-lab 的实际输出比较 + 计时
python check_levels.py   # 逐级: 以抓取的 pre_block 输出为起点，各级出口与 kernel 抓取比较
```

## 验收结果

| 对照 | 相关 | 平均差 |
|---|---|---|
| torch vs nr-lab 实际输出 (第 0 帧) | 0.99912 | 2.4/255 |
| torch vs numpy 参考 (research/net_ref.py) | 0.99960 | 1.4/255 |
| 网络改动量 (输出 - 输入) vs nr-lab | 0.990 | |

逐级出口 (起点为抓取的 pre 输出): 编码 1h→8h 0.9985 → 0.990，16h 0.984，解码 8h→1h 0.976 → 0.993。
误差来自逐块 fp8 舍入的累积 (求和顺序不同导致个别值落到相邻的 fp8 格点)，不是结构差异。

## 结构

```
pre_block   10 路输入 (颜色/重投影历史/3 路噪声/常数) -> 适配器 16->32 -> 全分辨率 1h swin -> 2x2 平均
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
| `dlss5/data/` | `schedule.json` (块类型/记录/窗口平移，由 `research/gen_schedule.py` 从执行轨迹导出)、`exec_order.json`、`adapter_map.json`、`maps_block1.json` |

几个与常见实现不同、容易写错的地方 (都已实测):

- 余弦注意力 `q̂ = τ·q/|q|` (τ 只乘 Q)，相对位置偏置作为 QK 累加器初值；1h-8h 用真 softmax。
- 16h 与 ViT 的 softmax 分子是**位运算近似 exp**，不减行最大值，logit 分别被截断在 [-6,6] / [-3,3]；
  16h 先归一化再乘 V，ViT 先乘 V 再归一化。
- 16h 的 FFN 是分组低秩的 (8 组，每组 512->64->256->64)。
- 整网在补齐到 384x640 的网格上运行，补齐行按 `y' = 718 - y` 镜像取样；重置帧以颜色充当历史。

## 数据来源

逆向过程、每条结论的证据与精度记录在 `tools/notes.jsonl`；numpy 逐块参考和探针在 `research/`。
早期的"形状骨架"(打包模型已被证伪) 移到了 `legacy/`，仅作存档。
