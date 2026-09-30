# DLSS5 逆向研究

对 NVIDIA DLSS5 降噪/增强模型 (`nvngx_dlssnr.dll` 310.8.0) 的独立逆向研究：

1. **RTX 30 系移植** —— 把只支持 sm_120 (RTX 50) 的 kernel 重写为 sm_86 (FP8 mma / TMA 等指令的软件模拟)，在 RTX 3060 上端到端运行 (1080p 约 45 ms)。
2. **网络结构完整还原** —— 71 个块逐一用探针 + 显存抓取解出，并写成可运行的参考实现：
   `torch/` 的 GPU 版与 nr-lab 在 3060 上的真实输出相关 0.9991 (平均差 2.4/255)。

> 本仓库**不包含**任何 NVIDIA 的二进制或数据：没有 dll、没有权重 (`WEIGHTS_HT.bin`)、没有从 dll 抽出的 fatbin/PTX/cubin、
> 没有补丁后的 dll，也没有显存抓取数据。要运行需要你自己合法持有的 `nvngx_dlssnr.dll` 310.8.0 (sha256 `e16bcf15…fc8e`)。
> 与 NVIDIA 无关，仅供研究与互操作性学习。

## 目录

| 路径 | 内容 |
|---|---|
| `torch/` | **DLSS5 的 PyTorch 参考实现** (GPU，一帧约 250 ms)，见 [torch/README.md](torch/README.md) |
| `research/` | 逐块逆向用的 numpy 参考、探针与对照脚本；`net_ref.py` 是整网 numpy 参考 |
| `tools/notes.jsonl` | 研究日志: 每条结论的证据、精度与更正 |
| `tools/dllq/` | DLL / fatbin / PTX / 权重资源的结构化检索工具 (提取 `WEIGHTS_HT.bin`) |
| `tools/ptx86/` | sm_120 -> sm_86 的 PTX 改写与 dll 重建工具链 |
| `harness/` | 独立测试程序 nr-lab (源自 DLSS5 ReShade AIO，Apache-2.0) + 性能/显存抓取钩子 (`nvprof.h`) |
| `plugin/` | RTX 30 安装包的构建脚本与安装器源码 (拒绝装到带反作弊的游戏) |
| `weights_map.json` | 权重记录表 (名字、偏移、大小)，不含权重数据 |

## 从自己的 dll 准备权重

```
# 把你的 nvngx_dlssnr.dll 310.8.0 放到 dlss5/nvngx_dlssnr.dll
python -m tools.dllq index        # 抽取 WEIGHTS_HT 资源 -> WEIGHTS_HT.bin，并建检索库
cd torch && python check.py       # 需要 research/out_f0.ppm (nr-lab --frames 1 的输出) 做端到端对照
```

```python
import sys; sys.path.insert(0, "torch")
from dlss5 import DLSS5
out = DLSS5()(color)              # color: (360, 640, 3) in [0, 1]
```

## 主要发现 (摘要)

- U-Net: 1h(32ch)→2h→4h→8h(256ch) swin 编码，16h 分组低秩 swin (512ch) + ViT-1d (1024ch, 96 token) 瓶颈，对称解码。
- swin-v2 式余弦注意力 (τ 只乘 Q)，MP 残差 `y = c⊙x + FFN(x)`，MpCubicSiLU 激活，权重矩阵 FP8 e4m3、系数 f16。
- pre_block 输入: 颜色、重投影历史 (5-tap Catmull-Rom)、3 路哈希高斯噪声；post_block 是带学习门控的时域累积器。
- 16h / ViT 的 softmax 用位运算近似 exp (不减行最大值、logit 截断)。

完整推导见 `tools/notes.jsonl`。
