# DLSS5 PyTorch 骨架

从 `nvngx_dlssnr.dll` 逆向出的 shape-correct 模型骨架 + 权重加载器。

```
python check.py             # 形状核对（零剩余 + 覆盖率）
python check.py -v          # 逐条记录列出 slot
python check.py --forward   # 跑已闭合 block 的前向
python -m dlss5.topology    # 71 个 block 的镜像拓扑
```

## 验收标准只有一个：零剩余

加载器不允许有「大概是这样」的余量。每条记录的字节必须被 slot 完全消费，
153 条相加必须等于 **73,841,889**。任何一处对不上就 assert 失败。

```
家族                       参数            已定名      覆盖率
----------------------------------------------------
single1h             95,616         36,936    38.6%
single            7,116,408        786,720    11.1%
cuckoo4          14,761,440     14,761,440   100.0%
cuckoo5           1,246,248      1,246,248   100.0%
split16h         50,348,616     50,348,616   100.0%
bridge              262,656        262,656   100.0%
output               10,905              9     0.1%
----------------------------------------------------
合计               73,841,889     67,442,625    91.3%
```

**91.3% 的参数已定名到具体 slot**，其余是单记录 block 里顺序未解的打包段。

## 置信度分档

代码里每处都标了档：

| 标注 | 含义 |
|---|---|
| `[实测]` | 从 PTX 或权重字节直接读出，可用 `dllq` 复查 |
| `[推断]` | 形状/计数闭合，但语义靠推理 |
| `[占位]` | 形状对，连接方式未定，forward 只是合理猜测 |

**`ops.py` 里的东西全是 `[实测]`** —— 每个常数都能用 `dllq const <值>` 查回 PTX 行号。

## 精确恢复的算子

```python
# 激活 MpCubicSiluActivation      fatbin_01.ptx:1975-2020
t = clamp(x, -4, +4)
y = x * (-0.055908203*t*|t| + 0.447265625*t + 0.894531250)

# 归一化 —— 没有均值减法（全 pre-block sub.f16x2 = 0）  fatbin_01.ptx:9216
y = x * rsqrt(max(sum(x^2), 6.2e-5))     # 下界钳位，非 +eps；未除以 N

# 注意力指数 Schraudolph 位运算       fatbin_06.ptx:22637-22664
s = clamp(logit*0.0895394683 + 1.70936143, 1.4394531, 1.9775391)
w = f16_bits((bits(s) << 4) + 0x3FFC4000)     # = 2^(16s-31) ∝ exp(logit)
# 有效 logit 区间 ±3.0
```

`schraudolph_exp_exact()` 走真正的位运算，`schraudolph_exp()` 是解析等价形式
（`2^(16s-31)`）。两者最大相对差 **5.74%**，正是 Schraudolph 的误差包络。

## 已完全闭合的两个 block

**`SplitSwin16HBlock`**（block31–38，占 68% 参数），5/5 精确：

```
layer0  ffn_expand    512→2048  ×2  + 8       2,097,160
layer1  ffn_contract  2048→512  ×2  + 1024    2,098,176
layer2  attn_scale H×2×2 + qkv 512→1536 ×2    1,572,928
layer3  attn_cos_skip 单标量                          1
layer4  final_head 512→1024 + bias 1024         525,312
```

**`CuckooBlock`**（block23–29 / 40–47），10/10 slot：

```
layer0  weight0 2W² ×2 · weight1 W² · weight2 W²
layer1  weight3 2W² ×2 · ffn_cos_skip 2W
layer2  qkv 3W²×2 · attn_bias H×2×64×64 · attn_scale H×2×2
layer3  projection_weight 2W² · attn_cos_skip 2W
```

前向冒烟（CUDA）：

```
SplitSwin16H  in (2, 64, 512) -> out (2, 64, 1024)   有限值 True
CrazyCuckoo   in (2, 64, 256) -> out (2, 64, 256)    有限值 True
```

## 没做的，以及为什么

**`SingleLayerBlock.forward()` 故意抛 NotImplementedError。**

单记录 block（block0–22 / 48–70，占 10% 参数）把整个块打包成一条记录。
参数量公式已闭合（`a·W² + 194W + 24`，尾部 8 个零填充），`attn_bias` 的段边界
也在多 block 投票中确认，但**其余段归属哪个 slot 未解**。随便定个顺序会静默出错——
宁可显式失败。

**端到端 forward 也没有。** DLL 里的 builder 走描述符 + 工厂分派，
序列化的激活层序列不是明文，所以 71 个 block 的执行顺序是结构推断而非证据。
`topology.py` 给的是结构，不是被证实的执行顺序。

## 复查任何一个数字

```
cd E:\DLSS5\tools
python -m dllq const 1063583744        # SiLU 的 C 系数
python -m dllq const 0x3FFC4000        # Schraudolph 偏置
python -m dllq unpack block23.layer2.layer --width 256
python -m dllq shape                   # C++ 模板配置
python -m dllq note find attention
```
