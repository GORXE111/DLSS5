# dllq — DLL / PTX 结构化检索

给 agent 用的二进制分析检索层。**结构索引优先，词法检索兜底，向量层可插拔。**

```
python -m dllq index          # 建索引，约 5 秒
python -m dllq                # 命令表
```

## 为什么不是标准 RAG

在二进制/PTX 分析里，实际的查询绝大多数是**精确查询**，不是语义查询：

| 真实问过的问题 | 向量相似度 | 本方案 |
|---|---|---|
| `0.894531` 在哪出现 | 几乎无区分度 | `const 0.894531` 秒级命中 |
| `%r2677` 在哪定值 | 完全无效 | `def %r2677 --kernel X` 0.14s |
| 谁用了 `ex2.approx` | 无效 | `ops ex2.approx` 精确计数 |
| `0x3FFC4000` 是什么 | 无效 | `const 0x3FFC4000` |

38 MB PTX 切块嵌入会产生 8 万+ chunk，而其中真正需要"语义"召回的部分——
"哪段在做归一化"——可以用**指令特征**精确回答（`rsqrt` + `shfl.bfly` + 无 `sub`），
比嵌入更准且可解释。

所以骨架是：**SQLite 结构索引 + 行级词法扫描 + 带引用输出**。
向量层留了接口（`notes.jsonl` 是天然的小语料），需要时再挂。

## 索引里有什么

| 表 | 内容 | 规模 |
|---|---|---|
| `file` | 目标文件 + SHA-256 | 1 |
| `pe_section` / `pe_resource` | PE 段与资源目录 | 7 / 2 |
| `fatbin` | 按魔数 `0xBA55ED50` 切出的容器 | 15 |
| `kernel` | PTX entry 名 + 行范围 | 231 |
| `instr` | 每 kernel 的指令频次 | ~9k |
| `konst` | 每条 `mov.b32` 立即数，已解码 f32/f16x2 | ~7.8k |
| `strtab` | 二进制字符串，已分类 | 62,819 |
| `wrec` | WEIGHTS_HT 记录：名字/字节/参数/偏移 | 153 |

## 设计原则（这几条决定了它好不好用）

1. **每条结果带引用。** 输出统一是 `fatbin_NN.ptx:行号`，可直接
   `sed -n '9236p' fatbins/fatbin_01.1.sm_120.ptx` 取原文。
2. **输出保持小。** 同一常数散在 40 个 kernel 变体里就折叠成一组，
   不往上下文里灌重复行。
3. **渐进披露。** `map` → `kernel` → `consts-in` → `def`，
   每层只给下一步需要的信息。
4. **命令按「想问什么」组织**，不按「数据结构」组织。
   `python -m dllq` 打出来的是问题清单不是 API 列表。
5. **零重依赖。** 只用 Python 标准库；`ripgrep` 有则加速，没有则纯 Python 回退。

## 常用路径

```
dllq map                                  # 全局地图，一页看完
dllq const 0.894531                       # 数值反查（自动试 f32/f16/i32 编码）
dllq const 0x3FFC4000
dllq kernel vit_attention -d              # kernel 指令构成
dllq consts-in cc_vit_attention           # 该 kernel 内全部立即数（已解码）
dllq ops rsqrt.approx                     # 谁用了这条指令，按 fatbin/kernel 汇总
dllq def %r2677 --kernel cc_..._pre_block_swin_1h_32_1
dllq grep "shl\.b32 .*, 4" --fatbin 6 -A 2
dllq str SkinStructure --cls param
dllq weights                              # 71 个 block 的参数分布
dllq weights --block 31
dllq isa sm_86                            # 让 ptxas 给出阻塞指令清单
dllq shape                                # 从 C++ 模板名读出层配置（硬证据）
dllq shape --weights --block 31           # 用参数量反推形状
dllq note add "..." --cite fatbin_06.ptx:22637 --tag attention
dllq note find softmax
dllq ask "0x3FFC4000 是什么"               # 不确定用哪个命令时的前门
```

## 典型工作流

找一个数字的来龙去脉：

```
dllq const 1069039616
  → fatbin_06.ptx:22637 在 cc_vit_attention 内
dllq consts-in cc_vit_attention
  → 一次看全该 kernel 的 9 个常数，发现配对的 1073553408
dllq grep "max\.f16x2" --fatbin 6 -B 4 -A 6
  → 拿到完整指令序列
dllq note add "attention clamp 区间 [1.4394,1.9775]" --cite fatbin_06.ptx:22637
```

## 换一个 target

路径写死在 `dllq/core.py` 顶部（`ROOT` / `DLL` / `FATBINS` / `CUDABIN`）。
换 DLL 改这四行再 `index` 即可。PTX 需要先用 `cuobjdump -ptx` 导出到 `FATBINS`，
命名为 `fatbin_NN.1.sm_XXX.ptx`。

## 层形状从哪来

不要靠参数量硬猜。PTX 的 `.shared` 符号里带着 C++ mangled 模板名，
配置整数就写在里面 —— 这是形状的硬证据：

```
FusedSwin2dFfwdConfig<512, 64, 4, 8, 8, ...>            核宽 512
CrazyCuckooFusedSwin2d4HConfig<256, 32, 4, 8, 8, 8, ...> W=256 headDim=32 heads=8
Conv2d1x1Config<1024, 512, ...>                          512 -> 1024（第一位是输出）
```

`dllq shape` 抽的就是这个（288 个实例 / 12 种 Config）。
`shape --weights` 再用参数量反推做交叉核对——两边对上才算数。

## 向量层（已装）

```
python -m pip install fastembed        # ONNX，约 120 MB
python -m dllq vec build               # 1,427 条 x 384 维
python -m dllq vec search "where is gaussian noise generated"
```

**它能干什么、不能干什么，实测过：**

| 查询 | 结果 |
|---|---|
| `skin and material detail control` | 命中 `DLSSNR.SkinStructureStrength` ✓ |
| `how are weights quantized` | 命中对应笔记 ✓ |
| `which kernel does normalization`（初版） | 返回 `cuda_capture_kernel` ✗ |

第三条失败是因为 kernel 名字里根本没有 "normalization" 这个词。
两步修复后才可用：

1. **不嵌入名字，嵌入能力** —— 用指令特征生成描述
   （`rsqrt.approx` → "normalization variance reduction rsqrt"）
2. **每个 (kernel, 能力) 单独成行** —— 挤在一个字符串里会互相稀释

修完 `where is gaussian noise generated` 才正确命中 pre-block。

**定位：语义层负责把你带到正确的邻域，精确答案仍然由结构索引给。**
不要用它回答"某常数在哪"这类问题。
