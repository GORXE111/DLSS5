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

## 2026-10-23 可复现性与验收

在仓库根目录运行 `python -m tools.verify_assets`，只读核对原 DLL SHA-256、PE 权重资源、权重文件、153 条记录和 71 步调度引用顺序。该命令不建索引、不运行网络，失败返回非零；可用 `--report path.json` 保存结果。

`python -m tools.ptx86.test_emu` 的失败现在返回非零。FP8/FP16 编解码、整数布局及伪量化/融合测试可验证改写局部行为；小数矩阵测试使用 CPU 数学参考和误差门槛，不是原生 sm120 的金标准。旧 NaN 转换差异由第三批专项修复，FP16 双段累加与原生的差异仍未关闭。

`build_dll.py` 缓存按输入内容和输出哈希复用，包含源 PTX/fatbin/原 cubin、rewrite/fuse/flow、构建逻辑、工具和数值选项。旧的无 manifest 缓存首次运行会重新构建。可先运行 `python -m tools.ptx86.test_cache` 验证缓存失效规则；不应通过修改时间戳强行授权缓存。

第二批新增 `--work` 显式指定构建/缓存目录，`--out` 与 `--report` 都按启动目录解析，避免相对路径随工作目录变化。报告路径及其临时文件先与输入、产物和缓存检查别名，通过检查后先建立 running 报告，再启动构建。逐 fatbin 记录内容缓存命中、提取/改写/编译/打包时间和产物 SHA；构建异常或工具失败会写回失败状态。命令示例（两个构建共用新建的独占 work，精度默认 f16/v1/v1）：

```powershell
python -m tools.ptx86.build_dll --work _dl/my-isolated-build --out _dl/my-isolated-build/first.dll --report research/build_first.json
python -m tools.ptx86.build_dll --work _dl/my-isolated-build --out _dl/my-isolated-build/second.dll --report research/build_second.json
python -m tools.ptx86.verify_build --candidate _dl/my-isolated-build/first.dll --second _dl/my-isolated-build/second.dll --baseline sm86_port/nvngx_dlssnr.dll --first-report research/build_first.json --second-report research/build_second.json --report research/build_verify.json
```

首次完整重建与第二次复用的本批证据集中在 `research/batch2_build_summary.json`。输出相同只证明本地构建可复现，不证明原生 sm120 数值等价；GPU smoke 也须使用相同测试程序及辅助 DLL 才能比较。

构建报告 schema 2 的验收会复核独立的两份 DLL、逐项缓存决策与汇总、完整成功阶段、当前源码/工具/PTX/原 cubin 和输出身份，记录输入报告自身 SHA。缺独立第二份产物、缺阶段、计数与逐项记录矛盾或输入哈希错误都会拒绝。初版 schema 1 报告作为历史保留，最终安全修订后使用 revision 报告重新验收。

DLL 高层只读研究可使用 `python -m tools.host_trace function 0x18001c920` 与 `python -m tools.host_trace refs 0x18001c920`；工具按 PE unwind 范围导航并打印原文件 SHA。常规 `dllq` 的包内动态导入已修正，仓库根目录 `python -m tools.dllq str DLSSNR.Style` 与 tools 目录 `python -m dllq str DLSSNR.Style` 两种入口均验证可用。

`tools/notes.jsonl` 是追加式研究历史。较新 correction 记录取代早期推断；不能把其中的历史“100%覆盖”或“待确认”直接当成当前状态。统一结论见根目录 `PLAN.md`。

## 第三批：逐步证据与 FP8 特殊值

`python -m tools.schedule_refs` 提供 `plan`、`capture`、`validate`：按固定 640×360 reset 调度，明确每个边界的发射序号、参数槽、大小与布局，分块捕获大缓冲；不搜索旧 tap 目录。25 批抓取绑定原 DLL/移植 DLL/程序/辅助组件、输入和最终图像，保存源码与捕获器身份快照。`torch/check_schedule.py` 的 71/71 步、79/79 比较通过，只证明这些明确边界的本机近似回归。

第三批修复了 `rewrite.py` / `fuse.py` 的 E4M3 特殊值：旧版本会把编码 0x7f/0xff 解成 ±480，部分编码/融合路径把 NaN 饱和成有限数。新实现保留 NaN 类别；有限值、无穷的 satfinite 饱和以及 RNE ties-to-even 用独立整数 oracle 穷举。NaN 不要求保留 payload 或符号，不把该合同扩张成 FP8 MMA 累加等价。依据为 [NVIDIA PTX cvt 规范](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)。

```powershell
python -m unittest tools.ptx86.test_numeric_special tools.ptx86.test_emu_exit
python -m tools.ptx86.test_emu --conversions-only
```

CPU 专项包括 65,536 个 half 位模式的双 lane 编码、65,536 个 packed E4M3 输入、三种编码及两种解码/融合/常量路径。另用 Torch CPU 独立交叉核对全部 half 模式，结果一致；E5M2 仅有规格常量和 oracle 自检，该版本 15 份 PTX 没有 E5M2 指令，不宣称实现了 E5M2 改写。`test_emu_exit` 的夹具已适配新命令行，仍用六个失败子用例检查非零退出传播。

完整构建证据为 `research/batch3_build_first.json`、`batch3_build_second.json`、`batch3_build_verified.json`。两份新 DLL 为 `f454090a97d18e2eb8e6008d42a5fa4e37c8e4bd251a86aedfff8901fef23382`，196,982,784 字节；与旧 d442 二进制不同，严格 verifier 如实记录该差异。首轮新空目录 15 miss、76 阶段，630.247 秒；第二轮 15 hit、16 阶段，91.098 秒，无编译工具阶段。并行研究期间的耗时不能当作速度基准。

六种实际 GPU 编解码/融合配置全部通过；默认 MMA 有限值及融合整数回归也通过，但不是 sm120 MMA 原生对照。新旧 full21、temporal12、motion2d19 共 52 场景、53 个输出文件逐完整文件 SHA 相同。默认 76 工件已整体推广，15 份缓存和内嵌 fatbin 再核验通过，旧目录保留。证据为 `batch3_numeric_gpu.json`、`batch3_gpu_regressions.json`、`batch3_promotion_revision.json` 与 `batch3_promotion_verified.json`。新安装包和默认 DLL 同步，具体交付记录见 PLAN。

## 第四批：报告输入保护与提交来源

`tools/report_paths.py` 是仅用标准库的报告路径保护。资产核验、通用 Torch 验收、迁移 host trace 和 temporal 分析在写报告前检查源文件、模型、运行组件、参考和捕获目录；目录之外的硬链接及 `.tmp` 别名也不能覆盖输入。动态登记参考会再次校验；不安全目标返回非零并保留原文件。报告先序列化再独占创建临时文件，失败只清理本次创建且身份相同的文件，不删除用户已有 `.tmp`。

```powershell
python -m unittest tools.test_report_paths
```

两处“报告覆盖输入却 PASS”的临时 fixture、修后拒绝及 13 项负例记录在 `research/validation/batch4_report_paths.json`。`.gitignore` 排除复制 runtime 与原始运行载荷，`.gitattributes` 保留原始字节换行；源码 SHA 对照依赖文件字节，不能以平台换行差异静默改写历史快照。具体提交文件以第四批发布审查的显式清单为准，NVIDIA 资源与提取产物不进入源码提交。

采样专项的自写硬件 RCP 封装、原 pre/post FFMA 证据及旧切片上下文更正见 `torch/README.md` 和 `research/batch4_sampling/arithmetic_findings.json`。它无需 nvcc 或额外包，仍要求正常 CUDA Driver；CPU fallback 不提供硬件数值等价。
