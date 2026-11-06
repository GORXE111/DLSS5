# 2026 年十月版 DLSS-NR：RTX Remix 310.9.16 原件调查

日期：2026-11-06。已下载并分析 NVIDIA 官方 RTX Remix 1.6.1 的 Windows x64 发布包。现用 310.8.0 3060 移植与安装包不变。本报告把官方发布口径、可复算的静态证据、编译探针和仍缺的动态证据分开。

## 原件与出处

NVIDIA [十月公告](https://www.nvidia.com/en-us/geforce/news/dlss-5-faster-ai-model-hogwarts-legacy-rtx-spark-laptops/)把约 1.18 亿参数、较九月 GPU 开销约减半的 NR 模型列入 10 月 23 日 RTX Remix 更新。[RTX Remix 官方 1.6.1](https://github.com/NVIDIAGameWorks/rtx-remix/releases/tag/remix-1.6.1)提供公开运行时；本地只提取目标 DLL，没有安装或执行该 DLL。校验详见[来源记录](validation/october_nr_remix_provenance.json)。

| 文件 | 本地核验 |
|---|---|
| Windows x64 release ZIP | 557,087,446 字节；SHA-256 `7dd1d44492a609050e44b0e3581cec31135d1ed88b4a91123b4faae91955716a`，与官方资产页一致 |
| ZIP 内 `nvngx_dlssnr.dll` | 266,424,944 字节；SHA-256 `2bc19bc09a7b89dd44f73e2ac0108fcc586e2154bdd0830f242f4f040e28789f`，与官方校验清单一致 |
| Windows PE 信息 | `310.9.16.0`，`NVIDIA DLSSNR - DVS PRODUCTION`，`CL 39025366`；Windows 验证 NVIDIA Authenticode 签名为 Valid |

## 两套权重与选择方式

| 资源 | 序列化大小 | 记录数 | 编号块 |
|---|---:|---:|---:|
| `WEIGHTS_HT` | 147,695,410 B | 153 | 71 |
| `WEIGHTS_CC_SMALL_FP8` | 113,893,343 B | 77 | 31 |

- 310.9.16 的 `WEIGHTS_HT` 与原 310.8.0 **整个资源逐字节相同**。这只证明保留了同一权重，不意味着新旧运行时在大档的输出一致。
- 小档载荷 113,887,490 B，约为大档的 77.1%。两档有 31 条同名记录，但没有相同的记录载荷（按同名逐字节对照，且跨名的大小与 SHA 也无重合）。小档所有 31 个块都能在大档找到相同的“记录后缀、各记录大小”组合，说明复用了结构形态，不能据此断定复用了权重。
- 小档块 11–18 保留 8 组与旧 ViT 块尺寸组合一致的记录，占小档载荷约 88.4%。其余重复块明显减少。**31 是序列化块数，不能写成 31 次 CUDA 发射**。记录尾部的计数等于载荷半字数，不能用来确认官方约 1.18 亿学习参数，也不能把约 1.14 亿个存储字节当作同一个量。
- 本地反汇编确认，`DLSSNR.Hint.Render.Preset=1` 选 `WEIGHTS_HT`，`=2` 选 `WEIGHTS_CC_SMALL_FP8`，查表不匹配时落到小档。官方 Remix 的公开 Vulkan 接入源码显式选轻量预设 B；界面的 A/B/C 外观控制另写入 `Style`。旧 `nr-lab` 默认 `--model 1` 会选大档，后续对照小档须明确 `--model 2` 并验证实际加载的资源。详见[主机接口证据](validation/october_nr_remix_host.json)及[权重报告](validation/october_nr_remix_weights.json)。

## CUDA 内核与接口

两版均有 15 个 CUDA 模块，每个都有 sm_120 PTX 与 cubin。模块 8–15 整个 fatbin 逐字节相同；模块 1–7 变化。静态 PTX 入口数从 **231 降至 120**，移除的 111 个名字均不含 FP8 标记，保留的 120 个名字全都在旧 DLL 出现。保留入口不能视作原代码：逐名比对 cubin 代码段，**71 个相同，49 个不同**。两个 pre FP8 入口的参数数组从 **264 B 增至 272 B**。这些是静态符号、参数布局与代码段的对照，不是运行时发射次数。[内核报告](validation/october_nr_remix_kernels.json)

主机新增 `DLSSNR.MaskBlendMode`。模式 0 保留显式 ControlMask 与 AutoMask 的旧互斥关系；存在显式 mask 且 AutoMask 有效时，新 pre 分支按模式 1/2/3 将对应强度取平均、最小或最大。主机和 PTX 各自的分支已静态确认；尚未完整追到最后一段启动参数绑定，也没有四种模式的逐像素验收。

## 3060 可复用性

用现有改写器，七个改动模块中 **六个**通过 sm_86 汇编。模块 6（含 ViT）初次失败：21 处 GPU 范围 acquire fence 不受 sm_86 支持。在独立副本里试用同范围 acquire-release fence 后，该模块的 31 个代码段均成功生成。**这只是编译通过**；未确认线程间同步、输出或速度等价，默认改写器没有修改。[编译探针](validation/october_nr_remix_compile.json) · [NVIDIA PTX fence 规范](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar)

旧构建脚本绑定 310.8.0 的原件哈希、宿主偏移与参数合同，不能把新版 DLL 改名替换。下一步应为 310.9.16 单独建验证目标：先在 RTX 50 分别锁定预设 1/2、捕获输出及真实发射序列，再做 sm_86 的同步微测、模块绑定和对照。当前没有新模型画面、3060 运行或速度测试，不能把官方的约 2 倍加速套到本项目。

## 本地复算

自行准备合法取得的两份原 DLL 和本地 CUDA 工具，将输入与输出放在 Git 忽略的目录。示例从仓库根目录执行；`--work` 必须是新的目录，里面会写出用于分析的 PTX/cubin，不要提交：

```powershell
python -B -m tools.inspect_nr_weights dlss5/nvngx_dlssnr.dll _dl/october_nr_20261106/original/nvngx_dlssnr.dll
python -B -m tools.compare_nr_kernels --baseline dlss5/nvngx_dlssnr.dll --candidate _dl/october_nr_20261106/original/nvngx_dlssnr.dll --cuobjdump cuda_tools/bin/cuobjdump.exe --work _dl/nr_kernel_comparison
```

两条命令均为离线静态分析。若要复现编译探针，可对相应候选模块的 PTX 使用 `python -m tools.ptx86.rewrite 原PTX 实验PTX`，再用本地 `ptxas -arch=sm_86` 编译。试验 fence 替换只作用于模块 6 的独立 PTX 副本。公开记录只保留来源、哈希、布局与分析结果。
