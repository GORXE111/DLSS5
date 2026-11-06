# DLSS-NR 310.8.0 逆向与 RTX 30 移植

更新：2026-11-06。现有移植和独立参考主线锁定 NVIDIA `nvngx_dlssnr.dll` **310.8.0** 的 Neural Rendering（NR，feature 18）。官方 RTX Remix **310.9.16.0** Small FP8 隔离实验版已在 RTX 3060 上完成 D3D12 整网评估及 4K 同尺寸捕获。

**状态：原始权重和完整主网络已经恢复，RTX 3060 移植可运行；独立 PyTorch 实现高度接近移植版。尚未完成与 RTX 50 原生输出的严格等价验收。**

## 两条运行路径

| 路径 | 已完成 | 验收边界 |
|---|---|---|
| `tools/ptx86/` + 原 NVIDIA DLL | 把 sm_120 指令改写为 sm_86，在 RTX 3060 上运行原权重；保留原 DLL 的封装和功能实现 | 默认 FP16 累加是速度优先的近似路径；FP32 变体也尚未与原生硬件逐位对齐 |
| `torch/dlss5/` 独立参考 | 解码原权重，执行 71 步网络、历史重投影、控制参数及已验证后处理 | 在已记录用例中按相关 ≥0.999、MAE ≤3/255 验收；逐例误差见报告，尚非逐值相同或实时替代后端 |

不得把整图相关系数当“还原百分比”。插件与 nr-lab 的逐字节一致，验证的是同一移植版经过不同接入路径的自洽性。

- [十月版 Small FP8 整网实测](research/october_nr_ampere_runtime.md)。13 个合成用例通过，77 条 GPU 权重记录与原件相符，四帧提交和 ViT 边界张量已核对。RTX 50 原版数值对照仍待验收。[静态图](research/october_nr_small_execution.md) · [原件调查](research/october_nr_remix.md)
- [离线同帧查看器](research/nr_viewer/README.md)：从本地捕获生成可拖动的前后对照、热图和原尺寸局部。已用同一次 Present 的 3840×2160 画面及 1920×1080 NR 工作纹理验证；[测量报告](research/validation/batch5_sponza_capture.json)。

## 数据与来源

Git 仓库只跟踪本项目源码、研究记录和文档，不跟踪 NVIDIA DLL、权重、提取/改写后的 PTX、cubin、显存抓取、测试画面或安装包。运行需要自行准备对应版本的合法本地资源。

- 原始 DLL SHA-256：`e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e`。
- `WEIGHTS_HT.bin`：147,695,410 字节，SHA-256 `836f445d06ecd2e59bb9f17b84b91c143396fd76ccda1c9dc7fe81d5edd548f4`。
- 153 条记录全部与原 DLL 的 `WEIGHTS_HT/1033` 对应，并在 71 步调度中各引用一次；这证明来源、范围和记录覆盖，不证明数值等价。

```powershell
# 先放置 dlss5/nvngx_dlssnr.dll，并准备本地 CUDA 工具链
python -m tools.dllq index
python -m tools.verify_assets
python -m tools.ptx86.test_emu
python -m tools.ptx86.build_dll
```

PyTorch 参考在本机使用 Python 3.11、NumPy、PyTorch 2.5.1+cu121；二进制构建使用本地 VS2022 / CUDA 工具及自备 SDK。相关本地路径见构建脚本，不会自动下载运行依赖。

## 验收入口

```powershell
# 在仓库根目录执行；缺资源或指标不达标须返回非零
python -m tools.verify_assets
python -m tools.ptx86.test_emu
python torch/check.py --report research/check_frame.json
python torch/check_frames.py --report research/check_frames.json
python torch/check_motion.py --report research/check_motion.json
python torch/check_res.py --report research/check_res.json
python torch/check_postprocess.py --report research/check_postprocess.json

# 先构建 fallback，再执行自动回归；不需要真实游戏
plugin\fallback\build.bat
powershell -NoProfile -ExecutionPolicy Bypass -File plugin/fallback/run_tests.ps1
```

默认数值门槛是**对移植版的回归门槛**，不意味着官方原生等价。逐块/逐级研究依赖本地抓取数据；缺少覆盖时必须报告缺项，不能作为“全部通过”。参见 [PyTorch 说明](torch/README.md)。

## RTX 50 原版对照

`harness/reference.py` 提供有版本和硬件证据的捕获、校验与比较流程：

```powershell
# 本机移植版：输出目录必须不存在
python harness/reference.py capture --dll sm86_port/nvngx_dlssnr.dll --kind port-sm86 --suite full --out harness/captures/port-3108

# RTX 50 机器：使用同一份测试程序、调用桥和辅助 DLL
python harness/reference.py capture --dll dlss5/nvngx_dlssnr.dll --kind official-sm120 --suite full --out harness/captures/native-3108

# 将两组输出放在同一机器后逐值比较
python harness/reference.py compare --reference harness/captures/native-3108 --candidate harness/captures/port-3108 --criterion bit-exact --report research/native_comparison.json
```

官方原版标签要求原始 DLL 哈希匹配、存在 Blackwell GPU，且执行日志确认选中了该 GPU；本机 RTX 3060 会明确拒绝生成该标签。捕获保留命令、硬件、DLL/程序哈希、输出及日志，不保留复制的 DLL。比较仅对捕获用例有效，不能外推所有输入、游戏或版本。`full` 目前包括 21 组重置、历史、运动、参数、遮罩/UI、分辨率和 HDR 合成用例。

## 游戏接入与交付

- `plugin/fallback/`：DX11/DX12 的 Present 兼容接入，使用图像光流、可选降分辨率及自定义合成。与官方引擎集成分开验收，Vulkan 游戏注入尚未实现。
- `plugin/manager/`：识别游戏、安装/卸载、调参数；OptiScaler 模式依赖其 DLSSNR 分支。
- `python plugin/build_plugin.py`：在本地生成安装包。包内存在 NVIDIA 与第三方组件，不属于本仓库的源码交付物；不能把本地打包成功当成已获得公开分发许可。

当前完成项、实测结果及剩余条件集中在 [PLAN.md](PLAN.md)。第一批模型数值/控制参数共 83 项、后处理专项 16 项、插件 29 项和安装安全 15 项通过。第二批修正 OKLab 迁移采样边界，扩展后 34/34 项、17/17 采样探针通过，仍是未接入默认网络的独立实验参考；新增 temporal 套件 12 组捕获及 6 项场景切换/重置关系检查通过。这些是各自范围内的回归验收。不启动联网游戏，不访问 `E:\ZZZ`。

第二批还完成安全修订后 15 份内核全量重编译、第二轮全部内容缓存复用及严格报告核验；两份 DLL 与当时现用/包内模型逐字节相同。新旧 DLL 的 3 项 smoke 对照也逐字节一致，构建路径与报告保护的 18 项 CPU 测试通过。完整证据见 `research/batch2_build_summary.json`。

第三批已接入 RTX 3060 实测的历史纹理过滤和重投影坐标，整网 20 项通过，Graph 与 eager 逐值一致；隔离采样仍有 2/12 项失败，专项没有标成全通过。新的二维运动套件 19 组捕获、35 项检查通过。全新参考的 71/71 调度步、79/79 比较通过。FP8 NaN 修复通过六种 GPU 配置及两次一致的全量构建；新旧 52 场景、53 个输出文件逐字节相同，默认 DLL 和安装包已同步为 `f454090a…fef23382`，旧版完整保留。具体范围和新增 Reset/MV 接口缺口见 [PLAN.md](PLAN.md)。

第四批修复独立网络的 Reset/MV 语义、前后处理融合乘加/硬件倒数，以及 pre 颜色和历史的分步 half 舍入。冻结源码最终 7 套回归通过：采样 16 探针 + 9 完整 post + 20 整网及 Graph、Reset、71 步/79 比较、二维运动 35 项、分辨率 3 项、控制 18 项、后处理 16 项；新源码 35 项 CPU 方法通过。门槛未变，旧失败和局部时序误差增加均保留。新增报告输入保护；自写微型 PTX 倒数经现有 CUDA Driver 执行，CPU fallback 不保证 CUDA 数值等价。各套件混用已标识的 d442/f454 本机参考，不能视作 RTX 50 原生验证。最新证据见 [汇总](research/validation/session_summary.json) 和 [PLAN.md](PLAN.md)。

## 目录

| 路径 | 内容 |
|---|---|
| `torch/` | 当前独立网络和后处理参考 |
| `research/` | 探针、消融、NumPy 参考及本地实验 |
| `tools/notes.jsonl` | 按时间追加的研究记录；后来的更正覆盖早期推断 |
| `tools/dllq/` | DLL、PTX、权重结构检索 |
| `tools/ptx86/` | 指令改写、融合、构建与仿真验收 |
| `harness/` | D3D12 / Vulkan 独立调用、原版对照工具 |
| `plugin/` | 兼容接入、管理器与本地打包 |
| `torch/legacy/` | 已被后续实验取代的早期骨架，不用于当前验收 |

本项目与 NVIDIA 无关，面向研究与互操作性学习。
