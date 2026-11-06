# 推进与验收记录

更新：2026-11-06。本文件区分“已实现”“本机验证”和“仍待外部条件”，不把相关系数、权重命名覆盖率或接口成功率换算为整体完成百分比。

## 当前结论

现有移植主线为 `nvngx_dlssnr.dll` 310.8.0，原始 SHA-256 为 `e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e`。原权重和 71 步主网络已恢复，3060 移植已运行，独立参考高度吻合该移植版。310.9.16.0 小模型已通过独立实验整网运行，两版均待严格 RTX 50 原生数值验收。

本地约束：不启动联网游戏，不访问 `E:\ZZZ`；NVIDIA 二进制、权重、提取/改写后的 PTX、cubin 与测试画面不进入 Git。此次研究使用 RTX 3060 12 GB、驱动 610.74。

## 2026-11-06 第十批 · 注意力指令与大档中间张量

- 对十月小档普通 attention 隔离插桩，实测分母、倒数、八次 K16 和量化前半精度。已将一个 FP8 边界的差异追到 PV 第 3 次 MMA，并确认 96 个有效 key 的连续 K16 顺序。32 次独立 MMA 重放逐位一致。[逐指令实测](research/october_nr_attention_instruction_trace.md)
- 同一个 scRGB 运动用例的旧版和十月版 HT 大档通过 12 份中间张量逐字节对照，共 20,447,232 B。[配对报告](research/validation/october_nr_ht_activation_taps.json)

## 2026-11-06 第九批 · 版本因子和自动采集

- 两版大档的同源 4K 静止、平移及三组 scRGB 输出逐字节一致。固定十月运行时后，小档的 4K 合成 MAE 分别为 5.624/255 和 5.460/255。[版本因子实测](research/october_nr_version_factors.md)
- 普通和 chained attention 对真实首尾 Q/K/V 的独立重放均逐字节匹配。已用独立 MMA 探针复现一处分组差和单次最近舍入差，另一处量化边界仍待分阶段取证。自动采集在 3060 上复现全部 73 个边界。[实测数据](research/validation/october_nr_version_factor_experiment.json)

## 2026-11-06 第八批 · 4K 接入和逐层数值

- 真实提交确认 standalone 的 copy 限于 1920×1080。紧凑 NR 串 DLSS SR 写满 3840×2160。DXGI 代理通过预设 2 在真实 Sponza 同帧将 1080p 模型改动合成回 4K。[4K 接入实测](research/october_nr_render_pipeline.md)
- 首帧已抓取 73 个受支持的张量边界。首尾 attention 的独立 CPU 数值分别有 99.2574% 和 98.1405% 精确相同。RTX 50 逐层取样和同角色比对是下一项验收。[逐层实测](research/october_nr_render_pipeline.md)

## 2026-11-06 第七批 · 十月小模型整网

- 310.9.16.0 预设 2 的独立 sm_86 实验版通过 13 个 D3D12 用例。四帧真实提交与静态图吻合，显存中的 77 条权重记录与原件逐条相符，首尾 ViT 重排和 CPU 独立计算逐字节一致。[整网实测](research/october_nr_ampere_runtime.md)
- 两个 pre 遮罩片段的 6,300 个 FP16 输出均逐位匹配。已取得同尺寸 4K 强度 0/1 对照，并记录 1080p 输入直接输出 4K 只覆盖左上四分之一的失败用例。[实测数据](research/validation/october_nr_ampere_d3d12.json)

## 2026-11-06 第六批：十月小模型执行研究

- 原版 310.9.16.0 预设 2 的主机构图和权重字段台账独立复算，31 块、77 条记录全部匹配。已追通 272 B pre 参数块中的遮罩模式 `+40` 和宿主 15 个模块引用。[执行研究及复算入口](research/october_nr_small_execution.md)
- RTX 3060 隔离运行双向重排入口，16 例逐字节一致。跨 CTA 同步微测的两种栅栏合计检查 1,048,576 个值，错误对照有效。下一验收点是 RTX 50 原版预设 2 的实际资源和发射追踪，以及新版 sm_86 逐层对照。[实测结果](research/validation/october_nr_sm86_primitives.json)

## 2026-11-06 第五批：4K 同帧对照与十月版调查

- 现有 310.8.0 移植已在离线 Godot Sponza 抓到真实 3840×2160 原画与最终合成，NR 工作分辨率为 1920×1080。两对数据分别比较同一次 Present 的全链与模型边界。滑块查看器和像素热图生成器纳入[源码](research/nr_viewer/README.md)，不提交截图。[捕获和验证结果](research/validation/batch5_sponza_capture.json)
- 官方 RTX Remix 1.6.1 包中的 NR DLL 已下载、验签和比对。310.9.16 同时保留逐字节相同的大档权重和新增小档；确认轻量预设 2、样式分离、内核删减及 pre 参数扩容。已隔离探测 sm_86 编译，尚未加载或移植新版 DLL。[十月版调查](research/october_nr_remix.md)

## 2026-10-23 第四批推进

本批继续处理上一批的 Reset/MV 缺口和采样边界，并为源码提交补齐输入保护。三个子任务并行，GPU 按 Reset、采样、最终集成回归顺序使用。

**Reset 合同。** raw `PreBlock`、`PostBlock` 和导出引擎外 `Core.inputs/finish` 在 History 或 MV 缺失时跳过历史分支；Reset 直接采用当前 Color，不再把 Color 作为可被非零 MV 重投影的历史。公共 `DLSS5` 继续把 `mv=None` 解释为全零运动，保留既有便利语义，不将其冒充原 NGX 空句柄。Graph 的历史存在状态固定：reset 图忽略显式 MV；调用时 History 是否存在须与捕获状态一致，矛盾输入明确拒绝。

**采样推论更正。** 完整原 sm120 与当前 sm86 的 pre/post 机器码都使用 FFMA 计算采样中心和分数。第三批的 pre 切片准备代码使用了非 FTZ 的 multiply，改变了可融合性；其通过只能证明该人工准备代码，不能证明完整原 pre 必须分离舍入。本批保留旧四组 pre 探针并标为 legacy split 诊断，另加四组保持原 FTZ 上下文的 pre_native；真实路径统一默认 fused。

在 53×37 的临界点，硬件 `rcp.approx(37)` 与普通 RN 倒数相差 1 ULP，足以跨过纹理过滤权重边界。动态 `w2/w12` 也需要原指令的 RCP.APPROX + FFMA。新增自写 `torch/dlss5/ptx_math.py`，通过 CUDA Driver 在当前 Torch stream 发射微型 PTX；模块按活动 context 缓存并保留，预热后可进入 Graph，无需为此安装 nvcc 或额外 Python 包。未按尺寸或误差点查表。CPU fallback 使用普通倒数，不保证与 CUDA 的数值相同。

固定门槛未变：最大差 0.005/255、MAE 0.0005/255。半精度归一化前的采样专项 `research/validation/batch4_sampling.json` 的 16 组采样、9 组完整 port cubin post、20 组整网及 Graph 已通过。16 组包含 4 个 legacy split 诊断，不能全部当成真实原内核上下文；9 组完整 post 仍采用零神经权重和指定 blend 隔离历史采样，新增两组空间变化分数 MV。最初真实失败保留于 `research/batch4_sampling/isolated_initial.json`，不是通过删除用例或放宽阈值取得通过。

**原 pre 舍入。** 最终集成首次出现控制 1.5 用例相关性 0.998989，低于既有 0.999 门槛，失败报告 `batch4_final_gpu.json` / `batch4_controls.json` 原样保留。追踪确认 d442/f454 两版在该用例输出相同，Reset 的 History/MV 句柄确实为空；原 pre 对颜色和历史执行先转 half、再 half 减法和乘法。独立实现原先将整个归一化式放在 f32，遗漏了中间舍入。现对 RGB/History 六列统一显式 half 分步归一化，两种模式均执行；噪声和控制列不变，没有针对 1.5 的分支。修复后该例相关性 0.999073、MAE 2.201170/255，通过原门槛。独立 CPU oracle 的 327,680 项 × 四模式均无失配，原 pre 特征探针由 7,064 个不同值降为零；这只是特定特征探针逐值相同。分数运动诊断第二帧 MAE 1.576978→1.626913/255，仍通过，但不能写成每例改善。证据为 `research/validation/batch4_pre_round_fix.json` 和 `batch4_rounding_review.json`。

**报告输入保护。** 独立审查用临时 fixture 复现了 `verify_assets --report` 覆盖权重映射、通用验收报告覆盖已登记参考图像却仍报告 PASS 的问题。新增 `tools/report_paths.py`，先检查源文件、模型、捕获目录、硬链接及临时文件别名，再原子写报告；序列化失败不残留自身临时文件，用户已有临时文件不被删除。资产核验、通用验收、host trace 和 temporal 分析已接入；13 项专项负例通过，证据为 `research/validation/batch4_report_paths.json`。

**冻结源码最终回归。** `research/validation/batch4_rounded_gpu.json` 的七套回归均通过，前后源码 SHA 一致。采样最新报告为 `batch4_sampling_rounded.json`；Reset 最新报告为 `batch4_reset_fused_rounded.json`。

| 专项 | 最新结果 | 范围 |
|---|---|---|
| 采样 | 16/16 探针、9/9 完整 post、20/20 整网，Graph 逐值一致 | 4 组 legacy split 仍仅作诊断 |
| Reset | 3 CPU、12 kernel、8 candidate、2 Graph、1 固定历史控制全部通过 | 旧语义 baseline 6/8，仅作非门控诊断 |
| 调度 | 71/71 步、79/79 比较 | 保留的 d442 单尺寸重置参考 |
| 二维运动 | 35/35，零跳过 | 保留的 d442 19 组捕获 |
| 分辨率 | 3/3 | 有 SHA 的历史 PPM 参考 |
| 控制及遮罩 | 18/18 | 旧 nr-lab/d442 runtime；高控制另有 f454 同输出诊断 |
| 后处理 | 8/8 整网、8/8 隔离 kernel | 有限本机参考，非原生对照 |

新源码 35 项 CPU 方法通过，独立报告路径保护 13 项通过；与历史重复的方法不再次累加成总覆盖数。当前原生 DLL/安装包仍为 f454，26 个包工件校验通过，本批 Python 参考修复没有重建该 DLL。各报告实际参考身份独立记录：不能把根报告中的现装 f454 哈希套用到所有历史参考。汇总、文件字节与语法核验入口为 `research/validation/session_summary.json` / `batch4_integrity.json`；发布清单为 `batch4_release_audit.json`。

**提交范围。** 源码、文档及结构化历史报告可以提交；复制 runtime、NVIDIA DLL/权重/PTX/cubin、原始抓取和测试图像继续留在本机。`.gitignore` 新增捕获 runtime 目录排除；`.gitattributes` 保留文件原始换行字节，避免跨平台 checkout 破坏报告绑定的 SHA。前几批的失败报告和自有源码快照保留。

## 2026-10-23 第三批推进（历史）

第三批由模型采样、逐步证据和指令数值三个子任务并行推进，独立复核与 GPU 工作串行协调。以下是本批已落地的结果；前两批报告作为历史保留。

| 工作 | 本批结果 | 范围与限制 |
|---|---|---|
| 历史纹理采样 | 已接入 RTX 3060 实测 TEX 精度和 FMA 坐标顺序 | pre/post 及引擎外 finish 一致；Catmull 的 div/rcp.approx 仍有近似 |
| 采样专项 | 隔离 10/12、完整 post 7/7、整网 20/20，Graph 逐值一致 | 两项随机坐标 post 边界仍失败，专项整体为 false，未改门槛 |
| 二维运动 | 19/19 新捕获、35/35 独立检查 | 32 项浮点/8 位历史比较和 3 项输入逐字节一致；不是游戏运动质量评分 |
| 71 步证据 | 71/71 调度步通过，79/79 比较，无跳过 | 25 批全新抓取、78 个命名缓冲；77 项步内输出/辅助分支及 2 项 repack；post 仅 sRGB8 边界 |
| FP8 特殊值 | 修复 E4M3 NaN 编解码及融合 roundtrip，六种 GPU 配置通过 | CPU 穷举、独立整数 oracle、第三方 Torch CPU 交叉核对通过；默认 FP16 MMA 累加策略未变 |
| 全量构建 | 两轮产物逐字节一致，严格 verifier 通过 | 新目录 15/15 实编译及原 cubin 提取；次轮 15/15 内容命中，无编译工具阶段 |
| 默认产物和安装包 | 已更新并保留旧版 | 52 场景、53 输出文件与旧版逐字节一致；76 个默认工件、15 份缓存、26 个包产物及 15 项安装回归通过 |

采样报告 `research/validation/batch3_sampling.json` 与推广后重跑的 `batch3_sampling_current.json` 均须按 `passed:false` 使用。后者绑定新默认 cubin，前者保留旧来源。失败项为 `post_53x37`（最大差 0.959403/255）和 `post_640x360`（最大差 0.919520/255、MAE 0.000900/255），保留最大差 0.005/255、MAE 0.0005/255 的原门槛。七项完整 post 测试将神经权重置零并指定 blend，用来隔离历史采样，不能替代任意网络状态下的完整 post 验收。

同一批十项整网前后对照的平均 MAE 净降 0.012978/255；静态四帧均改善，正确运动矢量第三帧略增 0.008050/255，零运动矢量对照三帧增 0.090250–0.163859/255。逐例结果保留，不写成“所有运动改善”。新的 motion2d 在相关 ≥0.999、MAE ≤3/255 的既有门槛下通过，MAE 1.3983–1.8859/255；这些是与 d442 sm86 移植版的比较，不是官方原生对照。

新测试程序 `harness/batch3/nr-lab.exe` 增加纵向平移、严格整数解析和分块 tap 地址。CPU 保护包括 26 项地址解析、2 项空参数检查、8 项非法平移拒绝；16 项捕获/比较与 4 项场景切换分析测试通过。二维运动检查的 4 项 CPU 测试和引擎外 finish 的 16 组合也通过。旧套件和前两批程序身份保持独立。

逐步参考 `research/batch3/schedule_capture` 记录了精确序号/参数槽/大小/布局、153 条权重来源、输入与 skip 指针连接、运行 DLL/程序、输出及每块哈希。71 步以各自实际抓取的输入独立执行，包含 pre 双输出、encoder 1h/2h、16 组 split16、8 组 ViT、block39 与 post。中间张量最低相关约 0.99903，最高 NRMSE 约 0.0442；近似门槛仍是相关 ≥0.98、NRMSE ≤0.25。最后 post 用量化 sRGB8 对照，不能把这项解释为内部 float surface 已逐值相等，也不能把 71 个调度步解释为全部 156 次发射的每个内部状态都已验收。

独立审查发现并修复了验证工具的报告覆盖抓取文件、非法阈值残留旧 PASS、执行合同遗漏，以及外部硬链接覆盖证据问题。捕获时源码和工具另存只读身份快照，之后源码或 live port 更新不会抹掉旧参考的来源。报告使用 `research/batch3/check_schedule.json`；CPU 拒绝测试与独立复核详见 `research/validation/batch3_independent_review.json`。

最终逐步 CPU 测试 21/21 通过。旧 `check_blocks` / `check_levels` 也改为显式 strict manifest；直接消费方 `check_numerics` 已补 `--capture`、`--harness` 和 `--levels`，帮助及缺失/无效捕获均在导入模型前处理，GPU 组合扫描未重跑。

新默认 DLL 为 `f454090a97d18e2eb8e6008d42a5fa4e37c8e4bd251a86aedfff8901fef23382`，196,982,784 字节。默认目录中的 PTX、cubin、fatbin 和缓存随 DLL 整体切换，15 份内嵌 fatbin 与严格核验产物一致；旧 79 文件完整保留于 `sm86_port.batch3-backup-0b84169e5487`。推广首次因报告路径键的斜杠形式不一致而在复制前安全中止，失败与修订报告均保留；后验再次核验全部 15 份 fatbin 与 GPU 测试 DLL 身份。

安装包 `plugin/dist/DLSS5-RTX30` 已重建并核验 26 个产物，模型与新默认 DLL 相同；旧包保留于 `DLSS5-RTX30.previous-zhf4pwbv`。15 项安装/卸载保护通过。代理及 NGX 桥的重建结果与此前实际 GPU 测试二进制除 PE 时间戳外相同，本批未把旧 29 项插件 GPU 回归说成重新执行。源码指纹仍为 `1b73c55f1a2c9419c1fb451810d6b41dcf85d51f124afb0964b82a378b810a13`，其范围是包构建输入；新模型另有完整数值构建来源报告。

本批发现的新增未修边界：独立网络在 `hist=None` 且显式提供非零 `mv` 时，pre 仍会重投影当前 Color；原 pre 的无 History 句柄分支直接使用 Color。当前已测默认/运动套件重置帧均不传运动矢量，不能覆盖该组合。后续需统一无历史/无运动的 pre、post、Graph 与导出接口语义，不能宣称这些分支已全部验收。

## 2026-10-23 第二批推进（历史）

第二批继续修改和验证第一批留下的缺口，以下结果取代第一批对应的未完成状态。

| 工作 | 本批状态 | 证据与限制 |
|---|---|---|
| OKLab 两类采样边界误差 | 已修正并扩展回归 | 原 28 项全部保留，扩展后 34/34；17/17 采样探针；最大差 0.003769/255，固定阈值未改 |
| 纹理坐标原因定位 | RTX 3060 测量模型通过 | X/Y 共 6,156 个边界样本，21 位归一化小数截断模型零失配；不是跨 GPU 的 API 保证 |
| 迁移高层启用条件 | 静态调用与四组动态轨迹吻合 | 310.8.0 唯一静态调用点传空；Style1/2、缩放覆盖探针的迁移字段均为零；不是可随意开启的公开功能 |
| 场景切换与中途 Reset | 已实现并验证 | 新 temporal 套件 12/12 捕获、6/6 关系检查；4 项重置结果与对应 fresh 输出逐字节相同 |
| 对照工具失败路径 | 已验证 | 15 项 capture/compare、4 项 temporal 分析 CPU 测试；10 项非法新参数在初始化 GPU 前拒绝 |
| 全量构建与内容缓存 | 已完成并严格验收 | 修订版 15/15 实编译、次轮 15/15 内容缓存命中且无工具阶段；两轮 DLL 与现用及包内模型逐字节相同 |
| 构建路径与报告验收保护 | 已修复并复测 | 18 项 CPU 测试；写报告前拒绝输入/产物/临时路径别名，逐项阶段、缓存计数及当前输入哈希交叉核验 |
| DLL 研究入口 | 已修复并验证 | `python -m tools.dllq` 与 tools 目录下 `python -m dllq` 都可用 |

新增证据：`research/validation/check_transfer_batch2.json`、`batch2_temporal.json`、`batch2_harness_cpu.json`、`transfer_host_static_batch2.json`，`research/transfer_host_runtime_batch2/report.json` 以及 `research/batch2_build_*.json`。第一批迁移失败报告保留，第二批报告通过 SHA 关联旧报告与新旧代码。

本批没有改变默认精度策略，没有把独立迁移 helper 接入默认网络。场景切换是合成输入与本机移植版的回归验证，不替代真实游戏或官方原生对照。

完整构建最初已在新空目录运行；独立审查发现报告路径可覆盖产物、初版 verifier 可接受矛盾报告，均在最终交付前修复。修订后的源码冻结，再进行 15 份全量实编译和一次完整缓存复用，两轮命令总耗时分别为 311.350 秒和 42.551 秒。修订首轮复用已核验的原 sm120 cubin，不宣称逐项重新提取；所有改写/编译/打包均实际执行。最终只以 schema 2 revision 报告验收，旧 schema 1 报告与哈希作为历史保留。数值策略为默认 f16/v1/v1；194,906,112 字节的两份产物、现用 DLL 和包内模型均为 `d442aaf801e5bb5ae1973b494995e9838f0a1dba8039cc79ab91dc9b6ea94ecd`。

使用相同 batch2 harness 的新旧 DLL 各 3/3 smoke 捕获通过，3 项比较 MAE 均为零、逐字节相同。严格 verifier 另核验 15 份原始 fatbin 来源、15 个 sm120 与 15 个 sm86 cubin 的嵌入内容、原权重资源和当前构建输入身份。构建自测期间一次目录隔离疏漏刷新了默认目录的 5 个缓存文件；现用 DLL 与包未改，事后对应 fatbin 与现用 DLL 内嵌字节相同，事件及未保留事前缓存哈希的事实已记录，未隐藏该过程。

## 2026-10-23 第一批收尾（历史）

| 工作 | 状态 | 验收依据 |
|---|---|---|
| 原 DLL / 权重 / 153 条映射 / 71 步顺序身份核验 | 已完成 | `python -m tools.verify_assets`：资源字节相同、记录无越界/重叠、每条各引用一次 |
| 遮罩 + Style 1/2、UICorrection / UIAlpha / UI.a / Backbuffer 后处理 | 已实现并专项验证 | 8 个整网组合及 8 个隔离原 kernel 对照；最终结果见下方本轮记录 |
| 数值回归阈值、JSON 报告与非零失败 | 已完成并复测 | 单帧、时序、运动、分辨率、控制参数；缺资源不得自动通过 |
| 显存抓取缺项与逐块覆盖报告 | 已完成并复测 | 完整模式遇缺项须失败；允许部分研究必须显式选择 |
| 指令仿真失败退出及构建缓存内容指纹 | 已完成并验证 | 编译参数、工具、源 PTX、rewrite/fuse/flow 等参与缓存判定 |
| 交换链共享 NGX 参数的并发隔离 | 已完成并复测 | 全局参数事务同步与并行 Present 回归 |
| HDR 自动数值验收 | 已完成并复测 | 有限性、亮部、alpha、画面实际改动，不仅检查日志 |
| 安全打包与交付指纹、同步最新产物 | 已完成并验包 | 先构建/验证、暂存后替换、文件 SHA-256、源码指纹 |
| RTX 50 原生参考捕获 / 跨机比较工具 | 已实现，本机 21 组捕获通过 | `harness/reference.py`；12 项证据完整性与失败路径测试；3060 原生标签拒绝已验证 |
| OKLab 迁移分支 | 独立实验实现，专项未通过 | 28 项迁移用例通过 26 项、17 项采样探针通过 15 项；保留失败，未接入默认网络 |
| 项目文档统一与历史错误降级 | 已完成 | 当前状态以本文件和当前实现为准，历史记录保留更正关系 |

## 第一批复测与证据

- 已重新核验原 DLL 与权重 SHA-256。权重 147,695,410 字节，载荷 147,683,778 字节，153 条记录、71 步、156 次记录内核发射。
- 修改前基线复测：640×360 单帧相关 0.99931、MAE 1.92/255、8 位通道逐值一致率 17.17%；1080p 单帧相关 0.99934、MAE 1.84/255。
- 修改前四帧递推：浮点历史 MAE 1.72–1.83/255，8 位历史 MAE 1.64–1.65/255（后续三帧）。这些比较对象是 3060 移植版的 8 位输出。
- 修改前最新 harness + 移植 DLL 连续四帧成功，校验和 `5B3A70119F21C9EE`，新结果与 `research/out_f3.ppm` 像素完全相同。
- 新后处理专项：整网组合 MAE 0.280–1.915/255、相关 0.999311–0.999985；隔离后处理最大差 0.000289/255。仅后处理误差不能替代整网或 RTX 50 验收。
- 修改后数值验收：单帧 1、多帧 8、运动 6、分辨率 3、逐级 9、指定逐块 38，共 65 项通过，0 跳过。逐块覆盖指定编码 4h/8h 和解码 Swin，不是所有 71 步完整逐块覆盖。
- 控制参数与 Style/Mask 组合 18 项通过，MAE 0.388–2.208/255，最低相关 0.999075。
- 指令局部仿真 5 项通过；缓存、非法数值门槛、缺参考和报告失败路径 CPU 验证通过。另用真实本地工具链在独占临时目录编译 fatbin_08，确认无变化时零工具调用、损坏 cubin 后自动重建且结果字节相同。此次未进行完整 DLL 重建；全量首次重建仍需在后续构建时执行。
- 固定 bench 的正常四帧 `5B3A70119F21C9EE` 和 Intensity=0 `D61DE47C7AECF78F` 均匹配。720p/1080p 各六帧仅用于短回归，不作为稳定性能测量。
- 最新插件 29/29 项、每项 300 帧通过，包含有起跑屏障的双线程 Present 和四项 HDR 数值检查；可选独占全屏未运行。
- HDR 亮部平均比为 f16 0.887019、scRGB×4 0.922648、PQ 0.876059、PQ×4 0.974358，满足自定义 0.75–1.25 包络。这是高光保留回归，不是官方色彩数值等价。
- 发布安全 CPU 10/10、最终包安装安全 15/15。26 个产物逐项校验，包与当前构建同步；格式整理重建的代理除 PE 时间戳外与实际 GPU 测试版本相同，证据保留在 integration 报告中。
- 原版对照工具 CPU 12 项证据完整性测试通过；sm86 完整 21 组实际捕获通过，包括奇数尺寸和原始 HDR 浮点输出。此报告的 reference_kind 为 port-sm86，不是 official-sm120。
- 两次独立 smoke 捕获的重置、第四帧历史、Intensity=0 三项输出逐字节相同；报告明确标为 official_reference=false。
- 后处理报告异常已修复为覆盖旧成功状态并非零退出；控制参数报告逐项保存参考 SHA，避免共享路径覆写丢失历史。
- OKLab 迁移独立实验：固定最大差 0.005/255、MAE 0.0005/255 的门槛下，28 项通过 26 项。`point_outside_rect` 最大差 255/255、MAE 1.657566/255；`linear_large_grid_fallback` 最大差 0.353138/255、MAE 0.001853/255。17 项采样探针另有 2 项失败，报告整体为 `passed:false`。不计入上述已通过的默认路径回归。

机器可读报告保存在本地 `research/`、`harness/captures/` 或插件测试目录，不包含进 Git 的私有输入和二进制。基准来源必须标注为 `port-sm86` 或 `official-sm120`。

具体证据入口：

- `research/validation/check*.json`：模型回归，默认近似门槛。
- `research/validation/bench.json`：固定校验和及短运行。
- `research/validation/cache_real.json`：真实编译工具的缓存复用与损坏恢复。
- `research/validation/port_repeatability.json`：两次独立捕获的三项逐字节一致。
- `research/model_postprocess_20261023.json`：独立后处理对照。
- `research/validation/check_transfer.json`：实验迁移与采样探针，保留全部失败、输入/输出及代码哈希。
- `research/integration_validation_20261023/cpu_summary.json` / `gpu_summary.json`：发布、安装、29 项插件回归。
- `research/integration_validation_20261023/lf_rebuild_equivalence.json`：重建前后代理等价检查。
- `harness/captures/port-3108-20261023-a/manifest.json`：21 组本机捕获（输出和日志留本地）。

## 研究结论与不确定性

1. **数值精度。** 默认快速移植把 FP8 MMA 拆为两段 FP16 累加；第三批修复 E4M3 转换与融合的 NaN 类别，不等于修复 MMA 累加精度。两种现存累加方案的同帧输出 MAE 1.62561/255、最大 14/255。这证明策略会影响输出，不证明哪种与原生一致。
2. **官方指令依据。** [NVIDIA PTX ISA 的 mma 说明](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma)规定 FP8 运算的中间累加至少单精度，但没有指定累加顺序、舍入和次正规数处理。因此 FP32 变体有更强规范依据，仍不自动保证 Blackwell 逐位相同。本轮保留现有快速默认值，避免未经评估的性能回归。
3. **误差归因。** FP8 舍入、rcp/rsqrt 近似、矩阵累加次序是有证据支持的来源，但尚未严格排除所有结构或边界差异。旧 README 中“不是结构差异”不再作为定论。
4. **兜底接入。** 后缓冲 + 图像光流 + 降分辨率改动回加 + 自制 HUD/HDR 处理是兼容路径。缺少深度本身不是否定官方等价的充分理由；真正需要对照的是实际输入、运动、历史、遮罩及最终合成。
5. **新版范围。** [NVIDIA 2026-10-13 更新公告](https://www.nvidia.com/en-us/geforce/news/dlss-5-faster-ai-model-hogwarts-legacy-rtx-spark-laptops/)说明 10 月 16 日引入约 1.18 亿参数的新模型。已取得 RTX Remix 中的 310.9.16 原件并完成[独立静态调查](research/october_nr_remix.md)；旧版成果不代表新版已能运行或数值等价。

## OKLab 迁移分支的研究结果

原 `fatbins/fatbin_14.1.sm_120.ptx` 第 43–60 行确认：仅当参数 +300 非零、+224/+256 两个纹理句柄非零、+304/+308 两个有符号网格尺寸均为正时，进入迁移。Style 不参与这个分支条件，低分辨率本身不是充分条件。

数据流确认 A=+224 是保留高频内容的基图，B=+256 提供迁移目标；普通 NR 纹理 +32 仅在分支禁用时读取。这是由实际读取和计算关系确定的角色，不是已知公开 API 参数名。

网格中心采样后，RGB 先饱和，经过 PTX 的 LMS/立方根矩阵变换得到 Lab（LMS 下限 1e-4，没有 sRGB 解码）。低网格保存亮度的受限对数比及色度差，再按输出像素的网格坐标双线性插值，回加到输出位置直接采样 A 的 Lab，逆变换后进入已有 grade/UI/mask 合成。

`torch/dlss5/style.py` 的独立 `transfer()` 没有接入 `DLSS5` 默认执行路径。`torch/check_transfer.py` 强制注入原 kernel 参数，覆盖开关、缺纹理、非正网格、奇数尺寸、暗部、子区域、点/线性采样及调色/UI/遮罩组合；参考来自原 kernel，不由 helper 生成。第一批保留的两类采样失败已在第二批修正：归一化坐标先按 RTX 3060 实测模型截断为 21 位小数，再转资源像素坐标；保留线性轴/交叉权重量化。扩展后 34/34 迁移及 17/17 采样探针通过，最大差 0.003769/255，门槛仍为最大 0.005/255、MAE 0.0005/255。6,156 个 X/Y 独立 TEX 样本支持该模型；20 位模型错 344 个、22 位错 256 个，21 位零失配。

[CUDA 文档的线性过滤说明](https://docs.nvidia.com/cuda/archive/12.6.2/cuda-c-programming-guide/index.html#linear-filtering)支持 8 位小数插值权重；21 位坐标截断、交叉权重和 tie 行为来自本机测量，不是文档保证，也不应外推所有 GPU。旧失败报告与新报告、代码/输入/输出身份均保留。

第二批已追踪 DLL 高层参数组装：`0x18001C920` 可构造 +224/+256 descriptor 并设置开关，但唯一静态直接调用点 `0x180019884` 向 arg8/arg9 传入零。清零点 `0x1800188D9` 支配该调用，路径之间未发现 RSI 写入，按 Windows x64 非易失寄存器规则推导两个 descriptor 为空。原始 rel32 扫描也只找到该调用点，未找到绝对函数指针引用；不能排除运行时计算的间接调用、其他版本或外部内部函数调用。静态证据支持此版本常规 Evaluate 路径未连接迁移。

四组正常 API 调用的动态轨迹与此相符：Style1、Style2、runtime-scale=0.5，以及输入 640×360 / 输出 960×540 的 Style1，捕获到的 376 字节后处理块在 +224/+256/+300/+304/+308 均为零。前三组执行通过，异尺寸组返回 5、完整输出覆盖失败，报告明确区分“观察到轨迹”和“执行通过”；后者不能列入可用缩放结果。模式参数只决定是否调色，不足以开启迁移。不能把独立强制注参测试冒充正式 NGX 端到端集成。

D3D12 后端的 sampler 创建和 descriptor heap 索引已进一步确定：+224 的 selector 0 为 LINEAR+CLAMP，+256 的 selector 2 为 POINT+CLAMP，独立 helper 对应 `base_filter="linear", target_filter="point"`。现有混合过滤用例覆盖该组合。原 DLL 与当前 port 的 Evaluate、参数组装、descriptor 构造及 sampler 初始化/绑定五段相关 host 代码逐字节相同；仍不证明其他版本或 Vulkan 后端合同相同。

## 尚未验收完成的工作

- [ ] 在 RTX 50 上，用已知原始 310.8.0 DLL 和相同测试程序捕获 `official-sm120` 完整参考。
- [ ] 对 sm86 FP16 / FP32 两种策略作逐用例、逐值及内部层对照，依据原生结果决定精度策略；不只比较整图相关。
- [x] 第四批在固定采样探针中关闭两项 Catmull post 失败，并统一 Reset 与显式非零 MV 的独立接口语义；范围限于报告记录的本机输入，不能外推任意画面或原生等价。
- [ ] OKLab 迁移分支：在 RTX 50 验证独立采样模型，并完成高层调用/采样器来源的范围核对。当前 310.8.0 常规 Evaluate 的静态路径传空 descriptor，不自行假设一个公开启用键或接入默认网络。
- [ ] 新构建的商业游戏矩阵：引擎运动矢量、遮挡/显露、透明物体、粒子、HUD、场景切换、HDR、长时序和不同驱动/GPU；未获授权或未准备的游戏不启动。
- [ ] Vulkan 游戏注入层；目前仅独立 Vulkan harness 可运行。
- [ ] 新版模型单独立项验收，不能复用 310.8.0 的字节偏移、权重布局和精度结论。
- [ ] 训练数据、损失与完整训练流程没有恢复；不宣称训练过程一比一。

这些是未完成条件，不会因本轮工程收尾而自动勾选。

## 历史阶段摘要（2026-10-09 至 10-12）

| 阶段 | 已有成果 | 证据边界 |
|---|---|---|
| A 交换链加固 | 格式、改大小、释放重建、Present1、双链、等待对象 | 旧版测试报告；不能覆盖本轮新增并发路径 |
| B Godot DX12 | Sponza、改窗口、热更新、运行 140 秒 | 本地引擎样例，不是官方 NR 游戏对照 |
| C 光流与历史 | 平移闪烁 3.30→0.74/255，正确矢量约 0.58 | 特定测试图；光流质量档开销略超原 3 ms 目标 |
| D 可选输入 | ControlMask、UIAlpha、UI、Backbuffer、UICorrection 语义实验 | 效果和输入含义限于该 DLL；独立接口由本轮继续补齐 |
| E HDR 兼容 | 模型域压缩、HDR delta 回加、高光保留 | 本项目自定义兼容路径，不是已证明的官方 HDR 等价 |
| DX11 / OptiScaler | 共享纹理桥接、FFX 示例接入 | 官方商业游戏矩阵未完成 |
| Superposition | 720p/0.35 档有效十分钟约 33,490 Present | 10 月 10 日构建；更早失活暂停测试已作废，不能混用 |
| EdgeUpsample | 沿图像边缘放大低分辨率改动 | 对当前三帧场景有局部改善，不等于原生分辨率输出 |
| Vulkan harness | Vulkan 与 D3D12 的移植输出相同 | 不是 Vulkan 游戏注入，也不是 RTX 50 原生对照 |
| F 提速研究 | 融合编码、不同寄存器/精度方案、TensorRT 比较 | 未完成后续优化周期；不把“只能蒸馏”当严格不可能性结论 |

详细原始研究仍在 `tools/notes.jsonl`。该文件按时间保留历史，较早关于 FP16 权重、softmax 或拓扑的推断已经被后续实验更正。
