# 十月 Small FP8 的 4K 接入与注意力复核

2026-11-06。RTX 3060 上的隔离实验版已在两条 4K 输出路径验证。独立 NR 的输出边界、1080p NR 串接 SR、以及真实 4K 后缓冲合成分别记录。

后续的同权重运行时对照、K16 分组修正和自动逐层采集见[版本因子实测](october_nr_version_factors.md)。本页保留上一批算术基线。

## 1080p 到 4K 的写入范围

| 实验 | NR 输入 / 输出 | 最终结果 |
| --- | --- | --- |
| 独立 NR，分配 4K 目标 | 有效输入 1920×1080，应用请求 3840×2160 输出矩形 | 两个末尾 copy 都以 1920×1080 为写入范围，原点是 `(0,0)`。结果只覆盖 25%，左上之外保留哨兵 |
| 紧凑 NR 串接 DLSS SR | NR 读写 1920×1080，SR 直接读取同一张 NR 输出纹理 | SR 写入独立的 3840×2160 目标。NR 强度 0 和 1 两组均覆盖四个象限 |
| DXGI 代理合成 | 从原 3840×2160 后缓冲取得 1920×1080 工作图 | 将放大的 NR 改动量回加到原 4K 画面，保留同一次 Present 的前后捕获 |

独立 NR 的 copy 参数 `+64/+68` 给出实际写入宽高，`+40/+44` 给出输出原点。实测左上区域与相同 NR DLL 的 1080p 全帧最大只差 1/255，两次使用了不同 lab 构建。原版比例回调的静态分支只接受质量值 2、4、5，分别返回 1.0、0.85、1.0。此处的覆盖结论来自真实 CUDA 提交与像素哨兵，适用于已测的 D3D12 参数合同。[4K 写入证据](validation/october_nr_4k_contract.json)

复现 NR 串 SR 时使用已核对哈希的隔离 DLL 和相同辅助 DLL。下面的命令在独立 lab 目录运行，给两个强度使用不同输出目录以保存两份结果。

```powershell
.\nr-lab.exe --compact-nr --input 1920x1080 --output 3840x2160 --model 2 --frames 1 --scene-variant 0 --intensity 1
```

## 真实 Sponza 画面

DXGI 代理新增 `RenderPreset`。在隔离 Sponza 环境用 `RenderPreset=2` 选择十月小档，`WorkingScale=0.5` 使模型在 1920×1080 工作。Present 56 同时留存 4K 原图与最终合成，以及 1080p 模型输入和输出。4K 合成前后 8 位 RGB 平均绝对差为 **7.423/255**。

另一组用 `fbtest` 回放同一张 Sponza 画面。310.8.0 预设 1 与 310.9.16.0 预设 2 的 4K 原始后缓冲和 1080p 模型输入都逐字节一致。两版关闭时序后，4K 合成结果平均绝对差为 **5.624/255**，1080p 模型结果为 **5.744/255**。比较涵盖两套运行时和模型的合并差异。五组对照的本地查看器已验证滑块、热图、原尺寸和手机触控。[场景、对照与查看器证据](validation/october_nr_4k_integration.json)

## 注意力算术与下一次原版对照

从 640×360、预设 2 的真实首帧抓到第一个和最后一个 ViT attention 的同次 Q、K、V 与输出。两次均为 96 个 token、32 个注意力头、每头 32 维。独立 CPU 计算按照实际缓冲布局、分组半精度规约和 E4M3 舍入预测输出。

| 注意力调用 | 数值完全一致 | 平均绝对误差（E4M3 解码值单位） | 误差在 1 个 E4M3 步长内 |
| --- | ---: | ---: | ---: |
| 首个 ViT，提交 23 | 99.2574% | 0.002090 | 100% |
| 最后一个 ViT，提交 58 | 98.1405% | 0.001407 | 99.9949% |

这两处还未达到逐位一致。CPU 模型对 MMA 内部舍入和近似倒数使用近似。另一次运行取样了首帧的 **73 个受支持边界，共 41,189,376 B**。它复现了早先 16 个选定张量的哈希，包括从 pre 到 post 内容一致的 skip 缓冲。取样后的最终 PPM 和 RGBA16F 与普通 3060 运行逐字节一致。[注意力算术结果](validation/october_nr_attention_cpu.json) · [边界取样记录](validation/october_nr_4k_integration.json)

逐层对照工具按各自的轨迹推导角色、实际地址和尺寸，校验 tap 日志并记录独立的捕获哈希。以下工具在仓库根目录运行。先用原链轨迹规划取样，再按输出的 `tap_request` 记录所选角色。两边使用相同 lab、桥接库、辅助库和用例。分别编译 Ampere 的 `tap.cubin`（`-arch=sm_86`）和 Blackwell 的 `tap.cubin`（`-arch=sm_120`）。

```powershell
python -B -m tools.compare_october_taps plan TRACE/nr-trace.tsv --frame 0
python -B -m tools.compare_october_taps seal CAPTURE
python -B -m tools.compare_october_taps compare NATIVE PORT --native-seal $nativeHash --port-seal $portHash --role block11.attention.in0 --role block11.attention.in1 --role block11.attention.in2 --role block11.attention.out
```

两次取样各保留一份 `run.json`，记录 `arguments`、`exit_code`、`gpus`、`tap_spec`、`tap_cubin_sha256`、`trace_sha256`，以及按文件名索引的 `runtime_sha256`、`evidence_sha256`、`outputs` 和 `tap_sha256`。保留原始文件供工具核验，其中日志包括 `nr-lab.log`、`stdout.log`、`stderr.log`，运行证据还包括 `nr-lab-result.json`、`nr-trace.tsv` 和 `tap_index.tsv`。在取样完成时分别记录两端的 `capture_seal_sha256`。

对照工具默认要求逐位相等，会单列输入角色及其差异，用于区分误差已在上游出现还是发生在本层。当前 RTX 50 端已完成取样内核编译，待在原版硬件上捕获。
