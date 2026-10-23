# DLSS-NR 独立调用与原生对照

更新：2026-10-23。测试目标为 `nvngx_dlssnr.dll` 310.8.0 / feature 18。

## 程序范围

- `nr-lab.cpp`：D3D12 的确定性合成输入、运动/控制/可选资源探针及输出读取。
- `nrvk.cpp`：独立 Vulkan 调用，尚不是 Vulkan 游戏注入。
- `nr11.cpp`：确认该 DLL 的原生 D3D11 接口不支持；插件使用自己的 DX11→DX12 共享纹理桥。
- `nvprof.h`：本地轨迹、计时与显存研究钩子，依赖指定 DLL 布局。

`nr-lab-result.json` 的 `outputCoverageValidated` 只说明 feature 创建、指定帧求值和目标区域写入通过。
旧字段 `upscalingValidated` 保留相同含义以兼容旧脚本。`fidelityValidated=false` 明确表示这个程序本身没有对照官方图像。

## 固定回归

`bench.ps1` 检查已知默认 sm86 快速变体的 360p 四帧校验和 `5B3A70119F21C9EE` 和 Intensity=0 基准。
它证明数值没有偏离本项目基线，不证明与 RTX 50 原生相同。其他精度变体不能套用快速变体的默认期望值。

## 有证据的跨硬件捕获

```powershell
# 依赖本地已编译的 nr-lab.exe / nvngx.dll，以及自备的辅助 DLL
python harness/reference.py capture --dll sm86_port/nvngx_dlssnr.dll --kind port-sm86 --suite full --out harness/captures/port-3108

# 在 RTX 50 上执行；必须使用未修改的已知 310.8.0 NR DLL
python harness/reference.py capture --dll dlss5/nvngx_dlssnr.dll --kind official-sm120 --suite full --out harness/captures/native-3108

python harness/reference.py compare --reference harness/captures/native-3108 --candidate harness/captures/port-3108 --criterion bit-exact --report research/native_comparison.json
```

两台机器要使用**相同哈希**的 `nr-lab.exe`、`nvngx.dll` 和辅助 `nvngx_dlss.dll` / `nvngx_dlssg.dll`；单独重新编译的时间戳可能改变程序哈希。
先复制同一份测试程序再测试。自备 NVIDIA 资源，不把 DLL 上传进源码仓库。`--harness`、`--bridge`、`--support` 可指定这些本地文件。

输出目录必须不存在。工具在专属子目录运行，保存每组参数、日志、结果和输出哈希，并删除它自己复制的可执行文件/DLL。
原生标签检查原始 DLL 哈希、Blackwell GPU 清单及实际选卡日志；RTX 3060 会在运行前拒绝该标签。

| 套件 | 用例 | 范围 |
|---|---:|---|
| smoke | 3 | 重置帧、第四帧历史、Intensity=0 |
| core | 13 | 增加中间历史帧、正确/零运动矢量、调色、结构/色调、遮罩 |
| full | 21 | 再增加 540p/720p/1080p/奇数尺寸、Style+Mask、UI 合成、scRGB 原始浮点输出 |
| temporal | 12 | 两种场景、16 帧历史、中途场景切换，以及带/不带 Reset 的对应前缀 |
| motion2d | 19 | 正负对角、垂直、分数/越界矢量、奇数尺寸、零矢量对照及三项 Intensity=0 输入证明 |

捕获只保留每个用例的最后一帧；历史和运动用例分别运行各个前缀，获得多个时刻的结果。
当前输入是确定性测试图，没有商业游戏画面，也没有遮挡/粒子/透明材质等完整内容矩阵。

## 场景切换与历史重置

编译当前 `nr-lab.cpp` 后，`--scene-variant 0|1` 选择两种通道置换场景，`--scene-cut-frame N` 在零起始帧 N 切换，`--reset-frame N` 在同一帧求值前设置 Reset / DLSSNR.Reset。-1 为禁用；帧号超出本次运行或非法参数在初始化 GPU 前拒绝。

```powershell
python harness/reference.py capture --dll sm86_port/nvngx_dlssnr.dll --kind port-sm86 --suite temporal --out harness/captures/temporal
python harness/check_temporal.py --capture harness/captures/temporal --report research/temporal.json
```

第二批使用专属 `harness/batch2/nr-lab.exe` 编译产物，12/12 捕获通过；4 项中途重置结果与对应 fresh 场景的 RGB8 输出逐字节相同。另要求两种场景确实不同、Reset 确实改变切换后的输出，避免相同画面带来的空通过。本机切换首帧不重置与重置的输出 MAE 为 45.421/255，下一帧约 7.089/255；这些是合成场景的观察，不是质量评分或通用阈值。16 帧历史另作观测，未将内部浮点状态或真实游戏切换质量宣称为已验证。

捕获同时核验每帧场景和 Reset 事件日志，拒绝遗漏事件或错误场景。`check_temporal.py` 检查上述 6 个关系，缺捕获或关系不符均覆盖旧成功报告并返回非零。旧 smoke/core/full 套件定义保持不变。

`bit-exact` 要求所有捕获输出逐字节一致；原始浮点数据连有符号零也参与比较。`approximate` 必须由调用者显式给出 `--max-mae` 和 `--min-psnr`，报告会保留该判据。
两种模式都输出 MAE、最大差、p99、逐值一致率、PSNR 和相关。相关不是还原百分比；原始 HDR 的误差按浮点值×255 报告，PSNR 以 1.0 为峰值参照。

比较会拒绝缺用例、不完整捕获、文件哈希变化或不同辅助程序。`official_reference` 明确区分原版和移植基准。
即使所有有限用例通过，也只对这些输入和版本成立，不能宣布所有画面、所有游戏或最新版 DLSS 5 等价。

## 工具自测

第三批 `--temporal-shift-y` 增加输入图案的纵向逐帧平移，与横向参数一起严格检查有符号整数范围。负平移按图像尺寸环绕；大帧号先用 64 位乘积取模，避免整数溢出。结果 JSON 新增 `temporalShiftYPixels` 和 `intensity`。旧套件定义保持原样，新套件要求这些字段和明确参数相符。

```powershell
python harness/reference.py capture --dll sm86_port/nvngx_dlssnr.dll --kind port-sm86 --suite motion2d --harness harness/batch3/nr-lab.exe --out harness/captures/motion2d
python torch/check_motion2d.py --capture harness/captures/motion2d --report research/motion2d.json
```

分数矢量是数值边界探针，不能解释为已知正确的场景运动。该检查分别递推浮点与 8 位历史，并要求三项强度为零的输出与 Python 输入图案逐字节相同。其余用例使用既有移植版近似门槛，不能替代真实游戏的遮挡与运动质量评估。

`nvprof.h` 的 tap 地址支持 `pN+byteOffset`，用于把大缓冲拆成受控小块；严格拒绝非法索引、空指针、溢出和尾随字符。26 项地址解析和 2 项空参数保护已通过，8 项非法横/纵平移参数在初始化 GPU 前返回 64。

```powershell
python harness/test_reference.py
python harness/test_temporal.py
```

CPU 测试覆盖像素首字节为空白、缺用例、输出被改动、原生标签误用、非有限浮点、有符号零及真实像素回归的失败路径。
当前为 16 项捕获/比较测试与 4 项 temporal 分析测试；新增场景/Reset 参数另有 10 项实际程序拒绝验证。`torch/test_motion2d.py` 的 4 项 CPU 测试覆盖输入环绕、报告路径保护、缺捕获和非法阈值时旧成功失效；引擎外 finish 另有 16 个组合与正常 PostBlock 逐值一致。
