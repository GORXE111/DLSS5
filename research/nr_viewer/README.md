# NR 离线图像对照查看器

用滑块看原图与处理结果，切换差异热图，或以原尺寸 / 200% 查看局部。页面不联网，也不重新执行推理。此目录只提交查看器源码；截图、模型、捕获日志和生成后的网页素材留在本机。

需要 Python 3.11+、NumPy 和 Pillow。先用不依赖游戏或模型的合成示例验证环境，在仓库根目录运行：

```powershell
python -m pip install numpy Pillow
python research/nr_viewer/generate_assets.py --demo --out research/nr_viewer/local-output
python -B research/nr_viewer/test_generate_assets.py
```

直接打开生成目录中的 `index.html`。移动生成物时保持 `index.html`、`datasets.js` 与 `assets/` 相对位置；无需 HTTP 服务。输出目录必须不存在，父目录须已存在。生成器不会改输入或替换已有文件夹。

## 使用自己的捕获

在本地捕获目录旁创建一个 JSON 配置（不需要提交）。输入路径相对于配置文件，也可指向本地绝对路径；生成的证据不记录输入路径。以下示例使用代理的第一个带数字后缀的转储：

```json
{
  "title": "我的离线 NR 对照",
  "meta": "本地 DXGI 代理样例",
  "default_dataset": "scene-present",
  "datasets": [
    {
      "id": "scene-present",
      "label": "全链",
      "stage": "present",
      "capture": "capture-folder",
      "slot": 1,
      "description": "同次 Present 的原后缓冲与最终合成。",
      "regions": [{"label": "中央细节", "x": 0.5, "y": 0.6}]
    },
    {
      "id": "scene-model",
      "label": "模型纹理",
      "stage": "model",
      "capture": "capture-folder",
      "slot": 1,
      "description": "送入模型的 Color 与模型输出，未经过放大回写。"
    }
  ]
}
```

```powershell
python research/nr_viewer/generate_assets.py --spec path/to/local-spec.json --out research/nr_viewer/local-output
```

`capture-folder` 应是一次完整的单交换链 [fallback](../../plugin/fallback/README.md) 运行目录，保留对应的 `dlss5fb.ini`、`dlss5fb.log`、运行 DLL 与原始 PPM。可用 `DumpFrame` 指定起始 Present，`DumpCount` 设为至少 2。未带后缀的是 `slot: 0`，`_1` 是 `DumpFrame + 1`。`present` 读取 `in/out`；`model` 读取 `lo/nr`，需要 `slot >= 1`。生成器检查转储顺序、尺寸、成功创建 feature 18 和交换链释放记录。可选 `expected_model_sha256` 进一步锁定本地模型身份；如果有 `process.json`，还检查退出码。这里验证本地捕获合同，并不独立证明日志或输入文件未被人工改写。

也支持自备同尺寸配对：设置 `stage: "pair"`，以 `before`、`after` 指向两份 PPM。支持代理输出形式的 8 位 RGB P6，三行头、无注释、完整像素。自备图不会标为同帧或模型捕获；合成示例也不会标为模型结果。局部 `x/y` 是从左上角起算的 0–1 相对位置。

原图 PNG 保留输入尺寸和像素。幅度热图是每像素三个 RGB 通道的平均绝对差；方向热图是 Rec.709 编码亮度 Y′ 的输出减输入，蓝暗橙亮。图例截到 40/255，统计保留完整差值。`assets/evidence.json` 记录像素尺寸、来源类别、哈希和数值范围。这些图显示改了什么，不证明细节恢复、HDR 等价或官方原生效果。
