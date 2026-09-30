"""降低内部分辨率的画质代价 (torch 参考实现，GPU)。

对每张测试画面 (缩放到 1920x1080 作为"原生"帧):
  基准  full = NR@1080p(原生帧)                                 (连跑 N 帧静止画面，取稳态)
  低分  low  = 放大到 1080p( NR@r(缩小到 r 的帧) )                r = 1600x900 / 1280x720 / 960x540
  对照  up   = 放大到 1080p( 缩小到 r 的帧 )                       (不跑 NR，只看分辨率本身的损失)
指标:
  PSNR(low, full)        低分辨率 NR 离全分辨率 NR 多远
  PSNR(up, 原生)          仅缩放的损失 (参照)
  NR 效果保留  corr(low - up, full - 原生)，幅度比 |low - up| / |full - 原生|
测试画面只在本地读取，不入库。
"""
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "torch"))
from dlss5 import DLSS5  # noqa: E402

FRAMES = 4
SIZES = [(900, 1600), (720, 1280), (540, 960)]


def load(path, H=1080, W=1920):
    """读图 -> 居中裁成 16:9 -> 缩放到 W x H (面积平均) -> (H, W, 3) [0,1]"""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    tw, th = (w, round(w * 9 / 16)) if w * 9 / 16 <= h else (round(h * 16 / 9), h)
    im = im.crop(((w - tw) // 2, (h - th) // 2, (w - tw) // 2 + tw, (h - th) // 2 + th)).resize((W, H), Image.BOX)
    return torch.tensor(np.asarray(im, np.float32) / 255, device="cuda")


def resize(img, H, W, mode):
    x = img.permute(2, 0, 1)[None]
    if mode == "area":
        y = F.interpolate(x, size=(H, W), mode="area")
    else:
        y = F.interpolate(x, size=(H, W), mode=mode, align_corners=False, antialias=False)
    return y[0].permute(1, 2, 0).clamp(0, 1)


def run(net, img, frames=FRAMES):
    out = None
    for f in range(frames):
        out = net(img, hist=out, frame=f)
    return out


def psnr(a, b):
    return float(10 * torch.log10(1 / ((a - b) ** 2).mean()))


def main(paths, out_json):
    net = DLSS5()
    rows = []
    for p in paths:
        src = load(p)
        t = time.time()
        full = run(net, src)
        eff_full = full - src
        row = {"image": os.path.basename(p), "nr_effect_full": float(eff_full.abs().mean())}
        for (h, w) in SIZES:
            small = resize(src, h, w, "area")
            low_small = run(net, small)
            for up_mode in ("bicubic", "bilinear"):
                low = resize(low_small, 1080, 1920, up_mode)
                up = resize(small, 1080, 1920, up_mode)
                eff_low = low - up
                key = f"{w}x{h}/{up_mode}"
                row[key] = {
                    "psnr_low_vs_full": psnr(low, full),
                    "psnr_up_vs_native": psnr(up, src),
                    "effect_corr": float(torch.corrcoef(torch.stack([eff_low.flatten(), eff_full.flatten()]))[0, 1]),
                    "effect_ratio": float(eff_low.abs().mean() / eff_full.abs().mean()),
                }
            np.save(os.path.join(HERE, "res_study_out", f"{os.path.splitext(row['image'])[0]}_{w}x{h}.npy"),
                    (low_small.cpu().numpy() * 255).astype(np.uint8))
        np.save(os.path.join(HERE, "res_study_out", f"{os.path.splitext(row['image'])[0]}_full.npy"), (full.cpu().numpy() * 255).astype(np.uint8))
        np.save(os.path.join(HERE, "res_study_out", f"{os.path.splitext(row['image'])[0]}_src.npy"), (src.cpu().numpy() * 255).astype(np.uint8))
        rows.append(row)
        print(f"{row['image']}: NR 改动量 {row['nr_effect_full']:.4f}  ({time.time() - t:.0f}s)")
        for k, v in row.items():
            if isinstance(v, dict) and k.endswith("bicubic"):
                print(f"   {k:18s} PSNR(低分NR,全分NR) {v['psnr_low_vs_full']:.2f} dB | PSNR(仅缩放,原生) {v['psnr_up_vs_native']:.2f} dB"
                      f" | NR 效果相关 {v['effect_corr']:.3f} 幅度比 {v['effect_ratio']:.3f}")
    json.dump(rows, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "res_study_out"), exist_ok=True)
    main(sys.argv[1:], os.path.join(HERE, "res_study_out", "results.json"))
