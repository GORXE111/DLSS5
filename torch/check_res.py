"""多分辨率验收: nr-lab 各分辨率第 0 帧 (重置帧) 输出 vs torch。需要 research/out_f0_<W>x<H>.ppm"""
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from check import color_pattern  # noqa: E402
from dlss5 import DLSS5  # noqa: E402
from dlss5.net import dims  # noqa: E402

R = os.path.join(HERE, "..", "research")


def ppm(path, W, H):
    raw = open(path, "rb").read()
    return np.frombuffer(raw[-W * H * 3:], np.uint8).reshape(H, W, 3).astype(np.float32) / 255


if __name__ == "__main__":
    net = DLSS5()
    for res in sys.argv[1:] or ["960x540", "1280x720", "1920x1080"]:
        W, H = map(int, res.split("x"))
        p = os.path.join(R, f"out_f0_{res}.ppm")
        if not os.path.exists(p):
            continue
        color = color_pattern(W, H)
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = net(color).cpu().numpy()
        dt = time.perf_counter() - t
        o8 = np.floor(out * 255 + 0.5) / 255
        ref = ppm(p, W, H)
        d, c = ref - color, o8 - color
        print(f"{res} (网格 {dims(H, W)}): 相关 {np.corrcoef(o8.ravel(), ref.ravel())[0, 1]:.5f}  平均差 {np.abs(o8 - ref).mean() * 255:.2f}/255"
              f"  改动量相关 {np.corrcoef(d.ravel(), c.ravel())[0, 1]:.4f}  改动幅度 kernel {np.abs(d).mean():.4f} torch {np.abs(c).mean():.4f}  {dt:.1f}s")
