"""多帧 (时域累积) 验收: nr-lab 连跑 N 帧 (合成输入不动、运动矢量 0)，最后一帧输出与 torch 逐帧递推比较。
需要 research/out_f{0..3}.ppm (nr-lab --frames 1..4 的输出)。"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from check import color_pattern  # noqa: E402
from dlss5 import DLSS5  # noqa: E402

R = os.path.join(HERE, "..", "research")


def ppm(i):
    raw = open(os.path.join(R, f"out_f{i}.ppm"), "rb").read()
    return np.frombuffer(raw[-640 * 360 * 3:], np.uint8).reshape(360, 640, 3).astype(np.float32) / 255


def q255(x):
    return torch.floor(x * 255 + 0.5) / 255


if __name__ == "__main__":
    net = DLSS5()
    color = torch.tensor(color_pattern(), device="cuda")
    for mode in ("浮点历史", "8 位历史"):
        hist = None
        for f in range(4):
            out = net(color, hist=hist, frame=f)
            o8 = q255(out).cpu().numpy()
            ref = ppm(f)
            d, c = ref - color.cpu().numpy(), o8 - color.cpu().numpy()
            print(f"{mode} 第 {f} 帧: 相关 {np.corrcoef(o8.ravel(), ref.ravel())[0, 1]:.5f}  平均差 {np.abs(o8 - ref).mean() * 255:.2f}/255"
                  f"  改动量相关 {np.corrcoef(d.ravel(), c.ravel())[0, 1]:.4f}  改动幅度 kernel {np.abs(d).mean():.4f} torch {np.abs(c).mean():.4f}")
            hist = out if mode == "浮点历史" else q255(out)
