"""验收: nr-lab 第 0 帧 (重置帧) 的合成输入 -> DLSS5 (GPU) -> 与 nr-lab 在 RTX 3060 上的实际输出比较，并与 numpy 参考比较、计时。

    python check.py            # 需要 research/out_f0.ppm (nr-lab --frames 1 的输出)
"""
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dlss5 import DLSS5  # noqa: E402

RESEARCH = os.path.join(HERE, "..", "research")


def color_pattern(W=640, H=360):
    """nr-lab 的合成输入 (MakeColorPattern, sRGB, R8G8B8A8_UNORM)"""
    y, x = np.mgrid[0:H, 0:W]
    checker = ((x // 12) ^ (y // 12)) & 1
    line = (x % 61 < 2) | (y % 47 < 2) | ((x + y) % 79 < 2)
    img = np.stack([np.where(line, 1.0, np.where(checker, 0.82, 0.06)),
                    np.where(line, 0.18, np.where(checker, 0.11, 0.68)),
                    np.where(line, 0.04, np.where(checker, 0.55, 0.09))], -1)
    return (np.floor(img * 255.0 + 0.5) / 255.0).astype(np.float32)


def stats(name, o, ref):
    o8 = np.floor(o * 255 + 0.5) / 255
    print(f"{name}: 相关 {np.corrcoef(o8.ravel(), ref.ravel())[0, 1]:.5f}  平均差 {np.abs(o8 - ref).mean() * 255:.2f}/255  "
          f"8 位一致 {(o8 == ref).mean():.4f}  差<=1 {(np.abs(o8 - ref) <= 1.01 / 255).mean():.4f}")


def main():
    t0 = time.time()
    net = DLSS5()
    torch.cuda.synchronize()
    print(f"加载+解码权重: {time.time() - t0:.1f}s")
    color = color_pattern()
    out = net(color, frame=0).cpu().numpy()
    raw = open(os.path.join(RESEARCH, "out_f0.ppm"), "rb").read()
    ref = np.frombuffer(raw[-640 * 360 * 3:], np.uint8).reshape(360, 640, 3).astype(np.float32) / 255
    stats("torch vs nr-lab 实际输出", out, ref)
    p = os.path.join(RESEARCH, "net_frame0_out.npy")
    if os.path.exists(p):
        stats("torch vs numpy 参考   ", out, np.floor(np.load(p) * 255 + 0.5) / 255)
    d, c = ref - color, np.floor(out * 255 + 0.5) / 255 - color
    print(f"网络改动量 (输出-输入) 相关: {np.corrcoef(d.ravel(), c.ravel())[0, 1]:.5f}")
    hist = out
    for i in range(3):
        torch.cuda.synchronize()
        t1 = time.time()
        hist = net(color, hist=hist, frame=i + 1)
        torch.cuda.synchronize()
        print(f"第 {i + 1} 帧 (带历史): {(time.time() - t1) * 1000:.0f} ms")


if __name__ == "__main__":
    main()
