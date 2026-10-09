"""精确模式 (f32 + 舍入模拟) / 快速模式 (半精度) / 快速模式 + CUDA Graph 的整帧耗时，以及与 nr-lab 实际输出的吻合度 (640x360 第 0 帧)
    python bench_modes.py [WxH ...]      默认 640x360 1280x720 1920x1080"""
import os
import sys
import time

import numpy as np
import torch

from check import RESEARCH, color_pattern
from dlss5 import DLSS5


def timeit(fn, n=5):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


def main():
    sizes = [tuple(map(int, s.split("x"))) for s in sys.argv[1:]] or [(640, 360), (1280, 720), (1920, 1080)]
    net = DLSS5()
    raw = open(os.path.join(RESEARCH, "out_f0.ppm"), "rb").read()
    ref = np.frombuffer(raw[-640 * 360 * 3:], np.uint8).reshape(360, 640, 3).astype(np.float32) / 255
    for precise in (True, False):
        net.precise, net.half = precise, not precise
        o = net(color_pattern(), frame=0).cpu().numpy()
        o = np.floor(o * 255 + 0.5) / 255
        print(f"{'精确' if precise else '快速'}模式 vs nr-lab: 相关 {np.corrcoef(o.ravel(), ref.ravel())[0, 1]:.5f}  "
              f"平均差 {np.abs(o - ref).mean() * 255:.2f}/255")
    for W, H in sizes:
        color = torch.rand(H, W, 3, device="cuda")
        hist = torch.rand(H, W, 3, device="cuda")
        row = [f"{W}x{H}"]
        for precise in (True, False):
            net.precise, net.half = precise, not precise
            row.append(f"{'精确' if precise else '快速'} {timeit(lambda: net(color, hist=hist, frame=1), 3 if precise else 5):6.0f} ms")
        run = net.graph(H, W)
        row.append(f"快速+Graph {timeit(lambda: run(color, hist=hist, frame=1)):6.0f} ms")
        print("  ".join(row))
        del run
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
