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
from acceptance import execute, parser, ppm as read_ppm  # noqa: E402

R = os.path.join(HERE, "..", "research")


def ppm(path, W, H):
    return read_ppm(path, W, H)


def main(report):
    # Check every requested reference before allocating the model; no silent resolution skips.
    refs = {}
    for res in report.args.sizes:
        W, H = map(int, res.split("x"))
        if min(W, H) <= 0:
            raise ValueError(f"Invalid resolution: {res}")
        p = report.args.reference_dir / f"out_f0_{res}.ppm"
        refs[res] = ppm(p, W, H)
        report.reference(p)
    net = DLSS5()
    for res, ref in refs.items():
        W, H = map(int, res.split("x"))
        color = color_pattern(W, H)
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = net(color).cpu().numpy()
        dt = time.perf_counter() - t
        o8 = np.floor(out * 255 + 0.5) / 255
        report.compare(res, out, ref, quantize=True)
        d, c = ref - color, o8 - color
        print(f"{res} (网格 {dims(H, W)}): 相关 {np.corrcoef(o8.ravel(), ref.ravel())[0, 1]:.5f}  平均差 {np.abs(o8 - ref).mean() * 255:.2f}/255"
              f"  改动量相关 {np.corrcoef(d.ravel(), c.ravel())[0, 1]:.4f}  改动幅度 kernel {np.abs(d).mean():.4f} torch {np.abs(c).mean():.4f}  {dt:.1f}s")


if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("sizes", nargs="*", default=["960x540", "1280x720", "1920x1080"])
    sys.exit(execute("check_res", p.parse_args(), main))
