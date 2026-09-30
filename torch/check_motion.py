"""运动画面验收: nr-lab --temporal-shift 4 (图案每帧右移 4 像素)，运动矢量 mv_x = -4 (正确) 或 0 (错误)。
torch 逐帧递推 (历史 = 上一帧输出) 与 nr-lab 各帧输出比较。需要 research/out_mvok_f*.ppm、out_mv0_f*.ppm"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from check import color_pattern  # noqa: E402
from check_frames import ppm, q255  # noqa: E402
from dlss5 import DLSS5  # noqa: E402

R = os.path.join(HERE, "..", "research")
SHIFT = 4

if __name__ == "__main__":
    net = DLSS5()
    outs = {}
    for case, mvx in (("mvok", -SHIFT), ("mv0", 0)):
        mv = torch.zeros(360, 640, 2, device="cuda")
        mv[..., 0] = mvx
        hist = None
        for f in range(4):
            color = torch.tensor(color_pattern(offset_x=f * SHIFT), device="cuda")
            out = net(color, hist=hist, mv=None if hist is None else mv, frame=f)
            hist = out
            outs[case, f] = q255(out).cpu().numpy()
            if f == 0:
                continue
            ref = ppm(f"{case}_f{f}") if False else np.frombuffer(open(os.path.join(R, f"out_{case}_f{f}.ppm"), "rb").read()[-640 * 360 * 3:], np.uint8).reshape(360, 640, 3).astype(np.float32) / 255
            o = outs[case, f]
            c = color.cpu().numpy()
            print(f"{case} (mv_x={mvx:+d}) 第 {f} 帧: 相关 {np.corrcoef(o.ravel(), ref.ravel())[0, 1]:.5f}  平均差 {np.abs(o - ref).mean() * 255:.2f}/255"
                  f"  改动量相关 {np.corrcoef((ref - c).ravel(), (o - c).ravel())[0, 1]:.4f}")
            outs["ref", case, f] = ref
    for f in (1, 2, 3):
        dk = np.abs(outs["ref", "mvok", f] - outs["ref", "mv0", f]).mean() * 255
        dt = np.abs(outs["mvok", f] - outs["mv0", f]).mean() * 255
        print(f"第 {f} 帧: 正确/错误运动矢量两种输出之差  kernel {dk:.2f}/255  torch {dt:.2f}/255")
