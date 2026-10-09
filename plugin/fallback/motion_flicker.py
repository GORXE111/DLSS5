"""motion_flicker.ps1 的计算: dlss5fb_out_<k>.ppm (k=1..n-1) 运动补偿后的帧间差 (跳过前 2 帧的建立期)，只算内部区域。"""
import os
import sys

import numpy as np


def rd(p):
    b = open(p, "rb").read()
    parts = b.split(maxsplit=4)
    w, h = int(parts[1]), int(parts[2])
    return np.frombuffer(b[-w * h * 3:], np.uint8).reshape(h, w, 3).astype(np.int16)


d, pan, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
outs = {k: rd(os.path.join(d, f"dlss5fb_out_{k}.ppm")) for k in range(1, n)}
ins = {k: rd(os.path.join(d, f"dlss5fb_in_{k}.ppm")) for k in range(1, n)}
m = 4 * abs(pan) + 8
diffs, big, effect = [], [], []
for k in range(3, n):
    a = np.roll(outs[k], -pan, axis=1)[8:-8, m:-m]
    b = outs[k - 1][8:-8, m:-m]
    e = np.abs(a - b).max(-1)
    diffs.append(np.abs(a - b).mean())
    big.append((e > 8).mean())
    effect.append(np.abs(outs[k] - ins[k]).mean())
print(f"运动补偿帧间差 {np.mean(diffs):.2f}/255  跳变>8 {np.mean(big):.2%}  DLSS5 改动量 {np.mean(effect):.1f}/255")
