"""mv_check.ps1 的比较部分 (第 1 帧 = 序列第 0 帧，重置帧)。
    python mv_check.py <dump 目录> <research 目录>   nr-lab 参考: out_f0 / out_mvok_f* (正确运动矢量) / out_mv0_f* (零运动矢量)
    python mv_check.py <dump 目录> --self            参考: 同目录 ref_nr_*.ppm (常数精确运动矢量跑出的结果)"""
import os
import sys

import numpy as np


def rd(p):
    b = open(p, "rb").read()
    parts = b.split(maxsplit=4)
    w, h = int(parts[1]), int(parts[2])
    return np.frombuffer(b[-w * h * 3:], np.uint8).reshape(h, w, 3).astype(np.int16)


d, r = sys.argv[1], sys.argv[2]
inner = (slice(8, -8), slice(16, -16))   # the wrap seam at the image border has no valid flow
if r == "--self":
    for k in (1, 2, 3):
        o, ref = rd(os.path.join(d, f"dlss5fb_nr_{k}.ppm")), rd(os.path.join(d, f"ref_nr_{k}.ppm"))
        print(f"f{k}: 光流运动矢量 vs 精确运动矢量 {np.abs(o - ref).mean():.2f}/255 (内部 {np.abs(o - ref)[inner].mean():.2f})")
else:
    print(f"f0 vs nr-lab: {np.abs(rd(os.path.join(d, 'dlss5fb_nr.ppm')) - rd(os.path.join(r, 'out_f0.ppm'))).mean():.2f}/255")
    for k in (1, 2, 3):
        o = rd(os.path.join(d, f"dlss5fb_nr_{k}.ppm"))
        ok = rd(os.path.join(r, f"out_mvok_f{k}.ppm"))
        z = rd(os.path.join(r, f"out_mv0_f{k}.ppm"))
        print(f"f{k}: vs 正确运动矢量参考 {np.abs(o - ok).mean():.2f}/255 (内部 {np.abs(o - ok)[inner].mean():.2f})   "
              f"vs 零运动矢量参考 {np.abs(o - z).mean():.2f}/255   (两个参考相差 {np.abs(ok - z).mean():.2f})")
