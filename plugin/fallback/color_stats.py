"""DLSS5 对颜色的影响: dlss5fb_in.ppm (原画面) vs dlss5fb_out.ppm (结果) 的饱和度、Lab 彩度、亮度、色相、冷暖与细节变化。
    python color_stats.py <dump 目录> [标签]"""
import os
import sys

import numpy as np


def rd(p):
    b = open(p, "rb").read()
    parts = b.split(maxsplit=4)
    w, h = int(parts[1]), int(parts[2])
    return np.frombuffer(b[-w * h * 3:], np.uint8).reshape(h, w, 3).astype(np.float64) / 255


def lab(c):
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def blur(x, r=4):
    k = np.ones(2 * r + 1) / (2 * r + 1)
    x = np.apply_along_axis(lambda v: np.convolve(v, k, "same"), 0, x)
    return np.apply_along_axis(lambda v: np.convolve(v, k, "same"), 1, x)


d = sys.argv[1]
tag = sys.argv[2] if len(sys.argv) > 2 else ""
a, b = rd(os.path.join(d, "dlss5fb_in.ppm")), rd(os.path.join(d, "dlss5fb_out.ppm"))
la, lb = lab(a), lab(b)
ca, cb = np.hypot(la[..., 1], la[..., 2]), np.hypot(lb[..., 1], lb[..., 2])
sat = lambda c: (c.max(-1) - c.min(-1)) / np.maximum(c.max(-1), 1e-6)
mask = ca > 8   # coloured pixels only for hue rotation
hue = np.degrees(np.angle(np.exp(1j * (np.arctan2(lb[..., 2], lb[..., 1]) - np.arctan2(la[..., 2], la[..., 1])))))[mask]
ga = la[..., 0] - blur(la[..., 0])
gb = lb[..., 0] - blur(lb[..., 0])
print(f"{tag:10s} 饱和度 x{sat(b).mean() / sat(a).mean():.3f}  Lab 彩度 x{cb.mean() / ca.mean():.3f}  "
      f"亮度 L {lb[..., 0].mean() - la[..., 0].mean():+.2f}  色相转动 {np.mean(np.abs(hue)):.1f}°  "
      f"冷暖 b* {lb[..., 2].mean() - la[..., 2].mean():+.2f}  细节 x{np.abs(gb).mean() / np.abs(ga).mean():.3f}  "
      f"平均改动 {np.abs(b - a).mean() * 255:.1f}/255")
