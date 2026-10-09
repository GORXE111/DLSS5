"""帧间闪烁量: DumpFrame / DumpCount 存下的连续帧 (dlss5fb_in*.ppm / dlss5fb_out*.ppm)，
算相邻两帧的平均差。静止镜头下输入几乎不变，输出的帧间差就是 DLSS5 带来的闪烁。
    python flicker.py <dump 目录> [n]"""
import os
import sys

import numpy as np


def rd(p):
    b = open(p, "rb").read()
    parts = b.split(maxsplit=4)
    w, h = int(parts[1]), int(parts[2])
    return np.frombuffer(b[-w * h * 3:], np.uint8).reshape(h, w, 3).astype(np.int16)


def main():
    d = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    name = lambda kind, k: os.path.join(d, f"dlss5fb_{kind}.ppm" if k == 0 else f"dlss5fb_{kind}_{k}.ppm")
    ins = [rd(name("in", k)) for k in range(n)]
    outs = [rd(name("out", k)) for k in range(n)]
    din = np.mean([np.abs(ins[k] - ins[k - 1]).mean() for k in range(1, n)])
    dout = np.mean([np.abs(outs[k] - outs[k - 1]).mean() for k in range(1, n)])
    big = np.mean([(np.abs(outs[k] - outs[k - 1]).max(-1) > 8).mean() for k in range(1, n)])
    print(f"帧间平均差: 输入 {din:.2f}/255  输出 {dout:.2f}/255  (输出里变化 >8/255 的像素 {big:.1%})")


if __name__ == "__main__":
    main()
