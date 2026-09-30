"""逐级对照: 以抓取的 pre_block 输出为起点跑 torch 整网，各级出口与 kernel 抓取比较 (需要 research/tapsnet 等抓取数据)"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dlss5 import DLSS5  # noqa: E402

R = os.path.join(HERE, "..", "research")
E4M3 = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float().numpy()


def img(path, C, H, W):
    v = E4M3[np.frombuffer(open(path, "rb").read(), np.uint8)[:C * H * W]]
    return v.reshape(C // 16, H, W, 16).transpose(1, 2, 0, 3).reshape(H * W, C)


def slope(r, k):
    """kernel ≈ a·torch 的最小二乘斜率 (幅度比)"""
    return float((r * k).sum() / (r * r).sum())


def find(seq, C, H, W):
    for d in ("tapsnet", "taps8t"):
        dd = os.path.join(R, d)
        fs = sorted((f for f in os.listdir(dd) if f.startswith(f"tap_s{seq}_") and os.path.getsize(os.path.join(dd, f)) >= C * H * W),
                    key=lambda f: os.path.getsize(os.path.join(dd, f)))
        if fs:
            return os.path.join(dd, fs[0])


# 调度步序号 -> (kernel 序号, 通道, H, W)
CHECK = {4: (5, 64, 96, 160), 8: (9, 128, 48, 80), 14: (15, 256, 24, 40), 22: (23, 512, 12, 20), 47: (131, 512, 12, 20),
         55: (139, 256, 24, 40), 61: (145, 128, 48, 80), 65: (149, 64, 96, 160), 69: (153, 32, 192, 320)}

if __name__ == "__main__":
    net = DLSS5()
    pre = torch.tensor(img(find(1, 32, 192, 320), 32, 192, 320), device="cuda")
    st0, m0 = net.steps[0]
    net.steps[0] = (st0, lambda *a: (pre, None))              # 用抓取的 pre 输出替换 pre_block
    net.steps = net.steps[:-1]                                  # 不跑 post
    tr = {}
    try:
        net(np.zeros((360, 640, 3), np.float32), trace=tr)
    except UnboundLocalError:
        pass
    for i, (seq, C, H, W) in CHECK.items():
        k = img(find(seq, C, H, W), C, H, W)
        r = tr[i].cpu().numpy()
        print(f"step{i:2d} (seq{seq}) {C}ch {H}x{W}: 相关 {np.corrcoef(r.ravel(), k.ravel())[0, 1]:.5f}  逐值精确 {(r == k).mean():.4f}"
              f"  幅度比 kernel/torch {slope(r, k):.4f}  std {k.std():.3f}/{r.std():.3f}")
