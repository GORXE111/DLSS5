"""逐块对照: 每块用上一块的真实抓取作输入，只测该块自身 (定位串联误差来源)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import klab  # noqa: E402
import net_ref as N  # noqa: E402
from act import E4M3, tin_to_chw  # noqa: E402

TD = os.path.join(os.path.dirname(__file__), "taps")
TR = os.path.join(TD, "nr-trace.tsv")


def tin(seq, addr, C, H, W):
    return tin_to_chw(np.frombuffer(open(os.path.join(TD, f"tap_s{seq}_{addr}.bin"), "rb").read(), np.uint8), C, H, W).reshape(C, -1).T


def img(seq, addr, C, H, W):
    v = E4M3[np.frombuffer(open(os.path.join(TD, f"tap_s{seq}_{addr}.bin"), "rb").read(), np.uint8)]
    return v.reshape(C // 16, H, W, 16).transpose(1, 2, 0, 3).reshape(H * W, C)


def cmp(nm, r, k):
    print(f"{nm}: 相关 {np.corrcoef(r.ravel(), k.ravel())[0, 1]:.5f} 精确 {(r == k).mean():.4f}")


if __name__ == "__main__":
    pre = img(1, "1c19b000", 32, 192, 320)
    W, H, Wd = N.LEVEL["1h"]
    x1 = tin(2, "1c37b000", 32, H, Wd)
    p = klab.trace_row(2, TR)["params"]
    sh = N.shift_of(p, H, Wd)
    r = N.std_block(N.img_to_frag(pre, 32), N.record("block1.layer0.layer"), 32, H, Wd, sh)
    cmp(f"block1 (平移 {sh})", r, x1)
    x2 = tin(3, "1c55b000", 32, H, Wd)
    sh = N.shift_of(klab.trace_row(3, TR)["params"], H, Wd)
    cmp(f"block2 (平移 {sh})", N.std_block(x1, N.record("block2.layer0.layer"), 32, H, Wd, sh), x2)
