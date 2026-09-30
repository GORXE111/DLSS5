"""整网串联对照: 输入 = pre_block 输出抓取 (s1)，逐级出口与抓取比较"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import net_ref as N  # noqa: E402
from act import E4M3  # noqa: E402

D = os.path.join(os.path.dirname(__file__), "tapsnet")
TRACE = os.path.join(D, "nr-trace.tsv")


def img(path, C, H, W):
    v = E4M3[np.frombuffer(open(path, "rb").read(), np.uint8)[:C * H * W]]
    return v.reshape(C // 16, H, W, 16).transpose(1, 2, 0, 3).reshape(H * W, C)


def find(seq, d=D, size=None):
    f = [n for n in os.listdir(d) if n.startswith(f"tap_s{seq}_")
         and (size is None or os.path.getsize(os.path.join(d, n)) >= size)]
    f.sort(key=lambda n: os.path.getsize(os.path.join(d, n)))
    return os.path.join(d, f[0]) if f else None


CHECK = {5: (64, 96, 160), 9: (128, 48, 80), 15: (256, 24, 40), 23: (512, 12, 20), 131: (512, 12, 20),
         139: (256, 24, 40), 145: (128, 48, 80), 149: (64, 96, 160), 153: (32, 192, 320)}
EXTRA = {23: "taps8t", 131: "taps8t"}

if __name__ == "__main__":
    t0 = time.time()
    pre = img(find(1), 32, 192, 320)
    np.save(os.path.join(os.path.dirname(__file__), "net_pre.npy"), pre)
    outs, cur, x = N.run_net(pre, TRACE, log=lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True))
    for seq, (C, H, W) in CHECK.items():
        p = find(seq, size=C * H * W) or find(seq, os.path.join(os.path.dirname(__file__), EXTRA.get(seq, "tapsnet")), C * H * W)
        if seq not in outs or p is None:
            print("缺", seq)
            continue
        k = img(p, C, H, W)
        r = outs[seq]
        print(f"seq{seq} 出口 ({C}ch {H}x{W}): 相关 {np.corrcoef(r.ravel(), k.ravel())[0, 1]:.5f}  "
              f"逐值精确 {(r == k).mean():.4f}")
    np.savez(os.path.join(os.path.dirname(__file__), "net_outs.npz"), **{str(k): v for k, v in outs.items()})
