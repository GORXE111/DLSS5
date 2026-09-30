"""16h ffwd_proj (seq29, block24.layer1): y = q8(c1[cols]⊙x + q8(Z) @ D[CMAP 行][:, cols])，D 512x512 [0,262144)，c1 f16 [262144,263168)。
Z = ffwd 输出 (tin 片段序)，x = 块输入。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import block1_ref as R  # noqa: E402
import k16  # noqa: E402
import krun  # noqa: E402
import swin_gen as G  # noqa: E402
from act import chw_to_tin, random_fp8, tin_to_chw  # noqa: E402
from exp_ffn import q8, unswizzle  # noqa: E402

SEQ = 29
COLS = G.group_map(G.COLS32, 512)
INV_COLS = np.argsort(COLS)


def weights(seq=SEQ):
    krun.TRACE = k16.TRACE
    return krun.weight_of(seq)[2]


def run(Z, X, w, seq=SEQ):
    p0, p8, p16 = k16.ptr(seq, 0), k16.ptr(seq, 8), k16.ptr(seq, 16)
    r = k16.run(seq, {p0: chw_to_tin(Z), p8: chw_to_tin(X), p16: 122880}, bytes(w))[p16]
    return tin_to_chw(np.frombuffer(r, np.uint8), 512, 12, 20)


def ref(Zf, Xf, w):
    """Zf, Xf: (N, 512) 片段序"""
    D = unswizzle(bytes(w[0:262144]), 512, 512)
    c1 = np.frombuffer(bytes(w[262144:263168]), np.float16).astype(np.float32)
    P = Zf[:, INV_COLS]                                     # 片段序 -> 隐层序
    return q8(R.f16(c1[COLS] * Xf + P @ D[G.group_map(G.CMAP32, 512)][:, COLS]))


if __name__ == "__main__":
    w = weights()
    for lo in (-3, -1):
        Z = random_fp8((512, 12, 20), lo=lo, hi=-lo, seed=11)
        X = random_fp8((512, 12, 20), lo=lo, hi=-lo, seed=12)
        k = run(Z, X, w).reshape(512, -1).T
        r = ref(Z.reshape(512, -1).T, X.reshape(512, -1).T, w)
        print(f"±{-lo}: 相关 {np.corrcoef(k.ravel(), r.ravel())[0, 1]:.5f}  逐值精确 {(k == r).mean():.4f}")
