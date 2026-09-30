"""16h 分组 swin 块 (block23-30, 40-47) 完整参考: 4 个 kernel 串联。
  Z  = ffwd(x)                      分组低秩 FFN (k16_ffwd)
  y  = q8(c1⊙x + q8(Z)·D)          ffwd_proj (k16_ffproj.ref)
  O  = attn(y)                      QKV + 近似 exp 的截断 softmax (k16_attn)
  out= q8(c2⊙y + q8(O)·Wproj)      proj (与 ffwd_proj 同式)
所有激活为 tin 片段序 (N=240 token, 512 通道)。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import k16  # noqa: E402
import k16_attn as A  # noqa: E402
import k16_ffproj as P  # noqa: E402
import k16_ffwd as F  # noqa: E402
import krun  # noqa: E402
import swin_gen as G  # noqa: E402
from act import tin_to_chw  # noqa: E402
from exp_ffn import q8, unswizzle  # noqa: E402

COLS = G.group_map(G.COLS32, 512)


def ffwd(Xf, w):
    M = F.mids(Xf, w, np.array(G.CMAP32))
    Pm = np.zeros((Xf.shape[0], 512), np.float32)
    for q in range(8):
        for j in range(8):
            Wb = unswizzle(bytes(w[393216 + q * 16384 + j * 2048:][:2048]), 32, 64)[G.CMAP32]
            Pm[:, 64 * q:64 * q + 64] += M[:, q, 32 * j:32 * j + 32] @ Wb
    return q8(Pm)[:, COLS]


def block(Xf, ws, shift):
    Z = ffwd(Xf, ws[0])
    y = P.ref(Z, Xf, ws[1])
    O = q8(A.attention(y, ws[2], shift))[:, COLS]
    return P.ref(O, y, ws[3])


def seqs_weights(first):
    krun.TRACE = k16.TRACE
    return [krun.weight_of(first + i)[2] for i in range(4)]


if __name__ == "__main__":
    first = 28                                                 # block24: seq 28-31
    X = tin_to_chw(np.frombuffer(k16.tap(first - 1), np.uint8), 512, 12, 20).reshape(512, -1).T
    T = {s: tin_to_chw(np.frombuffer(k16.tap(s), np.uint8), 512, 12, 20).reshape(512, -1).T for s in range(first, first + 4)}
    ws = seqs_weights(first)
    Z = ffwd(X, ws[0])
    print("ffwd 真实数据: 参考非零", int((Z != 0).sum()), " kernel 非零", int((T[first] != 0).sum()))
    y = P.ref(T[first], X, ws[1])
    O = q8(A.attention(T[first + 1], ws[2], A.shift(first + 2)))[:, COLS]
    o = P.ref(T[first + 2], T[first + 1], ws[3])
    for nm, r, k in (("ffwd_proj", y, T[first + 1]), ("attn", O, T[first + 2]), ("proj", o, T[first + 3])):
        print(f"{nm:10s} 单级: 相关 {np.corrcoef(r.ravel(), k.ravel())[0, 1]:.5f}  精确 {(r == k).mean():.4f}")
    full = block(X, ws, A.shift(first + 2))
    k = T[first + 3]
    print(f"整块串联: 相关 {np.corrcoef(full.ravel(), k.ravel())[0, 1]:.5f}  精确 {(full == k).mean():.4f}")
