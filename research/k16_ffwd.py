"""16h ffwd (FusedSwin2dFfwd 512) 解析: out = W2 · q8(act(W1 · x))，隐层 512。回归判定 W1 的输入序/swizzle，再定 W2。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import k16  # noqa: E402
import krun  # noqa: E402
from act import chw_to_tin, random_fp8, tin_to_chw  # noqa: E402
from exp_ffn import mp_cubic_silu, q8, unswizzle  # noqa: E402
import block1_ref as R  # noqa: E402
import swin_gen as G  # noqa: E402

SEQ = 28


def weights():
    krun.TRACE = k16.TRACE
    return krun.weight_of(SEQ)[2]


def run(X, w):
    q0, q8_ = k16.ptr(SEQ, 0), k16.ptr(SEQ, 8)
    r = k16.run(SEQ, {q0: chw_to_tin(X), q8_: 122880}, bytes(w))[q8_]
    return tin_to_chw(np.frombuffer(r, np.uint8), 512, 12, 20)


def dataset(w, n=6, lo=-3):
    Xs, Ys = [], []
    for s in range(n):
        X = random_fp8((512, 12, 20), lo=lo, hi=-lo, seed=100 + s)
        Xs.append(X.reshape(512, -1).T)
        Ys.append(run(X, w).reshape(512, -1).T)
    return np.vstack(Xs), np.vstack(Ys)


def r2(A, Y):
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    res = ((A @ coef - Y) ** 2).sum(0)
    tot = ((Y - Y.mean(0)) ** 2).sum(0)
    return 1 - res / np.maximum(tot, 1e-12), coef


def main():
    w = weights()
    X, Y = dataset(w)
    print("样本", X.shape)
    cands = {"tin 序": np.arange(512), "规范序 (canon_index)": G.canon_index(512),
             "逆规范序": np.argsort(G.canon_index(512))}
    for order in ("n_then_k", "k_then_n"):
        W1 = unswizzle(bytes(w[0:262144]), 512, 512, order)
        for nm, perm in cands.items():
            A = q8(mp_cubic_silu(R.f16(X[:, perm] @ W1)))
            rr, _ = r2(A, Y)
            print(f"W1 {order:9s} 输入 {nm:20s}: R² 中位 {np.median(rr):.4f}  最小 {rr.min():.4f}")


if __name__ == "__main__":
    main()


# ---- 结构 (原始 PTX o16_ffwd.ptx 读出): 8 组 q = (ctaid.z<<2)|(tid.y&3)，每组 64 个输出通道
#   h_q  = q8(x @ W1[:, 64q:64q+64])                      W1 块 (kc*16+hb)*1024，32x32，输入规范序
#   m_qj = q8(act(q8(h_q[:32]) @ Wa1 + q8(h_q[32:]) @ Wa2))  Wa1 = 262144+q*16384+j*1024，Wa2 = 270336+...，j = 0..7
#   out_q = Σ_j m_qj @ Wb_qj                               Wb = 393216+q*16384+j*2048 (32x64)
def w1_matrix(w):
    W1 = np.zeros((512, 512), np.float32)
    for kc in range(16):
        for hb in range(16):
            off = (kc * 16 + hb) * 1024
            W1[kc * 32:(kc + 1) * 32, hb * 32:(hb + 1) * 32] = unswizzle(bytes(w[off:off + 1024]), 32, 32)
    return W1


def mids(X, w, rowmap=None):
    """返回每组的中间激活 (N, 8组, 256)"""
    rowmap = np.arange(32) if rowmap is None else rowmap
    Xc = X[:, G.canon_index(512)]
    H = q8(R.f16(Xc @ w1_matrix(w)))
    out = np.zeros((X.shape[0], 8, 256), np.float32)
    for q in range(8):
        h1, h2 = H[:, 64 * q:64 * q + 32], H[:, 64 * q + 32:64 * q + 64]
        for j in range(8):
            a1 = unswizzle(bytes(w[262144 + q * 16384 + j * 1024:][:1024]), 32, 32)[rowmap]
            a2 = unswizzle(bytes(w[270336 + q * 16384 + j * 1024:][:1024]), 32, 32)[rowmap]
            out[:, q, 32 * j:32 * j + 32] = q8(mp_cubic_silu(R.f16(h1 @ a1 + h2 @ a2)))
    return out
