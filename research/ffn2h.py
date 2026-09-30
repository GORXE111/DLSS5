"""2h FFN 结构验证 (线性回归法): 隔离 FFN 后，若隐层激活 h 算对，输出应能被 [x, h] 的线性组合近乎完美拟合。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import krun  # noqa: E402
from act import TAPS, tin_to_chw  # noqa: E402
from exp_ffn import mp_cubic_silu, q8, unswizzle  # noqa: E402

SEQ, IN, OUT, NB = 7, 0x1C19B000, 0x1C28B000, 983040
C, H, W = 64, 96, 160
X_BYTES = open(os.path.join(TAPS, "tap_s6_1c19b000.bin"), "rb").read()
_, _, WREAL = krun.weight_of(SEQ)


def run(w, xbytes=None):
    out = krun.run(SEQ, {IN: xbytes or X_BYTES, OUT: NB}, bytes(w))[OUT]
    return tin_to_chw(np.frombuffer(out, np.uint8), C, H, W)


def isolated_ffn_weights():
    w = bytearray(WREAL)
    w[57520:61616] = bytes(4096)                                     # 投影清零
    w[61616:61744] = np.ones(64, np.float16).tobytes()               # c2 = 1
    w[28688:28816] = bytes(128)                                      # c1 = 0 -> 输出只剩 FFN
    return w


def hidden(Xc, w, khalf_order="AB", tile="n_then_k", warp_stride=8192):
    """按假设算 256 个隐层: warp w, 轮 j: A = w*warp_stride + j*1024 (k 0-31), B = +4096 (k 32-63)"""
    hs = []
    for wp in range(2):
        for j in range(4):
            a0 = wp * warp_stride + j * 1024
            A = unswizzle(w[a0:a0 + 1024], 32, 32, tile)
            Bm = unswizzle(w[a0 + 4096:a0 + 5120], 32, 32, tile)
            W1 = np.vstack([A, Bm]) if khalf_order == "AB" else np.vstack([Bm, A])
            hs.append(Xc @ W1)
    return np.hstack(hs)                                             # (N, 256)


def r2(Y, F):
    M, *_ = np.linalg.lstsq(F, Y, rcond=None)
    P = F @ M
    return 1 - ((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()


def main():
    from act import chw_to_tin, random_fp8
    x = random_fp8((C, H, W), -3, 3)                                  # 大幅度随机输入，激活进入非线性区
    Y = run(isolated_ffn_weights(), chw_to_tin(x)).reshape(C, -1).T
    X = x.reshape(C, -1).T
    idx = np.random.default_rng(0).choice(len(Y), 6000, replace=False)
    Y, X = Y[idx], X[idx]
    print(f"只用 x 的线性拟合 R² = {r2(Y, np.hstack([X, np.ones((len(X), 1))])):.4f}  (基线)")
    from exp_ffn import perm_in
    PI = [perm_in(c) for c in range(32)]
    canon_group = np.zeros(64, int)
    for gi in range(2):
        for c in range(32):
            canon_group[gi * 32 + PI[c]] = gi * 32 + c           # 规范通道 gi*32+PI[c] <- 片段通道 gi*32+c
    # ---- 第二层: 块对角 W2 (每 warp 128 隐层 -> 32) 验证
    Xc = X[:, canon_group]
    a = q8(mp_cubic_silu(hidden(Xc, WREAL, "AB", "n_then_k")))     # (N,256): 列 = warp*128 + j*32 + n
    import json
    Cmap = json.load(open(os.path.join(os.path.dirname(__file__), "maps_block1.json")))["W1col_to_W2row"][:32]
    for rowmap_name, rm in (("隐层同 1h 重排", Cmap), ("隐层不重排", list(range(32)))):
        zs = []
        for wp in range(2):
            z = 0
            for j in range(4):
                c0 = 16384 + wp * 4096 + j * 1024
                W2 = unswizzle(WREAL[c0:c0 + 1024], 32, 32)[rm, :]
                z = z + a[:, wp * 128 + j * 32: wp * 128 + (j + 1) * 32] @ W2
            zs.append(z)
        Z = np.hstack(zs)
        print(f"第二层 ({rowmap_name}): [x, z(64)] 拟合 R² = {r2(Y, np.hstack([X, Z, np.ones((len(X), 1))])):.5f}")
        print(f"            [x, q8(z)] 拟合 R² = {r2(Y, np.hstack([X, q8(Z), np.ones((len(X), 1))])):.5f}")
    variants = {"片段序": X, "组内 perm_in": X[:, canon_group]}
    for xname, Xc in variants.items():
        for ko in ("AB", "BA"):
            h = hidden(Xc, WREAL, ko, "n_then_k")
            F = np.hstack([X, q8(mp_cubic_silu(h)), np.ones((len(X), 1))])
            print(f"输入={xname:10s} K 半序={ko}: [x, act(h)] 拟合 R² = {r2(Y, F):.5f}")


if __name__ == "__main__":
    main()
