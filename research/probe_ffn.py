"""构造权重探针，精确解剖 block1 的 FFN 段。

W1 只在 (k0, j0) 为 1、W2 只在 (j0, n0) 为 1、残差系数清零、输出投影清零且 attn_cos_skip=1
=> 输出通道 n0 = g(x_k0)，g = 隐层激活 (以及可能的缩放/量化)。输入通道 k0 在像素间扫 -8..8 得到曲线。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import exp_block1 as B  # noqa: E402
from exp_ffn import mp_cubic_silu, q8  # noqa: E402
from act import E4M3  # noqa: E402

ENC = {float(v): i for i, v in enumerate(E4M3) if np.isfinite(v)}


def swizzle(M, tile_order="n_then_k"):
    """(K,N) 逻辑矩阵 (值须是 e4m3 可表示的) -> mma B 片段序字节 (unswizzle 的逆)"""
    K, N = M.shape
    ntile, ktile = N // 8, K // 32
    out = np.zeros(K * N, np.uint8)
    tiles = [(kt, nt) for kt in range(ktile) for nt in range(ntile)] if tile_order == "n_then_k" else \
            [(kt, nt) for nt in range(ntile) for kt in range(ktile)]
    for ti, (kt, nt) in enumerate(tiles):
        blk, half = divmod(ti, 2)
        for lane in range(32):
            g, t = lane >> 2, lane & 3
            for jj in range(8):
                k = 4 * t + jj % 4 + 16 * (jj // 4)
                out[blk * 512 + lane * 16 + half * 8 + jj] = ENC[float(M[kt * 32 + k, nt * 8 + g])]
    return out.tobytes()


def weights(W1=None, W2=None, c1=None, c2=None, proj_zero=True):
    w = bytearray(B.W_REAL)
    if W1 is not None:
        w[0:4096] = swizzle(W1)
    if W2 is not None:
        w[4096:8192] = swizzle(W2)
    if c1 is not None:
        w[8192:8288] = np.concatenate([np.zeros(8), c1, np.zeros(8)]).astype(np.float16).tobytes()
    if c2 is not None:
        w[20576:20672] = np.concatenate([np.zeros(8), c2, np.zeros(8)]).astype(np.float16).tobytes()
    if proj_zero:
        w[19552:20576] = bytes(1024)
    return bytes(w)


def ramp_input(k0):
    """输入内存通道 k0 在 (像素) 上扫所有有限 e4m3 值 [-8,8]，其余为 0"""
    vals = np.array(sorted(v for v in ENC if -8 <= v <= 8), np.float32)
    x = np.zeros((32, B.H, B.W), np.float32)
    flat = x[k0].reshape(-1)
    flat[:] = np.resize(vals, flat.size)
    return x, vals


def main():
    k0, j0, n0 = 3, 5, 7
    # 与下标对应无关的探针: W1 第 j0 列全 1 (所有输入汇入一个隐层单元)，W2 第 n0 列全 1 (所有隐层流向一个输出)
    W1 = np.zeros((32, 128), np.float32); W1[:, j0] = 1.0
    W2 = np.zeros((128, 32), np.float32); W2[:, n0] = 1.0
    x, vals = ramp_input(k0)
    y = B.run(B.act_bytes(x), weights(W1, W2, c1=np.zeros(32), c2=np.ones(32)))
    xs = x[k0].reshape(-1)
    live = [c for c in range(32) if np.abs(y[c]).max() > 0]
    print("有输出的通道 (片段序):", live)
    ch = live[0] if live else n0
    ys = y[ch].reshape(-1)
    print(" x        g(x)     mp_cubic   silu")
    for v in (-8, -4, -2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2, 4, 8):
        i = np.nonzero(xs == v)[0][0]
        mc = q8(mp_cubic_silu(np.array([v], np.float32)))[0]
        sl = q8(np.array([v / (1 + np.exp(-v))], np.float32))[0]
        print(f"{v:6.2f}  {ys[i]:8.4f}  {mc:8.4f}  {sl:8.4f}")
    np.save(os.path.join(os.path.dirname(__file__), "act_curve.npy"), np.stack([xs[:len(vals)], ys[:len(vals)]]))


if __name__ == "__main__":
    main()
