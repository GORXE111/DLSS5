"""隔离 block1 的 FFN 段: 输出投影清零 + attn_cos_skip 设 1 -> 输出 = FFN 段结果 (FP8 量化后)。
再用 PyTorch 按假设计算，对照。

权重在显存里按 mma B 片段预排: 每 512 字节 = 32 线程 x 16 字节 = 两个 32(K)x8(N) 块；
线程 (g=L>>2, t=L&3) 的第 j 字节 (j<8 属块 0，否则块 1；jj=j%8): k = 4t + jj%4 + 16*(jj//4), n = g。
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from act import E4M3  # noqa: E402
import exp_block1 as B  # noqa: E402


def unswizzle(raw: bytes, K: int, N: int, tile_order="n_then_k"):
    """片段序字节 -> (K, N) FP8 数值矩阵。块顺序待定，给两种候选。"""
    v = E4M3[np.frombuffer(raw, np.uint8)]
    ntile, ktile = N // 8, K // 32
    out = np.zeros((K, N), np.float32)
    tiles = [(kt, nt) for kt in range(ktile) for nt in range(ntile)] if tile_order == "n_then_k" else \
            [(kt, nt) for nt in range(ntile) for kt in range(ktile)]
    for ti, (kt, nt) in enumerate(tiles):
        blk, half = divmod(ti, 2)
        for lane in range(32):
            g, t = lane >> 2, lane & 3
            for jj in range(8):
                k = 4 * t + jj % 4 + 16 * (jj // 4)
                out[kt * 32 + k, nt * 8 + g] = v[blk * 512 + lane * 16 + half * 8 + jj]
    return out


def mp_cubic_silu(x):
    t = np.clip(x, -4, 4)
    return x * (-0.055908203 * t * np.abs(t) + 0.447265625 * t + 0.894531250)


def q8(x):
    """RNE 到 e4m3 (satfinite)"""
    return torch.from_numpy(np.clip(x, -448, 448).astype(np.float32)).to(torch.float8_e4m3fn).float().numpy()


def perm_in(c):
    """输出片段通道 c -> 输入内存通道 (实验测得，相关 0.996-1.000)"""
    return 4 * ((c % 8) // 2) + 16 * (c % 2) + c // 8


def perm_param(c):
    """输出片段通道 c -> 系数向量里的下标: 二进制位 new_b4=b0, new_b3=b4, new_b0=b3，b1/b2 不动"""
    b = [(c >> i) & 1 for i in range(5)]
    return b[3] | b[1] << 1 | b[2] << 2 | b[4] << 3 | b[0] << 4


def main2():
    """统一到输出片段序后测 FFN 段"""
    w = bytearray(B.W_REAL)
    w[19552:20576] = bytes(1024)
    w[20576:20672] = np.array([0] * 8 + [1.0] * 32 + [0] * 8, np.float16).tobytes()
    Y = B.run(B.INP_REAL, bytes(w)).reshape(32, -1).T
    Xin = B.load_input().reshape(32, -1).T
    PI = [perm_in(c) for c in range(32)]
    SG = [perm_param(c) for c in range(32)]
    c1 = np.frombuffer(B.W_REAL[8192:8288], np.float16).astype(np.float32)[8:40]
    Xf, cf = Xin[:, PI], c1[SG]
    W2 = unswizzle(B.W_REAL[4096:8192], 128, 32)
    for w1name, W1 in (("W1 行=输入序", unswizzle(B.W_REAL[0:4096], 32, 128)),
                       ("W1 行=片段序", unswizzle(B.W_REAL[0:4096], 32, 128)[np.argsort(PI)])):
        for tord in ("n_then_k", "k_then_n"):
            W2t = unswizzle(B.W_REAL[4096:8192], 128, 32, tord)
            for actname, act in (("mp_cubic", mp_cubic_silu), ("silu", lambda h: h / (1 + np.exp(-h)))):
                h = Xin @ W1
                for qh in (True, False):
                    pred = cf * Xf + (q8(act(h)) if qh else act(h)) @ W2t
                    err = np.abs(q8(pred) - Y)
                    r = np.corrcoef(pred.ravel(), Y.ravel())[0, 1]
                    print(f"{w1name} W2块序={tord:9s} {actname:8s} 隐层量化={qh!s:5s} 相关 {r:.4f} 逐位相等 {(err == 0).mean():.3f}")


def main():
    w = bytearray(B.W_REAL)
    w[19552:20576] = bytes(1024)                                     # 输出投影清零
    w[20576:20672] = np.array([0] * 8 + [1.0] * 32 + [0] * 8, np.float16).tobytes()  # attn_cos_skip = 1
    y = B.run(B.INP_REAL, bytes(w))                                  # (C,H,W)
    x = B.load_input()
    W1 = unswizzle(B.W_REAL[0:4096], 32, 128)
    W2 = unswizzle(B.W_REAL[4096:8192], 128, 32)
    c1 = np.frombuffer(B.W_REAL[8192:8288], np.float16).astype(np.float32)[8:40]
    X = x.reshape(32, -1).T                                          # (像素, 32)
    Y = y.reshape(32, -1).T
    cands = {}
    h = X @ W1
    cands["c1*x + W2(act(W1 x))"] = c1 * X + q8(mp_cubic_silu(h)) @ W2
    cands["x + W2(act(W1 x))"] = X + q8(mp_cubic_silu(h)) @ W2
    cands["c1*x + W2(silu(W1 x))"] = c1 * X + q8(h / (1 + np.exp(-h))) @ W2
    cands["W2(act(W1 x))"] = q8(mp_cubic_silu(h)) @ W2
    for name, pred in cands.items():
        err = np.abs(q8(pred) - Y)
        r = np.corrcoef(pred.ravel(), Y.ravel())[0, 1]
        print(f"{name:26s} 相关 {r:.4f}  平均误差 {err.mean():.4f}  逐位相等 {(err == 0).mean():.3f}")


if __name__ == "__main__":
    main()
