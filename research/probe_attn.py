"""block1 注意力段的权重探针。

实验条件: FFN 收缩清零 + c1=1 -> 注意力段输入 y = 原输入 (片段序)；c2=0 -> 输出 = proj(attn(y))。
qkv 区 [8288,11360) = 32(K)x96(N) fp8，投影区 [19552,20576) = 32x32 fp8，都按 mma B 片段序存放。
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import exp_block1 as B  # noqa: E402
from probe_ffn import swizzle  # noqa: E402

H, W = B.H, B.W
ONES32 = np.ones(32)


def weights_attn(Wqkv=None, Wproj=None, c2=np.zeros(32), bias=None, tau=None):
    """布局 (PTX 证据修正): [19552,19568) 4xf32 (头 0 的温度 τ=0.4088，其余 0) |
    [19568,20592) 输出投影 fp8 | [20592,20656) c2 32xf16 | [20656,20672) 零。tau=None 保留真实值"""
    w = bytearray(B.W_REAL)
    w[4096:8192] = bytes(4096)                                                  # FFN 收缩清零
    w[8192:8288] = np.concatenate([np.zeros(8), ONES32, np.zeros(8)]).astype(np.float16).tobytes()  # c1 = 1
    if Wqkv is not None:
        w[8288:11360] = swizzle(Wqkv)
    if bias is not None:
        w[11360:19552] = np.asarray(bias, np.float16).tobytes()
    if tau is not None:
        w[19552:19568] = np.array([tau, 0, 0, 0], np.float32).tobytes()
    if Wproj is not None:
        w[19568:20592] = swizzle(Wproj)
    w[20592:20656] = np.asarray(c2, np.float16).tobytes()
    return bytes(w)


def const_input(v=1.0):
    return B.act_bytes(np.full((32, H, W), v, np.float32))


def live(y, tol=0):
    return [c for c in range(32) if np.abs(y[c]).max() > tol]


def find_v_columns():
    x = const_input(1.0)
    Wproj = np.ones((32, 32), np.float32)
    vcols = []
    for n in range(96):
        Wqkv = np.zeros((32, 96), np.float32); Wqkv[:, n] = 1.0
        y = B.run(x, weights_attn(Wqkv, Wproj))
        if live(y):
            vcols.append((n, float(np.abs(y).max())))
    return vcols


SCALES = np.array([1 + (m % 8) / 8 for m in range(32)], np.float32) * np.array([2.0 ** (m // 8 - 2) for m in range(32)])


def measure_P(bias=None, Wqk=None, value=1.0):
    """测 64x64 注意力权重: 每个窗口第 j 个像素放信号 (全部通道=value)，V=第 64 列全一 (v = Σ 通道)，
    投影第 m 列 = SCALES[m] -> 32 个输出通道 = 不同比例的同一量，合起来估计 P[i, j]。
    Wqk: (32, 64) 的 Q/K 列 (默认全零，P = softmax(偏置))。返回 P[i_token, j_token]，token = py*8+px。"""
    Wqkv = np.zeros((32, 96), np.float32)
    Wqkv[:, 64] = 1.0
    if Wqk is not None:
        Wqkv[:, :64] = Wqk
    Wproj = np.tile(SCALES, (32, 1)).astype(np.float32)          # 第 m 列全为 SCALES[m]
    w = weights_attn(Wqkv, Wproj, bias=bias)
    colmap = json.load(open(os.path.join(os.path.dirname(__file__), "maps_block1.json")))["W2col_to_outfrag"]
    P = np.zeros((64, 64))
    v = 32 * value
    for j in range(64):
        x = np.zeros((32, H, W), np.float32)
        x[:, j // 8::8, j % 8::8] = value
        y = B.run(B.act_bytes(x), w)                                  # (32, H, W)，片段序通道
        win = y[:, 8:16, 8:16].reshape(32, 64)                        # 取一个完整窗口 (第 1 行第 1 列)
        est = np.stack([win[colmap[m]] / (v * SCALES[m]) for m in range(32)])   # (32, 64)
        P[:, j] = np.median(est, axis=0)
    return P


if __name__ == "__main__":
    v = find_v_columns()
    print(f"V 列 ({len(v)} 个):", [n for n, _ in v])
    print("对应输出幅度:", sorted({round(m, 3) for _, m in v}))
