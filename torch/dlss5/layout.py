"""权重/激活的布局解码 (numpy，只在加载权重时用一次)。

所有映射均由逆向实测得到，出处见 tools/notes.jsonl 与 research/*.py:
  - 权重矩阵是 FP8 e4m3，按 mma B 片段预排 (unswizzle)；系数向量是 f16
  - 激活的"片段序"(tin 内存序) 与"规范序"之间差 canon_index 排列
  - 各类通道重排 (COLS32 / CMAP32 / A_MAP / C_MAP) 都按 32 通道一组重复
"""
import json
import os

import numpy as np
import torch

DATA = os.path.join(os.path.dirname(__file__), "data")
_M = json.load(open(os.path.join(DATA, "maps_block1.json")))
A_MAP = np.array(_M["W2col_to_outfrag"])           # 1h: W2 列 -> 输出片段通道
C_MAP = np.array(_M["W1col_to_W2row"])             # 1h: W1 列 (隐层) -> W2 行
CMAP32 = C_MAP[:32].tolist()                       # 32 块内的隐层/累加器重排
COLS32 = [0, 16, 2, 18, 4, 20, 6, 22, 1, 17, 3, 19, 5, 21, 7, 23,
          8, 24, 10, 26, 12, 28, 14, 30, 9, 25, 11, 27, 13, 29, 15, 31]   # 输出片段通道 -> 系数下标 (= perm_param)
EPS = np.float32(6.2e-05)                          # 余弦注意力的范数下限


def perm_in(c):
    """片段通道 c -> 规范 (输入内存) 通道"""
    return 4 * ((c % 8) // 2) + 16 * (c % 2) + c // 8


PI = [perm_in(c) for c in range(32)]


def group_map(pat, W):
    return np.array([g * 32 + p for g in range(W // 32) for p in pat])


def canon_index(W):
    """规范序 = 片段序[:, canon_index(W)]"""
    cg = np.zeros(W, int)
    for gi in range(W // 32):
        for c in range(32):
            cg[gi * 32 + PI[c]] = gi * 32 + c
    return cg


E4M3 = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float().numpy()


def f16vec(raw):
    return np.frombuffer(bytes(raw), np.float16).astype(np.float32)


def unswizzle(raw, K, N):
    """mma B 片段序 FP8 字节 -> (K, N) 矩阵。每 512B = 32 线程 x 16B = 两个 32(K)x8(N) 块，块按 (k 块, n 块) 行优先；
    线程 (g=L>>2, t=L&3) 的第 jj 字节: k = 4t + jj%4 + 16*(jj//4)，n = g"""
    v = E4M3[np.frombuffer(bytes(raw), np.uint8, count=K * N)]
    ntile, ktile = N // 8, K // 32
    kt = np.arange(ktile)[:, None, None, None]
    nt = np.arange(ntile)[None, :, None, None]
    lane = np.arange(32)[None, None, :, None]
    jj = np.arange(8)[None, None, None, :]
    ti = kt * ntile + nt
    src = (ti // 2) * 512 + lane * 16 + (ti % 2) * 8 + jj
    k = kt * 32 + 4 * (lane & 3) + jj % 4 + 16 * (jj // 4)
    n = nt * 8 + (lane >> 2)
    src, k, n = np.broadcast_arrays(src, k, n)
    out = np.zeros((K, N), np.float32)
    out[k, n] = v[src]
    return out


def _token_of_pixel(py, px):
    return 16 * (2 * (py // 4) + px // 4) + 4 * (py % 4) + (px % 4)


def bias1h(raw):
    """1h-8h swin 的相对位置偏置 (单头 4096 个 f16，QK mma 累加器片段序) -> (64 查询, 64 键)，token = 窗口内行优先"""
    v = f16vec(raw)
    M = np.zeros((64, 64), np.float32)
    for ti in range(32):
        mt, nt = divmod(ti, 8)
        blk, half = divmod(ti, 2)
        for lane in range(32):
            g, t = lane >> 2, lane & 3
            base = blk * 256 + lane * 8 + half * 4
            for q, (r, c) in enumerate(((g, 2 * t), (g, 2 * t + 1), (g + 8, 2 * t), (g + 8, 2 * t + 1))):
                M[mt * 16 + r, nt * 8 + c] = v[base + q]
    order = [_token_of_pixel(p // 8, p % 8) for p in range(64)]
    return M[order][:, order]


def bias16(raw):
    """16h 偏置: 元素 e 的比特 -> 键 x = b0+2b3+4b8，键 y = b4+2b2+4b9，查询 x = b5+2b6+4b10，查询 y = b7+2b1+4b11"""
    v = f16vec(raw)
    e = np.arange(4096)
    b = [(e >> i) & 1 for i in range(12)]
    kx, ky = b[0] + 2 * b[3] + 4 * b[8], b[4] + 2 * b[2] + 4 * b[9]
    qx, qy = b[5] + 2 * b[6] + 4 * b[10], b[7] + 2 * b[1] + 4 * b[11]
    B = np.zeros((64, 64), np.float32)
    B[qy * 8 + qx, ky * 8 + kx] = v
    return B


def swin_layout(W):
    """2h-8h 标准 swin 记录的段偏移 (W = 64/128/256)"""
    heads = W // 32
    o = {"W1": 0, "W2": 4 * W * W}
    o["D"] = o["W2"] + 128 * W
    o["c1"] = o["D"] + W * W + 16
    o["qkv"] = o["c1"] + 2 * W + 16
    o["bias"] = o["qkv"] + 3 * W * W
    o["tau"] = o["bias"] + heads * 8192
    o["proj"] = o["tau"] + 16 * ((heads + 3) // 4)
    o["c2"] = o["proj"] + W * W
    o["end"] = o["c2"] + 2 * W + 16
    return o
