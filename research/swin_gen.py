"""通用 swin 块参考实现 (W = 64 / 128 / 256，头数 = W/32，头维 32)。

权重布局公式 (1h/2h/4h/8h 记录大小全部精确吻合):
  FFN:  W1  W x 4W   [0, 4W²)            warp w (共 W/32 个，每个 128 隐层) 的 K 块 kc、轮 j:
                                          w*(W/32)*4096 + kc*4096 + j*1024，每块 32x32
        W2  块对角    [4W², 4W²+128W)     warp w 轮 j: 4W² + w*4096 + j*1024 (32 隐层 -> 32)
        D   W x W     [4W²+128W, 5W²+128W) K 块主序
  c1 块 16B + W 个 f16 + 16B；qkv W x 3W (按头分块，每头 96 列 Q/K/V)；偏置 头数 x 8192；
  τ 16B x ceil(头数/4)；投影 W x W；c2 W 个 f16 + 16B
通道映射都是按 32 分组重复的同一族 (1h/2h 实测)。
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import block1_ref as R  # noqa: E402
from exp_ffn import mp_cubic_silu, perm_in, q8, unswizzle  # noqa: E402

_M1 = json.load(open(os.path.join(os.path.dirname(__file__), "maps_block1.json")))
CMAP32 = _M1["W1col_to_W2row"][:32]                                  # 隐层 32 块内重排
COLS32 = [0, 16, 2, 18, 4, 20, 6, 22, 1, 17, 3, 19, 5, 21, 7, 23,     # 输出片段通道 -> 参数下标 (32 块内)
          8, 24, 10, 26, 12, 28, 14, 30, 9, 25, 11, 27, 13, 29, 15, 31]


def layout(W):
    heads = W // 32
    o = {}
    o["W1"] = 0
    o["W2"] = 4 * W * W
    o["D"] = o["W2"] + 128 * W
    o["c1"] = o["D"] + W * W + 16
    o["qkv"] = o["c1"] + 2 * W + 16
    o["bias"] = o["qkv"] + 3 * W * W
    o["tau"] = o["bias"] + heads * 8192
    o["proj"] = o["tau"] + 16 * ((heads + 3) // 4)
    o["c2"] = o["proj"] + W * W
    o["end"] = o["c2"] + 2 * W + 16
    return o, heads


def group_map(pat, W):
    return [g * 32 + p for g in range(W // 32) for p in pat]


def canon_index(W):
    PI = [perm_in(c) for c in range(32)]
    cg = np.zeros(W, int)
    for gi in range(W // 32):
        for c in range(32):
            cg[gi * 32 + PI[c]] = gi * 32 + c
    return cg


def ffn(Xf, w, W):
    o, _ = layout(W)
    nw = W // 32
    Xc = Xf[:, canon_index(W)]
    zs = []
    for wp in range(nw):
        z = 0
        for j in range(4):
            W1 = np.vstack([unswizzle(w[wp * nw * 4096 + kc * 4096 + j * 1024:][:1024], 32, 32) for kc in range(nw)])
            a = q8(mp_cubic_silu(R.f16(Xc @ W1)))
            c0 = o["W2"] + wp * 4096 + j * 1024
            z = z + a @ unswizzle(w[c0:c0 + 1024], 32, 32)[CMAP32, :]
        zs.append(z)
    Z = q8(np.hstack(zs))
    D = unswizzle(w[o["D"]:o["D"] + W * W], W, W)
    return Z @ D[group_map(CMAP32, W)][:, group_map(COLS32, W)]


def attention(Yf, w, W, H, Wd, shift):
    o, heads = layout(W)
    Y = q8(Yf[:, canon_index(W)])
    Wqkv = unswizzle(w[o["qkv"]:o["qkv"] + 3 * W * W], W, 3 * W)
    oy, ox = -shift[0], -shift[1]
    ny, nx = (H + oy + 7) // 8, (Wd + ox + 7) // 8
    pad = np.zeros((ny * 8, nx * 8, W), np.float32)
    pad[oy:oy + H, ox:ox + Wd] = Y.reshape(H, Wd, W)
    Yw = pad.reshape(ny, 8, nx, 8, W).transpose(0, 2, 1, 3, 4).reshape(-1, 64, W)
    outs = []
    for h in range(heads):
        b = 96 * h
        qh, kh, vh = (R.f16(Yw @ Wqkv[:, b + s:b + s + 32]) for s in (0, 32, 64))
        tau = np.frombuffer(w[o["tau"] + 4 * h:o["tau"] + 4 * h + 4], np.float32)[0]
        qn = qh / np.sqrt(np.maximum((qh * qh).sum(-1, keepdims=True), R.EPS)) * tau
        kn = kh / np.sqrt(np.maximum((kh * kh).sum(-1, keepdims=True), R.EPS))
        fake = bytearray(20672)
        fake[11360:19552] = w[o["bias"] + h * 8192:o["bias"] + (h + 1) * 8192]
        L = R.f16(q8(qn) @ q8(kn).transpose(0, 2, 1) + R.bias_matrix(bytes(fake)))
        e = np.exp(L - L.max(-1, keepdims=True))
        outs.append(R.f16(q8(e / e.sum(-1, keepdims=True)) @ q8(vh)))
    O = np.concatenate(outs, -1)
    O = O.reshape(ny, nx, 8, 8, W).transpose(0, 2, 1, 3, 4).reshape(ny * 8, nx * 8, W)[oy:oy + H, ox:ox + Wd]
    return O.reshape(-1, W)


def block(Xf, w, W, H, Wd, shift=(0, 0)):
    """Xf: (H*Wd, W) 片段序 -> 片段序输出 (fp8 值)。shift = (y, x)"""
    o, _ = layout(W)
    assert len(w) == o["end"], (len(w), o["end"])
    cols = group_map(COLS32, W)
    c1 = np.frombuffer(w[o["c1"]:o["c1"] + 2 * W], np.float16).astype(np.float32)
    c2 = np.frombuffer(w[o["c2"]:o["c2"] + 2 * W], np.float16).astype(np.float32)
    Yf = R.f16(c1[cols] * Xf + ffn(Xf, w, W))
    O = attention(Yf, w, W, H, Wd, shift)
    Pm = unswizzle(w[o["proj"]:o["proj"] + W * W], W, W)[group_map(CMAP32, W)][:, cols]
    return q8(R.f16(c2[cols] * Yf + O @ Pm))


def shift_of(params_word):
    x = params_word & 0xFFFFFFFF
    y = params_word >> 32
    s = lambda v: v - (1 << 32) if v >= 1 << 31 else v
    return (s(y), s(x))


def block_up8(low_img, skip_f, w, H, Wd, shift=(0, 0)):
    """8h 解码上采样块 (block48，记录 820784B) —— 实测结构，与 1h 上采样块的写法不同:
      记录: FFN [0,360448) | W_up 512x256 [360448,491520) | s [491520,492032) | c1 [492032,492544) | qkv 492544 | 偏置 | τ 754688 | 投影 754720 | c2 820256 | 16B
      x = s⊙skip，t = c1⊙x + s⊙nn_up2x(low规范序·W_up)，y = t + FFN(t)，然后标准注意力与 c2 残差。
    low_img: (h*w, 2W) 规范序 (上一级 outview 的图像格式)，skip_f: (H*Wd, W) 片段序"""
    W = 256
    w = bytes(w)
    cols = np.array(group_map(COLS32, W))
    s = np.frombuffer(w[491520:492032], np.float16).astype(np.float32)[cols]
    c1 = np.frombuffer(w[492032:492544], np.float16).astype(np.float32)[cols]
    std = w[:360448] + bytes(16) + w[492032:492544] + bytes(16) + w[492544:820768] + bytes(16)
    h, wd = H // 2, Wd // 2
    up = low_img.reshape(h, wd, 2 * W).repeat(2, 0).repeat(2, 1).reshape(H * Wd, 2 * W)
    U = R.f16(up @ unswizzle(w[360448:491520], 2 * W, W))[:, cols]
    t = R.f16(c1 * R.f16(s * skip_f) + R.f16(s * U))
    Yf = R.f16(t + ffn(t, std, W))
    o, _ = layout(W)
    c2 = np.frombuffer(std[o["c2"]:o["c2"] + 2 * W], np.float16).astype(np.float32)
    O = attention(Yf, std, W, H, Wd, shift)
    Pm = unswizzle(std[o["proj"]:o["proj"] + W * W], W, W)[group_map(CMAP32, W)][:, cols]
    return q8(R.f16(c2[cols] * Yf + O @ Pm))
