"""16h qkv 核 (FusedSwin2dQKVAttn 512, 16 头): 记录 = qkv 512x1536 [0,786432) | 偏置 16x8192 | τ 16xf32 [917504,917568)。
输出 = 各头注意力结果拼接 (尚未投影)，写成 tin。窗口 8x8，平移在 (0,0)/(-4,-4)/x-4/y-4 间轮换 (参数 +32: x 低 32 位, y 高 32 位)。"""
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import block1_ref as R  # noqa: E402
import k16  # noqa: E402
import klab  # noqa: E402
import krun  # noqa: E402
import swin_gen as G  # noqa: E402
from act import chw_to_tin, random_fp8, tin_to_chw  # noqa: E402
from exp_ffn import q8, unswizzle  # noqa: E402

W, HEADS = 512, 16
QKV, BIAS, TAU = 0, 786432, 917504


def shift(seq):
    b = klab.trace_row(seq, k16.TRACE)["params"]
    x, y = struct.unpack_from("<2i", b, 32)
    return (y, x)


def weights(seq):
    krun.TRACE = k16.TRACE
    return krun.weight_of(seq)[2]


def run(Y, w, seq):
    p0, p8 = k16.ptr(seq, 0), k16.ptr(seq, 8)
    r = k16.run(seq, {p0: chw_to_tin(Y), p8: 122880}, bytes(w))[p8]
    return tin_to_chw(np.frombuffer(r, np.uint8), 512, 12, 20)


def bias16(raw):
    """16h 偏置 (单头 4096 个 f16) -> (64 查询, 64 键)，token = y*8+x (窗口内行优先)。
    元素下标 e 的比特 (单元素探针实测): 键 x = b0 + 2*b3 + 4*b8，键 y = b4 + 2*b2 + 4*b9，
    查询 x = b5 + 2*b6 + 4*b10，查询 y = b7 + 2*b1 + 4*b11"""
    v = np.frombuffer(raw, np.float16).astype(np.float32)
    e = np.arange(4096)
    b = [(e >> i) & 1 for i in range(12)]
    kx, ky = b[0] + 2 * b[3] + 4 * b[8], b[4] + 2 * b[2] + 4 * b[9]
    qx, qy = b[5] + 2 * b[6] + 4 * b[10], b[7] + 2 * b[1] + 4 * b[11]
    B = np.zeros((64, 64), np.float32)
    B[qy * 8 + qx, ky * 8 + kx] = v
    return B


C_MUL, C_ADD, C_LO, C_HI = (np.float16(v) for v in (0.04491037502884865, 1.3008946180343628, 1.03125, 1.5693359375))


def exp_fp(L):
    """kernel 的 softmax 分子 (原始 PTX 4930-4960): y = clamp(fma16(L, 0.0449, 1.3009), 1.03125, 1.5693)，
    把 y 的 f16 位左移 5 位 (尾数高 5 位变成指数) -> p ≈ 2^(32(y-1)-15) ≈ 0.024·e^L，L 实际被截断在 [-6, 6]。
    不减行最大值 -> 不具备平移不变性"""
    y = np.clip((L.astype(np.float16) * C_MUL + C_ADD).astype(np.float16), C_LO, C_HI)
    bits = (y.view(np.uint16).astype(np.uint32) << 5) & 0xFFFF
    return (bits ^ 0x8000).astype(np.uint16).view(np.float16).astype(np.float32)


def softmax_fp(L):
    p = exp_fp(L)
    s = p.astype(np.float16).sum(-1, keepdims=True, dtype=np.float16).astype(np.float32)
    inv = R.f16(1.0 / np.maximum(s, 6.1e-5))
    return R.f16(p * inv)


def attention(Yf, w, shift, H=12, Wd=20):
    """Yf (N,512) 片段序 -> 各头输出拼接 (N,512)，头内通道为 swin_gen 的顺序"""
    Y = q8(Yf[:, G.canon_index(W)])
    Wqkv = unswizzle(bytes(w[QKV:QKV + 3 * W * W]), W, 3 * W)
    oy, ox = -shift[0], -shift[1]
    ny, nx = (H + oy + 7) // 8, (Wd + ox + 7) // 8
    pad = np.zeros((ny * 8, nx * 8, W), np.float32)
    pad[oy:oy + H, ox:ox + Wd] = Y.reshape(H, Wd, W)
    Yw = pad.reshape(ny, 8, nx, 8, W).transpose(0, 2, 1, 3, 4).reshape(-1, 64, W)
    outs = []
    for h in range(HEADS):
        b = 96 * h
        qh, kh, vh = (R.f16(Yw @ Wqkv[:, b + s:b + s + 32]) for s in (0, 32, 64))
        tau = np.frombuffer(bytes(w[TAU + 4 * h:TAU + 4 * h + 4]), np.float32)[0]
        qn = qh / np.sqrt(np.maximum((qh * qh).sum(-1, keepdims=True), R.EPS)) * tau
        kn = kh / np.sqrt(np.maximum((kh * kh).sum(-1, keepdims=True), R.EPS))
        L = R.f16(q8(qn) @ q8(kn).transpose(0, 2, 1) + bias16(bytes(w[BIAS + h * 8192:BIAS + (h + 1) * 8192])))
        outs.append(R.f16(q8(softmax_fp(L)) @ q8(vh)))                  # 16h: 先归一化再 PV (与 ViT 相反，实测)
    O = np.concatenate(outs, -1)
    O = O.reshape(ny, nx, 8, 8, W).transpose(0, 2, 1, 3, 4).reshape(ny * 8, nx * 8, W)[oy:oy + H, ox:ox + Wd]
    return O.reshape(-1, W)


if __name__ == "__main__":
    seq = int(sys.argv[1]) if len(sys.argv) > 1 else 26
    w = weights(seq)
    print("seq", seq, "平移", shift(seq), "记录", len(w))
    Yin = random_fp8((512, 12, 20), lo=-2, hi=2, seed=21)
    k = run(Yin, w, seq).reshape(512, -1).T
    r = q8(attention(Yin.reshape(512, -1).T, w, shift(seq)))[:, G.group_map(G.COLS32, W)]   # 输出序 = COLS32 (实测)
    print("相关", np.corrcoef(r.ravel(), k.ravel())[0, 1].round(5), " 逐值精确", (r == k).mean().round(4))
