"""ViT-1d 瓶颈 (block31-38): 1024 通道、96 token (8x12 网格按行展平，60 个真实 + 36 个补齐)、32 头。
1D 布局 = 每 16 个 token 一块，块内按 mma 片段 (token g+8r)，每 32 通道 512 字节 —— 等价于把 token 当 4x4 小块的 tin。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import klab  # noqa: E402
import krun  # noqa: E402
from act import chw_to_tin, tin_to_chw  # noqa: E402

TAPS = os.path.join(os.path.dirname(__file__), "tapsvit1")
TRACE = os.path.join(TAPS, "nr-trace.tsv")
N = 96


def from1d(raw, C, n=N):
    """1D 布局字节 -> (n, C)，通道为片段序"""
    x = tin_to_chw(np.frombuffer(raw, np.uint8)[: n * C], C, 4, n // 4)       # (C, 4, n/4)，tile = 列/4
    x = x.reshape(C, 4, n // 16, 4).transpose(2, 1, 3, 0)                     # (tile, py, px, C)
    return x.reshape(n, C)


def to1d(X):
    n, C = X.shape
    x = X.reshape(n // 16, 4, 4, C).transpose(3, 1, 0, 2).reshape(C, 4, n // 4)
    return chw_to_tin(x)


def tap(seq, addr=None):
    f = [f for f in os.listdir(TAPS) if f.startswith(f"tap_s{seq}_") and (addr is None or addr in f)]
    return open(os.path.join(TAPS, f[0]), "rb").read()


def ptr(seq, off):
    import struct
    return struct.unpack_from("<Q", klab.trace_row(seq, TRACE)["params"], off)[0]


def weights(seq):
    krun.TRACE = TRACE
    return krun.weight_of(seq)[2]


def run(seq, bufs, w=None, extra=1 << 20, split_sync=True):
    """未指定的激活/临时缓冲指针自动分配 extra 字节的零缓冲 (如 qkv 的 +48、contract/projection 的 +40)"""
    import struct
    import k16
    k16.TRACE = TRACE
    krun.TRACE = TRACE
    wptr = krun.weight_of(seq)[0]
    page = k16.counters(seq)
    p = klab.trace_row(seq, TRACE)["params"]
    bufs = dict(bufs)
    for i in range(0, len(p) // 8 * 8, 8):
        v = struct.unpack_from("<Q", p, i)[0]
        if v > 0x10000000 and v < 0x80000000 and v not in bufs and v != wptr and not (page <= v < page + (1 << 20)):
            bufs[v] = extra
    return k16.run(seq, bufs, w, split_sync=split_sync)


def _bits(v, spec):
    """spec: [(源比特值数组, 目标权重), ...] -> 字节偏移"""
    return sum(((v >> i) & 1) * wgt for i, wgt in spec)


def k_offsets():
    """K 缓冲 (qkv +16): (token n, 头 h, 维 d) -> 字节。16 token 一块 (16384B)，块内: 头*512，
    n0/n1/n2 -> 64/128/256，n3 -> 8；d0 -> 1，d1 -> 16，d2 -> 32，d3 -> 2，d4 -> 4 (单元素探针)"""
    n, h, d = np.meshgrid(np.arange(N), np.arange(32), np.arange(32), indexing="ij")
    return (n // 16) * 16384 + h * 512 + _bits(n % 16, [(0, 64), (1, 128), (2, 256), (3, 8)]) + \
        _bits(d, [(0, 1), (1, 16), (2, 32), (3, 2), (4, 4)])


def v_offsets():
    """V 缓冲 (qkv +24): 32 token 一块 (32768B)，块内: 头*1024，d4 -> 512，d0/d1/d2 -> 64/128/256，d3 -> 8；
    n0 -> 1，n1 -> 16，n2 -> 32，n3 -> 2，n4 -> 4"""
    n, h, d = np.meshgrid(np.arange(N), np.arange(32), np.arange(32), indexing="ij")
    return (n // 32) * 32768 + h * 1024 + _bits(d, [(0, 64), (1, 128), (2, 256), (3, 8), (4, 512)]) + \
        _bits(n % 32, [(0, 1), (1, 16), (2, 32), (3, 2), (4, 4)])


def read_kv(raw, offs):
    from act import E4M3
    return E4M3[np.frombuffer(raw, np.uint8)[offs]]          # (N, 32 头, 32 维)


def exp_trick(L, mul, add, lo, hi, shift, addc):
    """位运算 exp: y = clamp(fma16(L, mul, add), lo, hi)，结果 = f16 位 ((y_bits << shift) + addc) 的低 16 位"""
    f = np.float16
    y = np.clip((L.astype(f) * f(mul) + f(add)).astype(f), f(lo), f(hi))
    bits = ((y.view(np.uint16).astype(np.uint32) << shift) + (addc & 0xFFFF)) & 0xFFFF
    return bits.astype(np.uint16).view(np.float16).astype(np.float32)


VIT_EXP = (0.08953946828842163, 1.7093614339828491, 1.439453125, 1.9775390625, 4, 0x3FFC4000)   # p ≈ 0.084·e^L，L 截断 [-3, 3]


def attention(Q, K, V, keys=N):
    """Q (N,32,32) 已含 τ√32；K (N,32,32) 单位化；V (N,32,32)。返回 (N,32,32)"""
    import block1_ref as R
    from exp_ffn import q8
    out = np.zeros_like(Q)
    for h in range(32):
        L = R.f16(Q[:, h] @ K[:keys, h].T)
        p = exp_trick(L, *VIT_EXP)
        s = p.astype(np.float16).sum(-1, keepdims=True, dtype=np.float16).astype(np.float32)
        out[:, h] = R.f16(R.f16(q8(p) @ V[:keys, h]) * R.f16(1.0 / np.maximum(s, 6.1e-5)))   # 先 PV 再乘 1/Σp (实测)
    return out


def block(X, ws):
    """ViT-1d 块 (block31-38) 参考: X (96, 1024) 片段序 (1D 布局解码)，ws = 5 个记录字节 (expand/contract/qkv/attn/proj)。
      H  = q8(act(x规范序 · W1))[:, COLS]                     W1 1024x4096 [0,4194304) + 16B 填充
      x1 = q8(c1⊙x + H·W2)                                     W2 4096x1024，c1 f16 在末尾
      Q  = q8(τ√32 · q/|q|)，K = q8(k/|k|)，V = q8(v)           qkv: τ f32x32 在记录开头 [0,128)，矩阵 [128,…) 每头 96 列
      O  = (q8(p)·V) · 1/Σp，p = 位运算 exp(Q·Kᵀ) (L 截断 [-3,3])，96 个键全参与 (含补齐 token)
      out= q8(c2⊙x1 + O·Wp)                                    attn 记录 (1 个 f16) 未被 kernel 读取"""
    import block1_ref as R
    import swin_gen as G
    from exp_ffn import mp_cubic_silu, q8, unswizzle
    C4, C1 = np.array(G.group_map(G.COLS32, 4096)), np.array(G.group_map(G.COLS32, 1024))
    CM4, CM1 = G.group_map(G.CMAP32, 4096), G.group_map(G.CMAP32, 1024)
    w1, w2, wq, _, wp = (bytes(w) for w in ws)
    Xc = X[:, G.canon_index(1024)]
    H = q8(mp_cubic_silu(R.f16(Xc @ unswizzle(w1[:4194304], 1024, 4096))))[:, C4]
    c1 = np.frombuffer(w2[4194304:], np.float16).astype(np.float32)
    x1 = q8(R.f16(c1[C1] * X + H[:, np.argsort(C4)] @ unswizzle(w2[:4194304], 4096, 1024)[CM4][:, C1]))
    tau = np.frombuffer(wq[:128], np.float32)
    A = R.f16(x1[:, G.canon_index(1024)] @ unswizzle(wq[128:], 1024, 3072)).reshape(N, 32, 96)
    q, k, v = A[:, :, :32], A[:, :, 32:64], A[:, :, 64:]
    nq = np.sqrt(np.maximum((q * q).sum(-1, keepdims=True), R.EPS))
    nk = np.sqrt(np.maximum((k * k).sum(-1, keepdims=True), R.EPS))
    Q = q8(R.f16(q / nq * tau[None, :, None] * np.sqrt(32)))
    O = q8(attention(Q, q8(k / nk), q8(v)).reshape(N, 1024))[:, C1]
    c2 = np.frombuffer(wp[1048576:], np.float16).astype(np.float32)
    return q8(R.f16(c2[C1] * x1 + O[:, np.argsort(C1)] @ unswizzle(wp[:1048576], 1024, 1024)[CM1][:, C1]))
