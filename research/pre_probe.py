"""pre_block (block0) 输入通道解剖: 纹理喂自造数据，Swin 主体短路，适配器换成已知满秩矩阵，最小二乘反解 16 个输入通道。

block0 布局 (21696B，PTX 读址): FFN [0,8192) | 16B | 适配器 f16 16x32 [8208,9232) (m16n8k16 的 B 操作数) |
16B + c1 [9232,9296) + 16B | qkv [9312,12384) | 偏置 [12384,20576) | τ [20576,20592) | 投影 [20592,21616) | c2 [21616,21680) | 16B
"""
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import klab  # noqa: E402
import krun  # noqa: E402
from act import tin_to_chw  # noqa: E402

TRACE = os.path.join(os.path.dirname(__file__), "tapsup", "nr-trace.tsv")
H, W = 360, 640
OH, OW = 192, 320


def adapter_bytes(A):
    """A: (16 输入, 32 输出)。适配器权重是 m16n8k16 的 **A 操作数** (输出通道为行, 输入通道为列; 像素作 B)，
    两个 m 块各 512B。线程 (g,t) 的第 j 个 f16: 行 = g + 8*((j//2)%2)，列 = 2t + j%2 + 8*(j//4)。
    (旧版按 B 操作数写，结果一列点亮 4 个输出 —— 探针实测 8 组 4x4 块结构)"""
    out = np.zeros(512, np.float16)
    for mt in range(2):
        for lane in range(32):
            g, t = lane >> 2, lane & 3
            for j in range(8):
                row = g + 8 * ((j // 2) % 2)
                col = 2 * t + j % 2 + 8 * (j // 4)
                out[mt * 256 + lane * 8 + j] = A[col, mt * 16 + row]
    return out.tobytes()


def bypass_weights(w, A):
    w = bytearray(w)
    w[4096:8192] = bytes(4096)                                                  # FFN 收缩 0
    w[8208:9232] = adapter_bytes(A)
    w[9232:9296] = np.ones(32, np.float16).tobytes()                            # c1 = 1
    w[20592:21616] = bytes(1024)                                                # 投影 0
    w[21616:21680] = np.ones(32, np.float16).tobytes()                          # c2 = 1
    return bytes(w)


def run(color, mv, hist, w, extra_param=None):
    krun.TRACE = TRACE
    row = klab.trace_row(1, TRACE)
    p = bytearray(row["params"])
    texs = [klab.texture(color), klab.texture(hist), klab.texture(mv, linear=False)]
    for off, (h, _) in zip((0, 8, 16), texs):
        struct.pack_into("<Q", p, off, h)
    if extra_param:
        for off, fmt, val in extra_param:
            struct.pack_into(fmt, p, off, val)
    fn = klab.function("01", row["kernel"])
    out = klab.gpu_bytes(b"", OH * OW * 32)
    wbuf = klab.gpu_bytes(w, 512)
    hist_feat = klab.gpu_bytes(b"", 16 << 20)
    cnt = klab.gpu_bytes(b"", 1 << 20)
    m = {struct.unpack_from("<Q", p, 248)[0]: out.data_ptr(), struct.unpack_from("<Q", p, 224)[0]: wbuf.data_ptr(),
         struct.unpack_from("<Q", p, 216)[0]: hist_feat.data_ptr()}
    for i in range(0, len(p) // 8 * 8, 8):
        v = struct.unpack_from("<Q", p, i)[0]
        if 0x1BA00000 <= v < 0x1BB00000 and v not in m:
            m[v] = cnt.data_ptr() + (v - 0x1BA00000)
    p = klab.patch_ptrs(bytes(p), m)
    klab.launch(fn, row["grid"], row["block"], row["smem"], p)
    # pre_block 输出是图像格式 [2][OH][OW][16] (即 block1 的输入格式)，不是 tin —— 按 tin 解会把 1 个通道读成 4 个
    from act import E4M3
    return E4M3[out.cpu().numpy()].reshape(2, OH, OW, 16).transpose(0, 3, 1, 2).reshape(32, OH, OW)


def recover(out, A, perm):
    """out (32,OH,OW) 片段序 -> 16 个输入通道: out[perm[n]] = Σ_k A[k,n] in[k]"""
    Y = out.reshape(32, -1)[perm]                         # 按 N 序
    X, *_ = np.linalg.lstsq(A.T, Y, rcond=None)
    return X.reshape(16, OH, OW)
