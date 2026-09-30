"""DLSS5 pre_block 的噪声输入 (PTX 逐条读出): 坐标 + 帧序号 -> PCG 哈希 -> 4 路均匀数 -> Box-Muller -> 3 个 N(0,1)。

  seed = (x*0x8DA6B343) ^ (frame*0x9E3779B9) ^ (y*0xD8163841) ^ 0x243F6A88
  h0   = rxs(seed) ^ (rxs(seed) >> 22)   rxs(s) = ((s >> ((s>>28)+4)) ^ s) * 277803737   (种子前无 LCG 步)
  u_i  = (rxs(h0*a_i + b_i) 取 (w>>30)^(w>>8) 的 24 位 + 1) * 2^-24，4 路 (a_i, b_i) 见 STREAMS
  n0 = √(-2 ln u0)·cos 2πu1，n1 = √(-2 ln u0)·sin 2πu1，n2 = √(-2 ln u2)·cos 2πu3
"""
import numpy as np

U = np.uint64
M32 = U(0xFFFFFFFF)
# PTX 里是有符号立即数 (-93469191 等)，按 & 0xFFFFFFFF 换算 (手算十六进制曾算错 3 个)
STREAMS = [(747796405, -1403630843 & 0xFFFFFFFF), (-93469191 & 0xFFFFFFFF, 1192405134),
           (-895109107 & 0xFFFFFFFF, 568162667), (-2094846927 & 0xFFFFFFFF, 878960812)]


def mul(a, b):
    return (a * U(b)) & M32


def rxs(s):
    return mul(((s >> ((s >> U(28)) + U(4))) ^ s), 277803737)


def uniform24(state):
    w = rxs(state)
    return (((w >> U(30)) ^ (w >> U(8))) & U(0xFFFFFF)).astype(np.float64) * 0 + \
        ((((w >> U(30)) ^ (w >> U(8))) & M32) + U(1)).astype(np.float64) * 2.0 ** -24


def noise(H, W, frame):
    y, x = np.mgrid[0:H, 0:W].astype(U)
    seed = mul(x, 0x8DA6B343) ^ (U(frame) * U(0x9E3779B9) & M32) ^ mul(y, 0xD8163841) ^ U(0x243F6A88)
    w = rxs(seed)                     # PTX 197-203: 种子直接做输出置换，前面没有 LCG 步
    h0 = (w >> U(22)) ^ w
    u = [uniform24((mul(h0, a) + U(b)) & M32) for a, b in STREAMS]
    r1 = np.sqrt(-2 * np.log(u[0]))
    r2 = np.sqrt(-2 * np.log(u[2]))
    t1, t2 = 2 * np.pi * u[1], 2 * np.pi * u[3]
    return np.stack([r1 * np.cos(t1), r1 * np.sin(t1), r2 * np.cos(t2)]).astype(np.float32)
