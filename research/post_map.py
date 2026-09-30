"""post_block 输出卷积 (f16 32->16 [20784,21808)) 排布: 旁路 swin 主体后, 每个输入通道 one-hot -> 3 个输出通道响应。"""
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import klab  # noqa: E402
import krun  # noqa: E402
import post_probe as PP  # noqa: E402
import post_ref as P  # noqa: E402
from act import E4M3, chw_to_tin  # noqa: E402

CODE1 = int(np.nanargmin(np.abs(E4M3 - 1.0)))   # 表里有 NaN (0x7f)，不能用 argmin


def bypass(w):
    wb = bytearray(w)
    wb[4096:8192] = bytes(4096)
    wb[8208:8272] = np.ones(32, np.float16).tobytes()
    wb[19680:20704] = bytes(1024)
    wb[20704:20768] = np.ones(32, np.float16).tobytes()
    return wb


def pix(c):
    return 8 + 16 * (c // 8), 8 + 16 * (c % 8)       # 半分辨率像素，各通道落在不同窗口


def net(wb, xb, sb):
    row = klab.trace_row(PP.SEQ, PP.TRACE)
    skip_va = struct.unpack_from("<Q", row["params"], 8)[0]
    zero = np.zeros((PP.H, PP.W, 3), np.float32)
    o = P.run_with_skip(zero, zero, np.zeros((PP.H, PP.W, 2), np.float32), bytes(wb), xb, sb, skip_va,
                        [(52, "<i", 0), (56, "<Q", 0)])
    return o.transpose(2, 0, 1)[:3] / 0.03125


def x_onehot():
    xb = np.zeros(2 * 192 * 320 * 16, np.uint8)
    for c in range(32):
        y, x = pix(c)
        xb[(((c // 16) * 192 + y) * 320 + x) * 16 + c % 16] = CODE1
    return xb.tobytes()


def s_onehot():
    s = np.zeros((32, 384, 640), np.float32)
    for c in range(32):
        y, x = pix(c)
        s[c, 2 * y, 2 * x] = 1.0
    return chw_to_tin(s)


def responses(wb):
    zx, zs = bytes(2 * 192 * 320 * 16), chw_to_tin(np.zeros((32, 384, 640), np.float32))
    n0 = net(wb, zx, zs)
    nx, ns = net(wb, x_onehot(), zs) - n0, net(wb, zx, s_onehot()) - n0
    Rx = np.array([nx[:, 2 * pix(c)[0], 2 * pix(c)[1]] for c in range(32)])
    Rs = np.array([ns[:, 2 * pix(c)[0], 2 * pix(c)[1]] for c in range(32)])
    return Rx, Rs


def main():
    krun.TRACE = PP.TRACE
    _, _, w = krun.weight_of(PP.SEQ)
    wb = bypass(w)
    Rx, Rs = responses(wb)
    f = lambda a, b: np.frombuffer(bytes(w[a:b]), np.float16).astype(np.float32)  # noqa: E731
    s1, s2 = f(8272, 8336), f(8336, 8400)
    print("Rx 非零通道", (np.abs(Rx).max(1) > 0).sum(), " Rs 非零通道", (np.abs(Rs).max(1) > 0).sum())
    # s1=s2=1 再测一遍，得到纯输出卷积 (输入通道 -> 输出)
    wb2 = bytearray(wb)
    wb2[8272:8400] = np.ones(64, np.float16).tobytes()
    Ux, Us = responses(wb2)
    print("s=1: x 与 skip 的纯卷积一致?", np.abs(Ux - Us).max())
    np.savez(os.path.join(os.path.dirname(__file__), "post_map.npz"), Rx=Rx, Rs=Rs, Ux=Ux, Us=Us, s1=s1, s2=s2)
    ratio_x = Rx / np.where(Ux == 0, np.nan, Ux)
    ratio_s = Rs / np.where(Us == 0, np.nan, Us)
    print("Rx/Ux (每通道应为常数 = s1[?])", np.nanstd(ratio_x, 1).max())
    print("Rs/Us", np.nanstd(ratio_s, 1).max())
    print("s1 值集合匹配:", np.sort(np.nanmean(ratio_x, 1))[:6], np.sort(s1)[:6])
    print("s2 值集合匹配:", np.sort(np.nanmean(ratio_s, 1))[:6], np.sort(s2)[:6])


if __name__ == "__main__":
    main()
