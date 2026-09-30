"""16h 瓶颈层 (512 通道, 20x12 token, 16 头) 的 kernel 单独运行工具。抓取数据: taps16 (同一次运行的轨迹)。"""
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import klab  # noqa: E402
import krun  # noqa: E402

TAPS = os.path.join(os.path.dirname(__file__), "taps16")
TRACE = os.path.join(TAPS, "nr-trace.tsv")
C, H, W = 512, 12, 20


def ptr(seq, off):
    return struct.unpack_from("<Q", klab.trace_row(seq, TRACE)["params"], off)[0]


def tap(seq):
    f = next(n for n in os.listdir(TAPS) if n.startswith(f"tap_s{seq}_"))
    return open(os.path.join(TAPS, f), "rb").read()


def counters(seq):
    """同步标志区 = 参数里落在权重之后 ~0x11C00000 处的指针所在的 1MB 页"""
    krun.TRACE = TRACE
    wb = krun.weight_base()
    row = klab.trace_row(seq, TRACE)["params"]
    vs = [struct.unpack_from("<Q", row, i)[0] for i in range(0, len(row) // 8 * 8, 8)]
    fl = [v for v in vs if wb + 0x11000000 < v < wb + 0x12000000]
    return fl[0] & ~0xFFFFF if fl else 0


def flags(seq):
    krun.TRACE = TRACE
    page = counters(seq)
    p = klab.trace_row(seq, TRACE)["params"]
    return page, {v for v in (struct.unpack_from("<Q", p, i)[0] for i in range(0, len(p) // 8 * 8, 8)) if page and page <= v < page + (1 << 20)}


def run(seq, buffers, weights=None, split_sync=False):
    """split_sync: 同步区初始化为 -1 (0xFF)，只把上一个 kernel 也用到的标志区 (= 输入就绪标志) 置 0。
    kernel 内 split-K 等跨 CTA 同步 (如 ViT qkv 的 z 两半) 需要它，否则后半不等待前半就开跑"""
    krun.TRACE = TRACE
    page, mine = flags(seq)
    cinit = None
    if split_sync and page:
        _, prev = flags(seq - 1)
        buf = bytearray([0xFF]) * (1 << 20)
        for v in mine & prev:
            buf[v - page:v - page + 512] = bytes(512)
        cinit = bytes(buf)
    return krun.run(seq, buffers, weights, counters=page, cinit=cinit)
