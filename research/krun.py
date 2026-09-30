"""通用 kernel 运行器: 按执行轨迹里第 seq 次发射的参数，替换指针后在 Python 里单独运行。

run(seq, buffers={旧指针: bytes 或 大小(int, 表示分配清零的输出)}, weights=bytes 或 None)
  -> {旧指针: 运行后的 bytes}
权重指针自动识别 (落在权重区的那个)，weights=None 时用真实权重记录。
"""
import json
import os
import struct

import klab

ROOT = klab.ROOT
WB, WE = 0x9E00000, 0x12B00000
_RECS = sorted(json.load(open(os.path.join(ROOT, "weights_map.json"))), key=lambda r: r["name"])
_GPU = {}
_o = 0
for _r in _RECS:
    _GPU[WB + _o] = _r
    _o = (_o + _r["C"] + 511) // 512 * 512


TRACE = os.path.join(ROOT, "research", "taps", "nr-trace.tsv")   # 每次运行显存地址不同，窃听数据须配同一次运行的轨迹


_BASE = {}


def weight_base():
    """本轨迹的权重缓冲基址: pre_block (seq1) 用 block0，指针即基址；用 seq2 (block1 起点 = 基址 + 22016) 交叉确认"""
    if TRACE not in _BASE:
        p1 = klab.trace_row(1, TRACE)["params"]
        p2 = klab.trace_row(2, TRACE)["params"]
        w1 = {struct.unpack_from("<Q", p1, i)[0] for i in range(0, len(p1) // 8 * 8, 8)}
        w2 = {struct.unpack_from("<Q", p2, i)[0] for i in range(0, len(p2) // 8 * 8, 8)}
        block1_off = [o for o, r in _GPU.items() if r["name"] == "block1.layer0.layer"][0] - WB
        _BASE[TRACE] = next(v for v in w1 if v + block1_off in w2)
    return _BASE[TRACE]


def weight_of(seq):
    """该次发射用的权重记录 (名字, 字节)"""
    row = klab.trace_row(seq, TRACE)
    shift = weight_base() - WB
    for i in range(0, len(row["params"]) // 8 * 8, 8):
        v = struct.unpack_from("<Q", row["params"], i)[0] - shift
        if v in _GPU:
            r = _GPU[v]
            v += shift
            with open(os.path.join(ROOT, "WEIGHTS_HT.bin"), "rb") as f:
                f.seek(r["off"])
                return v, r["name"], f.read(r["C"])
    return None, None, None


def run(seq, buffers, weights=None, fatbin=None, counters=0x1BA00000, cinit=None):
    row = klab.trace_row(seq, TRACE)
    fb = fatbin or {"_1h_": "01", "pre_block": "01", "post_block": "01", "_2h_": "02", "_4h_": "03",
                    "_8h_": "04", "split_swin": "05", "vit_1d": "06", "dec_input": "07"}
    if isinstance(fb, dict):
        fb = next(v for k, v in fb.items() if k in row["kernel"])
    fn = klab.function(fb, row["kernel"])
    wptr, _, wreal = weight_of(seq)
    g = {}
    mapping = {}
    for old, data in buffers.items():
        g[old] = klab.gpu_bytes(b"", data) if isinstance(data, int) else klab.gpu_bytes(data)
        mapping[old] = g[old].data_ptr()
    if wptr is not None:
        gw = klab.gpu_bytes(weights if weights is not None else wreal, 512)
        mapping[wptr] = gw.data_ptr()
    # 计数器/同步标志区: 同一次运行里的所有指针按偏移映射到一块缓冲 (上一 kernel 的完成标志也在这里)
    gc = klab.gpu_bytes(cinit, 0) if cinit is not None else klab.gpu_bytes(b"", 1 << 20)
    for i in range(0, len(row["params"]) // 8 * 8, 8):
        v = struct.unpack_from("<Q", row["params"], i)[0]
        if counters <= v < counters + (1 << 20):
            mapping[v] = gc.data_ptr() + (v - counters)
    p = klab.patch_ptrs(row["params"], mapping)
    klab.launch(fn, row["grid"], row["block"], row["smem"], p)
    return {old: t.cpu().numpy().tobytes() for old, t in g.items()}
