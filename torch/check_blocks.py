"""逐块隔离: 4h/8h 每块以 kernel 抓取的上一块输出为输入，只测该块 (相关 / 逐值精确 / 幅度比 kernel÷torch)。
需要 research/tapsblk 与 research/tapsnet 的抓取数据。"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dlss5 import DLSS5, ops  # noqa: E402
ops.PRECISE = True                     # 逐块对照: 模拟 kernel 的每处舍入
from dlss5.net import LEVEL  # noqa: E402

R = os.path.join(HERE, "..", "research")
E4M3 = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float().numpy()


def raw(seq, size):
    """第 seq 次发射后抓取的、大小不小于 size 的缓冲 (取最接近的一个，截到 size)"""
    best = None
    for d in ("tapsblk", "tapsdecb", "tapsnet", "taps8t", "taps"):
        dd = os.path.join(R, d)
        for f in sorted(os.listdir(dd)):
            n = os.path.getsize(os.path.join(dd, f))
            if f.startswith(f"tap_s{seq}_") and n >= size and (best is None or n < best[0]):
                best = (n, os.path.join(dd, f))
    return None if best is None else np.frombuffer(open(best[1], "rb").read(), np.uint8)[:size]


def tin(seq, C, H, W):
    """tin 缓冲 -> (H*W, C) 片段序 (向量化)"""
    v = E4M3[raw(seq, C * H * W)].reshape(H // 4, W // 4, C // 32, 32, 16)
    lane, b = np.meshgrid(np.arange(32), np.arange(16), indexing="ij")
    g, t = lane >> 2, lane & 3
    tok = g + 8 * ((b >> 2) % 2)
    ch = 8 * (b % 4) + 2 * t + (b >> 3)
    out = np.zeros((H // 4, W // 4, 4, 4, C // 32, 32), np.float32)
    out[:, :, tok // 4, tok % 4, :, ch] = v.transpose(3, 4, 0, 1, 2)          # 不相邻的高级索引维度排在最前
    return out.transpose(0, 2, 1, 3, 4, 5).reshape(H * W, C)


def img(seq, C, H, W):
    return E4M3[raw(seq, C * H * W)].reshape(C // 16, H, W, 16).transpose(1, 2, 0, 3).reshape(H * W, C)


def report(name, r, k):
    r, k = r.cpu().numpy() if torch.is_tensor(r) else r, k
    a = float((r * k).sum() / (r * r).sum())
    print(f"{name}: 相关 {np.corrcoef(r.ravel(), k.ravel())[0, 1]:.5f}  精确 {(r == k).mean():.4f}  幅度比 {a:.4f}")
    return a


def run(net, levels=("4h", "8h")):
    dev = "cuda"
    T = lambda a: torch.tensor(a, device=dev)  # noqa: E731
    res = {}
    for i, (st, m) in enumerate(net.steps):
        if st["kind"] != "swin" or st["level"] not in levels or i > 22:
            continue
        W, H, Wd = LEVEL[st["level"]]
        if raw(i + 1, W * H * Wd) is None or raw(i, W * H * Wd if st["variant"] != "inpview" else W * H * Wd // 2) is None:
            continue                                             # 缺抓取数据的块跳过
        seq = i + 1
        W, H, Wd = LEVEL[st["level"]]
        sh, var = tuple(st["shift"]), st["variant"]
        if var == "inpview":
            x = T(img(seq - 1, W, H, Wd))
            r = m(x[:, m.fi], H, Wd, sh)
        else:
            r = m(T(tin(seq - 1, W, H, Wd)), H, Wd, sh)
            if var == "ds":
                r, ds = r
                res[f"{seq}ds"] = report(f"seq{seq} {st['level']} 下采样", ds, img(seq, 2 * W, H // 2, Wd // 2))
        res[seq] = report(f"seq{seq} {st['level']} {var:7s}", r, tin(seq, W, H, Wd))
    return res


def seq_of_records():
    """权重记录名 -> kernel 序号 (exec_order.json)"""
    import json
    E = json.load(open(os.path.join(HERE, "dlss5", "data", "exec_order.json")))["launches"]
    return {w: L["seq"] for L in E for w in L["weights"]}


SKIP_SEQ = {"1h": 5, "2h": 9, "4h": 15, "8h": 23}      # 编码出口 (跳连) 的 kernel 序号


def run_decoder(net, levels=("8h", "4h", "2h", "1h")):
    """解码段逐块: 上采样入口 = (低一级出口图像, 同级编码跳连)，其余以 kernel 上一块输出为输入"""
    T = lambda a: torch.tensor(a, device="cuda")  # noqa: E731
    sq = seq_of_records()
    res = {}
    for st, m in net.steps:
        if st["kind"] != "swin" or st["level"] not in levels or st["variant"] not in ("up", "std", "outview"):
            continue
        seq = sq[st["record"]]
        if seq < 100:
            continue                                         # 编码段
        W, H, Wd = LEVEL[st["level"]]
        sh = tuple(st["shift"])
        if st["variant"] == "up":
            if raw(seq - 1, 2 * W * H * Wd // 4) is None or raw(SKIP_SEQ[st["level"]], W * H * Wd) is None:
                continue
            r = m(T(img(seq - 1, 2 * W, H // 2, Wd // 2)), T(tin(SKIP_SEQ[st["level"]], W, H, Wd)), H, Wd, sh)
            k = tin(seq, W, H, Wd)
        else:
            if raw(seq - 1, W * H * Wd) is None or raw(seq, W * H * Wd) is None:
                continue
            r = m(T(tin(seq - 1, W, H, Wd)), H, Wd, sh)
            if st["variant"] == "outview":
                r, k = r[:, m.ci], img(seq, W, H, Wd)
            else:
                k = tin(seq, W, H, Wd)
        res[seq] = report(f"seq{seq} {st['level']} {st['variant']:7s}", r, k)
    return res


if __name__ == "__main__":
    net = DLSS5(precise=True)
    run(net)
    run_decoder(net)
