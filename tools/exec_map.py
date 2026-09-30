"""从 nr-lab 执行轨迹 (nr-trace.tsv) 恢复 "每次 kernel 发射 -> 用的哪条权重记录"。

显存里的权重缓冲 = WEIGHTS_HT 的 153 条记录按名字顺序排列，每条占 C 字节 (纯 f16 数据)、512 对齐。
参数块里落在权重缓冲范围内的 64 位值即权重指针。实测 153 个不同指针与 153 个记录起点逐一吻合。

python -m tools.exec_map harness/rt_sm86/nr-trace.tsv [--out torch/dlss5/data/exec_order.json]
"""
import argparse
import json
import os
import struct

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def record_layout(recs, align=512):
    starts, o = {}, 0
    for r in sorted(recs, key=lambda r: r["name"]):
        starts[o] = r["name"]
        o = (o + r["C"] + align - 1) // align * align
    return starts, o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--out", default=os.path.join(ROOT, "torch", "dlss5", "data", "exec_order.json"))
    a = ap.parse_args()
    recs = json.load(open(os.path.join(ROOT, "weights_map.json")))
    starts, total = record_layout(recs)
    rows = [l.rstrip("\n").split("\t") for l in open(a.trace)]
    # 权重基址 = 所有 64 位字里、能让最多值落在记录起点上的那个 (pre_block 的第一个指针即基址)
    words = []
    for r in rows:
        b = bytes.fromhex(r[7])
        words.append([struct.unpack_from("<Q", b, j)[0] for j in range(0, len(b) // 8 * 8, 8)])
    base = min(v for ws in words for v in ws if 0x1000000 <= v < 1 << 40 and any(
        (v2 - v) in starts for v2 in ws))
    order, used = [], set()
    for (seq, chain, name, grid, block, smem, psize, _), ws in zip(rows, words):
        w = [starts[v - base] for v in ws if base <= v < base + total and (v - base) in starts]
        used.update(w)
        order.append({"seq": int(seq), "kernel": name, "grid": grid, "block": block, "smem": int(smem),
                      "param_bytes": int(psize), "weights": w})
    unused = sorted(set(starts.values()) - used)
    json.dump({"weight_base": hex(base), "weight_bytes": total, "launches": order, "unused_records": unused},
              open(a.out, "w"), indent=1)
    print(f"{len(order)} 次发射, 用到 {len(used)}/{len(starts)} 条权重记录, 未用 {len(unused)} -> {a.out}")


if __name__ == "__main__":
    main()
