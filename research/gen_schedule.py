"""从执行顺序 + 一次运行的轨迹导出整网调度表 torch/dlss5/data/schedule.json (纯数据，torch 版不再依赖轨迹/klab)。
每步: 块类型、所用权重记录名、窗口平移 (y, x)。平移是网络的静态属性 (与帧/分辨率无关)。"""
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(__file__))
import klab  # noqa: E402
import net_ref as N  # noqa: E402

TRACE = os.path.join(os.path.dirname(__file__), "tapsnet", "nr-trace.tsv")
OUT = os.path.join(N.ROOT, "torch", "dlss5", "data", "schedule.json")


def main():
    E = N.EXEC
    steps = []
    i = 0
    while i < len(E):
        L = E[i]
        k, wn, seq = L["kernel"], L["weights"], L["seq"]
        p = klab.trace_row(seq, TRACE)["params"] if wn else None
        if "pre_block" in k:
            steps.append({"kind": "pre", "record": wn[0]})
        elif "post_block" in k:
            steps.append({"kind": "post", "record": wn[0], "blend_record": wn[1]})
        elif any(f"_{lv}_" in k for lv in N.LEVEL):
            lv = next(lv for lv in N.LEVEL if f"_{lv}_" in k)
            _, H, W = N.LEVEL[lv]
            var = ("inpview" if "inpview" in k else "ds" if "ds_wait" in k else "up" if "upsample" in k
                   else "outview" if "outview" in k else "std")
            steps.append({"kind": "swin", "level": lv, "variant": var, "record": wn[0], "shift": list(N.shift_of(p, H, W))})
        elif "split_swin_16h" in k:
            recs = [E[i + j]["weights"][0] for j in range(4)]
            tail = E[i + 3]["kernel"]
            st = {"kind": "split16", "records": recs, "inpview": "inpview" in k,
                  "shift": list(N.shift16(klab.trace_row(E[i + 2]["seq"], TRACE)["params"])),
                  "tail": "pool" if "pool" in tail else "outview" if "outview" in tail else None}
            if st["tail"] == "pool":
                st["head_record"] = E[i + 4]["weights"][0]
                i += 1
            steps.append(st)
            i += 3
        elif "vit_1d_ffn_expand" in k:
            steps.append({"kind": "vit", "records": [E[i + j]["weights"][0] for j in range(5)]})
            i += 4
        elif "dec_input_upsample" in k:
            steps.append({"kind": "block39", "record": wn[0]})
        i += 1
    json.dump({"source": "exec_order.json + tapsnet/nr-trace.tsv", "steps": steps}, open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    kinds = {}
    for s in steps:
        kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
    print(OUT, len(steps), "步", kinds)


if __name__ == "__main__":
    main()
