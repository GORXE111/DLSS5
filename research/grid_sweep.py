"""补齐网格规则的黑盒扫描: 对一批 (W, H) 各跑一次 nr-lab (1 帧，只记轨迹)，读出 DLL 选的补齐网格与各级尺寸。
    python grid_sweep.py 1280x720 1344x720 ...        (或 --w 1024:1600:32 --h 720 这样的范围)
结果追加到 research/grid_sweep.jsonl"""
import json
import os
import struct
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
import klab  # noqa: E402

RT = os.path.join(klab.ROOT, "harness", "rt_sm86")
OUT = os.path.join(os.path.dirname(__file__), "grid_sweep.jsonl")


def probe(W, H):
    env = dict(os.environ, DLSS5_PROF="1", DLSS5_TRACE="0")
    trace = os.path.join(RT, "nr-trace.tsv")
    if os.path.exists(trace):
        os.remove(trace)
    subprocess.run([os.path.join(RT, "nr-lab.exe"), "--nr-only", "--input", f"{W}x{H}", "--output", f"{W}x{H}", "--frames", "1"],
                   cwd=RT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
    if not os.path.exists(trace):
        return None
    p1 = klab.trace_row(1, trace)["params"]
    row = {"W": W, "H": H, "in": struct.unpack_from("<2i", p1, 208), "grid": struct.unpack_from("<2i", p1, 240),
           "half": struct.unpack_from("<2i", p1, 256)}
    try:
        row["16h"] = struct.unpack_from("<2i", klab.trace_row(26, trace)["params"], 24)
        row["vit"] = struct.unpack_from("<2i", klab.trace_row(56, trace)["params"], 32)
    except Exception:
        pass
    return row


def parse(args):
    out = []
    ws, hs = None, None
    for i, a in enumerate(args):
        if a in ("--w", "--h"):
            lo, hi, st = map(int, args[i + 1].split(":")) if ":" in args[i + 1] else (int(args[i + 1]),) * 2 + (1,)
            rng = list(range(lo, hi + 1, st))
            if a == "--w":
                ws = rng
            else:
                hs = rng
        elif "x" in a and a[0].isdigit():
            w, h = map(int, a.split("x"))
            out.append((w, h))
    if ws and hs:
        out += [(w, h) for h in hs for w in ws]
    return out


if __name__ == "__main__":
    with open(OUT, "a", encoding="utf-8") as f:
        for W, H in parse(sys.argv[1:]):
            r = probe(W, H)
            if r:
                f.write(json.dumps(r) + "\n")
                f.flush()
                print(W, H, "->", r["grid"], r.get("16h"), r.get("vit"), flush=True)
            else:
                print(W, H, "-> 失败", flush=True)
