"""控制参数落在哪: 默认参数跑 1 帧，再逐个改一个参数跑 1 帧，对比每次 kernel launch 的参数字节 (nr-trace.tsv)。
    python param_diff.py [WxH]
只报告差异 (launch 序号、kernel 名、偏移、前后值按 f32/i32 解读)。指针差异 (每次运行都变) 自动忽略。"""
import os
import struct
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
import klab  # noqa: E402

RT = os.path.join(klab.ROOT, "harness", "rt_sm86")
RES = sys.argv[1] if len(sys.argv) > 1 else "640x360"
CASES = {"local-tone": ["--local-tone", "0.25"], "local-structure": ["--local-structure", "0.25"],
         "skin-structure": ["--skin-structure", "0.5"], "intensity": ["--intensity", "0.25"],
         "style": ["--style", "1"], "auto-mask": ["--auto-mask", "0"]}


def run(extra):
    trace = os.path.join(RT, "nr-trace.tsv")
    if os.path.exists(trace):
        os.remove(trace)
    env = dict(os.environ, DLSS5_PROF="1", DLSS5_TRACE="0")
    subprocess.run([os.path.join(RT, "nr-lab.exe"), "--nr-only", "--input", RES, "--output", RES, "--frames", "1"] + extra,
                   cwd=RT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
    rows = {}
    for line in open(trace):
        r = line.rstrip("\n").split("\t")
        rows[int(r[0])] = (r[2], bytes.fromhex(r[7]))
    return rows


def ptr_like(v):
    return 0x10000000000 <= v < 0x800000000000


base = run([])
for name, extra in CASES.items():
    rows = run(extra)
    diffs = []
    for seq, (kern, p) in rows.items():
        if seq not in base or base[seq][1] == p:
            continue
        q = base[seq][1]
        for off in range(0, min(len(p), len(q)) - 3, 4):
            a, b = q[off:off + 4], p[off:off + 4]
            if a == b:
                continue
            if off % 8 == 0 and off + 8 <= len(p) and ptr_like(struct.unpack_from("<Q", q, off)[0]):
                continue
            if off % 8 == 4 and ptr_like(struct.unpack_from("<Q", q, off - 4)[0]):
                continue
            diffs.append((seq, kern[:40], off, struct.unpack("<f", a)[0], struct.unpack("<f", b)[0],
                          struct.unpack("<i", a)[0], struct.unpack("<i", b)[0]))
    print(f"== {name} {' '.join(extra)}: {len(diffs)} 处")
    for d in diffs[:12]:
        print("  seq%-3d %-40s +%-4d f32 %g -> %g   i32 %d -> %d" % d)
