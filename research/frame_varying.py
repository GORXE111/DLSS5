"""Which kernel parameters change from frame to frame? (nr-trace.tsv written with DLSS5_TRACE=-2)
Compares launch k of frame f with launch k of frame f+1 and prints the differing 4-byte words.
    python frame_varying.py <nr-trace.tsv> [first frame]"""
import sys
from collections import defaultdict

path = sys.argv[1]
first = int(sys.argv[2]) if len(sys.argv) > 2 else 1
frames = defaultdict(list)
for line in open(path):
    c = line.rstrip("\n").split("\t")
    frames[int(c[0][1:])].append((c[3], bytes.fromhex(c[-1])))
fs = sorted(f for f in frames if f >= first)
summary = defaultdict(set)
for a, b in zip(fs, fs[1:]):
    A, B = frames[a], frames[b]
    if len(A) != len(B):
        print(f"frames {a},{b}: {len(A)} vs {len(B)} launches")
        continue
    for k, ((na, pa), (nb, pb)) in enumerate(zip(A, B)):
        if pa == pb:
            continue
        for o in range(0, min(len(pa), len(pb)) - 3, 4):
            wa, wb = int.from_bytes(pa[o:o + 4], "little"), int.from_bytes(pb[o:o + 4], "little")
            if wa != wb:
                summary[(k, na[:60], o)].add((a, wa))
                summary[(k, na[:60], o)].add((b, wb))
for (k, name, o), vals in sorted(summary.items()):
    v = " ".join(f"f{f}={w:#x}" for f, w in sorted(vals))
    print(f"launch {k:3d} +{o:<4d} {name:60s} {v}")
