"""cg2r_post_process_kernel 单独运行，与 torch/dlss5/style.grade 逐分支对照。
参数块取自 nr-lab --style 1 的轨迹 (harness/rt_sm86/nr-trace.tsv)，换上自己的贴图与调色参数。
    python pp_check.py"""
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "torch"))
import klab  # noqa: E402
from dlss5 import style  # noqa: E402

TRACE = os.path.join(klab.ROOT, "harness", "rt_sm86", "nr-trace.tsv")
KEYS = ["black", "white", "exposure", "gamma", "contrast", "saturation", "vibrance", "warm", "tint"]   # +316 起


def row():
    for line in open(TRACE):
        r = line.rstrip("\n").split("\t")
        if "post_process" in r[2]:
            return dict(kernel=r[2], grid=tuple(map(int, r[3].split(","))), block=tuple(map(int, r[4].split(","))),
                        smem=int(r[5]), params=bytes.fromhex(r[7]))
    raise SystemExit("轨迹里没有 post_process: 先跑 nr-lab --style 1 (DLSS5_PROF=1)")


def run(src, base, g, strength=1.0):
    R = row()
    H, W = src.shape[:2]
    p = bytearray(R["params"])
    ts, keep_s = klab.texture(src, linear=False)
    tb, keep_b = klab.texture(base, linear=False)
    out = klab.Surface(W, H)
    struct.pack_into("<Q", p, 0, out.handle)
    struct.pack_into("<Q", p, 32, ts)
    struct.pack_into("<Q", p, 64, tb)
    struct.pack_into("<f", p, 288, strength)
    for i, k in enumerate(KEYS):
        struct.pack_into("<f", p, 316 + 4 * i, g.get(k, 1.0 if k == "white" else 0.0))
    struct.pack_into("<5f", p, 352, *g.get("zones", (0.0,) * 5))
    klab.launch(klab.function("14", R["kernel"]), R["grid"], R["block"], R["smem"], bytes(p))
    torch.cuda.synchronize()
    return out.read()[..., :3]


def main():
    rng = np.random.default_rng(0)
    H, W = 360, 640
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    src = np.stack([xx / W, yy / H, 0.5 + 0.5 * np.sin(xx / 23.0) * np.cos(yy / 17.0)], -1).astype(np.float32)
    src[: H // 4] = rng.random((H // 4, W, 3), dtype=np.float32)          # 一块随机色，覆盖各色相
    base = rng.random((H, W, 3), dtype=np.float32)
    cases = {
        "style1": style.STYLES[1], "style2": style.STYLES[2],
        "black/white": dict(black=0.1, white=0.85),
        "exposure+": dict(exposure=0.4), "contrast+": dict(contrast=0.6),
        "warm+": dict(warm=0.4), "warm-": dict(warm=-0.4), "tint+": dict(tint=0.3), "tint-": dict(tint=-0.3),
        "zones": dict(zones=(0.3, -0.2, 0.15, -0.25, 0.2)), "gamma": dict(gamma=0.35),
        "saturation+": dict(saturation=0.5), "vibrance": dict(vibrance=0.6), "vibrance-": dict(vibrance=-0.4),
        "all": dict(black=0.05, white=0.95, exposure=-0.2, gamma=0.1, contrast=0.3, saturation=0.2, vibrance=0.3,
                    warm=0.2, tint=-0.1, zones=(0.1, 0.0, -0.1, 0.05, 0.0)),
    }
    for name, g in cases.items():
        k = run(src, base, g)
        t = style.grade(torch.from_numpy(src).cuda(), **{"zones": (0.0,) * 5, **g}).cpu().numpy()
        d = np.abs(k - t)
        print(f"{name:12s} 最大差 {d.max() * 255:6.3f}/255  平均差 {d.mean() * 255:.4f}/255  (调色改动 {np.abs(k - src).mean() * 255:5.2f}/255)")
    k = run(src, base, style.STYLES[1], strength=0.6)
    t = style.apply(torch.from_numpy(base).cuda(), torch.from_numpy(src).cuda(), 1, 0.6).cpu().numpy()
    print(f"强度 0.6     最大差 {np.abs(k - t).max() * 255:6.3f}/255")


if __name__ == "__main__":
    main()
