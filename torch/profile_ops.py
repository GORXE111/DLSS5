"""torch.profiler 统计一帧里各类 CUDA kernel 的时间 (矩阵乘 / 逐元素 / 拷贝 ...)
    python profile_ops.py [H W]"""
import sys

import torch
from torch.profiler import ProfilerActivity, profile

from check import color_pattern
from dlss5 import DLSS5


def main():
    H, W = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (360, 640)
    net = DLSS5()
    import numpy as np
    color = torch.tensor(np.resize(color_pattern(), (H, W, 3)), device="cuda")
    out = net(color)
    for _ in range(2):
        net(color, hist=out, frame=1)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        net(color, hist=out, frame=1)
        torch.cuda.synchronize()
    groups = {}
    total = 0
    for e in prof.key_averages():
        t = e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total
        if t <= 0 or e.key.startswith("aten::") or e.key.startswith("cuda"):
            continue
        k = e.key.lower()
        g = ("矩阵乘" if any(s in k for s in ("gemm", "sgemm", "cutlass", "ampere_", "sm80_", "sm86_", "magma", "gemv"))
             else "fp8/f16 转换" if "copy" in k or "to_copy" in k or "cast" in k
             else "归约" if "reduce" in k
             else "索引/拼接" if any(s in k for s in ("index", "gather", "scatter", "cat"))
             else "逐元素")
        groups[g] = groups.get(g, 0) + t
        total += t
    for g, t in sorted(groups.items(), key=lambda kv: -kv[1]):
        print(f"{g:12s} {t / 1000:7.1f} ms  {t / total:5.1%}")
    print(f"合计 GPU kernel 时间 {total / 1000:.1f} ms")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))


if __name__ == "__main__":
    main()
