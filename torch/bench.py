"""按块类型统计 GPU 时间 (每步前后同步计时)"""
import collections
import time

import torch

from check import color_pattern
from dlss5 import DLSS5


def main(n=3):
    net = DLSS5()
    color = torch.tensor(color_pattern(), device="cuda")
    out = net(color)
    T = collections.defaultdict(float)
    orig = list(net.steps)

    def wrap(st, m):
        key = st["kind"] + ("/" + st["level"] if st["kind"] == "swin" else "")

        class W:
            def __getattr__(self, a):
                return getattr(m, a)

            def __call__(self, *a, **k):
                torch.cuda.synchronize()
                t = time.perf_counter()
                r = m(*a, **k)
                torch.cuda.synchronize()
                T[key] += time.perf_counter() - t
                return r
        return W()

    net.steps = [(st, wrap(st, m)) for st, m in orig]
    for _ in range(n):
        T.clear()
        net(color, hist=out, frame=1)
    net.steps = orig
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        net(color, hist=out, frame=1)
    torch.cuda.synchronize()
    print(f"整帧 (不插同步) {(time.perf_counter() - t) / n * 1000:.0f} ms")
    for k, v in sorted(T.items(), key=lambda kv: -kv[1]):
        print(f"  {k:10s} {v * 1000:6.1f} ms")


if __name__ == "__main__":
    main()
