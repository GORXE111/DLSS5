"""逐级对照: 从显式 --capture manifest的pre输出起跑，不搜索旧tap目录。"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dlss5 import DLSS5, ops  # noqa: E402
from acceptance import execute, parser  # noqa: E402
from schedule_reference import ScheduleReference, arguments  # noqa: E402
ops.PRECISE = True                     # 逐块对照: 模拟 kernel 的每处舍入

R = os.path.join(HERE, "..", "research")
E4M3 = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float().numpy()
REFERENCE = None


def img(path, C, H, W):
    v = E4M3[np.frombuffer(open(path, "rb").read(), np.uint8)]
    return v.reshape(C // 16, H, W, 16).transpose(1, 2, 0, 3).reshape(H * W, C)


def slope(r, k):
    """kernel ≈ a·torch 的最小二乘斜率 (幅度比)"""
    return float((r * k).sum() / (r * r).sum())


def find(seq, C, H, W):
    if REFERENCE is None:
        raise ValueError("Initialize a ScheduleReference explicitly; legacy tap search is disabled")
    return str(REFERENCE.path(seq, C * H * W, shape=(H, W, C), layout="image16"))


# 调度步序号 -> (kernel 序号, 通道, H, W)
CHECK = {4: (5, 64, 96, 160), 8: (9, 128, 48, 80), 14: (15, 256, 24, 40), 22: (23, 512, 12, 20), 47: (131, 512, 12, 20),
         55: (139, 256, 24, 40), 61: (145, 128, 48, 80), 65: (149, 64, 96, 160), 69: (153, 32, 192, 320)}

def main(report):
    global REFERENCE
    REFERENCE = ScheduleReference(report.args.capture, report.args.harness)
    REFERENCE.annotate(report)
    paths = {i: find(seq, C, H, W) for i, (seq, C, H, W) in CHECK.items()}
    pre_path = find(1, 32, 192, 320)
    if pre_path is None:
        for i in CHECK:
            report.skip(f"step{i}", "Missing pre_block input tap")
        return
    report.reference(pre_path)
    for i, path in paths.items():
        if path is None:
            report.skip(f"step{i}", "Missing level output tap")
        else:
            report.reference(path)
    net = DLSS5(precise=True)
    pre = torch.tensor(img(pre_path, 32, 192, 320), device="cuda")
    st0, m0 = net.steps[0]
    net.steps[0] = (st0, lambda *a: (pre, None))              # 用抓取的 pre 输出替换 pre_block
    st_post, _ = net.steps[-1]
    # A deliberate no-op endpoint avoids swallowing unrelated UnboundLocalError failures.
    net.steps[-1] = (st_post, lambda *a: torch.empty(0, device="cuda"))
    tr = {}
    net(np.zeros((360, 640, 3), np.float32), trace=tr)
    for i, (seq, C, H, W) in CHECK.items():
        if paths[i] is None:
            continue
        k = img(paths[i], C, H, W)
        r = tr[i].cpu().numpy()
        report.compare(f"step{i}/seq{seq}", r, k)
        print(f"step{i:2d} (seq{seq}) {C}ch {H}x{W}: 相关 {np.corrcoef(r.ravel(), k.ravel())[0, 1]:.5f}  逐值精确 {(r == k).mean():.4f}"
              f"  幅度比 kernel/torch {slope(r, k):.4f}  std {k.std():.3f}/{r.std():.3f}")


if __name__ == "__main__":
    p = arguments(parser(__doc__, tensors=True))
    p.add_argument("--allow-missing-taps", action="store_true",
                   help="Diagnostic only: allow incomplete coverage; JSON still reports passed=false")
    args = p.parse_args()
    from check_schedule import safe_report_destination
    try:
        safe_report_destination(args.report or args.reference_dir / "validation/check_levels.json", args.capture, args.harness)
    except ValueError as error:
        p.error(str(error))
    sys.exit(execute("check_levels", args, main))
