"""各级注意力数值选项组合研究扫描；需显式fresh capture，本批未重跑GPU扫描。"""
import argparse
import contextlib
import io
import itertools
from pathlib import Path
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--harness", type=Path)
    parser.add_argument("--levels", nargs="+", choices=("1h", "2h", "4h", "8h"), default=["1h", "2h", "4h", "8h"])
    args = parser.parse_args()
    from schedule_reference import ScheduleReference
    try:
        reference = ScheduleReference(args.capture, args.harness)
    except (ValueError, OSError) as error:
        print(f"FAIL before GPU initialization: {error}", file=sys.stderr)
        return 1
    import check_blocks as CB
    from dlss5 import DLSS5, ops
    from dlss5.blocks import Swin
    CB.REFERENCE = reference
    ops.PRECISE = True
    net = DLSS5(precise=True)
    for lv in args.levels:
        for be, qr, qo in itertools.product((False, True), repeat=3):
            for _, module in net.steps:
                for block in (module, getattr(module, "block", None)):
                    if isinstance(block, Swin):
                        block.bit_exp, block.q_res, block.q_O = be, qr, qo
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                CB.run(net, levels=(lv,))
            exact = [float(line.split("精确 ")[1].split()[0]) for line in buf.getvalue().splitlines()
                     if "精确" in line and "下采样" not in line]
            print(f"{lv} 位运算exp={be:d} 残差q8={qr:d} O量化={qo:d}: 平均逐值精确 {np.mean(exact):.4f}  最低 {min(exact):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
