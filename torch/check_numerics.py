"""各级注意力数值选项的组合扫描 (逐块隔离，以逐值精确率为准)"""
import itertools
import sys

import numpy as np

import check_blocks as CB
from dlss5 import DLSS5
from dlss5.blocks import Swin

net = DLSS5()
levels = sys.argv[1:] or ["1h", "2h", "4h", "8h"]
for lv in levels:
    for be, qr, qo in itertools.product((False, True), repeat=3):
        for _, m in net.steps:
            for b in (m, getattr(m, "block", None)):
                if isinstance(b, Swin):
                    b.bit_exp, b.q_res, b.q_O = be, qr, qo
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            CB.run(net, levels=(lv,))
        ex = [float(l.split("精确 ")[1].split()[0]) for l in buf.getvalue().splitlines() if "精确" in l and "下采样" not in l]
        print(f"{lv} 位运算exp={be:d} 残差q8={qr:d} O量化={qo:d}: 平均逐值精确 {np.mean(ex):.4f}  最低 {min(ex):.4f}")
