"""控制参数对照: 每组 DLSSNR 参数让 nr-lab 跑 1 帧 (640x360 合成画面)，torch 用同样参数前向，比较输出。
    python check_controls.py
需要 harness/rt_sm86/nr-lab.exe。输出只在本地 (harness/rt_sm86/nr-lab-output-*.ppm)，不入库。"""
import os
import subprocess

import numpy as np

from check import color_pattern, stats
from dlss5 import DLSS5
from dlss5.net import control_inputs

RT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "harness", "rt_sm86")
CASES = [   # (nr-lab 参数, control_inputs 参数, intensity)
    ([], {}, 1.0),
    (["--local-tone", "0.25"], {"tone": 0.25}, 1.0),
    (["--local-structure", "0.3"], {"structure": 0.3}, 1.0),
    (["--skin-structure", "0.5"], {"skin": 0.5}, 1.0),
    (["--auto-mask", "0", "--local-structure", "0.6"], {"auto_mask": False, "structure": 0.6}, 1.0),
    (["--local-tone", "1.5", "--local-structure", "1.5"], {"tone": 1.5, "structure": 1.5}, 1.0),
    (["--intensity", "0.4"], {}, 0.4),
    (["--style", "1"], {"style": 1}, 1.0),
    (["--style", "2"], {"style": 2}, 1.0),
]


def nrlab(args):
    subprocess.run([os.path.join(RT, "nr-lab.exe"), "--nr-only", "--input", "640x360", "--output", "640x360", "--frames", "1"]
                   + args, cwd=RT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300, check=True)
    raw = open(os.path.join(RT, "nr-lab-output-model1-srgb.ppm"), "rb").read()
    return np.frombuffer(raw[-640 * 360 * 3:], np.uint8).reshape(360, 640, 3).astype(np.float32) / 255


def main():
    net = DLSS5()
    color = color_pattern()
    base = None
    for args, ctl, inten in CASES:
        ref = nrlab(args)
        out = net(color, frame=0, controls=control_inputs(**ctl), intensity=inten).cpu().numpy()
        base = ref if base is None else base
        stats(f"{' '.join(args) or '默认':42s}", out, ref)
        print(f"{'':44s}(nr-lab 相对默认的变化 {np.abs(ref - base).mean() * 255:.2f}/255)")


if __name__ == "__main__":
    main()
