"""控制参数对照: 每组 DLSSNR 参数让 nr-lab 跑 1 帧 (640x360 合成画面)，torch 用同样参数前向，比较输出。
    python check_controls.py
需要 harness/rt_sm86/nr-lab.exe。输出只在本地 (harness/rt_sm86/nr-lab-output-*.ppm)，不入库。"""
import os
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from acceptance import execute, parser, ppm
from check import color_pattern
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
    (["--style", "1", "--intensity", "0.5"], {"style": 1}, 0.5),
    (["--style", "2", "--local-structure", "0.5"], {"style": 2, "structure": 0.5}, 1.0),
]
# DLSSNR.ControlMask (RGBA16F): nr-lab 的图案 (--optional-variant 1 常数 / 2 中央方块 / 3 横向渐变) 与逐通道倍数
CM = ["--optional", "control-mask", "--optional-format", "rgba16f"]
MASKS = [   # (nr-lab 参数, 图案, 渐变通道位, RGBA 倍数, control_inputs 参数, intensity)
    (CM + ["--optional-variant", "3", "--ramp-channels", "2"], 3, 2, (1, 1, 1, 1), {}, 1.0),
    (CM + ["--optional-variant", "3", "--ramp-channels", "4", "--local-tone", "0.5"], 3, 4, (1, 1, 1, 1), {"tone": 0.5}, 1.0),
    (CM + ["--optional-variant", "3", "--ramp-channels", "1", "--intensity", "0.7"], 3, 1, (1, 1, 1, 1), {}, 0.7),
    (CM + ["--optional-variant", "2", "--optional-rgba", "1,0.5,1.5,1"], 2, 0, (1, 0.5, 1.5, 1), {}, 1.0),
    (CM + ["--optional-variant", "3", "--ramp-channels", "1", "--style", "1"], 3, 1, (1, 1, 1, 1), {"style": 1}, 1.0),
    (CM + ["--optional-variant", "3", "--ramp-channels", "1", "--style", "2"], 3, 1, (1, 1, 1, 1), {"style": 2}, 1.0),
    (CM + ["--optional-variant", "3", "--ramp-channels", "1", "--style", "1", "--intensity", "0.4"],
     3, 1, (1, 1, 1, 1), {"style": 1}, 0.4),
]


def mask_pattern(variant, ramp_bits, rgba, W=640, H=360):
    """nr-lab MakeOptionalPattern 的同款 (值 1)"""
    y, x = np.mgrid[0:H, 0:W]
    centre = (x >= W // 4) & (x < 3 * W // 4) & (y >= H // 4) & (y < 3 * H // 4)
    shape = {1: np.ones((H, W)), 2: centre.astype(np.float64), 3: (x + 0.5) / W}[variant]
    m = np.stack([(shape if variant != 3 or (ramp_bits >> c) & 1 else np.ones((H, W))) * rgba[c] for c in range(4)], -1)
    return m.astype(np.float32)


def nrlab(args):
    result, output = Path(RT) / "nr-lab-result.json", Path(RT) / "nr-lab-output-model1-srgb.ppm"
    for path in (result, output):
        path.unlink(missing_ok=True)
    subprocess.run([os.path.join(RT, "nr-lab.exe"), "--nr-only", "--model", "1", "--input", "640x360", "--output", "640x360", "--frames", "1"]
                   + args, cwd=RT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300, check=True)
    run = json.loads(result.read_text())
    if not run.get("created") or run.get("evaluationsSucceeded") != 1 or not run.get("upscalingValidated"):
        raise RuntimeError(f"nr-lab did not validate the requested frame: {run}")
    return ppm(output, 640, 360)


def main(report):
    report.data["runtime_sha256"] = {}
    for name in ("nr-lab.exe", "nvngx_dlssnr.dll", "nvngx.dll"):
        path = Path(RT) / name
        report.reference(path)
        report.data["runtime_sha256"][name] = report.data["references"][str(path.resolve())]
    net = DLSS5()
    color = color_pattern()
    base = None
    for i, (args, ctl, inten) in enumerate(CASES):
        ref = nrlab(args)
        reference = Path(RT) / "nr-lab-output-model1-srgb.ppm"
        report.reference(reference)
        reference_sha256 = report.data["references"][str(reference.resolve())]
        out = net(color, frame=0, controls=control_inputs(**ctl), intensity=inten).cpu().numpy()
        base = ref if base is None else base
        report.compare(f"controls_{i}_{' '.join(args) or 'default'}", out, ref, quantize=True)
        report.data["checks"][-1].update(reference_path=str(reference.resolve()), reference_sha256=reference_sha256)
        print(f"{'':44s}(nr-lab 相对默认的变化 {np.abs(ref - base).mean() * 255:.2f}/255)")
    for i, (args, variant, bits, rgba, ctl, inten) in enumerate(MASKS):
        ref = nrlab(args)
        reference = Path(RT) / "nr-lab-output-model1-srgb.ppm"
        report.reference(reference)
        reference_sha256 = report.data["references"][str(reference.resolve())]
        out = net(color, frame=0, controls=control_inputs(auto_mask=False, **ctl), intensity=inten,
                  control_mask=mask_pattern(variant, bits, rgba).astype(np.float16).astype(np.float32)).cpu().numpy()
        report.compare(f"mask_{i}_{' '.join(args[4:])}", out, ref, quantize=True)
        report.data["checks"][-1].update(reference_path=str(reference.resolve()), reference_sha256=reference_sha256)
        print(f"{'':44s}(nr-lab 相对默认的变化 {np.abs(ref - base).mean() * 255:.2f}/255)")


if __name__ == "__main__":
    sys.exit(execute("check_controls", parser(__doc__).parse_args(), main))
