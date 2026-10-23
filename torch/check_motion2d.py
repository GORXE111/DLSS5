"""Compare independent PyTorch history against a complete versioned motion2d capture.

The synthetic matrix includes vertical/negative/diagonal motion, fractional and
out-of-bounds vectors, odd dimensions, and exact Intensity=0 input checks.
Passing the default approximate gates is not native sm120 equivalence.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from acceptance import execute, parser
from check import color_pattern
from dlss5 import DLSS5

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
import reference  # noqa: E402


def dataset(folder):
    manifest = reference.load_capture(folder)
    if manifest["suite"] != "motion2d":
        raise ValueError("requires the complete motion2d capture suite")
    rows = []
    for case in manifest["cases"]:
        fields, _ = reference.case_contract(case)
        output = next(iter(case["outputs"]))
        path = folder / case["id"] / output
        rows.append((case["id"], fields, reference.read_ppm(path).astype(np.float32) / 255, path))
    return manifest, rows


def validate_report_target(target, capture):
    target, capture = target.resolve(), capture.resolve()
    if target.is_relative_to(capture):
        raise ValueError("report must be outside the capture directory")
    protected = [ROOT / "WEIGHTS_HT.bin", ROOT / "weights_map.json", ROOT / "harness/reference.py",
                 *ROOT.joinpath("torch").rglob("*.py"), *ROOT.joinpath("torch/dlss5/data").glob("*.json")]
    if capture.is_dir():
        protected.extend(p for p in capture.rglob("*") if p.is_file())
    for path in protected:
        if str(target).casefold() == str(path.resolve()).casefold() or (
                target.exists() and path.exists() and os.path.samefile(target, path)):
            raise ValueError(f"report aliases a capture or model input: {path}")


def main(report):
    manifest, rows = dataset(report.args.capture)
    report.data.update(reference_kind=manifest["reference_kind"],
                       official_reference=manifest["reference_kind"] == "official-sm120",
                       claim="independent_torch_vs_recorded_dll_synthetic_matrix",
                       reference_accumulation="recorded_dll_variant; inspect its SHA256",
                       runtime_sha256=manifest["runtime_sha256"],
                       capture_manifest_sha256=reference.digest(report.args.capture / "manifest.json"))
    report.reference(report.args.capture / "manifest.json")
    report.data["source_sha256"] = {name: reference.digest(ROOT / name) for name in
                                    ("torch/check_motion2d.py", "torch/check.py", "torch/dlss5/net.py", "torch/dlss5/ops.py",
                                     "torch/dlss5/ptx_math.py")}
    groups = {}
    for name, fields, values, path in rows:
        report.reference(path)
        width, height = fields["input"]["width"], fields["input"]["height"]
        frame = fields["evaluationsSucceeded"] - 1
        sx, sy = fields["temporalShiftPixels"], fields["temporalShiftYPixels"]
        if fields["intensity"] == 0:
            expected = color_pattern(width, height, frame * sx, frame * sy)
            m = report.compare(name + "/exact_input", expected, values, quantize=True)
            report.data["checks"][-1]["passed"] = m["exact_fraction"] == 1.0
            report.data["checks"][-1]["criterion"] = "bit_exact_input"
            continue
        key = (width, height, sx, sy, *fields["motionVector"])
        groups.setdefault(key, {})[frame] = (name, values)
    net = DLSS5()
    for (width, height, sx, sy, mx, my), refs in groups.items():
        mv = torch.empty((height, width, 2), device="cuda", dtype=torch.float32)
        mv[..., 0], mv[..., 1] = mx, my
        for mode in ("float_history", "uint8_history"):
            hist = None
            for frame in range(max(refs) + 1):
                color = color_pattern(width, height, frame * sx, frame * sy)
                output = net(color, hist=hist, mv=None if hist is None else mv, frame=frame)
                quantized = torch.floor(output * 255 + 0.5) / 255
                if frame in refs:
                    name, values = refs[frame]
                    report.compare(name + "/" + mode, quantized.cpu().numpy(), values, quantize=True)
                hist = output if mode == "float_history" else quantized


def run_cli():
    # Locate and protect the report before validating numerical thresholds, so
    # an invalid threshold cannot leave a previous successful report in place.
    if not any(flag in sys.argv[1:] for flag in ("-h", "--help")):
        early = argparse.ArgumentParser(add_help=False)
        early.add_argument("--capture", type=Path, default=ROOT / "harness/captures")
        early.add_argument("--report", type=Path)
        early.add_argument("--reference-dir", type=Path, default=ROOT / "research")
        known, _ = early.parse_known_args()
        target = known.report or known.reference_dir / "validation/check_motion2d.json"
        try:
            validate_report_target(target, known.capture)
        except ValueError as error:
            early.error(str(error))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"passed": False, "status": "running"}) + "\n", encoding="utf-8")
    p = parser(__doc__)
    p.add_argument("--capture", type=Path, required=True)
    args = p.parse_args()
    target = args.report or args.reference_dir / "validation" / "check_motion2d.json"
    try:
        validate_report_target(target, args.capture)
    except ValueError as error:
        p.error(str(error))
    target.parent.mkdir(parents=True, exist_ok=True)
    # A CUDA crash or process interruption must not leave the previous PASS.
    target.write_text(json.dumps({"passed": False, "status": "running"}) + "\n", encoding="utf-8")
    return execute("check_motion2d", args, main)


if __name__ == "__main__":
    sys.exit(run_cli())
