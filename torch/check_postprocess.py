"""Verify same-resolution Style/ControlMask/UI postprocessing against the sm_86 port.

Runs fresh nr-lab output for every end-to-end case and the original postprocess
kernel in isolation. This establishes port agreement, not native RTX 50 parity.
Usage: python check_postprocess.py [--report results.json] [--kernel-only]
Requires the local private runtime, weights, and sm86_port/fatbin_14.sm86.cubin.
"""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import torch

from check import color_pattern
from check_controls import mask_pattern
from dlss5 import DLSS5, style
from dlss5.net import control_inputs

ROOT = Path(__file__).resolve().parent.parent
RT = ROOT / "harness" / "rt_sm86"
sys.path.insert(0, str(ROOT / "research"))
import klab  # noqa: E402


def metrics(out, ref, quantized=False):
    if out.shape != ref.shape or not out.size or not np.isfinite(out).all() or not np.isfinite(ref).all():
        raise ValueError("Invalid shapes or non-finite postprocess output/reference")
    if quantized:
        out = np.floor(np.clip(out, 0, 1) * 255 + 0.5) / 255
    d = np.abs(out - ref) * 255
    corr = (float(np.corrcoef(out.ravel(), ref.ravel())[0, 1]) if out.std() > 0 and ref.std() > 0
            else float(np.array_equal(out, ref)))
    return dict(mae_255=float(d.mean()), max_255=float(d.max()),
                corr=corr,
                equal_fraction=float((out == ref).mean()))


def nrlab(args):
    # Missing/partial runs must never reuse a prior successful frame or JSON.
    result = RT / "nr-lab-result.json"
    output = RT / "nr-lab-output-model1-srgb.ppm"
    trace = RT / "nr-trace.tsv"
    for p in (result, output, trace):
        p.unlink(missing_ok=True)
    proc = subprocess.run([str(RT / "nr-lab.exe"), "--nr-only", "--input", "640x360",
                           "--output", "640x360", "--model", "1", "--frames", "1", *args],
                          cwd=RT, env=dict(os.environ, DLSS5_PROF="1", DLSS5_TRACE="0"),
                          capture_output=True, timeout=300)
    if proc.returncode or not result.is_file() or not output.is_file() or not trace.is_file():
        raise RuntimeError(f"nr-lab failed or produced incomplete artifacts: rc={proc.returncode}\n"
                           + proc.stdout.decode(errors="replace")[-2000:])
    report = json.loads(result.read_text())
    if not report.get("created") or report.get("evaluationsSucceeded") != 1 or not report.get("upscalingValidated"):
        raise RuntimeError(f"nr-lab did not validate the requested frame: {report}")
    raw = output.read_bytes()
    if not raw.startswith(b"P6\n640 360\n255\n") or len(raw) != 640 * 360 * 3 + 15:
        raise ValueError("Unexpected nr-lab PPM dimensions or size")
    return np.frombuffer(raw[-640 * 360 * 3:], np.uint8).reshape(360, 640, 3).astype(np.float32) / 255


def post_row():
    for line in (RT / "nr-trace.tsv").read_text().splitlines():
        r = line.split("\t")
        if r[2] == "cg2r_post_process_kernel":
            return dict(kernel=r[2], grid=tuple(map(int, r[3].split(","))),
                        block=tuple(map(int, r[4].split(","))), smem=int(r[5]), params=bytes.fromhex(r[7]))
    raise RuntimeError("No postprocess kernel in fresh trace")


def run_kernel(row, color, nr, style_id, intensity=1.0, control_mask=None,
               ui=None, ui_alpha=None, backbuffer=None):
    """Direct original kernel with independently assigned textures (no neural error)."""
    H, W = color.shape[:2]
    p = bytearray(row["params"])
    keep = []
    # Texture +64 is the additive base; +128 is the subtracted original color.
    images = {32: nr, 64: backbuffer if backbuffer is not None else color,
              96: control_mask, 128: color if backbuffer is not None else None,
              160: ui, 192: ui_alpha, 224: None, 256: None}
    for off, img in images.items():
        handle = 0
        if img is not None:
            handle, buf = klab.texture(img, linear=False)
            keep.append(buf)
        struct.pack_into("<Q6f", p, off, handle, 0, 0, W, H, 1 / W, 1 / H)
    out = klab.Surface(W, H)
    struct.pack_into("<Q", p, 0, out.handle)
    struct.pack_into("<fiiiii", p, 288, min(max(float(intensity), 0), 1), W, H, 0, 0, 0)
    g = style.STYLES.get(style_id, {})
    for i, k in enumerate(("black", "white", "exposure", "gamma", "contrast", "saturation", "vibrance", "warm", "tint")):
        struct.pack_into("<f", p, 316 + 4 * i, g.get(k, 1.0 if k == "white" else 0.0))
    struct.pack_into("<5f", p, 352, *g.get("zones", (0.0,) * 5))
    klab.launch(klab.function("14", row["kernel"]), ((W + 15) // 16, (H + 15) // 16, 1),
                row["block"], row["smem"], bytes(p))
    return out.read()[..., :3]


def kernel_cases(row):
    rng = np.random.default_rng(20261023)
    color = rng.random((37, 53, 3), dtype=np.float32)
    nr = rng.random(color.shape, dtype=np.float32)
    back = rng.random(color.shape, dtype=np.float32)
    ui = rng.random((*color.shape[:2], 4), dtype=np.float32)
    alpha = rng.uniform(-0.2, 1.2, (*color.shape[:2], 1)).astype(np.float32)
    mask = rng.uniform(-0.2, 1.5, (*color.shape[:2], 4)).astype(np.float32)
    cases = [
        ("style1_mask", 1, dict(control_mask=mask)),
        ("style2_mask_intensity", 2, dict(control_mask=mask, intensity=0.6)),
        ("ui_alpha", 0, dict(ui_alpha=alpha)),
        ("ui_rgba", 0, dict(ui=ui)),
        ("ui_alpha_priority", 1, dict(ui=ui, ui_alpha=alpha, backbuffer=back)),
        ("style2_ui_mask", 2, dict(ui=ui, ui_alpha=alpha, backbuffer=back, control_mask=mask, intensity=0.7)),
        ("backbuffer_no_alpha", 0, dict(backbuffer=back)),
        ("zero_intensity_backbuffer", 1, dict(backbuffer=back, intensity=0)),
    ]
    rows = []
    for name, sid, kw in cases:
        ref = run_kernel(row, color, nr, sid, **kw)
        tk = {k: torch.from_numpy(v).cuda() if isinstance(v, np.ndarray) else v for k, v in kw.items()}
        out = style.apply(torch.from_numpy(color).cuda(), torch.from_numpy(nr).cuda(), sid,
                          ui_correction=True, **tk).cpu().numpy()
        m = metrics(out, ref)
        m.update(name=name, passed=m["max_255"] < 0.001)
        rows.append(m)
        print(f"kernel {name:28s} max={m['max_255']:.6f}/255 mean={m['mae_255']:.6f}/255", flush=True)
    return rows


def frame_cases():
    color = color_pattern()
    mask = mask_pattern(3, 1, (1, 1, 1, 1)).astype(np.float16).astype(np.float32)
    rgba = mask_pattern(3, 15, (1, 1, 1, 1)).astype(np.float16).astype(np.float32)
    alpha = np.floor(mask_pattern(3, 1, (1, 1, 1, 1))[..., :1] * 255).astype(np.float32) / 255
    cm = ["--optional", "control-mask", "--optional-format", "rgba16f", "--optional-variant", "3", "--ramp-channels", "1"]
    ua = ["--optional", "ui-alpha", "--optional-format", "r8", "--optional-variant", "3"]
    ub = ["--optional", "ui-bundle", "--optional-format", "rgba16f", "--optional-variant", "3", "--bundle-backbuffer", "1"]
    back = np.zeros_like(color)
    back[..., 1] = 1
    cases = [
        ("style1_mask", cm + ["--style", "1"], 1, dict(control_mask=mask)),
        ("style2_mask", cm + ["--style", "2"], 2, dict(control_mask=mask)),
        ("style1_mask_intensity", cm + ["--style", "1", "--intensity", "0.4"], 1, dict(control_mask=mask, intensity=0.4)),
        ("ui_alpha", ua + ["--ui-correction", "1"], 0, dict(ui_correction=True, ui_alpha=alpha)),
        ("style1_ui_alpha", ua + ["--ui-correction", "1", "--style", "1"], 1, dict(ui_correction=True, ui_alpha=alpha)),
        ("ui_bundle_priority_backbuffer", ub + ["--ui-correction", "1"], 0, dict(ui_correction=True, ui=rgba, ui_alpha=alpha, backbuffer=back)),
        ("style2_ui_bundle", ub + ["--ui-correction", "1", "--style", "2", "--intensity", "0.6"], 2,
         dict(ui_correction=True, ui=rgba, ui_alpha=alpha, backbuffer=back, intensity=0.6)),
        ("ui_disabled", ub + ["--ui-correction", "0"], 0, dict(ui_correction=False, ui=rgba, ui_alpha=alpha, backbuffer=back)),
    ]
    net = DLSS5()
    rows = []
    row = None
    for name, args, sid, kw in cases:
        ref = nrlab(args)
        if row is None:
            row = post_row()
        controls = control_inputs(style=sid, auto_mask="control_mask" not in kw)
        out = net(color, frame=0, controls=controls, **kw).cpu().numpy()
        m = metrics(out, ref, quantized=True)
        # These are regression tolerances for the approximate PyTorch network.
        # The independent kernel comparisons above have a much tighter gate.
        m.update(name=name, passed=m["mae_255"] < 2.5 and m["corr"] > 0.998,
                 arguments=args, checksum_fnv1a64=json.loads((RT / "nr-lab-result.json").read_text())["checksumFnv1a64"])
        rows.append(m)
        print(f"frame  {name:28s} mean={m['mae_255']:.3f}/255 corr={m['corr']:.6f} pass={m['passed']}", flush=True)
    return rows, row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", "--output", dest="output", type=Path,
                        help="Write the complete verification report as JSON")
    parser.add_argument("--kernel-only", action="store_true")
    args = parser.parse_args()
    report = dict(scope="DLSS-NR 310.8.0 sm_86 port; no native RTX 50 baseline", gpu=None,
                  frame_tolerance=dict(mae_255_max=2.5, correlation_min=0.998), kernel_max_255=0.001,
                  frames=[], kernels=[], errors=[], passed=False)

    def save():
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    try:
        # Invalidate an earlier PASS before loading weights, launching the lab,
        # or touching CUDA. Any later failure leaves an explicit failed report.
        save()
        if args.kernel_only:
            nrlab(["--style", "1"])
            row = post_row()
        else:
            report["frames"], row = frame_cases()
        report["kernels"] = kernel_cases(row)
        report["gpu"] = torch.cuda.get_device_name()
        report["passed"] = all(r["passed"] for r in report["frames"] + report["kernels"])
        if not report["passed"]:
            report["errors"].append("One or more postprocess comparisons exceeded their acceptance thresholds")
    except Exception as exc:
        report["passed"] = False
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    try:
        save()
    except (OSError, ValueError) as exc:
        report["passed"] = False
        report["errors"].append(f"Could not save report: {type(exc).__name__}: {exc}")
    for error in report["errors"]:
        print(f"FAIL: {error}", file=sys.stderr)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
