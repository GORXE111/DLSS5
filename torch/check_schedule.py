"""Isolate every declared schedule step using one strict, fresh, hashed reference set.

python torch/check_schedule.py --capture research/batch3/schedule_capture --harness harness/batch3/nr-lab.exe --report research/batch3/check_schedule.json
Intermediate gates are port regression criteria; post is compared after sRGB8 quantization.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.schedule_refs import (SOURCE_FILES, digest, recorded_runtime, runtime_files, safe_path, save, validate)  # noqa: E402
from acceptance import correlation_threshold, metrics, nonnegative_finite, ppm  # noqa: E402


def decode(raw, description):
    """Exact-length byte layout decoding; no filename or minimum-size heuristics."""
    import torch
    h, w, c = description["shape_hwc"]
    if len(raw) != h * w * c:
        raise ValueError(f"Invalid exact tensor size: {len(raw)} != {h*w*c}")
    values = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float().numpy()[np.frombuffer(raw, np.uint8)]
    if not np.isfinite(values).all():
        raise ValueError("Non-finite captured FP8 tensor")
    layout = description["layout"]
    if layout == "image16":
        return values.reshape(c // 16, h, w, 16).transpose(1, 2, 0, 3).reshape(h * w, c)
    if layout == "tin1d":
        th, tw = 4, h * w // 4
    elif layout == "tin2d":
        th, tw = h, w
    else:
        raise ValueError(f"Unsupported explicit tensor layout: {layout}")
    v = values.reshape(th // 4, tw // 4, c // 32, 32, 16)
    lane, byte = np.meshgrid(np.arange(32), np.arange(16), indexing="ij")
    token = (lane >> 2) + 8 * ((byte >> 2) % 2)
    channel = 8 * (byte % 4) + 2 * (lane & 3) + (byte >> 3)
    out = np.empty((th // 4, tw // 4, 4, 4, c // 32, 32), np.float32)
    out[:, :, token // 4, token % 4, :, channel] = v.transpose(3, 4, 0, 1, 2)
    out = out.transpose(0, 2, 1, 3, 4, 5).reshape(th, tw, c)
    if layout == "tin1d":
        out = out.reshape(4, h * w // 16, 4, c).transpose(1, 0, 2, 3)
    return out.reshape(h * w, c)


def safe_report_destination(report, capture, harness):
    report = Path(report).resolve()
    if report.suffix.lower() != ".json":
        raise ValueError("Report must have a .json extension")
    if capture is not None and report.is_relative_to(Path(capture).resolve()):
        raise ValueError("Report must be outside the capture and its artifacts")
    protected = [ROOT / relative for relative in SOURCE_FILES]
    if capture is not None and Path(capture).is_dir():
        protected += [path for path in Path(capture).rglob("*") if path.is_file()]
    protected += list((ROOT / "torch/dlss5").glob("*.py"))
    protected += list((ROOT / "torch/dlss5/data").glob("*.json"))
    protected += [Path(__file__), ROOT / "torch/acceptance.py", ROOT / "tools/schedule_refs.py",
                  ROOT / "WEIGHTS_HT.bin", ROOT / "dlss5/nvngx_dlssnr.dll", ROOT / "harness/reference.py"]
    if harness is not None:
        protected += list(runtime_files(harness).values())
    for candidate in (report, report.with_suffix(report.suffix + ".tmp")):
        for source in protected:
            source = source.resolve()
            if candidate == source or candidate.exists() and source.exists() and candidate.samefile(source):
                raise ValueError("Report aliases a declared source/runtime input")
    return report


def main():
    early = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for name in ("--capture", "--harness", "--report"):
        early.add_argument(name, type=Path)
    early_args, _ = early.parse_known_args()
    if early_args.report is not None:
        try:
            destination = safe_report_destination(early_args.report, early_args.capture, early_args.harness)
            save(destination, {"schema": 1, "suite": "check_schedule", "passed": False,
                               "status": "argument_validation_pending", "errors": ["Arguments not yet validated"]})
        except (ValueError, OSError) as error:
            print(f"FAIL: unsafe/unwritable report: {error}", file=sys.stderr)
            return 2
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--bit-exact", action="store_true")
    parser.add_argument("--min-correlation", type=correlation_threshold, default=.98)
    parser.add_argument("--max-nrmse", type=nonnegative_finite, default=.25)
    args = parser.parse_args()
    report = {"schema": 1, "suite": "check_schedule", "passed": False, "checks": [], "errors": [],
              "scope": "Independent captured input -> output for the 640x360 reset schedule; default f16 sm86 port regression, not native parity",
              "thresholds": {"tensor_min_correlation": args.min_correlation, "tensor_max_nrmse": args.max_nrmse,
                             "post_min_correlation": .999, "post_max_mae_255": 3., "bit_exact": args.bit_exact},
              "post_boundary": "Final sRGB8 copy, not raw internal float surface",
              "coverage": {"schedule_steps_total": 71, "schedule_steps_compared": 0, "schedule_steps_passed": 0},
              "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__), ROOT / "tools/schedule_refs.py", ROOT / "torch/acceptance.py"]}}
    save(args.report, report)  # Invalidate any previous PASS before validation or CUDA initialization.
    try:
        dataset = validate(args.capture, expected_runtime=recorded_runtime(args.capture, args.harness))
        report["capture_manifest_sha256"] = digest(args.capture / "manifest.json")
        report["reference_runtime_sha256"] = dataset["runtime_sha256"]
        report["capture_source_sha256"] = dataset["source_sha256"]
        report["capture_tool_sha256"] = dataset["capture_tool_sha256"]
        report["source_sha256"]["harness/reference.py"] = digest(ROOT / "harness/reference.py")
        for path in sorted((ROOT / "torch/dlss5").glob("*.py")):
            report["source_sha256"][str(path.relative_to(ROOT))] = digest(path)
        # All provenance checks finish before importing the GPU implementation.
        import torch
        from dlss5 import DLSS5, ops
        from dlss5.net import dims
        ops.PRECISE = True
        net = DLSS5(precise=True)
        dimensions = dims(360, 640)
        report["gpu"] = torch.cuda.get_device_name()
        report["torch_version"] = torch.__version__
        cache = {}
        def array(name):
            if name not in cache:
                entry = dataset["buffers"][name]
                cache[name] = decode(safe_path(args.capture, entry["path"]).read_bytes(), dataset["plan"]["buffers"][name])
            return cache[name]
        def tensor(name):
            return torch.from_numpy(array(name)).cuda()
        def compare(step, role, prediction, name, *, post=False, bridge=False):
            output = prediction.detach().cpu().numpy() if torch.is_tensor(prediction) else prediction
            reference = ppm(safe_path(args.capture, dataset["final_output"]["path"]), 640, 360) if post else array(name)
            m = metrics(output, reference, quantize=post)
            if args.bit_exact or bridge:
                passed = m["exact_fraction"] == 1
            elif post:
                passed = m["correlation"] >= .999 and m["mae"] <= 3
            else:
                passed = m["correlation"] >= args.min_correlation and m["nrmse"] is not None and m["nrmse"] <= args.max_nrmse
            row = {"step": step, "role": role, "reference": name, "passed": bool(passed), **m,
                   "output_sha256": hashlib.sha256(np.ascontiguousarray(output).tobytes()).hexdigest(),
                   "reference_sha256": dataset["final_output"]["sha256"] if post else dataset["buffers"][name]["sha256"]}
            report["checks"].append(row)
            print(f"step{step} {role}: corr={m['correlation']:.6f} NRMSE={m['nrmse']} {'PASS' if passed else 'FAIL'}", flush=True)
            return passed
        color = torch.from_numpy(np.frombuffer(safe_path(args.capture, dataset["input"]["path"]).read_bytes(), np.uint8).copy().reshape(360, 640, 3)).cuda().float() / 255
        motion = torch.zeros(360, 640, 2, device="cuda")
        for left, right in dataset["plan"]["bridges"]:
            compare(None, "repack", array(left), right, bridge=True)
        with torch.no_grad():
            for entry, (step, module) in zip(dataset["plan"]["steps"], net.steps):
                index, kind = entry["step"], step["kind"]
                inputs, outputs = entry["inputs"], entry["outputs"]
                before = len(report["checks"])
                if kind == "pre":
                    result, skip = module(color, color, motion, 0)
                    compare(index, "skip", skip, outputs["skip"])
                elif kind == "swin":
                    h, w = dimensions[step["level"]]
                    x = tensor(inputs["input"])
                    shift, variant = tuple(step["shift"]), step["variant"]
                    if variant == "inpview":
                        result = module(x[:, module.fi], h, w, shift)
                    elif variant == "up":
                        result = module(x, tensor(inputs["skip"]), h, w, shift)
                    else:
                        result = module(x, h, w, shift)
                    if variant == "ds":
                        result, down = result
                        compare(index, "downsample", down, outputs["downsample"])
                    elif variant == "outview":
                        result = result[:, module.ci]
                elif kind == "split16":
                    x = tensor(inputs["input"])
                    result = module(x[:, module.fi] if step["inpview"] else x, tuple(step["shift"]), *dimensions["16h"])
                    if step["tail"] == "pool":
                        compare(index, "skip", result, outputs["skip"])
                        result = step["head"](result, dimensions["16h"], dimensions["vit"])
                    elif step["tail"] == "outview":
                        result = result[:, module.ci]
                elif kind == "vit":
                    result = module(tensor(inputs["input"]))
                elif kind == "block39":
                    result = module(tensor(inputs["input"]), tensor(inputs["skip"]), dimensions["vit"], dimensions["16h"])
                elif kind == "post":
                    result = module(tensor(inputs["input"]), tensor(inputs["skip"]), color, None, motion)
                else:
                    raise ValueError(f"Unsupported schedule kind {kind}")
                compare(index, "output", result, outputs["output"], post=kind == "post")
                report["coverage"]["schedule_steps_compared"] += 1
                report["coverage"]["schedule_steps_passed"] += int(all(c["passed"] for c in report["checks"][before:]))
                save(args.report, report)
        report["passed"] = (report["coverage"]["schedule_steps_compared"] == 71 and all(row["passed"] for row in report["checks"]))
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
        report["passed"] = False
        print(f"FAIL: {type(error).__name__}: {error}", file=sys.stderr)
    save(args.report, report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
