"""Shared acceptance reporting. Approximate gates are port-regression criteria, not native equivalence."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.report_paths import ReportPath, reference_inputs


def nonnegative_finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise argparse.ArgumentTypeError("must be a finite nonnegative number") from exc
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be a finite nonnegative number")
    return number


def correlation_threshold(value):
    number = nonnegative_finite(value)
    if number > 1:
        raise argparse.ArgumentTypeError("correlation must be between 0 and 1")
    return number


def parser(description, *, tensors=False):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--reference-dir", type=Path, default=Path(__file__).resolve().parents[1] / "research")
    p.add_argument("--report", type=Path, help="JSON report (default: research/validation/<script>.json)")
    p.add_argument("--bit-exact", action="store_true", help="Require all compared values to match")
    p.add_argument("--min-correlation", type=correlation_threshold, default=0.98 if tensors else 0.999)
    p.add_argument("--max-mae", type=nonnegative_finite, default=None if tensors else 3.0,
                   help="Maximum mean RGB error in 8-bit units")
    p.add_argument("--max-nrmse", type=nonnegative_finite, default=0.25 if tensors else None,
                   help="Maximum RMSE / reference RMS (intermediate tensors)")
    return p


def ppm(path, width, height):
    """Read the declared P6 raster, rejecting missing/truncated/mismatched references."""
    path = Path(path)
    raw = path.read_bytes()
    pos = 0
    tokens = []
    while len(tokens) < 4:
        while pos < len(raw) and raw[pos] in b" \t\r\n":
            pos += 1
        if pos < len(raw) and raw[pos] == 35:
            end = raw.find(b"\n", pos)
            if end < 0:
                raise ValueError(f"Unterminated PPM comment: {path}")
            pos = end + 1
            continue
        start = pos
        while pos < len(raw) and raw[pos] not in b" \t\r\n":
            pos += 1
        if start == pos:
            raise ValueError(f"Incomplete PPM header: {path}")
        tokens.append(raw[start:pos])
    if tokens != [b"P6", str(width).encode(), str(height).encode(), b"255"]:
        raise ValueError(f"Unexpected PPM format/dimensions: {path}: {tokens}")
    if pos >= len(raw):
        raise ValueError(f"Missing PPM raster: {path}")
    pos += 2 if raw[pos:pos + 2] == b"\r\n" else 1
    raster = raw[pos:]
    if len(raster) != width * height * 3:
        raise ValueError(f"PPM raster length mismatch: {path}")
    return np.frombuffer(raster, np.uint8).reshape(height, width, 3).astype(np.float32) / 255


def metrics(output, reference, *, quantize=False):
    a, b = np.asarray(output, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if a.shape != b.shape or not a.size:
        raise ValueError(f"Invalid comparison shapes: {a.shape} vs {b.shape}")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Non-finite output/reference")
    if quantize:
        a, b = np.floor(a * 255 + .5), np.floor(b * 255 + .5)
    d = np.abs(a - b)
    # Equality of constants is well-defined; unrelated constant fields fail correlation.
    corr = (float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
            if a.std() > 0 and b.std() > 0 else float(np.array_equal(a, b)))
    rms = float(np.sqrt(np.mean(b * b)))
    rmse = float(np.sqrt(np.mean(d * d)))
    return {"correlation": corr, "mae": float(d.mean()), "max_error": float(d.max()),
            "exact_fraction": float((a == b).mean()), "within_one_fraction": float((d <= 1).mean()),
            "nrmse": rmse / rms if rms else (0.0 if rmse == 0 else None),
            "units": "uint8_rgb" if quantize else "tensor"}


class Report:
    def __init__(self, name, args):
        self.args = args
        self.path = args.report or args.reference_dir / "validation" / f"{name}.json"
        capture = getattr(args, "capture", None)
        self.output = ReportPath(self.path, inputs=reference_inputs(args.reference_dir),
                                 input_dirs=(capture,) if capture is not None else ())
        self.data = {"schema": 1, "suite": name, "reference_kind": "RTX3060_port",
                     "reference_accumulation": "f16_two_mma_approximation",
                     "claim": "port_regression_only", "mode": "bit_exact" if args.bit_exact else "approximate",
                     "thresholds": {"min_correlation": args.min_correlation, "max_mae": args.max_mae,
                                    "max_nrmse": args.max_nrmse}, "checks": [], "skipped": [], "errors": []}

    def reference(self, path):
        path = Path(path)
        self.output.add_inputs(path)
        self.data.setdefault("references", {})[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()

    def compare(self, name, output, reference, *, quantize=False):
        m = metrics(output, reference, quantize=quantize)
        a = self.args
        passed = (m["exact_fraction"] == 1 if a.bit_exact else
                  m["correlation"] >= a.min_correlation and
                  (a.max_mae is None or m["mae"] <= a.max_mae) and
                  (a.max_nrmse is None or m["nrmse"] is not None and m["nrmse"] <= a.max_nrmse))
        self.data["checks"].append({"name": name, "passed": bool(passed), **m})
        print(f"{name}: corr={m['correlation']:.6f} MAE={m['mae']:.4f} max={m['max_error']:.4f} "
              f"exact={m['exact_fraction']:.4%} {'PASS' if passed else 'FAIL'}")
        return m

    def skip(self, name, reason):
        self.data["skipped"].append({"name": name, "reason": str(reason)})
        print(f"SKIP {name}: {reason}")

    def finish(self, *, allow_partial=False):
        count, skipped = len(self.data["checks"]), len(self.data["skipped"])
        self.data["coverage"] = {"compared": count, "skipped": skipped,
                                 "fraction": count / (count + skipped) if count + skipped else 0.0}
        good = count > 0 and not self.data["errors"] and all(c["passed"] for c in self.data["checks"])
        self.data["passed"] = good and not skipped
        self.data["status"] = "passed" if self.data["passed"] else "incomplete" if good else "failed"
        self.output.add_inputs(*(Path(path) for path in self.data.get("references", {})))
        self.output.write_json(self.data)
        print(f"{self.data['status'].upper()}: compared={count}, skipped={skipped}; report={self.path}")
        return 0 if good and (not skipped or allow_partial) else 1


def execute(name, args, fn):
    report = Report(name, args)
    try:
        fn(report)
    except Exception as exc:
        report.data["errors"].append(f"{type(exc).__name__}: {exc}")
        print(f"FAIL: {type(exc).__name__}: {exc}")
    return report.finish(allow_partial=getattr(args, "allow_missing_taps", False))
