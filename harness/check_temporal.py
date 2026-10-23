"""Measure reset reinitialization and scene-cut sensitivity within one recorded runtime.

This is a synthetic temporal regression probe, not a native NVIDIA fidelity test.
Reset equivalence is checked strictly; a failure remains visible for investigation.
"""
import argparse
import json
import sys
from pathlib import Path

import reference


def analyze(folder, manifest):
    if manifest["suite"] != "temporal":
        raise ValueError("temporal analysis requires the complete temporal capture suite")
    rows = {c["id"]: c for c in manifest["cases"]}
    expectations = [
        ("static_reset", "static_reset_f2", "fresh_a", True),
        ("cut_reset_first", "cut_reset_f2", "fresh_b", True),
        ("cut_reset_second", "cut_reset_f3", "fresh_b_f1", True),
        ("longer_history_reset", "cut_reset_f8", "fresh_b", True),
        ("distinct_scenes", "fresh_a", "fresh_b", False),
        ("reset_has_effect", "cut_f2", "cut_reset_f2", False),
    ]
    checks = []
    for name, left, right, equal in expectations:
        names = set(rows[left]["outputs"]) & set(rows[right]["outputs"])
        if not names or set(rows[left]["outputs"]) != set(rows[right]["outputs"]):
            raise ValueError("temporal relation has missing or different output inventories")
        for output in sorted(names):
            a, b = folder / left / output, folder / right / output
            if a.suffix != ".ppm":
                raise ValueError("temporal suite currently requires SDR PPM outputs")
            values = reference.metrics(reference.read_ppm(a), reference.read_ppm(b))
            checks.append({"name": name, "left": left, "right": right, "output": output,
                           "expected_bit_exact": equal, **values, "passed": values["bit_exact"] == equal})
    observations = []
    for left, right in (("history_a_f3", "history_a_f15"), ("cut_f3", "cut_reset_f3")):
        output = next(iter(rows[left]["outputs"]))
        observations.append({"left": left, "right": right,
                             **reference.metrics(reference.read_ppm(folder / left / output),
                                                 reference.read_ppm(folder / right / output))})
    return {"schema": 1, "scope": "synthetic scene/reset regression within the same runtime; not official native equivalence",
            "capture": str(folder.resolve()), "capture_manifest_sha256": reference.digest(folder / "manifest.json"),
            "reference_kind": manifest["reference_kind"], "runtime_sha256": manifest["runtime_sha256"],
            "checks": checks, "observations": observations, "passed": all(c["passed"] for c in checks)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    # Import by the repository root, including when invoked as a direct script.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.report_paths import ReportPath
    try:
        output = ReportPath(args.report, input_dirs=(args.capture,))
    except ValueError as error:
        parser.error(str(error))
    output.write_json({"schema": 1, "passed": False, "status": "running"})
    try:
        report = analyze(args.capture, reference.load_capture(args.capture))
        output.write_json(report)
        for check in report["checks"]:
            print(f"{'PASS' if check['passed'] else 'FAIL'} {check['name']}: max={check['max_255']:.5f}/255")
        return 0 if report["passed"] else 1
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
        output.write_json({"schema": 1, "passed": False, "status": "failed", "error": str(error)})
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
