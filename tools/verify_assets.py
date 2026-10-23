"""Read-only verification of the supported 310.8.0 assets and network schedule.

Run from the repository root: python -m tools.verify_assets [--report path.json]
This verifies provenance and record coverage, not numeric fidelity or training.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

from .dllq.index import parse_pe, parse_resources, parse_weights
from .report_paths import ReportPath, project_inputs

ROOT = Path(__file__).resolve().parents[1]
DLL_SHA256 = "e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e"
WEIGHTS_SHA256 = "836f445d06ecd2e59bb9f17b84b91c143396fd76ccda1c9dc7fe81d5edd548f4"


def verify(root):
    original = (root / "dlss5" / "nvngx_dlssnr.dll").read_bytes()
    weights = (root / "WEIGHTS_HT.bin").read_bytes()
    dll_hash, weights_hash = hashlib.sha256(original).hexdigest(), hashlib.sha256(weights).hexdigest()
    if dll_hash != DLL_SHA256:
        raise ValueError("original DLL is not the supported unmodified 310.8.0 build")
    if weights_hash != WEIGHTS_SHA256:
        raise ValueError("WEIGHTS_HT.bin does not match the supported 310.8.0 resource")
    sections, resource_rva = parse_pe(original)
    matches = [resource for resource in parse_resources(original, sections, resource_rva)
               if resource["rtype"] == "10" and resource["rname"] == "WEIGHTS_HT" and resource["rlang"] == "1033"]
    if len(matches) != 1:
        raise ValueError("expected one WEIGHTS_HT/1033 resource")
    resource = matches[0]
    if original[resource["off"]:resource["off"] + resource["size"]] != weights:
        raise ValueError("extracted weights differ from the original DLL resource")
    parsed = parse_weights(weights)
    mapping = json.loads((root / "weights_map.json").read_text(encoding="utf-8"))
    if len(parsed) != 153 or len(mapping) != 153 or len({row["name"] for row in mapping}) != 153:
        raise ValueError("expected 153 unique weight records")
    by_name = {row["name"]: row for row in mapping}
    end = 0
    for record in parsed:
        declared = by_name.get(record["name"], {})
        if (record["off"], record["bytes"], record["tag"]) != (declared.get("off"), declared.get("C"), declared.get("tag")):
            raise ValueError(f"weight map differs from parsed record: {record['name']}")
        if record["off"] < end or record["off"] + record["bytes"] > len(weights):
            raise ValueError(f"overlapping or out-of-bounds record: {record['name']}")
        end = record["off"] + record["bytes"]
    data = root / "torch" / "dlss5" / "data"
    schedule = json.loads((data / "schedule.json").read_text(encoding="utf-8"))["steps"]
    if len(schedule) != 71:
        raise ValueError("expected 71 schedule steps")
    scheduled = []
    for step in schedule:
        scheduled.extend(step.get("records", []))
        scheduled.extend(step[key] for key in ("record", "head_record", "blend_record") if key in step)
    usage = Counter(scheduled)
    if set(usage) != set(by_name) or any(count != 1 for count in usage.values()):
        raise ValueError("schedule must reference each of the 153 records exactly once")
    launches = json.loads((data / "exec_order.json").read_text(encoding="utf-8"))["launches"]
    execution = [name for launch in launches for name in launch["weights"]]
    if scheduled != execution:
        raise ValueError("schedule record order differs from recorded kernel execution order")
    return {"passed": True, "target": "DLSS-NR 310.8.0", "dll_sha256": dll_hash, "weights_sha256": weights_hash,
            "weights_bytes": len(weights), "payload_bytes": sum(record["bytes"] for record in parsed),
            "records": len(parsed), "steps": len(schedule), "step_kinds": dict(Counter(step["kind"] for step in schedule)),
            "kernel_launches": len(launches), "all_records_referenced_once": True,
            "scope": "original resource identity, byte ranges and schedule coverage; not numerical equivalence"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    files, directories = project_inputs(args.root)
    output = ReportPath(args.report, inputs=files, input_dirs=directories) if args.report else None
    try:
        report = verify(args.root.resolve())
    except (OSError, ValueError, KeyError) as error:
        report = {"passed": False, "error": str(error)}
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.report:
        output.write_json(report)
    print(text, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
