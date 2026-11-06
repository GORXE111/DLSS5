"""Verify every October preset-2 record in a GPU weight buffer dump.

Use the same tap run's trace and index. Bind the dump to the pre launch at
byte +232, verify the inferred pointer map, then hash each record separately
at its 512-byte-aligned location. Padding is excluded from record checks.
"""

import argparse
from hashlib import sha256
import json
from mmap import ACCESS_READ, mmap
from pathlib import Path
import struct
import sys

from ..align_nr_small_trace import GRAPH, GRAPH_SHA, align_frame, kernel_inventory, pinned_bytes, read_trace
from ..exec_map import record_layout
from .check_october_live_repack import tap_index


LAYOUT = GRAPH.with_name("october_nr_small_layout.json")
LAYOUT_SHA = "22ae030815b56994199d6ec3555e4eb552b40d8fdb0dbd4ff9936a31baad77cc"


def run(folder, frame):
    folder = folder.resolve()
    trace = (folder / "nr-trace.tsv").read_bytes()
    frames = read_trace(trace, kernel_inventory())
    if frame not in frames:
        raise ValueError("trace lacks the requested frame ID")
    graph = json.loads(pinned_bytes(GRAPH, GRAPH_SHA))
    aligned = align_frame(frame, frames[frame], graph, expect_mask_mode=None)
    if not aligned["weight_pointer_layout_complete"]:
        raise ValueError("expected complete October weight pointer layout")
    layout = json.loads(pinned_bytes(LAYOUT, LAYOUT_SHA))
    records = {row["name"]: row for row in layout["records"]}
    starts, packed_bytes = record_layout([{"name": r["name"], "C": r["payload_bytes"]} for r in records.values()])
    if len(records) != 77 or packed_bytes != aligned["packed_weight_bytes"]:
        raise ValueError("unexpected October weight record inventory")
    pre_seq = aligned["blocks"][0]["launches"][0]["sequence"]
    pre = next(row for row in frames[frame] if row.sequence == pre_seq)
    declared, dump = tap_index(folder, frame)[pre_seq, pre.pointer(232)]
    if declared != packed_bytes or dump.stat().st_size != packed_bytes:
        raise ValueError("weight dump is not the complete inferred GPU buffer")
    checks = []
    with dump.open("rb") as stream, mmap(stream.fileno(), 0, access=ACCESS_READ) as contents:
        with memoryview(contents) as payloads:
            packed_sha = sha256(payloads).hexdigest()
            for offset, name in starts.items():
                record = records[name]
                size = record["payload_bytes"]
                matched = sha256(payloads[offset:offset + size]).hexdigest() == record["payload_sha256"]
                checks.append({"name": name, "gpu_buffer_offset": offset, "bytes": size, "matched": matched})
    mismatches = [row["name"] for row in checks if not row["matched"]]
    return {"schema": "dlssnr_october_live_weights_v1", "scope": "GPU record payloads versus the pinned serialized ledger",
            "trace_sha256": sha256(trace).hexdigest(), "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "layout_sha256": LAYOUT_SHA, "frame": frame, "pre_sequence": pre_seq, "weight_pointer_parameter_offset": 232,
            "gpu_packed_bytes": packed_bytes, "gpu_packed_sha256": packed_sha,
            "records_checked": len(checks), "payload_bytes_checked": sum(row["bytes"] for row in checks),
            "mismatching_records": mismatches, "records": checks, "passed": not mismatches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args()
    try:
        report = run(args.capture, args.frame)
    except (OSError, ValueError, KeyError, struct.error) as error:
        report = {"passed": False, "error": str(error)}
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
