"""Replay October ViT's two TIN repacks, including the 4K storage padding."""

import argparse
import json
from pathlib import Path
import struct
import sys

import numpy as np

from ..align_nr_small_trace import GRAPH, GRAPH_SHA, kernel_inventory, pinned_bytes, read_trace
from ..compare_october_taps import PORT, seal_capture
from .probe_october_fence import sha
from .probe_october_repack import CHANNELS, NAMES, pack_tin


ROLES = (("block11.repack.input", "block11.repack.output"), ("block18.repack.input", "block18.out0"))


def repack(raw, height, width, *, to_1d):
    """Permute the valid bytes and add or drop the final full storage tile."""
    if height <= 0 or width <= 0 or height % 4 or width % 4:
        raise ValueError("repack requires positive dimensions aligned to 4×4 TIN tiles")
    tokens = height * width
    valid_bytes = tokens * CHANNELS
    padded_bytes = ((tokens + 31) // 32 * 32) * CHANNELS
    if len(raw) != (valid_bytes if to_1d else padded_bytes):
        raise ValueError("repack input length differs from its TIN storage geometry")
    labels = np.arange(valid_bytes, dtype="<u4").reshape(height, width, CHANNELS)
    flat = labels.reshape(tokens // 16, 4, 4, CHANNELS).transpose(1, 0, 2, 3).reshape(4, tokens // 4, CHANNELS)
    order_2d, order_1d = (np.frombuffer(pack_tin(grid), dtype="<u4") for grid in (labels, flat))
    source, destination = (order_2d, order_1d) if to_1d else (order_1d, order_2d)
    logical = np.empty(valid_bytes, np.uint8)
    logical[source] = np.frombuffer(raw, np.uint8, count=valid_bytes)
    result = logical[destination].tobytes()
    return result + bytes(padded_bytes - valid_bytes) if to_1d else result


def repack_2d_to_1d(raw, height, width):
    return repack(raw, height, width, to_1d=True)


def repack_1d_to_2d(raw, height, width):
    return repack(raw, height, width, to_1d=False)


def run(capture):
    inventory = kernel_inventory()
    graph = json.loads(pinned_bytes(GRAPH, GRAPH_SHA))
    sealed, files = seal_capture(capture, inventory, graph)
    if sealed["kind"] != PORT:
        raise ValueError("expected a sealed experimental sm86 capture")
    trace = pinned_bytes(capture / "nr-trace.tsv", sealed["artifact_sha256"]["nr-trace.tsv"])
    launches = {launch.sequence: launch for launch in read_trace(trace, inventory)[sealed["frame"]]}
    cases = []
    for index, roles in enumerate(ROLES):
        source, target = (sealed["boundaries"][role] for role in roles)
        height, width, channels = source["shape_hwc"]
        tokens = height * width
        storage = (tokens + 31) // 32 * 32
        to_1d = index == 0
        layouts = ("tin2d", "tin1d") if to_1d else ("tin1d", "tin2d")
        launch = launches[source["trace_sequence"]]
        if (source["trace_sequence"] != target["trace_sequence"] or source["shape_hwc"] != target["shape_hwc"]
                or channels != CHANNELS or (source["layout"], target["layout"]) != layouts
                or (source["parameter_offset"], target["parameter_offset"]) != (0, 8)
                or launch.name != NAMES[index] or launch.grid != (storage if to_1d else tokens, 1, 1)
                or launch.block != (256, 1, 1) or launch.shared_memory != 0):
            raise ValueError("sealed tensors do not bind the expected repack launch")
        src, dst, h, w = struct.unpack("<QQii", launch.params)
        if (h, w) != (height, width) or not src or not dst or src == dst:
            raise ValueError("invalid repack ABI")
        raw, expected = (files[role].read_bytes() for role in roles)
        if sha(raw) != source["sha256"] or sha(expected) != target["sha256"]:
            raise ValueError("capture changed after sealing")
        actual = repack(raw, height, width, to_1d=to_1d)
        if len(actual) != len(expected):
            raise ValueError("sealed output length differs from its TIN storage geometry")
        differing = int(np.count_nonzero(np.frombuffer(actual, np.uint8) != np.frombuffer(expected, np.uint8)))
        cases.append({"entry": launch.name, "trace_sequence": launch.sequence, "grid_xyz": launch.grid,
                      "shape_hwc": source["shape_hwc"], "logical_tokens": tokens, "tin1d_storage_tokens": storage,
                      "input_role": roles[0], "input_sha256": source["sha256"], "output_role": roles[1],
                      "output_sha256": sha(actual), "sealed_output_sha256": target["sha256"],
                      "bytes_compared": len(actual), "different_bytes": differing})
    return {"schema": "october_nr_vit_repack_v1", "capture_seal_sha256": sealed["capture_seal_sha256"],
            "frame": sealed["frame"], "source_sha256": sha(Path(__file__).read_bytes()), "cases": cases,
            "passed": all(case["different_bytes"] == 0 for case in cases)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    args = parser.parse_args()
    try:
        report = run(args.capture)
    except (OSError, ValueError, KeyError, struct.error) as error:
        report = {"passed": False, "error": str(error)}
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
