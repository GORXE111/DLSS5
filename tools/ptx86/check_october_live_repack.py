"""Check paired 2D/1D ViT buffers tapped from an October nr-lab frame.

Reads a frame-prefixed nr-trace.tsv and tap_index.tsv from the same tap run.
Matches the dump addresses to the two repack parameter blocks, then compares
each observed output against CPU packing of that launch's own input.
"""

import argparse
import json
from pathlib import Path
import re
import struct
import sys

import numpy as np

from ..align_nr_small_trace import kernel_inventory, read_trace
from .probe_october_fence import sha
from .probe_october_repack import CHANNELS, NAMES, pack_tin


def tap_index(folder, frame):
    log = (folder / "stdout.log").read_text(encoding="ascii", errors="replace")
    entries = re.findall(r"\[tap\] frame (\d+) seq (\d+) (0x[0-9a-f]+) \+(\d+) -> off \d+ \(launch=0\)", log)
    observed = {(int(seq), int(address, 16)): int(size)
                for captured_frame, seq, address, size in entries if int(captured_frame) == frame}
    if not observed:
        raise ValueError("no successful tap entries for the requested frame")
    rows = {}
    for line in (folder / "tap_index.tsv").read_text(encoding="ascii").splitlines():
        sequence, address, size, filename = line.split("\t")
        seq, ptr = int(sequence), int(address, 16)
        key = seq, ptr
        if key in rows or filename != f"tap_s{seq}_{ptr:x}.bin":
            raise ValueError("tap index contains an ambiguous address or unexpected filename")
        if observed.get(key) != int(size):
            raise ValueError("tap index does not agree with the successful tap log for this frame")
        path = (folder / filename).resolve()
        if path.parent != folder:
            raise ValueError("tap points outside the capture directory")
        rows[key] = int(size), path
    return rows


def compare_launch(launch, taps, direction):
    src, dst, h, w = struct.unpack("<QQii", launch.params)
    size = h * w * CHANNELS
    if h <= 0 or w <= 0 or h % 4 or w % 4 or h * w % 32 or size > 64 * 1024 * 1024 or src == dst:
        raise ValueError("unsupported repack dimensions, padding, or aliased buffers")
    pair = []
    for ptr in (src, dst):
        stated, path = taps[launch.sequence, ptr]
        if path.stat().st_size != size or stated != size:
            raise ValueError("tap size differs from the observed repack shape")
        pair.append(np.frombuffer(path.read_bytes(), dtype=np.uint8))
    labels = np.arange(size, dtype="<u4").reshape(h, w, CHANNELS)
    flat = labels.reshape(h * w // 16, 4, 4, CHANNELS).transpose(1, 0, 2, 3).reshape(4, h * w // 4, CHANNELS)
    order = [np.frombuffer(pack_tin(grid), dtype="<u4") for grid in (labels, flat)]
    logical = np.empty(size, dtype=np.uint8)
    logical[order[direction]] = pair[0]
    expected = logical[order[1 - direction]]
    return {"entry": launch.name, "sequence": launch.sequence, "height": h, "width": w, "channels": CHANNELS,
            "bytes_compared": size, "differing_bytes": int(np.count_nonzero(expected != pair[1])),
            "input_vs_output_differing_bytes": int(np.count_nonzero(pair[0] != pair[1])),
            "source_sha256": sha(pair[0].tobytes()), "expected_sha256": sha(expected.tobytes()),
            "observed_sha256": sha(pair[1].tobytes())}


def run(folder, frame):
    folder = folder.resolve()
    trace = (folder / "nr-trace.tsv").read_bytes()
    frames = read_trace(trace, kernel_inventory())
    if frame not in frames:
        raise ValueError("trace lacks the requested frame ID")
    taps = tap_index(folder, frame)
    cases = []
    for i, name in enumerate(NAMES):
        candidates = [row for row in frames[frame] if row.name == name]
        if len(candidates) != 1:
            raise ValueError(f"expected exactly one {name} submission")
        cases.append(compare_launch(candidates[0], taps, i))
    return {"schema": "dlssnr_october_live_repack_v1", "frame": frame, "trace_sha256": sha(trace),
            "scope": "Same-launch buffer bytes versus CPU TIN packing; trace and tap provenance is external",
            "source_sha256": sha(Path(__file__).read_bytes()), "cases": cases,
            "passed": all(row["differing_bytes"] == 0 for row in cases)}


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
