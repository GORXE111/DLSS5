"""Replay six aligned ViT token tiles from sealed 4K FFN inputs on sm86."""

import argparse
from functools import partial
import json
from pathlib import Path
import sys

import numpy as np

from ..align_nr_small_trace import GRAPH, GRAPH_SHA, kernel_inventory, pinned_bytes, read_trace
from ..compare_october_taps import PORT, seal_capture
from ..inspect_nr_small_host_graph import DLL_SHA, PRE_ENTRY
from .check_october_attention import f16, metrics
from .check_october_attention_tensor import gpu_split
from .check_october_live_weights import LAYOUT_SHA
from .mma_sm86_tiles import Sm86Mma
from .probe_october_fence import sha
from . import probe_october_ffn_expand as expand, probe_october_ffn_contract as contract


SHAPE = (36, 60, 1024)
STORAGE_TOKENS = 2176
TILES = (0, 15, 16, 127, 128, 134)


def gather(block, suffix, sealed, files):
    role = f"block{block}.{suffix}"
    boundary = sealed["boundaries"][role]
    raw = files[role].read_bytes()
    width = SHAPE[2]
    if (tuple(boundary["shape_hwc"]) != SHAPE or boundary["storage_tokens"] != STORAGE_TOKENS
            or len(raw) != STORAGE_TOKENS * width or sha(raw) != boundary["sha256"]):
        raise ValueError(f"{role}: expected intact padded 4K activation")
    tiles = np.frombuffer(raw, np.uint8).reshape(-1, 16 * width)[list(TILES)]
    if np.any((tiles & 127) == 127):
        raise ValueError(f"{role}: nonfinite E4M3 code in selected tiles")
    return boundary, tiles.tobytes()


def run(capture, dll, ordinal):
    inventory = kernel_inventory()
    graph = json.loads(pinned_bytes(GRAPH, GRAPH_SHA))
    sealed, files = seal_capture(capture, inventory, graph)
    if sealed["kind"] != PORT:
        raise ValueError("expected a sealed staged Ampere capture")
    trace = pinned_bytes(capture / "nr-trace.tsv", sealed["artifact_sha256"]["nr-trace.tsv"])
    launches = {row.sequence: row for row in read_trace(trace, inventory)[sealed["frame"]]}
    pre, = (row for row in launches.values() if row.name == PRE_ENTRY)
    weight_base = pre.pointer(232)
    blocks = (12, 18)
    first, offsets = expand.load_ffn_weights(dll, blocks, 0, expand.RECORD_BYTES)
    second, _ = expand.load_ffn_weights(dll, blocks, 1, contract.MATRIX_BYTES + 2 * contract.WIDTH)
    coordinates = expand.tin_coordinates(expand.TOKENS, contract.WIDTH)
    cases = []
    with Sm86Mma(ordinal) as mma:
        hardware = mma.hardware_info
        multiply = partial(gpu_split, mma)
        for block in blocks:
            source, activation = gather(block - 1, "out1", sealed, files)
            target, observed = gather(block, "qkv.input", sealed, files)
            prev, qkv = (launches[row["trace_sequence"]] for row in (source, target))
            step1, step2 = launches[qkv.sequence - 2], launches[qkv.sequence - 1]
            if (prev.sequence + 1 != step1.sequence or step1.name not in expand.LIVE_ENTRIES
                    or step2.name != contract.LIVE_ENTRY or prev.pointer(16) != step1.pointer(0)
                    or step1.pointer(16) != step2.pointer(0) or step1.pointer(0) != step2.pointer(8)
                    or step2.pointer(16) != qkv.pointer(0)
                    or step1.pointer(24) - weight_base != offsets[f"block{block}.layer0.layer"]
                    or step2.pointer(24) - weight_base != offsets[f"block{block}.layer1.layer"]):
                raise ValueError(f"block {block}: trace does not bind both FFN operands and records")
            a = expand.decode_tin(activation, expand.TOKENS, expand.INPUTS)
            middle = expand.predict(a, expand.decode_weights(first[block], expand.INPUTS, expand.OUTPUTS), multiply)
            scale = np.frombuffer(second[block], "<f2", offset=contract.MATRIX_BYTES)
            args = (expand.decode_tin(middle.tobytes(), expand.TOKENS, expand.OUTPUTS), f16(a * scale),
                    expand.decode_weights(second[block], expand.OUTPUTS, expand.INPUTS), coordinates)
            output = contract.predict(*args, multiply)
            cases.append({"block": block, "source_sha256": source["sha256"], "target_sha256": target["sha256"],
                          "input_tiles_sha256": sha(activation), "expand_tiles_sha256": sha(middle.tobytes()),
                          "contract_tiles_sha256": sha(output.tobytes()), "observed_tiles_sha256": sha(observed),
                          "weights_sha256": [sha(first[block]), sha(second[block])], "bytes": len(observed),
                          **metrics(output, np.frombuffer(observed, np.uint8))})
    names = (Path(__file__).name, "probe_october_ffn_expand.py", "probe_october_ffn_contract.py",
             "check_october_attention.py", "check_october_attention_tensor.py", "mma_sm86_tiles.py")
    return {"schema": "october_nr_ffn_4k_tiles_v1", "capture_seal_sha256": sealed["capture_seal_sha256"],
            "frame": sealed["frame"], "shape_hwc": SHAPE, "storage_tokens": STORAGE_TOKENS,
            "source_sha256": {name: sha(Path(__file__).with_name(name).read_bytes()) for name in names},
            "original_dll_sha256": DLL_SHA, "weight_layout_sha256": LAYOUT_SHA,
            "tiles_m16": TILES, "cases": cases, **hardware,
            "passed": all(row["different_byte_count"] == 0 for row in cases)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--original-dll", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.device < 0:
        parser.error("use a nonnegative device ordinal")
    try:
        report = run(args.capture, args.original_dll, args.device)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        report = {"passed": False, "error": str(error)}
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
