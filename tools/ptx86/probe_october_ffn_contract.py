"""Replay October ViT FFN contract from sealed 96-token activations and weights."""

import argparse
from contextlib import ExitStack
import ctypes as C
from functools import partial
import json
from pathlib import Path
import struct
import sys

import numpy as np

from ..align_nr_small_trace import GRAPH, GRAPH_SHA, kernel_inventory, pinned_bytes, read_trace
from ..compare_october_taps import PORT, seal_capture
from ..inspect_nr_small_host_graph import DLL_SHA, PRE_ENTRY
from .check_october_attention import encode_fp8, f16, metrics
from .check_october_attention_tensor import CPU_SPLIT, gpu_split
from .check_october_live_weights import LAYOUT_SHA
from .mma_sm86_tiles import Sm86Mma
from .probe_october_attention_projection import BLOCK, GRID, SHAPE, execute
from .probe_october_fence import Driver, TRIAL_CUBIN_SHA, sha
from .probe_october_ffn_expand import decode_tin, decode_weights, load_ffn_weights, tin_coordinates


ENTRY = "cc_vit_1d_ffn_contract_fp8"
LIVE_ENTRY = "cc_vit_1d_ffn_contract_chained_fp8"
TOKENS, WIDTH, EXPANDED = SHAPE[0] * SHAPE[1], SHAPE[2], 4096
MATRIX_BYTES = WIDTH * EXPANDED


def predict(expanded, seed, weights, coordinates, multiply):
    """Combine four K1024 partials in FP16 with the scaled residual in z=0."""
    total = None
    for partition in range(0, EXPANDED, 1024):
        result = seed if partition == 0 else 0
        for start in range(partition, partition + 1024, 32):
            result = multiply(expanded[..., start:start + 32], weights[:, start:start + 32], result)
        total = result if total is None else f16(total + result)
    return encode_fp8(total[0][coordinates]).reshape(-1)


def load_case(block, sealed, files, launches, weight_base, weight_offset):
    roles = [f"block{block}.{suffix}" for suffix in ("ffn.expand.output", "ffn.expand.input", "qkv.input")]
    rows = [sealed["boundaries"][role] for role in roles]
    data = [files[role].read_bytes() for role in roles]
    for row, raw, width in zip(rows, data, (EXPANDED, WIDTH, WIDTH)):
        if (tuple(row["shape_hwc"]) != (*SHAPE[:2], width) or len(raw) != TOKENS * width
                or sha(raw) != row["sha256"] or np.any((np.frombuffer(raw, np.uint8) & 127) == 127)):
            raise ValueError(f"block {block}: expected intact finite 96-token FFN taps")
    expand, residual, qkv = (launches[row["trace_sequence"]] for row in rows)
    contract = launches[qkv.sequence - 1]
    if ((contract.name, contract.grid, contract.block, contract.shared_memory) != (LIVE_ENTRY, GRID, BLOCK, 0)
            or contract.sequence != expand.sequence + 1 or residual != expand
            or len(contract.params) != 72 or struct.unpack_from("<ii", contract.params, 64) != SHAPE[:2]
            or contract.pointer(0) != expand.pointer(16) or contract.pointer(8) != expand.pointer(0)
            or contract.pointer(16) != qkv.pointer(0)
            or contract.pointer(24) - weight_base != weight_offset):
        raise ValueError(f"block {block}: contract trace does not bind the sealed tensors and layer1 weights")
    return contract, data


def run(capture, dll, trial_cubin, blocks, ordinal):
    inventory = kernel_inventory()
    graph = json.loads(pinned_bytes(GRAPH, GRAPH_SHA))
    sealed, files = seal_capture(capture, inventory, graph)
    if sealed["kind"] != PORT:
        raise ValueError("expected a sealed staged Ampere capture")
    trace = pinned_bytes(capture / "nr-trace.tsv", sealed["artifact_sha256"]["nr-trace.tsv"])
    launches = {row.sequence: row for row in read_trace(trace, inventory)[sealed["frame"]]}
    pre, = (launch for launch in launches.values() if launch.name == PRE_ENTRY)
    weights, offsets = load_ffn_weights(dll, blocks, 1, MATRIX_BYTES + 2 * WIDTH)
    cases, inputs = [], {}
    coordinates = tin_coordinates(TOKENS, WIDTH)
    cubin = pinned_bytes(trial_cubin, TRIAL_CUBIN_SHA)
    with ExitStack() as cleanup:
        device = Driver(ordinal)
        cleanup.callback(device.close)
        copy = device.lib.cuMemcpyHtoD_v2
        copy.argtypes, copy.restype = [C.c_uint64, C.c_void_p, C.c_size_t], C.c_int
        module = device.load(cubin)
        cleanup.callback(device.call, "cuModuleUnload", module)
        function = device.function(module, ENTRY)
        for block in blocks:
            name = f"block{block}.layer1.layer"
            contract, (expanded, residual, expected) = load_case(block, sealed, files, launches, pre.pointer(232), offsets[name])
            weight = weights[block]
            scale = np.frombuffer(weight, "<f2", offset=MATRIX_BYTES)
            seed = f16(decode_tin(residual, TOKENS, WIDTH) * scale)
            zero = encode_fp8(seed[0][coordinates] + np.float32(0)).tobytes()
            cases.append({"block": block, "sequence": contract.sequence,
                          "expanded_input_sha256": sha(expanded), "residual_input_sha256": sha(residual),
                          "weight_sha256": sha(weight),
                          "isolated": execute(device, function, expanded, residual, weight, expected),
                          "zero_expanded": execute(device, function, bytes(TOKENS * EXPANDED), residual, weight, zero)})
            inputs[block] = expanded, seed, weight, expected
    with Sm86Mma(ordinal) as mma:
        hardware = mma.hardware_info
        for row in cases:
            expanded, seed, weight, expected = inputs[row["block"]]
            args = (decode_tin(expanded, TOKENS, EXPANDED), seed, decode_weights(weight, EXPANDED, WIDTH), coordinates)
            for label, multiply in (("cpu", CPU_SPLIT), ("sm86", partial(gpu_split, mma))):
                output = predict(*args, multiply)
                row[f"independent_{label}"] = {**metrics(output, np.frombuffer(expected, np.uint8)),
                                               "output_sha256": sha(output.tobytes())}
    sources = [Path(__file__), *(Path(__file__).with_name(name) for name in
                                 ("probe_october_ffn_expand.py", "probe_october_attention_projection.py", "check_october_attention.py",
                                  "check_october_attention_tensor.py", "mma_sm86_tiles.py"))]
    isolated_exact = all(row["isolated"]["passed"] and row["zero_expanded"]["passed"] for row in cases)
    independent_exact = all(row["independent_sm86"]["different_byte_count"] == 0 for row in cases)
    return {"schema": "october_nr_ffn_contract_replay_v1", "capture_seal_sha256": sealed["capture_seal_sha256"],
            "frame": sealed["frame"], "source_sha256": {path.name: sha(path.read_bytes()) for path in sources},
            "trial_cubin_sha256": TRIAL_CUBIN_SHA, "original_dll_sha256": DLL_SHA,
            "weight_layout_sha256": LAYOUT_SHA, "entry": ENTRY, "grid_xyz": GRID, "block_xyz": BLOCK,
            "shape_hwc": SHAPE, "expanded_channels": EXPANDED, **hardware, "cases": cases,
            "passed": isolated_exact and independent_exact, "independent_bit_exact": independent_exact}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--original-dll", type=Path, required=True)
    parser.add_argument("--trial-cubin", type=Path, required=True)
    parser.add_argument("--blocks", type=int, nargs="+", choices=range(11, 19), default=list(range(11, 19)))
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if len(set(args.blocks)) != len(args.blocks) or args.device < 0:
        parser.error("use distinct block numbers and a nonnegative device ordinal")
    try:
        report = run(args.capture, args.original_dll, args.trial_cubin, args.blocks, args.device)
    except (OSError, ValueError, RuntimeError, KeyError, struct.error) as error:
        report = {"passed": False, "error": str(error)}
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
