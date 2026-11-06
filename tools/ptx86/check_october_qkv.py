"""Replay October 96-token QKV from sealed activations and verified weight records."""

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
from .check_october_attention import FP8, buffer_offsets, encode_fp8, f16, half_literal, metrics, sha
from .check_october_attention_tensor import CPU_SPLIT, gpu_split
from .check_october_live_repack import tap_index
from .check_october_live_weights import LAYOUT, LAYOUT_SHA, run as verify_weights
from .mma_sm86_tiles import Sm86Mma
from .probe_october_attention_replay import GUARD
from .probe_october_fence import Driver, TRIAL_CUBIN_SHA


ENTRY = "cc_vit_1d_qkv_fp8"
TOKENS, WIDTH = 96, 1024
CUDA_GRID, CUDA_BLOCK = (16, 1, 2), (32, 4, 1)
ROLES = ("q", "k", "v")
# Four native E4M3 K values belong to each lane in a K16 half. The rewritten
# half MMA interprets them as the lane's adjacent pair and pair eight higher.
SM86_ORDER = np.array([0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15] +
                      [16, 17, 20, 21, 24, 25, 28, 29, 18, 19, 22, 23, 26, 27, 30, 31])


def input_offsets():
    head, token, canonical = np.ogrid[:32, :TOKENS, :32]
    fragment = np.arange(32)
    canon_by_fragment = 4 * ((fragment % 8) // 2) + 16 * (fragment % 2) + fragment // 8
    fragment = np.argsort(canon_by_fragment)[canonical]
    lane = ((token % 8) << 2) + (fragment % 8) // 2
    byte = fragment // 8 + ((token % 16) // 8) * 4 + (fragment % 2) * 8
    return (token // 16) * 16384 + head * 512 + lane * 16 + byte


def weight_matrix(raw):
    """Decode K32/N8 B fragments, with adjacent N8 tiles packed in one 512 B page."""
    row, col = np.ogrid[:WIDTH, :3 * WIDTH]
    tile = (row // 32) * (3 * WIDTH // 8) + col // 8
    lane = (col % 8) * 4 + (row % 16) // 4
    byte = 4 * ((row % 32) // 16) + row % 4
    return FP8[np.frombuffer(raw, np.uint8)[(tile // 2) * 512 + lane * 16 + (tile % 2) * 8 + byte]]


def norm_sum(values):
    lanes = values.reshape(TOKENS, 32, 4, 4, 2)
    lower = lanes[..., :2, :, :].astype(np.float64)
    upper = lanes[..., 2:, :, :]
    # The staged SASS contracts each lower square with the rounded upper square.
    paired = f16(lower * lower + f16(upper * upper))
    local = f16(paired[..., 0, :, :] + paired[..., 1, :, :])
    shuffled = f16(f16(local[..., 0, :] + local[..., 2, :]) + f16(local[..., 1, :] + local[..., 3, :]))
    return f16(shuffled[..., :1] + shuffled[..., 1:])


def norm_inverse(values):
    return f16(1 / np.sqrt(np.maximum(norm_sum(values), half_literal(0x388205FF))))


def project_qkv(activation, weights, mma):
    matrix = weight_matrix(weights[128:])[None]
    value = FP8[np.frombuffer(activation, np.uint8)[input_offsets()]].transpose(1, 0, 2).reshape(1, TOKENS, WIDTH)
    halves = []
    for split in range(0, WIDTH, 512):
        acc = 0
        for start in range(split, split + 512, 32):
            acc = mma(value[..., start:start + 32], matrix[:, start:start + 32], acc)
        halves.append(acc)
    return f16(halves[0] + halves[1]).reshape(TOKENS, 32, 3, 32)


def predict(activation, weights, mma):
    projected = project_qkv(activation, weights, mma)
    q, k, v = (projected[:, :, i, :] for i in range(3))
    tau = f16(np.frombuffer(weights[:128], "<f4"))[None, :, None]
    q = f16(f16(f16(q * norm_inverse(q)) * np.float16(np.sqrt(32))) * tau)
    k = f16(k * norm_inverse(k))
    return [encode_fp8(x.transpose(1, 0, 2)) for x in (q, k, v)]


def physical_split(mma, a, b, acc):
    return gpu_split(mma, a[..., SM86_ORDER], b[:, SM86_ORDER, :], acc)


def isolated(device, function, shape, raw, weights):
    activation, *expected = raw
    poison = [bytes(np.frombuffer(output, np.uint8) ^ np.uint8(255)) for output in expected]
    # Each CTA(x) uses a private signed split-K sequence counter. z=1
    # waits for z=0's release of value 0 before it reads the three half planes.
    counters = b"\xff" * (4 * CUDA_GRID[0])
    scratch = bytes(6 * len(activation))
    payloads = [activation, *poison, weights, counters, scratch]
    with ExitStack() as cleanup:
        pointers = []
        for data in payloads:
            wrapped = GUARD + data + GUARD
            pointer = C.c_uint64()
            device.call("cuMemAlloc_v2", C.byref(pointer), len(wrapped))
            cleanup.callback(device.call, "cuMemFree_v2", pointer)
            device.call("cuMemcpyHtoD_v2", pointer, C.create_string_buffer(wrapped), len(wrapped))
            pointers.append(pointer)
        active = [p.value + len(GUARD) for p in pointers]
        blob = C.create_string_buffer(struct.pack("<9Qii", *active, 0, 0, *shape[:2]))
        args = (C.c_void_p * 1)(C.addressof(blob))
        device.call("cuLaunchKernel", function, *CUDA_GRID, *CUDA_BLOCK, 0, None, args, None)
        device.call("cuCtxSynchronize")
        results = []
        for pointer, data in zip(pointers, payloads):
            readback = C.create_string_buffer(len(data) + 2 * len(GUARD))
            device.call("cuMemcpyDtoH_v2", readback, pointer, len(readback))
            if readback.raw[:len(GUARD)] != GUARD or readback.raw[-len(GUARD):] != GUARD:
                raise ValueError("isolated QKV wrote across a buffer guard")
            results.append(readback.raw[len(GUARD):-len(GUARD)])
    if results[0] != activation or results[4] != weights:
        raise ValueError("isolated QKV changed its input or weights")
    if set(np.frombuffer(results[5], "<i4")) != {1}:
        raise ValueError("isolated QKV split-K completion counters differ")
    return {name: {"bytes": len(sealed), "output_sha256": sha(actual),
                   "different_bytes": int(np.count_nonzero(np.frombuffer(sealed, np.uint8) != np.frombuffer(actual, np.uint8)))}
            for name, sealed, actual in zip(ROLES, expected, results[1:4])}


def input_case(block, sealed, files, launches):
    roles = (f"block{block}.qkv.input", *(f"block{block}.attention.in{i}" for i in range(3)))
    rows = [sealed["boundaries"][role] for role in roles]
    raw = [files[role].read_bytes() for role in roles]
    shape = tuple(rows[0]["shape_hwc"])
    qkv = launches[rows[0]["trace_sequence"]]
    if ((qkv.name, qkv.grid, qkv.block, qkv.shared_memory) !=
            ("cc_vit_1d_qkv_chained_fp8", CUDA_GRID, CUDA_BLOCK, 0) or shape[0] * shape[1] != TOKENS or
            shape[2] != WIDTH or any(tuple(row["shape_hwc"]) != shape or len(data) != TOKENS * WIDTH or
                                     sha(data) != row["sha256"] for row, data in zip(rows, raw))):
        raise ValueError(f"block {block} requires intact 96-token October chained-QKV taps")
    if any(np.any((np.frombuffer(data, np.uint8) & 127) == 127) for data in raw):
        raise ValueError(f"block {block} contains non-finite E4M3 activations")
    return shape, qkv.sequence, raw


def run(capture, weight_capture, blocks, cubin_path, gpu_mma):
    inventory = kernel_inventory()
    graph = json.loads(pinned_bytes(GRAPH, GRAPH_SHA))
    weight_capture = weight_capture.resolve()
    sealed, files = seal_capture(capture, inventory, graph)
    if sealed["kind"] != PORT:
        raise ValueError("expected a sealed October staged Ampere capture")
    trace = (capture / "nr-trace.tsv").read_bytes()
    if sha(trace) != sealed["artifact_sha256"]["nr-trace.tsv"]:
        raise ValueError("trace changed after sealing")
    launches = {row.sequence: row for row in read_trace(trace, inventory)[sealed["frame"]]}
    weights_check = verify_weights(weight_capture, 0)
    if not weights_check["passed"]:
        raise ValueError("packed weight capture did not match the pinned October records")
    records = {row["name"]: row for row in weights_check["records"]}
    layout = {row["name"]: row for row in json.loads(pinned_bytes(LAYOUT, LAYOUT_SHA))["records"]}
    weight_rows = read_trace((weight_capture / "nr-trace.tsv").read_bytes(), inventory)[0]
    pre = next(row for row in weight_rows if row.sequence == weights_check["pre_sequence"])
    _, packed = tap_index(weight_capture, 0)[pre.sequence, pre.pointer(232)]
    cases = []
    cubin = cubin_path.read_bytes() if cubin_path else None
    if cubin is not None and sha(cubin) != TRIAL_CUBIN_SHA:
        raise ValueError("not the pinned October module 6 sm86 trial cubin")
    offsets = buffer_offsets(TOKENS)[:3]
    with ExitStack() as cleanup, packed.open("rb") as stream:
        hardware = None
        if cubin is not None:
            device = Driver(0)
            cleanup.callback(device.close)
            copy = device.lib.cuMemcpyHtoD_v2
            copy.argtypes, copy.restype = [C.c_uint64, C.c_void_p, C.c_size_t], C.c_int
            module = device.load(cubin)
            cleanup.callback(device.call, "cuModuleUnload", module)
            function = device.function(module, ENTRY)
            hardware = {"gpu": device.label, "compute_capability": device.capability, "cuda_driver_api_version": device.version}
        inputs = []
        for block in blocks:
            shape, sequence, raw = input_case(block, sealed, files, launches)
            name = f"block{block}.layer2.layer"
            record = records[name]
            stream.seek(record["gpu_buffer_offset"])
            weight = stream.read(record["bytes"])
            if len(weight) != 128 + 3 * WIDTH ** 2 or sha(weight) != layout[name]["payload_sha256"]:
                raise ValueError(f"block {block} QKV weight record changed after verification")
            observed = [np.frombuffer(data, np.uint8)[idx] for data, idx in zip(raw[1:], offsets)]
            inputs.append((raw, weight, observed))
            case = {"block": block, "sequence": sequence, "shape_hwc": shape, "input_sha256": sha(raw[0]),
                    "weight_sha256": sha(weight), "observed_sha256": {key: sha(data) for key, data in zip(ROLES, raw[1:])},
                    "cpu": {key: metrics(a, b) for key, a, b in zip(ROLES, predict(raw[0], weight, CPU_SPLIT), observed)}}
            if cubin is not None:
                case["isolated"] = isolated(device, function, shape, raw, weight)
            cases.append(case)
    if gpu_mma:
        with Sm86Mma() as mma:
            hardware = mma.hardware_info
            for case, (raw, weight, observed) in zip(cases, inputs):
                expected = predict(raw[0], weight, partial(physical_split, mma))
                case["independent_sm86"] = {key: metrics(a, b) for key, a, b in zip(ROLES, expected, observed)}
    return {"schema": "october_qkv_replay_v1", "source_sha256": sha(Path(__file__).read_bytes()),
            "capture_seal_sha256": sealed["capture_seal_sha256"], "frame": sealed["frame"],
            "packed_weights_sha256": weights_check["gpu_packed_sha256"], "layout_sha256": LAYOUT_SHA,
            "hardware": hardware, "trial_cubin_sha256": TRIAL_CUBIN_SHA if cubin is not None else None,
            "entry": ENTRY if cubin is not None else None, "grid_xyz": CUDA_GRID, "block_xyz": CUDA_BLOCK, "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("--trial-cubin", type=Path)
    parser.add_argument("--gpu-mma", action="store_true")
    parser.add_argument("--blocks", type=int, nargs="+", choices=range(11, 19), default=list(range(11, 19)))
    args = parser.parse_args()
    if len(set(args.blocks)) != len(args.blocks):
        parser.error("select each block at most once")
    try:
        result = run(args.capture, args.weights, args.blocks, args.trial_cubin, args.gpu_mma)
    except (OSError, ValueError, RuntimeError, KeyError, struct.error) as error:
        print(json.dumps({"error": str(error)}, indent=2))
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
