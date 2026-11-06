"""Replay October projection on sealed 96-token attention and residual taps."""

import argparse
from contextlib import ExitStack
import ctypes as C
from functools import partial
import json
from mmap import ACCESS_READ, mmap
from pathlib import Path
import struct
import sys

import numpy as np

from ..align_nr_small_trace import GRAPH, GRAPH_SHA, kernel_inventory, pinned_bytes, read_trace
from ..compare_october_taps import PORT, seal_capture
from ..dllq.index import parse_pe, parse_resources
from ..exec_map import record_layout
from ..inspect_nr_small_host_graph import DLL_SHA, PRE_ENTRY, RESOURCE_NAME
from ..inspect_nr_weights import payload_records
from .check_october_attention import FP8, encode_fp8, f16, metrics
from .check_october_attention_tensor import CPU_SPLIT, gpu_split
from .check_october_live_weights import LAYOUT, LAYOUT_SHA
from .mma_sm86_tiles import Sm86Mma
from .probe_october_attention_replay import GUARD
from .probe_october_fence import Driver, TRIAL_CUBIN_SHA, sha


ENTRY = "cc_vit_1d_projection_fp8"
LIVE_ENTRIES = ("cc_vit_1d_projection_chained_fp8", "cc_vit_1d_projection_wait_fp8")
SHAPE = (8, 12, 1024)
TOKENS = SHAPE[0] * SHAPE[1]
ELEMENTS = TOKENS * SHAPE[2]
MATRIX_BYTES = SHAPE[2] ** 2
GRID, BLOCK = (8, 1, 4), (32, 4, 1)


def source_weights(dll, blocks):
    layout = json.loads(pinned_bytes(LAYOUT, LAYOUT_SHA))
    records = {row["name"]: row for row in layout["records"]}
    names = [f"block{block}.layer4.layer" for block in blocks]
    with dll.open("rb") as stream, mmap(stream.fileno(), 0, access=ACCESS_READ) as data:
        if sha(data) != DLL_SHA:
            raise ValueError("expected the pinned original October NR DLL as a weight source")
        sections, rva = parse_pe(data)
        resource, = (row for row in parse_resources(data, sections, rva)
                     if (row["rtype"], row["rname"], row["rlang"]) == ("10", RESOURCE_NAME, "1033"))
        begin = resource["off"]
        with memoryview(data)[begin:begin + resource["size"]] as payload:
            parsed = payload_records(payload)
            weights = {}
            for block, name in zip(blocks, names):
                expected, actual = records[name], parsed[name]
                start, size = actual["payload_offset"], actual["payload_bytes"]
                if size != MATRIX_BYTES + 2 * SHAPE[2] or actual["payload_sha256"] != expected["payload_sha256"]:
                    raise ValueError(f"projection record did not match the pinned layout: {name}")
                weights[block] = bytes(payload[start:start + size])
    starts, _ = record_layout([{"name": name, "C": row["payload_bytes"]} for name, row in records.items()])
    return weights, {name: offset for offset, name in starts.items()}


def load_case(block, sealed, files, launches, weight_base, weight_offset):
    prefix = f"block{block}"
    destination = "repack.input" if block == 18 else "out1"
    roles = [f"{prefix}.{role}" for role in ("attention.out", "qkv.input", destination)]
    rows = [sealed["boundaries"][role] for role in roles]
    raw = [files[role].read_bytes() for role in roles]
    if any(tuple(row["shape_hwc"]) != SHAPE or len(data) != ELEMENTS or sha(data) != row["sha256"]
           for row, data in zip(rows, raw)):
        raise ValueError(f"block {block}: expected intact 96-token projection taps")
    attention, qkv, output = (launches[row["trace_sequence"]] for row in rows)
    projection = launches[attention.sequence + 1]
    if ((projection.name not in LIVE_ENTRIES) or (projection.grid, projection.block, projection.shared_memory) != (GRID, BLOCK, 0)
            or len(projection.params) != 72 or struct.unpack_from("<ii", projection.params, 64) != SHAPE[:2]
            or projection.pointer(0) != attention.pointer(24) or qkv.sequence != attention.sequence - 1
            or projection.pointer(8) != qkv.pointer(0) or projection.pointer(24) - weight_base != weight_offset
            or (block == 18 and (output.sequence != projection.sequence + 1 or projection.pointer(16) != output.pointer(0)))
            or (block != 18 and output != projection)):
        raise ValueError(f"block {block}: projection trace does not bind the sealed tensors and weight record")
    if any(np.any((np.frombuffer(data, np.uint8) & 127) == 127) for data in raw):
        raise ValueError(f"block {block}: nonfinite captured E4M3 values")
    return projection, raw


def projection_coordinates():
    """Map physical TIN byte coordinates to this port's MMA matrix coordinates."""
    channel_tile, lane, byte = np.ogrid[:32, :32, :16]
    # Each half2 scale starts at 2 * (lane & 3). The byte pair selects
    # an N8 tile via bits 1 and 3, and token row via bit 2.
    channels = 32 * channel_tile + 2 * (lane & 3) + (byte & 1) + 8 * ((byte >> 1) & 1) + 16 * (byte >> 3)
    tokens = 16 * np.arange(TOKENS // 16)[:, None, None, None] + (lane >> 2) + 8 * ((byte >> 2) & 1)
    return tokens, channels


def activation_matrix(raw, coordinates):
    matrix = np.empty((1, TOKENS, SHAPE[2]), np.float32)
    matrix[0][coordinates] = FP8[np.frombuffer(raw, np.uint8)].reshape(-1, 32, 32, 16)
    return matrix


def weight_matrix(raw):
    """Decode two N8 tiles per 512 B and the actual K16 register order."""
    row, column = np.ogrid[:SHAPE[2], :SHAPE[2]]
    tile = (row // 32) * (SHAPE[2] // 8) + column // 8
    lane = 4 * (column % 8) + (row % 8) // 2
    byte = 4 * ((row % 32) // 16) + 2 * ((row % 16) // 8) + row % 2
    indices = (tile // 2) * 512 + lane * 16 + (tile % 2) * 8 + byte
    return FP8[np.frombuffer(raw, np.uint8)[indices]][None]


def predict_projection(attention, residual_seed, weights, coordinates, mma):
    """Accumulate four K256 partials and reduce them in FP16 in fence order."""
    total = None
    for partition in range(0, SHAPE[2], 256):
        result = residual_seed if partition == 0 else 0
        for start in range(partition, partition + 256, 32):
            result = mma(attention[..., start:start + 32], weights[:, start:start + 32], result)
        total = result if total is None else f16(total + result)
    return encode_fp8(total[0][coordinates]).reshape(-1)


def execute(device, function, attention, residual, weight, expected):
    fence = struct.pack("<8i", *([-1] * GRID[0]))
    partial = b"\x00\x7e" * ELEMENTS
    poisoned = bytes(np.frombuffer(expected, np.uint8) ^ np.uint8(255))
    inputs = (attention, residual, poisoned, weight, fence, partial)
    with ExitStack() as cleanup:
        addresses = []
        for data in inputs:
            wrapped = GUARD + data + GUARD
            address = C.c_uint64()
            device.call("cuMemAlloc_v2", C.byref(address), len(wrapped))
            cleanup.callback(device.call, "cuMemFree_v2", address)
            device.call("cuMemcpyHtoD_v2", address, C.create_string_buffer(wrapped), len(wrapped))
            addresses.append(address)
        pointers = [pointer.value + len(GUARD) for pointer in addresses]
        abi = C.create_string_buffer(struct.pack("<8Q2i", *pointers, 0, 0, *SHAPE[:2]))
        params = (C.c_void_p * 1)(C.addressof(abi))
        device.call("cuLaunchKernel", function, *GRID, *BLOCK, 0, None, params, None)
        device.call("cuCtxSynchronize")
        readbacks = []
        for pointer, data in zip(addresses, inputs):
            target = C.create_string_buffer(len(data) + 2 * len(GUARD))
            device.call("cuMemcpyDtoH_v2", target, pointer, len(target))
            if target.raw[:len(GUARD)] != GUARD or target.raw[-len(GUARD):] != GUARD:
                raise ValueError("projection crossed an allocation guard")
            readbacks.append(target.raw[len(GUARD):-len(GUARD)])
    if (readbacks[:2] != list(inputs[:2]) or readbacks[3] != weight
            or readbacks[4] != struct.pack("<8i", *([3] * GRID[0]))
            or not np.isfinite(np.frombuffer(readbacks[5], "<f2")).all()):
        raise ValueError("projection changed a source or did not finish the split-K workspace")
    actual = readbacks[2]
    return {"expected_sha256": sha(expected), "actual_sha256": sha(actual), "bytes": len(actual),
            "different_bytes": int(np.count_nonzero(np.frombuffer(actual, np.uint8) != np.frombuffer(expected, np.uint8))),
            "passed": actual == expected}


def run(capture, dll, trial_cubin, blocks, ordinal):
    cubin = pinned_bytes(trial_cubin, TRIAL_CUBIN_SHA)
    graph = json.loads(pinned_bytes(GRAPH, GRAPH_SHA))
    inventory = kernel_inventory()
    sealed, files = seal_capture(capture, inventory, graph)
    if sealed["kind"] != PORT:
        raise ValueError("expected a sealed staged Ampere capture")
    trace = pinned_bytes(capture / "nr-trace.tsv", sealed["artifact_sha256"]["nr-trace.tsv"])
    launches = {launch.sequence: launch for launch in read_trace(trace, inventory)[sealed["frame"]]}
    pre, = (launch for launch in launches.values() if launch.name == PRE_ENTRY)
    weight_base = pre.pointer(232)
    weights, offsets = source_weights(dll, blocks)
    inputs = {block: load_case(block, sealed, files, launches, weight_base, offsets[f"block{block}.layer4.layer"])
              for block in blocks}
    report = {"schema": "october_nr_attention_projection_replay_v1", "capture_seal_sha256": sealed["capture_seal_sha256"],
              "frame": sealed["frame"], "trial_cubin_sha256": TRIAL_CUBIN_SHA, "original_dll_sha256": DLL_SHA,
              "probe_source_sha256": sha(Path(__file__).read_bytes()), "weight_layout_sha256": LAYOUT_SHA,
              "replay_source_sha256": {name: sha(Path(__file__).with_name(name).read_bytes()) for name in
                                       ("check_october_attention.py", "check_october_attention_tensor.py", "mma_sm86_tiles.py")},
              "scope": "Isolated projection and independent matrix replay on sealed 96-token inputs, plus a CPU residual-only oracle.",
              "entry": ENTRY, "shape_hwc": SHAPE, "grid_xyz": GRID, "block_xyz": BLOCK, "cases": []}
    coordinates = projection_coordinates()
    residual_seeds = {}
    with ExitStack() as cleanup:
        device = Driver(ordinal)
        cleanup.callback(device.close)
        copy = device.lib.cuMemcpyHtoD_v2
        copy.argtypes, copy.restype = [C.c_uint64, C.c_void_p, C.c_size_t], C.c_int
        module = device.load(cubin)
        cleanup.callback(device.call, "cuModuleUnload", module)
        function = device.function(module, ENTRY)
        report.update(gpu=device.label, compute_capability=device.capability, cuda_driver_api_version=device.version)
        for block, (projection, (attention, residual, expected)) in inputs.items():
            weight = weights[block]
            full = execute(device, function, attention, residual, weight, expected)
            # z0 seeds the first MMA with the half product. The other partitions add +0.
            scale = np.frombuffer(weight, "<f2", offset=MATRIX_BYTES)
            residual_seeds[block] = f16(activation_matrix(residual, coordinates) * scale)
            zero_expected = encode_fp8(residual_seeds[block][0][coordinates] + np.float32(0)).tobytes()
            zero = execute(device, function, bytes(ELEMENTS), residual, weight, zero_expected)
            report["cases"].append({"block": block, "trace_sequence": projection.sequence,
                                    "live_entry": projection.name, "weight_sha256": sha(weight),
                                    "attention_input_sha256": sha(attention), "residual_input_sha256": sha(residual),
                                    "full": full, "zero_attention": zero})
    with Sm86Mma(ordinal) as mma:
        hardware = partial(gpu_split, mma)
        for row in report["cases"]:
            block = row["block"]
            _, (attention, _, expected) = inputs[block]
            args = (activation_matrix(attention, coordinates), residual_seeds[block], weight_matrix(weights[block]), coordinates)
            for mode, multiply in (("cpu", CPU_SPLIT), ("sm86", hardware)):
                actual = predict_projection(*args, multiply)
                row[f"independent_{mode}"] = {**metrics(actual, np.frombuffer(expected, np.uint8)),
                                               "output_sha256": sha(actual.tobytes())}
    report["passed"] = all(row["full"]["passed"] and row["zero_attention"]["passed"] and
                           row["independent_sm86"]["different_byte_count"] == 0 for row in report["cases"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--original-dll", type=Path, required=True)
    parser.add_argument("--trial-cubin", type=Path, required=True)
    parser.add_argument("--blocks", nargs="+", type=int, choices=range(11, 19), default=list(range(11, 19)))
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if len(set(args.blocks)) != len(args.blocks) or args.device < 0:
        parser.error("use distinct blocks and a nonnegative device ordinal")
    try:
        report = run(args.capture, args.original_dll, args.trial_cubin, args.blocks, args.device)
    except (OSError, ValueError, RuntimeError, KeyError, struct.error) as error:
        report = {"passed": False, "error": str(error)}
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
