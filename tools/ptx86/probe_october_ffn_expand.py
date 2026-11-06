"""Replay the October Small FP8 ViT FFN expansion on sealed 96-token taps."""

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
from .check_october_attention import FP8, encode_fp8, f16, half_literal, metrics
from .check_october_attention_tensor import CPU_SPLIT, gpu_split
from .check_october_live_weights import LAYOUT, LAYOUT_SHA
from .mma_sm86_tiles import Sm86Mma
from .probe_october_attention_replay import GUARD
from .probe_october_fence import Driver, TRIAL_CUBIN_SHA, sha


ENTRY = "cc_vit_1d_ffn_expand_fp8"
LIVE_ENTRIES = (ENTRY, "cc_vit_1d_ffn_expand_publish_fp8", "cc_vit_1d_ffn_expand_chained_fp8")
SHAPE = (8, 12)
TOKENS = SHAPE[0] * SHAPE[1]
INPUTS, OUTPUTS = 1024, 4096
RECORD_BYTES = INPUTS * OUTPUTS + 16
GRID, BLOCK = (32, 1, 1), (32, 4, 1)
# MpCubicSiluActivation constants, as FP32 literals converted to FP16 by the entry.
POLY_BITS = (0xBD650000, 0x3EE50000, 0x3F650000)


def tin_coordinates(tokens, width):
    """Physical TIN storage [M16, N32, lane, byte] to logical token and channel."""
    tile_m, tile_n, lane, byte = np.ogrid[:tokens // 16, :width // 32, :32, :16]
    token = 16 * tile_m + (lane >> 2) + 8 * ((byte >> 2) & 1)
    channel = 32 * tile_n + 2 * (lane & 3) + (byte & 1) + 8 * ((byte >> 1) & 1) + 16 * (byte >> 3)
    return token, channel


def decode_tin(raw, tokens, width):
    matrix = np.empty((1, tokens, width), np.float32)
    matrix[0][tin_coordinates(tokens, width)] = FP8[np.frombuffer(raw, np.uint8)].reshape(tokens // 16, width // 32, 32, 16)
    return matrix


def decode_weights(raw, inputs, outputs):
    """Decode packed N8 weight tiles in the K16 lane order of the sm86 rewrite."""
    row, column = np.ogrid[:inputs, :outputs]
    tile = (row // 32) * (outputs // 8) + column // 8
    lane = 4 * (column % 8) + (row % 8) // 2
    byte = 4 * ((row % 32) // 16) + 2 * ((row % 16) // 8) + row % 2
    index = (tile // 2) * 512 + lane * 16 + (tile % 2) * 8 + byte
    return FP8[np.frombuffer(raw, np.uint8)[index]][None]


def load_ffn_weights(dll, blocks, layer, record_bytes):
    layout = json.loads(pinned_bytes(LAYOUT, LAYOUT_SHA))
    records = {row["name"]: row for row in layout["records"]}
    names = [f"block{block}.layer{layer}.layer" for block in blocks]
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
                row = parsed[name]
                start, size = row["payload_offset"], row["payload_bytes"]
                if size != record_bytes or row["payload_sha256"] != records[name]["payload_sha256"]:
                    raise ValueError(f"FFN record did not match the pinned layout: {name}")
                weights[block] = bytes(payload[start:start + size])
    starts, _ = record_layout([{"name": name, "C": row["payload_bytes"]} for name, row in records.items()])
    return weights, {name: offset for offset, name in starts.items()}


def cubic_silu(projected):
    # Both polynomial operations are fused and rounded once to binary16.
    a, b, c = map(half_literal, POLY_BITS)
    clipped = projected.clip(-4, 4).astype(np.float64)
    slope = f16(a * np.abs(clipped) + b)
    gain = f16(clipped * slope + c)
    return f16(projected * gain)


def predict(activation, weights, mma):
    result = 0
    for start in range(0, INPUTS, 32):
        result = mma(activation[..., start:start + 32], weights[:, start:start + 32], result)
    return encode_fp8(cubic_silu(result)[0][tin_coordinates(TOKENS, OUTPUTS)]).reshape(-1)


def load_case(block, sealed, files, launches, weight_base, weight_offset):
    roles = [f"block{block}.ffn.expand.{name}" for name in ("input", "output")]
    rows = [sealed["boundaries"][role] for role in roles]
    raw = [files[role].read_bytes() for role in roles]
    entry = launches[rows[0]["trace_sequence"]]
    for row, data, width in zip(rows, raw, (INPUTS, OUTPUTS)):
        if (tuple(row["shape_hwc"]) != (*SHAPE, width) or len(data) != TOKENS * width
                or row["trace_sequence"] != entry.sequence or sha(data) != row["sha256"]):
            raise ValueError(f"block {block}: expected intact 96-token FFN expand taps from one launch")
        if np.any((np.frombuffer(data, np.uint8) & 127) == 127):
            raise ValueError(f"block {block}: nonfinite captured E4M3 value")
    contract = launches[entry.sequence + 1]
    if (entry.name not in LIVE_ENTRIES or (entry.grid, entry.block, entry.shared_memory) != (GRID, BLOCK, 0)
            or len(entry.params) != 72 or struct.unpack_from("<ii", entry.params, 64) != SHAPE
            or rows[0]["parameter_offset"] != 0 or rows[1]["parameter_offset"] != 16
            or entry.pointer(24) - weight_base != weight_offset or not contract.name.startswith("cc_vit_1d_ffn_contract_")
            or entry.pointer(16) != contract.pointer(0) or entry.pointer(0) != contract.pointer(8)):
        raise ValueError(f"block {block}: expansion trace does not bind the sealed tensors and weight record")
    return entry, raw


def execute(device, function, activation, expected, weight):
    poisoned = bytes(np.frombuffer(expected, np.uint8) ^ np.uint8(255))
    inputs = activation, poisoned, weight
    with ExitStack() as cleanup:
        pointers = []
        for data in inputs:
            wrapped = GUARD + data + GUARD
            pointer = C.c_uint64()
            device.call("cuMemAlloc_v2", C.byref(pointer), len(wrapped))
            cleanup.callback(device.call, "cuMemFree_v2", pointer)
            device.call("cuMemcpyHtoD_v2", pointer, C.create_string_buffer(wrapped), len(wrapped))
            pointers.append(pointer)
        source, output, weights = (p.value + len(GUARD) for p in pointers)
        blob = C.create_string_buffer(struct.pack("<8Qii", source, 0, output, weights, 0, 0, 0, 0, *SHAPE))
        args = (C.c_void_p * 1)(C.addressof(blob))
        device.call("cuLaunchKernel", function, *GRID, *BLOCK, 0, None, args, None)
        device.call("cuCtxSynchronize")
        results = []
        for pointer, data in zip(pointers, inputs):
            buffer = C.create_string_buffer(len(data) + 2 * len(GUARD))
            device.call("cuMemcpyDtoH_v2", buffer, pointer, len(buffer))
            readback = buffer.raw
            if readback[:len(GUARD)] != GUARD or readback[-len(GUARD):] != GUARD:
                raise ValueError("isolated FFN expansion crossed an allocation guard")
            results.append(readback[len(GUARD):-len(GUARD)])
    if results[0] != activation or results[2] != weight:
        raise ValueError("isolated FFN expansion changed its input or weights")
    return results[1]


def run(capture, dll, trial_cubin, blocks, ordinal):
    cubin = pinned_bytes(trial_cubin, TRIAL_CUBIN_SHA)
    graph = json.loads(pinned_bytes(GRAPH, GRAPH_SHA))
    inventory = kernel_inventory()
    sealed, files = seal_capture(capture, inventory, graph)
    if sealed["kind"] != PORT:
        raise ValueError("expected a sealed staged Ampere capture")
    trace = pinned_bytes(capture / "nr-trace.tsv", sealed["artifact_sha256"]["nr-trace.tsv"])
    launches = {row.sequence: row for row in read_trace(trace, inventory)[sealed["frame"]]}
    pre, = (launch for launch in launches.values() if launch.name == PRE_ENTRY)
    weights, offsets = load_ffn_weights(dll, blocks, 0, RECORD_BYTES)
    inputs = {block: load_case(block, sealed, files, launches, pre.pointer(232), offsets[f"block{block}.layer0.layer"])
              for block in blocks}
    cases = {}
    with ExitStack() as cleanup:
        device = Driver(ordinal)
        cleanup.callback(device.close)
        copy = device.lib.cuMemcpyHtoD_v2
        copy.argtypes, copy.restype = [C.c_uint64, C.c_void_p, C.c_size_t], C.c_int
        module = device.load(cubin)
        cleanup.callback(device.call, "cuModuleUnload", module)
        function = device.function(module, ENTRY)
        for block, (entry, (activation, observed)) in inputs.items():
            isolated = execute(device, function, activation, observed, weights[block])
            cases[block] = {"block": block, "trace_sequence": entry.sequence, "live_entry": entry.name,
                            "input_sha256": sha(activation), "weight_sha256": sha(weights[block]),
                            "output_bytes": len(observed), "sealed_output_sha256": sha(observed),
                            "isolated_output_sha256": sha(isolated), "isolated_different_bytes": int(np.count_nonzero(
                                np.frombuffer(isolated, np.uint8) != np.frombuffer(observed, np.uint8)))}
    with Sm86Mma(ordinal) as mma:
        hardware = mma.hardware_info
        for block, (_, (activation, observed)) in inputs.items():
            a, b = decode_tin(activation, TOKENS, INPUTS), decode_weights(weights[block], INPUTS, OUTPUTS)
            for label, multiply in (("cpu", CPU_SPLIT), ("sm86", partial(gpu_split, mma))):
                prediction = predict(a, b, multiply)
                cases[block][label] = {**metrics(prediction, np.frombuffer(observed, np.uint8)),
                                       "output_sha256": sha(prediction.tobytes())}
    source_hashes = {name: sha(Path(__file__).with_name(name).read_bytes()) for name in
                     (Path(__file__).name, "check_october_attention.py", "check_october_attention_tensor.py", "mma_sm86_tiles.py")}
    return {"schema": "october_nr_ffn_expand_replay_v1", "capture_seal_sha256": sealed["capture_seal_sha256"],
            "frame": sealed["frame"], "source_sha256": source_hashes, "weight_layout_sha256": LAYOUT_SHA,
            "original_dll_sha256": DLL_SHA, "trial_cubin_sha256": TRIAL_CUBIN_SHA,
            "entry": ENTRY, "grid_xyz": GRID, "block_xyz": BLOCK, "tokens": TOKENS,
            "weight_shape_kn": [INPUTS, OUTPUTS], "weight_record_bytes": RECORD_BYTES,
            "poly_f32_literal_bits": [f"0x{x:08x}" for x in POLY_BITS],
            **hardware, "cases": list(cases.values()),
            "passed": all(x["isolated_different_bytes"] == 0 and x["sm86"]["different_byte_count"] == 0
                          for x in cases.values())}


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
