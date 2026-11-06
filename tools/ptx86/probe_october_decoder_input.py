"""Replay October block 19 on the sealed 96-token ViT and encoder skip taps."""

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
from .check_october_attention import FP8, encode_fp8, f16, metrics, sha
from .check_october_attention_tensor import CPU_SPLIT, gpu_split
from .check_october_live_weights import LAYOUT_SHA
from .mma_sm86_tiles import Sm86Mma
from .probe_october_attention_replay import GUARD
from .probe_october_fence import Driver
from .probe_october_ffn_expand import decode_weights, load_ffn_weights


ENTRY = "cc_dec_input_upsample_1024_512_fp8"
LIVE_ENTRY = "cc_dec_input_upsample_1024_512_tilesync_fp8"
CUBIN_SHA = "44a6963cfc3c99ebc7ad83e28495af8659a1fd9e19013d5dc8838bf0db528e01"
GRID, BLOCK = (6, 2, 4), (32, 2, 1)
LOW, HIGH = (8, 12, 1024), (12, 20, 512)
ROLES = ("block18.out0", "block10.out1", "block19.out1")
WEIGHT_BYTES = LOW[2] * HIGH[2]
COUNTERS = GRID[0] * GRID[1]
PARTIAL_BYTES = LOW[0] * LOW[1] * HIGH[2] * 2


def coordinates(shape):
    """Map 2D TIN storage to spatial rows and sm86 MMA-order channels."""
    h, w, channels = shape
    tile_y, tile_x, tile_c, lane, byte = np.ogrid[:h // 4, :w // 4, :channels // 32, :32, :16]
    token = (lane >> 2) + 8 * ((byte >> 2) & 1)
    channel = 32 * tile_c + 2 * (lane & 3) + (byte & 1) + 8 * ((byte >> 1) & 1) + 16 * (byte >> 3)
    return 4 * tile_y + token // 4, 4 * tile_x + token % 4, channel


def decode(raw, shape):
    h, w, channels = shape
    out = np.empty(shape, dtype=np.float32)
    out[coordinates(shape)] = FP8[np.frombuffer(raw, np.uint8)].reshape(h // 4, w // 4, channels // 32, 32, 16)
    return out


def scaled_skip(raw, weights):
    return f16(decode(raw, HIGH) * np.frombuffer(weights, "<f2", offset=WEIGHT_BYTES))


def encode(values):
    return encode_fp8(values[coordinates(HIGH)]).reshape(-1).tobytes()


def project(low, weights, multiply):
    h, w = HIGH[0] // 2, HIGH[1] // 2
    channels = LOW[2]
    activation = decode(low, LOW)[:h, :w].reshape(-1, channels)
    activation = np.pad(activation, ((0, (-len(activation)) % 16), (0, 0)))[None]
    matrix = decode_weights(weights, channels, HIGH[2])
    total = None
    for split in range(0, channels, 256):
        partial_sum = 0
        for start in range(split, split + 256, 32):
            partial_sum = multiply(activation[..., start:start + 32], matrix[:, start:start + 32], partial_sum)
        total = partial_sum if total is None else f16(total + partial_sum)
    return total[0, :h * w].reshape(h, w, HIGH[2]).repeat(2, axis=0).repeat(2, axis=1)


def merge(nearest, skip, weights):
    # Ptxas contracts the half skip-scale multiply and projection add into HFMA2.
    scale = np.frombuffer(weights, "<f2", offset=WEIGHT_BYTES)
    return encode(f16(decode(skip, HIGH).astype(np.float64) * scale + nearest))


def load_case(capture, dll):
    inventory = kernel_inventory()
    graph = json.loads(pinned_bytes(GRAPH, GRAPH_SHA))
    sealed, files = seal_capture(capture, inventory, graph)
    if sealed["kind"] != PORT:
        raise ValueError("expected a sealed experimental sm86 capture")
    trace = pinned_bytes(capture / "nr-trace.tsv", sealed["artifact_sha256"]["nr-trace.tsv"])
    launches = {row.sequence: row for row in read_trace(trace, inventory)[sealed["frame"]]}
    tap_rows = [sealed["boundaries"][role] for role in ROLES]
    low, skip, decoder = (launches[row["trace_sequence"]] for row in tap_rows)
    if (tuple(tuple(row["shape_hwc"]) for row in tap_rows) != (LOW, HIGH, HIGH)
            or any(row["layout"] != "tin2d" for row in tap_rows)
            or [row["parameter_offset"] for row in tap_rows] != [8, 16, 16]
            or decoder.name != LIVE_ENTRY or decoder.grid != GRID or decoder.block != BLOCK or decoder.shared_memory
            or len(decoder.params) != 80 or struct.unpack_from("<4i", decoder.params, 64) != (*LOW[:2], *HIGH[:2])
            or decoder.pointer(0) != low.pointer(8) or decoder.pointer(8) != skip.pointer(16)):
        raise ValueError("sealed taps do not bind the supported 96-token decoder ABI")
    records, offsets = load_ffn_weights(dll, (19,), 0, WEIGHT_BYTES + HIGH[2] * 2)
    pre, = (row for row in launches.values() if row.name == PRE_ENTRY)
    if decoder.pointer(56) - pre.pointer(232) != offsets["block19.layer0.layer"]:
        raise ValueError("decoder weight pointer differs from the pinned record")
    raw = [files[role].read_bytes() for role in ROLES]
    if any(sha(value) != row["sha256"] or len(value) != row["bytes"] for value, row in zip(raw, tap_rows)):
        raise ValueError("capture changed after sealing")
    if any(np.any((np.frombuffer(value, np.uint8) & 127) == 127) for value in raw):
        raise ValueError("decoder taps contain nonfinite E4M3 values")
    return sealed, decoder, raw, records[19]


def occupancy(device, function):
    query = device.lib.cuOccupancyMaxActiveBlocksPerMultiprocessor
    query.argtypes = [C.POINTER(C.c_int), C.c_void_p, C.c_int, C.c_size_t]
    query.restype = C.c_int
    blocks_per_sm, sms = C.c_int(), C.c_int()
    device.call("cuOccupancyMaxActiveBlocksPerMultiprocessor", C.byref(blocks_per_sm), function, 64, 0)
    device.call("cuDeviceGetAttribute", C.byref(sms), 16, device.device)
    resident = blocks_per_sm.value * sms.value
    if resident < COUNTERS * GRID[2]:
        raise ValueError("the bounded split-K launch must fit in the measured resident block capacity")
    return {"multiprocessors": sms.value, "active_blocks_per_multiprocessor": blocks_per_sm.value,
            "resident_capacity": resident, "blocks_launched": COUNTERS * GRID[2]}


def execute(device, function, low, skip, weights, expected):
    poisoned = bytes(np.frombuffer(expected, np.uint8) ^ np.uint8(255))
    counters = struct.pack(f"<{COUNTERS}i", *([-1] * COUNTERS))
    scratch = b"\x00\x7e" * (PARTIAL_BYTES // 2)
    payloads = (low, skip, poisoned, counters, scratch, weights)
    with ExitStack() as cleanup:
        pointers = []
        for payload in payloads:
            wrapped = GUARD + payload + GUARD
            pointer = C.c_uint64()
            device.call("cuMemAlloc_v2", C.byref(pointer), len(wrapped))
            cleanup.callback(device.call, "cuMemFree_v2", pointer)
            device.call("cuMemcpyHtoD_v2", pointer, C.create_string_buffer(wrapped), len(wrapped))
            pointers.append(pointer)
        a, b, output, fence, partial_sum, weight = (pointer.value + len(GUARD) for pointer in pointers)
        abi = C.create_string_buffer(struct.pack("<8Q4i", a, b, output, 0, fence, 0, partial_sum, weight,
                                               *LOW[:2], *HIGH[:2]))
        args = (C.c_void_p * 1)(C.addressof(abi))
        device.call("cuLaunchKernel", function, *GRID, *BLOCK, 0, None, args, None)
        device.call("cuCtxSynchronize")
        readbacks = []
        for pointer, payload in zip(pointers, payloads):
            buf = C.create_string_buffer(len(payload) + 2 * len(GUARD))
            device.call("cuMemcpyDtoH_v2", buf, pointer, len(buf))
            raw = buf.raw
            if raw[:len(GUARD)] != GUARD or raw[-len(GUARD):] != GUARD:
                raise ValueError("isolated decoder crossed an allocation guard")
            readbacks.append(raw[len(GUARD):-len(GUARD)])
    if readbacks[:2] != [low, skip] or readbacks[-1] != weights:
        raise ValueError("isolated decoder changed its inputs or weights")
    if struct.unpack(f"<{COUNTERS}i", readbacks[3]) != (3,) * COUNTERS:
        raise ValueError("decoder split-K counter did not complete all four partitions")
    if not np.isfinite(np.frombuffer(readbacks[4], "<f2")).all():
        raise ValueError("decoder split-K partial scratch was not fully written")
    return readbacks[2]


def compare(predicted, observed):
    return {"predicted_sha256": sha(predicted), "observed_sha256": sha(observed), "bytes_compared": len(observed),
            **metrics(np.frombuffer(predicted, np.uint8), np.frombuffer(observed, np.uint8))}


def run(capture, dll, cubin, ordinal):
    binary = pinned_bytes(cubin, CUBIN_SHA)
    sealed, launch, (low, skip, observed), weights = load_case(capture, dll)
    residual = encode(f16(scaled_skip(skip, weights) + np.float32(0)))
    with ExitStack() as cleanup:
        device = Driver(ordinal)
        cleanup.callback(device.close)
        copy = device.lib.cuMemcpyHtoD_v2
        copy.argtypes, copy.restype = [C.c_uint64, C.c_void_p, C.c_size_t], C.c_int
        module = device.load(binary)
        cleanup.callback(device.call, "cuModuleUnload", module)
        function = device.function(module, ENTRY)
        resident = occupancy(device, function)
        isolated = compare(execute(device, function, low, skip, weights, observed), observed)
        zero_low = compare(residual, execute(device, function, bytes(len(low)), skip, weights, residual))
    with Sm86Mma(ordinal) as mma:
        hardware = mma.hardware_info
        nearest = project(low, weights, partial(gpu_split, mma))
    gpu = compare(merge(nearest, skip, weights), observed)
    separate = compare(encode(f16(nearest + scaled_skip(skip, weights))), observed)
    cpu = compare(merge(project(low, weights, CPU_SPLIT), skip, weights), observed)
    sources = [Path(__file__), *(Path(__file__).with_name(name) for name in
               ("probe_october_ffn_expand.py", "check_october_attention.py", "check_october_attention_tensor.py", "mma_sm86_tiles.py"))]
    return {"schema": "october_nr_decoder_input_replay_v1", "capture_seal_sha256": sealed["capture_seal_sha256"],
            "frame": sealed["frame"], "trace_sequence": launch.sequence, "entry": ENTRY, "captured_entry": launch.name,
            "grid_xyz": GRID, "block_xyz": BLOCK, "low_shape_hwc": LOW, "skip_and_output_shape_hwc": HIGH,
            "occupancy_bound": resident, "trial_cubin_sha256": CUBIN_SHA, "original_dll_sha256": DLL_SHA,
            "weight_layout_sha256": LAYOUT_SHA, "weight_record_sha256": sha(weights),
            "source_sha256": {path.name: sha(path.read_bytes()) for path in sources}, **hardware,
            "input_roles": list(ROLES[:2]), "input_sha256": [sha(low), sha(skip)], "output_role": ROLES[-1],
            "isolated": isolated, "zero_low_cpu_residual": zero_low, "independent_sm86": gpu,
            "independent_sm86_separate_skip_multiply_add": separate, "independent_cpu": cpu,
            "passed": not any(row["different_byte_count"] for row in (isolated, zero_low, gpu))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--original-dll", required=True, type=Path)
    parser.add_argument("--trial-cubin", required=True, type=Path)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.device < 0:
        parser.error("device ordinal must be nonnegative")
    try:
        result = run(args.capture, args.original_dll, args.trial_cubin, args.device)
    except (OSError, ValueError, RuntimeError, KeyError, struct.error) as error:
        result = {"passed": False, "error": str(error)}
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
