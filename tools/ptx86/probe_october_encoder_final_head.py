"""Replay the October encoder's pooled 512-to-1024 projection on sealed sm86 taps."""

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
from .check_october_attention import encode_fp8, metrics
from .check_october_attention_tensor import CPU_SPLIT, gpu_split
from .mma_sm86_tiles import Sm86Mma
from .probe_october_attention_replay import GUARD
from .probe_october_fence import Driver, sha
from .probe_october_ffn_expand import decode_tin, decode_weights, load_ffn_weights, tin_coordinates


BLOCK_INDEX = 10
INPUTS, OUTPUTS = 512, 1024
RECORD_BYTES = INPUTS * OUTPUTS + 16
INPUT_ROLE, OUTPUT_ROLE = "block10.final_head.input", "block10.out0"
ENTRY = "cc_split_swin_16h_final_head_512_fp8"
LIVE_ENTRY = "cc_split_swin_16h_final_head_512_wait_fp8"
POOL_ENTRY = "cc_split_swin_16h_proj_pool_512_chained_fp8"
MODULE_SHA256 = "8334158563e86ad6146a19105746eece2a26495967741518fb1b08b280fc838c"
CUDA_BLOCK = (32, 8, 1)


def predict(activation, weights, tokens, multiply):
    # Each 4x4 TIN2D tile is contiguous. The pointwise convolution preserves
    # that tile order, so the 1D tile decoder can reuse the same lane mapping.
    values = decode_tin(activation, tokens, INPUTS)
    matrix = decode_weights(weights, INPUTS, OUTPUTS)
    accum = 0
    for start in range(0, INPUTS, 32):
        accum = multiply(values[..., start:start + 32], matrix[:, start:start + 32], accum)
    return encode_fp8(accum[0][tin_coordinates(tokens, OUTPUTS)]).reshape(-1)


def execute(device, function, activation, expected, weights, shape, grid):
    poisoned = bytes(np.frombuffer(expected, np.uint8) ^ np.uint8(255))
    payloads = activation, poisoned, weights
    with ExitStack() as cleanup:
        addresses = []
        for data in payloads:
            guarded = GUARD + data + GUARD
            address = C.c_uint64()
            device.call("cuMemAlloc_v2", C.byref(address), len(guarded))
            cleanup.callback(device.call, "cuMemFree_v2", address)
            device.call("cuMemcpyHtoD_v2", address, C.create_string_buffer(guarded), len(guarded))
            addresses.append(address)
        active = [address.value + len(GUARD) for address in addresses]
        blob = C.create_string_buffer(struct.pack("<4Qii", *active, 0, *shape))
        arguments = (C.c_void_p * 1)(C.addressof(blob))
        device.call("cuLaunchKernel", function, *grid, *CUDA_BLOCK, 0, None, arguments, None)
        device.call("cuCtxSynchronize")
        results = []
        for address, data in zip(addresses, payloads):
            readback = C.create_string_buffer(len(data) + 2 * len(GUARD))
            device.call("cuMemcpyDtoH_v2", readback, address, len(readback))
            if readback.raw[:len(GUARD)] != GUARD or readback.raw[-len(GUARD):] != GUARD:
                raise ValueError("isolated encoder head crossed an allocation guard")
            results.append(readback.raw[len(GUARD):-len(GUARD)])
    if results[0] != activation or results[2] != weights:
        raise ValueError("isolated encoder head changed its activation or weights")
    return results[1]


def load_case(capture, sealed, files, inventory, weight_offset):
    source, destination = (sealed["boundaries"][role] for role in (INPUT_ROLE, OUTPUT_ROLE))
    h, w, channels = destination["shape_hwc"]
    grid = (4 * ((w + 7) // 8), (h + 7) // 8, 1)
    trace = pinned_bytes(capture / "nr-trace.tsv", sealed["artifact_sha256"]["nr-trace.tsv"])
    launches = {launch.sequence: launch for launch in read_trace(trace, inventory)[sealed["frame"]]}
    head = launches[source["trace_sequence"]]
    pool = launches[head.sequence - 1]
    pre, = (launch for launch in launches.values() if launch.name == PRE_ENTRY)
    if ((h, w) not in ((8, 12), (36, 60)) or channels != OUTPUTS or source["shape_hwc"] != (h, w, INPUTS)
            or (source["layout"], destination["layout"]) != ("tin2d", "tin2d")
            or (source["parameter_offset"], destination["parameter_offset"]) != (0, 8)
            or destination["trace_sequence"] != head.sequence or head.name != LIVE_ENTRY
            or (head.grid, head.block, head.shared_memory) != (grid, CUDA_BLOCK, 0)
            or len(head.params) != 40 or struct.unpack_from("<ii", head.params, 32) != (h, w)
            or pool.name != POOL_ENTRY or pool.pointer(24) != head.pointer(0)
            or pool.params[72:80] != head.params[32:40]
            or head.pointer(16) - pre.pointer(232) != weight_offset):
        raise ValueError("sealed tensors do not bind the expected pooled encoder head")
    activation, expected = (files[role].read_bytes() for role in (INPUT_ROLE, OUTPUT_ROLE))
    for data, boundary, width in ((activation, source, INPUTS), (expected, destination, OUTPUTS)):
        if (len(data) != h * w * width or sha(data) != boundary["sha256"]
                or np.any((np.frombuffer(data, np.uint8) & 127) == 127)):
            raise ValueError("expected intact finite E4M3 encoder head taps")
    return (h, w), grid, activation, expected, head.sequence


def run(capture, dll, cubin_path, ordinal):
    inventory = kernel_inventory()
    graph = json.loads(pinned_bytes(GRAPH, GRAPH_SHA))
    sealed, files = seal_capture(capture, inventory, graph)
    if sealed["kind"] != PORT:
        raise ValueError("expected a sealed staged Ampere capture")
    cubin = pinned_bytes(cubin_path, MODULE_SHA256)
    records, offsets = load_ffn_weights(dll, (BLOCK_INDEX,), 4, RECORD_BYTES)
    weights = records[BLOCK_INDEX]
    shape, grid, activation, expected, sequence = load_case(capture, sealed, files, inventory,
                                                            offsets[f"block{BLOCK_INDEX}.layer4.layer"])
    with ExitStack() as cleanup:
        device = Driver(ordinal)
        cleanup.callback(device.close)
        copy = device.lib.cuMemcpyHtoD_v2
        copy.argtypes, copy.restype = [C.c_uint64, C.c_void_p, C.c_size_t], C.c_int
        module = device.load(cubin)
        cleanup.callback(device.call, "cuModuleUnload", module)
        isolated = execute(device, device.function(module, ENTRY), activation, expected, weights, shape, grid)
    tokens = shape[0] * shape[1]
    with Sm86Mma(ordinal) as mma:
        hardware = mma.hardware_info
        independent = predict(activation, weights, tokens, partial(gpu_split, mma))
    cpu = predict(activation, weights, tokens, CPU_SPLIT)
    observed = np.frombuffer(expected, np.uint8)
    results = {}
    for key, array in (("isolated", np.frombuffer(isolated, np.uint8)), ("independent_sm86", independent),
                       ("independent_cpu", cpu)):
        results[key] = {**metrics(array, observed), "output_sha256": sha(array.tobytes())}
    sources = (Path(__file__).name, "probe_october_ffn_expand.py", "check_october_attention.py",
               "check_october_attention_tensor.py", "mma_sm86_tiles.py")
    return {"schema": "october_nr_encoder_final_head_v1", "capture_seal_sha256": sealed["capture_seal_sha256"],
            "source_sha256": {name: sha(Path(__file__).with_name(name).read_bytes()) for name in sources},
            "original_dll_sha256": DLL_SHA, "module_cubin_sha256": MODULE_SHA256, "weight_sha256": sha(weights),
            **hardware, "frame": sealed["frame"], "sequence": sequence, "entry": ENTRY, "grid_xyz": grid,
            "block_xyz": CUDA_BLOCK, "input_shape_hwc": (*shape, INPUTS), "output_shape_hwc": (*shape, OUTPUTS),
            "input_role": INPUT_ROLE, "output_role": OUTPUT_ROLE, "input_sha256": sha(activation),
            "sealed_output_sha256": sha(expected), "output_bytes": len(expected), **results,
            "passed": isolated == expected and independent.tobytes() == expected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--original-dll", type=Path, required=True)
    parser.add_argument("--module-cubin", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.device < 0:
        parser.error("device must be nonnegative")
    try:
        report = run(args.capture, args.original_dll, args.module_cubin, args.device)
    except (OSError, RuntimeError, ValueError, KeyError, struct.error) as error:
        report = {"passed": False, "error": str(error)}
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
