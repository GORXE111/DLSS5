"""Replay the two staged October attention entries against real Q/K/V taps.

The original tap run supplies the launch geometry and expected output. The
chained entry receives already-ready counters, so this tests its arithmetic,
layout and completion stores in isolation without running an NR DLL.
"""

import argparse
from contextlib import ExitStack
import ctypes as C
import json
from pathlib import Path
import struct
import sys

import numpy as np

from ..align_nr_small_trace import GRAPH, GRAPH_SHA, kernel_inventory, pinned_bytes, read_trace
from ..compare_october_taps import PORT, seal_capture
from .probe_october_fence import Driver, TRIAL_CUBIN_SHA, sha


ENTRIES = ("cc_vit_1d_attention_fp8", "cc_vit_1d_attention_chained_fp8")
GUARD = bytes((index * 17 + 3) & 255 for index in range(256))
COUNTERS = 32


def execute(device, function, grid, block, h, w, taps, chained):
    source, expected = taps[:3], taps[3]
    poisoned = bytes(np.frombuffer(expected, np.uint8) ^ np.uint8(255))
    inputs = [*source, poisoned]
    if chained:
        inputs.extend([bytes(COUNTERS * 4), bytes([255]) * (COUNTERS * 4)])
    with ExitStack() as cleanup:
        addresses = []
        for data in inputs:
            pointer = C.c_uint64()
            device.call("cuMemAlloc_v2", C.byref(pointer), len(data) + 2 * len(GUARD))
            cleanup.callback(device.call, "cuMemFree_v2", pointer)
            contents = C.create_string_buffer(GUARD + data + GUARD)
            device.call("cuMemcpyHtoD_v2", pointer, contents, len(contents) - 1)
            addresses.append(pointer)
        q, k, v, out, *counters = [pointer.value + len(GUARD) for pointer in addresses]
        ready, completion = counters if chained else (0, 0)
        blob = C.create_string_buffer(struct.pack("<QQQQQQQii", q, k, v, out, 0, ready, completion, h, w))
        arguments = (C.c_void_p * 1)(C.addressof(blob))
        device.call("cuLaunchKernel", function, *grid, *block, 0, None, arguments, None)
        device.call("cuCtxSynchronize")
        results, guards = [], []
        for pointer, data in zip(addresses, inputs):
            readback = C.create_string_buffer(len(data) + 2 * len(GUARD))
            device.call("cuMemcpyDtoH_v2", readback, pointer, len(readback))
            results.append(readback.raw[len(GUARD):-len(GUARD)])
            guards.append(readback.raw[:len(GUARD)] == GUARD == readback.raw[-len(GUARD):])
    differences = int(np.count_nonzero(np.frombuffer(results[3], np.uint8) != np.frombuffer(expected, np.uint8)))
    source_unchanged = results[:3] == source
    completion_ready = not chained or results[4:] == [bytes(COUNTERS * 4)] * 2
    return {"output_sha256": sha(results[3]), "expected_sha256": sha(expected), "bytes": len(expected),
            "different_bytes": differences, "source_unchanged": source_unchanged, "guards_intact": all(guards),
            "counter_check": completion_ready if chained else None,
            "passed": differences == 0 and source_unchanged and all(guards) and completion_ready}


def run(capture, trial_cubin, ordinal):
    cubin = trial_cubin.read_bytes()
    if sha(cubin) != TRIAL_CUBIN_SHA:
        raise ValueError("not the pinned October module 6 sm86 trial cubin")
    inventory = kernel_inventory()
    graph = json.loads(pinned_bytes(GRAPH, GRAPH_SHA))
    sealed, files = seal_capture(capture, inventory, graph)
    if sealed["kind"] != PORT:
        raise ValueError("expected an independently captured staged Ampere run")
    trace = read_trace((capture / "nr-trace.tsv").read_bytes(), inventory)
    cases = []
    device = Driver(ordinal)
    try:
        copy = device.lib.cuMemcpyHtoD_v2
        copy.argtypes, copy.restype = [C.c_uint64, C.c_void_p, C.c_size_t], C.c_int
        module = device.load(cubin)
        try:
            for block_index in (11, 18):
                roles = [f"block{block_index}.attention.{suffix}" for suffix in ("in0", "in1", "in2", "out")]
                boundary = sealed["boundaries"][roles[0]]
                h, w, channels = boundary["shape_hwc"]
                launch, = (row for row in trace[sealed["frame"]] if row.sequence == boundary["trace_sequence"])
                if launch.name != ENTRIES[1] or launch.grid != (32, 1, 1) or launch.block != (32, 4, 1) or (
                    h, w, channels, launch.shared_memory
                ) != (8, 12, 1024, 0):
                    raise ValueError("this replay requires the observed 96-token, 32-head launch")
                taps = [files[role].read_bytes() for role in roles]
                if any(sha(data) != sealed["boundaries"][role]["sha256"] for role, data in zip(roles, taps)):
                    raise ValueError("tap changed after it was sealed")
                for entry in ENTRIES:
                    function = device.function(module, entry)
                    cases.append({"block_index": block_index, "sequence": launch.sequence, "entry": entry,
                                  "grid": launch.grid, "block": launch.block, "token_hw": [h, w],
                                  **execute(device, function, launch.grid, launch.block, h, w, taps, entry == ENTRIES[1])})
        finally:
            device.call("cuModuleUnload", module)
        return {"schema": "october_nr_attention_replay_v1", "capture_seal_sha256": sealed["capture_seal_sha256"],
                "source_sha256": sha(Path(__file__).read_bytes()), "trial_cubin_sha256": TRIAL_CUBIN_SHA,
                "gpu": device.label, "compute_capability": device.capability, "cuda_driver_api_version": device.version,
                "scope": "Isolated attention math on tapped Q/K/V; chained input flags start ready.",
                "cases": cases, "passed": len(cases) == 4 and all(row["passed"] for row in cases)}
    finally:
        device.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--trial-cubin", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    try:
        report = run(args.capture, args.trial_cubin, args.device)
    except (OSError, ValueError, RuntimeError, KeyError, struct.error) as error:
        report = {"passed": False, "error": str(error)}
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
