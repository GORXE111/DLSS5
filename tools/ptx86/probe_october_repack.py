"""Run only the two October ViT repack entries from the reviewed sm86 cubin.

Each direction is compared separately against a CPU TIN layout oracle with
independent input bytes. No NR DLL, tensor math, weights, or host scheduler
is loaded. The default cubin path must be supplied explicitly.
"""

import argparse
import ctypes as C
import json
from pathlib import Path
import struct
import sys

import numpy as np

from .probe_october_fence import Driver, TRIAL_CUBIN_SHA, sha

CHANNELS, GUARD = 1024, 256
CASES = ((8, 12), (12, 8), (4, 8), (8, 16))
SEEDS = (0x02062026, 0xDEADBEEF)
NAMES = ("cc_vit_1d_repack_2d_to_1d_fp8", "cc_vit_1d_repack_1d_to_2d_fp8")


def pack_tin(grid):
    h, w, c = grid.shape
    tiles = (grid.reshape(h // 4, 4, w // 4, 4, c // 32, 32)
             .transpose(0, 2, 4, 1, 3, 5).reshape(h // 4, w // 4, c // 32, 16, 32))
    lane = np.arange(32)[:, None]
    byte = np.arange(16)[None, :]
    token = (lane >> 2) + 8 * ((byte >> 2) % 2)
    channel = 8 * (byte % 4) + 2 * (lane & 3) + (byte >> 3)
    return np.ascontiguousarray(tiles[..., token, channel]).tobytes()


def payloads(h, w, seed):
    if h % 4 or w % 4 or h * w % 32:
        raise ValueError("probe shapes require 4x4 spatial tiles and no extra 1D token padding")
    grid = np.random.default_rng(seed).integers(0, 256, size=(h, w, CHANNELS), dtype=np.uint8)
    # Independently pack the same logical row-major sequence as 2D and 1D TIN.
    flattened = grid.reshape(h * w // 16, 4, 4, CHANNELS).transpose(1, 0, 2, 3)
    return pack_tin(grid), pack_tin(flattened.reshape(4, h * w // 4, CHANNELS))


def launch(device, fn, h, w, source, expected):
    assert len(source) == len(expected) == h * w * CHANNELS
    # Every unwritten output byte differs from its expected value. Guard both
    # buffers, and require that the complete source survives the launch.
    guard = bytes((i * 17 + 3) & 255 for i in range(GUARD))
    guarded_input = guard + source + guard
    guarded_output = guard + bytes(np.frombuffer(expected, np.uint8) ^ np.uint8(255)) + guard
    addresses = []
    try:
        for contents in (guarded_input, guarded_output):
            ptr = C.c_uint64()
            device.call("cuMemAlloc_v2", C.byref(ptr), len(contents))
            addresses.append(ptr)
            device.call("cuMemcpyHtoD_v2", ptr, C.create_string_buffer(contents), len(contents))
        # Captured legacy ABI: one 24-byte parameter structure; GPU pointers
        # at +0/+8, height/width at +16/+20. Both October entries retain it.
        blob = C.create_string_buffer(struct.pack("<QQii", addresses[0].value + GUARD,
                                                  addresses[1].value + GUARD, h, w))
        params = (C.c_void_p * 1)(C.addressof(blob))
        device.call("cuLaunchKernel", fn, h * w, 1, 1, 256, 1, 1, 0, None, params, None)
        device.call("cuCtxSynchronize")
        snapshots = []
        for ptr in addresses:
            readback = C.create_string_buffer(len(guarded_input))
            device.call("cuMemcpyDtoH_v2", readback, ptr, len(guarded_input))
            snapshots.append(readback.raw)
        actual = snapshots[1][GUARD:-GUARD]
        source_preserved = snapshots[0] == guarded_input
        guards_preserved = snapshots[1][:GUARD] == guard == snapshots[1][-GUARD:]
        differences = int(np.count_nonzero(np.frombuffer(actual, np.uint8) != np.frombuffer(expected, np.uint8)))
        return {"bytes": len(source), "input_sha256": sha(source), "expected_sha256": sha(expected),
                "actual_sha256": sha(actual), "differing_bytes": differences,
                "source_preserved": source_preserved, "destination_guards_preserved": guards_preserved,
                "passed": differences == 0 and source_preserved and guards_preserved}
    finally:
        for ptr in reversed(addresses):
            device.call("cuMemFree_v2", ptr)


def run(path, ordinal):
    cubin = path.read_bytes()
    if sha(cubin) != TRIAL_CUBIN_SHA:
        raise ValueError("not the reviewed module 6 sm86 compilation trial")
    device = Driver(ordinal)
    try:
        copy = device.lib.cuMemcpyHtoD_v2
        copy.argtypes, copy.restype = [C.c_uint64, C.c_void_p, C.c_size_t], C.c_int
        module = device.load(cubin)
        try:
            functions = [device.function(module, name) for name in NAMES]
            cases = []
            for h, w in CASES:
                for seed in SEEDS:
                    layouts = payloads(h, w, seed)
                    for i, name in enumerate(NAMES):
                        cases.append({"entry": name, "shape_hwc": [h, w, CHANNELS], "seed": seed,
                                      "grid": [h * w, 1, 1], "block": [256, 1, 1],
                                      **launch(device, functions[i], h, w, layouts[i], layouts[1 - i])})
        finally:
            device.call("cuModuleUnload", module)
        return {"scope": "Direct sm86 execution of two October ViT byte-repack kernels against CPU packing; no tensor math or NR DLL inference",
                "gpu": device.label, "compute_capability": device.capability, "cuda_driver_api_version": device.version,
                "trial_cubin_sha256": sha(cubin), "probe_source_sha256": sha(Path(__file__).read_bytes()),
                "driver_helper_source_sha256": sha(Path(__file__).with_name("probe_october_fence.py").read_bytes()),
                "cases": cases, "bytes_compared": sum(case["bytes"] for case in cases),
                "passed": len(cases) == 16 and all(case["passed"] for case in cases)}
    finally:
        device.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-cubin", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.device < 0:
        parser.error("device must be nonnegative")
    try:
        report = run(args.trial_cubin, args.device)
    except (OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"passed": False, "error": str(error)}, indent=2))
        return 1
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
