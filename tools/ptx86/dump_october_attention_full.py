"""Dump packed PV, reciprocal and pre-quantization half values for October attention.

Run the isolated ordinary sm86 entry on sealed Q/K/V. No NR DLL is
loaded. All compiled probe files, readbacks and report stay in a fresh _dl work
directory. Select sealed ViT blocks with --blocks 11 18 or any subset of 11..18.
"""

import argparse
from contextlib import ExitStack
import ctypes as C
import json
from pathlib import Path
import re
import struct
import subprocess
import sys

import numpy as np

from ..align_nr_small_trace import GRAPH, GRAPH_SHA, kernel_inventory, pinned_bytes, read_trace
from ..capture_october_taps import capture_path
from ..compare_october_taps import PORT, seal_capture
from .check_october_attention import encode_fp8
from .probe_october_attention_replay import ENTRIES, GUARD
from .probe_october_attention_stages import Driver, ENTRY, PTX_SHA, PTXAS_SHA, replace_once, sha


CHANNELS = 1024
QUERY_TILE = 16
QUERIES_PER_CTA = 256
CUDA_BLOCK = (32, 4, 1)
PLANES = ("pv", "inv", "norm")
PAIR_BASE = 983
PAIRS = range(PAIR_BASE, PAIR_BASE + 32)


def instrument(data, tokens=96):
    if sha(data) != PTX_SHA:
        raise ValueError("not the pinned rewritten October module 6 PTX")
    if tokens <= 0 or tokens % 32 or tokens * CHANNELS > 256 * 1024 * 1024:
        raise ValueError("expected a positive 32-aligned sealed token count")
    plane_bytes = 2 * tokens * CHANNELS
    source = data.decode("ascii")
    start = source.index(f".visible .entry {ENTRY}(")
    stop = source.index(".visible .entry ", start + 1)
    body = source[start:stop]
    args = ENTRY + "_param_0"
    if re.search(rf"\[{args}\+32\]", body):
        raise ValueError("ordinary attention already uses the scratch parameter slot")
    prologue = f"""
.reg .pred %nr_full_write;
.reg .b32 %nr_full_head, %nr_full_lane, %nr_full_warp, %nr_full_cta_y, %nr_full_base, %nr_full_tile, %nr_full_element, %nr_full_one;
.reg .b64 %nr_full_raw, %nr_full_scratch, %nr_full_byte, %nr_full_addr;
ld.param.b64 %nr_full_raw, [{args}+32];
cvta.to.global.u64 %nr_full_scratch, %nr_full_raw;
mov.u32 %nr_full_head, %ctaid.x;
mov.u32 %nr_full_lane, %tid.x;
mov.u32 %nr_full_warp, %tid.y;
mov.u32 %nr_full_cta_y, %ctaid.y;
mul.lo.u32 %nr_full_base, %nr_full_cta_y, {QUERIES_PER_CTA // QUERY_TILE};
mad.lo.u32 %nr_full_base, %nr_full_warp, 4, %nr_full_base;
mov.u32 %nr_full_one, 1;
"""
    body = replace_once(body, rf"^ld\.param\.b64 %rd1, \[{args}\];$", lambda m: prologue + m[0])
    seen = []

    def emit(match):
        dst = int(match[1])
        if dst not in PAIRS:
            return match[0]
        seen.append(dst)
        group, pair = divmod(dst - PAIR_BASE, 8)
        return match[0] + f"""
add.u32 %nr_full_tile, %nr_full_base, {group};
setp.lt.u32 %nr_full_write, %nr_full_tile, {tokens // QUERY_TILE};
mad.lo.u32 %nr_full_element, %nr_full_tile, 32, %nr_full_head;
mad.lo.u32 %nr_full_element, %nr_full_element, 32, %nr_full_lane;
mad.lo.u32 %nr_full_element, %nr_full_element, 16, {2 * pair};
mul.wide.u32 %nr_full_byte, %nr_full_element, 2;
add.u64 %nr_full_addr, %nr_full_scratch, %nr_full_byte;
@%nr_full_write st.global.b32 [%nr_full_addr], {match[2]};
@%nr_full_write st.global.b32 [%nr_full_addr+{plane_bytes}], {match[3]};
@%nr_full_write st.global.b32 [%nr_full_addr+{2 * plane_bytes}], %r{dst};
@%nr_full_write st.global.b32 [%nr_full_addr+{3 * plane_bytes}], %nr_full_one;
"""

    body = re.sub(r"^\{mul\.f16x2 %r(\d+),(%r\d+),(%r\d+);\n\}\n", emit, body, flags=re.M)
    if sorted(seen) != list(PAIRS):
        raise ValueError("expected one normalization anchor for each of the 32 packed half2 pairs")
    return (source[:start] + body + source[stop:]).encode("ascii")


def execute(device, function, raw, shape, grid):
    expected = raw[3]
    plane_bytes = 2 * len(expected)
    scratch_bytes = (len(PLANES) + 1) * plane_bytes
    payloads = [*raw[:3], bytes(np.frombuffer(expected, np.uint8) ^ np.uint8(255)), bytes(scratch_bytes)]
    with ExitStack() as cleanup:
        addresses = []
        for data in payloads:
            wrapped = GUARD + data + GUARD
            address = C.c_uint64()
            device.call("cuMemAlloc_v2", C.byref(address), len(wrapped))
            cleanup.callback(device.call, "cuMemFree_v2", address)
            device.call("cuMemcpyHtoD_v2", address, C.create_string_buffer(wrapped), len(wrapped))
            addresses.append(address)
        active = [address.value + len(GUARD) for address in addresses]
        abi = C.create_string_buffer(struct.pack("<QQQQQQQii", *active, 0, 0, *shape[:2]))
        params = (C.c_void_p * 1)(C.addressof(abi))
        device.call("cuLaunchKernel", function, *grid, *CUDA_BLOCK, 0, None, params, None)
        device.call("cuCtxSynchronize")
        results = []
        for address, data in zip(addresses, payloads):
            target = C.create_string_buffer(len(data) + 2 * len(GUARD))
            device.call("cuMemcpyDtoH_v2", target, address, len(target))
            if target.raw[:len(GUARD)] != GUARD or target.raw[-len(GUARD):] != GUARD:
                raise ValueError("GPU write crossed an input, output or scratch guard")
            results.append(target.raw[len(GUARD):-len(GUARD)])
    if results[:3] != raw[:3] or results[3] != expected:
        raise ValueError("instrumented launch changed its inputs or differs from the live FP8 output")
    scratch = results[4]
    markers = np.frombuffer(scratch[len(PLANES) * plane_bytes:], "<u4")
    if markers.size != len(expected) // 2 or not np.all(markers == 1):
        raise ValueError("missing packed half2 scratch writes")
    return {name: scratch[index * plane_bytes:(index + 1) * plane_bytes] for index, name in enumerate(PLANES)}


def load_block(block, sealed, files):
    roles = [f"block{block}.attention.{role}" for role in ("in0", "in1", "in2", "out")]
    boundaries = [sealed["boundaries"][role] for role in roles]
    shape = tuple(boundaries[0]["shape_hwc"])
    h, w, channels = shape
    sequence = boundaries[0]["trace_sequence"]
    elements = h * w * channels
    if h <= 0 or w <= 0 or h * w % 32 or channels != CHANNELS:
        raise ValueError(f"block {block}: expected positive 32-aligned tokens with {CHANNELS} channels")
    raw = [files[role].read_bytes() for role in roles]
    if any(tuple(row["shape_hwc"]) != shape or row["trace_sequence"] != sequence
           or len(data) != elements or sha(data) != row["sha256"]
           for row, data in zip(boundaries, raw)):
        raise ValueError(f"block {block}: expected intact attention taps from one launch")
    hashes = dict(zip(("q", "k", "v", "out"), (row["sha256"] for row in boundaries)))
    return raw, hashes, sequence, shape


def dump_block(device, function, block, raw, hashes, sequence, shape, grid, folder):
    planes = execute(device, function, raw, shape, grid)
    values = {name: np.frombuffer(data, "<f2") for name, data in planes.items()}
    if not all(np.isfinite(value).all() for value in values.values()):
        raise ValueError(f"block {block}: nonfinite captured half values")
    encoded = encode_fp8(values["norm"].astype(np.float32)).tobytes()
    if encoded != raw[3]:
        differing = np.count_nonzero(np.frombuffer(encoded, np.uint8) != np.frombuffer(raw[3], np.uint8))
        raise ValueError(f"block {block}: FP8 encoding of measured norm differs in {differing} packed bytes")
    artifacts = {}
    for name, data in planes.items():
        path = folder / f"block{block:02d}-{name}.f16"
        with path.open("xb") as stream:
            stream.write(data)
        artifacts[name] = {"file": path.name, "bytes": len(data), "sha256": sha(data)}
    return {"block": block, "trace_sequence": sequence, "tap_sha256": hashes, "packed_elements": len(encoded),
            "scratch_half2_writes": len(encoded) // 2, "instrumented_output_sha256": hashes["out"],
            "encoded_norm_sha256": sha(encoded), "planes": artifacts}


def run(capture, ptx, ptxas, work, blocks):
    if not blocks or len(set(blocks)) != len(blocks) or not set(blocks) <= set(range(11, 19)):
        raise ValueError("select distinct ViT blocks 11 through 18")
    folder = capture_path(work)
    if folder.is_relative_to(capture.resolve()):
        raise ValueError("probe output must be outside the input capture")
    ptxas = ptxas.resolve(strict=True)
    if sha(ptxas.read_bytes()) != PTXAS_SHA:
        raise ValueError("not the pinned ptxas 13.3 tool")
    graph = json.loads(pinned_bytes(GRAPH, GRAPH_SHA))
    inventory = kernel_inventory()
    sealed, files = seal_capture(capture, inventory, graph)
    if sealed["kind"] != PORT:
        raise ValueError("expected a sealed Ampere capture")
    inputs = {block: load_block(block, sealed, files) for block in blocks}
    shapes = {values[3] for values in inputs.values()}
    if len(shapes) != 1:
        raise ValueError("selected attention blocks must share one token shape")
    shape, = shapes
    tokens = shape[0] * shape[1]
    grid = (32, (tokens + QUERIES_PER_CTA - 1) // QUERIES_PER_CTA, 1)
    trace = (capture / "nr-trace.tsv").read_bytes()
    if sha(trace) != sealed["artifact_sha256"]["nr-trace.tsv"]:
        raise ValueError("trace changed after sealing")
    launches = {row.sequence: row for row in read_trace(trace, inventory)[sealed["frame"]]}
    for _, _, sequence, _ in inputs.values():
        launch = launches[sequence]
        if (launch.name, launch.grid, launch.block, launch.shared_memory) != (ENTRIES[1], grid, CUDA_BLOCK, 0):
            raise ValueError("sealed attention launch does not match the token shape")
    transformed = instrument(ptx.read_bytes(), tokens)
    folder.mkdir(parents=True, exist_ok=False)
    report = {"schema": "october_nr_attention_full_half_planes_v1", "passed": False,
              "source_sha256": sha(Path(__file__).read_bytes()), "capture_seal_sha256": sealed["capture_seal_sha256"],
              "rewritten_ptx_sha256": PTX_SHA, "ptxas_sha256": PTXAS_SHA, "instrumented_ptx_sha256": sha(transformed),
              "entry": ENTRY, "frame": sealed["frame"], "grid_xyz": grid, "block_xyz": CUDA_BLOCK, "shape_hwc": shape,
              "format": "Little-endian binary16 in packed live FP8 output element order", "cases": []}
    try:
        debug_ptx = folder / "module06.attention-full.ptx"
        cubin_path = folder / "module06.attention-full.cubin"
        with debug_ptx.open("xb") as stream:
            stream.write(transformed)
        with (folder / "compile.log").open("xb") as log:
            subprocess.run([str(ptxas), "-arch=sm_86", str(debug_ptx), "-o", str(cubin_path)],
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
        cubin = cubin_path.read_bytes()
        report["instrumented_cubin_sha256"] = sha(cubin)
        device = Driver(0)
        with ExitStack() as cleanup:
            cleanup.callback(device.close)
            copy = device.lib.cuMemcpyHtoD_v2
            copy.argtypes, copy.restype = [C.c_uint64, C.c_void_p, C.c_size_t], C.c_int
            module = device.load(cubin)
            cleanup.callback(device.call, "cuModuleUnload", module)
            function = device.function(module, ENTRY)
            report.update(gpu=device.label, compute_capability=device.capability, cuda_driver_api_version=device.version)
            for block, values in inputs.items():
                report["cases"].append(dump_block(device, function, block, *values, grid, folder))
        report["passed"] = True
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        report["error"] = str(error)
    with (folder / "report.json").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "ptx", "ptxas", "work"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--blocks", nargs="+", type=int, default=[11, 18])
    args = parser.parse_args(argv)
    try:
        report = run(args.capture, args.ptx, args.ptxas, args.work, args.blocks)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, struct.error) as error:
        report = {"passed": False, "error": str(error)}
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
