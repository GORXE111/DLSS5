"""Capture October attention QK logits and pre-E4M3 probabilities on sm86."""

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
from . import check_october_attention, check_october_attention_tensor, mma_sm86_tiles
from .check_october_attention import FP8, buffer_offsets, encode_fp8, probabilities
from .check_october_attention_tensor import CPU_SPLIT, gpu_split, half_metrics
from .mma_sm86_tiles import Sm86Mma
from .probe_october_attention_replay import ENTRIES, GUARD
from .probe_october_attention_stages import Driver, ENTRY, PTX_SHA, PTXAS_SHA, replace_once, sha


HEADS = 32
PLANES = ("logits", "prob")
GRID_BLOCK = (32, 4, 1)


def instrument(data):
    if sha(data) != PTX_SHA:
        raise ValueError("not the pinned rewritten October module 6 PTX")
    source = data.decode("ascii")
    start = source.index(f".visible .entry {ENTRY}(")
    stop = source.index(".visible .entry ", start + 1)
    body = source[start:stop]
    args = ENTRY + "_param_0"
    if f"[{args}+32]" in body:
        raise ValueError("ordinary attention already uses the scratch parameter slot")
    declarations = f"""
.reg .pred %nr_qk_write;
.reg .b32 %nr_qk_lane, %nr_qk_base, %nr_qk_cols, %nr_qk_tile, %nr_qk_index;
.reg .b64 %nr_qk_raw, %nr_qk_scratch, %nr_qk_offset, %nr_qk_address;
ld.param.b64 %nr_qk_raw, [{args}+32];
cvta.to.global.u64 %nr_qk_scratch, %nr_qk_raw;
"""
    body = replace_once(body, rf"^ld\.param\.b64 %rd1, \[{args}\];$", lambda m: declarations + m[0])

    # %r1 = head, %r10 = first query M16 of the warp, %r5 = query tile count.
    # %r20 = current K64 iteration and %r1392 = padded key tile count.
    loop_index = """
mov.u32 %nr_qk_lane, %laneid;
shr.u32 %nr_qk_index, %nr_qk_lane, 2;
mad.lo.u32 %nr_qk_base, %r1, %r5, %r10;
mad.lo.u32 %nr_qk_base, %nr_qk_base, 16, %nr_qk_index;
shl.b32 %nr_qk_cols, %r1392, 6;
shl.b32 %nr_qk_index, %r20, 6;
mad.lo.u32 %nr_qk_base, %nr_qk_base, %nr_qk_cols, %nr_qk_index;
and.b32 %nr_qk_index, %nr_qk_lane, 3;
mad.lo.u32 %nr_qk_base, %nr_qk_index, 2, %nr_qk_base;
"""
    body = replace_once(body, r"^mov\.b32 %r20, %r1417;$", lambda m: m[0] + "\n" + loop_index)

    fma = re.compile(r"^\{fma\.rn\.f16x2 %r\d+,(%r\d+),%r168,%r169;\n\}\n"
                     r"(?:\s*\{(?:max|min)\.f16x2 [^}]+\}){2}\s*"
                     r"shl\.b32 (%r\d+), %r\d+, 4;\n"
                     r"add\.s32 (%r\d+), \2, 1073496064;$", re.M)
    matches = list(fma.finditer(body))
    mma = re.compile(r"^mma\.sync\.aligned\.m16n8k16\.row\.col\.f16\.f16\.f16\.f16 "
                     r"\{(%r\d+), (%r\d+)\}, [^\n]+, \{%m_t0, %m_t1\};$", re.M)
    outputs = mma.findall(body[:matches[0].start()]) if matches else []
    if len(matches) != 64 or [m[1] for m in matches] != [reg for pair in outputs for reg in pair]:
        raise ValueError("expected 64 QK output/probability half2 pairs in MMA order")

    parts = []
    end = 0
    for index, match in enumerate(matches):
        tile, half = divmod(index, 2)
        query_tile, key_tile = divmod(tile, 8)
        query_row = query_tile * 16 + half * 8
        parts.append(body[end:match.end()])
        parts.append(f"""
add.u32 %nr_qk_tile, %r10, {query_tile};
setp.lt.u32 %nr_qk_write, %nr_qk_tile, %r5;
mad.lo.u32 %nr_qk_index, %nr_qk_cols, {query_row}, %nr_qk_base;
add.u32 %nr_qk_index, %nr_qk_index, {key_tile * 8};
mul.wide.u32 %nr_qk_offset, %nr_qk_index, 4;
add.u64 %nr_qk_address, %nr_qk_scratch, %nr_qk_offset;
@%nr_qk_write st.global.v2.b32 [%nr_qk_address], {{{match[1]}, {match[3]}}};
""")
        end = match.end()
    return (source[:start] + "".join(parts) + body[end:] + source[stop:]).encode("ascii")


def load_case(block, sealed, files, launches):
    roles = [f"block{block}.attention.{role}" for role in ("in0", "in1", "in2", "out")]
    rows = [sealed["boundaries"][role] for role in roles]
    raw = [files[role].read_bytes() for role in roles]
    h, w, channels = rows[0]["shape_hwc"]
    tokens = h * w
    shape = (h, w, channels)
    sequence = rows[0]["trace_sequence"]
    launch = launches[sequence]
    if (channels != 1024 or tokens <= 0 or tokens % 32 or tokens > 640 or
            any(row["shape_hwc"] != shape or row["trace_sequence"] != sequence or
                len(data) != tokens * channels or sha(data) != row["sha256"] for row, data in zip(rows, raw)) or
            (launch.name, launch.grid, launch.block, launch.shared_memory) !=
            (ENTRIES[1], (HEADS, (tokens + 255) // 256, 1), GRID_BLOCK, 0)):
        raise ValueError(f"block {block}: expected intact staged attention taps and supported launch geometry")
    return shape, launch, raw


def capture_planes(device, function, shape, launch, raw):
    h, w, _ = shape
    tokens = h * w
    keys = (tokens + 63) // 64 * 64
    # Each pair writes [logits.low, logits.high, prob.low, prob.high]. NaN
    # poison makes a missing pair detectable without a separate marker plane.
    poisoned_out = bytes(np.frombuffer(raw[3], np.uint8) ^ np.uint8(255))
    poisoned_scratch = b"\x00\x7e" * (HEADS * tokens * keys * 2)
    inputs = [*raw[:3], poisoned_out, poisoned_scratch]
    with ExitStack() as cleanup:
        pointers = []
        for data in inputs:
            buffer = C.create_string_buffer(GUARD + data + GUARD)
            pointer = C.c_uint64()
            device.call("cuMemAlloc_v2", C.byref(pointer), len(buffer) - 1)
            cleanup.callback(device.call, "cuMemFree_v2", pointer)
            device.call("cuMemcpyHtoD_v2", pointer, buffer, len(buffer) - 1)
            pointers.append(pointer)
        active = [pointer.value + len(GUARD) for pointer in pointers]
        abi = C.create_string_buffer(struct.pack("<QQQQQQQii", *active, 0, 0, h, w))
        args = (C.c_void_p * 1)(C.addressof(abi))
        device.call("cuLaunchKernel", function, *launch.grid, *launch.block, 0, None, args, None)
        device.call("cuCtxSynchronize")
        results = []
        for pointer, data in zip(pointers, inputs):
            buffer = C.create_string_buffer(len(data) + 2 * len(GUARD))
            device.call("cuMemcpyDtoH_v2", buffer, pointer, len(buffer))
            observed = buffer.raw
            if observed[:len(GUARD)] != GUARD or observed[-len(GUARD):] != GUARD:
                raise ValueError("GPU write crossed an input, output or scratch guard")
            results.append(observed[len(GUARD):-len(GUARD)])
    if results[:4] != raw:
        raise ValueError("instrumentation changed Q/K/V or the sealed FP8 output")
    words = np.frombuffer(results[4], "<f2").reshape(HEADS, tokens, keys // 2, 2, 2)
    if not np.all(np.isfinite(words)):
        raise ValueError("missing QK half2 scratch writes or nonfinite captured values")
    return {name: np.ascontiguousarray(words[..., i, :]).reshape(HEADS, tokens, keys)
            for i, name in enumerate(PLANES)}


def dump_case(block, sealed, shape, launch, raw, planes, work):
    logits, prob = planes["logits"], planes["prob"]
    _, tokens, keys = logits.shape
    reconstructed = probabilities(logits.astype(np.float32))
    files = {}
    for name, array in planes.items():
        filename = f"block{block:02d}-{name}.f16"
        data = array.tobytes()
        with (work / filename).open("xb") as stream:
            stream.write(data)
        files[name] = {"file": filename, "bytes": len(data), "sha256": sha(data)}
    tap_hashes = {name: sealed["boundaries"][f"block{block}.attention.{role}"]["sha256"]
                  for name, role in zip(("q", "k", "v", "out"), ("in0", "in1", "in2", "out"))}
    return {"block": block, "trace_sequence": launch.sequence, "tap_sha256": tap_hashes,
            "instrumented_output_sha256": sha(raw[3]), "shape_hwc": shape, "grid_xyz": launch.grid,
            "tensor_shape_hqk": logits.shape, "half2_writes_per_plane": logits.size // 2,
            "host_probability_from_measured_logits": half_metrics(reconstructed, prob),
            "padding": {"keys": keys - tokens, "nonzero_logits": int(np.count_nonzero(logits[..., tokens:]))},
            "planes": files}


def compare_qk(mma, case, raw, work):
    _, tokens, keys = case["tensor_shape_hqk"]
    offsets_q, offsets_k = buffer_offsets(tokens)[:2]
    q = FP8[np.frombuffer(raw[0], np.uint8)[offsets_q]]
    k = FP8[np.frombuffer(raw[1], np.uint8)[offsets_k]]
    if keys > tokens:
        k = np.pad(k, ((0, 0), (0, keys - tokens), (0, 0)))
    kt = k.transpose(0, 2, 1)
    logits, prob = (np.fromfile(work / case["planes"][name]["file"], "<f2").reshape(case["tensor_shape_hqk"])
                    for name in PLANES)
    observed_e4m3 = encode_fp8(prob.astype(np.float32))
    for label, predicted in (("cpu", CPU_SPLIT(q, kt)), ("sm86", gpu_split(mma, q, kt))):
        predicted_prob = probabilities(predicted)
        case[label] = {"logits": half_metrics(predicted, logits), "prob": half_metrics(predicted_prob, prob),
                       "derived_e4m3_prob_different_bytes": int(np.count_nonzero(
                           encode_fp8(predicted_prob) != observed_e4m3))}


def run(capture, ptx, ptxas, work, blocks):
    if len(set(blocks)) != len(blocks) or not blocks or not set(blocks) <= set(range(11, 19)):
        raise ValueError("select distinct ViT blocks 11 through 18")
    work = capture_path(work)
    if work.is_relative_to(capture.resolve()):
        raise ValueError("probe output must be outside the input capture")
    ptxas = ptxas.resolve(strict=True)
    if sha(ptxas.read_bytes()) != PTXAS_SHA:
        raise ValueError("not the pinned ptxas 13.3 tool")
    graph = json.loads(pinned_bytes(GRAPH, GRAPH_SHA))
    inventory = kernel_inventory()
    sealed, files = seal_capture(capture, inventory, graph)
    if sealed["kind"] != PORT:
        raise ValueError("expected a sealed Ampere capture")
    trace = (capture / "nr-trace.tsv").read_bytes()
    if sha(trace) != sealed["artifact_sha256"]["nr-trace.tsv"]:
        raise ValueError("trace changed after sealing")
    launches = {row.sequence: row for row in read_trace(trace, inventory)[sealed["frame"]]}
    inputs = {block: load_case(block, sealed, files, launches) for block in blocks}
    transformed = instrument(ptx.read_bytes())
    work.mkdir(parents=True, exist_ok=False)
    dependencies = {Path(module.__file__).name: sha(Path(module.__file__).read_bytes())
                    for module in (check_october_attention, check_october_attention_tensor, mma_sm86_tiles)}
    report = {"schema": "october_nr_attention_qk_planes_v1", "passed": False, "frame": sealed["frame"],
              "source_sha256": sha(Path(__file__).read_bytes()), "capture_seal_sha256": sealed["capture_seal_sha256"],
              "dependency_sha256": dependencies,
              "rewritten_ptx_sha256": PTX_SHA, "ptxas_sha256": PTXAS_SHA,
              "instrumented_ptx_sha256": sha(transformed), "entry": ENTRY, "block_xyz": GRID_BLOCK,
              "format": "Little-endian binary16 in logical [head,query,key] order including padded keys", "cases": []}
    try:
        debug_ptx = work / "module06.attention-qk.ptx"
        cubin_path = work / "module06.attention-qk.cubin"
        with debug_ptx.open("xb") as stream:
            stream.write(transformed)
        with (work / "compile.log").open("xb") as log:
            subprocess.run([str(ptxas), "-arch=sm_86", str(debug_ptx), "-o", str(cubin_path)],
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
        cubin = cubin_path.read_bytes()
        report["instrumented_cubin_sha256"] = sha(cubin)
        with ExitStack() as cleanup:
            device = Driver(0)
            cleanup.callback(device.close)
            copy = device.lib.cuMemcpyHtoD_v2
            copy.argtypes, copy.restype = [C.c_uint64, C.c_void_p, C.c_size_t], C.c_int
            module = device.load(cubin)
            cleanup.callback(device.call, "cuModuleUnload", module)
            function = device.function(module, ENTRY)
            report.update(gpu=device.label, compute_capability=device.capability, cuda_driver_api_version=device.version)
            for block, (shape, launch, raw) in inputs.items():
                planes = capture_planes(device, function, shape, launch, raw)
                report["cases"].append(dump_case(block, sealed, shape, launch, raw, planes, work))
        with Sm86Mma() as mma:
            for case in report["cases"]:
                compare_qk(mma, case, inputs[case["block"]][2], work)
        report["passed"] = True
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        report["error"] = str(error)
    with (work / "report.json").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "ptx", "ptxas", "work"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--blocks", nargs="+", type=int, default=[11, 18])
    args = parser.parse_args()
    try:
        report = run(args.capture, args.ptx, args.ptxas, args.work, args.blocks)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, struct.error) as error:
        report = {"passed": False, "error": str(error)}
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
