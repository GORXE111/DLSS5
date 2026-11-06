"""Compare CPU and Ampere-assisted attention across sealed October tensor taps."""

import argparse
from functools import partial
import json
from pathlib import Path
import sys

import numpy as np

from ..align_nr_small_trace import GRAPH, GRAPH_SHA, kernel_inventory, pinned_bytes
from ..compare_october_taps import PORT, seal_capture
from . import check_october_attention as model, mma_sm86_tiles
from .check_october_attention import (
    FP8, buffer_offsets, encode_fp8, f16, half_literal, half_sum, metrics, probabilities, sha, split_mma,
)
from .mma_sm86_tiles import Sm86Mma


K16 = (np.arange(16), np.arange(16, 32))
CPU_SPLIT = partial(split_mma, groups=K16)
ROLES = ("in0", "in1", "in2", "out")
MODES = ("cpu", "gpu_qk", "gpu_pv", "gpu_qk_pv")


def load_case(block, sealed, files, shape, layout):
    tensors = []
    for suffix, indices in zip(ROLES, layout):
        role = f"block{block}.attention.{suffix}"
        boundary = sealed["boundaries"][role]
        data = files[role].read_bytes()
        if boundary["shape_hwc"] != shape or len(data) != indices.size or sha(data) != boundary["sha256"]:
            raise ValueError(f"{role} requires an intact tap with the selected attention geometry")
        codes = np.frombuffer(data, np.uint8)[indices]
        if np.any((codes & 127) == 127):
            raise ValueError(f"{role} contains a non-finite E4M3 value")
        tensors.append(codes)
    return tensors


def gpu_split(mma, a, b, acc=0):
    c = None if np.isscalar(acc) and acc == 0 else acc
    low = mma(a[..., :16], b[:, :16, :], c)
    return mma(a[..., 16:], b[:, 16:, :], low).astype(np.float32)


def evaluate(q, k, v, qk, pv, *, tokens):
    """Stream 64-key tiles in kernel order without allocating a token-square matrix."""
    out = np.zeros_like(q)
    denominator = 0
    p_zero = probabilities(np.array(0, dtype=np.float32)).item()
    padding = (-tokens) % 64
    for start in range(0, tokens, 64):
        stop = min(start + 64, tokens)
        keys, values = k[:, start:stop], v[:, start:stop]
        if stop - start < 64:
            widths = ((0, 0), (0, padding), (0, 0))
            keys, values = np.pad(keys, widths), np.pad(values, widths)
        prob = probabilities(qk(q, keys.transpose(0, 2, 1)))
        denominator = f16(denominator + half_sum(prob))
        p8 = FP8[encode_fp8(prob)]
        out = pv(p8[..., :32], values[:, :32], out)
        out = pv(p8[..., 32:], values[:, 32:], out)
    denominator = f16(denominator - f16(p_zero * padding))
    reciprocal = 1.0 / np.maximum(denominator, half_literal(0x388205FF))
    return {"pv": out, "reciprocal_f32": reciprocal, "pre_quantized": f16(out * f16(reciprocal))}


def half_metrics(predicted, measured):
    a, b = np.asarray(predicted, dtype="<f2"), np.asarray(measured, dtype="<f2")
    if a.shape != b.shape:
        raise ValueError("stage tensor shapes differ")
    delta = np.abs(a.astype(np.float32) - b.astype(np.float32))
    return {"different_numeric_values": int(np.count_nonzero(delta)),
            "different_half_words": int(np.count_nonzero(a.view("<u2") != b.view("<u2"))),
            "mae": float(delta.mean(dtype=np.float64)), "max_abs": float(delta.max())}


def encode_output(values, tokens):
    encoded = encode_fp8(values)
    # Attention computes padded queries but clears their FP8 stores afterwards.
    encoded[:, tokens:] = 0
    return encoded


def read_dump(folder, row, expected, layout, tokens):
    arrays = {}
    for role in ("pv", "inv", "norm"):
        artifact = row["planes"][role]
        data = (folder / f"block{row['block']:02d}-{role}.f16").read_bytes()
        if len(data) != expected.size * 2 or len(data) != artifact["bytes"] or sha(data) != artifact["sha256"]:
            raise ValueError("full stage dump failed its size or SHA-256 check")
        arrays[role] = np.frombuffer(data, "<f2")[layout]
    if not np.array_equal(encode_output(arrays["norm"].astype(np.float32), tokens), expected):
        raise ValueError("captured pre-quantization tensor does not encode to the sealed output")
    return arrays


def diagnostic(stages, measured, observed, tokens):
    predicted_pv = stages["pv"].astype(np.float16)
    predicted_inv = np.broadcast_to(stages["reciprocal_f32"].astype(np.float16), predicted_pv.shape)
    return {"pv": half_metrics(predicted_pv, measured["pv"]), "reciprocal": half_metrics(predicted_inv, measured["inv"]),
            "pre_quantization": half_metrics(stages["pre_quantized"], measured["norm"]),
            "with_measured_pv": metrics(encode_output((measured["pv"] * predicted_inv).astype(np.float32), tokens), observed),
            "with_measured_reciprocal": metrics(encode_output((predicted_pv * measured["inv"]).astype(np.float32), tokens), observed),
            "with_both_measured": metrics(encode_output((measured["pv"] * measured["inv"]).astype(np.float32), tokens), observed)}


def run(capture, blocks, dump, device, modes=MODES):
    sealed, files = seal_capture(capture, kernel_inventory(), json.loads(pinned_bytes(GRAPH, GRAPH_SHA)))
    if sealed["kind"] != PORT:
        raise ValueError("expected a sealed staged Ampere capture")
    shape = sealed["boundaries"][f"block{blocks[0]}.attention.in0"]["shape_hwc"]
    h, w, channels = shape
    tokens = h * w
    if tokens <= 0 or channels != 1024:
        raise ValueError("expected positive attention tokens and 1024 channels")
    storage_tokens = (tokens + 31) // 32 * 32
    seal = sealed["capture_seal_sha256"]
    dump_sha = None
    if dump:
        raw = (dump / "report.json").read_bytes()
        report = json.loads(raw)
        if (report["schema"] != "october_nr_attention_full_half_planes_v1" or
                report["capture_seal_sha256"] != seal or report["passed"] is not True):
            raise ValueError("full stage capture does not belong to this attention tap")
        dump_cases = {row["block"]: row for row in report["cases"]}
        if len(dump_cases) != len(report["cases"]) or not set(blocks) <= dump_cases.keys():
            raise ValueError("full stage capture must include each selected block once")
        dump_sha = sha(raw)
    layout = buffer_offsets(storage_tokens)
    cases = []
    with Sm86Mma(ordinal=device) as mma:
        hardware_info = mma.hardware_info
        hardware = partial(gpu_split, mma)
        switches = dict(zip(MODES, ((CPU_SPLIT, CPU_SPLIT), (hardware, CPU_SPLIT),
                                    (CPU_SPLIT, hardware), (hardware, hardware))))
        for block in blocks:
            q, k, v, observed = load_case(block, sealed, files, shape, layout)
            q, k, v = FP8[q], FP8[k], FP8[v]
            measured = read_dump(dump, dump_cases[block], observed, layout[3], tokens) if dump else None
            models = {}
            for label in modes:
                qk, pv = switches[label]
                stages = evaluate(q, k, v, qk, pv, tokens=tokens)
                expected = encode_output(stages["pre_quantized"], tokens)
                models[label] = {"output": metrics(expected, observed)}
                if storage_tokens != tokens:
                    models[label]["valid_query_output"] = metrics(expected[:, :tokens], observed[:, :tokens])
                if measured is not None:
                    models[label]["against_measured_stages"] = diagnostic(stages, measured, observed, tokens)
            cases.append({"block": block, "values_compared": observed.size, "models": models})
    totals = {label: sum(row["models"][label]["output"]["different_numeric_count"] for row in cases) for label in modes}
    sources = {Path(path).name: sha(Path(path).read_bytes())
               for path in (__file__, model.__file__, mma_sm86_tiles.__file__)}
    return {"schema": "october_attention_tensor_comparison_v1", "source_sha256": sources,
            "capture_seal_sha256": seal, "full_dump_report_sha256": dump_sha, **hardware_info,
            "capture_frame": sealed["frame"], "shape_hwc": shape, "tokens": tokens, "storage_tokens": storage_tokens,
            "scope": "64-key streaming with Ampere split K16 MMA, host exponent, reduction and reciprocal.",
            "cases": cases, "total_different_numeric_values": totals,
            "bit_exact_gpu_mma": (all(row["models"]["gpu_qk_pv"]["output"]["different_byte_count"] == 0 for row in cases)
                                  if "gpu_qk_pv" in modes else None)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--blocks", type=int, nargs="+", choices=range(11, 19), default=list(range(11, 19)))
    parser.add_argument("--dump", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    args = parser.parse_args()
    if len(set(args.blocks)) != len(args.blocks) or len(set(args.modes)) != len(args.modes) or args.device < 0:
        parser.error("use distinct block numbers and modes, and a nonnegative device")
    try:
        report = run(args.capture, args.blocks, args.dump, args.device, args.modes)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(json.dumps({"error": str(error)}, indent=2))
        return 1
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
