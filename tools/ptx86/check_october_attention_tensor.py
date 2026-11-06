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
from .check_october_attention import FP8, attention_stages, buffer_offsets, encode_fp8, metrics, probabilities, sha, split_mma
from .mma_sm86_tiles import Sm86Mma


TOKENS = 96
K16 = (np.arange(16), np.arange(16, 32))
CPU_SPLIT = partial(split_mma, groups=K16)
ROLES = ("in0", "in1", "in2", "out")
MODES = ("cpu", "gpu_qk", "gpu_pv", "gpu_qk_pv")


def load_case(block, sealed, files, layout):
    tensors = []
    for suffix, indices in zip(ROLES, layout):
        role = f"block{block}.attention.{suffix}"
        boundary = sealed["boundaries"][role]
        data = files[role].read_bytes()
        if boundary["shape_hwc"] != (8, 12, 1024) or sha(data) != boundary["sha256"]:
            raise ValueError(f"{role} requires an intact 96-token tap")
        codes = np.frombuffer(data, np.uint8)[indices]
        if np.any((codes & 127) == 127):
            raise ValueError(f"{role} contains a non-finite E4M3 value")
        tensors.append(codes)
    return tensors


def gpu_split(mma, a, b, acc=0):
    c = None if np.isscalar(acc) and acc == 0 else acc
    low = mma(a[..., :16], b[:, :16, :], c)
    return mma(a[..., 16:], b[:, 16:, :], low).astype(np.float32)


def evaluate(q, k, v, qk, pv):
    pad_p = FP8[encode_fp8(probabilities(np.array(0, dtype=np.float32)))]
    pad_a, pad_b = np.full_like(q, pad_p), np.zeros_like(v[:, :32])
    next_tile = 0

    def padded_pv(a, b, acc):
        nonlocal next_tile
        next_tile += 1
        acc = pv(a, b, acc)
        if next_tile == TOKENS // 32:
            acc = pv(pad_a, pad_b, acc)
        return acc

    return attention_stages(q, k, v, qk=qk, pv=padded_pv)


def half_metrics(predicted, measured):
    a, b = np.asarray(predicted, dtype="<f2"), np.asarray(measured, dtype="<f2")
    if a.shape != b.shape:
        raise ValueError("stage tensor shapes differ")
    delta = np.abs(a.astype(np.float32) - b.astype(np.float32))
    return {"different_numeric_values": int(np.count_nonzero(delta)),
            "different_half_words": int(np.count_nonzero(a.view("<u2") != b.view("<u2"))),
            "mae": float(delta.mean(dtype=np.float64)), "max_abs": float(delta.max())}


def read_dump(folder, row, expected, layout):
    arrays = {}
    for role in ("pv", "inv", "norm"):
        artifact = row["planes"][role]
        data = (folder / f"block{row['block']:02d}-{role}.f16").read_bytes()
        if len(data) != expected.size * 2 or len(data) != artifact["bytes"] or sha(data) != artifact["sha256"]:
            raise ValueError("full stage dump failed its size or SHA-256 check")
        arrays[role] = np.frombuffer(data, "<f2")[layout]
    if not np.array_equal(encode_fp8(arrays["norm"].astype(np.float32)), expected):
        raise ValueError("captured pre-quantization tensor does not encode to the sealed output")
    return arrays


def diagnostic(stages, measured, observed):
    predicted_pv = stages["pv"].astype(np.float16)
    predicted_inv = np.broadcast_to(stages["reciprocal_f32"].astype(np.float16), predicted_pv.shape)
    return {"pv": half_metrics(predicted_pv, measured["pv"]), "reciprocal": half_metrics(predicted_inv, measured["inv"]),
            "pre_quantization": half_metrics(stages["pre_quantized"], measured["norm"]),
            "with_measured_pv": metrics(encode_fp8((measured["pv"] * predicted_inv).astype(np.float32)), observed),
            "with_measured_reciprocal": metrics(encode_fp8((predicted_pv * measured["inv"]).astype(np.float32)), observed),
            "with_both_measured": metrics(encode_fp8((measured["pv"] * measured["inv"]).astype(np.float32)), observed)}


def run(capture, blocks, dump, device):
    sealed, files = seal_capture(capture, kernel_inventory(), json.loads(pinned_bytes(GRAPH, GRAPH_SHA)))
    if sealed["kind"] != PORT or sealed["frame"] != 0:
        raise ValueError("expected a sealed staged Ampere frame-zero capture")
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
    layout = buffer_offsets(TOKENS)
    cases = []
    with Sm86Mma(ordinal=device) as mma:
        hardware_info = mma.hardware_info
        hardware = partial(gpu_split, mma)
        switches = ((CPU_SPLIT, CPU_SPLIT), (hardware, CPU_SPLIT), (CPU_SPLIT, hardware), (hardware, hardware))
        for block in blocks:
            q, k, v, observed = load_case(block, sealed, files, layout)
            measured = read_dump(dump, dump_cases[block], observed, layout[3]) if dump else None
            models = {}
            for label, (qk, pv) in zip(MODES, switches):
                stages = evaluate(FP8[q], FP8[k], FP8[v], qk, pv)
                expected = encode_fp8(stages["pre_quantized"])
                models[label] = {"output": metrics(expected, observed)}
                if measured is not None:
                    models[label]["against_measured_stages"] = diagnostic(stages, measured, observed)
            cases.append({"block": block, "values_compared": observed.size, "models": models})
    totals = {label: sum(row["models"][label]["output"]["different_numeric_count"] for row in cases) for label in MODES}
    sources = {Path(path).name: sha(Path(path).read_bytes())
               for path in (__file__, model.__file__, mma_sm86_tiles.__file__)}
    return {"schema": "october_attention_tensor_comparison_v1", "source_sha256": sources,
            "capture_seal_sha256": seal, "full_dump_report_sha256": dump_sha, **hardware_info,
            "scope": "Ampere split K16 MMA with host exponent, reduction and reciprocal; sealed 96-token inputs.",
            "cases": cases, "total_different_numeric_values": totals,
            "bit_exact_gpu_mma": all(row["models"]["gpu_qk_pv"]["output"]["different_byte_count"] == 0 for row in cases)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--blocks", type=int, nargs="+", choices=range(11, 19), default=list(range(11, 19)))
    parser.add_argument("--dump", type=Path)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if len(set(args.blocks)) != len(args.blocks) or args.device < 0:
        parser.error("use distinct block numbers and a nonnegative device")
    try:
        report = run(args.capture, args.blocks, args.dump, args.device)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(json.dumps({"error": str(error)}, indent=2))
        return 1
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
