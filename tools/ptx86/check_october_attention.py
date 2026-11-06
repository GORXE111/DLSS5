"""Compare October preset-2 attention taps with an independent CPU oracle.

Uses the existing capture, not another launch. The oracle models the FP8
layout, PTX half-precision reductions, and the staged port's split MMA. Exact
hardware accumulation and reciprocal rounding are measured, not assumed.
"""

import argparse
from functools import partial
from hashlib import sha256
import json
from pathlib import Path
import struct
import sys

import numpy as np

from ..align_nr_small_trace import kernel_inventory, read_trace
from .check_october_live_repack import tap_index


ENTRY = "cc_vit_1d_attention_chained_fp8"
STAGED_DLL_SHA256 = "2d3e405535cfb574828a5940f62d14952044f4feb1013caf9e7b9240385506ed"
ROLES = ("q", "k", "v", "out")
COLS32 = np.array([0, 16, 2, 18, 4, 20, 6, 22, 1, 17, 3, 19, 5, 21, 7, 23,
                   8, 24, 10, 26, 12, 28, 14, 30, 9, 25, 11, 27, 13, 29, 15, 31])
EXP = (0x3DB76078, 0x3FDACC5B, 0x3FB84000, 0x3FFD2000)


def mma_groups(bit):
    axis = np.arange(32)
    return np.flatnonzero((axis & (1 << bit)) == 0), np.flatnonzero(axis & (1 << bit))


# K components and V tokens both put their logical bit 3 in bit 1 of
# the physical byte offset. The sm86 rewrite splits bytes 0/1 from 2/3.
PHYSICAL_GROUPS = mma_groups(3)


def sha(raw):
    return sha256(raw).hexdigest()


def f16(x):
    return np.asarray(x, dtype=np.float16).astype(np.float32)


def half_literal(word):
    return np.float16(struct.unpack("<f", struct.pack("<I", word))[0]).item()


def fp8_table():
    byte = np.arange(256, dtype=np.uint8)
    exponent, fraction = (byte >> 3) & 15, (byte & 7) / 8.0
    magnitude = np.where(exponent == 0, fraction * 2 ** -6, np.exp2(exponent.astype(np.float32) - 7) * (1 + fraction))
    values = np.where(byte & 128, -magnitude, magnitude).astype(np.float32)
    values[(byte & 127) == 127] = np.nan
    return values


FP8 = fp8_table()


def encode_fp8(x):
    mag = np.abs(x).clip(0, 448)
    high = np.searchsorted(FP8[:127], mag).clip(0, 126)
    low = (high - 1).clip(0, 126)
    dl, dh = mag - FP8[low], FP8[high] - mag
    value = np.where((dl < dh) | ((dl == dh) & (low % 2 == 0)), low, high)
    return (value | (np.signbit(x).astype(np.uint8) << 7)).astype(np.uint8)


def bits(array, mapping):
    return sum(((array >> source) & 1) * weight for source, weight in enumerate(mapping))


def buffer_offsets(tokens):
    """Returns physical byte offsets for logical arrays [head, token, component]."""
    h, n, d = np.ogrid[:32, :tokens, :32]
    # To select logical component d from TIN, invert fragment->canonical.
    c = np.argsort(COLS32)[d]
    lane = ((n % 8) << 2) + (c % 8) // 2
    byte = (c // 8) + ((n % 16) // 8) * 4 + (c % 2) * 8
    qo = (n // 16) * 16384 + h * 512 + lane * 16 + byte
    k = (n // 16) * 16384 + h * 512 + bits(n, (64, 128, 256, 8)) + bits(d, (1, 16, 32, 2, 4))
    v = (n // 32) * 32768 + h * 1024 + bits(d, (64, 128, 256, 8, 512)) + bits(n, (1, 16, 32, 2, 4))
    return qo, k, v, qo


def split_mma(a, b, acc=0, *, groups=PHYSICAL_GROUPS, dtype=np.float32):
    """A logical K32 MMA with the port's low/high byte K16 subgroups."""
    low, high = groups
    a, b = a.astype(dtype, copy=False), b.astype(dtype, copy=False)
    half = f16(a[..., low] @ b[:, low, :] + acc)
    return f16(a[..., high] @ b[:, high, :] + half)


def probabilities(logits, *, fused=True):
    mul, bias, low, high = map(half_literal, EXP)
    # PTX uses fma.rn.f16x2. A double temporary retains the unfused product
    # bits until the one half rounding, unlike the old float tensor path.
    scaled = logits.astype(np.float64) * mul
    if not fused:
        scaled = f16(scaled)
    y = np.asarray(scaled + bias, dtype=np.float16).clip(np.float16(low), np.float16(high))
    bits = (y.view(np.uint16).astype(np.uint32) << 4) + 0x4000
    return bits.astype(np.uint16).view(np.float16).astype(np.float32)


def half_sum(probability):
    """Per 64-key tile: adjacent N8 pairs, four lane groups, then two FP16 halves."""
    p = probability.reshape(*probability.shape[:-1], 8, 4, 2)
    s = f16(f16(p[..., 0, :, :] + p[..., 1, :, :]) + f16(p[..., 2, :, :] + p[..., 3, :, :]))
    s = f16(s + f16(p[..., 4, :, :] + p[..., 5, :, :]))
    s = f16(s + f16(p[..., 6, :, :] + p[..., 7, :, :]))
    s = f16(f16(f16(s[..., 0, :] + s[..., 1, :]) + s[..., 2, :]) + s[..., 3, :])
    return f16(s[..., 0] + s[..., 1])[..., None]


def attention_stages(q, k, v, *, qk=split_mma, pv=split_mma, fused_exp=True, summation="ptx"):
    tokens = q.shape[1]
    logits = qk(q, k.transpose(0, 2, 1))
    prob = probabilities(logits, fused=fused_exp)
    qprob = FP8[encode_fp8(prob)]
    out = np.zeros_like(q)
    for start in range(0, tokens, 32):
        out = pv(qprob[:, :, start:start + 32], v[:, start:start + 32, :], out)

    padded = (-tokens) % 64
    p_zero = probabilities(np.array(0, dtype=np.float32)).item()
    sum_prob = 0
    if summation == "ideal":
        sum_prob = f16(prob.sum(-1, keepdims=True, dtype=np.float64))
    else:
        for start in range(0, tokens, 64):
            tile = prob[..., start:start + 64]
            if tile.shape[-1] != 64:
                tile = np.pad(tile, ((0, 0), (0, 0), (0, padded)),
                              constant_values=p_zero if summation == "ptx" else 0)
            sum_prob = f16(sum_prob + half_sum(tile))
        # Kernel pads the key loop to 64 and removes the zero-logit mass after
        # summing in half. Zero V tiles contribute no mathematical numerator.
        if summation == "ptx":
            sum_prob = f16(sum_prob - f16(p_zero * padded))
    reciprocal_f32 = 1.0 / np.maximum(sum_prob, half_literal(0x388205FF))
    return {"prob": prob, "pv": out, "sum": sum_prob, "reciprocal_f32": reciprocal_f32,
            "pre_quantized": f16(out * f16(reciprocal_f32))}


def oracle(q, k, v):
    return encode_fp8(attention_stages(q, k, v)["pre_quantized"])


def rank(code):
    return (code & 127).astype(np.int16) * np.where(code & 128, -1, 1)


def metrics(expected, observed):
    delta = np.abs(FP8[observed] - FP8[expected])
    steps = np.abs(rank(observed) - rank(expected))
    return {"different_byte_count": int(np.count_nonzero(expected != observed)),
            "different_numeric_count": int(np.count_nonzero(delta)), "exact_numeric_fraction": float(np.mean(delta == 0)),
            "mae": float(delta.mean(dtype=np.float64)), "max_abs": float(delta.max()),
            "within_one_e4m3_step_fraction": float(np.mean(steps <= 1)), "max_e4m3_steps": int(steps.max())}


def output_point(index, expected, observed, stages, offsets):
    h, n, d = map(int, index)
    index = h, n, d
    return {"head": h, "token": n, "component": d, "packed_byte_offset": int(offsets[index]),
            "expected_byte": f"0x{expected[index]:02x}", "observed_byte": f"0x{observed[index]:02x}",
            "expected_value": float(FP8[expected[index]]), "observed_value": float(FP8[observed[index]]),
            "pv_half": float(stages["pv"][index]), "sum_half": float(stages["sum"][h, n, 0]),
            "reciprocal_half": float(f16(stages["reciprocal_f32"][h, n, 0])),
            "pre_quantized_half": float(stages["pre_quantized"][index])}


def rcp_sensitivity(stages, expected, observed):
    inv = stages["reciprocal_f32"]
    values = inv.view(np.uint32).astype(np.int64)
    bands = []
    for distance in (1, 4, 8, 16, 64):
        low, high = (f16((values + side * distance).astype(np.uint32).view(np.float32)) for side in (-1, 1))
        possible = np.abs(rank(encode_fp8(f16(stages["pv"] * low))) - rank(
            encode_fp8(f16(stages["pv"] * high)))) != 0
        bands.append({"f32_ulps": distance, "half_reciprocal_rows_that_can_change": int(np.count_nonzero(low != high)),
                      "output_values_that_can_change": int(possible.sum()),
                      "baseline_residual_values_in_envelope": int(np.count_nonzero(possible & (rank(expected) != rank(observed))))})
    return bands


def zero_query_probe(index, q, v, expected, observed, stages, offsets, old_expected, old_stages):
    # These captured rows are independent of the QK dot and exponential input.
    if q.shape[1] != 96 or np.any(q[index[:2]]):
        return None
    h, n, d = index
    prob = stages["prob"][h, n, 0]
    prob8 = FP8[encode_fp8(np.array(prob))].item()
    vector = v[h, :, d]
    partials = []
    for group_name, groups in (("physical_byte_split", PHYSICAL_GROUPS), ("previous_logical_bit1", mma_groups(1)),
                               ("contiguous_halves", mma_groups(4))):
        accumulator, checkpoints = 0, []
        for start in range(0, len(vector), 32):
            for indices in groups:
                accumulator = f16(accumulator + (prob8 * vector[start + indices]).sum(dtype=np.float64)).item()
                checkpoints.append(accumulator)
        partials.append({"grouping": group_name, "half_accumulator_after_each_k16": checkpoints})
    original_pv = np.float16(stages["pv"][index])
    inv = f16(stages["reciprocal_f32"][h, n, 0])
    one_ulp = []
    for direction in (-np.inf, np.inf):
        pv = np.nextafter(original_pv, np.float16(direction))
        code = encode_fp8(f16(pv * inv)).item()
        one_ulp.append({"pv_half": float(pv), "output_byte": f"0x{code:02x}", "output_value": float(FP8[code])})
    return {"physical": output_point(index, expected, observed, stages, offsets),
            "previous_logical_bit1": output_point(index, old_expected, observed, old_stages, offsets),
            "all_q_components_zero": True, "probability_half": float(prob), "probability_e4m3": prob8,
            "decoded_v_by_token": vector.tolist(), "k16_partial_accumulators": partials,
            "output_if_pv_is_one_half_ulp_lower_or_higher": one_ulp}


def residual_diagnostics(q, k, v, observed, expected, stages, offsets):
    full_k32 = lambda a, b, acc=0: f16(a @ b + acc)
    old = partial(split_mma, groups=mma_groups(1))
    legacy_stages = attention_stages(q, k, v, qk=old, pv=old)
    legacy_expected = encode_fp8(legacy_stages["pre_quantized"])
    alternatives = {"previous_logical_bit1": legacy_expected}
    for label, change in (
        ("qk_float64_products", {"qk": partial(split_mma, dtype=np.float64)}),
        ("pv_float64_products", {"pv": partial(split_mma, dtype=np.float64)}),
        ("qk_single_k32_half_round", {"qk": full_k32}),
        ("pv_single_k32_half_round", {"pv": full_k32}),
        ("qk_and_pv_contiguous_k16_halves", {"qk": partial(split_mma, groups=mma_groups(4)),
                                            "pv": partial(split_mma, groups=mma_groups(4))}),
        ("unfused_exp", {"fused_exp": False}),
        ("ideal_sum", {"summation": "ideal"}),
        ("sum_without_zero_key_correction", {"summation": "no_tail"}),
    ):
        alternatives[label] = encode_fp8(attention_stages(q, k, v, **change)["pre_quantized"])
    numeric_error = rank(expected) != rank(observed)
    ablations = {label: {**metrics(value, observed),
                         "changed_from_physical_oracle": int(np.count_nonzero(rank(value) != rank(expected))),
                         "physical_residuals_resolved": int(np.count_nonzero(numeric_error & (rank(value) == rank(observed))))}
                 for label, value in alternatives.items()}
    expected_mag, observed_mag = np.abs(FP8[expected]), np.abs(FP8[observed])
    delta = np.abs(FP8[expected] - FP8[observed])
    steps = np.abs(rank(expected) - rank(observed))

    def worst(values, pred, pred_stages):
        rows = np.argwhere(values == values.max())[:4] if values.max() else []
        return [output_point(idx, pred, observed, pred_stages, offsets) for idx in rows]

    zero_q = (q == 0).all(-1)
    probes = [zero_query_probe(index, q, v, expected, observed, stages, offsets, legacy_expected, legacy_stages)
              for index in ((7, 10, 1), (28, 10, 29))]
    return {"ablations": ablations, "reciprocal_sensitivity_at_fixed_half_sum": rcp_sensitivity(stages, expected, observed),
            "mismatch_rows": int(np.count_nonzero(numeric_error.any(-1))),
            "rows_needing_magnitude_changes_in_both_directions_at_fixed_pv": int(np.count_nonzero(
                (expected_mag > observed_mag).any(-1) & (expected_mag < observed_mag).any(-1))),
            "mismatches_by_head": numeric_error.sum(axis=(1, 2)).tolist(),
            "zero_q_rows": int(zero_q.sum()), "zero_q_numeric_mismatches": int(numeric_error[zero_q].sum()),
            "previous_zero_q_numeric_mismatches": int((rank(legacy_expected[zero_q]) != rank(observed[zero_q])).sum()),
            "worst_absolute": worst(delta, expected, stages), "worst_e4m3_steps": worst(steps, expected, stages),
            "previous_worst_absolute": worst(np.abs(FP8[legacy_expected] - FP8[observed]), legacy_expected, legacy_stages),
            "previous_worst_e4m3_steps": worst(np.abs(rank(legacy_expected) - rank(observed)), legacy_expected, legacy_stages),
            "zero_query_probes": [probe for probe in probes if probe is not None]}


def verify_capture(folder, trace):
    run_path = folder / "run.json"
    run_bytes = run_path.read_bytes()
    run = json.loads(run_bytes)
    if (run["exit_code"] != 0 or run["report"]["model"] != 2 or run["report"]["pipeline"] != "nr-only"
            or not run["report"]["outputCoverageValidated"]
            or run["runtime_sha256"]["nvngx_dlssnr.dll"] != STAGED_DLL_SHA256
            or run["trace_sha256"] != sha(trace)):
        raise ValueError("run evidence does not identify a successful pinned October preset-2 capture")
    files = {**run["runtime_sha256"], **run["evidence_sha256"], **run["outputs"], "tap.cubin": run["tap_cubin_sha256"]}
    for name, expected in files.items():
        path = (folder / name).resolve()
        if path.parent != folder or sha(path.read_bytes()) != expected:
            raise ValueError(f"capture evidence hash mismatch: {name}")
    return {"run_sha256": sha(run_bytes), "pinned_staged_dll_sha256": STAGED_DLL_SHA256,
            "trace_sha256": sha(trace), "tap_index_sha256": files["tap_index.tsv"], "tap_spec": run["tap_spec"]}


def compare_launch(launch, taps, diagnose=False):
    h, w = struct.unpack_from("<ii", launch.params, 56)
    tokens = h * w
    if h <= 0 or w <= 0 or tokens % 32 or tokens > 1024:
        raise ValueError("attention oracle supports 32-token groups up to 1024 tokens")
    size = tokens * 1024
    addresses = struct.unpack_from("<4Q", launch.params)
    if min(addresses) <= 0 or any(b - a < size for a, b in zip(sorted(addresses), sorted(addresses)[1:])):
        raise ValueError("attention input/output buffers overlap")
    buffers, payloads = [], []
    orders = buffer_offsets(tokens)
    for role, ptr, offsets in zip(ROLES, addresses, orders, strict=True):
        declared, path = taps[launch.sequence, ptr]
        raw = path.read_bytes()
        if declared != size or len(raw) != size or np.unique(offsets).size != size:
            raise ValueError(f"invalid {role} capture length or buffer permutation")
        data = np.frombuffer(raw, dtype=np.uint8)[offsets]
        if ((data & 127) == 127).any():
            raise ValueError(f"NaN in {role} tap, outside this finite numerical oracle")
        buffers.append(data)
        payloads.append({"role": role, "address": f"0x{ptr:x}", "bytes": size, "sha256": sha(raw)})
    q, k, v = (FP8[data] for data in buffers[:3])
    stages = attention_stages(q, k, v)
    expected, observed = encode_fp8(stages["pre_quantized"]), buffers[3]
    packed = np.empty(size, dtype=np.uint8)
    packed[orders[3]] = expected
    result = {**launch.identity(), "height": h, "width": w, "tokens": tokens, "heads": 32, "head_dim": 32,
              "payloads": payloads, "expected_packed_output_sha256": sha(packed.tobytes()),
              "values": size, **metrics(expected, observed)}
    if diagnose:
        result["residuals"] = residual_diagnostics(q, k, v, observed, expected, stages, orders[3])
    return result


def run(folder, frame, sequences, diagnose=False):
    folder = folder.resolve()
    trace = (folder / "nr-trace.tsv").read_bytes()
    identity = verify_capture(folder, trace)
    frames = read_trace(trace, kernel_inventory())
    if frame not in frames:
        raise ValueError("trace lacks the requested frame ID")
    taps = tap_index(folder, frame)
    selected = [launch for launch in frames[frame] if launch.sequence in sequences and launch.name == ENTRY]
    if len(selected) != len(sequences) or len(set(sequences)) != len(sequences):
        raise ValueError("selected sequences do not identify unique October attention launches in that frame")
    chain_counts = {row.chain: sum(other.chain == row.chain for other in frames[frame]) for row in selected}
    if any(count != 1 for count in chain_counts.values()):
        raise ValueError("attention taps require single-kernel launch chains")
    cases = [compare_launch(launch, taps, diagnose) for launch in selected]
    requested_frame, *requests = identity["tap_spec"].split(";")
    if int(requested_frame) != frame or any(
        f"{case['sequence']}:p{slot}:{payload['bytes']}" not in requests
        for case in cases for slot, payload in enumerate(case["payloads"])
    ):
        raise ValueError("run evidence does not request the observed attention slots and byte sizes")
    return {"schema": "october_nr_attention_cpu_v2", "frame": frame, **identity,
            "source_sha256": sha(Path(__file__).read_bytes()), "cases": cases,
            "sm86_split_mma_k16_groups": [indices.tolist() for indices in PHYSICAL_GROUPS],
            "scope": "Offline arithmetic for the tapped staged sm86 attention only. CPU MMA internals and reciprocal are approximations. "
                     "Capture run hashes its metadata and logs; individual tensor hashes are recorded by this checker.",
            "bit_exact": all(case["different_byte_count"] == 0 for case in cases)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--sequence", type=int, nargs="+", default=(23, 58))
    parser.add_argument("--diagnose", action="store_true", help="include residual coordinates and numeric ablations")
    parser.add_argument("--require-bit-exact", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args.capture, args.frame, args.sequence, args.diagnose)
    except (OSError, ValueError, KeyError, struct.error) as error:
        print(json.dumps({"error": str(error)}, indent=2))
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return int(args.require_bit_exact and not result["bit_exact"])


if __name__ == "__main__":
    sys.exit(main())
