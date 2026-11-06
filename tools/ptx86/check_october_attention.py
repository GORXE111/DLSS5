"""Compare October preset-2 attention taps with an independent CPU oracle.

Uses the existing capture, not another launch. The oracle models the FP8
layout, PTX half-precision reductions, and the staged port's split MMA. Exact
hardware accumulation and reciprocal rounding are measured, not assumed.
"""

import argparse
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
MMA_LO = np.flatnonzero(np.arange(32) % 4 < 2)
MMA_HI = np.flatnonzero(np.arange(32) % 4 >= 2)


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


def split_mma(a, b, acc=0):
    """A logical K32 MMA with the port's low/high byte K16 subgroups."""
    half = f16(a[..., MMA_LO] @ b[:, MMA_LO, :] + acc)
    return f16(a[..., MMA_HI] @ b[:, MMA_HI, :] + half)


def probabilities(logits):
    mul, bias, low, high = map(half_literal, EXP)
    # PTX uses fma.rn.f16x2. A double temporary retains the unfused product
    # bits until the one half rounding, unlike the old float tensor path.
    y = np.asarray(logits.astype(np.float64) * mul + bias, dtype=np.float16).clip(np.float16(low), np.float16(high))
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


def oracle(q, k, v):
    tokens = q.shape[1]
    logits = split_mma(q, k.transpose(0, 2, 1))
    prob = probabilities(logits)
    qprob = FP8[encode_fp8(prob)]
    out = np.zeros_like(q)
    for start in range(0, tokens, 32):
        out = split_mma(qprob[:, :, start:start + 32], v[:, start:start + 32, :], out)

    padded = (-tokens) % 64
    p_zero = probabilities(np.array(0, dtype=np.float32)).item()
    sum_prob = 0
    for start in range(0, tokens, 64):
        tile = prob[..., start:start + 64]
        if tile.shape[-1] != 64:
            tile = np.pad(tile, ((0, 0), (0, 0), (0, padded)), constant_values=p_zero)
        sum_prob = f16(sum_prob + half_sum(tile))
    # Kernel pads the key loop to 64 and removes the zero-logit mass after
    # summing in half. The zero V tiles contribute no mathematical numerator.
    sum_prob = f16(sum_prob - f16(p_zero * padded))
    inv = f16(1.0 / np.maximum(sum_prob, half_literal(0x388205FF)))
    return encode_fp8(f16(out * inv))


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


def compare_launch(launch, taps):
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
    expected, observed = oracle(q, k, v), buffers[3]
    actual_values, expected_values = FP8[observed], FP8[expected]
    delta = np.abs(actual_values - expected_values)

    def rank(code):
        return (code & 127).astype(np.int16) * np.where(code & 128, -1, 1)

    steps = np.abs(rank(observed) - rank(expected))
    packed = np.empty(size, dtype=np.uint8)
    packed[orders[3]] = expected
    return {**launch.identity(), "height": h, "width": w, "tokens": tokens, "heads": 32, "head_dim": 32,
            "payloads": payloads, "expected_packed_output_sha256": sha(packed.tobytes()),
            "values": size, "different_byte_count": int(np.count_nonzero(expected != observed)),
            "different_numeric_count": int(np.count_nonzero(delta)), "exact_numeric_fraction": float(np.mean(delta == 0)),
            "mae": float(delta.mean(dtype=np.float64)), "max_abs": float(delta.max()),
            "within_one_e4m3_step_fraction": float(np.mean(steps <= 1)), "max_e4m3_steps": int(steps.max())}


def run(folder, frame, sequences):
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
    cases = [compare_launch(launch, taps) for launch in selected]
    requested_frame, *requests = identity["tap_spec"].split(";")
    if int(requested_frame) != frame or any(
        f"{case['sequence']}:p{slot}:{payload['bytes']}" not in requests
        for case in cases for slot, payload in enumerate(case["payloads"])
    ):
        raise ValueError("run evidence does not request the observed attention slots and byte sizes")
    return {"schema": "october_nr_attention_cpu_v1", "frame": frame, **identity,
            "source_sha256": sha(Path(__file__).read_bytes()), "cases": cases,
            "scope": "Offline arithmetic for the tapped staged sm86 attention only. CPU MMA internals and reciprocal are approximations. "
                     "Capture run hashes its metadata and logs; individual tensor hashes are recorded by this checker.",
            "bit_exact": all(case["different_byte_count"] == 0 for case in cases)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--sequence", type=int, nargs="+", default=(23, 58))
    parser.add_argument("--require-bit-exact", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args.capture, args.frame, args.sequence)
    except (OSError, ValueError, KeyError, struct.error) as error:
        print(json.dumps({"error": str(error)}, indent=2))
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return int(args.require_bit_exact and not result["bit_exact"])


if __name__ == "__main__":
    sys.exit(main())
