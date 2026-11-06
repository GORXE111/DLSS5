"""Instrument two selected points of the pinned ordinary October sm86 attention entry."""

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

from ..align_nr_small_trace import GRAPH, GRAPH_SHA, kernel_inventory, pinned_bytes
from ..capture_october_taps import capture_path
from ..compare_october_taps import PORT, seal_capture
from .check_october_attention import (
    COLS32, FP8, attention_stages, buffer_offsets, encode_fp8, metrics, probabilities, split_mma,
)
from .probe_october_attention_replay import GUARD
from .probe_october_fence import Driver, sha


PTX_SHA = "f6bd97ebc3db9d9c190e731144005ec2bda23b147325bcdbf77f784007a88f29"
PTXAS_SHA = "86b2ff48655c55f1547edc59b503d39a62aa5c71745b7ecf6be526c29af0b693"
ENTRY = "cc_vit_1d_attention_fp8"
# Logical head/token/component, writer lane, packed byte, half2 destination/PV/reciprocal.
POINTS = ((7, 10, 1, 8, 3717, "%r985", "%r1435", "%r922"),
          (28, 10, 29, 10, 14511, "%r990", "%r1429", "%r934"))
MMA_A = (("%r643_ql", "%r644_ql", "%r643_qh", "%r644_qh"),
         ("%r645_ql", "%r646_ql", "%r645_qh", "%r646_qh"),
         ("%r653_ql", "%r654_ql", "%r653_qh", "%r654_qh"),
         ("%r655_ql", "%r656_ql", "%r655_qh", "%r656_qh"))
MMA_ACC = ((("%r1436", "%r1435"), ("%r651", "%r652")),
           (("%r1430", "%r1429"), ("%r671", "%r672")))
MMA_STEPS, MMA_LANES, MMA_STRIDE = 8, 32, 48
MMA_BYTES_PER_POINT = MMA_STEPS * MMA_LANES * MMA_STRIDE
DEBUG_HEADER = 64
DEBUG_BYTES = DEBUG_HEADER + len(POINTS) * MMA_BYTES_PER_POINT
MARKER = 0x38


def replace_once(body, pattern, replacement):
    result, count = re.subn(pattern, replacement, body, flags=re.M)
    if count != 1:
        raise ValueError(f"expected exactly one instruction anchor, found {count}")
    return result


def instrument(data):
    if sha(data) != PTX_SHA:
        raise ValueError("not the pinned rewritten October module 6 PTX")
    source = data.decode("ascii")
    start = source.index(f".visible .entry {ENTRY}(")
    stop = source.index(".visible .entry ", start + 1)
    body = source[start:stop]
    args = ENTRY + "_param_0"
    if re.search(rf"\[{args}\+(32|40)\]", body):
        raise ValueError("the ordinary attention entry already uses a debug parameter slot")
    prologue = f"""
.reg .pred %nr_first, %nr_active, %nr_hit, %nr_marker, %nr_replace;
.reg .b32 %nr_head, %nr_warp, %nr_cta_y, %nr_lane, %nr_mode, %nr_index;
.reg .b64 %nr_address, %nr_debug, %nr_mma;
ld.param.b64 %nr_address, [{args}+32];
cvta.to.global.u64 %nr_debug, %nr_address;
ld.param.u32 %nr_mode, [{args}+40];
setp.eq.u32 %nr_marker, %nr_mode, 1;
mov.u32 %nr_head, %ctaid.x;
mov.u32 %nr_cta_y, %ctaid.y;
mov.u32 %nr_warp, %tid.y;
mov.u32 %nr_lane, %laneid;
or.b32 %nr_warp, %nr_warp, %nr_cta_y;
setp.eq.u32 %nr_first, %nr_warp, 0;
"""
    body = replace_once(body, rf"^ld\.param\.b64 %rd1, \[{args}\];$", lambda m: prologue + m[0])

    def stores(head, lane, offset, regs):
        lines = [f"setp.eq.u32 %nr_active, %nr_head, {head};",
                 f"setp.eq.u32 %nr_hit, %nr_lane, {lane};",
                 "and.pred %nr_hit, %nr_hit, %nr_active;", "and.pred %nr_hit, %nr_hit, %nr_first;"]
        lines.extend(f"@%nr_hit st.global.b32 [%nr_debug+{offset + 4*i}], {reg};"
                     for i, reg in enumerate(regs))
        return "\n".join(lines) + "\n"

    denominator = "".join(stores(p[0], 10, index * 32, ("%r1450", "%r910", "%r909"))
                          for index, p in enumerate(POINTS))
    raw_rcp = "".join(stores(p[0], 10, index * 32 + 24, ("fl",)) for index, p in enumerate(POINTS))
    body = replace_once(body, r"^rcp\.approx\.ftz\.f32 fu, fu;[ \t]*\n(?=cvt\.rn\.f16\.f32 hl, fl;[ \t]*\n"
                        r"cvt\.rn\.f16\.f32 hu, fu;[ \t]*\nmov\.b32 %r909,)",
                        lambda m: m[0] + raw_rcp)
    body = replace_once(body, r"^mov\.b32 %r909, \{hl, hu\};[ \t]*\n\}\n", lambda m: m[0] + denominator)
    for index, (head, _, _, lane, _, dst, pv, inv) in enumerate(POINTS):
        capture = stores(head, lane, index * 32 + 12, (pv, inv, dst))
        capture += (f"and.pred %nr_replace, %nr_hit, %nr_marker;\n"
                    f"@%nr_replace and.b32 {dst}, {dst}, 0x0000ffff;\n"
                    f"@%nr_replace or.b32 {dst}, {dst}, 0x3c000000;\n")
        anchor = rf"^\{{mul\.f16x2 {dst},{pv},{inv};\n\}}\n"
        body = replace_once(body, anchor, lambda m: m[0] + capture)
        carried, temporary = MMA_ACC[index]
        shared = ("%m_t0", "%m_t1")
        for step, (c, d, a) in enumerate(zip((carried, shared, temporary, shared),
                                            (shared, temporary, shared, carried), MMA_A)):
            b = ("%m_c2", "%m_c3") if step % 2 else ("%m_c0", "%m_c1")
            regs = lambda names: "{" + ", ".join(names) + "}"
            instruction = "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 " + ", ".join(map(regs, (d, a, b, c))) + ";"
            base = DEBUG_HEADER + index * MMA_BYTES_PER_POINT
            before = f"""
setp.eq.u32 %nr_active, %nr_head, {head};
setp.lt.u32 %nr_hit, %r20, 2;
and.pred %nr_hit, %nr_hit, %nr_active;
and.pred %nr_hit, %nr_hit, %nr_first;
mad.lo.u32 %nr_index, %r20, 4, {step};
mad.lo.u32 %nr_index, %nr_index, 32, %nr_lane;
mul.wide.u32 %nr_mma, %nr_index, {MMA_STRIDE};
add.u64 %nr_mma, %nr_debug, %nr_mma;
@%nr_hit st.global.v4.b32 [%nr_mma+{base}], {regs(a)};
@%nr_hit st.global.v4.b32 [%nr_mma+{base + 16}], {regs(b + c)};
"""
            after = f"@%nr_hit st.global.v2.b32 [%nr_mma+{base + 32}], {regs(d)};\n"
            body = replace_once(body, "^" + re.escape(instruction) + "\n", lambda m: before + m[0] + after)
    return (source[:start] + body + source[stop:]).encode("ascii")


def execute(device, function, inputs, expected, marker, *, poison=0):
    poisoned = (bytes(np.frombuffer(expected, np.uint8) ^ np.uint8(255))
                if expected is not None else bytes([poison]) * len(inputs[0]))
    original = [*inputs, poisoned, b"\xa5" * DEBUG_BYTES]
    with ExitStack() as cleanup:
        addresses = []
        for payload in original:
            wrapped = GUARD + payload + GUARD
            ptr = C.c_uint64()
            device.call("cuMemAlloc_v2", C.byref(ptr), len(wrapped))
            cleanup.callback(device.call, "cuMemFree_v2", ptr)
            device.call("cuMemcpyHtoD_v2", ptr, C.create_string_buffer(wrapped), len(wrapped))
            addresses.append(ptr)
        values = [ptr.value + len(GUARD) for ptr in addresses]
        abi = C.create_string_buffer(struct.pack("<QQQQQIIQii", *values, int(marker), 0, 0, 8, 12))
        params = (C.c_void_p * 1)(C.addressof(abi))
        device.call("cuLaunchKernel", function, 32, 1, 1, 32, 4, 1, 0, None, params, None)
        device.call("cuCtxSynchronize")
        results = []
        for pointer, payload in zip(addresses, original):
            readback = C.create_string_buffer(len(payload) + 2 * len(GUARD))
            device.call("cuMemcpyDtoH_v2", readback, pointer, len(readback))
            if readback.raw[:len(GUARD)] != GUARD or readback.raw[-len(GUARD):] != GUARD:
                raise ValueError("GPU write outside an allocated input, output, or debug payload")
            results.append(readback.raw[len(GUARD):-len(GUARD)])
    if results[:3] != inputs or (expected is not None and results[3] != expected):
        raise ValueError("isolated output differs from its full expected bytes or inputs were modified")
    return results[3], results[4]


def half_pair(word):
    return np.frombuffer(struct.pack("<I", word), "<f2").astype(float).tolist()


def mma_checkpoints(debug, index, lane, output_offset, logical_v, zero_q):
    start = DEBUG_HEADER + index * MMA_BYTES_PER_POINT
    words = np.frombuffer(debug[start:start + MMA_BYTES_PER_POINT], "<u4").reshape(MMA_STEPS, MMA_LANES, 12)
    if np.any(words[..., :10] == 0xA5A5A5A5) or np.any(words[..., 10:] != 0xA5A5A5A5):
        raise ValueError("incomplete MMA fragment or unexpected debug write")
    halves = words.view("<f2")
    lanes, slots = np.ogrid[:MMA_LANES, :8]
    group, thread = lanes // 4, lanes % 4
    a_row = group + 8 * ((slots % 4) >= 2)
    a_col = 2 * thread + slots % 2 + 8 * (slots >= 4)
    b_row = a_col[:, :4] + 8 * (slots[:, :4] >= 2)
    row, col = lane // 4 + 8, 2 * (lane % 4) + 1
    columns = np.arange(8)
    packed_bytes = ((output_offset % 16) & ~1) + columns % 2
    logical_columns = COLS32[8 * (packed_bytes % 4) + 2 * (columns // 2) + (packed_bytes >> 3)]
    fingerprint = logical_v[:, logical_columns]
    out, candidates = [], []
    previous = 0.0
    indices = np.arange(32)
    for i, fragment in enumerate(halves):
        a, b = np.empty((16, 16), dtype=np.float16), np.empty((16, 8), dtype=np.float16)
        a[a_row, a_col] = fragment[:, :8]
        b[b_row, group] = fragment[:, 8:12]
        c, d = float(fragment[lane, 15]), float(fragment[lane, 19])
        if c != previous:
            raise ValueError("MMA does not carry the previous half accumulator")
        previous = d
        exact = c + float(a[row].astype(np.float64) @ b[:, col].astype(np.float64))
        checkpoint = {"step": i + 1, "input_half": c, "output_half": d, "single_rne_from_measured_operands": float(np.float16(exact)),
                      "exact_sum_from_measured_operands": exact, "a_nonzero_values": int(np.count_nonzero(a[row])),
                      "b_nonzero_values": int(np.count_nonzero(b[:, col]))}
        matches = np.all(b[:, None, :] == fingerprint[None, :, :], axis=-1)
        candidates.append(matches)
        checkpoint["b_key_rows_matched"] = int(np.count_nonzero(matches.any(axis=1)))
        if zero_q:
            probability = FP8[encode_fp8(probabilities(np.array(0, np.float32)))]
            values = logical_v[i // 2 * 32:(i // 2 + 1) * 32, logical_columns[col]]
            expected = values[((indices & 8) != 0) == bool(i % 2)]
            checkpoint.update(a_matches_uniform_probability=bool(np.all(a[row] == probability)),
                              b_matches_logical_split=bool(np.array_equal(b[:, col], expected)),
                              b_values_match_up_to_permutation=bool(np.array_equal(np.sort(b[:, col]), np.sort(expected))))
        out.append(checkpoint)
    return out, np.stack(candidates)


def case(device, function, block, sealed, files, work):
    roles = [f"block{block}.attention.{r}" for r in ("in0", "in1", "in2", "out")]
    raw = [files[role].read_bytes() for role in roles]
    if any(sha(value) != sealed["boundaries"][role]["sha256"] or len(value) != 98304
           or sealed["boundaries"][role]["shape_hwc"] != (8, 12, 1024) for role, value in zip(roles, raw)):
        raise ValueError("expected intact 96-token live tap data")
    encoded = [np.frombuffer(value, np.uint8)[indices] for value, indices in zip(raw, buffer_offsets(96))]
    stages = attention_stages(*(FP8[codes] for codes in encoded[:3]))
    _, normal = execute(device, function, raw[:3], raw[3], False)
    marked = bytearray(raw[3])
    for head, token, channel, _, offset, *_ in POINTS:
        if buffer_offsets(96)[3][head, token, channel] != offset:
            raise ValueError("logical-to-packed output mapping differs")
        marked[offset] = MARKER
    _, marked_debug = execute(device, function, raw[:3], bytes(marked), True)
    if marked_debug != normal:
        raise ValueError("marker pass changed arithmetic before the marker injection")
    points, candidates = [], []
    for index, (head, token, channel, lane, offset, *_) in enumerate(POINTS):
        total, clamp, producer, pv, inv, norm, raw_rcp, pad = struct.unpack_from("<8I", normal, index * 32)
        if 0xA5A5A5A5 in (total, clamp, producer, pv, inv, norm, raw_rcp) or pad != 0xA5A5A5A5:
            raise ValueError("incomplete debug record or unexpected debug write")
        point = {"head": head, "token": token, "channel": channel, "writer_lane": lane, "denominator_lane": 10,
                 "output_offset": offset, "gpu_sum_half": half_pair(total)[0],
                 "gpu_clamped_half": half_pair(clamp)[0], "gpu_producer_reciprocal_half": half_pair(producer)[0],
                 "gpu_producer_reciprocal_f32": struct.unpack("<f", struct.pack("<I", raw_rcp))[0],
                 "gpu_pv_half": half_pair(pv)[1], "gpu_reciprocal_half": half_pair(inv)[1],
                 "gpu_pre_quantization_half": half_pair(norm)[1], "gpu_output_code": raw[3][offset],
                 "marker_changed_byte": marked[offset] != raw[3][offset]}
        if point["gpu_producer_reciprocal_half"] != point["gpu_reciprocal_half"]:
            raise ValueError("captured denominator lane does not supply the selected reciprocal")
        point["fp8_rne_from_gpu_half"] = int(encode_fp8(np.float32(point["gpu_pre_quantization_half"])))
        point["cpu_sum_half"] = float(stages["sum"][head, token, 0])
        point["cpu_pv_half"] = float(stages["pv"][head, token, channel])
        point["cpu_reciprocal_half"] = float(np.float16(stages["reciprocal_f32"][head, token, 0]))
        point["cpu_pre_quantization_half"] = float(stages["pre_quantized"][head, token, channel])
        point["cpu_output_code"] = int(encode_fp8(np.float32(point["cpu_pre_quantization_half"])))
        zero_q = not np.any(FP8[encoded[0][head, token]])
        v = np.pad(FP8[encoded[2][head]], ((0, 32), (0, 0)))
        point["mma_checkpoints"], keys = mma_checkpoints(normal, index, lane, offset, v, zero_q)
        candidates.append(keys)
        if point["mma_checkpoints"][-1]["output_half"] != point["gpu_pv_half"]:
            raise ValueError("final observed MMA accumulator differs from the captured PV")
        points.append(point)
    (work / f"block{block}-fragments.bin").write_bytes(normal)
    return ({"block": block, "unchanged_output_sha256": sha(raw[3]), "debug_sha256": sha(normal),
             "marker_output_sha256": sha(bytes(marked)), "points": points}, candidates)


def tagged_order(device, function, files, observations, work):
    """Make each logical key unique in one V column and verify against real operands."""
    q, k, v = [files[f"block11.attention.in{i}"].read_bytes() for i in range(3)]
    indices = buffer_offsets(96)[2]
    tagged = np.frombuffer(v, np.uint8).copy()
    for head, _, column, *_ in POINTS:
        tagged[indices[head, :, column]] = np.arange(1, 97, dtype=np.uint8)
    inputs = [q, k, tagged.tobytes()]
    output, debug = execute(device, function, inputs, None, False, poison=0x5A)
    if execute(device, function, inputs, None, False, poison=0xA5) != (output, debug):
        raise ValueError("tagged test output depends on output buffer initialization")
    (work / "tagged-v-fragments.bin").write_bytes(debug)
    tagged_matches = []
    for index, (head, _, _, lane, offset, *_) in enumerate(POINTS):
        logical = np.pad(FP8[tagged[indices[head]]], ((0, 32), (0, 0)))
        _, matches = mma_checkpoints(debug, index, lane, offset, logical, True)
        tagged_matches.append(matches)
    shared = np.logical_and.reduce(tagged_matches)
    real = np.logical_and.reduce(observations)
    order = []
    for step, rows in enumerate(shared):
        keys = []
        for slot, row in enumerate(rows):
            options = np.flatnonzero(row)
            if len(options) == 1:
                if not real[step, slot, options[0]]:
                    raise ValueError("tagged V key does not match all observed real operands")
                keys.append(int(options[0]))
            elif len(options) and options[0] >= 96 and real[step, slot, 96:].any():
                keys.append("padding")
            else:
                raise ValueError("tagged V did not identify a unique real key or known padding")
        order.append({"step": step + 1, "key_positions": keys})
    if sorted(key for entry in order for key in entry["key_positions"] if isinstance(key, int)) != list(range(96)):
        raise ValueError("tagged V order does not cover every real key exactly once")
    return {"steps": order, "tagged_debug_sha256": sha(debug), "tagged_output_sha256": sha(output),
            "real_operand_matches_checked": len(observations) * MMA_STEPS * 16}


def recheck_cpu(cases, order, sealed, files):
    key_groups = [np.asarray(entry["key_positions"]) for entry in order["steps"][:6]]
    for block_index in range(3):
        if sorted(np.concatenate(key_groups[2 * block_index:2 * block_index + 2])) != list(
            range(32 * block_index, 32 * (block_index + 1))
        ):
            raise ValueError("observed K16 groups cannot be replayed in independent 32-key tiles")
    for item in cases:
        block = item["block"]
        decoded = []
        for suffix, indices in zip(("in0", "in1", "in2", "out"), buffer_offsets(96)):
            role = f"block{block}.attention.{suffix}"
            data = files[role].read_bytes()
            if sha(data) != sealed["boundaries"][role]["sha256"]:
                raise ValueError("tap changed during CPU recheck")
            decoded.append(np.frombuffer(data, np.uint8)[indices])
        next_group = iter(zip(key_groups[::2], key_groups[1::2]))
        tile_start = 0

        def ordered_pv(a, b, acc):
            nonlocal tile_start
            low, high = next(next_group)
            result = split_mma(a, b, acc, groups=(low - tile_start, high - tile_start))
            tile_start += 32
            return result

        stages = attention_stages(*(FP8[buffer] for buffer in decoded[:3]), pv=ordered_pv)
        expected = encode_fp8(stages["pre_quantized"])
        item["cpu_with_measured_pv_order"] = metrics(expected, decoded[3])
        for point in item["points"]:
            at = point["head"], point["token"], point["channel"]
            point["cpu_with_measured_pv_order"] = {
                "pv_half": float(stages["pv"][at]), "pre_quantization_half": float(stages["pre_quantized"][at]),
                "output_code": int(expected[at]),
            }


def run(capture, ptx, ptxas, work):
    if sha(ptxas.read_bytes()) != PTXAS_SHA:
        raise ValueError("not the pinned ptxas 13.3 tool")
    graph = json.loads(pinned_bytes(GRAPH, GRAPH_SHA))
    sealed, files = seal_capture(capture, kernel_inventory(), graph)
    if sealed["kind"] != PORT or sealed["frame"] != 0:
        raise ValueError("expected a real Ampere frame-zero attention capture")
    transformed = instrument(ptx.read_bytes())
    folder = capture_path(work)
    folder.mkdir(parents=True)
    debug_ptx = folder / "module06.attention-stages.ptx"
    cubin_path = folder / "module06.attention-stages.cubin"
    debug_ptx.write_bytes(transformed)
    compile_result = subprocess.run([str(ptxas.resolve()), "-arch=sm_86", str(debug_ptx), "-o", str(cubin_path)],
                                    capture_output=True, timeout=120)
    (folder / "compile.log").write_bytes(compile_result.stdout + compile_result.stderr)
    if compile_result.returncode:
        raise ValueError(f"instrumented PTX compile failed, see {folder / 'compile.log'}")
    cubin = cubin_path.read_bytes()
    device = Driver(0)
    try:
        copy = device.lib.cuMemcpyHtoD_v2
        copy.argtypes, copy.restype = [C.c_uint64, C.c_void_p, C.c_size_t], C.c_int
        module = device.load(cubin)
        try:
            function = device.function(module, ENTRY)
            observed = [case(device, function, block, sealed, files, folder) for block in (11, 18)]
            cases = [item for item, _ in observed]
            key_order = tagged_order(device, function, files, [keys for _, candidates in observed for keys in candidates], folder)
            recheck_cpu(cases, key_order, sealed, files)
        finally:
            device.call("cuModuleUnload", module)
        if not all(point["marker_changed_byte"] for point in cases[0]["points"]):
            raise ValueError("the marker did not distinguish both reference points")
        return {"schema": "october_nr_attention_stages_v1", "gpu": device.label, "passed": True,
                "source_sha256": sha(Path(__file__).read_bytes()), "capture_seal_sha256": sealed["capture_seal_sha256"],
                "rewritten_ptx_sha256": PTX_SHA, "instrumented_ptx_sha256": sha(transformed),
                "instrumented_cubin_sha256": sha(cubin), "cases": cases, "measured_pv_key_order": key_order}
    finally:
        device.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "ptx", "ptxas", "work"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        report = run(args.capture, args.ptx, args.ptxas, args.work)
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.TimeoutExpired) as error:
        report = {"passed": False, "error": str(error)}
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
