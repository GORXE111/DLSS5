"""Exercise the two October pre-kernel mask slices with synthetic pre-sampled inputs.

The source PTX is hash-pinned and read locally. The probe keeps the 272-byte
parameter ABI, scalar mask branches, FP16 conversion and three style selects.
Texture sampling, dispatch and tensor math remain outside this experiment.
"""

import argparse
from collections import Counter
import ctypes as C
from itertools import product
import json
from pathlib import Path
import struct
import sys

import numpy as np

from .probe_october_fence import Driver, sha


MODULE_SHA = "cf24a2f92c2b398764de29474d66ee996bd54a07be0bdfee1239885f74fbefa4"
FIELDS = ("local_tone", "structure_gate", "skin", "structure", "style_lane_0", "style_lane_1", "style_lane_2")
VARIANTS = (
    {
        "entry": "cc_tinlayout_fused_pre_block_swin_1h_32_1_fp8", "lines": (19, 627),
        "regs": (24, 27), "param_reg": "%rd8",
        "setup": ((64, 65), (94, 95), (98, 98), (103, 105), (108, 108), (111, 118), (137, 138), (147, 147)),
        "sample": (580, 584), "sample_regs": ("%r5038", "%r5039"),
        "branch": (588, 627), "merge": (149, 170),
        "style": ((173, 175), (199, 199), (204, 204)),
        "outputs": ("%rs493", "%rs492", "%rs491", "%rs494", "%rs25", "%rs24", "%rs26"),
    },
    {
        "entry": "cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8", "lines": (14458, 15066),
        "regs": (14463, 14466), "param_reg": "%rd19",
        "setup": ((14502, 14503), (14532, 14533), (14536, 14536), (14541, 14543), (14546, 14546),
                  (14549, 14556), (14575, 14576), (14586, 14586)),
        "sample": (15019, 15023), "sample_regs": ("%r5565", "%r5566"),
        "branch": (15027, 15066), "merge": (14588, 14609),
        "style": ((14612, 14614), (14638, 14638), (14643, 14643)),
        "outputs": ("%rs502", "%rs501", "%rs500", "%rs503", "%rs33", "%rs32", "%rs34"),
    },
)


def slice_ptx(lines, variant):
    def excerpt(start, end):
        return "\n".join(lines[start - 1:end])

    start, _ = variant["lines"]
    if not lines[start - 1].startswith(f".visible .entry {variant['entry']}(") or "[272]" not in lines[start]:
        raise ValueError("unexpected pre entry or ABI")
    tone, structure = variant["sample_regs"]
    stores = "\n".join(f"st.global.b16 [%probe_out+{i * 2}], {reg};"
                       for i, reg in enumerate(variant["outputs"]))
    return ("\n".join((
        ".version 8.8\n.target sm_86\n.address_size 64",
        ".visible .entry mask_slice(.param .align 8 .b8 probe_args[272],",
        "    .param .b32 probe_sample_g, .param .b32 probe_sample_b, .param .b64 probe_result) {",
        excerpt(*variant["regs"]),
        ".reg .b64 %probe_out;",
        f"mov.b64 {variant['param_reg']}, probe_args;",
        *(excerpt(*span) for span in variant["setup"]),
        excerpt(*variant["sample"]),
        f"ld.param.b32 {tone}, [probe_sample_g];\nld.param.b32 {structure}, [probe_sample_b];",
        excerpt(*variant["branch"]),
        excerpt(*variant["merge"]),
        *(excerpt(*span) for span in variant["style"]),
        "ld.param.b64 %probe_out, [probe_result];",
        stores, "ret;\n}",
    )) + "\n").encode("ascii")


def fixtures():
    # The caller normalizes modes and applies its mode-0 mask policy. Probe
    # low-level parameters independently, then include a few invalid raw modes.
    for mode, present, automatic, (skin, structure), style in product(
        range(4), (False, True), (False, True), ((-1, -1), (-1, 0.75), (0.25, -1), (0.25, 0.75)), (0, 15, 16)
    ):
        yield (mode, present, automatic, style, 0.75, 0.5, skin, structure, 0.25, 0.75)
    rng = np.random.default_rng(310916)
    for mode in range(4):
        for tone, structure, skin_auto, structure_auto, sample_g, sample_b in rng.uniform(0, 1, (64, 6)):
            yield (mode, True, True, 16, tone, structure, skin_auto, structure_auto, sample_g, sample_b)
    for mode in (4, 0xFFFFFFFF):
        yield (mode, True, True, 0, 0.75, 0.5, 0.25, 0.75, 0.25, 0.75)


def oracle(case):
    mode, present, automatic, style, tone, structure, skin_auto, structure_auto, g, b = case
    tone, structure, skin_auto, structure_auto, g, b = map(np.float32, (tone, structure, skin_auto, structure_auto, g, b))
    if not present:
        g = b = np.float32(1)
    local_tone, explicit = g * tone, b * structure
    if not automatic:
        gate, skin, struct = explicit, 0, 0
    elif max(skin_auto, structure_auto) < 0:
        gate, skin, struct = explicit, -1, -1
    else:
        gate = 1
        skin = structure if skin_auto < 0 else skin_auto
        struct = structure if structure_auto < 0 else structure_auto
        if present and mode:
            if mode == 2:
                skin, struct = min(explicit, skin), min(explicit, struct)
            elif mode == 3:
                skin, struct = max(explicit, skin), max(explicit, struct)
            else:
                skin, struct = (explicit + skin) * np.float32(0.5), (explicit + struct) * np.float32(0.5)
    lanes = (skin, struct, 0) if style <= 15 else (0, skin, struct)
    return (local_tone, gate, skin, struct, *lanes)


def launch(device, fn, output, cases):
    word_count = len(cases) * len(FIELDS)
    device.call("cuMemsetD32_v2", output, 0xFFFFFFFF, (word_count + 1) // 2)
    for i, (mode, present, automatic, style, tone, structure, skin, struct_auto, g, b) in enumerate(cases):
        parameters = C.create_string_buffer(272)
        struct.pack_into("<QI", parameters, 32, present, mode)
        struct.pack_into("<ff", parameters, 176, tone, structure)
        struct.pack_into("<f", parameters, 188, skin)
        struct.pack_into("<fiI", parameters, 192, struct_auto, style, automatic)
        sample_g, sample_b = C.c_float(g), C.c_float(b)
        target = C.c_uint64(output.value + i * len(FIELDS) * 2)
        inputs = (C.c_void_p * 4)(C.addressof(parameters), C.addressof(sample_g), C.addressof(sample_b), C.addressof(target))
        device.call("cuLaunchKernel", fn, 1, 1, 1, 1, 1, 1, 0, None, inputs, None)
    device.call("cuCtxSynchronize")
    data = C.create_string_buffer(word_count * 2)
    device.call("cuMemcpyDtoH_v2", data, output, len(data))
    return np.frombuffer(data.raw, "<u2").reshape(len(cases), len(FIELDS))


def run(path, ordinal):
    data = path.read_bytes()
    if sha(data) != MODULE_SHA:
        raise ValueError("not the fixed October module 01 PTX")
    lines = data.decode("ascii").splitlines()
    cases = list(fixtures())
    expected = np.asarray([oracle(case) for case in cases], dtype="<f2").view("<u2")
    wrong_mask_channel = np.asarray([oracle((*case[:-1], case[-2])) for case in cases], dtype="<f2").view("<u2")
    wrong_style_threshold = np.asarray([oracle((*case[:3], case[3] + 1, *case[4:]))
                                       for case in cases], dtype="<f2").view("<u2")
    results = []
    device = Driver(ordinal)
    try:
        output = C.c_uint64()
        device.call("cuMemAlloc_v2", C.byref(output), (expected.nbytes + 3) & ~3)
        try:
            for variant in VARIANTS:
                ptx = slice_ptx(lines, variant)
                module = device.load(ptx)
                try:
                    actual = launch(device, device.function(module, "mask_slice"), output, cases)
                finally:
                    device.call("cuModuleUnload", module)
                differing = np.argwhere(actual != expected)
                summary = {}
                for mode in range(4):
                    case = (mode, True, mode != 0, 0, 0.75, 0.5, 0.25, 0.75, 0.25, 0.75)
                    summary[str(mode)] = dict(zip(FIELDS, map(float, actual[cases.index(case)].view("<f2"))))
                wrong_channel_count = int(np.count_nonzero(actual != wrong_mask_channel))
                wrong_threshold_count = int(np.count_nonzero(actual != wrong_style_threshold))
                results.append({
                    "entry": variant["entry"], "source_line_range": variant["lines"], "generated_ptx_sha256": sha(ptx),
                    "fp16_values_compared": int(actual.size), "differing_fp16_values": len(differing),
                    "first_difference_indices": differing[:8].tolist(), "output_sha256": sha(actual.tobytes()),
                    "mismatches_if_using_G_for_structure": wrong_channel_count,
                    "mismatches_if_using_greater_or_equal_15_for_style": wrong_threshold_count,
                    "example": summary,
                })
        finally:
            device.call("cuMemFree_v2", output)
        return {
            "schema": "dlssnr_october_pre_mask_slices_v1",
            "scope": "Standalone slices from original PTX with synthetic post-texture values; no vendor DLL execution",
            "gpu": device.label, "compute_capability": device.capability, "cuda_driver_api_version": device.version,
            "original_ptx_sha256": MODULE_SHA, "probe_source_sha256": sha(Path(__file__).read_bytes()),
            "driver_helper_source_sha256": sha(Path(__file__).with_name("probe_october_fence.py").read_bytes()),
            "parameter_bytes": 272, "mask_blend_mode_offset": 40, "style_lane_selector_offset": 196,
            "cases_per_entry": len(cases), "cases_per_mode": dict(Counter(str(case[0]) for case in cases)),
            "expected_output_sha256": sha(expected.tobytes()), "output_fields": FIELDS,
            "example_inputs": {"mask_g": 0.25, "mask_b": 0.75, "base_tone": 0.75, "base_structure": 0.5,
                               "auto_skin": 0.25, "auto_structure": 0.75,
                               "automatic_enabled_for_mode_zero": False},
            "entries": results,
            "passed": all(row["differing_fp16_values"] == 0 and row["mismatches_if_using_G_for_structure"] > 0
                          and row["mismatches_if_using_greater_or_equal_15_for_style"] > 0 for row in results),
        }
    finally:
        device.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--october-ptx", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.device < 0:
        parser.error("device must be nonnegative")
    try:
        report = run(args.october_ptx, args.device)
    except (OSError, ValueError, RuntimeError) as error:
        print(json.dumps({"passed": False, "error": str(error)}, indent=2))
        return 1
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
