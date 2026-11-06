"""Reproduce the static 310.9.16.0 preset-2 host graph report on stdout.

Run from the repository with `python -B -m tools.inspect_nr_small_host_graph`.
The default paths expect the original DLL and extracted SM120 PTX used by the
October investigation. This offline audit pins their hashes, parses the PE and
weight metadata, and checks the assembly-derived address ranges and topology.
The roles, phase conventions, routing, and launch sites encode manual analysis
of this release. The script does not disassemble arbitrary binaries or run CUDA.
"""

import argparse
from hashlib import sha256
import json
from mmap import ACCESS_READ, mmap
from pathlib import Path
import re
import struct
import sys

from .dllq.index import parse_pe, parse_resources
from .inspect_nr_weights import payload_records


ROOT = Path(__file__).resolve().parents[1]
OCTOBER = Path("_dl/october_nr_20261106")
DLL = OCTOBER / "original/nvngx_dlssnr.dll"
PTX = OCTOBER / "candidate_extract"
HOST = Path("research/validation/october_nr_remix_host.json")
WEIGHTS = Path("research/validation/october_nr_remix_weights.json")
OLD_SCHEDULE = Path("torch/dlss5/data/schedule.json")
IMAGE_BASE = 0x180000000
DLL_SHA = "2bc19bc09a7b89dd44f73e2ac0108fcc586e2154bdd0830f242f4f040e28789f"
SMALL_SHA = "f743e8eb320f42bc9bdc992ee2bf51f7bb7c8688ced4ecf5ce6807ec4eef7962"
PTX_MODULES = {
    "fatbin_01.1.sm_120.ptx": "cf24a2f92c2b398764de29474d66ee996bd54a07be0bdfee1239885f74fbefa4",
    "fatbin_06.1.sm_120.ptx": "3dc57ec3e2a68965129f831f889ddc451ba213cc8e0eb271af90668a25891ac7",
}
MODEL_NAME = "hnet-crazy-cuckoo-small-fp8"
RESOURCE_NAME = "WEIGHTS_CC_SMALL_FP8"
PRE_ENTRY = "cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8"

# Stage boundaries and roles are transcribed from the pinned host factory.
STAGES = (
    ("pre", 0, 0, 0x18005CA59, 0x18005CC26),
    ("encoder_1h", 1, 2, 0x18005CC26, 0x18005CF83),
    ("encoder_2h", 3, 4, 0x18005CFA8, 0x18005D1A8),
    ("encoder_4h", 5, 6, 0x18005D1CF, 0x18005D3C0),
    ("encoder_8h", 7, 8, 0x18005D3E7, 0x18005D5DE),
    ("encoder_split16", 9, 10, 0x18005D603, 0x18005D6D2),
    ("vit_1d", 11, 18, 0x18005D6F1, 0x18005D7C1),
    ("decoder_bridge", 19, 19, 0x18005D7DE, 0x18005D9B4),
    ("decoder_split16", 20, 21, 0x18005D9C3, 0x18005DA87),
    ("decoder_8h", 22, 23, 0x18005DAA8, 0x18005DCE8),
    ("decoder_4h", 24, 25, 0x18005DD09, 0x18005DF48),
    ("decoder_2h", 26, 27, 0x18005DF69, 0x18005E1B8),
    ("decoder_1h", 28, 29, 0x18005E1C9, 0x18005E406),
    ("post", 30, 30, 0x18005E423, 0x18005E52A),
)
SKIP_SITES = {
    "block22": ((0x18005D5DE, 0x18005D600), (0x18005DB24, 0x18005DB87)),
    "block24": ((0x18005D3C0, 0x18005D3E5), (0x18005DD85, 0x18005DDEB)),
    "block26": ((0x18005D1A8, 0x18005D1CD), (0x18005DFE5, 0x18005E044)),
    "block28": ((0x18005CF83, 0x18005CFA8), (0x18005E244, 0x18005E2A3)),
    "block30": ((0x18005E406, 0x18005E423), (0x18005E47E, 0x18005E4D9)),
}
SPLIT_CONSTRUCTORS = (
    (0, "ffwd", 0x180090060, 0x180063B5F),
    (1, "ffwd_projection", 0x1800906A0, 0x180063BAB),
    (2, "qkv_attention", 0x180091430, 0x180063EEF),
    (3, "projection_pool", 0x1800910F0, 0x18006415E),
    (3, "projection_or_outview", 0x180090EC0, 0x180064178),
    (4, "final_head", 0x180090B90, 0x1800642E7),
)
WINDOW_PHASES = ((-1, -1), (1, 1), (1, 0), (0, 1))
SHIFTS_YX = ((0, 0), (-4, -4), (0, -4), (-4, 0))


def read_identity(root, path):
    data = (root / path).read_bytes()
    return {"path": path.as_posix(), "sha256": sha256(data).hexdigest()}, data


def ptx_entry(module, data, name, parameter_bytes):
    declaration = rb"(?m)^\.visible \.entry " + re.escape(name.encode("ascii")) + rb"\(\s*"
    parameter = rb"\.param \.align 8 \.b8 \w+\[" + str(parameter_bytes).encode("ascii") + rb"\]\s*\)"
    if len(re.findall(declaration + parameter, data)) != 1:
        raise ValueError(f"expected one entry {name} with {parameter_bytes} parameter bytes in {module}")
    return {"name": name, "extracted_file": module, "sha256": PTX_MODULES[module]}


def previous(index):
    return {"block": index - 1, "selection": "previous_canonical_output"}


def slot(index, output):
    return {"block": index, "output": output, "selection": "explicit_output_slot"}


def logical_graph(records, depths):
    # These constructors and explicit input slots encode the manual routing audit.
    graph = []

    def append(kind, variant, inputs, components, phase=None):
        index = len(graph)
        stage = next(name for name, first, last, _, _ in STAGES if first <= index <= last)
        block_records = []
        for layer, component in enumerate(components):
            suffix = "layer0.blend_scale" if component == "blend_scale" else f"layer{layer}.layer"
            name = f"block{index}.{suffix}"
            block_records.append({"name": name, "payload_bytes": records[name]["payload_bytes"],
                                  "component": component})
        graph.append({"index": index, "stage": stage, "kind": kind, "variant": variant, "inputs": inputs,
                      "window_phase_xy": None if phase is None else WINDOW_PHASES[phase % 4],
                      "shift_yx": None if phase is None else SHIFTS_YX[phase % 4], "records": block_records})

    append("fused_pre_swin_1h", "downsample", [{"external": "color_history_motion_mask_and_pre_controls"}],
           ["pre_with_swin"])
    for level in range(4):
        index = len(graph)
        kind = f"fused_swin_{2 ** level}h"
        append(kind, "inpview", [previous(index)], ["swin"], 0)
        append(kind, "downsample", [slot(index, 0)], ["swin"], 1)
    split = ["ffwd", "ffwd_projection", "qkv_attention"]
    append("split_swin_16h", "inpview", [previous(9)], split + ["projection"], 0)
    append("split_swin_16h", "projection_pool_and_final_head", [slot(9, 0)],
           split + ["projection_pool", "final_head"], 1)
    vit = ["ffn_expand", "ffn_contract", "qkv", "attention", "projection"]
    append("vit_1d", "repack_input", [previous(11)], vit)
    for index in range(12, 18):
        append("vit_1d", "middle", [slot(index - 1, 1)], vit)
    append("vit_1d", "repack_output", [slot(17, 1)], vit)
    append("decoder_input_upsample", "1024_to_512", [slot(18, 0), slot(10, 1)], ["decoder_input_upsample"])
    append("split_swin_16h", "standard", [slot(19, 1)], split + ["projection"], depths[4])
    append("split_swin_16h", "outview", [slot(20, 0)], split + ["projection_outview"], depths[4] + 1)
    for level in reversed(range(4)):
        index = len(graph)
        kind = f"fused_swin_{2 ** level}h"
        append(kind, "upsample_skip", [slot(index - 1, 0), slot(2 + 2 * level, 1)], ["swin"], depths[level])
        append(kind, "outview", [slot(index, 0)], ["swin"], depths[level] + 1)
    append("fused_post_swin_1h", "image_output", [slot(29, 0), slot(0, 1)], ["post_with_swin", "blend_scale"])
    names = [record["name"] for block in graph for record in block["records"]]
    if len(graph) != 31 or len(names) != 77 or len(names) != len(set(names)) or set(names) != records.keys():
        raise ValueError("logical graph does not cover the 31 blocks and 77 serialized records exactly")
    if any(edge["block"] >= block["index"] for block in graph for edge in block["inputs"] if "block" in edge):
        raise ValueError("the graph contains a non-causal edge")
    return graph


def inspect(root=ROOT):
    host, _ = read_identity(root, HOST)
    weight_report, weight_data = read_identity(root, WEIGHTS)
    old_schedule, _ = read_identity(root, OLD_SCHEDULE)
    prior_weights = json.loads(weight_data)["candidate"]
    if prior_weights["sha256"] != DLL_SHA:
        raise ValueError("prior weight report targets a different DLL")
    prior_resource = next(row for row in prior_weights["resources"] if row["name"] == RESOURCE_NAME)
    modules = {}
    for name, expected in PTX_MODULES.items():
        identity, modules[name] = read_identity(root, PTX / name)
        if identity["sha256"] != expected:
            raise ValueError(f"unexpected SHA-256 for {PTX / name}")
    with (root / DLL).open("rb") as stream, mmap(stream.fileno(), 0, access=ACCESS_READ) as dll:
        return inspect_dll(dll, host, weight_report, prior_resource, old_schedule, modules)


def inspect_dll(dll, host, weight_report, prior_resource, old_schedule, modules):
    if sha256(dll).hexdigest() != DLL_SHA:
        raise ValueError(f"unexpected SHA-256 for {DLL}")
    sections, resource_rva = parse_pe(dll)

    def at_va(start, size):
        rva = start - IMAGE_BASE
        for section in sections:
            delta = rva - section["vaddr"]
            if delta >= 0 and delta + size <= section["rsize"]:
                offset = section["raddr"] + delta
                return dll[offset:offset + size]
        raise ValueError(f"VA range is outside the raw PE sections at {start:#x}")

    def code(start, end):
        size = end - start
        return {"va": hex(start), "bytes": size, "sha256": sha256(at_va(start, size)).hexdigest()}

    def depths(start, expected):
        values = struct.unpack("<5I", at_va(start, 20))
        if values != expected:
            raise ValueError(f"unexpected Swin stage depths at {start:#x}")
        return {**code(start, start + 20), "values": values}

    descriptor_va = 0x1800DC0C0
    descriptor = at_va(descriptor_va, 0x290)
    if struct.unpack_from("<I", descriptor, 8)[0] != 2:
        raise ValueError("unexpected preset identifier")
    for field, expected in ((0, MODEL_NAME), (16, RESOURCE_NAME), (24, "CC_SMALL_FP8")):
        va = struct.unpack_from("<Q", descriptor, field)[0]
        if at_va(va, len(expected) + 1) != expected.encode("ascii") + b"\0":
            raise ValueError(f"unexpected preset descriptor string at {va:#x}")
    large = depths(0x1800DE990, (4, 4, 6, 8, 8))
    small = depths(0x1800DE9A8, (2, 2, 2, 2, 2))
    matching = [row for row in parse_resources(dll, sections, resource_rva)
                if (row["rtype"], row["rname"], row["rlang"]) == ("10", RESOURCE_NAME, "1033")]
    if len(matching) != 1:
        raise ValueError("expected one preset-2 RCDATA resource")
    resource = matching[0]
    start, size = resource["off"], resource["size"]
    if start is None or not 0 <= start < start + size <= len(dll):
        raise ValueError("invalid preset-2 RCDATA extent")
    with memoryview(dll)[start:start + size] as blob:
        if sha256(blob).hexdigest() != SMALL_SHA:
            raise ValueError("unexpected SHA-256 for the preset-2 RCDATA resource")
        records = payload_records(blob)
    graph = logical_graph(records, small["values"])
    prior = {f"block{block['index']}.{record['suffix']}": record["payload_bytes"]
             for block in prior_resource["blocks"] for record in block["records"]}
    sizes = {name: record["payload_bytes"] for name, record in records.items()}
    if sizes != prior or sum(sizes.values()) != prior_resource["payload_bytes"]:
        raise ValueError("parsed records differ from the prior weight inventory")

    pre_module = "fatbin_01.1.sm_120.ptx"
    repack_module = "fatbin_06.1.sm_120.ptx"
    pre = ptx_entry(pre_module, modules[pre_module], PRE_ENTRY, 272)
    repack_in = ptx_entry(repack_module, modules[repack_module], "cc_vit_1d_repack_2d_to_1d_fp8", 24)
    repack_out = ptx_entry(repack_module, modules[repack_module], "cc_vit_1d_repack_1d_to_2d_fp8", 24)
    return {
        "schema": "october_nr_small_host_graph_v1",
        "scope": "Static reconstruction of the 310.9.16.0 preset-2 Crazy Cuckoo block graph and host launch binding. No DLL execution, runtime trace, GPU measurement or cross-version numerical equivalence.",
        "source": {
            "dll": {"version": "310.9.16.0", "path": DLL.as_posix(), "bytes": len(dll), "sha256": DLL_SHA,
                    "image_base_va": hex(IMAGE_BASE)},
            "previous_release_host_report": host,
            "previous_release_weights_report": weight_report,
            "resource": {"name": RESOURCE_NAME, "language": resource["rlang"], "file_offset": start, "bytes": size,
                         "sha256": SMALL_SHA, "serialized_records": len(records), "serialized_blocks": len(graph),
                         "record_payload_bytes": sum(sizes.values())},
            "prior_310_8_schedule_used_only_to_crosscheck_shift_convention": old_schedule,
        },
        "preset_selection": {
            "id": 2, "descriptor_name": MODEL_NAME, "model": "CC_SMALL_FP8",
            "descriptor": code(descriptor_va, descriptor_va + 0x290),
            "descriptor_name_matches_factory_config": True,
            "registry_construction": code(0x18006031C, 0x1800603AF),
            "registry_call": code(0x180021150, 0x18002115C),
            "selected_name_lookup": code(0x18002184A, 0x180021937),
            "factory_start": "0x18005c5a0", "large_swin_depths": large, "small_swin_depths": small,
            "swins_depth_order": ["1h", "2h", "4h", "8h", "16h"],
            "fixed_bottleneck_depth": 8, "factory_selects_vit_1d": True,
            "evidence": "The second registry construction passes the short depths and zero as the second argument to the shared factory, then assigns the same model name returned by preset selection. The create path matches the returned descriptor name against the registry. The fourth fused stage has 256 channels and the split stage uses 512. ViT loop bound 8 is independent of the five stage depths.",
        },
        "stages": [{"stage": stage, "first_block": first, "last_block": last, "code": code(a, b)}
                   for stage, first, last, a, b in STAGES],
        "blocks": graph,
        "routing_evidence": {
            "window_phase_encoder": code(0x180059C80, 0x180059CC7),
            "window_phase_decoder": code(0x1800568F0, 0x180056938),
            "phase_semantics": "Pairs are raw signed x/y constructor flags. Encoder phase is i mod 4, decoder phase is (stage depth + i) mod 4. The corresponding pixel shift y/x convention is cross-checked against the earlier large graph.",
            "range_chain_helper": code(0x180056940, 0x180056B8E),
            "stage_range_rewrite_calls": code(0x18005E714, 0x18005E83A),
            "default_previous_and_explicit_lookup": code(0x18005F904, 0x18005F9EF),
            "record_prefix_builder": code(0x1800599F0, 0x180059C77),
            "bridge_inputs": code(0x18005D7C1, 0x18005D890),
            "bridge_to_first_decoder_split_slot": code(0x18005E83A, 0x18005E8C6),
            "upsample_skip_sources": {name: [code(*site) for site in sites] for name, sites in SKIP_SITES.items()},
            "split_layer_constructors": [
                {"layer": layer, "type": kind, "constructor_va": hex(constructor), "factory_callsite": hex(callsite)}
                for layer, kind, constructor, callsite in SPLIT_CONSTRUCTORS
            ],
            "record_correlation": "Factory composes each block prefix from its zero-based block index. Class layer order and serialized suffixes agree for all 31 blocks and all 77 records. The separate block30 layer0.blend_scale record is used by the fused post operation.",
        },
        "boundary_repack_launches": [
            {"block": 11, "when": "before FFN expand", "parameter_bytes": 24, "entry": repack_in,
             "first_flag_call": code(0x180066857, 0x180066867),
             "constructor_name_selection": code(0x18009AEC8, 0x18009AEF6),
             "kernel_registration": code(0x18009CB06, 0x18009CB54),
             "launch_path": code(0x18009BB51, 0x18009BCB7)},
            {"block": 18, "when": "after projection", "parameter_bytes": 24, "entry": repack_out,
             "last_flag_call": code(0x1800670B9, 0x1800670C9),
             "constructor_name_selection": code(0x18009B205, 0x18009B233),
             "kernel_registration": code(0x18009CCA6, 0x18009CCF4),
             "launch_path": code(0x18009C02F, 0x18009C0C1)},
        ],
        "mask_blend_binding": {
            "key": "DLSSNR.MaskBlendMode", "pre_entry": pre, "parameter_bytes": 272,
            "mode_parameter_byte_offset": 40, "host_parameter_object_offset": 168, "host_eval_struct_offset": 244,
            "host_state_write": code(0x180083EE0, 0x180083EF7),
            "pre_prologue": code(0x1800834B0, 0x1800834E6),
            "pre_state_pointer_load": code(0x180083538, 0x18008353F),
            "pre_state_read_and_argument_store": code(0x18008360C, 0x180083616),
            "pre_dispatch": code(0x1800837A5, 0x180083800),
            "stack_accounting": {"push_bytes": 40, "local_allocation_bytes": 416, "rbp_below_entry_rsp_bytes": 200,
                                 "rbp_above_current_rsp_bytes": 256, "mode_store_relative_to_rbp": -104,
                                 "arguments_relative_to_rsp": 112, "mode_relative_to_arguments": 40},
            "ptx_read_lines": [14532, 14534],
            "verified": "Previously traced Evaluate getter reaches the pre layer object. Its +0xa8 value is copied into byte +40 of the 272-byte argument block passed to dispatch. The same field is read by the original SM120 pre PTX. This closes the static binder.",
        },
        "limits": [
            "Logical block topology and deterministic class/routing flags are static. Grid sizes, synchronization variant choice, GPU ordering, and D3D12/Vulkan launch behavior still require captured execution.",
            "The 31 logical blocks and 77 serialized records are different units from emitted CUDA launches. The two ViT repacks are unweighted, and the post blend scale shares its post operation.",
            "No numeric tensor decode, cross-device result, throughput, or speed claim is established by this report.",
        ],
    }


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    try:
        report = inspect()
    except (OSError, ValueError, struct.error, KeyError, StopIteration) as error:
        print(f"Preset-2 host graph inspection failed: {error}", file=sys.stderr)
        return 1
    sys.stdout.buffer.write((json.dumps(report, indent=2) + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
