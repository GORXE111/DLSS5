"""Bind the October preset-2 weights to a provisional legacy-tensor reference.

`python -B -m tools.october_nr_small_reference` audits binding without Torch.
Add `--numpy-constructors` to check all legacy constructors without GPU execution.
`build_approximate_reference()` is opt-in and has no role in the 310.8.0 defaults.
"""

import argparse
from collections import Counter
from hashlib import sha256
import json
from mmap import ACCESS_READ, mmap
from pathlib import Path
import struct
import sys

from .dllq.index import parse_pe, parse_resources
from .inspect_nr_small_layout import CANDIDATE, EXPECTED, ROOT, SMALL, check_constructors
from .inspect_nr_weights import payload_records


REPORT_DIR = ROOT / "research/validation"
GRAPH = REPORT_DIR / "october_nr_small_host_graph.json"
LAYOUT = REPORT_DIR / "october_nr_small_layout.json"
SWIN_VARIANTS = {"inpview": "inpview", "downsample": "ds", "upsample_skip": "up", "outview": "outview"}
SPLIT_TAILS = {"inpview": None, "projection_pool_and_final_head": "pool", "standard": None, "outview": "outview"}
SKIP_ROUTES = {19: (10, 1), 22: (8, 1), 24: (6, 1), 26: (4, 1), 28: (2, 1), 30: (0, 1)}
VIT_LAYERS = ("ffn_expand", "ffn_contract", "qkv", "attention", "projection")
SPLIT_LAYERS = ("ffwd", "ffwd_projection", "qkv_attention")
OPERATOR_FILES = ("torch/dlss5/net.py", "torch/dlss5/blocks.py", "torch/dlss5/layout.py",
                  "torch/dlss5/data/adapter_map.json", "torch/dlss5/data/control_map.json",
                  "torch/dlss5/data/maps_block1.json")


def json_source(path):
    raw = path.read_bytes()
    return json.loads(raw), sha256(raw).hexdigest()


def ordered_records(node, roles):
    block = node["index"]
    rows = node["records"]
    if tuple(row["component"] for row in rows) != roles:
        raise ValueError(f"unexpected record roles for block {block}")
    names = [row["name"] for row in rows]
    expected = [f"block{block}.layer{i}.layer" for i in range(len(roles))]
    if roles[-1] == "blend_scale":
        expected[-1] = f"block{block}.layer0.blend_scale"
    if names != expected:
        raise ValueError(f"unexpected record names or order for block {block}")
    return names


def translate_node(node):
    kind, variant, index = node["kind"], node["variant"], node["index"]
    if kind.startswith("fused_swin_"):
        step = {"kind": "swin", "level": kind.removeprefix("fused_swin_"), "variant": SWIN_VARIANTS[variant],
                "record": ordered_records(node, ("swin",))[0]}
    elif kind == "split_swin_16h":
        tail = SPLIT_TAILS[variant]
        last = {"pool": "projection_pool", "outview": "projection_outview"}.get(tail, "projection")
        names = ordered_records(node, SPLIT_LAYERS + (last,) + (("final_head",) if tail == "pool" else ()))
        step = {"kind": "split16", "records": names[:4], "inpview": variant == "inpview", "tail": tail}
        if tail == "pool":
            step["head_record"] = names[4]
    elif kind == "vit_1d":
        expected_variant = {11: "repack_input", 18: "repack_output"}.get(index, "middle")
        if variant != expected_variant:
            raise ValueError(f"unexpected 1D repack boundary for block {index}")
        step = {"kind": "vit", "records": ordered_records(node, VIT_LAYERS)}
    elif kind == "fused_pre_swin_1h" and variant == "downsample":
        step = {"kind": "pre", "record": ordered_records(node, ("pre_with_swin",))[0]}
    elif kind == "decoder_input_upsample" and variant == "1024_to_512":
        step = {"kind": "block39", "record": ordered_records(node, ("decoder_input_upsample",))[0]}
    elif kind == "fused_post_swin_1h" and variant == "image_output":
        record, blend = ordered_records(node, ("post_with_swin", "blend_scale"))
        step = {"kind": "post", "record": record, "blend_record": blend}
    else:
        raise ValueError(f"unexpected block {index}: {kind}/{variant}")
    shift = node["shift_yx"]
    if step["kind"] in ("swin", "split16"):
        phase = node["window_phase_xy"]
        mapped = [-4 if value == 1 else 0 for value in reversed(phase)]
        if len(phase) != 2 or any(value not in (-1, 0, 1) for value in phase) or shift != mapped:
            raise ValueError(f"inconsistent window phase for block {index}")
        step["shift"] = shift
    elif shift is not None:
        raise ValueError(f"unexpected window phase for block {index}")
    return step


def validate_routes(node, step):
    index = node["index"]
    if index == 0:
        expected = [{"external": "color_history_motion_mask_and_pre_controls"}]
    elif (step.get("variant") == "inpview" or step.get("inpview") or index == 11):
        expected = [{"block": index - 1, "selection": "previous_canonical_output"}]
    else:
        previous_slot = 1 if 12 <= index <= 18 or index == 20 else 0
        expected = [{"block": index - 1, "output": previous_slot, "selection": "explicit_output_slot"}]
    if index in SKIP_ROUTES:
        skip, slot = SKIP_ROUTES[index]
        expected.append({"block": skip, "output": slot, "selection": "explicit_output_slot"})
    if node["inputs"] != expected:
        raise ValueError(f"host input slots differ from reusable tensor route at block {index}")


def read_small_weights(path, resource_sha):
    with path.open("rb") as stream, mmap(stream.fileno(), 0, access=ACCESS_READ) as data:
        if sha256(data).hexdigest() != EXPECTED[1]:
            raise ValueError("candidate is not the pinned unmodified 310.9.16.0 DLL")
        sections, rva = parse_pe(data)
        resource, = [r for r in parse_resources(data, sections, rva)
                     if (r["rtype"], r["rname"], r["rlang"]) == ("10", SMALL, "1033")]
        start, size = resource["off"], resource["size"]
        with memoryview(data)[start:start + size] as blob:
            if sha256(blob).hexdigest() != resource_sha:
                raise ValueError("candidate weight resource differs from host graph and layout evidence")
            records = payload_records(blob)
            weights = {name: bytes(blob[row["payload_offset"]:row["payload_offset"] + row["payload_bytes"]])
                       for name, row in records.items()}
    return weights, records, start, size


def bind(path=CANDIDATE):
    graph, graph_sha = json_source(GRAPH)
    layout, layout_sha = json_source(LAYOUT)
    host_source = graph["source"]
    resource, decoded = host_source["resource"], layout["resource"]
    if (graph["schema"], layout["schema"]) != ("october_nr_small_host_graph_v1", "dlssnr_cc_small_fp8_storage_v1"):
        raise ValueError("unexpected October evidence schemas")
    if (host_source["dll"]["sha256"] != EXPECTED[1] or layout["candidate"]["sha256"] != EXPECTED[1] or
            graph["preset_selection"]["id"] != 2 or not layout["passed"] or resource["name"] != SMALL):
        raise ValueError("host graph or layout is not for the pinned October preset 2")
    nodes = graph["blocks"]
    if [node["index"] for node in nodes] != list(range(31)):
        raise ValueError("expected the 31 ordered October model blocks")
    schedule = [translate_node(node) for node in nodes]
    for node, step, decoder in zip(nodes, schedule, layout["blocks"], strict=True):
        validate_routes(node, step)
        if (node["index"], step["kind"], step.get("level"), len(node["records"])) != (
            decoder["index"], decoder["layout_family"], decoder.get("level"), decoder["record_count"]
        ) or step["kind"] == "swin" and step["variant"] not in decoder["baseline_variants"]:
            raise ValueError(f"legacy decoder family differs from host graph at block {node['index']}")
    families = Counter(step["kind"] for step in schedule)
    if families != {"pre": 1, "swin": 16, "split16": 4, "vit": 8, "block39": 1, "post": 1}:
        raise ValueError("unexpected October layer families")
    weights, records, start, size = read_small_weights(Path(path), resource["sha256"])
    if (start, size) != (resource["file_offset"], resource["bytes"]) or (
        start, size, resource["sha256"]
    ) != (decoded["file_offset"], decoded["size"], decoded["sha256"]):
        raise ValueError("field layout describes a different PE resource")
    host_rows = [row for node in nodes for row in node["records"]]
    decoded_rows = {row["name"]: row for row in layout["records"]}
    if (len(host_rows), len(decoded_rows), len(layout["records"]), len(records)) != (77, 77, 77, 77) or (
        {r["name"] for r in host_rows} != records.keys() or decoded_rows.keys() != records.keys()
    ):
        raise ValueError("host or field layout record inventory differs from the original PE resource")
    for row in host_rows:
        name = row["name"]
        if row["payload_bytes"] != records[name]["payload_bytes"] or any(
            decoded_rows[name][key] != value for key, value in records[name].items()
        ):
            raise ValueError(f"host or field layout does not match serialized payload {name}")
    source_hashes = {name: sha256((ROOT / name).read_bytes()).hexdigest() for name in OPERATOR_FILES}
    if any(layout["source_sha256"][name] != value for name, value in source_hashes.items()):
        raise ValueError("legacy constructor sources have changed since the field layout audit")
    source_hashes["tools/october_nr_small_reference.py"] = sha256(Path(__file__).read_bytes()).hexdigest()
    summary = {"passed": True, "blocks": len(nodes), "records": len(records), "families": dict(families),
               "payload_bytes": sum(len(payload) for payload in weights.values()), "dll_sha256": EXPECTED[1],
               "weight_resource_sha256": resource["sha256"], "host_graph_sha256": graph_sha,
               "field_layout_sha256": layout_sha, "source_sha256": source_hashes, "schedule": schedule,
               "scope": "Static slot binding and shape-compatible legacy tensor schedule; October forward fidelity unmeasured"}
    return summary, weights, layout, start


def build_approximate_reference(path=CANDIDATE, *, device="cpu", precise=True):
    """Construct the experimental legacy-operator approximation from the original preset-2 bytes."""
    summary, weights, _, _ = bind(path)
    sys.path.insert(0, str(ROOT / "torch"))
    try:
        from dlss5.net import DLSS5
    finally:
        sys.path.pop(0)
    return DLSS5(device=device, records=weights, schedule=summary["schedule"], precise=precise, half=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", type=Path, default=CANDIDATE)
    parser.add_argument("--numpy-constructors", action="store_true", help="check the translated constructor plan on CPU")
    args = parser.parse_args()
    try:
        if args.numpy_constructors and args.dll.resolve() != CANDIDATE.resolve():
            raise ValueError("--numpy-constructors uses the default candidate DLL path")
        summary, _, layout, start = bind(args.dll)
        if args.numpy_constructors:
            # Existing audit reads the pinned original path and walks these exact translated steps.
            steps = {f"block{index}.layer0.layer": step for index, step in enumerate(summary["schedule"])}
            rows = [{**row, "baseline_layout_matches": [row["name"]]} for row in layout["records"]]
            result = check_constructors(rows, start, steps)
            summary["legacy_constructor_check"] = {key: value for key, value in result.items()
                                                   if key != "expanded_bias_head_checks"}
    except (OSError, ValueError, TypeError, KeyError, ImportError, struct.error) as error:
        summary = {"passed": False, "error": str(error)}
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
