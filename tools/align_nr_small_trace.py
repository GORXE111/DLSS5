"""Check an existing D3D12 nr-trace.tsv against the 310.9.16.0 preset-2 graph.

Run `python -B -m tools.align_nr_small_trace PATH [--frame N]`. Captures with
DLSS5_TRACE=-2 carry frame IDs. A single-frame capture has no embedded frame ID.
The existing DLSS5_PROF=1 hook records host submissions and splits CUDA chains.
This offline checker does not capture, prove producer identity, read tensor
contents, or measure native timing. Preserve trace, DLL and harness identities,
command, logs and output together during any subsequent authorized capture.
"""

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import struct
import sys

from .exec_map import record_layout
from .inspect_nr_small_host_graph import DLL_SHA, PRE_ENTRY, PTX, ROOT


GRAPH = ROOT / "research/validation/october_nr_small_host_graph.json"
GRAPH_SHA = "01e3c597d638592da877d5c66b9bce019353bf79e453ff4785459b9050f3733f"
KERNELS = ROOT / "research/validation/october_nr_remix_kernels.json"
KERNELS_SHA = "bbf13b4916193f94427021e016e08675fbdd6175016c187e812267dc4693e324"
ENTRY = re.compile(rb"(?m)^\.visible \.entry ((?:cc_|cg2r_)\w+)\(\s*\.param \.align \d+ \.b8 \w+\[(\d+)\]")
DECIMAL = re.compile(r"[0-9]+")
HEX = re.compile(r"[0-9a-fA-F]*")
AUXILIARY = {"cc_cb_clear", "cg2r_copy_kernel", "cg2r_post_process_kernel"}
REPACK_IN = "cc_vit_1d_repack_2d_to_1d_fp8"
REPACK_OUT = "cc_vit_1d_repack_1d_to_2d_fp8"
SWIN_VARIANTS = {
    "inpview": ("inpview", "inpview_tilesync"),
    "downsample": ("ds", "ds_wait"),
    "upsample_skip": ("upsample", "upsample_tilesync"),
    "outview": ("outview", "outview_wait"),
}


@dataclass(frozen=True)
class Launch:
    sequence: int
    chain: int
    name: str
    grid: tuple
    block: tuple
    shared_memory: int
    params: bytes

    def pointer(self, offset):
        return struct.unpack_from("<Q", self.params, offset)[0]

    def identity(self):
        return {"sequence": self.sequence, "chain": self.chain, "kernel": self.name,
                "grid": self.grid, "block": self.block, "shared_memory": self.shared_memory,
                "parameter_bytes": len(self.params), "parameter_sha256": sha256(self.params).hexdigest()}


def pinned_bytes(path, expected):
    data = path.read_bytes()
    if sha256(data).hexdigest() != expected:
        raise ValueError(f"SHA-256 mismatch for {path.name}")
    return data


def kernel_inventory():
    audit = json.loads(pinned_bytes(KERNELS, KERNELS_SHA))
    if audit["candidate_dll_sha256"] != DLL_SHA:
        raise ValueError("kernel inventory targets a different DLL")
    entries = {}
    for row in audit["modules"]:
        number = row["module"]
        if number not in (1, 2, 3, 4, 5, 6, 7, 14, 15):
            continue
        source = ROOT / PTX / f"fatbin_{number:02}.1.sm_120.ptx"
        data = pinned_bytes(source, row["candidate"]["ptx_sha256"])
        for name, size in ENTRY.findall(data):
            name = name.decode("ascii")
            if name in entries:
                raise ValueError(f"duplicate PTX entry {name}")
            entries[name] = int(size)
    return entries


def integers(text, separator, count):
    fields = text.split(separator)
    if len(fields) != count or any(not DECIMAL.fullmatch(field) for field in fields):
        raise ValueError(f"expected {count} unsigned integers separated by {separator!r}")
    return tuple(map(int, fields))


def read_trace(data, inventory):
    frames = {}
    previous_frame = None
    chain, expected_index, chain_size = -1, 0, 0
    for seq, line in enumerate(data.decode("ascii").splitlines()):
        cells = line.split("\t")
        if len(cells) == 9 and re.fullmatch(r"f[0-9]+", cells[0]):
            frame = int(cells.pop(0)[1:])
        elif len(cells) == 8:
            frame = None
        else:
            raise ValueError(f"row {seq}: malformed profiler trace")
        if seq and frame != previous_frame and (expected_index != chain_size or frame is None
                                               or previous_frame is None or frame <= previous_frame):
            raise ValueError(f"row {seq}: inconsistent frame boundary")
        previous_frame = frame
        sequence, group, name, grid, block, shared, size, params = cells
        new_chain, index, count = integers(group, "/", 3)
        if integers(sequence, "/", 1)[0] != seq:
            raise ValueError(f"row {seq}: missing or reordered trace sequence")
        if new_chain != chain:
            if new_chain != chain + 1 or index or expected_index != chain_size:
                raise ValueError(f"row {seq}: missing or reordered launch chain")
            chain, expected_index, chain_size = new_chain, 0, count
        if not 0 <= index < count or index != expected_index or count != chain_size:
            raise ValueError(f"row {seq}: incomplete launch chain")
        expected_index += 1
        declared = integers(size, "/", 1)[0]
        if inventory.get(name) != declared or len(params) != declared * 2 or not HEX.fullmatch(params):
            raise ValueError(f"row {seq}: unexpected kernel or parameter ABI for {name} ({declared} B)")
        grid, block = integers(grid, ",", 3), integers(block, ",", 3)
        if min(*grid, *block) <= 0:
            raise ValueError(f"row {seq}: nonpositive launch dimensions")
        frames.setdefault(frame, []).append(Launch(seq, chain, name, grid, block,
                                                  integers(shared, "/", 1)[0], bytes.fromhex(params)))
    if not frames or expected_index != chain_size:
        raise ValueError("empty or truncated profiler trace")
    return frames


def symbols(stem, variants=("",)):
    return tuple(f"{stem}{'_' + variant if variant else ''}_fp8" for variant in variants)


def block_symbols(block):
    kind, variant = block["kind"], block["variant"]
    if kind == "fused_pre_swin_1h":
        return [(PRE_ENTRY,)]
    if kind == "fused_post_swin_1h":
        stem = "cc_tinlayout_fused_post_block_swin_1h_32_fp8"
        return [(stem, *(stem + suffix for suffix in ("_simple_blend", "_simple_blend_full_rect",
                                                      "_control_mask", "_control_mask_full_rect")))]
    if kind.startswith("fused_swin_"):
        heads = int(kind.removeprefix("fused_swin_").removesuffix("h"))
        return [symbols(f"cc_tinlayout_fused_swin_{heads}h_{32 * heads}_{heads}", SWIN_VARIANTS[variant])]
    if kind == "split_swin_16h":
        inpview = "_inpview" if variant == "inpview" else ""
        pool = variant == "projection_pool_and_final_head"
        projection = "proj_pool" if pool else "proj"
        projection_variants = ("outview", "outview_wait") if variant == "outview" else ("", "chained", "tilesync", "wait")
        result = [symbols(f"cc_split_swin_16h_ffwd{inpview}_512", ("", "chained", "tilesync", "wait")),
                  symbols(f"cc_split_swin_16h_ffwd_proj{inpview}_512", ("", "chained", "wait")),
                  symbols("cc_split_swin_16h_qkv_512", ("", "chained")),
                  symbols(f"cc_split_swin_16h_{projection}_512", projection_variants)]
        return result + [symbols("cc_split_swin_16h_final_head_512", ("", "wait"))] if pool else result
    if kind == "vit_1d":
        result = [symbols("cc_vit_1d_" + role, ("", "chained", alternate)) if alternate else
                  symbols("cc_vit_1d_" + role, ("", "chained"))
                  for role, alternate in (("ffn_expand", "publish"), ("ffn_contract", None), ("qkv", None),
                                          ("attention", None), ("projection", "wait"))]
        if variant == "repack_input":
            return [(REPACK_IN,)] + result
        return result + [(REPACK_OUT,)] if variant == "repack_output" else result
    if kind == "decoder_input_upsample":
        return [symbols("cc_dec_input_upsample_1024_512", ("", "tilesync"))]
    raise ValueError(f"unrecognized graph block type {kind}")


def output_slots(block, launches):
    kind, variant = block["kind"], block["variant"]
    if kind == "fused_pre_swin_1h":
        return {0: (launches[0], 256), 1: (launches[0], 224)}
    if kind.startswith("fused_swin_"):
        if variant == "downsample":
            return {0: (launches[0], 64 if kind == "fused_swin_1h" else 72), 1: (launches[0], 8)}
        return {0: (launches[0], 8)}
    if kind == "split_swin_16h":
        if variant == "projection_pool_and_final_head":
            return {0: (launches[-1], 8), 1: (launches[-2], 16)}
        return {0: (launches[-1], 16)}
    if kind == "vit_1d":
        return {0: (launches[-1], 8)} if variant == "repack_output" else {1: (launches[-1], 16)}
    if kind == "decoder_input_upsample":
        return {1: (launches[0], 16)}
    return {}


def check_edges(block, launches, outputs):
    links = []
    for edge in block["inputs"]:
        if "block" not in edge:
            continue
        source = outputs[edge["block"]][edge.get("output", 0)]
        position = 0
        if links:
            if block["kind"].startswith("fused_swin_"):
                position = 80 if block["kind"] == "fused_swin_1h" else 24
            else:
                position = 8
        links.append((*source, launches[0], position, "block_input"))
    if block["variant"] == "repack_input":
        links.append((launches[0], 8, launches[1], 0, "repack_to_ffn_expand"))
    elif block["variant"] == "repack_output":
        links.append((launches[-2], 16, launches[-1], 0, "projection_to_repack"))
    evidence = []
    for producer, out, consumer, inp, role in links:
        value = producer.pointer(out)
        if not value or value != consumer.pointer(inp):
            raise ValueError(f"block {block['index']}: {role} pointer mismatch at seq {producer.sequence}->{consumer.sequence}")
        evidence.append({"role": role, "producer_sequence": producer.sequence, "producer_offset": out,
                         "consumer_sequence": consumer.sequence, "consumer_offset": inp})
    return evidence


def check_shift(block, launches):
    if block["shift_yx"] is None:
        return None
    kind = block["kind"]
    launch = launches[2] if kind == "split_swin_16h" else launches[0]
    offset = 32 if kind in ("fused_swin_1h", "split_swin_16h") else 40
    x, y = struct.unpack_from("<ii", launch.params, offset)
    if [y, x] != block["shift_yx"]:
        raise ValueError(f"block {block['index']}: window shift {(y, x)} differs from {block['shift_yx']}")
    return [y, x]


def expected_records(block):
    names = [(record["name"],) for record in block["records"]]
    if block["variant"] == "repack_input":
        return [()] + names
    if block["variant"] == "repack_output":
        return names + [()]
    return [tuple(record["name"] for record in block["records"])] if block["kind"] == "fused_post_swin_1h" else names


def observe_weights(launch, expected, base, starts):
    words = (word[0] for word in struct.iter_unpack("<Q", launch.params[:len(launch.params) // 8 * 8]))
    observed = sorted({starts[word - base] for word in words if word - base in starts})
    return {"expected": sorted(expected), "observed": observed, "matched": sorted(expected) == observed}


def align_frame(frame, rows, graph, expect_mask_mode=None):
    blocks = graph["blocks"]
    starts, packed_bytes = record_layout([
        {"name": record["name"], "C": record["payload_bytes"]} for block in blocks for record in block["records"]
    ])
    if starts.get(0) != "block0.layer0.layer":
        raise ValueError("unexpected pre record in packed weight layout")
    network = [row for row in rows if row.name not in AUXILIARY]
    auxiliary = [row.identity() for row in rows if row.name in AUXILIARY]
    if not network:
        raise ValueError("no network submissions found")
    if any(network[0].sequence < row.sequence < network[-1].sequence and row.name.startswith("cg2r_") for row in rows):
        raise ValueError("copy or image postprocessing appeared inside the network")
    position, outputs, layers = 0, {}, []
    weight_base = None
    for block in blocks:
        index = block["index"]
        candidates = block_symbols(block)
        launches = network[position:position + len(candidates)]
        if len(launches) != len(candidates):
            raise ValueError(f"block {index}: trace ended before the expected launches")
        for launch, allowed in zip(launches, candidates):
            if launch.name not in allowed:
                raise ValueError(f"block {index}, seq {launch.sequence}: {launch.name} is outside the expected family {allowed}")
        if index == 0:
            weight_base = launches[0].pointer(232)
            mode = struct.unpack_from("<I", launches[0].params, 40)[0]
            if mode > 3 or (expect_mask_mode is not None and mode != expect_mask_mode):
                raise ValueError(f"pre launch observed unexpected MaskBlendMode {mode}")
        bindings = [observe_weights(launch, expected, weight_base, starts)
                    for launch, expected in zip(launches, expected_records(block))]
        layers.append({"block": index, "stage": block["stage"], "kind": block["kind"], "variant": block["variant"],
                       "shift_yx": check_shift(block, launches), "pointer_links": check_edges(block, launches, outputs),
                       "launches": [{**launch.identity(), "weight_pointer_layout": binding}
                                    for launch, binding in zip(launches, bindings)]})
        outputs[index] = output_slots(block, launches)
        position += len(launches)
    if position != len(network):
        raise ValueError(f"unexpected network submissions after block 30 at seq {network[position].sequence}")
    matched = sum(launch["weight_pointer_layout"]["matched"] for block in layers for launch in block["launches"])
    return {"frame": frame, "blocks_aligned": len(layers), "network_submissions": position, "mask_blend_mode": mode,
            "packed_weight_bytes": packed_bytes, "weight_pointer_layout_matches": matched,
            "weight_pointer_layout_complete": bool(weight_base and weight_base % 512 == 0 and matched == position),
            "pointer_links_checked": sum(len(block["pointer_links"]) for block in layers),
            "auxiliary_submissions": auxiliary, "blocks": layers}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--frame", type=int, help="check this frame from a frame-prefixed trace")
    parser.add_argument("--expect-mask-mode", choices=range(4), type=int)
    parser.add_argument("--require-weight-bindings", action="store_true",
                        help="also require every launch's parameters to match the inferred packed record addresses")
    args = parser.parse_args(argv)
    result = {"schema": "october_nr_small_trace_alignment_v1", "static_graph_sha256": GRAPH_SHA,
              "kernel_inventory_sha256": KERNELS_SHA, "reference_dll_sha256": DLL_SHA,
              "trace_producer_identity_verified": False, "tensor_values_checked": False,
              "scope": "Host submission alignment and pointer links against the pinned static preset-2 graph. Packed weight address matches are structural and do not prove the bytes uploaded. Timing and GPU completion are outside this check.",
              "aligned": False, "frames": []}
    exit_code = 1
    try:
        data = args.trace.read_bytes()
        result["trace_sha256"] = sha256(data).hexdigest()
        graph = json.loads(pinned_bytes(GRAPH, GRAPH_SHA))
        frames = read_trace(data, kernel_inventory())
        if args.frame is not None and args.frame not in frames:
            raise ValueError("requested frame is absent or the trace lacks embedded frame IDs")
        selected = frames.items() if args.frame is None else [(args.frame, frames[args.frame])]
        for frame, rows in selected:
            result["frames"].append(align_frame(frame, rows, graph, args.expect_mask_mode))
        result["aligned"] = True
        result["weight_pointer_layout_complete"] = all(frame["weight_pointer_layout_complete"] for frame in result["frames"])
        if args.require_weight_bindings and not result["weight_pointer_layout_complete"]:
            result["error"] = "one or more packed weight pointer bindings differ"
        else:
            exit_code = 0
    except (OSError, ValueError, KeyError, struct.error) as error:
        result["error"] = str(error)
    sys.stdout.buffer.write((json.dumps(result, indent=2) + "\n").encode("utf-8"))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
