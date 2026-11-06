"""Independently replay the October ViT from its 2D input to its 2D output at 4K."""

import argparse
from functools import partial
import json
from pathlib import Path
import sys

from ..align_nr_small_trace import GRAPH, GRAPH_SHA, kernel_inventory, pinned_bytes
from ..compare_october_taps import PORT, seal_capture
from ..inspect_nr_small_host_graph import DLL_SHA
from . import check_october_attention as attention, check_october_attention_tensor as tensor
from . import check_october_qkv as qkv, check_october_vit_chain as core, check_october_vit_repack as repack
from . import mma_sm86_tiles, october_vit_ffn_uhd as ffn, october_vit_qkv_projection_uhd as projections
from . import probe_october_attention_projection as projection
from . import probe_october_ffn_contract as contract, probe_october_ffn_expand as expand


HEIGHT, WIDTH = 36, 60
TOKENS = HEIGHT * WIDTH
STORAGE_TOKENS = 2176
INPUT_ROLE = "block11.repack.input"
OUTPUT_ROLE = "block18.out0"


def compare(raw, role, sealed, files):
    result = core.comparison(raw, role, sealed, files)
    boundary = sealed["boundaries"][role]
    result["storage_padding_bytes"] = (boundary.get("storage_tokens", TOKENS) - TOKENS) * boundary["shape_hwc"][2]
    return result


def run(capture, dll, ordinal):
    graph = json.loads(pinned_bytes(GRAPH, GRAPH_SHA))
    sealed, files = seal_capture(capture, kernel_inventory(), graph)
    if sealed["kind"] != PORT or sealed["boundaries"][INPUT_ROLE]["shape_hwc"] != (HEIGHT, WIDTH, core.WIDTH):
        raise ValueError("expected a sealed 36×60-token experimental sm86 capture")
    weights = {layer: expand.load_ffn_weights(dll, core.BLOCKS, layer, size)[0]
               for layer, size in core.RECORD_BYTES.items()}
    initial = files[INPUT_ROLE].read_bytes()
    if attention.sha(initial) != sealed["boundaries"][INPUT_ROLE]["sha256"]:
        raise ValueError("initial 2D ViT input changed after sealing")
    activation = repack.repack_2d_to_1d(initial, HEIGHT, WIDTH)
    input_repack = compare(activation, "block11.repack.output", sealed, files)
    offsets = attention.buffer_offsets(STORAGE_TOKENS)
    geometry = {"tokens": TOKENS, "storage_tokens": STORAGE_TOKENS}
    cases = []
    with mma_sm86_tiles.Sm86Mma(ordinal) as mma:
        hardware = mma.hardware_info
        multiply = partial(tensor.gpu_split, mma)
        qkv_multiply = partial(qkv.physical_split, mma)
        for block in core.BLOCKS:
            prefix = f"block{block}"
            checks = [compare(activation, f"{prefix}.ffn.expand.input", sealed, files)]
            expanded, residual = ffn.predict_ffn(activation, weights[0][block], weights[1][block], multiply, tokens=TOKENS)
            checks.append(compare(expanded, f"{prefix}.ffn.expand.output", sealed, files))
            checks.append(compare(residual, f"{prefix}.qkv.input", sealed, files))
            qkv_codes = projections.predict_qkv(residual, weights[2][block], qkv_multiply, **geometry)
            for i, codes in enumerate(qkv_codes):
                packed = core.pack_attention(codes, offsets[i])
                checks.append(compare(packed, f"{prefix}.attention.in{i}", sealed, files))
            stages = tensor.evaluate(*(attention.FP8[codes] for codes in qkv_codes), multiply, multiply, tokens=TOKENS)
            attended = core.pack_attention(tensor.encode_output(stages["pre_quantized"], TOKENS), offsets[3])
            checks.append(compare(attended, f"{prefix}.attention.out", sealed, files))
            activation = projections.predict_projection(attended, residual, weights[4][block], multiply, **geometry)
            suffix = "repack.input" if block == core.BLOCKS[-1] else "out1"
            checks.append(compare(activation, f"{prefix}.{suffix}", sealed, files))
            cases.append({"block": block, "boundaries": checks,
                          "weight_sha256": {f"layer{layer}": attention.sha(records[block]) for layer, records in weights.items()}})
            errors = {row["role"]: row["different_bytes"] for row in checks if row["different_bytes"]}
            print(f"ViT block {block}: {errors if errors else '8 sealed boundaries matched'}", file=sys.stderr, flush=True)

    final = repack.repack_1d_to_2d(activation, HEIGHT, WIDTH)
    output_repack = compare(final, OUTPUT_ROLE, sealed, files)
    derived = [input_repack, *(row for case in cases for row in case["boundaries"]), output_repack]
    paths = [Path(__file__), *(Path(module.__file__) for module in (attention, tensor, qkv, core, repack,
                             mma_sm86_tiles, ffn, projections, projection, contract, expand))]
    return {"schema": "october_nr_vit_chain_uhd_v1", "capture_seal_sha256": sealed["capture_seal_sha256"],
            "original_dll_sha256": DLL_SHA, "source_sha256": {p.name: attention.sha(p.read_bytes()) for p in paths},
            **hardware, "frame": sealed["frame"], "shape_hwc": [HEIGHT, WIDTH, core.WIDTH], **geometry,
            "input_role": INPUT_ROLE, "input_sha256": attention.sha(initial), "input_repack": input_repack,
            "cases": cases, "output_repack": output_repack, "derived_boundaries": len(derived),
            "derived_bytes": sum(row["bytes"] for row in derived),
            "storage_padding_bytes": sum(row["storage_padding_bytes"] for row in derived),
            "different_bytes": sum(row["different_bytes"] for row in derived),
            "passed": all(not row["different_bytes"] for row in derived)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--original-dll", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.device < 0:
        parser.error("device must be nonnegative")
    try:
        report = run(args.capture, args.original_dll, args.device)
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        report = {"passed": False, "error": str(error)}
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
