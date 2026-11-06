"""Compose all eight October ViT cores from the first sealed input on sm86."""

import argparse
from functools import partial
import json
from pathlib import Path
import sys

import numpy as np

from ..align_nr_small_trace import GRAPH, GRAPH_SHA, kernel_inventory, pinned_bytes
from ..compare_october_taps import PORT, seal_capture
from ..inspect_nr_small_host_graph import DLL_SHA
from . import check_october_attention as attention, check_october_attention_tensor as tensor
from . import check_october_qkv as qkv, mma_sm86_tiles, probe_october_attention_projection as projection
from . import probe_october_ffn_contract as contract, probe_october_ffn_expand as expand


TOKENS, WIDTH, EXPANDED = 96, 1024, 4096
BLOCKS = tuple(range(11, 19))
RECORD_BYTES = {0: WIDTH * EXPANDED + 16, 1: EXPANDED * WIDTH + 2 * WIDTH,
                2: 3 * WIDTH * WIDTH + 128, 4: WIDTH * WIDTH + 2 * WIDTH}


def pack_attention(codes, offsets):
    packed = np.empty(codes.size, np.uint8)
    packed[offsets] = codes
    return packed.tobytes()


def residual_seed(activation, weights, matrix_bytes):
    scale = np.frombuffer(weights, "<f2", offset=matrix_bytes)
    return attention.f16(activation * scale)


def comparison(raw, role, sealed, files):
    row = sealed["boundaries"][role]
    if len(raw) != row["bytes"]:
        raise ValueError(f"{role}: wrong predicted tensor size")
    predicted = attention.sha(raw)
    expected = row["sha256"]
    different = 0
    if predicted != expected:
        observed = files[role].read_bytes()
        if attention.sha(observed) != expected:
            raise ValueError(f"{role}: capture changed after sealing")
        different = int(np.count_nonzero(np.frombuffer(raw, np.uint8) != np.frombuffer(observed, np.uint8)))
    return {"role": role, "bytes": len(raw), "different_bytes": different,
            "predicted_sha256": predicted, "sealed_sha256": expected}


def run(capture, dll, ordinal):
    graph = json.loads(pinned_bytes(GRAPH, GRAPH_SHA))
    sealed, files = seal_capture(capture, kernel_inventory(), graph)
    if sealed["kind"] != PORT:
        raise ValueError("expected a sealed experimental sm86 capture")
    for block in BLOCKS:
        for suffix, width in (("ffn.expand.input", WIDTH), ("ffn.expand.output", EXPANDED)):
            row = sealed["boundaries"].get(f"block{block}.{suffix}")
            if row is None or row["shape_hwc"] != (8, 12, width):
                raise ValueError("ViT chain requires the sealed 96-token FFN geometry")

    weights = {layer: expand.load_ffn_weights(dll, BLOCKS, layer, size)[0] for layer, size in RECORD_BYTES.items()}
    offsets = attention.buffer_offsets(TOKENS)
    contract_coordinates = expand.tin_coordinates(TOKENS, WIDTH)
    projection_coordinates = projection.projection_coordinates()
    input_role = "block11.ffn.expand.input"
    activation = files[input_role].read_bytes()
    if attention.sha(activation) != sealed["boundaries"][input_role]["sha256"]:
        raise ValueError("initial ViT input changed after sealing")

    cases = []
    with mma_sm86_tiles.Sm86Mma(ordinal) as mma:
        hardware = mma.hardware_info
        multiply = partial(tensor.gpu_split, mma)
        qkv_multiply = partial(qkv.physical_split, mma)
        for block in BLOCKS:
            checks = []
            prefix = f"block{block}"
            checks.append(comparison(activation, f"{prefix}.ffn.expand.input", sealed, files))
            decoded = expand.decode_tin(activation, TOKENS, WIDTH)
            expanded = expand.predict(decoded, expand.decode_weights(weights[0][block], WIDTH, EXPANDED), multiply).tobytes()
            checks.append(comparison(expanded, f"{prefix}.ffn.expand.output", sealed, files))
            residual = contract.predict(expand.decode_tin(expanded, TOKENS, EXPANDED),
                                        residual_seed(decoded, weights[1][block], EXPANDED * WIDTH),
                                        expand.decode_weights(weights[1][block], EXPANDED, WIDTH),
                                        contract_coordinates, multiply).tobytes()
            checks.append(comparison(residual, f"{prefix}.qkv.input", sealed, files))
            qkv_codes = qkv.predict(residual, weights[2][block], qkv_multiply)
            for i, codes in enumerate(qkv_codes):
                checks.append(comparison(pack_attention(codes, offsets[i]), f"{prefix}.attention.in{i}", sealed, files))
            stages = tensor.evaluate(*(attention.FP8[codes] for codes in qkv_codes), multiply, multiply, tokens=TOKENS)
            attended = pack_attention(tensor.encode_output(stages["pre_quantized"], TOKENS), offsets[3])
            checks.append(comparison(attended, f"{prefix}.attention.out", sealed, files))
            residual_matrix = projection.activation_matrix(residual, projection_coordinates)
            activation = projection.predict_projection(projection.activation_matrix(attended, projection_coordinates),
                                                        residual_seed(residual_matrix, weights[4][block], WIDTH * WIDTH),
                                                        projection.weight_matrix(weights[4][block]),
                                                        projection_coordinates, multiply).tobytes()
            suffix = "repack.input" if block == BLOCKS[-1] else "out1"
            checks.append(comparison(activation, f"{prefix}.{suffix}", sealed, files))
            cases.append({"block": block, "boundaries": checks,
                          "weight_sha256": {f"layer{layer}": attention.sha(records[block]) for layer, records in weights.items()}})

    derived = [row for case in cases for row in case["boundaries"] if row["role"] != input_role]
    paths = [Path(__file__), *(Path(module.__file__) for module in
                             (attention, tensor, qkv, mma_sm86_tiles, expand, contract, projection))]
    return {"schema": "october_nr_vit_chain_v1", "capture_seal_sha256": sealed["capture_seal_sha256"],
            "original_dll_sha256": DLL_SHA, "source_sha256": {p.name: attention.sha(p.read_bytes()) for p in paths},
            "scope": "Independent core between the 2D repacks, with one initial activation and no intermediate substitutions.",
            **hardware, "frame": sealed["frame"], "input_role": input_role, "tokens": TOKENS,
            "derived_boundaries": len(derived), "derived_bytes": sum(row["bytes"] for row in derived),
            "different_bytes": sum(row["different_bytes"] for row in derived), "cases": cases,
            "passed": all(row["different_bytes"] == 0 for row in derived)}


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
