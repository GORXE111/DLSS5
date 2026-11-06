"""Verify the matched 640x360 HT motion taps from two Ampere DLL versions.

python -B -m tools.compare_ht_activation_taps OLD_CAPTURE OCTOBER_CAPTURE [--report NEW_JSON]

Both directories must retain the nr-lab runtime, logs, output and tap files
listed in run.json. The capture manifests are the operator's trust anchors.
"""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import struct
import sys

import numpy as np

from harness.reference import (
    OCTOBER_EXPERIMENT_SHA256, ROOT, validate_pixels, validate_report,
    validate_temporal_log, verify_gpu_log,
)
from .align_nr_small_trace import pinned_bytes, read_trace
from .compare_october_taps import EVIDENCE, RUNTIME, TAP_EVENT, check_hashes, local_file, parse_tap_spec
from .ptx86.check_october_live_repack import tap_index


SCHEMA = "dlssnr_ht_matched_activation_comparison_v1"
OLD_SHA256 = "f454090a97d18e2eb8e6008d42a5fa4e37c8e4bd251a86aedfff8901fef23382"
TAP_SHA256 = "080642fe9576c32924b4bf9746e350aa51b1a139c7a3adcc3294ffec985eada4"
ORDER = ROOT / "torch/dlss5/data/exec_order.json"
ORDER_SHA256 = "fbe2e4b0d28bf90f281e67bbb74cdf2efa9c0c7a415647fb9d1e277f43ed95d4"
ARGS = ["--nr-only", "--input", "640x360", "--output", "640x360", "--profile", "scrgb",
        "--model", "1", "--scene-variant", "0", "--frames", "4", "--temporal-shift", "4", "--motion-x", "-4"]


def motion_trace(data, order, october):
    inventory = {row["kernel"]: row["param_bytes"] for row in order}
    if october:
        inventory[order[1]["kernel"]] = 272
    frames = read_trace(data, inventory)
    if list(frames) != list(range(4)):
        raise ValueError("expected all four frame-prefixed HT motion traces")
    for frame, rows in frames.items():
        if len(rows) != len(order):
            raise ValueError(f"frame {frame}: incomplete HT launch sequence")
        for row, expected in zip(rows, order):
            contract = (expected["kernel"], tuple(map(int, expected["grid"].split(","))),
                        tuple(map(int, expected["block"].split(","))), expected["smem"])
            if (row.name, row.grid, row.block, row.shared_memory) != contract or row.chain != row.sequence:
                raise ValueError(f"frame {frame}: changed HT launch or chain contract at {row.sequence}")
    return frames


def requested_boundaries(rows, october):
    """Bind the requested roles to each run's producer/consumer pointers."""
    pre_down, pre_skip = (256, 224) if october else (248, 216)
    if (struct.unpack_from("<ii", rows[1].params, pre_down + 8) != (192, 320)
            or rows[1].pointer(pre_down) != rows[2].pointer(0)
            or rows[1].pointer(pre_skip) != rows[154].pointer(8)
            or rows[153].pointer(8) != rows[154].pointer(0)):
        raise ValueError("pre/downsample or post input does not match its consumer/producer")
    roles = {"pre.downsample": (1, pre_down, (192, 320, 32)),
             "pre.skip": (1, pre_skip, (384, 640, 32))}
    for label, seq in (("first", 61), ("last", 96)):
        qkv, attention, projection = rows[seq - 1:seq + 2]
        ptrs = [attention.pointer(i) for i in (0, 8, 16, 24)]
        geometry = struct.pack("<ii", 8, 12)
        if (len(set(ptrs)) != 4 or ptrs[:3] != [qkv.pointer(i) for i in (8, 16, 24)]
                or ptrs[3] != projection.pointer(0)
                or not qkv.params[72:80] == attention.params[56:64] == projection.params[64:72] == geometry):
            raise ValueError(f"{label} attention does not match QKV/projection pointers or geometry")
        for role, offset in zip(("q", "k", "v", "out"), (0, 8, 16, 24)):
            roles[f"attention.{label}.{role}"] = (seq, offset, (8, 12, 1024))
    roles.update({"post.decoder_input": (154, 0, (192, 320, 32)),
                  "post.pre_skip": (154, 8, (384, 640, 32))})
    return roles


def checked_taps(folder, run, rows, roles, stdout):
    frame, requests = parse_tap_spec(run["tap_spec"])
    expected = {(seq, offset, math.prod(shape)) for seq, offset, shape in roles.values()}
    if frame != 0 or requests != expected:
        raise ValueError("tap spec does not match all twelve frame-0 HT roles")
    indexed = tap_index(folder, frame)
    keys = {(seq, rows[seq].pointer(offset)) for seq, offset, _ in roles.values()}
    if len(keys) != len(roles) or not all(address for _, address in keys) or indexed.keys() != keys:
        raise ValueError("tap index does not match the distinct requested trace locations")
    events = TAP_EVENT.findall(stdout)
    actual = {(int(f), int(seq), int(address, 16), int(size), int(rc))
              for f, seq, address, size, rc in events}
    expected_events = {(frame, seq, rows[seq].pointer(offset), size, 0) for seq, offset, size in requests}
    if actual != expected_events or len(events) != len(expected_events):
        raise ValueError("successful tap events do not match the requested frame, pointers and sizes")
    captures, files = {}, {}
    for role, (seq, offset, shape) in roles.items():
        address = rows[seq].pointer(offset)
        declared, indexed_file = indexed[seq, address]
        path = local_file(folder, f"tap_s{seq}_{address:x}.bin")
        if indexed_file != path or any(path.samefile(other) for other in files.values()):
            raise ValueError(f"{role}: redirected or aliased tap")
        size = math.prod(shape)
        if declared != size or path.stat().st_size != size:
            raise ValueError(f"{role}: tap size does not match the tensor shape")
        captures[role] = {"sequence": seq, "kernel": rows[seq].name, "parameter_offset": offset,
                          "shape_hwc": shape, "bytes": size, "file": path.name,
                          "sha256": run["tap_sha256"].get(path.name)}
        files[role] = path
    if set(run["tap_sha256"]) != {path.name for path in files.values()}:
        raise ValueError("run tap hashes do not identify exactly the requested tensor files")
    return captures, files


def load_capture(folder, order, october):
    manifest = local_file(folder, "run.json").read_bytes()
    run = json.loads(manifest)
    if (run["arguments"] != ARGS or type(run["exit_code"]) is not int or run["exit_code"] != 0
            or ("passed" in run and run["passed"] is not True)):
        raise ValueError("unexpected HT motion arguments or unsuccessful run")
    runtime = run["runtime_sha256"]
    expected_dll = OCTOBER_EXPERIMENT_SHA256 if october else OLD_SHA256
    if set(runtime) != RUNTIME or runtime["nvngx_dlssnr.dll"] != expected_dll or run["tap_cubin_sha256"] != TAP_SHA256:
        raise ValueError("unexpected HT runtime or tap cubin identity")
    blobs = {name: local_file(folder, name).read_bytes() for name in EVIDENCE}
    evidence = {name: sha256(blob).hexdigest() for name, blob in blobs.items()}
    if run["evidence_sha256"] != evidence or run["trace_sha256"] != evidence["nr-trace.tsv"]:
        raise ValueError("evidence or trace hash mismatch")
    case = {"id": "ht-motion-tap", "args": ARGS}
    result = json.loads(blobs["nr-lab-result.json"])
    if run.get("report", result) != result:
        raise ValueError("run manifest and result sidecar disagree")
    outputs = run["outputs"]
    validate_report(case, result, outputs)
    check_hashes(folder, {**runtime, **outputs, "tap.cubin": TAP_SHA256})
    for name in outputs:
        validate_pixels(local_file(folder, name), result)
    log = blobs["nr-lab.log"].decode("utf-8", errors="replace")
    validate_temporal_log(case, log)
    completed = re.findall(r"frame (\d+) EvaluateFeature = 0x00000001 \(Success\)", log)
    if completed != list("0123") or not re.search(r"NVAPI GPU \d+: .*\barch=0x170\b.*\bmatch=yes\b", log):
        raise ValueError("missing successful four-frame Evaluate or selected Ampere evidence")
    verify_gpu_log([gpu for gpu in run["gpus"] if gpu["compute_capability"] == "8.6"], log, "Ampere")
    frames = motion_trace(blobs["nr-trace.tsv"], order, october)
    roles = requested_boundaries(frames[0], october)
    captures, files = checked_taps(folder, run, frames[0], roles, blobs["stdout.log"].decode("utf-8", errors="replace"))
    scaling = result["nrResolvedScalingRatio"]
    if not math.isfinite(scaling) or scaling <= 0:
        raise ValueError("invalid resolved scale")
    summary = {"run_sha256": sha256(manifest).hexdigest(), "runtime_sha256": runtime,
               "tap_cubin_sha256": TAP_SHA256, "tap_spec": run["tap_spec"],
               "evidence_sha256": dict(sorted(evidence.items())), "output_sha256": outputs,
               "resolved_scaling_ratio": scaling, "boundaries": captures}
    return summary, files, frames, run["gpus"]


def compare_pre(old, october):
    rows = []
    for frame in old:
        a, b = old[frame][1].params, october[frame][1].params
        mask_mode, style_lanes = struct.unpack_from("<I", b, 40)[0], struct.unpack_from("<I", b, 196)[0]
        if (mask_mode, style_lanes) != (0, 15):
            raise ValueError(f"frame {frame}: unexpected October mask-mode or style-lane selector")
        stripped = b[:40] + b[44:196] + b[200:]
        rows.append({"frame": frame, "october_mask_mode_at_40": mask_mode, "october_style_lanes_at_196": style_lanes,
                     "old_sha256": sha256(a).hexdigest(), "october_stripped_sha256": sha256(stripped).hexdigest(),
                     "raw_264_byte_match": a == stripped})
    return rows


def compare(old_folder, october_folder):
    order = json.loads(pinned_bytes(ORDER, ORDER_SHA256))["launches"]
    old, left, old_frames, old_gpus = load_capture(old_folder, order, False)
    october, right, october_frames, october_gpus = load_capture(october_folder, order, True)
    common = RUNTIME - {"nvngx_dlssnr.dll"}
    if any(old["runtime_sha256"][name] != october["runtime_sha256"][name] for name in common) or old_gpus != october_gpus:
        raise ValueError("harness, support artifacts or GPU inventory differ")
    comparisons = []
    for role, a in old["boundaries"].items():
        b = october["boundaries"][role]
        if any(left[role].samefile(path) for path in right.values()):
            raise ValueError(f"{role}: both runs refer to the same physical tap file")
        raw_a, raw_b = left[role].read_bytes(), right[role].read_bytes()
        if any(len(raw) != row["bytes"] or sha256(raw).hexdigest() != row["sha256"]
               for raw, row in ((raw_a, a), (raw_b, b))):
            raise ValueError(f"{role}: recorded tensor SHA-256 or length differs")
        different = int(np.count_nonzero(np.frombuffer(raw_a, np.uint8) != np.frombuffer(raw_b, np.uint8)))
        comparisons.append({"role": role, "bytes_compared": a["bytes"], "differing_bytes": different,
                            "old_sha256": a["sha256"], "october_sha256": b["sha256"]})
    preserved = [side["boundaries"]["pre.skip"]["sha256"] == side["boundaries"]["post.pre_skip"]["sha256"]
                 for side in (old, october)]
    output_match = old["output_sha256"] == october["output_sha256"]
    different = sum(row["differing_bytes"] for row in comparisons)
    return {"schema": SCHEMA, "passed": different == 0 and all(preserved) and output_match,
            "scope": "RTX 3060-class Ampere HT, one instrumented frame in the fixed four-frame scRGB motion case.",
            "arguments": ARGS, "tap_frame": 0, "matched_launches": sum(map(len, old_frames.values())),
            "execution_order_sha256": ORDER_SHA256, "tool_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "old": old, "october": october, "pre_abi_comparison": compare_pre(old_frames, october_frames),
            "output_hashes_match": output_match,
            "provider_resolved_scale_matches": old["resolved_scaling_ratio"] == october["resolved_scaling_ratio"],
            "pre_skip_preserved_to_post": dict(zip(("old", "october"), preserved)),
            "comparisons": comparisons, "bytes_compared": sum(row["bytes_compared"] for row in comparisons),
            "differing_bytes": different,
            "limits": ["Only the twelve instrumented roles and final outputs were compared.",
                       "Raw pre ABI equality includes runtime-local addresses and is diagnostic.",
                       "Both sides use Ampere ports. No Blackwell or original-chain intermediate equivalence is established."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old", type=Path)
    parser.add_argument("october", type=Path)
    parser.add_argument("--report", type=Path, help="optional new JSON file outside both capture directories")
    args = parser.parse_args(argv)
    try:
        old, october = (path.resolve(strict=True) for path in (args.old, args.october))
        if old == october or old.is_relative_to(october) or october.is_relative_to(old):
            raise ValueError("use two separate capture directories")
        target = args.report.resolve() if args.report else None
        if target and (target.exists() or target.is_relative_to(old) or target.is_relative_to(october)):
            raise ValueError("report must be a new file outside both capture directories")
        result = compare(old, october)
        payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if target:
            with target.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(payload)
        else:
            print(payload, end="")
        return 0 if result["passed"] else 1
    except (OSError, ValueError, KeyError, TypeError, AttributeError, struct.error) as error:
        print(json.dumps({"schema": SCHEMA, "passed": False, "error": str(error)}, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
