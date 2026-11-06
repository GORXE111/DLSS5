"""Plan, seal and compare captured 310.9.16.0 preset-2 activation boundaries.

  python -B -m tools.compare_october_taps plan CAPTURE/nr-trace.tsv --frame 0
  python -B -m tools.compare_october_taps seal CAPTURE
  python -B -m tools.compare_october_taps compare NATIVE PORT --native-seal SHA
      --port-seal SHA --role block11.repack.output [--role ...]

Use separate nr-lab directories with the same deterministic case, a frame-
prefixed trace, full-buffer taps, and run.json as in the local October tap
runs. Each architecture's tap cubin has its own verified hash. Keep each seal
SHA independently at capture time. This offline tool validates integrity
against those hashes, with the capture operator as the trust root. Results
cover only the requested instrumented tensor boundaries (up to 256 MiB each).
"""

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import struct
import sys

import numpy as np

from harness.reference import (
    OCTOBER_EXPERIMENT_KIND, OCTOBER_EXPERIMENT_SHA256, OCTOBER_SHA256, OCTOBER_VERSION,
    case_contract, digest, validate_mask_log, validate_pixels, validate_report, validate_temporal_log,
    verify_execution_gpu,
)
from .align_nr_small_trace import (
    GRAPH, GRAPH_SHA, REPACK_IN, REPACK_OUT, Launch, align_frame, kernel_inventory, output_slots, pinned_bytes, read_trace,
)
from .ptx86.check_october_live_repack import tap_index


RUNTIME = {"nr-lab.exe", "nvngx.dll", "nvngx_dlssnr.dll", "nvngx_dlss.dll", "nvngx_dlssg.dll"}
EVIDENCE = {"nr-lab-result.json", "nr-lab.log", "stdout.log", "stderr.log", "nr-trace.tsv", "tap_index.tsv"}
TAP_REQUEST = re.compile(r"([0-9]+):p([0-9]+):([0-9]+)")
TAP_EVENT = re.compile(r"\[tap\] frame (\d+) seq (\d+) (0x[0-9a-f]+) \+(\d+) -> off \d+ \(launch=(-?\d+)\)")
SCHEMA = "october_nr_activation_tap_v1"
NATIVE = "official-sm120"
PORT = OCTOBER_EXPERIMENT_KIND


@dataclass(frozen=True)
class Boundary:
    role: str
    local_sequence: int
    launch: Launch
    pointer_offset: int
    shape_hwc: tuple
    layout: str

    @property
    def size(self):
        return math.prod(self.shape_hwc)

    def describe(self):
        return {"role": self.role, "local_sequence": self.local_sequence, "trace_sequence": self.launch.sequence,
                "kernel": self.launch.name, "parameter_offset": self.pointer_offset,
                "parameter_sha256": sha256(self.launch.params).hexdigest(), "shape_hwc": self.shape_hwc,
                "layout": self.layout, "dtype": "e4m3fn", "bytes": self.size,
                "tap_request": f"{self.local_sequence}:p{self.pointer_offset // 8}:{self.size}"}


def spatial_output(block, slot, launch):
    kind, variant = block["kind"], block["variant"]
    if kind == "fused_pre_swin_1h":
        return (264, 32, "image16") if slot == 0 else (248, 32, "tin2d")
    if kind.startswith("fused_swin_"):
        heads = int(kind.removeprefix("fused_swin_").removesuffix("h"))
        if variant == "downsample" and slot == 0:
            return (72 if heads == 1 else 80), 64 * heads, "image16"
        return (24 if heads == 1 else 32), 32 * heads, ("image16" if variant == "outview" else "tin2d")
    if kind == "split_swin_16h":
        if variant == "projection_pool_and_final_head":
            return (32, 1024, "tin2d") if slot == 0 else (64, 512, "tin2d")
        return 32, 512, ("image16" if variant == "outview" else "tin2d")
    if kind == "vit_1d":
        return (16, 1024, "tin2d") if launch.name == REPACK_OUT else (64, 1024, "tin1d")
    if kind == "decoder_input_upsample":
        return 72, 512, "tin2d"
    raise ValueError(f"no activation ABI for {kind}")


def boundaries(frame, rows, graph):
    aligned = align_frame(frame, rows, graph)
    if not aligned["weight_pointer_layout_complete"]:
        raise ValueError("trace does not bind the complete preset-2 packed weight layout")
    by_sequence = {row.sequence: (i, row) for i, row in enumerate(rows)}
    roles = {}

    def add(role, launch, offset, dims_at, channels, layout):
        height, width = struct.unpack_from("<ii", launch.params, dims_at)
        if (role in roles or height <= 0 or width <= 0 or height % 4 or width % 4
                or (layout in {"tin1d", "vit1d_attention_storage"} and height * width % 32) or not launch.pointer(offset)
                or height * width * channels > 256 * 1024 * 1024):
            raise ValueError(f"unsupported or invalid tensor boundary {role}")
        roles[role] = Boundary(role, by_sequence[launch.sequence][0], launch, offset, (height, width, channels), layout)

    for block, matched in zip(graph["blocks"], aligned["blocks"]):
        prefix = f"block{block['index']:02}"
        launches = [by_sequence[entry["sequence"]][1] for entry in matched["launches"]]
        for slot, (launch, offset) in output_slots(block, launches).items():
            add(f"{prefix}.out{slot}", launch, offset, *spatial_output(block, slot, launch))
        if block["kind"] == "vit_1d":
            if launches[0].name == REPACK_IN:
                add(f"{prefix}.repack.input", launches[0], 0, 16, 1024, "tin2d")
                add(f"{prefix}.repack.output", launches[0], 8, 16, 1024, "tin1d")
            if launches[-1].name == REPACK_OUT:
                add(f"{prefix}.repack.input", launches[-1], 0, 16, 1024, "tin1d")
            position, = (i for i, launch in enumerate(launches) if launch.name.startswith("cc_vit_1d_attention_"))
            qkv, attention, projection = launches[position - 1:position + 2]
            pointers = [attention.pointer(offset) for offset in (0, 8, 16, 24)]
            if (len(set(pointers)) != 4 or pointers[:3] != [qkv.pointer(offset) for offset in (8, 16, 24)]
                    or pointers[3] != projection.pointer(0)
                    or not qkv.params[72:80] == attention.params[56:64] == projection.params[64:72]):
                raise ValueError(f"{prefix}: inconsistent ViT attention inputs, output or geometry")
            for role, offset in (("in0", 0), ("in1", 8), ("in2", 16), ("out", 24)):
                add(f"{prefix}.attention.{role}", attention, offset, 56, 1024, "vit1d_attention_storage")
        if block["kind"] == "fused_post_swin_1h":
            # The two post inputs are read-only activation VAs. Post's image
            # output is a surface handle and is never interpreted as a VA.
            for number, edge in enumerate(block["inputs"]):
                producer = roles[f"block{edge['block']:02}.out{edge['output']}"]
                roles[f"{prefix}.in{number}"] = Boundary(f"{prefix}.in{number}", by_sequence[launches[0].sequence][0],
                                                        launches[0], number * 8, producer.shape_hwc, producer.layout)
    return aligned, roles


def local_file(folder, name):
    if not isinstance(name, str) or Path(name).name != name or not name:
        raise ValueError(f"capture artifact is not a single file name: {name!r}")
    path = folder / name
    if path.resolve().parent != folder or path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or redirected capture artifact {name}")
    return path


def check_hashes(folder, expected):
    for name, recorded in expected.items():
        if digest(local_file(folder, name)) != recorded:
            raise ValueError(f"capture artifact SHA-256 mismatch: {name}")


def parse_tap_spec(text):
    first, *entries = text.split(";")
    if not re.fullmatch(r"[0-9]+", first) or not entries:
        raise ValueError("expected a single frame and full-buffer pN tap requests")
    requested = set()
    for entry in entries:
        match = TAP_REQUEST.fullmatch(entry)
        if match is None:
            raise ValueError(f"unsupported tap request {entry!r}")
        sequence, pointer_index, size = map(int, match.groups())
        key = (sequence, pointer_index * 8, size)
        if not size or key in requested:
            raise ValueError(f"empty or duplicate tap request {entry!r}")
        requested.add(key)
    return int(first), requested


def seal_capture(folder, inventory, graph):
    folder = folder.resolve()
    run_bytes = local_file(folder, "run.json").read_bytes()
    run = json.loads(run_bytes)
    runtime = run["runtime_sha256"]
    exit_code = run.get("exit_code", 0)
    if set(runtime) != RUNTIME or type(exit_code) is not int or exit_code != 0:
        raise ValueError("missing runtime inventory or unsuccessful tap run")
    check_hashes(folder, {**runtime, "tap.cubin": run["tap_cubin_sha256"], **run["outputs"]})
    dll_hash = runtime["nvngx_dlssnr.dll"]
    kind = {OCTOBER_SHA256: NATIVE, OCTOBER_EXPERIMENT_SHA256: PORT}.get(dll_hash)
    if kind is None:
        raise ValueError("unexpected October runtime SHA-256")
    raw_args = run.get("arguments", run.get("case", {}).get("args"))
    if not isinstance(raw_args, list) or any(not isinstance(value, str) for value in raw_args):
        raise ValueError("missing original nr-lab arguments")
    flags = [arg for arg in raw_args if arg.startswith("--")]
    if len(set(flags)) != len(flags) or "--nr-only" not in flags:
        raise ValueError("ambiguous nr-lab arguments or unsupported pipeline")
    if run.get("case", {}).get("args", raw_args) != raw_args:
        raise ValueError("conflicting nr-lab command metadata")
    case = {"id": "activation-tap", "args": raw_args}
    paths = {name: local_file(folder, name) for name in EVIDENCE}
    inputs = {name: path.read_bytes() for name, path in paths.items()}
    evidence = {name: sha256(raw).hexdigest() for name, raw in sorted(inputs.items())}
    recorded = run.get("evidence_sha256")
    if (recorded is not None and recorded != evidence) or run["trace_sha256"] != evidence["nr-trace.tsv"]:
        raise ValueError("recorded trace or capture evidence SHA-256 differs")
    result = json.loads(inputs["nr-lab-result.json"])
    if run.get("report", result) != result or result.get("model") != 2:
        raise ValueError("inconsistent result or non-small preset")
    scaling = result["nrResolvedScalingRatio"]
    if not math.isfinite(scaling) or scaling <= 0:
        raise ValueError("invalid resolved network scaling ratio")
    validate_report(case, result, run["outputs"])
    for name in run["outputs"]:
        validate_pixels(local_file(folder, name), result)
    log = inputs["nr-lab.log"].decode("utf-8", errors="replace")
    validate_temporal_log(case, log)
    validate_mask_log(case, log)
    completed = [int(frame) for frame in re.findall(r"frame (\d+) EvaluateFeature = 0x00000001 \(Success\)", log)]
    if completed != list(range(result["evaluationsSucceeded"])):
        raise ValueError("missing successful Evaluate evidence for one or more frames")
    if kind == NATIVE or "gpus" in run:
        if "exit_code" not in run or recorded is None:
            raise ValueError("versioned capture lacks exit status or recorded evidence hashes")
        verify_execution_gpu(kind, OCTOBER_VERSION, run["gpus"], dll_hash, log)
    elif not re.search(r"NVAPI GPU [0-9]+: .*\barch=0x170\b.*\bmatch=yes\b", log):
        raise ValueError("staged run log lacks matching Ampere hardware evidence")

    frame, requests = parse_tap_spec(run["tap_spec"])
    frames = read_trace(inputs["nr-trace.tsv"], inventory)
    if list(frames) != completed or frame not in frames:
        raise ValueError("complete frame-prefixed trace does not agree with the successful Evaluate log")
    aligned, roles = boundaries(frame, frames[frame], graph)
    expected, _ = case_contract(case)
    if aligned["mask_blend_mode"] != max(0, expected.get("maskBlendModeRequested", 0)):
        raise ValueError("pre kernel mask mode differs from requested mode")
    by_request = {(row.local_sequence, row.pointer_offset, row.size): row for row in roles.values()}
    if len(by_request) != len(roles) or not requests <= by_request.keys():
        raise ValueError("tap request does not select a supported full tensor boundary")
    taps = tap_index(folder, frame)
    requested = {(seq, frames[frame][seq].pointer(offset)): by_request[seq, offset, size] for seq, offset, size in requests}
    if len(requested) != len(requests) or requested.keys() != taps.keys():
        raise ValueError("tap index does not exactly match the requested trace locations")
    tap_events = TAP_EVENT.findall(inputs["stdout.log"].decode("utf-8", errors="replace"))
    expected_events = {(frame, seq, address, row.size, 0) for (seq, address), row in requested.items()}
    events = {(int(f), int(seq), int(addr, 16), int(size), int(rc)) for f, seq, addr, size, rc in tap_events}
    if events != expected_events or len(tap_events) != len(events):
        raise ValueError("successful tap log does not exactly match the selected trace locations")
    captures = {}
    files = {}
    for key, row in sorted(requested.items(), key=lambda entry: entry[1].role):
        declared, indexed = taps[key]
        sequence, address = key
        path = local_file(folder, f"tap_s{sequence}_{address:x}.bin")
        if indexed != path or any(path.samefile(other) for other in files.values()):
            raise ValueError(f"redirected or aliased tensor file for {row.role}")
        if declared != row.size or path.stat().st_size != row.size:
            raise ValueError(f"tap size differs from trace-derived geometry for {row.role}")
        files[row.role] = path
        captures[row.role] = {**row.describe(), "file": path.name, "sha256": digest(path)}
    recorded_taps = run.get("tap_sha256")
    if recorded_taps is not None and recorded_taps != {row["file"]: row["sha256"] for row in captures.values()}:
        raise ValueError("recorded tensor artifact SHA-256 differs")
    for producer, consumer in (("block29.out0", "block30.in0"), ("block00.out1", "block30.in1")):
        if producer in captures and consumer in captures and captures[producer]["sha256"] != captures[consumer]["sha256"]:
            raise ValueError(f"preserved post input changed between producer and consumer: {consumer}")
    pre_sequence = aligned["blocks"][0]["launches"][0]["sequence"]
    pre = next(launch for launch in frames[frame] if launch.sequence == pre_sequence)
    pre_contract = {"texture_slots_present": [bool(pre.pointer(offset)) for offset in range(0, 40, 8)],
                    "control_and_geometry_sha256": sha256(pre.params[40:224] + pre.params[248:256] +
                                                          pre.params[264:272]).hexdigest()}
    report = {"schema": SCHEMA, "kind": kind, "nr_version": OCTOBER_VERSION, "graph_sha256": GRAPH_SHA,
              "run_sha256": sha256(run_bytes).hexdigest(), "runtime_sha256": runtime,
              "tap_cubin_sha256": run["tap_cubin_sha256"], "producer_evidence_hashes_present": recorded is not None,
              "producer_exit_code_present": "exit_code" in run, "producer_tap_hashes_present": recorded_taps is not None,
              "artifact_sha256": evidence,
              "arguments": raw_args, "frame": frame, "observed_mask_blend_mode": aligned["mask_blend_mode"],
              "pre_contract": pre_contract, "resolved_scaling_ratio": scaling,
              "output_sha256": run["outputs"], "boundaries": captures}
    report["capture_seal_sha256"] = sha256(json.dumps(report, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return report, files


def e4m3_lookup():
    codes = np.arange(256, dtype=np.int32)
    exponent, fraction = (codes >> 3) & 15, (codes & 7) / 8
    values = np.ldexp(np.where(exponent == 0, fraction, 1 + fraction), np.maximum(exponent, 1) - 7)
    values[codes >= 128] *= -1
    values[[0x7F, 0xFF]] = np.nan
    return values


def tensor_metrics(reference, candidate):
    a, b = np.frombuffer(reference, dtype=np.uint8), np.frombuffer(candidate, dtype=np.uint8)
    if len(a) != len(b) or not len(a):
        raise ValueError("expected equally sized nonempty activation buffers")
    lookup = e4m3_lookup()
    different = nonfinite_a = nonfinite_b = finite_pairs = 0
    total = squared = peak = 0.0
    chunk_bytes = 1024 * 1024
    for offset in range(0, len(a), chunk_bytes):
        first, second = a[offset:offset + chunk_bytes], b[offset:offset + chunk_bytes]
        different += int(np.count_nonzero(first != second))
        x, y = lookup[first], lookup[second]
        finite_a, finite_b = np.isfinite(x), np.isfinite(y)
        nonfinite_a += int(x.size - np.count_nonzero(finite_a))
        nonfinite_b += int(y.size - np.count_nonzero(finite_b))
        finite = finite_a & finite_b
        delta = np.abs(x[finite] - y[finite])
        finite_pairs += int(delta.size)
        total += float(delta.sum())
        squared += float(np.square(delta).sum())
        peak = max(peak, float(delta.max(initial=0)))
    return {"elements": int(a.size), "different_bytes": different, "nonfinite_reference": nonfinite_a,
            "nonfinite_candidate": nonfinite_b, "finite_pairs": finite_pairs,
            "max_abs": peak if finite_pairs else None,
            "mae": total / finite_pairs if finite_pairs else None,
            "rmse": math.sqrt(squared / finite_pairs) if finite_pairs else None}


def selected_inputs(role, graph):
    if role.endswith(".attention.out"):
        return [role.removesuffix("out") + f"in{i}" for i in range(3)]
    if role == "block11.repack.output":
        return ["block11.repack.input"]
    if role == "block18.out0":
        return ["block18.repack.input"]
    if re.fullmatch(r"block\d+\.out[01]", role):
        index = int(role.split(".")[0].removeprefix("block"))
        if index:
            return [f"block{edge['block']:02}.out{edge.get('output', 0)}" for edge in graph["blocks"][index]["inputs"]]
    return []


def compare_captures(native, port, native_files, port_files, expected_roles, max_abs, mae, graph):
    if native["kind"] != NATIVE or port["kind"] != PORT:
        raise ValueError("comparison requires an original Blackwell capture and a staged Ampere capture")
    for key in ("arguments", "frame", "observed_mask_blend_mode", "pre_contract", "resolved_scaling_ratio"):
        if native[key] != port[key]:
            raise ValueError(f"comparison case differs in {key}")
    for name in RUNTIME - {"nvngx_dlssnr.dll"}:
        if native["runtime_sha256"][name] != port["runtime_sha256"][name]:
            raise ValueError(f"comparison harness or support artifact differs: {name}")
    if set(native["boundaries"]) != expected_roles or set(port["boundaries"]) != expected_roles:
        raise ValueError("capture boundary sets must both match the explicitly requested roles")
    comparisons = []
    for role in sorted(expected_roles):
        a, b = native["boundaries"][role], port["boundaries"][role]
        for key in ("shape_hwc", "dtype", "layout", "bytes"):
            if a[key] != b[key]:
                raise ValueError(f"{role}: activation {key} differs across captures")
        if native_files[role].samefile(port_files[role]):
            raise ValueError(f"{role}: both runs refer to the same physical tap file")
        raw = [path.read_bytes() for path in (native_files[role], port_files[role])]
        if any(len(data) != entry["bytes"] or sha256(data).hexdigest() != entry["sha256"] for data, entry in zip(raw, (a, b))):
            raise ValueError(f"{role}: a tap changed after its capture was sealed")
        metrics = tensor_metrics(*raw)
        finite = not (metrics["nonfinite_reference"] or metrics["nonfinite_candidate"])
        exact = metrics["different_bytes"] == 0
        passed = finite and (exact if max_abs is None else metrics["max_abs"] <= max_abs and metrics["mae"] <= mae)
        comparisons.append({"role": role, "native": a, "port": b, "bit_exact": exact, "metrics": metrics, "passed": passed})
    by_role = {row["role"]: row for row in comparisons}
    for row in comparisons:
        required = selected_inputs(row["role"], graph)
        missing = [role for role in required if role not in by_role]
        row["input_comparison"] = {"roles": required, "missing": missing,
                                   "all_present_inputs_bit_exact_and_finite": bool(required) and not missing and
                                   all(by_role[role]["bit_exact"] and not by_role[role]["metrics"]["nonfinite_reference"]
                                       for role in required)}
    return comparisons


def finite_nonnegative(text):
    value = float(text)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("threshold must be finite and nonnegative")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("trace", type=Path)
    plan.add_argument("--frame", type=int, default=0)
    seal = commands.add_parser("seal")
    seal.add_argument("capture", type=Path)
    compare = commands.add_parser("compare")
    compare.add_argument("native", type=Path)
    compare.add_argument("port", type=Path)
    compare.add_argument("--native-seal", required=True)
    compare.add_argument("--port-seal", required=True)
    compare.add_argument("--role", action="append", required=True)
    compare.add_argument("--max-abs", type=finite_nonnegative)
    compare.add_argument("--mae", type=finite_nonnegative)
    args = parser.parse_args(argv)
    if args.command == "compare" and ((args.max_abs is None) != (args.mae is None) or len(set(args.role)) != len(args.role)):
        parser.error("provide both numeric thresholds or neither, and select each role once")
    status = 1
    try:
        inventory = kernel_inventory()
        graph = json.loads(pinned_bytes(GRAPH, GRAPH_SHA))
        if args.command == "plan":
            trace = args.trace.read_bytes()
            frames = read_trace(trace, inventory)
            if args.frame not in frames:
                raise ValueError("trace lacks the requested embedded frame ID")
            _, roles = boundaries(args.frame, frames[args.frame], graph)
            report = {"schema": SCHEMA, "frame": args.frame, "trace_sha256": sha256(trace).hexdigest(),
                      "boundaries": [row.describe() for row in roles.values()]}
            status = 0
        elif args.command == "seal":
            report, _ = seal_capture(args.capture, inventory, graph)
            status = 0
        else:
            if args.native.resolve() == args.port.resolve():
                raise ValueError("captures must be separate directories")
            native, native_files = seal_capture(args.native, inventory, graph)
            port, port_files = seal_capture(args.port, inventory, graph)
            if (native["capture_seal_sha256"], port["capture_seal_sha256"]) != (args.native_seal, args.port_seal):
                raise ValueError("one or both capture seals differ from their independently recorded SHA-256")
            rows = compare_captures(native, port, native_files, port_files, set(args.role), args.max_abs, args.mae, graph)
            passed = all(row["passed"] for row in rows)
            report = {"schema": SCHEMA, "scope": "Selected instrumented activation boundaries and all stored lanes, including padding.",
                      "native_seal_sha256": args.native_seal, "port_seal_sha256": args.port_seal,
                      "native_tap_cubin_sha256": native["tap_cubin_sha256"], "port_tap_cubin_sha256": port["tap_cubin_sha256"],
                      "frame": native["frame"], "max_abs_threshold": args.max_abs, "mae_threshold": args.mae,
                      "comparisons": rows, "passed": passed}
            status = 0 if passed else 1
    except (OSError, ValueError, KeyError, TypeError, struct.error, IndexError, StopIteration) as error:
        report = {"schema": SCHEMA, "passed": False, "error": str(error)}
    print(json.dumps(report, indent=2, allow_nan=False))
    return status


if __name__ == "__main__":
    sys.exit(main())
