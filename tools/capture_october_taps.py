"""Capture one deterministic October preset-2 case with the D3D12 nr-lab.

Use a new --out directory below the ignored repository _dl directory and a
trace-only-capable harness. Supply the tap_copy cubin for the target SM.
Run one plain case, one trace-only case, and one instrumented tap case.
The output images must be byte-identical across those three runs. Feed the
result's tap directory and its independently saved seal SHA to the comparison
tool. Only the requested frame and supported activation roles are tapped.
"""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys

from harness.reference import (
    OCTOBER_VERSION, ROOT, capture_profile, case_contract, digest, gpu_inventory, profiled_cases,
    validate_mask_log, validate_pixels, validate_report, validate_temporal_log, verify_dll_identity,
    verify_execution_gpu, write_json,
)

from .align_nr_small_trace import GRAPH, GRAPH_SHA, align_frame, kernel_inventory, pinned_bytes, read_trace
from .compare_october_taps import EVIDENCE, NATIVE, PORT, boundaries, finite_nonnegative, seal_capture


DEFAULT_TAP_SM86_SHA = "080642fe9576c32924b4bf9746e350aa51b1a139c7a3adcc3294ffec985eada4"
TRACE_SUMMARY = re.compile(r"(?m)^\[trace\] chains=(\d+) rows=(\d+) file=written chain_hook=yes legacy_hook=(?:yes|no)\r?$")
SHA256 = re.compile(r"[0-9a-fA-F]{64}")


def capture_path(path):
    root = ROOT.resolve()
    ignored = (root / "_dl").resolve()
    output = path.resolve()
    if ignored != root / "_dl" or not output.is_relative_to(ignored) or output == ignored:
        raise ValueError("capture output must be a child of the repository's ignored _dl directory")
    if output.exists() or path.is_symlink():
        raise ValueError("capture output must be a fresh directory")
    return output


def tap_identity(path, expected_hash, sm):
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("tap_copy cubin is unexpectedly large")
    data = path.read_bytes()
    digest = sha256(data).hexdigest()
    if digest != expected_hash.lower():
        raise ValueError("tap_copy cubin SHA-256 differs from the requested artifact")
    if (len(data) < 64 or not data.startswith(b"\x7fELF\x02\x01") or struct.unpack_from("<H", data, 18)[0] != 190
            or ((struct.unpack_from("<I", data, 48)[0] >> 8) & 255) != sm or b"tap_copy\0" not in data):
        raise ValueError(f"expected a tap_copy CUDA ELF cubin for sm_{sm}")
    return digest


def verify_gpu(kind, gpus, dll_hash, log=None):
    if kind == NATIVE:
        gpus = [gpu for gpu in gpus if re.search(r"\bGeForce RTX 50[0-9]{2}", gpu["name"], re.IGNORECASE)]
        if not gpus:
            raise ValueError("official October capture requires a GeForce RTX 50-series GPU")
    verify_execution_gpu(kind, OCTOBER_VERSION, gpus, dll_hash, log)


def runtime_inputs(args):
    sources = {"nr-lab.exe": args.harness.resolve(), "nvngx.dll": args.bridge.resolve(),
               "nvngx_dlssnr.dll": args.dll.resolve(), "nvngx_dlss.dll": (args.support / "nvngx_dlss.dll").resolve(),
               "nvngx_dlssg.dll": (args.support / "nvngx_dlssg.dll").resolve()}
    for name, path in sources.items():
        if not path.is_file():
            raise ValueError(f"missing {name} capture input: {path}")
    hashes = {name: digest(path) for name, path in sources.items()}
    verify_dll_identity(args.kind, OCTOBER_VERSION, hashes["nvngx_dlssnr.dll"])
    tap = args.tap_cubin.resolve()
    tap_hash = args.tap_sha256 or (DEFAULT_TAP_SM86_SHA if args.kind == PORT else None)
    if tap_hash is None or not SHA256.fullmatch(tap_hash):
        raise ValueError("supply the 64-character --tap-sha256 of the target tap_copy cubin")
    tap_hash = tap_identity(tap, tap_hash, 120 if args.kind == NATIVE else 86)
    return sources, hashes, tap, tap_hash


def check_match_seal(path, kind, case, frame, hashes):
    if path is None:
        return None
    data = path.read_bytes()
    seal = json.loads(data)
    actual = seal.pop("capture_seal_sha256")
    computed = sha256(json.dumps(seal, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if (computed != actual or seal["graph_sha256"] != GRAPH_SHA or seal["nr_version"] != OCTOBER_VERSION
            or seal["kind"] != (PORT if kind == NATIVE else NATIVE)):
        raise ValueError("companion seal does not describe the opposite October capture")
    if seal["frame"] != frame or seal["arguments"] != case["args"]:
        raise ValueError("case arguments or frame differ from the companion seal")
    if any(seal["runtime_sha256"][name] != hashes[name] for name in hashes if name != "nvngx_dlssnr.dll"):
        raise ValueError("harness or support files differ from the companion seal")
    return set(seal["boundaries"])


def checked_run(folder, case, args, sources, hashes, gpus, stage, tap=None, tap_hash=None, tap_spec=None):
    folder.mkdir()
    temp = folder / "temp"
    temp.mkdir()
    instrumentation = {}
    if stage == "trace":
        instrumentation = {"DLSS5_TRACE_ONLY": "1", "DLSS5_TRACE": "-2"}
    elif stage == "tap":
        instrumentation = {"DLSS5_PROF": "1", "DLSS5_TRACE": "-2", "DLSS5_TAP": tap_spec}
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("DLSS5_")
           and key.upper() not in {"TMP", "TEMP"}}
    env.update(instrumentation, TMP=str(temp), TEMP=str(temp))
    run = {"stage": stage, "arguments": case["args"], "runtime_sha256": hashes, "gpus": gpus,
           "instrumentation": instrumentation, "passed": False, "exit_code": None}
    if stage == "tap":
        run.update(tap_spec=tap_spec, tap_cubin_sha256=tap_hash)
    try:
        for name, source in sources.items():
            shutil.copyfile(source, folder / name)
        expected = hashes.copy()
        if stage == "tap":
            shutil.copyfile(tap, folder / "tap.cubin")
            expected["tap.cubin"] = tap_hash
        if any(digest(folder / name) != checksum for name, checksum in expected.items()):
            raise ValueError("copied runtime identity differs from the preflight hashes")
        with (folder / "stdout.log").open("xb") as stdout, (folder / "stderr.log").open("xb") as stderr:
            completed = subprocess.run([str(folder / "nr-lab.exe"), *case["args"]], cwd=folder, env=env,
                                       stdout=stdout, stderr=stderr, timeout=args.timeout)
        run["exit_code"] = completed.returncode
        if completed.returncode != 0:
            raise ValueError(f"nr-lab exited {completed.returncode}; see {stage} logs")
        if any(digest(folder / name) != checksum for name, checksum in expected.items()):
            raise ValueError("runtime artifact changed during execution")
        result = json.loads((folder / "nr-lab-result.json").read_bytes())
        log = (folder / "nr-lab.log").read_text(encoding="utf-8", errors="replace")
        validate_temporal_log(case, log)
        validate_mask_log(case, log)
        verify_gpu(args.kind, gpus, hashes["nvngx_dlssnr.dll"], log)
        loaded = re.findall(r"runtime file: (.+?nvngx_dlssnr\.dll) \((\d+) bytes\)", log, re.IGNORECASE)
        dll = folder / "nvngx_dlssnr.dll"
        if len(loaded) != 1 or Path(loaded[0][0]).resolve() != dll or int(loaded[0][1]) != dll.stat().st_size:
            raise ValueError("execution log does not identify this run's staged NR DLL")
        frames = [int(frame) for frame in re.findall(r"frame (\d+) EvaluateFeature = 0x00000001 \(Success\)", log)]
        if frames != list(range(result["evaluationsSucceeded"])):
            raise ValueError("execution log lacks complete successful Evaluate coverage")
        outputs = sorted(path for path in folder.glob("nr-lab-output-*") if path.suffix in {".ppm", ".f16"})
        validate_report(case, result, [path.name for path in outputs])
        for path in outputs:
            validate_pixels(path, result)
        if stage == "plain" and ((folder / "nr-trace.tsv").exists() or (folder / "tap_index.tsv").exists()):
            raise ValueError("instrumentation leaked into the plain run")
        if stage == "trace" and (folder / "tap_index.tsv").exists():
            raise ValueError("tap instrumentation leaked into the trace-only run")
        run.update(report=result, outputs={path.name: digest(path) for path in outputs}, passed=True)
        return run
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired, KeyboardInterrupt) as error:
        run["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        run["evidence_sha256"] = {name: digest(folder / name) for name in sorted(EVIDENCE) if (folder / name).is_file()}
        write_json(folder / "run.json", run)


def checked_trace(folder, inventory, graph, result, mode, trace_only):
    data = (folder / "nr-trace.tsv").read_bytes()
    frames = read_trace(data, inventory)
    if list(frames) != list(range(result["evaluationsSucceeded"])):
        raise ValueError("missing frame-prefixed CUDA chain trace coverage")
    if trace_only:
        stdout = (folder / "stdout.log").read_text(encoding="utf-8", errors="replace")
        status = TRACE_SUMMARY.findall(stdout)
        rows = [row for frame in frames.values() for row in frame]
        if (len(status) != 1 or tuple(map(int, status[0])) != (rows[-1].chain + 1, len(rows))
                or "[prof]" in stdout or "[tap]" in stdout):
            raise ValueError("harness did not confirm an unchanged-chain trace-only run")
    for frame, rows in frames.items():
        aligned = align_frame(frame, rows, graph, mode)
        if not aligned["weight_pointer_layout_complete"]:
            raise ValueError(f"frame {frame} does not cover the full preset-2 record binding")
    return sha256(data).hexdigest(), frames


def trace_contract(frames):
    return {frame: [(row.sequence, row.chain, row.name, row.grid, row.block, row.shared_memory, len(row.params))
                    for row in rows] for frame, rows in frames.items()}


def capture(args):
    output = capture_path(args.out)
    profile = capture_profile(args.kind, OCTOBER_VERSION, 2)
    cases = {case["id"]: case for case in profiled_cases(args.suite, profile)}
    if args.case not in cases:
        raise ValueError(f"case {args.case!r} is not in suite {args.suite!r}")
    case = cases[args.case]
    contract, _ = case_contract(case)
    if not 0 <= args.frame < contract["evaluationsSucceeded"]:
        raise ValueError("tap frame is outside the requested case")
    if args.role and len(set(args.role)) != len(args.role):
        raise ValueError("select each role once")
    mode = max(0, contract.get("maskBlendModeRequested", 0))
    sources, hashes, tap_cubin, tap_hash = runtime_inputs(args)
    companion_roles = check_match_seal(args.match_seal, args.kind, case, args.frame, hashes)
    if args.role and companion_roles is not None and set(args.role) != companion_roles:
        raise ValueError("selected roles differ from the companion seal")
    gpus = gpu_inventory()
    verify_gpu(args.kind, gpus, hashes["nvngx_dlssnr.dll"])
    inventory = kernel_inventory()
    graph = json.loads(pinned_bytes(GRAPH, GRAPH_SHA))
    output.mkdir(parents=True, exist_ok=False)
    summary = {"schema": "october_nr_controlled_tap_capture_v1", "kind": args.kind, "capture_profile": profile,
               "case": case, "frame": args.frame, "runtime_sha256": hashes, "tap_cubin_sha256": tap_hash,
               "graph_sha256": GRAPH_SHA, "tool_sha256": digest(__file__), "gpus": gpus, "stage": "plain", "passed": False}
    write_json(output / "capture.json", summary)
    try:
        plain = checked_run(output / "plain", case, args, sources, hashes, gpus, "plain")
        summary["stage"] = "trace"
        traced = checked_run(output / "trace", case, args, sources, hashes, gpus, "trace")
        if plain["outputs"] != traced["outputs"]:
            raise ValueError("trace-only output differs from the plain run")
        trace_sha, frames = checked_trace(output / "trace", inventory, graph, traced["report"], mode, True)
        _, roles = boundaries(args.frame, frames[args.frame], graph)
        selected = set(roles) if args.all_roles else set(args.role)
        if not selected or not selected <= roles.keys():
            raise ValueError(f"trace cannot cover requested roles: {sorted(selected - roles.keys())}")
        if companion_roles is not None and selected != companion_roles:
            raise ValueError("planned roles differ from the companion seal")
        requests = [role.describe() for role in sorted((roles[key] for key in selected),
                                                       key=lambda row: (row.local_sequence, row.pointer_offset))]
        payload_bytes = sum((row["bytes"] + 255) // 256 * 256 for row in requests)
        spec = ";".join([str(args.frame), *(row["tap_request"] for row in requests)])
        if payload_bytes > args.max_tap_mib * 1024 ** 2 or len(spec) >= 32767:
            raise ValueError(f"tap plan needs {payload_bytes} staging bytes; select fewer roles or raise --max-tap-mib")
        write_json(output / "plan.json", {"frame": args.frame, "trace_sha256": trace_sha, "requests": requests,
                                          "allocated_tap_bytes": payload_bytes, "tap_spec": spec})
        summary.update(stage="tap", trace_only_sha256=trace_sha, roles=sorted(selected))
        tap = checked_run(output / "tap", case, args, sources, hashes, gpus, "tap", tap_cubin, tap_hash, spec)
        if plain["outputs"] != tap["outputs"]:
            raise ValueError("tap output differs from the plain run")
        tap_sha, tapped = checked_trace(output / "tap", inventory, graph, tap["report"], mode, False)
        if trace_contract(frames) != trace_contract(tapped):
            raise ValueError("tap submission order or launch contract differs from the original-chain trace")
        tap_folder = output / "tap"
        tap.update(trace_sha256=tap_sha, outputs_equal_to_plain=True,
                   tap_sha256={path.name: digest(path) for path in sorted(tap_folder.glob("tap_s*.bin"))})
        write_json(tap_folder / "run.json", tap)
        seal, _ = seal_capture(tap_folder, inventory, graph)
        if set(seal["boundaries"]) != selected:
            raise ValueError("sealed tap coverage differs from the plan")
        for planned in requests:
            observed = seal["boundaries"][planned["role"]]
            if any(planned[key] != observed[key] for key in ("tap_request", "shape_hwc", "layout", "dtype", "bytes")):
                raise ValueError(f"tapped geometry differs from the plan for {planned['role']}")
        write_json(tap_folder / "seal.json", seal)
        summary.update(stage="complete", passed=True, outputs_equal_to_plain=True, boundaries_sealed=len(selected),
                       capture_seal_sha256=seal["capture_seal_sha256"], tap_trace_sha256=tap_sha)
    except (OSError, ValueError, KeyError, TypeError, struct.error, subprocess.SubprocessError, KeyboardInterrupt) as error:
        summary["error"] = f"{type(error).__name__}: {error}"
    finally:
        write_json(output / "capture.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=(NATIVE, PORT), required=True)
    for name in ("dll", "harness", "tap-cubin", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--bridge", type=Path, default=ROOT / "harness/nvngx.dll")
    parser.add_argument("--support", type=Path, default=ROOT / "dlss5")
    parser.add_argument("--tap-sha256", help="required for native; defaults to the pinned sm86 tap cubin on Ampere")
    parser.add_argument("--match-seal", type=Path, help="check case, roles, and support identity against a companion seal")
    parser.add_argument("--suite", choices=("smoke", "october-mask", "temporal", "motion2d", "core", "full"), default="smoke")
    parser.add_argument("--case", default="reset")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--timeout", type=finite_nonnegative, default=120)
    parser.add_argument("--max-tap-mib", type=int, default=256)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--role", action="append")
    selection.add_argument("--all-roles", action="store_true")
    args = parser.parse_args(argv)
    if args.timeout <= 0 or not 1 <= args.max_tap_mib <= 1024:
        parser.error("timeout must be positive and --max-tap-mib must be between 1 and 1024")
    try:
        report = capture(args)
    except (OSError, ValueError, KeyError, TypeError, struct.error, subprocess.SubprocessError) as error:
        report = {"stage": "preflight", "passed": False, "error": f"{type(error).__name__}: {error}"}
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
