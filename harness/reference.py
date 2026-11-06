"""Capture and compare versioned DLSS-NR reference cases; never infer universal equivalence.

Uses the existing D3D12 laboratory executable. Captures contain outputs, logs,
commands and hashes, but no NVIDIA DLLs. An official-sm120 label requires a
pinned unmodified DLL and a matching Blackwell GPU in the execution log.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_SHA256 = "e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e"
OCTOBER_VERSION = "310.9.16.0"
OCTOBER_SHA256 = "2bc19bc09a7b89dd44f73e2ac0108fcc586e2154bdd0830f242f4f040e28789f"
OCTOBER_EXPERIMENT_SHA256 = "2d3e405535cfb574828a5940f62d14952044f4feb1013caf9e7b9240385506ed"
OCTOBER_EXPERIMENT_KIND = "experimental-sm86"
NATIVE_SHA256 = {"310.8.0": ORIGINAL_SHA256, OCTOBER_VERSION: OCTOBER_SHA256}
OCTOBER_PRESETS = {1: "WEIGHTS_HT", 2: "WEIGHTS_CC_SMALL_FP8"}
SCHEMA = 1


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def suite_cases(suite):
    def case(name, frames=1, size="640x360", extra=()):
        return {"id": name, "args": ["--nr-only", "--input", size, "--output", size,
                                      "--frames", str(frames), *extra]}
    if suite == "october-mask":
        common = ("--profile", "scrgb", "--scene-variant", "0", "--local-tone", "0.75",
                  "--local-structure", "0.5", "--skin-structure", "0.25")
        mask = ("--optional", "control-mask", "--optional-format", "rgba16f", "--optional-variant", "3",
                "--optional-rgba", "1,0.625,1,1", "--ramp-channels", "4")
        cases = [case(f"blend{mode}_f{frames - 1}", frames,
                      extra=(*common, *mask, "--auto-mask", "1", "--mask-blend", str(mode)))
                 for mode in range(4) for frames in (1, 4)]
        return [*cases, case("no_control_mask_f3", 4, extra=(*common, "--auto-mask", "1", "--mask-blend", "2")),
                case("auto_mask_disabled_f3", 4, extra=(*common, *mask, "--auto-mask", "0", "--mask-blend", "2"))]
    if suite == "motion2d":
        groups = [("diagonal", 3, 2, -3, -2, "640x360"),
                  ("negative", -3, -2, 3, 2, "640x360"),
                  ("vertical", 0, 5, 0, -5, "640x360"),
                  ("fractional_static", 0, 0, -0.5, 0.25, "640x360"),
                  ("fractional_diagonal", 3, 2, -3.5, -1.75, "640x360"),
                  ("large_clamped", 80, 60, -80, -60, "640x360"),
                  ("odd_diagonal", 3, 2, -3, -2, "641x361"),
                  ("zero_mv_control", 3, 2, 0, 0, "640x360")]
        cases = [case(f"{name}_f{f}", f + 1, size,
                     ("--scene-variant", "0", "--temporal-shift", str(sx), "--temporal-shift-y", str(sy),
                      "--motion-x", str(mx), "--motion-y", str(my)))
                for name, sx, sy, mx, my, size in groups for f in (1, 3)]
        cases += [case(f"input_{name}_i0", 4, size,
                       ("--scene-variant", "0", "--temporal-shift", str(sx), "--temporal-shift-y", str(sy),
                        "--motion-x", str(mx), "--motion-y", str(my), "--intensity", "0"))
                  for name, sx, sy, mx, my, size in groups if name in ("diagonal", "negative", "odd_diagonal")]
        return cases
    if suite == "temporal":
        def temporal(name, frames=1, variant=0, extra=()):
            return case(name, frames, extra=("--scene-variant", str(variant), *extra))
        return [temporal("fresh_a"), temporal("fresh_b", variant=1), temporal("fresh_b_f1", 2, 1),
                temporal("history_a_f3", 4), temporal("history_b_f3", 4, 1), temporal("history_a_f15", 16),
                temporal("static_reset_f2", 3, extra=("--reset-frame", "2")),
                temporal("cut_f2", 3, extra=("--scene-cut-frame", "2")),
                temporal("cut_f3", 4, extra=("--scene-cut-frame", "2")),
                temporal("cut_reset_f2", 3, extra=("--scene-cut-frame", "2", "--reset-frame", "2")),
                temporal("cut_reset_f3", 4, extra=("--scene-cut-frame", "2", "--reset-frame", "2")),
                temporal("cut_reset_f8", 9, extra=("--scene-cut-frame", "8", "--reset-frame", "8"))]
    if suite not in ("smoke", "core", "full"):
        raise ValueError(f"unknown suite: {suite}")
    cases = [case("reset"), case("history_f3", 4), case("intensity_zero", extra=("--intensity", "0"))]
    if suite in ("core", "full"):
        cases += [case(f"history_f{f - 1}", f) for f in (2, 3)]
        cases += [case(f"motion_f{f - 1}", f, extra=("--temporal-shift", "4", "--motion-x", "-4")) for f in (2, 3, 4)]
        cases += [case("motion_zero_mv", 4, extra=("--temporal-shift", "4",)),
                  case("style1", extra=("--style", "1")), case("style2", extra=("--style", "2")),
                  case("tone_structure", extra=("--local-tone", "0.25", "--local-structure", "0.3")),
                  case("mask_ramp", extra=("--optional", "control-mask", "--optional-format", "rgba16f",
                                           "--optional-variant", "3", "--ramp-channels", "1"))]
    if suite == "full":
        cases += [case(f"resolution_{size}", size=size) for size in ("960x540", "1280x720", "1920x1080", "641x361")]
        cases += [case(f"mask_style{s}", extra=("--style", str(s), "--optional", "control-mask",
                                                "--optional-format", "rgba16f", "--optional-variant", "3")) for s in (1, 2)]
        cases += [case("ui_bundle", extra=("--optional", "ui-bundle", "--bundle", "7",
                                          "--ui-correction", "1", "--optional-value", "0.5")),
                  case("hdr_scrgb", extra=("--profile", "scrgb", "--hdr-scale", "4"))]
    return cases


def case_contract(case):
    args = case["args"]
    def option(key, default):
        return args[args.index(key) + 1] if key in args else default
    width, height = map(int, option("--input", "640x360").split("x"))
    out_width, out_height = map(int, option("--output", "640x360").split("x"))
    profile = option("--profile", "srgb")
    model = int(option("--model", "1"))
    fields = {"featureId": 18, "created": True, "evaluationsSucceeded": int(option("--frames", "1")),
              "input": {"width": width, "height": height}, "output": {"width": out_width, "height": out_height},
              "model": model, "style": int(option("--style", "-1")), "pipeline": "nr-only", "compactNrResources": False,
              "profile": {"srgb": "sRGB", "scrgb": "scRGB", "hdr10": "HDR10/PQ"}[profile],
              "optionalProbe": option("--optional", "none"), "optionalVariant": int(option("--optional-variant", "1")),
              "useAutoMask": int(option("--auto-mask", "1")), "uiCorrection": int(option("--ui-correction", "0")),
              "temporalShiftPixels": int(option("--temporal-shift", "0")),
              "motionVector": [float(option("--motion-x", "0")), float(option("--motion-y", "0"))]}
    if "--optional-format" in args:
        fields["optionalFormat"] = option("--optional-format", "unknown")
    if "--scene-variant" in args:
        fields.update(sceneVariant=int(option("--scene-variant", "0")),
                      sceneCutFrame=int(option("--scene-cut-frame", "-1")),
                      resetFrame=int(option("--reset-frame", "-1")))
    if "--temporal-shift-y" in args:
        fields["temporalShiftYPixels"] = int(option("--temporal-shift-y", "0"))
        fields["intensity"] = float(option("--intensity", "1"))
    if "--mask-blend" in args:
        fields.update(maskBlendModeRequested=int(option("--mask-blend", "-1")),
                      rampChannels=int(option("--ramp-channels", "15")),
                      optionalRgba=list(map(float, option("--optional-rgba", "1,1,1,1").split(","))),
                      localTone=float(option("--local-tone", "1")),
                      localStructure=float(option("--local-structure", "1")),
                      skinStructure=float(option("--skin-structure", "-1")),
                      intensity=float(option("--intensity", "1")))
    names = {f"nr-lab-output-model{model}-{profile}.ppm"}
    if profile != "srgb":
        names.add(f"nr-lab-output-model{model}-raw.f16")
    return fields, names


def validate_report(case, report, output_names):
    expected, expected_names = case_contract(case)
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"case {case['id']} report {key} does not match requested contract")
    if not report.get("outputCoverageValidated", report.get("upscalingValidated", False)):
        raise ValueError("complete output coverage failed")
    if set(output_names) != expected_names:
        raise ValueError(f"case {case['id']} requires output inventory {sorted(expected_names)}")


def validate_temporal_log(case, log):
    """Require the requested per-frame scene/reset events, not just echoed options."""
    if "--scene-variant" not in case["args"]:
        return
    fields, _ = case_contract(case)
    expected = [(f, fields["sceneVariant"] ^ int(fields["sceneCutFrame"] >= 0 and f >= fields["sceneCutFrame"]),
                 int(f == 0 or f == fields["resetFrame"])) for f in range(fields["evaluationsSucceeded"])]
    observed = [tuple(map(int, row)) for row in re.findall(r"temporal input: frame=(\d+) scene=(\d+) reset=(\d+)", log)]
    if observed != expected:
        raise ValueError(f"case {case['id']} temporal input/reset events do not match requested contract")


def validate_mask_log(case, log):
    """Check the requested value actually reached the runtime's parameter getter each frame."""
    if "--mask-blend" not in case["args"]:
        return
    fields, _ = case_contract(case)
    mode, frames = fields["maskBlendModeRequested"], fields["evaluationsSucceeded"]
    events = re.findall(r"PARAM (?:Set I32 DLSSNR\.MaskBlendMode=(\d+)|"
                        r"Get I32 DLSSNR\.MaskBlendMode -> 0x([0-9A-Fa-f]{8}) value=(\d+))", log)
    reads = []
    for setting, result, value in events:
        if setting:
            if int(setting) != mode:
                raise ValueError(f"case {case['id']} mask blend set the wrong value")
            reads.append(0)
        elif reads:
            if int(result, 16) != 1 or int(value) != mode:
                raise ValueError(f"case {case['id']} mask blend getter failed or returned the wrong value")
            reads[-1] += 1
    if len(reads) != frames or not all(reads):
        raise ValueError(f"case {case['id']} mask blend is missing per-frame set/get evidence")


def validate_pixels(path, report):
    width, height = report["output"]["width"], report["output"]["height"]
    if path.suffix == ".ppm":
        if read_ppm(path).shape[:2] != (height, width):
            raise ValueError("PPM dimensions disagree with requested contract")
    elif path.stat().st_size != width * height * 8 or not np.isfinite(np.fromfile(path, dtype="<f2")).all():
        raise ValueError("invalid or non-finite raw RGBA16F output")


def read_ppm(path):
    data = Path(path).read_bytes()
    pos = 0
    tokens = []
    for _ in range(4):
        while pos < len(data):
            if data[pos:pos + 1].isspace():
                pos += 1
            elif data[pos] == 35:
                end = data.find(b"\n", pos)
                if end < 0:
                    raise ValueError(f"unterminated PPM comment: {path}")
                pos = end + 1
            else:
                break
        start = pos
        while pos < len(data) and not data[pos:pos + 1].isspace():
            pos += 1
        tokens.append(data[start:pos])
    if tokens[0] != b"P6" or tokens[3] != b"255" or pos >= len(data):
        raise ValueError(f"expected 8-bit P6 PPM: {path}")
    pos += 2 if data[pos:pos + 2] == b"\r\n" else 1
    width, height = int(tokens[1]), int(tokens[2])
    if width <= 0 or height <= 0 or len(data) - pos != width * height * 3:
        raise ValueError(f"invalid PPM dimensions/payload: {path}")
    return np.frombuffer(data, np.uint8, offset=pos).reshape(height, width, 3)


def metrics(left, right, scale=1.0):
    if left.shape != right.shape:
        raise ValueError(f"different output shapes: {left.shape} vs {right.shape}")
    a, b = left.astype(np.float64).ravel(), right.astype(np.float64).ravel()
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("non-finite output values")
    difference = np.abs(a - b) * scale
    mse = float(np.mean((difference / 255.0) ** 2))
    correlation = None
    if a.std() > 0 and b.std() > 0:
        correlation = float(np.corrcoef(a, b)[0, 1])
    return {"mae_255": float(difference.mean()), "max_255": float(difference.max()),
            "p99_255": float(np.percentile(difference, 99)), "exact_fraction": float(np.mean(a == b)),
            "psnr_db": None if mse == 0 else float(-10 * math.log10(mse)),
            "correlation": correlation, "bit_exact": bool(left.dtype == right.dtype and left.tobytes() == right.tobytes())}


def gpu_inventory():
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,compute_cap,driver_version", "--format=csv,noheader"],
                            capture_output=True, text=True, check=True, timeout=20)
    return [{"name": row[0].strip(), "uuid": row[1].strip(), "compute_capability": row[2].strip(),
             "driver": row[3].strip()} for row in csv.reader(io.StringIO(result.stdout)) if row]


def capture_profile(kind, version, model_preset):
    if kind not in ("official-sm120", "port-sm86", OCTOBER_EXPERIMENT_KIND) or not isinstance(version, str) or version not in NATIVE_SHA256:
        raise ValueError("unknown capture kind or NR version")
    if kind == OCTOBER_EXPERIMENT_KIND and (version != OCTOBER_VERSION or type(model_preset) is not int or model_preset != 2):
        raise ValueError("experimental-sm86 requires 310.9.16.0 and --model-preset 2")
    if version == OCTOBER_VERSION:
        if kind == "port-sm86":
            raise ValueError("no validated sm86 port is registered for 310.9.16.0; use the dedicated experimental profile")
        if type(model_preset) is not int or model_preset not in OCTOBER_PRESETS:
            raise ValueError("310.9.16.0 requires explicit --model-preset 1 (HT) or 2 (Small FP8)")
        return {"nr_version": version, "requested_model_preset": model_preset,
                "statically_mapped_weight_resource": OCTOBER_PRESETS[model_preset]}
    if model_preset is not None:
        raise ValueError("--model-preset is only for the 310.9.16.0 capture profile")
    return {}


def profiled_cases(suite, profile):
    if suite == "october-mask" and (not profile or profile.get("nr_version") != OCTOBER_VERSION):
        raise ValueError("october-mask requires the explicit 310.9.16.0 capture profile")
    cases = suite_cases(suite)
    if not profile:
        return cases
    return [{**case, "args": [*case["args"], "--model", str(profile["requested_model_preset"])]} for case in cases]


def manifest_profile(manifest):
    profile = manifest.get("capture_profile", {})
    if not isinstance(profile, dict):
        raise ValueError("invalid capture profile")
    kind = manifest["reference_kind"]
    version = profile.get("nr_version", "310.8.0")
    expected = capture_profile(kind, version, profile.get("requested_model_preset"))
    if expected != profile or manifest.get("target", f"DLSS-NR {version}") != f"DLSS-NR {version}":
        raise ValueError("inconsistent capture profile or target")
    dll_hash = manifest.get("runtime_sha256", {}).get("nvngx_dlssnr.dll")
    verify_dll_identity(kind, version, dll_hash)
    return profile


def verify_dll_identity(kind, version, dll_hash):
    if kind == "official-sm120" and dll_hash != NATIVE_SHA256[version]:
        raise ValueError(f"official-sm120 requires the known unmodified {version} DLL SHA-256")
    if kind == OCTOBER_EXPERIMENT_KIND and dll_hash != OCTOBER_EXPERIMENT_SHA256:
        raise ValueError("experimental-sm86 requires the pinned October staged DLL SHA-256")
    if version != OCTOBER_VERSION and dll_hash in (OCTOBER_SHA256, OCTOBER_EXPERIMENT_SHA256):
        raise ValueError("the October DLL requires its explicit 310.9.16.0 profile")


def verify_native(gpus, dll_hash, log=None, *, nr_version="310.8.0"):
    if not isinstance(nr_version, str) or nr_version not in NATIVE_SHA256 or dll_hash != NATIVE_SHA256[nr_version]:
        raise ValueError(f"official-sm120 requires the known unmodified {nr_version} DLL SHA-256")
    supported = [gpu for gpu in gpus if gpu["compute_capability"].startswith("12.")]
    if not supported:
        raise ValueError("official-sm120 requires a Blackwell compute capability 12.x GPU; this machine cannot supply that reference")
    verify_gpu_log(supported, log, "Blackwell")


def verify_gpu_log(supported, log, family):
    if log is not None:
        names = re.findall(r"GPU: (.+?) \(vendor=", log)
        if len(names) != 1 or names[0] not in {gpu["name"] for gpu in supported}:
            raise ValueError(f"execution log does not identify the selected {family} GPU")


def verify_execution_gpu(kind, version, gpus, dll_hash, log=None):
    if kind == "official-sm120":
        verify_native(gpus, dll_hash, log, nr_version=version)
    elif kind == OCTOBER_EXPERIMENT_KIND:
        supported = [gpu for gpu in gpus if gpu["compute_capability"] == "8.6"]
        if not supported:
            raise ValueError("experimental-sm86 requires an Ampere compute capability 8.6 GPU")
        verify_gpu_log(supported, log, "Ampere")


def capture(args):
    version = getattr(args, "nr_version", "310.8.0")
    profile = capture_profile(args.kind, version, getattr(args, "model_preset", None))
    cases = profiled_cases(args.suite, profile)
    files = {"nr-lab.exe": args.harness.resolve(), "nvngx.dll": args.bridge.resolve(),
             "nvngx_dlssnr.dll": args.dll.resolve(),
             "nvngx_dlss.dll": (args.support / "nvngx_dlss.dll").resolve(),
             "nvngx_dlssg.dll": (args.support / "nvngx_dlssg.dll").resolve()}
    for source in files.values():
        if not source.is_file():
            raise ValueError(f"missing capture dependency: {source}")
    gpus = gpu_inventory()
    hashes = {name: digest(path) for name, path in files.items()}
    verify_dll_identity(args.kind, version, hashes["nvngx_dlssnr.dll"])
    verify_execution_gpu(args.kind, version, gpus, hashes["nvngx_dlssnr.dll"])
    output = args.out.resolve()
    if output.exists():
        raise ValueError(f"capture output must be a new directory: {output}")
    output.mkdir(parents=True)
    runtime = output / "_runtime"
    runtime.mkdir()
    manifest = {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
                "target": f"DLSS-NR {version}", "reference_kind": args.kind, "suite": args.suite,
                "gpus": gpus, "runtime_sha256": hashes,
                "scope": "finite synthetic D3D12 cases; not a universal fidelity or game certification",
                "cases": [], "passed": False}
    if profile:
        manifest["capture_profile"] = profile
        manifest["scope"] += "; preset is the verified requested option, not an observed weight-load or kernel trace"
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("DLSS5_")} if profile else None
    failed = False
    try:
        for name, source in files.items():
            shutil.copy2(source, runtime / name)
        for case in cases:
            destination = output / case["id"]
            destination.mkdir()
            for name in ("nr-lab-result.json", "nr-lab.log"):
                (runtime / name).unlink(missing_ok=True)
            for pattern in ("nr-lab-output-*.ppm", "nr-lab-output-*-raw.f16"):
                for path in runtime.glob(pattern):
                    path.unlink()
            command = [str(runtime / "nr-lab.exe"), *case["args"]]
            row = {**case, "passed": False, "outputs": {}}
            try:
                result = subprocess.run(command, cwd=runtime, env=env, capture_output=True, timeout=args.timeout)
                (destination / "stdout.log").write_bytes(result.stdout)
                (destination / "stderr.log").write_bytes(result.stderr)
                row["exit_code"] = result.returncode
                for name in ("nr-lab-result.json", "nr-lab.log"):
                    if (runtime / name).exists():
                        shutil.copy2(runtime / name, destination / name)
                if result.returncode != 0:
                    raise ValueError(f"nr-lab exited {result.returncode}")
                report = json.loads((destination / "nr-lab-result.json").read_text(encoding="utf-8"))
                log = (destination / "nr-lab.log").read_text(errors="replace")
                validate_temporal_log(case, log)
                validate_mask_log(case, log)
                verify_execution_gpu(args.kind, version, gpus, hashes["nvngx_dlssnr.dll"], log)
                outputs = list(runtime.glob("nr-lab-output-*.ppm")) + list(runtime.glob("nr-lab-output-*-raw.f16"))
                validate_report(case, report, [path.name for path in outputs])
                for source in outputs:
                    validate_pixels(source, report)
                    target = destination / source.name
                    shutil.copy2(source, target)
                    row["outputs"][source.name] = digest(target)
                if not all((destination / name).is_file() for name in
                           ("nr-lab-result.json", "nr-lab.log", "stdout.log", "stderr.log")):
                    raise ValueError("capture is missing required result/log evidence")
                row["passed"] = True
            except (ValueError, OSError, KeyError, subprocess.TimeoutExpired) as error:
                row["error"] = str(error)
                failed = True
            row["evidence"] = {name: digest(destination / name) for name in
                               ("nr-lab-result.json", "nr-lab.log", "stdout.log", "stderr.log") if (destination / name).is_file()}
            manifest["cases"].append(row)
            print(f"{'PASS' if row['passed'] else 'FAIL'} {case['id']} {row.get('error', '')}", flush=True)
            write_json(output / "manifest.json", manifest)
    finally:
        # Only remove the exact copies made above, never a user-selected directory tree.
        for name in files:
            (runtime / name).unlink(missing_ok=True)
    manifest["passed"] = not failed
    write_json(output / "manifest.json", manifest)
    return 1 if failed else 0


def load_capture(folder):
    folder = folder.resolve()
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or not manifest.get("passed"):
        raise ValueError(f"unsupported or incomplete capture: {folder}")
    profile = manifest_profile(manifest)
    version = profile.get("nr_version", "310.8.0")
    expected = profiled_cases(manifest["suite"], profile)
    if [{"id": row["id"], "args": row["args"]} for row in manifest["cases"]] != expected:
        raise ValueError(f"capture case coverage or definitions changed: {folder}")
    for case in manifest["cases"]:
        if not case["passed"] or not case["outputs"]:
            raise ValueError(f"failed or empty case: {case['id']}")
        if set(case.get("evidence", {})) != {"nr-lab-result.json", "nr-lab.log", "stdout.log", "stderr.log"}:
            raise ValueError(f"missing case evidence: {case['id']}")
        for name, checksum in case["evidence"].items():
            path = (folder / case["id"] / name).resolve()
            if not path.is_relative_to(folder) or digest(path) != checksum:
                raise ValueError(f"case evidence was modified: {path}")
        report = json.loads((folder / case["id"] / "nr-lab-result.json").read_text(encoding="utf-8"))
        validate_report(case, report, case["outputs"])
        log = (folder / case["id"] / "nr-lab.log").read_text(errors="replace")
        validate_temporal_log(case, log)
        validate_mask_log(case, log)
        for name, checksum in case["outputs"].items():
            path = (folder / case["id"] / name).resolve()
            if not path.is_relative_to(folder) or digest(path) != checksum:
                raise ValueError(f"capture output was modified: {path}")
            validate_pixels(path, report)
    if manifest["reference_kind"] in ("official-sm120", OCTOBER_EXPERIMENT_KIND):
        kind = manifest["reference_kind"]
        hashes = manifest["runtime_sha256"]
        verify_execution_gpu(kind, version, manifest["gpus"], hashes["nvngx_dlssnr.dll"])
        for case in manifest["cases"]:
            verify_execution_gpu(kind, version, manifest["gpus"], hashes["nvngx_dlssnr.dll"],
                                 (folder / case["id"] / "nr-lab.log").read_text(errors="replace"))
    return manifest


def compare(args):
    args._report_safe = False
    if any(args.report.resolve().is_relative_to(folder.resolve()) for folder in (args.reference, args.candidate)):
        raise ValueError("comparison report must be outside both capture directories")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args._report_safe = True
    write_json(args.report, {"schema": SCHEMA, "passed": False, "status": "running"})
    reference, candidate = load_capture(args.reference), load_capture(args.candidate)
    if reference["suite"] != candidate["suite"]:
        raise ValueError("capture suites differ")
    if reference.get("capture_profile", {}) != candidate.get("capture_profile", {}):
        raise ValueError("capture versions or requested model presets differ")
    for name in ("nr-lab.exe", "nvngx.dll", "nvngx_dlss.dll", "nvngx_dlssg.dll"):
        if reference["runtime_sha256"][name] != candidate["runtime_sha256"][name]:
            raise ValueError(f"capture dependencies differ: {name}; use the same harness and support DLLs")
    if args.criterion == "approximate" and (args.max_mae is None or args.min_psnr is None):
        raise ValueError("approximate comparisons require explicit --max-mae (8-bit units) and --min-psnr")
    if args.criterion == "approximate" and (not math.isfinite(args.max_mae) or args.max_mae < 0 or
                                           not math.isfinite(args.min_psnr) or args.min_psnr < 0):
        raise ValueError("approximate thresholds must be finite and nonnegative")
    report = {"schema": SCHEMA, "criterion": args.criterion, "official_reference": reference["reference_kind"] == "official-sm120",
              "scope": "only the named synthetic cases and recorded runtime versions; correlation is not a restoration percentage",
              "reference": str(args.reference.resolve()), "candidate": str(args.candidate.resolve()), "cases": []}
    if reference.get("capture_profile"):
        report["capture_profile"] = reference["capture_profile"]
    passed = True
    for left, right in zip(reference["cases"], candidate["cases"]):
        if left["args"] != right["args"] or set(left["outputs"]) != set(right["outputs"]):
            raise ValueError(f"case arguments or output inventory differ: {left['id']}")
        for name in left["outputs"]:
            lp, rp = args.reference / left["id"] / name, args.candidate / right["id"] / name
            if lp.suffix == ".ppm":
                values = metrics(read_ppm(lp), read_ppm(rp))
            else:
                values = metrics(np.fromfile(lp, dtype="<f2"), np.fromfile(rp, dtype="<f2"), 255.0)
            ok = values["bit_exact"] if args.criterion == "bit-exact" else (
                values["mae_255"] <= args.max_mae and (values["psnr_db"] is None or values["psnr_db"] >= args.min_psnr))
            report["cases"].append({"case": left["id"], "output": name, **values, "passed": ok})
            passed &= ok
            print(f"{'PASS' if ok else 'FAIL'} {left['id']}/{name}: MAE={values['mae_255']:.5f}/255 exact={values['exact_fraction']:.5f}")
    report["passed"] = bool(passed)
    write_json(args.report, report)
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--dll", type=Path, required=True)
    cap.add_argument("--out", type=Path, required=True)
    cap.add_argument("--kind", choices=("official-sm120", "port-sm86", OCTOBER_EXPERIMENT_KIND), required=True)
    cap.add_argument("--nr-version", choices=tuple(NATIVE_SHA256), default="310.8.0")
    cap.add_argument("--model-preset", choices=(1, 2), type=int,
                     help="required for 310.9.16.0: 1 uses HT, 2 requests Small FP8")
    cap.add_argument("--suite", choices=("smoke", "core", "full", "temporal", "motion2d", "october-mask"), default="core")
    cap.add_argument("--harness", type=Path, default=ROOT / "harness" / "nr-lab.exe")
    cap.add_argument("--bridge", type=Path, default=ROOT / "harness" / "nvngx.dll")
    cap.add_argument("--support", type=Path, default=ROOT / "dlss5")
    cap.add_argument("--timeout", type=float, default=90)
    cmp = sub.add_parser("compare")
    cmp.add_argument("--reference", type=Path, required=True)
    cmp.add_argument("--candidate", type=Path, required=True)
    cmp.add_argument("--report", type=Path, required=True)
    cmp.add_argument("--criterion", choices=("bit-exact", "approximate"), default="bit-exact")
    cmp.add_argument("--max-mae", type=float)
    cmp.add_argument("--min-psnr", type=float)
    args = parser.parse_args()
    try:
        return capture(args) if args.command == "capture" else compare(args)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        if args.command == "compare" and getattr(args, "_report_safe", False):
            write_json(args.report, {"schema": SCHEMA, "passed": False, "status": "failed", "error": str(error)})
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
