"""Fresh, versioned 640x360 schedule references; never search legacy tap folders.

python -m tools.schedule_refs plan --report research/batch3/schedule_plan.json
python -m tools.schedule_refs capture --out research/batch3/schedule_capture --harness harness/batch3/nr-lab.exe
python -m tools.schedule_refs validate --capture research/batch3/schedule_capture
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 1
ORIGINAL_SHA256 = "e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e"
PORT_SHA256 = "d442aaf801e5bb5ae1973b494995e9838f0a1dba8039cc79ab91dc9b6ea94ecd"
MAX_TAP_BYTES = 1_966_080
SCHEDULE = ROOT / "torch/dlss5/data/schedule.json"
EXEC_ORDER = ROOT / "torch/dlss5/data/exec_order.json"
COMMAND = ["--nr-only", "--model", "1", "--input", "640x360", "--output", "640x360", "--frames", "1"]
OUTPUT = "nr-lab-output-model1-srgb.ppm"
SOURCE_FILES = ["harness/nr-lab.cpp", "harness/nvprof.h", "harness/tap.ptx",
                "torch/dlss5/data/schedule.json", "torch/dlss5/data/exec_order.json", "weights_map.json"]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def safe_path(folder, relative):
    folder = Path(folder).resolve()
    path = (folder / relative).resolve()
    if not path.is_relative_to(folder):
        raise ValueError(f"Artifact path escapes capture: {relative}")
    return path


def plan():
    schedule = json.loads(SCHEDULE.read_text(encoding="utf-8"))["steps"]
    launches = json.loads(EXEC_ORDER.read_text(encoding="utf-8"))["launches"]
    by_weight = {name: row["seq"] for row in launches for name in row["weights"]}
    buffers, steps, skips = {}, [], {}

    def buffer(name, seq, offset, layout, channels, height, width, role):
        if name in buffers:
            raise ValueError(f"Duplicate buffer {name}")
        buffers[name] = {"id": name, "seq": seq, "parameter_offset": offset,
                         "kernel": launches[seq]["kernel"], "layout": layout,
                         "shape_hwc": [height, width, channels], "bytes": channels * height * width,
                         "dtype": "float8_e4m3fn", "role": role}
        return name

    previous = None
    for index, step in enumerate(schedule):
        kind = step["kind"]
        names = step.get("records", [step.get("record")])
        seqs = [by_weight[name] for name in names]
        if "head_record" in step:
            seqs.append(by_weight[step["head_record"]])
        entry = {"step": index, "definition": step, "launches": seqs, "inputs": {}, "outputs": {}}
        prefix = f"s{index:02d}"
        if kind == "pre":
            entry["inputs"] = {"color": "deterministic_srgb_color", "history": None, "motion": "zero"}
            previous = buffer(prefix + "_out", 1, 248, "image16", 32, 192, 320, "downsample output")
            skips["pre"] = buffer(prefix + "_skip", 1, 216, "tin2d", 32, 384, 640, "full-resolution pre skip")
            entry["outputs"] = {"output": previous, "skip": skips["pre"]}
        elif kind == "swin":
            c, h, w = {"1h": (32, 192, 320), "2h": (64, 96, 160),
                       "4h": (128, 48, 80), "8h": (256, 24, 40)}[step["level"]]
            variant, seq = step["variant"], seqs[0]
            entry["inputs"]["input"] = previous
            if variant == "up":
                entry["inputs"]["skip"] = skips[step["level"]]
            layout = "image16" if variant == "outview" else "tin2d"
            result = buffer(prefix + "_out", seq, 8, layout, c, h, w, "swin output")
            entry["outputs"]["output"] = result
            if variant == "ds":
                skips[step["level"]] = result
                previous = buffer(prefix + "_down", seq, 64 if c == 32 else 72,
                                  "image16", 2 * c, h // 2, w // 2, "downsample output")
                entry["outputs"]["downsample"] = previous
            else:
                previous = result
        elif kind == "split16":
            entry["inputs"]["input"] = previous
            if step["tail"] == "pool":
                skips["16h"] = buffer(prefix + "_skip", seqs[-2], 16, "tin2d", 512, 12, 20, "encoder 16h skip")
                previous = buffer(prefix + "_out", seqs[-1], 8, "tin2d", 1024, 8, 12, "pooled and lifted head output")
                entry["outputs"] = {"output": previous, "skip": skips["16h"]}
            else:
                previous = buffer(prefix + "_out", seqs[-1], 16,
                                  "image16" if step["tail"] == "outview" else "tin2d", 512, 12, 20, "16h output")
                entry["outputs"]["output"] = previous
        elif kind == "vit":
            if index == 31:
                previous = buffer("repack57", 57, 8, "tin1d", 1024, 8, 12, "2d-to-1d repack")
            entry["inputs"]["input"] = previous
            previous = buffer(prefix + "_out", seqs[-1], 16, "tin1d", 1024, 8, 12, "ViT output")
            entry["outputs"]["output"] = previous
        elif kind == "block39":
            previous = buffer("repack98", 98, 8, "tin2d", 1024, 8, 12, "1d-to-2d repack")
            entry["inputs"] = {"input": previous, "skip": skips["16h"]}
            previous = buffer(prefix + "_out", seqs[0], 16, "tin2d", 512, 12, 20, "decoder 16h input")
            entry["outputs"]["output"] = previous
        elif kind == "post":
            entry["inputs"] = {"input": previous, "skip": skips["pre"], "color": "deterministic_srgb_color"}
            entry["outputs"] = {"output": "final_srgb8"}
            entry["limitation"] = "Output is the final sRGB8 copy of the post surface, not its internal floating-point surface."
        else:
            raise ValueError(f"Unsupported schedule kind: {kind}")
        steps.append(entry)
    if len(steps) != 71:
        raise ValueError("Unexpected schedule length")
    return {"schema": SCHEMA, "scope": "640x360 reset frame, default controls, exact named boundaries of 71 schedule steps",
            "schedule_sha256": digest(SCHEDULE), "exec_order_sha256": digest(EXEC_ORDER),
            "command": COMMAND, "steps": steps, "buffers": buffers,
            "bridges": [["s30_out", "repack57"], ["s38_out", "repack98"]],
            "post_boundary": "final_srgb8; not raw floating-point post surface"}


def trace_rows(path):
    result = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        cells = line.split("\t")
        if len(cells) != 8:
            raise ValueError("Expected single-frame 8-column profiler trace")
        seq = int(cells[0])
        if seq in result:
            raise ValueError(f"Duplicate trace sequence {seq}")
        data = bytes.fromhex(cells[7])
        if len(data) != int(cells[6]):
            raise ValueError(f"Parameter size mismatch at seq {seq}")
        result[seq] = {"kernel": cells[2], "grid": cells[3], "block": cells[4],
                       "smem": int(cells[5]), "params": data}
    expected = json.loads(EXEC_ORDER.read_text(encoding="utf-8"))["launches"]
    for row in expected:
        found = result.get(row["seq"])
        if found is None or any(found[key] != row[key] for key in ("kernel", "grid", "block", "smem")):
            raise ValueError(f"Launch identity mismatch at seq {row['seq']}")
        if len(found["params"]) != row["param_bytes"]:
            raise ValueError(f"Launch parameter size mismatch at seq {row['seq']}")
    return result


def chunks_for_plan(reference_plan):
    chunks = []
    for name, desc in reference_plan["buffers"].items():
        for offset in range(0, desc["bytes"], MAX_TAP_BYTES):
            chunks.append({"buffer": name, "seq": desc["seq"], "parameter_offset": desc["parameter_offset"],
                           "offset": offset, "bytes": min(MAX_TAP_BYTES, desc["bytes"] - offset)})
    jobs, current, used = [], [], 0
    # A filename is keyed by seq/address, so never combine two chunks from one buffer in a job.
    for chunk in chunks:
        if current and (used + chunk["bytes"] > MAX_TAP_BYTES or any(c["buffer"] == chunk["buffer"] for c in current)):
            jobs.append(current)
            current, used = [], 0
        current.append(chunk)
        used += chunk["bytes"]
    if current:
        jobs.append(current)
    return jobs


def validate_bindings(rows, reference_plan):
    """Confirm each declared dependency and weight identity in this run's own trace."""
    def pointer(seq, offset):
        return struct.unpack_from("<Q", rows[seq]["params"], offset)[0]
    def produced(name):
        descriptor = reference_plan["buffers"][name]
        return pointer(descriptor["seq"], descriptor["parameter_offset"])
    if struct.unpack_from("<I", rows[1]["params"], 200)[0] != 0:
        raise ValueError("Reference is not reset frame zero")
    for entry in reference_plan["steps"]:
        definition, inputs = entry["definition"], entry["inputs"]
        first = entry["launches"][0]
        if "input" in inputs and pointer(first, 0) != produced(inputs["input"]):
            raise ValueError(f"Input dependency pointer mismatch: step {entry['step']}")
        if "skip" in inputs:
            offset = 80 if definition["kind"] == "swin" and definition["level"] == "1h" else (24 if definition["kind"] == "swin" else 8)
            if pointer(first, offset) != produced(inputs["skip"]):
                raise ValueError(f"Skip dependency pointer mismatch: step {entry['step']}")
    records = sorted(json.loads((ROOT / "weights_map.json").read_text()), key=lambda record: record["name"])
    base, offset, addresses = pointer(1, 224), 0, {}
    for record in records:
        addresses[base + offset] = record["name"]
        offset = (offset + record["C"] + 511) // 512 * 512
    launches = json.loads(EXEC_ORDER.read_text())["launches"]
    for launch in launches:
        params = rows[launch["seq"]]["params"]
        weights = {addresses[value] for value in struct.unpack("<" + "Q" * (len(params) // 8), params[:len(params) // 8 * 8]) if value in addresses}
        if weights != set(launch["weights"]):
            raise ValueError(f"Weight pointer/record identity mismatch: seq {launch['seq']}")


def runtime_files(harness):
    return {"nr-lab.exe": Path(harness), "nvngx.dll": ROOT / "harness/rt_sm86/nvngx.dll",
            "nvngx_dlssnr.dll": ROOT / "sm86_port/nvngx_dlssnr.dll", "nvngx_dlss.dll": ROOT / "dlss5/nvngx_dlss.dll",
            "nvngx_dlssg.dll": ROOT / "dlss5/nvngx_dlssg.dll", "tap.cubin": ROOT / "harness/tap.cubin"}


def recorded_runtime(folder, harness=None):
    folder = Path(folder).resolve()
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    names = set(runtime_files(harness or folder / "runtime/nr-lab.exe"))
    if set(manifest.get("runtime_sha256", {})) != names:
        raise ValueError("Missing or unexpected runtime identity")
    actual = {name: digest(safe_path(folder, "runtime/" + name)) for name in names}
    if harness is not None and digest(harness) != actual["nr-lab.exe"]:
        raise ValueError("Explicit harness identity differs from the capture")
    return actual


def validate_execution(status, output_path):
    from harness.reference import validate_pixels, validate_report
    validate_report({"id": "schedule-reset", "args": COMMAND}, status, [OUTPUT])
    validate_pixels(output_path, status)


def color_rgb8():
    """The fixed frame-zero sRGB MakeColorPattern input, including UNORM rounding."""
    import numpy as np
    y, x = np.mgrid[:360, :640]
    checker = ((x // 12) ^ (y // 12)) & 1
    line = (x % 61 < 2) | (y % 47 < 2) | ((x + y) % 79 < 2)
    values = np.stack([np.where(line, 1., np.where(checker, .82, .06)),
                       np.where(line, .18, np.where(checker, .11, .68)),
                       np.where(line, .04, np.where(checker, .55, .09))], -1)
    return np.floor(values * 255 + .5).astype(np.uint8).tobytes()


def capture(folder, harness, expected_port_sha256=PORT_SHA256):
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    reference_plan = plan()
    manifest = {"schema": SCHEMA, "kind": "schedule-reference-sm86", "created_utc": datetime.now(timezone.utc).isoformat(),
                "passed": False, "errors": [], "plan": reference_plan, "runs": [], "buffers": {},
                "source_sha256": {p: digest(ROOT / p) for p in SOURCE_FILES},
                "original_dll_sha256": digest(ROOT / "dlss5/nvngx_dlssnr.dll"),
                "weights_sha256": digest(ROOT / "WEIGHTS_HT.bin"), "runtime_sha256": {},
                "capture_tool_sha256": digest(__file__), "reference_kind": "port-sm86", "official_reference": False}
    save(folder / "manifest.json", manifest)
    try:
        manifest["expected_port_sha256"] = expected_port_sha256
        manifest["source_snapshots"] = {}
        for relative, source_hash in {**manifest["source_sha256"], "tools/schedule_refs.py": manifest["capture_tool_sha256"]}.items():
            path = folder / "sources" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, path)
            if digest(path) != source_hash:
                raise ValueError(f"Source changed while freezing capture: {relative}")
            manifest["source_snapshots"][relative] = {"path": str(path.relative_to(folder)), "sha256": source_hash}
        color_path = folder / "color_rgb8.bin"
        color_path.write_bytes(color_rgb8())
        manifest["input"] = {"path": color_path.name, "bytes": color_path.stat().st_size,
                             "sha256": digest(color_path), "shape_hwc": [360, 640, 3],
                             "scope": "Generated from the pinned harness frame-zero pattern contract; not a texture readback"}
        if manifest["original_dll_sha256"] != ORIGINAL_SHA256:
            raise ValueError("Original DLL identity mismatch")
        runtime = folder / "runtime"
        runtime.mkdir()
        for name, source in runtime_files(harness).items():
            shutil.copy2(source, runtime / name)
            manifest["runtime_sha256"][name] = digest(source)
        if manifest["runtime_sha256"]["nvngx_dlssnr.dll"] != expected_port_sha256:
            raise ValueError("This reference contract requires the identified default f16 sm86 port")
        manifest["gpu_inventory"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,uuid,compute_cap,driver_version", "--format=csv,noheader"],
            check=True, capture_output=True, text=True).stdout.strip()
        part_paths = {name: [] for name in reference_plan["buffers"]}
        baseline_output = None
        for index, requests in enumerate(chunks_for_plan(reference_plan)):
            destination = folder / f"run{index:03d}"
            destination.mkdir()
            for old in runtime.iterdir():
                if old.name not in manifest["runtime_sha256"]:
                    if old.is_file():
                        old.unlink()
            spec = "0" + "".join(f";{r['seq']}:p{r['parameter_offset'] // 8}+{r['offset']}:{r['bytes']}" for r in requests)
            env = dict(os.environ, DLSS5_PROF="1", DLSS5_TRACE="0", DLSS5_TAP=spec)
            command = [str(runtime / "nr-lab.exe"), *COMMAND]
            with (destination / "stdout.txt").open("wb") as out, (destination / "stderr.txt").open("wb") as err:
                process = subprocess.run(command, cwd=runtime, env=env, stdout=out, stderr=err, timeout=300)
            for path in runtime.iterdir():
                if path.is_file() and path.name not in manifest["runtime_sha256"]:
                    shutil.copy2(path, destination / path.name)
            row = {"id": destination.name, "command": command, "environment": {k: env[k] for k in ("DLSS5_PROF", "DLSS5_TRACE", "DLSS5_TAP")},
                   "returncode": process.returncode, "passed": False, "requests": requests,
                   "artifacts": {p.name: {"path": str(p.relative_to(folder)), "bytes": p.stat().st_size, "sha256": digest(p)}
                                 for p in destination.iterdir() if p.is_file()}}
            manifest["runs"].append(row)
            save(folder / "manifest.json", manifest)
            status = json.loads((destination / "nr-lab-result.json").read_text(encoding="utf-8"))
            if process.returncode or not status.get("created") or status.get("evaluationsSucceeded") != 1 or not status.get("outputCoverageValidated"):
                raise ValueError(f"Execution failed in {destination.name}")
            validate_execution(status, destination / OUTPUT)
            rows = trace_rows(destination / "nr-trace.tsv")
            validate_bindings(rows, reference_plan)
            output_sha = digest(destination / OUTPUT)
            baseline_output = baseline_output or output_sha
            if output_sha != baseline_output:
                raise ValueError("Repeated reset captures produced different final outputs")
            text = (destination / "stdout.txt").read_text(errors="replace")
            outcomes = re.findall(r"\[tap\] frame 0 seq (\d+) 0x([0-9a-fA-F]+) \+(\d+) -> off \d+ \(launch=(-?\d+)\)", text)
            if len(outcomes) != len(requests) or any(int(v[3]) for v in outcomes):
                raise ValueError(f"Missing/failed tap launch in {destination.name}")
            for request in requests:
                params = rows[request["seq"]]["params"]
                pointer = struct.unpack_from("<Q", params, request["parameter_offset"])[0]
                address = pointer + request["offset"]
                name = f"tap_s{request['seq']}_{address:x}.bin"
                path = destination / name
                if not pointer or not path.is_file() or path.stat().st_size != request["bytes"]:
                    raise ValueError(f"Missing/exact-size/address mismatch for {name}; requires pN+offset tap support")
                data = path.read_bytes()
                if not any(data):
                    raise ValueError(f"All-zero reference refused: {name}")
                request.update(path=str(path.relative_to(folder)), sha256=digest(path), address=hex(address), base_pointer=hex(pointer))
                part_paths[request["buffer"]].append(dict(request))
            row["passed"] = True
            save(folder / "manifest.json", manifest)
            print(f"{destination.name}: {len(requests)} exact taps, output SHA consistent", flush=True)
        buffers_dir = folder / "buffers"
        buffers_dir.mkdir()
        for name, parts in part_paths.items():
            path = buffers_dir / f"{name}.bin"
            parts.sort(key=lambda p: p["offset"])
            path.write_bytes(b"".join(safe_path(folder, p["path"]).read_bytes() for p in parts))
            if path.stat().st_size != reference_plan["buffers"][name]["bytes"]:
                raise ValueError(f"Assembled size mismatch: {name}")
            manifest["buffers"][name] = {"path": str(path.relative_to(folder)), "bytes": path.stat().st_size,
                                          "sha256": digest(path), "chunks": parts}
        manifest["final_output"] = manifest["runs"][0]["artifacts"][OUTPUT]
        manifest["passed"] = True
    except Exception as error:
        manifest["errors"].append(f"{type(error).__name__}: {error}")
        manifest["passed"] = False
    save(folder / "manifest.json", manifest)
    return manifest


def validate(folder, *, expected_runtime=None, verify_sources=True):
    folder = Path(folder).resolve()
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    expected_plan = plan()
    if manifest.get("schema") != SCHEMA or manifest.get("kind") != "schedule-reference-sm86" or not manifest.get("passed") or manifest.get("errors"):
        raise ValueError("Incomplete or wrong-kind reference capture")
    if manifest.get("plan") != expected_plan:
        raise ValueError("Schedule/exec-order/plan identity mismatch")
    if manifest.get("original_dll_sha256") != ORIGINAL_SHA256 or manifest.get("runtime_sha256", {}).get("nvngx_dlssnr.dll") != manifest.get("expected_port_sha256", PORT_SHA256):
        raise ValueError("DLL identity mismatch")
    if manifest.get("official_reference") or manifest.get("reference_kind") != "port-sm86":
        raise ValueError("Invalid native reference claim")
    expected_runtime = recorded_runtime(folder) if expected_runtime is None else expected_runtime
    if manifest["runtime_sha256"] != expected_runtime:
        raise ValueError("Harness/bridge/runtime identity mismatch")
    if verify_sources:
        required = {**manifest.get("source_sha256", {}), "tools/schedule_refs.py": manifest.get("capture_tool_sha256")}
        if set(manifest.get("source_sha256", {})) != set(SOURCE_FILES):
            raise ValueError("Missing capture source identities")
        for relative, expected_hash in required.items():
            snapshot = manifest.get("source_snapshots", {}).get(relative)
            if (not snapshot or snapshot.get("sha256") != expected_hash or
                    digest(safe_path(folder, snapshot["path"])) != expected_hash):
                raise ValueError(f"Capture source snapshot identity mismatch: {relative}")
        if manifest.get("weights_sha256") != digest(ROOT / "WEIGHTS_HT.bin"):
            raise ValueError("Weight identity mismatch")
    input_entry = manifest.get("input", {})
    input_path = safe_path(folder, input_entry.get("path", "missing-input"))
    if (not input_path.is_file() or input_path.read_bytes() != color_rgb8() or
            input_entry.get("bytes") != 640 * 360 * 3 or digest(input_path) != input_entry.get("sha256")):
        raise ValueError("Missing or wrong deterministic input")
    expected_jobs = chunks_for_plan(expected_plan)
    if len(manifest.get("runs", [])) != len(expected_jobs):
        raise ValueError("Missing capture run")
    all_chunks = {name: [] for name in expected_plan["buffers"]}
    output_hash = None
    for run, requests in zip(manifest["runs"], expected_jobs):
        if not run.get("passed") or run.get("returncode") or run.get("command", [])[1:] != COMMAND:
            raise ValueError("Invalid capture execution")
        for entry in run["artifacts"].values():
            path = safe_path(folder, entry["path"])
            if not path.is_file() or path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
                raise ValueError(f"Missing/size/hash mismatch: {entry['path']}")
        if [{k: r[k] for k in request} for r, request in zip(run.get("requests", []), requests)] != requests or len(run.get("requests", [])) != len(requests):
            raise ValueError("Tap request plan mismatch")
        rows = trace_rows(safe_path(folder, run["artifacts"]["nr-trace.tsv"]["path"]))
        validate_bindings(rows, expected_plan)
        status = json.loads(safe_path(folder, run["artifacts"]["nr-lab-result.json"]["path"]).read_text())
        if not status.get("created") or status.get("evaluationsSucceeded") != 1 or not status.get("outputCoverageValidated"):
            raise ValueError("Execution result does not meet the capture contract")
        validate_execution(status, safe_path(folder, run["artifacts"][OUTPUT]["path"]))
        stdout = safe_path(folder, run["artifacts"]["stdout.txt"]["path"]).read_text(errors="replace")
        outcomes = re.findall(r"\[tap\] frame 0 seq (\d+) 0x([0-9a-fA-F]+) \+(\d+) -> off \d+ \(launch=(-?\d+)\)", stdout)
        if len(outcomes) != len(requests) or any(int(v[3]) for v in outcomes):
            raise ValueError("Missing/failed tap launches in capture log")
        current_output = run["artifacts"][OUTPUT]["sha256"]
        output_hash = output_hash or current_output
        if output_hash != current_output:
            raise ValueError("Inconsistent repeated reset output")
        for request in run["requests"]:
            base = struct.unpack_from("<Q", rows[request["seq"]]["params"], request["parameter_offset"])[0]
            if not base or request["base_pointer"] != hex(base) or request["address"] != hex(base + request["offset"]):
                raise ValueError("Tap/trace pointer mismatch")
            expected_name = f"tap_s{request['seq']}_{base + request['offset']:x}.bin"
            artifact = run["artifacts"].get(expected_name)
            if artifact is None or artifact["path"] != request["path"] or artifact["sha256"] != request["sha256"] or artifact["bytes"] != request["bytes"]:
                raise ValueError("Tap chunk/artifact identity mismatch")
            all_chunks[request["buffer"]].append(request)
    if set(manifest.get("buffers", {})) != set(expected_plan["buffers"]):
        raise ValueError("Missing or unexpected named buffer")
    for name, desc in expected_plan["buffers"].items():
        entry = manifest["buffers"][name]
        if entry.get("chunks") != sorted(all_chunks[name], key=lambda p: p["offset"]):
            raise ValueError(f"Missing/reordered/unbound chunks: {name}")
        path = safe_path(folder, entry["path"])
        if not path.is_file() or path.stat().st_size != desc["bytes"] or entry["bytes"] != desc["bytes"] or digest(path) != entry["sha256"]:
            raise ValueError(f"Missing/exact-size/hash mismatch: {name}")
        joined = b"".join(safe_path(folder, p["path"]).read_bytes() for p in entry["chunks"])
        if path.read_bytes() != joined:
            raise ValueError(f"Chunk assembly mismatch: {name}")
    if manifest.get("final_output") != manifest["runs"][0]["artifacts"][OUTPUT]:
        raise ValueError("Unbound final output")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--report", type=Path, required=True)
    p = sub.add_parser("capture")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--harness", type=Path, required=True)
    p.add_argument("--expected-port-sha256", default=PORT_SHA256,
                   help="Explicit identity of an alternate port; default pins the existing f16 port")
    p = sub.add_parser("validate")
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--harness", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "plan":
            value = plan()
            save(args.report, value)
            print(f"71 steps, {len(value['buffers'])} named buffers, {len(chunks_for_plan(value))} isolated tap batches")
            return 0
        if args.command == "capture":
            return 0 if capture(args.out, args.harness, args.expected_port_sha256)["passed"] else 1
        validate(args.capture, expected_runtime=recorded_runtime(args.capture, args.harness))
        print("Reference identity, exact lengths and hashes verified")
        return 0
    except Exception as error:
        print(f"FAIL: {type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
