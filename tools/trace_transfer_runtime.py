"""Observe normal NGX postprocess parameter blocks through the existing local profiler.

python -m tools.trace_transfer_runtime --out research/transfer_host_runtime_batch2
Copies the private runtime into a new independent directory; no DLL is patched.
Large private artifacts stay in that local directory, while report.json contains
only provenance, execution status, and the normalized transfer fields.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ("native_style1", "640x360", "640x360", 1, ["--style", "1"]),
    ("native_style2", "640x360", "640x360", 1, ["--style", "2"]),
    ("different_output_style1", "640x360", "960x540", 1, ["--style", "1"]),
    ("runtime_half_style1", "640x360", "640x360", 2, ["--style", "1", "--runtime-scale", "0.5"]),
]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def postprocess_blocks(path):
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        columns = line.split("\t")
        if len(columns) < 8 or columns[2] != "cg2r_post_process_kernel":
            continue
        block = bytes.fromhex(columns[7])
        if len(block) != 376:
            raise ValueError(f"Unexpected postprocess block size: {len(block)}")
        descriptor = lambda offset: dict(zip(("handle", "base_x", "base_y", "width", "height", "inverse_width", "inverse_height"),
                                               struct.unpack_from("<Q6f", block, offset)))
        a, b = descriptor(224), descriptor(256)
        enabled, width, height = struct.unpack_from("<iii", block, 300)
        rows.append({"sequence": columns[0], "parameter_sha256": hashlib.sha256(block).hexdigest(),
                     "parameter_size": len(block), "texture_224": a, "texture_256": b,
                     "enable_300": enabled, "grid_width_304": width, "grid_height_308": height,
                     "output_size": list(struct.unpack_from("<ii", block, 292)),
                     "transfer_predicate": enabled != 0 and a["handle"] != 0 and b["handle"] != 0 and width > 0 and height > 0})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runtime = output / "runtime"
    runtime.mkdir()
    sources = {"nr-lab.exe": ROOT / "harness/nr-lab.exe", "nvngx.dll": ROOT / "harness/rt_sm86/nvngx.dll",
               "nvngx_dlssnr.dll": ROOT / "sm86_port/nvngx_dlssnr.dll",
               "nvngx_dlss.dll": ROOT / "dlss5/nvngx_dlss.dll", "nvngx_dlssg.dll": ROOT / "dlss5/nvngx_dlssg.dll"}
    report = {"schema": 1, "scope": "normal NGX synthetic evaluations on the RTX 3060 sm86 port; no forced transfer parameters",
              "tool_sha256": sha(__file__), "runtime_sources": {}, "cases": [],
              "static_evidence": "research/validation/transfer_host_static_batch2.json",
              "errors": [], "observation_complete": False}
    report_path = output / "report.json"
    def save():
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    save()
    try:
        for name, source in sources.items():
            shutil.copy2(source, runtime / name)
            report["runtime_sources"][name] = {"path": str(source.resolve()), "sha256": sha(source)}
        env = dict(os.environ, DLSS5_PROF="1", DLSS5_TRACE="0")
        for name, input_size, output_size, frames, extras in CASES:
            case_dir = output / name
            case_dir.mkdir()
            for filename in ("nr-lab-result.json", "nr-trace.tsv"):
                (runtime / filename).unlink(missing_ok=True)
            command = [str(runtime / "nr-lab.exe"), "--nr-only", "--model", "1", "--input", input_size,
                       "--output", output_size, "--frames", str(frames), *extras]
            with (case_dir / "stdout.txt").open("wb") as stdout, (case_dir / "stderr.txt").open("wb") as stderr:
                process = subprocess.run(command, cwd=runtime, env=env, stdout=stdout, stderr=stderr, timeout=300)
            case = {"name": name, "command": command, "returncode": process.returncode,
                    "requested_frames": frames, "execution_passed": False, "postprocess_trace_observed": False,
                    "artifacts": {}, "postprocess_blocks": []}
            for filename in ("nr-lab-result.json", "nr-trace.tsv", "nr-lab.log"):
                source = runtime / filename
                if source.is_file():
                    shutil.copy2(source, case_dir / filename)
            for path in case_dir.iterdir():
                case["artifacts"][path.name] = {"path": str(path), "sha256": sha(path)}
            result_path = case_dir / "nr-lab-result.json"
            if result_path.is_file():
                result = json.loads(result_path.read_bytes())
                case["runtime_result"] = result
                case["execution_passed"] = (process.returncode == 0 and bool(result.get("created")) and
                                                result.get("evaluationsSucceeded") == frames and
                                                bool(result.get("outputCoverageValidated", result.get("upscalingValidated", False))))
            trace_path = case_dir / "nr-trace.tsv"
            if trace_path.is_file():
                case["postprocess_blocks"] = postprocess_blocks(trace_path)
                case["postprocess_trace_observed"] = bool(case["postprocess_blocks"])
            report["cases"].append(case)
            save()
            print(f"{name}: rc={process.returncode}, execution_passed={case['execution_passed']}, postprocess_blocks={len(case['postprocess_blocks'])}", flush=True)
        blocks = [block for case in report["cases"] for block in case["postprocess_blocks"]]
        report["observation_complete"] = all(case["postprocess_trace_observed"] for case in report["cases"])
        report["all_requested_executions_passed"] = all(case["execution_passed"] for case in report["cases"])
        report["observed_block_count"] = len(blocks)
        report["observed_transfer_enabled_count"] = sum(block["transfer_predicate"] for block in blocks)
        report["conclusion"] = ("All observed normal postprocess blocks disabled transfer; this is limited to the recorded synthetic cases and runtime identity."
                                if blocks and not report["observed_transfer_enabled_count"] else
                                "At least one transfer block was enabled, or no block was observed; inspect per-case evidence.")
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    save()
    return 0 if report["observation_complete"] and not report["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
