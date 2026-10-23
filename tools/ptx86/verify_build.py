"""CPU-only checks for DLL identity and independently recorded full/cache build evidence.

Report paths are validated before any report write. Evidence validation binds every
row and stage to current source/tool/input hashes; it does not attest official GPU parity.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import struct

import pefile

from tools.dllq.index import parse_pe, parse_resources
from tools.verify_assets import DLL_SHA256, WEIGHTS_SHA256
from . import build_dll as build

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OPTIONS = {"mode": "f16", "enc": "v1", "dec": "v1", "maxnreg": 0, "maxnreg_floor": 0}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def weights(data):
    sections, resource_rva = parse_pe(data)
    matches = [row for row in parse_resources(data, sections, resource_rva)
               if row["rtype"] == "10" and row["rname"] == "WEIGHTS_HT" and row["rlang"] == "1033"]
    if len(matches) != 1:
        raise ValueError("expected exactly one WEIGHTS_HT/1033 resource")
    row = matches[0]
    payload = data[row["off"]:row["off"] + row["size"]]
    if digest(payload) != WEIGHTS_SHA256:
        raise ValueError("candidate weights are not the supported unmodified resource")
    return {"sha256": digest(payload), "bytes": len(payload)}


def report_artifacts(path):
    """Read only path declarations for the pre-write alias guard; malformed data fails later."""
    if path is None:
        return []
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, dict):
        return []
    paths = []
    if isinstance(data.get("work"), str):
        paths.extend(build.cache_paths(data["work"]))
    if isinstance(data.get("output"), str):
        paths.append(Path(data["output"]))
    for row in data.get("fatbins", []) if isinstance(data.get("fatbins"), list) else []:
        if isinstance(row, dict) and isinstance(row.get("outputs"), dict):
            for output in row["outputs"].values():
                if isinstance(output, dict) and isinstance(output.get("path"), str):
                    paths.append(Path(output["path"]))
    return paths


def validate_selection(candidate, second, first_report, second_report):
    if second is not None and build.paths_alias(candidate, second):
        raise ValueError("second DLL must be a distinct file, not an alias of candidate")
    if bool(first_report) != bool(second_report):
        raise ValueError("both first and second build reports are required together")
    if first_report:
        if second is None:
            raise ValueError("report verification requires an independent second DLL")
        if build.paths_alias(first_report, second_report):
            raise ValueError("first and second reports must be independent files")


def validate_report_path(args):
    report = Path(args.report).resolve() if args.report else None
    protected = [args.original, args.candidate, args.second, args.baseline, args.first_report, args.second_report,
                 *build.input_paths(), *report_artifacts(args.first_report), *report_artifacts(args.second_report)]
    protected = [p for p in protected if p is not None]
    if report:
        build.require_disjoint([report, report.with_suffix(report.suffix + ".tmp")], protected)
    return report, protected


def finite_seconds(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def validate_stages(run, index):
    expected = ["full_dll_build"]
    for row in run["fatbins"]:
        expected.append(row["tag"] + "/total")
        if type(row.get("extracted_original")) is not bool:
            raise ValueError("missing original-cubin extraction decision")
        if row["extracted_original"]:
            if index == 1:
                raise ValueError("cached second build must not re-extract original cubins")
            expected.append(row["tag"] + "/extract_original")
        if index == 0:
            expected.extend(row["tag"] + "/" + name for name in ("rewrite", "ptxas", "fatbinary"))
    stages = run.get("stages")
    if not isinstance(stages, list) or [row.get("name") for row in stages] != expected:
        raise ValueError("missing, duplicate, unexpected or out-of-order build stages")
    if not finite_seconds(run.get("wall_seconds")):
        raise ValueError("invalid total build wall time")
    for row in stages:
        if row.get("status") != "passed" or not finite_seconds(row.get("wall_seconds")):
            raise ValueError("build evidence contains a failed or unfinished stage")
        try:
            started = datetime.fromisoformat(row["started_utc"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid stage start timestamp") from error
        if started.utcoffset() is None:
            raise ValueError("stage timestamp must include a timezone")
    by_name = {row["name"]: row for row in stages}
    total = by_name["full_dll_build"]["wall_seconds"]
    sum_fatbins = sum(by_name[row["tag"] + "/total"]["wall_seconds"] for row in run["fatbins"])
    if total + 0.1 < sum_fatbins or run["wall_seconds"] + 0.1 < total:
        raise ValueError("stage times are inconsistent with complete build totals")
    for row in run["fatbins"]:
        children = [stage for stage in stages if stage["name"].startswith(row["tag"] + "/") and not stage["name"].endswith("/total")]
        if by_name[row["tag"] + "/total"]["wall_seconds"] + 0.1 < sum(stage["wall_seconds"] for stage in children):
            raise ValueError("fatbin stage time is smaller than its completed child stages")


def validate_run(run, index, output, output_hash, source_blocks, fatbins, payloads):
    if run.get("schema") != 2 or run.get("report_kind") != "full-dll-build" or run.get("status") != "passed":
        raise ValueError("unsupported or incomplete full-build report")
    if run.get("numerical_options") != DEFAULT_OPTIONS:
        raise ValueError("validation requires unchanged default numerical options")
    if run.get("source_dll_sha256") != DLL_SHA256:
        raise ValueError("build report source DLL hash is not supported")
    if not build.paths_alias(run.get("output", ""), output) or run.get("output_sha256") != output_hash:
        raise ValueError("build report output path/hash differs from its verified DLL")
    if run.get("output_bytes") != Path(output).stat().st_size:
        raise ValueError("build report output size differs from its verified DLL")
    rows = run.get("fatbins")
    if not isinstance(rows, list) or len(rows) != 15 or [row.get("tag") for row in rows] != [f"fatbin_{n:02d}" for n in range(1, 16)]:
        raise ValueError("fatbin coverage or order differs from a complete build")
    counts = Counter(row.get("cache") for row in rows)
    expected_cache = "miss" if index == 0 else "hit"
    if counts != {expected_cache: 15}:
        raise ValueError("per-fatbin cache decisions do not prove a full first / cached second build")
    if type(run.get("cache_hits")) is not int or type(run.get("cache_misses")) is not int or (run["cache_hits"], run["cache_misses"]) != (counts.get("hit", 0), counts.get("miss", 0)):
        raise ValueError("aggregate cache counts differ from per-fatbin decisions")
    validate_stages(run, index)
    work = Path(run["work"]).resolve()
    for number, row in enumerate(rows):
        tag = row["tag"]
        if row.get("status") != "passed":
            raise ValueError("incomplete fatbin in build report")
        original = work / f"{tag}.1.sm_120.cubin"
        inputs = build.cache_inputs(Path(build.FATBINS, f"{tag}.1.sm_120.ptx"),
                                    Path(build.FATBINS, f"{tag}.fatbin"), original, options=DEFAULT_OPTIONS)
        if row.get("inputs") != inputs:
            raise ValueError(f"{tag}: recorded source/tool/PTX/original-cubin inputs or options differ from current content")
        if inputs["files"]["source_fatbin"] != digest(source_blocks[number]):
            raise ValueError("a build fatbin input does not match the supported original DLL")
        expected_outputs = {"ptx": work / f"{tag}.sm86.ptx", "cubin": work / f"{tag}.sm86.cubin", "fatbin": work / f"{tag}.sm86.fatbin"}
        if not isinstance(row.get("outputs"), dict) or set(row["outputs"]) != set(expected_outputs):
            raise ValueError("incomplete or unexpected cache artifact records")
        for name, expected_path in expected_outputs.items():
            recorded = row["outputs"][name]
            if not build.paths_alias(recorded["path"], expected_path):
                raise ValueError("cache artifact path differs from the recorded work directory")
            if build.file_sha256(expected_path) != recorded["sha256"] or expected_path.stat().st_size != recorded["bytes"]:
                raise ValueError("cache artifact differs from its recorded bytes / SHA256")
        if row["outputs"]["fatbin"]["sha256"] != fatbins[number]["sha256"]:
            raise ValueError("DLL section differs from its validated cached fatbin")
        original_bytes = original.read_bytes()
        compiled_bytes = expected_outputs["cubin"].read_bytes()
        if digest(original_bytes) != row.get("original_cubin_sha256") or original_bytes not in payloads[number]:
            raise ValueError("original sm_120 cubin is not preserved in its appended fatbin")
        if compiled_bytes not in payloads[number]:
            raise ValueError("compiled sm_86 cubin is not embedded in its appended fatbin")
    return run


def verify(original, candidate, second=None, first_report=None, second_report=None, baseline=None):
    validate_selection(candidate, second, first_report, second_report)
    source = Path(original).read_bytes()
    if digest(source) != DLL_SHA256:
        raise ValueError("source DLL is not the supported 310.8.0 original")
    data = Path(candidate).read_bytes()
    pe = pefile.PE(data=data, fast_load=True)
    nv = [section for section in pe.sections if section.Name.rstrip(b"\0") == b".nv86"]
    if len(nv) != 1:
        raise ValueError("candidate must contain exactly one .nv86 section")
    blob = nv[0].get_data()[:nv[0].Misc_VirtualSize]
    fatbins, payloads = [], []
    for match in re.finditer(re.escape(bytes.fromhex("50ed55ba")), blob):
        off = match.start()
        size = 16 + struct.unpack_from("<Q", blob, off + 8)[0]
        if off % 256 or off + size > len(blob):
            raise ValueError("misaligned or truncated appended fatbin")
        payloads.append(blob[off:off + size])
        fatbins.append({"offset_in_section": off, "bytes": size, "sha256": digest(payloads[-1])})
    if len(fatbins) != 15:
        raise ValueError("expected 15 appended fatbins")
    if pe.OPTIONAL_HEADER.DATA_DIRECTORY[4].VirtualAddress or pe.OPTIONAL_HEADER.DATA_DIRECTORY[4].Size:
        raise ValueError("modified DLL retains a stale Authenticode directory")
    result = {"scope": "CPU build evidence consistency and resource identity; not official output parity",
              "source_dll_sha256": digest(source), "candidate": str(candidate), "candidate_sha256": digest(data),
              "candidate_bytes": len(data), "weights": weights(data), "source_weights": weights(source), "fatbins": fatbins}
    if second:
        result["second_sha256"] = build.file_sha256(second)
        if result["second_sha256"] != result["candidate_sha256"]:
            raise ValueError("second cached build differs byte-for-byte from first complete build")
        result["two_builds_byte_identical"] = True
    if baseline:
        result["baseline_sha256"] = build.file_sha256(baseline)
        result["candidate_matches_existing_dll"] = result["baseline_sha256"] == result["candidate_sha256"]
    if first_report:
        report_bytes = [Path(path).read_bytes() for path in (first_report, second_report)]
        runs = [json.loads(value) for value in report_bytes]
        source_blocks = []
        for match in re.finditer(re.escape(bytes.fromhex("50ed55ba")), source):
            off = match.start()
            size = 16 + struct.unpack_from("<Q", source, off + 8)[0]
            source_blocks.append(source[off:off + size])
        if len(source_blocks) != 15:
            raise ValueError("original DLL fatbin count differs from supported version")
        for index, (run, output) in enumerate(zip(runs, (candidate, second))):
            validate_run(run, index, output, result["candidate_sha256"], source_blocks, fatbins, payloads)
        if not build.paths_alias(runs[0]["work"], runs[1]["work"]) or [row["inputs"] for row in runs[0]["fatbins"]] != [row["inputs"] for row in runs[1]["fatbins"]]:
            raise ValueError("work directory or content inputs changed between full and cached builds")
        result["build_report_sha256"] = {"first": digest(report_bytes[0]), "second": digest(report_bytes[1])}
        result["cache_validation"] = {"first_misses": 15, "second_hits": 15,
                                      "first_wall_seconds": runs[0]["wall_seconds"], "second_wall_seconds": runs[1]["wall_seconds"],
                                      "numerical_options": DEFAULT_OPTIONS, "original_sm120_cubins_preserved": 15,
                                      "compiled_sm86_cubins_embedded": 15, "source_fatbins_match_original_dll": 15,
                                      "second_compile_rewrite_extract_stages": 0, "current_inputs_revalidated": True,
                                      "all_per_fatbin_decisions_and_stages_revalidated": True}
    return result


def main(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    for name in ("candidate", "second", "baseline", "first-report", "second-report", "report"):
        pre.add_argument("--" + name, type=Path)
    pre.add_argument("--original", type=Path, default=ROOT / "dlss5/nvngx_dlssnr.dll")
    known, _ = pre.parse_known_args(argv)
    report_path, protected = validate_report_path(known)
    report = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat()}

    def save():
        if report_path:
            temporary = report_path.with_suffix(report_path.suffix + ".tmp")
            build.require_disjoint([report_path, temporary], protected)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
            temporary.replace(report_path)

    save()
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--original", type=Path, default=ROOT / "dlss5/nvngx_dlssnr.dll")
        parser.add_argument("--candidate", type=Path, required=True)
        for name in ("second", "baseline", "first-report", "second-report", "report"):
            parser.add_argument("--" + name, type=Path)
        args = parser.parse_args(argv)
        report.update(verify(args.original, args.candidate, args.second, args.first_report, args.second_report, args.baseline))
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        save()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
