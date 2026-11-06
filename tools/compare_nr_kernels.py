"""Compare CUDA kernel inventories from two locally supplied original NR DLLs.

This is a static comparison. It does not load either DLL or run device code.
Keep the new work directory out of Git: it contains extracted NVIDIA binaries.
The JSON on stdout contains hashes, symbols and counts, not PTX or cubin bytes.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess

from .dllq.index import carve_fatbins

ENTRY = re.compile(rb"^\.visible\s+\.entry\s+([^\s(]+)", re.M)
CLOSE = re.compile(rb"^}\r?$", re.M)
PTX_HEADER = re.compile(rb"^\.(?:version|target)\s+(.+?)\s*$", re.M)
MMA = re.compile(rb"\bmma(?:\.sp)?\.sync\.[\w.:]+")
PARAM = re.compile(rb"\.param\s+\.align\s+\d+\s+\.b8\s+\w+\[(\d+)\]")
ELF_SECTION = struct.Struct("<IIQQQQIIQQ")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ptx_metadata(path):
    data = path.read_bytes()
    functions = {}
    matches = list(ENTRY.finditer(data))
    for i, match in enumerate(matches):
        end = CLOSE.search(data, match.end(), matches[i + 1].start() if i + 1 < len(matches) else len(data))
        name = match[1].decode("ascii")
        if end is None or name in functions:
            raise ValueError(f"Unrecognized PTX entry layout in {path.name}: {name}")
        function = data[match.start():end.end()]
        functions[name] = {
            "sha256": sha(function),
            "param_arrays_bytes": [int(x) for x in PARAM.findall(function.split(b"{", 1)[0])],
        }
    if not functions:
        raise ValueError(f"No visible PTX entry in {path.name}")
    opcodes = Counter(token.decode("ascii") for token in MMA.findall(data))
    return functions, opcodes, [x.decode("ascii") for x in PTX_HEADER.findall(data)]


def elf_text(path):
    """Hash exactly the code sections of a little endian ELF64 CUDA image."""
    data = path.read_bytes()
    if len(data) < 64 or not data.startswith(b"\x7fELF\x02\x01"):
        raise ValueError(f"Expected little endian ELF64: {path.name}")
    table = struct.unpack_from("<Q", data, 40)[0]
    entry_size, count, strings_index = struct.unpack_from("<HHH", data, 58)
    if entry_size != ELF_SECTION.size or not 0 < strings_index < count or table + count * entry_size > len(data):
        raise ValueError(f"Invalid ELF section table: {path.name}")

    def section(i):
        return ELF_SECTION.unpack_from(data, table + i * entry_size)

    def contents(offset, size):
        if offset + size > len(data):
            raise ValueError(f"Out-of-range ELF section: {path.name}")
        return data[offset:offset + size]

    string_section = section(strings_index)
    strings = contents(*string_section[4:6])
    result = {}
    for i in range(count):
        name_offset, _, _, _, offset, size, *_ = section(i)
        name = strings[name_offset:strings.index(b"\0", name_offset)].decode("ascii")
        if name.startswith(".text."):
            if name in result:
                raise ValueError(f"Duplicate code section: {name}")
            result[name] = sha(contents(offset, size))
    if not result:
        raise ValueError(f"No named code sections: {path.name}")
    return result


def compare(a, b):
    shared = a.keys() & b.keys()
    removed, added = a.keys() - b.keys(), b.keys() - a.keys()
    return {
        "baseline": len(a), "candidate": len(b), "shared": len(shared),
        "identical_shared": sum(a[n] == b[n] for n in shared),
        "removed": len(removed), "added": len(added),
        "removed_names_containing_fp8": sum("fp8" in n.lower() for n in removed),
        "removed_examples": sorted(removed)[:3], "added_examples": sorted(added)[:3],
    }


def inspect(dll, cuobjdump, destination):
    entries = carve_fatbins(dll.read_bytes(), destination / "fatbins")
    if not entries:
        raise ValueError(f"No CUDA fatbins in {dll.name}")

    def module(entry):
        number = entry["idx"]
        binary = Path(entry["path"]).resolve()
        extracted = destination / f"module-{number:02d}"
        extracted.mkdir()
        for option in ("--extract-ptx", "--extract-elf"):
            subprocess.run([str(cuobjdump), option, "all", str(binary)], cwd=extracted,
                           check=True, capture_output=True, timeout=90)
        ptx, cubin = sorted(extracted.glob("*.ptx")), sorted(extracted.glob("*.cubin"))
        if len(ptx) != 1 or len(cubin) != 1:
            raise ValueError(f"Expected one PTX and cubin image in module {number}")
        functions, opcodes, header = ptx_metadata(ptx[0])
        return {
            "number": number, "fatbin_sha256": file_sha(binary), "fatbin_bytes": binary.stat().st_size,
            "ptx_sha256": file_sha(ptx[0]), "cubin_sha256": file_sha(cubin[0]),
            "ptx_header": header, "ptx_mma_static_opcode_counts": dict(sorted(opcodes.items())),
            "functions": functions, "elf_text": elf_text(cubin[0]),
        }

    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(module, entries))


def report(baseline, candidate, cuobjdump, work):
    work.mkdir()
    a, b = [inspect(dll, cuobjdump, work / tag)
            for dll, tag in ((baseline, "baseline"), (candidate, "candidate"))]
    if len(a) != len(b):
        raise ValueError("Module count differs; positional comparison would be misleading")
    rows = []
    for old, new in zip(a, b):
        assert old["number"] == new["number"]
        shared = old["functions"].keys() & new["functions"].keys()
        ptx = compare({k: v["sha256"] for k, v in old["functions"].items()},
                      {k: v["sha256"] for k, v in new["functions"].items()})
        ptx["parameter_array_size_changes"] = [
            {"entry": k, "baseline_bytes": old["functions"][k]["param_arrays_bytes"],
             "candidate_bytes": new["functions"][k]["param_arrays_bytes"]}
            for k in sorted(shared)
            if old["functions"][k]["param_arrays_bytes"] != new["functions"][k]["param_arrays_bytes"]
        ]
        rows.append({
            "module": old["number"], "fatbin_identical": old["fatbin_sha256"] == new["fatbin_sha256"],
            "ptx_entries": ptx, "cubin_code_sections": compare(old["elf_text"], new["elf_text"]),
            "baseline": {k: v for k, v in old.items() if k not in {"functions", "elf_text"}},
            "candidate": {k: v for k, v in new.items() if k not in {"functions", "elf_text"}},
        })
    summary = {key: {field: sum(m[key][field] for m in rows) for field in (
        "baseline", "candidate", "shared", "identical_shared", "removed", "added", "removed_names_containing_fp8"
    )} for key in ("ptx_entries", "cubin_code_sections")}
    return {
        "scope": "Positional CUDA module comparison; static code and symbols, not executed launches or numerical parity",
        "baseline_dll_sha256": file_sha(baseline), "candidate_dll_sha256": file_sha(candidate),
        "cuobjdump_sha256": file_sha(cuobjdump), "module_count": len(rows),
        "identical_fatbin_modules": [m["module"] for m in rows if m["fatbin_identical"]],
        "summary": summary, "modules": rows,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "cuobjdump", "work"):
        ap.add_argument("--" + name, type=Path, required=True)
    args = ap.parse_args()
    try:
        value = report(args.baseline.resolve(), args.candidate.resolve(), args.cuobjdump.resolve(), args.work.resolve())
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        ap.exit(1, f"Comparison failed: {exc}\n")
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
