"""Recompute the static x64 PE host map for DLSS-NR 310.8.0 and 310.9.16.0.

Run with the two unmodified DLLs as inputs:
    python -m tools.inspect_nr_port_host --baseline OLD_DLL --candidate NEW_DLL

Uses read-only file mappings and prints strict JSON with hashes and addresses.
It does not load, execute, patch or extract any part of either DLL.
"""
import argparse
import hashlib
import json
import mmap
from pathlib import Path
import re
import struct


BASELINE_SHA = "e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e"
OCTOBER_SHA = "2bc19bc09a7b89dd44f73e2ac0108fcc586e2154bdd0830f242f4f040e28789f"
FATBIN_MAGIC = bytes.fromhex("50ed55ba")
ARCH_PROBE = bytes.fromhex("8d88c0feffff")
RIP_LEA = re.compile(rb"[\x48\x4c]\x8d[\x05\x0d\x15\x1d\x25\x2d\x35\x3d]")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def occurrences(data, pattern):
    start = 0
    while (offset := data.find(pattern, start)) >= 0:
        yield offset
        start = offset + 1


def inspect_pe(data, expected_sha, *, october):
    require(sha(data) == expected_sha, "Expected the pinned unmodified source DLL SHA-256")

    def u16(offset):
        return struct.unpack_from("<H", data, offset)[0]

    def u32(offset):
        return struct.unpack_from("<I", data, offset)[0]

    def i32(offset):
        return struct.unpack_from("<i", data, offset)[0]

    def u64(offset):
        return struct.unpack_from("<Q", data, offset)[0]

    pe = u32(0x3c)
    coff = pe + 4
    optional = coff + 20
    require(data[:2] == b"MZ" and data[pe:coff] == b"PE\0\0" and u16(coff) == 0x8664
            and u16(optional) == 0x20b, "Expected an x64 PE32+ image")
    image = u64(optional + 24)
    first_header = optional + u16(coff + 16)
    sections = {}
    for i in range(u16(coff + 2)):
        name, vsize, rva, size, offset = struct.unpack_from("<8sIIII", data, first_header + 40*i)
        require(offset + size <= len(data), "Section extends past file end")
        sections[name.rstrip(b"\0").decode("ascii")] = dict(rva=rva, offset=offset, bytes=size, virtual_bytes=vsize)

    def to_rva(offset):
        found = [s for s in sections.values() if s["offset"] <= offset < s["offset"] + s["bytes"]]
        require(len(found) == 1, f"File offset is not uniquely backed by a section: {offset:#x}")
        return found[0]["rva"] + offset - found[0]["offset"]

    def to_offset(rva):
        found = [s for s in sections.values() if s["rva"] <= rva < s["rva"] + s["bytes"]]
        require(len(found) == 1, f"RVA is not uniquely backed by file bytes: {rva:#x}")
        return found[0]["offset"] + rva - found[0]["rva"]

    def at(rva, length):
        offset = to_offset(rva)
        require(any(s["offset"] <= offset and offset + length <= s["offset"] + s["bytes"]
                    for s in sections.values()), "Range crosses a section boundary")
        return data[offset:offset + length]

    text = sections[".text"]
    t0, tv = text["offset"], text["rva"]
    code = data[t0:t0 + text["bytes"]]
    reloc_rva, reloc_bytes = struct.unpack_from("<II", data, optional + 112 + 8*5)
    block = to_offset(reloc_rva)
    end = block + reloc_bytes
    dir64 = set()
    while block < end:
        page, size = struct.unpack_from("<II", data, block)
        require(size >= 8 and size % 2 == 0 and block + size <= end, "Invalid PE relocation block")
        for offset in range(block + 8, block + size, 2):
            entry = u16(offset)
            if entry >> 12 == 10:  # IMAGE_REL_BASED_DIR64
                dir64.add(page + (entry & 0xfff))
        block += size

    fatbins = list(occurrences(data, FATBIN_MAGIC))
    require(len(fatbins) == 15, "Expected exactly 15 fatbin magics in the pinned image")
    lea = {}
    for match in RIP_LEA.finditer(code):
        k = match.start()
        lea.setdefault(tv + k + 7 + i32(t0 + k + 3), []).append(tv + k)

    modules = []
    for n, offset in enumerate(fatbins, 1):
        version, header, payload = struct.unpack_from("<HHQ", data, offset + 4)
        length, target = header + payload, to_rva(offset)
        end = fatbins[n] if n < len(fatbins) else len(data)
        require(version == 1 and header == 16 and payload > 0 and offset + length <= end,
                f"Invalid fatbin header or bounds for module {n}")
        rip = lea.get(target, [])
        absolute = [to_rva(o) for o in occurrences(data, struct.pack("<Q", image + target))]
        require(len(rip) + len(absolute) == 1, f"Module {n} does not have exactly one host pointer reference")
        row = dict(module=n, file_offset=hex(offset), va=hex(image + target), bytes=length,
                   sha256=sha(data[offset:offset + length]))
        if n <= 13:
            require(len(rip) == 1, f"Module {n} must use a RIP-relative reference")
            ref = rip[0]
            width = 8 if n <= 7 else 4
            # The known loader forms place the length load next to the pointer load.
            size_mov = ref - 7 if width == 8 else ref + (10 if n == 8 else 11)
            size_ip = to_offset(size_mov)
            opcode = bytes.fromhex("4c8b0d") if width == 8 else bytes.fromhex("8b05")
            require(at(size_mov, len(opcode)) == opcode, f"Unexpected module {n} length-load encoding")
            size_rva = size_mov + len(opcode) + 4 + i32(size_ip + len(opcode))
            value = (u64 if width == 8 else u32)(to_offset(size_rva))
            require(value == length, f"Module {n} host length does not match its fatbin header")
            row.update(pointer_ref_kind="rip_lea", pointer_ref_va=hex(image + ref),
                       pointer_instruction_sha256=sha(at(ref, 7)), length_bytes=width,
                       length_va=hex(image + size_rva), length_load_va=hex(image + size_mov),
                       length_load_sha256=sha(at(size_mov, len(opcode) + 4)))
        else:
            require(len(absolute) == 1 and absolute[0] in dir64, f"Module {n} must have an actual DIR64 relocation")
            ref = absolute[0]
            row.update(pointer_ref_kind="absolute_dir64", pointer_ref_va=hex(image + ref),
                       descriptor_va=hex(image + ref - 8), descriptor_sha256=sha(at(ref - 8, 24)),
                       length_bytes=None, length_va=None)
        modules.append(row)
    size_table = int(modules[0]["length_va"], 16)
    require([int(row["length_va"], 16) for row in modules[:7]] == list(range(size_table, size_table + 56, 8)),
            "Expected seven consecutive QWORD lengths")

    check = 0x17b9a if october else 0x17e9a
    accepted = 0x17c2b if october else 0x17f2b
    require([tv + m.start() for m in re.finditer(re.escape(ARCH_PROBE), code)] == [check],
            "Unexpected architecture-probe address or signature count")
    architecture = dict(probe_va=hex(image + check), accepted_block_va=hex(image + accepted),
                        gate_range_bytes=accepted + 9 - check,
                        gate_range_sha256=sha(at(check, accepted + 9 - check)),
                        old_offset_prefix_matches=at(0x17e9a, 6) == ARCH_PROBE)
    if october:
        require(at(0x17bf4, 1) == b"\x7d" and 0x17bf6 + at(0x17bf5, 1)[0] == accepted,
                "October's architecture branch target did not match")
        destinations = {}
        for arch in (0x140, 0x160, 0x170, 0x180, 0x190, 0x1a0, 0x1b0):
            selector = at(0x18160 + arch - 0x140, 1)[0]
            destinations[hex(arch)] = hex(image + u32(to_offset(0x18140 + 4*selector)))
        require(destinations["0x1b0"] == architecture["accepted_block_va"], "Unexpected 0x1b0 jump target")
        architecture["switch_destinations"] = destinations

    cert, cert_bytes = struct.unpack_from("<II", data, optional + 112 + 4*8)
    require(cert + cert_bytes == len(data), "Unexpected certificate table extent")
    export_dir = to_offset(u32(optional + 112))
    export_count = u32(export_dir + 24)
    functions, names, ordinals = (to_offset(u32(export_dir + off)) for off in (28, 32, 36))
    exports = {}
    app_id = None
    for i in range(export_count):
        offset = to_offset(u32(names + 4*i))
        name = data[offset:data.find(b"\0", offset)].decode("ascii")
        fn = u32(functions + 4*u16(ordinals + 2*i))
        if name == "NVSDK_NGX_GetApplicationId":
            body = at(fn, 6)
            require(body[0] == 0xb8 and body[5] == 0xc3, "Unexpected application-ID getter encoding")
            app_id = hex(struct.unpack_from("<I", body, 1)[0])
        if name == "NVSDK_NGX_GetApplicationId" or (
                name.startswith(("NVSDK_NGX_D3D12_", "NVSDK_NGX_VULKAN_"))
                and name.endswith(("Init_Ext", "CreateFeature", "EvaluateFeature", "ReleaseFeature"))):
            exports[name] = hex(image + fn)
    require(app_id == "0xe658703", "Unexpected exported application ID")
    return dict(sha256=expected_sha, bytes=len(data), image_base=hex(image), architecture=architecture,
                modules=modules, number_of_sections=len(sections), sections=sections, export_count=export_count,
                selected_exports=exports, application_id_simple_return=app_id,
                certificate_offset=hex(cert), certificate_bytes=cert_bytes,
                can_fit_one_more_section_header=first_header + 40*(len(sections) + 1) <= u32(optional + 60))


def inspect(path, sha256, *, october=False):
    with path.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        return inspect_pe(data, sha256, october=october)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Unmodified 310.8.0 DLL")
    parser.add_argument("--candidate", type=Path, required=True, help="Unmodified 310.9.16.0 DLL")
    args = parser.parse_args(argv)
    try:
        old = inspect(args.baseline, BASELINE_SHA)
        new = inspect(args.candidate, OCTOBER_SHA, october=True)
        require(old["selected_exports"].keys() == new["selected_exports"].keys(), "Selected export names differ")
        unchanged = []
        for baseline, october in zip(old["modules"], new["modules"], strict=True):
            same = baseline["sha256"] == october["sha256"]
            october["matches_3108_fatbin"] = same
            if same:
                unchanged.append(october["module"])
        require(unchanged == list(range(8, 16)), "Unexpected identity across the two module inventories")
    except (OSError, ValueError, KeyError, struct.error) as error:
        parser.exit(1, f"FAIL: {error}\n")
    print(json.dumps(dict(schema=1, scope="static PE host map only; no runtime validation",
                          old=old, october=new), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
