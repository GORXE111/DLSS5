"""Stage or verify an isolated 310.9.16.0 host PE without loading any DLL.

Supply 15 prebuilt, uncompressed dual-ELF fatbins. The manifest is JSON:
  {"schema": 1, "source_sha256": "...", "modules": [
    {"module": 1, "fatbin": "relative-file.fatbin", "fatbin_sha256": "...", "sm86_sha256": "..."}
  ]}
Use --check-only for an in-memory check, --out for a new immediate child of
the repository's ignored _dl, or --verify-only to recheck a staged DLL.
No CUDA tools, DLL loading, or runtime validation are performed by this tool.
"""
import argparse
import json
import mmap
import os
from pathlib import Path
import struct
import sys

from tools.inspect_nr_port_host import FATBIN_MAGIC, OCTOBER_SHA, inspect_pe, require, sha


ROOT = Path(__file__).resolve().parents[2]
KERNEL_REPORT = ROOT / "research/validation/october_nr_remix_kernels.json"
KERNEL_REPORT_SHA = "bbf13b4916193f94427021e016e08675fbdd6175016c187e812267dc4693e324"
SECTION = struct.Struct("<8sIIIIIIHHI")
SECTION_FLAGS = 0x40000040  # initialized data, readable
ARCH_TABLE_RVA = 0x18160
ARCH_JUMP_RVA = 0x18140
ARCH_FIRST, ARCH_LAST, AMPERE = 0x140, 0x1b0, 0x170


def align(value, multiple):
    return (value + multiple - 1) // multiple * multiple


def pack(fmt, *values):
    return struct.pack("<" + fmt, *values)


def unpack(fmt, data, offset=0):
    return struct.unpack_from("<" + fmt, data, offset)


class PE:
    """Minimal section/range lookup independent from the original host map."""

    def __init__(self, data):
        self.data = data
        self.coff = unpack("I", data, 0x3c)[0] + 4
        self.opt = self.coff + 20
        require(data[:2] == b"MZ" and data[self.coff - 4:self.coff] == b"PE\0\0"
                and unpack("H", data, self.coff)[0] == 0x8664
                and unpack("H", data, self.opt)[0] == 0x20b, "Expected x64 PE32+")
        self.image = unpack("Q", data, self.opt + 24)[0]
        self.section_align, self.file_align = unpack("II", data, self.opt + 32)
        require(self.section_align >= self.file_align > 0, "Invalid PE alignment")
        count = unpack("H", data, self.coff + 2)[0]
        self.first_header = self.opt + unpack("H", data, self.coff + 16)[0]
        self.next_header = self.first_header + count * SECTION.size
        self.sections = {}
        for i in range(count):
            row = SECTION.unpack_from(data, self.first_header + i * SECTION.size)
            name, vsize, rva, size, offset, *_, flags = row
            name = name.rstrip(b"\0").decode("ascii")
            require(name not in self.sections and offset + size <= len(data), "Invalid PE section extent/name")
            self.sections[name] = (vsize, rva, size, offset, flags)
        self.security_field = self.opt + 112 + 8 * 4
        self.checksum_field = self.opt + 64

    def offset(self, rva, length=1):
        offsets = [o + rva - start for _, start, size, o, _ in self.sections.values()
                   if start <= rva and rva + length <= start + size]
        require(len(offsets) == 1, f"RVA range not uniquely backed by section bytes: {rva:#x}")
        return offsets[0]

    def rva_slice(self, rva, length):
        offset = self.offset(rva, length)
        return self.data[offset:offset + length]

    def section_slice(self, name):
        _, _, size, offset, _ = self.sections[name]
        return memoryview(self.data)[offset:offset + size]

    def directory(self, index):
        return unpack("II", self.data, self.opt + 112 + 8 * index)

    def dir64(self):
        rva, length = self.directory(5)
        start = self.offset(rva, length)
        end = start + length
        result = set()
        while start < end:
            page, size = unpack("II", self.data, start)
            require(8 <= size <= end - start and size % 2 == 0, "Malformed PE relocation block")
            for off in range(start + 8, start + size, 2):
                entry = unpack("H", self.data, off)[0]
                if entry >> 12 == 10:
                    result.add(page + (entry & 0xfff))
            start += size
        return result


def pe_checksum(data, offset):
    # PE images emitted by this tool have an even, file-aligned byte count.
    require(len(data) % 2 == 0 and offset % 2 == 0, "Unaligned PE checksum input")
    words = (memoryview(data).cast("H") if sys.byteorder == "little"
             else (value for (value,) in struct.iter_unpack("<H", data)))
    total = sum(words) - sum(unpack("HH", data, offset))
    while total >> 16:
        total = (total & 0xffff) + (total >> 16)
    return total + len(data)


def dual_elf_hashes(data):
    """Decode the specific uncompressed kind=elf container used by the lab."""
    magic, version, header, length = unpack("4sHHQ", data)
    require(magic == FATBIN_MAGIC and version == 1 and header == 16 and length == len(data) - 16,
            "Malformed fatbin outer size/header")
    result = {}
    pos = header
    while pos < len(data):
        kind, record_version, hdr, payload = unpack("HHIQ", data, pos)
        require(kind == 2 and record_version == 0x101 and hdr >= 64 and payload >= 64
                and pos + hdr + payload <= len(data), "Expected a bounded ELF fatbin record")
        compressed_size = unpack("I", data, pos + 16)[0]
        flags = unpack("Q", data, pos + 40)[0]
        decompressed_size = unpack("Q", data, pos + 56)[0]
        require(compressed_size == decompressed_size == 0 and not flags & 0x8000,
                "Only uncompressed ELF fatbin records are accepted")
        sm = unpack("I", data, pos + 28)[0]
        elf = data[pos + hdr:pos + hdr + payload]
        require(elf.startswith(b"\x7fELF\x02\x01") and unpack("H", elf, 18)[0] == 190
                and (unpack("I", elf, 48)[0] >> 8) & 0xff == sm, "ELF machine or SM does not match the record")
        table = unpack("Q", elf, 40)[0]
        entry, count = unpack("HH", elf, 58)
        require(sm in (86, 120) and sm not in result and entry == 64 and count > 0
                and table + entry * count <= len(elf), "Unexpected duplicate SM or ELF section table")
        result[sm] = sha(elf)
        pos += hdr + payload
    require(set(result) == {86, 120}, "Expected exactly one sm86 and one sm120 ELF")
    return result


def input_modules(manifest, host):
    evidence = KERNEL_REPORT.read_bytes()
    require(sha(evidence) == KERNEL_REPORT_SHA, "October kernel evidence hash changed")
    native = json.loads(evidence)
    require(native["candidate_dll_sha256"] == OCTOBER_SHA, "Kernel evidence targets a different DLL")
    doc = json.loads(manifest.read_text(encoding="utf-8-sig"))
    require(doc.get("schema") == 1 and doc.get("source_sha256") == OCTOBER_SHA,
            "Wrong module manifest schema or source hash")
    modules = doc["modules"]
    require([row["module"] for row in modules] == list(range(1, 16)), "Manifest must name modules 1–15 in order")
    seen = set()
    payloads = []
    for row, original, proof in zip(modules, host["modules"], native["modules"], strict=True):
        n = row["module"]
        require(type(n) is int and proof["module"] == n and proof["candidate"]["fatbin_sha256"] == original["sha256"],
                f"Module {n}: native cubin evidence is not bound to this original fatbin")
        path = (manifest.parent / row["fatbin"]).resolve(strict=True)
        require(path not in seen, "Two modules alias the same fatbin path")
        seen.add(path)
        data = path.read_bytes()
        digest = sha(data)
        require(digest == row["fatbin_sha256"], f"Module {n}: replacement fatbin hash mismatch")
        elfs = dual_elf_hashes(data)
        require(elfs[86] == row["sm86_sha256"] and elfs[120] == proof["candidate"]["cubin_sha256"],
                f"Module {n}: sm86 manifest or original sm120 identity mismatch")
        payloads.append((data, digest, elfs))
    return payloads


def gate_plan(pe, host, allow_ampere):
    count = ARCH_LAST - ARCH_FIRST + 1
    prior = pe.rva_slice(ARCH_TABLE_RVA, count)
    expected = bytearray(prior)
    probe = int(host["architecture"]["probe_va"], 16) - pe.image
    accepted = int(host["architecture"]["accepted_block_va"], 16) - pe.image
    size = host["architecture"]["gate_range_bytes"]
    require(sha(pe.rva_slice(probe, size)) == host["architecture"]["gate_range_sha256"], "Original gate identity changed")
    previous_target = unpack("I", pe.rva_slice(ARCH_JUMP_RVA + 4 * prior[AMPERE - ARCH_FIRST], 4))[0]
    require(previous_target != accepted and unpack("I", pe.rva_slice(ARCH_JUMP_RVA + 4 * prior[-1], 4))[0] == accepted,
            "Unexpected original Ampere/0x1b0 targets")
    if allow_ampere:
        expected[AMPERE - ARCH_FIRST] = prior[-1]
    return bytes(expected), (probe, size), previous_target, accepted


def plan(source, host, payloads, allow_ampere):
    pe = PE(source)
    cert, cert_size = pe.directory(4)
    require(cert_size > 0 and cert + cert_size == len(source) and ".nv86" not in pe.sections, "Unexpected original PE layout")
    require(pe.next_header + SECTION.size <= unpack("I", source, pe.opt + 60)[0]
            and not any(source[pe.next_header:pe.next_header + SECTION.size]), "No empty section-header slot")
    require(max(o + size for _, _, size, o, _ in pe.sections.values()) <= cert, "Certificate overlaps a section")
    rva = align(max(v + max(vs, size) for vs, v, size, _, _ in pe.sections.values()), pe.section_align)
    raw = align(cert, pe.file_align)
    blob = bytearray()
    edits = {}
    bindings = []
    relocations = pe.dir64()
    for row, (fat, digest, elfs) in zip(host["modules"], payloads, strict=True):
        offset = align(len(blob), 256)
        blob.extend(b"\0" * (offset - len(blob)))
        blob.extend(fat)
        dst = rva + offset
        ref = int(row["pointer_ref_va"], 16) - pe.image
        size_rva = int(row["length_va"], 16) - pe.image if row["length_va"] else None
        if row["pointer_ref_kind"] == "rip_lea":
            width = row["length_bytes"]
            edits[pe.offset(ref + 3, 4)] = pack("i", dst - ref - 7)
            edits[pe.offset(size_rva, width)] = pack("Q" if width == 8 else "I", len(fat))
        else:
            require(ref in relocations, "Original absolute pointer is missing its relocation")
            edits[pe.offset(ref, 8)] = pack("Q", pe.image + dst)
        bindings.append(dict(module=row["module"], reference_kind=row["pointer_ref_kind"], reference_rva=ref,
                             target_rva=dst, length_rva=size_rva, length_width=row["length_bytes"],
                             bytes=len(fat), sha256=digest, sm86_sha256=elfs[86], sm120_sha256=elfs[120]))
    raw_size = align(len(blob), pe.file_align)
    edits[pe.next_header] = SECTION.pack(b".nv86", len(blob), rva, raw_size, raw, 0, 0, 0, 0, SECTION_FLAGS)
    edits[pe.coff + 2] = pack("H", len(pe.sections) + 1)
    edits[pe.opt + 8] = pack("I", unpack("I", source, pe.opt + 8)[0] + raw_size)
    edits[pe.opt + 56] = pack("I", align(rva + len(blob), pe.section_align))
    edits[pe.security_field] = b"\0" * 8
    gate, code, old_target, accept = gate_plan(pe, host, allow_ampere)
    if allow_ampere:
        target = ARCH_TABLE_RVA + AMPERE - ARCH_FIRST
        edits[pe.offset(target)] = gate[AMPERE - ARCH_FIRST:AMPERE - ARCH_FIRST + 1]
    require(all(0 <= start < start + len(data) <= cert for start, data in edits.items()), "Unexpected edit outside original PE")
    return pe, (len(blob), rva, raw_size, raw, SECTION_FLAGS), blob, edits, bindings, (gate, code, old_target, accept)


def verify(source, staged, build_plan, allow_ampere):
    original, section, blob, edits, bindings, (gate, (probe, code_size), prior_target, accepted) = build_plan
    actual = PE(staged)
    require(actual.sections == {**original.sections, ".nv86": section} and actual.image == original.image,
            "Staged PE sections or image base differ from the plan")
    require(actual.directory(4) == (0, 0) and actual.directory(5) == original.directory(5),
            "Stale signature or changed relocation directory")
    require(actual.dir64() == original.dir64() and sha(actual.section_slice(".reloc")) == sha(original.section_slice(".reloc")),
            "Staged relocation entries or bytes changed")
    resources = sha(original.section_slice(".rsrc"))
    require(sha(actual.section_slice(".rsrc")) == resources, "Staged resource section differs from pinned original")
    require(actual.rva_slice(ARCH_TABLE_RVA, len(gate)) == gate
            and actual.rva_slice(probe, code_size) == original.rva_slice(probe, code_size), "Unexpected change to architecture gate")
    selector = actual.rva_slice(ARCH_TABLE_RVA + AMPERE - ARCH_FIRST, 1)[0]
    selected = unpack("I", actual.rva_slice(ARCH_JUMP_RVA + 4 * selector, 4))[0]
    require(selected == (accepted if allow_ampere else prior_target), "Ampere gate target differs from experiment policy")
    for row in bindings:
        ref, size = row["reference_rva"], row["bytes"]
        if row["reference_kind"] == "rip_lea":
            lea = actual.rva_slice(ref, 7)
            require(lea[:3] == original.rva_slice(ref, 3), "RIP LEA opcode changed")
            target = ref + 7 + unpack("i", lea, 3)[0]
            field = actual.rva_slice(row["length_rva"], row["length_width"])
            require(int.from_bytes(field, "little") == size, "Replayed host fatbin size mismatch")
        else:
            target = unpack("Q", actual.rva_slice(ref, 8))[0] - actual.image
        require(target == row["target_rva"] and sha(actual.rva_slice(target, size)) == row["sha256"],
                "Replayed host fatbin pointer or target bytes mismatch")

    checksum = pe_checksum(staged, actual.checksum_field)
    require(unpack("I", staged, actual.checksum_field)[0] == checksum, "Incorrect staged PE checksum")
    cert = original.directory(4)[0]
    size, _, raw_size, raw, _ = section
    require(len(staged) == raw + raw_size and staged[raw:raw + size] == blob
            and not any(staged[cert:raw]) and not any(staged[raw + size:]), "Staged payload/padding differs")
    allowed = {**edits, actual.checksum_field: pack("I", checksum)}
    with memoryview(source) as source_view, memoryview(staged) as staged_view:
        last = 0
        for start, replacement in sorted(allowed.items()):
            require(last <= start and staged_view[last:start] == source_view[last:start]
                    and staged_view[start:start + len(replacement)] == replacement, "Unexpected edit in original PE prefix")
            last = start + len(replacement)
        require(staged_view[last:cert] == source_view[last:cert], "Unexpected edit in original PE suffix")
    return dict(status="static_host_verified", source_sha256=OCTOBER_SHA, staged_sha256=sha(staged),
                bytes=len(staged), resource_section_sha256=resources, relocation_section_sha256=sha(actual.section_slice(".reloc")),
                architecture=dict(policy="ampere_only_experiment" if allow_ampere else "unchanged",
                                  ampere_target_rva=hex(selected), gate_code_unchanged=True),
                bindings=bindings, runtime_validated=False)


def render(source, build_plan):
    pe, (_, _, raw_size, raw, _), blob, edits, *_ = build_plan
    cert = pe.directory(4)[0]
    staged = bytearray(memoryview(source)[:cert])
    staged.extend(b"\0" * (raw - cert))
    staged.extend(blob)
    staged.extend(b"\0" * (raw_size - len(blob)))
    for start, value in edits.items():
        staged[start:start + len(value)] = value
    struct.pack_into("<I", staged, pe.checksum_field, pe_checksum(staged, pe.checksum_field))
    return staged


def output_dir(path):
    ignore = ROOT / "_dl"
    require(ignore.is_dir() and not ignore.is_symlink() and not getattr(ignore.lstat(), "st_file_attributes", 0) & 0x400,
            "Expected the repository's ordinary ignored _dl directory")
    path = path.absolute()
    require(path.parent == ignore and path.parent.resolve(strict=True) == ignore and not os.path.lexists(path),
            "Output must be a new immediate child of the repository's ignored _dl directory")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--modules", type=Path, required=True, help="JSON manifest for the 15 dual-ELF fatbins")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true", help="Stage and verify in memory without writing")
    mode.add_argument("--out", type=Path, help="New immediate child of the repo's ignored _dl directory")
    mode.add_argument("--verify-only", type=Path, metavar="STAGED_DLL", help="Verify a staged file without writing")
    parser.add_argument("--allow-ampere-experiment", action="store_true",
                        help="Only route the observed Ampere 0x170 selector to the accepted branch; off by default")
    args = parser.parse_args(argv)
    try:
        out = output_dir(args.out) if args.out else None
        with args.original.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as source:
            host = inspect_pe(source, OCTOBER_SHA, october=True)
            payloads = input_modules(args.modules.resolve(strict=True), host)
            build_plan = plan(source, host, payloads, args.allow_ampere_experiment)
            if args.verify_only:
                with args.verify_only.open("rb") as check, mmap.mmap(check.fileno(), 0, access=mmap.ACCESS_READ) as staged:
                    report = verify(source, staged, build_plan, args.allow_ampere_experiment)
            else:
                staged = render(source, build_plan)
                report = verify(source, staged, build_plan, args.allow_ampere_experiment)
                if out:
                    out.mkdir(exist_ok=False)
                    candidate = out / "nvngx_dlssnr.staged.dll"
                    with candidate.open("xb") as destination:
                        destination.write(staged)
                    del staged
                    with candidate.open("rb") as check, mmap.mmap(check.fileno(), 0, access=mmap.ACCESS_READ) as written:
                        require(verify(source, written, build_plan, args.allow_ampere_experiment) == report,
                                "Written candidate differs from the verified in-memory PE")
                    with (out / "host_report.json").open("x", encoding="utf-8", newline="\n") as evidence:
                        evidence.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError, KeyError, TypeError, struct.error) as error:
        parser.exit(1, f"FAIL: {error}\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
