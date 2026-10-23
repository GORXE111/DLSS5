"""生成 sm_86 版 nvngx_dlssnr.dll。

  1. 15 个 fatbin: PTX -> rewrite -> ptxas sm_86 cubin；原 sm_120 cubin 抽出来一起打包 (50 系照常可用)
  2. 新 fatbin 放进追加的 .nv86 节，把 13 处 lea 位移 + 2 处绝对指针改指过去，并改 13 个硬编码长度
  3. 架构检查 0x180017e9a: `lea ecx,[rax-0x140]` 改成 jmp 0x180017f2b (直接走通过分支)
  4. 去掉 Authenticode 签名 (改过字节后签名本来就失效)

python -m tools.ptx86.build_dll [--out sm86_port/nvngx_dlssnr.dll]
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pefile

from . import rewrite as rw
from .rewrite import rewrite

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DLL = os.path.join(ROOT, "dlss5", "nvngx_dlssnr.dll")
SRC_SHA = "e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e"
FATBINS = os.path.join(ROOT, "fatbins")
WORK = os.path.join(ROOT, "sm86_port")
BIN = os.path.join(ROOT, "cuda_tools", "bin")
BUILD_REPORT = None
BUILD_REPORT_PATH = None
BUILD_OUTPUT = None
BUILD_PROTECTED = []


def paths_alias(left, right):
    """Resolve junction/symlink spellings and Windows case, then detect hard links."""
    left, right = Path(left).resolve(), Path(right).resolve()
    if str(left).casefold() == str(right).casefold():
        return True
    try:
        return os.path.samefile(left, right)
    except FileNotFoundError:
        return False


def require_disjoint(writable, protected):
    writable = [Path(path).resolve() for path in writable if path is not None]
    protected = [Path(path).resolve() for path in protected if path is not None]
    for number, target in enumerate(writable):
        if target.exists() and target.is_dir():
            raise ValueError(f"Output must be a file, not a directory: {target}")
        for other in [*writable[:number], *protected]:
            if paths_alias(target, other):
                raise ValueError(f"Unsafe output alias: {target} aliases protected path {other}")


def cache_paths(work):
    paths = []
    for number in range(1, 16):
        tag = f"fatbin_{number:02d}"
        paths.extend(Path(work, tag + suffix) for suffix in
                     (".sm86.ptx", ".sm86.cubin", ".sm86.fatbin", ".1.sm_120.cubin",
                      ".sm86.fatbin.cache.json", ".sm86.fatbin.cache.json.tmp"))
    return paths


def input_paths():
    paths = [Path(SRC_DLL), *Path(__file__).resolve().parent.glob("*.py")]
    for number in range(1, 16):
        paths.extend(Path(FATBINS, f"fatbin_{number:02d}" + suffix) for suffix in (".1.sm_120.ptx", ".fatbin"))
    paths.extend(Path(BIN, name) for name in ("ptxas.exe", "fatbinary.exe", "cuobjdump.exe"))
    return paths


def work_output(args):
    # Preflight and the typed parser must derive the same path for 00 / -0 / +0.
    maxnreg, floor = int(args.maxnreg), int(args.maxnreg_floor)
    work = Path(args.work).resolve() if args.work else Path(ROOT, "sm86_port")
    if not args.work:
        suffix = ""
        if (args.enc, args.dec) != ("v1", "v1"):
            suffix += f"_e{args.enc}d{args.dec}"
        if floor:
            suffix += f"_f{floor}"
        if args.acc != "f16":
            suffix += "_" + args.acc
        if maxnreg:
            suffix += f"_r{maxnreg}"
        work = Path(str(work) + suffix)
    output = Path(args.out).resolve() if args.out else work / "nvngx_dlssnr.dll"
    return work.resolve(), output.resolve()


def validate_build_paths(args):
    work, output = work_output(args)
    protected = [*input_paths(), *cache_paths(work)]
    require_disjoint([output], protected)
    report = Path(args.report).resolve() if getattr(args, "report", None) else None
    if report is not None:
        # Existing DLLs in this cache are artifacts too; a progress report must never replace one.
        require_disjoint([report, report.with_suffix(report.suffix + ".tmp")],
                         [output, *protected, *work.glob("*.dll")])
    return work, output, report, protected


def write_report():
    if BUILD_REPORT_PATH is None:
        return
    temporary = BUILD_REPORT_PATH.with_suffix(BUILD_REPORT_PATH.suffix + ".tmp")
    require_disjoint([BUILD_REPORT_PATH, temporary], [BUILD_OUTPUT, *BUILD_PROTECTED])
    temporary.write_text(json.dumps(BUILD_REPORT, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, BUILD_REPORT_PATH)


def stage(name, action):
    """Persist a running stage before slow tools start, including SystemExit failures."""
    record = {"name": name, "status": "running", "started_utc": datetime.now(timezone.utc).isoformat()}
    if BUILD_REPORT is not None:
        BUILD_REPORT["stages"].append(record)
        write_report()
    print(f"[{name}] started", flush=True)
    started = time.perf_counter()
    try:
        result = action()
        record["status"] = "passed"
        return result
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        record["wall_seconds"] = time.perf_counter() - started
        write_report()
        print(f"[{name}] {record['status']} in {record['wall_seconds']:.3f}s", flush=True)

ARCH_CHECK_VA = 0x180017E9A
ARCH_OK_VA = 0x180017F2B
ARCH_CHECK_BYTES = bytes.fromhex("8d88c0feffff")  # lea ecx,[rax-0x140]
FATBIN_MAGIC = b"\x50\xed\x55\xba"
SIZE_TABLE_BIG = 0x1122590  # fatbin_01..07 的 qword 长度表 (0x18003D8C3 起逐个 mov r9,[rip+..])
SIZE_SMALL = [0x11250E0, 0x1127B88, 0x112A9A0, 0x112D2F0, 0x112F9E8, 0x1131388]  # fatbin_08..13 的 dword 长度


def run(*cmd, cwd=None):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    err = "\n".join(l for l in (r.stdout + r.stderr).splitlines() if "knob" not in l)
    if r.returncode:
        sys.exit(f"失败: {' '.join(cmd)}\n{err}")


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def cache_inputs(ptx, source_fatbin, original_cubin, options=None):
    """Content, tools and numerical options all participate; mtimes never authorize reuse."""
    sources = [Path(__file__), *(Path(__file__).with_name(f"{name}.py") for name in ("rewrite", "fuse", "flow"))]
    files = {f"code/{p.name}": file_sha256(p) for p in sources}
    files.update({"source_ptx": file_sha256(ptx), "source_fatbin": file_sha256(source_fatbin),
                  "original_cubin": file_sha256(original_cubin)})
    for name in ("ptxas.exe", "fatbinary.exe", "cuobjdump.exe"):
        files[f"tool/{name}"] = file_sha256(os.path.join(BIN, name))
    return {"schema": 1, "files": files, "rewrite_options": dict(rw.ACC if options is None else options),
            "ptxas_args": ["-arch=sm_86"], "fatbinary_args": ["-64", "--compress=false", "sm=86", "sm=120"]}


def cache_valid(manifest, inputs, outputs):
    return (manifest.get("inputs") == inputs and
            isinstance(manifest.get("outputs"), dict) and
            all(os.path.isfile(p) and manifest.get("outputs", {}).get(name) == file_sha256(p)
                for name, p in outputs.items()))


def build_fatbin(n):
    """返回 fatbin_NN 的新 fatbin 字节 (sm_86 + sm_120 两份 ELF)。有缓存。"""
    tag = f"fatbin_{n:02d}"
    ptx = os.path.join(FATBINS, f"{tag}.1.sm_120.ptx")
    out_ptx = os.path.join(WORK, f"{tag}.sm86.ptx")
    cubin = os.path.join(WORK, f"{tag}.sm86.cubin")
    fat = os.path.join(WORK, f"{tag}.sm86.fatbin")
    orig = os.path.join(WORK, f"{tag}.1.sm_120.cubin")
    source_fatbin = os.path.join(FATBINS, f"{tag}.fatbin")
    event = {"tag": tag, "status": "running", "cache": "checking", "extracted_original": False}
    if BUILD_REPORT is not None:
        BUILD_REPORT["fatbins"].append(event)
        write_report()
    manifest_path = fat + ".cache.json"
    try:
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, ValueError):
        manifest = {}
    if not isinstance(manifest, dict) or not isinstance(manifest.get("inputs", {}), dict):
        manifest = {}
    # An extracted cubin is also a cache: refresh it when its container or bytes changed.
    old_files = manifest.get("inputs", {}).get("files", {})
    if not isinstance(old_files, dict):
        old_files = {}
    if (not os.path.isfile(orig) or old_files.get("source_fatbin") != file_sha256(source_fatbin)
            or old_files.get("original_cubin") != file_sha256(orig)):
        event["extracted_original"] = True
        with tempfile.TemporaryDirectory(prefix=f"{tag}_", dir=WORK) as tmp:
            stage(f"{tag}/extract_original", lambda: run(os.path.join(BIN, "cuobjdump.exe"), "-xelf", "all", source_fatbin, cwd=tmp))
            os.replace(os.path.join(tmp, f"{tag}.1.sm_120.cubin"), orig)
    inputs = cache_inputs(ptx, source_fatbin, orig)
    outputs = {"ptx": out_ptx, "cubin": cubin, "fatbin": fat}
    hit = cache_valid(manifest, inputs, outputs)
    event.update(cache="hit" if hit else "miss", inputs=inputs)
    write_report()
    if not hit:
        src, _ = stage(f"{tag}/rewrite", lambda: rewrite(Path(ptx).read_text(encoding="utf-8")))
        with open(out_ptx, "w", encoding="utf-8", newline="\n") as f:
            f.write(src)
        stage(f"{tag}/ptxas", lambda: run(os.path.join(BIN, "ptxas.exe"), "-arch=sm_86", out_ptx, "-o", cubin))
        stage(f"{tag}/fatbinary", lambda: run(os.path.join(BIN, "fatbinary.exe"), "-64", "--compress=false", f"--create={fat}",
            f"--image3=kind=elf,sm=86,file={cubin}", f"--image3=kind=elf,sm=120,file={orig}"))
        print(f"  {tag}: cubin {os.path.getsize(cubin):>9,} -> fatbin {os.path.getsize(fat):>9,}")
        record = {"inputs": inputs, "outputs": {name: file_sha256(p) for name, p in outputs.items()}}
        with open(manifest_path + ".tmp", "w", encoding="utf-8", newline="\n") as f:
            json.dump(record, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(manifest_path + ".tmp", manifest_path)
    event.update(status="passed", outputs={name: {"path": p, "sha256": file_sha256(p), "bytes": os.path.getsize(p)}
                                            for name, p in outputs.items()}, original_cubin_sha256=file_sha256(orig))
    write_report()
    if hit:
        print(f"  {tag}: content cache hit", flush=True)
    return Path(fat).read_bytes()


def align(x, a):
    return (x + a - 1) // a * a


def build(args):
    global WORK
    work, output, _, _ = validate_build_paths(args)
    WORK = str(work)
    rw.ACC["enc"], rw.ACC["dec"] = args.enc, args.dec
    rw.ACC["mode"] = args.acc
    rw.ACC["maxnreg"] = args.maxnreg
    rw.ACC["maxnreg_floor"] = args.maxnreg_floor
    args.out = str(output)
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if BUILD_REPORT is not None:
        BUILD_REPORT.update(work=WORK, output=args.out, numerical_options=dict(rw.ACC))
        write_report()

    data = bytearray(open(SRC_DLL, "rb").read())
    if hashlib.sha256(data).hexdigest() != SRC_SHA:
        raise ValueError("源 dll 不是已知的 310.8.0 版本")
    if BUILD_REPORT is not None:
        BUILD_REPORT["source_dll_sha256"] = SRC_SHA
        write_report()
    pe = pefile.PE(data=bytes(data), fast_load=True)
    base = pe.OPTIONAL_HEADER.ImageBase
    text = pe.sections[0]
    t0, tv = text.PointerToRawData, text.VirtualAddress
    T = bytes(data[t0:t0 + text.SizeOfRawData])

    # 所有 rip 相对 lea: 目标 rva -> [指令文件偏移]
    leas = {}
    for m in re.finditer(rb"[\x48\x4c]\x8d[\x05\x0d\x15\x1d\x25\x2d\x35\x3d]", T):
        i = m.start()
        tgt = tv + i + 7 + struct.unpack_from("<i", T, i + 3)[0]
        leas.setdefault(tgt, []).append(t0 + i)

    fat_offs = [m.start() for m in re.finditer(re.escape(FATBIN_MAGIC), data)]
    assert len(fat_offs) == 15, fat_offs

    # ---- 1. 构建新 fatbin 并排进新节
    print("构建 fatbin:")
    blob = bytearray()
    places = []  # (原 rva, 新节内偏移)
    new_sizes = []
    for n, off in enumerate(fat_offs, 1):
        fb = stage(f"fatbin_{n:02d}/total", lambda: build_fatbin(n))
        new_sizes.append(len(fb))
        blob += b"\0" * (align(len(blob), 256) - len(blob))
        places.append((pe.get_rva_from_offset(off), len(blob)))
        blob += fb

    # ---- 2. 去签名，追加节
    sec_dir = pe.OPTIONAL_HEADER.DATA_DIRECTORY[4]
    if sec_dir.VirtualAddress:
        del data[sec_dir.VirtualAddress:]
    falign, salign = pe.OPTIONAL_HEADER.FileAlignment, pe.OPTIONAL_HEADER.SectionAlignment
    last = pe.sections[-1]
    new_rva = align(last.VirtualAddress + last.Misc_VirtualSize, salign)
    new_raw = align(len(data), falign)
    data += b"\0" * (new_raw - len(data))
    data += blob + b"\0" * (align(len(blob), falign) - len(blob))

    hdr_off = last.get_file_offset() + 40
    assert hdr_off + 40 <= pe.OPTIONAL_HEADER.SizeOfHeaders
    data[hdr_off:hdr_off + 40] = struct.pack(
        "<8sIIIIIIHHI", b".nv86", len(blob), new_rva, align(len(blob), falign), new_raw,
        0, 0, 0, 0, 0x40000040)  # INITIALIZED_DATA | MEM_READ
    fh = pe.FILE_HEADER.get_file_offset()
    struct.pack_into("<H", data, fh + 2, pe.FILE_HEADER.NumberOfSections + 1)
    oh = pe.OPTIONAL_HEADER.get_file_offset()
    struct.pack_into("<I", data, oh + 56, align(new_rva + len(blob), salign))  # SizeOfImage
    struct.pack_into("<II", data, sec_dir.get_file_offset(), 0, 0)             # 安全目录清零

    # ---- 3. 改引用
    for (old_rva, noff) in places:
        tgt = new_rva + noff
        refs = leas.get(old_rva, [])
        for fo in refs:
            rip = pe.get_rva_from_offset(fo) + 7
            struct.pack_into("<i", data, fo + 3, tgt - rip)
        absrefs = [m.start() for m in re.finditer(re.escape(struct.pack("<Q", base + old_rva)), data[:new_raw])]
        for fo in absrefs:
            struct.pack_into("<Q", data, fo, base + tgt)
        assert len(refs) + len(absrefs) == 1, (hex(old_rva), refs, absrefs)

    # ---- 3b. 长度字段。dll 把 (指针, 长度) 交给 NvAPI 的 D3D12 cubin 接口，长度是硬编码的全局量；
    #          只改指针不改长度 → 驱动按旧长度读穿新节 (nvwgf2umx.dll 里 AV)。
    #          fatbin_14/15 走 cuModuleLoadData，长度从 fatbin 头里读，没有长度字段。
    old_sizes = [16 + struct.unpack_from("<Q", data, o + 8)[0] for o in fat_offs]
    for i in range(7):
        fo = pe.get_offset_from_rva(SIZE_TABLE_BIG + 8 * i)
        assert struct.unpack_from("<Q", data, fo)[0] == old_sizes[i]
        struct.pack_into("<Q", data, fo, new_sizes[i])
    for i, rva in enumerate(SIZE_SMALL, 7):
        fo = pe.get_offset_from_rva(rva)
        assert struct.unpack_from("<I", data, fo)[0] == old_sizes[i]
        struct.pack_into("<I", data, fo, new_sizes[i])

    # ---- 4. 架构检查
    fo = pe.get_offset_from_rva(ARCH_CHECK_VA - base)
    assert data[fo:fo + 6] == ARCH_CHECK_BYTES
    data[fo:fo + 6] = b"\xe9" + struct.pack("<i", ARCH_OK_VA - (ARCH_CHECK_VA + 5)) + b"\x90"

    # 校验和 (dll 不强制，但保持一致)
    pe2 = pefile.PE(data=bytes(data), fast_load=True)
    struct.pack_into("<I", data, oh + 64, pe2.generate_checksum())

    open(args.out, "wb").write(data)
    print(f"写出 {args.out}  {len(data):,} 字节  sha256 {hashlib.sha256(data).hexdigest()[:16]}…")
    if BUILD_REPORT is not None:
        BUILD_REPORT.update(output_sha256=hashlib.sha256(data).hexdigest(), output_bytes=len(data),
                            cache_hits=sum(row["cache"] == "hit" for row in BUILD_REPORT["fatbins"]),
                            cache_misses=sum(row["cache"] == "miss" for row in BUILD_REPORT["fatbins"]))


def main(argv=None):
    global BUILD_REPORT, BUILD_REPORT_PATH, BUILD_OUTPUT, BUILD_PROTECTED
    BUILD_REPORT_PATH, BUILD_REPORT, BUILD_OUTPUT, BUILD_PROTECTED = None, None, None, []
    # Identify all destinations and reject aliases before invalidating any report.
    pre = argparse.ArgumentParser(add_help=False)
    for name in ("work", "out", "report"):
        pre.add_argument("--" + name)
    for name, default in (("acc", "f16"), ("enc", "v1"), ("dec", "v1"), ("maxnreg", "0"), ("maxnreg-floor", "0")):
        pre.add_argument("--" + name, default=default)
    known, _ = pre.parse_known_args(argv)
    _, BUILD_OUTPUT, BUILD_REPORT_PATH, BUILD_PROTECTED = validate_build_paths(known)
    BUILD_REPORT = {"schema": 2, "report_kind": "full-dll-build", "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                    "stages": [], "fatbins": []}
    if BUILD_REPORT_PATH is not None:
        BUILD_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_report()
    started = time.perf_counter()
    try:
        ap = argparse.ArgumentParser(description=__doc__)
        ap.add_argument("--acc", choices=["f16", "f32"], default="f16", help="FP8 mma 仿真的累加精度")
        ap.add_argument("--maxnreg", type=int, default=0)
        ap.add_argument("--maxnreg-floor", type=int, default=0)
        ap.add_argument("--enc", choices=["v1", "v2", "none"], default="v1")
        ap.add_argument("--dec", choices=["v1", "v2", "none"], default="v1")
        ap.add_argument("--work", help="Explicit cache/work directory (does not change numerical options)")
        ap.add_argument("--out")
        ap.add_argument("--report", help="Atomic progress and final JSON, invalidated at startup")
        args = ap.parse_args(argv)
        stage("full_dll_build", lambda: build(args))
        BUILD_REPORT["status"] = "passed"
    except BaseException as error:
        BUILD_REPORT.update(status="failed", error_type=type(error).__name__, error=str(error))
        for event in BUILD_REPORT["fatbins"]:
            if event["status"] == "running":
                event.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        BUILD_REPORT["wall_seconds"] = time.perf_counter() - started
        write_report()


if __name__ == "__main__":
    main()
