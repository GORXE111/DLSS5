"""生成 sm_86 版 nvngx_dlssnr.dll。

  1. 15 个 fatbin: PTX -> rewrite -> ptxas sm_86 cubin；原 sm_120 cubin 抽出来一起打包 (50 系照常可用)
  2. 新 fatbin 放进追加的 .nv86 节，把 13 处 lea 位移 + 2 处绝对指针改指过去，并改 13 个硬编码长度
  3. 架构检查 0x180017e9a: `lea ecx,[rax-0x140]` 改成 jmp 0x180017f2b (直接走通过分支)
  4. 去掉 Authenticode 签名 (改过字节后签名本来就失效)

python -m tools.ptx86.build_dll [--out sm86_port/nvngx_dlssnr.dll]
"""
import argparse
import hashlib
import os
import re
import struct
import subprocess
import sys

import pefile

from . import rewrite as rw
from .rewrite import rewrite

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DLL = os.path.join(ROOT, "dlss5", "nvngx_dlssnr.dll")
SRC_SHA = "e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e"
FATBINS = os.path.join(ROOT, "fatbins")
WORK = os.path.join(ROOT, "sm86_port")
BIN = os.path.join(ROOT, "cuda_tools", "bin")

ARCH_CHECK_VA = 0x180017E9A
ARCH_OK_VA = 0x180017F2B
ARCH_CHECK_BYTES = bytes.fromhex("8d88c0feffff")  # lea ecx,[rax-0x140]
FATBIN_MAGIC = b"\x50\xed\x55\xba"
SIZE_TABLE_BIG = 0x1122590  # fatbin_01..07 的 qword 长度表 (0x18003D8C3 起逐个 mov r9,[rip+..])
SIZE_SMALL = [0x11250E0, 0x1127B88, 0x112A9A0, 0x112D2F0, 0x112F9E8, 0x1131388]  # fatbin_08..13 的 dword 长度


def run(*cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    err = "\n".join(l for l in (r.stdout + r.stderr).splitlines() if "knob" not in l)
    if r.returncode:
        sys.exit(f"失败: {' '.join(cmd)}\n{err}")


def build_fatbin(n):
    """返回 fatbin_NN 的新 fatbin 字节 (sm_86 + sm_120 两份 ELF)。有缓存。"""
    tag = f"fatbin_{n:02d}"
    ptx = os.path.join(FATBINS, f"{tag}.1.sm_120.ptx")
    out_ptx = os.path.join(WORK, f"{tag}.sm86.ptx")
    cubin = os.path.join(WORK, f"{tag}.sm86.cubin")
    fat = os.path.join(WORK, f"{tag}.sm86.fatbin")
    if not os.path.exists(fat) or os.path.getmtime(fat) < os.path.getmtime(__file__.replace("build_dll", "rewrite")):
        src, _ = rewrite(open(ptx, encoding="utf-8").read())
        open(out_ptx, "w", encoding="utf-8", newline="\n").write(src)
        run(os.path.join(BIN, "ptxas.exe"), "-arch=sm_86", out_ptx, "-o", cubin)
        orig = os.path.join(WORK, f"{tag}.1.sm_120.cubin")
        if not os.path.exists(orig):
            run(os.path.join(BIN, "cuobjdump.exe"), "-xelf", "all", os.path.join(FATBINS, f"{tag}.fatbin"))
            os.replace(f"{tag}.1.sm_120.cubin", orig)
        run(os.path.join(BIN, "fatbinary.exe"), "-64", "--compress=false", f"--create={fat}",
            f"--image3=kind=elf,sm=86,file={cubin}", f"--image3=kind=elf,sm=120,file={orig}")
        print(f"  {tag}: cubin {os.path.getsize(cubin):>9,} -> fatbin {os.path.getsize(fat):>9,}")
    return open(fat, "rb").read()


def align(x, a):
    return (x + a - 1) // a * a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acc", choices=["f16", "f32"], default="f16", help="FP8 mma 仿真的累加精度")
    ap.add_argument("--maxnreg", type=int, default=0, help="把 .maxnreg 压到不超过此值 (0=不改)")
    ap.add_argument("--maxnreg-floor", type=int, default=0, help="把 .maxnreg 抬到至少此值 (0=不改)")
    ap.add_argument("--enc", choices=["v1", "v2"], default="v1", help="FP16->FP8 编码实现")
    ap.add_argument("--dec", choices=["v1", "v2"], default="v1", help="FP8->FP16 解码实现")
    ap.add_argument("--out")
    args = ap.parse_args()
    global WORK
    rw.ACC["enc"], rw.ACC["dec"] = args.enc, args.dec
    if (args.enc, args.dec) != ("v1", "v1"):
        WORK = WORK + f"_e{args.enc}d{args.dec}"
    rw.ACC["mode"] = args.acc
    rw.ACC["maxnreg"] = args.maxnreg
    rw.ACC["maxnreg_floor"] = args.maxnreg_floor
    if args.maxnreg_floor:
        WORK = WORK + f"_f{args.maxnreg_floor}"
    if args.acc != "f16":
        WORK = WORK + "_" + args.acc
    if args.maxnreg:
        WORK = WORK + f"_r{args.maxnreg}"
    args.out = args.out or os.path.join(WORK, "nvngx_dlssnr.dll")
    os.makedirs(WORK, exist_ok=True)
    os.chdir(WORK)

    data = bytearray(open(SRC_DLL, "rb").read())
    assert hashlib.sha256(data).hexdigest() == SRC_SHA, "源 dll 不是已知的 310.8.0 版本"
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
    for n, off in enumerate(fat_offs, 1):
        fb = build_fatbin(n)
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
    new_sizes = [len(build_fatbin(n)) for n in range(1, 16)]
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


if __name__ == "__main__":
    main()
