"""Disassemble the x64 function containing an RVA of nvngx_dlssnr.dll (bounds from .pdata).
    python fn_at.py <rva hex> [--grep regex] [--dll path]
Lines matching --grep are printed with a marker; without it the whole function is printed."""
import argparse
import re

import capstone
import pefile

ap = argparse.ArgumentParser()
ap.add_argument("rva", nargs="+")
ap.add_argument("--grep")
ap.add_argument("--dll", default=r"E:\DLSS5\dlss5\nvngx_dlssnr.dll")
a = ap.parse_args()

pe = pefile.PE(a.dll, fast_load=True)
pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_EXCEPTION"]])
funcs = sorted((e.struct.BeginAddress, e.struct.EndAddress) for e in pe.DIRECTORY_ENTRY_EXCEPTION)
md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
base = pe.OPTIONAL_HEADER.ImageBase
for r in a.rva:
    rva = int(r, 16)
    begin, end = next(((b, e) for b, e in funcs if b <= rva < e), (rva - 0x80, rva + 0x80))
    # chained unwind entries split one function into several ranges: extend to neighbours that touch
    code = pe.get_data(begin, end - begin)
    print(f"=== function {base + begin:#x} .. {base + end:#x} (contains {base + rva:#x}, size {end - begin})")
    for ins in md.disasm(code, base + begin):
        line = f"{ins.address:#x}  {ins.mnemonic} {ins.op_str}"
        mark = ">>" if ins.address + ins.size == base + rva else "  "
        if a.grep is None or re.search(a.grep, line) or mark == ">>":
            print(mark, line)
