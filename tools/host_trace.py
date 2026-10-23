"""Read-only PE/x64 navigation for local research; outputs are not vendored disassembly.

python -m tools.host_trace function 0x18001dac0 --limit 120
python -m tools.host_trace refs 0x18001dac0
python -m tools.host_trace refs 0x1800e00d8 --radius 32
"""
import argparse
import bisect
import hashlib
import json
from pathlib import Path

import capstone
from capstone import x86
import pefile


class Image:
    def __init__(self, path):
        self.path = Path(path)
        self.blob = self.path.read_bytes()
        self.pe = pefile.PE(data=self.blob, fast_load=True)
        self.pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_EXCEPTION']])
        self.base = self.pe.OPTIONAL_HEADER.ImageBase
        self.functions = [(self.base + e.struct.BeginAddress, self.base + e.struct.EndAddress)
                          for e in self.pe.DIRECTORY_ENTRY_EXCEPTION]
        self.starts = [s for s, _ in self.functions]
        self.md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
        self.md.detail = True
        self.md.skipdata = True

    def function(self, address):
        index = bisect.bisect_right(self.starts, address) - 1
        if index >= 0 and self.functions[index][0] <= address < self.functions[index][1]:
            return self.functions[index]
        raise ValueError(f"No exception-table function contains {address:#x}")

    def instructions(self, start, end):
        return self.md.disasm(self.pe.get_data(start - self.base, end - start), start)

    def annotation(self, ins):
        notes = []
        if ins.id == 0:
            return notes
        for op in ins.operands:
            if op.type == x86.X86_OP_MEM and op.mem.base == x86.X86_REG_RIP:
                address = ins.address + ins.size + op.mem.disp
                note = f"RIP->{address:#x}"
                try:
                    data = self.pe.get_data(address - self.base, 120).split(b"\0", 1)[0]
                    if data and all(32 <= v < 127 for v in data):
                        note += " " + repr(data.decode("ascii"))
                except Exception:
                    pass
                notes.append(note)
        return notes

    def references(self, target, radius=0):
        for section in self.pe.sections:
            if not section.Characteristics & 0x20000000:
                continue
            for ins in self.instructions(self.base + section.VirtualAddress,
                                         self.base + section.VirtualAddress + section.SizeOfRawData):
                if ins.id == 0:
                    continue
                for op in ins.operands:
                    address = None
                    if op.type == x86.X86_OP_MEM and op.mem.base == x86.X86_REG_RIP:
                        address = ins.address + ins.size + op.mem.disp
                    elif op.type == x86.X86_OP_IMM and ins.mnemonic in ("call", "jmp"):
                        address = op.imm
                    if address is not None and abs(address - target) <= radius:
                        yield ins, address


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("function", "refs"))
    parser.add_argument("address", type=lambda v: int(v, 0))
    parser.add_argument("--dll", type=Path, default=Path(__file__).resolve().parents[1] / "dlss5/nvngx_dlssnr.dll")
    parser.add_argument("--limit", type=int, default=250)
    parser.add_argument("--from-address", type=lambda v: int(v, 0))
    parser.add_argument("--radius", type=int, default=0)
    args = parser.parse_args()
    image = Image(args.dll)
    print(json.dumps({"source": str(args.dll), "sha256": hashlib.sha256(image.blob).hexdigest(),
                      "image_base": hex(image.base)}))
    if args.command == "function":
        start, end = image.function(args.address)
        print(f"function {start:#x}..{end:#x}")
        rows = image.instructions(start, end)
        if args.from_address is not None:
            rows = (ins for ins in rows if ins.address >= args.from_address)
        for index, ins in enumerate(rows):
            if index >= args.limit:
                break
            print(f"{ins.address:#x}: {ins.mnemonic:8s} {ins.op_str:48s} {'; '.join(image.annotation(ins))}")
    else:
        for ins, target in image.references(args.address, args.radius):
            try:
                function = image.function(ins.address)
            except ValueError:
                function = (0, 0)
            print(f"{ins.address:#x}: {ins.mnemonic} {ins.op_str} -> {target:#x}; function {function[0]:#x}..{function[1]:#x}")


if __name__ == "__main__":
    main()
