"""x86-64 反汇编 —— host 侧代码分析。

PTX 只能看到 GPU 侧。权重记录到 slot 的绑定、以及网络描述符的选择，
都在 host 的 x86 代码里。这个模块补上那一半。
"""
import os, re, struct
from .core import DLL, db

IMAGE_BASE = 0x180000000        # gist 用的基址，与本 DLL 的 OptionalHeader 一致


def _pe(blob):
    pe = struct.unpack_from("<I", blob, 0x3C)[0]
    nsec = struct.unpack_from("<H", blob, pe + 6)[0]
    optsz = struct.unpack_from("<H", blob, pe + 20)[0]
    magic = struct.unpack_from("<H", blob, pe + 24)[0]
    base = struct.unpack_from("<Q", blob, pe + 24 + 24)[0] if magic == 0x20B else \
        struct.unpack_from("<I", blob, pe + 24 + 28)[0]
    secs = []
    for i in range(nsec):
        b = pe + 24 + optsz + 40 * i
        name = blob[b:b + 8].rstrip(b"\0").decode("ascii", "replace")
        vsz, va, rsz, ra = struct.unpack_from("<IIII", blob, b + 8)
        chars = struct.unpack_from("<I", blob, b + 36)[0]
        secs.append(dict(name=name, va=va, vsz=vsz, ra=ra, rsz=rsz, exec=bool(chars & 0x20000000)))
    return base, secs


def load_image():
    blob = open(DLL, "rb").read()
    base, secs = _pe(blob)
    return blob, base, secs


def va2off(va, base, secs):
    rva = va - base
    for s in secs:
        if s["va"] <= rva < s["va"] + max(s["vsz"], s["rsz"]):
            return s["ra"] + (rva - s["va"])
    return None


def off2va(off, base, secs):
    for s in secs:
        if s["ra"] <= off < s["ra"] + s["rsz"]:
            return base + s["va"] + (off - s["ra"])
    return None


def disasm(va, n=60):
    from capstone import Cs, CS_ARCH_X86, CS_MODE_64
    blob, base, secs = load_image()
    off = va2off(va, base, secs)
    if off is None:
        raise SystemExit("地址 0x%X 不在任何节内（映像基址 0x%X）" % (va, base))
    md = Cs(CS_ARCH_X86, CS_MODE_64)
    md.detail = True
    out = []
    for i in md.disasm(blob[off:off + n * 16], va):
        out.append(i)
        if len(out) >= n:
            break
        if i.mnemonic in ("ret", "int3") and len(out) > 4:
            break
    return out, blob, base, secs


def _strings_map():
    """VA -> 字符串，用于给 lea 加注释。"""
    c = db()
    blob, base, secs = load_image()
    m = {}
    for r in c.execute("SELECT off, s FROM strtab"):
        va = off2va(r["off"], base, secs)
        if va:
            m[va] = r["s"]
    return m


def cmd_dis(args):
    va = int(args.addr, 16) if args.addr.lower().startswith("0x") else int(args.addr, 16)
    ins, blob, base, secs = disasm(va, args.n)
    smap = _strings_map() if args.strings else {}
    print("映像基址 0x%X   反汇编 0x%X  (%d 条)" % (base, va, len(ins)))
    for i in ins:
        note = ""
        # lea reg, [rip+disp] -> 目标 VA
        if i.mnemonic == "lea" and "rip" in i.op_str:
            m = re.search(r"\[rip \+ (0x[0-9a-f]+)\]|\[rip - (0x[0-9a-f]+)\]", i.op_str)
            if m:
                d = int(m.group(1), 16) if m.group(1) else -int(m.group(2), 16)
                tgt = i.address + i.size + d
                if tgt in smap:
                    note = '   ; "%s"' % smap[tgt][:60]
                else:
                    note = "   ; -> 0x%X" % tgt
        print("  0x%012X  %-8s %-42s%s" % (i.address, i.mnemonic, i.op_str, note))


def cmd_xref(args):
    """找出引用某个字符串的代码位置（扫 lea rip-relative）。"""
    from capstone import Cs, CS_ARCH_X86, CS_MODE_64
    c = db()
    blob, base, secs = load_image()
    rows = c.execute("SELECT off, s FROM strtab WHERE s LIKE ? LIMIT 40",
                     ("%" + args.text + "%",)).fetchall()
    if not rows:
        print("没有匹配的字符串")
        return
    targets = {}
    for r in rows:
        va = off2va(r["off"], base, secs)
        if va:
            targets[va] = r["s"]
    print("目标字符串 %d 条:" % len(targets))
    for va, s in list(targets.items())[:10]:
        print("  0x%012X  %s" % (va, s[:70]))
    md = Cs(CS_ARCH_X86, CS_MODE_64)
    hits = []
    for sec in secs:
        if not sec["exec"]:
            continue
        data = blob[sec["ra"]:sec["ra"] + sec["rsz"]]
        secva = base + sec["va"]
        print("\n扫描节 %s  (%s 字节)…" % (sec["name"], "{:,}".format(len(data))))
        for i in md.disasm(data, secva):
            if i.mnemonic != "lea" or "rip" not in i.op_str:
                continue
            m = re.search(r"\[rip \+ (0x[0-9a-f]+)\]|\[rip - (0x[0-9a-f]+)\]", i.op_str)
            if not m:
                continue
            d = int(m.group(1), 16) if m.group(1) else -int(m.group(2), 16)
            t = i.address + i.size + d
            if t in targets:
                hits.append((i.address, targets[t]))
    print("\n引用点 %d 处:" % len(hits))
    for a, s in hits[:40]:
        print("  0x%012X  ->  %s" % (a, s[:60]))
