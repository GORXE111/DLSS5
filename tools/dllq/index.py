"""dllq index — 从 DLL / fatbin / PTX / 权重资源抽取结构，写入 SQLite。"""
import os, re, io, struct, hashlib, collections
from .core import DLL, FATBINS, ROOT, FATBIN_MAGIC, db, decode_literal

# ---------------- PE ----------------

def parse_pe(data):
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe:pe + 4] != b"PE\0\0":
        raise ValueError("not a PE file")
    nsec = struct.unpack_from("<H", data, pe + 6)[0]
    optsz = struct.unpack_from("<H", data, pe + 20)[0]
    magic = struct.unpack_from("<H", data, pe + 24)[0]
    ddoff = pe + 24 + (112 if magic == 0x20B else 96)
    secs = []
    for i in range(nsec):
        b = pe + 24 + optsz + 40 * i
        name = data[b:b + 8].rstrip(b"\0").decode("ascii", "replace")
        vsz, va, rsz, ra = struct.unpack_from("<IIII", data, b + 8)
        secs.append(dict(name=name, vaddr=va, vsize=vsz, raddr=ra, rsize=rsz))
    rsrc_rva = struct.unpack_from("<I", data, ddoff + 16)[0]
    return secs, rsrc_rva


def rva2off(secs, rva):
    for s in secs:
        if s["vaddr"] <= rva < s["vaddr"] + max(s["vsize"], s["rsize"]):
            return s["raddr"] + (rva - s["vaddr"])
    return None


def parse_resources(data, secs, rsrc_rva):
    base = rva2off(secs, rsrc_rva)
    if base is None:
        return []

    def ents(off):
        nn, ni = struct.unpack_from("<HH", data, off + 12)
        return [struct.unpack_from("<II", data, off + 16 + 8 * i) for i in range(nn + ni)]

    def nm(x):
        if x & 0x80000000:
            o = base + (x & 0x7FFFFFFF)
            ln = struct.unpack_from("<H", data, o)[0]
            return data[o + 2:o + 2 + ln * 2].decode("utf-16le", "replace")
        return str(x)

    out = []
    for tid, toff in ents(base):
        for nid, noff in ents(base + (toff & 0x7FFFFFFF)):
            for lid, loff in ents(base + (noff & 0x7FFFFFFF)):
                drva, dsz, _, _ = struct.unpack_from("<IIII", data, base + loff)
                out.append(dict(rtype=nm(tid), rname=nm(nid), rlang=nm(lid),
                                size=dsz, off=rva2off(secs, drva)))
    return out


# ---------------- fatbin 切分 ----------------

def carve_fatbins(data, outdir):
    os.makedirs(outdir, exist_ok=True)
    found, i, n = [], 0, 0
    while True:
        i = data.find(FATBIN_MAGIC, i)
        if i < 0:
            break
        try:
            _, ver, hs, fs = struct.unpack_from("<IHHQ", data, i)
        except Exception:
            i += 4
            continue
        if ver == 1 and 8 <= hs <= 64 and 0 < fs < 300 * 1024 * 1024 and i + hs + fs <= len(data):
            n += 1
            p = os.path.join(outdir, "fatbin_%02d.fatbin" % n)
            if not os.path.exists(p):
                open(p, "wb").write(data[i:i + hs + fs])
            found.append(dict(idx=n, off=i, header=hs, payload=fs, path=p))
            i += hs + fs
        else:
            i += 4
    return found


# ---------------- PTX ----------------

ENTRY_RE = re.compile(r"^\.visible \.entry ([A-Za-z0-9_$]+)\(")
IMM_RE = re.compile(
    r"^\s*\{?\s*(mov\.b32|add\.s32|sub\.s32|mul\.lo\.s32|mad\.lo\.s32|and\.b32|or\.b32|xor\.b32)"
    r"\s+(%r\d+)\s*,(?:[^,;]+,)*\s*(-?\d{3,});")
OP_RE = re.compile(r"(?:^|[{\s])([a-z][a-z0-9]*(?:\.[a-z0-9_:]+)+)\s+[^\s,;)]")


def scan_ptx(path):
    """返回 (entries, consts, per-kernel 指令计数)。"""
    entries, consts = [], []
    counts = collections.defaultdict(collections.Counter)
    cur, start, ln = None, 0, 0
    with io.open(path, encoding="utf-8", errors="replace") as f:
        for ln, line in enumerate(f, 1):
            m = ENTRY_RE.match(line)
            if m:
                if cur:
                    entries.append((cur, start, ln - 1))
                cur, start = m.group(1), ln
                continue
            if cur is None:
                continue
            mm = IMM_RE.match(line)
            if mm:
                consts.append((cur, ln, mm.group(2), int(mm.group(3))))
            om = OP_RE.search(line)
            if om:
                counts[cur][om.group(1)] += 1
        if cur:
            entries.append((cur, start, ln))
    return entries, consts, counts


CFG_RE = re.compile(r"\d+([A-Za-z][A-Za-z0-9_]*Config)I((?:L[ib]n?\d+E)+)")
CFG_NUM = re.compile(r"L([ib])(n?)(\d+)E")


def scan_templates(path):
    """从 .shared 符号的 C++ mangled 名里抽出模板配置整数 —— 形状的硬证据。"""
    out = []
    with io.open(path, encoding="utf-8", errors="replace") as f:
        for ln, line in enumerate(f, 1):
            if not line.startswith(".shared "):
                continue
            for m in CFG_RE.finditer(line):
                ii, bb = [], []
                for t, neg, v in CFG_NUM.findall(m.group(2)):
                    (ii if t == "i" else bb).append(-int(v) if neg else int(v))
                out.append((ln, m.group(1), ii, bb))
    return out


# ---------------- 字符串分类 ----------------

STR_RE = re.compile(rb"[\x20-\x7e]{5,200}")


def classify(s):
    if s.startswith("DLSSNR."):
        return "param"
    if s.startswith("cc_") or s.startswith("CC"):
        return "kernel"
    if "CG2R" in s:
        return "runtime"
    if "://" in s or re.match(r"^[A-Za-z]:/", s):
        return "path"
    if "_weights" in s:
        return "weights"
    if "%" in s and ("fail" in s.lower() or "error" in s.lower()):
        return "log"
    if s.startswith(".nv.") or s.startswith(".text.") or s.startswith(".rel"):
        return "elfsec"
    if re.match(r"^[A-Za-z_][A-Za-z0-9_:<>,$]{6,}$", s):
        return "symbol"
    return "other"


# ---------------- WEIGHTS_HT ----------------

def parse_weights(blob):
    recs, off = [], 8
    while off < len(blob) - 8:
        nl = struct.unpack_from("<Q", blob, off)[0]
        if not (1 <= nl <= 128):
            break
        name = blob[off + 8:off + 8 + nl].decode("ascii", "replace")
        p = off + 8 + nl
        A, B, C = struct.unpack_from("<QQQ", blob, p)
        tag = struct.unpack_from("<I", blob, p + 24)[0]
        pay = p + 28
        Z = struct.unpack_from("<I", blob, pay + C + 16)[0]
        recs.append(dict(name=name, bytes=C, params=Z, off=pay, tag=tag))
        off = pay + C + 20
    return recs


# ---------------- 主流程 ----------------

def build(verbose=True):
    def say(*a):
        if verbose:
            print(*a, flush=True)

    c = db(create=True)
    for t in ("file", "pe_section", "pe_resource", "fatbin", "kernel",
              "instr", "konst", "strtab", "wrec", "tmpl"):
        c.execute("DELETE FROM " + t)

    say("读取", DLL)
    data = open(DLL, "rb").read()
    sha = hashlib.sha256(data).hexdigest()
    c.execute("INSERT INTO file(path,kind,size,sha256) VALUES(?,?,?,?)",
              (DLL, "dll", len(data), sha))
    fid = c.execute("SELECT last_insert_rowid() i").fetchone()["i"]
    say("  SHA-256", sha)

    secs, rrva = parse_pe(data)
    c.executemany("INSERT INTO pe_section VALUES(?,?,?,?,?,?)",
                  [(fid, s["name"], s["vaddr"], s["vsize"], s["raddr"], s["rsize"]) for s in secs])
    say("  PE 段          ", len(secs))

    res = parse_resources(data, secs, rrva)
    c.executemany("INSERT INTO pe_resource VALUES(?,?,?,?,?,?)",
                  [(fid, r["rtype"], r["rname"], r["rlang"], r["size"], r["off"]) for r in res])
    say("  PE 资源        ", len(res))

    seen = {}
    for m in STR_RE.finditer(data):
        s = m.group().decode("ascii")
        if s not in seen:
            seen[s] = m.start()
    c.executemany("INSERT INTO strtab VALUES(?,?,?,?)",
                  [(fid, o, classify(s), s) for s, o in seen.items()])
    say("  字符串         ", len(seen))

    for r in res:
        if r["rtype"] == "10" and r["rname"] == "WEIGHTS_HT":
            blob = data[r["off"]:r["off"] + r["size"]]
            wp = os.path.join(ROOT, "WEIGHTS_HT.bin")
            if not os.path.exists(wp):
                open(wp, "wb").write(blob)
            wr = parse_weights(blob)
            rows = []
            for i, w in enumerate(wr):
                mm = re.match(r"block(\d+)\.(.*)", w["name"])
                rows.append((i, w["name"], int(mm.group(1)) if mm else -1,
                             mm.group(2) if mm else "", w["bytes"], w["params"],
                             w["off"], w["tag"]))
            c.executemany("INSERT INTO wrec VALUES(?,?,?,?,?,?,?,?)", rows)
            say("  权重记录       ", len(rows), " 参数合计",
                "{:,}".format(sum(w["params"] for w in wr)))

    fbs = carve_fatbins(data, FATBINS)
    del data
    say("  fatbin         ", len(fbs))
    say("")

    kid = 0
    for fb in fbs:
        ptx = os.path.join(FATBINS, "fatbin_%02d.1.sm_120.ptx" % fb["idx"])
        if not os.path.exists(ptx):
            c.execute("INSERT INTO fatbin VALUES(?,?,?,?,?,?,?)",
                      (fb["idx"], fb["off"], fb["header"], fb["payload"], None, 0, 0))
            say("  fatbin_%02d  (无 PTX，跳过 — 先跑 dllq extract)" % fb["idx"])
            continue
        ents, ks, counts = scan_ptx(ptx)
        with io.open(ptx, encoding="utf-8", errors="replace") as f:
            nl = sum(1 for _ in f)
        c.execute("INSERT INTO fatbin VALUES(?,?,?,?,?,?,?)",
                  (fb["idx"], fb["off"], fb["header"], fb["payload"], ptx, nl, len(ents)))
        kmap = {}
        for name, a, b in ents:
            kid += 1
            kmap[name] = kid
            c.execute("INSERT INTO kernel VALUES(?,?,?,?,?,?)",
                      (kid, fb["idx"], name, a, b, b - a + 1))
        c.executemany("INSERT INTO instr VALUES(?,?,?)",
                      [(kmap[k], op, n) for k, cc in counts.items() for op, n in cc.items()])
        rows = []
        for kname, ln, reg, raw in ks:
            d = decode_literal(raw)
            rows.append((fb["idx"], kmap.get(kname), ln, reg, raw,
                         d["f32"], d["f16_lo"], d["f16_hi"]))
        c.executemany("INSERT INTO konst VALUES(?,?,?,?,?,?,?,?)", rows)
        # 模板配置：按行号归属到 kernel
        krng = sorted([(a, b, kmap[n]) for n, a, b in ents])
        trows = []
        for ln, cfg, ii, bb in scan_templates(ptx):
            kid_ = None
            for a, b, k_ in krng:
                if a <= ln <= b:
                    kid_ = k_
                    break
            trows.append((kid_, fb["idx"], ln, cfg,
                          ",".join(map(str, ii)), ",".join(map(str, bb))))
        c.executemany("INSERT INTO tmpl VALUES(?,?,?,?,?,?)", trows)
        say("  fatbin_%02d  %8s 行  %3d entry  %7s 常数"
            % (fb["idx"], "{:,}".format(nl), len(ents), "{:,}".format(len(rows))))

    c.execute("INSERT OR REPLACE INTO meta VALUES('built', datetime('now'))")
    c.commit()
    say("")
    say("索引完成 ->", os.path.join(ROOT, "tools", "index.db"))
