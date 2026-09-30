"""dllq core — 路径、编码解码、数据库 schema。"""
import os, sys, struct, sqlite3, subprocess, shutil

# ---- 就地写死的项目路径 ----
ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))   # 仓库根目录
DLL      = os.path.join(ROOT, "dlss5", "nvngx_dlssnr.dll")
FATBINS  = os.path.join(ROOT, "fatbins")
CUDABIN  = os.path.join(ROOT, "cuda_tools", "bin")
TOOLS    = os.path.join(ROOT, "tools")
DB       = os.path.join(TOOLS, "index.db")
NOTES    = os.path.join(TOOLS, "notes.jsonl")

FATBIN_MAGIC = b"\x50\xed\x55\xba"      # 0xBA55ED50


def out_utf8():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def rg():
    return shutil.which("rg") or shutil.which("rg.exe")


def cuda(tool):
    p = os.path.join(CUDABIN, tool + ".exe")
    return p if os.path.exists(p) else shutil.which(tool)


# ---------------- 数值编码/解码 ----------------

def i32_to_f32(i):
    return struct.unpack("<f", struct.pack("<i", i & 0xFFFFFFFF if i >= 0 else i))[0]

def u32_to_f32(u):
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]

def u16_to_f16(u):
    return struct.unpack("<e", struct.pack("<H", u & 0xFFFF))[0]

def f32_to_i32(x):
    return struct.unpack("<i", struct.pack("<f", x))[0]

def f16_to_u16(x):
    return struct.unpack("<H", struct.pack("<e", x))[0]


def decode_literal(raw):
    """把一个 PTX 整数字面量翻成所有可能的浮点读法。"""
    v = int(raw)
    u = v & 0xFFFFFFFF
    d = {"i32": v, "u32": u, "hex": "0x%08X" % u}
    try:
        d["f32"] = u32_to_f32(u)
    except Exception:
        d["f32"] = None
    # 当作 packed f16x2
    try:
        d["f16_lo"] = u16_to_f16(u & 0xFFFF)
        d["f16_hi"] = u16_to_f16((u >> 16) & 0xFFFF)
    except Exception:
        d["f16_lo"] = d["f16_hi"] = None
    return d


def encodings_of(value_str):
    """给一个人类写法的数（0.894531 / -4.0 / 0x3B28 / 1063583744），
    返回它在二进制里可能长什么样，用于反查。"""
    cands = []
    s = value_str.strip()
    try:
        if s.lower().startswith("0x"):
            u = int(s, 16)
            cands.append(("原始十六进制", "0x%X" % u))
            cands.append(("十进制(有符号)", str(u - (1 << 32) if u > 0x7FFFFFFF else u)))
            cands.append(("十进制(无符号)", str(u)))
            if u <= 0xFFFF:
                cands.append(("按 f16 读", "%.9g" % u16_to_f16(u)))
            cands.append(("按 f32 读", "%.9g" % u32_to_f32(u)))
            return cands
        x = float(s)
        if x == int(x) and abs(x) < 2**31 and "." not in s and "e" not in s.lower():
            iv = int(x)
            cands.append(("当作 i32 字面量 → f32", "%.9g" % i32_to_f32(iv)))
            cands.append(("当作 i32 → f16x2", "lo=%.9g hi=%.9g" % (
                u16_to_f16(iv & 0xFFFF), u16_to_f16((iv >> 16) & 0xFFFF))))
        i = f32_to_i32(x)
        cands.append(("f32 → i32 字面量", str(i)))
        cands.append(("f32 → hex", "0x%08X" % (i & 0xFFFFFFFF)))
        cands.append(("f32 → PTX 0f 形式", "0F%08X" % (i & 0xFFFFFFFF)))
        try:
            h = f16_to_u16(x)
            cands.append(("f16 → hex", "0x%04X" % h))
            cands.append(("f16 → i16 十进制", str(h - 65536 if h > 32767 else h)))
            cands.append(("f16 → f32 字面量(常见写法)", str(f32_to_i32(u16_to_f16(h)))))
        except Exception:
            pass
    except ValueError:
        pass
    return cands


# ---------------- 数据库 ----------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);

CREATE TABLE IF NOT EXISTS file(
  id INTEGER PRIMARY KEY, path TEXT UNIQUE, kind TEXT, size INTEGER, sha256 TEXT);

CREATE TABLE IF NOT EXISTS pe_section(
  file_id INT, name TEXT, vaddr INT, vsize INT, raddr INT, rsize INT);

CREATE TABLE IF NOT EXISTS pe_resource(
  file_id INT, rtype TEXT, rname TEXT, rlang TEXT, size INT, file_off INT);

CREATE TABLE IF NOT EXISTS fatbin(
  idx INT PRIMARY KEY, file_off INT, header_sz INT, payload_sz INT,
  ptx_path TEXT, ptx_lines INT, n_entries INT);

CREATE TABLE IF NOT EXISTS kernel(
  id INTEGER PRIMARY KEY, fatbin INT, name TEXT, line_start INT, line_end INT, n_lines INT);
CREATE INDEX IF NOT EXISTS ix_kernel_name ON kernel(name);

CREATE TABLE IF NOT EXISTS instr(
  kernel_id INT, op TEXT, cnt INT);
CREATE INDEX IF NOT EXISTS ix_instr_op ON instr(op);

CREATE TABLE IF NOT EXISTS konst(
  fatbin INT, kernel_id INT, line INT, reg TEXT, raw INTEGER,
  f32 REAL, f16lo REAL, f16hi REAL);
CREATE INDEX IF NOT EXISTS ix_konst_raw ON konst(raw);
CREATE INDEX IF NOT EXISTS ix_konst_f32 ON konst(f32);

CREATE TABLE IF NOT EXISTS strtab(
  file_id INT, off INT, cls TEXT, s TEXT);
CREATE INDEX IF NOT EXISTS ix_str_cls ON strtab(cls);

CREATE TABLE IF NOT EXISTS tmpl(
  kernel_id INT, fatbin INT, line INT, cfg TEXT, ints TEXT, bools TEXT);
CREATE INDEX IF NOT EXISTS ix_tmpl_cfg ON tmpl(cfg);

CREATE TABLE IF NOT EXISTS wrec(
  ord INT PRIMARY KEY, name TEXT, block INT, layer TEXT,
  bytes INT, params INT, off INT, tag INT);
CREATE INDEX IF NOT EXISTS ix_wrec_block ON wrec(block);
"""


def db(create=False):
    if create:
        os.makedirs(TOOLS, exist_ok=True)
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    if create:
        c.executescript(SCHEMA)
    return c
