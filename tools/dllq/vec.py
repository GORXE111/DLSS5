"""dllq vec — 可选的语义层。

只索引「值得语义检索」的小语料：分析笔记 + kernel 名 + 有意义的字符串。
**不索引 PTX 本体** —— 那部分靠结构索引，向量在那里没有区分度。
"""
import os, io, json, struct, sqlite3
from .core import db, TOOLS, NOTES

MODEL = "BAAI/bge-small-en-v1.5"
DIM = 384
VDB = os.path.join(TOOLS, "vec.db")

VSCHEMA = """
CREATE TABLE IF NOT EXISTS vec(
  id INTEGER PRIMARY KEY, src TEXT, ref TEXT, text TEXT, emb BLOB);
CREATE INDEX IF NOT EXISTS ix_vec_src ON vec(src);
"""



# 指令特征 → 能力描述。让 kernel 的语义检索基于「它做什么」而不是「它叫什么」。
CAPS = [
    ("normalization variance reduction rsqrt",      ["rsqrt.approx"], []),
    ("warp butterfly shuffle reduction",            ["shfl.sync"], []),
    ("tensor core matrix multiply mma fp16",        ["mma.sync"], []),
    ("fp8 e4m3 quantization dequantization",        ["cvt.rn.satfinite"], []),
    ("exponential transcendental exp2 log2",        ["ex2.approx"], []),
    ("gaussian noise box muller sin cos sqrt",      ["sin.approx", "cos.approx"], []),
    ("texture sampling image fetch",                ["tex.2d"], []),
    ("global atomic reduction accumulate",          ["red.global"], []),
    ("reciprocal division softmax denominator",     ["rcp.approx"], []),
    ("async bulk copy tma shared memory staging",   ["cp.async"], []),
    ("polynomial activation clamp silu gelu",       ["abs.f16x2"], []),
    ("barrier synchronization tile scheduling",     ["mbarrier"], []),
    ("surface write output store",                  ["sust", "st.global"], []),
]


def _caps(ops):
    """ops: {op: cnt}。返回该 kernel 的能力短语。"""
    out = []
    for phrase, need, _ in CAPS:
        if any(any(o.startswith(n) for o in ops) for n in need):
            out.append(phrase)
    return out


def _model():
    try:
        from fastembed import TextEmbedding
    except ImportError:
        raise SystemExit("未安装向量层。装:  python -m pip install fastembed")
    return TextEmbedding(model_name=MODEL)


def _pack(v):
    return struct.pack("<%df" % len(v), *v)


def _unpack(b):
    return struct.unpack("<%df" % (len(b) // 4), b)


def _norm(v):
    s = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / s for x in v]


def collect():
    """挑出值得语义检索的条目。每条都带引用。"""
    items = []
    if os.path.exists(NOTES):
        for i, l in enumerate(io.open(NOTES, encoding="utf-8")):
            l = l.strip()
            if not l:
                continue
            n = json.loads(l)
            ref = ",".join(n.get("cite") or []) or "note#%d" % i
            tags = " ".join(n.get("tag") or [])
            items.append(("note", ref, (n["text"] + " " + tags).strip()))
    c = db()
    for r in c.execute("SELECT k.id, k.name, k.fatbin, group_concat(i.op) ops FROM kernel k "
                       "LEFT JOIN instr i ON i.kernel_id = k.id GROUP BY k.id"):
        ops = set((r["ops"] or "").split(","))
        nm = r["name"].replace("_", " ")
        ref = "fatbin_%02d/%s" % (r["fatbin"], r["name"])
        # 每个能力单独成行 —— 挤在一个向量里会互相稀释
        for cap in _caps(ops):
            items.append(("kernel", ref, nm + ". " + cap))
    for r in c.execute("SELECT s, cls, off FROM strtab WHERE cls IN "
                       "('param','runtime','log','weights','path')"):
        items.append(("str", "%s@0x%X" % (r["cls"], r["off"]), r["s"]))
    return items


def build(verbose=True):
    items = collect()
    if verbose:
        print("待嵌入 %d 条" % len(items))
        from collections import Counter
        for k, n in Counter(i[0] for i in items).most_common():
            print("  %-8s %5d" % (k, n))
    m = _model()
    v = sqlite3.connect(VDB)
    v.executescript(VSCHEMA)
    v.execute("DELETE FROM vec")
    embs = m.embed([t for _, _, t in items])
    rows = []
    for (src, ref, text), e in zip(items, embs):
        rows.append((src, ref, text, _pack(_norm(list(map(float, e))))))
    v.executemany("INSERT INTO vec(src,ref,text,emb) VALUES(?,?,?,?)", rows)
    v.commit()
    if verbose:
        print("\n向量索引完成 -> %s  (%d 条 x %d 维)" % (VDB, len(rows), DIM))


def search(q, k=8, src=None):
    if not os.path.exists(VDB):
        return None
    m = _model()
    qv = _norm(list(map(float, next(iter(m.embed([q]))))))
    v = sqlite3.connect(VDB)
    v.row_factory = sqlite3.Row
    sql = "SELECT * FROM vec" + (" WHERE src=?" if src else "")
    out = []
    for r in v.execute(sql, (src,) if src else ()):
        e = _unpack(r["emb"])
        out.append((sum(a * b for a, b in zip(qv, e)), r["src"], r["ref"], r["text"]))
    out.sort(reverse=True)
    return out[:k]


def cmd_vec(args):
    if args.action == "build":
        build()
    elif args.action == "search":
        res = search(args.text or "", k=args.k, src=args.src)
        if res is None:
            print("向量索引不存在，先跑:  python -m dllq vec build")
            return
        for score, src, ref, text in res:
            print("  %.3f  [%s] %s" % (score, src, ref))
            print("         %s" % text[:130])
