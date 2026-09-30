"""编码->mma 融合 (函数级 PTX 变换)，在 rewrite.RULES 之前跑。

原版: f16 激活 --F2FP(1 条)--> e4m3 --打包/拷贝--> QMMA 直接吃 FP8。
逐条仿真: 编码 ~15 条 + mma 前再解码 ~5 条，且多占寄存器。

做法: 给 "值来自 FP8 编码" 的寄存器配一个 f16 影子，影子 = decode(该寄存器)，恒等成立:
  %rsN (2 个 e4m3)  -> %rsN_q        (f16x2)
  %rN  (4 个 e4m3)  -> %rN_ql, %rN_qh (字节 0,1 / 2,3 的 f16x2)
  编码    cvt.satfinite.e4m3x2 %rs, %x  -> 另算 %rs_q = fq(%x) = decode(encode(%x)) (伪量化，逐位相等)
  打包    mov.b32 %r, {%rsA, %rsB}      -> %r_ql = %rsA_q, %r_qh = %rsB_q
  拆包    mov.b32 {%rsA, %rsB}, %r      -> %rsA_q = %r_ql, %rsB_q = %r_qh
  拷贝 / selp / 立即数                    -> 影子同步做 (立即数在 Python 里解码成常量)
影子只给 "所有定义点都能同步出影子" 的寄存器 (最大不动点，循环里的 phi 也成立)。
mma 的 A/B 操作数若有影子直接用影子，省掉解码；没人再读的编码由 ptxas 死代码消除。
"""
import collections
import re

from . import rewrite as rw

REG = r"%r(?:s)?\d+"
PRED = r"((?:@!?%p\d+\s+)?)"
IMM = r"(?:-?\d+|0[xX][0-9a-fA-F]+)"

# 允许出现在影子链上的定义形式: (类别, 正则)。组 1 恒为谓词。
DEFS = [
    ("enc", re.compile(PRED + r"cvt\.rn\.satfinite\.e4m3x2\.f16x2\s+(%rs\d+),\s*(%r\d+)\s*;")),
    ("pack", re.compile(PRED + r"mov\.b32\s+(%r\d+),\s*\{\s*(%rs\d+),\s*(%rs\d+)\s*\}\s*;")),
    ("unpack", re.compile(PRED + r"mov\.b32\s+\{\s*(%rs\d+),\s*(%rs\d+)\s*\},\s*(%r\d+)\s*;")),
    ("copy", re.compile(PRED + r"mov\.(?:b32|u32|b16|u16)\s+(%rs?\d+),\s*(%rs?\d+)\s*;")),
    ("imm", re.compile(PRED + r"mov\.(?:b32|u32|b16|u16)\s+(%rs?\d+),\s*(" + IMM + r")\s*;")),
    ("selp", re.compile(PRED + r"selp\.(?:b32|b16)\s+(%rs?\d+),\s*(%rs?\d+|" + IMM + r"),\s*(%rs?\d+|" + IMM
                        + r"),\s*(%p\d+)\s*;")),
]
MMA = re.compile(PRED + r"mma\.sync\.aligned\.m16n8k32\.row\.col\.f16\.e4m3\.e4m3\.f16\s*"
                 r"\{([^}]*)\}\s*,\s*\{([^}]*)\}\s*,\s*\{([^}]*)\}\s*,\s*\{([^}]*)\}\s*;")
DEC = re.compile(PRED + r"cvt\.rn\.f16x2\.e4m3x2\s+(%r\d+),\s*(%rs\d+)\s*;")
ANY_DST = re.compile(r"^(?:@!?%p\d+\s+)?[a-z][\w.:]*\s+(\{[^}]*\}|%\w+)")


# ---------------------------------------------------------------- 常量解码

def _e4m3_to_f16_bits(b):
    s, e, m = b >> 7, (b >> 3) & 0xF, b & 7
    if e == 0:
        v = m / 512.0
    else:
        v = (1 + m / 8.0) * 2.0 ** (e - 7)
    import struct
    return struct.unpack("<H", struct.pack("<e", -v if s else v))[0]


def _imm_shadow(val, wide):
    """立即数的影子: 16 位 -> 1 个 f16x2；32 位 -> (lo, hi)"""
    v = int(val, 0) & (0xFFFFFFFF if wide else 0xFFFF)
    f = [_e4m3_to_f16_bits((v >> (8 * i)) & 0xFF) for i in range(4 if wide else 2)]
    lo = f[0] | f[1] << 16
    return (f"0x{lo:08X}", f"0x{f[2] | f[3] << 16:08X}") if wide else f"0x{lo:08X}"


# ---------------------------------------------------------------- 伪量化

def fq(dst, src):
    """dst = decode(encode(src))，f16x2 两半同时，逐位等于 e4m3 RNE satfinite 往返。
    正规段: 幅值位 +0x3F+lsb 后清低 7 位 = 尾数 RNE 到 3 位 (进位自然进指数)
    次正规段 (<2^-6): |x|+2.0-2.0，2.0 的 ulp 恰为 2^-9"""
    return (
        "{ .reg .b32 %f_a, %f_l, %f_r, %f_s, %f_m, %f_k;\n"
        f"and.b32 %f_a, {src}, 0x7FFF7FFF;\n"
        "mov.b32 %f_k, 0x5F005F00;\n"
        "min.f16x2 %f_a, %f_a, %f_k;\n"
        "shr.u32 %f_l, %f_a, 7;\n"
        "and.b32 %f_l, %f_l, 0x00010001;\n"
        "add.u32 %f_r, %f_a, 0x003F003F;\n"
        "add.u32 %f_r, %f_r, %f_l;\n"
        "and.b32 %f_r, %f_r, 0xFF80FF80;\n"
        "mov.b32 %f_k, 0x40004000;\n"
        "add.rn.f16x2 %f_s, %f_a, %f_k;\n"
        "sub.rn.f16x2 %f_s, %f_s, %f_k;\n"
        "mov.b32 %f_k, 0x24002400;\n"
        "set.lt.u32.f16x2 %f_m, %f_a, %f_k;\n"
        "and.b32 %f_s, %f_s, %f_m;\n"
        "not.b32 %f_m, %f_m;\n"
        "and.b32 %f_r, %f_r, %f_m;\n"
        "or.b32 %f_r, %f_r, %f_s;\n"
        f"and.b32 %f_s, {src}, 0x80008000;\n"
        f"or.b32 {dst}, %f_r, %f_s;\n"
        "}\n"
    )


# ---------------------------------------------------------------- 分析

def _q(r):
    return (f"{r}_ql", f"{r}_qh") if not r.startswith("%rs") else (f"{r}_q",)


def _stmts(body):
    for s in body.split(";"):
        s = " ".join(s.split()).lstrip("{").strip()
        s = re.sub(r"^(\$\w+:\s*)+", "", s)
        if s and not s.startswith("."):
            yield s + ";"


def _dsts(s):
    m = ANY_DST.match(s)
    if not m or s.split()[0].lstrip("@").startswith(("st.", "bra", "ret", "bar", "red.", "mbarrier", "cp.", "fence")):
        return []
    op = s.split()[1] if s.startswith("@") else s.split()[0]
    if op.startswith(("setp", "st.", "bra")):
        return []
    return re.findall(r"%r(?:s)?\d+\b", m.group(1))


def shadow_set(body):
    """最大不动点: 所有定义都是允许形式、且源也在集合里的寄存器；再与 "从编码可达" 求交。"""
    defs = collections.defaultdict(list)
    for s in _stmts(body):
        for d in _dsts(s):
            defs[d].append(s)
    kinds = {}
    for r, ss in defs.items():
        ks = []
        for s in ss:
            for kind, rx in DEFS:
                m = rx.fullmatch(s)
                if m and r in m.groups()[1:]:
                    ks.append((kind, m))
                    break
            else:
                ks = None
                break
        if ks:
            kinds[r] = ks

    def srcs(kind, m):
        g = m.groups()[1:]
        if kind == "pack":
            return [g[1], g[2]]
        if kind == "unpack":
            return [g[2]]
        if kind == "copy":
            return [g[1]]
        if kind == "selp":
            return [x for x in g[1:3] if x.startswith("%")]
        return []

    S = set(kinds)
    changed = True
    while changed:
        changed = False
        for r in list(S):
            for kind, m in kinds[r]:
                if kind == "copy" and (m.group(2).startswith("%rs") != m.group(3).startswith("%rs")):
                    S.discard(r); changed = True; break
                if any(x not in S for x in srcs(kind, m)):
                    S.discard(r); changed = True; break
    # 只保留从编码出发可达的
    fwd = collections.defaultdict(set)
    seeds = set()
    for r in S:
        for kind, m in kinds[r]:
            if kind == "enc":
                seeds.add(r)
            for x in srcs(kind, m):
                fwd[x].add(r)
    reach, todo = set(), list(seeds)
    while todo:
        r = todo.pop()
        if r in reach:
            continue
        reach.add(r)
        todo += [x for x in fwd[r] if x in S]
    return reach


# ---------------------------------------------------------------- 改写

def _pfx(pred):
    return pred.strip() + " " if pred and pred.strip() else ""


def _shadow_of_def(kind, m, S):
    pred = m.group(1)
    g = m.groups()[1:]
    p = _pfx(pred)
    if kind == "enc":
        return rw._guard(pred, fq(_q(g[0])[0], g[1])) if g[0] in S else ""
    if kind == "pack":
        if g[0] not in S:
            return ""
        lo, hi = _q(g[0])
        return f"{p}mov.b32 {lo}, {_q(g[1])[0]};\n{p}mov.b32 {hi}, {_q(g[2])[0]};\n"
    if kind == "unpack":
        out = ""
        for dst, half in ((g[0], 0), (g[1], 1)):
            if dst in S:
                out += f"{p}mov.b32 {_q(dst)[0]}, {_q(g[2])[half]};\n"
        return out
    if kind == "copy":
        if g[0] not in S:
            return ""
        return "".join(f"{p}mov.b32 {a}, {b};\n" for a, b in zip(_q(g[0]), _q(g[1])))
    if kind == "imm":
        if g[0] not in S:
            return ""
        wide = not g[0].startswith("%rs")
        v = _imm_shadow(g[1], wide)
        return "".join(f"{p}mov.b32 {a}, {b};\n" for a, b in zip(_q(g[0]), v if wide else (v,)))
    if kind == "selp":
        if g[0] not in S:
            return ""
        wide = not g[0].startswith("%rs")

        def side(x):
            if x.startswith("%"):
                return _q(x)
            v = _imm_shadow(x, wide)
            return v if wide else (v,)
        return "".join(f"{p}selp.b32 {d}, {a}, {b}, {g[3]};\n"
                       for d, a, b in zip(_q(g[0]), side(g[1]), side(g[2])))
    return ""


def _mma(m, S):
    pred = m.group(1)
    d, a, b, c = (rw._ops(m.group(i)) for i in range(2, 6))
    if not any(r in S for r in a + b):
        return None
    ops = {}
    body = "{ .reg .b32 %m_a<8>, %m_c<4>, %m_t<2>;\n"
    for reg, (lo, hi) in zip(a + b, [("%m_a0", "%m_a2"), ("%m_a1", "%m_a3"), ("%m_a4", "%m_a6"),
                                      ("%m_a5", "%m_a7"), ("%m_c0", "%m_c1"), ("%m_c2", "%m_c3")]):
        if reg in S:
            ops[lo], ops[hi] = _q(reg)
        else:
            body += rw._unpack4(reg, lo, hi)
            ops[lo], ops[hi] = lo, hi
    A = [ops[f"%m_a{i}"] for i in range(8)]
    B = [ops[f"%m_c{i}"] for i in range(4)]
    if rw.ACC["mode"] == "f32":
        body += (
            ".reg .b16 %m_h<4>; .reg .f32 %m_f<4>;\n"
            f"mov.b32 {{%m_h0, %m_h1}}, {c[0]};\nmov.b32 {{%m_h2, %m_h3}}, {c[1]};\n"
            "cvt.f32.f16 %m_f0, %m_h0;\ncvt.f32.f16 %m_f1, %m_h1;\ncvt.f32.f16 %m_f2, %m_h2;\ncvt.f32.f16 %m_f3, %m_h3;\n"
            f"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%m_f0, %m_f1, %m_f2, %m_f3}}, "
            f"{{{A[0]}, {A[1]}, {A[2]}, {A[3]}}}, {{{B[0]}, {B[1]}}}, {{%m_f0, %m_f1, %m_f2, %m_f3}};\n"
            f"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%m_f0, %m_f1, %m_f2, %m_f3}}, "
            f"{{{A[4]}, {A[5]}, {A[6]}, {A[7]}}}, {{{B[2]}, {B[3]}}}, {{%m_f0, %m_f1, %m_f2, %m_f3}};\n"
            f"cvt.rn.f16x2.f32 {d[0]}, %m_f1, %m_f0;\ncvt.rn.f16x2.f32 {d[1]}, %m_f3, %m_f2;\n}}\n"
        )
    else:
        body += (
            f"mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {{%m_t0, %m_t1}}, "
            f"{{{A[0]}, {A[1]}, {A[2]}, {A[3]}}}, {{{B[0]}, {B[1]}}}, {{{c[0]}, {c[1]}}};\n"
            f"mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {{{d[0]}, {d[1]}}}, "
            f"{{{A[4]}, {A[5]}, {A[6]}, {A[7]}}}, {{{B[2]}, {B[3]}}}, {{%m_t0, %m_t1}};\n}}\n"
        )
    return rw._guard(pred, body)


def fuse_function(body, stats):
    S = shadow_set(body)
    if not S:
        return body
    stats["shadow_regs"] += len(S)

    def on_def(kind):
        def f(m):
            extra = _shadow_of_def(kind, m, S)
            if extra:
                stats[f"shadow_{kind}"] += 1
            return m.group(0) + "\n" + extra if extra else m.group(0)
        return f

    for kind, rx in DEFS:
        body = rx.sub(on_def(kind), body)

    def on_mma(m):
        r = _mma(m, S)
        if r is None:
            return m.group(0)
        stats["mma_fused"] += 1
        return r
    body = MMA.sub(on_mma, body)

    def on_dec(m):
        if m.group(3) not in S:
            return m.group(0)
        stats["decode_fused"] += 1
        return f"{_pfx(m.group(1))}mov.b32 {m.group(2)}, {_q(m.group(3))[0]};"
    body = DEC.sub(on_dec, body)

    names = sorted(n for r in S for n in _q(r))
    decl = "".join(f".reg .b32 {', '.join(names[i:i + 16])};\n" for i in range(0, len(names), 16))
    i = body.index("{") + 1
    return body[:i] + "\n" + decl + body[i:]


def apply(src):
    stats = collections.Counter()
    out, pos = [], 0
    for m in re.finditer(r"\.(?:entry|func)\s+(?:\([^)]*\)\s*)?\w+", src):
        start = src.find("{", m.end())
        if start < pos:
            continue
        depth, i = 0, start
        while True:
            c = src[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        out.append(src[pos:start])
        out.append(fuse_function(src[start:i + 1], stats))
        pos = i + 1
    out.append(src[pos:])
    return "".join(out), stats
