"""地址求值器 —— 把 PTX 的地址算式在 tid/ctaid = 0 下求值。

融合 kernel 只收一个权重指针（param_0+16），块内各张量的起始偏移
被折进地址算术里，且与线程索引混在一起。把索引置零后常量部分就浮出来了，
那就是打包布局的边界。
"""
import re, io

ZERO_SREGS = ("%tid.x", "%tid.y", "%tid.z", "%ctaid.x", "%ctaid.y", "%ctaid.z",
              "%laneid", "%warpid", "%nctaid.x", "%nctaid.y", "%nctaid.z")
CONST_SREGS = {"WARP_SZ": 32, "%ntid.x": 32, "%ntid.y": 8, "%ntid.z": 1}


def eval_kernel(lines, weight_param_off=16):
    """返回 {字节偏移: 示例指令}，只含能在 tid=0 下解出常量的访问。"""
    R = {}                       # reg -> int | None
    wp = None
    for l in lines:
        m = re.search(r"ld\.param\.b64 (%rd\d+), \[\w+_param_0\+" + str(weight_param_off) + r"\]", l)
        if m:
            wp = m.group(1)
            R[wp] = 0
            break
    if wp is None:
        return {}, None

    def g(tok):
        tok = tok.strip()
        if tok in R:
            return R[tok]
        if tok in ZERO_SREGS:
            return 0
        if tok in CONST_SREGS:
            return CONST_SREGS[tok]
        if re.fullmatch(r'-?\d+', tok):
            return int(tok)
        if tok.startswith("0f"):
            return None
        return None

    BIN = {
        "add": lambda a, b: a + b, "sub": lambda a, b: a - b,
        "mul.lo": lambda a, b: (a * b) & 0xFFFFFFFF if abs(a * b) < 2**63 else a * b,
        "mul.wide": lambda a, b: a * b,
        "shl": lambda a, b: a << (b & 31), "shr": lambda a, b: a >> (b & 31),
        "and": lambda a, b: a & b, "or": lambda a, b: a | b, "xor": lambda a, b: a ^ b,
        "min": min, "max": max,
    }
    hits = {}
    for _ in range(6):                       # 多轮传播，处理乱序定值
        for l in lines:
            t = l.strip()
            m = re.match(r'(mov|cvt)\.[\w.]+\s+(%\w+),\s*([^;]+);', t)
            if m:
                v = g(m.group(3))
                if v is not None:
                    R[m.group(2)] = v
                continue
            m = re.match(r'([a-z]+(?:\.lo|\.wide)?)\.[\w.]*\s+(%\w+),\s*([^,]+),\s*([^,;]+);', t)
            if m and m.group(1).split('.')[0] in ("add", "sub", "shl", "shr", "and",
                                                  "or", "xor", "min", "max") \
                    or (m and m.group(1) in ("mul.lo", "mul.wide")):
                op = m.group(1) if m.group(1) in BIN else m.group(1).split('.')[0]
                f = BIN.get(op)
                a, b = g(m.group(3)), g(m.group(4))
                if f and a is not None and b is not None:
                    R[m.group(2)] = f(a, b)
                continue
            m = re.match(r'mad\.lo\.\w+\s+(%\w+),\s*([^,]+),\s*([^,]+),\s*([^,;]+);', t)
            if m:
                a, b, c = g(m.group(2)), g(m.group(3)), g(m.group(4))
                if None not in (a, b, c):
                    R[m.group(1)] = a * b + c
    for l in lines:
        for m in re.finditer(r'\[(%rd\d+)(?:\+(-?\d+))?\]', l):
            base = R.get(m.group(1))
            if base is not None and "global" in l:
                hits.setdefault(base + int(m.group(2) or 0), l.strip()[:70])
    return hits, wp


def cmd_addr(args):
    from .core import db, FATBINS
    import os
    c = db()
    from .query import _pick_kernel, _ptx
    k = _pick_kernel(c, args.kernel)
    if not k:
        print("找不到 kernel")
        return
    lines = io.open(_ptx(k["fatbin"]), encoding="utf-8", errors="replace").read().split("\n")
    body = lines[k["line_start"] - 1:k["line_end"]]
    hits, wp = eval_kernel(body, args.param)
    print("%s  fatbin_%02d:%d-%d" % (k["name"], k["fatbin"], k["line_start"], k["line_end"]))
    print("权重指针 = param_0+%d -> %s" % (args.param, wp))
    if not hits:
        print("在 tid=0 下没有解出常量偏移")
        return
    print("\n%d 个常量偏移（tid/ctaid = 0）:" % len(hits))
    prev = None
    for off in sorted(hits):
        d = "" if prev is None else "   Δ=%s 元素" % "{:,}".format((off - prev) // 2)
        print("  %+11s B = %+10s 元素%s" % ("{:,}".format(off), "{:,}".format(off // 2), d))
        print("        %s" % hits[off])
        prev = off
