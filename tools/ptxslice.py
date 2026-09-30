"""PTX 反向切片: 从某条指令的寄存器往回追定义链，打印决定它的全部指令 (到 param/特殊寄存器为止)。

python -m tools.ptxslice kernel.ptx --pattern "st.global" [--which 0] [--reg %rd12]
PTX 是 nvcc 产出的近似 SSA (循环里的寄存器可能多处定义，都会列出)。
"""
import argparse
import re

REG = re.compile(r"%[a-z]+\d+")


def statements(text):
    """按分号切成完整语句 (mma 这类跨行指令保持完整)，记起始行号"""
    out = []
    pos, line = 0, 1
    for part in text.split(";"):
        lead = len(part) - len(part.lstrip())
        start_line = line + part[:lead].count("\n")
        line += part.count("\n")
        s = " ".join(part.split())
        # 去掉内联块的花括号与标签前缀 (保留 mma 等指令里的 {..} 操作数)
        while s.startswith("{ ") or s == "{" or s.startswith("}"):
            s = s[1:].strip()
        s = re.sub(r"^(\$\w+:\s*)+", "", s)
        if s and not s.startswith((".reg", "//", ".param", ".maxnreg", ".visible", ".entry", ".shared", ".global",
                                   ".local", ".const", ")", "(")):
            out.append((start_line, s))
    return out


def dst_of(s):
    s2 = re.sub(r"^@!?%p\d+\s+", "", s)
    op = s2.split()[0] if s2.split() else ""
    if op.startswith(("st.", "bra", "ret", "bar.", "red.", "@", "mbarrier.arrive", "cp.", "fence")) or ":" in op:
        return []
    rest = s2[len(op):].strip()
    if rest.startswith("{"):
        return REG.findall(rest[: rest.find("}")])
    m = REG.match(rest)
    return [m.group(0)] if m else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ptx")
    ap.add_argument("--pattern", default="st.global")
    ap.add_argument("--which", type=int, default=0)
    ap.add_argument("--reg")
    ap.add_argument("--depth", type=int, default=60)
    a = ap.parse_args()
    st = statements(open(a.ptx, encoding="utf-8").read())
    defs = {}
    for i, (ln, s) in enumerate(st):
        for d in dst_of(s):
            defs.setdefault(d, []).append(i)
    hits = [i for i, (ln, s) in enumerate(st) if a.pattern in s]
    target = hits[a.which]
    ln, s = st[target]
    print(f"目标 (行 {ln}): {s}\n")
    seeds = [a.reg] if a.reg else REG.findall(s.split("[", 1)[1].split("]")[0]) if "[" in s else REG.findall(s)
    seen, order, todo = set(), [], [(r, 0) for r in seeds]
    while todo:
        r, dep = todo.pop()
        if r in seen or dep > a.depth:
            continue
        seen.add(r)
        for i in defs.get(r, []):
            if i >= target and len(defs[r]) > 1:
                continue
            order.append(i)
            _, s2 = st[i]
            srcs = [x for x in REG.findall(s2) if x not in dst_of(s2)]
            todo += [(x, dep + 1) for x in srcs]
    for i in sorted(set(order)):
        print(f"{st[i][0]:6d}  {st[i][1]}")


if __name__ == "__main__":
    main()
