"""PTX 数据流分析: FP8 编码结果 (cvt.satfinite.e4m3x2) 最终流向哪里。

沿 mov 拷贝 / 打包 (mov.b32 %r,{%rs,%rs}) / 拆包 (mov.b32 {%rs,%rs},%r) / prmt 追到终端使用者，
按类别统计: mma 操作数 (A/B)、st.shared、st.global、shfl、解码 (cvt.f16x2.e4m3x2) 等。
用来判断能否把 "编码->马上被 mma 解码" 融合成 f16 伪量化。

python -m tools.ptx86.flow fatbins/fatbin_01.1.sm_120.ptx [--kernel 名字子串]
"""
import argparse
import collections
import re

REG = re.compile(r"%[a-z]+\d+")
PASS_OPS = ("mov.b32", "mov.u32", "mov.b16", "mov.u16", "prmt.b32", "selp.b32", "selp.b16")


def functions(src):
    for m in re.finditer(r"\.(?:visible\s+)?\.?(?:entry|func)\s+(?:\([^)]*\)\s*)?(\w+)", src):
        start = src.find("{", m.end())
        depth, i = 0, start
        while i < len(src):
            c = src[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        yield m.group(1), src[start:i]


def statements(body):
    for s in re.split(r";", body):
        s = s.strip().lstrip("{").strip()
        if s and not s.startswith(".reg") and not s.startswith("//"):
            s = re.sub(r"^\$\w+:\s*", "", s)
            if s:
                yield " ".join(s.split())


def dst_src(s):
    op = s.split()[0]
    rest = s[len(op):].strip()
    if op.startswith("@"):
        op, rest = rest.split()[0], rest[len(rest.split()[0]):].strip()
    if op.startswith(("st.", "red.", "atom.", "mma.", "bar", "cp.", "mbarrier", "ret", "bra", "setp")) and not op.startswith("mma"):
        return op, [], REG.findall(rest)
    if op.startswith("mma"):
        groups = re.findall(r"\{([^}]*)\}", rest)
        d = REG.findall(groups[0]) if groups else []
        return op, d, [r for g in groups[1:] for r in REG.findall(g)]
    if rest.startswith("{") and "}" in rest:
        g = rest[1:rest.index("}")]
        return op, REG.findall(g), REG.findall(rest[rest.index("}") + 1:])
    regs = REG.findall(rest)
    return op, regs[:1], regs[1:]


def analyze(name, body):
    stmts = list(statements(body))
    uses = collections.defaultdict(list)
    for idx, s in enumerate(stmts):
        op, d, srcs = dst_src(s)
        for r in set(srcs):
            uses[r].append((idx, op, d, s))
    term = collections.Counter()
    for s in stmts:
        if "cvt.rn.satfinite.e4m3x2" not in s:
            continue
        _, d, _ = dst_src(s)
        seen, todo = set(), list(d)
        while todo:
            r = todo.pop()
            if r in seen:
                continue
            seen.add(r)
            for idx, op, dd, st in uses.get(r, []):
                if op.startswith(PASS_OPS):
                    todo += dd
                elif op.startswith("mma"):
                    groups = re.findall(r"\{([^}]*)\}", st)
                    role = "A" if r in REG.findall(groups[1]) else "B" if r in REG.findall(groups[2]) else "C"
                    term[f"mma.{role}:{op.split('.')[4] if len(op.split('.')) > 4 else ''}"] += 1
                elif op.startswith("st.shared") or op.startswith("st.") and "shared" in op:
                    term["st.shared"] += 1
                elif op.startswith("st."):
                    term[op.split(".")[0] + "." + op.split(".")[1]] += 1
                elif op.startswith("shfl"):
                    term["shfl"] += 1
                    todo += dd
                elif op.startswith("cvt.rn.f16x2.e4m3x2"):
                    term["decode"] += 1
                else:
                    term[op] += 1
    return term


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ptx")
    ap.add_argument("--kernel", default="")
    a = ap.parse_args()
    src = open(a.ptx, encoding="utf-8").read()
    total = collections.Counter()
    for name, body in functions(src):
        if a.kernel and a.kernel not in name:
            continue
        t = analyze(name, body)
        if t:
            total += t
            if a.kernel:
                print(name, dict(t.most_common()))
    print("合计:", dict(total.most_common()))


if __name__ == "__main__":
    main()
