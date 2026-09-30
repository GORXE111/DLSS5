"""把一个 kernel 的 PTX 压成 "阶段时间线": 按程序顺序列出关键事件并合并连续同类。

事件: 读权重 (地址源自权重指针参数，尽量算出常量偏移)、读/写激活、mma (按形状)、
      FP8 编解码、rsqrt/ex2/rcp 等特殊函数、shfl、bar.sync、分支标签。
python -m tools.ptxstages kernel.ptx --wparam 16
"""
import argparse
import re

from .ptxslice import REG, dst_of, statements


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ptx")
    ap.add_argument("--wparam", type=int, default=16, help="权重指针在参数块里的字节偏移")
    ap.add_argument("--inparam", type=int, default=0)
    ap.add_argument("--outparam", type=int, default=8)
    a = ap.parse_args()
    st = statements(open(a.ptx, encoding="utf-8").read())
    # 追踪每个寄存器 = (来源参数, 常量偏移或 None)
    origin = {}
    const = {}
    events = []
    for ln, s in st:
        m = re.match(r"ld\.param\.b64\s+(%rd\d+),\s*\[[%\w]+\+(\d+)\]", s)
        if m:
            origin[m.group(1)] = (int(m.group(2)), 0)
            continue
        m = re.match(r"(?:cvta\.to\.global\.u64|mov\.b64|mov\.u64)\s+(%rd\d+),\s*(%rd\d+)$", s)
        if m and m.group(2) in origin:
            origin[m.group(1)] = origin[m.group(2)]
            continue
        m = re.match(r"add\.s64\s+(%rd\d+),\s*(%rd\d+),\s*(%rd\d+|-?\d+)$", s)
        if m:
            a_, b_ = m.group(2), m.group(3)
            if a_ not in origin and b_ in origin:
                a_, b_ = b_, a_
            src = origin.get(a_)
            if src:
                # 另一边若是常量寄存器则累加，若是线程相关的寄存器则只保留常量部分 (近似)
                if not b_.startswith("%"):
                    off = src[1] + int(b_)
                else:
                    off = src[1] + const.get(b_, 0)
                origin[m.group(1)] = (src[0], off)
                continue
        m = re.match(r"mov\.(?:b64|u64|s64)\s+(%rd\d+),\s*(-?\d+)$", s)
        if m:
            const[m.group(1)] = int(m.group(2))
        m = re.match(r"(?:mul\.wide\.[su]32|cvt\.[su]64\.[su]32)\s+(%rd\d+),", s)
        if m and m.group(1) not in const:
            const[m.group(1)] = 0
        op = re.sub(r"^@!?%p\d+\s+", "", s).split()[0]
        ev = None
        if op.startswith(("ld.global", "ld.weak.global", "ld.relaxed.global", "ld.volatile.global")) or (
                op.startswith("ld.") and "global" in op):
            addr = s[s.find("[") + 1: s.find("]")]
            base = addr.split("+")[0].strip()
            imm = int(addr.split("+")[1]) if "+" in addr else 0
            src = origin.get(base)
            if src and src[0] == a.wparam:
                ev = f"W[{src[1] + imm if src[1] is not None else '?'}]"
            elif src and src[0] == a.inparam:
                ev = "IN"
            else:
                ev = "LDG?"
        elif op.startswith("st.global") or "st.global" in s:
            ev = "OUT"
        elif op.startswith("ld.shared") or op.startswith("ld.shared::cta"):
            ev = "lds"
        elif op.startswith("st.shared"):
            ev = "sts"
        elif op.startswith("mma"):
            ev = "mma." + ("fp8" if "e4m3" in op else "f16") + "." + op.split(".")[3]
        elif "e4m3x2.f16x2" in op:
            ev = "enc8"
        elif op.startswith("cvt.rn.f16x2.e4m3x2"):
            ev = "dec8"
        elif op.startswith(("rsqrt", "ex2", "rcp", "sqrt", "lg2", "tanh", "sin", "cos")):
            ev = op.split(".")[0]
        elif op.startswith("shfl"):
            ev = "shfl"
        elif op.startswith("bar.") or op.startswith("barrier"):
            ev = "bar"
        elif op.startswith("mbarrier") or op.startswith("cp.async"):
            ev = op.split(".")[0]
        if ev:
            events.append((ln, ev))
    # 合并连续同类
    merged = []
    for ln, ev in events:
        key = re.sub(r"\[\d+\]", "", ev) if ev.startswith("W[") and merged and merged[-1][1].startswith("W[") else ev
        if merged and (merged[-1][1] == ev or (ev.startswith("W[") and merged[-1][1].startswith("W["))):
            merged[-1][2] += 1
            if ev.startswith("W["):
                merged[-1][3].append(ev[2:-1])
        else:
            merged.append([ln, ev, 1, [ev[2:-1]] if ev.startswith("W[") else []])
    for ln, ev, n, offs in merged:
        extra = ""
        if offs:
            nums = sorted({int(o) for o in offs if o != "?"})
            extra = f"  偏移 {nums[0]}..{nums[-1]} ({len(nums)} 个)" if nums else "  偏移未知"
        print(f"{ln:6d}  {('W' if ev.startswith('W[') else ev):14s} x{n}{extra}")


if __name__ == "__main__":
    main()
