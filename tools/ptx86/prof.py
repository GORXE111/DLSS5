"""解析 nr-lab 的 [prof] 输出: 逐 kernel 耗时排行，名字去修饰并压缩模板参数，标出所在 fatbin。

python -m tools.ptx86.prof harness/rt_sm86/prof1080.txt [--top 25] [--full]
"""
import argparse
import glob
import os
import re
import shutil
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def demangle(names):
    exe = shutil.which("c++filt")
    if not exe:
        return names
    r = subprocess.run([exe], input="\n".join(names), capture_output=True, text=True)
    out = r.stdout.splitlines()
    return out if len(out) == len(names) else names


def short(name):
    """tin3_1::Conv2d1x1Layer<Conv2d1x1Config<512,1024,...>, ...>::forward_impl(...) -> Conv2d1x1Layer<512,1024,…>"""
    name = re.sub(r"\(.*\)$", "", name)
    head = re.match(r"(?:[\w:]+::)?(\w+)", name.replace("tin3_1::", ""))
    ints = re.findall(r"(?<![\w.])-?\d+(?![\w.])", name)
    base = head.group(1) if head else name[:40]
    method = name.rsplit("::", 1)[-1] if "::" in name else ""
    return f"{base}<{','.join(ints[:8])}{',…' if len(ints) > 8 else ''}>{'::' + method if method and method != base else ''}"


def fatbin_of():
    """kernel mangled 名 -> fatbin 编号 (扫 sm86 cubin 的字符串表)"""
    m = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "sm86_port", "fatbin_*.sm86.cubin"))):
        n = os.path.basename(p)[7:9]
        for s in set(re.findall(rb"_Z[\w$]{8,}", open(p, "rb").read())):
            m.setdefault(s.decode(), n)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--full", action="store_true", help="打印完整去修饰名")
    a = ap.parse_args()
    head, rows = None, []
    for line in open(a.log, encoding="utf-8", errors="replace"):
        if line.startswith("[prof] frames="):
            head = line.strip()
        elif line.startswith("[prof]\t"):
            _, ms, per, grid, block, smem, name = line.rstrip("\n").split("\t", 6)
            rows.append(dict(ms=float(ms), per=float(per), grid=grid, block=block, smem=int(smem), name=name))
    fb = fatbin_of()
    dem = demangle([r["name"] for r in rows])
    total = sum(r["ms"] for r in rows)
    print(head)
    fam = {}
    for r, d in zip(rows, dem):
        r["short"] = short(d)
        r["full"] = d
        r["fb"] = fb.get(r["name"], "??")
        k = re.sub(r"<.*", "", r["short"])
        fam[k] = fam.get(k, 0) + r["ms"]
    print(f"\n按 kernel 族 (共 {total:.2f} ms/帧):")
    for k, v in sorted(fam.items(), key=lambda x: -x[1]):
        print(f"  {v:8.3f} ms  {100 * v / total:5.1f}%  {k}")
    print(f"\n前 {a.top} 个 launch 配置:")
    print(f"  {'ms/帧':>8} {'占比':>6} {'次':>4} {'fb':>3} {'grid':>13} {'block':>10} {'smem':>6}  kernel")
    for r in rows[: a.top]:
        print(f"  {r['ms']:8.3f} {100 * r['ms'] / total:5.1f}% {r['per']:4.0f} {r['fb']:>3} {r['grid']:>13} "
              f"{r['block']:>10} {r['smem']:6d}  {r['full'] if a.full else r['short']}")


if __name__ == "__main__":
    main()
