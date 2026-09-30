"""dllq query — 面向 agent 的检索命令。每条结果都带引用，输出保持小。"""
import os, re, io, json, subprocess, collections, datetime
from .core import (db, DB, NOTES, FATBINS, ROOT, DLL, rg, cuda,
                   encodings_of, u16_to_f16, u32_to_f32)

MAXROWS = 40


def _c():
    if not os.path.exists(DB):
        raise SystemExit("索引不存在，先跑:  python -m dllq index")
    return db()


def _pick_kernel(c, name):
    """精确名 > 前缀 > 子串。避免 'pre_block' 选中 '_ds' 变体。"""
    r = c.execute("SELECT * FROM kernel WHERE name=?", (name,)).fetchone()
    if r:
        return r
    r = c.execute("SELECT * FROM kernel WHERE name LIKE ? ORDER BY length(name) LIMIT 1",
                  (name + "%",)).fetchone()
    if r:
        return r
    return c.execute("SELECT * FROM kernel WHERE name LIKE ? ORDER BY length(name) LIMIT 1",
                     ("%" + name + "%",)).fetchone()


def _cite(fatbin, line):
    return "fatbin_%02d.ptx:%d" % (fatbin, line)


def _ptx(fatbin):
    return os.path.join(FATBINS, "fatbin_%02d.1.sm_120.ptx" % fatbin)


def _line(fatbin, n):
    p = _ptx(fatbin)
    if not os.path.exists(p):
        return ""
    with io.open(p, encoding="utf-8", errors="replace") as f:
        for i, l in enumerate(f, 1):
            if i == n:
                return l.rstrip()
            if i > n:
                break
    return ""


# ------------------------------------------------------------------ map

def cmd_map(args):
    """一页看完整个 target：文件、资源、fatbin、kernel 家族、权重。"""
    c = _c()
    f = c.execute("SELECT * FROM file WHERE kind='dll'").fetchone()
    print("目标   %s" % f["path"])
    print("大小   {:,} B".format(f["size"]))
    print("SHA256 %s" % f["sha256"])
    b = c.execute("SELECT v FROM meta WHERE k='built'").fetchone()
    print("索引   %s" % (b["v"] if b else "?"))

    print("\nPE 资源")
    for r in c.execute("SELECT * FROM pe_resource ORDER BY size DESC LIMIT 6"):
        print("  %-6s %-14s %-6s %14s B  @0x%X"
              % (r["rtype"], r["rname"], r["rlang"], "{:,}".format(r["size"]), r["file_off"]))

    print("\nfatbin / PTX")
    print("  %-4s %10s %10s %5s  %s" % ("idx", "payload", "PTX行", "entry", "首个 kernel"))
    for fb in c.execute("SELECT * FROM fatbin ORDER BY idx"):
        k = c.execute("SELECT name FROM kernel WHERE fatbin=? ORDER BY id LIMIT 1",
                      (fb["idx"],)).fetchone()
        print("  %-4d %10s %10s %5d  %s"
              % (fb["idx"], "{:,}".format(fb["payload_sz"]),
                 "{:,}".format(fb["ptx_lines"]), fb["n_entries"],
                 (k["name"] if k else "-")[:46]))
    tot = c.execute("SELECT count(*) n FROM kernel").fetchone()["n"]
    print("  合计 %d 个 kernel entry" % tot)

    w = c.execute("SELECT count(*) n, sum(bytes) b, sum(params) p, "
                  "count(DISTINCT block) k FROM wrec").fetchone()
    if w["n"]:
        print("\nWEIGHTS_HT")
        print("  记录 %d   block %d   payload {:,} B   参数 {:,}"
              .format(w["b"], w["p"]) % (w["n"], w["k"]))

    print("\n字符串分类")
    for r in c.execute("SELECT cls, count(*) n FROM strtab GROUP BY cls ORDER BY n DESC"):
        print("  %-10s %6d" % (r["cls"], r["n"]))


# ------------------------------------------------------------------ const

def cmd_const(args):
    """数值反查：给一个人类写法的数，找出它在 PTX 里的所有出现。"""
    c = _c()
    q = args.value
    print("查询 %s" % q)
    print("可能编码:")
    for lab, v in encodings_of(q):
        print("  %-26s %s" % (lab, v))

    hits = []
    try:
        if q.lower().startswith("0x"):
            u = int(q, 16)
            iv = u - (1 << 32) if u > 0x7FFFFFFF else u
            hits = c.execute("SELECT * FROM konst WHERE raw=? OR raw=? LIMIT ?",
                             (iv, u, MAXROWS)).fetchall()
            if not hits and u <= 0xFFFF:
                tgt = u16_to_f16(u)
                hits = c.execute(
                    "SELECT * FROM konst WHERE abs(f32-?)<1e-9 OR abs(f16lo-?)<1e-9 LIMIT ?",
                    (tgt, tgt, MAXROWS)).fetchall()
        else:
            x = float(q)
            if float(x).is_integer() and "." not in q and "e" not in q.lower():
                hits = c.execute("SELECT * FROM konst WHERE raw=? LIMIT ?",
                                 (int(x), MAXROWS)).fetchall()
            if not hits:
                hits = c.execute(
                    "SELECT * FROM konst WHERE abs(f32-?)<=abs(?*1e-6)+1e-12 "
                    "OR abs(f16lo-?)<=abs(?*1e-3)+1e-9 LIMIT ?",
                    (x, x, x, x, MAXROWS)).fetchall()
    except ValueError:
        pass

    if not hits:
        print("\n无命中（PTX 常数表里没有）")
        return
    # 按 raw 值分组：同一个常数散布在几十个 kernel 变体里，没必要逐条列
    groups = collections.OrderedDict()
    for h in hits:
        k = c.execute("SELECT name FROM kernel WHERE id=?", (h["kernel_id"],)).fetchone()
        g = groups.setdefault(h["raw"], dict(f32=h["f32"], first=h, kernels=[], n=0))
        g["n"] += 1
        kn = k["name"] if k else "?"
        if kn not in g["kernels"]:
            g["kernels"].append(kn)
    print("\n命中 %d 处 / %d 个不同取值:" % (len(hits), len(groups)))
    for raw, g in groups.items():
        h = g["first"]
        print("\n  raw=%-13d  f32=%-15.9g hex=0x%08X   共 %d 处，%d 个 kernel"
              % (raw, g["f32"] if g["f32"] is not None else float("nan"),
                 raw & 0xFFFFFFFF, g["n"], len(g["kernels"])))
        print("    首处  %-22s %s" % (_cite(h["fatbin"], h["line"]), h["reg"]))
        print("          %s" % _line(h["fatbin"], h["line"]).strip()[:110])
        for kn in g["kernels"][:5]:
            print("    kernel  %s" % kn)
        if len(g["kernels"]) > 5:
            print("    kernel  … 另 %d 个变体" % (len(g["kernels"]) - 5))


# ------------------------------------------------------------------ kernel

def cmd_kernel(args):
    c = _c()
    rows = c.execute("SELECT * FROM kernel WHERE name LIKE ? ORDER BY n_lines DESC LIMIT ?",
                     ("%" + args.name + "%", MAXROWS)).fetchall()
    if not rows:
        print("没有匹配的 kernel")
        return
    if len(rows) > 1 and not args.detail:
        print("匹配 %d 个 kernel:" % len(rows))
        for r in rows:
            print("  fatbin_%02d  %7s 行  %s" % (r["fatbin"], "{:,}".format(r["n_lines"]), r["name"]))
        print("\n加 -d 看第一个的指令构成")
        return
    r = rows[0]
    print("kernel  %s" % r["name"])
    print("位置    fatbin_%02d.ptx:%d-%d  (%s 行)"
          % (r["fatbin"], r["line_start"], r["line_end"], "{:,}".format(r["n_lines"])))
    print("\n指令构成 (top 24):")
    for i in c.execute("SELECT op, cnt FROM instr WHERE kernel_id=? ORDER BY cnt DESC LIMIT 24",
                       (r["id"],)):
        print("  %-34s %6d" % (i["op"], i["cnt"]))
    n = c.execute("SELECT count(*) n FROM konst WHERE kernel_id=?", (r["id"],)).fetchone()["n"]
    print("\n该 kernel 内 mov.b32 立即数: %d 个  (dllq consts-in %s)" % (n, r["name"]))


def cmd_consts_in(args):
    """列出某 kernel 内的所有立即数常数，已解码。"""
    c = _c()
    k = _pick_kernel(c, args.name)
    if not k:
        print("没有匹配的 kernel")
        return
    print("%s   (fatbin_%02d:%d-%d)" % (k["name"], k["fatbin"], k["line_start"], k["line_end"]))
    rows = c.execute("SELECT * FROM konst WHERE kernel_id=? ORDER BY line", (k["id"],)).fetchall()
    seen = {}
    for r in rows:
        seen.setdefault(r["raw"], r)
    print("\n%d 个不同的立即数:" % len(seen))
    print("  %-7s %-14s %-16s %-11s %s" % ("行", "raw", "f32", "f16(lo)", "hex"))
    def _g(v, w, p_):
        return ("%%-%d.%dg" % (w, p_)) % v if v is not None else "-".ljust(w)
    for raw, r in sorted(seen.items(), key=lambda kv: kv[1]["line"])[:MAXROWS * 2]:
        print("  %-7d %-14d %s %s 0x%08X"
              % (r["line"], raw, _g(r["f32"], 16, 9), _g(r["f16lo"], 11, 6),
                 raw & 0xFFFFFFFF))


# ------------------------------------------------------------------ ops

def cmd_ops(args):
    """指令普查：哪些 kernel / fatbin 用了这条指令。"""
    c = _c()
    pat = "%" + args.op + "%"
    rows = c.execute(
        "SELECT k.fatbin fb, k.name kn, i.op, i.cnt FROM instr i JOIN kernel k ON k.id=i.kernel_id "
        "WHERE i.op LIKE ? ORDER BY i.cnt DESC LIMIT ?", (pat, MAXROWS)).fetchall()
    if not rows:
        print("没有 kernel 用到 %s" % args.op)
        return
    byfb = collections.Counter()
    for r in rows:
        byfb[r["fb"]] += r["cnt"]
    print("按 fatbin 汇总:")
    for fb, n in sorted(byfb.items()):
        print("  fatbin_%02d  %8d" % (fb, n))
    print("\n按 kernel (top %d):" % min(len(rows), MAXROWS))
    for r in rows:
        print("  %8d  %-30s fatbin_%02d  %s" % (r["cnt"], r["op"], r["fb"], r["kn"][:44]))


# ------------------------------------------------------------------ def / grep

def _pyscan(pattern, path, before=0, after=0, limit=MAXROWS, fixed=False, lo=1, hi=1 << 40):
    """纯 Python 行扫描回退。38MB 文件约 1 秒，够用。"""
    rx = re.compile(re.escape(pattern) if fixed else pattern)
    buf, out, pend = collections.deque(maxlen=before or 1), [], 0
    with io.open(path, encoding="utf-8", errors="replace") as f:
        for n, line in enumerate(f, 1):
            if n > hi:
                break
            line = line.rstrip("\n")
            if n < lo:
                continue
            if pend:
                out.append("%d-%s" % (n, line)); pend -= 1
            elif rx.search(line):
                for i, b in enumerate(buf):
                    out.append("%d-%s" % (n - len(buf) + i, b))
                out.append("%d:%s" % (n, line))
                pend = after
                if sum(1 for l in out if ":" in l.split("-")[0] or l[:20].count(":")) >= limit:
                    if len([l for l in out if re.match(r"^\d+:", l)]) >= limit:
                        break
            if before:
                buf.append(line)
    return "\n".join(out)


def _rg(pattern, path, before=0, after=0, limit=MAXROWS, fixed=False, lo=1, hi=1 << 40):
    exe = rg()
    if not exe:
        return _pyscan(pattern, path, before, after, limit, fixed, lo, hi)
    cmd = [exe, "-n", "--no-heading", "-m", str(limit)]
    if fixed:
        cmd.append("-F")
    if before:
        cmd += ["-B", str(before)]
    if after:
        cmd += ["-A", str(after)]
    cmd += [pattern, path]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return p.stdout


def cmd_def(args):
    """寄存器定值/使用追踪：%rN 在哪定义、在哪被用。"""
    c = _c()
    reg = args.reg if args.reg.startswith("%") else "%" + args.reg
    if args.kernel:
        k = _pick_kernel(c, args.kernel)
        if not k:
            print("没有匹配的 kernel")
            return
        fbs = [(k["fatbin"], k["line_start"], k["line_end"], k["name"])]
    elif args.fatbin:
        fbs = [(int(args.fatbin), 1, 10 ** 9, "(整个 fatbin)")]
    else:
        print("需要 --kernel 或 --fatbin 限定范围（38MB PTX 全扫太慢）")
        return

    esc = re.escape(reg)
    for fb, lo, hi, label in fbs:
        p = _ptx(fb)
        print("范围  %s   fatbin_%02d.ptx:%d-%d" % (label, fb, lo, min(hi, 10 ** 9)))
        out = _pyscan(esc + r"[,;}\s]", p, limit=600, lo=lo, hi=min(hi, 1 << 40))
        defs, uses = [], []
        for line in out.splitlines():
            m = re.match(r"^(\d+):(.*)$", line)
            if not m:
                continue
            n, txt = int(m.group(1)), m.group(2)
            if not (lo <= n <= hi):
                continue
            # 定值：目标寄存器出现在第一个操作数位置
            if re.search(r"[\s{]\S+\s+" + esc + r"\s*,", txt) or \
               re.search(r"\{\s*[^}]*" + esc + r"[^}]*\}\s*,\s*\[", txt) or \
               re.search(r"^\s*\{?\s*[a-z][\w.]*\s+" + esc + r"[,;]", txt):
                defs.append((n, txt.strip()))
            else:
                uses.append((n, txt.strip()))
        print("\n  定值 %d 处:" % len(defs))
        for n, t in defs[:12]:
            print("    %6d  %s" % (n, t[:110]))
        print("\n  使用 %d 处:" % len(uses))
        for n, t in uses[:12]:
            print("    %6d  %s" % (n, t[:110]))


def cmd_grep(args):
    """词法检索，带引用。可限定 fatbin。"""
    targets = ([_ptx(int(args.fatbin))] if args.fatbin
               else sorted(os.path.join(FATBINS, f) for f in os.listdir(FATBINS)
                           if f.endswith(".ptx")))
    total = 0
    for p in targets:
        if not os.path.exists(p):
            continue
        out = _rg(args.pattern, p, before=args.B, after=args.A, limit=args.n, fixed=args.F)
        if not out:
            continue
        name = os.path.basename(p).split(".")[0]
        for line in out.splitlines():
            print("  %-11s %s" % (name, line[:150]))
            total += 1
        if total >= args.n:
            break
    if not total:
        print("无命中")


# ------------------------------------------------------------------ strings

def cmd_str(args):
    c = _c()
    sql = "SELECT * FROM strtab WHERE s LIKE ?"
    p = ["%" + args.pattern + "%"]
    if args.cls:
        sql += " AND cls=?"
        p.append(args.cls)
    sql += " ORDER BY length(s) LIMIT ?"
    p.append(args.n)
    rows = c.execute(sql, p).fetchall()
    if not rows:
        print("无命中")
        return
    for r in rows:
        print("  %-9s @0x%-9X %s" % (r["cls"], r["off"], r["s"][:150]))
    print("\n%d 条" % len(rows))


# ------------------------------------------------------------------ weights

def cmd_weights(args):
    c = _c()
    if args.block is not None:
        rows = c.execute("SELECT * FROM wrec WHERE block=? ORDER BY ord",
                         (args.block,)).fetchall()
        print("block%d  %d 条记录" % (args.block, len(rows)))
        for r in rows:
            print("  %-34s %14s B  %13s 参数  @0x%X"
                  % (r["name"], "{:,}".format(r["bytes"]),
                     "{:,}".format(r["params"]), r["off"]))
        return
    agg = c.execute("SELECT block, count(*) n, sum(bytes) b, sum(params) p "
                    "FROM wrec GROUP BY block ORDER BY block").fetchall()
    tot = sum(a["p"] for a in agg)
    print("%-9s %5s %16s %15s %8s" % ("block", "记录", "payload", "参数", "占比"))
    for a in agg:
        print("  block%-4d %4d %16s %15s %7.2f%%"
              % (a["block"], a["n"], "{:,}".format(a["b"]),
                 "{:,}".format(a["p"]), 100.0 * a["p"] / tot))
    print("\n合计 %d block   参数 {:,}".format(tot) % len(agg))


# ------------------------------------------------------------------ isa

def cmd_isa(args):
    """让 ptxas 直接给出目标架构的阻塞指令清单。"""
    ptxas = cuda("ptxas")
    if not ptxas:
        print("找不到 ptxas（预期在 %s）" % ROOT)
        return
    tgt = args.target
    files = sorted(f for f in os.listdir(FATBINS) if f.endswith(".ptx"))
    tmp = os.path.join(ROOT, "tools", "_isa_tmp.ptx")
    feats = collections.Counter()
    ok, bad = [], []
    for f in files:
        src = io.open(os.path.join(FATBINS, f), encoding="utf-8", errors="replace").read()
        src = re.sub(r"\.version 9\.4", ".version 9.3", src)
        src = re.sub(r"\.target sm_\d+[af]?", ".target " + tgt, src)
        io.open(tmp, "w", encoding="utf-8").write(src)
        p = subprocess.run([ptxas, "-arch=" + tgt, tmp, "-o", os.devnull],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        err = (p.stdout or "") + (p.stderr or "")
        short = f.split(".")[0]
        if p.returncode == 0:
            ok.append(short)
        else:
            bad.append(short)
            for m in re.finditer(r"Feature '([^']+)' requires \.target (sm_\d+)", err):
                feats[(m.group(1), m.group(2))] += 1
    try:
        os.remove(tmp)
    except OSError:
        pass
    print("目标 %s   通过 %d/%d" % (tgt, len(ok), len(files)))
    print("\n通过: %s" % ", ".join(ok) if ok else "\n通过: (无)")
    print("失败: %s" % ", ".join(bad) if bad else "失败: (无)")
    if feats:
        print("\n阻塞指令:")
        for (feat, need), n in feats.most_common():
            print("  %8d  %-34s 需要 %s" % (n, feat, need))
        print("\n  合计 %d 条错误" % sum(feats.values()))


# ------------------------------------------------------------------ notes

def _load_notes():
    if not os.path.exists(NOTES):
        return []
    out = []
    for l in io.open(NOTES, encoding="utf-8"):
        l = l.strip()
        if l:
            out.append(json.loads(l))
    return out


def cmd_note(args):
    if args.action == "add":
        rec = dict(ts=datetime.datetime.now().isoformat(timespec="seconds"),
                   text=args.text, cite=args.cite or [], tag=args.tag or [])
        with io.open(NOTES, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print("已记录 (#%d)" % len(_load_notes()))
    elif args.action == "ls":
        for i, n in enumerate(_load_notes()):
            print("  #%-3d %s  %s" % (i, n["ts"][5:16], n["text"][:100]))
            if n["cite"]:
                print("        引用: %s" % ", ".join(n["cite"]))
    elif args.action == "find":
        q = (args.text or "").lower()
        for i, n in enumerate(_load_notes()):
            hay = (n["text"] + " " + " ".join(n["cite"]) + " " + " ".join(n["tag"])).lower()
            if q in hay:
                print("  #%-3d %s" % (i, n["text"][:130]))
                if n["cite"]:
                    print("        引用: %s" % ", ".join(n["cite"]))


# ------------------------------------------------------------------ ask (路由器)

NUM_RE = re.compile(r"^-?(0[xX][0-9a-fA-F]+|\d+\.?\d*([eE][-+]?\d+)?)$")
REG_RE = re.compile(r"%r\d+")


def cmd_ask(args):
    """agent 的前门：按问题形态路由到对应索引，返回小而带引用的答案。"""
    q = " ".join(args.words)
    c = _c()
    print("问题  %s" % q)
    print("-" * 66)

    m = REG_RE.search(q)
    if m:
        print("路由 → 寄存器追踪 (def)")
        print("提示: dllq def %s --kernel <名字片段>\n" % m.group())

    for tok in q.replace(",", " ").split():
        if NUM_RE.match(tok):
            print("路由 → 常数反查 (const %s)\n" % tok)
            class A: value = tok
            cmd_const(A)
            return

    # 指令助记符
    if re.search(r"\b[a-z]+\.[a-z0-9.]+\b", q):
        tok = re.search(r"\b[a-z]+\.[a-z0-9.]+\b", q).group()
        n = c.execute("SELECT count(*) n FROM instr WHERE op LIKE ?",
                      ("%" + tok + "%",)).fetchone()["n"]
        if n:
            print("路由 → 指令普查 (ops %s)\n" % tok)
            class A: op = tok
            cmd_ops(A)
            return

    # kernel 名
    for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]{5,}", q):
        r = c.execute("SELECT count(*) n FROM kernel WHERE name LIKE ?",
                      ("%" + tok + "%",)).fetchone()["n"]
        if r:
            print("路由 → kernel 概览 (kernel %s)\n" % tok)
            class A: name = tok; detail = False
            cmd_kernel(A)
            return

    # 落到字符串 + 笔记
    print("路由 → 字符串表 + 笔记\n")
    hits = 0
    for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_.]{3,}", q):
        rows = c.execute("SELECT * FROM strtab WHERE s LIKE ? LIMIT 8",
                         ("%" + tok + "%",)).fetchall()
        for r in rows:
            print("  [str/%s] %s" % (r["cls"], r["s"][:120]))
            hits += 1
        if hits >= 12:
            break
    for i, n in enumerate(_load_notes()):
        hay = (n["text"] + " " + " ".join(n.get("tag") or []) + " "
               + " ".join(n.get("cite") or [])).lower()
        if any(t.lower() in hay for t in q.split() if len(t) > 2):
            print("  [note #%d] %s" % (i, n["text"][:120]))
            if n["cite"]:
                print("             引用: %s" % ", ".join(n["cite"]))
            hits += 1
    if hits < 4:
        try:
            from .vec import search
            res = search(q, k=5)
            if res:
                print("")
                print("  -- 语义召回 --")
                for score, src, ref, text in res:
                    if score < 0.55:
                        continue
                    print("  %.3f [%s] %s" % (score, src, ref))
                    print("        %s" % text[:120])
                    hits += 1
        except SystemExit:
            pass
        except Exception:
            pass
    if not hits:
        print("  无命中。试试:  dllq str <片段>   /   dllq grep <正则>")


# ------------------------------------------------------------------ shape

# 从模板名推断的字段语义。位置 0 基本都是主宽度，位置 1 常是输入宽度或 head 相关。
CFG_HINT = {
    "FusedSwin2d1HConfig":            ["W", "?", "tileH", "tileW", "heads", "?"],
    "CrazyCuckooFusedSwin2d2HConfig": ["W_out", "W_in", "?", "tileH", "tileW", "heads"],
    "CrazyCuckooFusedSwin2d4HConfig": ["W_out", "W_in", "?", "tileH", "tileW", "heads"],
    "CrazyCuckooFusedSwin2d8HConfig": ["W_out", "W_in", "?", "tileH", "tileW", "heads"],
    "FusedSwin2dFfwdConfig":          ["W", "?", "?", "tileH", "tileW", "?"],
    "FusedSwin2dQKVAttnConfig":       ["W", "tileH", "tileW", "heads", "?", "headDim"],
    "Conv2d1x1Config":                ["C_out", "C_in", "tileH", "tileW", "?"],
    "Conv1d1x1Config":                ["C_out", "C_in", "?", "?", "?"],
    "Conv2dQKVConfig":                ["headDim", "headDim", "C_in", "heads", "?"],
    "Conv1dQKVConfig":                ["headDim", "headDim", "C_in", "?", "?"],
    "Attention2dConfig":              ["headDim", "headDim", "heads", "tileH", "tileW"],
    "Attention1dConfig":              ["headDim", "headDim", "tokens", "?", "?"],
}

CAND_W = [1, 8, 16, 32, 64, 96, 128, 160, 192, 256, 384, 512, 768,
          1024, 1536, 2048, 3072, 4096, 6144, 8192]


def _factor(p):
    """把参数量拆成 a*b (+bias) 的候选。返回 (a, b, bias, 说明) 列表。"""
    out = []
    for bias_name, bias in (("无偏置", 0),):
        pass
    for a in CAND_W:
        for b in CAND_W:
            prod = a * b
            d = p - prod
            if d < 0:
                continue
            if d == 0:
                out.append((a, b, 0, "%d x %d" % (a, b)))
            elif d == b:
                out.append((a, b, b, "%d x %d + bias(%d)" % (a, b, b)))
            elif d == a:
                out.append((a, b, a, "%d x %d + bias(%d)" % (a, b, a)))
            elif 0 < d <= 4096 and d in CAND_W:
                out.append((a, b, d, "%d x %d + %d" % (a, b, d)))
    # 偏好：乘积占比高、a/b 是常见宽度
    out.sort(key=lambda t: (-(t[0] * t[1]) / p, abs(t[0] - t[1])))
    return out[:4]


def cmd_shape(args):
    c = _c()
    if args.weights:
        _shape_weights(c, args)
        return
    rows = c.execute("SELECT cfg, ints, bools, fatbin, count(*) n, "
                     "group_concat(DISTINCT kernel_id) kids "
                     "FROM tmpl GROUP BY cfg, ints ORDER BY cfg, fatbin").fetchall()
    if not rows:
        print("tmpl 表为空 — 先跑 python -m dllq index")
        return
    cur = None
    for r in rows:
        if r["cfg"] != cur:
            cur = r["cfg"]
            hint = CFG_HINT.get(cur)
            print("\n%s" % cur)
            if hint:
                print("  字段(推断): %s" % " · ".join(hint))
        ii = r["ints"].split(",") if r["ints"] else []
        k = c.execute("SELECT name FROM kernel WHERE id=?",
                      (int(r["kids"].split(",")[0]),)).fetchone() if r["kids"] else None
        print("  fb%-3d <%s>" % (r["fatbin"], ", ".join(ii[:15])))
        print("        %dx  %s" % (r["n"], (k["name"] if k else "?")[:56]))


def _shape_weights(c, args):
    """用参数量反推层形状，并和模板配置交叉核对。"""
    if args.block is not None:
        rows = c.execute("SELECT * FROM wrec WHERE block=? ORDER BY ord",
                         (args.block,)).fetchall()
    else:
        rows = c.execute("SELECT * FROM wrec ORDER BY ord").fetchall()
    print("%-30s %13s   %s" % ("记录", "参数", "最可能的形状"))
    print("-" * 86)
    for r in rows:
        cands = _factor(r["params"])
        best = cands[0][3] if cands else "-"
        alt = ("   | " + cands[1][3]) if len(cands) > 1 else ""
        print("%-30s %13s   %s%s"
              % (r["name"], "{:,}".format(r["params"]), best, alt))


# ------------------------------------------------------------------ unpack

WEIGHTS_BIN = os.path.join(ROOT, "WEIGHTS_HT.bin")


def _f16(u):
    import struct as _s
    return _s.unpack("<e", _s.pack("<H", u))[0]


def _load_rec(c, name):
    import struct as _s
    r = c.execute("SELECT * FROM wrec WHERE name=?", (name,)).fetchone()
    if not r:
        r = c.execute("SELECT * FROM wrec WHERE name LIKE ? LIMIT 1",
                      ("%" + name + "%",)).fetchone()
    if not r:
        return None, None
    b = open(WEIGHTS_BIN, "rb").read(0) if False else None
    with open(WEIGHTS_BIN, "rb") as f:
        f.seek(r["off"])
        raw = f.read(r["bytes"])
    v = [_f16(u) for u, in _s.iter_unpack("<H", raw)]
    return r, v


def _segment(v, win=64, thr=0.5):
    import math
    n = len(v)
    prof = []
    for i in range(0, n, win):
        s = [abs(x) for x in v[i:i + win] if not math.isnan(x) and x != 0]
        prof.append(math.log10(sum(s) / len(s)) if s else -12)
    cuts = [0]
    for i in range(1, len(prof)):
        if abs(prof[i] - prof[i - 1]) > thr:
            cuts.append(i * win)
    cuts.append(n)
    # 合并太碎的段
    merged, out = [], []
    for a, z in zip(cuts, cuts[1:]):
        if merged and z - a < win * 4 and a - merged[-1][0] < win * 8:
            merged[-1] = (merged[-1][0], z)
        else:
            merged.append((a, z))
    import math as _m
    for a, z in merged:
        s = [abs(x) for x in v[a:z] if not _m.isnan(x)]
        out.append((a, z - a, sum(s) / len(s) if s else 0))
    return out


def cmd_unpack(args):
    """把一条权重记录切成子张量，并按 slot 词表和已知形状猜标签。"""
    import math
    c = _c()
    r, v = _load_rec(c, args.name)
    if not r:
        print("找不到记录")
        return
    W = args.width
    H = (W // 32) if W else None
    print("%s   %s 元素 / %s 字节" %
          (r["name"], "{:,}".format(r["params"]), "{:,}".format(r["bytes"])))
    if W:
        print("假定 W=%d  heads=%d  (head_dim 32)" % (W, H))
    print()
    known, bigknown = {}, {}
    if W:
        # 大幅度段优先按 bias/scale 解释；小幅度段按权重解释
        bigknown = {
            H * 4096:       "attn_bias  H×64×64（8×8 窗口内稠密偏置）",
            H * 4096 * 2:   "attn_bias  H×2×64×64（split ×2）",
            H * 4:          "attn_scale H×2×2",
            H * 2:          "attn_scale H×2",
            W:              "scale W",
            2 * W:          "scale 2W",
        }
        known = {
            H * 4096:       "attn_bias  H×64×64（8×8 窗口内稠密偏置）",
            H * 4096 * 2:   "attn_bias  H×2×64×64（split ×2）",
            H * 4:          "attn_scale H×2×2",
            H * 2:          "attn_scale H×2",
            2 * W:          "cos_skip   2W",
            W:              "cos_skip / bias  W",
            W * W:          "proj  W×W",
            W * W * 2:      "proj  W×W ×2",
            W * W * 3:      "qkv   W×3W",
            W * W * 3 * 2:  "qkv   W×3W ×2",
            W * W * 4:      "ffn   W×4W",
            W * W * 4 * 2:  "ffn   W×4W ×2",
            W * W * 8:      "ffn   W×4W 双向",
        }
    print("  %-11s %11s  %-11s %s" % ("起点", "长度", "平均|x|", "标签"))
    for a, ln, m in _segment(v, thr=args.thr):
        lbl = (bigknown.get(ln) if m > 1.0 else None) or known.get(ln, "")
        if not lbl and m > 1.0:
            lbl = "(大值 → bias/scale 类)"
        p2 = ""
        if ln > 0 and abs(math.log2(ln) - round(math.log2(ln))) < 1e-9:
            p2 = " =2^%d" % round(math.log2(ln))
        print("  %-11s %11s%-7s %-11.4g %s"
              % ("{:,}".format(a), "{:,}".format(ln), p2, m, lbl))
