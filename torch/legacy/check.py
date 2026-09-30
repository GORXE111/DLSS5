"""形状核对 —— 唯一的验收标准是零剩余。

    python check.py            总览
    python check.py -v         逐条记录列出 slot
    python check.py --forward  跑一遍已闭合的 block 的前向
"""
import sys, collections
import numpy as np

from dlss5.weights import load, tier_of, parse_records

TOTAL = 73_841_889


def main():
    verbose = "-v" in sys.argv
    print("加载 WEIGHTS_HT …")
    recs = load(verbose=verbose)

    by_fam = collections.Counter()
    solved = collections.Counter()
    for name, slots in recs.items():
        blk = int(name.split(".")[0][5:])
        W, fam = tier_of(blk)
        n = sum(a.size for k, a in slots.items())
        by_fam[fam] += n
        unresolved = sum(a.size for k, a in slots.items()
                         if k.startswith("packed") or k == "out_packed")
        solved[fam] += n - unresolved

    print("\n%-12s %14s %14s %8s" % ("家族", "参数", "已定名", "覆盖率"))
    print("-" * 52)
    tot = totsolved = 0
    for fam in ("single1h", "single", "cuckoo4", "cuckoo5", "split16h", "bridge", "output"):
        if not by_fam[fam]:
            continue
        tot += by_fam[fam]; totsolved += solved[fam]
        print("%-12s %14s %14s %7.1f%%"
              % (fam, "{:,}".format(by_fam[fam]), "{:,}".format(solved[fam]),
                 100.0 * solved[fam] / by_fam[fam]))
    print("-" * 52)
    print("%-12s %14s %14s %7.1f%%"
          % ("合计", "{:,}".format(tot), "{:,}".format(totsolved), 100.0 * totsolved / tot))
    assert tot == TOTAL, "总数不符"
    print("\n零剩余校验通过：%s == %s" % ("{:,}".format(tot), "{:,}".format(TOTAL)))

    if "--forward" in sys.argv:
        forward_smoke(recs)


def forward_smoke(recs):
    import torch
    from dlss5.blocks import SplitSwin16HBlock, CuckooBlock
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n前向冒烟（%s）" % dev)

    b = SplitSwin16HBlock(512).load_from(recs, 31, dev)
    x = torch.randn(2, 64, 512, device=dev)
    y = b(x)
    print("  SplitSwin16H  in %s -> out %s   有限值 %s"
          % (tuple(x.shape), tuple(y.shape), bool(torch.isfinite(y).all())))

    c = CuckooBlock(256).load_from(recs, 23, dev)
    x = torch.randn(2, 64, 256, device=dev)
    y = c(x)
    print("  CrazyCuckoo   in %s -> out %s   有限值 %s"
          % (tuple(x.shape), tuple(y.shape), bool(torch.isfinite(y).all())))

    from dlss5.ops import mp_cubic_silu, schraudolph_exp, schraudolph_exp_exact, LOGIT_RANGE
    t = torch.linspace(-6, 6, 13, device=dev)
    print("\n  MpCubicSilu(x) 对 x=[-6..6]:")
    print("   ", " ".join("%+.3f" % v for v in mp_cubic_silu(t).tolist()))
    print("  有效 logit 区间 [%.4f, %.4f]" % LOGIT_RANGE)
    L = torch.linspace(-3, 3, 7, device=dev)
    a = schraudolph_exp(L).tolist()
    e = schraudolph_exp_exact(L).tolist()
    print("  解析式 vs 位运算式（最大相对差 %.2f%%）"
          % (100 * max(abs(x - y) / max(y, 1e-9) for x, y in zip(a, e))))


if __name__ == "__main__":
    main()
