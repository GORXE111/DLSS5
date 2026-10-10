"""网络在做什么 (二): 时域、更多画面、全局上下文的风格方向。接 anatomy.py (分解与指标沿用)。

  temporal  静止画面连跑 10 帧 (历史 = 上一帧输出): 改动量、帧间闪烁、历史门控如何随帧变化；
            平移镜头 (每帧 8 px，正确 / 置 0 的运动矢量 / 每帧重置) 的帧间一致性与细节保留；
            场景切换 (A 6 帧后换 B): 旧画面残留随帧衰减
  survey    全部画面 (本地测试画面 + _dl/commons 下载的人物/易混淆物/场景/CG) 的默认改动指标，
            Skin 0 vs 2 的作用区域 (AutoMask 皮肤分割) 与按类别的统计
  style     全局上下文 (ViT 出口，每 token = 输入 64x64 像素一格) 的结构: 注意力熵、token 间相似度；
            全部画面平均 token 的主成分 (风格方向)，沿各方向推动后画面怎么变
测试画面与下载画面只在本地读取，输出写 anatomy_out/ (不入库)。
    python research/anatomy2.py temporal|survey|style [--n N]
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from anatomy import LUMA, OUT, Probe, analyze, bands, control_inputs, rms  # noqa: E402
import dlss5.blocks as B  # noqa: E402  (anatomy 已把 torch/ 加入路径)
import dlss5.net as NET  # noqa: E402
from res_study import load  # noqa: E402

COMMONS = os.path.join(HERE, "..", "_dl", "commons")
H, W = 1080, 1920
REC = {}


def hook_gate():
    orig = NET.PostBlock.__call__

    def call(self, x69, skip, color, hist, mv):
        n = self.net(x69, skip, *NET.dims(*color.shape[:2])["grid"])[:color.shape[0], :color.shape[1]]
        REC["gate"] = (torch.sigmoid(n[..., 3]) * self.blend).clamp(0, 1) if hist is not None else None
        return orig(self, x69, skip, color, hist, mv)
    NET.PostBlock.__call__ = call


def images(n=None):
    """{代号: (类别, 路径或数组)}，本地测试画面用 img1..，下载画面用文件名"""
    out = {}
    for i, p in enumerate(sorted(glob.glob(os.path.join(HERE, "behavior_out", "*_src.npy")))):
        out[f"img{i + 1}"] = ("local", p)
    for p in sorted(glob.glob(os.path.join(COMMONS, "*.jpg"))):
        k = os.path.splitext(os.path.basename(p))[0]
        out[k] = (k.split("_")[0], p)
    if n:
        out = dict(list(out.items())[:n])
    return out


def get(path, size=(H, W)):
    if path.endswith(".npy"):
        x = torch.tensor(np.load(path).astype(np.float32), device="cuda")
        if x.shape[:2] != size:
            x = F.interpolate(x.permute(2, 0, 1)[None], size=size, mode="bicubic", align_corners=False)[0].permute(1, 2, 0).clamp(0, 1)
        return x
    return load(path, *size)


def warmth(x):
    """平均 (R - B) / 2"""
    return float((x[..., 0] - x[..., 2]).mean() / 2)


def small(x, s=4):
    return (F.avg_pool2d(x.permute(2, 0, 1)[None], s)[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)


# ================================================================== 时域
def temporal(net, imgs):
    hook_gate()
    res = {"static": {}, "pan": {}, "cut": {}}
    L = LUMA.cuda()
    pick = [k for k in ("img1", "img2", "img3", "img4", "people_10", "scene_08", "cg_01", "scene_16") if k in imgs]
    print("== 静止画面 10 帧")
    for k in pick:
        src = get(imgs[k][1])
        out, rows, first = None, [], None
        for f in range(10):
            prev = out
            out = net(src, hist=out, frame=f)
            m = analyze(src, out)
            r = {"frame": f, "total": m["total"], "low": m["luma"]["low"], "mid": m["luma"]["mid"], "high": m["luma"]["high"],
                 "sat": m["saturation"], "detail_high": m["detail"]["high"],
                 "flicker": rms(out - prev) if prev is not None else 0.0,
                 "gate": float(REC["gate"].mean()) if REC.get("gate") is not None else 0.0}
            if f == 0:
                first = out
            r["vs_reset"] = rms(out - first)
            rows.append(r)
        res["static"][k] = rows
        print(f"  {k:10s} " + " ".join(f"f{r['frame']}:{r['total']:.1f}/{r['flicker']:.2f}/g{r['gate']:.2f}" for r in rows)
              + f"  | 第 9 帧 低/中/高 {rows[-1]['low']:.1f}/{rows[-1]['mid']:.1f}/{rows[-1]['high']:.1f} (第 0 帧 {rows[0]['low']:.1f}/{rows[0]['mid']:.1f}/{rows[0]['high']:.1f})"
              f" 细节比 {rows[-1]['detail_high']:.2f} (第 0 帧 {rows[0]['detail_high']:.2f})")
    print("== 平移镜头 (每帧画面左移 8 px，12 帧)")
    S, N = 8, 12
    for k in pick[:5]:
        big = get(imgs[k][1], (H, W + S * N))
        res["pan"][k] = {}
        for case in ("正确运动矢量", "运动矢量置 0", "每帧重置"):
            mv = torch.zeros(H, W, 2, device="cuda")
            if case == "正确运动矢量":
                mv[..., 0] = S
            out, fl, fi, det = None, [], [], []
            for f in range(N):
                src = big[:, f * S:f * S + W]
                prev = out
                out = net(src, hist=None if (out is None or case == "每帧重置") else out, mv=mv, frame=f)
                if prev is not None and f >= 4:
                    fl.append(rms(out[:, :-S] - prev[:, S:]))       # 运动补偿后的帧间差 (输入的这一项恒为 0)
                    det.append(analyze(src, out)["detail"]["high"])
                    fi.append(analyze(src, out)["total"])
            r = {"flicker": float(np.mean(fl)), "detail_high": float(np.mean(det)), "total": float(np.mean(fi))}
            res["pan"][k][case] = r
            print(f"  {k:10s} {case:8s} 补偿后帧间差 {r['flicker']:.2f}/255  高频细节比 {r['detail_high']:.3f}  改动 {r['total']:.1f}")
    print("== 场景切换 (A 6 帧 -> B 6 帧，旧画面残留 = |输出 - B 单独连跑的输出|)")
    pairs = [("img1", "img3"), ("img3", "img1"), ("scene_08", "scene_16")]
    for a, b in pairs:
        if a not in imgs or b not in imgs:
            continue
        A, Bi = get(imgs[a][1]), get(imgs[b][1])
        ref, o = [], None
        for f in range(6):
            o = net(Bi, hist=o, frame=f + 6)
            ref.append(o)
        o = None
        for f in range(6):
            o = net(A, hist=o, frame=f)
        rows = []
        for f in range(6):
            o = net(Bi, hist=o, frame=f + 6)
            rows.append({"frame": f, "residual": rms(o - ref[f]), "gate": float(REC["gate"].mean())})
        res["cut"][f"{a}->{b}"] = rows
        print(f"  {a}->{b}: " + "  ".join(f"切换后第 {r['frame']} 帧 {r['residual']:.1f}/255 (门控 {r['gate']:.2f})" for r in rows))
    return res


# ================================================================== 普查: 场景判断 + 皮肤分割
def survey(net, imgs):
    res = {}
    masks = {}
    ctx_mean = {}
    for i, (k, (cat, p)) in enumerate(imgs.items()):
        src = get(p)
        out, ctx = net.run(src, keep_ctx=True, controls=control_inputs())
        m = analyze(src, out)
        m.update(cat=cat, warmth_d=(warmth(out) - warmth(src)) * 255, in_luma=float((src @ LUMA.cuda()).mean()) * 255,
                 in_sat=float((src - (src @ LUMA.cuda())[..., None]).norm(dim=-1).mean()) * 255, in_warmth=warmth(src) * 255)
        s0 = net.run(src, controls=control_inputs(skin=0.0))
        s2 = net.run(src, controls=control_inputs(skin=2.0))
        x = (s2 - s0).abs().mean(-1)
        v = x.flatten().sort().values
        m["skin"] = {"p99": float(v[int(v.numel() * 0.99)]) * 255, "mean": float(v.mean()) * 255,
                     "area": float((x > 4 / 255).float().mean()), "top10_share": float(v[-v.numel() // 10:].sum() / v.sum())}
        res[k] = m
        masks[k] = (small(src), (x.clamp(0, 0.1) * 2550).byte().cpu().numpy()[::4, ::4], small(out))
        g = NET.dims(H, W)
        real = (g["16h_real"][0] // 2, g["16h_real"][1] // 2)
        c = ctx.float().view(g["vit"][0], g["vit"][1], -1)[:real[0], :real[1]].reshape(-1, ctx.shape[-1])
        ctx_mean[k] = c.mean(0).cpu().numpy()
        cn = F.normalize(c - c.mean(0), dim=-1)
        m["ctx_spread"] = float((c - c.mean(0)).norm(dim=-1).mean() / c.mean(0).norm())   # token 偏离均值 / 均值长度
        print(f"[{i + 1}/{len(imgs)}] {k:10s} {cat:6s} 改动 {m['total']:5.1f} 低频 {m['luma']['low']:5.1f} ΔY {m['mean_dY']:+5.1f} 对比 x{m['contrast']:.2f}"
              f" 饱和 x{m['saturation']:.2f} 暖 {m['warmth_d']:+.1f} | 皮肤 p99 {m['skin']['p99']:5.1f} 面积 {m['skin']['area'] * 100:4.1f}%"
              f" | token 离散 {m['ctx_spread']:.2f}")
    np.savez_compressed(os.path.join(OUT, "survey_masks.npz"), **{f"{k}_{j}": v for k, t in masks.items() for j, v in zip(("src", "skin", "out"), t)})
    np.save(os.path.join(OUT, "survey_ctx.npy"), np.stack([ctx_mean[k] for k in res]))
    json.dump(list(res), open(os.path.join(OUT, "survey_order.json"), "w"))
    print("\n== 按类别")
    for cat in sorted({m["cat"] for m in res.values()}):
        r = [m for m in res.values() if m["cat"] == cat]
        f = lambda key: np.mean([key(m) for m in r])   # noqa: E731
        print(f"  {cat:7s} n={len(r):2d} 改动 {f(lambda m: m['total']):5.1f} 低频占比 {f(lambda m: m['luma']['low'] / m['total']):.2f}"
              f" 对比 x{f(lambda m: m['contrast']):.3f} 饱和 x{f(lambda m: m['saturation']):.3f} 暖 {f(lambda m: m['warmth_d']):+.2f}"
              f" ΔY {f(lambda m: m['mean_dY']):+.2f} | 皮肤 p99 {f(lambda m: m['skin']['p99']):5.1f} 面积 {f(lambda m: m['skin']['area']) * 100:4.1f}%")
    ks = list(res)
    for a, b in (("in_luma", "mean_dY"), ("in_sat", "saturation"), ("in_warmth", "warmth_d"), ("in_luma", "contrast")):
        x = np.array([res[k][a] for k in ks]); y = np.array([res[k][b] for k in ks])
        print(f"  相关 {a} ~ {b}: {np.corrcoef(x, y)[0, 1]:+.2f}")
    return res


# ================================================================== 全局上下文的结构与风格方向
def vit_attention_stats(net, src):
    """最后一个 ViT 块的注意力: 归一化熵 (1 = 平均看所有 token)、最大注意力权重、有效 token 数"""
    stats = []
    orig = B.ViT.__call__

    def call(self, X):
        from dlss5.ops import EXPVIT, act, cos_norm, exp_bits, f16, q8
        Hh = q8(act(f16(X[:, self.ci] @ self.W1)))
        x1 = q8(f16(self.c1 * X + Hh @ self.W2))
        xc = x1[:, self.ci]
        n = X.shape[0]
        sp = lambda t: t.view(n, 32, 32).transpose(0, 1)   # noqa: E731
        q, k = sp(f16(xc @ self.Wq)), sp(f16(xc @ self.Wk))
        Q = q8(f16(cos_norm(q) * self.tau.view(32, 1, 1) * np.sqrt(32)))
        p = exp_bits(f16(Q @ q8(cos_norm(k)).transpose(-1, -2)), *EXPVIT).float()
        P = p / p.sum(-1, keepdim=True)
        ent = float((-(P * P.clamp_min(1e-12).log()).sum(-1)).mean() / math.log(n))
        eff = float((1 / (P * P).sum(-1)).mean())
        stats.append({"entropy": ent, "eff_tokens": eff, "n": n, "max_w": float(P.max(-1).values.mean())})
        return orig(self, X)
    B.ViT.__call__ = call
    try:
        net.run(src)
    finally:
        B.ViT.__call__ = orig
    return stats


def style(net, imgs):
    res = {}
    order = json.load(open(os.path.join(OUT, "survey_order.json")))
    C = np.load(os.path.join(OUT, "survey_ctx.npy")).astype(np.float64)
    sv = json.load(open(os.path.join(OUT, "survey.json"), encoding="utf-8"))
    print("== ViT 注意力 (8 块，img1)")
    st = vit_attention_stats(net, get(imgs["img1"][1]))
    for i, s in enumerate(st):
        print(f"  块 {i}: 归一化熵 {s['entropy']:.3f}  有效 token {s['eff_tokens']:.0f}/{s['n']}  平均最大权重 {s['max_w']:.3f}")
    res["attn"] = st
    mu = C.mean(0)
    U, S, Vt = np.linalg.svd(C - mu, full_matrices=False)
    var = S ** 2 / (S ** 2).sum()
    print("\n== 全局上下文 (每画面平均 token) 主成分: 方差占比 " + " ".join(f"{v * 100:.0f}%" for v in var[:8]))
    sc = (C - mu) @ Vt.T
    res["pca_var"] = var[:10].tolist()
    keys = ["in_luma", "in_sat", "in_warmth"]
    for j in range(5):
        cors = {k: float(np.corrcoef(sc[:, j], [sv[o][k] for o in order])[0, 1]) for k in keys}
        cors.update({o: float(np.corrcoef(sc[:, j], [sv[x][o] for x in order])[0, 1]) for o in ("mean_dY", "saturation", "warmth_d", "contrast")})
        top = [order[i] for i in np.argsort(sc[:, j])[-4:][::-1]]
        bot = [order[i] for i in np.argsort(sc[:, j])[:4]]
        print(f"  PC{j + 1}: " + " ".join(f"{k} {v:+.2f}" for k, v in cors.items()) + f" | 正端 {top} 负端 {bot}")
        res[f"pc{j + 1}"] = {"corr": cors, "top": top, "bottom": bot}
    print("\n== 沿主成分推动全局上下文 (所有 token 加 ±2σ·方向)，相对原输出")
    g = NET.dims(H, W)
    for base in [k for k in ("img1", "scene_05", "people_10") if k in imgs]:
        src = get(imgs[base][1])
        out0, ctx = net.run(src, keep_ctx=True)
        res[f"push_{base}"] = {}
        for j in range(5):
            d = torch.tensor(Vt[j] * S[j] / math.sqrt(len(C)), device="cuda", dtype=ctx.dtype)
            row = {}
            for sgn in (-2, 2):
                o = net.run(src, ctx=ctx + sgn * d)
                m = analyze(out0, o)
                row[sgn] = {"total": m["total"], "dY": m["mean_dY"], "sat": m["saturation"], "contrast": m["contrast"],
                            "warm": (warmth(o) - warmth(out0)) * 255, "low": m["luma"]["low"], "mid": m["luma"]["mid"],
                            "high": m["luma"]["high"], "detail_high": m["detail"]["high"]}
                np.save(os.path.join(OUT, f"push_{base}_pc{j + 1}_{'p' if sgn > 0 else 'm'}.npy"), small(o, 2))
            res[f"push_{base}"][j + 1] = row
            print(f"  {base:10s} PC{j + 1}: " + "  ".join(
                f"{'+' if s > 0 else '-'}2σ: 变化 {r['total']:4.1f} ΔY {r['dY']:+5.1f} 饱和 x{r['sat']:.3f} 暖 {r['warm']:+5.1f} 对比 x{r['contrast']:.3f} 低/中/高 {r['low']:.1f}/{r['mid']:.1f}/{r['high']:.1f}"
                for s, r in row.items()))
        np.save(os.path.join(OUT, f"push_{base}_base.npy"), small(out0, 2))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["temporal", "survey", "style"])
    ap.add_argument("--n", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    net = Probe()
    imgs = images(a.n)
    t0 = time.time()
    r = {"temporal": temporal, "survey": survey, "style": style}[a.what](net, imgs)
    json.dump(r, open(os.path.join(OUT, f"{a.what}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
