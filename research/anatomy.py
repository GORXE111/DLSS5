"""网络在做什么: 把 NR 的改动量 d = 输出 - 输入 拆成可解释的部分，并看每个参数 / 每一级改变了哪一部分。

改动量分解 (全部在网络所见的显示编码 [0,1] 空间，按 /255 报告):
  亮度 Y = Rec.709 加权，色度 = RGB - Y
  频段  低 = G16(d) (高斯 σ=16 px @1080p，光照/大面积色调)，中 = G2(d) - G16(d) (材质/局部对比)，高 = d - G2(d) (细节/颗粒)
  解释度
    逐点 LUT    以输入 RGB (16^3 格) 预测 d 的 R²：越高越像"调色"(只看像素自身颜色)
    锐化系数    各频段 dY_b = k·Y_b(输入) 的 k 与 R²：k>0 = 该尺度的对比被放大
  画面量  亮度均值变化、全局对比 std(Y) 比、饱和度 (平均 |色度|) 比
实验:
  sweep   LocalTone / LocalStructure / SkinStructure / AutoMask 各取几档 (重置帧，无历史)
  ctx     把画面 A 的全局上下文 (ViT 出口) 换给画面 B：全局信息管什么
  skip    去掉解码器某一级的跳连 (置 0)：那一级的空间信息管什么 (超出训练分布，只看趋势)
  mask    Skin 0 vs 2、LocalStructure 0 vs 2 (skin=1) 的逐像素差：两个参数各作用在画面哪里 (AutoMask 的分割)
测试画面只在本地读取，输出写 anatomy_out/ (不入库)。
    python research/anatomy.py [--quick]
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "torch"))
import dlss5.net as NET  # noqa: E402
from dlss5 import DLSS5  # noqa: E402
from dlss5.net import control_inputs  # noqa: E402

OUT = os.path.join(HERE, "anatomy_out")
LUMA = torch.tensor([0.2126, 0.7152, 0.0722])


# ------------------------------------------------------------------ 合成测试卡 (可公开)
def chart(H=1080, W=1920):
    """上: 灰阶横向渐变；中: 色相 x 饱和度色块；下: 不同空间频率的正弦光栅 (对比 ±0.15，底 0.5)"""
    img = np.zeros((H, W, 3), np.float32)
    h3 = H // 3
    img[:h3] = np.linspace(0, 1, W, dtype=np.float32)[None, :, None]
    y, x = np.mgrid[0:h3, 0:W]
    hue = x / W * 6
    sat = 1 - y / h3
    c = np.clip(np.stack([np.abs(hue - 3) - 1, 2 - np.abs(hue - 2), 2 - np.abs(hue - 4)], -1), 0, 1)
    img[h3:2 * h3] = 0.5 + (c - 0.5) * sat[..., None] * 0.8
    periods = [96, 48, 24, 12, 6, 3]
    bw = W // len(periods)
    yy = np.arange(H - 2 * h3)[:, None]
    for i, p in enumerate(periods):
        xx = np.arange(bw)[None, :]
        g = 0.5 + 0.15 * np.sin(2 * np.pi * (xx + yy * 0.25) / p)
        img[2 * h3:, i * bw:(i + 1) * bw] = g[..., None]
    return torch.tensor(np.floor(img * 255 + 0.5) / 255, device="cuda")


def sources(quick):
    srcs = {"chart": chart()}
    for i, p in enumerate(sorted(glob.glob(os.path.join(HERE, "behavior_out", "*_src.npy")))):
        srcs[f"img{i + 1}"] = torch.tensor(np.load(p).astype(np.float32), device="cuda")   # 本地测试画面，只用代号
        if quick and i >= 0:
            break
    return srcs


# ------------------------------------------------------------------ 分解
def gauss(x, s):
    """(H, W, C) 可分离高斯，反射边界"""
    r = int(3 * s)
    k = torch.exp(-(torch.arange(-r, r + 1, device=x.device, dtype=torch.float32) ** 2) / (2 * s * s))
    k = (k / k.sum()).view(1, 1, -1)
    C = x.shape[-1]
    t = x.permute(2, 0, 1)[None]
    t = F.conv2d(F.pad(t, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).expand(C, 1, 1, -1), groups=C)
    t = F.conv2d(F.pad(t, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).expand(C, 1, -1, 1), groups=C)
    return t[0].permute(1, 2, 0)


def bands(x):
    a, b = gauss(x, 2.0), gauss(x, 16.0)
    return {"low": b, "mid": a - b, "high": x - a}


def rms(x):
    return float(x.pow(2).mean().sqrt()) * 255


def lut_r2(inp, d, n=16):
    """d 能被"输入颜色的函数"解释的比例 (逐通道方差加权)"""
    q = (inp.clamp(0, 1) * (n - 1) + 0.5).long()
    idx = (q[..., 0] * n + q[..., 1]) * n + q[..., 2]
    idx = idx.flatten()
    dd = d.reshape(-1, 3)
    cnt = torch.zeros(n ** 3, device=d.device).index_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
    s = torch.zeros(n ** 3, 3, device=d.device).index_add_(0, idx, dd)
    pred = (s / cnt.clamp_min(1)[:, None])[idx]
    return float(1 - (dd - pred).pow(2).sum() / (dd - dd.mean(0)).pow(2).sum().clamp_min(1e-12))


def fit_gain(dy, y):
    """dy ≈ k·y 的 k 与 R²"""
    k = float((dy * y).sum() / (y * y).sum().clamp_min(1e-12))
    r2 = float(1 - (dy - k * y).pow(2).sum() / dy.pow(2).sum().clamp_min(1e-12))
    return k, r2


def analyze(inp, out):
    L = LUMA.to(inp.device)
    d = out - inp
    yi, yo, dy = inp @ L, out @ L, d @ L
    ci, co = inp - yi[..., None], out - yo[..., None]
    dc = d - dy[..., None]
    by, bc, bi = bands(dy[..., None]), bands(dc), bands(yi[..., None])
    m = {"total": rms(d), "luma": {}, "chroma": {}, "gain": {},
         "lut_r2": lut_r2(inp, d),
         "mean_dY": float(dy.mean()) * 255,
         "contrast": float(yo.std() / yi.std()),
         "saturation": float(co.norm(dim=-1).mean() / ci.norm(dim=-1).mean().clamp_min(1e-9))}
    bo = bands(yo[..., None])
    m["detail"] = {}
    for b in ("low", "mid", "high"):
        m["detail"][b] = rms(bo[b]) / max(rms(bi[b]), 1e-9)      # 输出/输入 该频段亮度能量之比
        m["luma"][b] = rms(by[b])
        m["chroma"][b] = rms(bc[b])
        m["gain"][b] = fit_gain(by[b], bi[b])
    return m


# ------------------------------------------------------------------ 消融用的前向 (DLSS5._forward 的副本 + 两个干预点)
class Probe(DLSS5):
    def run(self, color, ctx=None, drop_skip=None, keep_ctx=False, controls=None):
        """ctx: 用给定张量替换 ViT 出口；drop_skip: 解码器该级 ('8h'..'1h', 'pre') 的跳连置 0；
        keep_ctx: 返回 (输出, ViT 出口)"""
        NET.ops.PRECISE = self.precise
        self._ab = {"ctx": ctx, "drop": drop_skip, "rec": None}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=self.half, cache_enabled=False):
            out = self._forward_ab(color, controls).float()
        return (out, self._ab["rec"]) if keep_ctx else out

    def _forward_ab(self, color, controls):
        ab = self._ab
        mv = torch.zeros(*color.shape[:2], 2, device=self.dev)
        skips, x, cur, pre_skip = {}, None, None, None
        D = NET.dims(*color.shape[:2])
        for st, m in self.steps:
            k = st["kind"]
            if k == "pre":
                x, pre_skip = m(color, color, mv, 0, controls)
                if ab["drop"] == "pre":
                    pre_skip = torch.zeros_like(pre_skip)
            elif k == "swin":
                lv = st["level"]
                H, Wd = D[lv]
                sh, var = tuple(st["shift"]), st["variant"]
                if var == "inpview":
                    cur = m(x[:, m.fi], H, Wd, sh)
                elif var == "ds":
                    cur, x = m(cur, H, Wd, sh)
                    skips[lv] = cur
                elif var == "up":
                    if lv == "8h":
                        x = NET.pad_rows_cols(x, D["16h"], D["16h_real"])
                    s = skips[lv] if ab["drop"] != lv else torch.zeros_like(skips[lv])
                    cur = m(x, s, H, Wd, sh)
                elif var == "outview":
                    x = m(cur, H, Wd, sh)[:, m.ci]
                else:
                    cur = m(cur, H, Wd, sh)
            elif k == "split16":
                if st["inpview"]:
                    x = NET.pad_rows_cols(x, D["16h_real"], D["16h"])
                cur = m(x[:, m.fi] if st["inpview"] else cur, tuple(st["shift"]), *D["16h"])
                if st["tail"] == "pool":
                    skips["16h"] = cur if ab["drop"] != "16h" else torch.zeros_like(cur)
                    cur = st["head"](cur, D["16h"], D["vit"])
                elif st["tail"] == "outview":
                    x = cur[:, m.ci]
            elif k == "vit":
                cur = m(cur)
                last_vit = cur
            elif k == "block39":
                if ab["ctx"] is not None:
                    cur = ab["ctx"]
                ab["rec"] = last_vit
                cur = m(cur, skips["16h"], D["vit"], D["16h"])
            elif k == "post":
                return m(x, pre_skip, color, None, mv)


# ------------------------------------------------------------------ 实验
SWEEP = [("默认", {}),
         ("LocalTone 0", {"tone": 0.0}), ("LocalTone 0.5", {"tone": 0.5}), ("LocalTone 2", {"tone": 2.0}),
         ("LocalStructure 0", {"structure": 0.0}), ("LocalStructure 0.5", {"structure": 0.5}),
         ("LocalStructure 2", {"structure": 2.0}),
         ("Skin 0", {"skin": 0.0}), ("Skin 2", {"skin": 2.0}),
         ("AutoMask 关", {"auto_mask": False}),
         ("Tone 0 + Structure 0", {"tone": 0.0, "structure": 0.0})]
DROPS = ["pre", "1h", "2h", "4h", "8h", "16h"]


def chart_response(inp, out, H=1080, W=1920):
    """测试卡: 灰阶响应 (输入灰度 -> 输出亮度，11 点) 与各周期光栅的振幅比"""
    L = LUMA.to(inp.device)
    h3 = H // 3
    yi, yo = (inp[40:h3 - 40] @ L).mean(0), (out[40:h3 - 40] @ L).mean(0)
    curve = [(float(yi[x]), float(yo[x])) for x in np.linspace(0, W - 1, 11).astype(int)]
    bw = W // 6
    gr = {}
    for i, p in enumerate([96, 48, 24, 12, 6, 3]):
        a = inp[2 * h3 + 40:H - 40, i * bw + 40:(i + 1) * bw - 40] @ L
        b = out[2 * h3 + 40:H - 40, i * bw + 40:(i + 1) * bw - 40] @ L
        gr[p] = float((b - b.mean()).std() / (a - a.mean()).std())
    return {"curve": curve, "grating": gr}


def fmt(m):
    g = m["gain"]
    return (f"总 {m['total']:5.2f} | 亮度 低/中/高 {m['luma']['low']:5.2f} {m['luma']['mid']:5.2f} {m['luma']['high']:5.2f}"
            f" | 色度 {m['chroma']['low']:5.2f} {m['chroma']['mid']:5.2f} {m['chroma']['high']:5.2f}"
            f" | 增益 中 {g['mid'][0]:+.3f}({g['mid'][1]:.2f}) 高 {g['high'][0]:+.3f}({g['high'][1]:.2f})"
            f" | 细节比 {m['detail']['low']:.2f}/{m['detail']['mid']:.2f}/{m['detail']['high']:.2f}"
            f" | LUT {m['lut_r2']:.2f} | ΔY {m['mean_dY']:+5.2f} 对比 x{m['contrast']:.3f} 饱和 x{m['saturation']:.3f}")


def save(name, x):
    np.save(os.path.join(OUT, name + ".npy"), (x.clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="只跑测试卡 + 1 张画面")
    ap.add_argument("--only", default="sweep,ctx,skip,mask")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    net = Probe()
    srcs = sources(a.quick)
    mp = os.path.join(OUT, "metrics.json")
    res = json.load(open(mp, encoding="utf-8")) if os.path.exists(mp) else {}
    for k in a.only.split(","):
        res[k] = {}
    t0 = time.time()
    base, ctxs = {}, {}
    for n, img in srcs.items():
        save(f"{n}_src", img)
        base[n], ctxs[n] = net.run(img, keep_ctx=True, controls=control_inputs())
    if "sweep" in a.only:
        for n, img in srcs.items():
            print(f"\n== {n} 参数扫描")
            res["sweep"][n] = {}
            for lab, kw in SWEEP:
                out = base[n] if not kw else net.run(img, controls=control_inputs(**kw))
                m = analyze(img, out)
                m["vs_default"] = analyze(base[n], out)["total"] if kw else 0.0
                if n == "chart":
                    m["chart"] = chart_response(img, out)
                res["sweep"][n][lab] = m
                print(f"  {lab:22s} {fmt(m)}  与默认差 {m['vs_default']:.2f}")
                if n == "chart":
                    c = m["chart"]
                    print("      灰阶 " + " ".join(f"{a * 255:.0f}->{b * 255:.0f}" for a, b in c["curve"])
                          + "  | 光栅振幅比 " + " ".join(f"T{p}:{g:.2f}" for p, g in c["grating"].items()))
                save(f"{n}_sweep_{lab.replace(' ', '_')}", out)
    if "ctx" in a.only:
        names = [n for n in srcs if n != "chart"]
        print("\n== 全局上下文互换 (ViT 出口)")
        for b in names:
            for c in [n for n in srcs if n != b]:
                out = net.run(srcs[b], ctx=ctxs[c])
                m = analyze(base[b], out)
                m["own_effect"] = analyze(srcs[b], base[b])["total"]
                res["ctx"][f"{b}<-{c}"] = m
                print(f"  {b} 用 {c} 的上下文: 与原输出差 {fmt(m)}  (原 NR 改动 {m['own_effect']:.2f})")
                save(f"{b}_ctx_from_{c}", out)
    if "skip" in a.only:
        print("\n== 去掉跳连 (与原输出之差)")
        for n, img in srcs.items():
            res["skip"][n] = {}
            for lv in DROPS:
                out = net.run(img, drop_skip=lv)
                m = analyze(base[n], out)
                res["skip"][n][lv] = m
                print(f"  {n} 去掉 {lv:3s}: {fmt(m)}")
                save(f"{n}_skip_{lv}", out)
    if "mask" in a.only:
        print("\n== 参数作用区域 (|差| /255，前 10% 像素所占能量)")
        for n, img in srcs.items():
            res["mask"][n] = {}
            for lab, k0, k1 in (("skin", {"skin": 0.0}, {"skin": 2.0}),
                                ("structure", {"structure": 0.0, "skin": 1.0}, {"structure": 2.0, "skin": 1.0})):
                x = (net.run(img, controls=control_inputs(**k1)) - net.run(img, controls=control_inputs(**k0))).abs().mean(-1)
                v = x.flatten().sort().values
                r = {"median": float(v[v.numel() // 2]) * 255, "p99": float(v[int(v.numel() * 0.99)]) * 255,
                     "top10_share": float(v[-v.numel() // 10:].sum() / v.sum())}
                res["mask"][n][lab] = r
                print(f"  {n} {lab:9s} 中位 {r['median']:.2f}  p99 {r['p99']:.2f}  前 10% 占 {r['top10_share'] * 100:.0f}%")
                np.save(os.path.join(OUT, f"{n}_mask_{lab}.npy"), (x.clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8))
    json.dump(res, open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
