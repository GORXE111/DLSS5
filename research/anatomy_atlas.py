"""anatomy.py 的结果 -> 本地对比图集 anatomy_out/atlas.html (图像内嵌，含测试画面，不入库、不发布)。

每张画面一行组图: 输入 | NR 输出 | 亮度改动的低频部分 (光照/色调，红亮蓝暗) | 中高频部分 (材质/细节)
| 色度改动 | LocalTone=0 | LocalStructure=0 | Skin / Structure 作用区域 | 换用别的画面的全局上下文 | 去掉 4h 跳连
    python research/anatomy_atlas.py
"""
import base64
import io
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from anatomy import LUMA, OUT, bands  # noqa: E402

W = 640


def T(n):
    return torch.tensor(np.load(os.path.join(OUT, n + ".npy")).astype(np.float32) / 255, device="cuda")


def jpg(x):
    a = x if isinstance(x, np.ndarray) else (x.clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)
    im = Image.fromarray(a)
    im = im.resize((W, round(W * im.height / im.width)), Image.LANCZOS)
    b = io.BytesIO()
    im.save(b, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def diverge(d, scale):
    """有符号 -> 红 (正) / 蓝 (负)，灰底"""
    t = (d * scale).clamp(-1, 1)
    g = torch.full_like(t, 0.5)
    return torch.stack([g + 0.5 * t.clamp(min=0) - 0.25 * (-t).clamp(min=0),
                        g - 0.3 * t.abs(),
                        g + 0.5 * (-t).clamp(min=0) - 0.25 * t.clamp(min=0)], -1)


def tiles(n, m, other):
    L = LUMA.cuda()
    s, o = T(f"{n}_src"), T(f"{n}_sweep_默认")
    d = o - s
    dy = d @ L
    b = bands(dy[..., None])
    dc = d - dy[..., None]
    t = [("输入", s), ("NR 输出", o),
         ("亮度改动·低频 (σ≥16px) ×8", diverge(b["low"][..., 0], 8)),
         ("亮度改动·中高频 ×16", diverge((b["mid"] + b["high"])[..., 0], 16)),
         ("色度改动 ×8", (0.5 + dc * 8).clamp(0, 1)),
         ("LocalTone = 0", T(f"{n}_sweep_LocalTone_0")),
         ("LocalStructure = 0", T(f"{n}_sweep_LocalStructure_0"))]
    if other and os.path.exists(os.path.join(OUT, f"{n}_ctx_from_{other}.npy")):
        t.append((f"换用 {other} 的全局上下文", T(f"{n}_ctx_from_{other}")))
    for lab, cap in (("skin", "Skin 0→2 的作用区域 ×20"), ("structure", "LocalStructure 0→2 的作用区域 ×20")):
        if os.path.exists(os.path.join(OUT, f"{n}_mask_{lab}.npy")):
            t.append((cap, (T(f"{n}_mask_{lab}") * 20).clamp(0, 1)[..., None].expand(-1, -1, 3)))
    if os.path.exists(os.path.join(OUT, f"{n}_skip_4h.npy")):
        t.append(("去掉 4h 跳连", T(f"{n}_skip_4h")))
    sw = m["sweep"][n]["默认"]
    cap = (f"总改动 {sw['total']:.1f}/255 · 亮度 低/中/高 {sw['luma']['low']:.1f}/{sw['luma']['mid']:.1f}/{sw['luma']['high']:.1f}"
           f" · 色度 {sw['chroma']['low']:.1f}/{sw['chroma']['mid']:.1f}/{sw['chroma']['high']:.1f}"
           f" · 饱和度 ×{sw['saturation']:.2f} · 全局对比 ×{sw['contrast']:.2f} · 逐点 LUT 可解释 {sw['lut_r2'] * 100:.0f}%")
    return [(c, jpg(x)) for c, x in t], cap


def main():
    m = json.load(open(os.path.join(OUT, "metrics.json"), encoding="utf-8"))
    names = list(m["sweep"])
    rows = []
    for n in names:
        other = "chart" if n != "chart" else None
        ts, cap = tiles(n, m, other)
        cells = "".join(f'<figure><img src="{u}"><figcaption>{c}</figcaption></figure>' for c, u in ts)
        rows.append(f"<section><h2>{n}</h2><p>{cap}</p><div class=g>{cells}</div></section>")
    html = f"""<!doctype html><meta charset=utf-8><title>DLSS5 Anatomy</title>
<style>:root{{--bg:#111;--fg:#ddd;--mut:#999}}body{{background:var(--bg);color:var(--fg);font:14px system-ui;margin:16px}}
.g{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:8px}}figure{{margin:0}}img{{width:100%;display:block}}
figcaption{{color:var(--mut);font-size:12px;padding:2px 0}}h2{{margin:24px 0 4px}}p{{color:var(--mut);margin:0 0 8px}}</style>
<h1>DLSS5 网络在做什么</h1><p>重置帧 (无历史)，1080p，默认参数。红 = 变亮/正，蓝 = 变暗/负。</p>{''.join(rows)}"""
    p = os.path.join(OUT, "atlas.html")
    open(p, "w", encoding="utf-8").write(html)
    print(p, f"{os.path.getsize(p) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
