"""同一帧的处理前后对比图 (代理 DumpFrame 存下的 dlss5fb_in.ppm / dlss5fb_out.ppm)。

    python compare_frames.py 输出.png 原画面.ppm 结果.ppm[=标题] [结果2.ppm[=标题]]

  例:  compare_frames.py pair.png test/sp_cap/dlss5fb_in.ppm test/sp_cap/dlss5fb_out.ppm
       compare_frames.py three.png in.ppm out_s035.ppm="比例 0.35" out_s100.ppm="比例 1.0 (原生)"

各结果必须来自同一张原画面 (要比较不同的 WorkingScale，把存下的 dlss5fb_in.ppm 交给 fbtest 用不同设置各处理一遍)。
图上: 第一行 原画面与各结果；第二行 各结果的亮度改动 (红 = 变亮，蓝 = 变暗，满色 = 32/255)；
之后是改动 (两个结果时: 两者差别) 最大的两块局部，原始大小不缩放。同时打印每个结果的统计。
这类图含游戏 / 测试程序的画面，只放本地 (例如 _dl/compare/)，不入库。
"""
import re
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

LUM = np.array([0.299, 0.587, 0.114])


def ppm(path):
    d = open(path, "rb").read()
    m = re.match(rb"P6\s+(\d+)\s+(\d+)\s+255\s", d)
    w, h = int(m.group(1)), int(m.group(2))
    return np.frombuffer(d[-w * h * 3:], np.uint8).reshape(h, w, 3)


def font(size):
    for name in ("msyh.ttc", "simhei.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def label(img, text, f):
    im = Image.fromarray(np.ascontiguousarray(img))
    d = ImageDraw.Draw(im)
    box = d.textbbox((0, 0), text, font=f)
    d.rectangle((8, 8, 8 + box[2] + 16, 8 + box[3] + 14), fill=(0, 0, 0))
    d.text((16, 12), text, fill=(255, 255, 255), font=f)
    return np.asarray(im)


def diffmap(d):
    t = np.clip((d @ LUM) / 32.0, -1, 1)
    m = np.zeros(d.shape, np.uint8)
    m[..., 0] = 128 + 127 * np.clip(t, 0, 1) - 60 * np.clip(-t, 0, 1)
    m[..., 1] = 128 - 80 * np.abs(t)
    m[..., 2] = 128 + 127 * np.clip(-t, 0, 1) - 60 * np.clip(t, 0, 1)
    return m


def box_blur(x, k=9):
    pad = np.pad(x, k // 2, mode="edge")
    cs = np.cumsum(np.cumsum(np.pad(pad, ((1, 0), (1, 0))), 0), 1)
    return (cs[k:, k:] - cs[:-k, k:] - cs[k:, :-k] + cs[:-k, :-k]) / (k * k)


def stats(a, b):
    d = b.astype(int) - a.astype(int)
    ya, yb = a @ LUM, b @ LUM
    ca = np.linalg.norm(a - ya[..., None], axis=-1).mean()
    cb = np.linalg.norm(b - yb[..., None], axis=-1).mean()
    dl = (d @ LUM).astype(np.float32)
    return {"d": d, "mean": np.abs(d).mean(), "gt2": 100 * (np.abs(d).max(-1) > 2).mean(), "gt16": 100 * (np.abs(d).max(-1) > 16).mean(),
            "max": int(np.abs(d).max()), "contrast": yb.std() / ya.std(), "sat": cb / max(ca, 1e-6),
            "detail": np.abs(dl - box_blur(dl)).mean()}      # 改动里的细节成分: 改动减去它自己的 9x9 均值


def main():
    out, src = sys.argv[1], sys.argv[2]
    results = [(x.split("=", 1) + ["DLSS5"])[:2] for x in sys.argv[3:5]]
    a = ppm(src)
    imgs = [(ppm(p), t) for p, t in results]
    H, W = a.shape[:2]
    n = 1 + len(imgs)
    big, small = font(30), font(22)
    scale = 2 / 3 if n == 3 else 1
    fit = lambda x: np.asarray(Image.fromarray(np.ascontiguousarray(x)).resize((int(W * scale), int(H * scale)), Image.LANCZOS))   # noqa: E731
    st = [stats(a, b) for b, _ in imgs]
    rows = [np.concatenate([label(fit(a), "原画面 (DLSS5 关)", big)] + [label(fit(b), t, big) for b, t in imgs], 1),
            np.concatenate([label(fit(np.full_like(a, 128)), "亮度改动: 红 = 变亮，蓝 = 变暗", big)]
                           + [label(fit(diffmap(s["d"])), f"{t}: 平均改动 {s['mean']:.1f}/255", big) for s, (_, t) in zip(st, imgs)], 1)]
    # 两块局部: 两个结果时取两者差别最大的地方，否则取改动最大的地方
    dd = np.abs(imgs[1][0].astype(int) - imgs[0][0].astype(int)).max(-1) if len(imgs) == 2 else np.abs(st[0]["d"]).max(-1)
    cw, ch = W // n, H // 3
    score = sorted(((dd[y:y + ch, x:x + cw].mean(), y, x) for y in range(0, H - ch + 1, ch // 3) for x in range(0, W - cw + 1, cw // 3)), reverse=True)
    picked = []
    for s, y, x in score:
        if all(abs(y - py) >= ch or abs(x - px) >= cw for _, py, px in picked):
            picked.append((s, y, x))
        if len(picked) == 2:
            break
    for _, y, x in picked:
        rows.append(np.concatenate([label(a[y:y + ch, x:x + cw], "原 (局部，原始大小)", small)]
                                   + [label(b[y:y + ch, x:x + cw], t, small) for b, t in imgs], 1))
    wmax = max(r.shape[1] for r in rows)
    rows = [np.concatenate([r, np.zeros((r.shape[0], wmax - r.shape[1], 3), np.uint8)], 1) for r in rows]
    Image.fromarray(np.concatenate(rows, 0)).save(out)
    print(f"{W}x{H} -> {out}")
    for s, (_, t) in zip(st, imgs):
        print(f"  {t}: 平均改动 {s['mean']:.1f}/255，改动超过 2/255 的像素 {s['gt2']:.0f}%，超过 16/255 的 {s['gt16']:.0f}%，最大 {s['max']}；"
              f"对比度 x{s['contrast']:.2f}，饱和度 x{s['sat']:.2f}，改动里的细节成分 {s['detail']:.2f}/255")
    if len(imgs) == 2:
        print(f"  两个结果之间: 平均相差 {np.abs(imgs[1][0].astype(int) - imgs[0][0].astype(int)).mean():.1f}/255")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
