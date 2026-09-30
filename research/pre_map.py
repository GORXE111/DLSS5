"""pre_block 输入适配器全解析: 512 个 f16 位置逐个置 1，得到 (内存位置 -> 输出通道, 输入通道)。
输入通道由 '输出图像完全相同' 分组，再用平滑渐变纹理识别每组是颜色/历史的哪个分量。"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import krun  # noqa: E402
import pre_probe as P  # noqa: E402

H, W = 360, 640
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
COLOR = np.stack([xx / W, yy / H, 0.5 + 0.4 * np.sin(xx / 37.0)], -1)
HIST = np.stack([1 - yy / H, 0.5 + 0.4 * np.cos(yy / 23.0), (xx + yy) / (W + H)], -1)
MV = np.zeros((H, W, 2), np.float32)
REFS = {f"颜色{c}": COLOR[:, :, i] for i, c in enumerate("RGB")}
REFS.update({f"历史{c}": HIST[:, :, i] for i, c in enumerate("RGB")})


def main():
    krun.TRACE = P.TRACE
    _, _, w = krun.weight_of(1)
    base = bytearray(P.bypass_weights(w, np.zeros((16, 32))))
    groups = {}                     # 输出图像指纹 -> [内存位置]
    elem = {}
    for e in range(512):
        ww = bytearray(base)
        ww[8208 + 2 * e:8210 + 2 * e] = np.float16(1).tobytes()
        out = P.run(COLOR, MV, HIST, bytes(ww))
        lit = [c for c in range(32) if np.abs(out[c]).max() > 0]
        if not lit:
            elem[e] = (None, None)
            continue
        img = out[lit[0]]
        key = hash(np.round(img, 4).tobytes())
        groups.setdefault(key, {"img": img, "elems": []})["elems"].append(e)
        elem[e] = (lit[0], key)
    # 识别每组输入
    names = {}
    for key, gdat in groups.items():
        img = gdat["img"][:180]
        best, bv = "未知", 0.0
        for nm, ref in REFS.items():
            r = np.corrcoef(img.ravel(), ref.reshape(180, 2, 320, 2).mean((1, 3)).ravel())[0, 1]
            if abs(np.nan_to_num(r)) > abs(bv):
                best, bv = nm, r
        if img.std() < 1e-6:
            best, bv = f"常数 {img.mean():.3f}", 1.0
        elif abs(bv) < 0.5:
            best = f"噪声/其他 (std {img.std():.3f})"
        names[key] = (best, round(float(bv), 4), len(gdat["elems"]))
    for key, (nm, r, n) in sorted(names.items(), key=lambda kv: kv[1][0]):
        print(f"{nm:24s} 相关 {r:+.4f}  对应 {n} 个权重位置")
    json.dump({str(e): [c, names[k][0] if k is not None else None] for e, (c, k) in elem.items()},
              open(os.path.join(os.path.dirname(__file__), "adapter_map.json"), "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
