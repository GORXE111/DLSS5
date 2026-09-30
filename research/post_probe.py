"""post_block (block70, seq154) 末层合成: 纹理/surface 自造，受控实验验证 PTX 读出的公式。

PTX (kpost.ptx 13900-14126) 读出的合成 (每个全分辨率像素):
  net = (r, g, b, a) 网络输出 4 通道 (f16)
  cur = clamp(color + 8*s*net.rgb, 0, 1)            s = 参数 +48 的 f32 (0.03125)，即 color + 0.25*net.rgb
  gate = clamp(sigmoid(net.a) * blend_scale, 0, 1)   blend_scale = 参数 +104 指向的 f16 (block70.layer0.blend_scale = 0.7397)
  hist = CatmullRom5(history, uv + mv * (1/W, 1/H))  参数 +112 == 0 时不用运动矢量
  out  = cur + gate * (hist - cur)，alpha = 1        (无历史纹理或 blend_scale<=0 时 out = cur)
  参数 +52 == 0: 原始模式，不钳位不混合，out = color*0.125-0.0625 + s*net，alpha = 0
"""
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import klab  # noqa: E402
import krun  # noqa: E402

TRACE = os.path.join(os.path.dirname(__file__), "tapspost", "nr-trace.tsv")
SEQ = 154
H, W = 360, 640
OUT_CONV = (20784, 21808)


def real_input():
    return open(os.path.join(os.path.dirname(__file__), "tapspost", "tap_s154_1c93b000.bin"), "rb").read()


def run(color, hist, mv, w, x=None, blend=0.7397, extra=()):
    krun.TRACE = TRACE
    row = klab.trace_row(SEQ, TRACE)
    p = bytearray(row["params"])
    texs = [klab.texture(color), klab.texture(hist), klab.texture(mv, linear=False)]
    for off, (h, _) in zip((56, 88, 96), texs):
        struct.pack_into("<Q", p, off, h)
    surf = klab.Surface(W, H)
    surf.fill(-7.0)
    struct.pack_into("<Q", p, 16, surf.handle)
    for off, fmt, val in extra:
        struct.pack_into(fmt, p, off, val)
    q = lambda o: struct.unpack_from("<Q", p, o)[0]  # noqa: E731
    xin = klab.gpu_bytes(x if x is not None else real_input(), 0)
    wbuf = klab.gpu_bytes(w, 512)
    bs = klab.gpu_bytes(np.float16(blend).tobytes(), 510)
    cnt = klab.gpu_bytes(b"", 1 << 20)
    m = {q(0): xin.data_ptr(), q(24): wbuf.data_ptr(), q(104): bs.data_ptr()}
    for i in range(0, len(p) // 8 * 8, 8):
        v = struct.unpack_from("<Q", p, i)[0]
        if 0x1BA00000 <= v < 0x1BE00000 and v not in m:
            m[v] = cnt.data_ptr() + (v & 0xFFFFF)
    p = klab.patch_ptrs(bytes(p), m)
    fn = klab.function("01", row["kernel"])
    klab.launch(fn, row["grid"], row["block"], row["smem"], p)
    return surf.read()


def catmull_rom5(img, u, v):
    """Jimenez 5-tap Catmull-Rom (与 PTX 14000-14081 同式)，u,v 为归一化坐标; 用双线性 (clamp) 采样"""
    Hh, Ww = img.shape[:2]

    def bil(uu, vv):
        xx = np.clip(uu * Ww - 0.5, 0, Ww - 1)
        yy = np.clip(vv * Hh - 0.5, 0, Hh - 1)
        x0 = np.floor(xx).astype(int)
        y0 = np.floor(yy).astype(int)
        x1, y1 = np.minimum(x0 + 1, Ww - 1), np.minimum(y0 + 1, Hh - 1)
        fx, fy = (xx - x0)[..., None], (yy - y0)[..., None]
        return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy)
                + img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)

    px, py = u * Ww, v * Hh
    cx, cy = np.floor(px - 0.5) + 0.5, np.floor(py - 0.5) + 0.5
    fx, fy = np.clip(px - cx, 0, 1), np.clip(py - cy, 0, 1)
    w0x, w0y = fx * (-0.5 + fx * (1 - 0.5 * fx)), fy * (-0.5 + fy * (1 - 0.5 * fy))
    w1x, w1y = 1 + fx * fx * (-2.5 + 1.5 * fx), 1 + fy * fy * (-2.5 + 1.5 * fy)
    w2x, w2y = fx * (0.5 + fx * (2 - 1.5 * fx)), fy * (0.5 + fy * (2 - 1.5 * fy))
    w3x, w3y = fx * fx * (-0.5 + 0.5 * fx), fy * fy * (-0.5 + 0.5 * fy)
    w12x, w12y = w1x + w2x, w1y + w2y
    t12x = np.clip(cx + w2x / w12x, 0.5, Ww - 0.5) / Ww
    t12y = np.clip(cy + w2y / w12y, 0.5, Hh - 0.5) / Hh
    t0x, t0y = np.clip(cx - 1, 0.5, Ww - 0.5) / Ww, np.clip(cy - 1, 0.5, Hh - 0.5) / Hh
    t3x, t3y = np.clip(cx + 2, 0.5, Ww - 0.5) / Ww, np.clip(cy + 2, 0.5, Hh - 0.5) / Hh
    a, b = (w12x * w0y)[..., None], (w0x * w12y)[..., None]
    c, d, e = (w12x * w12y)[..., None], (w3x * w12y)[..., None], (w12x * w3y)[..., None]
    s = (bil(t12x, t0y) * a + bil(t0x, t12y) * b + bil(t12x, t12y) * c + bil(t3x, t12y) * d + bil(t12x, t3y) * e)
    return s / (a + b + c + d + e)


def scenes(seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    color = np.stack([xx / W, yy / H, 0.5 + 0.4 * np.sin(xx / 37.0)], -1)
    hist = rng.random((H, W, 3)).astype(np.float32)
    mv = np.stack([np.full((H, W), 3.25), np.full((H, W), -1.5)], -1).astype(np.float32)
    return color, hist, mv


def main():
    krun.TRACE = TRACE
    _, name, w = krun.weight_of(SEQ)
    print("权重记录", name, len(w))
    color, hist, mv = scenes()
    uv = (np.mgrid[0:H, 0:W][::-1].astype(np.float32) + 0.5) / np.array([W, H], np.float32)[:, None, None]
    hcr = catmull_rom5(hist, uv[0] + mv[..., 0] / W, uv[1] + mv[..., 1] / H)

    # 1) 输出卷积置 0 -> net = 0 -> out = lerp(color, hist, 0.5*blend)
    w0 = bytearray(w)
    w0[OUT_CONV[0]:OUT_CONV[1]] = bytes(OUT_CONV[1] - OUT_CONV[0])
    for blend in (0.7397, 0.0, 1.0):
        o = run(color, hist, mv, bytes(w0), blend=blend)
        g = min(max(0.5 * blend, 0), 1)
        ref = color + g * (hcr - color)
        print(f"输出卷积=0, blend={blend}: 最大差 {np.abs(o[..., :3] - ref).max():.5f}  alpha {np.unique(o[..., 3])[:4]}"
              f"  未写像素 {(o[..., 0] == -7).sum()}")
    # 2) 参数 +52 = 0 -> 原始模式: 不做 *8+0.5/钳位/历史，out = (color*0.125-0.0625) + s*net，alpha 0
    o = run(color, hist, mv, bytes(w0), extra=[(52, "<i", 0)])
    print("原始模式 (+52=0): 最大差", np.abs(o[..., :3] - (color * 0.125 - 0.0625)).max(), " alpha", np.unique(o[..., 3]))
    # 3) 真实权重 + 真实输入: 用 blend=0 (关历史) 得到 net.rgb，再用它预测开历史时的输出 (gate 需 net.a，另测)
    oc = run(color, hist, mv, w, blend=0.0)
    oh = run(color, hist, mv, w)
    net_rgb = (oc[..., :3] - color) / 0.25
    gate = (oh[..., :3] - oc[..., :3]) / (hcr - oc[..., :3] + 1e-9)
    ok = np.abs(hcr - oc[..., :3]) > 0.05
    print("真实网络: net.rgb 范围", net_rgb.min(), net_rgb.max(), " 推得 gate 通道间一致性",
          np.abs(gate[..., 0] - gate[..., 1])[ok.all(-1)].mean(), " gate 均值", gate[ok].mean())
    np.save(os.path.join(os.path.dirname(__file__), "post_real.npy"), np.stack([oc, oh]))


if __name__ == "__main__":
    main()
