"""post_block (block70) 网络部分参考实现 + 与 kernel 对照。

记录布局 (21808B，PTX 读址 + 本脚本验证):
  FFN [0,8192) | 16B | c1 [8208,8272) | s1 [8272,8336) | s2 [8336,8400) | qkv [8400,11472) | 偏置 [11472,19664) |
  τ [19664,19668) + 12B | 投影 [19680,20704) | c2 [20704,20768) | 16B | 输出卷积 f16 32->16 [20784,21808)
  = 标准 1h swin 记录在 c1 之后插入 s1/s2 (+112B)，末尾接输出卷积。
计算:
  m   = s1⊙x + s2⊙skip          x = block69 输出 (参数 +0)，skip = pre_block 写到 +216 的特征 (参数 +8)，均为 1h tin
  y   = swin_1h(m)  (FFN 的 W1 吃 q8(m)，残差用 f16 m；输出 y 不再量化)
  net = y @ Wout (32->16) -> 2x2 像素重排 -> 全分辨率 4 通道 (RGB 残差 + 历史门控 logit)
"""
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import block1_ref as R  # noqa: E402
import klab  # noqa: E402
import krun  # noqa: E402
import post_probe as PP  # noqa: E402
from act import chw_to_tin, random_fp8  # noqa: E402
from exp_ffn import mp_cubic_silu, perm_in, perm_param, q8, unswizzle  # noqa: E402

H2, W2 = 180, 320
PI = [perm_in(c) for c in range(32)]
SG = [perm_param(c) for c in range(32)]


def std_record(w):
    """post 记录 -> 标准 1h swin 记录 (20672B)，供 swin_ref 的偏移使用"""
    return bytes(w[:8272]) + bytes(16) + bytes(w[8400:20784])


MX = [8 * (c % 4) + 2 * ((c % 16) // 4) + c // 16 for c in range(32)]      # x 图像通道 c -> 片段位置 j (实测)
MXI = list(np.argsort(MX))                                                   # 片段位置 j <- x 图像通道


def out_pos(j, o):
    """输出卷积 f16 位置: 片段通道 j -> 输出 o (实测, 误差 0)"""
    return 256 * (j % 2) + 8 * ((j % 8) // 2) + j // 8 + 32 * o


def swin_post(x_img, skip_frag, w, H=360, W=640):
    """x_img: (32, 180, 320) 图像格式通道序 (半分辨率); skip_frag: (32, H, W) 全分辨率 tin 内存序 (= 片段序)。
    返回 net (4, H, W)"""
    ws = std_record(w)
    f = lambda a, b: np.frombuffer(bytes(w[a:b]), np.float16).astype(np.float32)  # noqa: E731
    s1, s2, c1, c2 = f(8272, 8336), f(8336, 8400), f(8208, 8272), f(20704, 20768)
    xu = x_img.repeat(2, 1).repeat(2, 2)[:, :H, :W]                            # 最近邻 2x
    Xf = xu.reshape(32, -1).T[:, MXI]                                            # 片段序
    Sf = skip_frag.reshape(32, -1).T
    M_frag = R.f16(s1[SG] * Xf + s2[SG] * Sf)
    M = np.zeros_like(M_frag)
    M[:, PI] = M_frag                                                           # 规范序 (W1 的输入)
    W1 = unswizzle(ws[0:4096], 32, 128)
    W2m = unswizzle(ws[4096:8192], 128, 32)[R.C_MAP, :]
    ff = q8(mp_cubic_silu(R.f16(q8(M) @ W1))) @ W2m
    ff_frag = np.zeros_like(M)
    ff_frag[:, R.A_MAP] = ff
    Y_frag = R.f16(c1[SG] * M_frag + ff_frag)
    Y = np.zeros_like(Y_frag)
    Y[:, PI] = Y_frag
    Y = q8(Y)
    Wqkv = unswizzle(ws[8288:11360], 32, 96)
    tau = np.frombuffer(ws[19552:19556], np.float32)[0]
    Wproj = unswizzle(ws[19568:20592], 32, 32)
    bias = R.bias_matrix(ws)
    vperm = np.argsort(R.C_MAP[:32])
    oy = ox = 4
    ny, nx = (H + oy + 7) // 8, (W + ox + 7) // 8
    pad = np.zeros((ny * 8, nx * 8, 32), np.float32)
    pad[oy:oy + H, ox:ox + W] = Y.reshape(H, W, 32)
    Yw = pad.reshape(ny, 8, nx, 8, 32).transpose(0, 2, 1, 3, 4).reshape(-1, 64, 32)
    q, k, v = R.f16(Yw @ Wqkv[:, :32]), R.f16(Yw @ Wqkv[:, 32:64]), R.f16(Yw @ Wqkv[:, 64:])
    qn = q / np.sqrt(np.maximum((q * q).sum(-1, keepdims=True), R.EPS)) * tau
    kn = k / np.sqrt(np.maximum((k * k).sum(-1, keepdims=True), R.EPS))
    L = R.f16(q8(qn) @ q8(kn).transpose(0, 2, 1) + bias)
    e = np.exp(L - L.max(-1, keepdims=True))
    P = e / e.sum(-1, keepdims=True)
    o = R.f16(q8(P) @ q8(v[:, :, vperm]))
    o_img = o.reshape(ny, nx, 8, 8, 32).transpose(0, 2, 1, 3, 4).reshape(ny * 8, nx * 8, 32)[oy:oy + H, ox:ox + W]
    pr = o_img.reshape(-1, 32) @ Wproj
    pr_frag = np.zeros_like(pr)
    pr_frag[:, R.A_MAP] = pr
    y = R.f16(c2[SG] * Y_frag + pr_frag)                                        # 不再量化，直接进 f16 输出卷积
    Wo = f(20784, 21808)
    Wm = np.array([[Wo[out_pos(j, oc)] for oc in range(4)] for j in range(32)])
    return R.f16(y @ Wm).T.reshape(4, H, W)


def x_bytes(x_img):
    """(32,180,320) -> 图像格式 [2][192][320][16] fp8"""
    from act import E4M3_CODE
    keys = np.array(sorted(E4M3_CODE), np.float32)
    codes = np.array([E4M3_CODE[float(k)] for k in keys], np.uint8)
    b = np.zeros((2, 192, 320, 16), np.uint8)
    b[:, :180] = codes[np.searchsorted(keys, x_img.reshape(2, 16, 180, 320).transpose(0, 2, 3, 1))]
    return b.tobytes()


def kernel_net(x_img, skip_frag, w):
    """原始模式 (+52=0) 且无颜色纹理 -> surface = s * net.rgb (s = 0.03125)。返回 (3, 360, 640)"""
    zero = np.zeros((PP.H, PP.W, 3), np.float32)
    row = klab.trace_row(PP.SEQ, PP.TRACE)
    skip_va = struct.unpack_from("<Q", row["params"], 8)[0]
    o = run_with_skip(zero, zero, np.zeros((PP.H, PP.W, 2), np.float32), w, x_bytes(x_img), skip_bytes(skip_frag),
                      skip_va, [(52, "<i", 0), (56, "<Q", 0)])
    return o.transpose(2, 0, 1)[:3] / 0.03125


def run_with_skip(color, hist, mv, w, xb, sb, skip_va, extra):
    orig = klab.gpu_bytes
    sk = klab.gpu_bytes(sb, 0)
    real_patch = klab.patch_ptrs

    def patch(p, m):
        m = dict(m)
        m[skip_va] = sk.data_ptr()
        return real_patch(p, m)
    klab.patch_ptrs = patch
    try:
        return PP.run(color, hist, mv, w, x=xb, extra=extra)
    finally:
        klab.patch_ptrs = real_patch
        klab.gpu_bytes = orig


def compose(net, color, hist, mv, blend=0.7397, scale=0.03125):
    """末层合成 (PTX 13900-14126): cur = clamp(color + 8*scale*net.rgb)；gate = clamp(sigmoid(net.a)*blend)；
    out = cur + gate*(CatmullRom(hist, uv + mv/size) - cur)"""
    H, W = color.shape[:2]
    cur = np.clip(color + 8 * scale * net[:3].transpose(1, 2, 0), 0, 1)
    gate = np.clip(1 / (1 + np.exp(-net[3])) * blend, 0, 1)[..., None]
    uv = (np.mgrid[0:H, 0:W][::-1].astype(np.float32) + 0.5) / np.array([W, H], np.float32)[:, None, None]
    hcr = PP.catmull_rom5(hist, uv[0] + mv[..., 0] / W, uv[1] + mv[..., 1] / H)
    return cur + gate * (hcr - cur)


def skip_bytes(skip_frag):
    """skip 缓冲按参数 +32 的 (384, 640) 分配: kernel 把 384 行都当有效 token 读，360 行以下须为 0 (真实流程由 pre_block 写 0)"""
    pad = np.zeros((32, 384, skip_frag.shape[2]), np.float32)
    pad[:, :skip_frag.shape[1]] = skip_frag
    return chw_to_tin(pad)


def kernel_full(x_img, skip_frag, w, color, hist, mv):
    row = klab.trace_row(PP.SEQ, PP.TRACE)
    skip_va = struct.unpack_from("<Q", row["params"], 8)[0]
    o = run_with_skip(color, hist, mv, w, x_bytes(x_img), skip_bytes(skip_frag), skip_va, [])
    return o[..., :3]


def main():
    krun.TRACE = PP.TRACE
    _, _, w = krun.weight_of(PP.SEQ)
    for seed, lo in ((1, -4), (3, -1)):
        x = random_fp8((32, 180, 320), lo=lo, hi=-lo, seed=seed)
        sk = random_fp8((32, 360, 640), lo=lo, hi=-lo, seed=seed + 1)
        k = kernel_net(x, sk, w)
        r = swin_post(x, sk, w)[:3]
        c = np.corrcoef(k.ravel(), r.ravel())[0, 1]
        print(f"随机输入 ±{-lo}: 相关 {c:.5f}  最大差 {np.abs(k - r).max():.4f}  平均差 {np.abs(k - r).mean():.5f}  "
              f"kernel std {k.std():.4f}")
    color, hist, mv = PP.scenes()
    x = random_fp8((32, 180, 320), lo=-2, hi=2, seed=5)
    sk = random_fp8((32, 360, 640), lo=-2, hi=2, seed=6)
    k = kernel_full(x, sk, w, color, hist, mv)
    r = compose(swin_post(x, sk, w), color, hist, mv)
    print(f"完整合成 (网络+门控+历史): 相关 {np.corrcoef(k.ravel(), r.ravel())[0, 1]:.5f}  平均差 {np.abs(k - r).mean():.5f}  "
          f"最大差 {np.abs(k - r).max():.4f}  99% 分位 {np.quantile(np.abs(k - r), 0.99):.4f}")


if __name__ == "__main__":
    main()
