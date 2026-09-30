"""GPU 数值原语。舍入点与 kernel / numpy 参考一致: f16() 对应 kernel 里的 f16 累加/存储，q8() 对应 FP8 e4m3 量化。"""
import math

import torch

EPS = 6.2e-05
ARITH_Q8 = False      # True: f16/fp8 舍入改用纯算术实现 (torch.compile 用: triton 在 sm_86 上不支持 fp8 dtype，inductor 会省掉 half 往返)；与转换逐位一致


def f16(x):
    """舍入到 f16 (保持 f32 存储)"""
    if ARITH_Q8:                                                          # torch.compile 下 inductor 会省掉 half 往返，改用算术舍入
        e = (((x.view(torch.int32) >> 23) & 0xFF) - 127).clamp(min=-14)  # 低于 2^-14 按非规格化数的固定步长
        s = torch.exp2((e - 10).float())                                  # f16: 10 位尾数 -> 步长 2^(e-10)
        r = torch.round(x / s) * s
        return torch.where(r.abs() > 65504.0, r.sign() * float("inf"), r)
    return x.half().float()




def q8(x):
    """RNE 到 e4m3 (satfinite)"""
    if ARITH_Q8:
        x = x.clamp(-448.0, 448.0)
        e = (((x.view(torch.int32) >> 23) & 0xFF) - 127).clamp(min=-6)   # 指数，低于 2^-6 按非规格化数的固定步长
        s = torch.exp2((e - 3).float())                                   # e4m3: 3 位尾数 -> 量化步长 2^(e-3)
        return torch.round(x / s) * s                                     # round 为四舍六入五成双 (= RNE)
    return x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()


def act(x):
    """MpCubicSiLU: x·(-0.0559·t|t| + 0.4473·t + 0.8945)，t = clamp(x, ±4)"""
    t = x.clamp(-4.0, 4.0)
    return x * (-0.055908203 * t * t.abs() + 0.447265625 * t + 0.894531250)


def cos_norm(x):
    return x / torch.sqrt((x * x).sum(-1, keepdim=True).clamp_min(EPS))


# ------------------------------------------------------------------ 8x8 窗口
def windows(Y, H, W, shift):
    """(H*W, C) 行优先 -> (窗口数, 64, C)。shift = (y, x)，越界 token 为 0"""
    C = Y.shape[1]
    oy, ox = -shift[0], -shift[1]
    ny, nx = (H + oy + 7) // 8, (W + ox + 7) // 8
    pad = Y.new_zeros(ny * 8, nx * 8, C)
    pad[oy:oy + H, ox:ox + W] = Y.view(H, W, C)
    return pad.view(ny, 8, nx, 8, C).permute(0, 2, 1, 3, 4).reshape(-1, 64, C), (ny, nx, oy, ox)


def unwindows(O, meta, H, W):
    ny, nx, oy, ox = meta
    C = O.shape[-1]
    img = O.reshape(ny, nx, 8, 8, C).permute(0, 2, 1, 3, 4).reshape(ny * 8, nx * 8, C)
    return img[oy:oy + H, ox:ox + W].reshape(H * W, C)


# ------------------------------------------------------------------ 位运算 exp (16h / ViT softmax 分子)
def _h(x):
    """Python 标量预先舍入到 f16 (与 kernel 里 cvt.rn.f16.f32 的常数一致)"""
    return float(torch.tensor(x, dtype=torch.float16))


def exp_bits(L, mul, add, lo, hi, shift, addc):
    """y = clamp(f16(f16(L)*mul + add), lo, hi)；结果 = f16 位 ((y_bits << shift) + addc) 的低 16 位。
    常数先舍入到 f16；half 张量乘/加 Python 标量时按 f32 计算后舍入到 f16 (= 两次 f16 运算)，不产生主机->显存拷贝"""
    y = f16(f16(f16(L) * _h(mul)) + _h(add)).clamp(_h(lo), _h(hi)).half()   # 两次 f16 运算 (与 numpy 参考一致)
    bits = ((y.view(torch.int16).int() & 0xFFFF) << shift) + (addc & 0xFFFF)
    bits = bits & 0xFFFF
    bits = torch.where(bits >= 32768, bits - 65536, bits).to(torch.int16)
    return bits.view(torch.float16).float()


EXP16 = (0.04491037502884865, 1.3008946180343628, 1.03125, 1.5693359375, 5, 0x7FF88000)   # p ≈ 0.024·e^L，L 截断 [-6, 6]
EXPVIT = (0.08953946828842163, 1.7093614339828491, 1.439453125, 1.9775390625, 4, 0x3FFC4000)  # p ≈ 0.084·e^L，L 截断 [-3, 3]


def inv_sum(p):
    s = p.half().sum(-1, keepdim=True).float()
    return f16(1.0 / s.clamp_min(6.1e-5))


# ------------------------------------------------------------------ 纹理采样 (CUDA 纹理同式: 坐标*size-0.5，clamp)
def bilinear(img, u, v):
    Hh, Ww = img.shape[:2]
    xx = (u * Ww - 0.5).clamp(0, Ww - 1)
    yy = (v * Hh - 0.5).clamp(0, Hh - 1)
    x0, y0 = xx.floor().long(), yy.floor().long()
    x1, y1 = (x0 + 1).clamp(max=Ww - 1), (y0 + 1).clamp(max=Hh - 1)
    fx, fy = (xx - x0)[..., None], (yy - y0)[..., None]
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy)
            + img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)


def catmull_rom5(img, u, v):
    """Jimenez 5-tap Catmull-Rom (pre/post 两处的历史重投影都用它)"""
    Hh, Ww = img.shape[:2]
    px, py = u * Ww, v * Hh
    cx, cy = (px - 0.5).floor() + 0.5, (py - 0.5).floor() + 0.5
    fx, fy = (px - cx).clamp(0, 1), (py - cy).clamp(0, 1)
    w0x, w0y = fx * (-0.5 + fx * (1 - 0.5 * fx)), fy * (-0.5 + fy * (1 - 0.5 * fy))
    w1x, w1y = 1 + fx * fx * (-2.5 + 1.5 * fx), 1 + fy * fy * (-2.5 + 1.5 * fy)
    w2x, w2y = fx * (0.5 + fx * (2 - 1.5 * fx)), fy * (0.5 + fy * (2 - 1.5 * fy))
    w3x, w3y = fx * fx * (-0.5 + 0.5 * fx), fy * fy * (-0.5 + 0.5 * fy)
    w12x, w12y = w1x + w2x, w1y + w2y
    t12x = (cx + w2x / w12x).clamp(0.5, Ww - 0.5) / Ww
    t12y = (cy + w2y / w12y).clamp(0.5, Hh - 0.5) / Hh
    t0x, t0y = (cx - 1).clamp(0.5, Ww - 0.5) / Ww, (cy - 1).clamp(0.5, Hh - 0.5) / Hh
    t3x, t3y = (cx + 2).clamp(0.5, Ww - 0.5) / Ww, (cy + 2).clamp(0.5, Hh - 0.5) / Hh
    a, b = (w12x * w0y)[..., None], (w0x * w12y)[..., None]
    c, d, e = (w12x * w12y)[..., None], (w3x * w12y)[..., None], (w12x * w3y)[..., None]
    s = (bilinear(img, t12x, t0y) * a + bilinear(img, t0x, t12y) * b + bilinear(img, t12x, t12y) * c
         + bilinear(img, t3x, t12y) * d + bilinear(img, t12x, t3y) * e)
    return s / (a + b + c + d + e)


# ------------------------------------------------------------------ pre_block 噪声 (PCG-RXS 哈希 + Box-Muller)
_M32 = 0xFFFFFFFF
_STREAMS = [(747796405, -1403630843 & _M32), (-93469191 & _M32, 1192405134),
            (-895109107 & _M32, 568162667), (-2094846927 & _M32, 878960812)]


def _mul(a, b):
    return (a * b) & _M32                  # int64 乘法溢出按补码回绕，低 32 位仍正确


def _rxs(s):
    return _mul((s >> ((s >> 28) + 4)) ^ s, 277803737)


def noise(H, W, frame, device):
    """(3, H, W) 个 N(0,1)。seed = x*0x8DA6B343 ^ frame*0x9E3779B9 ^ y*0xD8163841 ^ 0x243F6A88。
    frame 可以是 int 或 int64 标量张量 (CUDA Graph 下帧号由张量传入)"""
    y, x = torch.meshgrid(torch.arange(H, device=device, dtype=torch.int64),
                          torch.arange(W, device=device, dtype=torch.int64), indexing="ij")
    seed = _mul(x, 0x8DA6B343) ^ ((frame * 0x9E3779B9) & _M32) ^ _mul(y, 0xD8163841) ^ 0x243F6A88
    w = _rxs(seed)
    h0 = (w >> 22) ^ w
    u = []
    for a, b in _STREAMS:
        ww = _rxs((_mul(h0, a) + b) & _M32)
        u.append(((((ww >> 30) ^ (ww >> 8)) & _M32) + 1).double() * 2.0 ** -24)
    r1, r2 = torch.sqrt(-2 * torch.log(u[0])), torch.sqrt(-2 * torch.log(u[2]))
    t1, t2 = 2 * math.pi * u[1], 2 * math.pi * u[3]
    return torch.stack([r1 * torch.cos(t1), r1 * torch.sin(t1), r2 * torch.cos(t2)]).float()
