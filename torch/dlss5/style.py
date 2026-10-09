"""DLSSNR.Style 1/2 的调色后处理 (cg2r_post_process_kernel，fatbin_14 的无迁移分支，逐行对照 PTX)。

DLL 在 Style != 0 时多跑两个 kernel: 开头 cg2r_copy_kernel 留一份原输入，网络之后 cg2r_post_process_kernel:
    out = sat(input + intensity · (grade(nr) - input))
grade 的参数来自 DLL 常量区的预设表 (nvngx_dlssnr.dll 文件偏移 0xaffa8 起，每项 0x44 字节，kernel 参数 +316..+368):
    black, white, exposure, gamma, contrast, saturation, vibrance, warm, tint, zone[5]
"""
import torch

#            black white exposure gamma contrast saturation vibrance warm tint zones
STYLES = {1: dict(black=0.0, white=1.0, exposure=-0.1, gamma=0.0, contrast=-0.25, saturation=-0.1, vibrance=0.0,
                  warm=0.0, tint=0.0, zones=(0.0,) * 5),
          2: dict(black=0.0, white=1.0, exposure=0.0, gamma=0.0, contrast=0.0, saturation=-0.15, vibrance=0.0,
                  warm=0.0, tint=0.0, zones=(0.0,) * 5)}


def _sat(x):
    return x.clamp(0.0, 1.0)


def _smooth(t):
    t = _sat(t)
    return t * t * (3 - 2 * t)


def _hsl(c):
    """RGB -> (h, s, l)，与 PTX 相同的分支 (h ∈ [0,1))"""
    r, g, b = c.unbind(-1)
    mx, mn = c.max(-1).values, c.min(-1).values
    l = (mx + mn) * 0.5
    d = mx - mn
    chroma = mx > mn
    s = torch.where(l > 0.5, d / (2 - mx - mn), d / (mx + mn).clamp_min(1e-30))
    dd = d.clamp_min(1e-30)
    h = torch.where(mx == r, ((g - b) / dd + torch.where(g < b, 6.0, 0.0)) / 6,
                    torch.where(mx == g, ((b - r) / dd + 2) / 6, ((r - g) / dd + 4) / 6))
    return torch.where(chroma, h, 0.0), torch.where(chroma, s, 0.0), l


def _hue2rgb(p, q, t):
    t = torch.where(t < 0, t + 1, t)
    t = torch.where(t > 1, t - 1, t)
    return torch.where(t < 1 / 6, p + (q - p) * 6 * t,
                       torch.where(t < 0.5, q, torch.where(t < 2 / 3, p + (q - p) * (2 / 3 - t) * 6, p)))


def _from_hsl(h, s, l):
    q = torch.where(l < 0.5, l * (1 + s), l + s - l * s)
    p = 2 * l - q
    rgb = torch.stack([_hue2rgb(p, q, h + 1 / 3), _hue2rgb(p, q, h), _hue2rgb(p, q, h - 1 / 3)], -1)
    return torch.where((s > 0)[..., None], rgb, l[..., None].expand_as(rgb))


def _toward_hue(c, amount, pos, neg):
    """朝一个满饱和、同 HSL 亮度的颜色混 |amount|。pos/neg: (通道顺序, 色相分量) —— PTX 里把 HSL->RGB 展开成常数"""
    if abs(amount) < 1e-6:
        return c
    mx, mn = c.max(-1).values, c.min(-1).values
    l = (mx + mn) * 0.5
    q = torch.where(l < 0.5, 2 * l, torch.ones_like(l))
    p = 2 * l - q
    (hi, lo, mid), frac = pos if amount > 0 else neg
    tgt = torch.empty_like(c)
    tgt[..., hi], tgt[..., lo], tgt[..., mid] = q, p, p + (q - p) * 6 * frac
    return _sat(c + abs(amount) * (tgt - c))


def grade(c, black=0.0, white=1.0, exposure=0.0, gamma=0.0, contrast=0.0, saturation=0.0, vibrance=0.0,
          warm=0.0, tint=0.0, zones=(0.0,) * 5):
    """c: (..., 3) [0,1]，显示编码 (与网络输出同一空间)"""
    x = _sat((_sat(c) - black) / (white - black + 1e-10))
    x = _toward_hue(x, warm, ((0, 2, 1), 0.0611100010573864), ((2, 0, 1), 0.1055566668510437))     # 暖 (橙) / 冷 (蓝)
    x = _toward_hue(x, tint, ((1, 2, 0), 0.022223353385925293), ((2, 1, 0), 0.14444339275360107))   # 绿 / 品红
    x = _sat(x * 2.0 ** exposure)
    x = _sat(x + contrast * (_smooth(x) - x))
    w = torch.stack([_smooth((x - 0.25) / -0.25),
                     _smooth(x / 0.25) * _smooth((x - 0.5) / -0.25),
                     _smooth((x - 0.25) / 0.25) * _smooth((x - 0.75) / -0.25),
                     _smooth((x - 0.5) / 0.25) * _smooth((x - 1.0) / -0.25),
                     _smooth((x - 0.75) / 0.25)], -1)
    e = (w * torch.tensor(zones, dtype=x.dtype, device=x.device)).sum(-1)
    x = x.clamp_min(0) ** (2.0 ** -e)
    x = x.clamp_min(0) ** (2.0 ** -gamma)
    h, s, l = _hsl(x)
    x = _from_hsl(h, _sat(s * (1 + saturation)), l)
    h, s, l = _hsl(x)
    x = _from_hsl(h, _sat(s.clamp_min(0) ** (2.0 ** -vibrance)), l)
    return _sat(x)


def apply(color, nr, style, intensity=1.0):
    """Style != 0 时 DLL 的最终输出: sat(color + t · (grade(nr) - color))，t = Intensity (夹到 [0,1])"""
    t = min(max(float(intensity), 0.0), 1.0)
    return _sat(color + t * (grade(nr, **STYLES[style]) - color))
