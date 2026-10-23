"""DLSSNR.Style 1/2 调色/UI 后处理，以及采用实测 RTX 3060 采样模型的独立实验性 OKLab 迁移 helper。

DLL 在 Style != 0 时多跑两个 kernel: 开头 cg2r_copy_kernel 留一份原输入，网络之后 cg2r_post_process_kernel:
    out = sat(input + intensity · (grade(nr) - input))
grade 的参数来自 DLL 常量区的预设表 (nvngx_dlssnr.dll 文件偏移 0xaffa8 起，每项 0x44 字节，kernel 参数 +316..+368):
    black, white, exposure, gamma, contrast, saturation, vibrance, warm, tint, zone[5]
"""
import math

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
    for name, value in (("black", black), ("white", white), ("exposure", exposure), ("gamma", gamma),
                        ("contrast", contrast), ("saturation", saturation), ("vibrance", vibrance),
                        ("warm", warm), ("tint", tint)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be a finite real number")
    if len(zones) != 5 or not all(math.isfinite(float(v)) for v in zones):
        raise ValueError("zones must contain five finite real numbers")
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


def apply(color, nr, style=0, intensity=1.0, control_mask=None, ui_correction=False,
          ui=None, ui_alpha=None, backbuffer=None):
    """cg2r_post_process_kernel 的无迁移分支 (同尺寸、显示编码 RGB)。

    out = sat(base + sat(Intensity * mask.R * (1-sat(alpha))) * (grade(nr)-color))
    base = Backbuffer (仅 UICorrection 开启时)，否则 Color；alpha 优先取 UIAlpha.R，
    其次 UI.a，缺失为 0。ControlMask 不存在时其 R 为 1；Style 0 为恒等调色。
    UI/UIAlpha/Backbuffer 在 ui_correction=False 时与 DLL 一样被忽略。
    调色与空间控制只应用一次: ControlMask.G/B 已在网络入口应用，R 在此处应用。
    """
    intensity = float(intensity)
    if not math.isfinite(intensity):
        raise ValueError("intensity must be a finite real number")
    if style not in (0, 1, 2):
        raise ValueError("style must be 0, 1, or 2")
    t = min(max(intensity, 0.0), 1.0)
    if control_mask is not None:
        t = t * control_mask[..., :1]
    base = color
    if ui_correction:
        if backbuffer is not None:
            base = backbuffer[..., :3]
        if ui_alpha is not None:
            alpha = ui_alpha[..., None] if ui_alpha.ndim == color.ndim - 1 else ui_alpha[..., :1]
            t = t * (1 - _sat(alpha))
        elif ui is not None:
            t = t * (1 - _sat(ui[..., 3:4]))
    if torch.is_tensor(t):
        t = _sat(t)
    graded = grade(nr, **STYLES[style]) if style else _sat(nr)
    return _sat(base + t * (graded - color))


def _transfer_lab(rgb):
    """PTX RGB -> Lab, including the LMS floor and its cube-root refinement."""
    r, g, b = _sat(rgb[..., :3]).unbind(-1)
    lms = torch.stack([
        (g * 0.5363325476646423 + r * 0.4122214615345001) + b * 0.05144599452614784,
        (g * 0.6806995272636414 + r * 0.21190349757671356) + b * 0.10739696025848389,
        (g * 0.2817188501358032 + r * 0.08830246329307556) + b * 0.6299787163734436,
    ], -1).clamp_min(9.999999747378752e-05)
    inv = torch.exp2(torch.log2(lms) * -0.6666666865348816)
    root = lms * inv
    root = root + root * ((1 - root * root * inv) * 0.3333333432674408)
    l, m, s = root.unbind(-1)
    return torch.stack([
        (m * 0.7936177849769592 + l * 0.21045425534248352) - s * 0.004072046838700771,
        (l * 1.9779984951019287 - m * 2.4285922050476074) + s * 0.4505937099456787,
        (m * 0.7827717661857605 + l * 0.025904037058353424) - s * 0.8086757659912109,
    ], -1)


def _transfer_rgb(lab):
    l, a, b = lab.unbind(-1)
    lr = (l + a * 0.3963377773761749) + b * 0.21580375730991364
    mr = (l - a * 0.10556134581565857) - b * 0.0638541728258133
    sr = (l - a * 0.08948417752981186) - b * 1.2914855480194092
    ll, mm, ss = lr * (lr * lr), mr * (mr * mr), sr * (sr * sr)
    return _sat(torch.stack([
        (mm * -3.307711601257324 + ll * 4.076741695404053) + ss * 0.23096993565559387,
        (ll * -1.2684379816055298 + mm * 2.609757423400879) - ss * 0.34131938219070435,
        (mm * -0.7034186124801636 - ll * 0.004196086432784796) + ss * 1.7076146602630615,
    ], -1))


def _transfer_size(name, value, *, positive=True):
    if len(value) != 2 or any(not math.isfinite(float(v)) or int(v) != v for v in value):
        raise ValueError(f"{name} must contain integer (height, width)")
    size = tuple(map(int, value))
    if positive and min(size) <= 0:
        raise ValueError(f"{name} dimensions must be positive")
    return size


def _transfer_image(name, img, device):
    if not torch.is_tensor(img) or img.ndim != 3 or min(img.shape[:2]) <= 0 or img.shape[-1] not in (3, 4):
        raise ValueError(f"{name} must be a nonempty (H,W,3/4) tensor")
    if img.device != device or not img.is_floating_point() or not torch.isfinite(img).all():
        raise ValueError(f"{name} must contain finite floating values on the same device as nr")
    return img.float()


def _transfer_texture(img, u, v, rect, filtering):
    """Normalized clamp texture following the measured RTX 3060 coordinate/weight model."""
    h, w = img.shape[:2]
    rect = (0, 0, w, h, 1 / w, 1 / h) if rect is None else tuple(rect)
    if len(rect) == 4:
        rect += (1 / w, 1 / h)
    if len(rect) != 6 or not all(math.isfinite(float(x)) for x in rect):
        raise ValueError("subrect must be finite (x,y,width,height[,inverse_resource_width,inverse_resource_height])")
    x, y, rw, rh, iw, ih = rect
    # Each descriptor entry is a float32, even when the caller supplied a Python double.
    x, y, rw, rh, iw, ih = (torch.tensor(a, dtype=torch.float32, device=img.device) for a in (x, y, rw, rh, iw, ih))
    # PTX uses fma.rn for the descriptor affine transform. A double temporary
    # followed by one f32 rounding preserves its decision at point-filter edges.
    nx = ((u.double() * rw.double() + x.double()).float() * iw).double()
    ny = ((v.double() * rh.double() + y.double()).float() * ih).double()
    # Direct TEX probes on RTX 3060 match truncation of normalized coordinates
    # to 21 fractional bits (3,078 neighboring-boundary samples, 11 widths).
    # Apply that coordinate conversion before texel indexing/weight rounding.
    px = (torch.floor(nx * 2097152) / 2097152) * w
    py = (torch.floor(ny * 2097152) / 2097152) * h
    if filtering == "point":
        return img[py.floor().long().clamp(0, h - 1), px.floor().long().clamp(0, w - 1), :3]
    if filtering != "linear":
        raise ValueError("texture filtering must be 'point' or 'linear'")
    px, py = px - 0.5, py - 0.5
    ix, iy = px.floor().long(), py.floor().long()
    fx = (torch.floor((px - ix) * 256 + 0.5) / 256).float()[..., None]
    fy = (torch.floor((py - iy) * 256 + 0.5) / 256).float()[..., None]
    x0, x1, y0, y1 = ix.clamp(0, w - 1), (ix + 1).clamp(0, w - 1), iy.clamp(0, h - 1), (iy + 1).clamp(0, h - 1)
    # RTX 3060's float texture filter also quantizes the bilinear cross weight.
    # A 2x2 {cross, x-ramp, y-ramp} texture distinguishes this from full-precision
    # multiplication of the two quantized axis fractions (check_transfer.py).
    xy = torch.floor(fx * fy * 256 + 0.5) / 256
    return (img[y0, x0, :3] * (1 - fx - fy + xy) + img[y0, x1, :3] * (fx - xy)
            + img[y1, x0, :3] * (fy - xy) + img[y1, x1, :3] * xy)


def transfer(nr, base=None, target=None, *, output_size=None, grid_size=None,
             base_subrect=None, target_subrect=None, base_filter="linear", target_filter="linear", enabled=True):
    """Experimental independent cg2r OKLab transfer; does not activate NGX or the main network path.

    nr is the ordinary +32 fallback texture, already at output_size=(H,W).
    base is +224 (high-frequency base A), target is +256 (target B), and grid_size
    is the explicit (+308,+304) low-resolution transfer grid, defaulting to B's size.
    A/B subrects are pixel (x,y,width,height), optionally followed by the descriptor's
    inverse resource dimensions. Each texture has explicit point/linear filtering.
    Linear mode models CUDA texture weights with eight fractional bits; it does not
    mean linear-light RGB. No sRGB decoding occurs in the original kernel.

    False enabled (+300), either missing image, or either nonpositive grid dimension
    selects nr, as the original predicate does. Enabled transfer ignores nr values.
    Returns RGB before grade/UI/mask; compose with apply(color, result, ...).
    A == B need not preserve near-black RGB: the kernel floors reconstructed L at .1.
    Float32 exp/log/FMA differences remain; this is not a bit-exact RTX 50 claim.
    The recorded RTX 3060 tests model normalized coordinates with 21 fractional
    bits, followed by eight-bit axis/cross weights with half-way ties rounded up.
    The 21-bit coordinate conversion is an empirical device model, not a PTX/API
    guarantee. Other GPUs and untested descriptors remain outside its validation.
    """
    if not torch.is_tensor(nr):
        raise ValueError("nr must be a (H,W,3/4) tensor")
    nr = _transfer_image("nr", nr, nr.device)
    out_h, out_w = _transfer_size("output_size", output_size or nr.shape[:2])
    if nr.shape[:2] != (out_h, out_w):
        raise ValueError("nr must already match output_size")
    default_grid = target.shape[:2] if torch.is_tensor(target) else nr.shape[:2]
    low_h, low_w = _transfer_size("grid_size", grid_size if grid_size is not None else default_grid, positive=False)
    if not enabled or base is None or target is None or min(low_h, low_w) <= 0:
        return nr[..., :3]
    base, target = _transfer_image("base", base, nr.device), _transfer_image("target", target, nr.device)
    yy, xx = torch.meshgrid(torch.arange(low_h, device=nr.device, dtype=torch.float32),
                            torch.arange(low_w, device=nr.device, dtype=torch.float32), indexing="ij")
    ua, va = (xx + 0.5) / low_w, (yy + 0.5) / low_h
    la = _transfer_lab(_transfer_texture(base, ua, va, base_subrect, base_filter))
    lb = _transfer_lab(_transfer_texture(target, ua, va, target_subrect, target_filter))
    floor_l, ln2 = 0.10000000149011612, 0.6931471824645996
    log_delta = ((torch.log2(lb[..., :1].clamp_min(floor_l)) * ln2)
                 - (torch.log2(la[..., :1].clamp_min(floor_l)) * ln2)).clamp(-2, 2)
    delta = torch.cat([_sat(la[..., :1] / floor_l) * log_delta, lb[..., 1:] - la[..., 1:]], -1)
    yy, xx = torch.meshgrid(torch.arange(out_h, device=nr.device, dtype=torch.float32),
                            torch.arange(out_w, device=nr.device, dtype=torch.float32), indexing="ij")
    u, v = (xx + 0.5) / out_w, (yy + 0.5) / out_h
    qx, qy = u * low_w - 0.5, v * low_h - 0.5
    ix, iy = qx.floor().long(), qy.floor().long()
    fx, fy = (qx - ix)[..., None], (qy - iy)[..., None]
    x0, x1, y0, y1 = ix.clamp(0, low_w - 1), (ix + 1).clamp(0, low_w - 1), iy.clamp(0, low_h - 1), (iy + 1).clamp(0, low_h - 1)
    change = (delta[y0, x0] * (1 - fx) * (1 - fy) + delta[y0, x1] * fx * (1 - fy)
              + delta[y1, x0] * (1 - fx) * fy + delta[y1, x1] * fx * fy)
    high = _transfer_lab(_transfer_texture(base, u, v, base_subrect, base_filter))
    light = _sat(torch.exp2((torch.log2(high[..., :1].clamp_min(floor_l)) * ln2 + change[..., :1]) * 1.4426950216293335))
    return _transfer_rgb(torch.cat([light, high[..., 1:] + change[..., 1:]], -1))
