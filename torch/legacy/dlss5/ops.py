"""从 PTX 中精确恢复的算子。

每个常数都标注了来源引用（fatbin_NN.ptx:行号），可用 dllq 复查：
    dllq const <值>
    dllq grep "<指令>" --fatbin N
"""
import math
import torch
import torch.nn as nn

# ---------------------------------------------------------------- 激活

# fatbin_01.ptx:1975-2020, cc_tinlayout_fused_pre_block_swin_1h_32_1
SILU_CLAMP = 4.0            # %r645 / %r644
SILU_A = -0.055908203125    # %r648  int32 -1117454336
SILU_B = 0.447265625        # %r647  int32  1055195136
SILU_C = 0.894531250        # %r646  int32  1063583744


def mp_cubic_silu(x):
    """MpCubicSiluActivation。

        t = clamp(x, -4, +4)
        y = x · (A·t·|t| + B·t + C)

    等价形式 y = 2C · x · (½ + t/4 - t|t|/32)，即带 1.789 增益的 SiLU。
    C = 0.89453125 是 2/√5 在 fp16 下的最近可表示值。
    """
    t = x.clamp(-SILU_CLAMP, SILU_CLAMP)
    return x * (SILU_A * t * t.abs() + SILU_B * t + SILU_C)


class MpCubicSilu(nn.Module):
    def forward(self, x):
        return mp_cubic_silu(x)


# ---------------------------------------------------------------- 归一化

# fatbin_01.ptx:9216-9236  epsilon = 6.19999992e-05 (int32 948045311)
NORM_FLOOR = 6.19999992e-05


def l2_norm(x, dim=-1, weight=None):
    """PTX 里的归一化：**没有均值减法**（全 pre-block sub.f16x2 = 0）。

        y = x · rsqrt( max( Σx², 6.2e-5 ) )

    注意两点：
      · epsilon 是对平方和取下界（max），不是加法
      · 没有除以 N —— 严格说是 L2 归一化而非 RMS，1/√N 应被吸收进 gain
    """
    ss = (x.float() ** 2).sum(dim=dim, keepdim=True)
    y = x * torch.rsqrt(ss.clamp_min(NORM_FLOOR)).to(x.dtype)
    return y if weight is None else y * weight


class L2Norm(nn.Module):
    def __init__(self, width, learn_gain=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width)) if learn_gain else None

    def forward(self, x):
        return l2_norm(x, dim=-1, weight=self.weight)


# ---------------------------------------------------------------- 注意力指数

# fatbin_06.ptx:22637-22664, cc_vit_attention
EXP_SCALE = 0.0895394683    # int32 1035427960
EXP_BIAS = 1.70936143       # int32 1071303771
EXP_LO = 1.439453125        # int32 1069039616
EXP_HI = 1.9775390625       # int32 1073553408
EXP_MAGIC = 0x3FFC4000      # add.s32 立即数


def schraudolph_exp_exact(logits):
    """按位运算精确复现 PTX 的做法（含近似误差）。

        s = clamp(logits·SCALE + BIAS, LO, HI)
        w = f16_bits_reinterpret( (bits(s) << 4) + 0x3FFC4000 ) 的低 16 位

    偏置的作用是精确抵消 f16 指数域：0x3FFC4000 + (0x3C00 << 4) = 2^30，
    于是低 16 位恰好等于 16m（m 为尾数），按 f16 读回即得指数曲线。
    """
    s = (logits * EXP_SCALE + EXP_BIAS).clamp(EXP_LO, EXP_HI).to(torch.float16)
    bits = s.view(torch.int16).to(torch.int32) & 0xFFFF
    out = ((bits << 4) + EXP_MAGIC) & 0xFFFF
    return out.to(torch.int16).view(torch.float16).to(logits.dtype)


def schraudolph_exp(logits):
    """解析等价形式 w = 2^(16s - 31)。数值上更稳，误差 0~6.6%（同 PTX）。

    展开后 w ∝ exp(1.4326·L)；精确 exp 需 16·SCALE = 1/ln2 = 1.442695，
    实际 1.432631 偏小 0.70%，用来抵消 Schraudolph 近似的系统性高估。
    常数因子 2^-3.65 在 softmax 分母里约掉。
    """
    s = (logits * EXP_SCALE + EXP_BIAS).clamp(EXP_LO, EXP_HI)
    return torch.pow(2.0, 16.0 * s - 31.0)


#: clamp 反推出的有效 logit 区间（对称 ±3）
LOGIT_RANGE = ((EXP_LO - EXP_BIAS) / EXP_SCALE, (EXP_HI - EXP_BIAS) / EXP_SCALE)


def windowed_attention(q, k, v, attn_bias=None, attn_scale=None, exact_bits=False):
    """DLSS5 的窗口注意力。

    q/k/v : [B, heads, T, head_dim]，T = 64（8×8 窗口）
    attn_bias : [heads, T, T]  —— 每头一张稠密 64×64 表（不是相对位置共享）
    attn_scale : [heads] 或 [heads, 2]

    softmax 分子用 Schraudolph 位运算指数，分母走 rcp.approx。
    """
    logits = q @ k.transpose(-1, -2)
    if attn_scale is not None:
        logits = logits * attn_scale.view(1, -1, 1, 1)
    if attn_bias is not None:
        logits = logits + attn_bias.unsqueeze(0)
    w = (schraudolph_exp_exact if exact_bits else schraudolph_exp)(logits)
    w = w / w.sum(dim=-1, keepdim=True).clamp_min(NORM_FLOOR)
    return w @ v


# ---------------------------------------------------------------- 余弦门控跳连

def cos_skip(x, skip, coef):
    """ffn_cos_skip / attn_cos_skip。

    **不是运行时余弦**：融合 Swin kernel（fatbin_02–07）里 sin/cos 指令为 0，
    sqrt 也为 0，向量减法同样为 0。所以存的就是系数本身，取值实测全落在
    [-1, 1] 内（0% 越界，两处取到精确 1.0）。

    排除掉的可能：
      · 运行时 cos(θ)            —— 无 sin/cos 指令
      · sin = sqrt(1-cos²)      —— 无 sqrt 指令
      · (1-a) 型 lerp           —— 无 sub.f16x2
      · 2W = [cos, sin] 拼接     —— 实测 a²+b² ∈ [0.85, 1.87]，非 1

    剩下的只有单条 fma，两种操作数顺序无法进一步区分：
        out = coef·skip + branch      （本实现采用）
        out = skip + coef·branch
    """
    return skip * coef.to(x.dtype) + x


# ---------------------------------------------------------------- 内部高斯

# fatbin_01.ptx:29 起  0x9E3779B9 与 lowbias32 常数 277803737
GOLDEN = 0x9E3779B9
LOWBIAS32 = 277803737


def _hash32(x, mul, add):
    x = (x * mul + add) & 0xFFFFFFFF
    x ^= x >> (((x >> 28) + 4) & 31)
    x = (x * LOWBIAS32) & 0xFFFFFFFF
    return ((x >> 30) ^ (x >> 8)) & 0xFFFFFFFF


def box_muller_lanes(seed, tile_x, tile_y, device="cpu"):
    """pre-block 内部生成的三条高斯 lane。

    返回 [n0, n1, n2]。这三条与常数 1.0 一起构成进入首个 MMA 的
    shared memory 前 8 个 f16 lane 中的前四个：
        [n0, n1, n2, 1.0, c0, c1, c2, c3]
    后四条 c* 是纹理采样。
    """
    st = (int(seed) * GOLDEN) & 0xFFFFFFFF
    st ^= (int(tile_x) * 0x8DA6B343) & 0xFFFFFFFF
    st = (st + int(tile_y)) & 0xFFFFFFFF
    u = [_hash32(st, m, a) * 2.0 ** -24 for m, a in
         ((0xCAFEBABE, 0x9E3779B9), (-895109107 & 0xFFFFFFFF, 568162667),
          (0xDEADBEEF, 0x85EBCA6B), (-2094846927 & 0xFFFFFFFF, 878960812))]
    u = [min(max(x, 1e-7), 1.0) for x in u]
    r1 = math.sqrt(-2.0 * math.log(u[0]))
    r2 = math.sqrt(-2.0 * math.log(u[1]))
    t1, t2 = 2 * math.pi * u[2], 2 * math.pi * u[3]
    return torch.tensor([r1 * math.cos(t1), r1 * math.sin(t1), r2 * math.cos(t2)],
                        device=device)
