"""DLSS5 的各类块 (GPU 前向)。构造时把权重记录字节解码成已排好序的稠密张量，前向只做张量运算。

激活约定 (与 kernel 缓冲的内存序一致):
  tin 激活  (N, C) 片段序；图像激活 (N, C) 规范序；N = H*W 行优先
每个块的公式与验证数据见 tools/notes.jsonl；numpy 对照实现在 research/net_ref.py。
"""
import numpy as np
import torch

from . import layout as Lay
from .ops import act, cos_norm, exp_bits, EXP16, EXPVIT, f16, inv_sum, q8, unwindows, windows


class _Base:
    def __init__(self, device):
        self.dev = device

    def T(self, a, dtype=torch.float32):
        return torch.as_tensor(np.asarray(a), dtype=dtype, device=self.dev)

    def I(self, a):
        return torch.as_tensor(np.asarray(a), dtype=torch.long, device=self.dev)


# ================================================================== 1h-8h swin (U-Net 两侧)
class Swin(_Base):
    """标准 swin 块: y = c1⊙x + FFN(x)；out = q8(c2⊙y + proj(attn(y)))。W=32 (1h) 与 W>=64 的记录格式不同。
    ffn_in / ffn_scale 供上采样入口块复用 (见 SwinUp)。"""

    def __init__(self, w, W, device):
        super().__init__(device)
        self.W, self.heads = W, W // 32
        self.bit_exp, self.q_res, self.q_O = self.NUMERICS[W]
        ci = Lay.canon_index(W)
        self.ci, self.fi = self.I(ci), self.I(np.argsort(ci))
        if W == 32:
            self._init_1h(w)
        else:
            self._init_w(w)

    def _init_1h(self, w):
        W2 = Lay.unswizzle(w[4096:8192], 128, 32)[Lay.C_MAP]
        W2f = np.zeros((128, 32), np.float32)
        W2f[:, Lay.A_MAP] = W2                                   # 输出直接落在片段序
        self.W1, self.W2 = self.T(Lay.unswizzle(w[0:4096], 32, 128)), self.T(W2f)
        self.D = None
        self.c1 = self.T(Lay.f16vec(w[8208:8272])[Lay.COLS32])
        qkv = Lay.unswizzle(w[8288:11360], 32, 96)
        vperm = np.argsort(Lay.C_MAP[:32])
        self.Wq, self.Wk, self.Wv = self.T(qkv[:, :32]), self.T(qkv[:, 32:64]), self.T(qkv[:, 64:][:, vperm])
        self.tau = self.T(np.frombuffer(w[19552:19556], np.float32))
        self.bias = self.T(Lay.bias1h(w[11360:19552]))[None]     # (1 头, 64, 64)
        Wp = np.zeros((32, 32), np.float32)
        Wp[:, Lay.A_MAP] = Lay.unswizzle(w[19568:20592], 32, 32)
        self.Wp = self.T(Wp)
        self.c2 = self.T(Lay.f16vec(w[20592:20656])[Lay.COLS32])

    def _init_w(self, w):
        W, nw = self.W, self.W // 32
        o = Lay.swin_layout(W)
        cols = Lay.group_map(Lay.COLS32, W)
        W1 = np.zeros((W, 4 * W), np.float32)
        W2 = np.zeros((4 * W, W), np.float32)
        for wp in range(nw):                                      # warp wp 管 128 个隐层 (4 轮 x 32)，块对角收缩到 32 通道
            for j in range(4):
                h0 = (wp * 4 + j) * 32
                W1[:, h0:h0 + 32] = np.vstack([Lay.unswizzle(w[wp * nw * 4096 + kc * 4096 + j * 1024:][:1024], 32, 32)
                                               for kc in range(nw)])
                c0 = o["W2"] + wp * 4096 + j * 1024
                W2[h0:h0 + 32, wp * 32:wp * 32 + 32] = Lay.unswizzle(w[c0:c0 + 1024], 32, 32)[Lay.CMAP32]
        self.W1, self.W2 = self.T(W1), self.T(W2)
        self.D = self.T(Lay.unswizzle(w[o["D"]:o["D"] + W * W], W, W)[Lay.group_map(Lay.CMAP32, W)][:, cols])
        self.c1 = self.T(Lay.f16vec(w[o["c1"]:o["c1"] + 2 * W])[cols])
        qkv = Lay.unswizzle(w[o["qkv"]:o["qkv"] + 3 * W * W], W, 3 * W).reshape(W, self.heads, 3, 32)
        self.Wq, self.Wk, self.Wv = (self.T(qkv[:, :, s].reshape(W, -1)) for s in range(3))
        self.tau = self.T(np.frombuffer(w[o["tau"]:o["tau"] + 4 * self.heads], np.float32))
        self.bias = self.T(np.stack([Lay.bias1h(w[o["bias"] + h * 8192:o["bias"] + (h + 1) * 8192])
                                     for h in range(self.heads)]))
        self.Wp = self.T(Lay.unswizzle(w[o["proj"]:o["proj"] + W * W], W, W)[Lay.group_map(Lay.CMAP32, W)][:, cols])
        self.c2 = self.T(Lay.f16vec(w[o["c2"]:o["c2"] + 2 * W])[cols])

    # ---------------------------------------------------------------
    def ffn(self, Xf):
        """FFN 增量 (片段序)，不含残差"""
        Hh = q8(act(f16(Xf[:, self.ci] @ self.W1)))
        if self.D is None:
            return Hh @ self.W2
        return q8(Hh @ self.W2) @ self.D

    # 注意力数值细节 (torch/check_numerics.py 逐块扫描实测，取逐值一致率最高者):
    #   全部级别的 softmax 分子都是位运算 exp；
    #   1h: 残差用 f16 的 y、投影前 O 量化；2h: 残差读 fp8 的 y、O 量化；4h/8h: 残差读 fp8 的 y、O 不量化 (与量化几乎相同)
    NUMERICS = {32: (True, False, True), 64: (True, True, True), 128: (True, True, False), 256: (True, True, False)}

    def attn(self, Yf, H, W, shift):
        """余弦窗口注意力 + 投影 + c2 残差"""
        Y = q8(Yf[:, self.ci])
        Yw, meta = windows(Y, H, W, shift)
        nh = self.heads
        sp = lambda t: t.view(t.shape[0], 64, nh, 32).transpose(1, 2)     # noqa: E731  (窗口, 头, 64, 32)
        q, k, v = sp(f16(Yw @ self.Wq)), sp(f16(Yw @ self.Wk)), sp(f16(Yw @ self.Wv))
        qn = cos_norm(q) * self.tau.view(1, nh, 1, 1)
        L = f16(q8(qn) @ q8(cos_norm(k)).transpose(-1, -2) + self.bias[None])
        if self.bit_exp:
            p = exp_bits(L, *EXP16)                               # 位运算 exp (logit 截断 [-6,6])，先归一化再乘 V
            P = q8(f16(p * inv_sum(p)))
        else:
            P = q8(torch.softmax(L, -1))
        o = f16(P @ q8(v)).transpose(1, 2).reshape(-1, 64, nh * 32)
        O = unwindows(o, meta, H, W)
        if self.q_O:
            O = q8(O)                                             # 投影是 fp8 mma: O 先量化
        res = q8(Yf) if self.q_res else Yf                        # 残差读存进共享内存的 fp8 y
        return q8(f16(self.c2 * res + O @ self.Wp))

    def __call__(self, Xf, H, W, shift):
        return self.attn(f16(self.c1 * Xf + self.ffn(Xf)), H, W, shift)


class SwinDown(_Base):
    """编码出口: 标准块 (输出 = 跳连) + ds = q8(avg2x2(y规范序)·W_ds)[:, invCMAP] (下一级的图像格式输入)"""

    def __init__(self, w, W, device):
        super().__init__(device)
        self.W = W
        if W == 32:
            std, off = w[:20672], 20656                           # 1h: W_ds 紧跟 c2
        else:
            end = Lay.swin_layout(W)["end"]
            std, off = w[:end - 16] + bytes(16), end - 16
        self.block = Swin(std, W, device)
        self.Wds = self.T(Lay.unswizzle(w[off:off + 2 * W * W], W, 2 * W))
        self.oc = self.I(np.argsort(Lay.group_map(Lay.CMAP32, 2 * W)))

    def __call__(self, Xf, H, W, shift):
        y = self.block(Xf, H, W, shift)
        Yc = y[:, self.block.ci].view(H // 2, 2, W // 2, 2, self.W).mean((1, 3)).reshape(-1, self.W)
        return y, q8(f16(f16(Yc) @ self.Wds))[:, self.oc]


class SwinUp(_Base):
    """解码入口: z = c⊙skip + nn_up2x(low规范序·W_up)，y = s⊙z + FFN(z)，再注意力 (1h 与 8h 实测)"""

    def __init__(self, w, W, device):
        super().__init__(device)
        self.W = W
        if W == 32:                                               # W_up | 16B | s | 16B | c | qkv… (+2112)
            Wu = Lay.unswizzle(w[8192:10240], 64, 32)
            Wuf = np.zeros((64, 32), np.float32)
            Wuf[:, Lay.A_MAP] = Wu
            s, c = Lay.f16vec(w[10256:10320])[Lay.COLS32], Lay.f16vec(w[10336:10400])[Lay.COLS32]
            std = w[:8192] + bytes(16) + w[10336:10400] + bytes(16) + w[10400:22784]
        else:                                                     # 紧凑拼接 FFN | W_up | s | c | qkv | …
            o = Lay.swin_layout(W)
            fe, cols = o["c1"] - 16, Lay.group_map(Lay.COLS32, W)
            p = fe + 2 * W * W
            Wuf = Lay.unswizzle(w[fe:p], 2 * W, W)[:, cols]
            s, c = Lay.f16vec(w[p:p + 2 * W])[cols], Lay.f16vec(w[p + 2 * W:p + 4 * W])[cols]
            q0 = p + 4 * W
            std = w[:fe] + bytes(16) + w[p + 2 * W:p + 4 * W] + bytes(16) + w[q0:q0 + o["end"] - 16 - o["qkv"]] + bytes(16)
        self.block = Swin(std, W, device)
        self.Wu, self.s, self.c = self.T(Wuf), self.T(s), self.T(c)

    def __call__(self, low_img, skip_f, H, W, shift):
        h, w = H // 2, W // 2
        U = f16(low_img @ self.Wu).view(h, 1, w, 1, self.W).expand(h, 2, w, 2, self.W).reshape(H * W, self.W)
        z = f16(self.c * skip_f + U)
        zq = q8(z)                                                # FFN 第一层是 fp8 mma: z 先量化
        y = f16(self.s * (zq if self.block.q_res else z) + self.block.ffn(zq))   # 残差量化规则同该级普通块
        return self.block.attn(y, H, W, shift)


# ================================================================== 16h 分组 swin (512 通道、16 头、20x12 token)
class Split16(_Base):
    """4 个 kernel: ffwd (分组低秩 FFN 512->64->256 x8 组) / ffwd_proj (D + c1) / qkv+注意力 (位运算 exp，截断 softmax) / proj (+c2)"""

    def __init__(self, ws, device):
        super().__init__(device)
        w0, w1, w2, w3 = ws
        ci = Lay.canon_index(512)
        self.ci, self.fi = self.I(ci), self.I(np.argsort(ci))
        cols = Lay.group_map(Lay.COLS32, 512)
        self.cols, self.icols = self.I(cols), self.I(np.argsort(cols))
        W1 = np.zeros((512, 512), np.float32)
        for kc in range(16):
            for hb in range(16):
                off = (kc * 16 + hb) * 1024
                W1[kc * 32:(kc + 1) * 32, hb * 32:(hb + 1) * 32] = Lay.unswizzle(w0[off:off + 1024], 32, 32)
        Wa = np.zeros((512, 2048), np.float32)
        Wb = np.zeros((2048, 512), np.float32)
        for q in range(8):
            for j in range(8):
                c0 = q * 256 + 32 * j
                Wa[64 * q:64 * q + 32, c0:c0 + 32] = Lay.unswizzle(w0[262144 + q * 16384 + j * 1024:][:1024], 32, 32)[Lay.CMAP32]
                Wa[64 * q + 32:64 * q + 64, c0:c0 + 32] = Lay.unswizzle(w0[270336 + q * 16384 + j * 1024:][:1024], 32, 32)[Lay.CMAP32]
                Wb[c0:c0 + 32, 64 * q:64 * q + 64] = Lay.unswizzle(w0[393216 + q * 16384 + j * 2048:][:2048], 32, 64)[Lay.CMAP32]
        self.W1, self.Wa, self.Wb = self.T(W1), self.T(Wa), self.T(Wb)
        self.P1 = self._proj(w1)
        self.P3 = self._proj(w3)
        qkv = Lay.unswizzle(w2[:786432], 512, 1536).reshape(512, 16, 3, 32)
        self.Wq, self.Wk, self.Wv = (self.T(qkv[:, :, s].reshape(512, -1)) for s in range(3))
        self.tau = self.T(np.frombuffer(w2[917504:917568], np.float32))
        self.bias = self.T(np.stack([Lay.bias16(w2[786432 + h * 8192:786432 + (h + 1) * 8192]) for h in range(16)]))

    def _proj(self, w):
        cols = Lay.group_map(Lay.COLS32, 512)
        M = Lay.unswizzle(w[:262144], 512, 512)[Lay.group_map(Lay.CMAP32, 512)][:, cols]
        return self.T(M), self.T(Lay.f16vec(w[262144:263168])[cols])

    def proj(self, Z, X, P):
        M, c = P
        return q8(f16(c * X + Z[:, self.icols] @ M))

    def ffwd(self, Xf):
        Hh = q8(f16(Xf[:, self.ci] @ self.W1))
        mid = q8(act(f16(Hh @ self.Wa)))
        return q8(mid @ self.Wb)[:, self.cols]

    def attn(self, y, shift, H=12, W=20):
        Y = q8(y[:, self.ci])
        Yw, meta = windows(Y, H, W, shift)
        sp = lambda t: t.view(t.shape[0], 64, 16, 32).transpose(1, 2)     # noqa: E731
        q, k, v = sp(f16(Yw @ self.Wq)), sp(f16(Yw @ self.Wk)), sp(f16(Yw @ self.Wv))
        qn = cos_norm(q) * self.tau.view(1, 16, 1, 1)
        L = f16(q8(qn) @ q8(cos_norm(k)).transpose(-1, -2) + self.bias[None])
        p = exp_bits(L, *EXP16)
        P = f16(p * inv_sum(p))                                   # 16h: 先归一化再乘 V
        o = f16(q8(P) @ q8(v)).transpose(1, 2).reshape(-1, 64, 512)
        return q8(unwindows(o, meta, H, W))[:, self.cols]

    def __call__(self, Xf, shift):
        y = self.proj(self.ffwd(Xf), Xf, self.P1)
        return self.proj(self.attn(y, shift), y, self.P3)


class FinalHead(_Base):
    """block30 末: 16h 输出 2x2 平均池化到 6x10 (补齐为 8x12 = 96 token) -> 512->1024 升维 (ViT 输入)"""

    def __init__(self, w, device):
        super().__init__(device)
        self.ci = self.I(Lay.canon_index(512))
        self.cols = self.I(Lay.group_map(Lay.COLS32, 1024))
        self.W = self.T(Lay.unswizzle(w[:524288], 512, 1024))

    def __call__(self, x16):
        pooled = x16.new_zeros(8, 12, 512)
        pooled[:6, :10] = q8(f16(x16.view(6, 2, 10, 2, 512).mean((1, 3))))
        return q8(f16(pooled.view(96, 512)[:, self.ci] @ self.W))[:, self.cols]


# ================================================================== ViT-1d (1024 通道、32 头、96 token 全局注意力)
class ViT(_Base):
    def __init__(self, ws, device):
        super().__init__(device)
        w1, w2, wq, _, wp = ws                                    # attention 记录 (1 个 f16) kernel 不读
        C4, C1 = Lay.group_map(Lay.COLS32, 4096), Lay.group_map(Lay.COLS32, 1024)
        self.ci = self.I(Lay.canon_index(1024))
        self.W1 = self.T(Lay.unswizzle(w1[:4194304], 1024, 4096))                        # 隐层按原序 (输出时的 COLS 重排与 W2 的读取抵消)
        self.W2 = self.T(Lay.unswizzle(w2[:4194304], 4096, 1024)[Lay.group_map(Lay.CMAP32, 4096)][:, C1])
        self.c1 = self.T(Lay.f16vec(w2[4194304:])[C1])
        self.tau = self.T(np.frombuffer(wq[:128], np.float32))
        A = Lay.unswizzle(wq[128:], 1024, 3072).reshape(1024, 32, 3, 32)
        self.Wq, self.Wk, self.Wv = (self.T(A[:, :, s].reshape(1024, -1)) for s in range(3))
        self.Wp = self.T(Lay.unswizzle(wp[:1048576], 1024, 1024)[Lay.group_map(Lay.CMAP32, 1024)][:, C1])
        self.c2 = self.T(Lay.f16vec(wp[1048576:])[C1])

    def __call__(self, X):
        Hh = q8(act(f16(X[:, self.ci] @ self.W1)))
        x1 = q8(f16(self.c1 * X + Hh @ self.W2))
        xc = x1[:, self.ci]
        sp = lambda t: t.view(96, 32, 32).transpose(0, 1)          # noqa: E731  (头, token, 32)
        q, k, v = sp(f16(xc @ self.Wq)), sp(f16(xc @ self.Wk)), sp(f16(xc @ self.Wv))
        Q = q8(f16(cos_norm(q) * self.tau.view(32, 1, 1) * np.sqrt(32)))
        L = f16(Q @ q8(cos_norm(k)).transpose(-1, -2))
        p = exp_bits(L, *EXPVIT)
        o = f16(f16(q8(p) @ q8(v)) * inv_sum(p))                  # ViT: 先乘 V 再乘 1/Σp
        O = q8(o.transpose(0, 1).reshape(96, 1024))
        return q8(f16(self.c2 * x1 + O @ self.Wp))


class Block39(_Base):
    """解码入口 (1024 -> 512，6x10 -> 12x20): out = q8(s⊙skip + nn_up2x(low规范序·W_up)[:, COLS])"""

    def __init__(self, w, device):
        super().__init__(device)
        cols = Lay.group_map(Lay.COLS32, 512)
        self.ci = self.I(Lay.canon_index(1024))
        self.W = self.T(Lay.unswizzle(w[:524288], 1024, 512)[:, cols])
        self.s = self.T(Lay.f16vec(w[524288:])[cols])

    def __call__(self, low96, skip16):
        L = low96.view(8, 12, 1024)[:6, :10].reshape(60, 1024)[:, self.ci]
        U = f16(L @ self.W).view(6, 1, 10, 1, 512).expand(6, 2, 10, 2, 512).reshape(240, 512)
        return q8(f16(self.s * skip16 + U))
