"""把网络主体 (pre_block 适配器之后 -> post_block 输出卷积) 导出为 ONNX，供 TensorRT 构建引擎。

网络入口的采样 (颜色/历史重投影、噪声) 与出口的时域混合仍在 torch 里做 (Core.inputs / Core.finish)，引擎只跑中间:
    X (GH*GW, 16) f16 (15 路输入 + 1 路 0)  ->  n (GH, GW, 4) f16     (GH, GW = 补齐网格)

导出模式下的数值替换 (set_export):
  f16()      恒等 (整网本来就是半精度)
  q8()       q8="none": 去掉；q8="arith": 用 log2/floor/round 的纯算术 e4m3 伪量化 (ONNX 标准算子)
  exp_bits   exp(clamp(L))，截断范围与位运算版相同 (16h: [-6, 6]，ViT: [-3, 3])；常数因子在归一化里抵消
  windows / FinalHead / pad_rows_cols   原地写入改为 F.pad (ONNX 友好)
    net = DLSS5(); core = Core(net, 1080, 1920, q8="arith"); core.export("dlss5_1080.onnx")
"""
import math

import torch
import torch.nn.functional as F

from . import blocks as B
from . import net as NET
from . import ops

Q8_MODE = "arith"


def _q8(x):
    if Q8_MODE == "none":
        return x
    x = x.clamp(-448.0, 448.0)
    e = torch.floor(torch.log2(x.abs().float().clamp_min(2.0 ** -9))).clamp(min=-6.0)
    s = torch.pow(2.0, e - 3.0)
    return (torch.round(x.float() / s) * s).to(x.dtype)


def _f16(x):
    return x


def _exp_bits(L, mul, add, lo, hi, shift, addc):
    return torch.exp(L.clamp((lo - add) / mul, (hi - add) / mul))


def _inv_sum(p):
    return (1.0 / p.float().sum(-1, keepdim=True).clamp_min(6.1e-5)).to(p.dtype)


def _windows(Y, H, W, shift):
    C = Y.shape[1]
    oy, ox = -shift[0], -shift[1]
    ny, nx = (H + oy + 7) // 8, (W + ox + 7) // 8
    pad = F.pad(Y.view(H, W, C), (0, 0, ox, nx * 8 - W - ox, oy, ny * 8 - H - oy))
    return pad.view(ny, 8, nx, 8, C).permute(0, 2, 1, 3, 4).reshape(-1, 64, C), (ny, nx, oy, ox)


def _final_head(self, x16, d16=(12, 20), dvit=(8, 12)):
    h, w = d16[0] // 2, d16[1] // 2
    pooled = F.pad(B.q8(x16.view(h, 2, w, 2, 512).mean((1, 3))), (0, 0, 0, dvit[1] - w, 0, dvit[0] - h))
    return B.q8(pooled.view(-1, 512)[:, self.ci] @ self.W)[:, self.cols]


def _pad_rows_cols(x, src, dst):
    C = x.shape[1]
    h, w = min(src[0], dst[0]), min(src[1], dst[1])
    y = x.view(src[0], src[1], C)[:h, :w]
    return F.pad(y, (0, 0, 0, dst[1] - w, 0, dst[0] - h)).reshape(-1, C)


def set_export(q8="arith"):
    """把 ops / blocks / net 里的数值原语换成导出版 (进程内不可逆)"""
    global Q8_MODE
    Q8_MODE = q8
    ops.PRECISE = False
    for mod in (ops, B, NET):
        for name, fn in (("q8", _q8), ("f16", _f16), ("exp_bits", _exp_bits), ("inv_sum", _inv_sum), ("windows", _windows)):
            if hasattr(mod, name):
                setattr(mod, name, fn)
    B.FinalHead.__call__ = _final_head
    NET.pad_rows_cols = _pad_rows_cols
    B.Swin.CHUNK = 1 << 30                                        # 不分块 (引擎自己管理显存)


def to_half(obj, seen=None):
    """把块对象里的 f32 张量 (权重) 原地换成 f16，索引张量不动"""
    seen = seen if seen is not None else set()
    if id(obj) in seen:
        return
    seen.add(id(obj))
    d = getattr(obj, "__dict__", None)
    if d is None:
        return
    for k, v in list(d.items()):
        if torch.is_tensor(v) and v.dtype == torch.float32:
            d[k] = v.half()
        elif isinstance(v, tuple) and all(torch.is_tensor(t) for t in v):
            d[k] = tuple(t.half() if t.dtype == torch.float32 else t for t in v)
        elif isinstance(v, dict):
            for vv in v.values():
                to_half(vv, seen)
        elif hasattr(v, "__dict__") and not isinstance(v, type):
            to_half(v, seen)


# ================================================================== 规范序版本 (canon=True)
# 除矩阵乘外全部运算都逐通道 (c1⊙x、量化、激活、空间平均、窗口划分)，通道顺序可以任取。kernel 的激活是"片段序"，
# 各块进出都要按 ci / cols / oc 重排 (Gather)。规范序版本让所有激活都保持规范序，把重排一次性折进权重的行/列与逐通道向量，
# 前向里不再有 Gather (post 入口一处 32 通道的除外)。代数上与原版完全相同。
def _canon(m, kind, ci16, ci_vit):
    """给块加上 k_ 前缀的规范序权重"""
    def sw(b):                                                    # Swin: 输出列 / 逐通道向量按 ci 取
        ci = b.ci
        if b.D is None:
            b.k_W2 = b.W2[:, ci]
        else:
            b.k_D = b.D[:, ci]
        b.k_c1, b.k_c2, b.k_Wp = b.c1[ci], b.c2[ci], b.Wp[:, ci]
    if kind == "pre":
        sw(m.swin)
    elif kind == "post":
        b = m.swin
        sw(b)
        m.k_s1, m.k_s2, m.k_Wo = m.s1[b.ci], m.s2[b.ci], m.Wo[b.ci]
        m.k_g = m.mxi[b.ci]                                       # 上一级规范序 -> 本级规范序的通道映射 (半分辨率上 Gather)
    elif kind == "swin":
        if isinstance(m, B.SwinDown):
            sw(m.block)
            m.k_Wds = m.Wds[:, m.oc]
        elif isinstance(m, B.SwinUp):
            sw(m.block)
            ci = m.block.ci
            m.k_Wu, m.k_s, m.k_c = m.Wu[:, ci], m.s[ci], m.c[ci]
        else:
            sw(m)
    elif kind == "split16":
        ci = m.ci
        (M1, c1), (M3, c3) = m.P1, m.P3
        m.k_M1, m.k_c1, m.k_M3, m.k_c3 = M1[:, ci], c1[ci], M3[:, ci], c3[ci]
    elif kind == "head":
        m.k_W = m.W[:, m.cols[ci_vit]]
    elif kind == "vit":
        ci = m.ci
        m.k_W2, m.k_c1, m.k_Wp, m.k_c2 = m.W2[:, ci], m.c1[ci], m.Wp[:, ci], m.c2[ci]
    elif kind == "block39":
        m.k_W, m.k_s = m.W[:, ci16], m.s[ci16]


def _ffn_k(b, X):
    Hh = B.q8(B.act(X @ b.W1))
    return Hh @ b.k_W2 if b.D is None else B.q8(Hh @ b.W2) @ b.k_D


def _attn_k(b, y, H, W, shift):
    Yw, meta = B.windows(B.q8(y), H, W, shift)
    O = B.unwindows(b._window_attn(Yw), meta, H, W)
    if b.q_O:
        O = B.q8(O)
    return B.q8(b.k_c2 * (B.q8(y) if b.q_res else y) + O @ b.k_Wp)


def _swin_k(b, X, H, W, shift):
    return _attn_k(b, b.k_c1 * X + _ffn_k(b, X), H, W, shift)


def _down_k(m, X, H, W, shift):
    y = _swin_k(m.block, X, H, W, shift)
    Yc = y.view(H // 2, 2, W // 2, 2, m.W).mean((1, 3)).reshape(-1, m.W)
    return y, B.q8(Yc @ m.k_Wds)


def _up_k(m, low, skip, H, W, shift):
    h, w = H // 2, W // 2
    U = (low @ m.k_Wu).view(h, 1, w, 1, m.W).expand(h, 2, w, 2, m.W).reshape(H * W, m.W)
    z = m.k_c * skip + U
    zq = B.q8(z)
    y = m.k_s * (zq if m.block.q_res else z) + _ffn_k(m.block, zq)
    return _attn_k(m.block, y, H, W, shift)


def _split16_k(m, X, shift, H, W):
    Hh = B.q8(X @ m.W1)
    mid = B.q8(B.act(Hh @ m.Wa))
    y = B.q8(m.k_c1 * X + B.q8(mid @ m.Wb) @ m.k_M1)
    Yw, meta = B.windows(B.q8(y), H, W, shift)
    sp = lambda t: t.view(t.shape[0], 64, 16, 32).transpose(1, 2)     # noqa: E731
    q, k, v = sp(Yw @ m.Wq), sp(Yw @ m.Wk), sp(Yw @ m.Wv)
    L = B.q8(B.cos_norm(q) * m.tau.view(1, 16, 1, 1)) @ B.q8(B.cos_norm(k)).transpose(-1, -2) + m.bias[None]
    p = B.exp_bits(L, *B.EXP16)
    o = (B.q8(p * B.inv_sum(p)) @ B.q8(v)).transpose(1, 2).reshape(-1, 64, 512)
    return B.q8(m.k_c3 * y + B.q8(B.unwindows(o, meta, H, W)) @ m.k_M3)


def _head_k(m, x16, d16, dvit):
    h, w = d16[0] // 2, d16[1] // 2
    pooled = F.pad(B.q8(x16.view(h, 2, w, 2, 512).mean((1, 3))), (0, 0, 0, dvit[1] - w, 0, dvit[0] - h))
    return B.q8(pooled.view(-1, 512) @ m.k_W)


def _vit_k(m, X):
    Hh = B.q8(B.act(X @ m.W1))
    x1 = B.q8(m.k_c1 * X + Hh @ m.k_W2)
    n = X.shape[0]
    sp = lambda t: t.view(n, 32, 32).transpose(0, 1)               # noqa: E731
    q, k, v = sp(x1 @ m.Wq), sp(x1 @ m.Wk), sp(x1 @ m.Wv)
    Q = B.q8(B.cos_norm(q) * m.tau.view(32, 1, 1) * math.sqrt(32))
    p = B.exp_bits(Q @ B.q8(B.cos_norm(k)).transpose(-1, -2), *B.EXPVIT)
    o = (B.q8(p) @ B.q8(v)) * B.inv_sum(p)
    return B.q8(m.k_c2 * x1 + B.q8(o.transpose(0, 1).reshape(n, 1024)) @ m.k_Wp)


def _b39_k(m, low, skip16, dvit, d16):
    h, w = d16[0] // 2, d16[1] // 2
    L = low.view(dvit[0], dvit[1], 1024)[:h, :w].reshape(-1, 1024)
    U = (L @ m.k_W).view(h, 1, w, 1, 512).expand(h, 2, w, 2, 512).reshape(-1, 512)
    return B.q8(m.k_s * skip16 + U)


def _post_k(m, x69, skip, GH, GW):
    b = m.swin
    h, w = GH // 2, GW // 2
    X = x69[:, m.k_g].view(h, 1, w, 1, 32).expand(h, 2, w, 2, 32).reshape(-1, 32)
    M = m.k_s1 * X + m.k_s2 * skip
    Y = b.k_c1 * M + B.q8(B.act(B.q8(M) @ b.W1)) @ b.k_W2
    Yw, meta = B.windows(B.q8(Y), GH, GW, (-4, -4))
    y = b.k_c2 * Y + B.q8(B.unwindows(b._window_attn(Yw), meta, GH, GW)) @ b.k_Wp
    return (y @ m.k_Wo).view(GH, GW, 4)


class Core(torch.nn.Module):
    """网络主体。forward(X: (GH*GW, 16) f16) -> (GH, GW, 4) f16
    canon=True: 规范序版本 (重排折进权重，见 _canon)"""

    def __init__(self, net, H, W, q8="arith", canon=False):
        super().__init__()
        set_export(q8)
        self.net, self.H, self.W, self.canon = net, H, W, canon
        self.D = NET.dims(H, W)
        pre = net.steps[0][1]
        A16 = torch.cat([pre.A, torch.zeros(1, 32, device=pre.A.device)])   # 15 路输入补成 16 (K 为 8 的倍数才走 Tensor Core)
        self.A16 = A16.half()
        for st, m in net.steps:
            to_half(m)
            if "head" in st:
                to_half(st["head"])
        if canon:
            ci16, ci_vit = None, None
            for st, m in net.steps:
                if st["kind"] == "split16":
                    ci16 = m.ci
                elif st["kind"] == "vit":
                    ci_vit = m.ci
            for st, m in net.steps:
                _canon(m, st["kind"], ci16, ci_vit)
                if "head" in st:
                    _canon(st["head"], "head", ci16, ci_vit)

    def forward(self, X):
        return self._forward_k(X) if self.canon else self._forward(X)

    def _forward_k(self, X):
        D = self.D
        GH, GW = D["grid"]
        skips, x, cur, pre_skip = {}, None, None, None
        for st, m in self.net.steps:
            k = st["kind"]
            if k == "pre":
                pre_skip = _swin_k(m.swin, X, GH, GW, (0, 0))            # X = 适配器输出 (Core.inputs 里算)
                x = B.q8(pre_skip.view(GH // 2, 2, GW // 2, 2, 32).mean((1, 3)).reshape(-1, 32))
            elif k == "swin":
                lv = st["level"]
                H, Wd = D[lv]
                sh, var = tuple(st["shift"]), st["variant"]
                if var == "inpview":
                    cur = _swin_k(m, x, H, Wd, sh)
                elif var == "ds":
                    cur, x = _down_k(m, cur, H, Wd, sh)
                    skips[lv] = cur
                elif var == "up":
                    if lv == "8h":
                        x = NET.pad_rows_cols(x, D["16h"], D["16h_real"])
                    cur = _up_k(m, x, skips[lv], H, Wd, sh)
                elif var == "outview":
                    x = _swin_k(m, cur, H, Wd, sh)
                else:
                    cur = _swin_k(m, cur, H, Wd, sh)
            elif k == "split16":
                if st["inpview"]:
                    x = NET.pad_rows_cols(x, D["16h_real"], D["16h"])
                cur = _split16_k(m, x if st["inpview"] else cur, tuple(st["shift"]), *D["16h"])
                if st["tail"] == "pool":
                    skips["16h"] = cur
                    cur = _head_k(st["head"], cur, D["16h"], D["vit"])
                elif st["tail"] == "outview":
                    x = cur
            elif k == "vit":
                cur = _vit_k(m, cur)
            elif k == "block39":
                cur = _b39_k(m, cur, skips["16h"], D["vit"], D["16h"])
            elif k == "post":
                return _post_k(m, x, pre_skip, GH, GW)

    def _forward(self, X):
        D = self.D
        GH, GW = D["grid"]
        skips, x, cur, pre_skip = {}, None, None, None
        for st, m in self.net.steps:
            k = st["kind"]
            if k == "pre":
                a = B.q8(X @ self.A16)[:, m.fi]
                pre_skip = m.swin(a, GH, GW, (0, 0))
                x = B.q8(pre_skip[:, m.swin.ci].view(GH // 2, 2, GW // 2, 2, 32).mean((1, 3)).reshape(-1, 32))
            elif k == "swin":
                lv = st["level"]
                H, Wd = D[lv]
                sh, var = tuple(st["shift"]), st["variant"]
                if var == "inpview":
                    cur = m(x[:, m.fi], H, Wd, sh)
                elif var == "ds":
                    cur, x = m(cur, H, Wd, sh)
                    skips[lv] = cur
                elif var == "up":
                    if lv == "8h":
                        x = NET.pad_rows_cols(x, D["16h"], D["16h_real"])
                    cur = m(x, skips[lv], H, Wd, sh)
                elif var == "outview":
                    x = m(cur, H, Wd, sh)[:, m.ci]
                else:
                    cur = m(cur, H, Wd, sh)
            elif k == "split16":
                if st["inpview"]:
                    x = NET.pad_rows_cols(x, D["16h_real"], D["16h"])
                cur = m(x[:, m.fi] if st["inpview"] else cur, tuple(st["shift"]), *D["16h"])
                if st["tail"] == "pool":
                    skips["16h"] = cur
                    cur = st["head"](cur, D["16h"], D["vit"])
                elif st["tail"] == "outview":
                    x = cur[:, m.ci]
            elif k == "vit":
                cur = m(cur)
            elif k == "block39":
                cur = m(cur, skips["16h"], D["vit"], D["16h"])
            elif k == "post":
                return m.net(x, pre_skip, GH, GW)

    # ------------------------------------------------------------ 引擎外的部分 (torch)
    def inputs(self, color, hist, mv, frame, ctrl=None):
        X = F.pad(self.net.steps[0][1].inputs(color, color if hist is None else hist, mv, frame, ctrl).half(), (0, 1))
        if self.canon:                                            # 适配器 16->32 在引擎外算 (实测对引擎耗时几乎没有影响: 540p 45.1 -> 44.3 ms)
            return B.q8(X @ self.A16)
        return X

    def finish(self, n, color, hist, mv):
        """n: 引擎输出 (GH, GW, 4) -> 与 PostBlock.__call__ 相同的时域混合"""
        post = self.net.steps[-1][1]
        H, W = color.shape[:2]
        n = n[:H, :W].float()
        cur = (color + 8 * post.scale * n[..., :3]).clamp(0, 1)
        if hist is None:
            return cur
        gate = (torch.sigmoid(n[..., 3:]) * post.blend).clamp(0, 1)
        y, x = torch.meshgrid(torch.arange(H, device=color.device, dtype=torch.float32),
                              torch.arange(W, device=color.device, dtype=torch.float32), indexing="ij")
        h = ops.catmull_rom5(hist[..., :3], (x + 0.5 + mv[..., 0]) / W, (y + 0.5 + mv[..., 1]) / H)
        return cur + gate * (h - cur)

    def export(self, path, opset=17):
        GH, GW = self.D["grid"]
        X = torch.zeros(GH * GW, 32 if self.canon else 16, device="cuda", dtype=torch.float16)
        with torch.no_grad():
            torch.onnx.export(self, (X,), path, input_names=["X"], output_names=["n"], opset_version=opset,
                              do_constant_folding=True)
        return path
