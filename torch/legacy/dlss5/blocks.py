"""DLSS5 的 block 模块。

置信度分三档，代码里逐处标注：
  [实测]  从 PTX 或权重字节直接读出，可复查
  [推断]  形状/计数闭合但语义靠推理
  [占位]  形状对但连接方式未定，forward 只是合理猜测
"""
import numpy as np
import torch
import torch.nn as nn

from .ops import mp_cubic_silu, l2_norm, windowed_attention, cos_skip
from .weights import HEAD_DIM, WINDOW


def _t(a, device=None):
    return torch.from_numpy(np.ascontiguousarray(a).astype(np.float32)).to(device)


class SplitSwin16HBlock(nn.Module):
    """16H 中央核。占全部参数的 68%，重复 8 次（block31–38）。

    五条记录 5/5 精确闭合：
        layer0  ffn_expand    512→2048  ×2      [实测]
        layer1  ffn_contract  2048→512  ×2      [实测]
        layer2  qkv           512→1536  ×2      [实测]
        layer3  attn_cos_skip 单标量             [实测]
        layer4  final_head    512→1024           [实测]

    ×2 是 SplitSwin 的双路；final_head 在两路汇合之后，不翻倍。[推断]
    六种 host 层类型（Ffwd / FfwdProj / QKVAttn / Proj / ProjPool / FinalHead）
    与五条权重记录的对应关系未完全确定 —— 前向顺序属 [占位]。
    """

    def __init__(self, W=512, device=None):
        super().__init__()
        self.W, self.H = W, W // HEAD_DIM
        self.device_ = device

    def load_(self, rec, device=None):
        d = device or self.device_
        g = lambda k, s: _t(rec["block%d.%s.layer" % (self.blk, k)][s], d)
        return self

    def load_from(self, recs, blk, device=None):
        self.blk = blk
        d = device or self.device_
        r = lambda k: recs["block%d.%s.layer" % (blk, k)]
        self.ffn_expand = nn.Parameter(_t(r("layer0")["ffn_expand"], d), False)
        self.ffn_contract = nn.Parameter(_t(r("layer1")["ffn_contract"], d), False)
        self.ffn_cos_skip = nn.Parameter(_t(r("layer1")["ffn_cos_skip"], d), False)
        self.qkv = nn.Parameter(_t(r("layer2")["qkv_weight"], d), False)
        self.attn_scale = nn.Parameter(_t(r("layer2")["attn_scale"], d), False)
        self.attn_cos_skip = nn.Parameter(_t(r("layer3")["attn_cos_skip"], d), False)
        self.final_head = nn.Parameter(_t(r("layer4")["final_head"], d), False)
        self.final_bias = nn.Parameter(_t(r("layer4")["bias"], d), False)
        return self

    def forward(self, x):
        """x: [B, T, W]，T = 64（8×8 窗口）。[占位] 前向顺序为合理猜测。"""
        B, T, W = x.shape
        outs = []
        for p in range(2):                                   # SplitSwin 双路
            h = l2_norm(x)                                   # [实测] L2，无均值
            qkv = h @ self.qkv[p]                            # [实测] 512→1536
            q, k, v = qkv.reshape(B, T, 3, self.H, HEAD_DIM).permute(2, 0, 3, 1, 4)
            a = windowed_attention(q, k, v,
                                   attn_scale=self.attn_scale[p, :, 0])   # [实测] 指数式
            a = a.permute(0, 2, 1, 3).reshape(B, T, W)
            h = x + a                                        # [推断] 残差
            f = mp_cubic_silu(l2_norm(h) @ self.ffn_expand[p])  # [实测] 激活
            f = f @ self.ffn_contract[p]
            outs.append(cos_skip(h + f, h, self.ffn_cos_skip[p * W:(p + 1) * W]))
        y = outs[0] + outs[1]                                # [推断] 双路汇合
        return y @ self.final_head + self.final_bias         # [实测] 512→1024


class CuckooBlock(nn.Module):
    """CrazyCuckoo 4 记录组（block23–29 / 40–47）。十个 slot 全闭合。

        layer0  weight0 2W² ×2 · weight1 W² · weight2 W²   [实测]
        layer1  weight3 2W² ×2 · ffn_cos_skip 2W           [实测]
        layer2  qkv 3W²×2 · attn_bias H×2×64×64 · attn_scale H×2×2  [实测]
        layer3  projection_weight 2W² · attn_cos_skip 2W   [实测]

    weight0..3 的具体角色（FFN 的哪一半 / 深度卷积 / 通道混合）属 [推断]。
    """

    def __init__(self, W=256, device=None):
        super().__init__()
        self.W, self.H = W, W // HEAD_DIM

    def load_from(self, recs, blk, device=None):
        r = lambda k: recs["block%d.%s.layer" % (blk, k)]
        P = lambda a: nn.Parameter(_t(a, device), False)
        self.weight0, self.weight1, self.weight2 = (
            P(r("layer0")["weight0"]), P(r("layer0")["weight1"]), P(r("layer0")["weight2"]))
        self.weight3 = P(r("layer1")["weight3"])
        self.ffn_cos_skip = P(r("layer1")["ffn_cos_skip"])
        self.qkv = P(r("layer2")["qkv_weight"])
        self.attn_bias = P(r("layer2")["attn_bias"])
        self.attn_scale = P(r("layer2")["attn_scale"])
        self.projection = P(r("layer3")["projection_weight"])
        self.attn_cos_skip = P(r("layer3")["attn_cos_skip"])
        return self

    def forward(self, x):
        """x: [B, T, W]。[占位] weight0..3 的接法是猜测。"""
        B, T, W = x.shape
        outs = []
        for p in range(2):
            h = l2_norm(x)
            qkv = h @ self.qkv[p]
            q, k, v = qkv.reshape(B, T, 3, self.H, HEAD_DIM).permute(2, 0, 3, 1, 4)
            a = windowed_attention(q, k, v,
                                   attn_bias=self.attn_bias[p],          # [实测] 稠密 64×64
                                   attn_scale=self.attn_scale[p, :, 0])
            a = a.permute(0, 2, 1, 3).reshape(B, T, W)
            a = a @ self.projection[p]
            h = cos_skip(x + a, x, self.attn_cos_skip[p * W:(p + 1) * W])
            f = mp_cubic_silu(l2_norm(h) @ self.weight0[p]) @ self.weight3[p]
            outs.append(cos_skip(h + f, h, self.ffn_cos_skip[p * W:(p + 1) * W]))
        y = outs[0] + outs[1]
        return (y @ self.weight1) @ self.weight2                          # [占位]


class SingleLayerBlock(nn.Module):
    """CCSingleLayerBlock（block0–22 / 48–70）。整块打包成一条记录。

    非零内容 = a·W² + 194W + 24，a = 4.0（1H）/ 4.5（CrazyCuckoo 层级），尾部 8 个零填充。
    194 = 128(attn_bias) + 64(每通道张量) + 2(cos_skip)

    **只有 attn_bias 的边界被实测确认**（四个层级各自的段位置在多 block 投票中复现）。
    其余段的 slot 归属未定，因此 forward 未实现 —— 强行给一个顺序只会静默出错。
    """

    def __init__(self, W, device=None):
        super().__init__()
        self.W, self.H = W, W // HEAD_DIM

    def load_from(self, recs, blk, device=None):
        r = recs["block%d.layer0.layer" % blk]
        P = lambda a: nn.Parameter(_t(a, device), False)
        self.attn_bias = P(r["attn_bias"])                    # [实测]
        self.packed_head = P(r["packed_head"])                # [未解]
        self.packed_tail = P(r["packed_tail"])                # [未解]
        if "input_adapter_weight" in r:
            self.input_adapter = P(r["input_adapter_weight"])  # [推断] 16→W
        return self

    def forward(self, x):
        raise NotImplementedError(
            "单记录 block 的 slot 顺序未解出。已确认的只有 attn_bias（%d 元素）；"
            "其余 %s 个元素的归属未定，随便定个顺序会静默出错。"
            % (self.attn_bias.numel(),
               "{:,}".format(self.packed_head.numel() + self.packed_tail.numel())))
