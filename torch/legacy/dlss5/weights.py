"""WEIGHTS_HT 加载器 —— 把 153 条记录切成命名张量。

判据一律是「零剩余」：每条记录的字节必须被 slot 完全消费，
全部 153 条相加必须等于 73,841,889。跑 check.py 验证。

不依赖 torch（只用 struct + numpy），这样形状核对不需要 GPU。
"""
import os, re, struct
import numpy as np

WEIGHTS_BIN = os.path.join(os.path.dirname(__file__), "..", "..", "..", "WEIGHTS_HT.bin")

HEAD_DIM = 32           # 全网统一，由 2H/4H/8H/16H 四组模板配置互证
WINDOW = 64             # 8×8 tile
PAD_TAIL = 8            # 每条单记录 block 末尾的零填充


# ---------------------------------------------------------------- 记录框架

def parse_records(path=WEIGHTS_BIN):
    """走记录框架，返回 [(name, byte_off, n_bytes, n_params), ...]。"""
    blob = open(path, "rb").read()
    total = struct.unpack_from("<Q", blob, 0)[0]
    assert total == len(blob), "资源长度不符: %d vs %d" % (total, len(blob))
    out, off = [], 8
    while off < len(blob) - 8:
        nl = struct.unpack_from("<Q", blob, off)[0]
        if not (1 <= nl <= 128):
            break
        name = blob[off + 8:off + 8 + nl].decode("ascii")
        p = off + 8 + nl
        _A, _B, C = struct.unpack_from("<QQQ", blob, p)
        pay = p + 28
        Z = struct.unpack_from("<I", blob, pay + C + 16)[0]
        assert C == Z * 2, "%s: 字节/元素比不是 2" % name
        out.append((name, pay, C, Z))
        off = pay + C + 20
    assert off == len(blob), "解析有剩余: %d 字节" % (len(blob) - off)
    return blob, out


def _f16(blob, off, n):
    return np.frombuffer(blob, dtype="<f2", count=n, offset=off)


# ---------------------------------------------------------------- 层级识别

#: block ID → (width, 家族)。由镜像结构与参数量公式确定。
def tier_of(block):
    for lo, hi, W, fam in (
        (0, 4, 32, "single1h"), (5, 8, 64, "single"), (9, 14, 128, "single"),
        (15, 22, 256, "single"), (23, 29, 256, "cuckoo4"), (30, 30, 256, "cuckoo5"),
        (31, 38, 512, "split16h"), (39, 39, 512, "bridge"),
        (40, 47, 256, "cuckoo4"), (48, 55, 256, "single"), (56, 61, 128, "single"),
        (62, 65, 64, "single"), (66, 69, 32, "single1h"), (70, 70, 32, "output"),
    ):
        if lo <= block <= hi:
            return W, fam
    raise KeyError(block)


# ---------------------------------------------------------------- 各家族切分

def split_split16h(name, v, W=512):
    """16H 中央核。5 条记录，5/5 精确闭合。

    ×2 来自 SplitSwin 的双路；FinalHead 位于两路汇合之后，不翻倍。
    """
    H = W // HEAD_DIM
    L = name.split(".")[1]
    if L == "layer0":                       # FFN expand 512→2048 ×2 + 8
        return [("ffn_expand", v[:2 * W * 4 * W].reshape(2, W, 4 * W)),
                ("tail", v[2 * W * 4 * W:])]
    if L == "layer1":                       # FFN contract 2048→512 ×2 + bias
        n = 2 * 4 * W * W
        return [("ffn_contract", v[:n].reshape(2, 4 * W, W)),
                ("ffn_cos_skip", v[n:])]
    if L == "layer2":                       # attn_scale 在前，qkv 在后
        return [("attn_scale", v[:H * 4].reshape(2, H, 2)),
                ("qkv_weight", v[H * 4:].reshape(2, W, 3 * W))]
    if L == "layer3":
        return [("attn_cos_skip", v)]       # 单标量
    if L == "layer4":                       # FinalHead 512→1024 + bias
        n = W * 2 * W
        return [("final_head", v[:n].reshape(W, 2 * W)), ("bias", v[n:])]
    raise KeyError(name)


def split_cuckoo(name, v, W=256):
    """CrazyCuckoo 4/5 记录组。十个 slot，边界在 15/15 个同尺寸 block 中复现。"""
    H = W // HEAD_DIM
    L = name.split(".")[1]
    if L == "layer0":
        a, b = 2 * W * W, W * W
        return [("weight0", v[:a].reshape(2, W, W)),
                ("weight1", v[a:a + b].reshape(W, W)),
                ("weight2", v[a + b:].reshape(W, W))]
    if L == "layer1":
        n = 2 * W * W
        return [("weight3", v[:n].reshape(2, W, W)), ("ffn_cos_skip", v[n:])]
    if L == "layer2":
        q, bi = 2 * 3 * W * W, H * 2 * WINDOW * WINDOW
        return [("qkv_weight", v[:q].reshape(2, W, 3 * W)),
                ("attn_bias", v[q:q + bi].reshape(2, H, WINDOW, WINDOW)),
                ("attn_scale", v[q + bi:].reshape(2, H, 2))]
    if L == "layer3":
        n = 2 * W * W
        return [("projection_weight", v[:n].reshape(2, W, W)), ("attn_cos_skip", v[n:])]
    if L == "layer4":                        # 仅 block30 有
        return [("extra", v)]
    raise KeyError(name)


def split_single(name, v, W, is_1h, has_adapter):
    """单记录 block。

    slot **顺序**来自 host x86 代码（CCSingleLayerBlock 的绑定序列，
    用 dllq xref 扫 lea rip-relative 得到），**尺寸**来自参数量闭合：

        weight1            0.5W²    仅 W>=64（CrazyCuckoo 层级）
        weight2            64W      每通道 x 64 窗口 token
        ffn_cos_skip       W
        qkv_weight         3W²
        attn_scale         24       常数
        attn_bias          128W  = H x 64 x 64
        projection_weight  W²
        attn_cos_skip      W
        _pad               8

    四个层级 4/4 精确（W=256 有 +8 残留）。标注为 [推断]：
    顺序有代码依据、尺寸有计数闭合，但用幅度分段做交叉核对时边界对不齐，
    所以映射是强假设而非已证。
    """
    H = W // HEAD_DIM
    n = len(v)
    parts, o = [], 0

    def take(k, cnt, shape=None):
        nonlocal o
        a = v[o:o + cnt]
        parts.append((k, a.reshape(shape) if shape else a))
        o += cnt

    if has_adapter:
        take("input_adapter_weight", 16 * W, (16, W))
    if not is_1h:
        take("weight1", W * W // 2)
    take("weight2", 64 * W, (W, 64))
    take("ffn_cos_skip", W)
    take("qkv_weight", 3 * W * W, (W, 3 * W))
    take("attn_scale", 24)
    take("attn_bias", H * WINDOW * WINDOW, (H, WINDOW, WINDOW))
    take("projection_weight", W * W, (W, W))
    take("attn_cos_skip", W)
    if n - o - PAD_TAIL > 0:                     # W=256 的 +8 残留等
        take("_residual", n - o - PAD_TAIL)
    take("_pad", n - o)
    return parts


def split_bridge(name, v, W=512):
    """block39：decoder 输入过渡，512×512 + bias。"""
    n = W * W
    return [("inp_upsample_weight", v[:n].reshape(W, W)),
            ("inp_upsample_input_scale", v[n:])]


def split_output(name, v, W=32):
    if name.endswith("blend_scale"):
        return [("blend_scale", v)]
    return [("out_packed", v[:len(v) - PAD_TAIL]), ("_pad", v[len(v) - PAD_TAIL:])]


# ---------------------------------------------------------------- 顶层

def load(path=WEIGHTS_BIN, verbose=False):
    """返回 {记录名: {slot名: ndarray}}，并校验零剩余。"""
    blob, recs = parse_records(path)
    out, consumed = {}, 0
    for name, off, nb, nz in recs:
        v = _f16(blob, off, nz)
        blk = int(re.match(r"block(\d+)\.", name).group(1))
        W, fam = tier_of(blk)
        if fam == "split16h":
            parts = split_split16h(name, v, W)
        elif fam in ("cuckoo4", "cuckoo5"):
            parts = split_cuckoo(name, v, W)
        elif fam == "bridge":
            parts = split_bridge(name, v, W)
        elif fam == "output":
            parts = split_output(name, v, W)
        else:
            parts = split_single(name, v, W, fam == "single1h", blk == 0)
        got = sum(p.size for k, p in parts)
        assert got == nz, "%s: slot 合计 %d != 记录 %d" % (name, got, nz)
        out[name] = dict(parts)
        consumed += nz
        if verbose:
            print("  %-28s %10s  ->  %s" % (name, "{:,}".format(nz),
                                            ", ".join("%s[%s]" % (k, "×".join(map(str, p.shape)))
                                                      for k, p in parts)))
    assert consumed == 73_841_889, "总参数 %d != 73,841,889" % consumed
    return out
