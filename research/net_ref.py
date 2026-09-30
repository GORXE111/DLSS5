"""DLSS5 整网 numpy 参考 (按 exec_order 逐 kernel 串联已验证的块实现)。

激活一律按"缓冲内存序"传递:
  tin 缓冲  -> (N, C) 片段序 (= tin_to_chw 的通道序)，N = H*W 行优先
  图像缓冲 -> (N, C) 规范序 ([C/16][H][W][16] 的通道序)
整网在补齐后的 384x640 网格上运行: 1h 192x320, 2h 96x160, 4h 48x80, 8h 24x40, 16h 12x20, ViT 8x12 (6x10 真实)。
"""
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import block1_ref as R  # noqa: E402
import k16_attn as A16  # noqa: E402
import k16_block as B16  # noqa: E402
import k16_ffproj as P16  # noqa: E402
import k39  # noqa: E402
import klab  # noqa: E402
import krun  # noqa: E402
import swin_gen as G  # noqa: E402
import vit  # noqa: E402
from exp_ffn import mp_cubic_silu, perm_in, q8, unswizzle  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXEC = json.load(open(os.path.join(ROOT, "torch", "dlss5", "data", "exec_order.json")))["launches"]
LEVEL = {"1h": (32, 192, 320), "2h": (64, 96, 160), "4h": (128, 48, 80), "8h": (256, 24, 40)}


def record(name):
    for r in krun._RECS:
        if r["name"] == name:
            with open(os.path.join(ROOT, "WEIGHTS_HT.bin"), "rb") as f:
                f.seek(r["off"])
                return f.read(r["C"])
    raise KeyError(name)


def shift_of(params, H, W):
    """平移字 = 参数块里紧跟 (H, W) 尺寸字之后的那个 64 位字 (x 低 32 位, y 高 32 位)"""
    dims = H | (W << 32)
    words = [struct.unpack_from("<Q", params, i)[0] for i in range(0, len(params) // 8 * 8, 8)]
    return G.shift_of(words[words.index(dims) + 1])


# ------------------------------------------------------------------ 1h (W=32) 片段
PI = [perm_in(c) for c in range(32)]
SG = G.COLS32                                  # = perm_param


def ffn1h(Xf, w):
    """1h FFN 增量 (片段序)，不含 c1 残差"""
    Xc = Xf[:, G.canon_index(32)]
    W1 = unswizzle(w[0:4096], 32, 128)
    W2 = unswizzle(w[4096:8192], 128, 32)[R.C_MAP, :]
    ff = q8(mp_cubic_silu(R.f16(Xc @ W1))) @ W2
    out = np.zeros_like(Xf)
    out[:, R.A_MAP] = ff
    return out


def attn1h(Yf, w, H, Wd, shift):
    """1h 注意力 + c2 残差 (swin_ref 同式)。w 为标准 1h 记录"""
    Y = q8(Yf[:, G.canon_index(32)])
    Wqkv = unswizzle(w[8288:11360], 32, 96)
    tau = np.frombuffer(w[19552:19556], np.float32)[0]
    Wproj = unswizzle(w[19568:20592], 32, 32)
    c2 = np.frombuffer(w[20592:20656], np.float16).astype(np.float32)
    bias = R.bias_matrix(w)
    vperm = np.argsort(R.C_MAP[:32])
    oy, ox = -shift[0], -shift[1]
    ny, nx = (H + oy + 7) // 8, (Wd + ox + 7) // 8
    pad = np.zeros((ny * 8, nx * 8, 32), np.float32)
    pad[oy:oy + H, ox:ox + Wd] = Y.reshape(H, Wd, 32)
    Yw = pad.reshape(ny, 8, nx, 8, 32).transpose(0, 2, 1, 3, 4).reshape(-1, 64, 32)
    q, k, v = R.f16(Yw @ Wqkv[:, :32]), R.f16(Yw @ Wqkv[:, 32:64]), R.f16(Yw @ Wqkv[:, 64:])
    qn = q / np.sqrt(np.maximum((q * q).sum(-1, keepdims=True), R.EPS)) * tau
    kn = k / np.sqrt(np.maximum((k * k).sum(-1, keepdims=True), R.EPS))
    L = R.f16(q8(qn) @ q8(kn).transpose(0, 2, 1) + bias)
    e = np.exp(L - L.max(-1, keepdims=True))
    P = e / e.sum(-1, keepdims=True)
    o = R.f16(q8(P) @ q8(v[:, :, vperm]))
    o_img = o.reshape(ny, nx, 8, 8, 32).transpose(0, 2, 1, 3, 4).reshape(ny * 8, nx * 8, 32)[oy:oy + H, ox:ox + Wd]
    pr = o_img.reshape(-1, 32) @ Wproj
    pr_frag = np.zeros_like(pr)
    pr_frag[:, R.A_MAP] = pr
    return q8(R.f16(c2[SG] * Yf + pr_frag))


def block1h(Xf, w, H, Wd, shift):
    c1 = np.frombuffer(w[8208:8272], np.float16).astype(np.float32)
    Yf = R.f16(c1[SG] * Xf + ffn1h(Xf, w))
    return attn1h(Yf, w, H, Wd, shift)


# ------------------------------------------------------------------ 通用 swin 块 (各变体)
def std_block(Xf, w, W, H, Wd, shift):
    return block1h(Xf, w, H, Wd, shift) if W == 32 else G.block(Xf, w, W, H, Wd, shift)


def img_to_frag(img, W):
    return img[:, np.argsort(G.canon_index(W))]


def frag_to_img(Yf, W):
    return Yf[:, G.canon_index(W)]


def ds_block(Xf, w, W, H, Wd, shift):
    """编码出口: 返回 (跳连 tin 片段序, 下一级图像 (N/4, 2W) 规范序)"""
    o, _ = G.layout(W) if W > 32 else ({"end": 20672 + 16}, 1)
    y = std_block(Xf, w[:20672] if W == 32 else w[:o["end"] - 16] + bytes(16), W, H, Wd, shift)
    off = 20656 if W == 32 else o["end"] - 16             # 1h: W_ds 紧跟 c2 (记录尾 16B 填充)
    Wds = unswizzle(w[off:off + 2 * W * W], W, 2 * W)
    Yc = frag_to_img(y, W).reshape(H // 2, 2, Wd // 2, 2, W).mean((1, 3)).reshape(-1, W)
    ds = q8(R.f16(R.f16(Yc) @ Wds))[:, np.argsort(G.group_map(G.CMAP32, 2 * W))]
    return y, ds


def up_block(low_img, skip_f, w, W, H, Wd, shift):
    """解码入口 (1h 与 8h 实测，统一结构): z = c⊙skip + nn_up2x(low规范序·W_up)，y = s⊙z + FFN(z)，再标准注意力。
    记录里 s 在前、c 在后 (1h: s [10256,10320)，c [10336,10400)；W>=64: 紧凑拼接 W_up | s | c | qkv…)"""
    h, wd = H // 2, Wd // 2
    up = low_img.reshape(h, wd, 2 * W).repeat(2, 0).repeat(2, 1).reshape(H * Wd, 2 * W)
    if W == 32:
        Wu = unswizzle(w[8192:10240], 64, 32)
        U = np.zeros((H * Wd, 32), np.float32)
        U[:, R.A_MAP] = R.f16(up @ Wu)
        s = np.frombuffer(w[10256:10320], np.float16).astype(np.float32)[SG]
        c1 = np.frombuffer(w[10336:10400], np.float16).astype(np.float32)[SG]   # 1h: W_up | 16B | s | 16B | c1 | qkv (实测)
        std = w[:8192] + bytes(16) + w[10336:10400] + bytes(16) + w[10400:22784]
        z = R.f16(R.f16(c1 * skip_f) + U)
        Yf = R.f16(R.f16(s * z) + ffn1h(z, std))
        return attn1h(Yf, std, H, Wd, shift)
    o, _ = G.layout(W)
    fe = o["c1"] - 16
    cols = np.array(G.group_map(G.COLS32, W))
    p = fe + 2 * W * W
    s = np.frombuffer(w[p:p + 2 * W], np.float16).astype(np.float32)[cols]
    c1 = np.frombuffer(w[p + 2 * W:p + 4 * W], np.float16).astype(np.float32)[cols]
    q0 = p + 4 * W
    std = w[:fe] + bytes(16) + w[p + 2 * W:p + 4 * W] + bytes(16) + w[q0:q0 + o["end"] - 16 - o["qkv"]] + bytes(16)
    U = R.f16(up @ unswizzle(w[fe:fe + 2 * W * W], 2 * W, W))[:, cols]
    z = R.f16(R.f16(c1 * skip_f) + U)
    Yf = R.f16(R.f16(s * z) + G.ffn(z, std, W))
    c2 = np.frombuffer(std[o["c2"]:o["c2"] + 2 * W], np.float16).astype(np.float32)
    O = G.attention(Yf, std, W, H, Wd, shift)
    Pm = unswizzle(std[o["proj"]:o["proj"] + W * W], W, W)[G.group_map(G.CMAP32, W)][:, cols]
    return q8(R.f16(c2[cols] * Yf + O @ Pm))


# ------------------------------------------------------------------ 16h / ViT
def shift16(params):
    x, y = struct.unpack_from("<2i", params, 32)
    return (y, x)


def run_net(pre_img, trace, log=print, stop=None):
    """从 pre_block 输出 (图像, (192*320, 32) 规范序) 串到 block69 输出 (图像)。返回 {seq: 输出} 以便对照"""
    outs = {}
    skips = {}
    x = pre_img
    cur = None                                     # 当前级 tin 片段序
    i = 1
    while i < len(EXEC):
        L = EXEC[i]
        seq, k, wn = L["seq"], L["kernel"], L["weights"]
        if k.startswith("cc_tinlayout_fused_pre_block") or k.startswith("cc_cb_clear"):
            i += 1
            continue
        if "post_block" in k:
            break
        params = klab.trace_row(seq, trace)["params"]
        lv = next((lv for lv in LEVEL if f"_{lv}_" in k), None)
        if lv:
            W, H, Wd = LEVEL[lv]
            w = record(wn[0])
            sh = shift_of(params, H, Wd)
            if "inpview" in k:
                cur = std_block(img_to_frag(x, W), w, W, H, Wd, sh)
            elif "ds_wait" in k or k.endswith("_ds_fp8"):
                cur, x = ds_block(cur, w, W, H, Wd, sh)
                skips[lv] = cur
                outs[seq] = x
            elif "upsample" in k:
                cur = up_block(x, skips[lv], w, W, H, Wd, sh)
            elif "outview" in k:
                x = frag_to_img(std_block(cur, w, W, H, Wd, sh), W)
                outs[seq] = x
            else:
                cur = std_block(cur, w, W, H, Wd, sh)
            log(f"seq{seq} {k[3:50]} {lv} 平移{sh}")
            i += 1
        elif "split_swin_16h" in k:
            ws = [record(EXEC[i + j]["weights"][0]) for j in range(4)]
            k0 = k
            Xf = img_to_frag(x, 512) if "inpview" in k0 else cur
            Z = B16.ffwd(Xf, ws[0])
            y = P16.ref(Z, Xf, ws[1])
            p_attn = klab.trace_row(EXEC[i + 2]["seq"], trace)["params"]
            O = q8(A16.attention(y, ws[2], shift16(p_attn)))[:, B16.COLS]
            cur = P16.ref(O, y, ws[3])
            kp = EXEC[i + 3]["kernel"]
            if "pool" in kp:
                skips["16h"] = cur
                pooled = np.zeros((8, 12, 512), np.float32)
                pooled[:6, :10] = q8(R.f16(cur.reshape(6, 2, 10, 2, 512).mean((1, 3))))
                hw = record(EXEC[i + 4]["weights"][0])
                xc = pooled.reshape(96, 512)[:, G.canon_index(512)]
                cur = q8(R.f16(xc @ unswizzle(hw[:524288], 512, 1024)))[:, np.array(G.group_map(G.COLS32, 1024))]
                cur[np.repeat(np.arange(8) >= 6, 12) | np.tile(np.arange(12) >= 10, 8)] = 0
                i += 5
            elif "outview" in kp:
                x = frag_to_img(cur, 512)
                outs[EXEC[i + 3]["seq"]] = x
                i += 4
            else:
                i += 4
            log(f"seq{seq} 16h 块 {wn[0]}")
        elif "vit_1d_repack_2d_to_1d" in k:
            i += 1                                 # 8x12 行优先展平 = 我们的 (96, C) 表示，无需操作
        elif "vit_1d_ffn_expand" in k:
            ws = [record(EXEC[i + j]["weights"][0]) for j in range(5)]
            cur = vit.block(cur, ws)
            log(f"seq{seq} ViT 块 {wn[0]}")
            i += 5
        elif "vit_1d_repack_1d_to_2d" in k:
            i += 1
        elif "dec_input_upsample" in k:
            w = record(wn[0])
            low = cur.T.reshape(1024, 8, 12)
            cur = k39.ref(low, skips["16h"].T.reshape(512, 12, 20), w).reshape(512, -1).T
            log(f"seq{seq} block39")
            i += 1
        else:
            log(f"跳过 seq{seq} {k}")
            i += 1
        if stop is not None and seq >= stop:
            break
    return outs, cur, x


# ------------------------------------------------------------------ 整帧: 纹理 -> 最终画面
def color_pattern(W=640, H=360):
    """nr-lab 的合成输入 (MakeColorPattern, sRGB 配置, R8G8B8A8_UNORM)"""
    y, x = np.mgrid[0:H, 0:W]
    checker = ((x // 12) ^ (y // 12)) & 1
    line = (x % 61 < 2) | (y % 47 < 2) | ((x + y) % 79 < 2)
    r = np.where(line, 1.0, np.where(checker, 0.82, 0.06))
    g = np.where(line, 0.18, np.where(checker, 0.11, 0.68))
    b = np.where(line, 0.04, np.where(checker, 0.55, 0.09))
    img = np.stack([r, g, b], -1)
    return (np.floor(np.minimum(img, 1.0) * 255.0 + 0.5) / 255.0).astype(np.float32)


def run_frame(color, hist, mv, frame, trace, log=print):
    """一帧完整前向。hist/mv 为 None 表示重置帧 (kernel 以颜色充当历史，post 不混合历史)。返回 (输出 RGB (H,W,3), 中间量)"""
    import pre_ref as PRE
    import post_ref as POST
    H, W = color.shape[:2]
    w0 = record("block0.layer0.layer")
    X = PRE.inputs(color, color if hist is None else hist, np.zeros((H, W, 2), np.float32) if mv is None else mv, frame)
    a = q8(R.f16(X @ PRE.adapter_matrix(w0)))[:, np.argsort(G.canon_index(32))]
    y0 = block1h(a, w0[:8208] + w0[9232:], 384, 640, (0, 0))              # pre_block 的 swin 在全分辨率上跑
    pre_img = q8(R.f16(frag_to_img(y0, 32).reshape(192, 2, 320, 2, 32).mean((1, 3)).reshape(-1, 32)))
    log("pre_block 完成")
    outs, cur, x = run_net(pre_img, trace, log=log)
    net = POST.swin_post(x.T.reshape(32, 192, 320), y0.T.reshape(32, 384, 640), record("block70.layer0.layer"), H=384, W=640)
    net = net[:, :H, :W]
    if hist is None:
        out = np.clip(color + 0.25 * net[:3].transpose(1, 2, 0), 0, 1)
    else:
        out = POST.compose(net, color, hist, mv)
    return out, {"pre_img": pre_img, "y0": y0, "x69": x, "net": net, "outs": outs}
