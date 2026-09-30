"""1h/32 通道 swin 块的通用参考实现 (block1 结构推广: 支持窗口平移、任意权重记录)。

与 block1_ref 相同的计算; 新增:
  shift  (sy, sx): 窗口原点偏移 (block2 为 (-4,-4))，像素 (y,x) 属于窗口 ((y-sy)//8, (x-sx)//8)
  oob   : 窗口伸出图像部分的处理 —— "zero" 按零 token 参与注意力 / "mask" 从 softmax 中剔除
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import block1_ref as R  # noqa: E402
from exp_ffn import mp_cubic_silu, perm_in, perm_param, q8, unswizzle  # noqa: E402


def ffn_stage(X, w):
    """X: (N, 32) 输入 (规范通道序 = 输入内存序)。返回 (Y 规范序 fp8, Y_frag f16)"""
    PI = [perm_in(c) for c in range(32)]
    SG = [perm_param(c) for c in range(32)]
    W1 = unswizzle(w[0:4096], 32, 128)
    W2 = unswizzle(w[4096:8192], 128, 32)[R.C_MAP, :]
    c1 = np.frombuffer(w[8208:8272], np.float16).astype(np.float32)
    ff = q8(mp_cubic_silu(R.f16(X @ W1))) @ W2
    ff_frag = np.zeros_like(X)
    ff_frag[:, R.A_MAP] = ff
    Y_frag = R.f16(c1[SG] * X[:, PI] + ff_frag)
    Y = np.zeros_like(Y_frag)
    Y[:, PI] = Y_frag
    return q8(Y), Y_frag


def swin_block(x_canon, w, shift=(0, 0), oob="zero"):
    """x_canon: (32, H, W) 规范通道序。返回 (32, H, W) 输出片段序 (fp8 值)"""
    C, H, W = x_canon.shape
    SG = [perm_param(c) for c in range(32)]
    Y, Y_frag = ffn_stage(x_canon.reshape(32, -1).T, w)
    Wqkv = unswizzle(w[8288:11360], 32, 96)
    tau = np.frombuffer(w[19552:19556], np.float32)[0]
    Wproj = unswizzle(w[19568:20592], 32, 32)
    c2 = np.frombuffer(w[20592:20656], np.float16).astype(np.float32)
    bias = R.bias_matrix(w)
    vperm = np.argsort(R.C_MAP[:32])

    sy, sx = shift
    oy, ox = -sy, -sx                                             # 图像像素 (0,0) 在填充网格中的位置
    ny, nx = (H + oy + 7) // 8, (W + ox + 7) // 8
    Hp, Wp = ny * 8, nx * 8
    pad = np.zeros((Hp, Wp, 32), np.float32)
    valid = np.zeros((Hp, Wp), bool)
    pad[oy:oy + H, ox:ox + W] = Y.reshape(H, W, 32)
    valid[oy:oy + H, ox:ox + W] = True
    Yw = pad.reshape(ny, 8, nx, 8, 32).transpose(0, 2, 1, 3, 4).reshape(-1, 64, 32)
    Vm = valid.reshape(ny, 8, nx, 8).transpose(0, 2, 1, 3).reshape(-1, 64)
    q, k, v = R.f16(Yw @ Wqkv[:, :32]), R.f16(Yw @ Wqkv[:, 32:64]), R.f16(Yw @ Wqkv[:, 64:])
    qn = q / np.sqrt(np.maximum((q * q).sum(-1, keepdims=True), R.EPS)) * tau
    kn = k / np.sqrt(np.maximum((k * k).sum(-1, keepdims=True), R.EPS))
    L = R.f16(q8(qn) @ q8(kn).transpose(0, 2, 1) + bias)
    if oob == "mask":
        L = np.where(Vm[:, None, :], L, -np.inf)
    e = np.exp(L - L.max(-1, keepdims=True))
    P = e / e.sum(-1, keepdims=True)
    o = R.f16(q8(P) @ q8(v[:, :, vperm]))
    o_img = o.reshape(ny, nx, 8, 8, 32).transpose(0, 2, 1, 3, 4).reshape(Hp, Wp, 32)[oy:oy + H, ox:ox + W]
    pr = o_img.reshape(-1, 32) @ Wproj
    pr_frag = np.zeros_like(pr)
    pr_frag[:, R.A_MAP] = pr
    out = R.f16(c2[SG] * Y_frag + pr_frag)
    return q8(out).T.reshape(32, H, W)


def frag_to_canon(y_frag):
    """输出片段序 -> 规范通道序 (下一块的输入)"""
    PI = [perm_in(c) for c in range(32)]
    out = np.zeros_like(y_frag)
    out[PI] = y_frag
    return out
