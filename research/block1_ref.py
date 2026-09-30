"""block1 (cc_tinlayout_fused_swin_1h_32_1_inpview) 的完整参考实现，对照真实 kernel 输出。

权重布局 (20672 B，全部由 PTX + 探针实测):
  [0,4096)      FFN W1 32->128 fp8 (mma B 片段序)
  [4096,8192)   FFN W2 128->32 fp8
  [8192,8208)   4xf32 = 0 (预留)
  [8208,8272)   c1 32xf16 (FFN 残差系数)        [8272,8288) 零
  [8288,11360)  qkv 32->96 fp8: 列 0-31 Q, 32-63 K, 64-95 V，Q 第 d 列与 K 第 32+d 列配对
  [11360,19552) 注意力偏置 64x64 f16，按 QK mma 累加器片段序 (查询块在外)，token = 4x4 小块序
  [19552,19568) 4xf32: 头 0 的温度 τ (=0.4088)，其余 0
  [19568,20592) 输出投影 32->32 fp8
  [20592,20656) c2 32xf16 (注意力残差系数)      [20656,20672) 零
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import exp_block1 as B  # noqa: E402
from exp_ffn import mp_cubic_silu, perm_in, perm_param, q8, unswizzle  # noqa: E402

MAPS = json.load(open(os.path.join(os.path.dirname(__file__), "maps_block1.json")))
A_MAP = MAPS["W2col_to_outfrag"]      # W2 列 -> 输出片段通道
C_MAP = MAPS["W1col_to_W2row"]        # 隐层重排
EPS = np.float32(6.2e-05)


def f16(v):
    return np.asarray(v, np.float32).astype(np.float16).astype(np.float32)


def token_of_pixel(py, px):
    return 16 * (2 * (py // 4) + px // 4) + 4 * (py % 4) + (px % 4)


def bias_matrix(w):
    raw = np.frombuffer(w[11360:19552], np.float16).astype(np.float32)
    M = np.zeros((64, 64), np.float32)
    for ti in range(32):
        mt, nt = divmod(ti, 8)
        blk, half = divmod(ti, 2)
        for lane in range(32):
            g, t = lane >> 2, lane & 3
            base = blk * 256 + lane * 8 + half * 4
            for q, (r, c) in enumerate(((g, 2 * t), (g, 2 * t + 1), (g + 8, 2 * t), (g + 8, 2 * t + 1))):
                M[mt * 16 + r, nt * 8 + c] = raw[base + q]
    order = [token_of_pixel(p // 8, p % 8) for p in range(64)]
    return M[order][:, order]            # 行/列均为窗口内像素行优先序


def block1(x_in, w, v_rows="identity", proj_cols="A", tau_on_k=False):
    """x_in: (32, H, W) 输入 (输入内存通道序)。返回 (32, H, W) 输出 (输出片段通道序)。"""
    C, H, W = x_in.shape
    PI = [perm_in(c) for c in range(32)]
    SG = [perm_param(c) for c in range(32)]
    W1 = unswizzle(w[0:4096], 32, 128)
    W2 = unswizzle(w[4096:8192], 128, 32)[C_MAP, :]
    c1 = np.frombuffer(w[8208:8272], np.float16).astype(np.float32)
    Wqkv = unswizzle(w[8288:11360], 32, 96)
    tau = np.frombuffer(w[19552:19556], np.float32)[0]
    Wp = unswizzle(w[19568:20592], 32, 32)
    c2 = np.frombuffer(w[20592:20656], np.float16).astype(np.float32)
    bias = bias_matrix(w)

    X = x_in.reshape(32, -1).T                                    # (像素, 32) 输入内存序
    # ---- FFN 段，结果换到 "规范通道" = 输入内存序 (qkv 的行按它排列)
    ff = q8(mp_cubic_silu(f16(X @ W1))) @ W2                      # 列 = W2 列序
    ff_frag = np.zeros_like(X); ff_frag[:, A_MAP] = ff            # -> 输出片段序
    res_frag = c1[SG] * X[:, PI]
    Y_frag = f16(res_frag + ff_frag)
    Y = np.zeros_like(Y_frag); Y[:, PI] = Y_frag                  # 片段序 -> 输入内存序
    Y = q8(Y)                                                     # 注意力前量化为 fp8 (qkv mma 的 A 操作数)
    # ---- 注意力 (8x8 窗口)
    Yw = Y.reshape(H // 8, 8, W // 8, 8, 32).transpose(0, 2, 1, 3, 4).reshape(-1, 64, 32)
    q, k, v = f16(Yw @ Wqkv[:, :32]), f16(Yw @ Wqkv[:, 32:64]), f16(Yw @ Wqkv[:, 64:])
    qn = q / np.sqrt(np.maximum((q * q).sum(-1, keepdims=True), EPS)) * tau
    kn = k / np.sqrt(np.maximum((k * k).sum(-1, keepdims=True), EPS)) * (tau if tau_on_k else 1)
    L = f16(q8(qn) @ q8(kn).transpose(0, 2, 1) + bias)
    e = np.exp(L - L.max(-1, keepdims=True))
    P = e / e.sum(-1, keepdims=True)
    if v_rows == "C":
        v = v[:, :, np.argsort(C_MAP[:32])] if max(C_MAP[:32]) < 32 else v
    o = f16(q8(P) @ q8(v))                                        # (窗口, 64, 32)
    ow = o.reshape(H // 8, W // 8, 8, 8, 32).transpose(0, 2, 1, 3, 4).reshape(-1, 32)
    pr = ow @ Wp                                                  # 列 = 投影列序
    pr_frag = np.zeros_like(pr)
    if proj_cols == "A":
        pr_frag[:, A_MAP] = pr
    else:
        pr_frag = pr
    out = f16(c2[SG] * Y_frag + pr_frag)
    return q8(out).T.reshape(32, H, W)


def main():
    x = B.load_input()
    ref = B.run(B.INP_REAL, B.W_REAL)
    for v_rows in ("identity", "C"):
        for proj_cols in ("A", "identity"):
            for tk in (False, True):
                y = block1(x, B.W_REAL, v_rows, proj_cols, tk)
                err = np.abs(y - ref)
                r = np.corrcoef(y.ravel(), ref.ravel())[0, 1]
                print(f"V行={v_rows:8s} 投影列={proj_cols:8s} τ也乘K={tk!s:5s}: 相关 {r:.5f} 逐位相等 {(err == 0).mean():.4f} 最大误差 {err.max():.3f}")


if __name__ == "__main__":
    main()
