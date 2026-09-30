"""2h swin 块 (64 通道, 2 头) 的逐步复现。所有映射都由数据求出，求出后与权重字节配对。"""
import json
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment as lsa

sys.path.insert(0, os.path.dirname(__file__))
import ffn2h as F  # noqa: E402
from act import chw_to_tin, random_fp8  # noqa: E402
from exp_ffn import mp_cubic_silu, perm_in, q8, unswizzle  # noqa: E402

C, H, W = 64, 96, 160
WR = F.WREAL
CMAP = json.load(open(os.path.join(os.path.dirname(__file__), "maps_block1.json")))["W1col_to_W2row"][:32]
MAPS_PATH = os.path.join(os.path.dirname(__file__), "maps_2h.json")


def canon_index():
    """片段通道 -> 规范通道 (每 32 组内 perm_in)，返回 cg: X[:, cg] = 规范序"""
    PI = [perm_in(c) for c in range(32)]
    cg = np.zeros(64, int)
    for gi in range(2):
        for c in range(32):
            cg[gi * 32 + PI[c]] = gi * 32 + c
    return cg


def sorted_match(A, B, axis):
    """A、B 行列顺序都未知时，按 '排序后的向量' 做匈牙利配对。axis=1 配列，0 配行。返回 p: A 的第 i 个 <-> B 的第 p[i] 个"""
    if axis == 1:
        A, B = A.T, B.T
    cost = np.array([[np.abs(np.sort(a) - np.sort(b)).sum() for b in B] for a in A])
    return lsa(cost)[1]


def ffn_out(Xf, w=WR):
    """Xf: (N,64) 片段序输入 -> FFN 输出 (N,64) 片段序 (D 映射由 maps_2h.json 给出)"""
    m = json.load(open(MAPS_PATH))
    a = q8(mp_cubic_silu(F.hidden(Xf[:, canon_index()], w, "AB", "n_then_k")))
    Z = q8(np.hstack([sum(a[:, wp * 128 + j * 32: wp * 128 + (j + 1) * 32] @
                          unswizzle(w[16384 + wp * 4096 + j * 1024: 16384 + wp * 4096 + (j + 1) * 1024], 32, 32)[CMAP, :]
                          for j in range(4)) for wp in range(2)]))
    D = unswizzle(w[24576:28672], 64, 64)
    return Z @ D[m["D_rows"]][:, m["D_cols"]]


def step_c1():
    """注意力关掉 (投影 0, c2 1)，输出 = c1⊙x + FFN；减掉 FFN 后逐通道拟合 c1"""
    x = random_fp8((C, H, W), -3, 3, seed=2)
    w = F.isolated_ffn_weights()
    w[28688:28816] = WR[28688:28816]                                  # 恢复真实 c1
    Y = F.run(w, chw_to_tin(x)).reshape(C, -1).T
    X = x.reshape(C, -1).T
    R = Y - ffn_out(X)
    c1_fit = np.array([np.dot(X[:, c], R[:, c]) / np.dot(X[:, c], X[:, c]) for c in range(64)])
    c1 = np.frombuffer(WR[28688:28816], np.float16).astype(np.float32)
    cost = np.abs(c1_fit[:, None] - c1[None, :])
    p = lsa(cost)[1]
    print("c1: 片段通道 -> 存放下标:", p.tolist())
    print("    拟合值与配对后存放值最大差 %.4f" % np.abs(c1_fit - c1[p]).max())
    return p.tolist()


def attn_weights(c2=None):
    """FFN 关 (D=0)，c1=1 -> 注意力输入 = x；c2 默认 0 -> 输出只剩 proj(attn)"""
    w = bytearray(WR)
    w[24576:28672] = bytes(4096)
    w[28688:28816] = np.ones(64, np.float16).tobytes()
    w[61616:61744] = (np.zeros(64) if c2 is None else c2).astype(np.float16).tobytes()
    return w


def bias_head(w, h):
    """头 h 的 64x64 偏置 (像素行优先序)，解码同 1h"""
    import block1_ref as R
    fake = bytearray(20672)
    fake[11360:19552] = w[41120 + h * 8192: 41120 + (h + 1) * 8192]
    return R.bias_matrix(bytes(fake))


QK_LAYOUT = "contig"


def attention(Xf, w, heads=2, shift=(-4, -4)):
    """Xf: (N=H*W, 64) 片段序注意力输入 -> 各头输出拼接 (N, 64)，列 = 头*32 + V 维。
    shift: 窗口原点偏移 (seq7/block6 为 (-4,-4))，越界 token 按零 (block2 已验证)"""
    import block1_ref as R
    Y = q8(Xf[:, canon_index()])                                      # 规范序 fp8
    # qkv 按 K 的前后两半分开存 (PTX: 两组读址 28832.. 与 34976..，相差 32x192)
    Wqkv = np.vstack([unswizzle(w[28832:34976], 32, 192), unswizzle(w[34976:41120], 32, 192)])
    oy, ox = -shift[0], -shift[1]
    ny, nx = (H + oy + 7) // 8, (W + ox + 7) // 8
    pad = np.zeros((ny * 8, nx * 8, 64), np.float32)
    pad[oy:oy + H, ox:ox + W] = Y.reshape(H, W, 64)
    Yw = pad.reshape(ny, 8, nx, 8, 64).transpose(0, 2, 1, 3, 4).reshape(-1, 64, 64)
    # 按头分块 (探针实测): 头 h 占列 [96h, 96h+96)，其中 Q 0-31 / K 32-63 / V 64-95 —— 与 1h 单头同构
    outs = []
    for h in range(heads):
        base = 96 * h
        qh = R.f16(Yw @ Wqkv[:, base:base + 32])
        kh = R.f16(Yw @ Wqkv[:, base + 32:base + 64])
        vh = R.f16(Yw @ Wqkv[:, base + 64:base + 96])
        tau = np.frombuffer(w[57504 + 4 * h:57508 + 4 * h], np.float32)[0]
        qn = qh / np.sqrt(np.maximum((qh * qh).sum(-1, keepdims=True), R.EPS)) * tau
        kn = kh / np.sqrt(np.maximum((kh * kh).sum(-1, keepdims=True), R.EPS))
        L = R.f16(q8(qn) @ q8(kn).transpose(0, 2, 1) + bias_head(w, h))
        e = np.exp(L - L.max(-1, keepdims=True))
        P = e / e.sum(-1, keepdims=True)
        outs.append(R.f16(q8(P) @ q8(vh)))
    o = np.concatenate(outs, -1)
    o = o.reshape(ny, nx, 8, 8, 64).transpose(0, 2, 1, 3, 4).reshape(ny * 8, nx * 8, 64)[oy:oy + H, ox:ox + W]
    return o.reshape(-1, 64)


def step_attn():
    x = random_fp8((C, H, W), -3, 3, seed=3)
    Y = F.run(attn_weights(), chw_to_tin(x)).reshape(C, -1).T
    X = x.reshape(C, -1).T
    idx = np.random.default_rng(0).choice(len(Y), 8000, replace=False)
    O = attention(X, WR)
    M, *_ = np.linalg.lstsq(np.hstack([O[idx], np.ones((len(idx), 1))]), Y[idx], rcond=None)
    P = np.hstack([O, np.ones((len(O), 1))]) @ M
    r2 = 1 - ((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()
    print(f"注意力 (2 头, 头 h 用 Q/K/V 第 32h..32h+31 维): 输出 = O @ M 拟合 R² = {r2:.5f}")
    # 对照: 故意把两个头的偏置对调
    w2 = bytearray(WR)
    w2[41120:49312], w2[49312:57504] = WR[49312:57504], WR[41120:49312]
    O2 = attention(X, w2)
    M2, *_ = np.linalg.lstsq(np.hstack([O2[idx], np.ones((len(idx), 1))]), Y[idx], rcond=None)
    P2 = np.hstack([O2, np.ones((len(O2), 1))]) @ M2
    print(f"   对照 (两头偏置对调): R² = {1 - ((Y - P2) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum():.5f}")
    return M[:64]


def block(Xf, w, shift=(-4, -4)):
    """完整 2h swin 块。Xf: (N,64) 片段序输入 -> (N,64) 片段序输出 (fp8 值)"""
    import block1_ref as R
    m = json.load(open(MAPS_PATH))
    cols, rows = m["D_cols"], m["D_rows"]
    c1 = np.frombuffer(w[28688:28816], np.float16).astype(np.float32)
    c2 = np.frombuffer(w[61616:61744], np.float16).astype(np.float32)
    Yf = R.f16(c1[cols] * Xf + ffn_out(Xf, w))                      # FFN 段 (片段序)
    O = attention(Yf, w, shift=shift)                                 # 注意力输入 = FFN 段输出
    Pm = unswizzle(w[57520:61616], 64, 64)[rows][:, cols]
    return q8(R.f16(c2[cols] * Yf + O @ Pm))


def check_real(seq=7, shift=(-4, -4)):
    from act import TAPS, tin_to_chw
    import krun
    names = {6: ("tap_s5_1cafb000.bin", "tap_s6_1c19b000.bin"), 7: ("tap_s6_1c19b000.bin", "tap_s7_1c28b000.bin"),
             8: ("tap_s7_1c28b000.bin", "tap_s8_1c37b000.bin")}
    fi, fo = names[seq]
    _, wname, w = krun.weight_of(seq)
    X = tin_to_chw(np.frombuffer(open(os.path.join(TAPS, fi), "rb").read(), np.uint8), C, H, W).reshape(C, -1).T
    Yr = tin_to_chw(np.frombuffer(open(os.path.join(TAPS, fo), "rb").read(), np.uint8), C, H, W).reshape(C, -1).T
    Yp = block(X, w, shift)
    err = np.abs(Yp - Yr)
    print(f"seq{seq} {wname} 平移{shift}: 相关 {np.corrcoef(Yp.ravel(), Yr.ravel())[0, 1]:.5f}  "
          f"逐位相等 {(err == 0).mean():.4f}  最大误差 {err.max():.3f}")


def step_proj(M):
    """拟合出的投影 M (注意力维 x 输出片段通道) 与权重字节配对，两种存法都试"""
    cands = {"整块": unswizzle(WR[57520:61616], 64, 64),
             "K 分两半": np.vstack([unswizzle(WR[57520:59568], 32, 64), unswizzle(WR[59568:61616], 32, 64)])}
    best = None
    for name, P in cands.items():
        cols = sorted_match(M, P, 1)
        rows = sorted_match(M, P[:, cols], 0)
        Pm = P[rows][:, cols]
        r = np.corrcoef(M.ravel(), Pm.ravel())[0, 1]
        print(f"投影存法={name}: 配对后相关 {r:.5f}")
        print("   注意力维 -> 投影行:", rows.tolist())
        print("   输出通道 -> 投影列:", cols.tolist())
        if best is None or r > best[0]:
            best = (r, name, rows.tolist(), cols.tolist())
    return best


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "attn":
    M = step_attn()
    r, name, rows, cols = step_proj(M)
    m = json.load(open(MAPS_PATH))
    m.update({"proj_layout": name, "proj_rows": rows, "proj_cols": cols})
    json.dump(m, open(MAPS_PATH, "w"))
    sys.exit()

if __name__ == "__main__":
    maps = {"D_rows": None, "D_cols": None}
    # D 的行列映射 (上一步配对结果: z 维 -> D 行, 输出片段通道 -> D 列)
    maps["D_rows"] = [0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15, 16, 17, 20, 21, 24, 25, 28, 29, 18, 19, 22, 23,
                      26, 27, 30, 31, 32, 33, 36, 37, 40, 41, 44, 45, 34, 35, 38, 39, 42, 43, 46, 47, 48, 49, 52, 53, 56,
                      57, 60, 61, 50, 51, 54, 55, 58, 59, 62, 63]
    maps["D_cols"] = [0, 16, 2, 18, 4, 20, 6, 22, 1, 17, 3, 19, 5, 21, 7, 23, 8, 24, 10, 26, 12, 28, 14, 30, 9, 25, 11, 27,
                      13, 29, 15, 31, 32, 48, 34, 50, 36, 52, 38, 54, 33, 49, 35, 51, 37, 53, 39, 55, 40, 56, 42, 58, 44,
                      60, 46, 62, 41, 57, 43, 59, 45, 61, 47, 63]
    json.dump(maps, open(MAPS_PATH, "w"))
    maps["c1"] = step_c1()
    json.dump(maps, open(MAPS_PATH, "w"))
