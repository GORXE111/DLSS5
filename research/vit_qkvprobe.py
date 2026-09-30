import numpy as np, vit, swin_gen as G
from act import E4M3

ONE = int(np.nanargmin(np.abs(E4M3 - 1.0)))


def wpos(k, n, N):
    """unswizzle(n_then_k) 的逆: 矩阵元素 (k, n) 在字节流中的位置"""
    ntile = N // 8
    kt, nt = k // 32, n // 8
    ti = kt * ntile + nt
    blk, half = divmod(ti, 2)
    kk = k % 32
    t, jj = (kk % 16) // 4, 4 * (kk // 16) + kk % 4
    lane = (n % 8) * 4 + t
    return blk * 512 + lane * 16 + half * 8 + jj


SEQ = 60
CI = G.canon_index(1024)
outs = {"q": 8, "k": 16, "v": 24, "x48": 48}
bufs = {nm: vit.ptr(SEQ, o) for nm, o in outs.items()}
p0 = vit.ptr(SEQ, 0)
w0 = bytes(vit.weights(SEQ))


def probe(n, r, col):
    """token n 的规范通道 r = 1，W[r, col] = 1 -> 各输出缓冲的非零字节"""
    w = bytearray(w0[:128]) + bytearray(3145728)
    w[128 + wpos(r, col, 3072)] = ONE
    Y = np.zeros((96, 1024), np.float32)
    Y[n, CI[r]] = 1.0
    res = vit.run(SEQ, {p0: vit.to1d(Y), **{bufs[k]: (1 << 20 if k == 'x48' else 98304) for k in bufs}}, bytes(w))
    out = {}
    for k in bufs:
        b = np.frombuffer(res[bufs[k]], np.uint8)
        out[k] = [(int(i), float(E4M3[b[i]])) for i in np.nonzero(b)[0][:4]]
    return out


if __name__ == "__main__":
    import sys
    for kind, base in (("K", 32), ("V", 64)):
        for n, d, h in [(0, 0, 0), (0, 1, 0), (0, 2, 0), (0, 4, 0), (0, 8, 0), (0, 16, 0), (1, 0, 0), (2, 0, 0), (4, 0, 0), (8, 0, 0),
                        (16, 0, 0), (32, 0, 0), (64, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 16)]:
            r = probe(n, 0, 96 * h + base + d)
            print(kind, f"token {n} d {d} head {h}:", r["k" if kind == "K" else "v"])
