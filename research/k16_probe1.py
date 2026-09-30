"""W1 单字节探针: 只留一个 fp8=1 -> 输出 ∝ q8(act(x_i)) -> 由指纹定输入通道 i"""
import numpy as np
import k16_ffwd as F
from act import E4M3, random_fp8
from exp_ffn import mp_cubic_silu, q8

ONE = int(np.nanargmin(np.abs(E4M3 - 1.0)))


def probe(w, X, offsets):
    Xt = X.reshape(512, -1).T
    act = q8(mp_cubic_silu(Xt))
    res = {}
    for off in offsets:
        ww = bytearray(w)
        ww[0:262144] = bytes(262144)
        ww[off] = ONE
        Y = F.run(X, ww).reshape(512, -1).T
        n = int(np.argmax(np.abs(Y).sum(0)))
        y = Y[:, n]
        if np.abs(y).max() == 0:
            res[off] = (None, n, 0)
            continue
        c = np.nan_to_num([np.corrcoef(y, act[:, i])[0, 1] for i in range(512)])
        i = int(np.argmax(np.abs(c)))
        res[off] = (i, n, round(float(c[i]), 4))
    return res


if __name__ == "__main__":
    w = F.weights()
    X = random_fp8((512, 12, 20), lo=-3, hi=3, seed=7)
    offs = list(range(0, 16)) + list(range(16, 64, 4)) + [512, 528] + [kc * 1024 for kc in range(1, 16)] + \
        [16384, 16384 * 2, 16384 * 8, 131072]
    for off, v in probe(w, X, offs).items():
        print(off, "-> 输入通道", v[0], " (最亮输出", v[1], " 相关", v[2], ")")
