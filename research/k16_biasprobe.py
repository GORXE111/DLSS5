"""16h 注意力偏置排布探针: Q=K=0，头 0 偏置只有一个 f16 = +12 -> 该查询 token 的输出 ≈ 被关注键 token 的 V"""
import numpy as np
import k16_attn as A
import swin_gen as G
from probe_ffn import swizzle
from exp_ffn import unswizzle, q8
from act import random_fp8
import block1_ref as R

SEQ = 26
w = A.weights(SEQ)
Y = random_fp8((512, 12, 20), lo=-2, hi=2, seed=21)
cols = np.array(G.group_map(G.COLS32, 512))
M = unswizzle(bytes(w[:786432]), 512, 1536)
for h in range(16):
    M[:, 96 * h:96 * h + 64] = 0
base = bytearray(w)
base[:786432] = swizzle(M)
base[786432:917504] = bytes(131072)
Yc = q8(Y.reshape(512, -1).T[:, G.canon_index(512)])
V0 = R.f16(Yc @ M[:, 64:96])                           # 头 0 的 V (240, 32)，头内顺序同参考
k0 = A.run(Y, bytes(base), SEQ).reshape(512, -1).T[:, :32]


def probe(e):
    ww = bytearray(base)
    ww[786432 + 2 * e:786432 + 2 * e + 2] = np.float16(12).tobytes()
    k = A.run(Y, bytes(ww), SEQ).reshape(512, -1).T[:, :32]
    changed = np.nonzero(np.abs(k - k0).max(1) > 0)[0]
    out = []
    for tq in changed:
        v = k[tq][np.argsort(cols[:32])] if False else k[tq]
        c = [np.corrcoef(v, q8(V0[tk])[cols[:32]])[0, 1] for tk in range(240)]
        tk = int(np.nanargmax(c))
        out.append((int(tq), tk, round(float(np.nanmax(c)), 3)))
    return out


if __name__ == "__main__":
    for e in [0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
        r = probe(e)
        print(e, [(f"q({t // 20},{t % 20})", f"k({s // 20},{s % 20})", c) for t, s, c in r][:3], len(r))
