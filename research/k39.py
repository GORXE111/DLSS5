"""block39 dec_input_upsample (1024 -> 512，8x12 -> 12x20) 单独运行与解析"""
import os
import numpy as np
import k16
import krun
import vit
from act import chw_to_tin, tin_to_chw

T = os.path.join(os.path.dirname(__file__), "tapsdec")
k16.TAPS, k16.TRACE = T, os.path.join(T, "nr-trace.tsv")
vit.TAPS, vit.TRACE = k16.TAPS, k16.TRACE
krun.TRACE = k16.TRACE
SEQ = 99


def tap(s, a):
    return open(os.path.join(T, f"tap_s{s}_{a}.bin"), "rb").read()


def run(low, skip, w=None):
    """low (1024, 8, 12)，skip (512, 12, 20) 片段序 -> 输出 (512, 12, 20)"""
    p0, p8, p16 = (vit.ptr(SEQ, o) for o in (0, 8, 16))
    r = vit.run(SEQ, {p0: chw_to_tin(low), p8: chw_to_tin(skip), p16: 122880}, w)
    return tin_to_chw(np.frombuffer(r[p16], np.uint8), 512, 12, 20)


def ref(low, skip, w):
    """out = q8(s[COLS]⊙skip + nn_up2x(low规范序 · W_up)[:, COLS])。W_up 1024x512 [0,524288)，s f16x512 在末尾 (实测)"""
    import block1_ref as R
    import swin_gen as G
    from exp_ffn import q8, unswizzle
    w = bytes(w)
    COLS = np.array(G.group_map(G.COLS32, 512))
    s = np.frombuffer(w[524288:], np.float16).astype(np.float32)
    L = low[:, :6, :10].reshape(1024, -1).T[:, G.canon_index(1024)]
    U = R.f16(L @ unswizzle(w[:524288], 1024, 512))[:, COLS].T.reshape(512, 6, 10)
    up = U.repeat(2, 1).repeat(2, 2)
    return q8(R.f16(s[COLS][:, None, None] * skip + up))
