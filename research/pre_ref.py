"""pre_block (block0) 完整参考 + 探针工具。

kernel 输出两份: +248 = 半分辨率图像 [2][192][320][16] (进 block1)；+216 = 全分辨率适配器输出 tin (32, 384, 640) (post_block 的 skip)。
"""
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import klab  # noqa: E402
import krun  # noqa: E402
import pre_probe as P  # noqa: E402
from act import E4M3, tin_to_chw  # noqa: E402

AD0, AD1 = 8208, 9232                     # 适配器 512 个 f16
MAP = json.load(open(os.path.join(os.path.dirname(__file__), "adapter_map.json")))


def run_raw(color, mv, hist, w, frame=None):
    """返回 (图像输出 (32,192,320), 全分辨率适配器输出 (32,384,640))。w 为完整 block0 记录字节"""
    krun.TRACE = P.TRACE
    row = klab.trace_row(1, P.TRACE)
    p = bytearray(row["params"])
    texs = [klab.texture(color), klab.texture(hist) if hist is not None else (0, None),
            klab.texture(mv, linear=False) if mv is not None else (0, None)]
    for off, (h, _) in zip((0, 8, 16), texs):
        struct.pack_into("<Q", p, off, h)                  # 句柄 0 = 未提供 (第 0 帧/重置时历史与运动矢量都是 0)
    if frame is not None:
        struct.pack_into("<I", p, 200, frame)
    fn = klab.function("01", row["kernel"])
    out = klab.gpu_bytes(b"", 192 * 320 * 32)
    wbuf = klab.gpu_bytes(w, 512)
    skip = klab.gpu_bytes(b"", 32 * 384 * 640 + (1 << 20))
    cnt = klab.gpu_bytes(b"", 1 << 20)
    q = lambda o: struct.unpack_from("<Q", p, o)[0]  # noqa: E731
    m = {q(248): out.data_ptr(), q(224): wbuf.data_ptr(), q(216): skip.data_ptr()}
    for i in range(0, len(p) // 8 * 8, 8):
        v = struct.unpack_from("<Q", p, i)[0]
        if 0x1BA00000 <= v < 0x1BB00000 and v not in m:
            m[v] = cnt.data_ptr() + (v - 0x1BA00000)
    klab.launch(fn, row["grid"], row["block"], row["smem"], klab.patch_ptrs(bytes(p), m))
    img = E4M3[out.cpu().numpy()].reshape(2, 192, 320, 16).transpose(0, 3, 1, 2).reshape(32, 192, 320)
    sk = tin_to_chw(skip.cpu().numpy()[:32 * 384 * 640], 32, 384, 640)
    return img, sk


def bypass(w, adapter):
    """swin 主体短路 (FFN 收缩 0、c1=1、投影 0、c2=1)，适配器换成给定的 512 个 f16"""
    w = bytearray(w)
    w[4096:8192] = bytes(4096)
    w[AD0:AD1] = np.asarray(adapter, np.float16).tobytes()
    w[9232:9296] = np.ones(32, np.float16).tobytes()
    w[20592:21616] = bytes(1024)
    w[21616:21680] = np.ones(32, np.float16).tobytes()
    return bytes(w)


def elements(label):
    """adapter_map 里某标签的元素: {输出通道: [元素下标...]}"""
    d = {}
    for e, (oc, nm) in MAP.items():
        if nm == label:
            d.setdefault(oc, []).append(int(e))
    return d


INPUTS = ["颜色R", "颜色G", "颜色B", "历史R", "历史G", "历史B", "噪声0", "噪声1", "噪声2", "常数"]
_LAB = {"颜色R": "颜色R", "颜色B": "颜色B", "历史G": "历史G", "历史B": "历史B",
        "噪声/其他 (std 0.500)": "噪声0", "噪声/其他 (std 0.501)": "噪声1", "噪声/其他 (std 0.498)": "噪声2", "常数 1.000": "常数"}


def adapter_matrix(w):
    """(10 输入, 32 输出)。'历史R' 标签的元素每个输出通道有两个: 下标小的是颜色 G，大的是历史 R (探针实测)；5 个常数 1 输入合并"""
    v = np.frombuffer(w[AD0:AD1], np.float16).astype(np.float32)
    A = np.zeros((10, 32), np.float32)
    pairs = {}
    for e, (oc, nm) in MAP.items():
        e = int(e)
        if nm is None:
            continue
        if nm == "历史R":
            pairs.setdefault(oc, []).append(e)
            continue
        A[INPUTS.index(_LAB[nm]), oc] += v[e]
    for oc, es in pairs.items():
        es = sorted(es)
        A[INPUTS.index("颜色G"), oc] += v[es[0]]
        A[INPUTS.index("历史R"), oc] += v[es[1]]
    return A


def sample(img, u, v):
    """双线性 + clamp，归一化坐标 (与 CUDA 纹理同: 坐标 *size - 0.5，8 位小数权重忽略)"""
    Hh, Ww = img.shape[:2]
    xx = np.clip(u * Ww - 0.5, 0, Ww - 1)
    yy = np.clip(v * Hh - 0.5, 0, Hh - 1)
    x0, y0 = np.floor(xx).astype(int), np.floor(yy).astype(int)
    x1, y1 = np.minimum(x0 + 1, Ww - 1), np.minimum(y0 + 1, Hh - 1)
    fx, fy = (xx - x0)[..., None], (yy - y0)[..., None]
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy)
            + img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)


def inputs(color, hist, mv, frame, H=384, W=640):
    """每个全分辨率像素 (H x W 补齐网格) 的 10 路输入 (N, 10)"""
    import noise
    import post_probe as PP
    Hi, Wi = color.shape[:2]
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    y = np.where(y >= Hi, 2 * (Hi - 1) - y, y)          # 补齐行 = 以末行为轴的镜像 (不重复末行，实测 360->358)
    u, v = (x + 0.5) / Wi, (y + 0.5) / Hi
    c = sample(color, u, v)[..., :3]
    mvs = sample(mv, u, v) if False else mv[np.clip(y.astype(int), 0, Hi - 1), np.clip(x.astype(int), 0, Wi - 1)]
    h = PP.catmull_rom5(hist[..., :3], u + mvs[..., 0] / Wi, v + mvs[..., 1] / Hi)
    nz = noise.noise(H, W, frame)
    X = np.concatenate([(c - 0.5) * 0.125, (h - 0.5) * 0.125, nz.transpose(1, 2, 0), np.ones((H, W, 1))], -1)
    return X.reshape(H * W, 10).astype(np.float32)
