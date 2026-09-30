"""窃听数据的读取工具: FP8 激活按 [组][H][W][16ch] 布局还原成 (C, H, W) float32。"""
import json
import os

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TAPS = os.path.join(ROOT, "research", "taps")
E4M3 = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float().numpy()


def load_act(name, C, H, W, group=16):
    raw = np.frombuffer(open(os.path.join(TAPS, name), "rb").read(), np.uint8)
    G = C // group
    v = E4M3[raw[: G * H * W * group]]
    return v.reshape(G, H, W, group).transpose(0, 3, 1, 2).reshape(C, H, W)


def tin_to_chw(raw_u8, C, H, W):
    """tinlayout (swin kernel 的输出，由 PTX 写出地址反推 + 数据验证):
    4x4 像素小块按行排 (每行 W/4 块)；每块 = C/32 组 x 32 线程 x 16 字节 (C=32 时一组)；
    线程 L: g=L>>2, t=L&3；第 b 字节 (n=b%4, r=(b>>2)%2, c=b>>3):
        块内像素 g+8r (行优先 4x4)，通道 8n+2t+c —— 即 mma m16n8 累加器片段原样写出"""
    G = C // 32                                   # 通道组: 每组 32 通道 = 512 字节 (C=64 已由 PTX 写址 +512 证实)
    v = E4M3[raw_u8[: (H // 4) * (W // 4) * 512 * G]].reshape(H // 4, W // 4, G, 32, 16)
    out = np.zeros((C, H, W), np.float32)
    for gi in range(G):
        for lane in range(32):
            g, t = lane >> 2, lane & 3
            for b in range(16):
                n, r, c = b % 4, (b >> 2) % 2, b >> 3
                tok, ch = g + 8 * r, 8 * n + 2 * t + c
                out[gi * 32 + ch, tok // 4 :: 4, tok % 4 :: 4] = v[:, :, gi, lane, b]
    return out


E4M3_CODE = {float(v): i for i, v in enumerate(E4M3) if np.isfinite(v)}


def chw_to_tin(x):
    """(C,H,W) 值 (须为 e4m3 可表示) -> tinlayout 字节，tin_to_chw 的逆"""
    C, H, W = x.shape
    G = C // 32
    raw = np.zeros((H // 4, W // 4, G, 32, 16), np.uint8)
    code = np.vectorize(lambda v: E4M3_CODE[float(v)])
    for gi in range(G):
        for lane in range(32):
            g, t = lane >> 2, lane & 3
            for b in range(16):
                n, r, c = b % 4, (b >> 2) % 2, b >> 3
                tok, ch = g + 8 * r, 8 * n + 2 * t + c
                raw[:, :, gi, lane, b] = code(x[gi * 32 + ch, tok // 4 :: 4, tok % 4 :: 4])
    return raw.tobytes()


def random_fp8(shape, lo=-4, hi=4, seed=0):
    vals = np.array(sorted(v for v in E4M3_CODE if lo <= v <= hi), np.float32)
    return np.random.default_rng(seed).choice(vals, shape).astype(np.float32)


def load_tin(name, C, H, W):
    return tin_to_chw(np.frombuffer(open(os.path.join(TAPS, name), "rb").read(), np.uint8), C, H, W)


def weight_record(name):
    """返回某条权重记录的 f16 数组 (与显存内容逐字节一致: 文件 off 起 C 字节)"""
    rec = [r for r in json.load(open(os.path.join(ROOT, "weights_map.json"))) if r["name"] == name][0]
    with open(os.path.join(ROOT, "WEIGHTS_HT.bin"), "rb") as f:
        f.seek(rec["off"])
        return np.frombuffer(f.read(rec["C"]), np.float16).astype(np.float32)
