"""block1 (swin_1h_32 inpview) 控制实验: 感受野 + 权重区段消融。"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import klab  # noqa: E402
from act import E4M3, TAPS, tin_to_chw  # noqa: E402

ROOT = klab.ROOT
H, W, C = 192, 320, 32
row = klab.trace_row(2)
fn = klab.function("01", row["kernel"])
rec = [r for r in json.load(open(os.path.join(ROOT, "weights_map.json"))) if r["name"] == "block1.layer0.layer"][0]
with open(os.path.join(ROOT, "WEIGHTS_HT.bin"), "rb") as f:
    f.seek(rec["off"])
    W_REAL = f.read(rec["C"])
INP_REAL = open(os.path.join(TAPS, "tap_s1_1c19b000.bin"), "rb").read()
E4M3_ENC = {float(v): i for i, v in enumerate(E4M3) if np.isfinite(v)}


def run(inp: bytes, w: bytes):
    g_in, g_out = klab.gpu_bytes(inp), klab.gpu_bytes(b"", H * W * C)
    g_w, g_cnt = klab.gpu_bytes(w, 512), klab.gpu_bytes(b"", 1 << 16)
    p = klab.patch_ptrs(row["params"], {0x1c19b000: g_in.data_ptr(), 0x1c37b000: g_out.data_ptr(),
                                        0x9e05600: g_w.data_ptr(), 0x1ba00000: g_cnt.data_ptr()})
    klab.launch(fn, row["grid"], row["block"], row["smem"], p)
    return tin_to_chw(g_out.cpu().numpy(), C, H, W)


def load_input():
    """block1 的真实输入 ([2][H][W][16] FP8) -> (C,H,W)"""
    v = E4M3[np.frombuffer(INP_REAL, np.uint8)]
    return v.reshape(2, H, W, 16).transpose(0, 3, 1, 2).reshape(C, H, W)


def act_bytes(x):
    """(C,H,W) float -> 输入布局 [2][H][W][16] 的 FP8 字节 (取最近的 e4m3，只用于构造实验输入)"""
    vals = np.array(sorted(E4M3_ENC))
    idx = np.clip(np.searchsorted(vals, x), 1, len(vals) - 1)
    near = np.where(np.abs(vals[idx - 1] - x) <= np.abs(vals[idx] - x), vals[idx - 1], vals[idx])
    codes = np.vectorize(E4M3_ENC.get)(near).astype(np.uint8)
    return codes.reshape(2, 16, H, W).transpose(0, 2, 3, 1).tobytes()


def main():
    base = run(INP_REAL, W_REAL)
    zero_out = run(bytes(len(INP_REAL)), W_REAL)
    print(f"输入全零 -> 输出: 最大 |y| {np.abs(zero_out).max():.4f} (非零说明有偏置项)")

    # 感受野: 零输入 + 单点冲激 (像素 (100,160)，全部 32 通道 = 1.0)
    for py, px in ((100, 160), (100, 164)):
        x = np.zeros((C, H, W), np.float32)
        x[:, py, px] = 1.0
        d = np.abs(run(act_bytes(x), W_REAL) - zero_out).max(0)
        ys, xs = np.nonzero(d > 1e-6)
        print(f"冲激 @({py},{px}): 受影响像素 {len(ys)} 个, y {ys.min()}..{ys.max()}, x {xs.min()}..{xs.max()}")

    # 权重区段消融 (字节区间来自 PTX 阶段时间线)
    regions = [(0, 8192, "R0 前段反复 mma 用"), (8192, 8288, "R1 小向量"), (8288, 11360, "R2"),
               (11360, 19552, "R3 注意力阶段 (两半各 x8)"), (19552, 20672, "R4 归一化前/输出投影")]
    for a, b, name in regions:
        w = bytearray(W_REAL)
        w[a:b] = bytes(b - a)
        y = run(INP_REAL, bytes(w))
        rel = np.abs(y - base).mean() / np.abs(base).mean()
        print(f"清零 {name:22s} [{a:5d},{b:5d}) {((b - a) // 2):5d} 个 f16 -> 输出平均变化 {rel:.3f}")


if __name__ == "__main__":
    main()
