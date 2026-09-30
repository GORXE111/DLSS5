"""用权重探针实测 block1 FFN 的三个下标映射 (unswizzle 的行列 -> 真实通道)。

  A. W2 列 n     -> 输出片段通道
  B. W1 行 r     -> 输入内存通道
  C. W1 列 j     -> W2 行 i (隐层单元对应)
结果写到 research/maps_block1.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import exp_block1 as B  # noqa: E402
from exp_ffn import mp_cubic_silu, q8  # noqa: E402
from probe_ffn import weights  # noqa: E402

H, W = B.H, B.W


def const_input(vals):
    x = np.zeros((32, H, W), np.float32)
    for k, v in enumerate(vals):
        x[k] = v
    return B.act_bytes(x)


def live_channels(y):
    return [c for c in range(32) if np.abs(y[c]).max() > 0]


def main():
    ones_in = const_input([1.0] * 32)
    # A. W2 列 -> 输出通道 (W1 第 0 列全 1)
    mapA = []
    for n in range(32):
        W1 = np.zeros((32, 128), np.float32); W1[:, 0] = 1
        W2 = np.zeros((128, 32), np.float32); W2[:, n] = 1
        live = live_channels(B.run(ones_in, weights(W1, W2, c1=np.zeros(32), c2=np.ones(32))))
        mapA.append(live[0] if len(live) == 1 else live)
    print("A. W2 列 -> 输出片段通道:", mapA)

    # B. W1 行 -> 输入通道: 输入通道 k 取不同常数，看激活后的值
    cand = [0.25 * (k + 1) for k in range(32)]
    cand = [float(q8(np.array([c], np.float32))[0]) for c in cand]
    xin = const_input(cand)
    act_of = {round(float(q8(mp_cubic_silu(np.array([v], np.float32)))[0]), 4): k for k, v in enumerate(cand)}
    mapB = []
    for r in range(32):
        W1 = np.zeros((32, 128), np.float32); W1[r, 0] = 1
        W2 = np.zeros((128, 32), np.float32); W2[:, 0] = 1
        y = B.run(xin, weights(W1, W2, c1=np.zeros(32), c2=np.ones(32)))
        ch = live_channels(y)
        val = round(float(y[ch[0]].reshape(-1)[0]), 4) if ch else None
        mapB.append(act_of.get(val, val))
    print("B. W1 行 -> 输入内存通道:", mapB)

    # C. W1 列 j -> W2 行 i: W2 行 i 在列 (i % 32) 与 (i // 32) 两次编码
    mapC = []
    for j in range(128):
        W1 = np.zeros((32, 128), np.float32); W1[:, j] = 1
        res = []
        for code in (lambda i: i % 32, lambda i: i // 32):
            W2 = np.zeros((128, 32), np.float32)
            for i in range(128):
                W2[i, code(i)] = 1
            live = live_channels(B.run(ones_in, weights(W1, W2, c1=np.zeros(32), c2=np.ones(32))))
            res.append(mapA.index(live[0]) if len(live) == 1 else None)
        mapC.append(None if None in res else res[1] * 32 + res[0])
    print("C. W1 列 -> W2 行:", mapC)
    json.dump({"W2col_to_outfrag": mapA, "W1row_to_inmem": mapB, "W1col_to_W2row": mapC},
              open(os.path.join(os.path.dirname(__file__), "maps_block1.json"), "w"))


if __name__ == "__main__":
    main()
