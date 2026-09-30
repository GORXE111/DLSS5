"""端到端: nr-lab 第 0 帧 (重置帧) 的合成输入 -> Python 整网 -> 与 nr-lab 实际输出 (out_f0.ppm) 比较"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import net_ref as N  # noqa: E402

HERE = os.path.dirname(__file__)
t0 = time.time()
color = N.color_pattern()
out, mid = N.run_frame(color, None, None, 0, os.path.join(HERE, "trace_f0.tsv"), log=lambda s: None)
raw = open(os.path.join(HERE, "out_f0.ppm"), "rb").read()
ref = np.frombuffer(raw[-640 * 360 * 3:], np.uint8).reshape(360, 640, 3).astype(np.float32) / 255
o8 = np.floor(out * 255 + 0.5) / 255
print(f"用时 {time.time() - t0:.0f}s")
print("Python 整网 vs nr-lab 输出: 相关 %.5f  平均绝对差 %.4f (%.2f/255)  8 位完全一致 %.4f  差<=1/255 %.4f" % (
    np.corrcoef(o8.ravel(), ref.ravel())[0, 1], np.abs(o8 - ref).mean(), np.abs(o8 - ref).mean() * 255,
    (o8 == ref).mean(), (np.abs(o8 - ref) <= 1.01 / 255).mean()))
d, c = ref - color, o8 - color
print("网络改动量 (输出-输入) 的相关: %.5f   kernel 改动幅度 %.4f  参考改动幅度 %.4f" % (
    np.corrcoef(d.ravel(), c.ravel())[0, 1], np.abs(d).mean(), np.abs(c).mean()))
np.save(os.path.join(HERE, "net_frame0_out.npy"), out)
