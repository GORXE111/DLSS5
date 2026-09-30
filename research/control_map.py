"""pre_block 适配器里 5 路"常数"输入各是哪个控制量。
默认参数下这 5 路都等于 1 (adapter_map.json 里合并成"常数 1.000")。这里给 pre_block 参数的控制量
设成互不相同的值，逐个点亮"常数"权重位置，看适配器输出的值 (pre_probe 已旁路 swin)：
    +168 未知 f32   +172 LocalToneStrength   +176 LocalStructureStrength   +180 未知 (f16 化后入 shmem)
    +184 有效 Skin 强度 (AutoMask 关时 -1)   +188 有效 Structure 强度 (AutoMask 关时 -1)
结果写入 torch/dlss5/data/control_map.json: {权重位置: [输出通道, 控制量名]}"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import krun  # noqa: E402
import pre_probe as P  # noqa: E402
from pre_map import COLOR, HIST, MV  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AMAP = json.load(open(os.path.join(ROOT, "torch", "dlss5", "data", "adapter_map.json"), encoding="utf-8"))
CASES = {   # 名字 -> 参数设定 (偏移, 格式, 值)
    "on": [(168, "<f", 0.1875), (172, "<f", 0.3125), (176, "<f", 0.5), (180, "<f", 0.25), (184, "<f", 0.75), (188, "<f", 0.875)],
    "off": [(168, "<f", 0.1875), (172, "<f", 0.3125), (176, "<f", 0.5), (180, "<f", 0.25), (184, "<f", -1.0), (188, "<f", -1.0)],
}


def main():
    krun.TRACE = P.TRACE
    _, _, w = krun.weight_of(1)
    base = bytearray(P.bypass_weights(w, np.zeros((16, 32))))
    const = [int(e) for e, (oc, nm) in AMAP.items() if nm and nm.startswith("常数")]
    none = [int(e) for e, (oc, nm) in AMAP.items() if nm is None]
    res = {}
    for e in const + none:
        ww = bytearray(base)
        ww[8208 + 2 * e:8210 + 2 * e] = np.float16(1).tobytes()
        vals = {}
        for name, extra in CASES.items():
            out = P.run(COLOR, MV, HIST, bytes(ww), extra_param=extra)
            lit = [c for c in range(32) if np.abs(out[c]).max() > 0]
            vals[name] = (lit[0], float(np.median(out[lit[0]]))) if lit else (None, 0.0)
        res[e] = vals
        print(e, AMAP[str(e)][0], vals, flush=True)
    json.dump({str(e): v for e, v in res.items()}, open(os.path.join(os.path.dirname(__file__), "control_map_raw.json"), "w"))


if __name__ == "__main__":
    main()
