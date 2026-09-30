"""DLSS5 在真实画面上的内部行为 (torch 参考实现 + 观测钩子)。

对每张画面 (1080p，静止画面连跑 4 帧取稳态) 统计:
  gate       post_block 的历史混合门控 clamp(sigmoid(a)·0.7397) 的分布
  noise      噪声通道的影响: |out(噪声) - out(噪声=0)|
  attn       各级注意力的归一化熵 (1 = 窗口内完全均匀)、16h logit 落在截断下限 (-6) 的比例
  ffn16      16h 分组 FFN 中间激活非零比例 (0 = 该块 FFN 在此画面上完全不起作用)
结果写 behavior_out/metrics.json，图像写 behavior_out/*.npy (测试画面只在本地读取，不入库)。
"""
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "torch"))
sys.path.insert(0, HERE)
import dlss5.blocks as B  # noqa: E402
import dlss5.net as NET  # noqa: E402
from dlss5 import DLSS5  # noqa: E402
from dlss5.ops import EXP16, act, cos_norm, exp_bits, f16, inv_sum, q8, windows  # noqa: E402
from res_study import load  # noqa: E402

REC = {"attn": [], "ffn16": [], "gate": None}


def entropy_norm(P):
    """P (..., 64) 概率 -> 归一化熵 (除以 log 64)"""
    P = P.clamp_min(1e-12)
    return float((-(P * P.log()).sum(-1) / math.log(P.shape[-1])).mean())


def hook_swin():
    orig = B.Swin.attn

    def attn(self, Yf, H, W, shift):
        Y = q8(Yf[:, self.ci])
        Yw, _ = windows(Y, H, W, shift)
        nh = self.heads
        sp = lambda t: t.view(t.shape[0], 64, nh, 32).transpose(1, 2)     # noqa: E731
        q, k = sp(f16(Yw @ self.Wq)), sp(f16(Yw @ self.Wk))
        L = f16(q8(cos_norm(q) * self.tau.view(1, nh, 1, 1)) @ q8(cos_norm(k)).transpose(-1, -2) + self.bias[None])
        p = exp_bits(L, *EXP16)
        REC["attn"].append({"level": f"{self.W // 32}h", "entropy": entropy_norm(p / p.sum(-1, keepdim=True)),
                            "clamp_lo": float((L <= -6).float().mean()), "clamp_hi": float((L >= 6).float().mean())})
        return orig(self, Yf, H, W, shift)
    B.Swin.attn = attn


def hook_split16():
    orig_attn, orig_ffwd = B.Split16.attn, B.Split16.ffwd

    def attn(self, y, shift, H, W):
        Y = q8(y[:, self.ci])
        Yw, _ = windows(Y, H, W, shift)
        sp = lambda t: t.view(t.shape[0], 64, 16, 32).transpose(1, 2)     # noqa: E731
        q, k = sp(f16(Yw @ self.Wq)), sp(f16(Yw @ self.Wk))
        L = f16(q8(cos_norm(q) * self.tau.view(1, 16, 1, 1)) @ q8(cos_norm(k)).transpose(-1, -2) + self.bias[None])
        p = exp_bits(L, *EXP16)
        REC["attn"].append({"level": "16h", "entropy": entropy_norm(p / p.sum(-1, keepdim=True)),
                            "clamp_lo": float((L <= -6).float().mean()), "clamp_hi": float((L >= 6).float().mean())})
        return orig_attn(self, y, shift, H, W)

    def ffwd(self, Xf):
        Hh = q8(f16(Xf[:, self.ci] @ self.W1))
        mid = q8(act(f16(Hh @ self.Wa)))
        REC["ffn16"].append(float((mid != 0).float().mean()))
        return orig_ffwd(self, Xf)
    B.Split16.attn, B.Split16.ffwd = attn, ffwd


def hook_post():
    orig = NET.PostBlock.__call__

    def call(self, x69, skip, color, hist, mv):
        H, W = color.shape[:2]
        n = self.net(x69, skip, *NET.dims(H, W)["grid"])[:H, :W]
        REC["gate"] = (torch.sigmoid(n[..., 3]) * self.blend).clamp(0, 1)
        REC["residual"] = (8 * self.scale * n[..., :3])
        return orig(self, x69, skip, color, hist, mv)
    NET.PostBlock.__call__ = call


def run(net, img, frames=4, noise_on=True):
    orig_noise = NET.noise
    if not noise_on:
        NET.noise = lambda H, W, frame, dev: torch.zeros(3, H, W, device=dev)
    out = None
    try:
        for f in range(frames):
            REC["attn"].clear()
            REC["ffn16"].clear()
            out = net(img, hist=out, frame=f)
    finally:
        NET.noise = orig_noise
    return out


def summarize_attn(rows):
    by = {}
    for r in rows:
        by.setdefault(r["level"], []).append(r)
    return {lv: {k: float(np.mean([r[k] for r in v])) for k in ("entropy", "clamp_lo", "clamp_hi")} | {"n": len(v)}
            for lv, v in by.items()}


def main(paths):
    hook_swin()
    hook_split16()
    hook_post()
    net = DLSS5()
    outdir = os.path.join(HERE, "behavior_out")
    os.makedirs(outdir, exist_ok=True)
    res = []
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        src = load(p)
        out = run(net, src)
        gate, resid = REC["gate"].clone(), REC["residual"].clone()
        attn, ffn16 = summarize_attn(REC["attn"]), list(REC["ffn16"])
        out0 = run(net, src, noise_on=False)
        nz = (out - out0).abs().mean(-1)
        g = gate.cpu().numpy()
        row = {"image": name, "gate_mean": float(g.mean()), "gate_p10": float(np.percentile(g, 10)),
               "gate_p90": float(np.percentile(g, 90)), "gate_max_possible": 0.7397,
               "residual_mean": float(resid.abs().mean()), "noise_effect_mean": float(nz.mean()),
               "noise_effect_p99": float(torch.quantile(nz.flatten()[::7], 0.99)), "attn": attn, "ffn16_active": ffn16}
        res.append(row)
        for k, v in (("src", src), ("out", out), ("gate", gate), ("noise", nz), ("resid", resid)):
            np.save(os.path.join(outdir, f"{name}_{k}.npy"), v.cpu().numpy().astype(np.float16))
        print(f"{name}: 门控均值 {row['gate_mean']:.3f} (p10 {row['gate_p10']:.3f} p90 {row['gate_p90']:.3f})  残差 {row['residual_mean']:.4f}"
              f"  噪声影响 {row['noise_effect_mean']:.4f} (p99 {row['noise_effect_p99']:.4f})")
        print("   注意力熵: " + "  ".join(f"{lv} {a['entropy']:.3f} (截断下限 {a['clamp_lo']:.2f})" for lv, a in sorted(attn.items(), key=lambda kv: int(kv[0][:-1]))))
        print("   16h FFN 非零比例: " + " ".join(f"{x:.2f}" for x in ffn16))
    json.dump(res, open(os.path.join(outdir, "metrics.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
