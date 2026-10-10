"""DLSS5 整网 (GPU)。按 data/schedule.json 的 71 步执行: pre_block -> U-Net 编码 1h..8h -> 16h 分组 swin -> ViT-1d ->
解码 16h..1h -> post_block (时域合成)。整网在补齐网格上运行: 宽高各向上取到 64 的倍数 (640x360 -> 640x384)，
16h 与 ViT 两级再各自补到 4 的倍数 (见 dims())。支持任意输入尺寸。

    net = DLSS5()                                   # 读 WEIGHTS_HT.bin，解码全部权重到 GPU
    out = net(color, hist=None, mv=None, frame=0)   # color/hist: (360, 640, 3) [0,1]；mv: (360, 640, 2) 像素；返回 (360, 640, 3)
"""
import json
import os

import numpy as np
import torch

from . import layout as Lay
from .blocks import Block39, FinalHead, Split16, Swin, SwinDown, SwinUp, ViT, _Base
from . import ops
from . import style as style_mod
from .ops import act, bilinear, catmull_rom5, f16, noise, q8, unwindows, windows
from .weights import Records

LEVEL = {"1h": (32, 192, 320), "2h": (64, 96, 160), "4h": (128, 48, 80), "8h": (256, 24, 40)}   # 通道数 (及 640x360 时的尺寸)


def _up(v, m):
    return (v + m - 1) // m * m


def _n_down(x):
    """CCNetwork::SetResolution (sub_18003C580) 对一维尺寸统计的"变小的层"数。逐层形状推导:
    6 个 _ds 层 (pre, 1h, 2h, 4h, 8h, 16h 池化) 各输出 ceil4(ceil(x/2))；2h 的 _outview 层输出计算图记录的
    形状 ceil8(x)/4 (图在输入端 constant_pad_nd 到 8 的倍数)，它比链上的值小时再计一次。"""
    n, y, g2 = 0, x, _up(x, 8) // 4
    for k in range(1, 7):
        z = _up((y + 1) // 2, 4)
        n, y = n + (z < y), z
        if k == 2:
            n, y = n + (g2 < y), g2
    return n


def grid_for(H, W):
    """DLL 为输入 (H, W) 选的补齐网格 (sub_18003C580 反汇编，90 个实测尺寸全部吻合):
    m = 2^下采样层数 (通常 64，2h 形状取整错位时 128)，各维向上取到 m 的倍数、至少 320；
    两维都恰为 4m 的倍数时宽再加 m。"""
    mH, mW = 1 << _n_down(H), 1 << _n_down(W)
    GH, GW = max(_up(H, mH), 320), max(_up(W, mW), 320)
    if GH % (4 * mH) == 0 and GW % (4 * mW) == 0:
        GW += mW
    return GH, GW


def dims(H, W, grid=None):
    """输入 H x W -> 各级网格。grid: 全分辨率补齐网格 (默认按 DLL 规则 grid_for)；
    1h..8h = grid / 2..16；16h: 实际 grid/32，补到 4 的倍数；vit: 16h 补齐网格 /2 再补到 4 的倍数"""
    GH, GW = grid or grid_for(H, W)
    d = {"grid": (GH, GW), "1h": (GH // 2, GW // 2), "2h": (GH // 4, GW // 4), "4h": (GH // 8, GW // 8),
         "8h": (GH // 16, GW // 16), "16h_real": (GH // 32, GW // 32)}
    d["16h"] = (_up(GH // 32, 4), _up(GW // 32, 4))
    d["vit"] = (_up(d["16h"][0] // 2, 4), _up(d["16h"][1] // 2, 4))
    return d


def pad_rows_cols(x, src, dst):
    """(src_h*src_w, C) -> (dst_h*dst_w, C)，多出的区域补 0，多余的裁掉"""
    C = x.shape[1]
    out = x.new_zeros(dst[0], dst[1], C)
    h, w = min(src[0], dst[0]), min(src[1], dst[1])
    out[:h, :w] = x.view(src[0], src[1], C)[:h, :w]
    return out.view(-1, C)


def control_inputs(tone=1.0, structure=1.0, skin=-1.0, auto_mask=True, style=0):
    """DLSSNR 参数 -> pre_block 的 5 路控制输入 (每像素相同)。默认值即 DLL 默认值。
    tone/structure/skin = DLSSNR.LocalToneStrength / LocalStructureStrength / SkinStructureStrength (skin<0: 跟随 structure)，
    auto_mask = DLSSNR.UseAutoMask (提供 ControlMask 时 DLL 强制为 0)，style = DLSSNR.Style (0..2)。
    DLL (sub_18001A700) 写 pre 参数 +172 tone, +176 structure, +180 style/128, +184 有效 skin, +188 有效 structure
    (auto_mask 关时后两者为 -1)；kernel 再据此生成下面 5 路。"""
    skin_eff = (skin if skin >= 0 else structure) if auto_mask else -1.0
    struct_eff = structure if auto_mask else -1.0
    on = max(skin_eff, struct_eff) >= 0
    return {"LocalTone": tone, "StructureGate": 1.0 if on else structure,
            "Skin": (skin_eff if skin_eff >= 0 else structure) if on else -1.0,
            "Structure": (struct_eff if struct_eff >= 0 else structure) if on else -1.0,
            "Style": min(max(int(style), 0), 2) / 128, "style": min(max(int(style), 0), 2)}


# ================================================================== pre_block (block0)
class PreBlock(_Base):
    """每个全分辨率像素 16 路输入 -> 适配器 16->32 -> 全分辨率 1h swin (skip, 供 post_block) -> 2x2 平均 (进 block1)。
    输入: 颜色 RGB 与重投影历史 RGB 各 (c-0.5)*0.125，3 路噪声，常数 1，5 路控制量 (control_inputs)，1 路恒 0。
    补齐行/列按不重复边缘的镜像取样 (360p: y' = 718 - y)。历史缺失 (重置帧) 时以颜色充当历史。"""

    CONTROLS = ["LocalTone", "StructureGate", "Skin", "Structure", "Style"]
    INPUTS = ["颜色R", "颜色G", "颜色B", "历史R", "历史G", "历史B", "噪声0", "噪声1", "噪声2", "常数"] + CONTROLS
    _LAB = {"颜色R": "颜色R", "颜色B": "颜色B", "历史G": "历史G", "历史B": "历史B",
            "噪声/其他 (std 0.500)": "噪声0", "噪声/其他 (std 0.501)": "噪声1", "噪声/其他 (std 0.498)": "噪声2"}

    def __init__(self, w, device):
        super().__init__(device)
        amap = json.load(open(os.path.join(Lay.DATA, "adapter_map.json"), encoding="utf-8"))
        cmap = json.load(open(os.path.join(Lay.DATA, "control_map.json"), encoding="utf-8"))   # research/control_map.py
        v = Lay.f16vec(w[8208:9232])
        A = np.zeros((len(self.INPUTS), 32), np.float32)
        for e, (oc, nm) in cmap.items():                     # 常数与 5 路控制量 (默认参数下它们都是 1，adapter_map 分不开)
            A[self.INPUTS.index(nm), oc] += v[int(e)]
        pairs = {}
        for e, (oc, nm) in amap.items():
            e = int(e)
            if nm is None or nm.startswith("常数"):
                continue
            if nm == "历史R":                                  # 该标签混了颜色 G: 每个输出通道两个元素，下标小者 = 颜色 G
                pairs.setdefault(oc, []).append(e)
            else:
                A[self.INPUTS.index(self._LAB[nm]), oc] += v[e]
        for oc, es in pairs.items():
            es = sorted(es)
            A[1, oc] += v[es[0]]
            A[3, oc] += v[es[1]]
        self.A = self.T(A)                                     # 输出为规范序
        self.fi = self.I(np.argsort(Lay.canon_index(32)))
        self.swin = Swin(w[:8208] + w[9232:], 32, device)      # 去掉适配器 = 标准 1h 记录
        self._kc = {}

    def inputs(self, color, hist, mv, frame, ctrl=None):
        Hi, Wi = color.shape[:2]
        GH, GW = dims(Hi, Wi)["grid"]
        y, x = torch.meshgrid(torch.arange(GH, device=self.dev, dtype=torch.float32),
                              torch.arange(GW, device=self.dev, dtype=torch.float32), indexing="ij")
        y = torch.where(y >= Hi, 2 * (Hi - 1) - y, y)
        x = torch.where(x >= Wi, 2 * (Wi - 1) - x, x)
        u, v = (x + 0.5) / Wi, (y + 0.5) / Hi
        c = bilinear(color, u, v)[..., :3]
        mvs = mv[y.long().clamp(0, Hi - 1), x.long().clamp(0, Wi - 1)]
        h = catmull_rom5(hist[..., :3], u + mvs[..., 0] / Wi, v + mvs[..., 1] / Hi)
        nz = noise(GH, GW, frame, self.dev).permute(1, 2, 0)
        key = tuple((ctrl or control_inputs())[n] for n in self.CONTROLS)
        if key not in self._kc:                              # 缓存: CUDA Graph 捕获期间不能做主机->显存拷贝
            self._kc[key] = torch.tensor((1.0,) + key, device=self.dev)
        k = self._kc[key].expand(GH, GW, 1 + len(self.CONTROLS))
        mask = (ctrl or {}).get("mask")
        if mask is not None:                                 # DLSSNR.ControlMask: G、B 逐像素乘到 LocalTone、StructureGate
            m = mask[y.long(), x.long()]
            k = torch.cat([k[..., :1], k[..., 1:2] * m[..., 1:2], k[..., 2:3] * m[..., 2:3], k[..., 3:]], -1)
        return torch.cat([(c - 0.5) * 0.125, (h - 0.5) * 0.125, nz, k], -1).view(-1, len(self.INPUTS))

    def __call__(self, color, hist, mv, frame, ctrl=None):
        GH, GW = dims(*color.shape[:2])["grid"]
        a = q8(f16(f16(self.inputs(color, hist, mv, frame, ctrl)) @ self.A))[:, self.fi]   # 适配器是 f16 mma: 输入先舍入到 f16
        y0 = self.swin(a, GH, GW, (0, 0))                        # skip (全分辨率 tin 片段序)
        img = q8(f16(y0[:, self.swin.ci].view(GH // 2, 2, GW // 2, 2, 32).mean((1, 3)).reshape(-1, 32)))
        return img, y0


# ================================================================== post_block (block70)
class PostBlock(_Base):
    """m = s1⊙nn_up2x(block69) + s2⊙skip(pre) -> 全分辨率 1h swin -> f16 输出卷积 32->4 (RGB 残差 + 门控 logit)
    -> cur = clamp(color + 0.25·rgb)，gate = clamp(sigmoid(a)·blend)，out = cur + gate·(CatmullRom(hist) - cur)"""

    MX = [8 * (c % 4) + 2 * ((c % 16) // 4) + c // 16 for c in range(32)]    # 图像通道 c -> 片段位置

    def __init__(self, w, blend_raw, device, scale=0.03125):
        super().__init__(device)
        SG = Lay.COLS32
        f = lambda a, b: Lay.f16vec(w[a:b])                        # noqa: E731
        self.s1, self.s2 = self.T(f(8272, 8336)[SG]), self.T(f(8336, 8400)[SG])
        self.mxi = self.I(np.argsort(self.MX))
        self.swin = Swin(w[:8272] + bytes(16) + w[8400:20784], 32, device)   # 插入 s1/s2 之外 = 标准 1h 记录
        Wo = f(20784, 21808)
        self.Wo = self.T([[Wo[256 * (j % 2) + 8 * ((j % 8) // 2) + j // 8 + 32 * o] for o in range(4)] for j in range(32)])
        self.blend = float(Lay.f16vec(blend_raw)[0])
        self.scale = scale                                          # kernel 参数 +48 (f32)

    def net(self, x69, skip, GH, GW):
        b = self.swin
        h, w = GH // 2, GW // 2
        Xf = x69.view(h, 1, w, 1, 32).expand(h, 2, w, 2, 32).reshape(-1, 32)[:, self.mxi]
        M = f16(self.s1 * Xf + self.s2 * skip)
        ffn = lambda m: q8(act(f16(q8(m[:, b.ci]) @ b.W1))) @ b.W2     # noqa: E731  post 的 FFN 吃 q8(m)
        Y = f16(b.c1 * M + torch.cat([ffn(m) for m in M.split(b.CHUNK)]))   # 分块: 全分辨率中间张量很大
        Yw, meta = windows(q8(Y[:, b.ci]), GH, GW, (-4, -4))
        o = torch.cat([b._window_attn(y) for y in Yw.split(b.CHUNK // 64)])
        y = f16(b.c2 * Y + q8(unwindows(o, meta, GH, GW)) @ b.Wp)   # 1h 规则: O 量化、残差用 f16；y 不量化直接进 f16 输出卷积
        return f16(y @ self.Wo).view(GH, GW, 4)

    def __call__(self, x69, skip, color, hist, mv):
        H, W = color.shape[:2]
        n = self.net(x69, skip, *dims(H, W)["grid"])[:H, :W]
        cur = (color + 8 * self.scale * n[..., :3]).clamp(0, 1)
        if hist is None:                                            # 重置帧: 无历史混合
            return cur
        gate = (torch.sigmoid(n[..., 3:]) * self.blend).clamp(0, 1)
        y, x = torch.meshgrid(torch.arange(H, device=self.dev, dtype=torch.float32),
                              torch.arange(W, device=self.dev, dtype=torch.float32), indexing="ij")
        h = catmull_rom5(hist[..., :3], (x + 0.5 + mv[..., 0]) / W, (y + 0.5 + mv[..., 1]) / H)
        return cur + gate * (h - cur)


# ================================================================== 整网
class DLSS5:
    def __init__(self, device="cuda", records=None, schedule=None, precise=False, half=None):
        """precise=True: 逐处模拟 kernel 的 f16/fp8 舍入、f32 计算 (逐块对照用，慢)；
        False (默认): 不模拟舍入、半精度计算 (autocast)。整帧与 kernel 的吻合度两者相同 (见 bench_modes.py)。
        half: 是否用半精度 (默认 = not precise)"""
        torch.backends.cuda.matmul.allow_tf32 = False             # 与参考一致的 f32 matmul
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False   # 半精度 matmul 用 f32 累加
        self.dev = device
        self.precise = precise
        self.half = (not precise) if half is None else half
        R = records or Records()
        sched = schedule or json.load(open(os.path.join(Lay.DATA, "schedule.json"), encoding="utf-8"))["steps"]
        self.steps = []
        for st in sched:
            k = st["kind"]
            if k == "pre":
                m = PreBlock(R[st["record"]], device)
            elif k == "post":
                m = PostBlock(R[st["record"]], R[st["blend_record"]], device)
            elif k == "swin":
                W = LEVEL[st["level"]][0]
                cls = {"ds": SwinDown, "up": SwinUp}.get(st["variant"], Swin)
                m = cls(R[st["record"]], W, device)
            elif k == "split16":
                m = Split16([R[n] for n in st["records"]], device)
                if st["tail"] == "pool":
                    st = dict(st, head=FinalHead(R[st["head_record"]], device))
            elif k == "vit":
                m = ViT([R[n] for n in st["records"]], device)
            elif k == "block39":
                m = Block39(R[st["record"]], device)
            self.steps.append((st, m))

    @torch.no_grad()
    def __call__(self, color, hist=None, mv=None, frame=0, trace=None, controls=None, intensity=1.0, control_mask=None):
        """color/hist: (H, W, 3) float [0,1] (numpy 或 torch)，输出 f32；mv: (H, W, 2) 像素位移；hist=None 表示重置帧。
        controls: control_inputs(...) 的结果 (默认 = DLL 默认参数)。
        intensity: DLSSNR.Intensity，夹到 [0,1] 后在网络之外做 lerp(color, NR 输出, t)，与 DLL 一致；
        Style 1/2 时再经 style.grade 调色 (cg2r_post_process_kernel)。
        control_mask: DLSSNR.ControlMask，(H, W, 4) [R, G, B, A]，逐像素乘到全局参数上 (与 DLL 逐字节一致):
            R x Intensity (网络之外的混合)，G x LocalTone，B x LocalStructure (pre_block 的控制输入)，A 未使用。
            提供时 DLL 强制 UseAutoMask=0，所以 controls 须为 control_inputs(auto_mask=False, ...) (默认即此)。
        trace: 可选 dict，记录各级出口 (step 序号 -> 张量) 以便对照"""
        ops.PRECISE = self.precise
        t = lambda a: None if a is None else torch.as_tensor(np.asarray(a) if not torch.is_tensor(a) else a,  # noqa: E731
                                                              dtype=torch.float32, device=self.dev)
        color, hist, mv = t(color), t(hist), t(mv)
        if control_mask is not None:
            if controls is None:
                controls = control_inputs(auto_mask=False)
            if controls["Skin"] != -1.0 or controls["Structure"] != -1.0:
                raise ValueError("ControlMask forces UseAutoMask=0: use control_inputs(auto_mask=False, ...)")
            if controls.get("style", 0):
                raise NotImplementedError("ControlMask with Style 1/2 has not been checked against the DLL")
            control_mask = t(control_mask)
            controls = dict(controls, mask=control_mask)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.half, cache_enabled=False):
            out = self._forward(color, hist, mv, frame, trace, controls).float()
        style = (controls or {}).get("style", 0)                   # 网络之外的部分用 f32
        if style:                                                  # Style 1/2: DLL 的调色后处理 (含 Intensity 混合)
            return style_mod.apply(color[..., :3], out, style, intensity)
        k = min(max(float(intensity), 0.0), 1.0)
        if control_mask is not None:                               # 逐像素 Intensity (post_block 的 control_mask 变体)
            k = k * control_mask[..., :1]
            return color[..., :3] + k * (out - color[..., :3])
        return out if k == 1.0 else color[..., :3] + k * (out - color[..., :3])

    def _forward(self, color, hist, mv, frame, trace, controls):
        """网络本身 (pre -> 71 步 -> post)，快速模式下在 autocast 里运行"""
        if mv is None:
            mv = torch.zeros(*color.shape[:2], 2, device=self.dev)
        skips, x, cur, pre_skip = {}, None, None, None
        D = dims(*color.shape[:2])
        for i, (st, m) in enumerate(self.steps):
            k = st["kind"]
            if k == "pre":
                x, pre_skip = m(color, color if hist is None else hist, mv, frame, controls)
            elif k == "swin":
                lv = st["level"]
                H, Wd = D[lv]
                sh, var = tuple(st["shift"]), st["variant"]
                if var == "inpview":
                    cur = m(x[:, m.fi], H, Wd, sh)
                elif var == "ds":
                    cur, x = m(cur, H, Wd, sh)
                    skips[lv] = cur
                elif var == "up":
                    if lv == "8h":                                   # 16h 出口是补齐网格，8h 只读实际区域
                        x = pad_rows_cols(x, D["16h"], D["16h_real"])
                    cur = m(x, skips[lv], H, Wd, sh)
                elif var == "outview":
                    x = m(cur, H, Wd, sh)[:, m.ci]
                else:
                    cur = m(cur, H, Wd, sh)
            elif k == "split16":
                if st["inpview"]:                                    # 8h 下采样输出 (实际区域) 补 0 到 16h 补齐网格
                    x = pad_rows_cols(x, D["16h_real"], D["16h"])
                cur = m(x[:, m.fi] if st["inpview"] else cur, tuple(st["shift"]), *D["16h"])
                if st["tail"] == "pool":
                    skips["16h"] = cur
                    cur = st["head"](cur, D["16h"], D["vit"])
                elif st["tail"] == "outview":
                    x = cur[:, m.ci]
            elif k == "vit":
                cur = m(cur)
            elif k == "block39":
                cur = m(cur, skips["16h"], D["vit"], D["16h"])
            elif k == "post":
                out = m(x, pre_skip, color, hist, mv)
            if trace is not None:
                img_out = k == "pre" or (k == "swin" and st["variant"] in ("ds", "outview")) or \
                    (k == "split16" and st["tail"] == "outview")
                trace[i] = x if img_out else cur                     # 跨级的图像格式输出 / 级内的 tin 输出
        return out

    def graph(self, H=360, W=640, history=True, controls=None, intensity=1.0):
        """把整帧前向捕获成 CUDA Graph，返回 run(color, hist=None, mv=None, frame=0) -> 输出 (静态缓冲，下次调用会被覆盖)。
        history=False 捕获重置帧 (无历史) 的图。输入尺寸、controls 与 intensity 在捕获时固定。"""
        dev = self.dev
        sc = torch.zeros(H, W, 3, device=dev)
        sh = torch.zeros(H, W, 3, device=dev) if history else None
        sm = torch.zeros(H, W, 2, device=dev)
        sf = torch.zeros((), dtype=torch.int64, device=dev)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):                                     # 预热 (分配缓存、cuBLAS 句柄)
                self(sc, sh, sm, sf, controls=controls, intensity=intensity)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = self(sc, sh, sm, sf, controls=controls, intensity=intensity)

        def run(color, hist=None, mv=None, frame=0):
            sc.copy_(torch.as_tensor(color, device=dev))
            if history:
                sh.copy_(torch.as_tensor(hist, device=dev))
            sm.copy_(torch.as_tensor(mv, device=dev)) if mv is not None else sm.zero_()
            sf.fill_(int(frame))
            g.replay()
            return out
        return run
