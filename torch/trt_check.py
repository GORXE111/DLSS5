"""TensorRT 路线: 导出 -> 构建引擎 -> 与 torch 参考比较 + 计时。

    python trt_check.py eager  [--q8 arith|none] [--res 1080]   # 导出版网络在 torch 里跑 (不经 TRT)，与参考比较
    python trt_check.py build  [--q8 ...] [--res ...]           # 导出 ONNX 并构建引擎 (写到 ../_dl/trt/，不入库)
    python trt_check.py run    [--q8 ...] [--res ...]           # 跑引擎: 与参考比较、计时、运动矢量响应
参考 = DLSS5() 快速模式 (与 nr-lab 吻合 0.9993) 的同一输入输出。测试输入为 check.py 的合成图案 (放大到目标分辨率)。
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from check import color_pattern  # noqa: E402
from dlss5 import DLSS5  # noqa: E402

TRT_DIR = os.path.join(HERE, "..", "_dl", "trt")
RES = {"540": (540, 960), "720": (720, 1280), "1080": (1080, 1920)}


def test_input(H, W, shift=0):
    c = torch.tensor(color_pattern(640, 360, offset_x=shift), device="cuda").permute(2, 0, 1)[None]
    return F.interpolate(c, size=(H, W), mode="bilinear", align_corners=False)[0].permute(1, 2, 0).contiguous()


def frames(H, W, n=4, shift=4):
    """平移序列 (每帧右移 shift 个 360p 像素)，运动矢量 = -shift * W/640"""
    mv = torch.zeros(H, W, 2, device="cuda")
    mv[..., 0] = -shift * W / 640
    return [test_input(H, W, f * shift) for f in range(n)], mv


def compare(name, a, b):
    a8, b8 = (a * 255 + 0.5).floor(), (b * 255 + 0.5).floor()
    c = np.corrcoef(a8.flatten().cpu().numpy(), b8.flatten().cpu().numpy())[0, 1]
    print(f"  {name}: 相关 {c:.5f}  平均差 {(a8 - b8).abs().mean():.2f}/255  最大差 {(a8 - b8).abs().max():.0f}")


def reference(net, H, W, mvok=True):
    fs, mv = frames(H, W)
    outs, h = [], None
    for f, c in enumerate(fs):
        h = net(c, hist=h, mv=None if h is None else (mv if mvok else torch.zeros_like(mv)), frame=f)
        outs.append(h)
    return outs


def sequence(step, core, H, W, mvok=True):
    """step(X) -> n；与 reference 同样的 4 帧"""
    fs, mv = frames(H, W)
    mv = mv if mvok else torch.zeros_like(mv)
    outs, h = [], None
    for f, c in enumerate(fs):
        X = core.inputs(c, h, mv, f)
        h = core.finish(step(X), c, h, mv)
        outs.append(h)
    return outs


def engine_path(a):
    return os.path.join(TRT_DIR, f"dlss5_{a.res}_{a.q8}_c{a.canon}.engine")


def build(core, a):
    import tensorrt as trt
    os.makedirs(TRT_DIR, exist_ok=True)
    onnx_p = os.path.join(TRT_DIR, f"dlss5_{a.res}_{a.q8}_c{a.canon}.onnx")
    t0 = time.time()
    core.export(onnx_p)
    del core
    torch.cuda.empty_cache()                                     # 构建时只需要 ONNX，释放 torch 占的显存
    print(f"ONNX: {os.path.getsize(onnx_p) / 1e6:.0f} MB, {time.time() - t0:.0f} s")
    log = trt.Logger(trt.Logger.WARNING)
    b = trt.Builder(log)
    netw = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    p = trt.OnnxParser(netw, log)
    if not p.parse_from_file(onnx_p):
        for i in range(p.num_errors):
            print(p.get_error(i))
        raise SystemExit("ONNX 解析失败")
    cfg = b.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    cfg.builder_optimization_level = a.opt
    t0 = time.time()
    plan = b.build_serialized_network(netw, cfg)
    if plan is None:
        raise SystemExit("构建失败")
    open(engine_path(a), "wb").write(plan)
    print(f"引擎: {plan.nbytes / 1e6:.0f} MB, 构建 {time.time() - t0:.0f} s")


class Engine:
    def __init__(self, path):
        import tensorrt as trt
        self.rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.eng = self.rt.deserialize_cuda_engine(open(path, "rb").read())
        self.ctx = self.eng.create_execution_context()
        shp = tuple(self.eng.get_tensor_shape("n"))
        self.out = torch.empty(shp, dtype=torch.float16, device="cuda")
        self.ctx.set_tensor_address("n", self.out.data_ptr())
        self.stream = torch.cuda.current_stream()

    def __call__(self, X):
        X = X.contiguous()
        self.ctx.set_tensor_address("X", X.data_ptr())
        self.ctx.execute_async_v3(self.stream.cuda_stream)
        return self.out


def timeit(fn, n=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(n):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["eager", "build", "run"])
    ap.add_argument("--q8", default="arith", choices=["arith", "none"])
    ap.add_argument("--res", default="540", choices=list(RES))
    ap.add_argument("--canon", type=int, default=1, help="1: 规范序版本 (重排折进权重)")
    ap.add_argument("--opt", type=int, default=3, help="TensorRT builder_optimization_level")
    a = ap.parse_args()
    H, W = RES[a.res]
    net = DLSS5()
    ref = ref0 = None
    if a.what != "build":
        ref, ref0 = reference(net, H, W), reference(net, H, W, mvok=False)   # 必须在 set_export 之前算
    from dlss5.export import Core
    core = Core(net, H, W, q8=a.q8, canon=bool(a.canon))
    if a.what == "build":
        return build(core, a)
    step = core if a.what == "eager" else Engine(engine_path(a))
    with torch.no_grad():
        outs, outs0 = sequence(step, core, H, W), sequence(step, core, H, W, mvok=False)
    print(f"== {a.what} q8={a.q8} canon={a.canon} {W}x{H} vs torch 参考")
    for f in range(4):
        compare(f"第 {f} 帧", outs[f], ref[f])
    for f in (1, 2, 3):
        print(f"  第 {f} 帧 正确/错误运动矢量输出之差: 参考 {((ref[f] - ref0[f]).abs().mean() * 255):.2f}  本版 {((outs[f] - outs0[f]).abs().mean() * 255):.2f}")
    X = core.inputs(test_input(H, W), None, torch.zeros(H, W, 2, device="cuda"), 0)
    with torch.no_grad():
        print(f"  网络主体耗时: {timeit(lambda: step(X), 5 if a.what == 'eager' else 30):.1f} ms")
        c = test_input(H, W)
        mv = torch.zeros(H, W, 2, device="cuda")
        print(f"  入口采样+噪声: {timeit(lambda: core.inputs(c, c, mv, 1)):.1f} ms   出口混合: {timeit(lambda: core.finish(step.out if a.what == 'run' else torch.zeros(*core.D['grid'], 4, device='cuda', dtype=torch.float16), c, c, mv)):.1f} ms")


if __name__ == "__main__":
    main()
