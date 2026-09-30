"""在真卡(sm_86)上验证 rewrite.py 的仿真序列。

测试 PTX 用原始 sm_120 指令写，经过 rewrite() 同一条路径再加载，所以测的就是移植时实际用的代码。
  1. e4m3 -> f16x2: 全部 65536 种 16 位输入穷举
  2. f16x2 -> e4m3x2 satfinite: 全部 65536 个 f16 穷举 (NaN 除外)
  3. mma m16n8k32 e4m3: 随机 A/B/C 多轮，对照 f32 参考

python -m tools.ptx86.test_emu
"""
import ctypes
import numpy as np
import torch

from .rewrite import rewrite

PTX = r"""
.version 9.4
.target sm_120
.address_size 64

.visible .entry cvt_dec(.param .u64 pin, .param .u64 pout, .param .u32 n)
{
  .reg .b16 %rs1; .reg .b32 %r<4>; .reg .b64 %rd<6>; .reg .pred %p1;
  ld.param.u64 %rd1, [pin]; ld.param.u64 %rd2, [pout]; ld.param.u32 %r1, [n];
  mov.u32 %r2, %ctaid.x; mov.u32 %r3, %ntid.x; mov.u32 %r0, %tid.x; mad.lo.s32 %r2, %r2, %r3, %r0;
  setp.ge.u32 %p1, %r2, %r1; @%p1 bra DONE;
  mul.wide.u32 %rd3, %r2, 2; add.s64 %rd4, %rd1, %rd3; ld.global.u16 %rs1, [%rd4];
  cvt.rn.f16x2.e4m3x2 %r3, %rs1;
  mul.wide.u32 %rd3, %r2, 4; add.s64 %rd5, %rd2, %rd3; st.global.u32 [%rd5], %r3;
DONE:
  ret;
}

.visible .entry cvt_enc(.param .u64 pin, .param .u64 pout, .param .u32 n)
{
  .reg .b16 %rs1; .reg .b32 %r<4>; .reg .b64 %rd<6>; .reg .pred %p1;
  ld.param.u64 %rd1, [pin]; ld.param.u64 %rd2, [pout]; ld.param.u32 %r1, [n];
  mov.u32 %r2, %ctaid.x; mov.u32 %r3, %ntid.x; mov.u32 %r0, %tid.x; mad.lo.s32 %r2, %r2, %r3, %r0;
  setp.ge.u32 %p1, %r2, %r1; @%p1 bra DONE;
  mul.wide.u32 %rd3, %r2, 4; add.s64 %rd4, %rd1, %rd3; ld.global.u32 %r3, [%rd4];
  cvt.rn.satfinite.e4m3x2.f16x2 %rs1, %r3;
  mul.wide.u32 %rd3, %r2, 2; add.s64 %rd5, %rd2, %rd3; st.global.u16 [%rd5], %rs1;
DONE:
  ret;
}

.visible .entry mma_fp8(.param .u64 pa, .param .u64 pb, .param .u64 pc, .param .u64 pd)
{
  .reg .b32 %r<16>; .reg .b64 %rd<12>;
  ld.param.u64 %rd1, [pa]; ld.param.u64 %rd2, [pb]; ld.param.u64 %rd3, [pc]; ld.param.u64 %rd4, [pd];
  mov.u32 %r0, %tid.x;
  mul.wide.u32 %rd5, %r0, 16; add.s64 %rd6, %rd1, %rd5; ld.global.v4.u32 {%r1, %r2, %r3, %r4}, [%rd6];
  mul.wide.u32 %rd5, %r0, 8;  add.s64 %rd7, %rd2, %rd5; ld.global.v2.u32 {%r5, %r6}, [%rd7];
  add.s64 %rd8, %rd3, %rd5; ld.global.v2.u32 {%r7, %r8}, [%rd8];
  mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%r9, %r10},
  {%r1, %r2, %r3, %r4},
  {%r5, %r6},
  {%r7, %r8};
  add.s64 %rd9, %rd4, %rd5; st.global.v2.u32 [%rd9], {%r9, %r10};
  ret;
}

.visible .entry fq_roundtrip(.param .u64 pin, .param .u64 pout, .param .u32 n)
{
  .reg .b16 %rs1; .reg .b32 %r<6>; .reg .b64 %rd<6>; .reg .pred %p1;
  ld.param.u64 %rd1, [pin]; ld.param.u64 %rd2, [pout]; ld.param.u32 %r1, [n];
  mov.u32 %r2, %ctaid.x; mov.u32 %r3, %ntid.x; mov.u32 %r0, %tid.x; mad.lo.s32 %r2, %r2, %r3, %r0;
  setp.ge.u32 %p1, %r2, %r1; @%p1 bra DONE;
  mul.wide.u32 %rd3, %r2, 4; add.s64 %rd4, %rd1, %rd3; ld.global.u32 %r3, [%rd4];
  cvt.rn.satfinite.e4m3x2.f16x2 %rs1, %r3;
  cvt.rn.f16x2.e4m3x2 %r4, %rs1;
  add.s64 %rd5, %rd2, %rd3; st.global.u32 [%rd5], %r4;
DONE:
  ret;
}

.visible .entry mma_fused(.param .u64 pa, .param .u64 pb, .param .u64 pc, .param .u64 pd)
{
  .reg .b16 %rs<9>; .reg .b32 %r<32>; .reg .b64 %rd<12>;
  ld.param.u64 %rd1, [pa]; ld.param.u64 %rd2, [pb]; ld.param.u64 %rd3, [pc]; ld.param.u64 %rd4, [pd];
  mov.u32 %r0, %tid.x;
  mul.wide.u32 %rd5, %r0, 32; add.s64 %rd6, %rd1, %rd5;
  ld.global.v4.u32 {%r1, %r2, %r3, %r4}, [%rd6];
  ld.global.v4.u32 {%r11, %r12, %r13, %r14}, [%rd6+16];
  cvt.rn.satfinite.e4m3x2.f16x2 %rs1, %r1;
  cvt.rn.satfinite.e4m3x2.f16x2 %rs2, %r2;
  cvt.rn.satfinite.e4m3x2.f16x2 %rs3, %r3;
  cvt.rn.satfinite.e4m3x2.f16x2 %rs4, %r4;
  cvt.rn.satfinite.e4m3x2.f16x2 %rs5, %r11;
  cvt.rn.satfinite.e4m3x2.f16x2 %rs6, %r12;
  cvt.rn.satfinite.e4m3x2.f16x2 %rs7, %r13;
  cvt.rn.satfinite.e4m3x2.f16x2 %rs8, %r14;
  mov.b32 %r21, {%rs1, %rs2};
  mov.b32 %r22, {%rs3, %rs4};
  mov.b32 %r23, {%rs5, %rs6};
  mov.b32 %r25, {%rs7, %rs8};
  mov.b32 %r24, %r25;
  mul.wide.u32 %rd5, %r0, 8;  add.s64 %rd7, %rd2, %rd5; ld.global.v2.u32 {%r5, %r6}, [%rd7];
  add.s64 %rd8, %rd3, %rd5; ld.global.v2.u32 {%r7, %r8}, [%rd8];
  mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%r9, %r10},
  {%r21, %r22, %r23, %r24},
  {%r5, %r6},
  {%r7, %r8};
  add.s64 %rd9, %rd4, %rd5; st.global.v2.u32 [%rd9], {%r9, %r10};
  ret;
}
"""

cu = ctypes.WinDLL("nvcuda.dll")


def ck(r):
    if r:
        s = ctypes.c_char_p()
        cu.cuGetErrorString(r, ctypes.byref(s))
        raise RuntimeError(s.value.decode())


def load():
    torch.zeros(1, device="cuda")  # 让 torch 建好主上下文
    ctx = ctypes.c_void_p()
    ck(cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), 0))
    ck(cu.cuCtxSetCurrent(ctx))
    src, _ = rewrite(PTX)
    mod = ctypes.c_void_p()
    log = ctypes.create_string_buffer(8192)
    opts = (ctypes.c_int * 2)(5, 6)  # CU_JIT_ERROR_LOG_BUFFER, _SIZE_BYTES
    vals = (ctypes.c_void_p * 2)(ctypes.cast(log, ctypes.c_void_p), 8192)
    r = cu.cuModuleLoadDataEx(ctypes.byref(mod), src.encode() + b"\0", 2, opts, vals)
    if r:
        print(log.value.decode())
        ck(r)
    return mod


def fn(mod, name):
    f = ctypes.c_void_p()
    ck(cu.cuModuleGetFunction(ctypes.byref(f), mod, name.encode()))
    return f


def launch(f, grid, block, *args):
    holders = [ctypes.c_uint64(a.data_ptr()) if isinstance(a, torch.Tensor) else ctypes.c_uint32(a) for a in args]
    ptrs = (ctypes.c_void_p * len(holders))(*[ctypes.cast(ctypes.byref(h), ctypes.c_void_p) for h in holders])
    ck(cu.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, 0, None, ptrs, None))
    ck(cu.cuCtxSynchronize())


# ---------------------------------------------------------------- 参考实现

E4M3 = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float().numpy()  # 0x7F/0xFF = NaN


def ref_encode(h):
    """f16 -> e4m3 字节，RN + satfinite。"""
    x = h.astype(np.float32)
    c = np.clip(x, -448, 448)
    b = torch.from_numpy(c).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
    return b


def test_decode(mod):
    n = 65536
    src = torch.arange(n, dtype=torch.int32).to(torch.int16).cuda()
    out = torch.zeros(n, dtype=torch.int32, device="cuda")
    launch(fn(mod, "cvt_dec"), n // 256, 256, src, out, n)
    got = out.cpu().numpy().view(np.float16).reshape(n, 2).astype(np.float32)
    lo, hi = np.arange(n) & 0xFF, np.arange(n) >> 8
    exp = np.stack([E4M3[lo], E4M3[hi]], 1)
    nan = np.isnan(exp)
    bad = (got != exp) & ~nan
    print(f"[decode] 65536 组 / {bad.sum()} 错 (NaN 编码 {nan.any(1).sum()} 组跳过)")
    return bad.sum() == 0


def test_encode(mod):
    h = np.arange(65536, dtype=np.uint32).astype(np.uint16).view(np.float16)
    ok = ~np.isnan(h)
    pairs = np.stack([h, h[::-1]], 1).copy()  # lo/hi 两半都覆盖全部值
    src = torch.from_numpy(pairs.view(np.int32).reshape(-1)).cuda()
    out = torch.zeros(65536, dtype=torch.int16, device="cuda")
    launch(fn(mod, "cvt_enc"), 256, 256, src, out, 65536)
    got = out.cpu().numpy().view(np.uint16)
    g_lo, g_hi = got & 0xFF, got >> 8
    e_lo, e_hi = ref_encode(pairs[:, 0]), ref_encode(pairs[:, 1])
    bad = ((g_lo != e_lo) & ok) | ((g_hi != e_hi) & ok[::-1])
    print(f"[encode] 65536x2 个 f16 / {bad.sum()} 错")
    if bad.any():
        i = np.nonzero(bad)[0][:5]
        print("  样例", pairs[i], g_lo[i], e_lo[i], g_hi[i], e_hi[i])
    return bad.sum() == 0


def frag_a(A):  # A: 16x32 字节
    out = np.zeros((32, 16), np.uint8)
    for lane in range(32):
        g, t = lane >> 2, lane & 3
        for i in range(16):
            row = g + (8 if (4 <= i < 8 or i >= 12) else 0)
            col = t * 4 + (i & 3) + (16 if i >= 8 else 0)
            out[lane, i] = A[row, col]
    return out


def frag_b(B):  # B: 32x8 字节 (k x n)
    out = np.zeros((32, 8), np.uint8)
    for lane in range(32):
        g, t = lane >> 2, lane & 3
        for i in range(8):
            out[lane, i] = B[t * 4 + (i & 3) + (16 if i >= 4 else 0), g]
    return out


def frag_c(C):  # C: 16x8 f16 -> 每线程 4 个 f16
    out = np.zeros((32, 4), np.float16)
    for lane in range(32):
        g, t = lane >> 2, lane & 3
        for i in range(4):
            out[lane, i] = C[g + (8 if i >= 2 else 0), t * 2 + (i & 1)]
    return out


def unfrag_c(F):
    C = np.zeros((16, 8), np.float32)
    for lane in range(32):
        g, t = lane >> 2, lane & 3
        for i in range(4):
            C[g + (8 if i >= 2 else 0), t * 2 + (i & 1)] = F[lane, i]
    return C


def _run_mma(f, A, B, C):
    t = lambda x: torch.from_numpy(x.view(np.int32).copy()).cuda()
    d = torch.zeros(64, dtype=torch.int32, device="cuda")
    launch(f, 1, 32, t(frag_a(A)), t(frag_b(B)), t(frag_c(C)), d)
    return unfrag_c(d.cpu().numpy().view(np.float16).reshape(32, 4))


def test_mma(mod, rounds=200):
    """整数输入: 所有中间和在 f16 里都精确可表示，结果必须逐位相等 —— 布局/重标号错一个就会露馅。
    随机小数输入: 只看与精确值的差，量级应是 f16 舍入 (绝对误差 / 幅值 ~1e-3)。"""
    rng = np.random.default_rng(0)
    f = fn(mod, "mma_fp8")
    finite = np.nonzero(~np.isnan(E4M3))[0]
    ints = finite[(np.abs(E4M3[finite]) <= 4) & (E4M3[finite] == np.round(E4M3[finite]))]
    small = finite[np.abs(E4M3[finite]) <= 4]
    exact_bad, worst = 0, 0.0
    for _ in range(rounds):
        A = rng.choice(ints, (16, 32)).astype(np.uint8)
        B = rng.choice(ints, (32, 8)).astype(np.uint8)
        C = rng.integers(-8, 9, (16, 8)).astype(np.float16)
        got = _run_mma(f, A, B, C)
        exact_bad += (got != E4M3[A].astype(np.float64) @ E4M3[B] + C).sum()

        A = rng.choice(small, (16, 32)).astype(np.uint8)
        B = rng.choice(small, (32, 8)).astype(np.uint8)
        C = rng.uniform(-8, 8, (16, 8)).astype(np.float16)
        got = _run_mma(f, A, B, C)
        exp = E4M3[A].astype(np.float64) @ E4M3[B] + C
        worst = max(worst, (np.abs(got - exp) / np.abs(exp).max()).max())
    ok = exact_bad == 0 and worst < 2e-3
    print(f"[mma]    整数 {rounds} 轮逐位: {exact_bad} 错 / 小数 {rounds} 轮: 误差/幅值 {worst:.1e} ({'OK' if ok else 'FAIL'})")
    return ok


def test_fq(mod):
    """编码后立即解码 -> 融合后走 f16 伪量化影子；逐位对照 decode(encode(x))"""
    h = np.arange(65536, dtype=np.uint32).astype(np.uint16).view(np.float16)
    pairs = np.stack([h, h[::-1]], 1).copy()
    src = torch.from_numpy(pairs.view(np.int32).reshape(-1)).cuda()
    out = torch.zeros(65536, dtype=torch.int32, device="cuda")
    launch(fn(mod, "fq_roundtrip"), 256, 256, src, out, 65536)
    got = out.cpu().numpy().view(np.uint16).reshape(-1, 2)
    exp = np.stack([E4M3[ref_encode(pairs[:, 0])], E4M3[ref_encode(pairs[:, 1])]], 1).astype(np.float16).view(np.uint16)
    ok = ~np.isnan(pairs.astype(np.float32))
    bad = (got != exp) & ok
    print(f"[fq]     65536x2 个 f16 伪量化 / {bad.sum()} 错")
    if bad.any():
        i = np.nonzero(bad.any(1))[0][:5]
        print("  样例", pairs[i], got[i], exp[i])
    return bad.sum() == 0


def test_mma_fused(mod, rounds=100):
    """A 操作数由 f16 编码+打包+拷贝得到 (融合路径)，整数输入逐位对照"""
    rng = np.random.default_rng(1)
    f = fn(mod, "mma_fused")
    finite = np.nonzero(~np.isnan(E4M3))[0]
    ints = finite[(np.abs(E4M3[finite]) <= 4) & (E4M3[finite] == np.round(E4M3[finite]))]
    bad = 0
    for _ in range(rounds):
        A = rng.choice(ints, (16, 32)).astype(np.uint8)
        B = rng.choice(ints, (32, 8)).astype(np.uint8)
        C = rng.integers(-8, 9, (16, 8)).astype(np.float16)
        a16 = E4M3[frag_a(A)].astype(np.float16)  # 每线程 16 个 f16，编码后恰是原字节
        t = lambda x: torch.from_numpy(x.view(np.int32).copy()).cuda()
        d = torch.zeros(64, dtype=torch.int32, device="cuda")
        launch(f, 1, 32, t(a16), t(frag_b(B)), t(frag_c(C)), d)
        got = unfrag_c(d.cpu().numpy().view(np.float16).reshape(32, 4))
        bad += (got != E4M3[A].astype(np.float64) @ E4M3[B] + C).sum()
    print(f"[fused]  编码->打包->拷贝->mma {rounds} 轮逐位: {bad} 错")
    return bad == 0


if __name__ == "__main__":
    m = load()
    src, st = rewrite(PTX)
    print("fuse:", {k: v for k, v in st.items() if k.startswith("fuse:")})
    r = [test_decode(m), test_encode(m), test_mma(m), test_fq(m), test_mma_fused(m)]
    print("全部通过" if all(r) else "有失败")
