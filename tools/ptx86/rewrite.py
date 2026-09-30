"""sm_120 PTX -> sm_86 PTX 改写器。

DLSS5 的 fatbin_01..07/14 用到 8 类 Ampere 没有的指令，这里逐条换成 sm_80 能跑的等价序列:

  mma.m16n8k32 e4m3          -> 解包成 f16，两条 mma.m16n8k16 f16 (k 维重标号，A/B 一致即可)
  cvt.f16x2.e4m3x2           -> 位移 + 乘 2^8 (精确)
  cvt.satfinite.e4m3x2.f16x2 -> f32 整数域舍入 (RN, ties-even, 饱和到 448)
  cp.async.bulk (TMA)        -> cp.async 16B 循环 + cp.async.mbarrier.arrive
  mbarrier.expect_tx         -> 删除 (完成跟踪改由 cp.async.mbarrier.arrive 承担)
  mbarrier.arrive 带 count   -> arrive.noComplete(count-1) + arrive
  mbarrier.try_wait          -> mbarrier.test_wait (外层本来就是重试循环)
  elect.sync                 -> activemask 最低位 == lanemask_eq
  red.global.v4.f16x2        -> 4 条标量 red.f16x2
  min.relu.s32               -> min + max(.,0)
  fence.release.gpu          -> fence.acq_rel.gpu (更强，安全)

已知偏差: e4m3 NaN(0x7F) 解码成 480 而不是 NaN；f16 输入 NaN 量化成 448；
FP8 mma 的 f16 累加被拆成两次舍入。

用法: python -m tools.ptx86.rewrite in.ptx out.ptx
"""
import re
import sys

_label = [0]


def _uid():
    _label[0] += 1
    return _label[0]


def _ops(s):
    return [x.strip() for x in s.split(",")]


# ---------------------------------------------------------------- FP8 转换

def _dec_spread(dst, w):
    # w: 每半 = e4m3<<8。f16 位 = s<<15 | em<<7，再乘 256 修正指数偏置 7->15 (精确)
    return (
        "{ .reg .b32 %x_s, %x_k;\n"
        f"and.b32 %x_s, {w}, 0x80008000;\n"
        f"and.b32 {w}, {w}, 0x7F007F00;\n"
        f"shr.u32 {w}, {w}, 1;\n"
        f"or.b32 {w}, {w}, %x_s;\n"
        "mov.b32 %x_k, 0x5C005C00;\n"
        f"mul.rn.f16x2 {dst}, {w}, %x_k;\n"
        "}\n"
    )


def _e4m3x2_to_f16x2(dst, src16):
    return (
        "{ .reg .b32 %x_t;\n"
        f"cvt.u32.u16 %x_t, {src16};\n"
        "prmt.b32 %x_t, %x_t, 0, 0x1404;\n"
        + _dec_spread(dst, "%x_t")
        + "}\n"
    )


def _half_to_e4m3(h, out):
    # h: .b16 f16, out: .b32 结果字节 (0..0xFE)
    return (
        f"cvt.f32.f16 %y_f, {h};\n"
        "abs.f32 %y_a, %y_f;\n"
        "min.f32 %y_a, %y_a, 0f43E00000;\n"          # 448
        "setp.lt.f32 %y_p, %y_a, 0f3C800000;\n"      # 2^-6 以下走次正规
        "mov.b32 %y_u, %y_a;\n"
        "shr.u32 %y_l, %y_u, 20;\n"
        "and.b32 %y_l, %y_l, 1;\n"
        "add.u32 %y_u, %y_u, 0x7FFFF;\n"
        "add.u32 %y_u, %y_u, %y_l;\n"
        "shr.u32 %y_u, %y_u, 20;\n"
        "sub.u32 %y_u, %y_u, 960;\n"                 # (127-7)<<3
        "mul.f32 %y_s, %y_a, 0f44000000;\n"          # *512
        "cvt.rni.u32.f32 %y_v, %y_s;\n"
        "selp.b32 %y_u, %y_v, %y_u, %y_p;\n"
        "mov.b32 %y_g, %y_f;\n"
        "shr.u32 %y_g, %y_g, 24;\n"
        "and.b32 %y_g, %y_g, 0x80;\n"
        f"or.b32 {out}, %y_u, %y_g;\n"
    )


def _f16x2_to_e4m3x2(dst16, src):
    """打包整数版，两半同时算，无 f32 往返。
    正规段 (|x|>=2^-6): f16 幅值位 a 做 RNE 右移 7 位 -> (E16<<3|m3)，再减 8<<3 修正偏置 (用 +192 取低字节避免跨半借位)
    次正规段: |x|+2.0 在 f16 里恰好按 2^-9 网格 RNE 舍入一次，减去 2.0 的位模式即 round(|x|*2^9)"""
    if ACC.get("encode") == "f32":
        return _f16x2_to_e4m3x2_f32(dst16, src)
    if ACC.get("enc", "v1") == "v1":
        return _f16x2_to_e4m3x2_v1(dst16, src)
    # 字节位置约定: 结果先放在 32 位的 bit 0..7 (半0) 与 bit 16..23 (半1)，最后一条 prmt 取字节 0/2。
    #   正规: (a + lsb + 0x3F + 0x6000) >> 7。0x6000 = (8<<3)... 即 -64 mod 256 预先左移 7 位加进去，
    #         每半最大 0x5F00+0x6040 < 0x10000 不跨半进位；右移后半1 的字节恰落在 bit 16..23
    #   次正规: (a + 2.0) - 0x4000 的整数值 0..8 本来就在每半的低位
    #   符号: src >> 8 把 bit 15/31 送到 bit 7/23
    return (
        "{ .reg .b32 %z_a, %z_l, %z_r, %z_s, %z_m, %z_k;\n"
        f"and.b32 %z_a, {src}, 0x7FFF7FFF;\n"
        "mov.b32 %z_k, 0x5F005F00;\n"                     # 448
        "min.f16x2 %z_a, %z_a, %z_k;\n"
        "shr.u32 %z_l, %z_a, 7;\n"
        "and.b32 %z_l, %z_l, 0x00010001;\n"
        "add.u32 %z_r, %z_a, %z_l;\n"
        "add.u32 %z_r, %z_r, 0x603F603F;\n"
        "shr.u32 %z_r, %z_r, 7;\n"
        "mov.b32 %z_k, 0x40004000;\n"                     # 2.0
        "add.rn.f16x2 %z_s, %z_a, %z_k;\n"
        "sub.u32 %z_s, %z_s, 0x40004000;\n"
        "mov.b32 %z_k, 0x24002400;\n"                     # 2^-6
        "set.lt.u32.f16x2 %z_m, %z_a, %z_k;\n"
        "and.b32 %z_s, %z_s, %z_m;\n"
        "not.b32 %z_m, %z_m;\n"
        "and.b32 %z_r, %z_r, %z_m;\n"
        "or.b32 %z_r, %z_r, %z_s;\n"
        f"shr.u32 %z_s, {src}, 8;\n"
        "and.b32 %z_s, %z_s, 0x00800080;\n"
        "or.b32 %z_r, %z_r, %z_s;\n"
        "prmt.b32 %z_r, %z_r, 0, 0x0020;\n"
        f"cvt.u16.u32 {dst16}, %z_r;\n"
        "}\n"
    )


def _f16x2_to_e4m3x2_v1(dst16, src):
    # 先右移再掩码/加偏置 (中间量少一个)
    return (
        "{ .reg .b32 %z_a, %z_l, %z_r, %z_s, %z_m, %z_k;\n"
        f"and.b32 %z_a, {src}, 0x7FFF7FFF;\n"
        "mov.b32 %z_k, 0x5F005F00;\n"
        "min.f16x2 %z_a, %z_a, %z_k;\n"
        "shr.u32 %z_l, %z_a, 7;\n"
        "and.b32 %z_l, %z_l, 0x00010001;\n"
        "add.u32 %z_r, %z_a, 0x003F003F;\n"
        "add.u32 %z_r, %z_r, %z_l;\n"
        "shr.u32 %z_r, %z_r, 7;\n"
        "and.b32 %z_r, %z_r, 0x01FF01FF;\n"
        "add.u32 %z_r, %z_r, 0x00C000C0;\n"
        "mov.b32 %z_k, 0x40004000;\n"
        "add.rn.f16x2 %z_s, %z_a, %z_k;\n"
        "sub.u32 %z_s, %z_s, 0x40004000;\n"
        "mov.b32 %z_k, 0x24002400;\n"
        "set.lt.u32.f16x2 %z_m, %z_a, %z_k;\n"
        "and.b32 %z_s, %z_s, %z_m;\n"
        "not.b32 %z_m, %z_m;\n"
        "and.b32 %z_r, %z_r, %z_m;\n"
        "or.b32 %z_r, %z_r, %z_s;\n"
        f"prmt.b32 %z_s, {src}, 0, 0x0031;\n"
        "and.b32 %z_s, %z_s, 0x8080;\n"
        "prmt.b32 %z_r, %z_r, 0, 0x0020;\n"
        "or.b32 %z_r, %z_r, %z_s;\n"
        f"cvt.u16.u32 {dst16}, %z_r;\n"
        "}\n"
    )


def _f16x2_to_e4m3x2_f32(dst16, src):
    return (
        "{ .reg .b16 %y_h0, %y_h1; .reg .f32 %y_f, %y_a, %y_s;\n"
        ".reg .b32 %y_u, %y_l, %y_v, %y_g, %y_o0, %y_o1; .reg .pred %y_p;\n"
        f"mov.b32 {{%y_h0, %y_h1}}, {src};\n"
        + _half_to_e4m3("%y_h0", "%y_o0")
        + _half_to_e4m3("%y_h1", "%y_o1")
        + "shl.b32 %y_o1, %y_o1, 8;\n"
        "or.b32 %y_o0, %y_o0, %y_o1;\n"
        f"cvt.u16.u32 {dst16}, %y_o0;\n"
        "}\n"
    )


# ---------------------------------------------------------------- FP8 mma

def _unpack4(reg, lo, hi):
    if ACC.get("dec", "v1") == "v1":
        return _unpack4_v1(reg, lo, hi)
    return _unpack4_v2(reg, lo, hi)


def _unpack4_v1(reg, lo, hi):
    # 每对字节各自 prmt 摊开再解码 (寄存器活跃区间短)
    return (
        "{ .reg .b32 %m_w;\n"
        f"prmt.b32 %m_w, {reg}, 0, 0x1404;\n"
        + _dec_spread(lo, "%m_w")
        + f"prmt.b32 %m_w, {reg}, 0, 0x3424;\n"
        + _dec_spread(hi, "%m_w")
        + "}\n"
    )


def _unpack4_v2(reg, lo, hi):
    """32 位 4 个 e4m3 -> 两个 f16x2 (字节 0,1 -> lo; 2,3 -> hi)，四个字节共用移位/掩码。
    每个字节 b 的 f16 位 = s<<15 | (b&0x7F)<<7:
      高字节 HS = s<<7 | (b>>1)&0x3F   -> (reg>>1)&0x3F3F3F3F | reg&0x80808080
      低字节 L  = (b&1)<<7              -> (reg<<7)&0x80808080
    两条 prmt 交错拼出 [L0,HS0,L1,HS1] / [L2,HS2,L3,HS3]，再乘 256 修正指数偏置"""
    return (
        "{ .reg .b32 %u_h, %u_s, %u_l, %u_k;\n"
        f"shr.u32 %u_h, {reg}, 1;\n"
        "and.b32 %u_h, %u_h, 0x3F3F3F3F;\n"
        f"and.b32 %u_s, {reg}, 0x80808080;\n"
        "or.b32 %u_h, %u_h, %u_s;\n"
        f"shl.b32 %u_l, {reg}, 7;\n"
        "and.b32 %u_l, %u_l, 0x80808080;\n"
        "mov.b32 %u_k, 0x5C005C00;\n"
        "prmt.b32 %u_s, %u_l, %u_h, 0x5140;\n"
        f"mul.rn.f16x2 {lo}, %u_s, %u_k;\n"
        "prmt.b32 %u_s, %u_l, %u_h, 0x7362;\n"
        f"mul.rn.f16x2 {hi}, %u_s, %u_k;\n"
        "}\n"
    )


ACC = {"mode": "f16"}  # "f32": 两段在 f32 里累加、最后一次舍入回 f16 (更接近硬件，GA10x 上吞吐减半)


def _mma_fp8(m):
    pred = m.group(1)
    d, a, b, c = (_ops(m.group(i)) for i in range(2, 6))
    if ACC["mode"] == "f32":
        return _guard(pred, _mma_fp8_f32acc(d, a, b, c))
    # k32 线程持有 k={4t..4t+3} 与 {4t+16..4t+19}；k16 持有 {2t,2t+1} 与 {2t+8,2t+9}
    # 重标号 2t+j <-> 4t+j, 2t+8+j <-> 4t+2+j: A 取 lo/hi，B 同样取 lo/hi，两边一致则和不变
    body = (
        "{ .reg .b16 %m_b0, %m_b1;\n"
        ".reg .b32 %m_a<8>, %m_c<4>, %m_t<2>;\n"
        + _unpack4(a[0], "%m_a0", "%m_a2")
        + _unpack4(a[1], "%m_a1", "%m_a3")
        + _unpack4(a[2], "%m_a4", "%m_a6")
        + _unpack4(a[3], "%m_a5", "%m_a7")
        + _unpack4(b[0], "%m_c0", "%m_c1")
        + _unpack4(b[1], "%m_c2", "%m_c3")
        + "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%m_t0, %m_t1}, "
        "{%m_a0, %m_a1, %m_a2, %m_a3}, {%m_c0, %m_c1}, " f"{{{c[0]}, {c[1]}}};\n"
        f"mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {{{d[0]}, {d[1]}}}, "
        "{%m_a4, %m_a5, %m_a6, %m_a7}, {%m_c2, %m_c3}, {%m_t0, %m_t1};\n"
        "}\n"
    )
    return _guard(pred, body)


def _mma_fp8_f32acc(d, a, b, c):
    return (
        "{ .reg .b16 %m_b0, %m_b1, %m_h<4>;\n"
        ".reg .b32 %m_a<8>, %m_c<4>;\n"
        ".reg .f32 %m_f<4>;\n"
        + _unpack4(a[0], "%m_a0", "%m_a2")
        + _unpack4(a[1], "%m_a1", "%m_a3")
        + _unpack4(a[2], "%m_a4", "%m_a6")
        + _unpack4(a[3], "%m_a5", "%m_a7")
        + _unpack4(b[0], "%m_c0", "%m_c1")
        + _unpack4(b[1], "%m_c2", "%m_c3")
        # f16 累加器 {c0,c1} = {(r g, col 2t,2t+1), (r g+8, ...)}，与 f32 的 c0..c3 顺序一致
        + f"mov.b32 {{%m_h0, %m_h1}}, {c[0]};\nmov.b32 {{%m_h2, %m_h3}}, {c[1]};\n"
        "cvt.f32.f16 %m_f0, %m_h0;\ncvt.f32.f16 %m_f1, %m_h1;\ncvt.f32.f16 %m_f2, %m_h2;\ncvt.f32.f16 %m_f3, %m_h3;\n"
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%m_f0, %m_f1, %m_f2, %m_f3}, "
        "{%m_a0, %m_a1, %m_a2, %m_a3}, {%m_c0, %m_c1}, {%m_f0, %m_f1, %m_f2, %m_f3};\n"
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%m_f0, %m_f1, %m_f2, %m_f3}, "
        "{%m_a4, %m_a5, %m_a6, %m_a7}, {%m_c2, %m_c3}, {%m_f0, %m_f1, %m_f2, %m_f3};\n"
        f"cvt.rn.f16x2.f32 {d[0]}, %m_f1, %m_f0;\n"
        f"cvt.rn.f16x2.f32 {d[1]}, %m_f3, %m_f2;\n"
        "}\n"
    )


def _guard(pred, body):
    if not pred:
        return body
    p = pred.strip()[1:]
    neg = p.startswith("!")
    p = p.lstrip("!")
    lbl = f"$L__x86_skip{_uid()}"
    return f"@{'' if neg else '!'}{p} bra {lbl};\n{body}{lbl}:\n"


# ---------------------------------------------------------------- sm_90 同步/拷贝

def _bulk_copy(m):
    """TMA 批量拷贝 -> Ampere cp.async (16B/条，异步发射) + cp.async.mbarrier.arrive。
    后者在本线程此前的 cp.async 全部完成时对 mbarrier 做一次异步 arrive，并预先把 pending 计数 +1，
    净效果 = "拷贝没完成前该相位不能结束"，与 expect_tx/complete_tx 语义一致。"""
    pred = m.group(1)
    dst, src, size, bar = _ops(m.group(2))
    dst, src = dst.strip("[]"), src.strip("[]")
    n = _uid()
    body = (
        "{ .reg .b32 %k_i, %k_d; .reg .b64 %k_s; .reg .pred %k_p;\n"
        "mov.b32 %k_i, 0;\n"
        f"$L__x86_cp{n}:\n"
        f"setp.ge.u32 %k_p, %k_i, {size};\n"
        f"@%k_p bra $L__x86_cpd{n};\n"
        "cvt.u64.u32 %k_s, %k_i;\n"
        f"add.s64 %k_s, %k_s, {src};\n"
        f"add.u32 %k_d, {dst}, %k_i;\n"
        "cp.async.cg.shared.global [%k_d], [%k_s], 16;\n"
        "add.u32 %k_i, %k_i, 16;\n"
        f"bra.uni $L__x86_cp{n};\n"
        f"$L__x86_cpd{n}:\n"
        f"cp.async.mbarrier.arrive.shared.b64 {bar};\n"
        "}\n"
    )
    return _guard(pred, body)


def _arrive(m):
    pred = m.group(1)
    state, addr, cnt = _ops(m.group(2))
    if cnt == "1":
        body = f"mbarrier.arrive.shared.b64 {state}, {addr};\n"
    else:
        body = (
            "{ .reg .b32 %r_c; .reg .pred %r_p;\n"
            f"sub.u32 %r_c, {cnt}, 1;\n"
            "setp.ne.u32 %r_p, %r_c, 0;\n"
            f"@%r_p mbarrier.arrive.noComplete.shared.b64 {state}, {addr}, %r_c;\n"
            f"mbarrier.arrive.shared.b64 {state}, {addr};\n"
            "}\n"
        )
    return _guard(pred, body)


def _elect(m):
    pred_out, mask = _ops(m.group(1))
    p = pred_out.split("|", 1)[1]
    return (
        "{ .reg .b32 %e_m, %e_n, %e_l;\n"
        "activemask.b32 %e_m;\n"
        f"and.b32 %e_m, %e_m, {mask};\n"
        "neg.s32 %e_n, %e_m;\n"
        "and.b32 %e_m, %e_m, %e_n;\n"
        "mov.u32 %e_l, %lanemask_eq;\n"
        f"setp.eq.b32 {p}, %e_m, %e_l;\n"
        "}\n"
    )


def _red_v4(m):
    pred = m.group(1)
    addr, vals = m.group(2).strip(), _ops(m.group(3))
    base, _, off = addr.partition("+")
    off = int(off) if off else 0
    body = "".join(
        f"red.global.add.noftz.f16x2 [{base}+{off + 4 * i}], {v};\n" for i, v in enumerate(vals)
    )
    return _guard(pred, body)


def _min_relu(m):
    pred = m.group(1)
    d, a, b = _ops(m.group(2))
    return _guard(pred, f"min.s32 {d}, {a}, {b};\nmax.s32 {d}, {d}, 0;\n")


P = r"((?:@!?%p\d+\s+)?)"  # 可选谓词
RULES = [
    (re.compile(P + r"mma\.sync\.aligned\.m16n8k32\.row\.col\.f16\.e4m3\.e4m3\.f16\s*"
                r"\{([^}]*)\}\s*,\s*\{([^}]*)\}\s*,\s*\{([^}]*)\}\s*,\s*\{([^}]*)\}\s*;"), _mma_fp8),
    (re.compile(P + r"cvt\.rn\.satfinite\.e4m3x2\.f16x2\s+([^,;]+),\s*([^;]+);"),
     lambda m: _guard(m.group(1), _f16x2_to_e4m3x2(m.group(2).strip(), m.group(3).strip()))),
    (re.compile(P + r"cvt\.rn\.f16x2\.e4m3x2\s+([^,;]+),\s*([^;]+);"),
     lambda m: _guard(m.group(1), _e4m3x2_to_f16x2(m.group(2).strip(), m.group(3).strip()))),
    (re.compile(P + r"cp\.async\.bulk\.shared::cta\.global\.mbarrier::complete_tx::bytes\s+([^;]+);"), _bulk_copy),
    (re.compile(P + r"mbarrier\.expect_tx\.relaxed\.cta\.shared::cta\.b64\s+[^;]+;"), lambda m: ""),
    (re.compile(P + r"mbarrier\.arrive\.shared::cta\.b64\s+([^;]+);"), _arrive),
    (re.compile(r"mbarrier\.try_wait\.shared::cta\.b64"), lambda m: "mbarrier.test_wait.shared.b64"),
    (re.compile(r"elect\.sync\s+([^;]+);"), _elect),
    (re.compile(P + r"red\.global\.v4\.f16x2\.add\.noftz\s+\[([^\]]+)\]\s*,\s*\{([^}]*)\}\s*;"), _red_v4),
    (re.compile(P + r"min\.relu\.s32\s+([^;]+);"), _min_relu),
    (re.compile(r"fence\.release\.gpu\s*;"), lambda m: "fence.acq_rel.gpu;"),
]


def rewrite(src):
    src = re.sub(r"^\.version 9\.\d+", ".version 9.3", src, flags=re.M)
    src = re.sub(r"^\.target sm_1\d\d[a-z]*", ".target sm_86", src, flags=re.M)
    if ACC.get("maxnreg"):
        # 原版按 Blackwell 定的 168；仿真后指令链更长，调低换占用率 (只压 >= 该值的)
        cap = ACC["maxnreg"]
        src = re.sub(r"^\.maxnreg (\d+)", lambda m: f".maxnreg {min(int(m.group(1)), cap)}", src, flags=re.M)
    if ACC.get("maxnreg_floor"):
        # 反方向: 抬高上限减少溢出 (实测压低到 128/96 明显变慢，瓶颈是溢出不是占用率)
        floor = ACC["maxnreg_floor"]
        src = re.sub(r"^\.maxnreg (\d+)", lambda m: f".maxnreg {max(int(m.group(1)), floor)}", src, flags=re.M)
    stats = {}
    if ACC.get("fuse", True):
        from . import fuse
        src, fstats = fuse.apply(src)
        stats.update({f"fuse:{k}": v for k, v in fstats.items()})
    for rx, fn in RULES:
        src, n = rx.subn(fn, src)
        stats[rx.pattern[len(P):][:40] if rx.pattern.startswith(P) else rx.pattern[:40]] = n
    return src, stats


def main():
    src = open(sys.argv[1], encoding="utf-8").read()
    out, stats = rewrite(src)
    open(sys.argv[2], "w", encoding="utf-8", newline="\n").write(out)
    for k, v in stats.items():
        if v:
            print(f"{v:7d}  {k}")


if __name__ == "__main__":
    main()
