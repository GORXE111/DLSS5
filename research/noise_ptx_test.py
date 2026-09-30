"""把 pre_block 的噪声哈希指令原样搬进小 kernel，在 GPU 上对已知 (x, y, frame) 求值，与 noise.py 对照。"""
import ctypes
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import klab  # noqa: E402
import noise as N  # noqa: E402

PTX = r"""
.version 8.0
.target sm_86
.address_size 64
.visible .entry nz(.param .u64 pout, .param .u32 W, .param .u32 H, .param .u32 frame)
{
  .reg .b32 %r<260>; .reg .f32 %f<40>; .reg .b64 %rd<8>; .reg .pred %p;
  ld.param.u64 %rd1, [pout]; ld.param.u32 %r1, [W]; ld.param.u32 %r2, [H]; ld.param.u32 %r122, [frame];
  mov.u32 %r3, %ctaid.x; mov.u32 %r4, %ntid.x; mov.u32 %r5, %tid.x; mad.lo.s32 %r6, %r3, %r4, %r5;
  mul.lo.s32 %r7, %r1, %r2; setp.ge.u32 %p, %r6, %r7; @%p bra DONE;
  rem.u32 %r134, %r6, %r1;                 // x
  div.u32 %r156, %r6, %r1;                 // y
  mul.lo.s32 %r123, %r122, -1640531527;
  mul.lo.s32 %r138, %r134, -1918454973;
  xor.b32 %r50, %r138, %r123;
  mul.lo.s32 %r160, %r156, -669632447;
  xor.b32 %r161, %r50, %r160;
  xor.b32 %r162, %r161, 608135816;
  shr.u32 %r163, %r162, 28; add.s32 %r164, %r163, 4; shr.u32 %r165, %r162, %r164; xor.b32 %r166, %r165, %r162;
  mul.lo.s32 %r167, %r166, 277803737; shr.u32 %r168, %r167, 22; xor.b32 %r169, %r168, %r167;
  mad.lo.s32 %r170, %r169, 747796405, -1403630843;
  shr.u32 %r171, %r170, 28; add.s32 %r172, %r171, 4; shr.u32 %r173, %r170, %r172; xor.b32 %r174, %r173, %r170;
  mul.lo.s32 %r175, %r174, 277803737; shr.u32 %r176, %r175, 30; shr.u32 %r177, %r175, 8; xor.b32 %r178, %r176, %r177;
  add.s32 %r179, %r178, 1; cvt.rn.f32.u32 %f1, %r179; mul.ftz.f32 %f2, %f1, 0f33800000;
  mad.lo.s32 %r182, %r169, -93469191, 1192405134;
  shr.u32 %r183, %r182, 28; add.s32 %r184, %r183, 4; shr.u32 %r185, %r182, %r184; xor.b32 %r186, %r185, %r182;
  mul.lo.s32 %r187, %r186, 277803737; shr.u32 %r188, %r187, 30; shr.u32 %r189, %r187, 8; xor.b32 %r190, %r188, %r189;
  add.s32 %r191, %r190, 1; cvt.rn.f32.u32 %f3, %r191; mul.ftz.f32 %f4, %f3, 0f33800000;
  lg2.approx.ftz.f32 %f5, %f2; mul.ftz.f32 %f6, %f5, 0f3F317218; mul.ftz.f32 %f7, %f6, 0fC0000000; sqrt.approx.ftz.f32 %f8, %f7;
  mul.ftz.f32 %f9, %f4, 0f40C90FDB; cos.approx.ftz.f32 %f10, %f9; mul.ftz.f32 %f11, %f8, %f10;
  mul.wide.u32 %rd2, %r6, 4; add.s64 %rd3, %rd1, %rd2; st.global.f32 [%rd3], %f11;
DONE:
  ret;
}
"""


def main():
    klab.ctx()
    m = ctypes.c_void_p()
    klab.ck(klab.cu.cuModuleLoadData(ctypes.byref(m), PTX.encode() + b"\0"))
    f = ctypes.c_void_p()
    klab.ck(klab.cu.cuModuleGetFunction(ctypes.byref(f), m, b"nz"))
    H, W, frame = 64, 64, 2
    out = torch.zeros(H * W, dtype=torch.float32, device="cuda")
    args = [ctypes.c_uint64(out.data_ptr()), ctypes.c_uint32(W), ctypes.c_uint32(H), ctypes.c_uint32(frame)]
    ptrs = (ctypes.c_void_p * 4)(*[ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in args])
    klab.ck(klab.cu.cuLaunchKernel(f, (H * W + 255) // 256, 1, 1, 256, 1, 1, 0, None, ptrs, None))
    klab.ck(klab.cu.cuCtxSynchronize())
    g = out.cpu().numpy().reshape(H, W)
    ref = N.noise(H, W, frame)[0]
    print("GPU 原样指令 vs noise.py: 最大差 %.5f  相关 %.5f" % (np.abs(g - ref).max(), np.corrcoef(g.ravel(), ref.ravel())[0, 1]))
    print("GPU 前 4 个:", g[0, :4], " numpy:", ref[0, :4])


if __name__ == "__main__":
    main()
