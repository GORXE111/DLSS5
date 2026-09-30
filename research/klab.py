"""kernel 实验室: 在 Python 里直接加载 DLSS5 的 sm_86 cubin，自己填参数、自己给输入/权重来跑单个 kernel。

参数块布局取自执行轨迹 (nr-trace.tsv) 的原始字节，只把其中的指针换成我们自己分配的显存。
用途: 与窃听数据逐字节对照验证；改权重/改输入做可控实验，反推每段权重的作用。
"""
import ctypes
import os
import struct

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
cu = ctypes.WinDLL("nvcuda.dll")


def ck(r):
    if r:
        s = ctypes.c_char_p()
        cu.cuGetErrorString(r, ctypes.byref(s))
        raise RuntimeError(s.value.decode())


_ctx = None
_mods = {}


def ctx():
    global _ctx
    if _ctx is None:
        torch.zeros(1, device="cuda")
        c = ctypes.c_void_p()
        ck(cu.cuDevicePrimaryCtxRetain(ctypes.byref(c), 0))
        ck(cu.cuCtxSetCurrent(c))
        _ctx = c
    return _ctx


def function(fatbin, name):
    ctx()
    if fatbin not in _mods:
        data = open(os.path.join(ROOT, "sm86_port", f"fatbin_{fatbin}.sm86.cubin"), "rb").read()
        m = ctypes.c_void_p()
        ck(cu.cuModuleLoadData(ctypes.byref(m), data))
        _mods[fatbin] = (m, data)
    f = ctypes.c_void_p()
    ck(cu.cuModuleGetFunction(ctypes.byref(f), _mods[fatbin][0], name.encode()))
    return f


def trace_row(seq, path=os.path.join(ROOT, "research", "taps", "nr-trace.tsv")):
    for line in open(path):
        r = line.rstrip("\n").split("\t")
        if int(r[0]) == seq:
            return dict(seq=seq, kernel=r[2], grid=tuple(map(int, r[3].split(","))),
                        block=tuple(map(int, r[4].split(","))), smem=int(r[5]), params=bytes.fromhex(r[7]))
    raise KeyError(seq)


def launch(fn, grid, block, smem, params: bytes):
    """kernel 只有一个结构体参数 (param_0)，按原始字节传入"""
    buf = ctypes.create_string_buffer(params, len(params))
    ptrs = (ctypes.c_void_p * 1)(ctypes.cast(buf, ctypes.c_void_p))
    if smem:
        ck(cu.cuFuncSetAttribute(fn, 8, smem))  # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
    ck(cu.cuLaunchKernel(fn, *grid, *block, smem, None, ptrs, None))
    ck(cu.cuCtxSynchronize())


def patch_ptrs(params: bytes, mapping: dict) -> bytes:
    """把参数块里等于旧指针的 64 位字替换成新指针"""
    b = bytearray(params)
    for i in range(0, len(b) // 8 * 8, 8):
        v = struct.unpack_from("<Q", b, i)[0]
        if v in mapping:
            struct.pack_into("<Q", b, i, mapping[v])
    return bytes(b)


class _ResPitch2D(ctypes.Structure):
    _fields_ = [("devPtr", ctypes.c_uint64), ("format", ctypes.c_int), ("numChannels", ctypes.c_uint),
                ("width", ctypes.c_size_t), ("height", ctypes.c_size_t), ("pitchInBytes", ctypes.c_size_t)]


class _ResUnion(ctypes.Union):
    _fields_ = [("pitch2D", _ResPitch2D), ("reserved", ctypes.c_int * 32)]


class _ResDesc(ctypes.Structure):
    _fields_ = [("resType", ctypes.c_int), ("res", _ResUnion), ("flags", ctypes.c_uint)]


class _TexDesc(ctypes.Structure):
    _fields_ = [("addressMode", ctypes.c_int * 3), ("filterMode", ctypes.c_int), ("flags", ctypes.c_uint),
                ("maxAnisotropy", ctypes.c_uint), ("mipmapFilterMode", ctypes.c_int), ("mipmapLevelBias", ctypes.c_float),
                ("minMipmapLevelClamp", ctypes.c_float), ("maxMipmapLevelClamp", ctypes.c_float),
                ("borderColor", ctypes.c_float * 4), ("reserved", ctypes.c_int * 12)]


def texture(img, linear=True, normalized=True, clamp=True):
    """img: (H, W, C) float32，C<=4 (不足补到 4)。返回 (纹理句柄 int, 保活对象)。
    pre_block 用 tex.2d.v4.f32.f32 + 归一化坐标；历史的 5-tap Catmull-Rom 依赖硬件双线性过滤"""
    ctx()
    img = np.asarray(img, np.float32)
    if img.ndim == 2:
        img = img[:, :, None]
    H, W, C = img.shape
    full = np.zeros((H, W, 4), np.float32)
    full[:, :, :C] = img
    pitch = (W * 16 + 255) // 256 * 256
    t = torch.zeros((H, pitch // 4), dtype=torch.float32, device="cuda")
    t[:, : W * 4] = torch.from_numpy(full.reshape(H, W * 4)).cuda()
    rd = _ResDesc()
    rd.resType = 3                                     # CU_RESOURCE_TYPE_PITCH2D
    rd.res.pitch2D = _ResPitch2D(t.data_ptr(), 0x20, 4, W, H, pitch)   # CU_AD_FORMAT_FLOAT
    td = _TexDesc()
    td.addressMode[0] = td.addressMode[1] = td.addressMode[2] = 1 if clamp else 0
    td.filterMode = 1 if linear else 0
    td.flags = 2 if normalized else 0                  # CU_TRSF_NORMALIZED_COORDINATES
    h = ctypes.c_uint64()
    ck(cu.cuTexObjectCreate(ctypes.byref(h), ctypes.byref(rd), ctypes.byref(td), None))
    return h.value, t


def gpu_bytes(data: bytes, pad=0):
    t = torch.zeros(len(data) + pad, dtype=torch.uint8, device="cuda")
    if data:
        t[: len(data)] = torch.frombuffer(bytearray(data), dtype=torch.uint8).cuda()
    return t


class _Array3DDesc(ctypes.Structure):
    _fields_ = [("Width", ctypes.c_size_t), ("Height", ctypes.c_size_t), ("Depth", ctypes.c_size_t),
                ("Format", ctypes.c_int), ("NumChannels", ctypes.c_uint), ("Flags", ctypes.c_uint)]


class _Memcpy2D(ctypes.Structure):
    _fields_ = [("srcXInBytes", ctypes.c_size_t), ("srcY", ctypes.c_size_t), ("srcMemoryType", ctypes.c_int),
                ("srcHost", ctypes.c_void_p), ("srcDevice", ctypes.c_uint64), ("srcArray", ctypes.c_void_p),
                ("srcPitch", ctypes.c_size_t), ("dstXInBytes", ctypes.c_size_t), ("dstY", ctypes.c_size_t),
                ("dstMemoryType", ctypes.c_int), ("dstHost", ctypes.c_void_p), ("dstDevice", ctypes.c_uint64),
                ("dstArray", ctypes.c_void_p), ("dstPitch", ctypes.c_size_t), ("WidthInBytes", ctypes.c_size_t),
                ("Height", ctypes.c_size_t)]


class Surface:
    """float4 可写 surface (post_block 用 sust.p.2d.v4.b32 写最终画面)。.handle 填进参数块，.read() 取回 (H, W, 4)"""

    def __init__(self, W, H):
        ctx()
        self.W, self.H = W, H
        d = _Array3DDesc(W, H, 0, 0x20, 4, 0x02)          # FLOAT x4, CUDA_ARRAY3D_SURFACE_LDST
        self.arr = ctypes.c_void_p()
        ck(cu.cuArray3DCreate_v2(ctypes.byref(self.arr), ctypes.byref(d)))
        rd = _ResDesc()
        rd.resType = 0                                      # CU_RESOURCE_TYPE_ARRAY
        ctypes.memmove(ctypes.addressof(rd.res), ctypes.byref(ctypes.c_uint64(self.arr.value)), 8)
        h = ctypes.c_uint64()
        ck(cu.cuSurfObjectCreate(ctypes.byref(h), ctypes.byref(rd)))
        self.handle = h.value

    def _copy(self, host, to_host):
        m = _Memcpy2D()
        if to_host:
            m.srcMemoryType, m.srcArray, m.dstMemoryType = 3, self.arr, 1   # ARRAY -> HOST
            m.dstHost, m.dstPitch = host.ctypes.data, self.W * 16
        else:
            m.srcMemoryType, m.srcHost, m.srcPitch = 1, host.ctypes.data, self.W * 16
            m.dstMemoryType, m.dstArray = 3, self.arr
        m.WidthInBytes, m.Height = self.W * 16, self.H
        ck(cu.cuMemcpy2D_v2(ctypes.byref(m)))

    def fill(self, v):
        self._copy(np.full((self.H, self.W, 4), v, np.float32), False)

    def read(self):
        out = np.zeros((self.H, self.W, 4), np.float32)
        self._copy(out, True)
        return out
