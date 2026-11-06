"""Isolated FP16 m16n8k16 matrix tiles on sm86. Running the module checks layout on CPU."""

import argparse
from contextlib import ExitStack
import ctypes as C

import numpy as np

from .probe_october_fence import Driver


M, N, K, LANES = 16, 8, 16, 32

# NVIDIA PTX ISA, Matrix Fragments for mma.m16n8k16 with floating point type.
# https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16816-float
_lane, _slot = np.ogrid[:LANES, :8]
_group, _thread = _lane // 4, _lane % 4
_a_row = _group + 8 * (_slot % 4 >= 2)
_a_col = 2 * _thread + _slot % 2 + 8 * (_slot >= 4)
_b_row = 2 * _thread + _slot[:, :4] % 2 + 8 * (_slot[:, :4] >= 2)
_d_row = _group + 8 * (_slot[:, :4] >= 2)
_d_col = 2 * _thread + _slot[:, :4] % 2

PTX = b"""
.version 8.0
.target sm_86
.address_size 64
.visible .entry mma_tile(.param .u64 src, .param .u64 dst)
{
    .reg .b32 %r<13>;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd0, [src];
    ld.param.u64 %rd1, [dst];
    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %laneid;
    mad.lo.u32 %r2, %r0, 32, %r1;
    mul.wide.u32 %rd2, %r2, 32;
    add.u64 %rd2, %rd0, %rd2;
    mul.wide.u32 %rd3, %r2, 8;
    add.u64 %rd3, %rd1, %rd3;
    ld.global.v4.b32 {%r3, %r4, %r5, %r6}, [%rd2];
    ld.global.v4.b32 {%r7, %r8, %r9, %r10}, [%rd2+16];
    mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
        {%r11, %r12}, {%r3, %r4, %r5, %r6}, {%r7, %r8}, {%r9, %r10};
    st.global.v2.b32 [%rd3], {%r11, %r12};
    ret;
}
"""


def _half_array(value):
    value = np.asarray(value)
    if value.dtype.kind not in "bifu":
        raise TypeError("MMA tiles require real numeric arrays")
    return value.astype("<f2", copy=False)


def pack_fragments(a, b, c=None):
    """Pack A[H,M,16], B[H,16,N] and optional C[H,M,N] as eight uint32 words per lane."""
    a, b = _half_array(a), _half_array(b)
    if a.ndim != 3 or b.ndim != 3 or a.shape[2] != K or b.shape[1] != K or a.shape[0] != b.shape[0]:
        raise ValueError("expected A[H,M,16] and B[H,16,N] with equal H")
    heads, rows, cols = a.shape[0], a.shape[1], b.shape[2]
    if rows % M or cols % N:
        raise ValueError("M must be a multiple of 16 and N a multiple of 8")
    if c is not None:
        c = _half_array(c)
        if c.shape != (heads, rows, cols):
            raise ValueError("expected C[H,M,N]")
    row_tiles, col_tiles = rows // M, cols // N
    tile_count = heads * row_tiles * col_tiles
    if tile_count > (1 << 32) // LANES:
        raise ValueError("tile count exceeds the kernel's lane index range")
    a_fragments = a.reshape(heads, row_tiles, M, K)[..., _a_row, _a_col]
    b_tiles = b.reshape(heads, K, col_tiles, N).transpose(0, 2, 1, 3)
    b_fragments = b_tiles[..., _b_row, _group]
    words = np.empty((heads, row_tiles, col_tiles, LANES, 8), dtype="<u4")
    halves = words.view("<f2")
    halves[..., :8] = a_fragments[:, :, None]
    halves[..., 8:12] = b_fragments[:, None]
    if c is None:
        halves[..., 12:] = 0
    else:
        c_tiles = c.reshape(heads, row_tiles, M, col_tiles, N).transpose(0, 1, 3, 2, 4)
        halves[..., 12:] = c_tiles[..., _d_row, _d_col]
    return words


def unpack_fragments(words, shape):
    """Scatter two D words per lane into an FP16 array of the specified [H,M,N] shape."""
    heads, rows, cols = shape
    if min(shape) < 0 or rows % M or cols % N:
        raise ValueError("invalid D shape")
    row_tiles, col_tiles = rows // M, cols // N
    words = np.asarray(words)
    if words.dtype != np.dtype("<u4") or words.shape != (heads, row_tiles, col_tiles, LANES, 2):
        raise ValueError("unexpected D fragment shape or type")
    halves = np.ascontiguousarray(words).view("<f2")
    matrix = np.empty(shape, dtype="<f2")
    tiles = matrix.reshape(heads, row_tiles, M, col_tiles, N).transpose(0, 1, 3, 2, 4)
    tiles[..., _d_row, _d_col] = halves
    return matrix


class Sm86Mma:
    """Synchronous, same-thread research backend. Use as a context manager."""

    def __init__(self, ordinal=0):
        self.ordinal = ordinal
        self._resources = None

    def __enter__(self):
        if self._resources is not None:
            raise RuntimeError("MMA backend is already open")
        with ExitStack() as resources:
            device = Driver(self.ordinal)
            resources.callback(device.close)
            copy = device.lib.cuMemcpyHtoD_v2
            copy.argtypes, copy.restype = [C.c_uint64, C.c_void_p, C.c_size_t], C.c_int
            get_current = device.lib.cuCtxGetCurrent
            get_current.argtypes, get_current.restype = [C.POINTER(C.c_void_p)], C.c_int
            module = device.load(PTX)
            resources.callback(device.call, "cuModuleUnload", module)
            self._function = device.function(module, "mma_tile")
            self._device = device
            self._resources = resources.pop_all()
        return self

    def __exit__(self, *exc):
        resources, self._resources = self._resources, None
        return resources.__exit__(*exc)

    @property
    def hardware_info(self):
        if self._resources is None:
            raise RuntimeError("MMA backend must be opened with a context manager")
        device = self._device
        return {"gpu": device.label, "compute_capability": device.capability,
                "cuda_driver_api_version": device.version}

    def __call__(self, a, b, c=None):
        if self._resources is None:
            raise RuntimeError("MMA backend must be opened with a context manager")
        device = self._device
        current = C.c_void_p()
        device.call("cuCtxGetCurrent", C.byref(current))
        if current.value != device.ctx.value:
            raise RuntimeError("MMA backend's CUDA context is not current on this thread")
        packed = pack_fragments(a, b, c)
        heads, row_tiles, col_tiles = packed.shape[:3]
        shape = heads, row_tiles * M, col_tiles * N
        result = np.empty(packed.shape[:-1] + (2,), dtype="<u4")
        tile_count = heads * row_tiles * col_tiles
        if tile_count:
            with ExitStack() as buffers:
                src, dst = C.c_uint64(), C.c_uint64()
                for pointer, size in ((src, packed.nbytes), (dst, result.nbytes)):
                    device.call("cuMemAlloc_v2", C.byref(pointer), size)
                    buffers.callback(device.call, "cuMemFree_v2", pointer)
                device.call("cuMemcpyHtoD_v2", src, packed.ctypes.data, packed.nbytes)
                args = (C.c_void_p * 2)(C.addressof(src), C.addressof(dst))
                device.call("cuLaunchKernel", self._function, tile_count, 1, 1, LANES, 1, 1, 0, None, args, None)
                device.call("cuCtxSynchronize")
                device.call("cuMemcpyDtoH_v2", result.ctypes.data, dst, result.nbytes)
        return unpack_fragments(result, shape)


def cpu_self_check():
    a = np.arange(2 * 32 * K, dtype=np.float16).reshape(2, 32, K)
    b = np.arange(2 * K * 24, dtype=np.float16).reshape(2, 24, K).transpose(0, 2, 1)
    c = np.arange(2 * 32 * 24, dtype=np.float16).reshape(2, 32, 24)
    c[0, 0, 0] = -0.0
    packed = pack_fragments(a, b, c)
    halves = packed.view("<f2")
    # One interior tile with explicit coordinates for the first, middle and last lanes.
    tile = halves[1, 1, 2]
    np.testing.assert_array_equal(tile[0, :8], a[1, [16, 16, 24, 24, 16, 16, 24, 24], [0, 1, 0, 1, 8, 9, 8, 9]])
    np.testing.assert_array_equal(tile[10, 8:12], b[1, [4, 5, 12, 13], 18])
    np.testing.assert_array_equal(tile[31, 12:], c[1, [23, 23, 31, 31], [22, 23, 22, 23]])
    d = unpack_fragments(packed[..., 6:], c.shape)
    np.testing.assert_array_equal(d.view("<u2"), c.view("<u2"))
    np.testing.assert_array_equal(pack_fragments(a, b)[..., 6:], np.zeros_like(packed[..., 6:]))


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    cpu_self_check()
    print("CPU FP16 MMA fragment layout check passed")


if __name__ == "__main__":
    main()
