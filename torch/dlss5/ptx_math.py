"""Small authored PTX math operations for the inference sampler.

CUDA uses the active device's RCP approximation, not an emulation table. CPU
uses the ordinary Torch reciprocal and does not promise CUDA numerical parity.
Modules stay alive until process exit because captured graphs retain functions.
"""
import ctypes
import os
import threading

import torch


_PTX = b""".version 6.0
.target sm_50
.address_size 64
.visible .entry reciprocal(.param .u64 dst, .param .u64 src, .param .u32 count) {
    .reg .b64 %out, %in, %offset;
    .reg .b32 %i, %block, %threads, %thread, %n;
    .reg .f32 %x, %y;
    .reg .pred %outside;
    ld.param.u64 %out, [dst];
    ld.param.u64 %in, [src];
    ld.param.u32 %n, [count];
    mov.u32 %block, %ctaid.x;
    mov.u32 %threads, %ntid.x;
    mov.u32 %thread, %tid.x;
    mad.lo.u32 %i, %block, %threads, %thread;
    setp.ge.u32 %outside, %i, %n;
    @%outside bra done;
    mul.wide.u32 %offset, %i, 4;
    add.u64 %in, %in, %offset;
    add.u64 %out, %out, %offset;
    ld.global.f32 %x, [%in];
    rcp.approx.ftz.f32 %y, %x;
    st.global.f32 [%out], %y;
done:
    ret;
}
"""

_driver = None
_modules = {}
_lock = threading.Lock()


def _check(driver, result):
    if result:
        message = ctypes.c_char_p()
        driver.cuGetErrorString(result, ctypes.byref(message))
        detail = message.value.decode() if message.value else str(result)
        raise RuntimeError("CUDA reciprocal: " + detail)


def _library():
    global _driver
    if _driver is None:
        driver = ctypes.WinDLL("nvcuda.dll") if os.name == "nt" else ctypes.CDLL("libcuda.so.1")
        signatures = {
            "cuCtxGetCurrent": [ctypes.POINTER(ctypes.c_void_p)],
            "cuModuleLoadData": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p],
            "cuModuleGetFunction": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p],
            "cuLaunchKernel": [ctypes.c_void_p] + [ctypes.c_uint] * 7 +
                              [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p],
            "cuGetErrorString": [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)],
        }
        for name, args in signatures.items():
            function = getattr(driver, name)
            function.argtypes, function.restype = args, ctypes.c_int
        _driver = driver
    return _driver


def _function():
    # Called within torch.cuda.device(input.device); never creates another CUDA
    # context or changes Torch's active stream. Compilation is eager-only.
    driver = _library()
    context = ctypes.c_void_p()
    _check(driver, driver.cuCtxGetCurrent(ctypes.byref(context)))
    if context.value is None:
        raise RuntimeError("CUDA reciprocal requires Torch's active CUDA context")
    with _lock:
        if context.value not in _modules:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm up rcp_approx on this CUDA context before graph capture")
            module, function = ctypes.c_void_p(), ctypes.c_void_p()
            _check(driver, driver.cuModuleLoadData(ctypes.byref(module), ctypes.c_char_p(_PTX)))
            _check(driver, driver.cuModuleGetFunction(ctypes.byref(function), module, b"reciprocal"))
            _modules[context.value] = (module, function)
        return driver, _modules[context.value][1]


def rcp_approx(x):
    """RCP.APPROX.FTZ on the current CUDA stream; CPU fallback is ordinary RN.

    This inference-only primitive accepts float32 tensors. It JITs one tiny PTX
    module per CUDA context during warmup, with no nvcc or extra Python package.
    Warmed calls allocate through Torch and launch on its current stream, so they
    can be recorded in a CUDA Graph. CUDA errors never select the CPU fallback.
    """
    if x.dtype != torch.float32:
        raise TypeError("rcp_approx requires float32")
    if x.requires_grad and torch.is_grad_enabled():
        raise RuntimeError("rcp_approx is an inference-only operation")
    if x.device.type == "cpu":
        return x.reciprocal()
    if x.device.type != "cuda":
        raise ValueError("rcp_approx supports CPU and CUDA tensors")
    if x.numel() > 0xFFFFFFFF:
        raise ValueError("rcp_approx input exceeds its 32-bit launch count")
    with torch.cuda.device(x.device):
        x = x.contiguous()
        output = torch.empty_like(x)
        if not x.numel():
            return output
        driver, function = _function()
        stream = torch.cuda.current_stream(x.device)
        destination, source = ctypes.c_uint64(output.data_ptr()), ctypes.c_uint64(x.data_ptr())
        count = ctypes.c_uint32(x.numel())
        args = (ctypes.c_void_p * 3)(ctypes.addressof(destination), ctypes.addressof(source), ctypes.addressof(count))
        _check(driver, driver.cuLaunchKernel(function, (x.numel() + 127) // 128, 1, 1, 128, 1, 1,
                                            0, ctypes.c_void_p(stream.cuda_stream), args, None))
        x.record_stream(stream)
        return output
