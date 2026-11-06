"""Compare the QKV FP16 inverse norm to the actual sm86 rsqrt instruction."""

import argparse
from contextlib import ExitStack
import ctypes as C
from fractions import Fraction
import json
from pathlib import Path
import sys

import numpy as np

from .check_october_attention import sha
from .check_october_attention_reciprocal import CLAMP_F32_BITS, LAST_FINITE_F16, f16_code, f16_value
from .probe_october_fence import Driver


FIRST = 0x0410
COUNT = LAST_FINITE_F16 - FIRST + 1
THREADS = 128
PTX = f"""
.version 8.0
.target sm_86
.address_size 64
.visible .entry half_rsqrt(.param .u64 out)
{{
    .reg .pred %p;
    .reg .b16 %h;
    .reg .b32 %cta, %thread, %i, %code;
    .reg .f32 %f;
    .reg .b64 %base, %address;
    mov.u32 %cta, %ctaid.x;
    mov.u32 %thread, %tid.x;
    mad.lo.u32 %i, %cta, {THREADS}, %thread;
    setp.ge.u32 %p, %i, {COUNT};
    @%p ret;
    add.u32 %code, %i, {FIRST};
    cvt.u16.u32 %h, %code;
    cvt.f32.f16 %f, %h;
    rsqrt.approx.ftz.f32 %f, %f;
    cvt.rn.f16.f32 %h, %f;
    ld.param.u64 %base, [out];
    mul.wide.u32 %address, %i, 2;
    add.u64 %address, %base, %address;
    st.global.b16 [%address], %h;
    ret;
}}
""".encode("ascii")


def measure(ordinal):
    result = np.empty(COUNT, dtype="<u2")
    with ExitStack() as cleanup:
        device = Driver(ordinal)
        cleanup.callback(device.close)
        module = device.load(PTX)
        cleanup.callback(device.call, "cuModuleUnload", module)
        output = C.c_uint64()
        device.call("cuMemAlloc_v2", C.byref(output), result.nbytes)
        cleanup.callback(device.call, "cuMemFree_v2", output)
        args = (C.c_void_p * 1)(C.addressof(output))
        device.call("cuLaunchKernel", device.function(module, "half_rsqrt"), (COUNT + THREADS - 1) // THREADS,
                    1, 1, THREADS, 1, 1, 0, None, args, None)
        device.call("cuCtxSynchronize")
        device.call("cuMemcpyDtoH_v2", result.ctypes.data, output, result.nbytes)
        hardware = {"gpu": device.label, "compute_capability": device.capability, "cuda_driver_api_version": device.version}
    return result, hardware


def rounding_bound(codes):
    # NVIDIA bounds relative error by 2**-22.9. Use the looser rational 2**-22.
    # Compare squared positive endpoints to avoid evaluating an irrational value.
    relative = Fraction(1, 2 ** 22)
    low, high = (1 - relative) ** 2, (1 + relative) ** 2
    candidates = []
    exact = []
    for code in codes:
        operand = f16_value(int(code))
        nominal = f16_code(1 / np.sqrt(float(operand)))
        midpoint_low = (f16_value(nominal - 1) + f16_value(nominal)) / 2
        midpoint_high = (f16_value(nominal) + f16_value(nominal + 1)) / 2
        below = operand * midpoint_low ** 2
        above = operand * midpoint_high ** 2
        if not (below <= 1 <= above) or (nominal % 2 and 1 in (below, above)):
            raise ValueError("host FP64 failed the exact FP16 midpoint check")
        exact.append(nominal)
        if below >= low or above <= high:
            candidates.append(int(code))
    return np.array(exact, dtype="<u2"), candidates


def run(ordinal):
    codes = np.arange(FIRST, LAST_FINITE_F16 + 1, dtype="<u2")
    half = codes.view("<f2").astype(np.float32)
    host = (np.float32(1) / np.sqrt(half)).astype("<f2").view("<u2")
    exact, candidates = rounding_bound(codes)
    actual, hardware = measure(ordinal)
    differing = (actual != host) | (actual != exact)
    return {"schema": "october_nr_qkv_rsqrt_v1", "source_sha256": sha(Path(__file__).read_bytes()),
            "ptx_sha256": sha(PTX), "ptx_spec": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-rsqrt",
            **hardware, "clamp_f32_bits": f"0x{CLAMP_F32_BITS:08x}", "finite_inputs": COUNT,
            "tested_f16_codes_inclusive": [f"0x{FIRST:04x}", f"0x{LAST_FINITE_F16:04x}"],
            "conservative_relative_error_bound": "2^-22",
            "denominators_near_midpoint_under_bound": [f"0x{x:04x}" for x in candidates],
            "gpu_vs_host_f32_different_words": int(np.count_nonzero(actual != host)),
            "gpu_vs_exact_rne_different_words": int(np.count_nonzero(actual != exact)),
            "differences": [{"input": f"0x{a:04x}", "gpu": f"0x{b:04x}", "host_f32": f"0x{c:04x}", "exact_rne": f"0x{d:04x}"}
                            for a, b, c, d in zip(codes[differing], actual[differing], host[differing], exact[differing])]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.device < 0:
        parser.error("device must be nonnegative")
    try:
        report = run(args.device)
    except (OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"error": str(error)}, indent=2))
        return 1
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
