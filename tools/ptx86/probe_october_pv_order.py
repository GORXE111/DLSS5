"""Test the PV split order at two real zero-Q October attention outputs."""

import argparse
import ctypes as C
from contextlib import ExitStack
import json
from pathlib import Path
import sys

import numpy as np

from ..align_nr_small_trace import GRAPH, GRAPH_SHA, kernel_inventory, pinned_bytes
from ..compare_october_taps import PORT, seal_capture
from .check_october_attention import FP8, buffer_offsets, encode_fp8, probabilities
from .probe_october_fence import Driver, sha


PTX = b"""
.version 8.0
.target sm_86
.address_size 64
.visible .entry pv_order(.param .u64 input, .param .u64 output,
                         .param .b32 a_pair, .param .f32 denominator)
{
    .reg .b32 %r<9>;
    .reg .b64 %rd<4>;
    .reg .pred %p<3>;
    .reg .f32 %f<3>;
    .reg .b16 %inv;
    ld.param.u64 %rd0, [input];
    ld.param.u64 %rd1, [output];
    ld.param.b32 %r0, [a_pair];
    mov.u32 %r1, %tid.x;
    and.b32 %r2, %r1, 3;
    mul.wide.u32 %rd2, %r2, 4;
    add.u64 %rd0, %rd0, %rd2;
    setp.eq.u32 %p0, %r1, 0;
    mov.u32 %r3, 0;
    mov.u32 %r4, 0;
    mov.u32 %r5, 0;
PV:
    ld.global.b32 %r6, [%rd0];
    ld.global.b32 %r7, [%rd0+16];
    mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
        {%r3, %r4}, {%r0, %r0, %r0, %r0}, {%r6, %r7}, {%r3, %r4};
    @%p0 st.global.v2.b32 [%rd1], {%r3, %r4};
    add.u64 %rd0, %rd0, 32;
    add.u64 %rd1, %rd1, 8;
    add.u32 %r5, %r5, 1;
    setp.lt.u32 %p1, %r5, 6;
    @%p1 bra PV;
    ld.param.f32 %f0, [denominator];
    rcp.approx.ftz.f32 %f1, %f0;
    cvt.rn.f16.f32 %inv, %f1;
    mov.b32 %r8, {%inv, %inv};
    mul.rn.f16x2 %r3, %r3, %r8;
    mul.rn.f16x2 %r4, %r4, %r8;
    @%p0 st.global.v2.b32 [%rd1], {%r3, %r4};
    ret;
}
"""


def project(device, function, order, v, probability, denominator):
    columns = np.arange(32)
    indices = np.concatenate((columns[(columns & order) == 0], columns[(columns & order) != 0]))
    fragments = v.reshape(3, 32)[:, indices].astype("<f2").reshape(6, 16)
    payload = fragments.tobytes()
    word = int(np.array(probability, dtype="<f2").view("<u2"))
    readback = C.create_string_buffer(7 * 8)
    with ExitStack() as cleanup:
        inp, out = C.c_uint64(), C.c_uint64()
        for ptr, size in ((inp, len(payload)), (out, len(readback))):
            device.call("cuMemAlloc_v2", C.byref(ptr), size)
            cleanup.callback(device.call, "cuMemFree_v2", ptr)
        device.call("cuMemcpyHtoD_v2", inp, C.create_string_buffer(payload), len(payload))
        a_pair, denominator = C.c_uint32(word | (word << 16)), C.c_float(denominator)
        arguments = (C.c_void_p * 4)(*(C.addressof(x) for x in (inp, out, a_pair, denominator)))
        device.call("cuLaunchKernel", function, 1, 1, 1, 32, 1, 1, 0, None, arguments, None)
        device.call("cuCtxSynchronize")
        device.call("cuMemcpyDtoH_v2", readback, out, len(readback))
    values = np.frombuffer(readback.raw, "<f2").reshape(7, 4)
    if not np.all(values == values[:, :1]):
        raise ValueError("the repeated rows and columns did not give identical half accumulators")
    previous = np.r_[0.0, values[:5, 0].astype(np.float64)]
    exact_sum = previous + (fragments.astype(np.float64) * probability).sum(axis=-1)
    rounded = exact_sum.astype(np.float16)
    different_steps = [{"step": int(i) + 1, "previous_measured_half": float(previous[i]),
                        "exact_sum": float(exact_sum[i]), "single_rne_half": float(rounded[i]),
                        "hardware_half": float(values[i, 0])}
                       for i in np.flatnonzero(values[:6, 0] != rounded)]
    return {"k_bit": order.bit_length() - 1, "pv_half_accumulators": values[:-1, 0].astype(float).tolist(),
            "steps_different_from_single_rne": different_steps,
            "normalized_half": float(values[-1, 0]), "encoded_output": int(encode_fp8(values[-1, 0]))}


def run(capture):
    graph = json.loads(pinned_bytes(GRAPH, GRAPH_SHA))
    sealed, files = seal_capture(capture, kernel_inventory(), graph)
    if sealed["kind"] != PORT or sealed["frame"] != 0:
        raise ValueError("expected a real staged frame-zero capture")
    roles = [f"block11.attention.{name}" for name in ("in0", "in2", "out")]
    tensors = []
    for role, order in zip(roles, (0, 2, 3)):
        item = sealed["boundaries"][role]
        raw = files[role].read_bytes()
        if sha(raw) != item["sha256"] or item["shape_hwc"] != (8, 12, 1024):
            raise ValueError("expected intact 96-token taps from the known attention run")
        tensors.append(np.frombuffer(raw, np.uint8)[buffer_offsets(96)[order]])
    q, v, output = tensors
    p = float(probabilities(np.array(0, np.float32)))
    probability = float(FP8[encode_fp8(np.float32(p))])
    denominator = float(np.float16(p * 96))
    cases = []
    device = Driver(0)
    try:
        copy = device.lib.cuMemcpyHtoD_v2
        copy.argtypes, copy.restype = [C.c_uint64, C.c_void_p, C.c_size_t], C.c_int
        module = device.load(PTX)
        try:
            function = device.function(module, "pv_order")
            for head, token, channel in ((7, 10, 1), (28, 10, 29)):
                if np.any(FP8[q[head, token]] != 0):
                    raise ValueError("the selected query is not the known all-zero row")
                observed = int(output[head, token, channel])
                variants = [project(device, function, order, FP8[v[head, :, channel]], probability, denominator)
                            for order in (2, 8)]
                cases.append({"position": {"head": head, "token": token, "channel": channel},
                              "observed_output": observed, "variants": variants,
                              "physical_split_matches_observed": variants[1]["encoded_output"] == observed})
        finally:
            device.call("cuModuleUnload", module)
        return {"schema": "october_nr_pv_order_v1", "gpu": device.label,
                "capture_seal_sha256": sealed["capture_seal_sha256"],
                "source_sha256": sha(Path(__file__).read_bytes()), "probe_ptx_sha256": sha(PTX),
                "scope": "Authored uniform-Q PV fragment probe at two captured positions. The first is a split-order control; the second checks a rounding boundary.",
                "probability_e4m3_value": probability, "half_denominator": denominator, "cases": cases,
                "passed": cases[0]["variants"][0]["encoded_output"] != cases[0]["observed_output"]
                and cases[0]["physical_split_matches_observed"]}
    finally:
        device.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    args = parser.parse_args()
    try:
        report = run(args.capture)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        report = {"passed": False, "error": str(error)}
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
