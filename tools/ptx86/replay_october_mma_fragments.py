"""Replay the captured October PV MMA operands, one independent warp per step."""

import argparse
from contextlib import ExitStack
import ctypes as C
import json
from pathlib import Path
import re
import struct
import sys

from .probe_october_fence import Driver, sha


REWRITTEN_PTX_SHA = "f6bd97ebc3db9d9c190e731144005ec2bda23b147325bcdbf77f784007a88f29"
INSTRUMENTED_CUBIN_SHA = "2ca399d05d22492163eb08ad47b15e06a888e5e6b5ac501388ca235c1462fe49"
BLOCKS = (11, 18)
POINTS = ((7, 10, 1), (28, 10, 29))
STEPS, LANES, HEADER, RECORD = 8, 32, 64, 48
GUARD = bytes((i * 17 + 3) & 255 for i in range(256))
CANARY = b"\xa5"
PTX = b"""
.version 8.0
.target sm_86
.address_size 64
.visible .entry replay_mma(.param .u64 src, .param .u64 dst)
{
    .reg .b32 %r<13>;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd0, [src];
    ld.param.u64 %rd1, [dst];
    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %laneid;
    mad.lo.u32 %r2, %r0, 32, %r1;
    mul.wide.u32 %rd2, %r2, 48;
    add.u64 %rd2, %rd0, %rd2;
    mul.wide.u32 %rd3, %r2, 8;
    add.u64 %rd3, %rd1, %rd3;
    ld.global.v4.b32 {%r3, %r4, %r5, %r6}, [%rd2+64];
    ld.global.v4.b32 {%r7, %r8, %r9, %r10}, [%rd2+80];
    mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
        {%r11, %r12}, {%r3, %r4, %r5, %r6}, {%r7, %r8}, {%r9, %r10};
    st.global.v2.b32 [%rd3], {%r11, %r12};
    ret;
}
"""


def prepare(raw):
    count = len(POINTS) * STEPS * LANES
    if len(raw) != HEADER + count * RECORD:
        raise ValueError("unexpected fragment file size")
    replay = bytearray(raw)
    expected = bytearray()
    for index in range(count):
        offset = HEADER + index * RECORD
        words = struct.unpack_from("<12I", raw, offset)
        if 0xA5A5A5A5 in words[:10] or words[10:] != (0xA5A5A5A5,) * 2:
            raise ValueError(f"incomplete fragment or invalid canary at record {index}")
        observed = raw[offset + 32:offset + 40]
        expected.extend(observed)
        # The observed D words are kept off the device input.
        replay[offset + 32:offset + 40] = bytes(8)
    replay[:HEADER] = bytes(HEADER)
    return bytes(replay), bytes(expected)


def load_inputs(capture, report, report_sha):
    report_bytes = report.read_bytes()
    if sha(report_bytes) != report_sha:
        raise ValueError("unexpected stage report SHA-256")
    stage = json.loads(report_bytes)
    if stage["schema"] != "october_nr_attention_stages_v1" or stage["passed"] is not True:
        raise ValueError("expected a successful pinned stage report")
    if (stage["rewritten_ptx_sha256"] != REWRITTEN_PTX_SHA
            or stage["instrumented_cubin_sha256"] != INSTRUMENTED_CUBIN_SHA):
        raise ValueError("unexpected instrumented October module")
    for key in ("source_sha256", "capture_seal_sha256", "instrumented_ptx_sha256"):
        if not re.fullmatch(r"[a-f0-9]{64}", stage[key]):
            raise ValueError(f"invalid stage identity: {key}")
    cases = stage["cases"]
    if tuple(case["block"] for case in cases) != BLOCKS:
        raise ValueError("unexpected stage report blocks")
    inputs = []
    for case in cases:
        points = tuple((p["head"], p["token"], p["channel"]) for p in case["points"])
        if points != POINTS:
            raise ValueError("unexpected point ordering in stage report")
        raw = (capture / f"block{case['block']}-fragments.bin").read_bytes()
        if sha(raw) != case["debug_sha256"]:
            raise ValueError(f"fragment SHA-256 differs for block {case['block']}")
        replay, expected = prepare(raw)
        inputs.append((case, replay, expected))
    return stage, inputs


def execute(device, function, replay, expected):
    # Each output word starts different from its expected value.
    poison = bytes(byte ^ 255 for byte in expected)
    original = (GUARD + replay + GUARD, GUARD + poison + GUARD)
    with ExitStack() as cleanup:
        pointers = []
        for payload in original:
            ptr = C.c_uint64()
            device.call("cuMemAlloc_v2", C.byref(ptr), len(payload))
            cleanup.callback(device.call, "cuMemFree_v2", ptr)
            device.call("cuMemcpyHtoD_v2", ptr, C.create_string_buffer(payload), len(payload))
            pointers.append(ptr)
        args = [C.c_uint64(ptr.value + len(GUARD)) for ptr in pointers]
        params = (C.c_void_p * 2)(*(C.addressof(arg) for arg in args))
        device.call("cuLaunchKernel", function, len(POINTS) * STEPS, 1, 1, LANES, 1, 1, 0, None, params, None)
        device.call("cuCtxSynchronize")
        readbacks = []
        for ptr, payload in zip(pointers, original):
            result = C.create_string_buffer(len(payload))
            device.call("cuMemcpyDtoH_v2", result, ptr, len(payload))
            readbacks.append(result.raw)
    if readbacks[0] != original[0]:
        raise ValueError("GPU modified replay input or its guard")
    if readbacks[1][:len(GUARD)] != GUARD or readbacks[1][-len(GUARD):] != GUARD:
        raise ValueError("GPU modified an output guard")
    return readbacks[1][len(GUARD):-len(GUARD)]


def compare(expected, replayed):
    if len(expected) != len(replayed) or len(expected) != len(POINTS) * STEPS * LANES * 8:
        raise ValueError("unexpected MMA output length")
    points = []
    for point_index, (head, token, channel) in enumerate(POINTS):
        steps = []
        for step in range(STEPS):
            start = (point_index * STEPS + step) * LANES * 8
            differing_words = 0
            differing_lanes = []
            for lane in range(LANES):
                offset = start + lane * 8
                count = sum(expected[i:i + 4] != replayed[i:i + 4] for i in (offset, offset + 4))
                if count:
                    differing_lanes.append(lane)
                    differing_words += count
            steps.append({"step": step + 1, "differing_words": differing_words, "differing_lanes": differing_lanes})
        points.append({"head": head, "token": token, "channel": channel, "steps": steps})
    return points


def run(capture, report, report_sha, verify_only):
    stage, inputs = load_inputs(capture, report, report_sha)
    result = {
        "schema": "october_nr_mma_fragment_replay_v1",
        "scope": "fragment_input_validation" if verify_only else "independent_sm86_one_warp_per_captured_step",
        "source_sha256": sha(Path(__file__).read_bytes()),
        "stage_report_sha256": report_sha,
        "stage_source_sha256": stage["source_sha256"],
        "capture_seal_sha256": stage["capture_seal_sha256"],
        "instrumented_ptx_sha256": stage["instrumented_ptx_sha256"],
        "instrumented_cubin_sha256": stage["instrumented_cubin_sha256"],
        "authored_ptx_sha256": sha(PTX),
        "cases": [
            {"block": case["block"], "fragment_sha256": case["debug_sha256"],
             "replay_input_sha256": sha(replay), "expected_d_sha256": sha(expected)}
            for case, replay, expected in inputs
        ],
        "compared_d_words": 0,
        "passed": True,
    }
    if verify_only:
        return result
    with ExitStack() as cleanup:
        device = Driver(0)
        cleanup.callback(device.close)
        copy = device.lib.cuMemcpyHtoD_v2
        copy.argtypes, copy.restype = [C.c_uint64, C.c_void_p, C.c_size_t], C.c_int
        module = device.load(PTX)
        cleanup.callback(device.call, "cuModuleUnload", module)
        function = device.function(module, "replay_mma")
        result.update(gpu=device.label, capability=device.capability, driver_version=device.version)
        for output, (_, replay, expected) in zip(result["cases"], inputs):
            actual = execute(device, function, replay, expected)
            output.update(replayed_d_sha256=sha(actual), points=compare(expected, actual), passed=actual == expected)
            result["compared_d_words"] += len(actual) // 4
    result["passed"] = all(case["passed"] for case in result["cases"])
    return result


def sha_argument(value):
    if not re.fullmatch(r"[a-fA-F0-9]{64}", value):
        raise argparse.ArgumentTypeError("expected a 64-digit SHA-256")
    return value.lower()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True, help="instrumentation work directory")
    parser.add_argument("--report", type=Path, required=True, help="pinned stage report to read")
    parser.add_argument("--report-sha256", type=sha_argument, required=True, help="recorded SHA-256 of the stage report")
    parser.add_argument("--verify-only", action="store_true", help="validate files without using CUDA")
    args = parser.parse_args()
    try:
        result = run(args.capture, args.report, args.report_sha256, args.verify_only)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        result = {"schema": "october_nr_mma_fragment_replay_v1", "passed": False, "error": str(error)}
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
