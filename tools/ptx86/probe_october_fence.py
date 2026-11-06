"""Run an authored sm86 producer/consumer probe for the October ViT fence.

No NR DLL is loaded. The optional previously compiled module is only loaded
and queried for symbols; its kernels are never launched. This finite stress
test cannot establish equivalence for the production launch schedule.
"""

import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

from tools.compare_nr_kernels import elf_text

ORIGINAL_PTX_SHA = "3dc57ec3e2a68965129f831f889ddc451ba213cc8e0eb271af90668a25891ac7"
TRIAL_CUBIN_SHA = "61c9b4686e760369137608307641cff64c53e62960e26cbb2d5d916cca583183"
LANES = 32
SEEDS = (0x00C0FFEE, 0x87654321, 0x10293847, 0xFEEDFACE)
MODES = {"acq_rel": "fence.acq_rel.gpu;", "sc": "fence.sc.gpu;"}

# Two resident CTAs exchange one tile at a time. The elected reader polls a
# relaxed, non-allocating flag; the whole CTA crosses a barrier and then
# executes the tested fence before loading data. An acknowledgement prevents
# the producer from overwriting the tile until all readers finish that epoch.
PROBE_PTX = r"""
.version 8.0
.target sm_86
.address_size 64
.visible .entry exchange(
    .param .u64 tile, .param .u64 control, .param .u64 observed,
    .param .u32 seed, .param .u32 rounds, .param .u32 spin_limit, .param .u32 inject
) {
    .reg .pred %p<10>;
    .reg .u32 %lane, %cta, %seed, %n, %epoch, %limit, %mode, %value, %poll, %status, %dummy, %spins;
    .reg .u64 %tile, %control, %result, %lane_offset, %address, %row, %error_address;
    .shared .align 4 .u32 mailbox;
    ld.param.u64 %tile, [tile];
    ld.param.u64 %control, [control];
    ld.param.u64 %result, [observed];
    ld.param.u32 %seed, [seed];
    ld.param.u32 %n, [rounds];
    ld.param.u32 %limit, [spin_limit];
    ld.param.u32 %mode, [inject];
    mov.u32 %lane, %tid.x;
    mov.u32 %cta, %ctaid.x;
    mul.wide.u32 %lane_offset, %lane, 4;
    add.u64 %tile, %tile, %lane_offset;
    setp.eq.u32 %p0, %lane, 0;
    mov.u32 %epoch, 1;
    setp.ne.u32 %p1, %cta, 0;
    @%p1 bra.uni consumer;

producer:
    mul.lo.u32 %value, %epoch, 2654435761;
    mul.lo.u32 %dummy, %lane, 2246822519;
    xor.b32 %value, %value, %dummy;
    xor.b32 %value, %value, %seed;
    setp.eq.u32 %p2, %mode, 1;
    setp.eq.u32 %p3, %epoch, 1;
    setp.eq.u32 %p4, %lane, 7;
    and.pred %p2, %p2, %p3;
    and.pred %p2, %p2, %p4;
    @%p2 xor.b32 %value, %value, 1;
    st.global.u32 [%tile], %value;
    bar.sync 0;
    @!%p0 bra prod_gate;
    setp.ne.u32 %p5, %mode, 2;
    @%p5 atom.release.gpu.global.add.u32 %dummy, [%control], 1;
    mov.u32 %spins, 0;
prod_wait:
    ld.acquire.gpu.global.u32 %poll, [%control+4];
    setp.ge.u32 %p6, %poll, %epoch;
    @%p6 bra prod_ready;
    add.u32 %spins, %spins, 1;
    setp.ge.u32 %p7, %spins, %limit;
    @%p7 bra prod_expired;
    nanosleep.u32 128;
    bra prod_wait;
prod_ready:
    st.shared.u32 [mailbox], 0;
    bra prod_gate;
prod_expired:
    st.shared.u32 [mailbox], 1;
    st.global.u32 [%control+8], 1;
prod_gate:
    bar.sync 0;
    ld.shared.u32 %status, [mailbox];
    setp.ne.u32 %p8, %status, 0;
    @%p8 bra.uni done;
    add.u32 %epoch, %epoch, 1;
    setp.le.u32 %p9, %epoch, %n;
    @%p9 bra.uni producer;
    bra.uni done;

consumer:
    @!%p0 bra cons_gate;
    mov.u32 %spins, 0;
cons_wait:
    ld.relaxed.gpu.global.L1::no_allocate.u32 %poll, [%control];
    setp.ge.u32 %p6, %poll, %epoch;
    @%p6 bra cons_ready;
    add.u32 %spins, %spins, 1;
    setp.ge.u32 %p7, %spins, %limit;
    @%p7 bra cons_expired;
    nanosleep.u32 128;
    bra cons_wait;
cons_ready:
    st.shared.u32 [mailbox], 0;
    bra cons_gate;
cons_expired:
    st.shared.u32 [mailbox], 1;
    st.global.u32 [%control+12], 1;
cons_gate:
    bar.sync 0;
    ld.shared.u32 %status, [mailbox];
    setp.ne.u32 %p8, %status, 0;
    @%p8 bra.uni done;
    __TEST_FENCE__
    ld.global.ca.u32 %value, [%tile];
    sub.u32 %dummy, %epoch, 1;
    mul.wide.u32 %row, %dummy, 128;
    add.u64 %address, %result, %row;
    add.u64 %address, %address, %lane_offset;
    st.global.u32 [%address], %value;
    bar.sync 0;
    @%p0 st.release.gpu.global.u32 [%control+4], %epoch;
    add.u32 %epoch, %epoch, 1;
    setp.le.u32 %p9, %epoch, %n;
    @%p9 bra.uni consumer;
done:
    ret;
}
"""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def site_inventory(path):
    data = path.read_bytes()
    if sha(data) != ORIGINAL_PTX_SHA:
        raise ValueError("module 6 PTX does not match the retained 310.9.16.0 evidence")
    name, previous, rows = None, "", []
    for number, line in enumerate(data.decode("ascii").splitlines(), 1):
        line = line.strip()
        match = re.match(r"\.visible\s+\.entry\s+([^\s(]+)", line)
        if match:
            name = match[1]
        if line == "fence.acquire.gpu;":
            rows.append({"entry": name, "line": number, "immediately_after_cta_barrier": previous.startswith("bar.sync ")})
        if line:
            previous = line
    if len(rows) != 21 or not all(row["immediately_after_cta_barrier"] for row in rows):
        raise ValueError("fence inventory does not match the intended probe shape")
    return {"ptx_sha256": sha(data), "sites": rows}


class Driver:
    def __init__(self, ordinal):
        if os.name != "nt":
            raise RuntimeError("This probe currently uses the Windows CUDA driver")
        lib = C.WinDLL("nvcuda.dll")
        ptr, integer, u32, u64, size = C.c_void_p, C.c_int, C.c_uint, C.c_uint64, C.c_size_t
        p = C.POINTER
        self.lib = lib
        args = {
            "cuInit": [u32], "cuDriverGetVersion": [p(integer)],
            "cuGetErrorString": [integer, p(C.c_char_p)],
            "cuDeviceGet": [p(integer), integer], "cuDeviceGetName": [ptr, integer, integer],
            "cuDeviceGetAttribute": [p(integer), integer, integer],
            "cuDevicePrimaryCtxRetain": [p(ptr), integer],
            "cuDevicePrimaryCtxRelease_v2": [integer], "cuCtxSetCurrent": [ptr], "cuCtxSynchronize": [],
            "cuModuleLoadData": [p(ptr), ptr], "cuModuleGetFunction": [p(ptr), ptr, C.c_char_p],
            "cuModuleUnload": [ptr], "cuMemAlloc_v2": [p(u64), size], "cuMemFree_v2": [u64],
            "cuMemsetD32_v2": [u64, u32, size], "cuMemcpyDtoH_v2": [ptr, u64, size],
            "cuLaunchKernel": [ptr] + [u32] * 7 + [ptr, p(ptr), ptr],
        }
        for name, values in args.items():
            fun = getattr(lib, name)
            fun.argtypes, fun.restype = values, integer
        self.call("cuInit", 0)
        self.device = integer()
        self.call("cuDeviceGet", C.byref(self.device), ordinal)
        label = C.create_string_buffer(256)
        self.call("cuDeviceGetName", label, len(label), self.device)
        self.label = label.value.decode()
        self.capability = []
        for attribute in (75, 76):
            out = integer()
            self.call("cuDeviceGetAttribute", C.byref(out), attribute, self.device)
            self.capability.append(out.value)
        if self.capability != [8, 6]:
            raise ValueError("This experiment specifically requires compute capability 8.6")
        version = integer()
        self.call("cuDriverGetVersion", C.byref(version))
        self.version = version.value
        self.ctx = ptr()
        self.call("cuDevicePrimaryCtxRetain", C.byref(self.ctx), self.device)
        try:
            self.call("cuCtxSetCurrent", self.ctx)
        except RuntimeError:
            self.call("cuDevicePrimaryCtxRelease_v2", self.device)
            raise

    def call(self, name, *args):
        code = getattr(self.lib, name)(*args)
        if code:
            message = C.c_char_p()
            self.lib.cuGetErrorString(code, C.byref(message))
            raise RuntimeError(f"{name}: {message.value.decode() if message.value else code}")

    def close(self):
        self.call("cuCtxSetCurrent", None)
        self.call("cuDevicePrimaryCtxRelease_v2", self.device)

    def load(self, data):
        module = C.c_void_p()
        self.call("cuModuleLoadData", C.byref(module), C.create_string_buffer(data))
        return module

    def function(self, module, name):
        fn = C.c_void_p()
        self.call("cuModuleGetFunction", C.byref(fn), module, name.encode())
        return fn

    def exchange(self, fn, seed, rounds, spin_limit, inject=0):
        pointers = []
        try:
            for words in (LANES, 4, rounds * LANES):
                ptr = C.c_uint64()
                self.call("cuMemAlloc_v2", C.byref(ptr), words * 4)
                pointers.append(ptr)
                self.call("cuMemsetD32_v2", ptr, 0, words)
            values = pointers + [C.c_uint(x) for x in (seed, rounds, spin_limit, inject)]
            params = (C.c_void_p * len(values))(*(C.addressof(value) for value in values))
            begin = time.monotonic()
            self.call("cuLaunchKernel", fn, 2, 1, 1, LANES, 1, 1, 0, None, params, None)
            self.call("cuCtxSynchronize")
            elapsed = time.monotonic() - begin
            control, observed = (C.c_uint * 4)(), (C.c_uint * (LANES * rounds))()
            self.call("cuMemcpyDtoH_v2", control, pointers[1], C.sizeof(control))
            self.call("cuMemcpyDtoH_v2", observed, pointers[2], C.sizeof(observed))
            differences = []
            count = 0
            for i, actual in enumerate(observed):
                epoch, lane = divmod(i, LANES)
                expected = (((epoch + 1) * 2654435761) ^ (lane * 2246822519) ^ seed) & 0xFFFFFFFF
                if actual != expected:
                    count += 1
                    if len(differences) < 4:
                        differences.append({"epoch": epoch + 1, "lane": lane, "expected": expected, "actual": actual})
            return {"rounds": rounds, "seed": seed, "inject": inject, "control": list(control),
                    "values": len(observed), "mismatches": count, "first_mismatches": differences,
                    "observed_sha256": sha(bytes(observed)), "seconds": elapsed}
        finally:
            for ptr in reversed(pointers):
                self.call("cuMemFree_v2", ptr)


def run(args):
    result = {
        "scope": "Authored two-CTA repeated tile handoff, not an NR inference or an original Blackwell comparison",
        "hardware": None, "ptx_inventory": site_inventory(args.october_ptx) if args.october_ptx else None,
        "compiled_module_symbol_check": None, "tests": [], "passed": False,
    }
    device = Driver(args.device)
    try:
        result["hardware"] = {"gpu": device.label, "compute_capability": device.capability,
                              "cuda_driver_api_version": device.version}
        if args.trial_cubin:
            contents = args.trial_cubin.read_bytes()
            if sha(contents) != TRIAL_CUBIN_SHA:
                raise ValueError("not the reviewed sm86 module 6 compilation trial")
            symbols = [name.removeprefix(".text.") for name in elf_text(args.trial_cubin)]
            if len(symbols) != 31:
                raise ValueError("module 6 code section count differs from compilation evidence")
            module = device.load(contents)
            try:
                for name in symbols:
                    device.function(module, name)
                result["compiled_module_symbol_check"] = {"sha256": sha(contents), "resolved": len(symbols), "launched": 0}
            finally:
                device.call("cuModuleUnload", module)
        for kind, fence in MODES.items():
            ptx = PROBE_PTX.replace("__TEST_FENCE__", fence).encode("ascii")
            module = device.load(ptx)
            try:
                fn = device.function(module, "exchange")
                for seed in SEEDS:
                    row = device.exchange(fn, seed, args.rounds, 50000)
                    row["passed"] = row["mismatches"] == 0 and row["control"] == [args.rounds, args.rounds, 0, 0]
                    result["tests"].append({"mode": kind, "test": "handoff", "probe_ptx_sha256": sha(ptx), **row})
                if kind == "acq_rel":
                    corrupt = device.exchange(fn, SEEDS[0], 2, 50000, 1)
                    corrupt["passed"] = (corrupt["control"] == [2, 2, 0, 0] and corrupt["mismatches"] == 1
                                         and corrupt["first_mismatches"][0]["epoch"] == 1
                                         and corrupt["first_mismatches"][0]["lane"] == 7)
                    result["tests"].append({"mode": kind, "test": "detect_injected_mismatch", **corrupt})
                    stalled = device.exchange(fn, SEEDS[0], 1, 2048, 2)
                    stalled["passed"] = stalled["control"] == [0, 0, 1, 1]
                    result["tests"].append({"mode": kind, "test": "bounded_missing_publication", **stalled})
            finally:
                device.call("cuModuleUnload", module)
        result["passed"] = len(result["tests"]) == 10 and all(row["passed"] for row in result["tests"])
        return result
    finally:
        device.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=2048)
    parser.add_argument("--october-ptx", type=Path)
    parser.add_argument("--trial-cubin", type=Path)
    args = parser.parse_args()
    if not 1 <= args.rounds <= 4096 or args.device < 0:
        parser.error("rounds must be 1–4096 and device nonnegative")
    try:
        report = run(args)
    except (OSError, ValueError, RuntimeError) as error:
        print(json.dumps({"passed": False, "error": str(error)}, indent=2))
        return 1
    report["probe_source_sha256"] = sha(Path(__file__).read_bytes())
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
