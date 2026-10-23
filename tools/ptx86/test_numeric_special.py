"""CPU bit-level references and execution checks for emitted FP8 conversion PTX.

The oracle enumerates exact integer values in units of 2**-24, independently
of the rewrite's bit tricks. The tiny interpreter executes emitted straight-line
PTX, not a separately transcribed conversion formula. NaN sign/payload is not
an acceptance requirement; finite values, infinities and signed zero are exact.
No CUDA context, tools, torch conversions, or GPU are used.
"""
import argparse
import bisect
import hashlib
import json
from pathlib import Path
import re
import unittest
from unittest.mock import patch

import numpy as np

from . import fuse, rewrite as rw


SOURCES = [
    {"url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#alternate-floating-point-data-formats",
     "contract": "PTX E4M3 has no infinity; only 0x7f/0xff are NaN. E5M2 has 5 exponent and 2 mantissa bits."},
    {"url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt",
     "contract": "cvt .rn ties to even; .satfinite preserves NaN class and saturates out-of-range magnitudes to sign-preserved MAX_NORM."},
    {"url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions",
     "contract": "Half add/mul without .ftz support subnormals."},
    {"url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma",
     "contract": "MMA accumulation order, rounding and handling of subnormal inputs are unspecified; conversion checks do not establish sm120 MMA parity."},
]


def half_is_nan(bits):
    return (int(bits) & 0x7FFF) > 0x7C00


def half_units(bits):
    """Nonnegative finite binary16 as an exact integer times 2**-24."""
    exponent, mantissa = (bits >> 10) & 31, bits & 1023
    if exponent == 31:
        raise ValueError("Nonfinite half")
    return mantissa if exponent == 0 else (1024 + mantissa) << (exponent - 1)


def fp8_units(bits, kind="e4m3"):
    """Nonnegative finite FP8 magnitude in the same exact units."""
    bits &= 127
    if kind == "e4m3":
        if bits == 127:
            raise ValueError("E4M3 NaN")
        exponent, mantissa = bits >> 3, bits & 7
        return mantissa << 15 if exponent == 0 else (8 + mantissa) << (exponent + 14)
    if kind == "e5m2":
        exponent, mantissa = bits >> 2, bits & 3
        if exponent == 31:
            raise ValueError("Nonfinite E5M2")
        return mantissa << 8 if exponent == 0 else (4 + mantissa) << (exponent + 7)
    raise ValueError(kind)


HALF_BY_UNITS = {half_units(bits): bits for bits in range(0x7C00)}
FP8_VALUES = {kind: [fp8_units(bits, kind) for bits in range(limit)]
              for kind, limit in (("e4m3", 127), ("e5m2", 124))}


def reference_decode(bits, kind="e4m3"):
    magnitude, sign = int(bits) & 127, (int(bits) & 128) << 8
    if kind == "e4m3" and magnitude == 127:
        return sign | 0x7E00
    if kind == "e5m2" and magnitude >= 124:
        return sign | (0x7C00 if magnitude == 124 else 0x7E00)
    return sign | HALF_BY_UNITS[fp8_units(magnitude, kind)]


def reference_encode(bits, kind="e4m3"):
    magnitude, sign = int(bits) & 0x7FFF, (int(bits) >> 8) & 128
    if magnitude > 0x7C00:
        return sign | (127 if kind == "e4m3" else 126)
    values = FP8_VALUES[kind]
    if magnitude == 0x7C00:
        return sign | (len(values) - 1)
    units = half_units(magnitude)
    hi = bisect.bisect_left(values, units)
    if hi == 0 or hi == len(values):
        chosen = min(hi, len(values) - 1)
    else:
        lo = hi - 1
        dl, dh = units - values[lo], values[hi] - units
        chosen = lo if dl < dh or (dl == dh and lo % 2 == 0) else hi
    return sign | chosen


def half_equivalent(actual, expected):
    a, b = np.asarray(actual, dtype=np.uint16), np.asarray(expected, dtype=np.uint16)
    return (a == b) | (((a & 0x7FFF) > 0x7C00) & ((b & 0x7FFF) > 0x7C00))


def byte_equivalent(actual, expected):
    a, b = np.asarray(actual), np.asarray(expected)
    return (a == b) | (((a & 127) == 127) & ((b & 127) == 127))


def execute(code, inputs):
    """Vectorized interpreter for the exact instruction subset emitted below.

    Unsupported instructions fail closed. This is a CPU semantic check, and is
    supplemented separately by ptxas and real-device conversion tests.
    """
    count = len(next(iter(inputs.values())))
    regs = {k: np.asarray(v, dtype=np.uint32) for k, v in inputs.items()}

    def value(token):
        token = token.strip()
        if token.startswith("%"):
            return regs[token]
        return np.full(count, int(token[2:], 16) if token.startswith("0f") else int(token, 0), np.uint32)

    def halves(v):
        return np.stack([v & 65535, v >> 16], 1).astype(np.uint16).view(np.float16)

    def pack_halves(v):
        b = np.asarray(v, dtype=np.float16).view(np.uint16).astype(np.uint32)
        return b[:, 0] | b[:, 1] << 16

    def f32(v):
        return v.view(np.float32)

    # Remove declarations and braces, then operate on semicolon statements.
    code = re.sub(r"\.reg\s+[^;]+;", "", code).replace("{", "").replace("}", "")
    with np.errstate(all="ignore"):
        for stmt in code.split(";"):
            stmt = stmt.strip()
            if not stmt:
                continue
            pred = None
            if stmt.startswith("@"):
                p, stmt = stmt.split(None, 1)
                pred = value(p[1:]).astype(bool)
            op, operands = stmt.split(None, 1)
            args = [a.strip() for a in operands.split(",")]
            dst, src = args[0], args[1:]
            # Tuple mov after brace removal.
            if op == "mov.b32" and len(args) == 3:
                regs[dst], regs[args[1]] = value(args[2]) & 65535, value(args[2]) >> 16
                continue
            v = [value(a) for a in src]
            if op.startswith("mov."):
                result = v[0]
            elif op in ("and.b32", "or.b32", "not.b32"):
                result = v[0] & v[1] if op == "and.b32" else v[0] | v[1] if op == "or.b32" else ~v[0]
            elif op in ("shr.u32", "shl.b32"):
                result = v[0] >> v[1] if op == "shr.u32" else v[0] << v[1]
            elif op in ("add.u32", "sub.u32"):
                result = v[0] + v[1] if op == "add.u32" else v[0] - v[1]
            elif op == "prmt.b32":
                selector = int(src[2], 0)
                result = np.zeros(count, np.uint32)
                for i in range(4):
                    index = (selector >> (i * 4)) & 15
                    if index >= 8:
                        raise ValueError("Unsupported prmt sign replication")
                    result |= ((v[index // 4] >> (8 * (index % 4))) & 255) << (8 * i)
            elif op in ("cvt.u16.u32", "cvt.u32.u16"):
                result = v[0] & 65535
            elif op == "cvt.f32.f16":
                result = (v[0] & 65535).astype(np.uint16).view(np.float16).astype(np.float32).view(np.uint32)
            elif op == "cvt.rni.u32.f32":
                result = np.rint(f32(v[0])).astype(np.uint32)
            elif op == "selp.b32":
                result = np.where(v[2].astype(bool), v[0], v[1])
            elif op in ("abs.f32", "min.f32", "mul.f32"):
                result = (np.abs(f32(v[0])) if op == "abs.f32" else
                          np.fmin(f32(v[0]), f32(v[1])) if op == "min.f32" else f32(v[0]) * f32(v[1])).astype(np.float32).view(np.uint32)
            elif op.startswith("setp."):
                comparison = op.split(".")[1]
                result = np.isnan(f32(v[0])) | np.isnan(f32(v[1])) if comparison == "nan" else f32(v[0]) < f32(v[1])
            elif op.startswith("set."):
                comparison = op.split(".")[1]
                a, b = halves(v[0]), halves(v[1])
                result = ((np.isnan(a) | np.isnan(b)) if comparison == "nan" else a == b if comparison == "eq" else a < b).astype(np.uint32) * 65535
                result = result[:, 0] | result[:, 1] << 16
            elif op in ("min.f16x2", "add.rn.f16x2", "sub.rn.f16x2", "mul.rn.f16x2"):
                a, b = halves(v[0]), halves(v[1])
                result = pack_halves(np.fmin(a, b) if op == "min.f16x2" else
                                     a + b if op.startswith("add") else a - b if op.startswith("sub") else a * b)
            else:
                raise ValueError("Unsupported instruction: " + op)
            result = np.asarray(result, dtype=np.uint32)
            regs[dst] = result if pred is None else np.where(pred, result, value(dst)).astype(np.uint32)
    return regs


class NumericSpecialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h = np.arange(65536, dtype=np.uint32)
        cls.pairs = cls.h | (cls.h[::-1] << 16)
        cls.expected_bytes = np.array([reference_encode(i) for i in range(65536)], np.uint32)
        cls.expected_half = np.array([reference_decode(reference_encode(i)) for i in range(65536)], np.uint16)

    def test_oracle_landmarks_and_e5m2_scope(self):
        for byte, half in ((0, 0), (128, 0x8000), (1, 0x1800), (8, 0x2400), (126, 0x5F00)):
            self.assertEqual(reference_decode(byte), half)
        self.assertEqual(reference_encode(0x7C00), 126)
        self.assertEqual(reference_encode(0xFC00), 254)
        self.assertEqual(reference_encode(0x1400), 0)  # halfway to min FP8 subnormal
        self.assertEqual(reference_encode(0x1A00), 2)  # halfway between 1 and 2: even 2
        self.assertEqual(reference_decode(0x7B, "e5m2"), 0x7B00)
        self.assertEqual(reference_decode(0x7C, "e5m2"), 0x7C00)
        for bits in range(256):
            half = reference_decode(bits, "e5m2")
            if (bits & 127) < 124:
                self.assertEqual(reference_encode(half, "e5m2"), bits)
        self.assertEqual(sum("e5m2" in p.read_text() for p in Path("fatbins").glob("*.ptx")), 0)

    def test_decode_all_65536_packed_pairs(self):
        got = execute(rw._e4m3x2_to_f16x2("%out", "%in"), {"%in": self.h})["%out"]
        for shift in (0, 8):
            expected = np.array([reference_decode((i >> shift) & 255) for i in range(65536)], np.uint16)
            actual = (got >> (2 * shift)).astype(np.uint16)
            self.assertTrue(half_equivalent(actual, expected).all())

    def test_mma_unpack_both_variants_all_bytes_all_positions(self):
        inputs = self.h | (self.h[::-1] << 16)
        for variant in (rw._unpack4_v1, rw._unpack4_v2):
            got = execute(variant("%in", "%lo", "%hi"), {"%in": inputs})
            for byte in range(4):
                expected = np.array([reference_decode(int(i >> (byte * 8)) & 255) for i in inputs], np.uint16)
                actual = (got["%lo" if byte < 2 else "%hi"] >> (16 * (byte % 2))).astype(np.uint16)
                self.assertTrue(half_equivalent(actual, expected).all(), variant.__name__)

    def test_encode_all_65536_halves_both_lanes_all_variants(self):
        for enc, encode in (("v1", None), ("v2", None), ("v1", "f32")):
            with patch.dict(rw.ACC, {"enc": enc, "encode": encode}, clear=True):
                code = rw._f16x2_to_e4m3x2("%out", "%in")
            got = execute(code, {"%in": self.pairs})["%out"]
            self.assertTrue(byte_equivalent(got & 255, self.expected_bytes).all(), (enc, encode, "lo"))
            self.assertTrue(byte_equivalent(got >> 8, self.expected_bytes[::-1]).all(), (enc, encode, "hi"))

    def test_fused_roundtrip_all_65536_halves_both_lanes(self):
        got = execute(fuse.fq("%out", "%in"), {"%in": self.pairs})["%out"]
        self.assertTrue(half_equivalent(got.astype(np.uint16), self.expected_half).all())
        self.assertTrue(half_equivalent((got >> 16).astype(np.uint16), self.expected_half[::-1]).all())

    def test_fused_immediate_all_bytes_and_positions(self):
        for bits in range(256):
            self.assertTrue(bool(half_equivalent(fuse._e4m3_to_f16_bits(bits), reference_decode(bits))))
            wide = bits * 0x01010101
            lo, hi = fuse._imm_shadow(hex(wide), True)
            expected = reference_decode(bits)
            for packed in (int(lo, 16), int(hi, 16)):
                self.assertTrue(bool(half_equivalent(packed & 65535, expected)))
                self.assertTrue(bool(half_equivalent(packed >> 16, expected)))

    def test_nan_negative_fixtures_detected(self):
        old_decode = rw._e4m3x2_to_f16x2("%out", "%in").replace("or.b32 %out, %out, %x_n;", "")
        got = execute(old_decode, {"%in": np.array([0xFF7F], np.uint32)})["%out"][0]
        self.assertFalse(bool(half_equivalent(got & 65535, reference_decode(127))))
        old_encode = rw._f16x2_to_e4m3x2_v1("%out", "%in").replace("or.b32 %z_r, %z_r, %z_n;", "")
        got = execute(old_encode, {"%in": np.array([0xFE017C01], np.uint32)})["%out"][0]
        self.assertFalse(bool(byte_equivalent(got & 255, 127)))
        old_fq = fuse.fq("%out", "%in").replace("or.b32 %out, %out, %f_n;", "")
        got = execute(old_fq, {"%in": np.array([0xFE017C01], np.uint32)})["%out"][0]
        self.assertFalse(bool(half_equivalent(got & 65535, 0x7E00)))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    scope = "CPU emitted-PTX conversion semantics; not sm120 MMA output parity"
    if args.report:
        # Reject before writes; never replace an unrelated user file or an input.
        from .build_dll import require_disjoint, input_paths, cache_paths, ROOT
        temporary = args.report.with_suffix(args.report.suffix + ".tmp")
        protected = [*input_paths(), *cache_paths(Path(ROOT, "sm86_port")),
                     Path(ROOT, "sm86_port/nvngx_dlssnr.dll"),
                     Path(ROOT, "plugin/dist/DLSS5-RTX30/payload/nvngx_dlssnr.dll")]
        require_disjoint([args.report, temporary], protected)
        if args.report.suffix.lower() != ".json":
            raise ValueError("A numeric report must have .json suffix")
        for path in (args.report, temporary):
            if path.exists():
                previous = json.loads(path.read_text(encoding="utf-8"))
                if previous.get("scope") != scope:
                    raise ValueError("Refusing to replace an unrelated file: " + str(path))
    report = {"schema": 1, "report_kind": "fp8-cpu-numeric-special", "status": "running", "scope": scope, "sources": SOURCES}
    def save():
        if args.report:
            require_disjoint([args.report, temporary], protected)
            temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
            temporary.replace(args.report)
    save()
    try:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(NumericSpecialTests))
        report.update(status="passed" if result.wasSuccessful() else "failed", tests=result.testsRun,
                      failures=len(result.failures), errors=len(result.errors),
                      source_sha256={str(Path(__file__).with_name(n)): hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                                     for n in ("rewrite.py", "fuse.py", "test_numeric_special.py")},
                      nan_acceptance="category only; payload/sign unspecified", finite_acceptance="bit exact including signed zero, saturation and subnormals",
                      e5m2="Independent oracle landmarks/roundtrip only; absent from source PTX and unsupported by current rewriter")
        save()
        return 0 if result.wasSuccessful() else 1
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
