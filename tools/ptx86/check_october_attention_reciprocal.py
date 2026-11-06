"""Check the October attention reciprocal's FP16 rounding with exact rationals."""

import argparse
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import struct
import sys


CLAMP_F32_BITS = 0x388205FF
LAST_FINITE_F16 = 0x7BFF
PTX_RCP_SPEC = "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-rcp"


def f16_value(code):
    return Fraction(struct.unpack("<e", struct.pack("<H", code))[0])


def f16_code(value):
    return struct.unpack("<H", struct.pack("<e", value))[0]


def spacing(value, fraction_bits, min_exponent):
    """Spacing in the exact value's binade, using the upper-side spacing at powers of two."""
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if value < Fraction(2) ** exponent:
        exponent -= 1
    return Fraction(2) ** (max(exponent, min_exponent) - fraction_bits)


def round_binary(value, fraction_bits, min_exponent):
    unit = spacing(value, fraction_bits, min_exponent)
    return round(value / unit) * unit


def run():
    clamp_f32 = struct.unpack("<f", struct.pack("<I", CLAMP_F32_BITS))[0]
    first = f16_code(clamp_f32)
    differences = dict.fromkeys(("exact_reciprocal_plus_or_minus_one_ulp",
                                 "rounded_f32_plus_or_minus_one_ulp", "host_f32_then_f16"), 0)
    closest = None
    for code in range(first, LAST_FINITE_F16 + 1):
        denominator = f16_value(code)
        exact = 1 / denominator
        expected = round_binary(exact, 10, -14)
        exact_ulp = spacing(exact, 23, -126)
        host_f32 = round_binary(exact, 23, -126)
        host_ulp = spacing(host_f32, 23, -126)
        checks = {"exact_reciprocal_plus_or_minus_one_ulp": (exact - exact_ulp, exact + exact_ulp),
                  "rounded_f32_plus_or_minus_one_ulp": (host_f32 - host_ulp, host_f32 + host_ulp),
                  "host_f32_then_f16": (host_f32,)}
        for key, endpoints in checks.items():
            differences[key] += any(round_binary(endpoint, 10, -14) != expected for endpoint in endpoints)

        rounded_code = f16_code(float(expected))
        lower_midpoint = (f16_value(rounded_code - 1) + expected) / 2
        upper_midpoint = (f16_value(rounded_code + 1) + expected) / 2
        distance = min(exact - lower_midpoint, upper_midpoint - exact) / exact_ulp
        if closest is None or distance < closest[0]:
            closest = distance, code

    distance, closest_code = closest
    infinity_code = f16_code(1.0 / float("inf"))
    return {"schema": "october_attention_reciprocal_bound_v1",
            "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "ptx_spec": PTX_RCP_SPEC,
            "ptx_reciprocal_max_error_f32_ulp": 1,
            "clamp_f32_bits": f"0x{CLAMP_F32_BITS:08x}",
            "tested_f16_codes_inclusive": [f"0x{first:04x}", f"0x{LAST_FINITE_F16:04x}"],
            "tested_finite_denominators": LAST_FINITE_F16 - first + 1,
            "ulp_definition": "Spacing in the reference reciprocal's FP32 binade; at a binade boundary use the upper-side spacing.",
            "rounding": "Exact rational RNE for binary32 and binary16 including binary16 subnormals; monotone rounding bounds all values between the endpoints.",
            "different_f16_values": differences,
            "nearest_f16_midpoint_distance_f32_ulp": {"numerator": distance.numerator, "denominator": distance.denominator,
                                                       "value": float(distance), "input_f16_code": f"0x{closest_code:04x}"},
            "positive_infinity": {"ptx_expected_f16_code": "0x0000", "host_f16_code": f"0x{infinity_code:04x}"},
            "passed": not any(differences.values()) and infinity_code == 0}


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    report = run()
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
