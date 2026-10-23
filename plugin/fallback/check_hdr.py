"""Validate raw fallback HDR dumps in display-linear nits; no GPU or third-party dependencies."""
import argparse
import json
import math
from pathlib import Path
import struct


def read_hdr(path, width, height, encoding):
    data = Path(path).read_bytes()
    stride = 8 if encoding == "f16" else 4
    if len(data) != width * height * stride:
        raise ValueError(f"{path}: expected {width * height * stride} bytes, got {len(data)}")
    if encoding == "f16":
        pixels = list(struct.iter_unpack("<4e", data))
        if any(not math.isfinite(v) for pixel in pixels for v in pixel):
            raise ValueError(f"{path}: non-finite half-float texel")
        if any(abs(p[3] - 1) > 0.001 for p in pixels):
            raise ValueError(f"{path}: alpha is not opaque")
        return [tuple(v * 80 for v in p[:3]) for p in pixels]
    pixels = [v[0] for v in struct.iter_unpack("<I", data)]
    if any(v >> 30 != 3 for v in pixels):
        raise ValueError(f"{path}: alpha is not opaque")

    def pq(code):
        p = (code / 1023) ** (1 / 78.84375)
        return 10000 * (max(p - 0.8359375, 0) / (18.8515625 - 18.6875 * p)) ** (1 / 0.1593017578125)

    lut = [pq(i) for i in range(1024)]
    return [tuple(lut[(v >> (10 * k)) & 1023] for k in range(3)) for v in pixels]


def validate(before, after, highlight_nits):
    if len(before) != len(after) or not before:
        raise ValueError("HDR inputs must have the same non-zero dimensions")
    a = [max(p) for p in before]
    b = [max(p) for p in after]
    bright = [i for i, v in enumerate(a) if v >= highlight_nits]
    if len(bright) < max(1, len(a) // 1000):
        raise ValueError("fixture has too few highlights to validate preservation")
    ratio = sum(b[i] for i in bright) / sum(a[i] for i in bright)
    changed = sum(abs(x - y) for p, q in zip(before, after) for x, y in zip(p, q)) / (3 * len(a))
    metrics = {"highlight_pixels": len(bright), "highlight_mean_ratio": ratio,
               "input_peak_nits": max(a), "output_peak_nits": max(b), "mean_abs_nits": changed}
    if not all(math.isfinite(v) for v in metrics.values()):
        raise ValueError("non-finite HDR metrics")
    # This is a fallback regression envelope, not a claim of official DLSS parity.
    if not 0.75 <= ratio <= 1.25 or max(b) < highlight_nits:
        raise ValueError(f"highlights lost or amplified excessively: {metrics}")
    if changed < 0.001:
        raise ValueError("HDR output was not processed (mean absolute difference < 0.001 nit)")
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path)
    ap.add_argument("--encoding", choices=("f16", "pq"), required=True)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--highlight-nits", type=float, required=True)
    a = ap.parse_args()
    try:
        before = read_hdr(a.directory / f"dlss5fb_in.{a.encoding}", a.width, a.height, a.encoding)
        after = read_hdr(a.directory / f"dlss5fb_out.{a.encoding}", a.width, a.height, a.encoding)
        print(json.dumps({"status": "pass", **validate(before, after, a.highlight_nits)}, sort_keys=True))
    except (OSError, ValueError, OverflowError) as exc:
        print(json.dumps({"status": "fail", "reason": str(exc)}))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
