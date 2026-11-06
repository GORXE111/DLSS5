"""Audit storage layout hypotheses for the 310.9.16.0 CC_SMALL_FP8 model.

Run with `python -B -m tools.inspect_nr_small_layout [--numpy-constructors] [--report PATH]`.
Uses old validated constructor layouts, the new serialized records and the new
PTX entry inventory. This does not establish the new host's execution schedule.
"""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from importlib import import_module
import json
from math import prod
from mmap import ACCESS_READ, mmap
from pathlib import Path
import re
import struct
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from .inspect_nr_weights import compare_files
from .report_paths import ReportPath


ROOT = Path(__file__).resolve().parents[1]
OCTOBER = ROOT / "_dl/october_nr_20261106"
BASELINE = ROOT / "dlss5/nvngx_dlssnr.dll"
CANDIDATE = OCTOBER / "original/nvngx_dlssnr.dll"
PTX = OCTOBER / "candidate_extract"
DATA = ROOT / "torch/dlss5/data"
INPUTS = [ROOT / "tools/inspect_nr_weights.py", ROOT / "tools/dllq/index.py", ROOT / "tools/dllq/core.py",
          ROOT / "torch/dlss5/net.py", ROOT / "torch/dlss5/blocks.py",
          ROOT / "torch/dlss5/layout.py", DATA / "schedule.json", DATA / "exec_order.json",
          DATA / "adapter_map.json", DATA / "control_map.json", DATA / "maps_block1.json"]
CONSTRUCTOR_IMPORTS = [ROOT / "torch/dlss5" / name for name in
                       ("ops.py", "ptx_math.py", "style.py", "weights.py")]
EXPECTED = ("e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e",
            "2bc19bc09a7b89dd44f73e2ac0108fcc586e2154bdd0830f242f4f040e28789f")
SMALL = "WEIGHTS_CC_SMALL_FP8"
DTYPE_BYTES = {"e4m3": 1, "f16": 2, "f32": 4}
ENTRY = re.compile(rb"(?m)^\.visible \.entry (\w+)\(")


def field(role, offset, dtype, shape, **extra):
    return {"role": role, "offset": offset, "dtype": dtype, "shape": list(shape),
            "bytes": prod(shape) * DTYPE_BYTES[dtype], **extra}


def swin_fields(width, variant):
    h = width // 32
    if width == 32:
        fields = [field("ffn_expand", 0, "e4m3", (32, 128)), field("ffn_contract", 4096, "e4m3", (128, 32))]
        c1, qkv = (10336, 10400) if variant == "up" else (8208, 8288)
        projection_bias = qkv + 3072
        temperature = projection_bias + 8192
        projection = temperature + 16
    else:
        d = 4 * width ** 2 + 128 * width
        fields = [field("ffn_expand", 0, "e4m3", (width, 4 * width)),
                  field("ffn_contract_grouped", 4 * width ** 2, "e4m3", (h, 128, 32),
                        materialized_dense_shape=[4 * width, width]),
                  field("ffn_project", d, "e4m3", (width, width))]
        c1 = d + width ** 2 + 16
        if variant == "up":
            c1 += 2 * width ** 2 + 2 * width - 16
        qkv = c1 + 2 * width + (0 if variant == "up" else 16)
        projection_bias = qkv + 3 * width ** 2
        temperature = projection_bias + 8192 * h
        projection = temperature + 16 * ((h + 3) // 4)
    fields += [field("ffn_residual_scale", c1, "f16", (width,)),
               field("attention_qkv", qkv, "e4m3", (width, h, 3, 32)),
               field("attention_position_bias_expanded", projection_bias, "f16", (h, 64, 64)),
               field("attention_temperature", temperature, "f32", (h,)),
               field("attention_projection", projection, "e4m3", (width, width)),
               field("attention_residual_scale", projection + width ** 2, "f16", (width,))]
    if variant == "ds":
        fields.append(field("downsample_projection", projection + width ** 2 + 2 * width,
                            "e4m3", (width, 2 * width)))
    if variant == "up":
        up = 8192 if width == 32 else c1 - 2 * width ** 2 - 2 * width
        fields += [field("upsample_projection", up, "e4m3", (2 * width, width)),
                   field("upsample_scale", c1 - 2 * width - (16 if width == 32 else 0), "f16", (width,))]
    return fields


def adapter_indices():
    adapter = read_json(DATA / "adapter_map.json")
    controls = read_json(DATA / "control_map.json")
    return sorted(set(map(int, controls)) | {int(i) for i, (_, name) in adapter.items()
                                            if name is not None and not name.startswith("常数")})


def layout_for_step(step):
    kind = step["kind"]
    if kind == "swin":
        return {step["record"]: swin_fields(32 * int(step["level"][:-1]), step["variant"])}
    if kind in {"pre", "post"}:
        fields = swin_fields(32, "std")
        if kind == "pre":
            for row in fields:
                if row["offset"] >= 8208:
                    row["offset"] += 1024
            fields.append(field("input_adapter", 8208, "f16", (16, 32), logical_shape=[15, 32],
                                active_storage_indices=adapter_indices()))
            return {step["record"]: fields}
        for row in fields:
            if row["offset"] >= 8288:
                row["offset"] += 112
        indices = sorted(256 * (j % 2) + 8 * ((j % 8) // 2) + j // 8 + 32 * o
                         for j in range(32) for o in range(4))
        fields += [field("predecessor_upsample_scale", 8272, "f16", (32,)),
                   field("pre_skip_scale", 8336, "f16", (32,)),
                   field("output_adapter", 20784, "f16", (512,), logical_shape=[32, 4],
                         active_storage_indices=indices)]
        return {step["record"]: fields, step["blend_record"]: [field("history_blend_scale", 0, "f16", (1,))]}
    if kind == "split16":
        fields = [
            [field("ffn_input_projection", 0, "e4m3", (512, 512)),
             field("ffn_expand_grouped", 262144, "e4m3", (8, 64, 256), materialized_dense_shape=[512, 2048]),
             field("ffn_contract_grouped", 393216, "e4m3", (8, 256, 64), materialized_dense_shape=[2048, 512])],
            [field("ffn_output_projection", 0, "e4m3", (512, 512)),
             field("ffn_residual_scale", 262144, "f16", (512,))],
            [field("attention_qkv", 0, "e4m3", (512, 16, 3, 32)),
             field("attention_position_bias_expanded", 786432, "f16", (16, 64, 64)),
             field("attention_temperature", 917504, "f32", (16,))],
            [field("attention_projection", 0, "e4m3", (512, 512)),
             field("attention_residual_scale", 262144, "f16", (512,))],
        ]
        records = dict(zip(step["records"], fields, strict=True))
        if "head_record" in step:
            records[step["head_record"]] = [field("vit_entry_projection", 0, "e4m3", (512, 1024))]
        return records
    if kind == "vit":
        fields = [[field("ffn_expand", 0, "e4m3", (1024, 4096))],
                  [field("ffn_contract", 0, "e4m3", (4096, 1024)),
                   field("ffn_residual_scale", 4194304, "f16", (1024,))],
                  [field("attention_temperature", 0, "f32", (32,)),
                   field("attention_qkv", 128, "e4m3", (1024, 32, 3, 32))],
                  [],
                  [field("attention_projection", 0, "e4m3", (1024, 1024)),
                   field("attention_residual_scale", 1048576, "f16", (1024,))]]
        return dict(zip(step["records"], fields, strict=True))
    if kind == "block39":
        return {step["record"]: [field("vit_exit_projection", 0, "e4m3", (1024, 512)),
                                 field("skip_scale", 524288, "f16", (512,))]}
    raise ValueError(f"unsupported baseline layout {kind}")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as stream:
        return sha256_file(stream)


def sha256_file(stream):
    sha = sha256()
    for chunk in iter(lambda: stream.read(1 << 20), b""):
        sha.update(chunk)
    return sha.hexdigest()


def kernel_index(paths):
    index = {}
    for path in paths:
        raw = path.read_bytes()
        for match in ENTRY.finditer(raw):
            name = match[1].decode("ascii")
            if name in index:
                raise ValueError(f"duplicate PTX entry {name}")
            index[name] = {"entry": name, "file": path.name, "line": raw.count(b"\n", 0, match.start()) + 1}
    return index


def unknown_interval(payload, start, end, role):
    return {"offset": start, "bytes": end - start, "role": role, "all_zero": not any(payload[start:end]),
            "raw_hex": payload[start:end].hex()}


def audit_fields(payload, fields):
    audit, unknown = [], []
    cursor = 0
    for row in sorted(fields, key=lambda row: row["offset"]):
        start, end = row["offset"], row["offset"] + row["bytes"]
        if not cursor <= start < end <= len(payload):
            raise ValueError(f"overlapping or out of bounds field {row['role']}")
        if start > cursor:
            unknown.append(unknown_interval(payload, cursor, start, "gap_outside_baseline_fields"))
        data = payload[start:end]
        dtype = row["dtype"]
        count = prod(row["shape"])
        indices = row.get("active_storage_indices", range(count))
        if len(indices) < count:
            omitted = set(range(count)) - set(indices)
            if prod(row["logical_shape"]) != len(indices) or any(any(data[i * 2:i * 2 + 2]) for i in omitted):
                raise ValueError(f"nonzero or inconsistent inactive slots in {row['role']}")
        if dtype == "e4m3":
            nonfinite = data.count(b"\x7f") + data.count(b"\xff")
        else:
            exp = 0x7C00 if dtype == "f16" else 0x7F800000
            code = "<H" if dtype == "f16" else "<I"
            nonfinite = sum((number & exp) == exp for number, in struct.iter_unpack(code, data))
        audit.append({**row, "storage_slots": count, "active_baseline_slots": len(indices),
                      "nonfinite_slots_if_decoded_as_declared_dtype": nonfinite})
        cursor = end
    if cursor < len(payload):
        unknown.append(unknown_interval(payload, cursor, len(payload),
                                        "untyped_attention_record" if not fields else "tail_outside_baseline_fields"))
    return audit, unknown


class TracedPayload(bytes):
    """Carry byte provenance through slices and adapter splices in the old constructors."""

    def __new__(cls, raw, spans):
        data = super().__new__(cls, raw)
        data.spans = spans
        return data

    def __getitem__(self, item):
        raw = super().__getitem__(item)
        if not isinstance(item, slice):
            return raw
        start, end, stride = item.indices(len(self))
        if stride != 1:
            raise ValueError("constructor payload slices must be contiguous")
        return TracedPayload(raw, [(max(a, start) - start, min(b, end) - start, name, orig + max(a, start) - a)
                                   for a, b, name, orig in self.spans if max(a, start) < min(b, end)])

    def __add__(self, other):
        return TracedPayload(bytes(self) + bytes(other), self.spans + shifted_spans(other, len(self)))

    def __radd__(self, other):
        return TracedPayload(bytes(other) + bytes(self), shifted_spans(self, len(other)))


def shifted_spans(data, amount):
    return [(a + amount, b + amount, name, orig) for a, b, name, orig in getattr(data, "spans", ())]


def numpy_torch(np):
    """Import-only stand-in for the few Torch services used in NumPy constructors."""
    bits = np.arange(256, dtype=np.uint8)
    exponent = (bits >> 3) & 15
    fraction = (bits & 7) / 8.0
    lut = np.where(exponent == 0, fraction * 2 ** -6, np.exp2(exponent.astype(np.float32) - 7) * (1 + fraction))
    lut = np.where(bits & 128, -lut, lut).astype(np.float32)
    lut[(bits & 127) == 127] = np.nan
    shim = ModuleType("torch")
    shim.float32, shim.long = np.float32, np.int64
    shim.uint8, shim.float8_e4m3fn = np.uint8, object()
    shim.no_grad = lambda: (lambda function: function)
    shim.as_tensor = lambda array, *, dtype, device: np.asarray(array, dtype=dtype)
    shim.arange = lambda *args, **kwargs: SimpleNamespace(
        view=lambda dtype: SimpleNamespace(float=lambda: SimpleNamespace(numpy=lambda: lut)))
    return shim


def merge_intervals(rows):
    result = []
    for start, end, dtype in sorted(rows):
        if result and result[-1][2] == dtype and result[-1][1] >= start:
            start, previous, dtype = result.pop()
            end = max(end, previous)
        result.append((start, end, dtype))
    return result


def check_constructors(records, resource_offset, steps_by_name):
    import numpy as np

    observed = defaultdict(list)
    bias_runs = []
    package = ModuleType("_nr_small_legacy")
    package.__path__ = [str(ROOT / "torch/dlss5")]
    index = np.arange(64)
    deltas = (index[None, :] // 8 - index[:, None] // 8 + 7) * 15 + index[None, :] % 8 - index[:, None] % 8 + 7
    bins = [deltas == delta for delta in range(225)]

    def track(raw, dtype):
        for start, end, name, original in getattr(raw, "spans", ()):
            observed[name].append((original, original + end - start, dtype))

    def bias_decoder(function):
        def read(raw):
            track(raw, "f16")
            value = function(bytes(raw))
            if isinstance(raw, TracedPayload) and len(raw.spans) == 1:
                _, _, name, start = raw.spans[0]
                inconsistent = sum(np.any(value[mask] != value[mask][0]) for mask in bins)
                bias_runs.append({"record": name, "offset": start, "position_delta_classes": 225,
                                  "classes_with_multiple_stored_values": int(inconsistent)})
            return value
        return read

    with patch.dict(sys.modules, {"torch": numpy_torch(np), package.__name__: package}):
        net = import_module(f"{package.__name__}.net")
        lay = net.Lay
        fp8, f16, frombuffer = lay.unswizzle, lay.f16vec, np.frombuffer

        def read_fp8(raw, k, n):
            track(raw[:k * n], "e4m3")
            return fp8(bytes(raw), k, n)

        def read_f16(raw):
            track(raw, "f16")
            return f16(bytes(raw))

        def read_buffer(raw, dtype=float, count=-1, offset=0, *, like=None):
            if isinstance(raw, TracedPayload):
                element = np.dtype(dtype)
                if element != np.dtype("<f4"):
                    raise ValueError(f"unexpected direct constructor payload dtype {element}")
                end = len(raw) if count == -1 else offset + count * element.itemsize
                track(raw[offset:end], "f32")
            return frombuffer(raw, dtype, count, offset, like=like)

        blocks = defaultdict(dict)
        for record in records:
            block, suffix = record["name"].split(".", 1)
            blocks[block][suffix] = record
        with (patch.multiple(lay, unswizzle=read_fp8, f16vec=read_f16,
                             bias1h=bias_decoder(lay.bias1h), bias16=bias_decoder(lay.bias16)),
              patch.object(np, "frombuffer", read_buffer), CANDIDATE.open("rb") as stream,
              mmap(stream.fileno(), 0, access=ACCESS_READ) as data):
            for group in blocks.values():
                step = steps_by_name[group["layer0.layer"]["baseline_layout_matches"][0]]

                def get(name):
                    row = group[name.split(".", 1)[1]]
                    start, size = resource_offset + row["payload_offset"], row["payload_bytes"]
                    return TracedPayload(data[start:start + size], [(0, size, row["name"], 0)])

                kind = step["kind"]
                if kind == "swin":
                    decoder = {"up": net.SwinUp, "ds": net.SwinDown}.get(step["variant"], net.Swin)
                    decoder(get(step["record"]), 32 * int(step["level"][:-1]), "cpu")
                elif kind in {"pre", "post", "block39"}:
                    decoder = {"pre": net.PreBlock, "post": net.PostBlock, "block39": net.Block39}[kind]
                    args = [get(step["record"])]
                    if kind == "post":
                        args.append(get(step["blend_record"]))
                    decoder(*args, "cpu")
                elif kind in {"split16", "vit"}:
                    decoder = net.Split16 if kind == "split16" else net.ViT
                    decoder([get(name) for name in step["records"]], "cpu")
                    if "head_record" in step:
                        net.FinalHead(get(step["head_record"]), "cpu")

    for row in records:
        expected = merge_intervals((f["offset"], f["offset"] + f["bytes"], f["dtype"]) for f in row["fields"])
        actual = merge_intervals(observed[row["name"]])
        if expected != actual:
            raise ValueError(f"legacy constructor read range differs for {row['name']}: {expected} vs {actual}")
        row["constructor_read_intervals"] = [{"start": a, "end_exclusive": b, "dtype": t} for a, b, t in actual]
    return {"mode": "legacy Python/NumPy constructors, tensor storage and import-time E4M3 table supplied by NumPy",
            "numpy_version": np.__version__, "blocks_constructed": len(blocks), "record_read_ranges_checked": len(records),
            "runtime_forward_executed": False, "expanded_bias_heads": len(bias_runs),
            "expanded_bias_heads_exactly_tied_by_relative_offset": sum(
                row["classes_with_multiple_stored_values"] == 0 for row in bias_runs),
            "expanded_bias_head_checks": bias_runs}


def inspect_small(ptx_dir, numpy_constructors=False):
    pair = compare_files(BASELINE, CANDIDATE, include_records=True)
    if (pair["baseline"]["sha256"], pair["candidate"]["sha256"]) != EXPECTED:
        raise ValueError("baseline or candidate DLL does not match pinned original SHA-256")
    comparison, = (row for row in pair["comparisons"] if row["candidate_resource"] == [SMALL, "1033"])
    small, = (r for r in pair["candidate"]["resources"] if (r["name"], r["language"]) == (SMALL, "1033"))
    if (len(small["blocks"]), small["record_count"]) != (31, 77):
        raise ValueError("unexpected small model inventory")
    matches = {row["candidate_block"]: row["baseline_blocks"] for row in comparison["block_storage_signature_matches"]}
    fields_by_name = {}
    steps_by_name = {}
    for step in read_json(DATA / "schedule.json")["steps"]:
        layout = layout_for_step(step)
        fields_by_name.update(layout)
        steps_by_name.update({name: step for name in layout})
    baseline_launches = defaultdict(set)
    for launch in read_json(DATA / "exec_order.json")["launches"]:
        for record in launch["weights"]:
            baseline_launches[record].add(launch["kernel"])
    paths = sorted(ptx_dir.glob("*.ptx"))
    if len(paths) != 15:
        raise ValueError("expected all 15 extracted candidate PTX modules")
    ptx_index = kernel_index(paths)

    records = []
    blocks = []
    for block in small["blocks"]:
        index = block["index"]
        steps = [steps_by_name[f"block{i}.layer0.layer"] for i in matches[index]]
        kind, = {step["kind"] for step in steps}
        block_role = {"index": index, "baseline_layout_matches": matches[index], "layout_family": kind,
                      "record_count": len(block["records"])}
        if kind == "swin":
            level, = {step["level"] for step in steps}
            block_role.update(level=level, baseline_variants=sorted({step["variant"] for step in steps}))
        blocks.append(block_role)
    with CANDIDATE.open("rb") as stream, mmap(stream.fileno(), 0, access=ACCESS_READ) as data:
        for record in small["records"]:
            block, suffix = record["name"].split(".", 1)
            baseline_names = [f"block{i}.{suffix}" for i in matches[int(block[5:])]]
            if not baseline_names or any(fields_by_name[name] != fields_by_name[baseline_names[0]]
                                         for name in baseline_names[1:]):
                raise ValueError(f"missing or conflicting baseline layouts for {record['name']}")
            start = small["file_offset"] + record["payload_offset"]
            payload = data[start:start + record["payload_bytes"]]
            if sha256(payload).hexdigest() != record["payload_sha256"]:
                raise ValueError(f"candidate payload changed: {record['name']}")
            fields, unresolved = audit_fields(payload, fields_by_name[baseline_names[0]])
            old_kernels = set().union(*(baseline_launches[name] for name in baseline_names))
            entries = [ptx_index[name] for name in sorted(old_kernels & ptx_index.keys())]
            if not entries:
                raise ValueError(f"no candidate PTX entry matches baseline launch names for {record['name']}")
            records.append({**record, "baseline_layout_matches": baseline_names, "candidate_ptx_entries": entries,
                            "baseline_launch_names_not_present_in_candidate_ptx": sorted(old_kernels - ptx_index.keys()),
                            "fields": fields, "unresolved_intervals": unresolved})

    storage, active = Counter(), Counter()
    unresolved_bytes, zero_gap_bytes, nonfinite = 0, 0, 0
    for record in records:
        for row in record["fields"]:
            storage[row["dtype"]] += row["storage_slots"]
            active[row["dtype"]] += row["active_baseline_slots"]
            nonfinite += row["nonfinite_slots_if_decoded_as_declared_dtype"]
        for row in record["unresolved_intervals"]:
            unresolved_bytes += row["bytes"]
            if row["all_zero"]:
                zero_gap_bytes += row["bytes"]
    typed_bytes = sum(count * DTYPE_BYTES[dtype] for dtype, count in storage.items())
    if typed_bytes + unresolved_bytes != small["payload_bytes"]:
        raise ValueError("field and unresolved byte ledger does not balance")
    evidence_paths = [Path(__file__), *INPUTS, *paths]
    compatibility = None
    if numpy_constructors:
        compatibility = check_constructors(records, small["file_offset"], steps_by_name)
        evidence_paths.extend(CONSTRUCTOR_IMPORTS)
    return {"schema": "dlssnr_cc_small_fp8_storage_v1", "passed": True,
            "scope": "Static transfer of 310.8.0 decoder fields supported by matching record signatures. "
                     "Named PTX entries exist in the 310.9.16.0 package; per-record dispatch is unverified.",
            "baseline": {**{k: v for k, v in pair["baseline"].items() if k not in {"resources", "path"}},
                         "path": BASELINE.relative_to(ROOT).as_posix()},
            "candidate": {**{k: v for k, v in pair["candidate"].items() if k not in {"resources", "path"}},
                          "path": CANDIDATE.relative_to(ROOT).as_posix()},
            "resource": {k: v for k, v in small.items() if k not in ("records", "blocks")},
            "source_sha256": {str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path).replace("\\", "/"):
                              digest(path) for path in evidence_paths},
            "summary": {"blocks": len(blocks), "records": len(records), "typed_bytes": typed_bytes,
                        "typed_storage_slots_by_dtype": dict(storage), "active_baseline_slots_by_dtype": dict(active),
                        "typed_storage_slots": sum(storage.values()), "active_baseline_slots": sum(active.values()),
                        "typed_bytes_in_zero_inactive_slots": sum((storage[t] - active[t]) * DTYPE_BYTES[t]
                                                                  for t in storage),
                        "unresolved_bytes": unresolved_bytes, "unresolved_all_zero_intervals_bytes": zero_gap_bytes,
                        "nonfinite_slots_if_decoded_as_declared_dtype": nonfinite,
                        "learned_parameter_count": None},
            "legacy_constructor_check": compatibility, "blocks": blocks, "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ptx-dir", type=Path, default=PTX)
    parser.add_argument("--numpy-constructors", action="store_true", help="trace legacy constructor reads on CPU")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    ptx_dir = args.ptx_dir.resolve()
    output = None
    if args.report:
        try:
            output = ReportPath(args.report, inputs=[BASELINE, CANDIDATE], input_dirs=[ptx_dir])
            output.write_json({"passed": False, "state": "running"})
        except (OSError, ValueError) as error:
            print(f"Cannot write layout report: {error}", file=sys.stderr)
            return 1
    try:
        report = inspect_small(ptx_dir, args.numpy_constructors)
    except (OSError, ValueError, KeyError, ImportError, struct.error) as error:
        report = {"passed": False, "error": str(error)}
    if output:
        output.write_json(report)
    print(json.dumps(report.get("summary", report) if output else report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
