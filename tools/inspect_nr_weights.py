"""Inspect and compare NVIDIA DLSS-NR PE weight resources without extracting files.

Run from the repository root with two original DLL paths:
    python -m tools.inspect_nr_weights baseline.dll candidate.dll [--records]

The JSON on stdout reports serialized storage. Tensor shapes and independent
learned parameter counts need separate kernel or checkpoint evidence.
"""

import argparse
from collections import Counter
from hashlib import sha256
import json
from mmap import ACCESS_READ, mmap
from pathlib import Path
import re
import struct
import sys

from .dllq.index import parse_pe, parse_resources


BLOCK_RECORD = re.compile(r"block(\d+)\.(.+)")


def payload_records(blob):
    if len(blob) < 8 or struct.unpack_from("<Q", blob)[0] != len(blob):
        raise ValueError("weight resource length header differs from PE resource size")
    records = {}
    offset = 8
    while offset < len(blob):
        if offset + 8 > len(blob):
            raise ValueError(f"truncated record name length at {offset}")
        name_length = struct.unpack_from("<Q", blob, offset)[0]
        fields = offset + 8 + name_length
        payload = fields + 28
        if not 1 <= name_length <= 128 or payload > len(blob):
            raise ValueError(f"invalid record header at {offset}")
        name = bytes(blob[offset + 8:fields]).decode("ascii")
        if not BLOCK_RECORD.fullmatch(name) or name in records:
            raise ValueError(f"invalid or duplicate record name {name!r}")
        field_a, field_b, size, tag = struct.unpack_from("<QQQI", blob, fields)
        end = payload + size
        if not size or end + 20 > len(blob):
            raise ValueError(f"invalid payload extent for {name}")
        reserved0, reserved1, marker, reserved2, units = struct.unpack_from("<IIIII", blob, end)
        if (field_a, field_b, tag) != (size + 40, size + 40, 1) or (
            reserved0, reserved1, marker, reserved2
        ) != (0, 0, 1, 0):
            raise ValueError(f"unsupported record envelope for {name}")
        with blob[payload:end] as raw:
            records[name] = {
                "name": name,
                "payload_offset": payload,
                "payload_bytes": size,
                "payload_sha256": sha256(raw).hexdigest(),
                "trailer_count": units,
            }
        offset = end + 20
    if not records:
        raise ValueError("empty weight resource")
    return records


def inspect(data, path):
    sections, resource_rva = parse_pe(data)
    weights = {}
    for resource in parse_resources(data, sections, resource_rva):
        name = resource["rname"]
        if resource["rtype"] != "10" or not name.startswith("WEIGHTS_"):
            continue
        key = name, resource["rlang"]
        start, size = resource["off"], resource["size"]
        if key in weights or start is None or not 0 <= start < start + size <= len(data):
            raise ValueError(f"duplicate or invalid PE resource {key!r}")
        with memoryview(data)[start:start + size] as blob:
            records = payload_records(blob)
            digest = sha256(blob).hexdigest()
        block_records = {}
        for record in records.values():
            match = BLOCK_RECORD.fullmatch(record["name"])
            block_records.setdefault(int(match[1]), []).append({
                "suffix": match[2],
                "payload_bytes": record["payload_bytes"],
            })
        blocks = [{"index": index, "records": block_records[index]} for index in sorted(block_records)]
        total = sum(record["payload_bytes"] for record in records.values())
        weights[key] = {
            "name": name,
            "language": resource["rlang"],
            "file_offset": start,
            "size": size,
            "sha256": digest,
            "record_count": len(records),
            "payload_bytes": total,
            "envelope_bytes": size - total,
            "records_with_trailer_count_times_two_equal_payload_bytes": sum(
                record["trailer_count"] * 2 == record["payload_bytes"] for record in records.values()
            ),
            "blocks": blocks,
            "records": records,
        }
    if not weights:
        raise ValueError(f"no WEIGHTS_* RCDATA resource in {path}")
    return {"path": str(path), "size": len(data), "sha256": sha256(data).hexdigest()}, weights


def block_signature(block):
    return tuple(sorted((row["suffix"], row["payload_bytes"]) for row in block["records"]))


def compare(left_data, left, right_data, right):
    left_start, right_start = left["file_offset"], right["file_offset"]
    with (memoryview(left_data)[left_start:left_start + left["size"]] as left_blob,
          memoryview(right_data)[right_start:right_start + right["size"]] as right_blob):
        same_resource = left_blob == right_blob

    left_records, right_records = left["records"], right["records"]
    common = left_records.keys() & right_records.keys()
    same_size = [name for name in sorted(common)
                 if left_records[name]["payload_bytes"] == right_records[name]["payload_bytes"]]
    same_payload = []
    for name in same_size:
        a = left_start + left_records[name]["payload_offset"]
        b = right_start + right_records[name]["payload_offset"]
        size = left_records[name]["payload_bytes"]
        with (memoryview(left_data)[a:a + size] as left_raw,
              memoryview(right_data)[b:b + size] as right_raw):
            if left_raw == right_raw:
                same_payload.append(name)

    def payload_keys(records):
        return Counter((record["payload_bytes"], record["payload_sha256"]) for record in records.values())

    baseline_blocks = {}
    for block in left["blocks"]:
        baseline_blocks.setdefault(block_signature(block), []).append(block["index"])
    block_matches = [{"candidate_block": block["index"],
                      "baseline_blocks": baseline_blocks.get(block_signature(block), [])}
                     for block in right["blocks"]]

    return {
        "baseline_resource": [left["name"], left["language"]],
        "candidate_resource": [right["name"], right["language"]],
        "resource_byte_identical": same_resource,
        "same_record_name_count": len(common),
        "same_name_and_size_count": len(same_size),
        "same_name_and_size": same_size,
        "same_name_and_payload_byte_identical_count": len(same_payload),
        "same_name_and_payload_byte_identical": same_payload,
        "matching_payload_digests_regardless_of_name_count": sum(
            (payload_keys(left_records) & payload_keys(right_records)).values()
        ),
        "candidate_blocks_with_baseline_storage_signature_count": sum(bool(row["baseline_blocks"])
                                                                     for row in block_matches),
        "block_storage_signature_matches": block_matches,
    }


def compare_files(baseline, candidate, include_records=False):
    baseline, candidate = baseline.resolve(), candidate.resolve()
    with (baseline.open("rb") as baseline_stream, candidate.open("rb") as candidate_stream,
          mmap(baseline_stream.fileno(), 0, access=ACCESS_READ) as left,
          mmap(candidate_stream.fileno(), 0, access=ACCESS_READ) as right):
        left_file, left_weights = inspect(left, baseline)
        right_file, right_weights = inspect(right, candidate)
        comparisons = [compare(left, a, right, b) for a in left_weights.values() for b in right_weights.values()]
    for file, weights in ((left_file, left_weights), (right_file, right_weights)):
        file["resources"] = list(weights.values())
        for resource in file["resources"]:
            records = resource.pop("records")
            if include_records:
                resource["records"] = list(records.values())
    return {
        "baseline": left_file,
        "candidate": right_file,
        "comparisons": comparisons,
        "scope": "PE resource and record storage; trailer counts are not learned parameter counts",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--records", action="store_true", help="include each payload offset, size and SHA-256")
    args = parser.parse_args()
    try:
        report = compare_files(args.baseline, args.candidate, args.records)
    except (OSError, ValueError, struct.error) as error:
        print(f"Weight inspection failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
