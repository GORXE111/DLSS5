"""Record minimal read-only evidence for the 310.8.0 host transfer parameter binding.

python -m tools.trace_transfer_host --report research/validation/transfer_host_static_batch2.json
No CUDA context, executable invocation, patching, or proprietary disassembly dump is used.
"""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import struct

from capstone import x86

from .host_trace import Image
from .report_paths import ReportPath


SOURCE_SHA256 = "e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e"
ROOT = Path(__file__).resolve().parents[1]


def instruction_graph(instructions):
    addresses = {i.address for i in instructions}
    edges = {}
    for ins in instructions:
        successors = []
        if ins.mnemonic not in ("ret", "retf", "int3", "ud2"):
            if ins.mnemonic != "jmp":
                successors.append(ins.address + ins.size)
            if (ins.mnemonic.startswith("j") or ins.mnemonic.startswith("loop")) and ins.operands:
                if ins.operands[0].type == x86.X86_OP_IMM:
                    successors.append(ins.operands[0].imm)
                elif ins.mnemonic == "jmp":
                    raise ValueError("Indirect jump prevents this simple dominance check")
        edges[ins.address] = [s for s in successors if s in addresses]
    return edges


def reachable(edges, start, *, blocked=None, stop=None):
    found, queue = set(), deque([start])
    while queue:
        address = queue.popleft()
        if address in found or address == blocked:
            continue
        found.add(address)
        if address != stop:
            queue.extend(edges.get(address, []))
    return found


def record(image, address):
    raw = image.pe.get_data(address - image.base, 16)
    ins = next(image.md.disasm(raw, address))
    return {"va": hex(address), "size": ins.size,
            "sha256": hashlib.sha256(ins.bytes).hexdigest()}


def analyze(image):
    source_hash = hashlib.sha256(image.blob).hexdigest()
    if source_hash != SOURCE_SHA256:
        raise ValueError("Only the identified original 310.8.0 image is supported")
    call, zero, callee = 0x180019884, 0x1800188D9, 0x18001C920
    start, end = image.function(call)
    instructions = list(image.instructions(start, end))
    graph = instruction_graph(instructions)
    reverse = {a: [] for a in graph}
    for src, destinations in graph.items():
        for dst in destinations:
            reverse[dst].append(src)
    on_path = reachable(graph, zero, stop=call) & reachable(reverse, call)
    rsi_registers = {x86.X86_REG_RSI, x86.X86_REG_ESI, x86.X86_REG_SI, x86.X86_REG_SIL}
    writes = [ins for ins in instructions if ins.address in on_path and ins.address != zero and
              set(ins.regs_access()[1]) & rsi_registers]
    direct_refs = [(ins, target) for ins, target in image.references(callee) if ins.mnemonic in ("call", "jmp")]
    raw_refs = []
    for section in image.pe.sections:
        if not section.Characteristics & 0x20000000:
            continue
        data, base = section.get_data(), image.base + section.VirtualAddress
        for index in range(len(data) - 4):
            if data[index] in (0xE8, 0xE9) and base + index + 5 + struct.unpack_from("<i", data, index + 1)[0] == callee:
                raw_refs.append(hex(base + index))
    absolute_refs = []
    offset = 0
    while (hit := image.blob.find(struct.pack("<Q", callee), offset)) >= 0:
        absolute_refs.append(hex(image.base + image.pe.get_rva_from_offset(hit)))
        offset = hit + 1
    evidence_addresses = {
        "caller_zero_and_null_slots": [zero, 0x18001985D, 0x180019862, call],
        "callee_optional_descriptor_sources": [0x18001C929, 0x18001C949, 0x18001C953,
                                                0x18001C970, 0x18001C980, 0x18001C985,
                                                0x18001C989, 0x18001C98E, 0x18001C994, 0x18001C999, 0x18001C99F],
        "parameter_base": [0x18001CABF, 0x18001CAA0],
        "base_texture_and_sampler_selector": [0x18001CDDA, 0x18001CDE6, 0x18001CDF6,
                                               0x18001CDFC, 0x18001CE03, 0x18001CE11],
        "target_texture_and_sampler_selector": [0x18001CE27, 0x18001CE36, 0x18001CE5B,
                                                 0x18001CE61, 0x18001CE68, 0x18001CE76],
        "enable_and_transfer_grid": [0x18001CEA3, 0x18001CEAE, 0x18001CEB2, 0x18001CEB6,
                                      0x18001CEBC, 0x18001CEC2, 0x18001CECB, 0x18001CED4,
                                      0x18001CEDC, 0x18001CEE4],
        "subrect_descriptor_builder": [0x18001C53E, 0x18001C574, 0x18001C596,
                                         0x18001C5A0, 0x18001C5AF, 0x18001C5B4,
                                         0x18001C5BF, 0x18001C5D0, 0x18001C5DD,
                                         0x18001C5EA, 0x18001C5FF, 0x18001C608,
                                         0x18001C619, 0x18001C622],
        "d3d12_sampler_heap_and_selector_slots": [0x18005DF6F, 0x18005E004,
                                                  0x18005E026, 0x18005E04D,
                                                  0x18005E052, 0x18005E095,
                                                  0x18005E09C, 0x18005E0A2,
                                                  0x18005E0E1, 0x18005E0F0,
                                                  0x18005E0F6, 0x18005E135,
                                                  0x18005E145, 0x18005E170],
        "d3d12_sampler_lookup_and_nvapi_binding": [0x18005D6C4, 0x18005D7AC,
                                                   0x18005D7BC, 0x18005D7DA,
                                                   0x18005D7E2, 0x18005D7E5,
                                                   0x18005D7ED, 0x18005D7F1,
                                                   0x18005D805, 0x18005D823,
                                                   0x18005D83D, 0x18005D844,
                                                   0x18005D848],
        "d3d12_backend_constructor_vtable": [0x18005B53D, 0x18005B544],
    }
    sampler_constants = {}
    for name, address in (("point", 0x1800B7770), ("linear", 0x1800B7780),
                          ("anisotropic", 0x1800B7790)):
        raw = image.pe.get_data(address - image.base, 16)
        filter_value, address_u, address_v, address_w = struct.unpack("<4I", raw)
        sampler_constants[name] = {"va": hex(address), "size": len(raw),
                                   "sha256": hashlib.sha256(raw).hexdigest(),
                                   "filter": filter_value,
                                   "address_u_v_w": [address_u, address_v, address_w]}
    result = {
        "schema": 1, "scope": "CPU static evidence only; original 310.8.0 image",
        "source_dll": str(image.path.resolve()), "source_sha256": source_hash,
        "image_base": hex(image.base), "postprocess_host_entry": hex(callee),
        "caller": {"va": hex(call), "function": [hex(start), hex(end)],
                   "null_arg8_store": "0x180019862: [caller rsp+0x38] = rsi",
                   "null_arg9_store": "0x18001985d: [caller rsp+0x40] = rsi",
                   "zero_va": hex(zero), "zero_dominates_call_in_direct_cfg": call not in reachable(graph, start, blocked=zero),
                   "call_reachable_in_direct_cfg": call in reachable(graph, start),
                   "rsi_writes_on_paths_between_zero_and_call": [record(image, ins.address) for ins in writes],
                   "abi_assumption": "Windows x64 callees preserve nonvolatile RSI"},
        "references": {"decoded_direct_call_or_jump_sites": [hex(ins.address) for ins, _ in direct_refs],
                       "raw_rel32_call_or_jump_sites": raw_refs,
                       "absolute_64bit_pointer_sites": absolute_refs,
                       "limitation": "This scan cannot exclude runtime-computed indirect targets, self-modifying code, or external callers into internal functions."},
        "parameter_contract": {
            "stack_parameter_base": "rbp-0x30, zero-initialized 0x178 bytes",
            "224": "texture from resource descriptor arg8, selector 0 via backend vtable+0xa8",
            "256": "texture from resource descriptor arg9, selector 2 via backend vtable+0xa8",
            "300": "set to 1 only after both textures are non-null and target subrect width/height are positive",
            "304": "int conversion of target descriptor width",
            "308": "int conversion of target descriptor height",
            "descriptor_fields": "resource qword+0, base int2+8, optional extent uint2+16; builder emits base float2, extent float2, inverse native resource float2",
            "descriptor_builder_va": "0x18001c520",
            "sampler_selector_meaning": "D3D12 backend selector 0 is MIN_MAG_MIP_LINEAR with CLAMP; selector 2 is MIN_MAG_MIP_POINT with CLAMP (see d3d12_sampler_contract)",
            "public_ngx_keys": "No public transfer-specific key has been established; the sole observed internal caller explicitly passes null transfer descriptors."},
        "d3d12_sampler_contract": {
            "backend_vtable_va": "0x1800b6988",
            "vtable_slots": {
                "init": {"slot_va": "0x1800b6990", "target_va": hex(struct.unpack("<Q", image.pe.get_data(0xB6990, 8))[0])},
                "get_input_texture_view_handle64": {"slot_va": "0x1800b6a30", "target_va": hex(struct.unpack("<Q", image.pe.get_data(0xB6A30, 8))[0])}},
            "get_input_texture_view_handle64_vtable_offset": "0xa8",
            "get_input_texture_view_handle64_entry": "0x18005d640",
            "init_entry": "0x18005dca0",
            "init_method_name_string_va": "0x1800b6c88",
            "sampler_heap": "Init creates three sampler descriptors per GPU node; CPU handles are stored at this+0x480, with a 24-byte stride per node",
            "create_sampler_device_vtable_offset": "0xb0",
            "creation_order": [
                {"call_va": "0x18005e09c", "selector_slot": 2, "constant": "point"},
                {"call_va": "0x18005e0f0", "selector_slot": 0, "constant": "linear"},
                {"call_va": "0x18005e145", "selector_slot": 1, "constant": "anisotropic"}],
            "lookup": "this+0x480 + (3*active_node_index + selector)*8; active_node_index is read from this+0x154",
            "raw_path": "NvAPI_D3D12_GetCudaMergedTextureSamplerObject receives selected sampler at struct+0x20; call 0x18005d7f1",
            "legacy_path": "NvAPI_D3D12_GetCudaTextureObject receives selected sampler in r8; call 0x18005d848",
            "transfer_base_224": {"selector": 0, "filter": "MIN_MAG_MIP_LINEAR", "address": "CLAMP"},
            "transfer_target_256": {"selector": 2, "filter": "MIN_MAG_MIP_POINT", "address": "CLAMP"},
            "constants": sampler_constants,
            "enum_basis": "Windows SDK d3d12.h: D3D12_FILTER values point=0, linear=0x15, anisotropic=0x55; D3D12_TEXTURE_ADDRESS_MODE_CLAMP=3",
            "limitation": "Static D3D12 construction and lookup evidence. CUDA/D3D11/Vulkan sampler creation is not traced here; no enabled transfer dispatch was obtained through normal NGX Evaluate."},
        "conclusion": "The retained host helper can build transfer parameters, but its sole statically identified NGX Evaluate caller passes null arg8/arg9. This is evidence of a disconnected path in this build, not proof about other DLL versions or dynamically constructed calls.",
        "dynamic_verification": "not performed by this tool",
        "evidence": {group: [record(image, address) for address in addresses] for group, addresses in evidence_addresses.items()},
    }
    if (not result["caller"]["call_reachable_in_direct_cfg"] or
            not result["caller"]["zero_dominates_call_in_direct_cfg"] or writes or raw_refs != [hex(call)]):
        raise ValueError("The expected null-binding evidence changed")
    return result


def compare_port_host_ranges(image, port):
    ranges = []
    for name, start, end in (
            ("evaluate", 0x180018620, 0x180019E67),
            ("postprocess_binding", 0x18001C920, 0x18001D1D2),
            ("descriptor_builder", 0x18001C520, 0x18001C642),
            ("sampler_init", 0x18005DCA0, 0x18005E274),
            ("sampler_binding", 0x18005D640, 0x18005D88E)):
        original = image.pe.get_data(start - image.base, end - start)
        transformed = port.pe.get_data(start - port.base, end - start)
        ranges.append({"name": name, "start_va": hex(start), "end_va_exclusive": hex(end),
                       "size": end - start, "identical": original == transformed,
                       "original_sha256": hashlib.sha256(original).hexdigest(),
                       "port_sha256": hashlib.sha256(transformed).hexdigest()})
    return {"port_dll": str(port.path.resolve()),
            "port_sha256": hashlib.sha256(port.blob).hexdigest(),
            "all_recorded_ranges_identical": all(row["identical"] for row in ranges),
            "ranges": ranges,
            "limitation": "Equality covers the recorded host code ranges, not every DLL byte or native GPU behavior."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", type=Path, default=ROOT / "dlss5/nvngx_dlssnr.dll")
    parser.add_argument("--port", type=Path, default=ROOT / "sm86_port/nvngx_dlssnr.dll")
    parser.add_argument("--report", type=Path, default=ROOT / "research/validation/transfer_host_static_batch2.json")
    args = parser.parse_args()
    output = ReportPath(args.report, inputs=(args.dll, args.port, Path(__file__)))
    original = Image(args.dll)
    result = analyze(original)
    result["port_host_comparison"] = compare_port_host_ranges(original, Image(args.port))
    result["tool_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.write_json(result)
    print(json.dumps({"report": str(args.report), "source_sha256": result["source_sha256"],
                      "zero_dominates": result["caller"]["zero_dominates_call_in_direct_cfg"],
                      "direct_call_sites": result["references"]["decoded_direct_call_or_jump_sites"]}))


if __name__ == "__main__":
    main()
