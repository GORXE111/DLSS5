"""Read projected Q/K, norm denominators, inverse norms and normalized halves."""

import argparse
from contextlib import ExitStack
import ctypes as C
from functools import partial
import json
from pathlib import Path
import re
import struct
import subprocess
import sys

import numpy as np

from ..align_nr_small_trace import GRAPH, GRAPH_SHA, kernel_inventory, pinned_bytes, read_trace
from ..capture_october_taps import capture_path
from ..compare_october_taps import PORT, seal_capture
from .check_october_attention import buffer_offsets, encode_fp8, f16, half_literal, sha
from .check_october_attention_tensor import half_metrics
from .check_october_live_repack import tap_index
from .check_october_live_weights import LAYOUT, LAYOUT_SHA, run as verify_weights
from . import check_october_qkv
from .check_october_qkv import CUDA_BLOCK, CUDA_GRID, ENTRY, TOKENS, WIDTH, input_case, physical_split, project_qkv
from .mma_sm86_tiles import Sm86Mma
from .probe_october_attention_replay import GUARD
from .probe_october_attention_stages import Driver, PTX_SHA, PTXAS_SHA, replace_once


PLANES = ('projected', 'denominator', 'inverse', 'normalized')
ELEMENTS = TOKENS * WIDTH
PLANE_BYTES = ELEMENTS * 2


def instrument(data):
    if sha(data) != PTX_SHA:
        raise ValueError('not the pinned rewritten October module 6 PTX')
    source = data.decode('ascii')
    start = source.index(f'.visible .entry {ENTRY}(')
    stop = source.index('.visible .entry ', start + 1)
    body = source[start:stop]
    arg = f'{ENTRY}_param_0'
    if f'[{arg}+56]' in body:
        raise ValueError('QKV already uses the proposed scratch argument')
    prologue = f'''
.reg .pred %nr_norm_valid;
.reg .b32 %nr_norm_head, %nr_norm_tile0, %nr_norm_tile, %nr_norm_element;
.reg .b64 %nr_norm_raw, %nr_norm_scratch, %nr_norm_offset, %nr_norm_addr;
ld.param.b64 %nr_norm_raw, [{arg}+56];
cvta.to.global.u64 %nr_norm_scratch, %nr_norm_raw;
'''
    body = replace_once(body, rf'^ld\.param\.b64 %rd1, \[{arg}\];$', lambda m: prologue + m[0])
    base = '''
and.b32 %nr_norm_head, %r8, 1;
mad.lo.u32 %nr_norm_head, %r5, 2, %nr_norm_head;
shl.b32 %nr_norm_tile0, %r8, 1;
and.b32 %nr_norm_tile0, %nr_norm_tile0, -4;
add.u32 %nr_norm_tile0, %nr_norm_tile0, %r6;
'''
    body = replace_once(body, r'^mov\.u32 %r8, %tid\.y;$', lambda m: m[0] + base)
    inverse = dict((dst, src) for src, dst in re.findall(
        r'mov\.b32 \{hl, hu\}, (%r\d+);\s+cvt\.f32\.f16 fl, hl;\s+cvt\.f32\.f16 fu, hu;\s+'
        r'rsqrt\.approx\.ftz\.f32 fl, fl;\s+rsqrt\.approx\.ftz\.f32 fu, fu;\s+'
        r'cvt\.rn\.f16\.f32 hl, fl;\s+cvt\.rn\.f16\.f32 hu, fu;\s+mov\.b32 (%r\d+), \{hl, hu\};', body))
    if len(inverse) != 16:
        raise ValueError('expected exactly sixteen Q/K inverse-norm register pairs')
    mul = list(re.finditer(r'^\{mul\.f16x2 (%r\d+),(%r\d+),(%r\d+);\n\}', body, re.M))
    normalize = [match for match in mul if match[3] in inverse]
    q_scale = {match[2]: match[1] for match in mul if match[3] == '%r708'}
    q_tau = {match[2]: match for match in mul if match[3] == '%r774'}
    quant = {register: int(slot) for register, slot in re.findall(
        r'\{ \.reg \.b32 %z_a[^}]*?and\.b32 %z_a, (%r\d+), 0x7FFF7FFF;[^}]*?cvt\.u16\.u32 %rs(\d+), %z_r;\n\}', body)}
    if len(normalize) != 64 or len(q_scale) != 32 or len(q_tau) != 32:
        raise ValueError('unexpected Q/K normalization, scale or temperature anchors')
    inserts = {}
    used = [set(), set()]
    for index, match in enumerate(normalize):
        role = index // 32
        final = q_tau[q_scale[match[1]]] if role == 0 else match
        slot = quant[final[1]] - (9 if role == 0 else 57)
        used[role].add(slot)
        tile, pair = divmod(slot, 8)
        addr = f'''
add.u32 %nr_norm_tile, %nr_norm_tile0, {tile};
setp.lt.u32 %nr_norm_valid, %nr_norm_tile, %r1457;
mad.lo.u32 %nr_norm_element, %nr_norm_tile, 32, %nr_norm_head;
mad.lo.u32 %nr_norm_element, %nr_norm_element, 32, %r65;
mad.lo.u32 %nr_norm_element, %nr_norm_element, 16, {pair * 2};
mul.wide.u32 %nr_norm_offset, %nr_norm_element, 2;
add.u64 %nr_norm_addr, %nr_norm_scratch, %nr_norm_offset;
'''
        offset = role * len(PLANES) * PLANE_BYTES
        stores = (match[2], inverse[match[3]], match[3], final[1])
        for stage, register in enumerate(stores):
            anchor = final.end() if stage == 3 else match.end()
            if anchor not in inserts:
                inserts[anchor] = addr
            inserts[anchor] += f'@%nr_norm_valid st.global.b32 [%nr_norm_addr+{offset + stage * PLANE_BYTES}], {register};\n'
    if used != [set(range(32)), set(range(32))]:
        raise ValueError('Q/K scratch locations do not match output lane packing')
    for pos, inserted in sorted(inserts.items(), reverse=True):
        body = body[:pos] + inserted + body[pos:]
    return (source[:start] + body + source[stop:]).encode('ascii')


def execute(device, function, shape, raw, weight):
    activation, *expected = raw
    poison = [bytes(np.frombuffer(data, np.uint8) ^ np.uint8(255)) for data in expected]
    nan = b'\x00\x7e' * (2 * len(PLANES) * ELEMENTS)
    payloads = [activation, *poison, weight, b'\xff' * (4 * CUDA_GRID[0]), bytes(6 * ELEMENTS), nan]
    with ExitStack() as cleanup:
        pointers = []
        for data in payloads:
            wrapped = GUARD + data + GUARD
            ptr = C.c_uint64()
            device.call('cuMemAlloc_v2', C.byref(ptr), len(wrapped))
            cleanup.callback(device.call, 'cuMemFree_v2', ptr)
            device.call('cuMemcpyHtoD_v2', ptr, C.create_string_buffer(wrapped), len(wrapped))
            pointers.append(ptr)
        active = [ptr.value + len(GUARD) for ptr in pointers]
        abi = C.create_string_buffer(struct.pack('<9Qii', *active, 0, *shape[:2]))
        args = (C.c_void_p * 1)(C.addressof(abi))
        device.call('cuLaunchKernel', function, *CUDA_GRID, *CUDA_BLOCK, 0, None, args, None)
        device.call('cuCtxSynchronize')
        outputs = []
        for ptr, data in zip(pointers, payloads):
            host = C.create_string_buffer(len(data) + 2 * len(GUARD))
            device.call('cuMemcpyDtoH_v2', host, ptr, len(host))
            if host.raw[:len(GUARD)] != GUARD or host.raw[-len(GUARD):] != GUARD:
                raise ValueError('instrumented QKV wrote beyond an allocation')
            outputs.append(host.raw[len(GUARD):-len(GUARD)])
    if outputs[:5] != [*raw, weight] or set(np.frombuffer(outputs[5], '<i4')) != {1}:
        raise ValueError('instrumented QKV differs from sealed activation, weights, outputs or split-K completion')
    planes = np.frombuffer(outputs[-1], '<f2').reshape(2, len(PLANES), ELEMENTS)
    if not np.isfinite(planes).all():
        raise ValueError('missing or nonfinite values in the Q/K normalization planes')
    return planes


def host_denominator(projected, fused):
    value = projected.astype(np.float64).reshape(32, TOKENS, 4, 4, 2)
    square = f16(value * value)
    if fused == 'lower':
        paired = f16(value[:, :, :2] ** 2 + square[:, :, 2:])
    elif fused == 'upper':
        paired = f16(square[:, :, :2] + value[:, :, 2:] ** 2)
    else:
        paired = f16(square[:, :, :2] + square[:, :, 2:])
    local = f16(paired[:, :, 0] + paired[:, :, 1])
    shuffled = f16(f16(local[..., 0, :] + local[..., 2, :]) + f16(local[..., 1, :] + local[..., 3, :]))
    summed = f16(shuffled[..., :1] + shuffled[..., 1:])
    return np.maximum(summed, half_literal(0x388205FF))


def dump_case(block, shape, sequence, raw, weight, packed, work):
    report = {'block': block, 'sequence': sequence, 'shape_hwc': shape, 'weight_sha256': sha(weight),
              'tap_sha256': dict(zip(('input', 'q', 'k', 'v'), map(sha, raw))), 'roles': {}}
    for role, data, indices, expected in zip(('q', 'k'), packed, buffer_offsets(TOKENS), raw[1:]):
        arrays = dict(zip(PLANES, (np.ascontiguousarray(plane[indices]) for plane in data)))
        projected, denominator, inverse, normalized = arrays.values()
        if (not np.all(denominator == denominator[..., :1]) or not np.all(inverse == inverse[..., :1]) or
                not np.array_equal(encode_fp8(normalized.astype(np.float32)), np.frombuffer(expected, np.uint8)[indices])):
            raise ValueError(f'captured {role} planes disagree with output or per-head normalization')
        recalc = f16(projected.astype(np.float32) * inverse)
        if role == 'q':
            temperature = f16(np.frombuffer(weight[:128], '<f4'))[:, None, None]
            recalc = f16(f16(recalc * np.float16(np.sqrt(32))) * temperature)
        denom = denominator[..., :1]
        row = {'host_inverse_from_measured_denominator': half_metrics(f16(1 / np.sqrt(denom.astype(np.float32))), inverse[..., :1]),
               'host_norm_from_measured_inverse': half_metrics(recalc, normalized),
               'host_denominator_from_measured_projection': {
                   name: half_metrics(host_denominator(projected, name), denom) for name in ('separate', 'lower', 'upper')},
               'planes': {}}
        for name, plane in arrays.items():
            filename = f'block{block:02d}-{role}-{name}.f16'
            payload = plane.tobytes()
            with (work / filename).open('xb') as stream:
                stream.write(payload)
            row['planes'][name] = {'file': filename, 'bytes': len(payload), 'sha256': sha(payload)}
        report['roles'][role] = row
    return report


def run(capture, weights, ptx, ptxas, work, blocks):
    if not blocks or len(set(blocks)) != len(blocks) or not set(blocks) <= set(range(11, 19)):
        raise ValueError('select distinct blocks 11 through 18')
    work = capture_path(work)
    if any(work.is_relative_to(path.resolve()) for path in (capture, weights)):
        raise ValueError('work must be outside the sealed activation and weight inputs')
    ptxas = ptxas.resolve(strict=True)
    if sha(ptxas.read_bytes()) != PTXAS_SHA:
        raise ValueError('not the pinned ptxas tool')
    inventory = kernel_inventory()
    graph = json.loads(pinned_bytes(GRAPH, GRAPH_SHA))
    sealed, files = seal_capture(capture, inventory, graph)
    if sealed['kind'] != PORT:
        raise ValueError('expected a sealed October staged Ampere capture')
    trace = pinned_bytes(capture / 'nr-trace.tsv', sealed['artifact_sha256']['nr-trace.tsv'])
    launches = {row.sequence: row for row in read_trace(trace, inventory)[sealed['frame']]}
    weight_check = verify_weights(weights.resolve(), 0)
    if not weight_check['passed']:
        raise ValueError('failed to verify the staged QKV weights')
    weight_trace = read_trace((weights / 'nr-trace.tsv').read_bytes(), inventory)[0]
    pre = next(row for row in weight_trace if row.sequence == weight_check['pre_sequence'])
    _, packed = tap_index(weights.resolve(), 0)[pre.sequence, pre.pointer(232)]
    records = {row['name']: row for row in weight_check['records']}
    layout = {row['name']: row for row in json.loads(pinned_bytes(LAYOUT, LAYOUT_SHA))['records']}
    inputs = {}
    with packed.open('rb') as stream:
        for block in blocks:
            name = f'block{block}.layer2.layer'
            record = records[name]
            stream.seek(record['gpu_buffer_offset'])
            weight = stream.read(record['bytes'])
            if len(weight) != 128 + 3 * WIDTH ** 2 or sha(weight) != layout[name]['payload_sha256']:
                raise ValueError(f'block {block} QKV weight changed after verification')
            inputs[block] = (*input_case(block, sealed, files, launches), weight)
    transformed = instrument(ptx.read_bytes())
    work.mkdir(parents=True, exist_ok=False)
    source = work / 'module06.qkv-norm.ptx'
    cubin = work / 'module06.qkv-norm.cubin'
    source.write_bytes(transformed)
    with (work / 'compile.log').open('wb') as log:
        subprocess.run([str(ptxas), '-arch=sm_86', str(source), '-o', str(cubin)],
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
    report = {'schema': 'october_nr_qkv_norm_planes_v1', 'source_sha256': sha(Path(__file__).read_bytes()),
              'capture_seal_sha256': sealed['capture_seal_sha256'], 'frame': sealed['frame'],
              'packed_weights_sha256': weight_check['gpu_packed_sha256'], 'rewritten_ptx_sha256': PTX_SHA,
              'ptxas_sha256': PTXAS_SHA, 'instrumented_ptx_sha256': sha(transformed),
              'instrumented_cubin_sha256': sha(cubin.read_bytes()), 'entry': ENTRY,
              'projection_source_sha256': sha(Path(check_october_qkv.__file__).read_bytes()),
              'format': 'little-endian binary16 in logical [head,token,channel] order',
              'plane_shape': [32, TOKENS, 32], 'cases': []}
    with ExitStack() as cleanup:
        device = Driver(0)
        cleanup.callback(device.close)
        copy = device.lib.cuMemcpyHtoD_v2
        copy.argtypes, copy.restype = [C.c_uint64, C.c_void_p, C.c_size_t], C.c_int
        module = device.load(cubin.read_bytes())
        cleanup.callback(device.call, 'cuModuleUnload', module)
        function = device.function(module, ENTRY)
        report.update(gpu=device.label, compute_capability=device.capability, cuda_driver_api_version=device.version)
        for block, (shape, sequence, raw, weight) in inputs.items():
            measured = execute(device, function, shape, raw, weight)
            report['cases'].append(dump_case(block, shape, sequence, raw, weight, measured, work))
    with Sm86Mma() as mma:
        multiply = partial(physical_split, mma)
        for case in report['cases']:
            _, _, raw, weight = inputs[case['block']]
            projected = project_qkv(raw[0], weight, multiply).transpose(2, 1, 0, 3)
            for role, predicted in zip(('q', 'k'), projected):
                role_report = case['roles'][role]
                path = work / role_report['planes']['projected']['file']
                observed = np.fromfile(path, '<f2').reshape(predicted.shape)
                role_report['independent_sm86_projection'] = half_metrics(predicted, observed)
    report['passed'] = True
    (work / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('capture', 'weights'):
        parser.add_argument(name, type=Path)
    for name in ('ptx', 'ptxas', 'work'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--blocks', nargs='+', type=int, default=list(range(11, 19)))
    args = parser.parse_args()
    try:
        report = run(args.capture, args.weights, args.ptx, args.ptxas, args.work, args.blocks)
    except (OSError, ValueError, KeyError, RuntimeError, struct.error, subprocess.SubprocessError) as error:
        print(json.dumps({'passed': False, 'error': str(error)}, indent=2))
        return 1
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
