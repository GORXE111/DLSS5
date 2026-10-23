"""Bounded diagnosis of the high-control Reset regression; no acceptance override.

This records current/previous DLL identities and keeps every failing candidate.
Pre-rounding is an experiment implemented only by an in-process method wrapper.
Frozen model sources, parameters and numerical thresholds are never changed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from acceptance import metrics
from check import color_pattern
from dlss5 import DLSS5
from dlss5 import net as N
from harness import reference as REF
from tools.report_paths import ReportPath


def sha(value):
    if torch.is_tensor(value):value=value.detach().cpu().numpy()
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def capture_one(folder,dll,high):
    extra=['--local-tone','1.5','--local-structure','1.5'] if high else []
    case=dict(id='high' if high else 'default',args=['--nr-only','--model','1','--input','640x360','--output','640x360',
        '--frames','1','--scene-variant','0',*extra])
    args=SimpleNamespace(out=folder,suite='batch4-controls-diagnosis',kind='port-sm86',harness=ROOT/'harness/batch3/nr-lab.exe',
        bridge=ROOT/'harness/nvngx.dll',dll=dll,support=ROOT/'dlss5',timeout=90)
    with patch.object(REF,'suite_cases',lambda suite:[case]),patch.dict(os.environ,{'DLSS5_PROF':'1','DLSS5_TRACE':'0','DLSS5_TAP':''}):
        if not folder.exists() and REF.capture(args):raise RuntimeError('Reference capture failed')
        manifest=REF.load_capture(folder)
    trace=folder/'_runtime/nr-trace.tsv'
    rows=[]
    for line in trace.read_text().splitlines():
        fields=line.split('\t')
        if fields[2] not in ('cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8','cc_tinlayout_fused_post_block_swin_1h_32_fp8'):continue
        blob=bytes.fromhex(fields[7]);pre='pre_block' in fields[2]
        offsets=(8,16) if pre else (88,96)
        row=dict(seq=int(fields[0]),kernel=fields[2],parameter_bytes=len(blob),
            history_handle_hex=hex(struct.unpack_from('<Q',blob,offsets[0])[0]),motion_handle_hex=hex(struct.unpack_from('<Q',blob,offsets[1])[0]),
            raw_parameters_sha256=hashlib.sha256(blob).hexdigest())
        if pre:row.update(frame=struct.unpack_from('<I',blob,200)[0],control_parameters={str(o):struct.unpack_from('<f',blob,o)[0] for o in (172,176,180,184,188)})
        rows.append(row)
    if len(rows)!=2:raise ValueError('Missing pre/post parameter evidence')
    output=folder/case['id']/next(iter(manifest['cases'][0]['outputs']))
    log=(folder/case['id']/'nr-lab.log').read_text(errors='replace')
    expected='localTone=1.500 localStructure=1.500' if high else 'localTone=1.000 localStructure=1.000'
    if expected not in log:raise ValueError('Tuning log does not match request')
    return REF.read_ppm(output).astype(np.float32)/255,dict(path=str(folder),manifest_sha256=REF.digest(folder/'manifest.json'),
        runtime_sha256=manifest['runtime_sha256'],output_sha256=REF.digest(output),trace_sha256=REF.digest(trace),parameters=rows)


def run(report,args,save):
    names=['torch/diagnose_controls.py','torch/dlss5/net.py','torch/dlss5/ops.py','torch/dlss5/ptx_math.py','torch/dlss5/blocks.py',
        'harness/reference.py','harness/batch3/nr-lab.exe','harness/nvngx.dll','sm86_port/nvngx_dlssnr.dll',
        'harness/rt_sm86/nvngx_dlssnr.dll','WEIGHTS_HT.bin','weights_map.json']
    report['source_sha256']={n:REF.digest(ROOT/n) for n in names}
    report['reference']={}
    refs={}
    for name,dll,high in [('current_high',ROOT/'sm86_port/nvngx_dlssnr.dll',True),('current_default',ROOT/'sm86_port/nvngx_dlssnr.dll',False),
                          ('previous_high',ROOT/'harness/rt_sm86/nvngx_dlssnr.dll',True)]:
        refs[name],report['reference'][name]=capture_one(args.capture/name,dll,high)
        save()
    report['dll_output_comparison']=dict(high_old_new_bit_exact=bool(np.array_equal(refs['current_high'],refs['previous_high'])),
        original_failed_case_reference_sha256='3b1246c39c00bd2df6bdf2fd74c1ac304548adc0c5cacddcab1f365ad043a38a',
        current_matches_original_failed_reference=report['reference']['current_high']['output_sha256']=='3b1246c39c00bd2df6bdf2fd74c1ac304548adc0c5cacddcab1f365ad043a38a')
    model=DLSS5()
    original=N.PreBlock.inputs
    def wrapped(kind):
        def inputs(self,color,hist,mv,frame,ctrl=None):
            if kind=='old_color_history' and hist is None:hist=color
            x=original(self,color,hist,mv,frame,ctrl)
            if kind=='ptx_input_rounding':
                # Original pre: CVT.RN.F16(Color/History), SUB.F16(.5), MUL.F16(.125).
                rgb=(x[:,:6]*8+.5).half()
                rounded=((rgb-.5)*.125).float()
                x=torch.cat((rounded,x[:,6:]),-1)
            return x
        return inputs
    configurations=[('new_fast','new',False,True),('old_fast','old_color_history',False,True),
        ('new_precise','new',True,False),('old_precise','old_color_history',True,False),
        ('rounded_fast','ptx_input_rounding',False,True),('rounded_precise','ptx_input_rounding',True,False),('fast_f32','new',False,False)]
    color=color_pattern();report['input_sha256']=sha(color)
    outputs={}
    for high in (True,False):
        control=N.control_inputs(**({'tone':1.5,'structure':1.5} if high else {}))
        for name,kind,precise,half in configurations:
            if not high and name not in ('new_fast','old_fast','rounded_fast'):continue
            model.precise,model.half=precise,half
            begin=time.perf_counter()
            with patch.object(N.PreBlock,'inputs',wrapped(kind)):
                out=model(color,hist=None,mv=None,frame=0,controls=control)
            torch.cuda.synchronize()
            output=out.cpu().numpy()
            m=metrics(output,refs['current_high' if high else 'current_default'],quantize=True)
            passed=m['correlation']>=.999 and m['mae']<=3
            key=('high' if high else 'default')+'/'+name
            report['variants'].append(dict(name=key,precise=precise,half=half,controls=control,pre_experiment=kind,
                metrics=m,meets_fixed_acceptance=passed,output_sha256=sha(output),seconds=time.perf_counter()-begin))
            outputs[key]=output
            save();print(key,m,'PASS' if passed else 'FAIL',flush=True)
    report['pairs']={}
    for level in ('high','default'):
        a=outputs[level+'/new_fast']
        for alternative in ('old_fast','rounded_fast'):
            b=outputs[level+'/'+alternative]
            report['pairs'][level+'/'+alternative]=dict(mae_255=float(np.abs(a-b).mean()*255),max_255=float(np.abs(a-b).max()*255),
                changed_values=int(np.count_nonzero(a!=b)))
    report['complete']=len(report['variants'])==10
    report['unresolved_acceptance_failure']=not next(row['meets_fixed_acceptance'] for row in report['variants'] if row['name']=='high/new_fast')
    report['status']='unresolved' if report['unresolved_acceptance_failure'] else 'diagnosed_no_current_failure'
    report['passed']=not report['unresolved_acceptance_failure']
    report['diagnostic_complete']=report['complete']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,default=ROOT/'research/validation/batch4_controls_diagnosis.json')
    parser.add_argument('--capture',type=Path,default=ROOT/'research/batch4/controls_diagnosis')
    args=parser.parse_args()
    output=ReportPath(args.report,input_dirs=(args.capture,))
    report=dict(passed=False,complete=False,diagnostic_complete=False,errors=[],variants=[],scope='Bounded current port diagnosis; no native parity claim',
        thresholds=dict(min_correlation=.999,max_mae_255=3.0),frozen_model_sources_modified=False)
    def save():output.write_json(report)
    save()
    try:
        report['device']=torch.cuda.get_device_name(0)
        run(report,args,save)
    except Exception as error:report['errors'].append(f'{type(error).__name__}: {error}')
    finally:save()
    return 0 if report['passed'] else 1


if __name__=='__main__':
    torch.set_num_threads(4)
    raise SystemExit(main())
