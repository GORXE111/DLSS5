"""Finite odd-size / fractional-history follow-up for original pre RGB rounding."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from acceptance import metrics
from check import color_pattern
from check_reset import pre_reference
from test_reset import pre_block
from dlss5 import DLSS5
from dlss5 import net as N
from harness import reference as REF
from tools.report_paths import ReportPath


def sha(value):
    if torch.is_tensor(value):value=value.detach().cpu().numpy()
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def capture(folder,case):
    args=SimpleNamespace(out=folder,suite='batch4-pre-round',kind='port-sm86',harness=ROOT/'harness/batch3/nr-lab.exe',
        bridge=ROOT/'harness/nvngx.dll',dll=ROOT/'sm86_port/nvngx_dlssnr.dll',support=ROOT/'dlss5',timeout=90)
    with patch.object(REF,'suite_cases',lambda suite:[case]),patch.dict(os.environ,{'DLSS5_PROF':'1','DLSS5_TRACE':'0','DLSS5_TAP':''}):
        if not folder.exists() and REF.capture(args):raise ValueError('Failed capture')
        manifest=REF.load_capture(folder)
    path=folder/case['id']/next(iter(manifest['cases'][0]['outputs']))
    return REF.read_ppm(path).astype(np.float32)/255,dict(path=str(folder),manifest_sha256=REF.digest(folder/'manifest.json'),
        output_sha256=REF.digest(path),runtime_sha256=manifest['runtime_sha256'],args=case['args'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,default=ROOT/'research/validation/batch4_pre_round_diagnosis.json')
    p.add_argument('--capture',type=Path,default=ROOT/'research/batch4/pre_round_diagnosis')
    args=p.parse_args()
    guard=ReportPath(args.report,input_dirs=(args.capture,))
    report=dict(passed=False,complete=False,errors=[],checks=[],references={},frozen_model_sources_modified=False,
        thresholds=dict(min_correlation=.999,max_mae_255=3.0),scope='Bounded global pre rounding experiment; no native parity claim')
    def save():guard.write_json(report)
    save()
    try:
        names=['torch/diagnose_pre_round.py','torch/check_reset.py','torch/dlss5/net.py','torch/dlss5/ops.py',
               'torch/dlss5/ptx_math.py','torch/dlss5/blocks.py','fatbins/fatbin_01.1.sm_120.ptx',
               'sm86_port/fatbin_01.sm86.cubin','sm86_port/nvngx_dlssnr.dll','WEIGHTS_HT.bin','weights_map.json']
        report['source_sha256']={n:REF.digest(ROOT/n) for n in names}
        specs=[('odd_reset',641,361,1,0,0,0,0),('fraction_f0',640,360,1,3,2,-3.5,-1.75),('fraction_f1',640,360,2,3,2,-3.5,-1.75)]
        refs={}
        for name,w,h,frames,sx,sy,mx,my in specs:
            case=dict(id=name,args=['--nr-only','--model','1','--input',f'{w}x{h}','--output',f'{w}x{h}','--frames',str(frames),
                '--scene-variant','0','--temporal-shift',str(sx),'--temporal-shift-y',str(sy),'--motion-x',str(mx),'--motion-y',str(my)])
            refs[name],report['references'][name]=capture(args.capture/name,case)
            save()
        original=N.PreBlock.inputs
        def rounded(self,color,hist,mv,frame,ctrl=None):
            x=original(self,color,hist,mv,frame,ctrl)
            rgb=(x[:,:6]*8+.5).half()
            return torch.cat((((rgb-.5)*.125).float(),x[:,6:]),-1)
        # Compare original PTX preprocessing features before the neural adapter.
        h,w=37,53;color=color_pattern(w,h);tc=torch.from_numpy(color).cuda()
        block=pre_block('cuda');base=original(block,tc,None,None,0)
        after=rounded(block,tc,None,None,0)
        gh,gw=N.dims(h,w)['grid']
        reference,ptx_sha=pre_reference(color,None,None)
        before_features=base.view(gh,gw,15)[:h,:w,:6].half().float().cpu().numpy()
        after_features=after.view(gh,gw,15)[:h,:w,:6].cpu().numpy()
        report['pre_kernel_feature_check']=dict(size_wh=[w,h],probe_ptx_sha256=ptx_sha,
            before_output_sha256=sha(before_features),output_sha256=sha(after_features),reference_sha256=sha(reference),
            before_mismatch_values=int(np.count_nonzero(before_features!=reference)),
            after_mismatch_values=int(np.count_nonzero(after_features!=reference)),
            noise_controls_bit_exact=bool(torch.equal(base[:,6:],after[:,6:])),
            passed=bool(np.array_equal(after_features,reference) and torch.equal(base[:,6:],after[:,6:])))
        model=DLSS5()
        for variant in ('before','rounded'):
            with patch.object(N.PreBlock,'inputs',original if variant=='before' else rounded):
                out=model(color_pattern(641,361),frame=0)
                for name,image in [('odd_reset',out)]:
                    m=metrics(image.cpu().numpy(),refs[name],quantize=True)
                    report['checks'].append(dict(name=name,variant=variant,metrics=m,output_sha256=sha(image),passed=m['correlation']>=.999 and m['mae']<=3))
                    save();print(variant,name,m,flush=True)
                hist=None
                mv=torch.empty(360,640,2,device='cuda');mv[...,0],mv[...,1]=-3.5,-1.75
                for frame in (0,1):
                    color=color_pattern(offset_x=3*frame,offset_y=2*frame)
                    out=model(color,hist=hist,mv=mv,frame=frame)
                    name=f'fraction_f{frame}';m=metrics(out.cpu().numpy(),refs[name],quantize=True)
                    report['checks'].append(dict(name=name,variant=variant,metrics=m,output_sha256=sha(out),input_sha256=sha(color),
                        history_sha256=None if hist is None else sha(hist),motion_sha256=sha(mv),passed=m['correlation']>=.999 and m['mae']<=3))
                    hist=out
                    save();print(variant,name,m,flush=True)
        report['complete']=len(report['checks'])==6
        report['passed']=report['complete'] and report['pre_kernel_feature_check']['passed'] and all(row['passed'] for row in report['checks'])
    except Exception as error:report['errors'].append(f'{type(error).__name__}: {error}')
    finally:save()
    return 0 if report['passed'] else 1


if __name__=='__main__':
    torch.set_num_threads(4)
    raise SystemExit(main())
