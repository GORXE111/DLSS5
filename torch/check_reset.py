"""Reset/resource-presence evidence against retained PTX and the local sm86 DLL.

The public DLSS5 API keeps mv=None as a zero-motion default. Raw Pre/Post/Core
interfaces treat None as a missing resource. This is a port regression suite,
not native sm120 equivalence. Fixed sampling/network gates are reused unchanged.
"""
import argparse
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
from acceptance import metrics
from check import color_pattern
from dlss5 import DLSS5, ops
from dlss5 import net as N
from dlss5.export import Core
from harness import reference as REF
from test_reset import core_parts, pre_block


def sha(data):
    return hashlib.sha256(data).hexdigest()


def array_sha(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return sha(np.ascontiguousarray(value).tobytes())


def baseline_cpu(report):
    """Reproduce the old Color-as-History substitution, without model weights."""
    pre = pre_block()
    rows = []
    for h,w,dx,dy in ((37,53,3.25,-1.5),(37,53,-2.375,1.625),(63,95,0,0)):
        color=torch.from_numpy(color_pattern(w,h))
        mv=torch.empty(h,w,2); mv[...,0],mv[...,1]=dx,dy
        old=pre.inputs(color,color,mv,0)
        current=pre.inputs(color,None,mv,0)
        rows.append(dict(size_wh=[w,h],motion=[dx,dy],input_sha256=array_sha(color),
            old_output_sha256=array_sha(old),output_sha256=array_sha(current),
            old_history_values_different_from_color=int(torch.count_nonzero(old[:,:3]!=old[:,3:6])),
            new_history_values_different_from_color=int(torch.count_nonzero(current[:,:3]!=current[:,3:6])),
            max_history_change=float((current[:,3:6]-old[:,3:6]).abs().max()),
            passed=bool(torch.equal(current[:,:3],current[:,3:6]))))
    report['cpu']=rows
    report['baseline_contract']=dict(source='research/validation/batch3_independent_review.json',
        source_sha256=REF.digest(ROOT/'research/validation/batch3_independent_review.json'),
        legacy_pre='Pre.inputs(Color, Color if History is None else History, MV, frame)',
        public_default='MV=None is converted to a real zero tensor by DLSS5._forward; preserved',
        raw_resources='Pre/Post/Core: either History=None or MV=None disables history',
        reset_graph='history=False ignores explicit MV; contradictory History presence raises ValueError')


def safe_report(target,capture):
    target=target.resolve()
    if target.is_relative_to(capture.resolve()):raise ValueError('report must be outside capture')
    for directory in (ROOT/'torch',ROOT/'fatbins',ROOT/'sm86_port',ROOT/'harness',ROOT/'dlss5'):
        if target.is_relative_to(directory.resolve()):raise ValueError(f'report must be outside input directory: {directory}')
    protected=[*ROOT.joinpath('torch').rglob('*.py'),ROOT/'harness/reference.py',ROOT/'research/klab.py',
        ROOT/'fatbins/fatbin_01.1.sm_120.ptx',ROOT/'sm86_port/fatbin_01.sm86.cubin',ROOT/'sm86_port/nvngx_dlssnr.dll',
        ROOT/'research/validation/batch3_independent_review.json',ROOT/'research/tapspost/nr-trace.tsv',
        ROOT/'WEIGHTS_HT.bin',ROOT/'weights_map.json',ROOT/'harness/batch3/nr-lab.exe',ROOT/'harness/nvngx.dll',
        ROOT/'dlss5/nvngx_dlss.dll',ROOT/'dlss5/nvngx_dlssg.dll']
    if capture.is_dir():protected.extend(path for path in capture.rglob('*') if path.is_file())
    for source in protected:
        if str(target).casefold()==str(source.resolve()).casefold() or (target.exists() and source.exists() and os.path.samefile(target,source)):
            raise ValueError(f'report aliases reference/model input: {source}')


def pre_probe_ptx():
    text=(ROOT/'fatbins/fatbin_01.1.sm_120.ptx').read_text()
    fn=text.split('.visible .entry cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8(',1)[1].split('.visible .entry',1)[0]
    start='setp.eq.b64 %p8, %rd2, 0;'
    body=start+fn.split(start,1)[1].split('$L__BB3_6:',1)[0]+'\n$L__BB3_6:\n'
    # The entire original resource predicate, optional-depth branch and history
    # sampling remain intact. Only the supplied descriptors/coordinates differ.
    setup=r'''.version 8.0
.target sm_86
.address_size 64
.visible .entry reset_pre_probe(.param .align 8 .b8 cfg[80]) {
.reg .pred %p<271>;
.reg .b16 %rs<501>;
.reg .b32 %r<5570>;
.reg .b64 %rd<407>;
.reg .b64 %out,%ctex,%coord,%addr,%off;
.reg .u32 %i,%n,%t,%b,%nt;
.reg .f32 %u,%v,%cr,%cg,%cb,%a,%hr,%hg,%hb;
ld.param.u64 %out,[cfg]; ld.param.u64 %ctex,[cfg+8];
ld.param.u64 %rd2,[cfg+16]; ld.param.u64 %rd3,[cfg+24];
mov.u64 %rd4,0;
ld.param.u64 %coord,[cfg+32]; ld.param.u32 %n,[cfg+40];
mov.u32 %b,%ctaid.x; mov.u32 %nt,%ntid.x; mov.u32 %t,%tid.x;
mad.lo.u32 %i,%b,%nt,%t; setp.ge.u32 %p0,%i,%n; @%p0 bra done;
mul.wide.u32 %off,%i,8; add.u64 %addr,%coord,%off;
ld.global.v2.f32 {%r51,%r59},[%addr];
ld.param.f32 %r8,[cfg+44]; ld.param.f32 %r9,[cfg+48];
mov.f32 %r25,%r8; mov.f32 %r26,%r9;
mov.f32 %r37,%r8; mov.f32 %r38,%r9;
rcp.approx.ftz.f32 %r29,%r8; rcp.approx.ftz.f32 %r30,%r9;
mov.f32 %r31,%r29; mov.f32 %r32,%r30;
mov.f32 %r35,%r29; mov.f32 %r36,%r30;
mov.f32 %r41,%r29; mov.f32 %r42,%r30;
mov.f32 %r27,0.0; mov.f32 %r28,0.0;
mov.f32 %r39,0.0; mov.f32 %r40,0.0;
add.f32 %r33,%r8,-0.5; add.f32 %r34,%r9,-0.5;
tex.2d.v4.f32.f32 {%cr,%cg,%cb,%a},[%ctex,{%r51,%r59}];
cvt.rn.f16.f32 %rs29,0.5; cvt.rn.f16.f32 %rs31,0.125;
cvt.rn.f16.f32 %rs18,%cr; sub.f16 %rs19,%rs18,%rs29; mul.f16 %rs2,%rs19,%rs31;
cvt.rn.f16.f32 %rs20,%cg; sub.f16 %rs21,%rs20,%rs29; mul.f16 %rs3,%rs21,%rs31;
cvt.rn.f16.f32 %rs22,%cb; sub.f16 %rs23,%rs22,%rs29; mul.f16 %rs4,%rs23,%rs31;
'''
    tail=r'''
cvt.f32.f16 %cr,%rs2; cvt.f32.f16 %cg,%rs3; cvt.f32.f16 %cb,%rs4;
cvt.f32.f16 %hr,%rs495; cvt.f32.f16 %hg,%rs494; cvt.f32.f16 %hb,%rs496;
mul.wide.u32 %off,%i,24; add.u64 %addr,%out,%off;
st.global.v3.f32 [%addr],{%cr,%cg,%cb};
st.global.v3.f32 [%addr+12],{%hr,%hg,%hb};
done: ret;
}'''.replace('st.global.v3.f32 [%addr],{%cr,%cg,%cb};','st.global.f32 [%addr],%cr; st.global.f32 [%addr+4],%cg; st.global.f32 [%addr+8],%cb;').replace('st.global.v3.f32 [%addr+12],{%hr,%hg,%hb};','st.global.f32 [%addr+12],%hr; st.global.f32 [%addr+16],%hg; st.global.f32 [%addr+20],%hb;')
    return (setup+body+tail).encode()


def pre_reference(color,history,motion):
    import klab
    klab.ctx()
    h,w=color.shape[:2]
    y,x=np.mgrid[:h,:w]
    coords=np.stack([(x+.5)/w,(y+.5)/h],-1).astype(np.float32).reshape(-1,2)
    out=torch.empty(h*w,6,device='cuda'); tc=torch.from_numpy(coords).cuda()
    module,fn=ctypes.c_void_p(),ctypes.c_void_p()
    ptx=pre_probe_ptx(); textures=[];keep=[]
    klab.ck(klab.cu.cuModuleLoadData(ctypes.byref(module),ptx))
    try:
        klab.ck(klab.cu.cuModuleGetFunction(ctypes.byref(fn),module,b'reset_pre_probe'))
        handles=[]
        for image,linear in ((color,False),(history,True),(motion,False)):
            handle=0
            if image is not None:
                handle,buffer=klab.texture(image,linear=linear);textures.append(handle);keep.append(buffer)
            handles.append(handle)
        cfg=bytearray(80)
        struct.pack_into('<5QI2f',cfg,0,out.data_ptr(),*handles,tc.data_ptr(),h*w,w,h)
        klab.launch(fn,((h*w+127)//128,1,1),(128,1,1),0,bytes(cfg))
        return out.cpu().numpy().reshape(h,w,6),sha(ptx)
    finally:
        for handle in textures:klab.ck(klab.cu.cuTexObjectDestroy(ctypes.c_uint64(handle)))
        klab.ck(klab.cu.cuModuleUnload(module))


def post_reference(color,history,motion):
    """Full retained post cubin, zero neural head and explicit blend=0.75."""
    import klab
    klab.ctx()
    h,w=color.shape[:2];gh,gw=N.dims(h,w)['grid']
    row=klab.trace_row(154,str(ROOT/'research/tapspost/nr-trace.tsv'))
    params=bytearray(row['params']);textures=[];keep=[];surface=klab.Surface(w,h)
    surface.fill(float('nan'))
    try:
        for offset,image,linear in ((56,color,False),(88,history,True),(96,motion,False)):
            handle=0
            if image is not None:
                handle,buffer=klab.texture(image,linear=linear);textures.append(handle);keep.append(buffer)
            struct.pack_into('<Q',params,offset,handle)
        struct.pack_into('<Q',params,16,surface.handle)
        struct.pack_into('<ii',params,32,gh,gw)
        struct.pack_into('<6f',params,64,0,0,w,h,1/w,1/h)
        struct.pack_into('<I6f',params,112,1,0,0,w,h,1/w,1/h)
        struct.pack_into('<8f',params,140,0,0,w,h,1/w,1/h,1/w,1/h)
        struct.pack_into('<ii',params,172,w,h)
        xin=klab.gpu_bytes(b'',32*(gh//2)*(gw//2));skip=klab.gpu_bytes(b'',32*gh*gw)
        weights=klab.gpu_bytes(bytes(21808),512);blend=klab.gpu_bytes(np.float16(.75).tobytes(),510)
        counter=klab.gpu_bytes(b'',1<<20)
        q=lambda offset:struct.unpack_from('<Q',params,offset)[0]
        mapping={q(0):xin.data_ptr(),q(8):skip.data_ptr(),q(24):weights.data_ptr(),q(104):blend.data_ptr()}
        for offset in range(0,len(params)//8*8,8):
            value=q(offset)
            if 0x1BA00000<=value<0x1BE00000 and value not in mapping:mapping[value]=counter.data_ptr()+(value&0xfffff)
        klab.launch(klab.function('01',row['kernel']),(gw//8+1,gh//8+1,1),row['block'],row['smem'],klab.patch_ptrs(bytes(params),mapping))
        return surface.read()[...,:3]
    finally:
        for handle in textures:klab.ck(klab.cu.cuTexObjectDestroy(ctypes.c_uint64(handle)))
        klab.ck(klab.cu.cuSurfObjectDestroy(ctypes.c_uint64(surface.handle)))
        klab.ck(klab.cu.cuArrayDestroy(surface.arr))


def kernel_checks(report):
    rng=np.random.default_rng(2026102401)
    for h,w,dx,dy in ((37,53,3.25,-1.5),(63,95,-2.375,1.625),(360,640,0,0)):
        color=color_pattern(w,h)
        history=rng.random((h,w,3),dtype=np.float32)
        motion=np.empty((h,w,2),np.float32);motion[...,0],motion[...,1]=dx,dy
        for mode in ('both_missing','history_missing','motion_missing','both_present'):
            hist=None if mode in ('both_missing','history_missing') else history
            mv=None if mode in ('both_missing','motion_missing') else motion
            disabled=hist is None or mv is None
            tc,th,tm=[None if value is None else torch.from_numpy(value).cuda() for value in (color,hist,mv)]
            pre,post,core,n=core_parts(h,w,'cuda')
            pref,ptx_sha=pre_reference(color,hist,mv)
            gh,gw=N.dims(h,w)['grid']
            candidate=pre.inputs(tc,th,tm,0).view(gh,gw,15)[:h,:w,:6].cpu().numpy()
            pm=metrics(candidate,pref)
            pre_ok=(np.array_equal(pref[...,:3],pref[...,3:]) and np.array_equal(candidate[...,:3],candidate[...,3:])) if disabled else pm['correlation']>=.98 and pm['nrmse']<=.25
            ref=post_reference(color,hist,mv)
            output=post(None,None,tc,th,tm).cpu().numpy()
            finished=core.finish(n,tc,th,tm).cpu().numpy()
            delta=np.abs(output.astype(np.float64)-ref)*255
            post_ok=bool(delta.max()<=.005 and delta.mean()<=.0005)
            row=dict(name=f'{w}x{h}/{mode}',motion=[dx,dy],resources=dict(history=hist is not None,motion=mv is not None),
                inputs_sha256=dict(color=array_sha(color),history=None if hist is None else array_sha(hist),motion=None if mv is None else array_sha(mv)),
                pre=dict(metrics=pm,probe_ptx_sha256=ptx_sha,output_sha256=array_sha(candidate),reference_sha256=array_sha(pref),
                         criterion='color/history bit-exact inside each implementation' if disabled else 'existing intermediate corr>=.98, NRMSE<=.25',passed=bool(pre_ok)),
                post=dict(mae_255=float(delta.mean()),max_255=float(delta.max()),output_sha256=array_sha(output),reference_sha256=array_sha(ref),passed=post_ok),
                core_finish_bit_exact=bool(np.array_equal(output,finished)))
            row['passed']=bool(pre_ok and post_ok and row['core_finish_bit_exact'])
            report['kernel'].append(row)
            print(f"kernel {row['name']}: pre={pre_ok} post={post_ok} max={delta.max():.6f}/255",flush=True)


def cases():
    rows=[]
    for name,w,h,dx,dy in (('zero',640,360,0,0),('fraction',640,360,3.25,-1.5),('negative',640,360,-2.375,1.625),('odd',641,361,-2.375,1.625)):
        for frames in (1,2):
            rows.append(dict(id=f'{name}_f{frames-1}',args=['--nr-only','--input',f'{w}x{h}','--output',f'{w}x{h}','--frames',str(frames),
                '--scene-variant','0','--motion-x',str(dx),'--motion-y',str(dy)]))
    return rows


def network_checks(report,capture):
    args=SimpleNamespace(out=capture,suite='batch4-reset',kind='port-sm86',harness=ROOT/'harness/batch3/nr-lab.exe',
        bridge=ROOT/'harness/nvngx.dll',dll=ROOT/'sm86_port/nvngx_dlssnr.dll',support=ROOT/'dlss5',timeout=90)
    with patch.object(REF,'suite_cases',lambda suite:cases()):
        if not capture.exists() and REF.capture(args):raise ValueError('Reset capture failed')
        manifest=REF.load_capture(capture)
    if manifest['runtime_sha256']['nvngx_dlssnr.dll']!=REF.digest(args.dll):raise ValueError('Capture must match current local DLL')
    report['capture']=dict(path=str(capture),manifest_sha256=REF.digest(capture/'manifest.json'),runtime_sha256=manifest['runtime_sha256'])
    model=DLSS5(); original_pre=N.PreBlock.inputs
    def legacy_pre(self,color,hist,mv,frame,ctrl=None):
        return original_pre(self,color,color if hist is None else hist,mv,frame,ctrl)
    for mode in ('baseline','candidate'):
        with patch.object(N.PreBlock,'inputs',legacy_pre if mode=='baseline' else original_pre):
            for group in ('zero','fraction','negative','odd'):
                rows=[row for row in manifest['cases'] if row['id'].startswith(group+'_')]
                fields,_=REF.case_contract(rows[0]);w,h=fields['input']['width'],fields['input']['height']
                dx,dy=fields['motionVector'];mv=torch.empty(h,w,2,device='cuda');mv[...,0],mv[...,1]=dx,dy
                hist=None
                for frame,row in enumerate(rows):
                    out=model(color_pattern(w,h),hist=hist,mv=mv,frame=frame)
                    path=capture/row['id']/next(iter(row['outputs']))
                    ref=REF.read_ppm(path).astype(np.float32)/255
                    m=metrics(out.cpu().numpy(),ref,quantize=True)
                    report['network'].append(dict(name=row['id'],mode=mode,metrics=m,output_sha256=array_sha(out),reference_sha256=REF.digest(path),
                        gating=mode=='candidate',passed=m['correlation']>=.999 and m['mae']<=3.0))
                    print(f"{mode} {row['id']}: MAE={m['mae']:.6f}/255 corr={m['correlation']:.8f}",flush=True)
                    hist=out
    # Hold the same History fixed: changes caused by a different previous reset
    # output must not be mistaken for a change to the normal-history operation.
    h,w=37,53;tc=torch.from_numpy(color_pattern(w,h)).cuda()
    tm=torch.empty(h,w,2,device='cuda');tm[...,0],tm[...,1]=3.25,-1.5
    with patch.object(N.PreBlock,'inputs',legacy_pre):
        before=model(tc,hist=1-tc,mv=tm,frame=1)
    current=model(tc,hist=1-tc,mv=tm,frame=1)
    report['history_control']=dict(name='same_fixed_history_and_fractional_mv',size_wh=[w,h],motion=[3.25,-1.5],
        input_sha256=array_sha(tc),history_sha256=array_sha(1-tc),motion_sha256=array_sha(tm),
        baseline_sha256=array_sha(before),output_sha256=array_sha(current),passed=bool(torch.equal(before,current)))
    h,w=37,53;color=torch.from_numpy(color_pattern(w,h)).cuda()
    mv=torch.empty(h,w,2,device='cuda');mv[...,0],mv[...,1]=-2.375,1.625
    eager=model(color,hist=None,mv=mv,frame=0)
    zero=model(color,hist=None,mv=None,frame=0)
    run=model.graph(h,w,history=False)
    output=run(color,mv=mv).clone()
    try:run(color,hist=color)
    except ValueError:contradiction=True
    else:contradiction=False
    report['graph'].append(dict(name='reset',eager_sha256=array_sha(eager),graph_sha256=array_sha(output),
        nonzero_vs_default_bit_exact=bool(torch.equal(eager,zero)),eager_vs_graph_bit_exact=bool(torch.equal(eager,output)),
        contradictory_history_rejected=contradiction,passed=bool(torch.equal(eager,zero) and torch.equal(eager,output) and contradiction)))
    del run;gc.collect();torch.cuda.empty_cache()
    eager=model(color,hist=color,mv=None,frame=1)
    run=model.graph(h,w,history=True)
    output=run(color,hist=color,mv=None,frame=1).clone()
    try:run(color)
    except ValueError:contradiction=True
    else:contradiction=False
    report['graph'].append(dict(name='history_default_zero_mv',eager_sha256=array_sha(eager),graph_sha256=array_sha(output),
        contradictory_history_rejected=contradiction,passed=bool(torch.equal(eager,output) and contradiction)))
    del run;gc.collect();torch.cuda.empty_cache()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,default=ROOT/'research/validation/batch4_reset.json')
    p.add_argument('--capture',type=Path,default=ROOT/'research/batch4/reset_capture')
    p.add_argument('--cpu-only',action='store_true')
    args,unknown=p.parse_known_args(argv)
    try:safe_report(args.report,args.capture)
    except ValueError as error:p.error(str(error))
    report=dict(passed=False,errors=[],cpu=[],kernel=[],network=[],graph=[],gpu_used=not args.cpu_only,
        execution_mode='cpu_only' if args.cpu_only else 'full',suite_complete=False,requested_checks_complete=False,
        scope='RTX3060 f16 port reset regression, not native sm120 equivalence',
        acceptance_scope='CPU + kernel + candidate network + Graph + fixed-History control; legacy baseline failures are preserved observations, not candidate acceptance',
        thresholds=dict(post_max_255=.005,post_mae_255=.0005,network_min_correlation=.999,network_max_mae_255=3.0,pre_min_correlation=.98,pre_max_nrmse=.25))
    report['limitations']=['Pre probe retains the original resource predicate and history slice, but is not the complete pre neural kernel',
        'Full post uses zero neural weights and blend=.75 to isolate resource-dependent temporal composition',
        'Network baseline reproduces the prior Color-as-History argument substitution using the current sampler arithmetic',
        'CUDA Graph history presence is fixed at capture; use a separate reset Graph to reset state',
        'Public MV=None means zero motion; raw Pre/Post/Core MV=None means resource absent']
    def save():
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    try:
        save()
        if unknown:
            report['errors'].append('Unrecognized arguments: '+' '.join(unknown))
            p.error(report['errors'][-1])
        names=['torch/check_reset.py','torch/test_reset.py','torch/dlss5/net.py','torch/dlss5/export.py','torch/dlss5/ops.py','torch/dlss5/ptx_math.py',
               'fatbins/fatbin_01.1.sm_120.ptx','sm86_port/fatbin_01.sm86.cubin','sm86_port/nvngx_dlssnr.dll',
               'harness/reference.py','research/klab.py','research/tapspost/nr-trace.tsv','WEIGHTS_HT.bin','weights_map.json',
               'harness/batch3/nr-lab.exe','harness/nvngx.dll','dlss5/nvngx_dlss.dll','dlss5/nvngx_dlssg.dll']
        report['source_sha256']={name:REF.digest(ROOT/name) for name in names}
        baseline_cpu(report)
        if not args.cpu_only:
            report['device']=torch.cuda.get_device_name(0)
            kernel_checks(report)
            network_checks(report,args.capture)
        report['coverage']={name:dict(compared=len(report[name]),passed=sum(row['passed'] for row in report[name])) for name in ('cpu','kernel','graph')}
        report['coverage']['network']={mode:dict(compared=sum(row['mode']==mode for row in report['network']),
            passed=sum(row['mode']==mode and row['passed'] for row in report['network'])) for mode in ('baseline','candidate')}
        checks=report['cpu']+report['kernel']+[row for row in report['network'] if row['gating']]+report['graph']+([report['history_control']] if 'history_control' in report else [])
        full=(len(report['cpu'])==3 and len(report['kernel'])==12 and len(report['network'])==16 and len(report['graph'])==2 and 'history_control' in report)
        report['suite_complete']=full
        report['requested_checks_complete']=len(report['cpu'])==3 if args.cpu_only else full
        report['passed']=report['requested_checks_complete'] and all(row['passed'] for row in checks)
    except Exception as error:
        report['errors'].append(f'{type(error).__name__}: {error}')
    finally:save()
    return 0 if report['passed'] else 1


if __name__=='__main__':
    torch.set_num_threads(4)
    raise SystemExit(main())
