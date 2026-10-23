"""Independent TEX and retained pre/post Catmull-Rom sampling comparisons.

References execute authored TEX instructions or the original sampling instruction
slice, extracted at runtime from the identified local PTX. No helper output is a
reference. The measured sampler contract is specific to the recorded RTX 3060.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import klab  # noqa: E402
from dlss5 import ops  # noqa: E402


def sha(data):
    return hashlib.sha256(data).hexdigest()


def guard_report(path):
    """Reject direct paths, directory aliases and hardlinks to reference inputs."""
    path=Path(path).resolve()
    if path.suffix.lower()!='.json' or path.is_dir():
        raise ValueError('Report must be a JSON file')
    folders=[ROOT/'torch',ROOT/'fatbins',ROOT/'sm86_port',ROOT/'dlss5',
             ROOT/'research/tapspost',ROOT/'harness/captures']
    protected=[ROOT/'WEIGHTS_HT.bin',ROOT/'weights_map.json',ROOT/'research/klab.py',
               ROOT/'research/batch4_sampling/post_arithmetic_static.json']
    protected += list((ROOT/'research').glob('out_*.ppm'))
    protected += list((ROOT/'research/validation').glob('batch3_sampling*.json'))
    for folder in folders:
        if folder.is_dir():
            protected += [p for p in folder.rglob('*') if p.is_file()]
    targets=[path,path.with_suffix(path.suffix+'.tmp')]
    for target in targets:
        if any(target.is_relative_to(folder.resolve()) for folder in folders):
            raise ValueError('Report must be outside source/runtime/reference directories')
        for source in protected:
            source=source.resolve()
            if target==source or target.exists() and source.exists() and target.samefile(source):
                raise ValueError('Report aliases a protected source/runtime/reference input')
    return path


def post_sass_evidence():
    """CPU-only instruction identities; proprietary disassembly is not saved."""
    result={'scope':'Static original sm120 and current sm86 complete pre/post scalar instructions; not GPU execution parity',
            'original_dll_sha256':sha((ROOT/'dlss5/nvngx_dlssnr.dll').read_bytes()),'kernels':[]}
    for role,relative,offsets in (('pre','sm86_port/fatbin_01.1.sm_120.cubin',(0x10d0,0x10e0,0x1120,0x1150)),
                                  ('pre','sm86_port/fatbin_01.sm86.cubin',(0x0f80,0x0f90,0x0fd0,0x1000)),
                                  ('post','sm86_port/fatbin_01.1.sm_120.cubin',(0xa290,0xa2a0,0xa2e0,0xa310)),
                                  ('post','sm86_port/fatbin_01.sm86.cubin',(0x22670,0x22680,0x226c0,0x226f0))):
        name='cc_tinlayout_fused_'+('pre_block_swin_1h_32_1_ds_fp8' if role=='pre' else 'post_block_swin_1h_32_fp8')
        path=ROOT/relative
        blob=path.read_bytes()
        shoff=struct.unpack_from('<Q',blob,40)[0]
        entsize,count,strings=struct.unpack_from('<HHH',blob,58)
        headers=[struct.unpack_from('<IIQQQQIIQQ',blob,shoff+i*entsize) for i in range(count)]
        string_header=headers[strings]
        table=blob[string_header[4]:string_header[4]+string_header[5]]
        section=next(h for h in headers if table[h[0]:].split(b'\0',1)[0].decode()=='.text.'+name)
        text=subprocess.run([str(ROOT/'cuda_tools/bin/cuobjdump.exe'),'--dump-sass','--function',name,str(path)],
                            check=True,capture_output=True,text=True).stdout
        decoded={int(match[0],16):(match[1],bytes.fromhex(match[2])[::-1]+bytes.fromhex(match[3])[::-1]) for match in
                 re.findall(r'/\*([0-9a-f]+)\*/\s+(?:@\S+\s+)?([A-Z0-9_.]+)\s+[^;]+;\s*/\* 0x([0-9a-f]{16}) \*/\s*\n\s*/\* 0x([0-9a-f]{16}) \*/',text)}
        rows=[]
        for offset,instruction_role in zip(offsets,('center_x=fma(u,width,-.5)','center_y=fma(v,height,-.5)',
                                       'fraction_x=fma(u,width,-center_x)','fraction_y=fma(v,height,-center_y)')):
            opcode,encoding=decoded[offset]
            raw=blob[section[4]+offset:section[4]+offset+16]
            if opcode!='FFMA.FTZ' or raw!=encoding:
                raise ValueError('Original pre/post instruction identity changed')
            rows.append(dict(section_offset=hex(offset),file_offset=hex(section[4]+offset),opcode=opcode,
                             instruction_sha256=sha(raw),role=instruction_role))
        result['kernels'].append(dict(kernel_role=role,path=relative,sha256=sha(blob),section='.text.'+name,instructions=rows))
    result['cuobjdump_sha256']=sha((ROOT/'cuda_tools/bin/cuobjdump.exe').read_bytes())
    return result


def legacy_bilinear(img, u, v):
    h, w = img.shape[:2]
    xx, yy = (u*w-.5).clamp(0,w-1), (v*h-.5).clamp(0,h-1)
    x0, y0 = xx.floor().long(), yy.floor().long()
    x1, y1 = (x0+1).clamp(max=w-1), (y0+1).clamp(max=h-1)
    fx, fy = (xx-x0)[...,None], (yy-y0)[...,None]
    return (img[y0,x0]*(1-fx)*(1-fy)+img[y0,x1]*fx*(1-fy)
            +img[y1,x0]*(1-fx)*fy+img[y1,x1]*fx*fy)


def legacy_catmull(img, u, v, *, arithmetic="fused"):
    h,w=img.shape[:2]
    px,py=u*w,v*h
    cx,cy=(px-.5).floor()+.5,(py-.5).floor()+.5
    fx,fy=(px-cx).clamp(0,1),(py-cy).clamp(0,1)
    w0x,w0y=fx*(-.5+fx*(1-.5*fx)),fy*(-.5+fy*(1-.5*fy))
    w1x,w1y=1+fx*fx*(-2.5+1.5*fx),1+fy*fy*(-2.5+1.5*fy)
    w2x,w2y=fx*(.5+fx*(2-1.5*fx)),fy*(.5+fy*(2-1.5*fy))
    w3x,w3y=fx*fx*(-.5+.5*fx),fy*fy*(-.5+.5*fy)
    wx,wy=w1x+w2x,w1y+w2y
    tx=(cx+w2x/wx).clamp(.5,w-.5)/w
    ty=(cy+w2y/wy).clamp(.5,h-.5)/h
    x0,y0=(cx-1).clamp(.5,w-.5)/w,(cy-1).clamp(.5,h-.5)/h
    x3,y3=(cx+2).clamp(.5,w-.5)/w,(cy+2).clamp(.5,h-.5)/h
    a,b,c,d,e=[x[...,None] for x in (wx*w0y,w0x*wy,wx*wy,w3x*wy,wx*w3y)]
    return (legacy_bilinear(img,tx,y0)*a+legacy_bilinear(img,x0,ty)*b
            +legacy_bilinear(img,tx,ty)*c+legacy_bilinear(img,x3,ty)*d
            +legacy_bilinear(img,tx,y3)*e)/(a+b+c+d+e)


def probe_ptx(kind):
    """Retain the original arithmetic/TEX instructions; only supply their inputs."""
    text = (ROOT / "fatbins/fatbin_01.1.sm_120.ptx").read_text()
    if kind in ("pre", "pre_native"):
        function = text.split(".visible .entry cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8(",1)[1].split(".visible .entry",1)[0]
        body = function.split("add.ftz.f32 %r302, %r299, 0fBF000000;",1)[1].split("mul.ftz.f32 %r287, %r412, %r427;",1)[0]
        body = "add.ftz.f32 %r302, %r299, 0fBF000000;" + body + "mul.ftz.f32 %r287, %r412, %r427;"
        setup = """
        mul.f32 %r299,%f0,%f2; mul.f32 %r301,%f1,%f3;
        add.f32 %r33,%f2,-0.5; add.f32 %r34,%f3,-0.5;
        rcp.approx.ftz.f32 %r35,%f2; rcp.approx.ftz.f32 %r36,%f3;
        mov.b32 %r37,%f2; mov.b32 %r38,%f3;
        mov.f32 %r39,0.0; mov.f32 %r40,0.0;
        ld.param.f32 %r41,[cfg+132]; ld.param.f32 %r42,[cfg+136];
        mov.u64 %rd2,%tex;
        """
        result = "%r285,%r286,%r287,%f4"
        if kind == "pre_native":
            # The complete original pre multiplies with FTZ. The historical
            # probe omitted FTZ, preventing the compiler's actual FFMA contraction.
            setup = setup.replace("mul.f32 %r299", "mul.ftz.f32 %r299").replace("mul.f32 %r301", "mul.ftz.f32 %r301")
    elif kind == "post":
        function = text.split(".visible .entry cc_tinlayout_fused_post_block_swin_1h_32_fp8(",1)[1].split(".visible .entry",1)[0]
        body = function.split("mul.ftz.f32 %r4735, %r4733, %r32;",1)[1].split("mul.ftz.f32 %r4870, %r4869, %r4852;",1)[0]
        body = "mul.ftz.f32 %r4735, %r4733, %r32;" + body + "mul.ftz.f32 %r4870, %r4869, %r4852;"
        setup = """
        mov.b32 %r4733,%f0; mov.b32 %r4734,%f1;
        mov.b32 %r32,%f2; mov.b32 %r34,%f3;
        add.f32 %r35,%f2,-0.5; add.f32 %r36,%f3,-0.5;
        rcp.approx.ftz.f32 %r37,%f2; rcp.approx.ftz.f32 %r38,%f3;
        ld.param.f32 %r4716,[cfg+136]; ld.param.u64 %rd397,[cfg+112];
        mov.u64 %rd9,%tex; mov.u64 %rd11,cfg;
        """
        result = "%r4858,%r4864,%r4870,%f4"
    elif kind == "linear":
        body = "tex.2d.v4.f32.f32 {%f5,%f6,%f7,%f4},[%tex,{%f0,%f1}];"
        setup, result = "", "%f5,%f6,%f7,%f4"
    else:
        raise ValueError(kind)
    if len(re.findall(r"tex\.2d",body)) != (1 if kind == "linear" else 5):
        raise ValueError("Original sampling PTX slice changed")
    header = """.version 8.0
    .target sm_86
    .address_size 64
    .visible .entry sampling_probe(.param .align 8 .b8 cfg[224]) {
    .reg .pred %p;
    .reg .b32 %r<5000>;
    .reg .b64 %rd<410>;
    .reg .b64 %out,%coord,%tex,%offset,%address;
    .reg .u32 %i,%ct,%nt,%threadid,%n;
    .reg .f32 %f<8>;
    ld.param.u64 %out,[cfg+184]; ld.param.u64 %coord,[cfg+192];
    ld.param.u64 %tex,[cfg+88]; ld.param.u32 %n,[cfg+200];
    ld.param.f32 %f2,[cfg+204]; ld.param.f32 %f3,[cfg+208];
    mov.u32 %ct,%ctaid.x; mov.u32 %nt,%ntid.x; mov.u32 %threadid,%tid.x;
    mad.lo.u32 %i,%ct,%nt,%threadid;
    setp.ge.u32 %p,%i,%n; @%p bra done;
    mul.wide.u32 %offset,%i,8; add.u64 %address,%coord,%offset;
    ld.global.v2.f32 {%f0,%f1},[%address]; mov.f32 %f4,0.0;
    """
    tail = f"""
    mul.wide.u32 %offset,%i,16; add.u64 %address,%out,%offset;
    st.global.v4.f32 [%address],{{{result}}};
    done: ret;
    }}
    """
    return (header+setup+body+tail).encode()


def run_probe(img, coords, kind):
    klab.ctx()
    ptx = probe_ptx(kind)
    module,fn=ctypes.c_void_p(),ctypes.c_void_p()
    klab.ck(klab.cu.cuModuleLoadData(ctypes.byref(module),ptx))
    texture = None
    try:
        klab.ck(klab.cu.cuModuleGetFunction(ctypes.byref(fn),module,b"sampling_probe"))
        texture,keep=klab.texture(img,linear=True)
        tc=torch.from_numpy(np.asarray(coords,np.float32)).cuda()
        out=torch.empty(len(coords),4,device="cuda")
        h,w=img.shape[:2]
        cfg=bytearray(224)
        struct.pack_into("<Q",cfg,88,texture)
        struct.pack_into("<I6f",cfg,112,1,0,0,w,h,1/w,1/h)
        struct.pack_into("<QQIff",cfg,184,out.data_ptr(),tc.data_ptr(),len(coords),w,h)
        klab.launch(fn,((len(coords)+127)//128,1,1),(128,1,1),0,bytes(cfg))
        return out[:,:3].cpu().numpy(), sha(ptx)
    finally:
        if texture is not None:
            klab.ck(klab.cu.cuTexObjectDestroy(ctypes.c_uint64(texture)))
        klab.ck(klab.cu.cuModuleUnload(module))


def error_metrics(output,reference):
    a,b=np.asarray(output,np.float64),np.asarray(reference,np.float64)
    if a.shape!=b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Invalid sampling output/reference")
    d=np.abs(a-b)*255
    return dict(mae_255=float(d.mean()),max_255=float(d.max()),p99_255=float(np.percentile(d,99)),
                output_sha256=sha(np.asarray(output,np.float32).tobytes()),reference_sha256=sha(np.asarray(reference,np.float32).tobytes()))


def isolated_checks(report):
    rng=np.random.default_rng(2026102303)
    for h,w in ((2,2),(19,27),(37,53),(360,640)):
        img=rng.random((h,w,3),dtype=np.float32)
        coords=rng.uniform(-.1,1.1,(4096,2)).astype(np.float32)
        y,x=np.mgrid[:min(h,37),:min(w,53)]
        centers=np.stack([(x+.5)/w,(y+.5)/h],-1).astype(np.float32).reshape(-1,2)
        coords=np.concatenate([coords,centers],0)
        ti,tc=torch.from_numpy(img).cuda(),torch.from_numpy(coords).cuda()
        for kind in ("linear","pre","pre_native","post"):
            ref,ptx_sha=run_probe(img,coords,kind)
            fn=ops.bilinear if kind=="linear" else ops.catmull_rom5
            old=legacy_bilinear if kind=="linear" else legacy_catmull
            current=(fn(ti,tc[:,0],tc[:,1]) if kind=='linear' else
                     fn(ti,tc[:,0],tc[:,1],arithmetic='split' if kind=='pre' else 'fused')).cpu().numpy()
            previous=old(ti,tc[:,0],tc[:,1]).cpu().numpy()
            metrics=error_metrics(current,ref)
            row=dict(name=f"{kind}_{w}x{h}",samples=len(coords),kind=kind,metrics=metrics,
                     baseline=error_metrics(previous,ref),probe_ptx_sha256=ptx_sha,
                     input_sha256=sha(img.tobytes()),coordinates_sha256=sha(coords.tobytes()))
            if kind=='post':
                row['before_post_arithmetic']=error_metrics(ops.catmull_rom5(ti,tc[:,0],tc[:,1],arithmetic='split').cpu().numpy(),ref)
            row['reference_scope']=('Historical artificial pre setup without FTZ multiply; split comparison only, not the complete pre kernel'
                                    if kind=='pre' else 'Sampling slice with original FTZ coordinate context' if kind=='pre_native'
                                    else 'Original post sampling slice' if kind=='post' else 'Authored raw TEX instruction')
            row['arithmetic']='split' if kind=='pre' else 'fused' if kind!='linear' else 'TEX'
            row['passed']=metrics['max_255']<=report['thresholds']['isolated_max_255'] and metrics['mae_255']<=report['thresholds']['isolated_mae_255']
            report['isolated'].append(row)
            print(f"{kind} {w}x{h}: old mean/max {row['baseline']['mae_255']:.6f}/{row['baseline']['max_255']:.6f}; new {metrics['mae_255']:.6f}/{metrics['max_255']:.6f}",flush=True)


def full_post_reference(history,motion):
    """Execute the full original cubin with a zero network head and gate=1/2."""
    h,w=history.shape[:2]
    from dlss5.net import dims
    gh,gw=dims(h,w)['grid']
    row=klab.trace_row(154,str(ROOT/'research/tapspost/nr-trace.tsv'))
    params=bytearray(row['params'])
    textures=[];keep=[];surface=klab.Surface(w,h)
    surface.fill(float('nan'))
    try:
        for offset,image,linear in ((56,np.zeros_like(history),True),(88,history,True),(96,motion,False)):
            handle,buffer=klab.texture(image,linear=linear)
            textures.append(handle);keep.append(buffer)
            struct.pack_into('<Q',params,offset,handle)
        struct.pack_into('<Q',params,16,surface.handle)
        struct.pack_into('<ii',params,32,gh,gw)
        struct.pack_into('<6f',params,64,0,0,w,h,1/w,1/h)
        struct.pack_into('<I6f',params,112,1,0,0,w,h,1/w,1/h)
        struct.pack_into('<8f',params,140,0,0,w,h,1/w,1/h,1/w,1/h)
        struct.pack_into('<ii',params,172,w,h)
        xin=klab.gpu_bytes(b'',32*(gh//2)*(gw//2))
        skip=klab.gpu_bytes(b'',32*gh*gw)
        weights=klab.gpu_bytes(bytes(21808),512)
        blend=klab.gpu_bytes(np.float16(1).tobytes(),510)
        counter=klab.gpu_bytes(b'',1<<20)
        q=lambda offset:struct.unpack_from('<Q',params,offset)[0]
        mapping={q(0):xin.data_ptr(),q(8):skip.data_ptr(),q(24):weights.data_ptr(),q(104):blend.data_ptr()}
        for offset in range(0,len(params)//8*8,8):
            value=q(offset)
            if 0x1BA00000<=value<0x1BE00000 and value not in mapping:
                mapping[value]=counter.data_ptr()+(value&0xFFFFF)
        klab.launch(klab.function('01',row['kernel']),(gw//8+1,gh//8+1,1),row['block'],row['smem'],
                    klab.patch_ptrs(bytes(params),mapping))
        return surface.read()[...,:3]*2
    finally:
        for handle in textures:
            klab.ck(klab.cu.cuTexObjectDestroy(ctypes.c_uint64(handle)))
        klab.ck(klab.cu.cuSurfObjectDestroy(ctypes.c_uint64(surface.handle)))
        klab.ck(klab.cu.cuArrayDestroy(surface.arr))


def full_kernel_checks(report):
    rng=np.random.default_rng(2026102304)
    hist=rng.random((360,640,3),dtype=np.float32)
    image=torch.from_numpy(hist).cuda()
    y,x=torch.meshgrid(torch.arange(360,device='cuda',dtype=torch.float32),torch.arange(640,device='cuda',dtype=torch.float32),indexing='ij')
    for name,mvx,mvy in (('zero',0,0),('integer',-4,0),('fractional',3.25,-1.5),('negative_fraction',-2.375,1.625),('outside',650,-370)):
        mv=np.empty((360,640,2),np.float32);mv[...,0]=mvx;mv[...,1]=mvy
        reference=full_post_reference(hist,mv)
        u,v=(x+.5+mvx)/640,(y+.5+mvy)/360
        previous=legacy_catmull(image,u,v).cpu().numpy()
        before_coordinates=ops.catmull_rom5(image,u,v,arithmetic='fused').cpu().numpy()
        uf,vf=ops.reproject_uv((x+.5)/640,(y+.5)/360,torch.from_numpy(mv).cuda(),640,360)
        current=ops.catmull_rom5(image,uf,vf,arithmetic='fused').cpu().numpy()
        m=error_metrics(current,reference)
        row=dict(name='full_post/'+name,metrics=m,baseline=error_metrics(previous,reference),
                 before_coordinate_fix=error_metrics(before_coordinates,reference),motion=[mvx,mvy],input_sha256=sha(hist.tobytes()),
                 motion_sha256=sha(mv.tobytes()))
        row['passed']=m['max_255']<=report['thresholds']['isolated_max_255'] and m['mae_255']<=report['thresholds']['isolated_mae_255']
        report['full_kernel'].append(row)
        print(f"{row['name']}: old {row['baseline']['mae_255']:.6f}/{row['baseline']['max_255']:.6f}; current {m['mae_255']:.6f}/{m['max_255']:.6f}",flush=True)
    for h,w in ((37,53),(63,95)):
        hist=rng.random((h,w,3),dtype=np.float32)
        mv=np.full((h,w,2),0,np.float32);mv[...,0]=-2.375;mv[...,1]=1.625
        ref=full_post_reference(hist,mv)
        y,x=torch.meshgrid(torch.arange(h,device='cuda',dtype=torch.float32),torch.arange(w,device='cuda',dtype=torch.float32),indexing='ij')
        u,v=ops.reproject_uv((x+.5)/w,(y+.5)/h,torch.from_numpy(mv).cuda(),w,h)
        current=ops.catmull_rom5(torch.from_numpy(hist).cuda(),u,v,arithmetic='fused').cpu().numpy()
        previous=legacy_catmull(torch.from_numpy(hist).cuda(),(x+.5-2.375)/w,(y+.5+1.625)/h).cpu().numpy()
        m=error_metrics(current,ref)
        row=dict(name=f'full_post/odd_{w}x{h}',metrics=m,baseline=error_metrics(previous,ref),motion=[-2.375,1.625],
                 input_sha256=sha(hist.tobytes()),motion_sha256=sha(mv.tobytes()))
        row['passed']=m['max_255']<=report['thresholds']['isolated_max_255'] and m['mae_255']<=report['thresholds']['isolated_mae_255']
        report['full_kernel'].append(row)
        print(f"{row['name']}: old {row['baseline']['mae_255']:.6f}/{row['baseline']['max_255']:.6f}; current {m['mae_255']:.6f}/{m['max_255']:.6f}",flush=True)
    # Spatially varying fractional fields exercise arbitrary coordinate boundaries
    # through the complete cubin, rather than only the re-JIT sampling slices.
    for h,w in ((37,53),(360,640)):
        hist=rng.random((h,w,3),dtype=np.float32)
        mv=rng.uniform(-3.75,3.75,(h,w,2)).astype(np.float32)
        ref=full_post_reference(hist,mv)
        y,x=torch.meshgrid(torch.arange(h,device='cuda',dtype=torch.float32),torch.arange(w,device='cuda',dtype=torch.float32),indexing='ij')
        image=torch.from_numpy(hist).cuda()
        u,v=ops.reproject_uv((x+.5)/w,(y+.5)/h,torch.from_numpy(mv).cuda(),w,h)
        current=ops.catmull_rom5(image,u,v,arithmetic='fused').cpu().numpy()
        previous=ops.catmull_rom5(image,u,v,arithmetic='split').cpu().numpy()
        m=error_metrics(current,ref)
        row=dict(name=f'full_post/spatial_{w}x{h}',metrics=m,before_post_arithmetic=error_metrics(previous,ref),
                 input_sha256=sha(hist.tobytes()),motion_sha256=sha(mv.tobytes()),
                 coordinates_sha256=sha(torch.stack((u,v),-1).cpu().numpy().tobytes()))
        row['passed']=m['max_255']<=report['thresholds']['isolated_max_255'] and m['mae_255']<=report['thresholds']['isolated_mae_255']
        report['full_kernel'].append(row)
        print(f"{row['name']}: split arithmetic {row['before_post_arithmetic']['mae_255']:.6f}/{row['before_post_arithmetic']['max_255']:.6f}; fused {m['mae_255']:.6f}/{m['max_255']:.6f}",flush=True)


def network_checks(report):
    from acceptance import metrics, ppm
    from check import color_pattern
    from dlss5 import DLSS5
    from dlss5 import net as net_module
    model=DLSS5()
    current_bilinear,current_catmull=net_module.bilinear,net_module.catmull_rom5
    current_uv=ops.reproject_uv
    current_post=net_module.PostBlock.__call__
    def legacy_post(block,x69,skip,color,hist,mv):
        h,w=color.shape[:2]
        n=block.net(x69,skip,*net_module.dims(h,w)['grid'])[:h,:w]
        cur=(color+8*block.scale*n[...,:3]).clamp(0,1)
        if hist is None:return cur
        gate=(torch.sigmoid(n[...,3:])*block.blend).clamp(0,1)
        y,x=torch.meshgrid(torch.arange(h,device=block.dev,dtype=torch.float32),torch.arange(w,device=block.dev,dtype=torch.float32),indexing='ij')
        previous=legacy_catmull(hist[...,:3],(x+.5+mv[...,0])/w,(y+.5+mv[...,1])/h)
        return cur+gate*(previous-cur)
    outputs={}
    try:
        for mode in ('baseline','candidate'):
            net_module.bilinear=legacy_bilinear if mode=='baseline' else current_bilinear
            net_module.catmull_rom5=legacy_catmull if mode=='baseline' else current_catmull
            ops.reproject_uv=(lambda u,v,mv,w,h:(u+mv[...,0]/w,v+mv[...,1]/h)) if mode=='baseline' else current_uv
            net_module.PostBlock.__call__=legacy_post if mode=='baseline' else current_post
            for case,mvx in (('static',0),('mvok',-4),('mv0',0)):
                hist=None
                mv=torch.zeros(360,640,2,device='cuda');mv[...,0]=mvx
                for frame in range(4):
                    color=color_pattern(offset_x=frame*4 if case!='static' else 0)
                    out=model(color,hist=hist,mv=None if hist is None else mv,frame=frame)
                    hist=out
                    if case!='static' and frame==0:
                        continue
                    path=ROOT/'research'/(f'out_f{frame}.ppm' if case=='static' else f'out_{case}_f{frame}.ppm')
                    ref=ppm(path,640,360)
                    image=out.cpu().numpy()
                    key=f'{case}/frame{frame}'
                    m=metrics(image,ref,quantize=True)
                    row=dict(name=key,mode=mode,metrics=m,output_sha256=sha(image.tobytes()),
                             reference_path=str(path.relative_to(ROOT)),reference_sha256=sha(path.read_bytes()))
                    row['passed']=m['correlation']>=report['thresholds']['min_correlation'] and m['mae']<=report['thresholds']['max_mae_255']
                    outputs[mode,key]=image
                    report['network'].append(row)
                    print(f"{mode} {key}: MAE={m['mae']:.6f}/255 corr={m['correlation']:.8f}",flush=True)
        pairs=[]
        for row in report['network']:
            if row['mode']!='candidate': continue
            previous=next(x for x in report['network'] if x['mode']=='baseline' and x['name']==row['name'])
            pairs.append(dict(name=row['name'],baseline_mae_255=previous['metrics']['mae'],candidate_mae_255=row['metrics']['mae'],
                              delta_mae_255=row['metrics']['mae']-previous['metrics']['mae'],
                              changed_values=int(np.count_nonzero(outputs['candidate',row['name']]!=outputs['baseline',row['name']]))))
        report['network_comparison']=dict(checks=pairs,mean_delta_mae_255=float(np.mean([x['delta_mae_255'] for x in pairs])))
        # CUDA Graph execution must agree with eager execution on the same inputs.
        net_module.bilinear,net_module.catmull_rom5=current_bilinear,current_catmull
        ops.reproject_uv=current_uv
        net_module.PostBlock.__call__=current_post
        color=torch.from_numpy(color_pattern()).cuda()
        hist=torch.from_numpy(outputs['candidate','static/frame0']).cuda()
        eager=model(color,hist=hist,frame=1)
        run=model.graph(360,640,history=True)
        graph=run(color,hist=hist,frame=1).clone()
        d=(eager-graph).abs()
        report['cuda_graph']=dict(max_error=float(d.max()),mean_error=float(d.mean()),passed=bool(torch.equal(eager,graph)),
                                  eager_sha256=sha(eager.cpu().numpy().tobytes()),graph_sha256=sha(graph.cpu().numpy().tobytes()))
    finally:
        net_module.bilinear,net_module.catmull_rom5=current_bilinear,current_catmull
        ops.reproject_uv=current_uv
        net_module.PostBlock.__call__=current_post


def main(argv=None):
    # Resolve and invalidate the safe destination before parsing the complete
    # CLI, so even an unknown option cannot leave an earlier PASS behind.
    preliminary=argparse.ArgumentParser(add_help=False)
    preliminary.add_argument('--report',type=Path,default=ROOT/'research/validation/batch4_sampling.json')
    early,_=preliminary.parse_known_args(argv)
    try:
        destination=guard_report(early.report)
    except (ValueError,OSError) as error:
        print(f'FAIL unsafe report: {error}',file=sys.stderr)
        return 2
    if not any(x in ('-h','--help') for x in (sys.argv[1:] if argv is None else argv)):
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text('{"passed":false,"complete":false,"errors":["CLI or suite has not completed"]}\n',encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,default=ROOT/'research/validation/batch4_sampling.json')
    parser.add_argument('--network-only',action='store_true')
    parser.add_argument('--isolated-only',action='store_true')
    args=parser.parse_args(argv)
    try:
        args.report=guard_report(args.report)
    except (ValueError,OSError) as error:
        print(f'FAIL unsafe report: {error}',file=sys.stderr)
        return 2
    report=dict(passed=False,complete=False,requested_suite_passed=False,errors=[],isolated=[],full_kernel=[],network=[],scope='RTX 3060 measured sampling; not official RTX 50 parity',
                thresholds=dict(isolated_max_255=.005,isolated_mae_255=.0005,min_correlation=.999,max_mae_255=3.0))
    report['sampler_contract']=dict(normalized_coordinate_fraction_bits=21,axis_weight_fraction_bits=8,
        cross_weight_fraction_bits=8,weight_ties='half upward',addressing='normalized clamp',filtering='linear',
        scope='Empirical RTX 3060 TEX model; coordinate/cross/tie details are not an API or cross-GPU guarantee')
    report['limitations']=['CUDA RCP uses a small authored PTX driver shim; CPU fallback is ordinary reciprocal without CUDA numerical parity',
                          'Catmull coefficient/accumulation rounding remains approximate; finite passing cases do not certify all coordinates/GPUs',
                          'PTX slices are re-JIT-compiled sampling instructions; full-kernel evidence is listed separately',
                          'Full post tests zero the neural weights and use blend=1, recovering history as 2*RGB output',
                          'Network comparisons are against the sm86 f16 port and are not official sm120 output parity']
    report['execution']=dict(isolated=not args.network_only,network=not args.isolated_only)
    report['network_baseline']='Legacy bilinear/Catmull sampler under the current Reset contract; not a replay of the complete batch3 network'
    report['catmull_arithmetic']=dict(split='historical artificial pre probe: separate pixel multiply/add rounding',fused='complete original pre/post pixel coordinate FFMA via float64 temporary then float32')
    def save():
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    try:
        save()
        report['sources']={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in
                           (Path(__file__),ROOT/'torch/dlss5/ops.py',ROOT/'torch/dlss5/net.py',ROOT/'fatbins/fatbin_01.1.sm_120.ptx',
                            ROOT/'sm86_port/fatbin_01.sm86.cubin',ROOT/'research/tapspost/nr-trace.tsv',
                            ROOT/'torch/test_sampling.py',ROOT/'dlss5/nvngx_dlssnr.dll',ROOT/'sm86_port/nvngx_dlssnr.dll',
                            ROOT/'research/klab.py',ROOT/'torch/dlss5/ptx_math.py')}
        evidence=ROOT/'research/batch4_sampling/post_arithmetic_static.json'
        if evidence.exists():
            report['post_machine_code_evidence']=dict(path=str(evidence.relative_to(ROOT)),sha256=sha(evidence.read_bytes()))
        baseline=ROOT/'research/validation/batch3_sampling_current.json'
        if baseline.exists() and baseline.resolve()!=args.report.resolve():
            report['baseline_report']=dict(path=str(baseline.relative_to(ROOT)),sha256=sha(baseline.read_bytes()))
        report['device']=torch.cuda.get_device_name(0)
        report['torch_version']=torch.__version__
        if args.network_only and args.isolated_only:
            raise ValueError('Select at most one partial suite')
        if not args.network_only:
            isolated_checks(report)
            full_kernel_checks(report)
        if not args.isolated_only:
            network_checks(report)
        checks=report['isolated']+report['full_kernel']+report['network']+([report['cuda_graph']] if 'cuda_graph' in report else [])
        report['coverage']={name:dict(compared=len(report[name]),passed=sum(x['passed'] for x in report[name]),
                                     failed=sum(not x['passed'] for x in report[name]))
                            for name in ('isolated','full_kernel','network')}
        requested_complete=((args.network_only or len(report['isolated'])==16 and len(report['full_kernel'])==9)
                            and (args.isolated_only or len(report['network'])==20 and 'cuda_graph' in report))
        report['complete']=not args.network_only and not args.isolated_only and requested_complete
        report['requested_suite_passed']=requested_complete and bool(checks) and all(x['passed'] for x in checks)
        report['passed']=report['complete'] and report['requested_suite_passed']
        if not report['complete']:
            report['scope']='Partial diagnostic sampling suite; not full acceptance'
    except Exception as exc:
        report['errors'].append(f'{type(exc).__name__}: {exc}')
    finally:
        save()
    return 0 if report['requested_suite_passed'] and not report['errors'] else 1


if __name__=='__main__':
    sys.exit(main())
