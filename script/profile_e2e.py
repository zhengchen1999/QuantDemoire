"""Attribute complete-network CUDA kernel time; retain the raw Kineto trace."""
import os
os.environ['TORCH_CUDNN_V8_API_DISABLED']='1'
import argparse
from collections import defaultdict
import json
from pathlib import Path
import torch
from torch.profiler import profile, ProfilerActivity
from quantdemoire.deploy import PackedConv,sha256
from quantdemoire.precision import configure,metadata
from quantdemoire.experiment import model_for,pad_image,timing,write_json


def category(name,operation):
    if 'attention_residual_kernel' in name:return 'pointwise_fused'
    if any(s in name for s in ('prepare_kernel','prepare_pair_kernel','prepare_quad_kernel')):return 'quantize_smooth_pack'
    if 'JointOutput' in name:return 'joint_integer_sparse_output'
    if any(s in name for s in ('sparse_half_kernel<4>','sparse_half_kernel<8>','sparse_half_kernel<4,','sparse_half_kernel<8,')):return 'fp16_sparse_merge_and_next_quant'
    if any(s in name for s in ('sparse_tuned_kernel<4,','sparse_tuned_kernel<8,')):return 'fp16_sparse_merge_and_next_quant'
    if 'sparse_tuned_kernel' in name:return 'fp16_sparse_merge'
    if any(s in name for s in ('sparse_half_kernel','sparse_epilogue_kernel','sparse_tc')):return 'fp16_sparse_merge'
    if 'cutlass' in name.lower() and 'cudnn' not in name.lower():return 'integer_convolution_and_epilogue'
    if 'conv' in operation or any(s in name.lower() for s in ('cudnn','xmma','implicit_convolve','winograd','ampere_fp16')):return 'fp16_convolution'
    if 'upsample' in operation or 'upsample' in name:return 'interpolation'
    if 'pixel_shuffle' in operation or 'pixel_unshuffle' in operation or 'pixel_shuffle' in name:return 'pixel_shuffle'
    if 'cat' in operation or 'CatArray' in name or 'Cat' in name:return 'concatenation'
    if any(s in operation for s in ('pool','mean','sum')) or 'reduce_kernel' in name:return 'pooling_reduction'
    if any(s in operation for s in ('aten::add','aten::mul','relu','sigmoid','threshold','clamp')):return 'pointwise'
    if any(s in operation for s in ('copy','contiguous','clone')):return 'layout_copy'
    return 'other'


def summarize_trace(trace,frames):
    events=json.loads(Path(trace).read_text())['traceEvents']
    # Kineto's CUDA External id may identify the runtime call, not an ATen op.
    # Correlate the kernel with its launch, then inspect enclosing CPU scopes.
    runtimes={e.get('args',{}).get('correlation'):e for e in events if e.get('cat')=='cuda_runtime'}
    scopes=defaultdict(list)
    for e in events:
        if e.get('cat')=='cpu_op' and e.get('ph')=='X':scopes[(e['pid'],e['tid'])].append(e)
    kernels=[e for e in events if e.get('cat')=='kernel' and e.get('ph')=='X']
    if not kernels:raise RuntimeError('No CUDA kernel activity in trace; verify CUPTI availability')
    groups=defaultdict(lambda:dict(total_ms=0.,launches=0))
    names=defaultdict(lambda:dict(total_ms=0.,launches=0))
    for e in kernels:
        launch=runtimes.get(e.get('args',{}).get('correlation'))
        ancestors=[]
        if launch:
            ancestors=[s for s in scopes[(launch['pid'],launch['tid'])] if s['ts']<=launch['ts']<=s['ts']+s['dur']]
        operation=' > '.join(s['name'] for s in sorted(ancestors,key=lambda x:-x['dur'])) or 'unscoped_extension'
        group=category(e['name'],operation)
        for row in (groups[group],names[(e['name'],operation,group)]):
            row['total_ms']+=e['dur']/1000/frames;row['launches']+=1/frames
    total=sum(r['total_ms'] for r in groups.values())
    for row in groups.values():row['fraction']=row['total_ms']/total
    return dict(profiled_cuda_kernel_sum_ms=total,
                profiled_gpu_span_ms=(max(e['ts']+e['dur'] for e in kernels)-min(e['ts'] for e in kernels))/1000/frames,
                groups=dict(sorted(groups.items(),key=lambda x:x[1]['total_ms'],reverse=True)),
                kernels=[dict(name=key[0],cpu_operation=key[1],group=key[2],**row) for key,row in sorted(names.items(),key=lambda x:x[1]['total_ms'],reverse=True)])


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--backend',choices=['fp16','cuda'])
    p.add_argument('--model')
    p.add_argument('--height',type=int,default=2160)
    p.add_argument('--width',type=int,default=3840)
    p.add_argument('--frames',type=int,default=3)
    p.add_argument('--disable-sparse',action='store_true',help='timing-only ablation: changes network values, not a quality-valid model')
    p.add_argument('--output',required=True)
    p.add_argument('--reanalyze',action='store_true',help='reclassify a saved trace on CPU; no inference')
    a=p.parse_args()
    if a.reanalyze:
        result=json.loads(Path(a.output).read_text())
        result.update(summarize_trace(result['trace'],result['profiled_frames']))
        write_json(a.output,result)
        print(result['groups']);return
    if not a.backend or not a.model:p.error('profiling requires --backend and --model')
    configure();torch.manual_seed(123)
    model=model_for(a.backend,a.model)
    if a.disable_sparse:
        if a.backend!='cuda':p.error('--disable-sparse requires --backend cuda')
        for m in model.modules():
            if isinstance(m,PackedConv):m.sparse_enabled=False
    x,_=pad_image(torch.rand(1,3,a.height,a.width,device='cuda',dtype=torch.float16))
    x=x.contiguous(memory_format=torch.channels_last)
    measured=timing(lambda:model(x),warmup=5,repeats=20)
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
        for _ in range(a.frames):model(x)
        torch.cuda.synchronize()
    trace=Path(a.output).with_suffix('.trace.json')
    trace.parent.mkdir(parents=True,exist_ok=True)
    prof.export_chrome_trace(str(trace))
    result=dict(backend=a.backend,model=a.model,input=list(x.shape),environment=metadata(),
                weights_sha256=sha256(Path(a.model)/'model.safetensors' if a.backend=='cuda' else a.model),
                config_sha256=sha256(Path(a.model)/'config.json') if a.backend=='cuda' else None,
                sparse_enabled=not a.disable_sparse,accuracy_valid=not a.disable_sparse,
                unprofiled_timing=measured,profiled_frames=a.frames,**summarize_trace(trace,a.frames),
                trace=str(trace),note='CUDA kernel durations are profiled; use unprofiled_timing for latency. No claim of measured DRAM bandwidth or occupancy.')
    write_json(a.output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='kernels'},indent=2),flush=True)


if __name__=='__main__':main()
