"""Compare graph fusions with their unfused path on every UHDM test image."""
import os
os.environ['TORCH_CUDNN_V8_API_DISABLED']='1'
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from quantdemoire.esdnet import ChannelScaleAttention,ResidualDenseBlock
from quantdemoire.experiment import model_for,pad_image,write_json
from quantdemoire.deploy import sha256,PackedConv
from quantdemoire.precision import configure,metadata
from quantdemoire.runtime import CudaGraphRunner


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['cuda','fp16'],required=True)
    p.add_argument('--model',required=True);p.add_argument('--output',required=True)
    p.add_argument('--data',default='data/UHDM/test');p.add_argument('--limit',type=int)
    p.add_argument('--graph',action='store_true')
    p.add_argument('--csr-kernel',choices=['legacy','linear','lut','wide','linear32','divmod'])
    p.add_argument('--joint-warps',type=int,choices=[0,4,8])
    p.add_argument('--runtime-profile',action='store_true',help='apply the CUDA model folder runtime.json before loading')
    p.add_argument('--compare-feature-release',action='store_true',help='also compare retained reference features against early release in the candidate')
    p.add_argument('--autotune',action='store_true',help='validate with the cuDNN search policy used by deployment benchmarks')
    a=p.parse_args()
    if a.runtime_profile:
        if a.backend!='cuda':p.error('--runtime-profile requires the CUDA backend')
        from quantdemoire.profiles import apply_execution_profile
        apply_execution_profile(a.model)
    if a.csr_kernel is None:a.csr_kernel=os.environ.get('QUANTDEMOIRE_CSR_KERNEL','legacy')
    if a.joint_warps is None:a.joint_warps=int(os.environ.get('QUANTDEMOIRE_JOINT_WARPS','0'))
    configure();torch.backends.cudnn.benchmark=a.autotune
    model=model_for(a.backend,a.model);runner=None
    attention=[m for m in model.modules() if isinstance(m,ChannelScaleAttention)]
    residual=[m for m in model.modules() if isinstance(m,ResidualDenseBlock)]
    quantized=[m for m in model.modules() if isinstance(m,PackedConv)]
    candidate_warps={id(m):m.joint_warps if os.environ.get('QUANTDEMOIRE_JOINT_PLAN') else a.joint_warps for m in quantized}
    def mode(enabled):
        if a.compare_feature_release:
            model.release_features=enabled;model.encoder.release_features=enabled
            os.environ['QUANTDEMOIRE_RELEASE_FEATURES']=str(int(enabled))
        for m in attention:m.fuse_attention=enabled
        for m in residual:m.fuse_residual=enabled
        for m in quantized:
            m.csr_kernel=a.csr_kernel if enabled else 'legacy'
            m.joint_warps=candidate_warps[id(m)] if enabled else 0
    paths=sorted(Path(a.data).rglob('*_moire.jpg'))
    if a.limit:paths=paths[:a.limit]
    if not paths:raise ValueError('empty dataset')
    result=dict(backend=a.backend,model=a.model,cudnn_autotune=a.autotune,compare_feature_release=a.compare_feature_release,model_sha256=sha256(Path(a.model)/'model.safetensors' if a.backend=='cuda' else a.model),
                config_sha256=sha256(Path(a.model)/'config.json') if a.backend=='cuda' else None,graph=a.graph,rows=[],attention_modules=len(attention),residual_modules=len(residual),
                csr_kernel=a.csr_kernel,joint_warps=a.joint_warps,note='Reference disables attention/residual fusions and uses legacy CSR; candidate enables both and the requested CSR/joint kernel. Joint prefix production is controlled by QUANTDEMOIRE_JOINT_NEXT_QUANT and the optional per-layer plan; all numerical quantization parameters are identical.')
    for i,path in enumerate(paths):
        with Image.open(path) as im:x=torch.from_numpy(np.asarray(im.convert('RGB'),dtype=np.float32)/255).permute(2,0,1)[None].cuda()
        x,_=pad_image(x);x=x.half().contiguous(memory_format=torch.channels_last)
        mode(False);expected=model(x)
        mode(True);actual=model(x)
        row=dict(image=str(path.relative_to(a.data)),input_sha256=sha256(path),bitwise_equal=[torch.equal(u,v) for u,v in zip(actual,expected)])
        if not all(row['bitwise_equal']):row['max_abs']=[float((u-v).abs().max()) for u,v in zip(actual,expected)]
        if a.graph:
            if runner is None:runner=CudaGraphRunner(model,x)
            captured=runner(x);row['graph_bitwise_equal']=[torch.equal(u,v) for u,v in zip(captured,expected)]
        result['rows'].append(row);result['count']=len(result['rows']);result['complete']=result['count']==len(paths)
        result['environment']=dict(metadata(),fuse_attention=True,fuse_residual=True,csr_kernel=a.csr_kernel,joint_warps=a.joint_warps)
        write_json(a.output,result)
        if not all(row['bitwise_equal']) or not all(row.get('graph_bitwise_equal',[True])):raise AssertionError(row)
        if (i+1)%10==0:print(i+1,'all outputs bitwise equal',flush=True)
        del x,actual,expected


if __name__=='__main__':main()
