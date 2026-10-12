"""Quality metrics and fused / CUDA Graph equivalence checks."""
import os
os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"
import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from quantdemoire.precision import configure, metadata
from quantdemoire.experiment import model_for, pad_image, write_json
from quantdemoire.deploy import sha256, PackedConv
from quantdemoire.esdnet import ChannelScaleAttention, ResidualDenseBlock
from quantdemoire.runtime import CudaGraphRunner
from quantdemoire.metrics import QualityMetrics
from quantdemoire.evaluation_subset import selected_pairs


@torch.inference_mode()
def evaluate_main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["fp32", "fp16", "fake", "fake_fp16", "cuda"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--data", default="data/UHDM/test")
    p.add_argument("--limit", type=int)
    p.add_argument('--manifest',help='optional dataset manifest; verify pair hashes before evaluation')
    p.add_argument('--subset',default='test',choices=('test','calibration','heldout_training'))
    p.add_argument("--crop", type=int, help="center crop for development only; omitted means original resolution")
    p.add_argument("--output", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument('--runtime-profile',action='store_true',help='apply the saved CUDA execution profile before model loading')
    p.add_argument("--autotune", action="store_true", help="cuDNN algorithm search; useful for many equal-sized images")
    p.add_argument('--memory-format', choices=('channels_last','contiguous'), default='channels_last',
                   help='convolution layout; contiguous is substantially faster for strict FP32 on RTX 4090')
    a = p.parse_args()
    a.data=str(Path(a.data).resolve())
    paths=selected_pairs(a.data,a.manifest,a.subset,a.limit)
    selection_hash=sha256(a.manifest) if a.manifest else None
    if a.backend == 'cuda' and a.memory_format != 'channels_last':
        p.error('integer CUDA deployment requires channels_last features')
    if a.runtime_profile:
        if a.backend!='cuda':p.error('--runtime-profile requires --backend cuda')
        from quantdemoire.profiles import apply_execution_profile
        apply_execution_profile(a.model)
    configure()
    torch.backends.cudnn.benchmark = a.autotune
    model = model_for(a.backend, a.model)
    layout = torch.channels_last if a.memory_format == 'channels_last' else torch.contiguous_format
    model.to(memory_format=layout)
    if a.backend == 'cuda':
        from kernel.ops import extension
        extension()  # Record the loaded source hash before writing evaluation metadata.
    metrics = QualityMetrics(torch.device("cuda"))
    weight_path = Path(a.model)/'model.safetensors' if a.backend in ('cuda','fake','fake_fp16') else Path(a.model)
    weight_hash = sha256(weight_path)
    config_path=Path(a.model)/'config.json'
    config_hash=sha256(config_path) if config_path.is_file() else None
    rows = []
    if a.resume and Path(a.output).is_file():
        previous = json.loads(Path(a.output).read_text())
        if previous['backend'] != a.backend or previous['model'] != str(Path(a.model).resolve()) or previous['crop'] != a.crop:
            raise ValueError("resume configuration differs")
        if previous['model_sha256'] != weight_hash or previous['split_root'] != str(Path(a.data).resolve()):
            raise ValueError("resume model weights or dataset differ")
        if previous.get('config_sha256')!=config_hash:
            raise ValueError('resume deployment configuration differs')
        if previous.get('selection_manifest_sha256')!=selection_hash or previous.get('selection_subset','test')!=a.subset:
            raise ValueError('resume evaluation subset differs')
        current_environment=metadata()
        for key in ('torch','cuda','cudnn','tf32','fp16_reduced_precision_reduction','fp16_convolution_accumulator',
                    'sparse_implementation','csr_kernel','joint_warps','joint_variant','shared_skip_planar','joint_next_quant',
                    'balance_csr','shared_retain_stages','few_channels','packed_csr_meta','shared_skip_prefix_planar',
                    'joint_plan_sha256','main_half_override','compact_sparse','prepare_implementation','fuse_next_quant',
                    'fuse_second_quant','omit_cached_features','release_features','fuse_attention','fuse_residual',
                    'deployment_source_sha256','kernel_source_sha256'):
            if previous.get('environment',{}).get(key)!=current_environment.get(key):
                raise ValueError(f'resume numerical environment differs: {key}')
        if previous.get('cudnn_autotune',False)!=a.autotune:
            raise ValueError('resume cuDNN algorithm policy differs')
        if previous.get('memory_format','channels_last')!=a.memory_format:
            raise ValueError('resume convolution layout differs')
        rows = previous['rows']
        selected = {str(p.relative_to(a.data)): p for p in paths}
        for row in rows:
            gt = selected.get(row['image'])
            if gt is None or sha256(gt) != row['gt_sha256'] or sha256(gt.with_name(gt.name.replace('_gt.jpg','_moire.jpg'))) != row['input_sha256']:
                raise ValueError("resume image subset or image contents differ")
    evaluated = {r['image'] for r in rows}
    result = dict(backend=a.backend, model=str(Path(a.model).resolve()), environment=metadata(),
                  model_sha256=weight_hash, dataset_scope="available local UHDM split; image hashes below define exact subset",
                  config_sha256=config_hash,
                  selection_manifest_sha256=selection_hash,selection_subset=a.subset,
                  runtime_profile_sha256=sha256(Path(a.model)/'runtime.json') if a.runtime_profile else None,
                  cudnn_autotune=a.autotune,
                  memory_format=a.memory_format,
                  split_root=str(Path(a.data).resolve()), crop=a.crop, requested_count=len(paths),
                  protocol="RGB [0,1], clamped prediction, valid-window SSIM, AlexNet LPIPS", rows=rows)
    for i, gt in enumerate(paths):
        if str(gt.relative_to(a.data)) in evaluated:
            continue
        inp = gt.with_name(gt.name.replace("_gt.jpg", "_moire.jpg"))
        images = []
        for f in (inp, gt):
            with Image.open(f) as im:
                im = im.convert("RGB")
                if a.crop:
                    w, h = im.size
                    if min(w, h) < a.crop:
                        raise ValueError("crop exceeds image dimensions")
                    l, t = (w-a.crop)//2, (h-a.crop)//2
                    im = im.crop((l,t,l+a.crop,t+a.crop))
                images.append(torch.from_numpy(np.array(im, dtype=np.float32) / 255).permute(2,0,1)[None].cuda())
        x, (t,l,h,w) = pad_image(images[0])
        dtype = torch.float16 if a.backend in ("fp16", "cuda", "fake_fp16") else torch.float32
        torch.cuda.synchronize(); started=time.monotonic()
        pred = model(x.to(dtype).contiguous(memory_format=layout))[0][:,:,t:t+h,l:l+w].float()
        torch.cuda.synchronize(); model_seconds=time.monotonic()-started
        lpips, psnr, ssim = metrics.compute(pred, images[1])
        torch.cuda.synchronize(); metric_seconds=time.monotonic()-started-model_seconds
        rows.append(dict(image=str(gt.relative_to(a.data)), shape=list(images[0].shape),
                         gt_sha256=sha256(gt), input_sha256=sha256(inp), psnr=psnr, ssim=ssim, lpips=lpips,
                         model_seconds=model_seconds,metric_seconds=metric_seconds))
        result["count"] = len(rows)
        result["complete"] = len(rows) == len(paths)
        result["mean"] = {k:float(np.mean([r[k] for r in rows])) for k in ("psnr","ssim","lpips")}
        write_json(a.output, result)
        print(f"{i+1}/{len(paths)} {gt.name} {result['mean']}", flush=True)


@torch.inference_mode()
def verify_main():
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

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        del sys.argv[1]
        verify_main()
    else:
        evaluate_main()
