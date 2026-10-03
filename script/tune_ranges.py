"""Select a global activation range multiplier using held-out TRAINING images."""
import os
os.environ["TORCH_CUDNN_V8_API_DISABLED"]="1"
import argparse
import copy
import json
import random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from safetensors.torch import save_file

from quantdemoire.deploy import load_model, PackedConv, sha256
from quantdemoire.experiment import model_for, write_json
from quantdemoire.precision import configure


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True)
    p.add_argument("--checkpoint",default="checkpoints/uhdm_checkpoint.pth")
    p.add_argument("--pool",required=True)
    p.add_argument("--calibration",default="data/UHDM/train")
    p.add_argument("--count",type=int,default=24)
    p.add_argument("--crop",type=int,default=512)
    p.add_argument("--factors",type=float,nargs="+",default=[1,1.15,1.3,1.5,2])
    p.add_argument("--output",required=True)
    p.add_argument("--backend",choices=['cuda','fake'],default='cuda')
    a=p.parse_args();configure();torch.backends.cudnn.benchmark=False
    excluded={str(x.relative_to(a.calibration)) for x in Path(a.calibration).rglob('*_gt.jpg')}
    candidates=[x for x in sorted(Path(a.pool).rglob('*_gt.jpg')) if str(x.relative_to(a.pool)) not in excluded]
    random.Random(2026).shuffle(candidates)
    if len(candidates)<a.count:raise ValueError("insufficient held-out training images")
    teacher=model_for('fp16',a.checkpoint)
    samples=[]
    for gt in candidates[:a.count]:
        images=[]
        for path in (gt.with_name(gt.name.replace('_gt.jpg','_moire.jpg')),gt):
            with Image.open(path) as im:
                w,h=im.size;l,t=(w-a.crop)//2,(h-a.crop)//2
                im=im.convert('RGB').crop((l,t,l+a.crop,t+a.crop))
                images.append(torch.from_numpy(np.array(im,dtype=np.float32)/255).permute(2,0,1)[None].cuda())
        x=images[0].half().contiguous(memory_format=torch.channels_last)
        ref=teacher(x)[0].float()
        samples.append((gt,x,images[1],ref))
    del teacher
    model=load_model(a.model,a.backend)
    original={n:{k:v.clone() for k,v in m.named_buffers()} for n,m in model.named_modules() if isinstance(m,PackedConv)}
    trials=[];best=None
    for factor in a.factors:
        for name,m in model.named_modules():
            if isinstance(m,PackedConv):
                for key in ('activation','output_scale','offset_sums'):
                    getattr(m,key).copy_(original[name][key])
                m.activation[0]*=factor;m.output_scale.mul_(factor);m.offset_sums.mul_(factor)
                m.refresh_runtime()
        rows=[]
        for gt,x,target,ref in samples:
            pred=model(x if a.backend=='cuda' else x.float())[0].float().clamp(0,1);ref=ref.clamp(0,1)
            psnr=float(-10*torch.log10((pred-target).square().mean()))
            baseline=float(-10*torch.log10((ref-target).square().mean()))
            rows.append(dict(image=str(gt.relative_to(a.pool)),gt_sha256=sha256(gt),psnr=psnr,baseline_psnr=baseline,
                             teacher_mse=float((pred-ref).square().mean())))
        trial=dict(factor=factor,psnr=float(np.mean([r['psnr'] for r in rows])),
                   gap=float(np.mean([r['baseline_psnr']-r['psnr'] for r in rows])),rows=rows)
        trials.append(trial);print({k:v for k,v in trial.items() if k!='rows'},flush=True)
        if best is None or trial['psnr']>best['psnr']:best=trial
        write_json(Path(a.output)/'range_search.json',dict(split='held-out training',crop=a.crop,trials=trials,best_factor=best['factor']))
    factor=best['factor']
    for name,m in model.named_modules():
        if isinstance(m,PackedConv):
            for key in ('activation','output_scale','offset_sums'):getattr(m,key).copy_(original[name][key])
            m.activation[0]*=factor;m.output_scale.mul_(factor);m.offset_sums.mul_(factor)
            m.refresh_runtime()
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    save_file({k:v.cpu().contiguous() for k,v in model.state_dict().items()},str(output/'model.safetensors'),metadata={'format':'quantdemoire-v1'})
    config=json.loads((Path(a.model)/'config.json').read_text())
    config['weights_sha256']=sha256(output/'model.safetensors')
    config['activation_range_multiplier']=factor
    config['range_search']='range_search.json'
    write_json(output/'config.json',config)


if __name__=='__main__':main()
