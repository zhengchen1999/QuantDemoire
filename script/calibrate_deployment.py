"""Training-only weighted activation MSE calibration and optional bias correction."""
import os
os.environ['TORCH_CUDNN_V8_API_DISABLED']='1'
import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from safetensors.torch import save_file

from quantdemoire.deploy import PackedConv,ChannelConcat,load_model,_replace,sha256
from quantdemoire.experiment import model_for,write_json
from quantdemoire.data import create_dataset
from quantdemoire.precision import configure


class Observer(nn.Module):
    def __init__(self,q,ref,factors):
        super().__init__();self.q=q;self.ref=ref;self.stage=1
        self.factors=torch.tensor(factors,device='cuda')
        self.errors=torch.zeros(len(factors),device='cuda')
        self.bias_sum=torch.zeros(q.spec['shape'][0],device='cuda')
        self.batches=0
        smooth=q.inverse_smooth.reciprocal()
        self.importance=(ref.weight.float()*smooth[None,:,None,None]).square().sum((0,2,3))
        self.importance/=self.importance.mean().clamp_min(1e-12)

    def forward(self,x):
        materialized=torch.cat(x.parts,1) if isinstance(x,ChannelConcat) else x
        if self.stage==1:
            flat=materialized.permute(0,2,3,1).reshape(-1,materialized.shape[1])
            indices=torch.randint(flat.shape[0],(min(128,flat.shape[0]),),device=flat.device)
            sample=flat[indices].float()*self.q.inverse_smooth
            a=self.q.activation[0]*self.factors[:,None,None];z=self.q.activation[1]
            reconstructed=((sample[None]/a+z).round().clamp(0,2**self.q.spec['bits']-1)-z)*a
            self.errors+=((reconstructed-sample[None]).square()*self.importance).mean((1,2))
        ref=self.ref(materialized)
        if self.stage==2:
            actual=self.q(x)
            self.bias_sum+=(ref.float()-actual.float()).mean((0,2,3))
            self.batches+=1
        return ref


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--checkpoint',default='checkpoints/uhdm_checkpoint.pth')
    p.add_argument('--data',default='data/UHDM/train')
    p.add_argument('--output',required=True)
    p.add_argument('--count',type=int,default=200)
    p.add_argument('--crop',type=int,default=512)
    p.add_argument('--no-bias-correction',action='store_true')
    a=p.parse_args();configure();torch.backends.cudnn.benchmark=False
    torch.manual_seed(123);random.seed(123);np.random.seed(123)
    model=load_model(a.model,'cuda',fuse_relu=False)
    teacher=model_for('fp16',a.checkpoint)
    observers={}
    for name,m in list(model.named_modules()):
        if isinstance(m,PackedConv):
            factors=[.5,.65,.8,1.,1.2,1.4,1.7,2.,2.5,3.,4.,6.]
            observer=Observer(m,teacher.get_submodule(name),factors)
            observers[name]=observer;_replace(model,name,observer)
    args=SimpleNamespace(DATA_TYPE='UHDM',LOADER='crop',CROP_SIZE=a.crop,BATCH_SIZE=1,WORKER=4)
    statistics={}
    for stage in (1,2):
        if stage==2 and a.no_bias_correction:break
        for observer in observers.values():observer.stage=stage
        loader=create_dataset(args,a.data,'train',sample_num=a.count,batch_size=1,shuffle=False,drop_last=False)
        for i,batch in enumerate(loader):
            model(batch['in_img'].cuda().half().contiguous(memory_format=torch.channels_last))
            if i%20==0:print(f'pass {stage}: {i+1}/{len(loader)}',flush=True)
        if stage==1:
            for name,obs in observers.items():
                index=int(obs.errors.argmin());factor=float(obs.factors[index])
                obs.q.activation[0]*=factor;obs.q.output_scale*=factor;obs.q.offset_sums*=factor
                obs.q.refresh_runtime()
                statistics[name]=dict(multiplier=factor,weighted_mse=obs.errors.cpu().tolist())
        else:
            for name,obs in observers.items():
                correction=obs.bias_sum/max(obs.batches,1)
                obs.q.bias+=correction
                statistics[name]['bias_correction_l2']=float(correction.norm())
    for name,obs in observers.items():_replace(model,name,obs.q)
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    save_file({k:v.cpu().contiguous() for k,v in model.state_dict().items()},str(output/'model.safetensors'),metadata={'format':'quantdemoire-v1'})
    config=json.loads((Path(a.model)/'config.json').read_text())
    config['weights_sha256']=sha256(output/'model.safetensors')
    config['deployment_calibration']='weighted activation MSE; '+('no bias correction' if a.no_bias_correction else 'FP16 teacher bias correction')
    write_json(output/'config.json',config)
    write_json(output/'calibration.json',dict(seed=123,config=vars(a),layers=statistics,
        manifest=[str(x.relative_to(a.data)) for x in sorted(Path(a.data).rglob('*_gt.jpg'))[:a.count]]))


if __name__=='__main__':main()
