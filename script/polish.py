"""Refine activation steps against an FP16/FP32-accumulator ESDNet teacher.

Both outlier tails stay fixed. By default only activation steps are trained;
--train-weights also updates normal integer weights and biases. Training uses
the deployed CSR structure and FP16 sparse operands. No test images are used.
"""
import os
os.environ["TORCH_CUDNN_V8_API_DISABLED"]="1"
import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import save_file

from quantdemoire.deploy import load_model, PackedConv, sha256, _replace, pack_signed
from quantdemoire.experiment import model_for, write_json
from quantdemoire.data import create_dataset
from quantdemoire.quantization.observers import ste_round
from quantdemoire.quantization.frequency_loss import gaussian_lowpass
from quantdemoire.precision import configure


class TrainConv(PackedConv):
    def __init__(self, layer, train_weights=False):
        super().__init__(layer.spec,{k:v for k,v in layer.named_buffers()},"fake")
        self.log_step=nn.Parameter(torch.zeros((),device=self.activation.device))
        self.dense_weight()
        self.train_weights=train_weights
        if train_weights:
            self.normal_weight=nn.Parameter((self._dense_cache-self._sparse_cache).clone())
            self.train_bias=nn.Parameter(self.bias.clone())
            mask=torch.ones_like(self._dense_cache,dtype=torch.bool).permute(0,2,3,1).contiguous()
            for oc in range(self.spec['shape'][0]):
                mask[oc].flatten()[self.indices[self.row[oc]:self.row[oc+1]].long()]=False
            self.normal_mask=mask.permute(0,3,1,2).contiguous()

    def normal_codes(self):
        return ste_round(self.normal_weight/self.weight_scale[:,None,None,None]).clamp(
            -(1<<(self.spec['bits']-1)),(1<<(self.spec['bits']-1))-1)*self.normal_mask

    def forward(self,x):
        s=self.spec;a=self.activation[0]*self.log_step.exp();z=self.activation[1]
        q=ste_round(x*(self.inverse_smooth/a)[None,:,None,None]+z).clamp(0,2**s['bits']-1)
        dq=(q-z)*a
        if self.train_weights:
            weight=self.normal_codes()*self.weight_scale[:,None,None,None]+self._sparse_cache
            # Fast FP16 training surrogate. Evaluation always uses the exported
            # integer codes and exact CUDA/fake-deployment arithmetic.
            y=F.conv2d(dq.half(),weight.half(),self.train_bias.half(),s['stride'],s['padding'],s['dilation']).float()
            return F.relu(y) if s.get('relu',False) else y
        y=F.conv2d(dq,self._dense_cache,self.bias,s['stride'],s['padding'],s['dilation'])
        y=y+F.conv2d(dq.half().float()-dq,self._sparse_cache,None,s['stride'],s['padding'],s['dilation'])
        if s.get('relu',False):y=F.relu(y)
        return y.half().float()


def save(model,source,output,epoch,history):
    tensors={k:v.detach().cpu().contiguous().clone() for k,v in model.state_dict().items() if not k.endswith(('.log_step','.normal_weight','.train_bias'))}
    for name,m in model.named_modules():
        if isinstance(m,TrainConv):
            factor=float(m.log_step.exp().detach())
            tensors[name+'.activation'][0]*=factor
            tensors[name+'.output_scale']*=factor
            tensors[name+'.offset_sums']*=factor
            if m.train_weights:
                codes=m.normal_codes().detach().permute(0,2,3,1).contiguous().cpu()
                cp=m.spec['padded_channels'];c=m.spec['shape'][1]
                tensors[name+'.packed_weight']=pack_signed(F.pad(codes,(0,cp-c)).to(torch.int8),m.spec['bits'])
                tensors[name+'.offset_sums']=codes.sum(-1)*tensors[name+'.output_scale'][:,None,None]
                tensors[name+'.bias']=m.train_bias.detach().cpu().contiguous()
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    save_file(tensors,str(output/'model.safetensors'),metadata={'format':'quantdemoire-v1'})
    config=json.loads((Path(source)/'config.json').read_text())
    config['weights_sha256']=sha256(output/'model.safetensors')
    config['polish_epoch']=epoch
    config['polish_method']='activation log-step distillation + Gaussian low/mid-frequency loss; fixed CSR/code weights'
    if any(isinstance(m,TrainConv) and m.train_weights for m in model.modules()):
        config['polish_method']='integer normal-weight/bias and activation-step distillation; fixed original FP16 tails; FP16 training surrogate'
    write_json(output/'config.json',config)
    write_json(output/'training.json',history)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--checkpoint',default='checkpoints/uhdm_checkpoint.pth')
    p.add_argument('--data',default='data/UHDM/train')
    p.add_argument('--output',required=True)
    p.add_argument('--epochs',type=int,default=4)
    p.add_argument('--crop',type=int,default=512)
    p.add_argument('--count',type=int,default=200)
    p.add_argument('--lr',type=float,default=.01)
    p.add_argument('--train-weights',action='store_true')
    p.add_argument('--weight-lr',type=float,default=1e-5)
    a=p.parse_args();configure();torch.manual_seed(123);random.seed(123);np.random.seed(123)
    teacher=model_for('fp16',a.checkpoint)
    teacher.requires_grad_(False)
    student=load_model(a.model,'fake')
    student.requires_grad_(False)
    for name,m in list(student.named_modules()):
        if isinstance(m,PackedConv):_replace(student,name,TrainConv(m,a.train_weights))
        elif isinstance(m,nn.Conv2d):
            m.weight.data=m.weight.data.half().float()
            m.bias.data=m.bias.data.half().float()
    params=[m.log_step for m in student.modules() if isinstance(m,TrainConv)]
    groups=[dict(params=params,lr=a.lr)]
    if a.train_weights:
        groups.append(dict(params=[v for m in student.modules() if isinstance(m,TrainConv) for v in (m.normal_weight,m.train_bias)],lr=a.weight_lr))
    opt=torch.optim.Adam(groups)
    scaler=torch.cuda.amp.GradScaler(enabled=True)
    args=SimpleNamespace(DATA_TYPE='UHDM',LOADER='crop',CROP_SIZE=a.crop,BATCH_SIZE=1,WORKER=4)
    loader=create_dataset(args,a.data,'train',sample_num=a.count,batch_size=1,shuffle=True,drop_last=False)
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:.05+.95*(1+np.cos(np.pi*step/(a.epochs*len(loader))))/2)
    history=dict(seed=123,config=vars(a),manifest=[str(x.relative_to(a.data)) for x in sorted(Path(a.data).rglob('*_gt.jpg'))[:a.count]],epochs=[])
    student.train()
    for epoch in range(a.epochs):
        total=0.
        for i,batch in enumerate(loader):
            x=batch['in_img'].cuda().contiguous(memory_format=torch.channels_last)
            with torch.no_grad():refs=teacher(x.half())
            outputs=student(x.half().float())
            loss=sum(F.mse_loss(y,t.float())+F.mse_loss(gaussian_lowpass(y,3),gaussian_lowpass(t.float(),3)) for y,t in zip(outputs,refs))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward();scaler.step(opt);scaler.update();scheduler.step()
            with torch.no_grad():
                for v in params:v.clamp_(-.7,.7)
            total+=loss.item()
            if i%20==0:print(f'epoch {epoch+1} step {i+1}/{len(loader)} loss {total/(i+1):.7f}',flush=True)
        history['epochs'].append(dict(epoch=epoch+1,loss=total/len(loader),min_multiplier=min(float(v.exp()) for v in params),max_multiplier=max(float(v.exp()) for v in params)))
        save(student,a.model,Path(a.output)/f'epoch{epoch+1}',epoch+1,history)
        print(history['epochs'][-1],flush=True)


if __name__=='__main__':main()
