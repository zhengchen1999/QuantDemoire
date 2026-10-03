"""Restore a real image with portable integer weights; optionally capture a graph."""
import os
os.environ['TORCH_CUDNN_V8_API_DISABLED']='1'
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from quantdemoire.precision import configure
from quantdemoire.deploy import load_model
from quantdemoire.experiment import pad_image
from quantdemoire.runtime import CudaGraphRunner


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--input',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--graph',action='store_true')
    p.add_argument('--runtime-profile',action='store_true',
                   help='apply the model folder execution profile before loading')
    a=p.parse_args()
    if a.runtime_profile:
        from quantdemoire.profiles import apply_execution_profile
        apply_execution_profile(a.model)
    configure()
    with Image.open(a.input) as im:
        x=torch.from_numpy(np.array(im.convert('RGB'),dtype=np.float32)/255).permute(2,0,1)[None].cuda().half()
    x,(t,l,h,w)=pad_image(x);x=x.contiguous(memory_format=torch.channels_last)
    model=load_model(a.model,'cuda')
    inference=CudaGraphRunner(model,x) if a.graph else model
    result=inference(x)[0][0,:,t:t+h,l:l+w].float().clamp(0,1)
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray((result.permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)).save(output)


if __name__=='__main__':main()
