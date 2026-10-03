"""Estimate sparse staging volume under alternative smoothing; does not claim speed or quality."""
import argparse
import json
import math
from pathlib import Path
import torch
from quantdemoire.deploy import sha256
from quantdemoire.experiment import write_json
from quantdemoire.quantization.observers import rank_tail_mask


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--quant-state',required=True)
    p.add_argument('--shapes',required=True,help='benchmark_branch.py report for the same ESDNet geometry')
    p.add_argument('--source-alpha',type=float,default=.5)
    p.add_argument('--alphas',type=float,nargs='+',default=[.5,.65,.8,1.])
    p.add_argument('--tail-fraction',type=float,default=.0025)
    p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(4)
    if not 0<a.source_alpha<=1 or any(not 0<=v<=1 for v in a.alphas):p.error('invalid smoothing exponent')
    if not .0025<=a.tail_fraction<.5:p.error('tail fraction must retain at least .25% of each tail')
    state=torch.load(a.quant_state,map_location='cpu',weights_only=True)
    weights=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    records={r['layer']:r for r in json.loads(Path(a.shapes).read_text())['rows']}
    rows=[]
    for alpha in a.alphas:
        layers=[]
        for name,old in state.items():
            if old['skip_quantization']:continue
            weight=weights[name+'.weight'];peak=weight.abs().amax((0,2,3)).clamp_min(1e-6)
            magnitude=(old['smooth_scale']*peak.pow(1-a.source_alpha)).pow(1/a.source_alpha)
            smooth=old['smooth_scale'] if alpha==a.source_alpha else (magnitude.pow(alpha)/peak.pow(1-alpha)).clamp_min(1e-6)
            mask=rank_tail_mask(weight*smooth[None,:,None,None],a.tail_fraction)
            channels=int(mask.any(dim=(0,2,3)).sum());k,c,r,_=weight.shape
            n,actual_c,h,w=records[name]['input']
            assert c==actual_c and mask.sum()>=2*math.ceil(weight.numel()*a.tail_fraction)
            counts=mask.reshape(k,-1).sum(1)
            layers.append(dict(name=name,shape=list(weight.shape),sparse_channels=channels,
                full_channels=c,nnz=int(mask.sum()),max_row_nnz=int(counts.max()),
                spatial_planar_bytes=n*channels*h*w if r>1 else 0))
        rows.append(dict(alpha=alpha,layers=layers,
            spatial_planar_bytes=sum(r['spatial_planar_bytes'] for r in layers),nnz=sum(r['nnz'] for r in layers)))
    write_json(a.output,dict(note=__doc__,checkpoint_sha256=sha256(a.checkpoint),quant_state_sha256=sha256(a.quant_state),
        source_alpha=a.source_alpha,tail_fraction=a.tail_fraction,rows=rows,
        caveat='Reconstructs sampled channel statistics from saved smoothing. New activations require recalibration; no weights or profiles are modified.'))
    for row in rows:print({k:v for k,v in row.items() if k!='layers'})


if __name__=='__main__':main()
