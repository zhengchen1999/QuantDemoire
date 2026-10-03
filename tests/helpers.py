import math
import torch
from torch.nn import functional as F
from quantdemoire.deploy import PackedConv, pack_signed


def synthetic_layer(record, bits, zero_point=None, tail_fraction=.0025):
    k,c,r,_ = record["weight"]
    cp = math.ceil(c/(32 if bits==4 else 16))*(32 if bits==4 else 16)
    # Both signed tails are retained, with an exact rank count including ties.
    raw = torch.randn(k,r,r,c)*.04
    count = math.ceil(raw.numel()*tail_fraction)
    order = raw.flatten().argsort(stable=True)
    mask = torch.zeros(raw.numel(),dtype=torch.bool)
    mask[order[:count]]=True
    mask[order[-count:]]=True
    mask=mask.reshape(k,-1)
    scale = torch.full((k,),.1 / ((1<<(bits-1))-1))
    q = torch.round(raw/scale[:,None,None,None]).clamp(-(1<<(bits-1)),(1<<(bits-1))-1)
    q.reshape(k,-1)[mask]=0
    a=torch.tensor(.1 if bits==4 else .01)
    z=torch.tensor(float((1<<(bits-1))-2 if zero_point is None else zero_point))
    row=torch.cat([torch.zeros(1,dtype=torch.long),mask.sum(1).cumsum(0)]).int()
    params=dict(packed_weight=pack_signed(F.pad(q,(0,cp-c)),bits), row=row,
                indices=torch.cat([torch.where(m)[0] for m in mask]).int(),
                values=raw.flatten()[mask.flatten()].half(), weight_scale=scale,
                output_scale=scale*a, inverse_smooth=torch.rand(c)*.5+.75,
                activation=torch.stack([a,z]), bias=torch.randn(k)*.01,
                offset_sums=q.sum(-1)*scale[:,None,None]*a)
    spec=dict(bits=bits,shape=[k,c,r,r],padded_channels=cp,stride=record["stride"],
              padding=record["padding"],dilation=record["dilation"],relu=False)
    return PackedConv(spec,params)
