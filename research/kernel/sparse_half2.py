"""Experimental FP16 accumulation of the FP16 branch using packed half2 FMA.

This deliberately changes sparse accumulation precision. It must pass its own
serial FP64/FP16 oracle and quality evaluation; FP32-accumulator equivalence is
not claimed. The integer branch and serialized sparse weights are unchanged.
"""
from functools import lru_cache
import torch
from kernel.diagnostic import generated_sources, build_sources
from .sparse_prefetch import sources as host_sources
from .spatial_packed import convolution as invoke


def sources():
    header=generated_sources()['diagnostic_common.cuh']
    start=header.index('       int oc=oc0+local;float acc[PixelsPerLane]={};')
    end=header.index('\n     }\n     __syncthreads();',start)
    body='''       int oc=oc0+local;
       half2 acc[PixelsPerLane/2];
       #pragma unroll
       for(int t=0;t<PixelsPerLane/2;++t)acc[t]=__float2half2_rn(0.f);
       if(oc<p_.k) {
         for(int j=p_.row[oc];j<p_.row[oc+1];++j) {
           Index off;half weight;
           if constexpr(Variant>=35&&Variant<=46) {
             uint64_t entry=static_cast<uint64_t>(p_.offsets[j]);
             off=Index(int32_t(uint32_t(entry)));
             weight=__ushort_as_half(uint16_t(entry>>32));
           } else {off=Index(p_.offsets[j]);weight=p_.value[j];}
           half2 ww=__half2half2(weight);
           if(inside) {
             #pragma unroll
             for(int t=0;t<PixelsPerLane;t+=2) {
               int c0=int(p_.x[base[t]+off]),c1=int(p_.x[base[t+1]+off]);
               float a0,a1;
               if constexpr(SharedW4Lookup){a0=operand_table[c0+8];a1=operand_table[c1+8];}
               else {a0=(float(c0)+delta)*scale;a1=(float(c1)+delta)*scale;}
               acc[t/2]=__hfma2(__floats2half2_rn(a0,a1),ww,acc[t/2]);
             }
           } else {
             int loc=p_.location[j],ic=loc&4095,dx=((loc>>12)&15)*p_.dil,dy=((loc>>16)&15)*p_.dil;
             #pragma unroll
             for(int t=0;t<PixelsPerLane;t+=2) {
               float aa[2]={};
               #pragma unroll
               for(int u=0;u<2;++u) {
                 int iy=iy0[t+u]+dy,ix=ix0[t+u]+dx;
                 if(valid[t+u]&&iy>=0&&iy<p_.h&&ix>=0&&ix<p_.w)
                   aa[u]=(float(p_.x[((Index(batch[t+u])*p_.c+ic)*p_.h+iy)*p_.w+ix])+delta)*scale;
               }
               acc[t/2]=__hfma2(__floats2half2_rn(aa[0],aa[1]),ww,acc[t/2]);
             }
           }
         }
       }
       #pragma unroll
       for(int t=0;t<PixelsPerLane;t+=2) {
         float2 aa=__half22float2(acc[t/2]);
         sparse_tile_[(lane+t*32)*(TileN+1)+local]=aa.x;
         sparse_tile_[(lane+(t+1)*32)*(TileN+1)+local]=aa.y;
       }'''
    header=header[:start]+body+header[end:]
    result=host_sources();result['spatial_packed_common.cuh']=header
    return result


@lru_cache(None)
def extension():return build_sources('sparse_half2',sources())


def convolution(m,packed,planar,**kwargs):
    return invoke(m,packed,planar,_backend=extension(),_locations=m.compact_locations,**kwargs)


@torch.inference_mode()
def reference(m,x):
    """Independent serial branch: exact FP64 products/addition, round each FMA to FP16."""
    from torch.nn import functional as F
    sparse,relu=m.sparse_enabled,m.spec.get('relu',False)
    try:
        m.sparse_enabled=False;m.spec['relu']=False
        dense=m(x).float()
    finally:
        m.sparse_enabled=sparse;m.spec['relu']=relu
    s=m.spec;a,z=m.activation
    code=(x.float()*m.quant_multiplier[None,:,None,None]+z).round().clamp(0,2**s['bits']-1)
    # Match the kernel's centered code expression before half rounding.
    center=1<<(s['bits']-1)
    values=((code-center+(center-z))*a).half().double()
    columns=F.unfold(values,s['shape'][2],dilation=s['dilation'],padding=s['padding'],stride=s['stride'])
    branch=torch.zeros_like(dense,dtype=torch.float64).flatten(2)
    row=m.row.cpu().tolist();locations=m.locations.cpu().tolist();r=s['shape'][2]
    for oc in range(s['shape'][0]):
        acc=branch[:,oc]
        for j in range(row[oc],row[oc+1]):
            loc=locations[j];ic=loc&4095;dx=(loc>>12)&15;dy=(loc>>16)&15
            acc=(acc+columns[:,(ic*r+dy)*r+dx]*m.values[j].double()).half().double()
        branch[:,oc]=acc
    output=dense+branch.reshape_as(dense).float()
    return (F.relu(output) if relu else output).half()
