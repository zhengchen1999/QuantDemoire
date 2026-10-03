"""Experimental coalesced FP16 MMA sparse branch; changes accumulation order.

Exact CSR coordinates/values are zero-padded into groups of eight output rows.
Planar activation loads are coalesced across pixels before shared-memory MMA
operand assembly. This standalone screen has no consumer-prefix fusion.
"""
from functools import lru_cache
import torch
from kernel.diagnostic import build_sources

CUDA = r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include "cutlass/fast_math.h"
__global__ void branch(const int8_t* x,const half* dense,half* out,
 const int* row,const int* loc,const half* weights,const float* params,
 int n,int c,int h,int w,int k,int p,int q,int stride,int pad,int dil,int center,bool relu,
 cutlass::FastDivmod q_div,cutlass::FastDivmod pq_div) {
 __shared__ half tile[16][65];
 int lane=threadIdx.x&31,warp=threadIdx.x>>5,group=lane>>2,tid=lane&3;
 int pb=blockIdx.y*64,oc0=blockIdx.x*8;
 int px=threadIdx.x&63,pos=pb+px,b=0,local=pos,yy,xx;
 if(n!=1)pq_div(b,local,pos);
 q_div(yy,xx,local);yy=yy*stride-pad;xx=xx*stride-pad;
 float d0=0,d1=0,d2=0,d3=0,delta=float(center)-params[1],scale=params[0];
 for(int start=row[blockIdx.x];start<row[blockIdx.x+1];start+=16) {
   #pragma unroll
   for(int u=0;u<8;++u) {
     int kk=(threadIdx.x>>6)+u*2,location=loc[start+kk];
     int ic=location&4095,iy=yy+((location>>16)&15)*dil,ix=xx+((location>>12)&15)*dil;
     half a=__float2half(0.f);
     if(location>=0&&pos<n*p*q&&iy>=0&&iy<h&&ix>=0&&ix<w)
       a=__float2half_rn((float(x[((int64_t(b)*c+ic)*h+iy)*w+ix])+delta)*scale);
     tile[kk][px]=a;
   }
   __syncthreads();
   uint32_t aa[4],bb[2];
   #pragma unroll
   for(int z=0;z<4;++z) {
     int rr=warp*16+group+((z&1)?8:0),kk=tid*2+((z&2)?8:0);
     aa[z]=uint32_t(__half_as_ushort(tile[kk][rr]))|(uint32_t(__half_as_ushort(tile[kk+1][rr]))<<16);
   }
   #pragma unroll
   for(int z=0;z<2;++z) {
     int kk=tid*2+z*8;
     bb[z]=uint32_t(__half_as_ushort(weights[(start+kk)*8+group]))|(uint32_t(__half_as_ushort(weights[(start+kk+1)*8+group]))<<16);
   }
   asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
     : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
     : "r"(aa[0]),"r"(aa[1]),"r"(aa[2]),"r"(aa[3]),"r"(bb[0]),"r"(bb[1]));
   __syncthreads();
 }
 float dd[4]={d0,d1,d2,d3};
 #pragma unroll
 for(int z=0;z<4;++z) {
   int pixel=pb+warp*16+group+(z>=2?8:0),oc=oc0+tid*2+(z&1);
   if(pixel<n*p*q&&oc<k) {
     int64_t dst=int64_t(pixel)*k+oc;
     float v=__half2float(dense[dst])+dd[z];out[dst]=__float2half_rn(relu?fmaxf(0.f,v):v);
   }
 }
}
torch::Tensor finish(torch::Tensor x,torch::Tensor dense,torch::Tensor row,
 torch::Tensor loc,torch::Tensor weight,torch::Tensor params,int64_t bits,int64_t stride,int64_t pad,int64_t dil,bool relu) {
 TORCH_CHECK(x.is_cuda()&&dense.is_cuda()&&x.device()==dense.device()&&x.scalar_type()==at::kChar&&dense.scalar_type()==at::kHalf,"invalid input");
 TORCH_CHECK(x.dim()==4&&dense.dim()==4&&x.is_contiguous()&&dense.is_contiguous(at::MemoryFormat::ChannelsLast),"invalid layout");
 int n=x.size(0),c=x.size(1),h=x.size(2),w=x.size(3),k=dense.size(1),p=dense.size(2),q=dense.size(3);
 auto check=[&](torch::Tensor t,at::ScalarType dtype){TORCH_CHECK(t.device()==x.device()&&t.scalar_type()==dtype&&t.is_contiguous(),"invalid parameter");};
 check(row,at::kInt);check(loc,at::kInt);check(weight,at::kHalf);check(params,at::kFloat);
 TORCH_CHECK(n==dense.size(0)&&row.numel()==(k+7)/8+1&&loc.numel()*8==weight.numel()&&params.numel()==2&&(bits==4||bits==8)&&stride>0&&dil>0&&pad>=0,"invalid shapes");
 c10::cuda::CUDAGuard guard(x.device());auto stream=c10::cuda::getCurrentCUDAStream(x.get_device());
 auto out=torch::empty_like(dense);
 branch<<<dim3((k+7)/8,(int64_t(n)*p*q+63)/64),128,0,stream>>>(x.data_ptr<int8_t>(),reinterpret_cast<half*>(dense.data_ptr()),reinterpret_cast<half*>(out.data_ptr()),row.data_ptr<int>(),loc.data_ptr<int>(),reinterpret_cast<half*>(weight.data_ptr()),params.data_ptr<float>(),n,c,h,w,k,p,q,stride,pad,dil,1<<(bits-1),relu,cutlass::FastDivmod(q),cutlass::FastDivmod(p*q));
 C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
'''
DECL='torch::Tensor finish(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,bool);'

def sources():
    return {'branch.cu':CUDA,'bindings.cpp':'#include <torch/extension.h>\n'+DECL+'\nPYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("finish",&finish);}\n'}

@lru_cache(None)
def extension():return build_sources('sparse_tc_coalesced',sources())

@torch.no_grad()
def prepare_metadata(m):
    if hasattr(m,'_tc_coalesced'):return m._tc_coalesced
    row=m.row.cpu().tolist();loc=m.compact_locations.cpu().tolist();val=m.values.cpu().tolist()
    rows=[0];locations=[];weights=[];k=m.spec['shape'][0]
    for start in range(0,k,8):
        union=sorted(set(loc[row[start]:row[min(start+8,k)]]));lookup={v:i for i,v in enumerate(union)}
        padded=(len(union)+15)//16*16;matrix=torch.zeros(padded,8,dtype=torch.float16)
        for oc in range(start,min(start+8,k)):
            for j in range(row[oc],row[oc+1]):matrix[lookup[loc[j]],oc-start]=val[j]
        locations.extend(union+[-1]*(padded-len(union)));weights.append(matrix);rows.append(len(locations))
    m._tc_coalesced=(torch.tensor(rows,dtype=torch.int32,device=m.values.device),torch.tensor(locations,dtype=torch.int32,device=m.values.device),torch.cat(weights).to(m.values.device))
    return m._tc_coalesced

def convolution(m,packed,planar):
    from quantdemoire.branch_diagnostic import dense_core
    spec=m.spec;relu=spec.get('relu',False);spec['relu']=False
    try:dense=dense_core(m,packed,planar,m.compact_locations)
    finally:spec['relu']=relu
    return extension().finish(planar,dense,*prepare_metadata(m),m.activation,spec['bits'],spec['stride'],spec['padding'],spec['dilation'],relu)
