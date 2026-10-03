"""Experimental exact CSR branch on a concurrent CUDA stream, with explicit merge.

The two compute paths share prepared inputs, then join before output rounding.
The temporary FP32 branch output is included in measured time and memory.
No deployment integration or consumer-prefix support is enabled by this module.
"""
from functools import lru_cache
from pathlib import Path
import torch
from kernel.diagnostic import build_sources


def sources():
    original=(Path(__file__).resolve().parents[2]/'kernel/csrc/sparse_tuned.cu').read_text()
    source=original[:original.index('std::vector<torch::Tensor> finish_half_tuned')]
    source=source.replace('const half* dense,half* result,','const half* dense,float* result,')
    start=source.index('     float v=__half2float(dense[dst])+tile[px][ch];')
    end=source.index('     if constexpr(NextBits>0) {',start)
    source=source[:start]+'     result[dst]=tile[px][ch];\n     half rounded=__float2half(0.f);\n'+source[end:]
    source+=r'''
__global__ void merge_kernel(const half* dense,const float* branch,half* out,int64_t count,bool relu) {
 int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
 if(i<count){float value=__half2float(dense[i])+branch[i];out[i]=__float2half_rn(relu?fmaxf(0.f,value):value);}
}
torch::Tensor branch_only(torch::Tensor x,torch::Tensor shape,torch::Tensor row,torch::Tensor index,torch::Tensor offsets,
 torch::Tensor value,torch::Tensor params,int64_t bits,int64_t stride,int64_t pad,int64_t dil,int64_t r) {
 TORCH_CHECK(x.is_cuda()&&shape.is_cuda()&&x.device()==shape.device()&&x.scalar_type()==at::kChar&&shape.scalar_type()==at::kHalf,"invalid input");
 TORCH_CHECK(x.dim()==4&&shape.dim()==4&&x.is_contiguous()&&shape.is_contiguous(at::MemoryFormat::ChannelsLast),"invalid layout");
 int n=x.size(0),c=x.size(1),h=x.size(2),w=x.size(3),k=shape.size(1),p=shape.size(2),q=shape.size(3);
 auto check=[&](torch::Tensor t,at::ScalarType dtype){TORCH_CHECK(t.device()==x.device()&&t.scalar_type()==dtype&&t.is_contiguous(),"invalid parameter");};
 check(row,at::kInt);check(index,at::kInt);check(offsets,at::kLong);check(value,at::kHalf);check(params,at::kFloat);
 TORCH_CHECK(n==shape.size(0)&&row.numel()==k+1&&index.numel()==value.numel()&&offsets.numel()==value.numel()&&params.numel()==2&&(bits==4||bits==8)&&stride>0&&dil>0&&pad>=0&&int64_t(n)*c*h*w<INT_MAX,"invalid geometry");
 c10::cuda::CUDAGuard guard(x.device());auto stream=c10::cuda::getCurrentCUDAStream(x.get_device());
 auto out=torch::empty_like(shape,shape.options().dtype(at::kFloat));
 sparse_tuned_kernel<0,false,2,false,true,2><<<dim3((k+15)/16,(int64_t(n)*p*q+63)/64),512,0,stream>>>(
 x.data_ptr<int8_t>(),nullptr,out.data_ptr<float>(),row.data_ptr<int>(),index.data_ptr<int>(),offsets.data_ptr<int64_t>(),reinterpret_cast<half*>(value.data_ptr()),params.data_ptr<float>(),
 n,c,h,w,k,p,q,stride,pad,dil,r,1<<(bits-1),false,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,0,0,cutlass::FastDivmod(q),cutlass::FastDivmod(p*q));
 C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
torch::Tensor merge(torch::Tensor dense,torch::Tensor branch,bool relu) {
 TORCH_CHECK(dense.is_cuda()&&branch.is_cuda()&&dense.device()==branch.device()&&dense.scalar_type()==at::kHalf&&branch.scalar_type()==at::kFloat&&dense.sizes()==branch.sizes(),"invalid merge");
 TORCH_CHECK(dense.is_contiguous(at::MemoryFormat::ChannelsLast)&&branch.is_contiguous(at::MemoryFormat::ChannelsLast),"invalid merge layout");
 c10::cuda::CUDAGuard guard(dense.device());auto stream=c10::cuda::getCurrentCUDAStream(dense.get_device());auto out=torch::empty_like(dense);
 merge_kernel<<<(out.numel()+255)/256,256,0,stream>>>(reinterpret_cast<half*>(dense.data_ptr()),branch.data_ptr<float>(),reinterpret_cast<half*>(out.data_ptr()),out.numel(),relu);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
'''
    bindings='''#include <torch/extension.h>
torch::Tensor branch_only(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,int64_t);
torch::Tensor merge(torch::Tensor,torch::Tensor,bool);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("branch_only",&branch_only);m.def("merge",&merge);}
'''
    return {'branch.cu':source,'bindings.cpp':bindings}

@lru_cache(None)
def extension():return build_sources('sparse_overlap',sources())

@torch.inference_mode()
def convolution(m,packed,planar,*,parallel=True):
    from quantdemoire.branch_diagnostic import dense_core
    if not hasattr(m,'_sparse_overlap_stream'):m._sparse_overlap_stream=torch.cuda.Stream(device=packed.device)
    op=extension();s=m.spec;relu=s.get('relu',False)
    offsets=m.sparse_linear_offsets(m.compact_locations,*planar.shape[-2:],s['dilation'],packed=False)
    current=torch.cuda.current_stream(packed.device);aux=m._sparse_overlap_stream
    if parallel:aux.wait_stream(current)
    s['relu']=False
    try:dense=dense_core(m,packed,planar,m.compact_locations)
    finally:s['relu']=relu
    def branch():return op.branch_only(planar,dense,m.row,m.compact_locations,offsets,m.values,m.activation,s['bits'],s['stride'],s['padding'],s['dilation'],s['shape'][2])
    if parallel:
        with torch.cuda.stream(aux):value=branch()
        current.wait_stream(aux)
        value.record_stream(current);planar.record_stream(aux)
    else:value=branch()
    return op.merge(dense,value,relu)
