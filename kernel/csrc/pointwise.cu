#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <climits>

// Preserve every half rounding boundary of PyTorch's three products and
// left-associated additions. The _rn intrinsics prohibit FMA contraction.
__global__ void attention_residual_kernel(const half2* full,const half2* mid,
 const half2* low,const half2* weights,const half2* residual,half2* out,
 int count,int channels,int spatial) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=count)return;
 int ch=i%channels,batch=i/(channels*spatial),wi=batch*channels*3+ch;
 half2 a=__hmul2_rn(full[i],weights[wi]);
 half2 b=__hmul2_rn(mid[i],weights[wi+channels]);
 half2 c=__hmul2_rn(low[i],weights[wi+channels*2]);
 out[i]=__hadd2_rn(residual[i],__hadd2_rn(__hadd2_rn(a,b),c));
}

torch::Tensor attention_residual(torch::Tensor full,torch::Tensor mid,torch::Tensor low,
 torch::Tensor weights,torch::Tensor residual) {
 TORCH_CHECK(full.is_cuda()&&full.dim()==4&&full.scalar_type()==at::kHalf,"attention expects CUDA FP16 NCHW tensors");
 for(const auto& x:{full,mid,low,residual}) {
   TORCH_CHECK(x.device()==full.device()&&x.scalar_type()==at::kHalf&&x.sizes()==full.sizes()&&x.is_contiguous(at::MemoryFormat::ChannelsLast),"invalid attention feature shape, device or layout");
 }
 int64_t n=full.size(0),c=full.size(1),spatial=full.size(2)*full.size(3);
 TORCH_CHECK(n>0&&c>0&&c%2==0&&spatial>0&&full.numel()<INT_MAX,"unsupported attention shape");
 TORCH_CHECK(weights.is_cuda()&&weights.device()==full.device()&&weights.scalar_type()==at::kHalf&&weights.is_contiguous()&&weights.dim()==4&&weights.size(0)==n&&weights.size(1)==3*c&&weights.size(2)==1&&weights.size(3)==1,"invalid attention weights");
 c10::cuda::CUDAGuard guard(full.device());auto stream=c10::cuda::getCurrentCUDAStream(full.get_device());
 auto result=torch::empty_like(full);
 attention_residual_kernel<<<(full.numel()/2+255)/256,256,0,stream>>>(
   reinterpret_cast<half2*>(full.data_ptr()),reinterpret_cast<half2*>(mid.data_ptr()),reinterpret_cast<half2*>(low.data_ptr()),
   reinterpret_cast<half2*>(weights.data_ptr()),reinterpret_cast<half2*>(residual.data_ptr()),reinterpret_cast<half2*>(result.data_ptr()),full.numel()/2,c/2,spatial);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return result;
}
