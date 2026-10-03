#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Each block of eight output channels stores the union of their exact CSR
// coordinates. Zero-filled FP16 coefficients preserve the original sparsity.
// One warp computes 16 pixels x 8 channels using FP16 MMA / FP32 accumulation.
__device__ __forceinline__ half operand(const int8_t* x,int loc,int64_t pos,
 int n,int c,int h,int w,int p,int q,int stride,int pad,int dil,float delta,float scale) {
 if(loc<0 || pos>=int64_t(n)*p*q)return __float2half(0.f);
 int ic=loc&4095,rx=(loc>>12)&15,ry=(loc>>16)&15;
 int batch=pos/(p*q),iy=((pos/q)%p)*stride-pad+ry*dil,ix=(pos%q)*stride-pad+rx*dil;
 if(iy<0||iy>=h||ix<0||ix>=w)return __float2half(0.f);
 return __float2half_rn((float(x[((int64_t(batch)*c+ic)*h+iy)*w+ix])+delta)*scale);
}
__global__ void sparse_tc_kernel(const int8_t* x,const float* y,half* out,
 const int* row,const int* locs,const half* weights,const float* sums,const float* bias,const float* params,
 int n,int c,int h,int w,int k,int p,int q,int r,int stride,int pad,int dil,int center,bool relu) {
 int lane=threadIdx.x&31,warp=threadIdx.x>>5,group=lane>>2,tid=lane&3;
 int oc0=blockIdx.x*8;
 int64_t pb=int64_t(blockIdx.y)*64+warp*16;
 float d0=0.f,d1=0.f,d2=0.f,d3=0.f;
 float delta=float(center)-params[1],as=params[0];
 for(int start=row[blockIdx.x];start<row[blockIdx.x+1];start+=16) {
   uint32_t aa[4],bb[2];
   #pragma unroll
   for(int z=0;z<4;++z) {
     int rr=group+((z&1)?8:0),kk=tid*2+((z&2)?8:0);
     half av0=operand(x,locs[start+kk],pb+rr,n,c,h,w,p,q,stride,pad,dil,delta,as);
     half av1=operand(x,locs[start+kk+1],pb+rr,n,c,h,w,p,q,stride,pad,dil,delta,as);
     aa[z]=uint32_t(__half_as_ushort(av0)) | (uint32_t(__half_as_ushort(av1))<<16);
   }
   #pragma unroll
   for(int z=0;z<2;++z) {
     int kk=tid*2+z*8;
     bb[z]=uint32_t(__half_as_ushort(weights[(start+kk)*8+group])) | (uint32_t(__half_as_ushort(weights[(start+kk+1)*8+group]))<<16);
   }
   asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
     : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
     : "r"(aa[0]),"r"(aa[1]),"r"(aa[2]),"r"(aa[3]),"r"(bb[0]),"r"(bb[1]));
 }
 float dd[4]={d0,d1,d2,d3};
 #pragma unroll
 for(int z=0;z<4;++z) {
   int64_t pos=pb+group+(z>=2?8:0);int oc=oc0+tid*2+(z&1);
   if(pos<int64_t(n)*p*q && oc<k) {
     float v=dd[z]+bias[oc];
     if(delta!=0.f) {
       int iy0=((pos/q)%p)*stride-pad,ix0=(pos%q)*stride-pad;
       for(int ry=0;ry<r;++ry)for(int rx=0;rx<r;++rx) {
         int iy=iy0+ry*dil,ix=ix0+rx*dil;
         if(iy>=0&&iy<h&&ix>=0&&ix<w)v+=delta*sums[(oc*r+ry)*r+rx];
       }
     }
     v+=y[pos*k+oc];out[pos*k+oc]=__float2half_rn(relu?fmaxf(v,0.f):v);
   }
 }
}

torch::Tensor finish_tc(torch::Tensor x,torch::Tensor y,torch::Tensor row,torch::Tensor locations,
 torch::Tensor weights,torch::Tensor sums,torch::Tensor bias,torch::Tensor params,
 int64_t bits,int64_t r,int64_t stride,int64_t pad,int64_t dil,bool relu) {
 TORCH_CHECK(x.is_cuda()&&y.is_cuda()&&x.device()==y.device()&&x.scalar_type()==at::kChar&&y.scalar_type()==at::kFloat,"invalid tensors");
 TORCH_CHECK(x.is_contiguous()&&y.is_contiguous(at::MemoryFormat::ChannelsLast),"invalid layout");
 int n=x.size(0),c=x.size(1),h=x.size(2),w=x.size(3),k=y.size(1),p=y.size(2),q=y.size(3);
 TORCH_CHECK(row.numel()==(k+7)/8+1&&locations.numel()*8==weights.numel(),"invalid block sparse shape");
 c10::cuda::CUDAGuard guard(y.device());auto stream=c10::cuda::getCurrentCUDAStream(y.get_device());
 auto result=torch::empty_like(y,y.options().dtype(at::kHalf));
 sparse_tc_kernel<<<dim3((k+7)/8,(int64_t(n)*p*q+63)/64),128,0,stream>>>(
 x.data_ptr<int8_t>(),y.data_ptr<float>(),reinterpret_cast<half*>(result.data_ptr()),row.data_ptr<int>(),locations.data_ptr<int>(),reinterpret_cast<half*>(weights.data_ptr()),sums.data_ptr<float>(),bias.data_ptr<float>(),params.data_ptr<float>(),n,c,h,w,k,p,q,r,stride,pad,dil,1<<(bits-1),relu);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return result;
}
