#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
// Corrected dense output is stored in FP16; sparse products accumulate in FP32.
template<int NextBits=0,bool Residual=false>
__global__ void sparse_half_kernel(const int8_t* x,const half* y,half* result,const int* row,
 const int* index,const half* value,const float* sums,const float* bias,const float* params,
 int n,int c,int h,int w,int k,int p,int q,int r,int stride,int pad,int dil,int center,bool relu,bool sparse,
 uint8_t* next_packed=nullptr,int8_t* next_planar=nullptr,const float* next_multiplier=nullptr,
 const float* next_params=nullptr,const int* next_selection=nullptr,int next_cp=0,int next_sc=0,const half* residual=nullptr) {
 __shared__ float tile[64][17];
 int lane=threadIdx.x&31,warp=threadIdx.x>>5;
 int o=int(blockIdx.x)*16+warp;
 int pb=blockIdx.y*64;
 int pos0=pb+lane*2;
 float acc[2]={0.f,0.f};
 float delta=float(center)-params[1],as=params[0];
 int iy0[2],ix0[2],batch[2];bool valid[2],inside[2];
 int row0=pos0/q,col0=pos0%q,batch0=n==1?0:row0/p;row0=n==1?row0:row0%p;
 #pragma unroll
 for(int t=0;t<2;++t) {
   int pos=pos0+t;valid[t]=pos<n*p*q && o<k;
   if(q>=2) {
     int xx=col0+t,yy=row0;
     if(xx>=q){xx-=q;++yy;}
     batch[t]=batch0;if(n>1&&yy>=p){yy-=p;++batch[t];}
     iy0[t]=yy*stride-pad;ix0[t]=xx*stride-pad;
   } else {
     batch[t]=n==1?0:pos/(p*q);iy0[t]=((pos/q)%p)*stride-pad;ix0[t]=(pos%q)*stride-pad;
   }
   inside[t]=iy0[t]>=0&&ix0[t]>=0&&iy0[t]+(r-1)*dil<h&&ix0[t]+(r-1)*dil<w;
 }
 if(o<k) {
   if(sparse) for(int j=row[o];j<row[o+1];++j) {
     int loc=index[j],ic=loc&4095,dx=((loc>>12)&15)*dil,dy=((loc>>16)&15)*dil;
     float val=__half2float(value[j]);
     #pragma unroll
     for(int t=0;t<2;++t) if(valid[t]) {
       int iy=iy0[t]+dy,ix=ix0[t]+dx;
       if(inside[t]||(iy>=0&&iy<h&&ix>=0&&ix<w)) {
         float av=(float(x[((int64_t(batch[t])*c+ic)*h+iy)*w+ix])+delta)*as;
         acc[t]=fmaf(__half2float(__float2half_rn(av)),val,acc[t]);
       }
     }
   }
 }
 #pragma unroll
 for(int t=0;t<2;++t) tile[lane*2+t][warp]=acc[t];
 __syncthreads();
 #pragma unroll
 for(int t=0;t<2;++t) {
   int i=threadIdx.x+t*512,px=i/16,ch=i%16,oc=blockIdx.x*16+ch;
   int pos=pb+px;
   int quantized=0;
   if(pos<int64_t(n)*p*q&&oc<k) {
     int64_t dst=pos*k+oc;
     float v=__half2float(y[dst])+tile[px][ch];
     half rounded=__float2half_rn(relu?fmaxf(0.f,v):v);
     if constexpr(Residual)rounded=__hadd_rn(rounded,residual[dst]);
     result[dst]=rounded;
     if constexpr(NextBits>0) {
       // Preserve the FP16 producer boundary and separate FP32 mul/add.
       float scaled=__fadd_rn(__fmul_rn(__half2float(rounded),next_multiplier[oc]),next_params[1]);
       quantized=__float2int_rn(fminf(float((1<<NextBits)-1),fmaxf(0.f,scaled)))-(1<<(NextBits-1));
     }
   }
   if constexpr(NextBits>0) {
     // Every cell has one owner, so it is safe to reuse the accumulator tile.
     tile[px][ch]=float(quantized);
     if constexpr(NextBits==4) {
       int other=__shfl_xor_sync(0xffffffff,quantized,1);
       if(pos<int64_t(n)*p*q&&oc<k&&(ch&1)==0)
         next_packed[(int64_t(pos)*next_cp+oc)/2]=uint8_t((quantized&15)|((other&15)<<4));
     } else if(pos<int64_t(n)*p*q&&oc<k) next_packed[int64_t(pos)*next_cp+oc]=uint8_t(quantized);
   }
 }
 if constexpr(NextBits>0) {
   __syncthreads();
   int target=o<k?next_selection[o]:-1;
   #pragma unroll
   for(int t=0;t<2;++t) {
     int pos=pb+lane+t*32;
     if(target>=0&&target<next_sc&&pos<int64_t(n)*p*q) {
       int b=n==1?0:pos/(p*q),sp=n==1?pos:pos%(p*q);
       next_planar[(int64_t(b)*next_sc+target)*p*q+sp]=int8_t(tile[lane+t*32][warp]);
     }
   }
 }
}



torch::Tensor finish_half(torch::Tensor x,torch::Tensor y,torch::Tensor row,torch::Tensor index,
 torch::Tensor value,torch::Tensor params,int64_t bits,int64_t r,int64_t stride,int64_t pad,int64_t dil,bool relu) {
 TORCH_CHECK(x.is_cuda()&&y.is_cuda()&&x.scalar_type()==at::kChar&&y.scalar_type()==at::kHalf,"invalid half branch tensors");
 TORCH_CHECK(x.device()==y.device()&&x.is_contiguous()&&y.is_contiguous(at::MemoryFormat::ChannelsLast),"invalid half branch layout");
 int n=x.size(0),c=x.size(1),h=x.size(2),w=x.size(3),k=y.size(1),p=y.size(2),q=y.size(3);
 c10::cuda::CUDAGuard guard(y.device());auto stream=c10::cuda::getCurrentCUDAStream(y.get_device());
 auto result=torch::empty_like(y);
 sparse_half_kernel<0><<<dim3((k+15)/16,(int64_t(n)*p*q+63)/64),512,0,stream>>>(
   x.data_ptr<int8_t>(),reinterpret_cast<half*>(y.data_ptr()),reinterpret_cast<half*>(result.data_ptr()),row.data_ptr<int>(),index.data_ptr<int>(),reinterpret_cast<half*>(value.data_ptr()),
   nullptr,nullptr,params.data_ptr<float>(),n,c,h,w,k,p,q,r,stride,pad,dil,1<<(bits-1),relu,true);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return result;
}

torch::Tensor finish_half_residual(torch::Tensor x,torch::Tensor y,torch::Tensor row,torch::Tensor index,
 torch::Tensor value,torch::Tensor params,int64_t bits,int64_t r,int64_t stride,int64_t pad,int64_t dil,bool relu,torch::Tensor residual) {
 TORCH_CHECK(x.is_cuda()&&y.is_cuda()&&x.device()==y.device()&&x.scalar_type()==at::kChar&&y.scalar_type()==at::kHalf,"invalid half residual tensors");
 TORCH_CHECK(x.is_contiguous()&&y.is_contiguous(at::MemoryFormat::ChannelsLast),"invalid half residual layout");
 TORCH_CHECK(residual.is_cuda()&&residual.device()==y.device()&&residual.scalar_type()==at::kHalf&&residual.sizes()==y.sizes()&&residual.is_contiguous(at::MemoryFormat::ChannelsLast),"residual must match output");
 int n=x.size(0),c=x.size(1),h=x.size(2),w=x.size(3),k=y.size(1),p=y.size(2),q=y.size(3);
 c10::cuda::CUDAGuard guard(y.device());auto stream=c10::cuda::getCurrentCUDAStream(y.get_device());
 auto result=torch::empty_like(y);
 sparse_half_kernel<0,true><<<dim3((k+15)/16,(int64_t(n)*p*q+63)/64),512,0,stream>>>(
   x.data_ptr<int8_t>(),reinterpret_cast<half*>(y.data_ptr()),reinterpret_cast<half*>(result.data_ptr()),row.data_ptr<int>(),index.data_ptr<int>(),reinterpret_cast<half*>(value.data_ptr()),
   nullptr,nullptr,params.data_ptr<float>(),n,c,h,w,k,p,q,r,stride,pad,dil,1<<(bits-1),relu,true,
   nullptr,nullptr,nullptr,nullptr,nullptr,0,0,reinterpret_cast<half*>(residual.data_ptr()));
 C10_CUDA_KERNEL_LAUNCH_CHECK();return result;
}

std::vector<torch::Tensor> finish_half_next(torch::Tensor x,torch::Tensor y,torch::Tensor row,torch::Tensor index,
 torch::Tensor value,torch::Tensor params,int64_t bits,int64_t r,int64_t stride,int64_t pad,int64_t dil,bool relu,
 torch::Tensor multiplier,torch::Tensor next_params,torch::Tensor selection,int64_t next_bits,int64_t next_cp,int64_t next_sc) {
 TORCH_CHECK(x.is_cuda()&&y.is_cuda()&&x.device()==y.device()&&x.scalar_type()==at::kChar&&y.scalar_type()==at::kHalf,
             "invalid next-quantization tensors");
 TORCH_CHECK(x.is_contiguous()&&y.is_contiguous(at::MemoryFormat::ChannelsLast),"invalid next-quantization layout");
 int n=x.size(0),c=x.size(1),h=x.size(2),w=x.size(3),k=y.size(1),p=y.size(2),q=y.size(3);
 TORCH_CHECK((bits==4||bits==8)&&(next_bits==4||next_bits==8)&&k%16==0,"invalid bits or producer channels");
 TORCH_CHECK(multiplier.is_cuda()&&multiplier.device()==y.device()&&multiplier.scalar_type()==at::kFloat&&multiplier.is_contiguous(),"invalid next multiplier");
 TORCH_CHECK(next_params.is_cuda()&&next_params.device()==y.device()&&next_params.scalar_type()==at::kFloat&&next_params.is_contiguous()&&next_params.numel()==2,"invalid next params");
 TORCH_CHECK(selection.is_cuda()&&selection.device()==y.device()&&selection.scalar_type()==at::kInt&&selection.is_contiguous(),"invalid next selection");
 TORCH_CHECK(multiplier.numel()>k&&selection.numel()==multiplier.numel()&&next_cp>=multiplier.numel()&&next_cp%(next_bits==4?32:16)==0&&next_sc>0&&next_sc<=multiplier.numel(),"invalid next quantizer shape");
 c10::cuda::CUDAGuard guard(y.device());auto stream=c10::cuda::getCurrentCUDAStream(y.get_device());
 auto result=torch::empty_like(y);
 auto packed=torch::empty({int64_t(n)*p*q*next_cp*next_bits/8},y.options().dtype(at::kByte));
 auto planar=torch::empty({n,next_sc,p,q},y.options().dtype(at::kChar).memory_format(at::MemoryFormat::Contiguous));
 #define NEXT_ARGS x.data_ptr<int8_t>(),reinterpret_cast<half*>(y.data_ptr()),reinterpret_cast<half*>(result.data_ptr()),row.data_ptr<int>(),index.data_ptr<int>(),reinterpret_cast<half*>(value.data_ptr()),nullptr,nullptr,params.data_ptr<float>(),n,c,h,w,k,p,q,r,stride,pad,dil,1<<(bits-1),relu,true,packed.data_ptr<uint8_t>(),planar.data_ptr<int8_t>(),multiplier.data_ptr<float>(),next_params.data_ptr<float>(),selection.data_ptr<int>(),next_cp,next_sc
 if(next_bits==4) sparse_half_kernel<4><<<dim3((k+15)/16,(int64_t(n)*p*q+63)/64),512,0,stream>>>(NEXT_ARGS);
 else sparse_half_kernel<8><<<dim3((k+15)/16,(int64_t(n)*p*q+63)/64),512,0,stream>>>(NEXT_ARGS);
 #undef NEXT_ARGS
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {result,packed,planar};
}
