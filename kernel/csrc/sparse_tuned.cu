// Experimental CSR variants: coalesced pixels, linear interior offsets,
// optional shared dequantization table, and two/four pixels per lane.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <climits>
#include <type_traits>
#include "cutlass/fast_math.h"

template<bool Lookup>
__device__ __forceinline__ float sparse_operand(int code,const float* table,int center,float delta,float scale) {
 if constexpr(Lookup)return table[code+center];
 else return __half2float(__float2half_rn((float(code)+delta)*scale));
}

template<int NextBits,bool Residual,int T,bool Lookup,bool NextPlanar=true,int Variant=0>
__global__ void sparse_tuned_kernel(const int8_t* x,const half* dense,half* result,
 const int* row,const int* index,const int64_t* offsets,const half* value,const float* params,
 int n,int c,int h,int w,int k,int p,int q,int stride,int pad,int dil,int r,int center,bool relu,
 const half* residual,uint8_t* packed,int8_t* planar,const float* multiplier,
 const float* next_params,const int* selection,int cp,int sc,cutlass::FastDivmod q_div,cutlass::FastDivmod pq_div) {
 constexpr int Pixels=32*T;
 __shared__ float tile[Pixels][17];
 __shared__ float table[Lookup?256:1];
 int lane=threadIdx.x&31,warp=threadIdx.x>>5,o=blockIdx.x*16+warp,pb=blockIdx.y*Pixels;
 float delta=float(center)-params[1],scale=params[0];
 if constexpr(Lookup) {
   if(threadIdx.x<2*center)table[threadIdx.x]=__half2float(__float2half_rn((float(int(threadIdx.x)-center)+delta)*scale));
   __syncthreads();
 }
 using Index=typename std::conditional<(Variant>=1),int,int64_t>::type;
 float acc[T];int iy0[T],ix0[T],batch[T];Index base[T];bool valid[T],interior=true;
 #pragma unroll
 for(int t=0;t<T;++t) {
   int pos=pb+lane+t*32;
   valid[t]=pos<int64_t(n)*p*q&&o<k;acc[t]=0.f;
   if constexpr(Variant>=2) {
     int local=pos,y,x0;batch[t]=0;
     if(n!=1)pq_div(batch[t],local,pos);
     q_div(y,x0,local);iy0[t]=y*stride-pad;ix0[t]=x0*stride-pad;
   } else {
     batch[t]=n==1?0:pos/(p*q);
     iy0[t]=(n==1?pos/q:(pos/q)%p)*stride-pad;ix0[t]=(pos%q)*stride-pad;
   }
   base[t]=Index(batch[t])*c*h*w+Index(iy0[t])*w+ix0[t];
   interior=interior&&valid[t]&&iy0[t]>=0&&ix0[t]>=0&&iy0[t]+(r-1)*dil<h&&ix0[t]+(r-1)*dil<w;
 }
 if(o<k) {
   int begin=row[o],end=row[o+1];
   if(interior) {
     for(int j=begin;j<end;++j) {
       Index off=Index(offsets[j]);float weight=__half2float(value[j]);
       #pragma unroll
       for(int t=0;t<T;++t)acc[t]=fmaf(sparse_operand<Lookup>(int(x[base[t]+off]),table,center,delta,scale),weight,acc[t]);
     }
   } else {
     for(int j=begin;j<end;++j) {
       int loc=index[j],ic=loc&4095,dx=((loc>>12)&15)*dil,dy=((loc>>16)&15)*dil;
       float weight=__half2float(value[j]);
       #pragma unroll
       for(int t=0;t<T;++t)if(valid[t]) {
         int iy=iy0[t]+dy,ix=ix0[t]+dx;
         if(iy>=0&&iy<h&&ix>=0&&ix<w)
           acc[t]=fmaf(sparse_operand<Lookup>(int(x[((Index(batch[t])*c+ic)*h+iy)*w+ix]),table,center,delta,scale),weight,acc[t]);
       }
     }
   }
 }
 #pragma unroll
 for(int t=0;t<T;++t)tile[lane+t*32][warp]=acc[t];
 __syncthreads();
 #pragma unroll
 for(int t=0;t<T;++t) {
   int i=threadIdx.x+t*512,px=i/16,ch=i%16,oc=blockIdx.x*16+ch,pos=pb+px,code=0;
   if(pos<int64_t(n)*p*q&&oc<k) {
     int64_t dst=int64_t(pos)*k+oc;
     float v=__half2float(dense[dst])+tile[px][ch];
     half rounded=__float2half_rn(relu?fmaxf(v,0.f):v);
     if constexpr(Residual)rounded=__hadd_rn(rounded,residual[dst]);
     result[dst]=rounded;
     if constexpr(NextBits>0) {
       float scaled=__fadd_rn(__fmul_rn(__half2float(rounded),multiplier[oc]),next_params[1]);
       code=__float2int_rn(fminf(float((1<<NextBits)-1),fmaxf(0.f,scaled)))-(1<<(NextBits-1));
     }
   }
   if constexpr(NextBits>0) {
     if constexpr(NextPlanar)tile[px][ch]=float(code);
     if constexpr(NextBits==4) {
       int other=__shfl_xor_sync(0xffffffff,code,1);
       if(pos<int64_t(n)*p*q&&oc<k&&(ch&1)==0)packed[(int64_t(pos)*cp+oc)/2]=uint8_t((code&15)|((other&15)<<4));
     } else if(pos<int64_t(n)*p*q&&oc<k)packed[int64_t(pos)*cp+oc]=uint8_t(code);
   }
 }
 if constexpr(NextBits>0&&NextPlanar) {
   __syncthreads();int target=o<k?selection[o]:-1;
   #pragma unroll
   for(int t=0;t<T;++t) {
     int pos=pb+lane+t*32;
     if(target>=0&&target<sc&&pos<int64_t(n)*p*q) {
       int b=0,sp=pos;
       if constexpr(Variant>=2) {if(n!=1)pq_div(b,sp,pos);}
       else {b=n==1?0:pos/(p*q);sp=n==1?pos:pos%(p*q);}
       planar[(int64_t(b)*sc+target)*p*q+sp]=int8_t(tile[lane+t*32][warp]);
     }
   }
 }
}

std::vector<torch::Tensor> finish_half_tuned(torch::Tensor x,torch::Tensor y,torch::Tensor row,
 torch::Tensor index,torch::Tensor offsets,torch::Tensor value,torch::Tensor params,
 int64_t bits,int64_t r,int64_t stride,int64_t pad,int64_t dil,bool relu,
 torch::Tensor residual,bool add_residual,torch::Tensor multiplier,torch::Tensor next_params,
 torch::Tensor selection,int64_t next_bits,int64_t cp,int64_t sc,int64_t mode,bool next_planar) {
 TORCH_CHECK(x.is_cuda()&&y.is_cuda()&&x.device()==y.device()&&x.dim()==4&&y.dim()==4&&x.scalar_type()==at::kChar&&y.scalar_type()==at::kHalf,"invalid tuned CSR input");
 TORCH_CHECK(x.is_contiguous()&&y.is_contiguous(at::MemoryFormat::ChannelsLast),"invalid tuned CSR layout");
 auto check=[&](const torch::Tensor& t,at::ScalarType dtype) {TORCH_CHECK(t.is_cuda()&&t.device()==y.device()&&t.scalar_type()==dtype&&t.is_contiguous(),"invalid tuned CSR parameter");};
 check(row,at::kInt);check(index,at::kInt);check(offsets,at::kLong);check(value,at::kHalf);check(params,at::kFloat);
 int n=x.size(0),c=x.size(1),h=x.size(2),w=x.size(3),k=y.size(1),p=y.size(2),q=y.size(3);
 TORCH_CHECK(n==y.size(0)&&row.numel()==k+1&&index.numel()==value.numel()&&offsets.numel()==value.numel()&&params.numel()==2,"invalid tuned CSR shapes");
 TORCH_CHECK((bits==4||bits==8)&&(next_bits==0||next_bits==4||next_bits==8)&&mode>=1&&mode<=5,"invalid tuned CSR mode");
 TORCH_CHECK(int64_t(n)*p*q<INT_MAX-128&&r>0&&dil>0&&stride>0&&pad>=0,"invalid tuned CSR geometry");
 if(add_residual)TORCH_CHECK(!next_bits&&residual.is_cuda()&&residual.device()==y.device()&&residual.scalar_type()==at::kHalf&&residual.sizes()==y.sizes()&&residual.is_contiguous(at::MemoryFormat::ChannelsLast),"invalid tuned residual");
 if(next_bits) {
   check(multiplier,at::kFloat);check(next_params,at::kFloat);check(selection,at::kInt);
   TORCH_CHECK(k%16==0&&multiplier.numel()>k&&selection.numel()==multiplier.numel()&&next_params.numel()==2&&cp>=multiplier.numel()&&cp%(next_bits==4?32:16)==0&&sc>0&&sc<=multiplier.numel(),"invalid tuned next quantizer");
 }
 c10::cuda::CUDAGuard guard(y.device());auto stream=c10::cuda::getCurrentCUDAStream(y.get_device());
 auto result=torch::empty_like(y);torch::Tensor packed,planar;
 if(next_bits) {
   packed=torch::empty({int64_t(n)*p*q*cp*next_bits/8},y.options().dtype(at::kByte));
   planar=next_planar?torch::empty({n,sc,p,q},y.options().dtype(at::kChar).memory_format(at::MemoryFormat::Contiguous)):
     torch::empty({1},y.options().dtype(at::kChar)).as_strided({n,sc,p,q},{0,0,0,0});
 }
 cutlass::FastDivmod q_div(q),pq_div(p*q);
 #define ARGS x.data_ptr<int8_t>(),reinterpret_cast<half*>(y.data_ptr()),reinterpret_cast<half*>(result.data_ptr()),row.data_ptr<int>(),index.data_ptr<int>(),offsets.data_ptr<int64_t>(),reinterpret_cast<half*>(value.data_ptr()),params.data_ptr<float>(),n,c,h,w,k,p,q,stride,pad,dil,r,1<<(bits-1),relu,add_residual?reinterpret_cast<half*>(residual.data_ptr()):nullptr,next_bits?packed.data_ptr<uint8_t>():nullptr,next_bits?planar.data_ptr<int8_t>():nullptr,next_bits?multiplier.data_ptr<float>():nullptr,next_bits?next_params.data_ptr<float>():nullptr,next_bits?selection.data_ptr<int>():nullptr,cp,sc,q_div,pq_div
 #define RUN(B,R,T,L,P,V) sparse_tuned_kernel<B,R,T,L,P,V><<<dim3((k+15)/16,(int64_t(n)*p*q+32*T-1)/(32*T)),512,0,stream>>>(ARGS)
 #define NEXT(B,T,L,V) if(next_planar){RUN(B,false,T,L,true,V);}else{RUN(B,false,T,L,false,V);}
 #define CHOOSE(T,L,V) if(next_bits==4){NEXT(4,T,L,V);}else if(next_bits==8){NEXT(8,T,L,V);}else if(add_residual){RUN(0,true,T,L,true,V);}else{RUN(0,false,T,L,true,V);}
 if(mode==4&&int64_t(n)*c*h*w<INT_MAX){CHOOSE(2,false,1);}else if(mode==5&&int64_t(n)*c*h*w<INT_MAX){CHOOSE(2,false,2);}else if(mode==1||mode>=4){CHOOSE(2,false,0);}else if(mode==2){CHOOSE(2,true,0);}else{CHOOSE(4,true,0);}
 #undef ARGS
 #undef RUN
 #undef CHOOSE
 #undef NEXT
 C10_CUDA_KERNEL_LAUNCH_CHECK();
 if(next_bits)return {result,packed,planar};return {result};
}
