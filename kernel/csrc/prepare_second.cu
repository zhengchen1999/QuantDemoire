// Complete a consumer input using its prepared first prefix and second feature cache.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <climits>
namespace {
struct Segments { const half* pointer[8]; int channels[8]; int ends[8]; int count; };
// Four channels per lane: 64-bit feature loads and four int8 codes per shared word.
template<int Bits,bool Sparse,int Pixels,bool Partial=false,bool Unroll=true,int Channels=128>
__global__ void prepare_cached_second_kernel(Segments segments,const float* multiplier,const float* params,
 uint8_t* packed,int8_t* planar,int n,int h,int w,int c,int cp,const int* selection,int sc,int skip,const uint8_t* second,int second_channels) {
 constexpr int Lanes=Channels/4,Rows=256/Lanes;
 __shared__ unsigned tile[Pixels][Channels==64?16:33];
 int lane=threadIdx.x,group=threadIdx.y,cb=blockIdx.x*Channels+(Partial?skip:0);
 int pb=(blockIdx.z*gridDim.y+blockIdx.y)*Pixels,ch=cb+lane*4,seg=0;
 while(seg+1<segments.count&&ch>=segments.ends[seg])++seg;
 int offset=seg?segments.ends[seg-1]:0,local_ch=ch-offset;
 const half* x=segments.pointer[seg];int input_c=segments.channels[seg];
 constexpr int Center=1<<(Bits-1),Max=(1<<Bits)-1;
 float m[4],z=params[1];
 #pragma unroll
 for(int j=0;j<4;++j)m[j]=ch+j<c?multiplier[ch+j]:0.f;
 #pragma unroll (Unroll ? Pixels/Rows : 1)
 for(int t=0;t<Pixels/Rows;++t) {
   int pp=group+t*Rows,pix=pb+pp;unsigned codes=0,nibbles=0;
   if(pix<int64_t(n)*h*w&&ch<c) {
     if(ch>=skip&&ch<skip+second_channels) {
       int64_t index=int64_t(pix)*second_channels+ch-skip;
       if constexpr(Bits==4) {
         nibbles=*reinterpret_cast<const uint16_t*>(second+index/2);
         if constexpr(Sparse) {
           #pragma unroll
           for(int j=0;j<4;++j) {
             int code=(int((nibbles>>(4*j))&15)^8)-8;
             codes|=unsigned(code&255)<<(8*j);
           }
         }
       } else codes=*reinterpret_cast<const uint32_t*>(second+index);
     } else {
     uint2 raw=*reinterpret_cast<const uint2*>(x+int64_t(pix)*input_c+local_ch);
     float v[4]={__half2float(__ushort_as_half(raw.x&65535)),__half2float(__ushort_as_half(raw.x>>16)),
                 __half2float(__ushort_as_half(raw.y&65535)),__half2float(__ushort_as_half(raw.y>>16))};
     #pragma unroll
     for(int j=0;j<4;++j) {
       int code=__float2int_rn(fminf(float(Max),fmaxf(0.f,__fadd_rn(__fmul_rn(v[j],m[j]),z))))-Center;
       codes|=unsigned(code&255)<<(j*8);nibbles|=unsigned(code&15)<<(j*4);
     }
   }
   }
   if constexpr(Sparse)tile[pp][lane ^ (Channels==64?((pp>>1)&15):0)]=codes;
   if(ch<cp&&pix<int64_t(n)*h*w) {
     if constexpr(Bits==4)*reinterpret_cast<uint16_t*>(packed+(int64_t(pix)*cp+ch)/2)=uint16_t(nibbles);
     else *reinterpret_cast<uint32_t*>(packed+int64_t(pix)*cp+ch)=codes;
   }
 }
 if constexpr(Sparse) {
   __syncthreads();
   #pragma unroll (Unroll ? Pixels*Channels/256 : 1)
   for(int t=0;t<Pixels*Channels/256;++t) {
     int linear=int(threadIdx.y)*Lanes+lane;
     int cc=linear/32+(t%(Channels/8))*8,ic=cb+cc,pp=linear%32+(t/(Channels/8))*32,pix=pb+pp;
     if(ic<c&&pix<int64_t(n)*h*w) {
       int target=selection?selection[ic]:ic;
       if(target>=0&&target<sc) {
         int batch=n==1?0:pix/(h*w),sp=n==1?pix:pix%(h*w);
         planar[(int64_t(batch)*sc+target)*h*w+sp]=int8_t(tile[pp][(cc/4) ^ (Channels==64?((pp>>1)&15):0)]>>((cc&3)*8));
       }
     }
   }
 }
}

}
std::vector<torch::Tensor> prepare_cached_second(std::vector<torch::Tensor> inputs,
 torch::Tensor multiplier,torch::Tensor params,int64_t bits,int64_t cp,
 torch::Tensor selection,int64_t sc,torch::Tensor packed,torch::Tensor planar,
 int64_t skip,bool sparse,torch::Tensor second) {
 TORCH_CHECK(inputs.size()>=2&&inputs.size()<=8,"expected 2..8 segments");
 auto x=inputs[0];TORCH_CHECK(x.is_cuda()&&x.dim()==4&&x.scalar_type()==at::kHalf,"invalid first input");
 int n=x.size(0),h=x.size(2),w=x.size(3),c=0;
 auto feature_metadata=[](const torch::Tensor& t) {
   return t.dim()==4&&t.stride(0)==0&&t.stride(1)==0&&t.stride(2)==0&&t.stride(3)==0&&t.storage().nbytes()==sizeof(half);
 };
 // The first segment is skipped and the complete second segment is read
 // from its packed cache. Only these two pointers may be metadata placeholders.
 Segments segments{};segments.count=inputs.size();
 for(int i=0;i<inputs.size();++i) {
   auto t=inputs[i];TORCH_CHECK(t.device()==x.device()&&t.scalar_type()==at::kHalf&&t.dim()==4&&(t.is_contiguous(at::MemoryFormat::ChannelsLast)||(i<2&&feature_metadata(t))),"invalid segment");
   TORCH_CHECK(t.size(0)==n&&t.size(2)==h&&t.size(3)==w&&t.size(1)>0&&t.size(1)%4==0,"unaligned/mismatched segment");
   segments.pointer[i]=reinterpret_cast<const half*>(t.data_ptr());
   segments.channels[i]=t.size(1);c+=t.size(1);segments.ends[i]=c;
 }
 auto check=[&](torch::Tensor t,at::ScalarType type){TORCH_CHECK(t.device()==x.device()&&t.scalar_type()==type&&t.is_contiguous(),"invalid metadata");};
 check(multiplier,at::kFloat);check(params,at::kFloat);check(selection,at::kInt);check(packed,at::kByte);check(second,at::kByte);
 TORCH_CHECK((bits==4||bits==8)&&cp>=c&&cp%(bits==4?32:16)==0,"invalid packing geometry");
 TORCH_CHECK(multiplier.numel()==c&&params.numel()==2&&selection.numel()==c&&sc>0&&sc<=c,"invalid quantizer");
 TORCH_CHECK(skip==inputs[0].size(1)&&skip%16==0&&skip<c,"invalid first prefix");
 int second_channels=inputs[1].size(1);
 TORCH_CHECK(second_channels%(bits==4?32:16)==0&&second.numel()==int64_t(n)*h*w*second_channels*bits/8,"invalid second feature cache");
 TORCH_CHECK(packed.numel()==int64_t(n)*h*w*cp*bits/8&&planar.device()==x.device()&&planar.scalar_type()==at::kChar&&planar.dim()==4&&planar.size(0)==n&&planar.size(1)==sc&&planar.size(2)==h&&planar.size(3)==w,"invalid output buffers");
 TORCH_CHECK(!sparse||planar.is_contiguous(),"invalid planar output layout");
 TORCH_CHECK(n>0&&h>0&&w>0&&int64_t(n)*h*w<INT_MAX-128,"invalid spatial extent");
 c10::cuda::CUDAGuard guard(x.device());auto stream=c10::cuda::getCurrentCUDAStream(x.get_device());
 int pixels=n*h*w>128?128:32,tiles=(n*h*w+pixels-1)/pixels;
 dim3 blocks((cp-skip+63)/64,std::min(tiles,65535),(tiles+65534)/65535);
 #define ARGS segments,multiplier.data_ptr<float>(),params.data_ptr<float>(),packed.data_ptr<uint8_t>(),planar.data_ptr<int8_t>(),n,h,w,c,cp,selection.data_ptr<int>(),int(sc),int(skip),second.data_ptr<uint8_t>(),second_channels
 #define RUN(B,S,P) prepare_cached_second_kernel<B,S,P,true,false,64><<<blocks,dim3(16,16),0,stream>>>(ARGS)
 #define CHOOSE(P) if(bits==4){if(sparse){RUN(4,true,P);}else{RUN(4,false,P);}}else{if(sparse){RUN(8,true,P);}else{RUN(8,false,P);}}
 if(pixels==128){CHOOSE(128);}else{CHOOSE(32);}
 #undef CHOOSE
 #undef RUN
 #undef ARGS
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {packed,planar};
}
