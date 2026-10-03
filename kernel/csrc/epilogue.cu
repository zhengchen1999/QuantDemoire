#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>
#include <climits>


void check_tensor(const torch::Tensor& t, const torch::Tensor& ref, at::ScalarType dtype) {
  TORCH_CHECK(t.is_cuda() && t.device()==ref.device() && t.scalar_type()==dtype && t.is_contiguous(),
              "invalid tensor device, dtype or layout");
}

// One thread owns one packed byte. No write races for the two INT4 nibbles.
// Fold smoothing into quantization; also produce a spatially contiguous INT8
// view for coalesced sparse gathers. Zero padding in channels represents q=0.
struct Segments { const half* pointer[8]; int channels[8]; int ends[8]; int count; };
template <int Bits, bool Sparse, int Pixels, bool Compact=false, bool Cached=false>
__global__ void prepare_kernel(Segments segments,const float* inv_s,const float* params,
 uint8_t* packed,int8_t* planar,int n,int h,int w,int c,int cp,const int* selection,int sc) {
 __shared__ int tile[Pixels][33];
 int lane=threadIdx.x,group=threadIdx.y;
 int cb=blockIdx.x*32;
 int pb=(blockIdx.z*gridDim.y+blockIdx.y)*Pixels;
 int ch=cb+lane;
 int seg=0;
 while(seg+1<segments.count && ch>=segments.ends[seg]) ++seg;
 int offset=seg?segments.ends[seg-1]:0;
 int local_ch=ch-offset;
 const half* x=segments.pointer[seg];int input_c=segments.channels[seg];
 constexpr int Center=1<<(Bits-1),Max=(1<<Bits)-1;
 float inv=ch<c?inv_s[ch]:0.f;
 float a=params[0],z=params[1],multiplier=Cached?inv:inv/a;
 #pragma unroll
 for(int t=0;t<Pixels/8;++t) {
   int pp=group+t*8;int pix=pb+pp;
   int v=0;
   if(pix<int64_t(n)*h*w && ch<c) {
     float scaled=__fmul_rn(__half2float(x[int64_t(pix)*input_c+local_ch]),multiplier);
     v=__float2int_rn(fminf(float(Max),fmaxf(0.f,__fadd_rn(scaled,z))))-Center;
   }
   if constexpr(Sparse) tile[pp][lane]=v;
   if constexpr(Bits==4) {
     int other=__shfl_xor_sync(0xffffffff,v,1);
     if((lane&1)==0 && ch<cp && pix<int64_t(n)*h*w)
       packed[(int64_t(pix)*cp+ch)/2]=uint8_t((v&15)|((other&15)<<4));
   } else {
     if(ch<cp && pix<int64_t(n)*h*w) packed[int64_t(pix)*cp+ch]=uint8_t(v);
   }
 }
 if constexpr(Sparse) {
 __syncthreads();
 #pragma unroll
 for(int t=0;t<Pixels/8;++t) {
   int cc=group+(t%4)*8,ic=cb+cc,pp=lane+(t/4)*32;int pix=pb+pp;
   if(ic<c && pix<int64_t(n)*h*w) {
     int batch=n==1?0:pix/(h*w),sp=n==1?pix:pix%(h*w);
     int target=Compact?selection[ic]:ic;
     if(target>=0&&target<sc)planar[(int64_t(batch)*sc+target)*h*w+sp]=int8_t(tile[pp][cc]);
   }
 }
 }
}

// Two FP16 channels per lane. Store both quantized bytes in one shared-memory
// word, keeping the transpose conflict-free while filling INT4 store sectors.
template<int Bits,bool Sparse,int Pixels,bool Partial=false>
__global__ void prepare_pair_kernel(Segments segments,const float* multiplier,const float* params,
 uint8_t* packed,int8_t* planar,int n,int h,int w,int c,int cp,const int* selection,int sc,int skip=0) {
 __shared__ unsigned tile[Pixels][33];
 int lane=threadIdx.x,group=threadIdx.y,cb=blockIdx.x*64+(Partial?skip:0);
 int pb=(blockIdx.z*gridDim.y+blockIdx.y)*Pixels;
 int ch=cb+lane*2,seg=0;
 while(seg+1<segments.count&&ch>=segments.ends[seg])++seg;
 int offset=seg?segments.ends[seg-1]:0,local_ch=ch-offset;
 const half* x=segments.pointer[seg];int input_c=segments.channels[seg];
 constexpr int Center=1<<(Bits-1),Max=(1<<Bits)-1;
 float m0=ch<c?multiplier[ch]:0.f,m1=ch+1<c?multiplier[ch+1]:0.f,z=params[1];
 #pragma unroll
 for(int t=0;t<Pixels/8;++t){
   int pp=group+t*8,pix=pb+pp,v0=0,v1=0;
   if(pix<int64_t(n)*h*w&&ch<c){
     float2 v=__half22float2(*reinterpret_cast<const half2*>(x+int64_t(pix)*input_c+local_ch));
     v0=__float2int_rn(fminf(float(Max),fmaxf(0.f,__fadd_rn(__fmul_rn(v.x,m0),z))))-Center;
     v1=__float2int_rn(fminf(float(Max),fmaxf(0.f,__fadd_rn(__fmul_rn(v.y,m1),z))))-Center;
   }
   if constexpr(Sparse)tile[pp][lane]=unsigned(v0&255)|(unsigned(v1&255)<<8);
   if(ch<cp&&pix<int64_t(n)*h*w){
     if constexpr(Bits==4)packed[(int64_t(pix)*cp+ch)/2]=uint8_t((v0&15)|((v1&15)<<4));
     else *reinterpret_cast<uint16_t*>(packed+int64_t(pix)*cp+ch)=uint16_t((v0&255)|((v1&255)<<8));
   }
 }
 if constexpr(Sparse){
   __syncthreads();
   #pragma unroll
   for(int t=0;t<Pixels/4;++t){
     int cc=group+(t%8)*8,ic=cb+cc,pp=lane+(t/8)*32,pix=pb+pp;
     if(ic<c&&pix<int64_t(n)*h*w){
       int target=selection?selection[ic]:ic;
       if(target>=0&&target<sc){
         int batch=n==1?0:pix/(h*w),sp=n==1?pix:pix%(h*w);
         planar[(int64_t(batch)*sc+target)*h*w+sp]=int8_t(tile[pp][cc/2]>>((cc&1)*8));
       }
     }
   }
 }
}

// Four channels per lane: 64-bit feature loads and four int8 codes per shared word.
template<int Bits,bool Sparse,int Pixels,bool Partial=false,bool Unroll=true,int Channels=128>
__global__ void prepare_quad_kernel(Segments segments,const float* multiplier,const float* params,
 uint8_t* packed,int8_t* planar,int n,int h,int w,int c,int cp,const int* selection,int sc,int skip=0) {
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
     uint2 raw=*reinterpret_cast<const uint2*>(x+int64_t(pix)*input_c+local_ch);
     float v[4]={__half2float(__ushort_as_half(raw.x&65535)),__half2float(__ushort_as_half(raw.x>>16)),
                 __half2float(__ushort_as_half(raw.y&65535)),__half2float(__ushort_as_half(raw.y>>16))};
     #pragma unroll
     for(int j=0;j<4;++j) {
       int code=__float2int_rn(fminf(float(Max),fmaxf(0.f,__fadd_rn(__fmul_rn(v[j],m[j]),z))))-Center;
       codes|=unsigned(code&255)<<(j*8);nibbles|=unsigned(code&15)<<(j*4);
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

std::vector<torch::Tensor> prepare_cat_impl(std::vector<torch::Tensor> inputs,torch::Tensor inv_s,torch::Tensor params,int64_t bits,int64_t cp,bool sparse,torch::Tensor selection,int sc,int mode=0,torch::Tensor packed=torch::Tensor(),torch::Tensor planar=torch::Tensor(),int skip=0) {
 TORCH_CHECK(inputs.size()>0&&inputs.size()<=8,"expected 1..8 concat inputs");
 auto x=inputs[0];
 auto feature_metadata=[](const torch::Tensor& t) {
   return t.dim()==4&&t.stride(0)==0&&t.stride(1)==0&&t.stride(2)==0&&t.stride(3)==0&&t.storage().nbytes()==sizeof(half);
 };
 // Partial quad preparation starts at skip==inputs[0].C, so it never
 // dereferences this complete first segment. Uncached operands stay contiguous.
 bool omitted_first=skip>0&&mode==5&&feature_metadata(x);
 TORCH_CHECK(bits==4||bits==8,"bits must be 4 or 8");
 TORCH_CHECK(x.is_cuda() && x.scalar_type()==at::kHalf && x.dim()==4 && (x.is_contiguous(at::MemoryFormat::ChannelsLast)||omitted_first),"input must be channels-last CUDA FP16");
 check_tensor(inv_s,x,at::kFloat);check_tensor(params,x,at::kFloat);
 int n=x.size(0),c=0,h=x.size(2),w=x.size(3);
 Segments segments{};segments.count=inputs.size();bool pair_aligned=true,quad_aligned=true;
 for(int i=0;i<inputs.size();++i) {
   auto part=inputs[i];
   TORCH_CHECK(part.is_cuda()&&part.device()==x.device()&&part.scalar_type()==at::kHalf&&part.dim()==4&&(part.is_contiguous(at::MemoryFormat::ChannelsLast)||(i==0&&omitted_first)),"invalid concat operand");
   TORCH_CHECK(part.size(0)==n&&part.size(2)==h&&part.size(3)==w,"concat shapes differ");
   segments.pointer[i]=reinterpret_cast<const half*>(part.data_ptr());
   segments.channels[i]=part.size(1);c+=part.size(1);segments.ends[i]=c;
   pair_aligned=pair_aligned&&(part.size(1)%2==0);
   quad_aligned=quad_aligned&&(part.size(1)%4==0);
 }
 TORCH_CHECK(cp>=c && cp%(bits==4?32:16)==0 && inv_s.numel()==c && params.numel()==2,"invalid quantizer shape");
 TORCH_CHECK(!omitted_first||quad_aligned,"omitted feature requires aligned quad segments");
 bool compact=selection.defined()&&selection.numel()>0;
 if(compact){check_tensor(selection,x,at::kInt);TORCH_CHECK(selection.numel()==c&&sc>0&&sc<=c,"invalid compact sparse channels");}
 else sc=c;
 c10::cuda::CUDAGuard guard(x.device());
 if(skip) {
   TORCH_CHECK((mode>=2&&mode<=5)&&pair_aligned&&compact&&skip>0&&skip<c&&skip%16==0&&skip==inputs[0].size(1),"invalid prepared prefix");
   check_tensor(packed,x,at::kByte);
   if(sparse)check_tensor(planar,x,at::kChar);
   else TORCH_CHECK(planar.is_cuda()&&planar.device()==x.device()&&planar.scalar_type()==at::kChar,"invalid unused prepared planar metadata");
   TORCH_CHECK(packed.numel()==int64_t(n)*h*w*cp*bits/8&&planar.dim()==4&&planar.size(0)==n&&planar.size(1)==sc&&planar.size(2)==h&&planar.size(3)==w,"invalid prepared output shapes");
 } else {
   packed=torch::empty({int64_t(n)*h*w*cp*bits/8},x.options().dtype(at::kByte));
   planar=sparse?torch::empty({n,sc,h,w},x.options().dtype(at::kChar).memory_format(at::MemoryFormat::Contiguous)):
     torch::empty({1},x.options().dtype(at::kChar)).as_strided({n,sc,h,w},{0,0,0,0});
 }
 auto stream=c10::cuda::getCurrentCUDAStream(x.get_device());
 TORCH_CHECK(int64_t(n)*h*w < INT_MAX-128,"too many spatial positions");
 bool quad=mode>=3&&quad_aligned,pair=mode>=2&&pair_aligned;
 int pixels=n*h*w>128?(quad&&mode!=5?64:128):32;int tile_count=(n*h*w+pixels-1)/pixels;
 int channel_tile=quad&&mode!=5?128:(pair?64:32);
 dim3 blocks((cp-skip+channel_tile-1)/channel_tile,std::min(tile_count,65535),(tile_count+65534)/65535);
 #define ARGS segments,inv_s.data_ptr<float>(),params.data_ptr<float>(),packed.data_ptr<uint8_t>(),planar.data_ptr<int8_t>(),n,h,w,c,cp,compact?selection.data_ptr<int>():nullptr,sc
 #define PREP(B,S,P,C,F) prepare_kernel<B,S,P,C,F><<<blocks,dim3(32,8),0,stream>>>(ARGS)
 #define CHOOSE(P,F) if(sparse&&compact){if(bits==4)PREP(4,true,P,true,F);else PREP(8,true,P,true,F);}else if(sparse) { if(bits==4) PREP(4,true,P,false,F);else PREP(8,true,P,false,F); } else { if(bits==4) PREP(4,false,P,false,F);else PREP(8,false,P,false,F); }
 #define PAIR(B,S,P) prepare_pair_kernel<B,S,P><<<blocks,dim3(32,8),0,stream>>>(ARGS)
 #define CHOOSE_PAIR(P) if(sparse){if(bits==4)PAIR(4,true,P);else PAIR(8,true,P);}else{if(bits==4)PAIR(4,false,P);else PAIR(8,false,P);}
 #define PART(B,P,S) prepare_pair_kernel<B,S,P,true><<<blocks,dim3(32,8),0,stream>>>(ARGS,skip)
 #define QUAD(B,S,P,Q) do{if(mode==5){prepare_quad_kernel<B,S,P,Q,false,64><<<blocks,dim3(16,16),0,stream>>>(ARGS,skip);}else if(mode==4){prepare_quad_kernel<B,S,P,Q,false><<<blocks,dim3(32,8),0,stream>>>(ARGS,skip);}else{prepare_quad_kernel<B,S,P,Q,true><<<blocks,dim3(32,8),0,stream>>>(ARGS,skip);}}while(0)
 #define CHOOSE_QUAD(P) if(skip){if(sparse){if(bits==4)QUAD(4,true,P,true);else QUAD(8,true,P,true);}else{if(bits==4)QUAD(4,false,P,true);else QUAD(8,false,P,true);}}else if(sparse){if(bits==4)QUAD(4,true,P,false);else QUAD(8,true,P,false);}else{if(bits==4)QUAD(4,false,P,false);else QUAD(8,false,P,false);}
 if(quad){if(pixels==128){CHOOSE_QUAD(128);}else if(pixels==64){CHOOSE_QUAD(64);}else{CHOOSE_QUAD(32);}}
 else if(skip){if(sparse){if(pixels==128){if(bits==4)PART(4,128,true);else PART(8,128,true);}else{if(bits==4)PART(4,32,true);else PART(8,32,true);}}else{if(pixels==128){if(bits==4)PART(4,128,false);else PART(8,128,false);}else{if(bits==4)PART(4,32,false);else PART(8,32,false);}}}
 else if(pair){if(pixels==128){CHOOSE_PAIR(128);}else{CHOOSE_PAIR(32);}}
 else if(mode){if(pixels==128){CHOOSE(128,true);}else{CHOOSE(32,true);}}
 else {if(pixels==128){CHOOSE(128,false);}else{CHOOSE(32,false);}}
 #undef ARGS
 #undef QUAD
 #undef CHOOSE_QUAD
 #undef PART
 #undef PAIR
 #undef CHOOSE_PAIR
 #undef CHOOSE
 #undef PREP
 C10_CUDA_KERNEL_LAUNCH_CHECK();
 return {packed,planar};
}

std::vector<torch::Tensor> prepare_next_cat(std::vector<torch::Tensor> inputs,torch::Tensor multiplier,torch::Tensor params,int64_t bits,int64_t cp,torch::Tensor selection,int64_t sc,torch::Tensor packed,torch::Tensor planar,int64_t skip,int64_t mode,bool sparse) {
 TORCH_CHECK(skip>0,"expected prepared prefix");
 TORCH_CHECK(mode>=2&&mode<=5,"prepared prefix mode must be pair, quad, quad_loop or quad_swizzle");
 return prepare_cat_impl(inputs,multiplier,params,bits,cp,sparse,selection,sc,mode,packed,planar,skip);
}

std::vector<torch::Tensor> prepare_cached_cat(std::vector<torch::Tensor> inputs,torch::Tensor multiplier,torch::Tensor params,int64_t bits,int64_t cp,torch::Tensor selection,int64_t sc,int64_t mode,bool sparse){
 TORCH_CHECK(mode>=1&&mode<=5,"cached prepare mode must be 1 (scalar), 2 (pair), 3 (quad), 4 (quad_loop) or 5 (quad_swizzle)");
 return prepare_cat_impl(inputs,multiplier,params,bits,cp,sparse,selection,sc,mode);
}

std::vector<torch::Tensor> prepare_cat(std::vector<torch::Tensor> inputs,torch::Tensor inv_s,torch::Tensor params,int64_t bits,int64_t cp,bool sparse) {
 return prepare_cat_impl(inputs,inv_s,params,bits,cp,sparse,torch::Tensor(),0);
}

std::vector<torch::Tensor> prepare_compact_cat(std::vector<torch::Tensor> inputs,torch::Tensor inv_s,torch::Tensor params,int64_t bits,int64_t cp,torch::Tensor selection,int64_t sc) {
 return prepare_cat_impl(inputs,inv_s,params,bits,cp,true,selection,sc);
}

std::vector<torch::Tensor> prepare(torch::Tensor x,torch::Tensor inv_s,torch::Tensor params,int64_t bits,int64_t cp,bool sparse) {
 return prepare_cat({x},inv_s,params,bits,cp,sparse);
}

// CSR rows are output channels; indices encode (kernel_y, kernel_x, input_c).
// A warp spans adjacent output pixels, making input reads coalesced.
__global__ void sparse_epilogue_kernel(const int8_t* x,const float* y,half* result,const int* row,
 const int* index,const half* value,const float* sums,const float* bias,const float* params,
 int n,int c,int h,int w,int k,int p,int q,int r,int stride,int pad,int dil,int center,bool relu,bool sparse,bool precomputed) {
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
   float total=0.f;
   if(delta!=0.f) { if(precomputed)total=sums[k*r*r+o];else for(int j=0;j<r*r;++j)total+=sums[o*r*r+j]; }
   #pragma unroll
   for(int t=0;t<2;++t) if(valid[t]) {
     acc[t]=bias[o];
     if(inside[t]) acc[t]+=delta*total;
     else if(delta!=0.f) {
       for(int ry=0;ry<r;++ry) for(int rx=0;rx<r;++rx) {
         int iy=iy0[t]+ry*dil,ix=ix0[t]+rx*dil;
         if(iy>=0&&iy<h&&ix>=0&&ix<w)acc[t]+=delta*sums[(o*r+ry)*r+rx];
       }
     }
   }
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
   if(pos<int64_t(n)*p*q&&oc<k) {
     int64_t dst=pos*k+oc;
     float v=y[dst]+tile[px][ch];
     result[dst]=__float2half_rn(relu?fmaxf(0.f,v):v);
   }
 }
}


__global__ void dense_epilogue_kernel(const float* y,half* result,const float* sums,const float* bias,const float* params,
 int n,int h,int w,int k,int p,int q,int r,int stride,int pad,int dil,int center,bool relu,bool precomputed) {
 int64_t dst=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
 if(dst>=int64_t(n)*p*q*k)return;
 int oc=dst%k;int pos=dst/k;
 float value=y[dst]+bias[oc],delta=float(center)-params[1];
 if(delta!=0.f) {
   int iy0=((pos/q)%p)*stride-pad,ix0=(pos%q)*stride-pad;
   if(precomputed&&iy0>=0&&ix0>=0&&iy0+(r-1)*dil<h&&ix0+(r-1)*dil<w) value+=delta*sums[k*r*r+oc];
   else for(int ry=0;ry<r;++ry)for(int rx=0;rx<r;++rx) {
     int iy=iy0+ry*dil,ix=ix0+rx*dil;
     if(iy>=0&&iy<h&&ix>=0&&ix<w)value+=delta*sums[(oc*r+ry)*r+rx];
   }
 }
 result[dst]=__float2half_rn(relu?fmaxf(0.f,value):value);
}

torch::Tensor finish(torch::Tensor x,torch::Tensor y,torch::Tensor row,torch::Tensor index,
 torch::Tensor value,torch::Tensor sums,torch::Tensor bias,torch::Tensor params,
 int64_t bits,int64_t r,int64_t stride,int64_t pad,int64_t dil,bool relu,bool sparse) {
 if(sparse) check_tensor(x,y,at::kChar);check_tensor(row,y,at::kInt);check_tensor(index,y,at::kInt);
 check_tensor(value,y,at::kHalf);check_tensor(sums,y,at::kFloat);check_tensor(bias,y,at::kFloat);check_tensor(params,y,at::kFloat);
 TORCH_CHECK(y.is_cuda() && y.dim()==4 && x.dim()==4 && y.scalar_type()==at::kFloat && y.is_contiguous(at::MemoryFormat::ChannelsLast),"invalid output");
 int n=x.size(0),c=x.size(1),h=x.size(2),w=x.size(3),k=y.size(1),p=y.size(2),q=y.size(3);
 TORCH_CHECK((bits==4||bits==8) && r>0 && stride>0 && pad>=0 && dil>0,"invalid convolution parameters");
 TORCH_CHECK(y.size(0)==n && row.numel()==k+1 && index.numel()==value.numel() && (sums.numel()==k*r*r||sums.numel()==k*r*r+k) && bias.numel()==k && params.numel()==2,"invalid epilogue buffers");
 c10::cuda::CUDAGuard guard(y.device());
 auto stream=c10::cuda::getCurrentCUDAStream(y.get_device());
 auto result=torch::empty_like(y,y.options().dtype(at::kHalf));
 if(sparse) sparse_epilogue_kernel<<<dim3((k+15)/16,(int64_t(n)*p*q+63)/64),512,0,stream>>>(
   x.data_ptr<int8_t>(),y.data_ptr<float>(),reinterpret_cast<half*>(result.data_ptr()),row.data_ptr<int>(),index.data_ptr<int>(),reinterpret_cast<half*>(value.data_ptr()),
   sums.data_ptr<float>(),bias.data_ptr<float>(),params.data_ptr<float>(),n,c,h,w,k,p,q,r,stride,pad,dil,1<<(bits-1),relu,sparse,sums.numel()>k*r*r);
 else dense_epilogue_kernel<<<(int64_t(n)*p*q*k+255)/256,256,0,stream>>>(
   y.data_ptr<float>(),reinterpret_cast<half*>(result.data_ptr()),sums.data_ptr<float>(),bias.data_ptr<float>(),params.data_ptr<float>(),
   n,h,w,k,p,q,r,stride,pad,dil,1<<(bits-1),relu,sums.numel()>k*r*r);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return result;
}
