#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>

#include "cutlass/cutlass.h"
#include "cutlass/conv/kernel/default_conv2d_fprop_with_broadcast.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/numeric_conversion.h"

// CUTLASS supplies the conflict-free INT32 -> FP16 iterator for eight-wide
// stores. The generic four-wide path is also valid for our K % 4 tail layers,
// but its catch-all template only admits integer outputs, so define the
// equivalent non-mixed iterators locally.
namespace cutlass {
namespace epilogue {
namespace threadblock {
namespace detail {

template <
    typename ThreadblockShape,
    typename WarpShape,
    typename InstructionShape,
    typename ThreadMap>
struct DefaultIteratorsTensorOp<
    half_t,
    int32_t,
    4,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    ThreadMap> {
  using WarpTileIterator = cutlass::epilogue::warp::TileIteratorTensorOp<
      WarpShape,
      InstructionShape,
      int32_t,
      cutlass::layout::RowMajor>;
  using SharedLoadIterator = cutlass::epilogue::threadblock::SharedLoadIterator<
      ThreadMap,
      int32_t,
      16>;
  static int const kFragmentsPerIteration = 1;
};

}  // namespace detail
}  // namespace threadblock
}  // namespace epilogue
}  // namespace cutlass


namespace {
using Layout = cutlass::layout::TensorNHWC;
struct SparseParams {
 const int8_t* x; const int* row; const int* location; const half* value;
 const float* sums; const float* bias; const float* activation;
 int n,c,h,w,k,p,q,r,stride,pad,dil,center;bool relu,sparse,precomputed;
};
template<int Access> CUTLASS_DEVICE cutlass::MatrixCoord output_coord(int ordinal);

template<int Access, bool SparseCode> class FusedOutput {
 public:
 using ElementOutput=cutlass::half_t;using ElementC=cutlass::half_t;
 using ElementAccumulator=int32_t;using ElementCompute=float;
 using ElementZ=cutlass::half_t;using ElementT=cutlass::half_t;using ElementVector=float;
 static int const kElementsPerAccess=Access,kCount=Access;
 static bool const kIsSingleSource=true,kStoreZ=true,kStoreT=false;
 using FragmentAccumulator=cutlass::Array<int32_t,Access>;
 using FragmentCompute=cutlass::Array<float,Access>;
 using FragmentC=cutlass::Array<cutlass::half_t,Access>;
 using FragmentZ=cutlass::Array<cutlass::half_t,Access>;
 using FragmentT=FragmentZ;
 struct Params { SparseParams sparse; CUTLASS_HOST_DEVICE Params():sparse{} {} CUTLASS_HOST_DEVICE Params(SparseParams s):sparse(s) {} };
 private:
 SparseParams p_;mutable int ordinal_;
 public:
 CUTLASS_HOST_DEVICE explicit FusedOutput(Params const& p):p_(p.sparse),ordinal_(0) {}
 CUTLASS_HOST_DEVICE bool is_source_needed() const {return false;}
 CUTLASS_HOST_DEVICE void set_k_partition(int,int) {}
 CUTLASS_DEVICE void operator()(FragmentZ& out,FragmentT& tmp,FragmentAccumulator const& acc,FragmentC const&,FragmentCompute const& scale) const {(*this)(out,tmp,acc,scale);}
 CUTLASS_DEVICE void operator()(FragmentZ& out,FragmentT&,FragmentAccumulator const& acc,FragmentCompute const& scale) const {
   auto coord=output_coord<Access>(ordinal_++);
   int pos=coord.row(),oc=coord.column();
   if(pos>=p_.n*p_.p*p_.q) {out.clear();return;}
   int batch=pos/(p_.p*p_.q),yy=(pos/p_.q)%p_.p,xx=pos%p_.q;
   int iy0=yy*p_.stride-p_.pad,ix0=xx*p_.stride-p_.pad;
   float delta=float(p_.center)-p_.activation[1],a=p_.activation[0];
   bool interior=iy0>=0 && ix0>=0 && iy0+(p_.r-1)*p_.dil<p_.h && ix0+(p_.r-1)*p_.dil<p_.w;
   #pragma unroll
   for(int t=0;t<Access;++t) {
     int o=oc+t;float v=0;
     if(o<p_.k) {
       v=float(acc[t])*scale[t]+p_.bias[o];
       if(delta!=0.f) {
         float total=0.f;
         if(interior&&p_.precomputed)total=p_.sums[p_.k*p_.r*p_.r+o];
         else for(int ry=0;ry<p_.r;++ry) for(int rx=0;rx<p_.r;++rx) {
           int iy=iy0+ry*p_.dil,ix=ix0+rx*p_.dil;
           if(interior || (iy>=0&&iy<p_.h&&ix>=0&&ix<p_.w)) total+=p_.sums[(o*p_.r+ry)*p_.r+rx];
         }
         v+=delta*total;
       }
       if constexpr(SparseCode) { if(p_.sparse) {
         for(int j=p_.row[o];j<p_.row[o+1];++j) {
           int loc=p_.location[j],ic=loc&4095,rx=(loc>>12)&15,ry=(loc>>16)&15;
           int iy=iy0+ry*p_.dil,ix=ix0+rx*p_.dil;
           if(interior || (iy>=0&&iy<p_.h&&ix>=0&&ix<p_.w)) {
             float av=(float(p_.x[((int64_t(batch)*p_.c+ic)*p_.h+iy)*p_.w+ix])+delta)*a;
             v=fmaf(__half2float(__float2half_rn(av)),__half2float(p_.value[j]),v);
           }
         }
       }
       }
       if(p_.relu) v=fmaxf(0.f,v);
     }
     out[t]=cutlass::half_t(v);
   }
 }
};
template <int Bits> struct Traits;
template <> struct Traits<8> { using Input = int8_t; static constexpr int K = 64; static constexpr int Align = 16; };
template <> struct Traits<4> { using Input = cutlass::int4b_t; static constexpr int K = 128; static constexpr int Align = 32; };
template <int Bits, int Access, bool SparseCode=true>
using FusedKernel = typename cutlass::conv::kernel::DefaultConv2dFpropWithBroadcast<
    typename Traits<Bits>::Input, Layout, typename Traits<Bits>::Input, Layout,
    cutlass::half_t, Layout, int32_t, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128,32,Traits<Bits>::K>,cutlass::gemm::GemmShape<32,32,Traits<Bits>::K>,
    cutlass::gemm::GemmShape<16,8,Traits<Bits>::K/2>,FusedOutput<Access,SparseCode>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,3,
    cutlass::arch::OpMultiplyAddSaturate,cutlass::conv::IteratorAlgorithm::kOptimized,
    cutlass::conv::StrideSupport::kStrided,Traits<Bits>::Align,Traits<Bits>::Align>::Kernel;

template<int Access> CUTLASS_DEVICE cutlass::MatrixCoord output_coord(int ordinal) {
 using TM=typename FusedKernel<4,Access>::Epilogue::OutputTileIterator::ThreadMap;
 auto initial=TM::initial_offset(threadIdx.x);
 int frag=ordinal%TM::Iterations::kCount,iter=ordinal/TM::Iterations::kCount;
 int col=frag%TM::Iterations::kColumn,u=frag/TM::Iterations::kColumn;
 int row=u%TM::Iterations::kRow;u/=TM::Iterations::kRow;
 int group=u%TM::Iterations::kGroup,cluster=u/TM::Iterations::kGroup;
 int advance=iter*TM::Shape::kRow;
 advance+=(iter/TM::Count::kRow)*(TM::Shape::kGroup-1)*TM::Shape::kRow*TM::Count::kRow;
 advance+=(iter/(TM::Count::kRow*TM::Count::kGroup))*TM::Count::kGroup*TM::Shape::kGroup*TM::Count::kRow*TM::Shape::kRow;
 advance+=(iter/(TM::Count::kRow*TM::Count::kGroup*TM::Count::kCluster))*TM::Shape::kGroup*TM::Shape::kRow*TM::Shape::kCluster*TM::Shape::kTile;
 return {int(blockIdx.x)*128+initial.row()+advance+row*TM::Delta::kRow+group*TM::Delta::kGroup+cluster*TM::Delta::kCluster,
         int(blockIdx.y)*32+initial.column()+col*TM::Delta::kColumn};
}

template<int Bits,int Access,bool SparseCode>
void run_fused(torch::Tensor x,torch::Tensor weight,torch::Tensor scale,torch::Tensor out,int cp,SparseParams p,cudaStream_t stream) {
 using Op=cutlass::conv::device::ImplicitGemmConvolution<FusedKernel<Bits,Access,SparseCode>>;
 using Input=typename Traits<Bits>::Input;
 cutlass::Tensor4DCoord xs(p.n,p.h,p.w,cp),ws(p.k,p.r,p.r,cp),ys(p.n,p.p,p.q,p.k);
 cutlass::conv::Conv2dProblemSize problem(xs,ws,{p.pad,p.pad,p.pad,p.pad},{p.stride,p.stride},{p.dil,p.dil},ys,cutlass::conv::Mode::kCrossCorrelation,1);
 typename Op::Arguments args(problem,
  {reinterpret_cast<Input*>(x.data_ptr()),Layout::packed(xs)},
  {reinterpret_cast<Input*>(weight.data_ptr()),Layout::packed(ws)},
  {nullptr,Layout::Stride(0)},
  {reinterpret_cast<cutlass::half_t*>(out.data_ptr()),Layout::packed(ys)},
  typename Op::EpilogueOutputOp::Params(p),cutlass::conv::SplitKMode::kSerial,static_cast<void*>(scale.data_ptr<float>()));
 Op op;
 TORCH_CHECK(op.can_implement(args)==cutlass::Status::kSuccess,"unsupported fused convolution");
 auto size=op.get_workspace_size(args);auto workspace=torch::empty({int64_t(size)},x.options().dtype(at::kByte));
 auto status=op.initialize(args,size?workspace.data_ptr():nullptr,stream);
 TORCH_CHECK(status==cutlass::Status::kSuccess,"fused initialize failed");
 status=op(stream);TORCH_CHECK(status==cutlass::Status::kSuccess,"fused run failed");
}
} // namespace

torch::Tensor fused_conv(torch::Tensor x,torch::Tensor planar,torch::Tensor weight,torch::Tensor scale,
 torch::Tensor row,torch::Tensor location,torch::Tensor value,torch::Tensor sums,torch::Tensor bias,torch::Tensor activation,
 int64_t bits,int64_t cp,int64_t k,int64_t r,int64_t stride,int64_t pad,int64_t dil,bool relu,bool sparse) {
 TORCH_CHECK(bits==4||bits==8,"invalid bit width");
 TORCH_CHECK(planar.is_cuda()&&planar.dim()==4&&planar.scalar_type()==at::kChar&&(!sparse||planar.is_contiguous()),"invalid planar input");
 auto check=[&](torch::Tensor t,at::ScalarType type){TORCH_CHECK(t.device()==planar.device()&&t.scalar_type()==type&&t.is_contiguous(),"invalid fused buffer");};
 check(x,at::kByte);check(weight,at::kByte);check(scale,at::kFloat);check(row,at::kInt);check(location,at::kInt);check(value,at::kHalf);check(sums,at::kFloat);check(bias,at::kFloat);check(activation,at::kFloat);
 int n=planar.size(0),c=planar.size(1),h=planar.size(2),w=planar.size(3);
 TORCH_CHECK(k>0&&r>0&&stride>0&&dil>0&&pad>=0&&cp>=c&&cp%(bits==4?32:16)==0&&k%4==0,"invalid fused dimensions");
 int p=(h+2*pad-dil*(r-1)-1)/stride+1,q=(w+2*pad-dil*(r-1)-1)/stride+1;
 TORCH_CHECK(p>0&&q>0&&x.numel()==int64_t(n)*h*w*cp*bits/8&&weight.numel()==k*r*r*cp*bits/8,"invalid packed size");
 TORCH_CHECK(row.numel()==k+1&&location.numel()==value.numel()&&(sums.numel()==k*r*r||sums.numel()==k*r*r+k)&&bias.numel()==k&&scale.numel()==k&&activation.numel()==2,"invalid fused metadata");
 c10::cuda::CUDAGuard guard(x.device());auto stream=c10::cuda::getCurrentCUDAStream(x.get_device());
 auto out=torch::empty({n,k,p,q},x.options().dtype(at::kHalf).memory_format(at::MemoryFormat::ChannelsLast));
 SparseParams param{planar.data_ptr<int8_t>(),row.data_ptr<int>(),location.data_ptr<int>(),reinterpret_cast<half*>(value.data_ptr()),sums.data_ptr<float>(),bias.data_ptr<float>(),activation.data_ptr<float>(),n,c,h,w,int(k),p,q,int(r),int(stride),int(pad),int(dil),1<<(bits-1),relu,sparse,sums.numel()>k*r*r};
 #define RUN(B,A,S) run_fused<B,A,S>(x,weight,scale,out,cp,param,stream)
 #define CHOOSE(S) if(bits==4){if(k%8==0)RUN(4,8,S);else RUN(4,4,S);}else{if(k%8==0)RUN(8,8,S);else RUN(8,4,S);}
 if(sparse){CHOOSE(true);}else{CHOOSE(false);}
 #undef CHOOSE
 #undef RUN
 C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
