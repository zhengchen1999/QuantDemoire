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

using ElementOutput = float;
using Layout = cutlass::layout::TensorNHWC;
/// Fused D[channel] = accumulator * scale[channel] + bias[channel].
template <int ElementsPerAccess>
class ScaleBiasOutputOp {
 public:
  using ElementOutput = float;
  using ElementC = float;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;
  using ElementZ = float;
  using ElementT = float;
  using ElementVector = float;
  static int const kElementsPerAccess = ElementsPerAccess;
  static int const kCount = kElementsPerAccess;
  static bool const kIsSingleSource = true;
  static bool const kStoreZ = true;
  static bool const kStoreT = false;

  using FragmentAccumulator =
      cutlass::Array<ElementAccumulator, kElementsPerAccess>;
  using FragmentCompute = cutlass::Array<ElementCompute, kElementsPerAccess>;
  using FragmentC = cutlass::Array<ElementC, kElementsPerAccess>;
  using FragmentZ = cutlass::Array<ElementZ, kElementsPerAccess>;
  using FragmentT = cutlass::Array<ElementT, kElementsPerAccess>;

  struct Params {
    bool has_bias;

    CUTLASS_HOST_DEVICE
    explicit Params(bool has_bias_ = false) : has_bias(has_bias_) {}
  };

 private:
  bool has_bias_;

 public:
  CUTLASS_HOST_DEVICE
  explicit ScaleBiasOutputOp(Params const& params) : has_bias_(params.has_bias) {}

  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    return has_bias_;
  }

  CUTLASS_HOST_DEVICE
  void set_k_partition(int, int) {}

  CUTLASS_HOST_DEVICE
  void operator()(
      FragmentZ& output,
      FragmentT&,
      FragmentAccumulator const& accumulator,
      FragmentC const& bias,
      FragmentCompute const& scale) const {
    FragmentCompute accumulator_float =
        cutlass::NumericArrayConverter<
            ElementCompute,
            ElementAccumulator,
            kElementsPerAccess>()(accumulator);
    FragmentCompute bias_float =
        cutlass::NumericArrayConverter<
            ElementCompute,
            ElementC,
            kElementsPerAccess>()(bias);
    FragmentCompute result;
    CUTLASS_PRAGMA_UNROLL
    for (int index = 0; index < kElementsPerAccess; ++index) {
      result[index] = accumulator_float[index] * scale[index] + bias_float[index];
    }
    output = cutlass::NumericArrayConverter<
        ElementZ,
        ElementCompute,
        kElementsPerAccess>()(result);
  }

  CUTLASS_HOST_DEVICE
  void operator()(
      FragmentZ& output,
      FragmentT&,
      FragmentAccumulator const& accumulator,
      FragmentCompute const& scale) const {
    FragmentCompute accumulator_float =
        cutlass::NumericArrayConverter<
            ElementCompute,
            ElementAccumulator,
            kElementsPerAccess>()(accumulator);
    FragmentCompute result;
    CUTLASS_PRAGMA_UNROLL
    for (int index = 0; index < kElementsPerAccess; ++index) {
      result[index] = accumulator_float[index] * scale[index];
    }
    output = cutlass::NumericArrayConverter<
        ElementZ,
        ElementCompute,
        kElementsPerAccess>()(result);
  }
};


template <int Bits> struct Traits;
template <> struct Traits<8> { using Input = int8_t; static constexpr int K = 64; static constexpr int Align = 16; };
template <> struct Traits<4> { using Input = cutlass::int4b_t; static constexpr int K = 128; static constexpr int Align = 32; };
template <int Bits, int Access>
using ConvKernel = typename cutlass::conv::kernel::DefaultConv2dFpropWithBroadcast<
    typename Traits<Bits>::Input, Layout, typename Traits<Bits>::Input, Layout,
    ElementOutput, Layout, int32_t, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128,32,Traits<Bits>::K>,
    cutlass::gemm::GemmShape<32,32,Traits<Bits>::K>,
    cutlass::gemm::GemmShape<16,8,Traits<Bits>::K/2>,
    ScaleBiasOutputOp<Access>, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3, cutlass::arch::OpMultiplyAddSaturate, cutlass::conv::IteratorAlgorithm::kOptimized,
    cutlass::conv::StrideSupport::kStrided, Traits<Bits>::Align, Traits<Bits>::Align>::Kernel;

template <int Bits, int Access>
void run_conv(torch::Tensor x, torch::Tensor weight, torch::Tensor scale, torch::Tensor out,
              int n,int h,int w,int c,int k,int r,int stride,int pad,int dil,int p,int q,
              cudaStream_t stream) {
  using Op = cutlass::conv::device::ImplicitGemmConvolution<ConvKernel<Bits, Access>>;
  using Input = typename Traits<Bits>::Input;
  cutlass::Tensor4DCoord xs(n,h,w,c), ws(k,r,r,c), ys(n,p,q,k);
  cutlass::conv::Conv2dProblemSize problem(xs,ws,{pad,pad,pad,pad},{stride,stride},{dil,dil},ys,
                                        cutlass::conv::Mode::kCrossCorrelation,1);
  typename Op::Arguments args(problem,
    {reinterpret_cast<Input*>(x.data_ptr()),Layout::packed(xs)},
    {reinterpret_cast<Input*>(weight.data_ptr()),Layout::packed(ws)},
    {nullptr,Layout::Stride(0)},
    {reinterpret_cast<ElementOutput*>(out.data_ptr()),Layout::packed(ys)},
    typename Op::EpilogueOutputOp::Params(false), cutlass::conv::SplitKMode::kSerial,
    static_cast<void*>(scale.data_ptr<float>()));
  Op op;
  TORCH_CHECK(op.can_implement(args)==cutlass::Status::kSuccess,"unsupported CUTLASS convolution");
  auto size=op.get_workspace_size(args);
  auto workspace=torch::empty({int64_t(size)},x.options().dtype(at::kByte));
  auto status=op.initialize(args,size?workspace.data_ptr():nullptr,stream);
  TORCH_CHECK(status==cutlass::Status::kSuccess,"CUTLASS initialize: ",cutlassGetStatusString(status));
  status=op(stream);
  TORCH_CHECK(status==cutlass::Status::kSuccess,"CUTLASS run: ",cutlassGetStatusString(status));
}

void check_tensor(const torch::Tensor& t, const torch::Tensor& ref, at::ScalarType dtype) {
  TORCH_CHECK(t.is_cuda() && t.device()==ref.device() && t.scalar_type()==dtype && t.is_contiguous(),
              "invalid tensor device, dtype or layout");
}

} // namespace

torch::Tensor integer_conv(torch::Tensor x, torch::Tensor weight, torch::Tensor scale,
  int64_t bits,int64_t n,int64_t h,int64_t w,int64_t c,int64_t k,int64_t r,
  int64_t stride,int64_t pad,int64_t dil) {
  TORCH_CHECK(bits==4 || bits==8,"bits must be 4 or 8");
  TORCH_CHECK(n>0 && h>0 && w>0 && c>0 && k>0 && r>0 && stride>0 && dil>0 && pad>=0,"invalid dimensions");
  check_tensor(x,x,at::kByte);check_tensor(weight,x,at::kByte);check_tensor(scale,x,at::kFloat);
  TORCH_CHECK(c%(bits==4?32:16)==0 && k%4==0,"invalid channel alignment");
  TORCH_CHECK(x.numel()==n*h*w*c*bits/8 && weight.numel()==k*r*r*c*bits/8 && scale.numel()==k,"invalid buffer size");
  int p=(h+2*pad-dil*(r-1)-1)/stride+1, q=(w+2*pad-dil*(r-1)-1)/stride+1;
  TORCH_CHECK(p>0 && q>0,"invalid output shape");
  c10::cuda::CUDAGuard guard(x.device());
  auto out=torch::empty({n,k,p,q},x.options().dtype(at::kFloat).memory_format(at::MemoryFormat::ChannelsLast));
  auto stream=c10::cuda::getCurrentCUDAStream(x.get_device());
  #define RUN(B,A) run_conv<B,A>(x,weight,scale,out,n,h,w,c,k,r,stride,pad,dil,p,q,stream)
  if(bits==4) { if(k%8==0) RUN(4,8); else RUN(4,4); }
  else { if(k%8==0) RUN(8,8); else RUN(8,4); }
  #undef RUN
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}


