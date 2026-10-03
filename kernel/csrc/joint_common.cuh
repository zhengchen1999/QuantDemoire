#pragma once
// Shared implementation; second-successor output is compiled only in joint_two.cu.
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
#include "cutlass/fast_math.h"
/***************************************************************************************************
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Templates implementing loading of convolution tiles mapped to GEMM B (filter tile)
    matrix from memory.

    This iterator assumes TensorNHWC or TensorCxRSKx<Interleave> layout of tensors in Global Memory.

    The iterator is specialized for each of the three convolution operators: forward propagation (Fprop),
    backward data gradient (Dgrad), and backward weight gradient (Wgrad).
*/



#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/coord.h"
#include "cutlass/predicate_vector.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/tensor_view.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/threadblock/conv2d_params.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Shape_,
  typename Element_,
  typename Layout_,
  typename ThreadMap_,
  typename AccessType_ = cutlass::AlignedArray<Element_, ThreadMap_::kElementsPerAccess>
>
class QuantDemoirePackedFilterIterator {
public:

  //
  // Types
  //

  using Shape = Shape_;
  using Element = Element_;
  using Layout = Layout_;
  static_assert(platform::is_same<Layout, layout::TensorNHWC>::value,
                "packed KRSC filter specialization requires TensorNHWC");
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;
  using TensorRef = cutlass::TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;
  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  static IteratorAlgorithm const kIteratorAlgorithm = conv::IteratorAlgorithm::kFewChannels;
  static StrideSupport const kStrideSupport = conv::StrideSupport::kStrided;
  static int const kConvDim = 2;
  using ConvProblemSize = typename conv::Conv2dProblemSize;

  static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;
  static int const kPositionsPerTile = Shape::kRow;

  static int const kAccessesPerVector = ThreadMap::kElementsPerAccess / AccessType::kElements;

  static bool const kUseFastDivmodPrologue = true;
  static bool const kUseFastDivmodMainloop = true;

  static_assert(!(ThreadMap::kElementsPerAccess % AccessType::kElements),
    "Vectors implied by the thread map must be divisible by the access type.");

  //
  // Simplifying assertions
  //
  static_assert(ThreadMap::Iterations::kContiguous == 1,
    "Require Iterations::kContiguous == 1");

  //
  // Parameters structure
  //

  using Params = Conv2dFewChannelsParams<Layout>;

private:

  Params const &params_;
  Conv2dProblemSize const &problem_size_;
  LongIndex iteration_contiguous_;
  LongIndex iteration_strided_;
  LongIndex iteration_vector_;
  char const *pointer_;

  int rsc_index_;

  int offset_k_[ThreadMap::Iterations::kStrided];

public:

  CUTLASS_HOST_DEVICE
  QuantDemoirePackedFilterIterator(
    Params const &params,
    Conv2dProblemSize const &problem_size,
    Element const *ptr,
    int thread_idx,
    MatrixCoord const &threadblock_offset = MatrixCoord()
  ):
    params_(params),
    problem_size_(problem_size),
    pointer_(reinterpret_cast<char const *>(ptr)),
    rsc_index_(0) {

    layout::PitchLinearCoord thread_coord = ThreadMap::initial_offset(thread_idx);

    rsc_index_ = (threadblock_offset.row() + thread_coord.contiguous());

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      offset_k_[s] = threadblock_offset.column() + thread_coord.strided() + s * ThreadMap::Delta::kStrided;
    }

    set_iteration_index(0);
  }

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(Index index) {
    iteration_vector_ = index % kAccessesPerVector;
    int residual_access = index / kAccessesPerVector;
    iteration_contiguous_ = residual_access % ThreadMap::Iterations::kContiguous;
    iteration_strided_ = residual_access / ThreadMap::Iterations::kContiguous;
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    pointer_ += pointer_offset * 8 / sizeof_bits<Element>::value;
  }

  CUTLASS_HOST_DEVICE
  void advance() {
    // moves to the next tile
    rsc_index_ += kPositionsPerTile * problem_size_.split_k_slices;
  }

  /// Returns the coordinate in the filter tensor W that is currently pointed to
  /// by the iterator.
  CUTLASS_HOST_DEVICE
  TensorCoord at() const {

    int rsc_index = rsc_index_ + iteration_vector_ * AccessType::kElements;

    int c = 0;
    int s = 0;
    int r = 0;

    if (kUseFastDivmodMainloop) {
      int rs_index = params_.divmod_C.divmod(c, rsc_index);
      r = params_.divmod_S.divmod(s, rs_index);
    }
    else {
      c = (rsc_index % problem_size_.C);
      int rs_index = (rsc_index / problem_size_.C);

      s = (rs_index % problem_size_.S);
      r = (rs_index / problem_size_.S);
    }

    int k = offset_k_[iteration_strided_];

    return TensorCoord(k, r, s, c);
  }

  /// Returns true if the current coordinate is within the activations tensor W
  CUTLASS_HOST_DEVICE
  bool valid() const {
    int rsc = rsc_index_ + iteration_vector_ * AccessType::kElements;
    return offset_k_[iteration_strided_] < problem_size_.K && rsc >= 0 &&
           rsc < problem_size_.R * problem_size_.S * problem_size_.C;
  }

  /// Contiguous packed KRSC: avoid decomposing and rebuilding the RSC index.
  CUTLASS_HOST_DEVICE
  AccessType const *get() const {
    int rsc = rsc_index_ + iteration_vector_ * AccessType::kElements;
    int32_t offset = offset_k_[iteration_strided_] * params_.stride_n + rsc;
    return reinterpret_cast<AccessType const *>(pointer_ + offset * sizeof_bits<Element>::value / 8);
  }

  /// Increments to the next memory access
  CUTLASS_HOST_DEVICE
  QuantDemoirePackedFilterIterator &operator++() {
    ++iteration_vector_;
    if (iteration_vector_ < kAccessesPerVector) {
      return *this;
    }
    iteration_vector_ = 0;

    ++iteration_contiguous_;
    if (iteration_contiguous_ < ThreadMap::Iterations::kContiguous) {
      return *this;
    }
    iteration_contiguous_ = 0;

    ++iteration_strided_;
    if (iteration_strided_ < ThreadMap::Iterations::kStrided) {
      return *this;
    }
    iteration_strided_ = 0;

    return *this;
  }

  /// Determines whether the Implicit GEMM can execute the given problem.
  CUTLASS_HOST_DEVICE
  static Status can_implement(Conv2dProblemSize const &problem_size) {

    // check alignment constraint on iterator's contiguous dimension
    if (problem_size.C % AccessType::kElements) {
      return Status::kErrorInvalidProblem;
    }

    if (platform::is_same<Layout, layout::TensorCxRSKx<32>>::value) {
      if (problem_size.K % 32) {
        return Status::kErrorInvalidProblem;
      }
    }

    if (platform::is_same<Layout, layout::TensorCxRSKx<64>>::value) {
      if (problem_size.K % 64) {
        return Status::kErrorInvalidProblem;
      }
    }

    return Status::kSuccess;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace conv
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////


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



// Experimental cooperative CSR epilogue. Accumulate an entire output tile
// with spatially contiguous warp gathers before the CUTLASS output transform.
namespace {
using Layout=cutlass::layout::TensorNHWC;
struct JointParams {
 const int8_t* x;const int* row;const int* location;const int64_t* offsets;const half* value;
 const float* sums;const float* bias;const float* activation;const half* residual;
 int n,c,h,w,k,p,q,r,stride,pad,dil,center;bool relu,add_residual;int shared_offset;
 cutlass::FastDivmod q_div,pq_div;
 const float* next_multiplier=nullptr;const float* next_activation=nullptr;
 const int* selection=nullptr;uint8_t* next_packed=nullptr;int8_t* next_planar=nullptr;
 int next_bits=0,next_cp=0,next_sc=0;
#ifdef QUANTDEMOIRE_TWO_SUCCESSOR
 const float* future_multiplier=nullptr;const float* future_activation=nullptr;
 uint8_t* future_packed=nullptr;int future_bits=0,future_offset=0;
#endif
};
template<int Access,int Warps,bool Sparse,int Variant> CUTLASS_DEVICE cutlass::MatrixCoord joint_coord(int ordinal);
template<int Bits,int Access,bool Balanced=false> CUTLASS_DEVICE void joint_shared_pointwise(JointParams const& p,float* tile);

template<int Access,int Warps,bool Sparse,int Variant,bool SharedW4Lookup=false> class JointOutput {
 public:
 using ElementOutput=cutlass::half_t;using ElementC=cutlass::half_t;
 using ElementAccumulator=int32_t;using ElementCompute=float;
 using ElementZ=cutlass::half_t;using ElementT=cutlass::half_t;using ElementVector=float;
 static int const kElementsPerAccess=Access,kCount=Access;
 static bool const kTransposeNext=Variant==13||Variant==14||Variant==19||Variant==20||Variant==25||Variant==26||Variant==33||Variant==34||Variant==39||Variant==40||Variant==45||Variant==46;
 // The omitted-feature translation units still run this output operator,
 // including its FP16 rounding and consumer-specific quantization, but do not
 // write the full FP16 destination. They must always produce an owned prefix.
#ifdef QUANTDEMOIRE_OMIT_FEATURE
 static bool const kIsSingleSource=true,kStoreZ=false,kStoreT=false;
#else
 static bool const kIsSingleSource=true,kStoreZ=true,kStoreT=false;
#endif
 using FragmentAccumulator=cutlass::Array<int32_t,Access>;
 using FragmentCompute=cutlass::Array<float,Access>;
 using FragmentC=cutlass::Array<cutlass::half_t,Access>;
 using FragmentZ=cutlass::Array<cutlass::half_t,Access>;
 using FragmentT=FragmentZ;
 struct Params {JointParams data;CUTLASS_HOST_DEVICE Params():data{}{} CUTLASS_HOST_DEVICE Params(JointParams p):data(p){}};
 private:
 JointParams p_;mutable int ordinal_;float* sparse_tile_;
 public:
 CUTLASS_HOST_DEVICE explicit JointOutput(Params const& p):p_(p.data),ordinal_(0),sparse_tile_(nullptr) {
 #if defined(__CUDA_ARCH__)
   if constexpr(Sparse) {
     if constexpr(Variant==9||Variant==12||Variant==14||Variant==16||Variant==18||Variant==20||(Variant>=21&&Variant<=26)) {
       extern __shared__ int joint_storage[];
       sparse_tile_=reinterpret_cast<float*>(reinterpret_cast<char*>(joint_storage)+p_.shared_offset);
       if(p_.center==8)joint_shared_pointwise<4,Access,((Variant>=15&&Variant<=20)||Variant==22||Variant==24||Variant==26||Variant==36||Variant==38||Variant==40||Variant==42||Variant==44||Variant==46||Variant==30||Variant==32||Variant==34)>(p_,sparse_tile_);
       else joint_shared_pointwise<8,Access,((Variant>=15&&Variant<=20)||Variant==22||Variant==24||Variant==26||Variant==36||Variant==38||Variant==40||Variant==42||Variant==44||Variant==46||Variant==30||Variant==32||Variant==34)>(p_,sparse_tile_);
     } else {
     constexpr int TileM=Warps==8?64:128,TileN=Variant==6?64:32,PixelsPerLane=TileM/32;
     if constexpr(Variant>=2) {
       extern __shared__ int joint_storage[];
       sparse_tile_=reinterpret_cast<float*>(reinterpret_cast<char*>(joint_storage)+p_.shared_offset);
       // The integer main loop has finished. Wait for every warp before
       // reusing its shared memory beyond the epilogue's own scratch space.
       __syncthreads();
     } else {
       __shared__ float tile[TileM][TileN+1];sparse_tile_=&tile[0][0];
     }
     int lane=threadIdx.x&31,warp=threadIdx.x>>5,pb=blockIdx.x*TileM,oc0=blockIdx.y*TileN;
     using Index=typename std::conditional<(Variant>=7),int,int64_t>::type;
     int iy0[PixelsPerLane],ix0[PixelsPerLane],batch[PixelsPerLane];Index base[PixelsPerLane];bool valid[PixelsPerLane],inside=true;
     float delta=float(p_.center)-p_.activation[1],scale=p_.activation[0];
     // Sixteen half-rounded W4 operands, private to each warp. The main-loop
     // barrier above retires previous shared users; prefix storage is disjoint.
     float* operand_table=nullptr;
     if constexpr(SharedW4Lookup) {
       static_assert(Warps==8,"shared W4 lookup requires eight warps");
       extern __shared__ int joint_operand_storage[];
       operand_table=reinterpret_cast<float*>(reinterpret_cast<char*>(joint_operand_storage)
         +p_.shared_offset+64*33*(sizeof(float)+sizeof(int8_t)))+warp*16;
       if(lane<16)operand_table[lane]=__half2float(__float2half_rn((float(lane-8)+delta)*scale));
       __syncwarp();
     }
     // W4 has only sixteen activation codes. One exact half-rounded value
     // per lane replaces repeated conversions in the sparse dot product.
     float lookup=0.f;
     if constexpr(Variant==5)lookup=__half2float(__float2half_rn((float((lane&15)-8)+delta)*scale));
     #pragma unroll
     for(int t=0;t<PixelsPerLane;++t) {
       int pos=pb+lane+t*32;valid[t]=pos<int64_t(p_.n)*p_.p*p_.q;
       if constexpr(Variant>=8) {
         int local=pos,y,x;
         batch[t]=0;
         if(p_.n!=1)p_.pq_div(batch[t],local,pos);
         p_.q_div(y,x,local);
         iy0[t]=y*p_.stride-p_.pad;ix0[t]=x*p_.stride-p_.pad;
       } else {
         batch[t]=p_.n==1?0:pos/(p_.p*p_.q);
         iy0[t]=(p_.n==1?pos/p_.q:(pos/p_.q)%p_.p)*p_.stride-p_.pad;
         ix0[t]=(pos%p_.q)*p_.stride-p_.pad;
       }
       base[t]=Index(batch[t])*p_.c*p_.h*p_.w+Index(iy0[t])*p_.w+ix0[t];
       inside=inside&&valid[t]&&iy0[t]>=0&&ix0[t]>=0&&iy0[t]+(p_.r-1)*p_.dil<p_.h&&ix0[t]+(p_.r-1)*p_.dil<p_.w;
     }
     // Every lane must execute the same shuffles, including border tiles.
     if constexpr(Variant==5)inside=__all_sync(0xffffffff,inside);
     #pragma unroll
     for(int group=0;group<TileN/Warps;++group) {
       int local=warp+group*Warps;
       // A derived row permutation redistributes whole rows; each CSR FMA sequence is unchanged.
       if constexpr((Variant>=15&&Variant<=20)||Variant==22||Variant==24||Variant==26||Variant==36||Variant==38||Variant==40||Variant==42||Variant==44||Variant==46||Variant==30||Variant==32||Variant==34)local=p_.row[p_.k+1+oc0+local];
       int oc=oc0+local;float acc[PixelsPerLane]={};
       if(oc<p_.k) {
         if(inside) {
           for(int j=p_.row[oc];j<p_.row[oc+1];++j) {
             Index off;float weight;
             if constexpr((Variant>=35&&Variant<=46)) {
               uint64_t entry=static_cast<uint64_t>(p_.offsets[j]);
               off=Index(int32_t(uint32_t(entry)));
               weight=__half2float(__ushort_as_half(uint16_t(entry>>32)));
             } else {off=Index(p_.offsets[j]);weight=__half2float(p_.value[j]);}
             #pragma unroll
             for(int t=0;t<PixelsPerLane;++t) {
               int code=int(p_.x[base[t]+off]);float operand;
               if constexpr(Variant==5)operand=__shfl_sync(0xffffffff,lookup,code+8);
               else if constexpr(SharedW4Lookup)operand=operand_table[code+8];
               else operand=__half2float(__float2half_rn((float(code)+delta)*scale));
               acc[t]=fmaf(operand,weight,acc[t]);
             }
           }
         } else {
           for(int j=p_.row[oc];j<p_.row[oc+1];++j) {
             int loc=p_.location[j],ic=loc&4095,dx=((loc>>12)&15)*p_.dil,dy=((loc>>16)&15)*p_.dil;
             float weight=__half2float(p_.value[j]);
             #pragma unroll
             for(int t=0;t<PixelsPerLane;++t) {
               int iy=iy0[t]+dy,ix=ix0[t]+dx;
               bool load=valid[t]&&iy>=0&&iy<p_.h&&ix>=0&&ix<p_.w;
               if constexpr(Variant==5) {
                 int code=load?int(p_.x[((Index(batch[t])*p_.c+ic)*p_.h+iy)*p_.w+ix]):0;
                 float operand=__shfl_sync(0xffffffff,lookup,code+8);
                 if(load)acc[t]=fmaf(operand,weight,acc[t]);
               } else if(load) {
                 float av=(float(p_.x[((Index(batch[t])*p_.c+ic)*p_.h+iy)*p_.w+ix])+delta)*scale;
                 acc[t]=fmaf(__half2float(__float2half_rn(av)),weight,acc[t]);
               }
             }
           }
         }
       }
       #pragma unroll
       for(int t=0;t<PixelsPerLane;++t)sparse_tile_[(lane+t*32)*(TileN+1)+local]=acc[t];
     }
     __syncthreads();
     }
   }
 #endif
 }
 CUTLASS_HOST_DEVICE bool is_source_needed() const {return false;}
 CUTLASS_HOST_DEVICE void set_k_partition(int,int){}
 CUTLASS_DEVICE void operator()(FragmentZ& out,FragmentT& tmp,FragmentAccumulator const& acc,FragmentC const&,FragmentCompute const& scale) const {(*this)(out,tmp,acc,scale);}
 CUTLASS_DEVICE void operator()(FragmentZ& out,FragmentT&,FragmentAccumulator const& acc,FragmentCompute const& scale) const {
   auto coord=joint_coord<Access,Warps,Sparse,Variant>(ordinal_++);int pos=coord.row(),oc=coord.column();
   if(pos>=p_.n*p_.p*p_.q){out.clear();return;}
   int yy,xx;
   if constexpr(Variant>=8) {
     int local=pos,batch=0;
     if(p_.n!=1)p_.pq_div(batch,local,pos);
     p_.q_div(yy,xx,local);
   } else {yy=(pos/p_.q)%p_.p;xx=pos%p_.q;}
   int iy0=yy*p_.stride-p_.pad,ix0=xx*p_.stride-p_.pad;
   bool interior=iy0>=0&&ix0>=0&&iy0+(p_.r-1)*p_.dil<p_.h&&ix0+(p_.r-1)*p_.dil<p_.w;
   float delta=float(p_.center)-p_.activation[1];
   #pragma unroll
   for(int t=0;t<Access;++t) {
     int o=oc+t;float v=0.f;
     if(o<p_.k) {
       v=float(acc[t])*scale[t]+p_.bias[o];
       if(delta!=0.f) {
         float total=0.f;
         if(interior)total=p_.sums[p_.k*p_.r*p_.r+o];
         else for(int ry=0;ry<p_.r;++ry)for(int rx=0;rx<p_.r;++rx) {
           int iy=iy0+ry*p_.dil,ix=ix0+rx*p_.dil;
           if(iy>=0&&iy<p_.h&&ix>=0&&ix<p_.w)total+=p_.sums[(o*p_.r+ry)*p_.r+rx];
         }
         v+=delta*total;
       }
       if constexpr(Sparse) {
         // Match the existing stored-half dense result before adding the
         // independently accumulated CSR result; do not accumulate into v.
         v=__half2float(__float2half_rn(v))+sparse_tile_[(pos-int(blockIdx.x)*(Warps==8?64:128))*((Variant==6?64:32)+1)+o-int(blockIdx.y)*(Variant==6?64:32)];
       }
       if(p_.relu)v=fmaxf(0.f,v);
       if(p_.add_residual)v=__half2float(__hadd_rn(__float2half_rn(v),p_.residual[int64_t(pos)*p_.k+o]));
     }
     out[t]=cutlass::half_t(v);
   }
   if constexpr((Variant>=11&&Variant<=14)||(Variant>=17&&Variant<=20)||(Variant>=23&&Variant<=26)||(Variant>=31&&Variant<=34)||(Variant>=37&&Variant<=40)||(Variant>=43&&Variant<=46)) {
     static_assert(Access==4,"joint producer currently uses four-wide output");
     if(oc<p_.k) {
       int code[4];
       #pragma unroll
       for(int t=0;t<4;++t) {
         float scaled=__fadd_rn(__fmul_rn(float(out[t]),p_.next_multiplier[oc+t]),p_.next_activation[1]);
         code[t]=__float2int_rn(fminf(float((1<<p_.next_bits)-1),fmaxf(0.f,scaled)))-(1<<(p_.next_bits-1));
       }
       if(p_.next_bits==4) {
         uint16_t packed=uint16_t((code[0]&15)|((code[1]&15)<<4)|((code[2]&15)<<8)|((code[3]&15)<<12));
         *reinterpret_cast<uint16_t*>(p_.next_packed+(int64_t(pos)*p_.next_cp+oc)/2)=packed;
       } else {
         uint32_t packed=uint32_t(code[0]&255)|(uint32_t(code[1]&255)<<8)|(uint32_t(code[2]&255)<<16)|(uint32_t(code[3]&255)<<24);
         *reinterpret_cast<uint32_t*>(p_.next_packed+int64_t(pos)*p_.next_cp+oc)=packed;
       }
#ifdef QUANTDEMOIRE_TWO_SUCCESSOR
       int future_code[4];
       #pragma unroll
       for(int t=0;t<4;++t) {
         float scaled=__fadd_rn(__fmul_rn(float(out[t]),p_.future_multiplier[p_.future_offset+oc+t]),p_.future_activation[1]);
         future_code[t]=__float2int_rn(fminf(float((1<<p_.future_bits)-1),fmaxf(0.f,scaled)))-(1<<(p_.future_bits-1));
       }
       if(p_.future_bits==4) {
         uint16_t packed=uint16_t((future_code[0]&15)|((future_code[1]&15)<<4)|((future_code[2]&15)<<8)|((future_code[3]&15)<<12));
         *reinterpret_cast<uint16_t*>(p_.future_packed+(int64_t(pos)*p_.k+oc)/2)=packed;
       } else {
         uint32_t packed=uint32_t(future_code[0]&255)|(uint32_t(future_code[1]&255)<<8)|(uint32_t(future_code[2]&255)<<16)|(uint32_t(future_code[3]&255)<<24);
         *reinterpret_cast<uint32_t*>(p_.future_packed+int64_t(pos)*p_.k+oc)=packed;
       }
#endif
       if(p_.next_planar) {
         if constexpr(Variant==13||Variant==14||Variant==19||Variant==20||Variant==25||Variant==26||Variant==33||Variant==34||Variant==39||Variant==40||Variant==45||Variant==46) {
           extern __shared__ int joint_storage[];
           auto tile=reinterpret_cast<int8_t*>(joint_storage)+p_.shared_offset+64*33*sizeof(float);
           #pragma unroll
           for(int t=0;t<4;++t)tile[(pos-int(blockIdx.x)*64)*33+oc-int(blockIdx.y)*32+t]=int8_t(code[t]);
         } else {
         int batch=0,sp=pos;
         if(p_.n!=1)p_.pq_div(batch,sp,pos);
         #pragma unroll
         for(int t=0;t<4;++t) {
           int target=p_.selection[oc+t];
           if(target>=0&&target<p_.next_sc)p_.next_planar[(int64_t(batch)*p_.next_sc+target)*p_.p*p_.q+sp]=int8_t(code[t]);
         }
         }
       }
     }
   }
 }
};

template<int Bits>struct JointTraits;
template<>struct JointTraits<4>{using Input=cutlass::int4b_t;static constexpr int K=128,Align=32;};
template<>struct JointTraits<8>{using Input=int8_t;static constexpr int K=64,Align=16;};
template<int Bits,int Access,int Warps,bool Sparse,int Variant,bool SharedW4Lookup=false>
using JointKernelDefault=typename cutlass::conv::kernel::DefaultConv2dFpropWithBroadcast<
 typename JointTraits<Bits>::Input,Layout,typename JointTraits<Bits>::Input,Layout,
 cutlass::half_t,Layout,int32_t,cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
 cutlass::gemm::GemmShape<Warps==8?64:128,Variant==6?64:32,JointTraits<Bits>::K*(Warps==8&&Variant!=6?2:1)>,
 cutlass::gemm::GemmShape<Warps==8?16:32,Warps==8&&Variant!=6?16:32,JointTraits<Bits>::K*(Warps==8&&Variant!=6?2:1)>,
 cutlass::gemm::GemmShape<16,8,JointTraits<Bits>::K/2>,JointOutput<Access,Warps,Sparse,Variant,SharedW4Lookup>,
 cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,Warps==8&&(Variant==0||Variant>=4)?2:3,cutlass::arch::OpMultiplyAddSaturate,
 ((Variant>=29&&Variant<=34)||(Variant>=41&&Variant<=46))?cutlass::conv::IteratorAlgorithm::kFewChannels:cutlass::conv::IteratorAlgorithm::kOptimized,cutlass::conv::StrideSupport::kStrided,
 JointTraits<Bits>::Align,JointTraits<Bits>::Align>::Kernel;

// CUTLASS's Stages=2 default selects synchronous pipelining. Obtain its
// asynchronous iterator/policy types from Stages=3, then explicitly use a
// two-stage cp.async main loop with the same validated large K tile.
template<class Kernel,bool Async>struct SelectJointKernel {using Type=Kernel;};
template<class Kernel>struct SelectJointKernel<Kernel,true> {
 using Old=typename Kernel::Mma;
 using Mma=cutlass::conv::threadblock::ImplicitGemmMultistage<typename Old::Shape,
   typename Old::IteratorA,typename Old::SmemIteratorA,Old::kCacheOpA,
   typename Old::IteratorB,typename Old::SmemIteratorB,Old::kCacheOpB,typename Old::Policy,2>;
 using Type=cutlass::conv::kernel::ImplicitGemmConvolutionWithFusedEpilogue<Mma,
   typename Kernel::Epilogue,typename Kernel::ThreadblockSwizzle,
   Kernel::kConvolutionalOperator,typename Kernel::ConvProblemSize>;
};
/***************************************************************************************************
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

template<class Old> class RetainedPointwiseMma : public Old {
 public:
 using Base=typename Old::Base;
 using Operator=typename Old::Operator;
 using Policy=typename Old::Policy;
 using WarpFragmentA=typename Operator::FragmentA;
 using WarpFragmentB=typename Operator::FragmentB;
 using FragmentA=typename Old::FragmentA;
 using FragmentB=typename Old::FragmentB;
 using TransformA=typename Old::TransformA;
 using TransformB=typename Old::TransformB;
 using FragmentC=typename Old::FragmentC;
 using IteratorA=typename Old::IteratorA;
 using IteratorB=typename Old::IteratorB;
 CUTLASS_DEVICE RetainedPointwiseMma(typename Old::SharedStorage& storage,int thread,int warp,int lane)
  :Old(storage,thread,warp,lane){}
  CUTLASS_DEVICE
  void operator()(
    int gemm_k_iterations,                            ///< number of iterations of the mainloop
    FragmentC &accum,                                 ///< destination accumulator tile
    IteratorA iterator_A,                             ///< iterator over A operand in global memory
    IteratorB iterator_B,                             ///< iterator over B operand in global memory
    FragmentC const &src_accum,                       ///< source accumulator tile
    int gemm_k_iterations_per_channel = 0,             ///< number of iterations per channel
    TransformA transform_A = TransformA(),            ///< transformation applied to A fragment
    TransformB transform_B = TransformB()) {          ///< transformation applied to B fragment

    //
    // Prologue
    //

    // Perform accumulation in the 'd' output operand
    accum = src_accum;

    FragmentA tb_frag_A;
    FragmentB tb_frag_B;

    tb_frag_A.clear();
    tb_frag_B.clear();

    // The last kblock is loaded in the prolog
    iterator_A.load(tb_frag_A);
    iterator_B.load(tb_frag_B);

    ++iterator_A;
    ++iterator_B;

    this->smem_iterator_A_.store(transform_A(tb_frag_A));
    this->smem_iterator_B_.store(transform_B(tb_frag_B));

    ++this->smem_iterator_A_;
    ++this->smem_iterator_B_;

    __syncthreads();

    // Pair of fragments used to overlap shared memory loads and math instructions
    WarpFragmentA warp_frag_A[2];
    WarpFragmentB warp_frag_B[2];

    this->warp_tile_iterator_A_.set_kgroup_index(0);
    this->warp_tile_iterator_B_.set_kgroup_index(0);

    this->warp_tile_iterator_A_.load(warp_frag_A[0]);
    this->warp_tile_iterator_B_.load(warp_frag_B[0]);

    ++this->warp_tile_iterator_A_;
    ++this->warp_tile_iterator_B_;

    Operator warp_mma;

    int smem_write_stage_idx = 1;

    // Issue loads during the first warp-level matrix multiply-add *AFTER* issuing 
    // shared memory loads (which have the tightest latency requirement).

    //
    // Mainloop
    //

    // Note: The main loop does not support Base::kWarpGemmIterations == 2.
    CUTLASS_GEMM_LOOP
    for (; gemm_k_iterations > 0; --gemm_k_iterations) {
      //
      // Loop over GEMM K dimension
      //

      CUTLASS_PRAGMA_UNROLL
      for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations; ++warp_mma_k) {

        // Load warp-level tiles from shared memory, wrapping to k offset if this is the last group
        // as the case may be.

        if (warp_mma_k == Base::kWarpGemmIterations - 1) {

          // Write fragments to shared memory
          // The final prefetched tile is never consumed. Preserve both A stages
          // for pointwise CSR work after the integer accumulators are complete.
          if (gemm_k_iterations > 1) {
            this->smem_iterator_A_.store(transform_A(tb_frag_A));
            this->smem_iterator_B_.store(transform_B(tb_frag_B));
          }

          __syncthreads();
          
          ++this->smem_iterator_A_;
          ++this->smem_iterator_B_;

          // Add negative offsets to return iterators to the 'start' of the circular buffer in shared memory
          if (smem_write_stage_idx == 1) {
            this->smem_iterator_A_.add_tile_offset({0, -Base::kStages});
            this->smem_iterator_B_.add_tile_offset({-Base::kStages, 0});
          }
          else {
            this->warp_tile_iterator_A_.add_tile_offset(
                {0, -Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations});
            this->warp_tile_iterator_B_.add_tile_offset(
                {-Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations,
                 0});
          }

          smem_write_stage_idx ^= 1;
        }

        this->warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
        this->warp_tile_iterator_B_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
        
        this->warp_tile_iterator_A_.load(warp_frag_A[(warp_mma_k + 1) % 2]);
        this->warp_tile_iterator_B_.load(warp_frag_B[(warp_mma_k + 1) % 2]);

        ++this->warp_tile_iterator_A_;
        ++this->warp_tile_iterator_B_;

        if (warp_mma_k == 0) {

          iterator_A.load(tb_frag_A);
          iterator_B.load(tb_frag_B);
    
          ++iterator_A;
          ++iterator_B;
        }

        warp_mma(accum, warp_frag_A[warp_mma_k % 2],
                 warp_frag_B[warp_mma_k % 2], accum);
      }
    }

  }
};
template<class Kernel,bool Active>struct SelectRetainedKernel {using Type=Kernel;};
template<class Kernel>struct SelectRetainedKernel<Kernel,true> {
 using Mma=RetainedPointwiseMma<typename Kernel::Mma>;
 using Type=cutlass::conv::kernel::ImplicitGemmConvolutionWithFusedEpilogue<Mma,
   typename Kernel::Epilogue,typename Kernel::ThreadblockSwizzle,
   Kernel::kConvolutionalOperator,typename Kernel::ConvProblemSize>;
};

// Keep the proven FewChannels pipeline and replace only filter addressing.
template<class Kernel,bool Active>struct SelectPackedFilterKernel {using Type=Kernel;};
template<class Kernel>struct SelectPackedFilterKernel<Kernel,true> {
 using Old=typename Kernel::Mma;
 using OldAccess=typename Old::IteratorB::TileAccessIterator;
 using Access=cutlass::conv::threadblock::QuantDemoirePackedFilterIterator<
   typename OldAccess::Shape,typename OldAccess::Element,typename OldAccess::Layout,
   typename OldAccess::ThreadMap,typename OldAccess::AccessType>;
 using Iterator=cutlass::conv::threadblock::TileIterator<Access>;
 using Mma=cutlass::conv::threadblock::ImplicitGemmPipelined<typename Old::Shape,
   typename Old::IteratorA,typename Old::SmemIteratorA,Iterator,typename Old::SmemIteratorB,
   typename Old::ElementC,typename Old::LayoutC,typename Old::Policy,
   typename Old::TransformA,typename Old::TransformB>;
 using Type=cutlass::conv::kernel::ImplicitGemmConvolutionWithFusedEpilogue<Mma,
   typename Kernel::Epilogue,typename Kernel::ThreadblockSwizzle,
   Kernel::kConvolutionalOperator,typename Kernel::ConvProblemSize>;
};

template<int Bits,int Access,int Warps,bool Sparse,int Variant,bool SharedW4Lookup=false>
using JointKernel=typename SelectPackedFilterKernel<typename SelectRetainedKernel<typename SelectJointKernel<JointKernelDefault<Bits,Access,Warps,Sparse,Variant,SharedW4Lookup>,(Variant>0&&Variant<4)>::Type,(Variant>=21&&Variant<=26)>::Type,((Variant>=29&&Variant<=34)||(Variant>=41&&Variant<=46))>::Type;

template<int Access,int Warps,bool Sparse,int Variant> CUTLASS_DEVICE cutlass::MatrixCoord joint_coord(int ordinal) {
 using TM=typename JointKernel<4,Access,Warps,Sparse,Variant>::Epilogue::OutputTileIterator::ThreadMap;
 auto initial=TM::initial_offset(threadIdx.x);
 int frag=ordinal%TM::Iterations::kCount,iter=ordinal/TM::Iterations::kCount;
 int col=frag%TM::Iterations::kColumn,u=frag/TM::Iterations::kColumn;
 int row=u%TM::Iterations::kRow;u/=TM::Iterations::kRow;
 int group=u%TM::Iterations::kGroup,cluster=u/TM::Iterations::kGroup;
 int advance=iter*TM::Shape::kRow;
 advance+=(iter/TM::Count::kRow)*(TM::Shape::kGroup-1)*TM::Shape::kRow*TM::Count::kRow;
 advance+=(iter/(TM::Count::kRow*TM::Count::kGroup))*TM::Count::kGroup*TM::Shape::kGroup*TM::Count::kRow*TM::Shape::kRow;
 advance+=(iter/(TM::Count::kRow*TM::Count::kGroup*TM::Count::kCluster))*TM::Shape::kGroup*TM::Shape::kRow*TM::Shape::kCluster*TM::Shape::kTile;
 return {int(blockIdx.x)*(Warps==8?64:128)+initial.row()+advance+row*TM::Delta::kRow+group*TM::Delta::kGroup+cluster*TM::Delta::kCluster,
         int(blockIdx.y)*(Variant==6?64:32)+initial.column()+col*TM::Delta::kColumn};
}


// For pointwise convolution with only one channel tile, the synchronous
// two-stage main loop leaves its original activation tile in stage zero.
// Read all required operands before reusing that storage for the CSR output.
// No FP32 sparse accumulator is live during the integer main loop.
template<int Bits,int Access,bool Balanced> CUTLASS_DEVICE void joint_shared_pointwise(JointParams const& p,float* tile) {
 using Kernel=JointKernel<Bits,Access,8,true,9>;
 using Storage=typename Kernel::Mma::SharedStorage;
 using Input=typename JointTraits<Bits>::Input;
 extern __shared__ int joint_storage[];
 auto& storage=*reinterpret_cast<Storage*>(joint_storage);
 __syncthreads();
 auto input=storage.operand_A_ref();
 int lane=threadIdx.x&31,warp=threadIdx.x>>5;
 float delta=float(p.center)-p.activation[1],scale=p.activation[0],acc[4][2]={};
 #pragma unroll
 for(int group=0;group<4;++group) {
   int local=warp+group*8;
   if constexpr(Balanced)local=p.row[p.k+1+int(blockIdx.y)*32+local];
   int oc=int(blockIdx.y)*32+local;
   if(oc<p.k)for(int j=p.row[oc];j<p.row[oc+1];++j) {
     // The caller supplies original channel indices for this variant.
     int channel=p.location[j]&4095;float weight=__half2float(p.value[j]);
     #pragma unroll
     for(int t=0;t<2;++t) {
       int code=int(Input(input.at(cutlass::MatrixCoord(lane+t*32,channel))));
       float operand=__half2float(__float2half_rn((float(code)+delta)*scale));
       acc[group][t]=fmaf(operand,weight,acc[group][t]);
     }
   }
 }
 // CSR output overlaps old MMA input storage: every reader must finish first.
 __syncthreads();
 #pragma unroll
 for(int group=0;group<4;++group) {
   int local=warp+group*8;
   if constexpr(Balanced)local=p.row[p.k+1+int(blockIdx.y)*32+local];
   #pragma unroll
   for(int t=0;t<2;++t)tile[(lane+t*32)*33+local]=acc[group][t];
 }
 __syncthreads();
}

// Bound the eight-warp launch to four resident blocks where resources permit.
// The compiler may rematerialize values or spill; always measure this variant.
CUTLASS_DEVICE void store_joint_next_planar(JointParams const& p) {
 if(!p.next_planar)return;
 __syncthreads();
 extern __shared__ int joint_storage[];
 auto tile=reinterpret_cast<int8_t*>(joint_storage)+p.shared_offset+64*33*sizeof(float);
 int lane=threadIdx.x&31,warp=threadIdx.x>>5;
 #pragma unroll
 for(int group=0;group<4;++group) {
   int oc=int(blockIdx.y)*32+warp+group*8,target=oc<p.k?p.selection[oc]:-1;
   if(target>=0&&target<p.next_sc) {
     #pragma unroll
     for(int t=0;t<2;++t) {
       int pos=int(blockIdx.x)*64+lane+t*32;
       if(pos<p.n*p.p*p.q) {
         int batch=0,sp=pos;
         if(p.n!=1)p.pq_div(batch,sp,pos);
         p.next_planar[(int64_t(batch)*p.next_sc+target)*p.p*p.q+sp]=tile[(lane+t*32)*33+warp+group*8];
       }
     }
   }
 }
}

template<class Kernel>
__global__ __launch_bounds__(256,4) void joint_limited_kernel(typename Kernel::Params params) {
 extern __shared__ int storage[];
 Kernel op;op(params,*reinterpret_cast<typename Kernel::SharedStorage*>(storage));
 if constexpr(Kernel::EpilogueOutputOp::kTransposeNext)store_joint_next_planar(params.output_op.data);
}

// A wider N tile permits a smaller reduction tile with the default 256-thread
// loaders. Allocate enough dynamic memory for the epilogue and CSR tile,
// including when their combined size exceeds main-loop storage.
template<class Kernel,int Warps,bool Sparse,int Variant>constexpr int joint_shared_bytes() {
 constexpr int main=sizeof(typename Kernel::SharedStorage);
 constexpr int merged=sizeof(typename Kernel::Epilogue::SharedStorage)+(Warps==8?64:128)*(Variant==6?65:33)*sizeof(float);
 return Variant>=2&&Sparse&&merged>main?merged:main;
}

template<int Bits,int Access,int Warps,bool Sparse,int Variant,bool SharedW4Lookup=false>
void run_joint(torch::Tensor x,torch::Tensor weight,torch::Tensor scale,torch::Tensor out,int cp,JointParams p,cudaStream_t stream) {
 using Kernel=JointKernel<Bits,Access,Warps,Sparse,Variant,SharedW4Lookup>;
 constexpr int SharedBytes=joint_shared_bytes<Kernel,Warps,Sparse,Variant>();
 if constexpr(SharedW4Lookup) {
   static_assert(Bits==4&&Warps==8&&Sparse&&(Variant==36||Variant==38||Variant==40||Variant==42||Variant==44||Variant==46),"unsupported shared W4 lookup specialization");
   static_assert(SharedBytes>=sizeof(typename Kernel::Epilogue::SharedStorage)
     +64*33*(sizeof(float)+sizeof(int8_t))+8*16*sizeof(float),"shared operand table exceeds reusable storage");
 }
 if constexpr(Variant==13||Variant==14||Variant==19||Variant==20||Variant==25||Variant==26||Variant==33||Variant==34||Variant==39||Variant==40||Variant==45||Variant==46)static_assert(SharedBytes>=sizeof(typename Kernel::Epilogue::SharedStorage)+64*33*(sizeof(float)+sizeof(int8_t)),"quantized prefix tile exceeds shared storage");
 p.shared_offset=sizeof(typename Kernel::Epilogue::SharedStorage);
 if constexpr(Variant>=2&&Sparse) {
   static_assert(SharedBytes>=sizeof(typename Kernel::Epilogue::SharedStorage)+(Warps==8?64:128)*(Variant==6?65:33)*sizeof(float),"CSR tile does not fit the reusable main-loop storage");
   static_assert(sizeof(typename Kernel::Epilogue::SharedStorage)%16==0,"unaligned reused CSR tile");
 }
 using Op=cutlass::conv::device::ImplicitGemmConvolution<Kernel>;using Input=typename JointTraits<Bits>::Input;
 cutlass::Tensor4DCoord xs(p.n,p.h,p.w,cp),ws(p.k,p.r,p.r,cp),ys(p.n,p.p,p.q,p.k);
 cutlass::conv::Conv2dProblemSize problem(xs,ws,{p.pad,p.pad,p.pad,p.pad},{p.stride,p.stride},{p.dil,p.dil},ys,cutlass::conv::Mode::kCrossCorrelation,1);
 typename Op::Arguments args(problem,{reinterpret_cast<Input*>(x.data_ptr()),Layout::packed(xs)},
  {reinterpret_cast<Input*>(weight.data_ptr()),Layout::packed(ws)},{nullptr,Layout::Stride(0)},
  {reinterpret_cast<cutlass::half_t*>(out.data_ptr()),Layout::packed(ys)},typename Op::EpilogueOutputOp::Params(p),
  cutlass::conv::SplitKMode::kSerial,static_cast<void*>(scale.data_ptr<float>()));
 Op op;TORCH_CHECK(op.can_implement(args)==cutlass::Status::kSuccess,"unsupported cooperative joint convolution");
 if constexpr(Variant>=3) {
   TORCH_CHECK(op.get_workspace_size(args)==0,"limited joint launch requires no split-K workspace");
   static thread_local int configured_device=-1;
   if(configured_device!=x.get_device()) {
     C10_CUDA_CHECK(cudaFuncSetAttribute(joint_limited_kernel<Kernel>,cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes));
     configured_device=x.get_device();
   }
   typename Kernel::Params launch_params(args,nullptr);
   // The fused-epilogue wrapper in CUTLASS 3.5.1 omits the algorithm argument.
   // FewChannels advances flattened R*S*C and requires this iteration count.
   if constexpr((Variant>=29&&Variant<=34)||(Variant>=41&&Variant<=46))
     launch_params.gemm_k_iterations=cutlass::conv::implicit_gemm_k_iterations(
       cutlass::conv::Operator::kFprop,Kernel::ThreadblockShape::kK,problem,
       cutlass::conv::IteratorAlgorithm::kFewChannels);
   typename Kernel::ThreadblockSwizzle swizzle;
   joint_limited_kernel<Kernel><<<swizzle.get_grid_shape(launch_params.grid_tiled_shape),Kernel::kThreadCount,SharedBytes,stream>>>(launch_params);
 } else {
 // Include the statically allocated CSR tile when opting into shared memory.
 static thread_local int configured_device=-1;
 if(configured_device!=x.get_device()) {
   C10_CUDA_CHECK(cudaFuncSetAttribute(cutlass::Kernel<Kernel>,cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes));
   configured_device=x.get_device();
 }
 auto size=op.get_workspace_size(args);auto workspace=torch::empty({int64_t(size)},x.options().dtype(at::kByte));
 auto status=op.initialize(args,size?workspace.data_ptr():nullptr,stream);
 TORCH_CHECK(status==cutlass::Status::kSuccess,"joint initialize failed");
 status=op(stream);TORCH_CHECK(status==cutlass::Status::kSuccess,"joint run failed");
 }
}
// Host selection keeps the measured spatial specialization separate from all
// pointwise, W8, dense-only, unbalanced, and unpacked-metadata kernels.
template<int Bits,int Access,int Warps,bool Sparse,int Variant>
void run_joint_selected(torch::Tensor x,torch::Tensor weight,torch::Tensor scale,
 torch::Tensor out,int cp,JointParams p,cudaStream_t stream) {
 if constexpr(Bits==4&&Warps==8&&Sparse&&(Variant==36||Variant==38||Variant==40||Variant==42||Variant==44||Variant==46)) {
   if(p.r>1){run_joint<Bits,Access,Warps,Sparse,Variant,true>(x,weight,scale,out,cp,p,stream);return;}
 }
 run_joint<Bits,Access,Warps,Sparse,Variant,false>(x,weight,scale,out,cp,p,stream);
}
template<int Bits,int Access,int Warps,bool Sparse,int Variant,bool SharedW4Lookup=false>
std::vector<int64_t> joint_resource_row() {
 using Kernel=JointKernel<Bits,Access,Warps,Sparse,Variant,SharedW4Lookup>;
 constexpr int SharedBytes=joint_shared_bytes<Kernel,Warps,Sparse,Variant>();
 if constexpr(SharedW4Lookup) {
   static_assert(Bits==4&&Warps==8&&Sparse&&(Variant==36||Variant==38||Variant==40||Variant==42||Variant==44||Variant==46),"unsupported shared W4 lookup specialization");
   static_assert(SharedBytes>=sizeof(typename Kernel::Epilogue::SharedStorage)
     +64*33*(sizeof(float)+sizeof(int8_t))+8*16*sizeof(float),"shared operand table exceeds reusable storage");
 }
 if constexpr(Variant==13||Variant==14||Variant==19||Variant==20||Variant==25||Variant==26||Variant==33||Variant==34||Variant==39||Variant==40||Variant==45||Variant==46)static_assert(SharedBytes>=sizeof(typename Kernel::Epilogue::SharedStorage)+64*33*(sizeof(float)+sizeof(int8_t)),"quantized prefix tile exceeds shared storage");
 cudaFuncAttributes attr{};int blocks=0;
 if constexpr(Variant>=3) {
 C10_CUDA_CHECK(cudaFuncSetAttribute(joint_limited_kernel<Kernel>,cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes));
 C10_CUDA_CHECK(cudaFuncGetAttributes(&attr,joint_limited_kernel<Kernel>));
 C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,joint_limited_kernel<Kernel>,Warps*32,SharedBytes));
 } else {
 C10_CUDA_CHECK(cudaFuncSetAttribute(cutlass::Kernel<Kernel>,cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes));
 C10_CUDA_CHECK(cudaFuncGetAttributes(&attr,cutlass::Kernel<Kernel>));
 C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,cutlass::Kernel<Kernel>,Warps*32,SharedBytes));
 }
 return {attr.numRegs,int64_t(attr.sharedSizeBytes),SharedBytes,
   sizeof(typename Kernel::Epilogue::SharedStorage),int64_t(attr.localSizeBytes),Warps*32,blocks};
}
} // namespace

