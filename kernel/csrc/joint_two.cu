#define QUANTDEMOIRE_TWO_SUCCESSOR 1
#include "joint_common.cuh"

std::vector<torch::Tensor> joint_next_two(torch::Tensor x,torch::Tensor planar,torch::Tensor weight,torch::Tensor scale,
 torch::Tensor row,torch::Tensor location,torch::Tensor offsets,torch::Tensor value,torch::Tensor sums,
 torch::Tensor bias,torch::Tensor activation,torch::Tensor residual,int64_t bits,int64_t cp,int64_t k,
 int64_t r,int64_t stride,int64_t pad,int64_t dil,bool relu,bool sparse,bool add_residual,int64_t warps,int64_t variant,
 torch::Tensor next_multiplier,torch::Tensor next_activation,torch::Tensor selection,
 int64_t next_bits,int64_t next_cp,int64_t sc,bool next_planar,int64_t strategy,bool balance,bool retain_shared,bool few_channels,bool packed_metadata,torch::Tensor future_multiplier,
 torch::Tensor future_activation,int64_t future_bits,int64_t future_offset) {
 TORCH_CHECK(bits==4||bits==8,"invalid joint bit width");
 TORCH_CHECK(r>1&&packed_metadata&&balance,"prototype requires spatial balanced packed-metadata CSR");
 TORCH_CHECK((future_bits==4||future_bits==8)&&future_offset>=0&&k%(future_bits==4?32:16)==0,"invalid future bit width or segment");
 TORCH_CHECK(warps==8&&sparse&&!add_residual&&(variant==8||variant==9),"unsupported joint producer");
 TORCH_CHECK(strategy==0||strategy==1,"invalid joint producer store strategy");
 TORCH_CHECK((next_bits==4||next_bits==8)&&k%16==0,"invalid next quantization bit width or prefix");
 TORCH_CHECK(warps==4||warps==8,"joint warps must be 4 or 8");
 TORCH_CHECK(variant>=0&&variant<=9,"joint variant must be between 0 and 9");
 TORCH_CHECK(!balance||(warps==8&&(variant==8||variant==9)),"CSR balancing requires eight-warp variant 8 or 9");
 bool balanced=balance;bool metadata=packed_metadata;bool few=few_channels&&warps==8&&(variant==8||variant==9)&&r>1&&cp<=(bits==4?64:48);
 bool retain_input=retain_shared&&variant==9&&warps==8&&r==1&&stride==1&&pad==0&&cp<=(bits==4?512:256);
 bool shared_input=(variant==9&&warps==8&&r==1&&stride==1&&pad==0&&cp<=(bits==4?256:128))||retain_input;
 TORCH_CHECK(!metadata||(warps==8&&(variant==8||variant==9)&&!shared_input),"packed metadata requires a regular eight-warp CSR input");
 TORCH_CHECK(planar.is_cuda()&&planar.dim()==4&&planar.scalar_type()==at::kChar&&(!sparse||shared_input||planar.is_contiguous()),"invalid joint planar input");
 auto check=[&](torch::Tensor t,at::ScalarType type){TORCH_CHECK(t.device()==planar.device()&&t.scalar_type()==type&&t.is_contiguous(),"invalid joint buffer");};
 check(x,at::kByte);check(weight,at::kByte);check(scale,at::kFloat);check(row,at::kInt);check(location,at::kInt);
 check(offsets,at::kLong);check(value,at::kHalf);check(sums,at::kFloat);check(bias,at::kFloat);check(activation,at::kFloat);
 check(future_multiplier,at::kFloat);check(future_activation,at::kFloat);
 TORCH_CHECK(future_multiplier.numel()>=future_offset+k&&future_activation.numel()==2,"invalid future quantizer");
 check(next_multiplier,at::kFloat);check(next_activation,at::kFloat);check(selection,at::kInt);
 TORCH_CHECK(next_multiplier.numel()>k&&selection.numel()==next_multiplier.numel()&&next_activation.numel()==2&&next_cp>=next_multiplier.numel()&&next_cp%(next_bits==4?32:16)==0&&sc>0&&sc<=next_multiplier.numel(),"invalid next quantizer");
 int n=planar.size(0),c=planar.size(1),h=planar.size(2),w=planar.size(3);
 TORCH_CHECK(k>0&&r>0&&stride>0&&dil>0&&pad>=0&&cp>=c&&cp%(bits==4?32:16)==0&&k%4==0,"invalid joint dimensions");
 few=few&&int64_t(n)*c*h*w<2147483647;
 TORCH_CHECK(!metadata||int64_t(n)*c*h*w<2147483647,"packed metadata requires 32-bit element offsets");
 int p=(h+2*pad-dil*(r-1)-1)/stride+1,q=(w+2*pad-dil*(r-1)-1)/stride+1;
 TORCH_CHECK(p>0&&q>0&&x.numel()==int64_t(n)*h*w*cp*bits/8&&weight.numel()==k*r*r*cp*bits/8,"invalid joint packed size");
 TORCH_CHECK(row.numel()==k+1+(balanced?(k+31)/32*32:0)&&location.numel()==value.numel()&&offsets.numel()==value.numel()&&sums.numel()==k*r*r+k&&bias.numel()==k&&scale.numel()==k&&activation.numel()==2,"invalid joint metadata");
 if(add_residual)TORCH_CHECK(residual.is_cuda()&&residual.device()==planar.device()&&residual.scalar_type()==at::kHalf&&residual.is_contiguous(at::MemoryFormat::ChannelsLast)&&residual.dim()==4&&residual.size(0)==n&&residual.size(1)==k&&residual.size(2)==p&&residual.size(3)==q,"invalid joint residual");
 c10::cuda::CUDAGuard guard(x.device());auto stream=c10::cuda::getCurrentCUDAStream(x.get_device());
 auto out=torch::empty({n,k,p,q},x.options().dtype(at::kHalf).memory_format(at::MemoryFormat::ChannelsLast));
 JointParams params{planar.data_ptr<int8_t>(),row.data_ptr<int>(),location.data_ptr<int>(),offsets.data_ptr<int64_t>(),reinterpret_cast<half*>(value.data_ptr()),sums.data_ptr<float>(),bias.data_ptr<float>(),activation.data_ptr<float>(),add_residual?reinterpret_cast<half*>(residual.data_ptr()):nullptr,n,c,h,w,int(k),p,q,int(r),int(stride),int(pad),int(dil),1<<(bits-1),relu,add_residual,0};
 params.q_div=cutlass::FastDivmod(q);params.pq_div=cutlass::FastDivmod(p*q);
 auto packed=torch::empty({int64_t(n)*p*q*next_cp*next_bits/8},x.options().dtype(at::kByte));
 auto auxiliary=next_planar?torch::empty({n,sc,p,q},x.options().dtype(at::kChar).memory_format(at::MemoryFormat::Contiguous)):
   torch::empty({1},x.options().dtype(at::kChar)).as_strided({n,sc,p,q},{0,0,0,0});
 params.next_multiplier=next_multiplier.data_ptr<float>();params.next_activation=next_activation.data_ptr<float>();
 params.selection=selection.data_ptr<int>();params.next_packed=packed.data_ptr<uint8_t>();
 params.next_planar=next_planar?auxiliary.data_ptr<int8_t>():nullptr;
 params.next_bits=next_bits;params.next_cp=next_cp;params.next_sc=sc;
 auto future=torch::empty({int64_t(n)*p*q*k*future_bits/8},x.options().dtype(at::kByte));
 params.future_multiplier=future_multiplier.data_ptr<float>();params.future_activation=future_activation.data_ptr<float>();
 params.future_packed=future.data_ptr<uint8_t>();params.future_bits=future_bits;params.future_offset=future_offset;
 #define RUN(B,A,W,S,V) run_joint_selected<B,A,W,S,V>(x,weight,scale,out,cp,params,stream)
 if(few) {
   if(strategy==0) {if(bits==4){RUN(4,4,8,true,44);}else{RUN(8,4,8,true,44);}}
   else {if(bits==4){RUN(4,4,8,true,46);}else{RUN(8,4,8,true,46);}}
 } else {
   if(strategy==0) {if(bits==4){RUN(4,4,8,true,38);}else{RUN(8,4,8,true,38);}}
   else {if(bits==4){RUN(4,4,8,true,40);}else{RUN(8,4,8,true,40);}}
 }
 #undef RUN
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {out,packed,auxiliary,future};
}
