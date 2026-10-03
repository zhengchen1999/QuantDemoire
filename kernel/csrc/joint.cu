#include "joint_common.cuh"

torch::Tensor joint_conv(torch::Tensor x,torch::Tensor planar,torch::Tensor weight,torch::Tensor scale,
 torch::Tensor row,torch::Tensor location,torch::Tensor offsets,torch::Tensor value,torch::Tensor sums,
 torch::Tensor bias,torch::Tensor activation,torch::Tensor residual,int64_t bits,int64_t cp,int64_t k,
 int64_t r,int64_t stride,int64_t pad,int64_t dil,bool relu,bool sparse,bool add_residual,int64_t warps,int64_t variant,bool balance,bool retain_shared,bool few_channels,bool packed_metadata) {
 TORCH_CHECK(bits==4||bits==8,"invalid joint bit width");
 TORCH_CHECK(warps==4||warps==8,"joint warps must be 4 or 8");
 TORCH_CHECK(variant>=0&&variant<=9,"joint variant must be between 0 and 9");
 TORCH_CHECK(!balance||(warps==8&&(variant==8||variant==9)),"CSR balancing requires eight-warp variant 8 or 9");
 bool balanced=balance;bool metadata=packed_metadata;
 bool few=few_channels&&warps==8&&(variant==8||variant==9)&&r>1&&cp<=(bits==4?64:48);
 bool retain_input=retain_shared&&variant==9&&warps==8&&r==1&&stride==1&&pad==0&&cp<=(bits==4?512:256);
 bool shared_input=(variant==9&&warps==8&&r==1&&stride==1&&pad==0&&cp<=(bits==4?256:128))||retain_input;
 TORCH_CHECK(!metadata||(warps==8&&(variant==8||variant==9)&&!shared_input),"packed metadata requires a regular eight-warp CSR input");
 TORCH_CHECK(planar.is_cuda()&&planar.dim()==4&&planar.scalar_type()==at::kChar&&(!sparse||shared_input||planar.is_contiguous()),"invalid joint planar input");
 auto check=[&](torch::Tensor t,at::ScalarType type){TORCH_CHECK(t.device()==planar.device()&&t.scalar_type()==type&&t.is_contiguous(),"invalid joint buffer");};
 check(x,at::kByte);check(weight,at::kByte);check(scale,at::kFloat);check(row,at::kInt);check(location,at::kInt);
 check(offsets,at::kLong);check(value,at::kHalf);check(sums,at::kFloat);check(bias,at::kFloat);check(activation,at::kFloat);
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
 #define RUN(B,A,W,S,V) run_joint_selected<B,A,W,S,V>(x,weight,scale,out,cp,params,stream)
 #define BITS4(S) if(bits==4){if(k%8==0){RUN(4,8,4,S,0);}else{RUN(4,4,4,S,0);}}else{if(k%8==0){RUN(8,8,4,S,0);}else{RUN(8,4,4,S,0);}}
 #define WARP8(S,V) if(bits==4){RUN(4,4,8,S,V);}else{RUN(8,4,8,S,V);}
 #define WARPS(S) if(warps==4){BITS4(S);}else if(variant==0){WARP8(S,0);}else if(variant==1){WARP8(S,1);}else if(variant==2){WARP8(S,2);}else if(variant==3){WARP8(S,3);}else if(few){if(metadata){if(balanced){WARP8(S,42);}else{WARP8(S,41);}}else if(balanced){WARP8(S,30);}else{WARP8(S,29);}}else if(metadata){if(balanced){WARP8(S,36);}else{WARP8(S,35);}}else if(retain_input){if(balanced){WARP8(S,22);}else{WARP8(S,21);}}else if(balanced){if(shared_input){WARP8(S,16);}else if(int64_t(n)*c*h*w<2147483647){WARP8(S,15);}else{WARP8(S,4);}}else if(variant==9&&r==1&&stride==1&&pad==0&&cp<=(bits==4?256:128)){WARP8(S,9);}else if(variant>=8&&int64_t(n)*c*h*w<2147483647){WARP8(S,8);}else if(variant==7&&int64_t(n)*c*h*w<2147483647){WARP8(S,7);}else if(variant>=7){WARP8(S,4);}else if(variant==6){WARP8(S,6);}else if(variant==4||bits==8){WARP8(S,4);}else{RUN(4,4,8,S,5);}
 if(sparse){WARPS(true);}else{WARPS(false);}
 #undef RUN
 #undef BITS4
 #undef WARP8
 #undef WARPS
 C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

std::vector<torch::Tensor> joint_next(torch::Tensor x,torch::Tensor planar,torch::Tensor weight,torch::Tensor scale,
 torch::Tensor row,torch::Tensor location,torch::Tensor offsets,torch::Tensor value,torch::Tensor sums,
 torch::Tensor bias,torch::Tensor activation,torch::Tensor residual,int64_t bits,int64_t cp,int64_t k,
 int64_t r,int64_t stride,int64_t pad,int64_t dil,bool relu,bool sparse,bool add_residual,int64_t warps,int64_t variant,
 torch::Tensor next_multiplier,torch::Tensor next_activation,torch::Tensor selection,
 int64_t next_bits,int64_t next_cp,int64_t sc,bool next_planar,int64_t strategy,bool balance,bool retain_shared,bool few_channels,bool packed_metadata) {
 TORCH_CHECK(bits==4||bits==8,"invalid joint bit width");
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
 #define RUN(B,A,W,S,V) run_joint_selected<B,A,W,S,V>(x,weight,scale,out,cp,params,stream)
 if(metadata) {
   if(few) {
     if(strategy==0) {
       if(bits==4){if(balanced){RUN(4,4,8,true,44);}else{RUN(4,4,8,true,43);}}
       else {if(balanced){RUN(8,4,8,true,44);}else{RUN(8,4,8,true,43);}}
     } else {
       if(bits==4){if(balanced){RUN(4,4,8,true,46);}else{RUN(4,4,8,true,45);}}
       else {if(balanced){RUN(8,4,8,true,46);}else{RUN(8,4,8,true,45);}}
     }
   } else if(strategy==0) {
     if(bits==4){if(balanced){RUN(4,4,8,true,38);}else{RUN(4,4,8,true,37);}}
     else {if(balanced){RUN(8,4,8,true,38);}else{RUN(8,4,8,true,37);}}
   } else {
     if(bits==4){if(balanced){RUN(4,4,8,true,40);}else{RUN(4,4,8,true,39);}}
     else {if(balanced){RUN(8,4,8,true,40);}else{RUN(8,4,8,true,39);}}
   }
 } else if(few) {
   if(strategy==0) {
     if(bits==4){if(balanced){RUN(4,4,8,true,32);}else{RUN(4,4,8,true,31);}}
     else {if(balanced){RUN(8,4,8,true,32);}else{RUN(8,4,8,true,31);}}
   } else {
     if(bits==4){if(balanced){RUN(4,4,8,true,34);}else{RUN(4,4,8,true,33);}}
     else {if(balanced){RUN(8,4,8,true,34);}else{RUN(8,4,8,true,33);}}
   }
 } else if(retain_input) {
   if(strategy==0) {
     if(bits==4){if(balanced){RUN(4,4,8,true,24);}else{RUN(4,4,8,true,23);}}
     else {if(balanced){RUN(8,4,8,true,24);}else{RUN(8,4,8,true,23);}}
   } else {
     if(bits==4){if(balanced){RUN(4,4,8,true,26);}else{RUN(4,4,8,true,25);}}
     else {if(balanced){RUN(8,4,8,true,26);}else{RUN(8,4,8,true,25);}}
   }
 } else if(balanced) {
   if(strategy==0) {
     if(bits==4){if(shared_input){RUN(4,4,8,true,18);}else{RUN(4,4,8,true,17);}}
     else {if(shared_input){RUN(8,4,8,true,18);}else{RUN(8,4,8,true,17);}}
   } else {
     if(bits==4){if(shared_input){RUN(4,4,8,true,20);}else{RUN(4,4,8,true,19);}}
     else {if(shared_input){RUN(8,4,8,true,20);}else{RUN(8,4,8,true,19);}}
   }
 } else if(strategy==0) {
   if(bits==4){if(shared_input){RUN(4,4,8,true,12);}else{RUN(4,4,8,true,11);}}
   else {if(shared_input){RUN(8,4,8,true,12);}else{RUN(8,4,8,true,11);}}
 } else {
   if(bits==4){if(shared_input){RUN(4,4,8,true,14);}else{RUN(4,4,8,true,13);}}
   else {if(shared_input){RUN(8,4,8,true,14);}else{RUN(8,4,8,true,13);}}
 }
 #undef RUN
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {out,packed,auxiliary};
}

// Static launch-resource limit from the CUDA occupancy API; this is not a
// measurement of achieved occupancy, memory bandwidth, or elapsed time.
std::vector<int64_t> joint_resources(int64_t bits,int64_t warps,int64_t access,
 int64_t variant,bool sparse,int64_t device) {
 TORCH_CHECK((bits==4||bits==8)&&(warps==4||warps==8)&&(access==4||access==8)&&variant>=0&&variant<=46&&variant!=10&&variant!=27&&variant!=28,"invalid resource-query configuration");
 TORCH_CHECK(warps!=8||access==4,"eight-warp output uses four elements per access");
 int count=0;C10_CUDA_CHECK(cudaGetDeviceCount(&count));TORCH_CHECK(device>=0&&device<count,"invalid device");
 c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device));
 #define QUERY(B,A,W,S,V) return joint_resource_row<B,A,W,S,V>()
 #define Q4(S) if(bits==4){if(access==4){QUERY(4,4,4,S,0);}else{QUERY(4,8,4,S,0);}}else{if(access==4){QUERY(8,4,4,S,0);}else{QUERY(8,8,4,S,0);}}
 #define Q8(S,V) if(bits==4){QUERY(4,4,8,S,V);}else{QUERY(8,4,8,S,V);}
 #define CONFIG(S) if(warps==4){Q4(S);}else if(variant==0){Q8(S,0);}else if(variant==1){Q8(S,1);}else if(variant==2){Q8(S,2);}else if(variant==3){Q8(S,3);}else if(variant==11){Q8(S,11);}else if(variant==12){Q8(S,12);}else if(variant==13){Q8(S,13);}else if(variant==14){Q8(S,14);}else if(variant==15){Q8(S,15);}else if(variant==16){Q8(S,16);}else if(variant==17){Q8(S,17);}else if(variant==18){Q8(S,18);}else if(variant==19){Q8(S,19);}else if(variant==20){Q8(S,20);}else if(variant==21){Q8(S,21);}else if(variant==22){Q8(S,22);}else if(variant==23){Q8(S,23);}else if(variant==24){Q8(S,24);}else if(variant==25){Q8(S,25);}else if(variant==26){Q8(S,26);}else if(variant==35){Q8(S,35);}else if(variant==36){Q8(S,36);}else if(variant==37){Q8(S,37);}else if(variant==38){Q8(S,38);}else if(variant==39){Q8(S,39);}else if(variant==40){Q8(S,40);}else if(variant==41){Q8(S,41);}else if(variant==42){Q8(S,42);}else if(variant==43){Q8(S,43);}else if(variant==44){Q8(S,44);}else if(variant==45){Q8(S,45);}else if(variant==46){Q8(S,46);}else if(variant==29){Q8(S,29);}else if(variant==30){Q8(S,30);}else if(variant==31){Q8(S,31);}else if(variant==32){Q8(S,32);}else if(variant==33){Q8(S,33);}else if(variant==34){Q8(S,34);}else if(variant==9){Q8(S,9);}else if(variant==8){Q8(S,8);}else if(variant==7){Q8(S,7);}else if(variant==6){Q8(S,6);}else if(variant==4||bits==8){Q8(S,4);}else{QUERY(4,4,8,S,5);}
 if(sparse){CONFIG(true);}else{CONFIG(false);}
 #undef QUERY
 #undef Q4
 #undef Q8
 #undef CONFIG
}

// Explicit resource query for the spatial W4 lookup specialization.
std::vector<int64_t> joint_lut_resources(int64_t variant) {
 if(variant==36)return joint_resource_row<4,4,8,true,36,true>();
 if(variant==38)return joint_resource_row<4,4,8,true,38,true>();
 if(variant==40)return joint_resource_row<4,4,8,true,40,true>();
 if(variant==42)return joint_resource_row<4,4,8,true,42,true>();
 if(variant==44)return joint_resource_row<4,4,8,true,44,true>();
 if(variant==46)return joint_resource_row<4,4,8,true,46,true>();
 TORCH_CHECK(false,"invalid shared W4 lookup resource variant");
}
