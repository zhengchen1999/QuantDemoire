#include <torch/extension.h>
torch::Tensor integer_conv(torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);
std::vector<torch::Tensor> prepare_cat(std::vector<torch::Tensor>,torch::Tensor,torch::Tensor,int64_t,int64_t,bool);
std::vector<torch::Tensor> prepare(torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,bool);
std::vector<torch::Tensor> prepare_compact_cat(std::vector<torch::Tensor>,torch::Tensor,torch::Tensor,int64_t,int64_t,torch::Tensor,int64_t);
std::vector<torch::Tensor> prepare_cached_cat(std::vector<torch::Tensor>,torch::Tensor,torch::Tensor,int64_t,int64_t,torch::Tensor,int64_t,int64_t,bool);
std::vector<torch::Tensor> prepare_next_cat(std::vector<torch::Tensor>,torch::Tensor,torch::Tensor,int64_t,int64_t,torch::Tensor,int64_t,torch::Tensor,torch::Tensor,int64_t,int64_t,bool);
std::vector<torch::Tensor> finish_half_next(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,int64_t,bool,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t);
torch::Tensor finish_tc(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,int64_t,bool);
torch::Tensor finish_half(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,int64_t,bool);
torch::Tensor finish_half_residual(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,int64_t,bool,torch::Tensor);
torch::Tensor attention_residual(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor);
std::vector<torch::Tensor> finish_half_tuned(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,int64_t,bool,torch::Tensor,bool,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,bool);
torch::Tensor joint_conv(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,bool,bool,bool,int64_t,int64_t,bool,bool,bool,bool);
torch::Tensor finish(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,int64_t,bool,bool);
torch::Tensor fused_conv(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,bool,bool);
std::vector<torch::Tensor> joint_next(torch::Tensor x,torch::Tensor planar,torch::Tensor weight,torch::Tensor scale,
 torch::Tensor row,torch::Tensor location,torch::Tensor offsets,torch::Tensor value,torch::Tensor sums,
 torch::Tensor bias,torch::Tensor activation,torch::Tensor residual,int64_t bits,int64_t cp,int64_t k,
 int64_t r,int64_t stride,int64_t pad,int64_t dil,bool relu,bool sparse,bool add_residual,int64_t warps,int64_t variant,
 torch::Tensor next_multiplier,torch::Tensor next_activation,torch::Tensor selection,
 int64_t next_bits,int64_t next_cp,int64_t sc,bool next_planar,int64_t strategy,bool balance,bool retain_shared,bool few_channels,bool packed_metadata);
std::vector<torch::Tensor> joint_next_two(torch::Tensor x,torch::Tensor planar,torch::Tensor weight,torch::Tensor scale,
 torch::Tensor row,torch::Tensor location,torch::Tensor offsets,torch::Tensor value,torch::Tensor sums,
 torch::Tensor bias,torch::Tensor activation,torch::Tensor residual,int64_t bits,int64_t cp,int64_t k,
 int64_t r,int64_t stride,int64_t pad,int64_t dil,bool relu,bool sparse,bool add_residual,int64_t warps,int64_t variant,
 torch::Tensor next_multiplier,torch::Tensor next_activation,torch::Tensor selection,
 int64_t next_bits,int64_t next_cp,int64_t sc,bool next_planar,int64_t strategy,bool balance,bool retain_shared,bool few_channels,bool packed_metadata,torch::Tensor future_multiplier,
 torch::Tensor future_activation,int64_t future_bits,int64_t future_offset);
std::vector<torch::Tensor> prepare_cached_second(std::vector<torch::Tensor>,torch::Tensor,torch::Tensor,int64_t,int64_t,torch::Tensor,int64_t,torch::Tensor,torch::Tensor,int64_t,bool,torch::Tensor);
std::vector<torch::Tensor> joint_next_omitted(torch::Tensor x,torch::Tensor planar,torch::Tensor weight,torch::Tensor scale,
 torch::Tensor row,torch::Tensor location,torch::Tensor offsets,torch::Tensor value,torch::Tensor sums,
 torch::Tensor bias,torch::Tensor activation,torch::Tensor residual,int64_t bits,int64_t cp,int64_t k,
 int64_t r,int64_t stride,int64_t pad,int64_t dil,bool relu,bool sparse,bool add_residual,int64_t warps,int64_t variant,
 torch::Tensor next_multiplier,torch::Tensor next_activation,torch::Tensor selection,
 int64_t next_bits,int64_t next_cp,int64_t sc,bool next_planar,int64_t strategy,bool balance,bool retain_shared,bool few_channels,bool packed_metadata);
std::vector<torch::Tensor> joint_next_two_omitted(torch::Tensor x,torch::Tensor planar,torch::Tensor weight,torch::Tensor scale,
 torch::Tensor row,torch::Tensor location,torch::Tensor offsets,torch::Tensor value,torch::Tensor sums,
 torch::Tensor bias,torch::Tensor activation,torch::Tensor residual,int64_t bits,int64_t cp,int64_t k,
 int64_t r,int64_t stride,int64_t pad,int64_t dil,bool relu,bool sparse,bool add_residual,int64_t warps,int64_t variant,
 torch::Tensor next_multiplier,torch::Tensor next_activation,torch::Tensor selection,
 int64_t next_bits,int64_t next_cp,int64_t sc,bool next_planar,int64_t strategy,bool balance,bool retain_shared,bool few_channels,bool packed_metadata,torch::Tensor future_multiplier,
 torch::Tensor future_activation,int64_t future_bits,int64_t future_offset);
std::vector<torch::Tensor> finish_half_omitted(torch::Tensor x,torch::Tensor y,torch::Tensor row,
 torch::Tensor index,torch::Tensor offsets,torch::Tensor value,torch::Tensor params,
 int64_t bits,int64_t r,int64_t stride,int64_t pad,int64_t dil,bool relu,
 torch::Tensor residual,bool add_residual,torch::Tensor multiplier,torch::Tensor next_params,
 torch::Tensor selection,int64_t next_bits,int64_t cp,int64_t sc,int64_t mode,bool next_planar);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {

 m.def("joint_next_omitted",&joint_next_omitted);
 m.def("joint_next_two_omitted",&joint_next_two_omitted);
 m.def("finish_half_omitted",&finish_half_omitted);
 m.def("joint_next_two",&joint_next_two);
 m.def("prepare_cached_second",&prepare_cached_second);
 m.def("fused_conv",&fused_conv);m.def("integer_conv",&integer_conv);m.def("finish",&finish);m.def("finish_tc",&finish_tc);
 m.def("finish_half",&finish_half);
 m.def("finish_half_residual",&finish_half_residual);m.def("attention_residual",&attention_residual);
 m.def("finish_half_tuned",&finish_half_tuned,pybind11::arg("x"),pybind11::arg("y"),pybind11::arg("row"),pybind11::arg("index"),pybind11::arg("offsets"),pybind11::arg("value"),pybind11::arg("params"),pybind11::arg("bits"),pybind11::arg("r"),pybind11::arg("stride"),pybind11::arg("pad"),pybind11::arg("dil"),pybind11::arg("relu"),pybind11::arg("residual"),pybind11::arg("add_residual"),pybind11::arg("multiplier"),pybind11::arg("next_params"),pybind11::arg("selection"),pybind11::arg("next_bits"),pybind11::arg("cp"),pybind11::arg("sc"),pybind11::arg("mode"),pybind11::arg("next_planar")=true);
 m.def("joint_conv",&joint_conv,pybind11::arg("x"),pybind11::arg("planar"),pybind11::arg("weight"),pybind11::arg("scale"),pybind11::arg("row"),pybind11::arg("location"),pybind11::arg("offsets"),pybind11::arg("value"),pybind11::arg("sums"),pybind11::arg("bias"),pybind11::arg("activation"),pybind11::arg("residual"),pybind11::arg("bits"),pybind11::arg("cp"),pybind11::arg("k"),pybind11::arg("r"),pybind11::arg("stride"),pybind11::arg("pad"),pybind11::arg("dil"),pybind11::arg("relu"),pybind11::arg("sparse"),pybind11::arg("add_residual"),pybind11::arg("warps"),pybind11::arg("variant")=0,pybind11::arg("balance")=false,pybind11::arg("retain_shared")=false,pybind11::arg("few_channels")=false,pybind11::arg("packed_metadata")=false);
 m.def("joint_next",&joint_next,pybind11::arg("x"),pybind11::arg("planar"),pybind11::arg("weight"),pybind11::arg("scale"),pybind11::arg("row"),pybind11::arg("location"),pybind11::arg("offsets"),pybind11::arg("value"),pybind11::arg("sums"),pybind11::arg("bias"),pybind11::arg("activation"),pybind11::arg("residual"),pybind11::arg("bits"),pybind11::arg("cp"),pybind11::arg("k"),pybind11::arg("r"),pybind11::arg("stride"),pybind11::arg("pad"),pybind11::arg("dil"),pybind11::arg("relu"),pybind11::arg("sparse"),pybind11::arg("add_residual"),pybind11::arg("warps"),pybind11::arg("variant"),pybind11::arg("next_multiplier"),pybind11::arg("next_activation"),pybind11::arg("selection"),pybind11::arg("next_bits"),pybind11::arg("next_cp"),pybind11::arg("sc"),pybind11::arg("next_planar"),pybind11::arg("strategy"),pybind11::arg("balance")=false,pybind11::arg("retain_shared")=false,pybind11::arg("few_channels")=false,pybind11::arg("packed_metadata")=false);
 m.def("finish_half_next",&finish_half_next);
 m.def("prepare_next_cat",&prepare_next_cat,pybind11::arg("inputs"),pybind11::arg("multiplier"),pybind11::arg("params"),pybind11::arg("bits"),pybind11::arg("cp"),pybind11::arg("selection"),pybind11::arg("sc"),pybind11::arg("packed"),pybind11::arg("planar"),pybind11::arg("skip"),pybind11::arg("mode")=2,pybind11::arg("sparse")=true);
 m.def("prepare_compact_cat",&prepare_compact_cat);
 m.def("prepare_cached_cat",&prepare_cached_cat);
 m.def("prepare_cat",&prepare_cat,pybind11::arg("inputs"),pybind11::arg("inv_s"),pybind11::arg("params"),pybind11::arg("bits"),pybind11::arg("cp"),pybind11::arg("sparse")=true);
 m.def("prepare",&prepare,pybind11::arg("x"),pybind11::arg("inv_s"),pybind11::arg("params"),pybind11::arg("bits"),pybind11::arg("cp"),pybind11::arg("sparse")=true);
}
