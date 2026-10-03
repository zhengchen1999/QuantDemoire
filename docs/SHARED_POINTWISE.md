# 从整数卷积共享内存读取稀疏分支输入

variant `9` 对 `1×1 / stride=1 / padding=0` 且输入仅占一个主循环 K tile 的层，
直接读取整数卷积留下的 shared A。W4 的 padded C 上限为 256，W8 为 128；其余
形状回退 variant 8。同步双缓冲流水线在只有一轮迭代时仍保留 stage 0 输入。

稀疏计算放在整数 MMA 结束之后、epilogue 改写共享内存之前。每线程先保存八个
FP32 稀疏累加结果，所有读取完成后同步，再将结果写入复用的 CSR 输出 tile。
稀疏操作数仍舍入为 FP16，各行贡献按原 CSR 顺序执行 FMA，普通分支的 FP16
中间舍入也保留。不需要在整数 MMA 期间保留这些稀疏累加器。

这一分支必须使用**原始 CSR 输入通道编号**，因为 shared A 保存完整输入通道。
`PackedConv` 和 `benchmark_joint.py` 会按相同条件选择原始 / compact locations；
直接调用底层 `joint_conv` 的实验代码也必须遵守这个约定。

## 只改变计算路径的对照

首先保留所有辅助 planar 输入准备，只替换读取方式。完整 564 项测试通过，包含
尾部输出通道、两种位宽、batch、stream、跨层量化、残差和变化输入的 CUDA Graph。
原型编译资源与 variant 8 相同：64 registers/thread、8 bytes local memory/thread，
不能据此代替实际性能验证。

当前调度下，W4 有 23 个符合条件的联合层，其中 20 个处理空间特征图；W8 仅有
5 个小型注意力层。两轮整网对照：

| 路径 | W4A4 ms | W8A8 ms |
|---|---:|---:|
| variant 8 | 86.870 | 100.332 |
| variant 9，保留辅助输入准备 | **85.337** | 100.384 |

W4 改善约 1.53 ms，W8 没有稳定收益。此轮没有重新测 FP16，因此不把其他轮次
基线混入这个表。记录：[joint_shared_compute_e2e.json](../results/joint_shared_compute_e2e.json)。

## 跳过无用的辅助输入准备

`QUANTDEMOIRE_SHARED_SKIP_PLANAR=1` 仅对上述 shared-A 路径生效。完整准备时只生成
主分支 packed 输入，保留零 stride 的 metadata view；补齐量化前缀时只写 packed
输入，保留已有的 planar 分配但不补齐其内容。此阶段生产者仍生成 planar 前缀，
因此不应宣称已消除所有辅助分配和写入。

底层仅允许确实不读取 planar 数据的 shared-A kernel 接受 metadata view；回退
到常规稀疏 kernel 时仍要求连续且真实存储的 planar 输入。正确性、完整形状和
整网收益继续独立验证，未通过前不替换已验证的模型包配置。

加入跳过路径后完整 568 项测试通过，包括 metadata view、已有前缀、输入更新的
Graph，以及不满足条件时回退到正常辅助输入的案例。固定同一 W4 调度，双轮结果：

| 路径 | W4A4 ms |
|---|---:|
| variant 8 | 86.790 |
| variant 9，仅改变稀疏读取 | 85.301 |
| variant 9，跳过辅助准备 | **84.532** |

峰值 allocated 均为 2577.22 MiB，没有额外峰值显存收益。生产者仍为前缀分配
完整 planar 张量，因此准备阶段少写数据不等于消除了该内存分配。
记录：[joint_shared_skip_e2e.json](../results/joint_shared_skip_e2e.json)。
后续全形状对照对符合条件的候选直接传入单字节存储的 metadata view，避免让
无意中的辅助输入读取被仍然存在的完整数据掩盖。

## 推荐组合的复核

W4 的 157 个实际量化层形状全部通过，包括符合条件时仅用单字节辅助存储；
W8 保留之前已全形状验证的 variant 8。当前代码下两种位宽均完成本地全部
100 图、三个输出和变化输入 CUDA Graph 的一致性验证，见
[joint_shared_validation.json](../results/joint_shared_validation.json)。权重、稀疏位置和
质量指标均未改变。

采用两个独立进程轮次、第二轮倒序执行的推荐组合测试：

| 模式 | 两轮中位时延的平均值 ms | 相对 FP16 | 峰值 allocated MiB |
|---|---:|---:|---:|
| W4 variant 9 + skip planar | 84.851 | 1.526× | 2577.22 |
| W8 variant 8 | 100.607 | 1.287× | 2797.60 |
| FP16 + 相同适用的注意力融合 | 129.465 | 1.000× | 3102.45 |

见 [shared_selected_e2e.json](../results/shared_selected_e2e.json)。本地模型包的
runtime.json 和 config/experiments/optimized.json 已采用此配置；没有上传。
W4 全联合模式的逐层计算增量约 44.8%，仍高于 30% 目标；这一计算消融
不是推荐混合调度的整网开销验收。详见
[joint_shared_all_shapes_4.json](../results/joint_shared_all_shapes_4.json)。

## 删除生产者的辅助前缀

实验开关 `QUANTDEMOIRE_SHARED_SKIP_PREFIX_PLANAR=1` 与 shared skip planar 同时启用时，
线性 CSR 生产者检查下一层是否走 shared-A 路径。若满足条件，它仍生成主分支的
packed 前缀，但不再分配、写入辅助 planar 前缀，也跳过为该写入准备的共享内存
存储和同步。不满足条件或使用 legacy CSR 时保留原路径；默认关闭实验开关。

572 项完整测试通过，额外两个 CPU 调度测试通过。真实 4K 网络的 50 条融合连接
均通过 packed 码检查，其中 10 条省掉辅助存储；另验证了三个整网输出和变化输入
Graph。这里的 4K 前缀检查仅使用一张图，不能代替后续全 100 图验证。
见 [shared_prefix_buffers.json](../results/shared_prefix_buffers.json)。

| W4 当前调度 | 两轮平均中位时延 ms | 峰值 allocated MiB |
|---|---:|---:|
| 保留辅助前缀 | 84.511 | 2577.22 |
| 删除辅助前缀 | 84.366 | 2473.62 |

[shared_prefix_e2e.json](../results/shared_prefix_e2e.json) 显示峰值减少 103.6 MiB。
时延差异约 0.17%，应按基本持平看待，不将它描述为稳定加速。
此轮未重测 FP16，不混用其他轮次计算加速比。

跨层调度器现在还测量下一层的模式，因为它决定上一层是否需要生成辅助前缀。
动态规划同时考虑前一层、当前层和下一层选择；用 168 组短链与穷举结果比对，
兼容旧的逐层计时记录。候选调度仍须通过独立整网计时和完整图像验证。

最新候选调度将 W4 联合层从 90 个增至 131 个，保留 26 个 separate 层。
157 层的所有 full/prefix 输入与下一层模式组合通过数值检查。
在 [shared_retuned_e2e.json](../results/shared_retuned_e2e.json) 的双轮整网对照中，
旧调度为 84.384 ms，新调度为 83.950 ms，两轮方向一致，峰值 allocated 均为
2473.62 MiB。该候选保存在 `config/kernel_plans/uhdm_w4_shared.json`；完整 100 图
与同批 FP16/W8 复核完成前不替换推荐模型包。

## 删除前缀并重调度后的完整复核

候选已通过本地全部 100 图、每图三个输出及变化输入 CUDA Graph 验证；W8 同时
用当前代码复核。新 W4 调度含 24 条融合连接，其中 7 条只保留单字节辅助存储。
[shared_prefix_validation.json](../results/shared_prefix_validation.json) 汇总了前缀、
完整形状与整网检查。FP16 操作数/FP32 累加、两侧原始尾部和权重均未改变。

| 当前推荐组合 | 两轮平均中位时延 ms | 相对同批 FP16 | 峰值 allocated MiB |
|---|---:|---:|---:|
| W4 shared-A + 无辅助前缀 + 新调度 | 84.182 | 1.538× | 2473.62 |
| W8 variant 8 | 100.419 | 1.289× | 2797.60 |
| FP16 + 相同适用融合 | 129.431 | 1.000× | 3102.45 |

见 [shared_prefix_selected_e2e.json](../results/shared_prefix_selected_e2e.json)。W4/W8
峰值 allocated 相对 FP16 减少约 20.27% / 9.83%；这是独立进程中的活跃分配峰值，
不是 CUDA reserved memory 或整个进程显存。当前推荐配置和本地模型 runtime.json
均已更新，没有上传。30% 稀疏分支开销目标仍未完成。

进一步的 [3×3 主循环共享输入原型](SHARED_SPATIAL_PROTOTYPE.md) 数值检查通过，但
代表层显著变慢，已保留失败记录，未纳入推荐配置。
