# 联合输出生成下一层量化前缀

已验证的联合 kernel 原先只能消费上一层前缀，生产者必须使用 separate 路径才能
融合下一层量化。新候选在联合输出已经完成 FP16 舍入后，使用下一层的 smoothing
与激活量化参数生成 packed 前缀，继续保留供 DenseBlock 其他层使用的 FP16 特征。
权重、离群值、每行 FMA 顺序和浮点舍入语义均不变。

## 两种写法

`QUANTDEMOIRE_JOINT_NEXT_QUANT=0` 默认关闭；`1` 直接生成 packed 与辅助 planar
前缀；`2` 先把辅助码写入共享内存中的小 tile，再以空间连续的 warp 写入 planar。
后一种 tile 使用 MMA 结束后的空闲区域，不增加原本 24 KiB 的动态共享内存分配。
当消费者使用 shared-A 且开启辅助前缀省略时，两种策略均只生成 packed 前缀。

仅八-warp、主变体 8/9 的联合生产者启用此功能；当前 DenseBlock 生产者输出 32
通道。其他路径保留原行为。内部生产者特化编号 11–14 不是全局 JOINT_VARIANT
选项。底层默认关闭；经下述整网验收后，推荐 runtime profile 显式开启并附带逐层计划。

## 独立原型结果

64 组两位宽、交叉位宽、空间尺寸、stream 和变化输入 Graph 检查通过。
编译资源仍是 64 registers/thread、8-byte stack；这不是 achieved occupancy 测量。
随后两种位宽、两种策略分别通过两张真实图的整网和 Graph 一致性检查。

| 两轮平均中位时延 ms | 原联合 kernel | 直接写入 | 共享内存重排 |
|---|---:|---:|---:|
| W4 | 83.755 | 83.647 | 83.558 |
| W8 | 100.232 | 99.538 | 99.717 |

记录：[原型整网筛选](../results/joint_next_screen_e2e.json)。该轮未测 FP16，因此
不计算跨批次加速比。W4 峰值 allocated 增加约 1.37 MiB，W8 减少约 1.38 MiB；
没有明显峰值显存变化。全部统一开启的收益较小，继续测试按层选择。

## 接入与按层选择

接入后完整 702 项测试通过，其中新增 128 项涵盖生产者先消费一个已有前缀，
再继续生成下一层前缀。另有一个独立 CPU 调度测试，用 30 组短链的穷举结果
验证四种执行选择的动态规划。

`script/tune_joint_next.py` 比较 separate、joint 无前缀、joint 直接前缀、joint
重排前缀，并计入后继层是否得到准备好的输入、以及是否需要辅助 planar。
它还逐例核对输出和完整填充后的前缀码。旧的两模式调度器在启用联合生产者时
会拒绝运行，避免遗漏新的前缀依赖。

候选计划继续使用 weight-bound joint-plan-v1，新增可选的 `joint_next_modes`，
为每个量化层保存 0/1/2。全局开关为 0 时禁用这些联合生产者设置；开启后再按
层选择或禁用。原有计划仍有效，量化权重格式不变。调优成本不能预测所有整网
缓存和内存分配效应，候选必须再做完整网络计时和 100 图验证。

## 逐层调优与峰值显存修正

两位宽全部 157 个形状的四种执行选择及可用前缀组合均已通过输出/码值检查：
[W4 记录](../results/joint_next_chain_shapes_4.json)、
[W8 记录](../results/joint_next_chain_shapes_8.json)。

| 同批次两轮中位时延 ms | 原计划，无联合前缀 | 原计划，全局开启 | 四模式逐层选择 |
|---|---:|---:|---:|
| W4 | 83.671 | 83.599 | 82.819 |
| W8 | 100.059 | 99.484 | 98.435 |

记录：[逐层选择整网对照](../results/joint_next_tuned_e2e.json)。逐层选择的峰值
allocated 比原计划增加约 66.87 / 25.03 MiB。分段 allocator 追踪表明，峰值出在
最高分辨率解码 DenseBlock：联合生产者运行期间必须同时持有当前输入与下一层
前缀，单层成本最小化没有考虑这部分生命周期重叠。

仅关闭 W4 `decoder.decoder_1.sam_blocks.0.basic_block.conv_layers.4.conv.0`
与 W8 对应 `conv_layers.3.conv.0` 的联合前缀生产，保留它们的联合卷积，获得：

| 另一批两轮对照 | 未修正 ms | 显存修正后 ms | 修正后峰值 allocated MiB |
|---|---:|---:|---:|
| W4 | 82.653 | 82.737 | 2473.88 |
| W8 | 98.023 | 98.297 | 2798.48 |

记录：[峰值修正整网对照](../results/joint_next_memory_e2e.json)。修正后峰值相对
原方案仅增加约 0.25 / 0.88 MiB；W4 约 0.1% 的时延差异不作为确定收益/退化结论。
两次表格是独立批次，不交叉计算加速比。修正计划保存为
`config/kernel_plans/uhdm_w{4,8}_joint_next_memory.json`，其中记录父计划和显存修正。
该修正计划已通过下述原生前缀、100 图 Graph 数值验证和同批 FP16 对照。

## 复测

依赖和 CUDA/CUTLASS 环境设置见 [REPRODUCE.md](REPRODUCE.md)。保留全部候选的
端到端计时可直接运行：

```bash
uv run python script/benchmark_variants.py \
  --config config/experiments/joint_next_tuned.json \
  --output results/joint_next_tuned_retest.json --rounds 2 --repeats 30
uv run python script/benchmark_variants.py \
  --config config/experiments/joint_next_memory.json \
  --output results/joint_next_memory_retest.json --rounds 2 --repeats 30
```

重新逐层调优时，使用该位宽候选的环境变量，清空 `QUANTDEMOIRE_JOINT_PLAN`，
设置 `QUANTDEMOIRE_JOINT_NEXT_QUANT=2`，然后调用：

```bash
uv run python script/tune_joint_next.py \
  --model checkpoints/hf/esdnet-uhdm-w4a4 \
  --shapes results/prepare_vectors_all_shapes_4.json \
  --output results/joint_next_retune_4.json \
  --plan config/kernel_plans/uhdm_w4_joint_next_retune.json --repeats 8
```

新计划不能直接替代已验收 profile：应再检查 `verify_prepare.py` 的
`--next-quant --check-prefixes --graph` 和 `verify_graph_fusions.py --graph`，并
重测整网峰值显存。校准、fakequant 及本地权重结构均不因执行计划而变化。

## 推荐配置验收

| 同批次两轮 eager 中位数 | 时延 ms | 相对 FP16 加速 | 峰值 allocated MiB |
|---|---:|---:|---:|
| FP16 操作数 / FP32 累加 | 129.412 | 1.000× | 3102.45 |
| W4A4 + FP16 稀疏分支 | 83.322 | 1.553× | 2473.88 |
| W8A8 + FP16 稀疏分支 | 98.858 | 1.309× | 2798.48 |

输入补齐到 2176×3840，batch=1；全部三个输出、输入准备、量化、稀疏计算均计入，
不含图像读取和质量指标。FP16 也启用适用的注意力/残差融合。峰值显存分别减少
20.26% / 9.80%，不是模型常驻显存压缩率。记录：
[本轮同批整网测量](../results/joint_next_selected_e2e.json)。

两种位宽各 157 个量化形状的四模式检查、原生分辨率实际前缀码比较，以及本地
100 张图的三个输出和变化输入 CUDA Graph 均通过。W4 检查 71 条融合边，其中
63 条为联合生产者、14 条省略辅助 planar；W8 为 70 / 50 / 0 条。
记录：[汇总](../results/joint_next_validation.json)、
[W4 前缀](../results/joint_next_prefix_buffers_4.json)、
[W8 前缀](../results/joint_next_prefix_buffers_8.json)、
[W4 整网](../results/joint_next_equivalence_4.json)、
[W8 整网](../results/joint_next_equivalence_8.json)。

推荐配置已写入 `config/experiments/optimized.json` 和本地模型包的 runtime profile，
权重与量化配置哈希不变，已有同数据质量测量继续适用。完整测试为 702 项，另有
1 项新增 CPU 调度测试单独通过。此轮不重新定义分支开销分母，也不代表 30% 目标
达成；继续保留 [之前的独立开销诊断](CSR_ADDRESS.md)。未 push 或上传。
