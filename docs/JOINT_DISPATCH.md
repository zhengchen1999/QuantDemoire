# 包含跨层量化成本的联合 kernel 调度

分离路径的 CSR 输出可以同时为下一层生成 32 通道的量化前缀，联合路径目前只能
消费此前缀、不能生产。因此，某层独立卷积更快，并不保证选择它后整网也更快。

`script/tune_joint_dispatch.py` 使用实际网络的 157 个输入形状与拼接顺序，对每层
测量分离和 eight-warp 联合两种实现，并在 DenseBlock 链内增加已有前缀的输入状态。
每次计时包括该层输入准备、卷积、稀疏合并和适用的前缀生产；RDB 最后一个卷积也
包含残差加法。随机输入只用于计时，各路径须与完整准备的分离实现精确相等。

每条链通过动态规划选择路径：前驱为分离 kernel 时，本层按部分准备成本计；前驱
为联合 kernel 时，本层按完整准备成本计。默认给联合选择加入 1% / 0.001 ms 的
小幅成本余量，降低微小计时波动导致调度改变的可能。单层注意力卷积等独立节点
使用完整准备成本。求解器已经过 120 组小链穷举最优解对照。

```bash
QUANTDEMOIRE_PREPARE_IMPL=quad_swizzle \
QUANTDEMOIRE_CSR_KERNEL=linear \
QUANTDEMOIRE_FUSE_NEXT_QUANT=1 \
QUANTDEMOIRE_FUSE_ATTENTION=1 \
QUANTDEMOIRE_FUSE_RESIDUAL=1 \
QUANTDEMOIRE_JOINT_VARIANT=4 \
QUANTDEMOIRE_JOINT_WARPS=0 QUANTDEMOIRE_JOINT_PLAN='' \
uv run python script/tune_joint_dispatch.py \
  --model checkpoints/hf/esdnet-uhdm-w4a4 \
  --shapes results/prepare_vectors_all_shapes_4.json \
  --output results/joint_chain_shapes_4.json \
  --plan config/kernel_plans/uhdm_w4_chain.json
```

输出计划仍绑定权重 SHA-256，使用现有 `quantdemoire-joint-plan-v1` 格式。计划内的
`measured_environment` 记录测试的 kernel variant，但运行时必须显式设置对应
`QUANTDEMOIRE_JOINT_VARIANT`。输入尺寸改变后应重新计时。

该成本模型不包含跨层缓存和 allocator 状态，也不预测 Python 调度间隙。生成计划
后仍须独立进行正反顺序的整网计时、显存记录及完整图像 / CUDA Graph 验证，才能
更新推荐配置和本地权重包。

## 实测结果

两种位宽各 157 层的完整准备 / 前缀准备、分离 / 联合输出均通过精确对照。W4
候选为 90 个联合层、67 个分离层，W8 为 93 / 64。两轮正反顺序整网计时：

| 配置 | W4A4 ms | W8A8 ms |
|---|---:|---:|
| 上轮较快的调度，variant 4 | 88.072 | 101.477 |
| 包含跨层成本的调度，variant 4 | **87.981** | **101.193** |

同轮 FP16 为 129.170 ms，对应加速为 1.468× / 1.276×。调度本身带来的改善较小，
尤其 W4 的 0.09 ms，不应夸大其稳健性。计时不含图片 I/O 和质量指标，包含完整
三个尺度输出、输入准备和稀疏分支。峰值 allocated 为 W4 2577.22 MiB、W8
2797.60 MiB，较同轮 FP16 减少 16.93% / 9.83%。

新的两种配置均在本地全部 100 张 UHDM 图像上通过三个输出的精确对照，以及
切换输入的 CUDA Graph 验证。权重、量化参数和稀疏索引不变，前述质量结果保持。
这不表示稀疏分支 30% 开销目标已经达到。

- [整网计时及显存](../results/joint_chain_e2e.json)
- [W4 全图验证](../results/joint_chain_equivalence_4.json)
- [W8 全图验证](../results/joint_chain_equivalence_8.json)
- 逐层报告：`results/joint_chain_shapes_{4,8}.json`
- 推荐矩阵：`config/experiments/optimized.json`

对应计划也包含在本地权重目录的 `runtime.json` / `joint_plan.json` 中，采用
[可搬移执行配置](FORMAT.md)。没有上传或修改权重文件。
