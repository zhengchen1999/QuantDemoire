# 整数卷积与 FP16 稀疏分支的联合输出

`QUANTDEMOIRE_JOINT_WARPS=4|8` 实验性地在同一个 CUTLASS kernel 中执行整数
主分支和 CSR FP16 分支，省去主分支 FP16 中间结果的全局写入、读取及独立合并 launch。
默认值仍为 `0`，保留已验证的独立分支实现。

## 数值与实现

整数 Tensor Core 使用 INT32 累加。整数 MMA 完成后，线程按连续空间位置计算 CSR
分支，使用 FP16 操作数和 FP32 FMA；结果暂存在 shared memory，再由 CUTLASS
输出线程读取。稀疏坐标及每个像素的累加顺序保持不变。主分支缩放、bias 和零点修正
的结果仍先舍入至 FP16，再与独立的 CSR FP32 结果相加；ReLU 和残差的舍入位置
也保持不变。因此，消除中间张量不会同时改变其数值语义。

两种配置的线程块及整数 K tile 不同：

| 配置 | 输出 tile M×N | warp tile M×N | INT4/INT8 K tile | pipeline stages |
|---|---|---|---|---|
| 4 warps | 128×32 | 32×32 | 128/64 | 3 |
| 8 warps | 64×32 | 16×16 | 256/128 | 2 |

这个版本不产生下一层量化前缀，完整网络测量包含因此增加的下一层 prepare 成本。
4-warp 版本明显更慢，不能假定减少 launch 和内存读写必然带来加速。

## 端到端消融

RTX 4090，batch=1，原生 2160×3840，模型补齐至 2176×3840，eager。
每个独立进程预热 7 次、测量 30 次；两轮反转配置顺序。
所有行均开启注意力和 RDB 残差融合，独立分支使用 linear CSR。
模型与权重未修改。

| 路径 | W4A4 ms | W8A8 ms |
|---|---:|---:|
| 独立分支 + 下一层量化前缀 | 93.216 | 106.888 |
| 独立分支，关闭量化前缀 | 94.459 | 108.788 |
| 联合输出，4 warps | 114.913 | 125.823 |
| 联合输出，8 warps | **91.418** | **104.253** |

相对于同轮最佳独立分支，8-warp 联合输出延迟减少约 1.93% / 2.46%。
峰值 allocated 从 2789.39 / 3012.51 MiB 降至 2534.89 / 2756.89 MiB。
完整记录及源码哈希：[joint_e2e.json](../results/joint_e2e.json)。
编译资源：[joint_compiled_resources.txt](../results/joint_compiled_resources.txt)。

完整单元测试集 65 项通过，其中 16 项覆盖联合输出的两种位宽、两种配置、边界、
stride/dilation、batch=2、ReLU、残差、稀疏关闭、非默认 stream 和变更输入的
CUDA Graph。全部真实层形状与数据集等价检查另存逐层/逐图报告；完成状态以
报告中的 `complete` 字段为准。

## 复现

```bash
uv run python script/benchmark_variants.py --config config/experiments/joint.json \
  --output results/joint_retest.json

uv run python script/benchmark_joint.py --model checkpoints/hf/esdnet-uhdm-w4a4 \
  --shapes-report results/kernel_compact_4.json --output results/joint_shapes4.json

QUANTDEMOIRE_FUSE_NEXT_QUANT=1 uv run python script/verify_graph_fusions.py \
  --backend cuda --model checkpoints/hf/esdnet-uhdm-w4a4 --csr-kernel linear \
  --joint-warps 8 --graph --output results/joint_verify4.json
```

逐层微基准排除 prepare，也不产生下一层量化前缀，不能用逐层耗时相加代替整网
结果。`joint8_sparse_increment_ratio` 记录同配置开关稀疏分支的增量耗时比例；
编译器在关闭稀疏时可以减少资源使用，所以它是消融测量，不能解释为一个可单独
计时的、与整数分支完全独立的 FP16 kernel 耗时。30% 目标仍需实测检查。

## 全部层形状与按层选择

两种位宽的全部 157 个量化层形状中，4-warp、8-warp 与独立分支均逐位一致。
记录：[W4A4](../results/joint_all_shapes_4.json)、[W8A8](../results/joint_all_shapes_8.json)。
8-warp 开关稀疏分支的逐层合计增量比例为 64.07% / 40.38%，仍高于 30% 目标。

`script/build_joint_plan.py` 从完整形状报告生成按层配置，默认只有在微基准同时改善
至少 3% 和 0.002 ms 时才选择联合输出。配置绑定模型权重 SHA256，并覆盖所有量化层，
通过 `QUANTDEMOIRE_JOINT_PLAN` 加载。它是需要整网验证的候选方案，不是理论加速保证。
其他输入尺寸仍可计算，但这些选择只针对已记录的 UHDM 原生形状调优。

联合输出层现在也能接收上一层产生的量化前缀；只有联合输出层本身不能产生下一层
前缀。混合位宽、不同空间大小及两种联合配置的链式 CUDA Graph 测试均已加入，
完整测试集 81 项通过。

```bash
uv run python script/build_joint_plan.py --report results/joint_all_shapes_4.json \
  --output config/kernel_plans/uhdm_w4_mixed.json
uv run python script/benchmark_variants.py --config config/experiments/joint_mixed.json \
  --output results/joint_mixed_retest.json
```

混合配置的测量与 FP16 对照见 [joint_mixed_e2e.json](../results/joint_mixed_e2e.json)。

两轮结果的中位数：

| 路径 | W4A4 ms | W8A8 ms |
|---|---:|---:|
| 独立分支 + 跨层量化 | 93.155 | 106.816 |
| 所有量化层联合输出 | 91.372 | 103.992 |
| 仅选中的 1×1 层联合输出 | 90.772 | 104.269 |
| **逐层选择** | **89.400** | **102.938** |

逐层选择的方案中，W4 有 76 层使用联合输出、81 层使用独立分支；W8 为 78 / 79 层。
同轮、同样开启注意力融合的 FP16 对照为 129.542 ms，因此完整网络加速为
**1.449× / 1.258×**。峰值 allocated 为 2576.72 / 2799.10 MiB，对比 FP16 的
3102.45 MiB 节约 16.95% / 9.78%。这是激活和工作区在内的运行峰值，不能与模型
权重文件压缩率混用。

最终逐层方案在两种位宽的全部 100 张本地 UHDM 图像上，三个输出及变更输入的
CUDA Graph 重放均与独立 legacy 路径逐位一致：
[W4](../results/joint_mixed_equivalence_4.json)、[W8](../results/joint_mixed_equivalence_8.json)。
验证汇总：[joint_validation.json](../results/joint_validation.json)。相同权重和本地数据
此前的质量指标继续适用。完整网络 profiling 记录为 `results/profile_joint_mixed_{4,8}.json`。
`config/experiments/optimized.json` 提供仅包含当前选定 W4、W8、FP16 对照的复测配置。
