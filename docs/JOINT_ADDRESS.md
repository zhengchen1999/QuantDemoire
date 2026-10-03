# 联合 kernel 的查表、分块与寻址消融

所有候选保留相同整数权重、CSR 坐标、FP16 舍入位置与 FP32 稀疏累加顺序。
本轮扩展后完整单元测试 500 项通过，包括 batch、空间边界、跨层量化、非默认
stream、切换输入的 CUDA Graph，以及跨越 64 输出通道分块的形状。

`QUANTDEMOIRE_JOINT_VARIANT` 新增以下 eight-warp 候选：

- `5`：W4 每个 warp 计算 16 个已舍入的反量化值，通过 shuffle 查表。内部 / 边界
  分支以整个 warp 一致的条件选择，边界上无效的 lane 仍参与 shuffle。W8 使用 `4`。
- `6`：输出 tile 从 M64 N32 改为 M64 N64，使默认 CUTLASS 256-thread 加载映射
  能使用一半大小的 K tile。动态共享内存容量显式取主循环与 epilogue/CSR 所需
  容量的较大值；没有改变实际张量的输出通道数。
- `7`：在输入 extent 小于 INT_MAX 时，用 32 位整数计算 CSR 的输入元素下标。
  全局指针仍按 CUDA 指针宽度寻址，尺寸不满足条件时回退 `4`。
- `8`：在 `7` 基础上使用 host 预计算的 CUTLASS `FastDivmod` 分解输出位置，
  避免在稀疏计算和 epilogue 中反复做运行时整数除法。

静态资源记录见 [joint_candidates_resources.json](../results/joint_candidates_resources.json)。
variant 6 使用 25856 bytes 动态共享内存，理论上限为 3 blocks/SM；其余新候选为
24576 bytes、4 blocks/SM。variant 5/6 每线程 local memory 为 16 bytes，7/8 为
8 bytes。这些数据不能替代实测 occupancy 或耗时。

## 首轮整网筛选

固定已经验证的跨层调度，单轮 20 次计时：

| kernel | W4A4 ms | W8A8 ms |
|---|---:|---:|
| variant 4 对照 | 87.533 | 101.171 |
| variant 5，W4 查表 | 90.784 | — |
| variant 6，较宽输出分块 | 89.055 | 103.995 |
| variant 7，32 位索引 | 87.029 | 100.542 |
| variant 8，再加入 FastDivmod | **86.958** | **100.404** |

查表和宽分块在此调度下变慢。32 位索引及 FastDivmod 候选进入下一轮；目前这是
筛选结果，尚不作为全形状、完整图像验证或 30% 分支开销验收的替代。
记录：[joint_candidates_screen.json](../results/joint_candidates_screen.json)。

```bash
uv run python script/benchmark_variants.py --config config/experiments/joint_address.json \
  --output results/joint_address_retest.json --rounds 2 --repeats 30
```

## 双轮复测与全形状检查

| kernel | W4A4 ms | W8A8 ms |
|---|---:|---:|
| variant 4，同一逐层调度 | 87.761 | 101.225 |
| variant 8，同一逐层调度 | **86.934** | **100.518** |

同轮 FP16 为 129.261 ms，约为 1.487× / 1.286× 加速。峰值 allocated 不变，
W4 / W8 为 2577.22 / 2797.60 MiB。记录为
[joint_address_e2e.json](../results/joint_address_e2e.json)。

两种位宽各 157 个实际量化层形状均与分离实现精确相等。汇总计算时间不含输入准备：

| 指标 | W4A4 | W8A8 |
|---|---:|---:|
| 全部层使用联合 kernel，含稀疏分支 ms | 39.051 | 48.894 |
| 相同联合 kernel 关闭稀疏分支 ms | 25.925 | 37.732 |
| 上述两项的增量比例 | 50.63% | 29.58% |
| 单层增量不超过 30% 的层数 | 13 / 157 | 33 / 157 |
| 按当前混合调度累加的计算阶段增量 | 80.68% | 43.99% |

最后一行按计划逐层取联合或分离计时及其无稀疏主路径；它不包含跨层前缀生产和
输入准备，仍是计算阶段诊断值。分离路径的增量还包含输出合并和激活处理。不能
把 W8 全联合模式的 29.58% 当作所有层或实际混合部署已经达到 30% 目标。

记录：`results/joint_address_all_shapes_{4,8}.json`。两种位宽均已完成全部 100 图及
切换输入的 CUDA Graph 验证，三个输出均与参考路径精确相等，见
`results/joint_address_equivalence_{4,8}.json`。模型权重、量化参数、稀疏索引均未改变，
质量结果保持；推荐矩阵及本地权重目录的执行配置已更新到 variant 8。
