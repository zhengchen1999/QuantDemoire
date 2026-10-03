# 1×1 卷积结束后保留两级 A 缓冲

独立候选修改同步双缓冲 MMA 主循环：最后一轮不再把不会被消费的预取片段写入
共享内存，保留两个输入 tile。主循环结束后，CSR 分支读取这些输入，继续按原行
顺序做 FP16 操作数、FP32 FMA。没有把稀疏累加器移入整数主循环。

这把可复用范围从 W4 的 Cp≤256、W8 的 Cp≤128 扩大到 Cp≤512 / Cp≤256；仅用于
1×1、stride=1、padding=0。其他情况保留原路径。与之前的 shared-A 模式一样，
可省略消费者不再使用的辅助 planar 准备和生产者前缀。量化参数、权重、稀疏结构
及输出舍入语义均不变。派生 MMA 保留了复制方法对应的 CUTLASS BSD 版权声明。

## 独立原型检查

208 组数值检查通过，覆盖一个 tile、两个 tile、尾部通道、超过容量的回退、
残差、稀疏关闭、stream、跨位宽前缀和变化输入 Graph。记录：
[原型检查](../results/joint_retained_prototype_tests.json)。

静态资源仍为 64 registers/thread、8-byte stack、0 static shared；动态 shared
仍用既有的 24 KiB。它们不是 achieved occupancy 或实际 spill traffic 测量。

真实权重的适用 pointwise 层测量同时包含输入准备，两个方案都启用 CSR 行平衡：

| 适用层耗时合计 ms | 原路径 | 保留两级输入 + 省略辅助准备 |
|---|---:|---:|
| W4，41 层 | 12.310 | 12.255 |
| W8，36 层 | 17.694 | 14.900 |

W4 的总差异只有约 0.05 ms；W8 的改善几乎全部来自需要两个输入 tile 的 26 层。
逐层计时未计入前驱生成下一层前缀的相互作用，不能直接替代整网结果。记录：
[逐层数据](../results/joint_retained_screen.json)、
[汇总](../results/joint_retained_shape_summary.json)。

原型还通过两张原生图及变化输入 Graph 检查，整网开关对照已完成，见下。
接入后的完整 1,018 项测试、原生形状和 100 图正式验收均已通过；W8 推荐
runtime profile 已启用此选项，W4 保持此前执行设置。

## 原型整网对照

| 两轮中位时延 ms | 当前方案 | 扩展共享输入复用 | 峰值 allocated 变化 |
|---|---:|---:|---:|
| W4 | 82.384 | 82.274 | 无变化 |
| W8 | 98.115 | 95.954 | 减少 102.35 MiB |

记录：[原型整网筛选](../results/joint_retained_screen_e2e.json)。W4 的约 0.13%
变化不足以确认收益，因此本轮计划只更新 W8 的推荐执行配置。W8 同时扩大共享
输入复用范围并省略消费者不再需要的辅助准备/前缀，不能把全部收益只归因于
省略主循环最后一次写入。生产消融将另测原先单 tile 的共享复用设置。

生产开关是 `QUANTDEMOIRE_SHARED_RETAIN_STAGES=1`，配合八-warp、主变体 9
和相应的 planar 省略选项。底层默认关闭；C++ 原调用的新增尾部 `retain_shared`
参数默认 false。内部编号 21–26 不是全局 `JOINT_VARIANT` 选项。

## 接入后的生产消融

完整 1,018 项测试、两种位宽各 157 个实际形状、原生前缀及 100 张图的三个输出和
变化输入 Graph 均已通过。W4 保留不启用新开关的推荐设置。W8 的两轮整网对照为：

| W8 设置 | 中位时延 ms | 峰值 allocated bytes |
|---|---:|---:|
| 上一版：变体 8 | 98.447 | 2934528512 |
| 单 tile 共享输入：变体 9，retain=0 | 98.508 | 2934528512 |
| 两个 tile 共享输入：变体 9，retain=1 | 96.235 | 2827205632 |

记录：[生产三方案消融](../results/retained_pointwise_e2e.json)。原先的单 tile 复用
没有带来整网改善；扩大到两个 tile 后收益保留，同时降低峰值显存。

W8 当前逐层执行计划的独立计算阶段，主计算合计 36.391 ms、含稀疏合计 46.939 ms，
增量为 **28.99%**。W4 对应增量仍约 46.31%。这些数字排除了准备与前缀工作，不能
据此宣布完整分支预算已经不超过 30%。记录：
[W4 形状及计算](../results/retained_all_shapes_4.json)、
[W8 形状及计算](../results/retained_all_shapes_8.json)。

## 已验收的同批 FP16 对照

| 两轮 eager 中位数 | 时延 ms | 相对 FP16 加速 | 峰值 allocated MiB |
|---|---:|---:|---:|
| FP16 操作数 / FP32 累加 | 129.573 | 1.000× | 3102.45 |
| W4A4 + FP16 稀疏分支，保留原设置 | 82.753 | 1.566× | 2473.99 |
| W8A8 + FP16 稀疏分支，复用两个 tile | 96.263 | 1.346× | 2696.23 |

输入补齐到 2176×3840，batch=1；包含全网络、三个输出及全部量化/稀疏准备，
不含图像读取和质量指标。FP16 也启用适用的融合。峰值显存分别减少 20.26% / 13.09%。
记录：[同批整网测量](../results/retained_selected_e2e.json)、
[验收汇总](../results/retained_validation.json)、
[W4 整网及 Graph](../results/retained_equivalence_4.json)、
[W8 整网及 Graph](../results/retained_equivalence_8.json)。

原生前缀检查中，W4 的 71 条融合边有 14 条省略辅助 planar；W8 的 70 条中有
16 条省略。完整 packed 码与需要使用的 planar 均和独立完整准备相同。
配置已写入 `config/experiments/optimized.json`、`models.json` 和两份本地模型包。
权重、量化配置哈希不变，三个输出和旧路径逐元素相同，既有质量指标继续适用。
前一版配置保存在 `optimized_csr_balance.json`；没有 push 或上传。

```bash
uv run python script/benchmark_variants.py \
  --config config/experiments/retained_pointwise.json \
  --output results/retained_pointwise_retest.json --rounds 2 --repeats 30
uv run python script/benchmark_variants.py \
  --config config/experiments/retained_selected.json \
  --output results/retained_selected_retest.json --rounds 2 --repeats 30
```
