# CSR 完整行的 warp 负载分配

候选将每个 32 输出通道 tile 的 CSR 行按条目数排序，依次放入当前累计工作量最少、
尚未分到四行的 warp。每个 warp 最终仍处理四行，结果写回原输出通道；不改变原始
CSR 指针、行内索引顺序或 FP16 数值，也不改变每像素的 FP32 FMA 顺序。

排列以原始 CSR row pointer 为前缀、追加每个 tile 的 32 个局部输出通道编号。
它是 `persistent=False` 的派生运行时 buffer，不写入 safetensors。原权重、两侧
至少各 0.25% 的原始离群值以及 fakequant 的稀疏结构均不变。

`QUANTDEMOIRE_BALANCE_CSR=1` 仅作用于八-warp、变体 8/9 的联合输出路径；其他
路径保留原行为。生产者前缀的两种写法也有对应特化。底层默认关闭，旧调用的新增
尾部 `balance` 参数默认 false。内部编号 15–20 用于这些特化，不是全局
`QUANTDEMOIRE_JOINT_VARIANT` 选项。

## 独立原型筛选

156 项边界、残差、稀疏关闭、辅助 planar 省略、交叉位宽前缀、非默认 stream 和
变化输入 Graph 测试通过。全部 157 个真实量化层的输出检查也通过。静态编译资源
与原变体相同：64 registers/thread、8-byte stack、0 static shared，动态 shared
仍为 24 KiB。这些不是 achieved occupancy 或实际 spill traffic 的测量。

| 全部层均使用联合计算，耗时合计 ms | 原排列 | 平衡排列 |
|---|---:|---:|
| W4 | 37.170 | 36.410 |
| W8 | 49.181 | 48.615 |

该微基准不含输入准备、下一层前缀生产，不能作为整网速度或分支增量的指标。
当前推荐计划中实际选为联合执行的 145 / 126 层，计算合计分别从 32.310 / 40.660
ms 降到 31.710 / 40.239 ms。记录：
[形状汇总](../results/joint_balanced_shape_summary.json)、
[W4 逐层](../results/joint_balanced_shapes_4.json)、
[W8 逐层](../results/joint_balanced_shapes_8.json)。

两张原生图和变化输入 Graph 检查通过后，在上一版联合前缀推荐计划上进行两轮
整网对照，保留全网络、三个输出和全部准备工作：

| 两轮中位时延 ms | 原排列 | 平衡排列 |
|---|---:|---:|
| W4 | 83.077 | 82.379 |
| W8 | 98.441 | 98.123 |

峰值 allocated 增加 113,664 bytes，约 0.108 MiB；包括为量化层创建的派生排列。
记录：[独立原型整网筛选](../results/joint_balanced_screen_e2e.json)。这轮没有 FP16
对照，不跨批次计算相对 FP16 的加速比。原型使用单独链接的库和进程内适配器，
不是可直接发布的执行配置。

生产接入已通过下述完整单元测试、全部形状、100 图及 Graph 验证和同批 FP16
测量，推荐 runtime profile 已显式启用。30% 分支开销目标仍未验收达成。

## 生产接入的配对消融

完整 861 项单元测试通过。两种位宽分别通过全部 157 个实际形状、原生前缀检查，
以及 100 张本地图像的三个输出和变化输入 Graph 比较。接入后的同批开关消融为：

| 两轮中位时延 ms | balance=0 | balance=1 |
|---|---:|---:|
| W4 | 83.372 | 82.824 |
| W8 | 98.893 | 98.467 |

记录：[生产开关对照](../results/csr_balance_e2e.json)。峰值增量仍为 113,664 bytes。
该表不是带 FP16 基线的最终比较。相比独立原型，生产 API 通过显式尾部布尔参数
控制平衡，保留原全局变体编号及旧调用兼容性。

按当前 145 / 126 个联合层和其余分离层的实际选择，独立计算阶段增量为：

| 当前执行计划的计算诊断 | 主计算合计 ms | 含稀疏合计 ms | 增量 / 主计算 |
|---|---:|---:|---:|
| W4 | 24.891 | 36.557 | 46.87% |
| W8 | 36.140 | 48.034 | 32.91% |

这一诊断不包含输入准备和下一层前缀生产；分离路径的增量含合并、激活。
全部层强制联合时为 40.94% / 27.54%，但不是当前最快整网计划，因此不能用后者
宣布当前 W8 的 30% 目标已达成。记录：
[W4 计算](../results/csr_balance_all_shapes_4.json)、
[W8 计算](../results/csr_balance_all_shapes_8.json)。

复测生产消融：

```bash
uv run python script/benchmark_variants.py \
  --config config/experiments/csr_balance.json \
  --output results/csr_balance_retest.json --rounds 2 --repeats 30
```

## 已验收的同批 FP16 对照

| 两轮 eager 中位数 | 时延 ms | 相对 FP16 加速 | 峰值 allocated MiB |
|---|---:|---:|---:|
| FP16 操作数 / FP32 累加 | 129.606 | 1.000× | 3102.45 |
| W4A4 + FP16 稀疏分支 | 82.808 | 1.565× | 2473.99 |
| W8A8 + FP16 稀疏分支 | 98.484 | 1.316× | 2798.58 |

输入补齐到 2176×3840、batch=1，包含全网络、三个输出和全部量化准备工作，
不含图像读取和质量指标。FP16 也启用适用的融合。峰值显存分别减少 20.26% / 9.79%。
记录：[同批整网测量](../results/csr_balance_selected_e2e.json)、
[验收汇总与环境](../results/csr_balance_validation.json)、
[W4 整网及 Graph](../results/csr_balance_equivalence_4.json)、
[W8 整网及 Graph](../results/csr_balance_equivalence_8.json)。

配置已写入 `config/experiments/optimized.json` 和本地模型包；前版保存在
`optimized_joint_next.json`。权重和量化配置哈希不变，三个网络输出与旧参考路径
逐元素完全一致，因此此前同数据的质量结果继续适用。没有 push 或上传。
