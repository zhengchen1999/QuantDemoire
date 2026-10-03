# 激活准备阶段优化

已完成第一轮改进：预计算常量并成对读写 FP16/整数值。默认实现切换为 `pair`，
模型文件、离群值、稀疏位置和量化定义保持不变。

## 本轮实测

RTX 4090，原生 2160×3840 输入，补齐到 2176×3840。逐层测试使用完整网络
实际执行时的 157 个量化输入签名，保留 DenseBlock 的多个输入段。

| 实现 | W4A4 准备时间合计 ms | W8A8 准备时间合计 ms |
|---|---:|---:|
| 原 scalar | 33.303 | 36.287 |
| 只预计算系数 cached | 32.894 | 36.119 |
| 预计算 + 成对读写 pair | **29.927** | **33.805** |

`pair` 分别减少约 **10.1% / 6.8%** 的准备耗时。
这些是逐层 CUDA Graph 微基准的总和，不是完整推理时间。
原始记录：[W4A4](../results/prepare_layers_4.json)、[W8A8](../results/prepare_layers_8.json)。

| 同轮独立进程 eager 整网计时 | W4A4 ms | W8A8 ms |
|---|---:|---:|
| scalar | 116.557 | 130.065 |
| cached | 116.297 | 129.921 |
| pair | **113.962** | **128.263** |

整网改善约 **2.2% / 1.4%**。来源为 `results/e2e_prepare_{scalar,cached,pair}_{4,8}.json`。
完整网络 profiling 另保存在 [W4A4](../results/profile_prepare_pair_4.json)、
[W8A8](../results/profile_prepare_pair_8.json)，不把不同轮次的 GPU 时钟波动混入改进比例。

## 改了什么

- 把固定的 `inverse_smooth / activation_step` 保存为 FP32 运行时派生系数。
  每层系数只在加载或校准参数改变时更新，不写进 safetensors。
  原实现每个空间块都重复除法；消融表明消除这项运算的收益较小。
- 成对加载相邻 FP16 通道；INT4 直接形成一个 packed byte，INT8 合并为 16-bit 写入。
  每个 shared-memory word 同时放两个 INT8 值，重排时提取相应字节。
  在同样 shared-memory 容量下，处理更多通道，并改善 INT4 写入事务的利用率。
- 遇到奇数通道段时使用标量 fallback。保留分开的 FP32 multiply/add 和 round-to-even，
  不使用改变舍入规则的 fast-math。

合并访问和 shared-memory 转置的原理见
[NVIDIA CUDA 内存性能文档](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#memory-performance)。

## 正确性

- 23 项测试通过，包括奇数通道 fallback、batch=2、稀疏映射、边界和 CUDA Graph。
- 两种位宽各 157 个真实输入签名的 packed 整数和 compact planar 数据均逐位一致。
- 两种位宽在全部 100 张 UHDM 测试图上的三个整网输出均与 scalar 逐位一致：
  [W4A4](../results/prepare_network_equivalence_4.json)、
  [W8A8](../results/prepare_network_equivalence_8.json)。因此此前这批图的质量指标保持不变。

## 更大幅度优化需要什么

目前仍会把 FP16 中间特征写回显存，再由下一层 prepare 读入，输出整数卷积
布局及稀疏 planar 布局。单纯缩短除法、改善指令和访存效率不能消除这些数据流。

优先实验方向是 **producer epilogue + 下一层量化融合**：在上一层合并稀疏结果、
ReLU、舍入到 FP16 时直接生成下一层所需的整数数据。为保持当前数值定义，
必须先在寄存器里模拟原来的 FP16 输出舍入，再执行下一层的乘法、加零点和舍入。
对于单消费者边，可以省掉后续 prepare 的 FP16 重读；有其他浮点消费者时仍可能
需要保存 FP16 输出。

ESDNet 的 DenseBlock 存在多消费者和不同的 smoothing / activation scale，
不能直接跨层共享同一份量化激活。需要按消费者生成数据、分阶段融合，或重新
约束并校准量化参数。后者会改变数值行为，必须重测质量，不能沿用等价验证结果。

另一个方向是统一主分支和稀疏分支的输入布局。当前 planar 准备支持按空间连续
读取稀疏输入；简单删除它、直接从 NHWC packed 激活随机读取，可能增加访存事务。
必须连同稀疏 kernel 的线程安排或分块结构一起验证。原始两侧离群值各至少 0.25%
的覆盖约束保持不变。

其中 producer epilogue 与下一层量化融合已实现为可选实验路径，针对 DenseBlock
新增的 32 通道前缀，测得额外约 1.3% / 2.0% 的整网延迟改善；完整结果及显存代价见
[跨层量化融合报告](NEXT_QUANT_FUSION.md)。本页上表仍保留融合之前的 prepare 消融。
统一输入布局尚待实验。即使消除整个 prepare，稀疏合并和其余网络算子仍存在；
不能将 prepare 的优化百分比直接当成整网加速比。

```bash
# 保留旧实现用于对照；默认已是 pair。
QUANTDEMOIRE_PREPARE_IMPL=scalar uv run python script/benchmark_e2e.py \
  --backend cuda --model checkpoints/hf/esdnet-uhdm-w4a4 --output results/scalar4.json
uv run python script/benchmark_prepare.py --model checkpoints/hf/esdnet-uhdm-w4a4 --output results/prepare4.json
uv run python script/verify_prepare.py --model checkpoints/hf/esdnet-uhdm-w4a4 --output results/equivalence4.json
```
