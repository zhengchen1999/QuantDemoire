# 原生 4K 加速瓶颈拆解

当前主要代价是每层的激活处理，以及 FP16 稀疏计算和结果合并。整数卷积核心
本身已经比相应 FP16 卷积快，但占完整推理的比例有限。

下表保留 `QUANTDEMOIRE_PREPARE_IMPL=scalar` 的 profiling。后续已完成成对读写
优化并设为默认 `pair`；准备阶段和端到端的新对比见 [准备阶段优化](PREPARE_OPTIMIZATION.md)。

## 整网 profiling

RTX 4090，输入 2160×3840，补齐到 2176×3840，batch=1；FP16 操作数、FP32
浮点卷积累加。量化模型使用本地 HF 包、FP16 主分支中间结果、压缩 planar 通道。
预热后记录 3 次完整推理的 CUDA kernel 时间，另在 profiler 之外计时 20 次。
原始 Kineto trace 保存在本地 `results/*.trace.json`，不加入 Git。

| GPU 时间分类，ms/图 | FP16 | W4A4 + FP16 | W8A8 + FP16 |
|---|---:|---:|---:|
| FP16 卷积（包括 bias / 库内部辅助 kernel） | 58.579 | 2.771 | 2.769 |
| 整数卷积、缩放、零点校正和 bias | — | 21.150 | 31.868 |
| 量化、smoothing、packing、稀疏通道准备 | — | 33.605 | 36.154 |
| FP16 稀疏计算、结果合并及融合 ReLU | — | 33.683 | 33.805 |
| 显式 concat | 46.145 | 0.009 | 0.009 |
| 独立逐点运算 | 19.405 | 13.577 | 13.588 |
| 双线性插值 | 9.406 | 9.412 | 9.413 |
| 池化、pixel shuffle / unshuffle | 3.126 | 3.128 | 3.122 |
| **CUDA kernel 时间合计** | **136.661** | **117.334** | **130.729** |
| 同轮未开启 profiler 的端到端中位数 | 136.677 | 117.198 | 130.698 |

来源：[FP16](../results/profile_fp16.json)、[W4A4](../results/profile_4.json)、
[W8A8](../results/profile_8.json)。不同轮次的时钟/缓存状态会产生小幅变化，因此
这组分项不强行拼接到另一轮 137.75 / 117.08 / 130.64 ms 的总时间上。

分类通过 CUDA correlation ID 找到 CPU launch，再按包围该 launch 的 ATen
调用区间归类。自定义 kernel 按名字单独识别。cuDNN 内部使用 CUTLASS 的 FP16
kernel 仍计入 FP16 卷积；不会因名字包含 CUTLASS 而误归为整数运算。
W4A4/W8A8 均记录到每次推理 157 个 prepare、157 个整数卷积、157 个稀疏合并
kernel，与网络量化层数一致。没有未归类的 CUDA kernel 时间。

## 为什么没有更高加速

1. **整数算力收益只作用于部分耗时。** W4A4 的整数主核占整网 GPU kernel 时间
   18.0%；量化准备和稀疏合并合计占 57.3%。即使假设整数主核耗时变为零，其他
   项完全不变，总时间仍约 96.2 ms，对本轮 FP16 的加速约为 1.42×。这只是固定
   其余项的代数上限，不是硬件上限。
2. **网络层间仍传递 FP16 特征。** 每个量化卷积重新读取 FP16 输入，做 smoothing、
   舍入/裁剪、packing，并准备另一种稀疏输入布局。已离线打包的权重仅约 4/7 MB，
   不能代表中间特征的读写规模。
3. **0.5% 的离群权重不等于 0.5% 的运行时间。** 当前 CSR 分支用 FP16 操作数
   转成 FP32 做标量 FMA，没有走 Tensor Core。它还要解码位置、读取分散的输入、
   处理边界，读写完整输出。稀疏权重在所有空间位置重复使用，名义稀疏乘加总数
   仍约为 110.5 亿次/图。
4. **W8A8 的固定开销与 W4A4 接近。** 两者的稀疏分支约 33.7–33.8 ms，但 W8A8
   整数主核约 31.9 ms，W4A4 约 21.1 ms；W8 packing 还会写出更多数据，因此净收益小。
5. **部分收益来自图融合。** 原始 FP16 算子图的显式 concat 约 46.1 ms；量化路径
   将读取多个输入段和量化融合，几乎消除了这些独立 concat。ReLU 融合也节约了
   一部分逐点处理。当前加速比反映整个部署改造收益，不能全部归因于整数位宽。
   同等图优化程度的 FP16 对照仍是后续需要补充的比较。

输入/输出张量形状推算的逻辑访问量如下，**不是硬件计数器测得的 DRAM 流量**：

| 每次推理的逻辑访问量 | W4A4 | W8A8 |
|---|---:|---:|
| prepare 读取 FP16 输入 | 18.42 GB | 18.42 GB |
| 写入打包整数激活 | 4.84 GB | 9.21 GB |
| 写入压缩 planar 激活 | 3.08 GB | 3.08 GB |
| 上述 prepare 访问合计 | 26.34 GB | 30.71 GB |
| 稀疏合并读取主分支结果 + 写最终输出 | 10.99 GB | 10.99 GB |

计数见 [W4A4](../results/traffic_4.json)、[W8A8](../results/traffic_8.json)。这些数字
还未计入整数卷积内部和其他网络算子的访问。没有用这些逻辑字节数声称显存带宽
已饱和，也没有将缓存命中或寄存器占用的推测写成实测结论。
算术强度、内存访问与实际卷积吞吐的关系可参见
[NVIDIA 卷积性能指南](https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html)。

## 稀疏开销的独立验证

只为计时关闭稀疏分支后，W4A4 从 117.20 ms 降到 76.33 ms，W8A8 从 130.70 ms
降到 90.11 ms。这个消融同时取消稀疏通道准备和结果合并；输出数值改变，不能
作为满足质量要求的部署模型。结果见 [W4A4 消融](../results/profile_4_no_sparse.json)、
[W8A8 消融](../results/profile_8_no_sparse.json)。

逐层 CUDA Graph 计时独立得出：

| 157 个实际量化层汇总 | W4A4 | W8A8 |
|---|---:|---:|
| 整数卷积核心，ms | 20.869 | 31.601 |
| 不含稀疏分支的完整主路径，ms | 47.721 | 61.371 |
| 含稀疏分支，ms | 88.664 | 101.935 |
| 分支增量，ms | 40.943 | 40.564 |
| 分支增量 / 整数核心 | **196.2%** | **128.4%** |
| 分支增量 / 完整主路径 | **85.8%** | **66.1%** |

因此 30% 要求仍未达到；按整数核心分母，每层通过数为 W4A4 0/157、W8A8 1/157。
所有数值检查仍然通过：两种位宽各 159 个原生 4K 卷积形状，包含两个浮点边界层
的额外整数形状测试。完整报告见 [逐层汇总](../results/kernel_compact_summary.json)。

## 后续优化优先级

- 先针对最高空间分辨率的 21 个量化层优化：它们占逐层混合路径耗时约 57%。
  18 个空间尺寸为 1×1 的注意力卷积合计约 0.13 ms，不是主要时间来源。
- 优先减少 CSR 分支的输入准备和整张输出读写，实验在整数 epilogue 中协作计算
  稀疏项；改变稀疏结构时仍需保留原始两侧各至少 0.25%，并与伪量化共用索引。
- 研究减少跨层重复量化/布局转换。DenseBlock 的不同消费者具有不同 smoothing
  和激活参数，不能未经验证就复用同一份低比特激活。
- 然后按实际层形状调整整数卷积 tile，同时补充同等图优化的 FP16 对照。

CUDA Graph 已实测：FP16/W4A4/W8A8 分别约 136.36/116.45/129.87 ms，包含每次
GPU 输入复制。量化模型相对 eager 只再减少约 0.6–0.8 ms，当前优先级低于上述
数据处理开销。CUDA Graph 主要减少提交开销，见
[NVIDIA 文档](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)。

```bash
export QUANTDEMOIRE_PREPARE_IMPL=scalar
uv run python script/profile_e2e.py --backend fp16 --model checkpoints/uhdm_checkpoint.pth --output results/profile_fp16.json
uv run python script/profile_e2e.py --backend cuda --model checkpoints/hf/esdnet-uhdm-w4a4 --output results/profile_4.json
uv run python script/profile_e2e.py --backend cuda --model checkpoints/hf/esdnet-uhdm-w8a8 --output results/profile_8.json
# 仅计时消融，输出不满足原模型质量要求：
uv run python script/profile_e2e.py --backend cuda --model checkpoints/hf/esdnet-uhdm-w4a4 --disable-sparse --output results/profile_4_no_sparse.json
```
