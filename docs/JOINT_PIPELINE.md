# 联合 kernel 的加载流水线与共享内存复用

`QUANTDEMOIRE_JOINT_VARIANT` 为 eight-warp 联合输出提供可独立比较的实现：

- `0`：此前已验证的同步 two-stage CUTLASS pipeline。
- `1`：显式构造 two-stage `cp.async` main loop。使用默认 three-stage kernel 的
  iterator/policy 类型，但保持已验证的 K tile，重新实例化 two-stage main loop。
- `2`：在 `1` 的基础上，整数 MMA 完成后复用其共享内存，存储 CSR 输出 tile。
  tile 放在 epilogue scratch 之后，有编译期容量及对齐检查，并在复用前同步。

4-warp 路径保持原实现。权重、CSR 坐标、累加顺序、FP16 舍入位置均不改变。
本轮完整单元测试 235 项通过，包含非默认 stream、边界、batch、残差、混合位宽、
所有 prepare 候选及跨层量化与 CUDA Graph 的组合。

## 资源检查

`script/joint_resources.py` 通过 `cudaFuncGetAttributes` 和 CUDA occupancy API 记录
静态资源及理论驻留上限。它不测量运行期间的实际 occupancy 或 DRAM 带宽。

RTX 4090，8 warps/block，两种位宽的启用 CSR 路径如下：

| variant | registers/thread | static shared bytes | dynamic shared bytes | 最大 blocks/SM |
|---|---:|---:|---:|---:|
| 0 | 72 | 8448 | 24576 | 3 |
| 1 | 70 | 8448 | 24576 | 3 |
| 2 | 70 | 0 | 24576 | 3 |

三者均无 local-memory spill。epilogue scratch 为 5120 bytes，CSR tile 为 8448
bytes，两者合计可放入复用的 24576 bytes main-loop storage。
共享内存减少没有增加理论驻留块数，不能据此推断端到端加速。
报告：[joint_async_resources.json](../results/joint_async_resources.json)。

## 复测

```bash
uv run python script/joint_resources.py --output results/joint_resources_retest.json
uv run python script/benchmark_variants.py --config config/experiments/joint_async.json \
  --output results/joint_async_retest.json
```

完整网络消融同时比较旧的逐层选择、异步逐层选择、共享内存复用逐层选择、全部
量化层使用复用版本，并包含同轮 FP16 对照。结果以
[joint_async_e2e.json](../results/joint_async_e2e.json) 中的完成状态及源码哈希为准。
这些是实验选项，不自动替换已经验证的推荐配置。

两轮完成后的整网结果：

| 路径 | W4A4 ms | W8A8 ms |
|---|---:|---:|
| variant 0，已验证逐层配置 | **88.628** | **102.261** |
| variant 1，同一逐层配置 | 90.611 | 103.905 |
| variant 2，同一逐层配置 | 90.739 | 103.843 |
| variant 2，全部量化层 | 94.914 | 106.474 |

本轮异步与复用候选均没有改善，因此保留 variant 0。同轮 FP16 对照为 129.676 ms，
现有推荐路径约为 1.46× / 1.27×；浮点对照也启用适用的注意力融合。
对于更慢的实验候选，未额外宣称已完成全部 100 图质量确认。

## 限制寄存器

variant `3` 为异步加载加复用，variant `4` 为同步加载加复用；两者使用
`__launch_bounds__(256,4)` 的单独 kernel 入口。它们都达到 64 registers/thread，
理论驻留上限增至 4 blocks/SM。variant 3 没有 local memory，variant 4 为每线程
8 bytes，因此仍需检查额外访存与驻留数量的权衡。
记录：[joint_bounded_resources.json](../results/joint_bounded_resources.json)。

本轮完整 CUDA 测试 315 项通过，另有 5 项本地执行配置的 CPU 测试通过。
两轮原生 UHDM 整网计时：

| 路径 | W4A4 ms | W8A8 ms |
|---|---:|---:|
| variant 0，原逐层配置 | 88.555 | 102.142 |
| variant 3，原逐层配置 | 89.140 | 102.546 |
| variant 4，原逐层配置 | **88.094** | **101.589** |
| variant 4，全部量化层 | 88.966 | 102.837 |

同步受限版本得到约 0.5 ms 改善，异步受限版本仍未超过原版。新的逐层选择与
完整图像验证仍是采用前需要完成的工作。同轮 FP16 为 129.417 ms。
记录：[joint_bounded_e2e.json](../results/joint_bounded_e2e.json)，配置为
`config/experiments/joint_bounded.json`。

进一步测完两种位宽各 157 个实际形状，variant 4 的结果均与分离实现精确相等。
纯计算部分（排除输入准备）的联合稀疏增量仍为 W4 57.03%、W8 32.98%，尚未满足
两种位宽均不超过 30% 的目标；此比例的分母是同配置的无稀疏联合 kernel。
记录为 `joint_bounded_all_shapes_{4,8}.json`。

按新 kernel 的逐层耗时重新选层后，再与旧调度交叉比较：

| 调度 / kernel | W4A4 ms | W8A8 ms |
|---|---:|---:|
| 旧调度 / variant 0 | 88.493 | 101.885 |
| 旧调度 / variant 4 | 88.050 | **101.436** |
| 新调度 / variant 0 | 88.843 | 102.174 |
| 新调度 / variant 4 | **87.974** | 101.718 |

同轮 FP16 为 129.525 ms。新调度 W4 采用 95 个联合层，W8 采用 84 个，但 W8
整网反而不如旧调度：逐层计算计时不含输入准备，也未计算联合输出失去量化前缀
生产能力的影响。不能把逐层耗时之和直接当作整网结果。
记录：[joint_retuned_e2e.json](../results/joint_retuned_e2e.json)。
