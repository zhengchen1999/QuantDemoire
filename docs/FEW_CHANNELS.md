# 小通道空间卷积的归约合并

当前八 warp INT4 / INT8 CTA 的 K tile 分别为 256 / 128。普通 optimized 卷积按
`R*S*ceil(Cp/K)` 次循环遍历卷积核位置；CUTLASS FewChannels 改为连续遍历
`R*S*Cp`，循环次数为 `ceil(R*S*Cp/K)`。权重仍是原来的 packed KRSC 布局，
输入量化、稀疏结构、CSR 行内 FP32 FMA 顺序和输出 FP16 舍入不变。

本地 CUTLASS 3.5.1 的 fused-epilogue Params 构造函数没有传迭代器类型参数，
因此新路径在 launch 前显式设置合并后的循环次数。没有修改安装的 CUTLASS。

减少整数补零计算并不保证变快。代表层中，W4 最小通道层的完整准备加联合卷积
改善约 5%–8%；较宽层普遍变慢。10 个高分辨率代表层全开时，W4 / W8 的
准备加联合卷积合计分别慢约 4.25% / 9.02%，故不全局使用。

记录：[代表层](../results/joint_few_channels_screen.json)、
[200 组独立原型检查](../results/joint_few_channels_tests.json)、
[W4 全形状](../results/joint_few_all_shapes_4.json)、
[W8 全形状](../results/joint_few_all_shapes_8.json)。两种位宽各 157 个量化层的
输出均与原路径逐元素相同；独立原型检查包括 stride、dilation、边界、两种 CSR
调度、残差、稀疏关闭、跨位宽前缀及变化输入 Graph。

## 只启用小通道层的独立进程对照

两轮、每轮 30 次 eager 测量，第二轮反向执行次序。输入为补齐后的原生 UHDM
形状，包含三个网络输出及完整输入准备，不含图像 I/O 和质量指标。

| 模式 | 原路径 ms | 小通道 FewChannels ms | 峰值 allocated bytes |
|---|---:|---:|---:|
| W4，Cp≤64 的空间联合卷积 | 82.387 | 82.057 | 2594161152，未改变 |
| W8，Cp≤48 的空间联合卷积 | 96.005 | 96.305 | 2827205632，未改变 |

记录：[原型整网对照](../results/joint_few_screen_e2e.json)。仅 W4 保留约 0.4%
改善；W8 不采用。不能将这次端到端改善描述为完整稀疏分支达到 30% 预算。

## 正式接口与验收状态

开关 `QUANTDEMOIRE_FEW_CHANNELS=1` 默认关闭。需要主变体 8/9、八 warp、
R>1 和上述 Cp 上限；不满足条件或稀疏索引超出 32-bit 安全范围时使用原路径。
兼容原调用的 C++ 尾部 `few_channels` 参数默认 false。内部编号 29–34 不是
`QUANTDEMOIRE_JOINT_VARIANT` 的公开选项。

正式代码的 1,019 项测试已全部通过，其中一个测试包含 200 组原生接口检查。
两种位宽各 157 个实际量化层、原生前缀和 100 张本地图像的三个输出及变化输入
Graph 均通过逐元素相同检查。权重与量化配置哈希未变，原质量指标继续适用。
本地模型包及 optimized 配置仅为 W4 开启新选项，W8 保持关闭。

```bash
uv run python script/benchmark_variants.py \
  --config config/experiments/few_channels.json \
  --output results/few_channels_retest.json --rounds 2 --repeats 30
uv run python script/benchmark_variants.py \
  --config config/experiments/few_selected.json \
  --output results/few_selected_retest.json --rounds 2 --repeats 30
```

## 正式同批性能与验收

| 两轮 eager 中位数 | 时延 ms | 相对 FP16 加速 | 峰值 allocated MiB |
|---|---:|---:|---:|
| FP16 操作数 / FP32 累加 | 129.565 | 1.000× | 3102.45 |
| W4A4 + FP16 分支 | 82.363 | 1.573× | 2473.98 |
| W8A8 + FP16 分支 | 96.264 | 1.346× | 2696.23 |

记录：[生产开关对照](../results/few_channels_e2e.json)、[同批 FP16 对照](../results/few_selected_e2e.json)、[验收汇总](../results/few_validation.json)。
输入补齐到 2176×3840，batch=1，包含三个输出及所有准备，不含图像 I/O 和质量指标。
FP16 同样启用适用的融合。100 张图像是本地 UHDM 子集，不是完整官方测试集。

W4 当前计划的独立计算阶段：主计算 24.515 ms，含稀疏 36.080 ms，增量 47.17%。
W8 当前计划的独立计算阶段：主计算 36.297 ms，含稀疏 46.906 ms，增量 29.23%。
这些计算阶段数字不包含准备与生产者前缀，不能用于宣称完整分支 ≤30% 预算已经达成。
上一推荐配置已保存在 `config/experiments/optimized_retained.json`；没有 push 或上传。
