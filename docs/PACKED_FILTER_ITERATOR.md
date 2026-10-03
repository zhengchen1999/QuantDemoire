# 小通道卷积的连续权重寻址

FewChannels 主循环把归约维度展平为 R×S×C。通用 filter iterator 在每次读取时，
仍会把该下标拆成 R/S/C，再按三个 stride 还原线性地址。这里的权重始终按连续
KRSC 存储，因此可以直接计算 `输出通道 * stride_n + 展平归约下标`。

实现只替换 FewChannels 的 filter iterator；激活 iterator、分块、流水线、INT32
累加、FP16 主输出舍入、CSR FP32 累加顺序及下一层量化逻辑不变。INT4 继续使用
相同的位地址换算。代码保留 CUTLASS 的 BSD 许可证，并通过类型约束和现有 host
分派条件限制适用范围；host 始终传入 `Layout::packed(ws)`。

没有新增权重格式或运行时开关。现有 `QUANTDEMOIRE_FEW_CHANNELS=1` 在符合条件
的小通道空间卷积中使用该实现，其余路径保持原分派。推荐设置继续为 W4 开启、
W8 关闭。原始两侧离群值、CSR 索引、权重值和 fakequant 结构均未改变。

## 独立原型结果

600 组组合检查通过，包括与非 FewChannels 参考路径逐元素一致的输出、metadata
组合、边界、尾通道、stride/dilation、融合前缀、独立 stream 和变化输入 Graph。
两种位宽还各通过两个实际图像、三个输出及 Graph 的筛查。

抽测符合条件的大分辨率实际层：

| 位宽 | 层数 | 原卷积与稀疏合并合计 | 连续寻址 | 含准备的原合计 | 连续寻址含准备 |
|---|---:|---:|---:|---:|---:|
| W4 | 9 | 2.512 ms | 2.329 ms | 3.953 ms | 3.786 ms |
| W8 | 4 | 1.639 ms | 1.632 ms | 2.486 ms | 2.485 ms |

这组数据只代表抽测热点层，不能作为整网加速比。若干变体的静态 local storage
变小；没有把静态资源数据解释为实测溢出流量或实测 occupancy。

两轮正反顺序、独立进程、每轮 30 个样本的整网筛查：

| 模式 | 当前版本 | 连续寻址候选 | 峰值 allocated |
|---|---:|---:|---:|
| W4，现有 FewChannels 开关 | 81.546 ms | 80.978 ms | 均为 2,594,161,152 bytes |
| W8，候选额外开启 FewChannels | 95.440 ms | 95.424 ms | 均为 2,827,205,632 bytes |

W4 两轮均改善，平均减少约 0.568 ms（0.697%）。W8 的差异很小且两轮方向不一，
不采用全局开启 FewChannels。输入补齐为 2176×3840，batch=1，计入三个输出和
所有输入准备，不含图像 I/O 与质量指标。

正式代码的 1,022 项测试通过。两种位宽各 157 个量化形状、原生前缀、
100 张本地 UHDM 图像的三个输出及变化输入 Graph 均通过逐元素相同检查。
本地模型执行配置已更新，W8 继续关闭 FewChannels。权重与量化配置哈希不变，
既有质量结果继续适用。完整稀疏分支 30% 目标仍未验收通过。

- [抽测结果](../results/flat_filter_screen_summary.json)
- [原型整网对照](../results/flat_filter_e2e.json)
- `outputs/prototypes/make_flat_filter.py`：独立原型生成器。
- `outputs/run_flat_filter_final.py`：正式回归和整网测试。
- `outputs/pair_flat_filter_production.py`：正式 W4 与保存的旧版本对照。

## 正式同批性能与验收

| 两轮 eager 中位数 | 时延 ms | 相对 FP16 加速 | 峰值 allocated MiB |
|---|---:|---:|---:|
| FP16 操作数 / FP32 累加 | 129.885 | 1.000× | 3102.45 |
| W4A4 + FP16 分支 | 81.131 | 1.601× | 2473.98 |
| W8A8 + FP16 分支 | 95.376 | 1.362× | 2696.23 |

[正式 W4 与旧版本对照](../results/flat_filter_production_pair.json)、[同批 FP16 对照](../results/flat_filter_selected_e2e.json)、[验收汇总](../results/flat_filter_validation.json)。
100 张图像是本地 UHDM 子集。整网时延不能替代单独的分支预算。

W4 当前计划的计算阶段：主计算 24.380 ms，含稀疏 35.350 ms，增量 44.99%。
W8 当前计划的计算阶段：主计算 36.209 ms，含稀疏 46.217 ms，增量 27.64%。
这些计算诊断排除了准备与生产者前缀，不能单独证明完整预算达标。
上一推荐配置保存在 `config/experiments/optimized_metadata.json`；没有 push 或上传。
