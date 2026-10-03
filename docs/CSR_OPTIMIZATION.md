# CSR 分支：地址预计算与空间访问优化

在已经开启跨层量化、注意力融合和 RDB 残差融合的基础上，新增三种 CSR kernel。
`QUANTDEMOIRE_CSR_KERNEL=legacy` 保留原版，`linear`、`lut`、`wide` 用于消融。
目前实测 `linear` 最快，开关仍为显式选择，不改变模型文件。

## 改动

- `linear`：每个 warp 按连续像素读取输入；为实际输入 H/W 和 dilation 缓存每个
  CSR 非零项的线性地址偏移。内部像素使用简化循环，图像边界保留逐项检查。
- `lut`：在 linear 的基础上，用 shared-memory 表保存量化整数对应的 FP16
  反量化操作数，W4 用 16 项，W8 用 256 项。测量包含建表和同步成本。
- `wide`：在查表版本上，把每个线程的像素数从 2 增至 4，扩大复用范围。

三种实现保持 CSR 非零项顺序、逐像素 FP32 FMA 累加顺序、FP16 操作数舍入和
主分支 FP16 中间舍入。输出端同时支持下一层量化前缀及 RDB 残差融合。
固定权重和稀疏结构完全相同，两侧各至少 0.25% 的原始离群值保留约束保持不变。
地址缓存按布局、H/W、dilation 更新，为非持久运行时 buffer，不写入 safetensors。

## 独立进程整网比较

RTX 4090，输入 2160×3840，补齐到 2176×3840，batch=1，eager；包含全部三个输出、
量化和稀疏分支，不包含图像 I/O。每项预热 7 次、测 30 次，做两轮并反转配置顺序。
下表是每轮中位数的中位数。

| CSR 实现 | W4A4 ms | W8A8 ms |
|---|---:|---:|
| legacy | 103.115 | 116.467 |
| **linear** | **93.438** | **107.288** |
| lut | 93.611 | 107.411 |
| wide | 97.189 | 111.048 |

linear 额外减少 **9.39% / 7.88%** 的整网延迟，运行时地址缓存使峰值 allocated
仅增加约 **0.267 MiB**。原始样本、实际环境和源码哈希见
[整网消融](../results/csr_e2e.json)，配置见 [csr.json](../config/experiments/csr.json)。

独立测最高输入分辨率的 21 个量化层，其 CSR 输出处理包含必要的下一层量化或残差：

| 21 层 CSR kernel 中位数合计，ms | W4A4 | W8A8 |
|---|---:|---:|
| legacy | 21.815 | 21.808 |
| linear | 15.813 | 16.257 |
| lut | 15.922 | 16.445 |
| wide | 17.536 | 17.997 |

微基准使用 CUDA Graph，每次包含 10 次操作，输入为随机张量，权重为实际部署权重。
它隔离 CSR 处理，不包含 prepare 和整数卷积，不能代替整网计时。
记录：[W4A4](../results/csr_hotspots_4.json)、[W8A8](../results/csr_hotspots_8.json)。

查表未带来额外收益，因此选择不查表的 linear。`cuobjdump` 静态资源报告显示，
带下一层量化的 linear 使用 40 registers/thread、4352 bytes shared/block；
wide 使用 48 registers/thread、9728 bytes shared/block。它们均没有 local-memory
spill。这表明 wide 使用更多资源，但此报告不是实际 occupancy 或 DRAM 带宽测量，
不能把耗时差异完全归因于寄存器。报告见
[编译资源](../results/csr_compiled_resources.txt)。

## 复现

```bash
uv run python script/benchmark_variants.py --config config/experiments/csr.json \
  --output results/csr_retest.json

QUANTDEMOIRE_FUSE_NEXT_QUANT=1 QUANTDEMOIRE_FUSE_ATTENTION=1 \
QUANTDEMOIRE_FUSE_RESIDUAL=1 QUANTDEMOIRE_CSR_KERNEL=linear \
  uv run python script/benchmark_e2e.py --backend cuda \
  --model checkpoints/hf/esdnet-uhdm-w4a4 --output results/linear4.json

uv run python script/benchmark_sparse.py --model checkpoints/hf/esdnet-uhdm-w4a4 \
  --shapes-report results/kernel_compact_4.json --all --output results/sparse_shapes4.json

QUANTDEMOIRE_FUSE_NEXT_QUANT=1 uv run python script/verify_graph_fusions.py \
  --backend cuda --model checkpoints/hf/esdnet-uhdm-w4a4 --csr-kernel linear \
  --graph --output results/linear_verify4.json
```

替换位宽可复测 W8A8。这些变体仍未解决 FP16 分支不超过主分支 30% 的目标，
后续会实验整数主分支与稀疏分支的协作输出，以进一步减少中间结果读写。

## 完整验证与 profiling

两种位宽分别覆盖全部 157 个量化层的实际权重、原生 UHDM 形状，三个 CSR 候选的
输出及适用的下一层 packed/planar 前缀均与 legacy 逐位一致。两个保留的浮点边界层
没有修改 CUDA 卷积实现。记录：[W4](../results/csr_all_shapes_4.json)、
[W8](../results/csr_all_shapes_8.json)。

选中的 linear 在全部 100 张本地 UHDM 图像上，三个网络输出及更新输入的 CUDA Graph
重放均与 legacy（同时关闭本轮注意力/残差融合）逐位一致：
[W4](../results/csr_linear_equivalence_4.json)、[W8](../results/csr_linear_equivalence_8.json)。
既有同权重、同数据的质量指标继续适用。

完整网络 profiling 中，CSR 合并、残差及附带下一层量化的总耗时，W4 从
37.184 ms 降到 27.099 ms，W8 从 37.155 ms 降到 27.786 ms。
prepare 分别约为 25.4 / 27.8 ms，基本保持不变；稀疏阶段减少量与无 profiler 的
整网延迟下降相符。原始报告为 `results/profile_csr_{legacy,linear}_{4,8}.json`。
