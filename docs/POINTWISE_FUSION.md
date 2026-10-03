# ESDNet 注意力加权与残差融合

本轮新增两个独立开关，默认关闭：

- `QUANTDEMOIRE_FUSE_ATTENTION=1`：SAM 三路加权、求和和外层残差合并为一个 CUDA kernel。
  适用于量化部署和 FP16 对照。
- `QUANTDEMOIRE_FUSE_RESIDUAL=1`：量化 RDB 的最终 CSR 合并直接执行 residual add。
  适用于 FP16 主分支中间结果和 CSR 路径；FP16 cuDNN 基线仍使用原来的 RDB 加法。

卷积依然使用规定的 FP16 操作数 / FP32 浮点累加。逐点运算保留原 PyTorch 算子
之间的 FP16 舍入：每次乘法和加法都分别舍入，使用 `_rn` half/half2 intrinsic
阻止 FMA contraction。RDB 先生成与原来完全相同的 FP16 卷积输出，再执行残差加法。
权重、激活量化参数、稀疏索引和两侧离群值覆盖均保持不变。

## 原生 UHDM 端到端消融

RTX 4090，2160×3840 输入补齐到 2176×3840，eager，三个输出，包含全部准备、
整数卷积和稀疏分支，不含图像 I/O 与指标计算。量化各组均开启上一轮的
`QUANTDEMOIRE_FUSE_NEXT_QUANT=1`，使用 pair prepare 和 compact CSR。

每种配置在独立进程中预热 7 次、计时 30 次，共两轮，第二轮反向执行配置列表。
下表为两轮中位数的中位数；“两者”对应注意力融合和 RDB 残差融合。

| 配置 | W4A4 ms | W8A8 ms | FP16 ms |
|---|---:|---:|---:|
| 两项均关闭 | 112.417 | 125.625 | 136.932 |
| 仅注意力融合 | 104.917 | 118.041 | **129.604** |
| 仅 RDB 残差融合 | 110.722 | 123.730 | — |
| 两者开启 | **103.091** | **116.362** | — |

量化整网延迟分别下降 **8.30% / 7.37%**；FP16 注意力融合下降 **5.35%**。
相对使用相同注意力融合的 FP16 对照，本轮 W4A4 / W8A8 约为 **1.257× / 1.114×**。
这是图优化与整数部署的共同结果，不把通用注意力融合收益全部归给量化。

| 峰值 allocated MiB | W4A4 | W8A8 | FP16 |
|---|---:|---:|---:|
| 两项均关闭 | 2851.396 | 3013.374 | 3104.450 |
| 本轮最佳延迟配置 | 2789.122 | 3012.245 | 3102.450 |

原始样本及环境、权重、源码哈希见 [完整消融](../results/pointwise_e2e.json)，
配置见 [pointwise.json](../config/experiments/pointwise.json)。

## 复现

先按 [环境文档](REPRODUCE.md) 配置 CUDA 12.1 / CUTLASS 3.5.1。

```bash
uv run python script/benchmark_variants.py --config config/experiments/pointwise.json \
  --output results/pointwise_retest.json

QUANTDEMOIRE_FUSE_NEXT_QUANT=1 QUANTDEMOIRE_FUSE_ATTENTION=1 QUANTDEMOIRE_FUSE_RESIDUAL=1 \
  uv run python script/benchmark_e2e.py --backend cuda \
  --model checkpoints/hf/esdnet-uhdm-w4a4 --output results/optimized4.json

QUANTDEMOIRE_FUSE_NEXT_QUANT=1 uv run python script/verify_graph_fusions.py \
  --backend cuda --model checkpoints/hf/esdnet-uhdm-w4a4 --graph \
  --output results/pointwise_verify4.json

uv run python script/verify_graph_fusions.py --backend fp16 \
  --model checkpoints/uhdm_checkpoint.pth --graph --output results/pointwise_verify_fp16.json
```

将 `w4a4` 改为 `w8a8` 可验证另一位宽。新增的点运算和 residual 测试覆盖 batch=2、
通道尾部、dilation、stride、半精度微小值、非默认 stream 和更新输入的 CUDA Graph。

## 完整数值验证

38 项单元测试通过。W4A4、W8A8 和 FP16 各自全部 100 张本地 UHDM 图像，
三个输出及更新输入后的 CUDA Graph 重放均与关闭两项融合时逐位一致。
记录：[W4A4](../results/pointwise_equivalence_4.json)、
[W8A8](../results/pointwise_equivalence_8.json)、
[FP16](../results/pointwise_equivalence_fp16.json)。因此同权重、同一批图的既有质量指标保持不变。
