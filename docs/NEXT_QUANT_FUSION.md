# 上一层输出与下一层量化融合

已实现可选的 producer epilogue 融合路径。RTX 4090 上的原生 UHDM 输入测试中，
W4A4 整网延迟降低约 **1.3%**，W8A8 降低约 **2.0%**。默认关闭，使用
`QUANTDEMOIRE_FUSE_NEXT_QUANT=1` 开启，便于在其他设备上独立比较。

## 实现范围与数值语义

ESDNet 的 24 个 DenseBlock/RDB 里共有 **108 条相邻卷积边**。每层产生 32 个
新通道，并将它们放到下一层 concat 输入的最前面。本实现针对这部分输入：

1. 在上一层的 FP16 sparse merge 输出处理里，完成合并、ReLU 和 FP16 舍入。
2. 在寄存器中将已舍入的 FP16 值转换回 FP32，按下一层的
   `inverse_smooth / activation_step` 乘法、加零点、round-to-even 和饱和规则量化。
3. 同时写出下一层的 packed INT4/INT8 前缀，以及 compact planar 稀疏输入前缀。
4. 下一层 prepare 只处理剩余通道和尾部 padding，直接复用已分配的输出缓冲。

后续其他卷积仍需访问这 32 个通道，所以继续保存原有 FP16 特征。
不同消费者的 smoothing 和量化参数独立，不能把同一份整数激活直接共享给其他层。
稀疏索引、值、离群值覆盖和模型文件格式保持不变，无需重新导出权重或校准。

支持条件是 CUDA backend、FP16 dense intermediate、compact CSR sparse branch、
`pair` prepare 和已融合的 ReLU；不满足条件的边保留原有执行方式。
发布目录 `checkpoints/hf/esdnet-uhdm-w4a4` 和 `esdnet-uhdm-w8a8` 满足这些条件。
这是部分输入融合，仍有 **157 次 prepare 启动**；108 次 sparse merge 启动改为
同时生成下一层前缀的版本，启动总数保持不变。

## 端到端结果

RTX 4090，batch=1，输入 2160×3840、补齐后 2176×3840。两边都使用 `pair` prepare、
相同权重和全部三个网络输出。每种模式独立进程预热 7 次、计时 30 次，做两轮，
顺序为 off/on/on/off。下表为两轮中位数的中位数，计时包含所有 CUDA 网络操作，
不包含图片读取、传输和指标计算；执行模式为 eager。

| 指标 | W4A4 | W8A8 |
|---|---:|---:|
| 关闭融合，ms | 113.773 | 128.264 |
| 开启融合，ms | **112.306** | **125.663** |
| 延迟减少 | **1.29%** | **2.03%** |
| 关闭融合，峰值 allocated MiB | 2850.021 | 3012.374 |
| 开启融合，峰值 allocated MiB | 2851.396 | 3013.374 |
| 额外峰值 allocated MiB | 1.375 | 1.000 |

原始记录：[W4A4](../results/next_quant_lifetime_e2e_4.json)、
[W8A8](../results/next_quant_lifetime_e2e_8.json)，各记录链接到独立运行的样本、
峰值显存、权重/配置哈希、CUDA 源码哈希及 Python 部署源码哈希。

首版曾使 W8A8 峰值显存增加约 168 MiB。原因包括 concat 对已消费整数输入的引用
及循环局部变量延长了缓存生命周期。当前版本在消费后清空前缀缓存，并在当前层
整数卷积完成使用后允许同一 stream 复用 packed 输入存储。前缀缓存按单次调用
生成，不存放在模块上，避免跨图像或跨消费者复用旧数据。首版记录仍保留在
`results/next_quant_e2e_{4,8}.json`，上表使用生命周期修正后的记录。

## 为什么收益有限

同轮完整网络 CUDA profiling（3 帧平均 kernel duration）给出：

| GPU kernel 分组 | W4A4 关闭→开启，ms | W8A8 关闭→开启，ms |
|---|---:|---:|
| 独立 quantize/smoothing/packing/planar prepare | 30.235 → 25.519 | 33.707 → 27.967 |
| sparse merge，含融合后附加的下一层量化 | 33.645 → 36.863 | 33.797 → 36.847 |
| 这两项的合计节省 | **1.498** | **2.691** |

准备阶段分别少了 4.716 / 5.740 ms，但输出处理增加 3.218 / 3.049 ms，抵消了
大部分收益。量化算术和整数输出写入仍然存在，部分工作只是搬到了 producer。
此表不能把融合后的 sparse merge 耗时全部归因于稀疏计算。

按实际 108 条边的张量尺寸计算，每次推理省去约 **3.356 GB 的逻辑 FP16 输入读取**，
约占全部准备阶段 FP16 输入读取量的 **18.2%**。这不是实测 DRAM 流量，缓存命中和
内存事务会影响实际访问量。其余旧特征仍需按下一层参数量化；当前实现也仍保留
FP16 输出、packed 输出和 compact planar 输出，不能把整个准备阶段全部消除。

profiling 原始记录：[W4 关闭](../results/profile_next_quant_0_4.json)、
[W4 开启](../results/profile_next_quant_1_4.json)、
[W8 关闭](../results/profile_next_quant_0_8.json)、
[W8 开启](../results/profile_next_quant_1_8.json)。profiling 用来拆解成本，整网延迟
使用上面的独立无 profiler 计时。没有将寄存器占用或 DRAM 带宽推测当成测量结果。

这一步没有解决 FP16 稀疏分支低于主分支 30% 的目标。进一步优化需要改善融合输出
布局、稀疏读取/写回，或覆盖更多消费者；对不同消费者采用统一量化参数会改变数值
行为，需要独立校准和质量验证。

## 正确性验证

- 两种位宽分别覆盖本地全部 100 张 UHDM 图像。融合前后的三个整网输出均逐位一致，
  每张图对应的 CUDA Graph 更新输入重放结果也逐位一致。
- 首张原生分辨率图像检查了全部 108 条融合边，packed INT4/INT8 和 compact planar
  整数缓冲均与单独 prepare 的结果逐位一致，包括下一层补齐的剩余通道和 padding。
- 新增 8 项单元测试覆盖连续融合、W4/W8 及混合位宽、batch=2、奇数空间尺寸、1×1
  输入、dilation、ReLU、饱和、不同消费者、非默认 stream 和更新输入的 CUDA Graph。

验证记录：[W4A4](../results/next_quant_network_equivalence_4.json)、
[W8A8](../results/next_quant_network_equivalence_8.json)。由于这 100 张图的输出逐位一致，
此前相同权重的质量指标继续适用；没有重新宣称复现完整官方测试集或进行新一轮校准。

## 复现

先按 [环境配置](REPRODUCE.md) 设置 CUDA 12.1 和 CUTLASS 3.5.1。

```bash
# 交替开关融合的整网延迟和峰值显存比较；将 w4a4 换为 w8a8 测试另一位宽。
uv run python script/benchmark_next_quant.py \
  --model checkpoints/hf/esdnet-uhdm-w4a4 --output results/next_quant_compare4.json

# 单独开启融合。
QUANTDEMOIRE_FUSE_NEXT_QUANT=1 uv run python script/benchmark_e2e.py \
  --backend cuda --model checkpoints/hf/esdnet-uhdm-w4a4 --output results/fused_next4.json

# 全部本地 UHDM 图像：三个输出、108 条边的 packed/planar、更新输入的 CUDA Graph。
uv run python script/verify_prepare.py --model checkpoints/hf/esdnet-uhdm-w4a4 \
  --reference pair --candidate pair --next-quant --check-prefixes --graph \
  --output results/next_quant_verify4.json

uv run pytest tests/test_next_quant.py -q
```

CUDA 实现见 [half_branch.cu](../kernel/csrc/half_branch.cu) 和
[epilogue.cu](../kernel/csrc/epilogue.cu)。所有操作仅作用于本地工作目录。
