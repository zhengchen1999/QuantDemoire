# 独立 CSR 的地址计算优化

`linear32`（mode 4）沿用 `linear` 的每 lane 两个空间位置与 FP32 FMA 顺序，
但在输入总元素数小于 INT_MAX 时使用 32 位中间地址计算。`divmod`（mode 5）
再将输出坐标和 batch 拆解换成预计算的 CUTLASS FastDivmod。超过地址范围时回退
原 64 位 linear 路径。权重、稀疏索引、半精度操作数舍入与输入量化均未改变。
辅助前缀写入、残差、跨位宽量化和 metadata-only 输入继续由原条件控制。

## 独立原型筛选

原型单独链接，没有替换生产 JIT binary。14 项 CSR 检查通过，随后测试当前调度
中所有独立 CSR 层；W4 为 26 层，W8 为 64 层，包含实际需要的残差和下一层前缀。
输出与前缀有效区域均和旧路径一致。

| 独立 CSR 层耗时之和 ms | linear | linear32 | divmod |
|---|---:|---:|---:|
| W4，26 层 | 9.770 | 9.600 | 9.310 |
| W8，64 层 | 10.244 | 10.097 | 9.882 |

这是 CSR 输出阶段的计时，不能当成整网延迟或全部 157 层开销。
[W4 原始记录](../results/csr_address_shapes_4.json)、
[W8 原始记录](../results/csr_address_shapes_8.json) 包含每层信息。

两个顺序反转的独立进程轮次得到：

| 整网 ms | 原 linear | 候选 divmod |
|---|---:|---:|
| W4 | 84.075 | 83.499 |
| W8 | 100.199 | 99.857 |

峰值 allocated 不变。见 [原型端到端筛选](../results/csr_address_screen_e2e.json)。
这些报告显式标记 isolated_prototype，不能直接写入发布包的已验证 runtime profile。

## 接入后的验收

两种模式已接入可选 CUDA 路径，完整 574 项测试通过。接入后按当前 W4/W8 调度
重新测试每种位宽全部 157 个量化层、真实前缀、100 图和变化输入 Graph，再使用
同批 FP16 对照计时。通过前保持上一份推荐 runtime profile；最终记录另行更新。

完整验收已通过：每种位宽均覆盖 157 个量化层及全部 100 图、三个输出和变化输入
Graph，另检查真实融合前缀。见 [验收汇总](../results/csr_divmod_validation.json)。

接入后的 [新旧路径配对计时](../results/csr_divmod_e2e.json) 两轮方向一致：
W4 linear / divmod 为 84.171 / 83.685 ms，W8 为 100.340 / 99.968 ms；峰值不变。
这组配对实验不含 FP16，不与别轮基线拼接计算加速比。

[同批 FP16 对照](../results/csr_divmod_selected_e2e.json) 得到：

| 当前推荐 | 两轮平均中位时延 ms | 相对 FP16 | 峰值 allocated MiB |
|---|---:|---:|---:|
| W4 + divmod CSR | 84.028 | 1.539× | 2473.62 |
| W8 + divmod CSR | 100.096 | 1.292× | 2797.60 |
| FP16 + 相同适用融合 | 129.308 | 1.000× | 3102.45 |

模型 runtime.json、models.json 与 optimized.json 已更新，权重不变，没有上传。
当前混合调度的逐层计算增量约 W4 62.3%、W8 41.9%；该诊断排除 prepare 和
生产者前缀，并把 separate 路径的 merge/activation 包含在增量中。
它不是整网开销；30% 分支目标仍未完成，也不应借全联合模式的另一组分母宣称达标。
