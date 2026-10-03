# 输入准备的四通道向量化消融

本轮保持已验证的按层联合输出配置、权重和稀疏结构不变，继续优化 smoothing、
量化、packing 和稀疏 planar 输入的生成。

`QUANTDEMOIRE_PREPARE_IMPL=quad` 每线程处理四个通道：使用 64-bit FP16 输入
读取，将四个 INT8 code 存入一个 shared-memory word，INT4/INT8 输出分别用
16-bit/32-bit 写入。大图使用 64 个像素、128 个通道的 tile。原 pair 使用
128 个像素、64 个通道的 tile，两个 code 占一个 shared-memory word。
不满足四通道对齐时自动回退到 pair，仍不满足双通道对齐时使用 scalar。
已准备的 32 通道前缀可与新 kernel 配合使用。

## 首轮结果：完全展开的 quad 更慢

RTX 4090，原生 UHDM 尺寸，eager；每项独立进程，7 次预热、30 次计时，两轮
反转配置顺序。使用 `config/experiments/prepare_quad.json`。

| 输入准备 | W4A4 ms | W8A8 ms |
|---|---:|---:|
| pair | **89.457** | **102.798** |
| quad | 91.704 | 104.703 |

四通道版本未获得端到端收益，因此不替换当前选定的 pair。峰值 allocated 保持相同。
记录：[prepare_quad_e2e.json](../results/prepare_quad_e2e.json)。完整测试 107 项通过，
但本轮未为更慢的候选另做 100 图质量确认；不能把测试覆盖描述为已完成全数据验证。

`cuobjdump` 显示，大图且生成稀疏输入时，pair 使用 48 registers/thread、16896
bytes shared/block，quad 使用 72 registers/thread、8448 bytes shared/block，均没有
local-memory spill。减少 shared memory 并不代表总资源限制一定缓解；这些数据为
编译资源统计，不是实际 occupancy 或 DRAM 带宽测量。
记录：[prepare_quad_compiled_resources.txt](../results/prepare_quad_compiled_resources.txt)。

后续 `quad_loop` 保留四通道处理，但禁止两个外层循环展开，用于检验寄存器与循环
控制开销的权衡。最终是否采用以完整网络测量及数值验证为准。

```bash
uv run python script/benchmark_variants.py --config config/experiments/prepare_quad_loop.json \
  --output results/quad_loop_retest.json
uv run python script/benchmark_prepare.py --model checkpoints/hf/esdnet-uhdm-w4a4 \
  --implementations pair quad quad_loop --output results/prepare_vectors4.json
```

## 减少循环展开

`quad_loop` 完整单元测试 131 项通过。大图 W4 kernel 的寄存器降为 35（完整
输入）或 33（补齐前缀以外的输入），shared memory 仍为 8448 bytes/block，
没有 local-memory spill。两轮整网结果如下：

| 输入准备 | W4A4 ms | W8A8 ms |
|---|---:|---:|
| pair | 89.515 | 102.744 |
| quad | 91.763 | 104.250 |
| quad_loop | **89.048** | **102.236** |

改善约 0.47 / 0.51 ms，只有约 0.5%。记录：
[整网](../results/prepare_quad_loop_e2e.json)、
[编译资源](../results/prepare_quad_loop_compiled_resources.txt)。当前默认和已验证推荐
配置当时仍保持 pair，后续经过下述完整验证才更新推荐配置。

`quad_swizzle` 继续尝试 64 通道、128 像素的 tile，以匹配 pair 的通道分块，
避免 128 通道 tile 在部分层增加空线程。线程块调整为 16×16，并通过 XOR
重排 shared-memory 列索引，使输入写入与 planar 转置读取使用同一紧凑布局。
它是待测候选，不预先假定比 pair 或 quad_loop 更快。

155 项单元测试通过后的两轮测量：

| 输入准备 | W4A4 ms | W8A8 ms |
|---|---:|---:|
| pair | 89.391 | 102.548 |
| quad_loop | 89.033 | 102.185 |
| quad_swizzle | **88.459** | **101.870** |

相同轮次中改善约 1.04% / 0.66%，峰值 allocated 不变。新 kernel 使用约 35 个
寄存器及 8192 bytes shared/block，没有 local-memory spill。记录：
[整网测量](../results/prepare_quad_swizzle_e2e.json)、
[编译资源](../results/prepare_quad_swizzle_compiled_resources.txt)。

两种位宽的全部 157 个真实输入形状及 concat 组合中，四种候选的 packed/planar
整数内容均与 scalar 参考逐位一致。记录：
[W4](../results/prepare_vectors_all_shapes_4.json)、[W8](../results/prepare_vectors_all_shapes_8.json)。
这些微基准测完整输入准备，不含跨层前缀；与整网中的部分输入准备及缓存状态不同，
不能把其合计时间当作整网对应阶段的精确耗时。

最终 quad_swizzle 在两种位宽各 100 张本地图像上，三个网络输出和更换输入的
CUDA Graph 重放均逐位一致。首张图像还逐条核对实际启用的跨层 packed/planar
缓存内容：[W4](../results/quad_swizzle_equivalence_4.json)、
[W8](../results/quad_swizzle_equivalence_8.json)。
汇总见 [quad_swizzle_validation.json](../results/quad_swizzle_validation.json)。
模型权重及既有质量指标不变，`config/experiments/optimized.json` 已更新为该实现；
上轮 pair 推荐配置另存 `optimized_pair.json`。程序本身的默认环境开关仍为 pair，
每个历史消融的配置文件均保留。
