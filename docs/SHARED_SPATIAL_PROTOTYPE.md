# 3×3 主循环共享输入原型（未采用）

独立原型在 CUTLASS 同步双缓冲主循环的每个 RS 步骤读取 shared A，并维持八个
FP32 稀疏累加器。只支持一个输入通道 tile，按 RS/C 原顺序累加，显式跳过 padding；
其余形状回退到已验证路径。原型没有修改生产 kernel 或模型配置，复制的 CUTLASS
代码保留 BSD-3-Clause 许可。

16 组两位宽、尾部通道、stride/dilation/padding、残差、stream 与变化输入 Graph
检查通过，见 [原型检查](../results/shared_spatial_prototype_tests.json)。随后用真实
权重和 4K 网络中的 16 个代表层筛选，全部更慢，见
[计时报告](../results/shared_spatial_screen.json)。这些仅是候选筛选形状，不是完整
157 层覆盖；由于已经显著退步，没有推进到整网部署。

| 代表层：encoder.encoder_1.rdb.conv_layers.0.conv.0 | 已验证 kernel ms | 原型 ms |
|---|---:|---:|
| W4 仅计算 | 0.747 | 2.282 |
| W4 包括完整输入准备 | 1.070 | 2.464 |
| W8 仅计算 | 0.784 | 1.795 |
| W8 包括完整输入准备 | 1.149 | 2.137 |

编译资源为 64 registers/thread、40-byte stack、160-byte static shared，原 variant 8/9
为 8-byte stack；这不是实际 spill 流量或 achieved occupancy 的测量。持续稀疏累加
与额外寻址进入主循环后，即使删掉辅助输入准备也无法抵消开销。该原型不进入
推荐配置。未来若重试，需要先降低跨 MMA 存活的状态或提供新的资源/计时证据。
