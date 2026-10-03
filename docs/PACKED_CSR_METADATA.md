# 合并读取 CSR 地址与 FP16 权重

常用 interior 路径把每个稀疏元素的 32-bit 线性地址和 FP16 权重位模式组合成
一个 64-bit 派生条目。边界路径继续使用原始位置和权重。所有 CSR 行内 FMA
顺序、FP16 操作数舍入、FP32 累加及输出语义保持不变。

模型里的 safetensors、CSR 位置、两侧离群值和量化参数不变。缓存按输入几何
生成，不是新的权重存储格式；fakequant 仍使用原来的同一份稀疏权重。

## 独立计算原型

72 组几何、通道、残差和变化输入 Graph 检查通过。10 个高分辨率空间层的
合计准备加联合计算时间，W4 从 12.808 降到 12.592 ms，W8 从 15.426 降到
15.231 ms。这是层级筛选，不能直接代替整网结果。静态元数据生成在预热时完成，
不包含在逐次推理计时中，与原地址缓存的计时方式相同。

原型 SASS 出现了 `LDG.E.64`，原有独立 32/16-bit 读取数相应减少。所查询的
普通联合输出核仍为 64 registers/thread、24 KiB dynamic shared、8 bytes local
storage；这些静态数据不等于实际占用率或 spill 流量。

记录：[原型数值检查](../results/joint_packed_metadata_tests.json)、
[代表层计时](../results/joint_packed_metadata_screen.json)、
[统计汇总](../results/joint_packed_metadata_summary.json)、
[静态指令检查](../results/packed_metadata_sass_static.json)。

## 前缀与小通道路径组合

组合版保持 pointwise 共享输入路径不变，为其余联合核支持普通输出、直接前缀、
转置前缀以及 FewChannels 归约。400 组组合检查、两种位宽各 157 个量化层和
两张完整图像的三路输出及变化输入 Graph 均逐元素相同。

两轮独立进程、每轮 30 次 eager、第二轮反向顺序：

| 模式 | 原路径 ms | 元数据合并 ms | 原型峰值增加 |
|---|---:|---:|---:|
| W4A4 + FP16 分支 | 82.126 | 81.816 | 227840 bytes |
| W8A8 + FP16 分支 | 96.318 | 95.728 | 192512 bytes |

记录：[400 组组合检查](../results/joint_packed_metadata_full_tests.json)、
[原型整网对照](../results/packed_metadata_screen_e2e.json)、
[W4 全形状](../results/packed_metadata_all_shapes_4.json)、
[W8 全形状](../results/packed_metadata_all_shapes_8.json)。原型同时保留了原地址
和派生元数据，所以上表包含额外缓存；正式实现计划复用原来的地址缓存。

## 正式接口与验收状态

`QUANTDEMOIRE_PACKED_CSR_META=1` 默认关闭，仅对八 warp、变体 8/9、需要读取
planar 的 CSR 路径启用。超出 32-bit 安全寻址范围时，Python 调度使用原格式；
底层会拒绝不符合约束的 packed 缓冲，避免把它误当成普通 64-bit 地址。
新增的 native 尾部 `packed_metadata` 参数默认 false。

正式实现使用既有的非持久化地址缓存，按几何和表示方式切换重建；切换时不修改
调用者此前拿到的缓冲。刷新运行时模型缓冲会使缓存失效。部署权重按不可变数据
使用，不承诺任意原地修改权重后无需刷新即可复用派生缓存。

正式代码的 1,022 项测试通过，包括 400 组原生组合检查及两种位宽的缓存切换检查。
两种位宽各 157 个实际量化层、原生前缀和 100 张本地图像的三个输出与变化输入
Graph 均通过逐元素相同检查。本地推荐模型包已启用该选项；权重与量化配置哈希
不变，既有质量结果继续适用。完整稀疏分支 ≤30% 预算仍是待验收项。

```bash
uv run python script/benchmark_variants.py \
  --config config/experiments/packed_metadata.json \
  --output results/packed_metadata_retest.json --rounds 2 --repeats 30
uv run python script/benchmark_variants.py \
  --config config/experiments/metadata_selected.json \
  --output results/metadata_selected_retest.json --rounds 2 --repeats 30
```

## 正式同批性能与验收

| 两轮 eager 中位数 | 时延 ms | 相对 FP16 加速 | 峰值 allocated MiB |
|---|---:|---:|---:|
| FP16 操作数 / FP32 累加 | 130.152 | 1.000× | 3102.45 |
| W4A4 + FP16 分支 | 81.759 | 1.592× | 2473.98 |
| W8A8 + FP16 分支 | 95.480 | 1.363× | 2696.23 |

记录：[生产开关对照](../results/metadata_e2e.json)、[同批 FP16 对照](../results/metadata_selected_e2e.json)、[验收汇总](../results/metadata_validation.json)。
输入补齐到 2176×3840，batch=1，包含三个输出及所有准备，不含图像 I/O 和质量指标。
FP16 同样启用适用的融合。100 张图像是本地 UHDM 子集，不是完整官方测试集。

W4 当前计划的独立计算阶段：主计算 24.439 ms，含稀疏 35.608 ms，增量 45.70%。
W8 当前计划的独立计算阶段：主计算 36.308 ms，含稀疏 46.351 ms，增量 27.66%。
这些数字排除了准备与生产者前缀，不能单独证明完整分支预算达标。
上一推荐配置已保存在 `config/experiments/optimized_few.json`；没有 push 或上传。
