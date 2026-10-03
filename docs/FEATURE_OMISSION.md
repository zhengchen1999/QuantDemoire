# 省略已被量化缓存覆盖的 FP16 特征

ESDNet DenseBlock 的最后一个新特征只被 `conv_post` 使用；倒数第二个新特征
只被最后一层和 `conv_post` 使用。当前第一、第二消费者缓存各自使用消费者的
smoothing 和激活参数。这些缓存覆盖全部后续用途时，可以省去完整 FP16 特征的
分配和写出。更早的特征、残差输入和 `conv_post` 的输出仍保留。

## 实测结果

RTX 4090，固定 uv 环境，batch1，输入补齐为2176×3840，保留三个网络输出。
两轮独立进程、每轮30次 eager 推理，第二轮反转顺序；计时包含完整网络、量化、
packing、缓存和 FP16 分支，不含图片读取和指标计算。峰值为 PyTorch allocated
memory，包含激活和工作区。FP16 对照保留适用的注意力及残差融合。

| 模式 | 耗时 ms | 相对 FP16 加速 | 峰值 MiB | 相对 FP16 峰值节约 |
|---|---:|---:|---:|---:|
| FP16 操作数 / FP32 累加 | 129.910 | 1.000× | 3102.45 | — |
| W4A4 + FP16 branch | 79.965 | 1.625× | 2377.61 | 23.36% |
| W8A8 + FP16 branch | 94.775 | 1.371× | 2567.73 | 17.24% |

数据：[同批完整网络测量](../results/feature_omission_selected_e2e.json)。
这些数字针对已验证的本地100图子集及硬件，不是论文完整测试集结果或其他设备承诺。

另用备份的原 CUDA 二进制、原 `PackedConv.forward`、`DenseBlock.forward` 和
`ChannelConcat` 构造过程做四轮交替对比，每次使用新进程、10次预热、50次测量：

| 模式 | 原版中位数 ms | 新版中位数 ms | 四轮成对节约 ms | 峰值下降 MiB |
|---|---:|---:|---|---:|
| W4A4 | 80.748 | 80.005 | 0.633, 0.818, 0.759, 0.727 | 96.37 |
| W8A8 | 95.466 | 94.849 | 0.618, 0.569, 0.600, 0.686 | 126.75 |

见[成对原始记录](../results/feature_omission_production_pair.json)。不同测量批次有设备
漂移，不能把上面两张表相减归因于优化。

## 实现与调度

联合 kernel 的 output functor 仍完成原有 FP16 舍入和各消费者的量化。
专用编译单元关闭 CUTLASS 的完整 FP16 destination store，只保留缓存写出。
独立 CSR 版本仍读取整数主分支的 FP16 中间结果，保持 FP32 FMA 次序和最终
FP16 舍入，只省去最终特征的完整写出。

内部返回值为2字节、全零 stride 的形状占位 Tensor。它没有有效的完整特征内容。
`ChannelConcat` 记录是否含此类占位数据，消费者在任何 contiguous 转换前核对
缓存覆盖和所有权。第一段必须被完整 first-prefix 跳过；第二段必须有对应的
完整 second cache。原生输入准备仅对这些完全缓存的段接受占位布局，未缓存段
仍须连续。缺少覆盖、错误消费者或无效占位元数据均报错。

`QUANTDEMOIRE_OMIT_CACHED_FEATURES`：0关闭，1仅允许联合输出省略，2也允许独立
CSR 输出省略。原始默认值为0；模型目录的验证配置显式选择：

- W4使用1，恢复此前因峰值显存关闭的最终 producer prefix。省去对应 FP16
  特征后，这项融合的显存代价改变，实测峰值反而下降。
- W8使用2并保留原调度。恢复 W8 那处被排除的融合虽略快，峰值明显上升，因此未选。
- W4独立 CSR 省略的额外时延收益不稳定，峰值只少约0.52MiB，因此未选。

原型对比保留在[联合输出省略](../results/omit_feature_pair.json)、
[恢复被排除的融合](../results/omit_feature_restored_screen.json)和
[独立 CSR 扩展](../results/omit_feature_all_screen.json)。累计少写约1GB特征不代表
峰值下降1GB；峰值由同时存活的张量、其他层和分配器复用共同决定。

## 验证与质量

- 1331项单元测试通过；集成版本290项
  Compute Sanitizer memcheck 通过，关闭分配缓存时0错误。
- 两种位宽各157个量化权重形状通过；首末两个浮点卷积的先前覆盖保持不变。
- W4核对72条 first-prefix、34条 second-cache 边及26个被省略的特征；
  W8核对70条 first-prefix、32条 second-cache 边及27个被省略的特征。
- 占位数据的 oracle 来自未融合参考运行保存在CPU的真实特征，绝不读取占位数据
  来生成自身参考。验证包括未缓存后缀、缺失所有权、重复消费、非默认 stream
  和更换输入的 CUDA Graph。
- 两种位宽各100张本地 UHDM 图像的三个输出及 Graph 回放均逐位一致。
  原有PSNR、SSIM、LPIPS结果由这种等价性关联，未重新训练或改变测试集。

详见[完整验证及源码哈希](../results/feature_omission_validation.json)。整数码、
FP16离群值、稀疏结构、两侧各至少0.25%的原始排序尾部、smoothing和激活参数均未改动；
fakequant与部署继续读取同一份权重。占位数据只存在于运行时，不进入权重格式。
**完整 FP16 分支相对主分支≤30%的目标仍未认证。**

## 复现

按[环境和数据说明](REPRODUCE.md)准备CUDA12.1、CUTLASS3.5.1及本地模型，然后运行：

```bash
uv run pytest -q
uv run python script/benchmark_variants.py --config config/experiments/feature_omission_selected.json --output results/omission_recheck.json --rounds 2 --repeats 30
uv run python script/benchmark_e2e.py --backend cuda --model checkpoints/hf/esdnet-uhdm-w4a4 --runtime-profile --output results/omission_w4_e2e.json
uv run python script/verify_prepare.py --model checkpoints/hf/esdnet-uhdm-w4a4 --runtime-profile --reference scalar --candidate quad_swizzle --next-quant --check-prefixes --graph --output results/omission_w4_cache_check.json
uv run python script/verify_graph_fusions.py --backend cuda --model checkpoints/hf/esdnet-uhdm-w8a8 --runtime-profile --graph --output results/omission_w8_image_check.json
```

互换模型目录即可检查另一位宽。速度和峰值显存均在完整网络benchmark的JSON中记录。
`runtime.json`及模型卡已本地更新；未执行远程push或Hugging Face上传。
