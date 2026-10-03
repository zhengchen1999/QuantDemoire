# W4 稀疏操作数共享查表

W4 激活只有16种整数码。联合 kernel 在整数主分支完成后，每个 warp 将这16种
码按照原激活参数反量化、舍入到 FP16，再转为 FP32 放入共享表。内部像素的
CSR 循环读取表中的值，省去重复转换；每一项 FP32 FMA 的顺序保持不变。
边界处理继续使用原始坐标与 FP16 舍入。没有改动量化参数、离群值或权重格式。

## 当前测量

RTX4090，固定 uv 环境，batch1、补齐输入2176×3840，保留三个输出。
两轮独立进程、每轮30次 eager 推理，第二轮反转顺序。计时包含完整网络、
量化、packing、稀疏分支和融合缓存，排除图片 I/O 与指标计算。FP16 对照采用
FP16 操作数、FP32 卷积累加，并保留适用的注意力/残差融合。峰值为 PyTorch
allocated memory，包含激活与工作区。

| 模式 | 耗时 ms | 相对 FP16 加速 | 峰值 MiB | 相对 FP16 峰值节约 |
|---|---:|---:|---:|---:|
| FP16 / FP32 累加 | 129.876 | 1.000× | 3102.45 | — |
| W4A4 + FP16 分支 | 79.583 | 1.632× | 2377.61 | 23.36% |
| W8A8 + FP16 分支 | 94.595 | 1.373× | 2567.73 | 17.24% |

见[同批整网结果](../results/shared_lut_selected_e2e.json)。针对查表本身另做四轮
交替的独立进程对照：加载备份的旧 CUDA 二进制，Python、模型和执行计划完全
相同；每次10次预热、50次测量。W4中位数从80.086ms
变为79.849ms；四轮中3轮更快，成对节省分别为
-0.058, 0.316, 0.158, 0.217ms（负值表示变慢），平均节省0.158ms，峰值不增加。
见[原版二进制对照](../results/shared_lut_production_pair.json)。收益较小；不能将
不同批次的绝对耗时相减来归因于此优化。W8没有新增查表路径。

## 实现边界

`run_joint_selected` 只给 W4、八个warp、空间卷积、平衡CSR行及打包元数据的
专门编译实例启用共享表，涵盖普通/FewChannels、第一/第二消费者前缀和
[省略已缓存FP16特征](FEATURE_OMISSION.md)。其他实例保留原实现。

表共512字节，位于主循环退出后复用的共享区域，避开epilogue工作区、CSR
输出tile及转置前缀tile；编译期检查它位于原24KiB分配内。每个warp独占自己的
16项，初始化后使用warp同步。没有将稀疏累加器放进整数主循环。

显式的 `joint_lut_resources`、`joint_two_lut_resources`、
`joint_omit_lut_resources`、`joint_omit_two_lut_resources` 用于查询这些实例。
旧资源查询继续返回未启用查表的实例。实际资源见
[资源记录](../results/shared_lut_integrated_resources.json)；静态local字节数
不等于实测spill流量。

## 验证与复现

- 1379项回归测试通过；包含极小尺度、不同零点、奇数边界、
  空间步长/膨胀、大通道1×1回退、非默认stream和变更输入的CUDA Graph。
- 集成版本338项uncached Compute Sanitizer memcheck，0错误。
- W4/W8各157种量化卷积形状通过；首末两个浮点卷积仍由既有覆盖检查。
- W4验证72条第一消费者缓存、34条第二消费者缓存及26个省略特征；
  W8为70、32、27。占位特征的参考来自未融合运行保存的真实数据。
- 两种位宽各100张本地UHDM图像的三个输出及Graph回放逐位一致；原有PSNR、
  SSIM和LPIPS通过这种等价性关联。测试集是本地子集，不是完整官方划分。
- 新模型目录的runtime profile通过公开验证CLI检查，全部artifact manifest哈希通过。

详见[完整验证与源码哈希](../results/shared_lut_validation.json)和
[可移植配置检查](../results/shared_lut_portable_profile_smoke.json)。
原始两侧排序尾部各至少0.25%，fakequant与CUDA共享整数码和CSR离群值，均未改变。
**完整FP16分支相对主分支≤30%的目标仍未认证。**

按[环境说明](REPRODUCE.md)准备数据和模型后：

```bash
uv run pytest -q
uv run python script/benchmark_variants.py --config config/experiments/shared_lut_selected.json --output results/lut_recheck.json --rounds 2 --repeats 30
uv run python script/benchmark_e2e.py --backend cuda --model checkpoints/hf/esdnet-uhdm-w4a4 --runtime-profile --output results/lut_w4.json
uv run python script/verify_prepare.py --model checkpoints/hf/esdnet-uhdm-w4a4 --runtime-profile --reference scalar --candidate quad_swizzle --next-quant --check-prefixes --graph --output results/lut_prefixes.json
```

模型权重和config哈希保持不变，runtime profile及模型卡只在本地更新，未push或上传。
