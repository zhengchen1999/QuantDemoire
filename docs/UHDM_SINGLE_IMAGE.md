# 单张真实 UHDM 图像：速度和峰值显存

使用 `data/UHDM/test/0000_moire.jpg`，原图 2160×3840，补齐后输入为
`1×3×2176×3840`。GPU 为 RTX 4090，PyTorch 2.2.2 + CUDA 12.1。
按本次要求固定已有 FP16 基线：FP16 操作数、FP32 卷积累加，保持已有 attention
融合，不启用新增的特征释放。W4/W8 启用特征释放，量化权重和 CUDA 算术保持不变。

| 模式 | 延迟 ms | 相对 FP16 加速 | 峰值 allocated GiB | 峰值显存节约 |
|---|---:|---:|---:|---:|
| FP16 / FP32 累加 | 129.88 | 1.000× | 3.032 | — |
| W4A4 + FP16 稀疏分支 | 79.31 | 1.638× | 1.355 | 55.32% |
| W8A8 + FP16 稀疏分支 | 94.50 | 1.374× | 1.608 | 46.96% |

使用 eager 完整网络推理，保留三个输出，计入 smoothing、量化、packing 和 FP16
稀疏分支，排除图像 I/O、输入传输/补齐和质量指标。每种模式在两个独立进程中各测
一轮，第二轮反转测试顺序；每轮预热 10 次，CUDA events 计时 50 次。
延迟取两个轮次中位数的中位数，显存取各轮预热后 `max_memory_allocated()` 的最大值。
三个模式的峰值 allocated 在两轮中分别完全一致。

- 加速比 = FP16 延迟 / 量化模式延迟。
- 显存节约 = 1 − 量化模式峰值 allocated / FP16 峰值 allocated。
- 这里包含模型、输入、输出和推理临时张量的显存；CUDA context 与 allocator
  reserved 缓存不计入 allocated。原始记录同时保留 reserved 数值。

两轮延迟分别为 FP16 **129.8099 / 129.9579 ms**、W4 **78.9079 / 79.7184 ms**、
W8 **94.2822 / 94.7159 ms**。W4 本轮主要新增收益来自及时释放已用完的中间特征；
此改动的目的是降低显存，不能将相对旧批次的亚毫秒波动解释为稳定加速。

## 复现

在仓库根目录、已按 [环境说明](REPRODUCE.md) 配置 CUDA/CUTLASS 后运行：

```bash
PYTHONPATH=. uv run python script/benchmark_variants.py \
  --config config/experiments/uhdm_single_image.json \
  --image data/UHDM/test/0000_moire.jpg \
  --rounds 2 --warmup 10 --repeats 50 \
  --output results/uhdm_single_image_e2e.json
```

`--image` 使用真实图像原生尺寸，不传时保留原有随机输入测试。
配置明确设置 W4/W8 的 `QUANTDEMOIRE_RELEASE_FEATURES=1`，FP16 为 `0`。
本次没有修改 `optimized.json` 或权重目录的默认 runtime profile。

[完整计时与配置](../results/uhdm_single_image_e2e.json) 保留六个独立进程的原始文件链接，
[比例汇总](../results/uhdm_single_image_summary.json) 保留计算口径、图像及配置哈希。
新增特征释放实现的 [1381 项测试](../results/feature_lifetime_integrated_pytest.json) 已通过。
该图像在特征释放开关前后的三个输出均逐位一致，CUDA Graph 输出也一致：
[W4 核对](../results/uhdm_single_image_equivalence_w4a4.json)、
[W8 核对](../results/uhdm_single_image_equivalence_w8a8.json)。
本次仅测试这一张图像的速度和显存；没有重跑数据集质量评估。
