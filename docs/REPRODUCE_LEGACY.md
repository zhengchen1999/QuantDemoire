# ESDNet / UHDM 本地复现

**进度更新（2026-09-30）**：完整 500 对测试集已经补齐，FP32 主表结果已复现。
新的论文 PTQ 配方、随机 200 对训练清单、从头运行命令见
[PAPER_PROTOCOL.md](PAPER_PROTOCOL.md)，统一入口为 `script/reproduce_ptq.py`。
下文记录历史 100 图部署实验和额外 QAT 流程，不能替代新的论文复现结果。

当前实现只覆盖 ESDNet(48,32,64,32,1) 和 UHDM。默认浮点部署精度为 FP16
操作数、FP32 卷积累加；FP32 网络只用于质量参考。安装环境固定在 Python 3.11、
PyTorch 2.2.2+cu121、CUDA toolkit 12.1、CUTLASS 3.5.1。

```bash
uv sync --frozen
export CUDA_HOME=/path/to/cuda-12.1
bash script/setup_kernel.sh
uv run pytest -q
```

`CUDA_HOME` 必须指向包含 `nvcc` 的 toolkit；PyTorch wheel 内的 CUDA runtime
不足以编译扩展。已有 CUTLASS 时可设置 `CUTLASS_ROOT`，直接运行 Python 脚本。
已验证硬件为 RTX 4090；其他 SM8x GPU 仍需要独立的正确性和性能验证。

## 数据及原始权重

将 UHDM 的配对文件放在 `data/UHDM/train` 和 `data/UHDM/test`，文件名分别为
`*_moire.jpg`、`*_gt.jpg`。原始 checkpoint 放在 `checkpoints/uhdm_checkpoint.pth`。
模型和数据均不进入 Git。数据来源及 ESDNet 原始实现见
[CVMI-Lab/UHDM](https://github.com/CVMI-Lab/UHDM)。

本地实验使用现有的 200 对校准图和 100 对测试图，后者不是完整官方测试集。
另从训练池中选取 24 对未参与校准的图像，用于范围搜索和候选选取。
所有测试结果记录输入、GT、权重及配置哈希；不能把子集上的绝对 PSNR 当作
论文完整测试集的复现值。主表参考差距为 W8A8 0.12 dB、W4A4 1.04 dB，
见 [arXiv 表 4](https://arxiv.org/html/2510.04066v1#S4.T4)。

## 校准和导出

可从原始权重重新生成传统校准状态：

```bash
uv run python calibrate.py --config config/w4a4.yaml --calibrate-only
uv run python calibrate.py --config config/w8a8.yaml --calibrate-only
```

YAML 指定数据路径、随机种子、范围采样、smoothing 及频率校准。
当前实验起点是已有本地校准状态；重新校准需要保持相同数据及配置，结果仍须验证。
传统状态仅作为转换输入，发布后的推理不依赖它。示例：

```bash
uv run python script/export.py --checkpoint checkpoints/uhdm_checkpoint.pth \
  --quant-state /path/to/w4_quant_state.pth --bits 4 --tail-fraction 0.0025 \
  --weight-range mse --output checkpoints/base-w4a4
uv run python script/export.py --checkpoint checkpoints/uhdm_checkpoint.pth \
  --quant-state /path/to/w8_quant_state.pth --bits 8 --tail-fraction 0.0025 \
  --weight-range minmax --output checkpoints/base-w8a8
uv run python script/tune_ranges.py --model checkpoints/base-w4a4 \
  --pool /path/to/uhdm_training_pool --calibration data/UHDM/train \
  --factors 0.9 1 1.15 1.3 --output checkpoints/tuned-w4a4
```

导出按原始 smoothed weight 的排序保留两侧各至少 `ceil(0.0025*numel)` 个极值。
INT4/INT8 权重在这些位置为零，FP16 分支保留这些位置的权重；fakequant 与 CUDA
从同一份 safetensors 读取这些整数码、稀疏索引和离群值。

可选的补充量化感知微调使用 FP16 teacher、200 对训练图、512 裁剪，固定离群值
位置及其 FP16 值，训练普通整数权重、bias 和激活步长。它属于额外的部署优化，
并非声称严格复现论文原始 PTQ 配方：

```bash
uv run python script/polish.py --model checkpoints/tuned-w4a4 \
  --output checkpoints/qat-w4 --train-weights --epochs 4 --lr 0.001 --weight-lr 0.00001
```

仅使用留出的训练图选择 epoch 和范围参数。不得根据测试集分数选择候选。

以下入口依次执行导出、范围搜索、W4A4 补充训练及全部 epoch 的 CUDA 留出集
选择，随后评估和计时。W8A8 不执行这一步补充训练。`export` 以现有校准状态
为输入；从头校准仍使用上面的 `calibrate.py`。自动流程会重新选取候选，不能
保证不同硬件或重新校准后的结果与本地已保存权重逐位相同。

```bash
uv run python script/reproduce.py --stage export \
  --state4 /path/to/w4_quant_state.pth --state8 /path/to/w8_quant_state.pth \
  --train-pool /path/to/uhdm_training_pool
uv run python script/reproduce.py --stage polish --train-pool /path/to/uhdm_training_pool
uv run python script/reproduce.py --stage configure --main-output float16
uv run python script/reproduce.py --stage evaluate
uv run python script/reproduce.py --stage benchmark
```

`polish` 会替换 `checkpoints/reproduced/w4a4`，每个 epoch 和其留出集结果保存在
`checkpoints/reproduced/qat-w4a4`。训练/推理仍读取同一套离群值位置及 FP16 值。
`configure` 只修改主分支中间结果的存储精度，必须在评估之前运行；INT32 整数累加
与 FP32 校正/稀疏累加保持不变。发布包采用 `main_output=float16`。

## 质量、速度和显存

```bash
uv run python script/evaluate.py --backend fp32 --model checkpoints/uhdm_checkpoint.pth --output results/fp32.json
uv run python script/evaluate.py --backend fp16 --model checkpoints/uhdm_checkpoint.pth --output results/fp16.json
uv run python script/evaluate_fakequant.py --model checkpoints/final-w4a4 --output results/fake4.json
uv run python script/evaluate_cuda.py --model checkpoints/final-w4a4 --output results/cuda4.json
uv run python script/evaluate_fakequant.py --model checkpoints/final-w8a8 --output results/fake8.json
uv run python script/evaluate_cuda.py --model checkpoints/final-w8a8 --output results/cuda8.json
uv run python script/compare_quality.py --fp32 results/fp32.json \
  --results results/fp16.json results/fake4.json results/cuda4.json results/fake8.json results/cuda8.json \
  --output results/quality_summary.json
```

默认 fakequant 为 FP16 操作数/FP32 累加；加 `--fp32` 使用较慢的 FP32 伪量化参考。
后者也是逐层 CUDA 数值测试的 oracle。完整评估不裁剪原图，RGB PSNR、有效窗口
SSIM 和预训练 AlexNet LPIPS 均在裁剪回原尺寸、clamp 到 [0,1] 后计算。
`--limit`、`--crop` 仅用于开发；`--resume` 会校验数据及模型哈希。

```bash
uv run python script/benchmark_kernel.py --model checkpoints/final-w4a4 --bits 4 --output results/kernel4.json
uv run python script/benchmark_kernel.py --model checkpoints/final-w8a8 --bits 8 --output results/kernel8.json
uv run python script/benchmark_e2e.py --backend fp16 --model checkpoints/uhdm_checkpoint.pth --graph --output results/e2e_fp16.json
uv run python script/benchmark_e2e.py --backend cuda --model checkpoints/final-w4a4 --graph --output results/e2e4.json
uv run python script/benchmark_e2e.py --backend cuda --model checkpoints/final-w8a8 --graph --output results/e2e8.json
uv run python script/benchmark_memory.py --checkpoint checkpoints/uhdm_checkpoint.pth \
  --w4 checkpoints/final-w4a4 --w8 checkpoints/final-w8a8 --output results/memory.json
```

端到端默认输入 2160×3840，网络实际接收 2176×3840 的补齐输入。计时包括完整网络、
量化、packing、稀疏分支及所有输出；排除图片 I/O 和指标计算。比较双方需使用相同
CUDA Graph/eager 模式，测试时 GPU 不能同时运行其他任务。
显存测试在独立进程中分别报告模型常驻量、峰值 allocated/reserved；权重压缩比例
不能代替整网峰值显存节约。详细的开销定义和形状覆盖见 [kernel 说明](../kernel/README.md)。
完整网络的 CUDA kernel 分项 profiling 由 `script/profile_e2e.py` 提供，命令和
已测得的瓶颈见 [性能拆解](PERFORMANCE_BREAKDOWN.md)。
上一层输出与下一层量化的可选融合实验由 `QUANTDEMOIRE_FUSE_NEXT_QUANT=1` 开启；
完整对照计时、显存记录和验证命令见 [跨层量化融合](NEXT_QUANT_FUSION.md)。
后续的 [注意力与残差融合](POINTWISE_FUSION.md) 同时提供量化路径与 FP16 对照，
可用 `script/benchmark_variants.py` 按 JSON 配置在独立进程中顺序复测。
`QUANTDEMOIRE_CSR_KERNEL=linear` 进一步优化稀疏地址和空间访问，完整命令与全形状
验证见 [CSR 优化](CSR_OPTIMIZATION.md)。
整数卷积与稀疏分支协作输出的延迟、显存消融及复现命令见
[联合输出实验](JOINT_OUTPUT.md)。
输入准备的四通道向量化及失败候选记录见 [向量宽度消融](PREPARE_VECTOR_WIDTH.md)。
推荐的完整调度使用 [跨层成本调优](JOINT_DISPATCH.md)，复测当前选择可运行：

```bash
uv run python script/benchmark_variants.py --config config/experiments/optimized.json \
  --output results/optimized_retest.json --rounds 2 --repeats 30
uv run python script/benchmark_memory.py --checkpoint checkpoints/uhdm_checkpoint.pth \
  --w4 checkpoints/hf/esdnet-uhdm-w4a4 --w8 checkpoints/hf/esdnet-uhdm-w8a8 \
  --runtime-profile --output results/optimized_memory_retest.json
```

显存脚本的 `--runtime-profile` 为两份量化权重分别读取各自调度，并为 FP16 对照
开启适用的相同融合。单独测某份模型也可给 `benchmark_e2e.py` 添加此参数。
已有显式环境变量仍优先于模型包配置，实际生效设置会记录在结果中。

## 本地发布目录

```bash
uv run python script/verify_release.py --model checkpoints/final-w4a4 \
  --checkpoint checkpoints/uhdm_checkpoint.pth --output results/format4.json
uv run python script/package_model.py --model checkpoints/final-w4a4 \
  --output checkpoints/hf/esdnet-uhdm-w4a4 --quality results/cuda4.json
```

将位宽换为 8 得到另一个包。产物包含 safetensors、配置、模型卡、提供的评估记录和哈希清单；
只使用本地文件即可推理。Hugging Face 下载链接在实际发布后填写，当前没有上传或
创建远程模型仓库，也不执行 Git push。

当前已生成的本地包为 `checkpoints/hf/esdnet-uhdm-w4a4`、
`checkpoints/hf/esdnet-uhdm-w8a8`，索引见 [models.json](../models.json)。
默认仅为实际稀疏索引涉及的输入通道保存 planar 激活；设置
`QUANTDEMOIRE_COMPACT_SPARSE=0` 可复测压缩前实现。这不会删除或改变任何离群值。

已经验证的加速配置可随本地权重包保存，重建方式如下（W8 使用 variant `w8_metadata` 和对应的验证报告）：

```bash
uv run python script/attach_runtime_profile.py \
  --model checkpoints/hf/esdnet-uhdm-w4a4 \
  --benchmark results/two_successors_selected_e2e.json --variant w4_two_successors \
  --validation results/two_successors_equivalence_4.json
```

它增加 `runtime.json` 和 `joint_plan.json` 并更新模型卡、文件清单，不改变权重或
量化配置。下载或移动整个目录后，先调用 `apply_execution_profile` 再加载模型即可；
示例见 [权重格式](FORMAT.md#portable-execution-settings)。
