# ESDNet / UHDM 复现与部署

本页是当前入口；[历史部署实验](REPRODUCE_LEGACY.md)包含旧 100 图评估、范围搜索及
额外 QAT，不能用于描述当前 PTQ 权重。当前发布范围仅为 ESDNet / UHDM。

## 安装

已验证环境：RTX 4090、Python 3.11、PyTorch 2.2.2+cu121、CUDA toolkit 12.1、
CUTLASS 3.5.1。SM80/86 需要独立验证；不支持设备不会回退到浮点卷积。

```bash
uv sync --frozen
export CUDA_HOME=/path/to/cuda-12.1
bash script/setup_kernel.sh
uv run pytest -q
```

`CUDA_HOME` 需含 `nvcc`；可用 `CUTLASS_ROOT` 指定固定版本的已有源码。
独立研究候选位于 [research/](../research/README.md)，不属于默认测试和 wheel。

## 数据与原始 checkpoint

从 [UHDM 官方仓库](https://github.com/CVMI-Lab/UHDM)取得 ESDNet 的 UHDM checkpoint，
保存为 `checkpoints/uhdm_checkpoint.pth`。本地参考 SHA256：
`254235cd25f90a3f1785885385dc6cb3f2178e053291ab53d1943bd7c2f7de65`。

也可以使用下面的镜像下载入口。它记录来源、文件大小、逐文件 SHA256 和随机抽样：

```bash
uv run --script script/download_uhdm.py --split train --count 200 --seed 123 \
  --output data/UHDM-paper-calibration/train
uv run --script script/download_uhdm.py --output data/UHDM-full/test
```

测试集必须为完整 500 对；校准集从完整 4,500 对训练清单选择 200 对。Kaggle 镜像
可能需要用户配置访问凭据。本次尚未核对官方压缩包哈希，不能据此声称镜像与其
密码学一致。`results/release_20260930/paper_data_manifest.json` 固定本地校准、
24 对训练留出和测试图的内容哈希，并验证三者不重叠。

## 从原始权重校准

最终部署配方保留每侧至少 0.25% 的原始 smoothed weight 极值；alpha=0.5，
W4 gamma=0.01、W8 gamma=0.05，1×1 池化特征使用完整统计。这些选择来自训练留出集。
FAC 只更新激活上下界，普通权重、稀疏 mask、FP16 尾部及 smoothing 均被冻结。

```bash
uv run python script/reproduce_ptq.py --recipe deploy --bits 4 \
  --small-tensor-policy full_if_pooled --sampling-rate 0.01
uv run python script/reproduce_ptq.py --recipe deploy --bits 8 \
  --small-tensor-policy full_if_pooled --sampling-rate 0.05
```

该命令串联校准、原样导出、格式校验、FP32 fakequant 与 CUDA 质量评估。
`--stage calibrate|export|evaluate` 可分步执行；已有状态不会被默认覆盖。
默认产物位于 `checkpoints/reproduced-ptq/deploy-w4a4-full_if_pooled-g0.01` 和
`checkpoints/reproduced-ptq/deploy-w8a8-full_if_pooled-g0.05`。

把 `--recipe deploy` 改成 `--recipe paper` 可复测每侧 0.5% 的调整 PTQ；原论文
gamma=0.001 的配置也保留。[论文协议](PAPER_PROTOCOL.md)记录跨图统计、采样和
学习率细节的实现假设。量化分数与论文仍有差距，不能把上述命令称作精确数值复现。

## 最终本地模型包

[models.json](../models.json)是当前本地发布模型的唯一索引。HF repository 和 URL 在上传前
保持 `null`，`published=false`。包采用 safetensors，推理不读取原始 pickle checkpoint
或校准状态；它不是 Transformers `AutoModel` 格式。

```python
from quantdemoire.profiles import apply_execution_profile
from quantdemoire.precision import configure
from quantdemoire.deploy import load_model

folder = "checkpoints/hf/esdnet-uhdm-w4a4-ptq-20261003"
apply_execution_profile(folder)
configure()
model = load_model(folder, backend="cuda")
```

输入为 CUDA FP16 RGB tensor，范围 [0,1]，padding 至 32 的倍数；三个输出中的第一个
为最终图像。已有显式 `QUANTDEMOIRE_*` 环境变量优先于模型 profile；为了复测保存的
默认配置，应使用未设置这些覆盖项的新进程。完整格式见 [FORMAT.md](FORMAT.md)。

## 全量质量评估

以下命令使用相同导出权重、整数码和 FP16 稀疏 mask。将模型目录中的 w4a4 换成
w8a8 即可评估 W8；`--fp32` 选择 FP32 fakequant 数值参考。

```bash
uv run python script/evaluate_cuda.py \
  --model checkpoints/hf/esdnet-uhdm-w4a4-ptq-20261003 --runtime-profile \
  --data data/UHDM-full/test --autotune --output results/retest_w4_cuda.json
uv run python script/evaluate_fakequant.py --fp32 \
  --model checkpoints/hf/esdnet-uhdm-w4a4-ptq-20261003 \
  --data data/UHDM-full/test --autotune --output results/retest_w4_fake.json
uv run python script/evaluate.py --backend fp32 \
  --model checkpoints/uhdm_checkpoint.pth --data data/UHDM-full/test \
  --memory-format contiguous --output results/retest_fp32.json
```

不使用 `--crop` 或 `--limit`。指标为有效原图区域的 RGB PSNR、valid-window SSIM 和
AlexNet LPIPS；预测先 clamp 到 [0,1]。记录保留每张输入和 GT 的哈希。
`--resume` 只接受权重、配置、数据和数值环境均匹配的历史结果。

## 速度、显存与分支成本

```bash
uv run python script/benchmark_e2e.py --backend cuda \
  --model checkpoints/hf/esdnet-uhdm-w4a4-ptq-20261003 --runtime-profile \
  --image data/UHDM-full/test/0000_moire.jpg --graph --output results/retest_w4_graph.json
uv run python script/benchmark_memory.py --checkpoint checkpoints/uhdm_checkpoint.pth \
  --w4 checkpoints/hf/esdnet-uhdm-w4a4-ptq-20261003 \
  --w8 checkpoints/hf/esdnet-uhdm-w8a8-ptq-20261003 \
  --runtime-profile --include-fp32 --image data/UHDM-full/test/0000_moire.jpg \
  --graph --output results/retest_memory.json
uv run python script/benchmark_branch.py \
  --model checkpoints/hf/esdnet-uhdm-w4a4-ptq-20261003 \
  --image data/UHDM-full/test/0000_moire.jpg --output results/retest_branch_w4.json
```

省略 `--graph` 复测 eager。`benchmark_variants.py --config config/release/deployment.json`
可在隔离进程中交替测量 FP16/W4/W8；该配置指向最终本地模型包，移动模型后需更新
其中 model 和 joint plan 路径。`benchmark_e2e.py --backend fake_fp16` 可测伪量化速度。

计时包括完整网络、全部三个输出、smoothing、输入量化、packing 和稀疏分支，
排除图片读取、主机到设备传输、padding 和指标。普通 Graph 计时使用常驻输入；
`--changing-input` 另含设备内输入复制。FP16 对照使用相同适用的融合和特征释放。
显存分别报告模型常驻、峰值 allocated/reserved，Graph 与 eager 不混用。

严格分支比例为 `sum(mixed-main_only)/sum(integer_core)`，包含稀疏准备与合并。
它是逐层隔离计时之和，与整网延迟分开报告；当前 30% 目标尚未达到。
当前候选验收结果保存到 `results/release_20261001/`，历史权重的数字不能用于这些包。


## 从导出目录生成模型包

原始导出目录只包含权重、配置与校准来源。先创建带模型卡的本地包，再将该包的
数值验证和测速结果交给 `attach_runtime_profile.py`。下面是可独立使用的打包步骤：

```bash
uv run python script/package_model.py \
  --model checkpoints/reproduced-ptq/deploy-w4a4-full_if_pooled-g0.01 \
  --output checkpoints/hf/my-esdnet-uhdm-w4a4
```

如需保存质量分数，添加 `--quality /path/to/completed-quality.json`，该结果必须与包内
权重及配置哈希一致。目标目录必须为空，避免旧配置残留。原始导出没有最终调度时，
不能直接套用发布模型的速度；应先验证、测量自己的调度。

如果已有同架构的本地参考包，可以用 `script/propose_runtime.py --source <reference>
--model <new-package>` 生成明确标为 `unmeasured_proposal` 的调度。随后在新包上运行
`verify_graph_fusions.py` 和 `benchmark_variants.py`，最后执行：

```bash
uv run python script/attach_runtime_profile.py \
  --model checkpoints/hf/my-esdnet-uhdm-w4a4 \
  --benchmark results/my_e2e.json --variant w4a4 \
  --validation results/my_validation.json
```

最终实测、包位置及未达标项见 [发布结果](RELEASE_RESULTS.md)。模型权重尚未上传，
Git 仓库不包含 checkpoint；`models.json` 中的本地目录不会在 clone 时自动生成。
复现实验的原始 checkpoint、训练/测试数据和按需下载的感知网络权重需单独准备。
