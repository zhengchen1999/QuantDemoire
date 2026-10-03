# ESDNet/UHDM arXiv v1 复现配方

依据 [QuantDemoire arXiv v1](https://arxiv.org/html/2510.04066v1)，范围仅为 ESDNet/UHDM。
配置为 `config/ptq/{paper,deploy}_w{4,8}a{4,8}.yaml`（实际文件名例如 `paper_w4a4.yaml`）。
旧 `config/w4a4.yaml`、`config/w8a8.yaml` 保留作为历史配置，不能用于声称严格论文配方。

## 已确认的 FP32 主结果

`results/release_20260930/quality_fp32_500_nchw.json` 包含完整 500 对逐图指标和哈希：
PSNR **22.11899375**、SSIM **0.7955921984**、LPIPS **0.2550998695**。
按论文小数位四舍五入后为 **22.12 / 0.7956 / 0.2551**，与主表一致。
权重为原始 `uhdm_checkpoint.pth`，未启用 TF32；FP32 卷积采用 NCHW，feature release 开启。
前三张与 NHWC 指标差异最大约 0.000002 dB；NCHW 的约 0.38 秒/图只用于缩短质量评估，
不作为 INT4/INT8 对比所用的 FP16 速度基线。

## 固定设置

| 项目 | 新配方 |
|---|---|
| 校准数据 | 从完整 4,500 对训练清单中用 Python Random(123) 抽取 200 对，保存路径和双图哈希 |
| 两次统计与 FAC | 512×512 随机裁剪，batch size 1 |
| 采样 | 通道幅值和激活边界均按 0.001 比例；采样后的激活使用 min/max |
| Smoothing | alpha=0.5，按已有实现汇总每张图的采样统计均值 |
| 权重 | 对称、逐输出通道，排除尾部后 min/max；激活为非对称、逐 tensor |
| FP16 尾部 | 校准前稳定排序选择每侧 ceil(N×beta) 项，保存为 FP16；FAC 中 mask 与值固定 |
| paper / deploy | beta 分别为 0.005 / 0.0025；部署变体单独报告 |
| 优化 | 只更新激活 lower/upper bound；Adam (0.9,0.999)，4 epochs，初始 lr=0.001 |
| 调度 | CosineAnnealingWarmRestarts，每 epoch 一个周期，T_mult=1，eta_min=1e-6 |
| 频率提取 | 论文 3×3 binomial kernel，三个阶段，dilation=2,4,8 |
| 损失 | 滤波结果上的 L1 + VGG16 L1；采用 ESDNet 的多尺度监督 |
| 浮点策略 | TF32 关闭，PyTorch 2.2.2+cu121；fakequant 和导出状态保持可追溯 |

原有实现有几处偏差：激活采用采样后的额外 percentile 截断、普通权重二次 percentile
截断、采样数量固定、无学习率调度、频率 dilation=1,2,4。新配方显式设置相应参数，
旧 loss 的默认参数保留，历史产物不会因此被冒称为新配方结果。

## 明示的实现假设

- 论文没有给出原始 200 对文件列表、随机种子和完整校准状态，因此新选择独立固定为
  seed 123；具体像素和裁剪随机状态均由本地运行记录确定。
- 论文未说明循环周期和最低学习率；此处采用一 epoch 周期及 1e-6，不能声称作者原值如此。
- 式 (11) 的 i 从 1 开始、dilation=2^i，因此采用 2,4,8。
- 论文引用的 [ESDNet loss](https://github.com/CVMI-Lab/UHDM/blob/main/utils/loss_util.py)
  默认选择 VGG 第三段末端（features[15]），先归一化再缩放到 224×224、不 clamp；
  新配方照此实现。原先 [3,8,15] 且 clamp 的配置保留为历史选项。
- 第一/最后卷积保留浮点，沿用已有实现。统计跨图取均值、随机采样有放回和三尺度
  loss 汇总亦沿用已有代码；正文没有完整说明这些实现细节。
- 稳定 rank 选择保证并列值时两侧数量仍达标；无并列值时对应原论文 percentile 尾部。
  导出不得再次改变 mask，也不得在测试集上选择校准参数。

## 命令与证据

```bash
uv run --script script/download_uhdm.py --split train --count 200 --seed 123 \
  --output data/UHDM-paper-calibration/train
uv run --script script/download_uhdm.py --output data/UHDM-full/test
uv run python calibrate.py --config config/ptq/paper_w4a4.yaml --calibrate-only
uv run python calibrate.py --config config/ptq/paper_w8a8.yaml --calibrate-only
uv run python calibrate.py --config config/ptq/deploy_w4a4.yaml --calibrate-only
uv run python calibrate.py --config config/ptq/deploy_w8a8.yaml --calibrate-only
```

完整的 raw-checkpoint → FAC → 原样导出 → 格式校验 → fakequant/CUDA 评估入口：

```bash
uv run python script/reproduce_ptq.py --recipe paper --bits 4 8
uv run python script/reproduce_ptq.py --recipe deploy --bits 4 8
```

可用 `--stage calibrate|export|evaluate` 分步执行；已有校准状态不会被默认覆盖。
该入口不使用导出时的 `--tail-fraction` 或范围搜索，保证 FAC 和导出保留相同 mask。
默认 fake oracle 使用 FP32 特征及 FP16 主分支中间结果舍入；CUDA 使用 FP16 特征，
两个结果分别记录，不能冒称逐位相同。历史 `script/reproduce.py` 包含额外权重 QAT，
仍是独立的部署调优流程。

### 小 tensor 采样与经授权的 gamma 调整

首轮 W4 发现 1×1 池化特征按 γ=0.001 只采到一个值；逐图 min/max 后取均值会形成
近零宽度区间。`ACT_SMALL_TENSOR_POLICY=legacy` 保留此行为；独立候选
`full_if_single` 仅在 tensor 多于一个值而采样数为一时使用完整小 tensor。
它不是已确认的原论文规则，不能据此声称严格恢复了作者实现。大 tensor 的采样和
随机数消耗不变。对照配置 `paper_w4a4_guard.yaml` 和原始首轮状态均保留。

```bash
uv run python script/reproduce_ptq.py --recipe paper --bits 4 8 \
  --small-tensor-policy full_if_single
uv run python script/evaluate.py --backend cuda --model /path/to/model \
  --data /path/to/training_pool --manifest /path/to/data_manifest.json \
  --subset heldout_training --crop 512 --output results/holdout.json
```

留出集入口按 manifest 的顺序和双图哈希检查样本，不把测试图用于选择该规则。

2026-10-01 用户确认没有原始校准实现，并允许适当提高 gamma。增加 0.01 / 0.05
对照（仍保留 0.001 基线），使用 `--sampling-rate` 同时设置两个 gamma。改变采样率
的结果标为 adjusted recipe；先比较固定训练留出集，再进行完整测试集评估。

第一轮增大 gamma 时发现 `full_if_single` 的保护会在采样数变成 2–3 时停止。
新增 `full_if_pooled`，对 1×1 池化特征始终使用完整 tensor，空间特征仍按 gamma
采样。后续 gamma 对照采用此固定规则；旧 `full_if_single` 结果保留，不能混称为同一配方。

```bash
uv run python script/reproduce_ptq.py --stage calibrate --recipe paper --bits 4 \
  --small-tensor-policy full_if_pooled --sampling-rate 0.01
```

校准保存输入/原始 checkpoint/配置/源码哈希、每 epoch 的 quant_state 和训练记录。
每 epoch 校验冻结参数、量化权重及尾部 mask/值未改变。epoch 快照目前用于恢复与
审计模型状态，不宣称已实现优化器/RNG 的精确续训。

### 完整测试前冻结的选择

在相同的 `full_if_pooled` 规则下比较两组 gamma，训练留出 PSNR：

| 位宽 | γ=.01 | γ=.05 | 选择 |
|---|---:|---:|---|
| W4A4 | 21.1676 | 20.8362 | .01 |
| W8A8 | 21.8294 | 21.9682 | .05 |

选择依据是 24 对互斥训练留出图的 PSNR；FP32 同子集为 22.3031 dB。
`results/release_20260930/paper_adjusted_selection.json` 在完整测试评估前写入，
保存两个候选及最终权重/配置哈希。alpha=.5、两侧各 .5%、仅激活边界 FAC 保持不变。
原始 gamma .001、小样本保护和提高 gamma 的结果独立记录，不能声称逐参数严格复现。

```bash
uv run python script/reproduce_ptq.py --recipe paper --bits 4 \
  --small-tensor-policy full_if_pooled --sampling-rate 0.01
uv run python script/reproduce_ptq.py --recipe paper --bits 8 \
  --small-tensor-policy full_if_pooled --sampling-rate 0.05
```

量化质量仍需完整评估；本页的 FP32 成功和训练留出结果不代表 W4/W8 已复现。
