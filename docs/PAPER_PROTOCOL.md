# ESDNet / UHDM calibration protocol

The base configurations are `config/ptq/{paper,deploy}_w{4,8}a{4,8}.yaml`.
The [arXiv v1 paper](https://arxiv.org/html/2510.04066v1) and the
[ESDNet loss implementation](https://github.com/CVMI-Lab/UHDM/blob/main/utils/loss_util.py)
provide the recipe references.

| Setting | Reproduction |
|---|---|
| Calibration | 200 pairs sampled from 4,500 training pairs, seed 123 |
| Input | 512×512 random crops, batch size 1 |
| Smoothing | alpha=0.5; mean of per-image sampled statistics |
| Weight quantization | symmetric per output channel, min/max excluding FP16 tails |
| Activation quantization | asymmetric per tensor, sampled min/max |
| FP16 tails per side | paper: 0.5%; deployment: 0.25%; stable rank selection |
| FAC | activation bounds only; weights, smoothing and sparse mask/values frozen |
| Optimizer | Adam (0.9, 0.999), four epochs, learning rate 0.001 |
| Scheduler | cosine warm restarts each epoch, minimum learning rate 1e-6 |
| Frequency filters | three binomial stages, dilation 2, 4, 8 |
| Loss | filtered L1 + VGG16 features[15] L1, three output scales |
| Precision | TF32 disabled; FP32 fakequant reported separately from CUDA |

## Adjusted settings

The checked-in base YAML files retain gamma=0.001. The public metrics script
reproduces the adjusted results using gamma=0.01 (W4) / 0.05 (W8), with complete
statistics for pooled 1×1 tensors. These settings were selected on 24 held-out
training pairs before the 500-pair test evaluation. The original calibration
implementation and exact calibration subset were unavailable.

The paper does not fully specify cross-image statistics, the restart period,
minimum learning rate, pooled-feature handling or sampling seed. These are
explicit implementation assumptions. The public `--recipe paper` comparison
also uses the adjusted gamma and pooled-feature rule; its name denotes the
0.5%-tail setting and does not claim identical paper parameters.

```bash
bash script/reproduce_metrics.sh                 # 0.25%-tail deployment
bash script/reproduce_metrics.sh --recipe paper  # adjusted 0.5%-tail comparison
```

Calibration records the checkpoint, configuration, input and source hashes and
checks frozen parameters after every epoch. Export preserves the calibrated mask,
integer codes, FP16 tails, activation parameters and smoothing values. Validation
of these identities is part of the metrics workflow. CUDA/fakequant quality is
measured separately; the two arithmetic paths are not claimed to be bitwise equal.

## Reference outcomes

Full-test FP32 is 22.118994 PSNR / 0.795592 SSIM / 0.255100 LPIPS, matching the
paper's reported precision. Adjusted 0.5%-tail CUDA PSNR is 20.973432 (W4) /
21.868181 (W8), below the paper's 21.08 / 22.00. The separate 0.25%-tail deployment
results are 21.093776 / 21.874563 and do not establish exact paper reproduction.
[Release results](RELEASE_RESULTS.md) contain all metrics and per-image evidence.
