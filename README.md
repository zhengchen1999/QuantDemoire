# QuantDemoire: Quantization with Outlier Aware for Image Demoiréing
[Zheng Chen](https://zhengchen1999.github.io/), [Kewei Zhang](https://github.com/Pushazf), [Xiaoyang Liu](https://xyliu339.github.io/), [Weihang Zhang](), [Mengfan Wang](), [Yifan Fu](), [Yulun Zhang](http://yulunzhang.com/),"QuantDemoire: Quantization with Outlier Aware for Image Demoiréing", arXiv, 2025



<div>
<a href="https://github.com/zhengchen1999/QuantDemoire/releases" target='_blank' style="text-decoration: none;"><img src="https://img.shields.io/github/downloads/zhengchen1999/QuantDemoire/total?color=green&style=flat"></a>
<a href="https://github.com/zhengchen1999/QuantDemoire" target='_blank' style="text-decoration: none;"><img src="https://visitor-badge.laobi.icu/badge?page_id=zhengchen1999/QuantDemoire"></a>
<a href="https://github.com/zhengchen1999/QuantDemoire/stargazers" target='_blank' style="text-decoration: none;"><img src="https://img.shields.io/github/stars/zhengchen1999/QuantDemoire?style=social"></a>
</div>


[[arXiv](https://arxiv.org/abs/2510.04066)] [[supplementary material](https://github.com/zhengchen1999/QuantDemoire/releases/download/v1/Supplementary_Material.pdf)] [[project](https://zheng-chen.cn/QuantDemoire/)] 

#### 🔥🔥🔥 News

- **2025-10-05:** This repo is released.

---

> **Abstract:** Demoiréing aims to remove moiré artifacts that often occur in images. While recent deep learning-based methods have achieved promising results, they typically require substantial computational resources, limiting their deployment on edge devices. Model quantization offers a compelling solution. However, directly applying existing quantization methods to demoiréing models introduces severe performance degradation. The main reasons are distribution outliers and weakened representations in smooth regions. To address these issues, we propose QuantDemoire, a post-training quantization framework tailored to demoiréing. It contains two key components. **First**, we introduce an outlier-aware quantizer to reduce errors from outliers. It uses sampling-based range estimation to reduce activation outliers, and keeps a few extreme weights in FP16 with negligible cost. **Second**, we design a frequency-aware calibration strategy. It emphasizes low- and mid-frequency components during fine-tuning, which mitigates banding artifacts caused by low-bit quantization. Extensive experiments validate that our QuantDemoire achieves large reductions in parameters and computation while maintaining quality. Meanwhile, it outperforms existing quantization methods by over **4 dB** on W4A4.

<img src="figs/compare.png" width="100%"/>

## ⚒️ TODO

* [x] Add ESDNet/UHDM calibration, portable weights, and CUDA experiment code
* [ ] Publish validated pretrained models and Hugging Face download links
* [ ] Test our quantization method on more models

## ESDNet / UHDM implementation

Packed W4A4 INT4 and W8A8 INT8 Tensor Core convolution cover 157 quantized layers.
The FP16 outlier branch accumulates in FP32, and the two boundary convolutions
remain floating point. Fakequant and CUDA load the same integer codes, sparse
indices and FP16 tail values. Smoothing, quantization and packing are included
in complete-network inference.

### Validated results

All quality results below use the full 500-pair UHDM test split. Adjusted PTQ
uses gamma 0.01 (W4) / 0.05 (W8), alpha=0.5 and full statistics for pooled 1×1
features. Parameters were selected on held-out training images.

| Model | Tail fraction per side | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|
| FP32 | — | 22.118994 | 0.795592 | 0.255100 |
| FP16 | — | 22.118892 | 0.795589 | 0.255058 |
| W4A4 adjusted PTQ CUDA | 0.5% | 20.973432 | 0.760425 | 0.315438 |
| W8A8 adjusted PTQ CUDA | 0.5% | 21.868181 | 0.792626 | 0.256231 |
| W4A4 deployment CUDA | 0.25% | 21.093776 | 0.762535 | 0.316172 |
| W8A8 deployment CUDA | 0.25% | 21.874563 | 0.792815 | 0.256027 |

Final 0.25%-tail deployment packages, RTX 4090, one native 4K image padded to
2176×3840, three alternating-process rounds with 30 samples per round:

| Model | Graph ms | Eager ms | Graph speedup | Graph peak allocated GiB |
|---|---:|---:|---:|---:|
| FP16 | 127.835 | 128.795 | 1.000× | 2.065 |
| W4A4 | 78.084 | 78.813 | 1.637× | 1.340 |
| W8A8 | 93.158 | 94.155 | 1.372× | 1.631 |

FP16 uses the same applicable fusions and feature release. Graph peak allocated
memory savings are 35.13% (W4) / 21.06% (W8).
Timing excludes image I/O, host-to-device transfer, padding and quality metrics.
These measurements are specific to the recorded hardware, input and profiles.

The strict sparse-branch overhead target of **≤30% is not met**: W4
is **57.45%** and W8 **39.03%**.
The denominator is integer-core time, and the increment includes sparse
preparation and merge. Complete-network speedup does not imply this target passes.
Adjusted 0.5%-tail PTQ PSNR remains below the paper's W4/W8 values of
21.08/22.00 dB. Deployment uses a different tail fraction; its scores are reported
separately and are not evidence of exact paper reproduction.

See [release evidence and validation](docs/RELEASE_RESULTS.md),
[calibration assumptions](docs/PAPER_PROTOCOL.md) and [documentation index](docs/README.md).

### Install and reproduce

Validated with Python 3.11, PyTorch 2.2.2+cu121, CUDA toolkit 12.1, CUTLASS 3.5.1,
and an RTX 4090. Other SM8x devices require independent validation.

```bash
uv sync --frozen
export CUDA_HOME=/path/to/cuda-12.1
bash script/setup_kernel.sh
uv run pytest -q
```

Prepare the raw checkpoint and data as described in [REPRODUCE.md](docs/REPRODUCE.md).
Run calibration, unchanged-mask export, format checks and full fakequant/CUDA
evaluation from the raw checkpoint:

```bash
uv run python script/reproduce_ptq.py --recipe deploy --bits 4 \
  --small-tensor-policy full_if_pooled --sampling-rate 0.01
uv run python script/reproduce_ptq.py --recipe deploy --bits 8 \
  --small-tensor-policy full_if_pooled --sampling-rate 0.05
```

Use `--recipe paper` for the separate 0.5%-tail adjusted PTQ comparison.
The original gamma=0.001 recipe remains available as an explicitly different
experiment. Ordinary network weights are frozen during this PTQ process.

[models.json](models.json) identifies the validated local safetensors packages
and their exact reports. **Weights are not included in Git and have not been
uploaded to Hugging Face**; download links are null. Cloning this repository
does not create those model folders. The [guide](docs/REPRODUCE.md) describes
reproduction and local packaging. Once a local package is available:

```bash
uv run python script/infer.py \
  --model checkpoints/hf/esdnet-uhdm-w4a4-ptq-20261003 --runtime-profile \
  --input /path/to/moire.jpg --output outputs/restored.png --graph
```

| Entry point in `script/` | Purpose |
|---|---|
| `evaluate_fakequant.py`, `evaluate_cuda.py` | Full PSNR, SSIM and LPIPS evaluation |
| `benchmark_e2e.py`, `benchmark_variants.py` | Eager / CUDA Graph network latency |
| `benchmark_memory.py` | Resident, peak allocated and reserved memory |
| `benchmark_branch.py` | Strict branch cost with a matched main-only comparison |
| `export.py`, `verify_release.py`, `package_model.py` | Packed weights, calibration identity and local packages |

[kernel/](kernel/README.md) contains the production CUDA sources and build logic.
[research/](research/README.md) preserves rejected candidates outside the runtime
wheel and default test suite. [Third-party notices](docs/THIRD_PARTY_NOTICES.md)
document external code and dependencies.

## 🔎 Method Overview

<img src="figs/method.png" width="100%"/>

## <a name="results"></a>🔎 Results

<details open>
<summary>Quantitative Results (click to expand)</summary>

- Results in Tab. 4 of the main paper

<p align="center">
  <img width="900" src="figs/quantitative.png">
</p>

- Results (Compression Ratio) in Tab. 3 of the supplementary material

<p align="center">
  <img width="900" src="figs/size.png">
</p>

</details>

<details open>
<summary>Qualitative Results (click to expand)</summary>

- Results in Fig. 6 of the main paper

<p align="center">
  <img width="900" src="figs/visual.png">
</p>
<details>
<summary>More Qualitative Results</summary>

- More results in Fig. 1 of the supplementary material

<p align="center">
  <img width="900" src="figs/visual_supp1.png">
</p>

- More results in Fig. 2 of the supplementary material

<p align="center">
  <img width="900" src="figs/visual_supp2.png">
</p>

</details>

</details>

## <a name="citation"></a>📎 Citation

If you find the code helpful in your research or work, please cite our work.

```
@article{chen2025quantdemoire,
    title={QuantDemoire: Quantization with Outlier Aware for Image Demoiréing},
    author={Chen, Zheng and Zhang, Kewei and Liu, Xiaoyang and Zhang, Weihang and Wang, Mengfan and Fu, Yifan and Zhang, Yulun},
    journal={arXiv preprint arXiv:2510.04066},
    year={2025}
}
```
