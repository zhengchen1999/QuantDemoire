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

## Reproduction

ESDNet on UHDM, tested with an RTX 4090, Python 3.11, PyTorch 2.2.2+cu121,
CUDA toolkit 12.1 and CUTLASS 3.5.1.

### Environment

```bash
uv sync --frozen
export CUDA_HOME=/path/to/cuda-12.1
uv run --frozen python -m kernel
```

### Quality metrics

Prepare the original ESDNet checkpoint, 200 calibration pairs and the 500-pair
UHDM test set using the [data guide](docs/REPRODUCE.md). Then run:

```bash
bash script/reproduce_metrics.sh
```

This calibrates and exports W4A4/W8A8, verifies CUDA execution, and reports
PSNR, SSIM and LPIPS for FP32, FP16, fakequant and integer CUDA inference.
Models are saved under `checkpoints/reproduction/deploy/`. To evaluate existing
exports again, add `--stage evaluate`. Pretrained weights are not bundled with Git.

### Speedup and GPU memory

```bash
bash script/benchmark_speed.sh
bash script/benchmark_memory.sh
```

Both commands use the reproduced models and report CUDA Graph and eager results.
Speedup and memory savings use an FP16 baseline with the same applicable fusions;
the memory test also includes FP32. Reports are saved under `results/reproduction/`.
Use `--help` for custom model paths, input images and output directories.

[Reference results and limitations](docs/RELEASE_RESULTS.md) ·
[Reproduction details](docs/REPRODUCE.md) · [Calibration protocol](docs/PAPER_PROTOCOL.md)

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
