# QuantDemoire: Quantization with Outlier Aware for Image Demoiréing
[Zheng Chen](https://zhengchen1999.github.io/), [Kewei Zhang](https://github.com/Pushazf), [Xiaoyang Liu](https://xyliu339.github.io/), [Weihang Zhang](), [Mengfan Wang](), [Yifan Fu](), [Yulun Zhang](http://yulunzhang.com/),"QuantDemoire: Quantization with Outlier Aware for Image Demoiréing", arXiv, 2025

[[project page]()] [[paper]()] [[supplementary material]()]

#### 🔥🔥🔥 News

- **2025-09-30:** This repo is released.

---

> **Abstract:** Demoiréing aims to remove moiré artifacts that often occur in images. While recent deep learning–based methods have achieved promising results, they typically require substantial computational resources, limiting their deployment on edge devices. Model quantization offers a compelling solution with its advantages of compactness and efficiency. However, directly applying existing quantization methods to demoiréing models introduces severe performance degradation. The main reasons are distribution outliers and weakened representations in smooth regions.
To address these issues, we propose QuantDemoire, a post-training quantization framework tailored to demoiréing. It contains two key components. <b>First</b>, we introduce an outlier-aware quantizer to reduce errors from outliers. It uses sampling-based range estimation to reduce activation outliers, and keeps a few extreme weights in FP16 with negligible cost. <b>Second</b>, we design a frequency-aware calibration strategy. It emphasizes low- and mid-frequency components during fine-tuning, which mitigates banding artifacts caused by low-bit quantization. Extensive experiments validate that our QuantDemoire achieves large reductions in parameters and computation while maintaining quality. Meanwhile, it outperforms existing quantization methods by over <b>6.24 dB</b> on W4A4. Code will be released.

<img src="figs/Performance.png" width="100%"/>

## ⚒️ TODO

* [ ] Release code and pretrained models
* [ ] Test our quantization method on more models

## 🔎 Method Overview

<img src="figs/Overview.png" width="100%"/>

## <a name="results"></a>🔎 Results

<details open>
<summary>&ensp;Quantitative Comparisons (click to expand) </summary>
<li> Quantitative results on UHDM, FHDMi and LCDMoiré dataset. 
<p align="center">
<img src="figs/comparison.png" >
</p>
</li>
</details>
<details close>
<summary>&ensp;Visual Comparisons (click to expand) </summary>
<p align="center">
<img src="figs/visual.png">
<img src="figs/visual_supp1.png">
<img src="figs/visual_supp2.png">
</p>
</details>


## <a name="citation"></a>📎 Citation

If you find the code helpful in your research or work, please cite our work.

```
```
