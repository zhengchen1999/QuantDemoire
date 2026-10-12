# ESDNet / UHDM release evidence — 2026-10-03

Scope: ESDNet, UHDM, W4A4/W8A8 and floating references. Model packages are local;
the measurements below describe the accepted October 3 weights.
Machine-readable current results are in [summary.json](../results/release_20261003/summary.json).

The simplified reproduction entry points were checked on October 12:
[workflow validation](../results/reproduction_validation/summary.json) records
1,412 passing production tests, unchanged exported weights/configs, a two-image
native-4K quality/Graph smoke, and Graph/eager speed and memory runs. These checks
do not replace the full-test reference measurements below.

## Full test quality

| Model | Tail fraction per side | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|
| FP32 | — | 22.118994 | 0.795592 | 0.255100 |
| FP16 | — | 22.118892 | 0.795589 | 0.255058 |
| W4A4 adjusted PTQ CUDA | 0.5% | 20.973432 | 0.760425 | 0.315438 |
| W8A8 adjusted PTQ CUDA | 0.5% | 21.868181 | 0.792626 | 0.256231 |
| W4A4 deployment CUDA | 0.25% | 21.093776 | 0.762535 | 0.316172 |
| W8A8 deployment CUDA | 0.25% | 21.874563 | 0.792815 | 0.256027 |

Each row covers the same 500 paired images without a crop or limit. Metrics use
RGB [0,1], clamped predictions, valid-window SSIM and AlexNet LPIPS. FP32 and
FP32 fakequant use contiguous layout. The 200 calibration pairs, 24 held-out
training pairs and 500 test pairs have disjoint pair hashes. The test data were
obtained from the documented mirror; the official archive hash was not verified.

| Deployment | FP32 fakequant PSNR / SSIM / LPIPS | CUDA PSNR / SSIM / LPIPS |
|---|---|---|
| W4A4, 0.25% per tail | 21.089903 / 0.762456 / 0.316270 | 21.093776 / 0.762535 / 0.316172 |
| W8A8, 0.25% per tail | 21.874959 / 0.792832 / 0.256069 | 21.874563 / 0.792815 / 0.256027 |

Per-image metrics and input/GT hashes: [W4 CUDA](../results/release_20261001/quality_w4_cuda_500.json),
[W4 fakequant](../results/release_20261001/quality_w4_fake_500.json),
[W8 CUDA](../results/release_20261001/quality_w8_cuda_500.json),
[W8 fakequant](../results/release_20261001/quality_w8_fake_500.json).
The paper-setting comparison is documented in [PAPER_PROTOCOL.md](PAPER_PROTOCOL.md).

## Complete-network latency and memory

| Model | Graph ms | Eager ms | Graph speedup | Graph peak allocated GiB |
|---|---:|---:|---:|---:|
| FP16 | 127.835 | 128.795 | 1.000× | 2.065 |
| W4A4 | 78.084 | 78.813 | 1.637× | 1.340 |
| W8A8 | 93.158 | 94.155 | 1.372× | 1.631 |

Three independent-process rounds alternate variant order, with seven warmup
calls and 30 CUDA-event samples in each round. The table uses the median of
round medians; the JSON retains every sample. The single 2160×3840 RGB image is
padded to 2176×3840. Timing includes all three outputs, smoothing, quantization,
packing and the FP16 sparse branch. Image I/O, host-to-device transfer, padding
and metrics are excluded. Baselines use the same applicable generic fusions and
feature release. TF32 is disabled; FP16 convolution accumulates in FP32.

[Graph samples](../results/release_20261001/e2e_graph.json), [eager samples](../results/release_20261001/e2e_eager.json).
Speed measurements were taken on October 1; quality and remaining acceptance
were completed on October 3 with unchanged weight/config and production-source hashes.

Independent memory measurements include a strict FP32 baseline, model residency,
peak allocated memory and peak reserved memory. These are not compressed model
file sizes; Graph and eager measurements are kept separate.

| Mode | Model | Resident MiB | Peak allocated GiB | Peak reserved GiB |
|---|---|---:|---:|---:|
| graph | FP32 | 23.067 | 4.131 | 9.326 |
| graph | FP16 | 11.758 | 2.065 | 4.818 |
| graph | W4A4 | 5.454 | 1.340 | 3.564 |
| graph | W8A8 | 8.206 | 1.631 | 2.920 |
| eager | FP32 | 23.067 | 4.129 | 9.088 |
| eager | FP16 | 11.758 | 2.066 | 3.771 |
| eager | W4A4 | 5.454 | 1.339 | 3.061 |
| eager | W8A8 | 8.206 | 1.630 | 2.721 |

Sources: [Graph memory](../results/release_20261001/memory_graph.json), [eager memory](../results/release_20261001/memory_eager.json).

## Strict sparse-branch budget

`R_core = sum(mixed - matched_main_only) / sum(integer_core)`.
The numerator includes sparse-specific input preparation, lookup, arithmetic
and merging. Shared work is kept in the matched comparison. Layer sums are
isolated Graph measurements and must not be described as whole-network latency.

| Model | R_core | Layers ≤30% | Aggregate ≤30% |
|---|---:|---:|---|
| W4A4 | 57.45% | 33/157 | Failed |
| W8A8 | 39.03% | 32/157 | Failed |

All layers and samples: [W4](../results/release_20261001/branch_w4.json), [W8](../results/release_20261001/branch_w8.json).

## Correctness, environment and packaging

Both final weights passed 157-layer packing, exact calibration identity and
ranked-tail coverage checks. On all 500 test inputs, fusion/feature-release
comparisons and changing-input Graph replay matched all three outputs bitwise:
[W4 validation](../results/release_20261001/validation_w4.json), [W8 validation](../results/release_20261001/validation_w8.json).
This is equivalence between CUDA execution configurations; fakequant and CUDA
quality are evaluated separately and are not claimed to match bitwise.

The original October 3 software acceptance is recorded in [final software validation](../results/release_20261003/software_validation.json)
for production tests, clean uv installation, wheel/source-distribution checks,
portable loading and CLI smoke tests. Runtime profiles are bound to model,
config and joint-plan checksums. The final model cards include the measurements
and limitations, with hashes for every packaged file. Inference does not require
the original pickle checkpoint or calibration state.

## Known limitations

The strict sparse-branch overhead target of **≤30% is not met**: W4
is **57.45%** and W8 **39.03%**.
The denominator is integer-core time, and the increment includes sparse
preparation and merge. Complete-network speedup does not imply this target passes.
Adjusted 0.5%-tail PTQ PSNR remains below the paper's W4/W8 values of
21.08/22.00 dB. Deployment uses a different tail fraction; its scores are reported
separately and are not evidence of exact paper reproduction.

The evidence applies to the recorded RTX 4090 environment. The source supports
SM8x compilation, but SM80/86 have not received equivalent device validation.
Hugging Face links remain unset until an actual upload. Calibration and result
files retain historical paths as provenance; public commands use relative paths.
Earlier 100-image/QAT/range-search experiments and rejected candidates are
available in Git history at commit `368422d`; they do not describe the current PTQ packages.
