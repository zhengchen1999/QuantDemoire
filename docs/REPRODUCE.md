# ESDNet / UHDM reproduction

The public entry points are `script/reproduce_metrics.sh`,
`script/benchmark_speed.sh` and `script/benchmark_memory.sh`. Their Python
implementation lives in `script/_internal/`.

## Environment

```bash
uv sync --frozen
export CUDA_HOME=/path/to/cuda-12.1
uv run --frozen python -m kernel
```

The build command downloads pinned CUTLASS 3.5.1 and compiles the CUDA extension.
To reuse an existing checkout, export `CUTLASS_ROOT` before building and keep it
set for subsequent commands. Validation hardware is RTX 4090 / SM89; SM80/86
require independent device validation. CUDA_HOME must contain `bin/nvcc`.

## Data and checkpoint

Obtain the ESDNet UHDM checkpoint from the [official UHDM repository](https://github.com/CVMI-Lab/UHDM)
and save it as `checkpoints/uhdm_checkpoint.pth`. Reference SHA256:
`254235cd25f90a3f1785885385dc6cb3f2178e053291ab53d1943bd7c2f7de65`.

Use 200 calibration pairs and all 500 test pairs. The optional mirror downloader
selects calibration pairs with seed 123 from the complete 4,500-pair training list:

```bash
uv run --script script/_internal/download_uhdm.py --split train --count 200 --seed 123 \
  --output data/UHDM-paper-calibration/train
uv run --script script/_internal/download_uhdm.py --output data/UHDM-full/test
```

Each pair has matching `*_moire.jpg` and `*_gt.jpg` filenames. The downloader
records file sizes and hashes; Kaggle access may require credentials. The official
archive hash was not verified. Data and checkpoints are not included in Git.

## Quality metrics

```bash
bash script/reproduce_metrics.sh
```

This performs calibration, export, integer/mask checks, fusion/Graph validation,
and full FP32, FP16, W4/W8 fakequant and CUDA evaluation. The default adjusted
deployment recipe keeps each 0.25% weight tail in FP16, uses alpha=0.5,
gamma=0.01 (W4) / 0.05 (W8), and full statistics for pooled 1×1 features.
Ordinary network weights remain frozen during calibration.

- Calibration states: `outputs/reproduction/deploy/`.
- Exported weights and runtime settings: `checkpoints/reproduction/deploy/w4a4/` and `w8a8/`.
- Per-image results, validation and `metrics.json`: `results/reproduction/deploy/`.

Use `--stage calibrate`, `--stage export` or `--stage evaluate` to run a single
stage. Existing calibration states are protected from overwrite. Evaluation
resumes only matching reports. `--models`, `--artifacts`, `--output`, `--checkpoint`,
`--calibration` and `--data` override paths. `--bits 4` selects W4 only.
Paths supplied to the shell scripts are relative to the repository root.

`--recipe paper` runs the separate adjusted 0.5%-tail comparison and writes to
parallel `paper/` directories. It is not exact numerical reproduction of the
published quantized scores; see [the calibration protocol](PAPER_PROTOCOL.md).
Without `--limit`, evaluation requires all 500 pairs. `--limit N` is a development
check and its report is explicitly marked as a subset.

## Speedup and memory

```bash
bash script/benchmark_speed.sh
bash script/benchmark_memory.sh
```

Both commands consume the default deployment exports. For existing local packages:

```bash
bash script/benchmark_speed.sh --w4 /path/to/w4a4 --w8 /path/to/w8a8
bash script/benchmark_memory.sh --w4 /path/to/w4a4 --w8 /path/to/w8a8
```

Each model must contain a checksum-bound `runtime.json` and `joint_plan.json`.
The quality reproduction command creates these automatically. New exports receive
execution settings without inheriting old measurements or a validated status.
Their new quality, equivalence, speed and memory reports define their results.

The default image is `data/UHDM-full/test/0000_moire.jpg`; set `--image` to change it.
Both Graph and eager are measured; `--mode graph` or `--mode eager` selects one.
Speed uses three alternating-process rounds, seven warmups and 30 samples per round.
`speed.json` reports the median of round medians and speedup against FP16.
`memory.json` reports resident, peak allocated/reserved bytes and savings against
FP16/FP32. Output directories are `results/reproduction/speed/` and `memory/`.

Timing includes the complete network and all three outputs, smoothing, packing,
input quantization and the sparse branch. Image I/O, transfers, padding and metrics
are excluded. FP16 uses the same applicable fusion and feature release settings.
The scripts isolate QuantDemoire environment overrides for comparable measurements.

[Reference results](RELEASE_RESULTS.md) contain the historical full-test measurements
and known limitations. `models.json` indexes the validated local reference packages;
they are not automatically created by cloning the repository.
