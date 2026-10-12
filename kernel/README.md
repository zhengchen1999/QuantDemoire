# Integer CUDA backend

The production backend implements packed W4A4 INT4 and W8A8 INT8 Tensor Core
convolutions for ESDNet, with an FP16 sparse outlier branch accumulating in FP32.
Integer main paths accumulate in INT32. CUDA source files live in `csrc/`.

## Build

```bash
export CUDA_HOME=/path/to/cuda-12.1
uv run --frozen python -m kernel
```

The command checks the CUDA toolkit, prepares pinned CUTLASS 3.5.1 and JIT-compiles
the extension. `CUTLASS_ROOT` can select an existing checkout at commit
`f7b19de32c5d1f3cedfc735c2849f12b537522ee`. Keep CUDA_HOME and any custom
CUTLASS_ROOT set during reproduction. RTX 4090 / SM89 is validated; other SM8x
hardware requires independent validation.

The reproduction workflow applies per-model runtime settings for smoothing,
input quantization, packing, sparse computation, fusion and feature release.
No dense floating fallback replaces the integer computation.

Use the three [reproduction scripts](../README.md#reproduction) for quality,
full-network speedup and memory.

The Python `diagnostic.py` and main-only diagnostic helpers support numerical
tests; they do not define a quality-valid inference model.
