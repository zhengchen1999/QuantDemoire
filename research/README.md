# Archived kernel candidates

These isolated candidates did not improve complete hotspot latency and are not
loaded or distributed by the production `kernel` package. The production CSR
path and its matched branch-cost diagnostic remain in `kernel/`.

The implementations, numerical references and screening entry point are retained
here for research. Run from the repository root with the CUDA toolchain set:

```bash
uv run pytest research/tests --run-experimental -q
uv run python -m research.script.benchmark_spatial_packed --help
```

Candidates include packed NHWC gathers, prefetch, product lookup, FP16 half2,
planar INT4, 16-lane CSR, grouped Tensor Core and concurrent sparse execution.
FP16 half2 and Tensor Core grouping change the accumulation arithmetic; passing
their independent references does not establish image quality equivalence.
Screening uses synthetic activations at actual layer shapes and excludes some
consumer caches. It is neither whole-network latency nor the strict branch test.

Historical measurements and rejected choices are indexed in
[the execution record](../docs/EXECUTION_20260930.md). Source hashes in older
records identify their original locations before this directory reorganization.
