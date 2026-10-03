# Deployment experiment index

> 历史优化实验索引。当前模型、全量质量与性能见 [最终发布结果](RELEASE_RESULTS.md)。


Historical experiments retain their original weights, protocols, and measurement scope.
Use [the current execution record](EXECUTION_20260930.md) for the latest status.

## ESDNet / UHDM deployment experiments

The local implementation includes packed W4A4 INT4 and W8A8 INT8 CUDA
convolution, an FP16 outlier branch with FP32 accumulation, matching sparse
fakequant structure, and a pinned `uv` environment. Floating ESDNet deployment
uses FP16 operands with FP32 convolution accumulation. The first and final
convolutions remain floating point.

See [reproduction commands](REPRODUCE.md), [CUDA kernels and shape
coverage](../kernel/README.md), and the [portable weight format](FORMAT.md).
Measured quality, latency, memory, and the remaining performance targets are
documented in the [local experiment report](EXPERIMENTS.md).
The [2026-09-30 progress audit](AUDIT_20260930.md) checks the saved artifacts,
reruns tests and end-to-end measurements, and identifies the remaining release requirements.
The [execution plan](EXECUTION_PLAN.md) defines the remaining implementation,
paper reproduction, benchmarking, and packaging work and their acceptance criteria.
The [execution log](EXECUTION_20260930.md) records clean builds, the complete
UHDM test split, and matched branch-cost measurements. The [paper protocol](PAPER_PROTOCOL.md)
includes the reproduced 500-image FP32 result and the separate PTQ configurations.
An optional [producer-output / next-layer quantization fusion](NEXT_QUANT_FUSION.md)
includes measured latency, peak memory, and reproducible ablations.
Further [attention and residual fusion experiments](POINTWISE_FUSION.md)
include the corresponding FP16 baseline and bitwise output validation.
The [CSR address and access optimization](CSR_OPTIMIZATION.md) records the
subsequent sparse-kernel ablation and full-resolution shape coverage.
The [joint integer/sparse output experiment](JOINT_OUTPUT.md) compares
cooperative CUTLASS epilogues, their latency, and reduced intermediate memory.
The [input vectorization experiments](PREPARE_VECTOR_WIDTH.md) retain both
failed candidates and the verified shared-memory swizzle. The currently selected
W4/W8 and matching FP16 benchmark settings are in `config/experiments/optimized.json`.
Further [joint-kernel pipeline experiments](JOINT_PIPELINE.md) and
[dispatch including cross-layer quantization costs](JOINT_DISPATCH.md) record
their separate ablations and adoption checks.
The [joint-kernel addressing ablation](JOINT_ADDRESS.md) and
[shared-input reuse experiment](SHARED_POINTWISE.md) include full-shape checks,
complete-image validation, and the remaining branch-overhead gap.
The current profiles also use the verified [separate CSR address optimization](CSR_ADDRESS.md)
and [joint-output / next-layer quantization fusion](JOINT_NEXT_FUSION.md),
with per-layer scheduling and a measured peak-memory adjustment.
A verified [CSR warp workload assignment](CSR_BALANCE.md) further reduces
the joint sparse stage without changing CSR row order or model weights.
The W8 profile also [retains two pointwise input stages](RETAINED_POINTWISE.md),
expanding shared-input reuse and reducing auxiliary memory.
The W4 profile additionally uses [flattened reduction for selected small-channel
spatial convolutions](FEW_CHANNELS.md), with full-image equivalence and a paired timing ablation.
Both profiles also [combine CSR addresses and FP16 values in the runtime cache](PACKED_CSR_METADATA.md),
while preserving the serialized weights and sparse accumulation order.
W4 also [uses contiguous packed-weight addressing in its FewChannels iterator](PACKED_FILTER_ITERATOR.md),
with unchanged layouts, full-image equivalence and a paired integrated-backend comparison.
The verified profiles also [quantize eligible features for two successive consumers](TWO_SUCCESSOR_QUANTIZATION.md),
preserving each consumer's own quantizer and including cache costs in whole-network timing.
The W4 spatial CSR kernels use a [warp-private operand lookup](SHARED_W4_LOOKUP.md),
validated with exact outputs and paired end-to-end measurements.
The selected profiles also [omit FP16 features fully covered by consumer-specific caches](FEATURE_OMISSION.md),
with exact image outputs, guarded cache ownership and measured peak-memory reduction.
The [logical C80 experiment](LOGICAL80_EXPERIMENT.md) records a numerically
verified specialization that was not selected because its end-to-end gain was too small and inconsistent.
`script/` contains quality, end-to-end latency, kernel overhead, memory, export,
and local model-packaging entry points. Actual measured deployment results and
unmet targets are recorded separately from the paper's tables. No Hugging Face
upload or remote push is performed by these scripts.
