# CUDA implementation

For the current fused execution profiles, use the matched branch diagnostic:

```bash
uv run python script/benchmark_branch.py --model checkpoints/hf/esdnet-uhdm-w4a4-ptq-20261003 \
  --image data/UHDM-full/test/0000_moire.jpg --output results/branch_w4.json
uv run python script/benchmark_branch.py --model checkpoints/hf/esdnet-uhdm-w8a8-ptq-20261003 \
  --image data/UHDM-full/test/0000_moire.jpg --output results/branch_w8.json
```

Final strict measurements are **57.45% (W4)** and
**39.03% (W8)** of the integer core, including sparse
preparation and merge. **Neither meets 30%.** See [release results](../docs/RELEASE_RESULTS.md).
The older `benchmark_kernel.py` is useful for shape coverage and unfused
ablations; it does not represent the final cross-layer schedule. Historical
optimization sections below refer to their own weights and timing batches.

Isolated candidate modules (`spatial_packed`, `sparse_prefetch`, `sparse_products`,
`sparse_half2`, `planar_int4`, `sparse_subwarp`) are not loaded by the production
backend. Their tests require `pytest research/tests --run-experimental`; ordinary deployment
tests do not compile these extra research libraries. Candidate speed and quality
claims must use their own reports. In particular, `sparse_half2` uses FP16 sparse
accumulation and is not an arithmetic replacement for the FP32 accumulator.

- [INT4 and INT8 Tensor Core convolution](csrc/conv.cu): CUTLASS 3.5.1 implicit
  GEMM with signed operands, INT32 accumulation and FP32 scaling.
- [Quantization and CSR FP16 branch](csrc/epilogue.cu): fused concatenation,
  smoothing and packing; coalesced sparse gathers; FP16 operands and FP32 FMA;
  boundary-aware zero-point correction, bias and optional ReLU.
- [FP16 Tensor Core sparse branch](csrc/sparse_tc.cu): experimental exact
  eight-output-channel grouping, `mma.m16n8k16.f32.f16.f16.f32`.
- [Integer convolution with a fused sparse epilogue](csrc/fused.cu): experimental
  alternative. Correctness alone does not establish a performance benefit.
- [FP16 intermediate and sparse merge](csrc/half_branch.cu): corrected integer
  output stored in FP16, with the sparse contribution accumulated in FP32.

The tested device is an RTX 4090, SM89. The extension targets SM8x and CUDA 12.1;
SM80/86 still need device-specific validation. The build fails explicitly if
the supported backend cannot be loaded. There is no dense floating fallback
for the 157 quantized layers.

```bash
export CUDA_HOME=/path/to/cuda-12.1
bash script/setup_kernel.sh
uv run pytest -q
uv run python script/benchmark_kernel.py --output results/kernel.json
uv run python script/benchmark_kernel.py --model checkpoints/release-w4a4 --bits 4 --output results/kernel_actual4.json
uv run python script/benchmark_kernel.py --model checkpoints/release-w8a8 --bits 8 --output results/kernel_actual8.json
```

The default input of 224x224 traverses all 159 convolution layers and 114 unique
runtime signatures. Actual-model mode tests every layer's deployed weights;
the two unquantized boundary convolutions additionally receive synthetic
integer tests. Use `--height 2160 --width 3840` to test full-resolution runtime
shapes. Dilation, stride, image edges, batch size two, non-default CUDA streams,
tail output channels, saturation and round-to-even are also unit-tested.

Per-layer timing captures 50 operations in each CUDA Graph and divides the
event time by 50, avoiding a Python submission floor. The reported incremental
branch ratio is `(mixed - without_sparse) / integer_core`. The dense comparison
omits planar preparation and uses a dedicated dense output kernel, so branch
preparation is charged to the sparse path. Reports retain negative samples
caused by timing noise and all individual timing samples. A ratio above 0.30
is a failed target; it must not be silently excluded because a layer is small.

Set `QUANTDEMOIRE_SPARSE_IMPL=tensorcore` to benchmark the optional block branch.
The default CSR path and experimental alternatives use identical sparse values
and coordinates. End-to-end benchmarks include all preparation and all three
network outputs, and compare the same eager or CUDA Graph execution mode.

The CSR path stores planar activations only for input channels referenced by
the sparse weights. The derived channel map and remapped coordinates are
runtime buffers and are not part of the portable checkpoint. Set
`QUANTDEMOIRE_COMPACT_SPARSE=0` for an ablation. Both settings produced bitwise
identical complete-network outputs on all 100 local UHDM test images for both
bit widths; see [the equality report](../results/compact_network_equivalence.json).

Activation preparation now defaults to `QUANTDEMOIRE_PREPARE_IMPL=pair`:
precompute `inverse_smooth / activation_step`, load adjacent FP16 channels
together, and place two INT8 values in each shared-memory word before the
planar transpose. Odd channel boundaries fall back to scalar preparation
using the same cached coefficient. `cached` isolates coefficient caching;
`scalar` selects the earlier implementation. Every mode preserves rounding
order and uses the same sparse positions and values. Calibration must call
`PackedConv.refresh_runtime()` after changing activation or smoothing scales.
See [measurements and next steps](../docs/PREPARE_OPTIMIZATION.md).

`QUANTDEMOIRE_FUSE_NEXT_QUANT=1` enables an optional DenseBlock producer
epilogue that prepares the next convolution's new 32-channel prefix. It
preserves FP16 output rounding, per-consumer quantizers, and CSR structure;
the next prepare kernel fills only the remaining channels. It requires the
compact CSR / half-intermediate path with pair or quad preparation and defaults to off. See the
[implementation, measurements and reproduction commands](../docs/NEXT_QUANT_FUSION.md).

`QUANTDEMOIRE_FUSE_ATTENTION=1` combines SAM weighting and its residual into
one kernel, preserving the rounding of each half operation. It also applies
to the FP16 baseline. `QUANTDEMOIRE_FUSE_RESIDUAL=1` adds RDB residuals inside
the quantized CSR output kernel. Both are optional; see the
[complete ablation and equality checks](../docs/POINTWISE_FUSION.md).

`QUANTDEMOIRE_CSR_KERNEL=linear` selects spatially contiguous warp gathers
and cached linear addresses for interior pixels. `lut` and `wide` retain
the slower lookup-table and larger per-thread variants for reproducible
ablation; `legacy` is the prior implementation. Sparse values, coordinates,
per-pixel accumulation order, and output rounding are unchanged. See the
[CSR measurements and shape checks](../docs/CSR_OPTIMIZATION.md).

`QUANTDEMOIRE_JOINT_WARPS=8` selects an experimental cooperative integer/CSR
output kernel, eliminating the globally stored dense intermediate. `4` is a
slower tile configuration retained for ablation; `0` is the separated default.
See [joint output measurements](../docs/JOINT_OUTPUT.md).

`QUANTDEMOIRE_PREPARE_IMPL=quad|quad_loop|quad_swizzle` compares four-channel
preparation, loop unrolling, and shared-memory layout. The verified profiles use
`quad_swizzle`; the unrolled candidate was slower. See the
[vector-width ablation](../docs/PREPARE_VECTOR_WIDTH.md).

`QUANTDEMOIRE_JOINT_VARIANT` retains variants 0 through 9 for reproducible
experiments. Both current profiles use variant 9, which retains variant 8's 32-bit addressing
and FastDivmod and additionally reuses shared A for eligible
single-tile pointwise convolutions and falls back to variant 8 otherwise.
The weight-bound `QUANTDEMOIRE_JOINT_PLAN` chooses joint/separate execution per
layer; the variant must also be set. Earlier asynchronous candidates remain
available for [pipeline ablation](../docs/JOINT_PIPELINE.md).

For eligible shared-input consumers, `QUANTDEMOIRE_SHARED_SKIP_PLANAR=1` omits
unused auxiliary input preparation. With `QUANTDEMOIRE_SHARED_SKIP_PREFIX_PLANAR=1`,
a tuned CSR producer also omits the auxiliary prefix allocation and writes.
These flags preserve quantization codes, original outlier positions, half
rounding, and FP32 sparse accumulation. See the
[shared-input measurements and validation](../docs/SHARED_POINTWISE.md).

Use `config/release/deployment.json` or the model's verified `runtime.json`
for the complete tested combination. Defaults and individual experimental
flags do not by themselves reproduce the recommended end-to-end result.

The current profiles select `QUANTDEMOIRE_CSR_KERNEL=divmod` for separate output
kernels: 32-bit intermediate addresses when safe, plus FastDivmod for spatial
coordinates. `linear32` isolates the address-width change; oversized inputs
fall back to the original 64-bit linear path. See the
[paired timing and full validation](../docs/CSR_ADDRESS.md).

`QUANTDEMOIRE_JOINT_NEXT_QUANT=1|2` lets eligible eight-warp joint producers
prepare the consumer's quantized prefix after output-half rounding. Mode 1 writes
the auxiliary planar prefix directly; mode 2 transposes a small shared-memory
tile before writing it. The recommended profiles use a per-layer
`joint_next_modes` map and disable prefix production on the observed memory-peak
layer. The raw flag defaults to 0. Internal producer specializations 11–14 are
not global `JOINT_VARIANT` options. See the
[full-shape, prefix, image, Graph and timing checks](../docs/JOINT_NEXT_FUSION.md).

`QUANTDEMOIRE_BALANCE_CSR=1` redistributes complete CSR output rows among the
eight warps in variants 8/9, including their next-prefix specializations.
The derived permutation is a nonpersistent runtime buffer; each row's original
indices, FP16 values and FP32 FMA order remain unchanged. The recommended
profiles enable this after full-shape, 100-image and changing-input Graph checks.
The raw flag defaults to 0. See [paired measurements and remaining branch
cost](../docs/CSR_BALANCE.md). Internal specializations 15–20 are not global
`JOINT_VARIANT` options.

`QUANTDEMOIRE_SHARED_RETAIN_STAGES=1` extends variant 9 pointwise shared-A reuse
to two input tiles, Cp <= 512 for W4 or <= 256 for W8, by skipping the final
unused main-loop shared store. Sparse accumulation still starts after the integer
main loop. The W8 profile enables it with auxiliary planar omission; W4 keeps
its previous settings. Internal specializations 21–26 are not global variant
options. The copied CUTLASS main-loop method retains its BSD notice. See
[boundary, Graph, full-image and paired measurements](../docs/RETAINED_POINTWISE.md).

Small spatial convolutions can opt into `QUANTDEMOIRE_FEW_CHANNELS=1` (eight warps,
variant 8/9, R>1, Cp≤64 W4 / Cp≤48 W8). The verified profile enables this only for
W4. Other cases retain the original path. [Validation and timing](../docs/FEW_CHANNELS.md)
include all quantized shapes and the complete-network ablation.

`QUANTDEMOIRE_PACKED_CSR_META=1` combines an interior sparse offset with its FP16
weight bits in the existing nonpersistent address cache. Shared pointwise inputs
retain their prior path; separate CSR uses ordinary offsets. [Evidence and
constraints](../docs/PACKED_CSR_METADATA.md) include prefix composition, cache
switching, all quantized shapes, and complete-image equivalence.

The existing FewChannels path now [addresses packed KRSC filters directly](../docs/PACKED_FILTER_ITERATOR.md),
while retaining its activation iterator, pipeline, output semantics and eligibility.
There is no new serialized layout or runtime flag; W4 enables FewChannels and W8 keeps it disabled.

`QUANTDEMOIRE_FUSE_SECOND_QUANT=1` extends eligible spatial joint producers to
[prepare two consumer-specific quantized feature segments](../docs/TWO_SUCCESSOR_QUANTIZATION.md).
The extra output is compiled separately in [joint_two.cu](csrc/joint_two.cu),
using [shared joint helpers](csrc/joint_common.cuh); [prepare_second.cu](csrc/prepare_second.cu)
consumes the compact second cache. The raw flag defaults to 0; both verified profiles enable it.

The isolated `sparse_tc_coalesced.py` experiment stages coalesced planar loads
into shared memory before FP16 MMA, using zero-padded exact coordinate unions.
It changes accumulation order and is not selected for deployment: 12 independent
FP64-reference/Graph cases pass, but the W4 top-20 screen is 29.398 ms for the
production path versus 34.146 ms for the candidate. This is a synthetic-input
screen, not end-to-end timing. No sparse mask or production kernel is changed.
