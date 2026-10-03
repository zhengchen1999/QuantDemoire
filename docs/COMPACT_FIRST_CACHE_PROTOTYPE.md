# Compact first-consumer cache prototype

This experiment is isolated under `outputs/prototypes`; it has not replaced the
selected runtime. Weights, consumer quantizers, both 0.25% weight tails, CSR order,
FP16 rounding, and FP32 sparse accumulation are unchanged.

The existing first-prefix producer allocates the next consumer's entire packed
input and, when needed, its planar sparse input, although it only writes its own
32 output channels. The compact prototype allocates just those codes and delays
complete packed/planar allocation until consumer preparation. It emits no first
planar prefix. Preparation copies and decodes the compact first cache, optionally
the second and third caches, and quantizes all remaining real FP16 segments.

`CompactFirstCache` explicitly distinguishes a complete compact cache from a
partially initialized full input. Consumer identity, channel count, bit width,
complete cache sizes, and metadata coverage are checked. Nested concatenations
do not inherit caches, and a consumed metadata feature cannot be read again.
Only currently eligible omitted spatial joint producers are compacted; separate
CSR producers and all other paths remain unchanged.

Validation before timing:

- 1,152 producer cases and 216 preparation cases: exact first/second/third
  consumer-specific codes, different same-bit quantizers, FP16 subnormal scales,
  borders, odd/tiny geometry, independent streams, and changing-input Graphs.
- 32 DenseBlock cases: original output equality, mixed compact/full caches,
  producer allowlists, residuals, three metadata features, cache consumption.
- Four configurations (W4/W8, with/without third cache), three local UHDM images
  each: all three outputs and Graph replay match exactly. Actual compact
  producer counts are 26/43 for W4 and 18/29 for W8.
- Uncached ComputeSanitizer: 264 native plus 32 pipeline cases, zero errors.

The all-eligible exploratory screen uses two alternating fresh-process rounds.
Positive savings mean faster inference; memory changes are relative to that
round's unchanged selected model.

| Mode | Bits | Mean paired latency saving, ms | Peak saved, MiB |
|---|---:|---:|---:|
| Compact first cache only | 4 | -0.06920 | 38.21484 |
| Compact + block-selected third cache | 4 | 0.03963 | 38.21484 |
| Compact first cache only | 8 | -0.76848 | 0.25 |
| Compact + block-selected third cache | 8 | -0.69324 | 126.50293 |

Global enablement adds copying where it does not lower the network peak. A
separate allocator profile locates the peak in
`decoder.decoder_1.sam_blocks.0.basic_block`. Instrumentation records peak bytes
inside each quantized convolution and between convolutions; these hook runs are
not latency benchmarks. A narrower candidate changes only:

- W4: compact the first cache of `conv_layers.3.conv.0`, with the original plan.
- W8: restore the first prefix of `conv_layers.3.conv.0`, enable the third cache
  from `conv_layers.2.conv.0`, and compact these two producers only.

Both narrowed candidates pass another three images and Graph replay per bit,
with the expected one/two compact producers and zero/one third-cache producers.
Four alternating fresh-process pairs give:

| Bits | Paired savings, ms | Mean saving, ms | Peak saved, MiB |
|---|---|---:|---:|
| 4 | -0.34171, 0.04773, 0.05420, 0.13676 | -0.02575 | 37.21875 |
| 8 | 0.04661, 0.21466, 0.13482, 0.03971 | 0.10895 | 127.49902 |

The W8 result is a promising memory candidate with a small observed latency
benefit. The W4 timing result is inconclusive. Neither is adopted yet. All timing
uses batch 1 at padded 2176×3840, all three outputs, 10 warmups and 50 eager
measurements per process. Image I/O and metrics are excluded. Both sides use the
same Python adapter. These results do not certify the complete sparse-branch
30% budget, or performance on other GPUs/shapes.

Evidence and reproduction sources:

- `results/compact_first_tests.json`, `compact_first_pipeline_tests.json`
- `results/compact_first_memcheck.json`
- `results/compact_first_image_smoke_{4,8}_{0,1}.json`
- `results/compact_first_screen_pair.json`
- `results/compact_first_memory_profile_{4,8}_{0,1}_{0,1}.json` (six measured modes)
- `results/compact_critical_images_{4,8}.json`, `compact_critical_pair.json`
- `outputs/prototypes/make_compact_first.py`, `make_compact_first_runtime.py`
- `outputs/prototypes/run_compact_first_screen.py`, `run_compact_critical_screen.py`

The independent outer-network feature-lifetime experiment discovered during this
work must be measured on FP16 as well as both quantized models. Its effects must
not be attributed to compact-cache kernels. Combination measurements are still
pending.
