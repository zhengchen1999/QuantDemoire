# Two-successor quantization

The second-successor fusion is integrated and enabled in the local verified profiles. All 1,041 tests passed, including 16 native preparation and 64 native producer cases inside the suite. Both bit widths passed all 157 actual quantized shapes, all first/second cache edges, and all three outputs on the 100-image local UHDM subset with changing-input CUDA Graph replay. Weights and quantization configurations are unchanged.

A DenseBlock feature appears first in the next convolution input, then as the second 32-channel segment of the following input. The producer can quantize its already-rounded FP16 output with both consumers' distinct smoothing multipliers and activation parameters. It writes the existing first-consumer prefix and one compact packed-only cache for the second consumer. That later consumer reads the cached codes and constructs any required planar sparse input while preparing its remaining features.

No quantizer is shared between consumers. FP16 feature tensors remain available for later DenseBlock inputs. Original CSR locations, FP16 outlier weights, the main-output half rounding, and sparse FP32 FMA order are unchanged. The positive/negative weight tail selection and fakequant representation are therefore unchanged.

## Isolated experiment

Consumer-only timing initially excluded cache production. Adding the producer measurement and then testing the whole network progressively accounts for the actual costs:

| Measurement | W4A4 | W8A8 |
| --- | ---: | ---: |
| Eligible second-consumer edges | 33 | 32 |
| Consumer preparation saving, cache already available | 1.173 ms | 0.832 ms |
| Extra producer cost, separate layer measurement | 0.392 ms | 0.458 ms |
| Layer-screen net saving | 0.781 ms | 0.374 ms |
| Whole-network baseline, two rounds | 80.810 ms | 95.330 ms |
| Whole-network candidate, two rounds | 80.136 ms | 94.906 ms |
| Whole-network saving | 0.674 ms | 0.424 ms |
| Baseline peak allocated | 2,594,161,152 bytes | 2,827,205,632 bytes |
| Candidate peak allocated | 2,594,161,152 bytes | 2,825,370,624 bytes |

The whole-network experiment used fresh processes, reversed the second round, and measured 30 eager iterations after seven warmups per process. It includes producer quantization, all cache reads/writes, and allocations. These results apply to the recorded RTX 4090 configuration and batch-1 padded UHDM input; the consumer-only savings are not an end-to-end prediction.

The caches add 470,016,000 bytes of writes per W4 inference and 848,117,760 bytes per W8 inference across the eligible edges. The largest single cache is 33,423,360 / 66,846,720 bytes. Total writes are not peak memory: a cache survives only until its specific second consumer prepares its input. The measured peak above includes those lifetimes. No peak reduction is guaranteed for other shapes or allocators.

Evidence:

- `results/prepare_cached_second_tests.json`: 16 native preparation cases.
- `results/joint_next_two_tests.json`: 64 producer cases, including mixed consumer bit widths, both prefix store strategies, planar omission, another stream, and changing-input CUDA Graph replay.
- `results/prepare_cached_second_screen.json`: actual consumer headroom.
- `results/joint_next_two_screen.json`: producer cost and numerical checks on all eligible edges.
- `results/joint_next_two_e2e.json`: paired network timing and peak allocation.
- `results/joint_next_two_equivalence_screen_{4,8}.json`: two real images per bit, all three outputs and changing-input Graph replay match the reference exactly.

## Integrated implementation

`QUANTDEMOIRE_FUSE_SECOND_QUANT=1` enables the path when the producer is an eligible eight-warp spatial CSR joint kernel with balanced rows and packed metadata, the middle layer can produce its usual prefix, and all three layers use `quad_swizzle`. Producer and middle outputs must each have 32 channels. Unsupported chains retain the existing preparation path. Large shapes outside the existing 32-bit sparse-offset guard also retain it.

`kernel/csrc/joint_common.cuh` contains the shared joint implementation. `joint_two.cu` enables the extra consumer output at compile time; ordinary joint kernels do not contain that output code. `prepare_second.cu` consumes the compact second cache. The JIT content fingerprint includes the shared header as well as compiled sources.

`ChannelConcat` carries the first prefix and optional second cache for one invocation. The consuming layer clears both after preparation. A nested concatenation never inherits another consumer's caches. The final DenseBlock convolution may consume a second cache but does not leave one pending.

This optimization does not certify the complete FP16-branch overhead target of 30%. It preserves the sparse arithmetic and reduces input preparation; the complete branch budget still requires its own matched, independently tuned main-only comparison.

## Integrated results

| Same-batch two-round eager timing | Median ms | Speedup over FP16 | Peak allocated MiB |
| --- | ---: | ---: | ---: |
| FP16 operands / FP32 accumulation | 129.859 | 1.000x | 3102.45 |
| W4A4 + FP16 branch | 80.220 | 1.619x | 2473.98 |
| W8A8 + FP16 branch | 95.117 | 1.365x | 2694.48 |

A separate paired experiment uses the preserved previous CUDA library and original forward methods:

- W4: 81.142 to 80.223 ms, saving 0.920 ms; peak allocated does not increase.
- W8: 95.379 to 95.076 ms, saving 0.303 ms; peak allocated does not increase.

Evidence: [selected same-batch benchmark](../results/two_successors_selected_e2e.json), [preserved-baseline pair](../results/two_successors_production_pair.json), and [validation summary](../results/two_successors_validation.json). The previous recommended configuration is retained in `config/experiments/optimized_flat_filter.json`. No remote push or upload occurred.

The compute-only selected-plan diagnostic excludes all preparation and outgoing prefixes:

- W4: main 24.436 ms, with sparse 35.288 ms, increment 44.41%.
- W8: main 36.302 ms, with sparse 46.364 ms, increment 27.72%.

These diagnostics are not complete-branch certification. The earlier matched-schedule main-only diagnostic predates this fusion.

```bash
uv run python script/benchmark_variants.py --config config/experiments/two_successors_selected.json \
  --output results/two_successors_retest.json --rounds 2 --repeats 30
```
