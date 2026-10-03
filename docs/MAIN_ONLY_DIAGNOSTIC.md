# Main-only branch-cost diagnostic

This is a counterfactual timing experiment, not a deployable quality-valid
model. Production weights, sparse masks and runtime profiles are unchanged.
Both original weight tails remain present in the deployed full model.

The adapter disables sparse arithmetic and planar auxiliary preparation while
retaining applicable next-consumer quantization, packed prefixes, residual
fusion, FP16 outputs and common floating operators. Dense prefixes include the
same FP16 rounding before next-consumer quantization. The current layer and
producer-edge schedule is inherited from the verified full model; a separately
tuned main-only schedule may be faster and increase the reported ratios.

An initial four-warp producer accidentally used the eight-warp launch bound
`__launch_bounds__(256,4)`. Static resources showed 64 registers and a 232-byte
stack. Its W8 aggregate time was 24.112 ms across 20 launches, making the first
main-only network slower than the full model. Those measurements are preserved
in `results/main_only_network_diagnostic.json`, with the limitation recorded in
`results/main_only_initial_diagnostic_audit.json`. They must not be used to
certify a branch budget.

The repaired isolated library uses the ordinary CUTLASS launch for four-warp
producers. Their static resources are 114 registers and zero stack, compared
with 116 registers and zero stack for the existing four-warp dense convolution.
Static resources are not measurements of achieved occupancy or memory traffic.

Validation passed 144 native prefix cases covering both bit widths, four/eight
warps, geometry, tails, separate streams and changing-input CUDA Graph replay.
Two actual UHDM images per bit width matched the same-mode unfused-output
reference at all three model outputs, including changing-input Graph replay.
The independent native oracle checks fused prefixes against complete preparation.

Fresh-process E2E timing used two rounds of 30 samples, reversing order in the
second round, batch one, padded 2176 x 3840 inputs and all three outputs:

| Format | Full model | Main-only | Network increment |
|---|---:|---:|---:|
| W4A4 | 81.471 ms | 65.487 ms | 24.409% |
| W8A8 | 95.349 ms | 80.510 ms | 18.431% |

These network denominators include common floating operators. A separate
three-frame CUDA trace sums only quantized-layer kernels, including input
preparation and fused output / next-consumer prefixes:

| Format | Full quantized-layer sum | Main-only sum | Increment |
|---|---:|---:|---:|
| W4A4 | 61.983 ms | 46.049 ms | 34.603% |
| W8A8 | 75.767 ms | 61.075 ms | 24.056% |

Common floating kernels account for about 19.3–19.4 ms in each trace. Profiled
kernel sums and unprofiled wall latency are different measurements. Neither
aggregate ratio certifies every layer, or the standalone compute-only budget.
W4 still exceeds 30% in this preparation-inclusive aggregate comparison.

Artifacts:

- `results/main_only_launch_fix_diagnostic.json`: raw E2E runs and scope.
- `results/main_only_launch_fix_profile_summary.json`: categories and CUDA sums.
- `results/dense_prefix_launch_fix_resources.json`: static resources.
- `results/dense_prefix_launch_fix_tests.json`: native prefix checks.
- `outputs/prototypes/run_dense_prefix_launch_fix.py`: validation / timing runner.
- `outputs/prototypes/profile_dense_prefix_launch_fix.py`: profiling runner.
- `outputs/prototypes/make_dense_prefix_launch_fix.py`: isolated launch fix.

The isolated build links production objects from the verified local JIT cache;
its source and binary hashes are included in the reports. It is not a portable
deployment profile. No production source or artifact was replaced by this
diagnostic.
