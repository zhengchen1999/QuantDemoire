# Coalesced sparse FP16 Tensor Core experiment

The isolated prototype is not adopted. Even after precomputing spatial offsets and packing MMA coefficient fragments, it is slower than the current CSR branch on the 20 measured ESDNet spatial convolution hotspots. These are layer measurements on RTX 4090, not whole-network measurements.

The prototype derives an exact union of the immutable CSR coordinates for each eight-output group. It preserves every original tail weight, fills absent coefficients with zero, and cooperatively gathers neighboring pixels into a shared FP16 tile. Four warps compute FP16 multiplication with FP32 accumulation. The first version uses coordinate decoding and loads coefficients individually; the second version adds optional precomputed interior offsets and prepacked MMA B fragments. Metadata construction is outside timing.

| Sum over 10 measured layers per bit width | W4A4 | W8A8 |
| --- | ---: | ---: |
| Current CSR merge | 6.499 ms | 6.440 ms |
| First Tensor Core version | 8.151 ms | 8.036 ms |
| Best second version | 7.547 ms | 7.428 ms |
| Best version versus CSR | 16.14% slower | 15.34% slower |
| Preparation + integer main + current CSR | 20.147 ms | 23.824 ms |
| Preparation + integer main + best candidate | 21.987 ms | 26.085 ms |

The eight-output union increases coefficient slots to roughly 8.61 times the original CSR entries in the prior structural analysis. That expansion and the shared-memory gather/synchronization remain costs even though Tensor Cores execute the dot products. The measurements do not establish how much time each cost contributes.

The first version passed 48 cases against explicit sparse FP64 product sums, checking the FP32 accumulators, final FP16 rounding, residual/ReLU semantics, another stream, and changing-input CUDA Graph replay. An earlier FP64 convolution reference produced tiny nonzero residues when every exact sparse product was zero; the audit is preserved, and the test was corrected to sum the products directly without enlarging its acceptance bound. The second version passed 192 cases with exact FP16 and FP32 equality to the first version.

FP32 reduction order differs from CSR. Maximum observed FP16 output differences on the measured real layers were 0.001953125 for W4 and 0.00390625 for W8. These are not full-image quality results. The [CUDA 12.1 PTX specification](https://docs.nvidia.com/cuda/archive/12.1.1/pdf/ptx_isa_8.1.pdf) leaves aspects of MMA accumulation order and rounding unspecified; the numerical bounds here are acceptance checks on the tested hardware. Full-image quality validation would be required before adoption.

Evidence:

- `results/sparse_tc_coalesced_error_audit.json`
- `results/sparse_tc_coalesced_tests.json`
- `results/sparse_tc_coalesced_v2_tests.json`
- `results/sparse_tc_coalesced_screen.json`
- `results/sparse_tc_coalesced_v2_screen.json`
- `results/sparse_tc_coalesced_v2_summary.json`
- Sources, loaders, benchmark scripts, and union/fragment derivation under `outputs/prototypes/`.

## Joint epilogue follow-up

Three isolated joint kernels were subsequently implemented after the selected feature-omission baseline. They run sparse MMA after the integer main loop, retain the original half-rounded main result, and avoid the standalone main-output round trip. None is adopted. Each passed 144 numerical, stream and changing-input Graph cases against the independently checked standalone MMA implementation. This establishes candidate/reference agreement on those cases, not CSR bitwise equality or full-image quality.

| Candidate; 20 actual spatial hotspots per bit | W4 CSR → candidate compute | W8 CSR → candidate compute | W4 preparation + compute | W8 preparation + compute |
| --- | ---: | ---: | ---: | ---: |
| Two paired union groups with CTA synchronization | 14.210 → 20.068 ms | 20.221 → 25.866 ms | 24.630 → 30.813 ms | 32.760 → 38.620 ms |
| Independent M32/N8 warp tiles | 14.128 → 17.468 ms | 20.246 → 22.532 ms | 24.644 → 28.192 ms | 32.771 → 35.359 ms |
| Warp tiles with balanced exact unions | 14.225 → 16.315 ms | 20.267 → 21.523 ms | 24.707 → 26.995 ms | 32.884 → 34.396 ms |

Each row uses its own same-batch CSR control. Results across rows come from separate timing batches. The best balanced candidate still increases computation by 14.69% / 6.20% and preparation-plus-computation by 9.26% / 4.60%. All screens exclude outgoing quantized-prefix production and feature omission; they are not the selected network's latency.

The warp versions use 62 registers, zero static local bytes and 24 KiB dynamic shared memory, compared with 64 registers / 16 static local bytes for the first joint candidate. These compiler/runtime resource values are not measured spill traffic. Balancing greedily groups up to eight original output rows inside each 32-output tile and maps results back to their canonical columns. Every original CSR location and FP16 value bit is checked and retained; dummy coefficients are zero. The reference explicitly pads and permutes dense/residual channels, including partial-group cases with K=44, then inverts the permutation.

Evidence is recorded in `results/joint_tc_union{,_warp,_balanced}_{tests,screen,summary}.json` and `results/joint_tc_union_group_audit.json`. The warp summary corrects a source-hash field accidentally inherited from the v1 benchmark generator; the tests and checked extension record the actual v2 source. Prototype sources and generators are under `outputs/prototypes/`. Because all three candidates remain slower, no producer-prefix integration, full-image quality run, or deployment adoption was performed.
