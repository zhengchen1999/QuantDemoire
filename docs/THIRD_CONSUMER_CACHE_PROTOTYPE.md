# Third consumer cache experiment

This is an isolated, unadopted experiment under `outputs/prototypes`. The selected
production profile remains the shared W4 operand lookup described in
[SHARED_W4_LOOKUP.md](SHARED_W4_LOOKUP.md). Serialized weights, quantizers, CSR
entries, and their FMA order are unchanged. No files were uploaded or pushed.

The third-last feature of an ESDNet DenseBlock has exactly three remaining
consumers. Its producer can emit three distinct consumer-specific quantized
caches and omit the FP16 feature. The first cache uses the existing full-size
prefix buffers; the second and third each contain just the new 32 channels.
Only fully covered features may be represented by metadata. Caches are owned by
one consumer invocation and must not propagate implicitly through nested concats.

Native producer validation covers 512 cases, including different scales and zero
points for same-bit consumers, FP16 subnormal operands, borders, streams, and
changing-input CUDA Graph replay. Consumer preparation covers 72 cases against
fully materialized real-input preparation. The DenseBlock adapter covers 32
cases, including residual output, mixed full and metadata inputs, cache reuse
guards, and producer selection. Initial uncached memcheck covers 168 cases with
zero errors. The allowlist extension was subsequently retested in all 32 pipeline
cases, also with zero memcheck errors.

On the original saved plans, five W4 and six W8 producers are eligible. Three
local UHDM images per bit match all three reference outputs exactly, with Graph
replay. Four alternating fresh-process timing pairs show no consistent gain:

| Dispatch | Bit width | Mean paired saving, ms | Faster pairs | Peak change, MiB |
|---|---:|---:|---:|---:|
| Original plans | 4 | -0.03698 | 1 / 4 | 0 |
| Original plans | 8 | -0.06044 | 1 / 4 | -0.5 |
| Block-selected restored prefixes | 4 | 0.05726 | 3 / 4 | 0 |
| Block-selected restored prefixes | 8 | 0.26409 | 4 / 4 | +189.25781 |
| W8 without high-memory decoder block | 8 | -0.01710 | 2 / 4 | 0 |

Positive savings mean lower latency. The last result includes one relatively
large negative first pair; the remaining pairs are mixed as well. No dispatch
has been adopted. Small differences and differences between independently run
batches should not be treated as reliable speedups.

The block-selected experiment restores only previously disabled first-prefix
edges in derived experimental plans. Six edges for W4 and five for W8 are
restored; all joint/separate convolution choices are preserved. Selection used
two block-level Graph timing rounds, reversed order, and a predeclared threshold
of at least 0.002 ms and 0.5% saved in both rounds. Ten W4 and seven W8 candidates
passed. Their aggregate block savings did not predict an equivalent whole-network
gain. Three images per bit and all three outputs remain exact. Removing the
selected W8 `decoder.decoder_1.sam_blocks.0.basic_block` restoration removes the
peak increase but also loses a stable timing benefit.

Evidence:

- `results/third_terminal_pair.json`, `third_terminal_original_assessment.json`
- `results/third_restored_block_screen.json`
- `results/third_selected_pair.json`, `third_selected_assessment.json`
- `results/third_memory_pair.json`
- `results/third_terminal_image_smoke_{4,8}.json`
- `results/third_selected_image_smoke_{4,8}.json`
- `results/joint_next_three_tests.json`, `prepare_third_tests.json`
- `results/third_terminal_pipeline_tests.json`
- `results/third_terminal_memcheck.json`, `third_selected_pipeline_memcheck.json`

All timing pairs use batch 1 at padded 2176×3840, all three output scales,
10 warmups and 50 eager measurements per fresh process, with reversed mode order
on alternate rounds. Both sides use the same Python routing adapter. The
memory-constrained W8 controller completed every benchmark; an aggregation loop
that incorrectly also expected W4 was fixed after the run, using the saved raw
measurements. The report records this bookkeeping correction.

The next isolated experiment stores the first cache compactly too, and delays
allocation of the complete packed and planar input until the consumer executes.
This can reduce overlapping allocations, at the cost of an extra compact-cache
copy during preparation. It is not yet a validated production optimization.
