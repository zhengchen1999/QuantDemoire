# Retuning after retained stages, FewChannels and packed CSR metadata

The saved joint-next plans predated these kernel changes. The current
`script/tune_joint_next.py` already times `PackedConv.forward`, so all current
flags, derived metadata, residual processing and successor-specific prefix
requirements participate in the measurement. No production tuner patch was
needed. Inputs came from `results/prepare_vectors_all_shapes_{4,8}.json`, whose
hashes match the shape provenance of the previous plans.

Both bit widths completed all 157 shapes. Every alternative output and prefix
was checked against complete preparation and the separate reference. The
dynamic program compares separate, joint, direct-prefix and transpose-prefix
paths, including whether a predecessor has supplied a quantized prefix.

The unconstrained W4 plan selects 151 joint layers and six separate layers;
W8 selects 144 joint layers and 13 separate layers. These are not performance
claims. Relative to the current plans, 33 and 45 layer-mode pairs changed.
Candidate copies also retain the previous peak-memory exclusions, reducing
those counts to 32 and 44. Both candidates per bit width passed a two-image,
three-output and changing-input CUDA Graph screen.

Two fresh-process rounds, 30 samples per round, reversed order in round two:

| Format / plan | Median latency | Peak allocated bytes |
|---|---:|---:|
| W4 current | 81.631 ms | 2,594,161,152 |
| W4 retuned | 81.506 ms | 2,661,790,720 |
| W4 retuned with previous memory exclusions | 81.617 ms | 2,592,453,120 |
| W8 current | 95.579 ms | 2,827,205,632 |
| W8 retuned | 95.308 ms | 2,957,761,024 |
| W8 retuned with previous memory exclusions | 95.362 ms | 2,826,157,056 |

No candidate was adopted. The constrained W4 gain is negligible. The
constrained W8 saving varies from 0.013 to 0.421 ms across rounds, so its
aggregate 0.227% saving remains inconclusive. The unconstrained alternatives
add 67.6 MB and 130.6 MB of peak allocation for small latency differences.
Further paired measurements may revisit W8; no full-image adoption validation
has been claimed for these candidate plans.

The outcome illustrates a limit of summing isolated-layer timings: the model
does not capture all cross-layer cache and allocator effects. More joint
layers or a smaller predicted cost does not guarantee a faster full network.

Artifacts and runners:

- `results/retune_metadata_{4,8}.json`: measured alternatives.
- `results/retune_metadata_e2e.json`: same-batch E2E comparison.
- `results/retune_metadata_plan_differences.json`: changed layer modes.
- `results/retune_metadata_assessment.json`: differences and adoption decision.
- `outputs/prototypes/run_retune_metadata.py`: tuner runner.
- `outputs/prototypes/screen_retuned_metadata.py`: output/Graph and E2E screen.

Weights, quantizers, sparse masks and the recommended runtime profiles remain
unchanged. All work is local.
