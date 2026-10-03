# Logical C80 experiment: not selected

This candidate reduced four W4 spatial convolutions to a logical C80 integer
matrix while retaining physical C96 activation rows and both consumer-cache
layouts. Compact integer weights were derived at runtime; serialized weights,
outlier masks, quantizers and sparse FP32 FMA order were unchanged.

The earlier compact-input and static-address alternatives were also screened.
Only the dynamic FewChannels iterator with sixteen INT4 elements per access
showed enough isolated benefit to justify integrating first/second-prefix
production for evaluation. Its four actual layer modes summed 2.285101 to
2.211280 ms, a saving of about 0.074 ms before whole-network effects.

The integrated candidate passed 1,044 tests, including 76 native cases and
derived-cache checks. W4 and W8 each passed all 157 quantized shapes, all actual
first/second-cache edges, and all three outputs on the 100-image local UHDM
subset with changing-input CUDA Graph replay. W8 did not enable the C80
specialization. Peak allocated memory increased by 47,104 bytes for W4.

Performance evidence was too small and inconsistent for default adoption:

| Comparison | Result |
| --- | --- |
| Four fresh-process paired rounds | Savings +0.261677, −0.193027, +0.035313, +0.070190 ms |
| Fresh-process per-mode medians | Previous 80.237783 ms; candidate 80.283136 ms |
| Same-model old/new binary and forward-method interleaving | Five of six blocks faster; mean saving 0.028922 ms |
| Interleaved block savings | +0.056252, +0.075615, −0.057842, +0.040607, +0.029202, +0.029697 ms |

The interleaved test shared cudnn state and kept derived weights resident in both
modes; it is latency-only. Independent fresh processes supplied the memory
comparison. No repeat was selected as the headline improvement. The numerical
checks establish correctness of the candidate, not a stable speed advantage.

The previous verified two-successor implementation, native binary and portable
profiles were restored, with their saved hashes checked. The recommended
configuration remains unchanged. The complete evaluated candidate is preserved
in `outputs/snapshots/logical80_candidate/sources.tar.gz`, with its binary and
manifest beside it. Historical adoption scripts are disabled.

Evidence: [decision and provenance](../results/logical80_assessment.json),
[fresh-process pair](../results/logical80_production_pair.json),
[integrated interleaving](../results/logical80_production_interleaved.json),
and [candidate same-batch timing](../results/logical80_selected_e2e.json).
The latter is an experimental result, not the selected runtime profile.
