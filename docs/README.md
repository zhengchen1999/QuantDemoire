# ESDNet / UHDM documentation

The current release covers ESDNet on UHDM, W4A4/W8A8 integer CUDA inference,
and the matching FP16 outlier branch.

- [Reproduce, evaluate, benchmark and package models](REPRODUCE.md)
- [Final release results and known limitations](RELEASE_RESULTS.md)
- [Calibration protocol and deviations from the paper](PAPER_PROTOCOL.md)
- [Portable model and execution-profile format](FORMAT.md)
- [CUDA implementation](../kernel/README.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

The dated audits, execution plans, legacy reproduction guide and
[optimization experiment index](DEPLOYMENT_EXPERIMENT_INDEX.md) are historical
research records. Their model identities, datasets and timing denominators can
differ from the final release. Use RELEASE_RESULTS.md and models.json for the
current weight-bound results. Superseded checkpoints are not public downloads.

The release retains the measured failures: the strict sparse-branch overhead
target of 30% is not met, and the adjusted quantized paper results are not an
exact numerical reproduction. A complete acceptance run does not change those
research outcomes.
