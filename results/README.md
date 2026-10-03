# Measurement evidence

Use [the final summary](release_20261003/summary.json) and
[release results](../docs/RELEASE_RESULTS.md) for current claims.

- `release_20261001/`: final deployment format checks, 500-image equivalence,
  alternating-process timing, full CUDA/fakequant quality, branch cost and memory.
  The queue began October 1 and completed October 3; original names are preserved.
- `release_20261003/`: current summary and software/package validation.
- `release_20260930/`: floating references, separate 0.5%-tail adjusted PTQ,
  training-only selection, dataset provenance and clean-build evidence.
- Other retained files support explicitly historical kernel/optimization documents;
  their weights, sample counts and denominators may differ from the final release.

`artifact_manifest.json` hashes every retained raw evidence file. Report contents
are preserved, including historical paths and source hashes. These paths are
provenance, not inference dependencies. Uncited exploration, profiler traces and
long logs remain local and are excluded by the explicit Git allowlist. Model
weights, datasets and environments are excluded from the source repository.
